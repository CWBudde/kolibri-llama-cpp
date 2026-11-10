package locality

import (
	"math/rand/v2"
	"slices"
	"testing"
)

func TestPerfectRecall(t *testing.T) {
	// 5 layers x 300 tokens, Top-6 of 24 experts; two answers, decoded token by token
	r := rand.New(rand.NewPCG(41, 42))
	sel := randomSelections(r, 5, 300, 6, 24)
	answers := []Answer{{Prompt: 70, Tokens: make([]int32, 60)}, {Prompt: 180, Tokens: make([]int32, 100)}}
	spans, err := OnlineSpans(300, answers, true, 32)
	if err != nil {
		t.Fatal(err)
	}
	cut, err := SplitAfter(spans, answers, 1)
	if err != nil {
		t.Fatal(err)
	}
	pad := MarkovParams{HalfLife: 8, Mix: 0.5}
	ks := []int{6, 12, 24}
	ranking := RankWith(NewPerfect(sel, NewMarkov(sel.Layers, 24, pad)), sel, spans, 24, SameToken, 24)
	markov := Rank(sel, spans, 24, SameToken, pad, 24)
	for _, rg := range [][2]int{{0, cut}, {cut, len(spans)}} {
		decodedTokens := 0
		for _, s := range spans[rg[0]:rg[1]] {
			if s.Phase == Decode {
				decodedTokens++
			}
		}
		// the same token's horizon predicts layers 2 to L-1 of every decoded token, one step ahead:
		// all of them, every expert, at k 6
		predicted := decodedTokens * (sel.Layers - 2)
		got := ranking.Recall(sel, spans, rg[0], rg[1], ks)
		for i, k := range ks {
			x := got[i]
			if x.Steps != decodedTokens*sel.Layers || x.Candidates != k*predicted || x.Exact != predicted ||
				x.Covered != sel.K*predicted {
				t.Fatalf("spans %v, k %d: %+v; want %d steps, %d predicted, every one covered", rg, k, x,
					decodedTokens*sel.Layers, predicted)
			}
		}
		if got[0].Covered != got[0].Needed-2*sel.K*decodedTokens {
			t.Fatalf("spans %v: %d of %d needed covered at k 6; only layers 0 and 1 may be missed", rg,
				got[0].Covered, got[0].Needed)
		}
	}
	// before every step: the target layer's true set first, then the padding predictor's ranks
	// without them, in its order
	for si, s := range spans {
		for l := range sel.Layers {
			step := si*sel.Layers + l
			layer, top := ranking.Top(step, 24)
			ml, mtop := markov.Top(step, 24)
			if (top == nil) != (mtop == nil) {
				t.Fatalf("step %d: predicted %v, Markov %v: the horizons' steps differ", step, top, mtop)
			}
			if top == nil {
				continue
			}
			truth := stepExperts(sel, s, l+1, nil)
			if layer != l+1 || ml != layer || !slices.Equal(top[:6], truth) {
				t.Fatalf("step %d: layer %d, top 6 %v; want layer %d, %v", step, layer, top[:6], l+1, truth)
			}
			want := slices.Clone(truth)
			for _, e := range mtop {
				if !slices.Contains(want, e) {
					want = append(want, e)
				}
			}
			if !slices.Equal(top, want[:24]) {
				t.Fatalf("step %d: ranked %v, want %v", step, top, want[:24])
			}
		}
	}
	// nothing for the next token's horizon
	if _, got := NewPerfect(sel, NewMarkov(sel.Layers, 24, pad)).Predict(NextToken, StepInfo{1, spans[len(spans)-1], 1},
		6, nil); got != nil {
		t.Fatalf("next token: %v", got)
	}
}
