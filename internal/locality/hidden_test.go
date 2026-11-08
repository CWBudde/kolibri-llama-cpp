package locality

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"reflect"
	"slices"
	"testing"
)

// randomPred is a capture file of horizon h for layers x tokens: w distinct experts of nExpert in
// every row that has a target block, -1 in the rows that have none (row + h >= layers).
func randomPred(r *rand.Rand, layers, tokens, w, nExpert, h int) Selections {
	p := Selections{Layers: layers, Tokens: tokens, K: w, IDs: make([]int32, layers*tokens*w)}
	for row := range layers {
		for t := range tokens {
			ids := p.IDs[(row*tokens+t)*w : (row*tokens+t+1)*w]
			if row+h >= layers {
				for i := range ids {
					ids[i] = -1
				}
				continue
			}
			for i, e := range r.Perm(nExpert)[:w] {
				ids[i] = int32(e)
			}
		}
	}
	return p
}

// hiddenTrace is a small workload: 6 layers x 240 tokens, two answers decoded token by token.
func hiddenTrace(t *testing.T, r *rand.Rand, k, nExpert int) (Selections, []Span, []Answer) {
	t.Helper()
	sel := randomSelections(r, 6, 240, k, nExpert)
	answers := []Answer{{Prompt: 60, Tokens: make([]int32, 40)}, {Prompt: 150, Tokens: make([]int32, 50)}}
	spans, err := OnlineSpans(240, answers, true, 32)
	if err != nil {
		t.Fatal(err)
	}
	return sel, spans, answers
}

func TestHiddenStatePredictsTheRowBefore(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	sel, spans, _ := hiddenTrace(t, r, 6, 24)
	pred := randomPred(r, sel.Layers, sel.Tokens, 12, 24, 2)
	p, err := NewHiddenState(pred, 24)
	if err != nil {
		t.Fatal(err)
	}
	predicted, decoded := 0, 0
	for si, s := range spans {
		if s.Phase == Decode {
			decoded++
		}
		for l := range sel.Layers {
			info := StepInfo{si*sel.Layers + l, s, l}
			for _, k := range []int{1, 6, 12} {
				layer, got := p.Predict(SameToken, info, k, nil)
				// the same token's horizon: before layer l of a decoded token, layer l+1, from row
				// l-1 of the token, the residual stream after block l-1, which has run
				if s.Phase != Decode || l < 1 || l+1 >= sel.Layers {
					if got != nil {
						t.Fatalf("step %d (%s, layer %d): predicted %v, want nothing", info.Index, s.Phase, l, got)
					}
					continue
				}
				row := pred.IDs[((l-1)*pred.Tokens+s.First)*pred.K:]
				want := make([]int, k)
				for i := range want {
					want[i] = int(row[i])
				}
				if layer != l+1 || !slices.Equal(got, want) {
					t.Fatalf("step %d (token %d, layer %d), k %d: layer %d, %v; want layer %d, %v", info.Index,
						s.First, l, k, layer, got, l+1, want)
				}
				if k == 6 {
					predicted++
				}
			}
			if _, got := p.Predict(NextToken, info, 6, nil); got != nil {
				t.Fatalf("step %d: next token predicted %v", info.Index, got)
			}
			p.Observe(info, stepExperts(sel, s, l, nil))
		}
	}
	if want := decoded * (sel.Layers - 2); decoded == 0 || predicted != want {
		t.Fatalf("%d steps predicted, want layers 2 to %d of %d decoded tokens, %d", predicted, sel.Layers-1, decoded,
			want)
	}
}

// Control M104: reading row l before layer l (the residual stream after block l, which has not
// run) is future knowledge; TestHiddenStateCausal fails on it.
func TestHiddenStateCausal(t *testing.T) {
	// what may decide the prediction before layer l of decoded token t: rows below l of token t and
	// any row of an earlier token. Rewriting everything else must not change it.
	r := rand.New(rand.NewPCG(3, 4))
	sel, spans, _ := hiddenTrace(t, r, 6, 24)
	pred := randomPred(r, sel.Layers, sel.Tokens, 12, 24, 2)
	p, err := NewHiddenState(pred, 24)
	if err != nil {
		t.Fatal(err)
	}
	for si, s := range spans {
		if s.Phase != Decode || si%3 != 0 {
			continue
		}
		for l := 1; l+1 < sel.Layers; l++ {
			info := StepInfo{si*sel.Layers + l, s, l}
			_, want := p.Predict(SameToken, info, 12, nil)
			future := pred
			future.IDs = slices.Clone(pred.IDs)
			other := randomPred(r, sel.Layers, sel.Tokens, 12, 24, 2)
			for row := range sel.Layers {
				for tok := range sel.Tokens {
					if tok > s.First || (tok == s.First && row >= l) {
						i := (row*sel.Tokens + tok) * pred.K
						copy(future.IDs[i:i+pred.K], other.IDs[i:i+pred.K])
					}
				}
			}
			q, err := NewHiddenState(future, 24)
			if err != nil {
				t.Fatal(err)
			}
			if _, got := q.Predict(SameToken, info, 12, nil); !slices.Equal(got, want) {
				t.Fatalf("token %d, layer %d: %v with the future rewritten, %v without", s.First, l, got, want)
			}
		}
	}
}

// synthPred is a pred-h2 file built from the selections themselves: row r of token t holds the
// Top-K of layer r+2 at t, -1 where layer r+2 does not exist; written as int16 .npy and read back.
func synthPred(t *testing.T, sel Selections) Selections {
	t.Helper()
	ids := make([]int16, len(sel.IDs))
	for row := range sel.Layers {
		for tok := range sel.Tokens {
			for j := range sel.K {
				v := int16(-1)
				if row+2 < sel.Layers {
					v = int16(sel.IDs[((row+2)*sel.Tokens+tok)*sel.K+j])
				}
				ids[(row*sel.Tokens+tok)*sel.K+j] = v
			}
		}
	}
	hdr := fmt.Sprintf("{'descr': '<i2', 'fortran_order': False, 'shape': (%d, %d, %d), }", sel.Layers, sel.Tokens,
		sel.K)
	p, err := ReadNPY(bytes.NewReader(npy(hdr, ids)))
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// Control M105: an off-by-one token index in the predictor (token t-1's row before token t's
// step) fails the mapping check.
func TestHiddenStateMapping(t *testing.T) {
	// the pre-registered mapping check: the selections themselves, fed through the file-backed
	// predictor, replay exactly as the perfect same-token oracle at k = 6, the Top-6
	r := rand.New(rand.NewPCG(5, 6))
	sel, spans, answers := hiddenTrace(t, r, 6, 24)
	cut, err := SplitAfter(spans, answers, 1)
	if err != nil {
		t.Fatal(err)
	}
	p, err := NewHiddenState(synthPred(t, sel), 24)
	if err != nil {
		t.Fatal(err)
	}
	hidden := RankWith(p, sel, spans, 24, SameToken, 6)
	perfect := RankWith(NewPerfect(sel, NewMarkov(sel.Layers, 24, MarkovParams{HalfLife: 8, Mix: 0.5})), sel, spans,
		24, SameToken, 6)
	for step := range hidden.Steps() {
		hl, ht := hidden.Top(step, 6)
		pl, pt := perfect.Top(step, 6)
		if hl != pl || !slices.Equal(ht, pt) {
			t.Fatalf("step %d: file-backed layer %d %v, oracle layer %d %v", step, hl, ht, pl, pt)
		}
	}
	eb := sizes(sel.Layers, 100)
	tm := Timing{Links: []Link{{Name: "ssd", BytesPerSecond: 1e4}}, DecodeStep: 1e-3, PrefillToken: 1e-4}
	for _, c := range []int{2, 6, 12, 24} {
		var budget int64
		for _, b := range eb {
			budget += int64(c) * b
		}
		got, err := ReplaySegments(sel, spans, []int{cut}, 24, eb, NewRanked(NewSharedLRU(24, eb, budget), hidden, 6),
			tm)
		if err != nil {
			t.Fatal(err)
		}
		want, err := ReplaySegments(sel, spans, []int{cut}, 24, eb, NewRanked(NewSharedLRU(24, eb, budget), perfect, 6),
			tm)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("%d experts per layer: the file-backed replay %+v, the oracle's %+v", c, got[1].Decode, want[1].Decode)
		}
		if c < 24 && got[1].Decode.PrefetchBytes == 0 { // at 24 every expert stays resident
			t.Fatalf("%d experts per layer: nothing prefetched", c)
		}
	}
}

func TestHiddenRecall(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	sel, spans, answers := hiddenTrace(t, r, 6, 24)
	cut, err := SplitAfter(spans, answers, 1)
	if err != nil {
		t.Fatal(err)
	}
	ks := []int{1, 6, 12}
	// pred-h2 is the same token's horizon: its recall is the ranking's, step for step
	pred := randomPred(r, sel.Layers, sel.Tokens, 12, 24, 2)
	p, err := NewHiddenState(pred, 24)
	if err != nil {
		t.Fatal(err)
	}
	ranking := RankWith(p, sel, spans, 24, SameToken, 12)
	for _, rg := range [][2]int{{0, cut}, {cut, len(spans)}, {0, len(spans)}} {
		all, predicted := HiddenRecall(pred, 2, sel, spans, rg[0], rg[1], ks)
		if want := ranking.Recall(sel, spans, rg[0], rg[1], ks); !reflect.DeepEqual(all, want) {
			t.Fatalf("spans %v: %+v, the ranking's %+v", rg, all, want)
		}
		for i := range ks {
			a, x := all[i], predicted[i]
			decoded := a.Steps / sel.Layers
			if x.Steps != decoded*(sel.Layers-2) || x.Needed != x.Steps*sel.K || x.Candidates != a.Candidates ||
				x.Covered != a.Covered || x.Exact != a.Exact {
				t.Fatalf("spans %v, k %d: predicted layers only %+v, all %+v", rg, ks[i], x, a)
			}
		}
	}
	// pred-h1, by hand: layer l of token t from row l-1, layer 0 not predicted
	sel = Selections{Layers: 3, Tokens: 2, K: 2, IDs: []int32{
		0, 1, 2, 3, // layer 0, tokens 0 and 1
		4, 5, 6, 7, // layer 1
		8, 9, 10, 11, // layer 2
	}}
	pred = Selections{Layers: 3, Tokens: 2, K: 3, IDs: []int32{
		5, 9, 4, 7, 6, 1, // row 0: block 1 on token 0 and 1: covers {4, 5} at k 3, {6, 7} at k 2
		8, 0, 1, 11, 3, 2, // row 1: block 2: covers 8 of {8, 9}, 11 of {10, 11}
		-1, -1, -1, -1, -1, -1, // row 2: no block 3
	}}
	spans = []Span{{Prefill, 0, 1}, {Decode, 1, 2}}
	all, predicted := HiddenRecall(pred, 1, sel, spans, 0, 2, []int{2, 3})
	// decode only: token 1. Layer 0 missed, layer 1 both at k 2, layer 2 one of two
	want := []Recall{{Steps: 3, Candidates: 4, Needed: 6, Covered: 3, Exact: 1},
		{Steps: 3, Candidates: 6, Needed: 6, Covered: 3, Exact: 1}}
	if !reflect.DeepEqual(all, want) {
		t.Fatalf("pred-h1: %+v, want %+v", all, want)
	}
	if predicted[0] != (Recall{Steps: 2, Candidates: 4, Needed: 4, Covered: 3, Exact: 1}) {
		t.Fatalf("pred-h1, predicted layers only: %+v", predicted[0])
	}
	// every token as a decoded step: token 0 adds layer 1 both at k 3 and layer 2 one
	all, _ = HiddenRecall(pred, 1, sel, TraceOrder(2), 0, 2, []int{2, 3})
	if want := (Recall{Steps: 6, Candidates: 12, Needed: 12, Covered: 6, Exact: 2}); all[1] != want {
		t.Fatalf("pred-h1, every token at k 3: %+v, want %+v", all[1], want)
	}
}

func TestNewHiddenStateRejects(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 10))
	good := randomPred(r, 4, 5, 3, 8, 2)
	if _, err := NewHiddenState(good, 8); err != nil {
		t.Fatal(err)
	}
	for name, edit := range map[string]func(p *Selections){
		"expert out of range": func(p *Selections) { p.IDs[0] = 8 },
		"negative expert":     func(p *Selections) { p.IDs[1] = -2 },
		"repeated expert":     func(p *Selections) { p.IDs[1] = p.IDs[0] },
		"-1 in a target row":  func(p *Selections) { p.IDs[2] = -1 },
		"a row past the last": func(p *Selections) { p.IDs[len(p.IDs)-1] = 0 },
	} {
		p := good
		p.IDs = slices.Clone(good.IDs)
		edit(&p)
		if _, err := NewHiddenState(p, 8); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}
