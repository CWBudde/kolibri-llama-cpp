package locality

import (
	"math/rand/v2"
	"slices"
	"testing"
)

// prefetchLog records what the policy it wraps prefetches, step by step.
type prefetchLog struct {
	Policy
	sets [][]int
}

func (p *prefetchLog) Prefetch(s StepInfo) []int {
	out := p.Policy.Prefetch(s)
	set := slices.Clone(out)
	slices.Sort(set)
	p.sets = append(p.sets, slices.Compact(set))
	return out
}

// experts is layer l's experts in span s, as keys, sorted and once each.
func experts(sel Selections, nExpert, l int, s Span) []int {
	var out []int
	for tok := s.First; tok < s.End; tok++ {
		for _, e := range sel.IDs[(l*sel.Tokens+tok)*sel.K : (l*sel.Tokens+tok+1)*sel.K] {
			out = append(out, l*nExpert+int(e))
		}
	}
	slices.Sort(out)
	return slices.Compact(out)
}

func TestLastN(t *testing.T) {
	r := rand.New(rand.NewPCG(13, 14))
	sel, spans, _ := onlineSplit(r)
	bytes := sizes(sel.Layers, 10)
	for _, n := range []int{1, 2, 5} {
		p := &prefetchLog{Policy: NewLastN(NewSharedLRU(12, bytes, 1<<40), sel.Layers, 12, n)}
		if _, err := Replay(sel, spans, 12, bytes, p); err != nil {
			t.Fatal(err)
		}
		// before layer l of span i: layer l+1's experts in spans i-n to i-1; before the last layer,
		// layer 0's in spans i-n+1 to i, the last of them the running span
		step := 0
		for i := range spans {
			for l := range sel.Layers {
				next, last := l+1, i
				if next == sel.Layers {
					next, last = 0, i+1
				}
				var want []int
				for j := max(last-n, 0); j < last; j++ {
					want = append(want, experts(sel, 12, next, spans[j])...)
				}
				slices.Sort(want)
				want = slices.Compact(want)
				if !slices.Equal(p.sets[step], want) {
					t.Fatalf("n %d, span %d, layer %d: prefetches %v, want %v", n, i, l, p.sets[step], want)
				}
				step++
			}
		}
	}
	// on a shared LRU it keeps the LRU's budget, and prefetches while the budget holds fewer than
	// all 36 experts (396 bytes)
	for _, budget := range []int64{0, 25, 100, 200} {
		res, err := Replay(sel, spans, 12, bytes, NewLastN(NewSharedLRU(12, bytes, budget), sel.Layers, 12, 4))
		if err != nil {
			t.Fatal(err)
		}
		if res.PeakBytes > budget || res.Decode.Prefetches == 0 && budget > 0 {
			t.Fatalf("budget %d: peak %d, %d prefetches", budget, res.PeakBytes, res.Decode.Prefetches)
		}
	}
}

func TestLastNRecall(t *testing.T) {
	// one layer, 4 of 6 experts; a ubatch {0,1} {1,2}, then decoded {2,3} {0,4} {4,5}
	sel := Selections{Layers: 1, Tokens: 5, K: 2, IDs: []int32{0, 1, 1, 2, 2, 3, 0, 4, 4, 5}}
	spans := []Span{{Prefill, 0, 2}, {Decode, 2, 3}, {Decode, 3, 4}, {Decode, 4, 5}}
	// from span 2 on, with n 1: {0,4} against {2,3}, nothing covered; {4,5} against {0,4}, 4
	got := LastNRecall(sel, spans, 2, 6, 1)
	want := Recall{Steps: 2, Candidates: 4, Needed: 4, Covered: 1, Exact: 0}
	if got != want {
		t.Fatalf("n 1: %+v, want %+v", got, want)
	}
	// with n 2: {0,4} against {0,1,2,3}, 0; {4,5} against {0,2,3,4}, 4
	got = LastNRecall(sel, spans, 2, 6, 2)
	want = Recall{Steps: 2, Candidates: 8, Needed: 4, Covered: 2, Exact: 0}
	if got != want {
		t.Fatalf("n 2: %+v, want %+v", got, want)
	}
	// from span 1, n 1: {2,3} against the ubatch's {0,1,2}, 2
	got = LastNRecall(sel, spans, 1, 6, 1)
	want = Recall{Steps: 3, Candidates: 7, Needed: 6, Covered: 2, Exact: 0}
	if got != want {
		t.Fatalf("from 1: %+v, want %+v", got, want)
	}
	// a token whose experts all came before: {1,2} after the ubatch {0,1,2}
	sel.IDs[4], sel.IDs[5] = 1, 2
	if got := LastNRecall(sel, spans, 1, 6, 1); got.Exact != 1 || got.Covered != 3 {
		t.Fatalf("an exact set: %+v", got)
	}
}
