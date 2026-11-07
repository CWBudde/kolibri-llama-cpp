package locality

import (
	"math/bits"
	"math/rand/v2"
	"testing"
)

// heldOutAccess is one replay access of a key, and whether it costs anything to a held-out objective.
type heldOutAccess struct {
	key     int
	counted bool
}

// heldOutAccesses lists a trace's accesses for keys of layers [first, first+layers), as bits over the key
// less the first's, in replay order: a step's union once, in the order its tokens select it.
func heldOutAccesses(sel Selections, spans []Span, cut, nExpert, first, layers int) []heldOutAccess {
	var out []heldOutAccess
	for i, s := range spans {
		for l := first; l < first+layers; l++ {
			var seen uint32
			for t := s.First; t < s.End; t++ {
				for _, e := range sel.IDs[(l*sel.Tokens+t)*sel.K : (l*sel.Tokens+t+1)*sel.K] {
					k := (l-first)*nExpert + int(e)
					if seen&(1<<k) == 0 {
						seen |= 1 << k
						out = append(out, heldOutAccess{k, i >= cut && s.Phase == Decode})
					}
				}
			}
		}
	}
	return out
}

// optimalCountedMisses is the fewest counted misses of a demand cache of c equal experts, by search
// over its contents after each access: on a miss it keeps the expert, evicting any other, or uses it
// once.
func optimalCountedMisses(as []heldOutAccess, c int) int {
	cur := map[uint32]int{0: 0}
	set := func(m map[uint32]int, s uint32, v int) {
		if old, ok := m[s]; !ok || v < old {
			m[s] = v
		}
	}
	for _, a := range as {
		next := map[uint32]int{}
		bit := uint32(1) << a.key
		for s, m := range cur {
			if s&bit != 0 {
				set(next, s, m)
				continue
			}
			if a.counted {
				m++
			}
			set(next, s, m)
			if bits.OnesCount32(s) < c {
				set(next, s|bit, m)
			} else if c > 0 {
				for r := s; r != 0; r &= r - 1 {
					set(next, s&^(r&-r)|bit, m)
				}
			}
		}
		cur = next
	}
	best := -1
	for _, m := range cur {
		if best < 0 || m < best {
			best = m
		}
	}
	return best
}

// optimalCountedStepMisses is optimalStepMisses when only the misses of counted steps cost anything.
func optimalCountedStepMisses(steps []uint32, counted []bool, c int) int {
	cur := map[uint32]int{0: 0}
	for i, need := range steps {
		next := map[uint32]int{}
		for set, misses := range cur {
			m := misses
			if counted[i] {
				m += bits.OnesCount32(need &^ set)
			}
			kept := set &^ need
			for s := kept; ; s = (s - 1) & kept {
				if bits.OnesCount32(s)+bits.OnesCount32(need) <= c {
					if v, ok := next[s|need]; !ok || m < v {
						next[s|need] = m
					}
				}
				if s == 0 {
					break
				}
			}
		}
		cur = next
	}
	best := -1
	for _, m := range cur {
		if best < 0 || m < best {
			best = m
		}
	}
	return best
}

// heldOutFixture is a trace of turns: a prefill ubatch, then decoded tokens, cut before a turn.
func heldOutFixture(r *rand.Rand) (Selections, []Span, int, int) {
	layers, nExpert := 1+r.IntN(2), 3+r.IntN(3)
	k := min(1+r.IntN(2), nExpert)
	tokens := 8 + r.IntN(8)
	sel := randomSelections(r, layers, tokens, k, nExpert)
	var spans []Span
	cut := 0
	turns := 2 + r.IntN(2)
	for t, turn := 0, 0; t < tokens; turn++ {
		if turn == turns/2 || turns == 0 {
			cut = len(spans)
		}
		first := min(t+1+r.IntN(3), tokens)
		spans = append(spans, Span{Prefill, t, first})
		t = first
		for d := 1 + r.IntN(4); d > 0 && t < tokens; d-- {
			spans = append(spans, Span{Decode, t, t + 1})
			t++
		}
	}
	if cut == 0 {
		cut = len(spans) / 2
	}
	return sel, spans, cut, nExpert
}

// stepsOf are the steps' unions and whether each costs anything to a held-out objective.
func stepsOf(sel Selections, spans []Span, cut, nExpert, first, layers int) ([]uint32, []bool) {
	steps := stepSets(sel, spans, nExpert, first, layers)
	counted := make([]bool, 0, len(steps))
	for i, s := range spans {
		for range layers {
			counted = append(counted, i >= cut && s.Phase == Decode)
		}
	}
	return steps, counted
}

func testDecode(t *testing.T, sel Selections, spans []Span, cut, nExpert int, bytes []int64, o *Oracle) Counts {
	t.Helper()
	segs, err := ReplaySegments(sel, spans, []int{cut}, nExpert, bytes, o, Timing{})
	if err != nil || o.Err() != nil {
		t.Fatal(err, o.Err())
	}
	return segs[1].Decode
}

// TestHeldOutOracleWholeTraceSliceIsNoBound is the case the slice gets wrong: with room for one
// expert and A, B, A | B, MIN of the whole trace keeps A and misses B in the suffix, at 3 misses in
// all, while keeping B costs the prefix a miss and the suffix none, also at 3.
func TestHeldOutOracleWholeTraceSliceIsNoBound(t *testing.T) {
	sel := Selections{Layers: 1, Tokens: 4, K: 1, IDs: []int32{0, 1, 0, 1}}
	spans, bytes := TraceOrder(4), uniform(1, 2)
	if got := testDecode(t, sel, spans, 3, 2, bytes, NewOracle(sel, spans, 2, bytes, 2)).Misses; got != 1 {
		t.Fatalf("the whole-trace oracle misses %d in the suffix, want 1", got)
	}
	if got := testDecode(t, sel, spans, 3, 2, bytes, NewHeldOutOracle(sel, spans, 3, 2, bytes, 2)).Misses; got != 0 {
		t.Fatalf("the held-out oracle misses %d in the suffix, want 0", got)
	}
}

// TestHeldOutOracleCountsDecodeOnly: a prefill that needs room between two decoded uses is free to
// use its experts once, which MIN of all bytes need not do.
func TestHeldOutOracleCountsDecodeOnly(t *testing.T) {
	// room for one: decode A, prefill B, B, decode A. MIN of all bytes may keep either; only A is free.
	sel := Selections{Layers: 1, Tokens: 4, K: 1, IDs: []int32{0, 1, 1, 0}}
	spans := []Span{{Decode, 0, 1}, {Prefill, 1, 2}, {Prefill, 2, 3}, {Decode, 3, 4}}
	bytes := uniform(1, 2)
	if got := testDecode(t, sel, spans, 1, 2, bytes, NewHeldOutOracle(sel, spans, 1, 2, bytes, 2)).Misses; got != 0 {
		t.Fatalf("the held-out oracle misses %d in decode, want 0", got)
	}
}

// TestHeldOutOracleOptimal holds the held-out oracles to exhaustive search with equal experts: the
// plain ones to the demand-cache optimum, the pinned ones, wherever every step fits, to the
// event-atomic optimum and to no step evictions anywhere.
func TestHeldOutOracleOptimal(t *testing.T) {
	r := rand.New(rand.NewPCG(31, 32))
	below := 0
	for trial := range 80 {
		sel, spans, cut, nExpert := heldOutFixture(r)
		bytes := uniform(sel.Layers, 2)
		all := heldOutAccesses(sel, spans, cut, nExpert, 0, sel.Layers)
		steps, counted := stepsOf(sel, spans, cut, nExpert, 0, sel.Layers)
		for c := range sel.Layers*nExpert + 1 {
			budget := int64(c) * 2
			got := testDecode(t, sel, spans, cut, nExpert, bytes, NewHeldOutOracle(sel, spans, cut, nExpert, bytes, budget)).Misses
			want := optimalCountedMisses(all, c)
			if got != want {
				t.Fatalf("trial %d, %d experts: the held-out oracle misses %d, the optimum %d", trial, c, got, want)
			}
			slice := testDecode(t, sel, spans, cut, nExpert, bytes, NewOracle(sel, spans, nExpert, bytes, budget)).Misses
			if slice < got {
				t.Fatalf("trial %d, %d experts: the whole-trace slice misses %d, below the bound's %d", trial, c, slice, got)
			}
			if slice > got {
				below++
			}
			p := NewPinnedHeldOutOracle(sel, spans, cut, nExpert, bytes, budget)
			segs, err := ReplaySegments(sel, spans, []int{cut}, nExpert, bytes, p, Timing{})
			if err != nil || p.Err() != nil {
				t.Fatal(err, p.Err())
			}
			if e := segs[0].Prefill.StepEvictions + segs[0].Decode.StepEvictions + segs[1].Prefill.StepEvictions +
				segs[1].Decode.StepEvictions; e != 0 {
				t.Fatalf("trial %d, %d experts: %d step evictions", trial, c, e)
			}
			if c < maxBits(steps) {
				continue
			}
			if got, want := segs[1].Decode.Misses, optimalCountedStepMisses(steps, counted, c); got != want {
				t.Fatalf("trial %d, %d experts: the pinned oracle misses %d, the event-atomic optimum %d", trial, c, got, want)
			}
		}
		for c := range nExpert + 1 {
			want, pinned, fits := 0, 0, true
			for l := range sel.Layers {
				want += optimalCountedMisses(heldOutAccesses(sel, spans, cut, nExpert, l, 1), c)
				ls, lc := stepsOf(sel, spans, cut, nExpert, l, 1)
				if c < maxBits(ls) {
					fits = false
					continue
				}
				pinned += optimalCountedStepMisses(ls, lc, c)
			}
			got := testDecode(t, sel, spans, cut, nExpert, bytes, NewHeldOutLayerOracle(sel, spans, cut, nExpert, bytes, c)).Misses
			if got != want {
				t.Fatalf("trial %d, %d per layer: the held-out oracle misses %d, the optimum %d", trial, c, got, want)
			}
			if !fits {
				continue
			}
			got = testDecode(t, sel, spans, cut, nExpert, bytes, NewPinnedHeldOutLayerOracle(sel, spans, cut, nExpert, bytes, c)).Misses
			if got != pinned {
				t.Fatalf("trial %d, %d per layer: the pinned oracle misses %d, the event-atomic optimum %d", trial, c, got, pinned)
			}
		}
	}
	if below == 0 {
		t.Fatal("no trace has a whole-trace slice above the bound, so this test tells them apart nowhere")
	}
}

// stepHolder fails if its Oracle evicts an expert of the running step, used yet or not, to load another.
type stepHolder struct {
	*Oracle
	t     *testing.T
	sel   Selections
	spans []Span
	need  map[int]bool
}

func (h *stepHolder) Prefetch(s StepInfo) []int {
	h.need = map[int]bool{}
	for t := s.First; t < s.End; t++ {
		for _, e := range h.sel.IDs[(s.Layer*h.sel.Tokens+t)*h.sel.K : (s.Layer*h.sel.Tokens+t+1)*h.sel.K] {
			h.need[s.Layer*h.Oracle.nExpert+int(e)] = true
		}
	}
	return h.Oracle.Prefetch(s)
}

func (h *stepHolder) Load(k int) []int {
	ev := h.Oracle.Load(k)
	for _, v := range ev {
		if v != k && h.need[v] {
			h.t.Fatalf("evicted %d of the running step to load %d", v, k)
		}
	}
	return ev
}

// TestPinnedHeldOutHoldsTheStep: an expert of a free step ranks like one never used again, so the
// step has to pin it before anything loads, or a free step evicts and reloads its own experts.
func TestPinnedHeldOutHoldsTheStep(t *testing.T) {
	r := rand.New(rand.NewPCG(41, 42))
	for range 80 {
		sel, spans, cut, nExpert := heldOutFixture(r)
		bytes := uniform(sel.Layers, 2)
		steps := stepSets(sel, spans, nExpert, 0, sel.Layers)
		for c := maxBits(steps); c <= sel.Layers*nExpert; c++ {
			h := &stepHolder{Oracle: NewPinnedHeldOutOracle(sel, spans, cut, nExpert, bytes, int64(c)*2), t: t, sel: sel,
				spans: spans}
			if _, err := ReplaySegments(sel, spans, []int{cut}, nExpert, bytes, h, Timing{}); err != nil || h.Err() != nil {
				t.Fatal(err, h.Err())
			}
		}
	}
}
