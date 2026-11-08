package locality

import (
	"math/bits"
	"math/rand/v2"
	"reflect"
	"testing"
)

func TestStepUnions(t *testing.T) {
	// two layers of three tokens with two experts each
	sel := Selections{Layers: 2, Tokens: 3, K: 2, IDs: []int32{
		0, 1, 1, 2, 2, 3, // layer 0
		4, 5, 4, 5, 4, 5, // layer 1
	}}
	spans := []Span{{Prefill, 0, 2}, {Decode, 2, 3}}
	if got, want := StepUnions(sel, spans, 8, Prefill), []int{3, 2}; !reflect.DeepEqual(got, want) {
		t.Fatalf("prefill unions %v, want %v", got, want)
	}
	if got, want := StepUnions(sel, spans, 8, Decode), []int{2, 2}; !reflect.DeepEqual(got, want) {
		t.Fatalf("decode unions %v, want %v", got, want)
	}
}

// TestStepUnionsMatchReplay holds StepUnions to Replay's own accesses: a step accesses each of its
// union's experts once, so their sum is the phase's accesses.
func TestStepUnionsMatchReplay(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	for range 20 {
		layers, nExpert, k := 1+r.IntN(3), 8+r.IntN(16), 1+r.IntN(6)
		tokens := 20 + r.IntN(100)
		sel := randomSelections(r, layers, tokens, k, nExpert)
		spans, err := OnlineSpans(tokens, []Answer{{Prompt: tokens / 2, Tokens: make([]int32, tokens/4)}}, false,
			2+r.IntN(8))
		if err != nil {
			t.Fatal(err)
		}
		res, err := Replay(sel, spans, nExpert, uniform(layers, 1), &NoCache{})
		if err != nil {
			t.Fatal(err)
		}
		for _, x := range []struct {
			p Phase
			c Counts
		}{{Prefill, res.Prefill}, {Decode, res.Decode}} {
			u := StepUnions(sel, spans, nExpert, x.p)
			sum := 0
			for _, v := range u {
				sum += v
			}
			if sum != x.c.Accesses || len(u) != x.c.Steps {
				t.Fatalf("%s: %d unions summing to %d, the replay has %d steps and %d accesses", x.p, len(u), sum,
					x.c.Steps, x.c.Accesses)
			}
		}
	}
}

// optimalStepMisses is the fewest misses of a schedule that holds every step's whole union resident
// together, by search over the cache's contents after each step, for at most c equal experts. It is
// the event-atomic optimum Belady's MIN need not reach, since MIN may evict within a step.
func optimalStepMisses(steps []uint32, c int) int {
	cur := map[uint32]int{0: 0}
	for _, need := range steps {
		next := map[uint32]int{}
		for set, misses := range cur {
			m := misses + bits.OnesCount32(need&^set)
			kept := set &^ need
			// any subset of the others that fits beside the union may stay
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

// stepSets are the steps' unions as bit sets over key - lo, for keys of [lo, lo+width).
func stepSets(sel Selections, spans []Span, nExpert, first, layers int) []uint32 {
	var out []uint32
	for _, s := range spans {
		for l := first; l < first+layers; l++ {
			var set uint32
			for t := s.First; t < s.End; t++ {
				for _, e := range sel.IDs[(l*sel.Tokens+t)*sel.K : (l*sel.Tokens+t+1)*sel.K] {
					set |= 1 << ((l-first)*nExpert + int(e))
				}
			}
			out = append(out, set)
		}
	}
	return out
}

func maxBits(steps []uint32) int {
	m := 0
	for _, s := range steps {
		m = max(m, bits.OnesCount32(s))
	}
	return m
}

// TestPinnedOracleOptimal holds the step-pinned oracles to the search above with equal experts,
// wherever every step's union fits, and to no step evictions anywhere.
func TestPinnedOracleOptimal(t *testing.T) {
	r := rand.New(rand.NewPCG(21, 22))
	worse := 0
	for trial := range 60 {
		layers, nExpert, k := 1+r.IntN(2), 3+r.IntN(4), 1+r.IntN(3)
		k = min(k, nExpert)
		tokens := 2 + r.IntN(10)
		sel := randomSelections(r, layers, tokens, k, nExpert)
		spans := TraceOrder(tokens)
		if trial%2 == 1 { // a prefill ubatch of up to 3 tokens, then decoded tokens
			first := min(3, tokens-1)
			spans = append([]Span{{Prefill, 0, first}}, TraceOrder(tokens)[first:]...)
		}
		bytes := uniform(layers, 2)
		steps := stepSets(sel, spans, nExpert, 0, layers)
		for c := range layers*nExpert + 1 {
			o := NewPinnedOracle(sel, spans, nExpert, bytes, int64(c)*2)
			res, err := Replay(sel, spans, nExpert, bytes, o)
			if err != nil || o.Err() != nil {
				t.Fatal(err, o.Err())
			}
			if got := res.Prefill.StepEvictions + res.Decode.StepEvictions; got != 0 {
				t.Fatalf("trial %d, %d experts: %d step evictions", trial, c, got)
			}
			if c < maxBits(steps) {
				continue
			}
			got := res.Prefill.Misses + res.Decode.Misses
			if want := optimalStepMisses(steps, c); got != want {
				t.Fatalf("trial %d, %d experts: the pinned oracle misses %d, the event-atomic optimum %d", trial, c,
					got, want)
			}
			u := NewOracle(sel, spans, nExpert, bytes, int64(c)*2)
			ur, err := Replay(sel, spans, nExpert, bytes, u)
			if err != nil {
				t.Fatal(err)
			}
			free := ur.Prefill.Misses + ur.Decode.Misses
			if free > got {
				t.Fatalf("trial %d, %d experts: MIN misses %d, more than the pinned oracle's %d", trial, c, free, got)
			}
			if free < got {
				worse++
			}
		}
		for c := range nExpert + 1 {
			o := NewPinnedLayerOracle(sel, spans, nExpert, bytes, c)
			res, err := Replay(sel, spans, nExpert, bytes, o)
			if err != nil || o.Err() != nil {
				t.Fatal(err, o.Err())
			}
			if got := res.Prefill.StepEvictions + res.Decode.StepEvictions; got != 0 {
				t.Fatalf("trial %d, %d per layer: %d step evictions", trial, c, got)
			}
			want, fits := 0, true
			for l := range layers {
				ls := stepSets(sel, spans, nExpert, l, 1)
				if c < maxBits(ls) {
					fits = false
					break
				}
				want += optimalStepMisses(ls, c)
			}
			if !fits {
				continue
			}
			if got := res.Prefill.Misses + res.Decode.Misses; got != want {
				t.Fatalf("trial %d, %d per layer: the pinned oracle misses %d, the event-atomic optimum %d", trial, c,
					got, want)
			}
		}
	}
	if worse == 0 {
		t.Fatal("MIN never beat the pinned oracle: pinning changed nothing, so these traces do not test it")
	}
}

// TestPinnedOracleSizes runs the oracles on unequal experts, where neither is proven optimal: the
// replay must follow the trace, keep to the budget and hold no step evictions.
func TestPinnedOracleSizes(t *testing.T) {
	r := rand.New(rand.NewPCG(31, 32))
	for range 20 {
		layers, nExpert, k := 2+r.IntN(3), 8+r.IntN(16), 1+r.IntN(6)
		tokens := 20 + r.IntN(100)
		sel := randomSelections(r, layers, tokens, k, nExpert)
		spans, err := OnlineSpans(tokens, []Answer{{Prompt: tokens / 2, Tokens: make([]int32, tokens/4)}}, false,
			2+r.IntN(8))
		if err != nil {
			t.Fatal(err)
		}
		bytes := sizes(layers, 2)
		var total int64
		for _, b := range bytes {
			total += b
		}
		for _, c := range []int{0, 1, k, k + 1, nExpert / 2, nExpert} {
			for _, o := range []*Oracle{NewPinnedOracle(sel, spans, nExpert, bytes, int64(c)*total),
				NewPinnedLayerOracle(sel, spans, nExpert, bytes, c)} {
				res, err := Replay(sel, spans, nExpert, bytes, o)
				if err != nil || o.Err() != nil {
					t.Fatal(err, o.Err())
				}
				if got := res.Prefill.StepEvictions + res.Decode.StepEvictions; got != 0 {
					t.Fatalf("%d experts: %d step evictions", c, got)
				}
				if res.PeakBytes > int64(c)*total+slicesMax(bytes) {
					t.Fatalf("%d experts: peak %d over the budget %d", c, res.PeakBytes, int64(c)*total)
				}
			}
		}
	}
}

func slicesMax(v []int64) int64 {
	var m int64
	for _, x := range v {
		m = max(m, x)
	}
	return m
}
