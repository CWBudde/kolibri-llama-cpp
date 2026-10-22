package locality

import (
	"math"
	"math/rand/v2"
	"reflect"
	"slices"
	"testing"
)

func newSLRUs(t *testing.T, nExpert int, bytes []int64, capacity int, fraction float64) (layer, shared *SLRU) {
	t.Helper()
	var budget int64
	for _, b := range bytes {
		budget += int64(capacity) * b
	}
	layer, err := NewLayerSLRU(nExpert, bytes, capacity, fraction)
	if err != nil {
		t.Fatal(err)
	}
	shared, err = NewSLRU(nExpert, bytes, budget, fraction)
	if err != nil {
		t.Fatal(err)
	}
	return layer, shared
}

func TestPolicySLRU(t *testing.T) {
	// 4 experts, protected at most 2. 0-3 miss into probation; the hits on 0, 1, 2 promote them,
	// and the third overflows protected: its least recent, 0, is demoted to probation's front. So 4
	// evicts probation's least recent, 3, then 5 evicts 0 and 6 evicts 4, and the protected 1 and 2
	// still hit. An LRU of 4 evicts 3, 0, 1 and 2 instead and misses both.
	seq := sequence(0, 1, 2, 3, 0, 1, 2, 4, 5, 6, 1, 2)
	layer, shared := newSLRUs(t, 8, []int64{4}, 4, 0.5)
	for _, p := range []*SLRU{layer, shared} {
		e := &evictLog{Policy: p}
		res, err := Replay(seq, TraceOrder(12), 8, []int64{4}, e)
		if err != nil || res.Decode.Hits != 5 || !slices.Equal(e.evicted, []int{3, 0, 4}) {
			t.Fatalf("SLRU, per layer %v: %d hits, evicted %v, %v", p.perLayer, res.Decode.Hits, e.evicted, err)
		}
		if p.prot[0] != 8 || p.l.size[1] != 2 {
			t.Fatalf("protected holds %d bytes in %d experts, want 2 experts", p.prot[0], p.l.size[1])
		}
	}
	lru, _ := Replay(seq, TraceOrder(12), 8, []int64{4}, NewLayerLRU(1, 8, 4))
	if lru.Decode.Hits != 3 {
		t.Fatalf("LRU: %d hits", lru.Decode.Hits)
	}
	// all protected: with probation empty, the victim is protected's least recent
	layer, _ = newSLRUs(t, 4, []int64{4}, 2, 1)
	e := &evictLog{Policy: layer}
	if _, err := Replay(sequence(0, 1, 0, 1, 2), TraceOrder(5), 4, []int64{4}, e); err != nil ||
		!slices.Equal(e.evicted, []int{0}) || layer.l.size[0] != 1 || layer.l.size[1] != 1 {
		t.Fatalf("fraction 1: evicted %v, probation %d, protected %d, %v", e.evicted, layer.l.size[0],
			layer.l.size[1], err)
	}
	// a hit in protected moves it to protected's front: 0 and 1 are protected, 0 is refreshed, so
	// when 2's promotion overflows protected, 1 is demoted and the probation victim of 4 is 1, not 0
	layer, _ = newSLRUs(t, 8, []int64{4}, 4, 0.5)
	e = &evictLog{Policy: layer}
	if _, err := Replay(sequence(0, 1, 0, 1, 0, 2, 2, 3, 4), TraceOrder(9), 8, []int64{4}, e); err != nil ||
		!slices.Equal(e.evicted, []int{1}) {
		t.Fatalf("protected refresh: evicted %v, %v", e.evicted, err)
	}
	for _, f := range []float64{-0.1, 1.1, math.NaN()} {
		if _, err := NewSLRU(4, []int64{4}, 8, f); err == nil {
			t.Errorf("fraction %v accepted", f)
		}
	}
}

// TestPolicySLRUZeroIsLRU is control M40: with a protected fraction of 0 the SLRU is an LRU that
// never evicts the running step's experts, so on random traces, wherever no step needs more
// experts than the cache holds, it replays exactly as the LRU: the per-layer SLRU as LayerLRU and
// the shared one as SharedLRU, counts, evictions and peaks alike. Where a step needs more, the
// LRU evicts experts the step has used and the SLRU does not; and a protected fraction of 0.5
// differs from the LRU, so the comparison can fail.
func TestPolicySLRUZeroIsLRU(t *testing.T) {
	r := rand.New(rand.NewPCG(40, 41))
	compared, departed, protectedDiffers := 0, 0, 0
	for trial := range 40 {
		layers, nExpert, k := 1+r.IntN(4), 8+r.IntN(24), 1+r.IntN(6)
		tokens := 10 + r.IntN(200)
		sel := randomSelections(r, layers, tokens, k, nExpert)
		spans := TraceOrder(tokens)
		if trial%2 == 1 {
			var err error
			spans, err = OnlineSpans(tokens, []Answer{{Prompt: tokens / 2, Tokens: make([]int32, tokens/4)}}, false,
				1+r.IntN(6))
			if err != nil {
				t.Fatal(err)
			}
		}
		bytes := uniform(layers, 3)
		if trial%4 == 3 {
			bytes = sizes(layers, 2)
		}
		// the most experts a step needs, in any layer
		most := 0
		for _, s := range spans {
			for l := range layers {
				var u []int32
				for tok := s.First; tok < s.End; tok++ {
					for _, e := range sel.IDs[(l*tokens+tok)*k : (l*tokens+tok+1)*k] {
						if !slices.Contains(u, e) {
							u = append(u, e)
						}
					}
				}
				most = max(most, len(u))
			}
		}
		for _, c := range []int{1, k, most, most + 1, nExpert / 2, nExpert} {
			layer, shared := newSLRUs(t, nExpert, bytes, c, 0)
			var budget int64
			for _, b := range bytes {
				budget += int64(c) * b
			}
			for _, pair := range []struct {
				name     string
				slru     Policy
				lru      Policy
				fits     bool
				halfSLRU Policy
			}{
				{"per layer", layer, NewLayerLRUSizes(nExpert, slices.Repeat([]int{c}, layers)), c >= most,
					func() Policy { p, _ := NewLayerSLRU(nExpert, bytes, c, 0.5); return p }()},
				// a step is one layer: the shared cache holds its experts if it holds c*layers >= most
				// of the largest size
				{"shared", shared, NewSharedLRU(nExpert, bytes, budget),
					int64(most)*slices.Max(bytes) <= budget,
					func() Policy { p, _ := NewSLRU(nExpert, bytes, budget, 0.5); return p }()},
			} {
				got, err := Replay(sel, spans, nExpert, bytes, pair.slru)
				if err != nil {
					t.Fatal(err)
				}
				want, err := Replay(sel, spans, nExpert, bytes, pair.lru)
				if err != nil {
					t.Fatal(err)
				}
				if got.Prefill.StepEvictions+got.Decode.StepEvictions != 0 {
					t.Fatalf("trial %d, %s at %d: the SLRU evicts the running step's experts: %+v", trial, pair.name,
						c, got)
				}
				if pair.fits {
					compared++
					if !reflect.DeepEqual(got, want) {
						t.Fatalf("trial %d, %s at %d: SLRU(0) %+v, LRU %+v", trial, pair.name, c, got, want)
					}
					half, err := Replay(sel, spans, nExpert, bytes, pair.halfSLRU)
					if err != nil {
						t.Fatal(err)
					}
					if half.Prefill.Hits+half.Decode.Hits != want.Prefill.Hits+want.Decode.Hits {
						protectedDiffers++
					}
				} else if want.Prefill.StepEvictions+want.Decode.StepEvictions > 0 {
					departed++
				}
			}
		}
	}
	if compared < 100 || departed == 0 || protectedDiffers == 0 {
		t.Fatalf("%d comparisons, %d where the LRU evicts the running step's experts, %d where fraction 0.5 "+
			"hits otherwise", compared, departed, protectedDiffers)
	}
}

func TestPolicyHotness(t *testing.T) {
	// two experts fit. 0 is used at tokens 0-2, 1 at 3; at token 4, 2 evicts one of them. Without
	// decay 0 scores 3 against 1's 1 and stays. Halved every token, 0 scores 1/16+1/8+1/4 = 0.4375
	// at token 4 against 1's 0.5: 0 goes.
	seq := sequence(0, 0, 0, 1, 2)
	for _, c := range []struct {
		halfLife float64
		evicted  int
	}{{1e9, 1}, {1, 0}} {
		h, err := NewHotness(1, 3, 2, c.halfLife)
		if err != nil {
			t.Fatal(err)
		}
		e := &evictLog{Policy: h}
		res, err := Replay(seq, TraceOrder(5), 3, []int64{4}, e)
		if err != nil || !slices.Equal(e.evicted, []int{c.evicted}) || res.Decode.Hits != 2 {
			t.Fatalf("half-life %v: evicted %v after %d hits, want [%d] after 2, %v", c.halfLife, e.evicted,
				res.Decode.Hits, c.evicted, err)
		}
		if c.halfLife == 1 {
			if s := h.Score(1, 4); math.Abs(s-0.5) > 1e-12 {
				t.Errorf("score of 1 at token 4: %v, want 0.5", s)
			}
			if s := h.Score(2, 4); s != 1 {
				t.Errorf("score of 2 at token 4: %v, want 1", s)
			}
		}
	}
	// 0 at token 0 scores 0.4375 at token 4 with the decay above; as of token 3 (one ubatch's first
	// token) it scored 0.875, as of token 5 0.21875
	h, _ := NewHotness(1, 3, 2, 1)
	if _, err := Replay(sequence(0, 0, 0), TraceOrder(3), 3, []int64{4}, h); err != nil {
		t.Fatal(err)
	}
	if s := h.Score(0, 3); math.Abs(s-0.875) > 1e-12 {
		t.Errorf("score of 0 at token 3: %v", s)
	}
	// equal scores, the least recent goes: one ubatch uses both at the same token, the first
	// selected is the least recent
	for _, c := range []struct {
		ids     []int32
		evicted int
	}{{[]int32{0, 1, 2}, 0}, {[]int32{1, 0, 2}, 1}} {
		h, _ := NewHotness(1, 3, 2, 8)
		e := &evictLog{Policy: h}
		spans := []Span{{Prefill, 0, 2}, {Decode, 2, 3}}
		if _, err := Replay(Selections{Layers: 1, Tokens: 3, K: 1, IDs: c.ids}, spans, 3, []int64{4}, e); err != nil ||
			!slices.Equal(e.evicted, []int{c.evicted}) {
			t.Errorf("ties %v: evicted %v, want [%d], %v", c.ids, e.evicted, c.evicted, err)
		}
	}
	// per layer: layer 1's experts never evict layer 0's
	sel := Selections{Layers: 2, Tokens: 3, K: 1, IDs: []int32{0, 0, 0, 1, 2, 1}}
	h, _ = NewHotness(2, 3, 1, 8)
	res, err := Replay(sel, TraceOrder(3), 3, []int64{4, 4}, h)
	if err != nil || res.Decode.Hits != 2 {
		t.Fatalf("per layer: %d hits, want layer 0's 2, %v", res.Decode.Hits, err)
	}
	for _, hl := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		if _, err := NewHotness(1, 3, 2, hl); err == nil {
			t.Errorf("half-life %v accepted", hl)
		}
	}
}

func TestStepEvictionsLFU(t *testing.T) {
	// one layer, two experts fit, tokens {0,1} {0,1} {2,3}. LFU loads 2 with its old count 0,
	// evicting 0 (counts 2 and 2, 0 less recent); 2 is used and counts 1, so 3 evicts 2, which the
	// running step has used: one step eviction. Pinned, 3 evicts 1 instead.
	sel := Selections{Layers: 1, Tokens: 3, K: 2, IDs: []int32{0, 1, 0, 1, 2, 3}}
	for _, c := range []struct {
		p       *LFU
		evicted []int
		step    int
	}{{NewLFU(4, []int64{4}, 8, 1000), []int{0, 2}, 1}, {NewPinnedLFU(4, []int64{4}, 8, 1000), []int{0, 1}, 0}} {
		e := &evictLog{Policy: c.p}
		res, err := Replay(sel, TraceOrder(3), 4, []int64{4}, e)
		if err != nil || !slices.Equal(e.evicted, c.evicted) || res.Decode.StepEvictions != c.step ||
			res.Decode.Evictions != 2 {
			t.Fatalf("pinned %v: evicted %v, %d step evictions, want %v and %d; %+v, %v", c.p.pinned, e.evicted,
				res.Decode.StepEvictions, c.evicted, c.step, res.Decode, err)
		}
	}
	// a step that needs more than the cache holds: the pinned LFU uses the third expert once
	sel = Selections{Layers: 1, Tokens: 1, K: 3, IDs: []int32{0, 1, 2}}
	res, err := Replay(sel, TraceOrder(1), 4, []int64{4}, NewPinnedLFU(4, []int64{4}, 8, 1000))
	if err != nil || res.Decode.Evictions != 0 || res.Decode.StepEvictions != 0 || res.PeakBytes != 8 {
		t.Fatalf("an oversized step: %+v, %v", res, err)
	}
	res, err = Replay(sel, TraceOrder(1), 4, []int64{4}, NewLFU(4, []int64{4}, 8, 1000))
	if err != nil || res.Decode.StepEvictions != 1 {
		t.Fatalf("an oversized step, unpinned: %+v, %v", res, err)
	}
}

// stepWatch wraps a policy and counts its evictions of keys the running step has used.
type stepWatch struct {
	Policy
	step       int
	used       map[int]bool
	stepEvicts int
}

func (w *stepWatch) Prefetch(s StepInfo) []int {
	if s.Index != w.step || w.used == nil {
		w.step, w.used = s.Index, map[int]bool{}
	}
	return w.Policy.Prefetch(s)
}

func (w *stepWatch) Load(k int) []int {
	out := w.Policy.Load(k)
	for _, v := range out {
		if v != k && w.used[v] {
			w.stepEvicts++
		}
	}
	return out
}

func (w *stepWatch) Use(s StepInfo, k int) {
	w.used[k] = true
	w.Policy.Use(s, k)
}

// TestStepPinned: on random traces with ubatches and small caches, none of the step-pinned
// policies evicts an expert its running step has used, and the replay's StepEvictions count
// exactly such evictions for the policies that do.
func TestStepPinned(t *testing.T) {
	r := rand.New(rand.NewPCG(42, 43))
	unpinnedSeen := 0
	for trial := range 30 {
		layers, nExpert, k := 1+r.IntN(4), 8+r.IntN(24), 1+r.IntN(6)
		tokens := 20 + r.IntN(200)
		sel := randomSelections(r, layers, tokens, k, nExpert)
		spans, err := OnlineSpans(tokens, []Answer{{Prompt: tokens / 2, Tokens: make([]int32, tokens/4)}}, false,
			2+r.IntN(8))
		if err != nil {
			t.Fatal(err)
		}
		bytes := uniform(layers, 3)
		if trial%3 == 2 {
			bytes = sizes(layers, 2)
		}
		for _, c := range []int{0, 1, 2, k, nExpert / 2} {
			var budget int64
			for _, b := range bytes {
				budget += int64(c) * b
			}
			layer, shared := newSLRUs(t, nExpert, bytes, c, 0.5)
			layer1, _ := newSLRUs(t, nExpert, bytes, c, 1)
			_, shared0 := newSLRUs(t, nExpert, bytes, c, 0)
			hot, _ := NewHotness(layers, nExpert, c, 1+float64(r.IntN(64)))
			pinned := []Policy{layer, shared, layer1, shared0, hot, NewPinnedLFU(nExpert, bytes, budget, 1+r.IntN(32))}
			arc, err := NewARC(nExpert, uniform(layers, 3), int64(c)*3*int64(layers))
			if err != nil {
				t.Fatal(err)
			}
			others := []Policy{NewLayerLRU(layers, nExpert, c), NewSharedLRU(nExpert, bytes, budget),
				NewLFU(nExpert, bytes, budget, 1+r.IntN(32))}
			if trial%3 != 2 {
				others = append(others, arc)
				pinnedARC, err := NewPinnedARC(nExpert, bytes, budget)
				if err != nil {
					t.Fatal(err)
				}
				pinned = append(pinned, pinnedARC)
			}
			for i, p := range append(pinned, others...) {
				w := &stepWatch{Policy: p}
				res, err := Replay(sel, spans, nExpert, bytes, w)
				if err != nil {
					t.Fatal(err)
				}
				got := res.Prefill.StepEvictions + res.Decode.StepEvictions
				if got != w.stepEvicts {
					t.Fatalf("trial %d, policy %d at %d: %d step evictions counted, %d seen", trial, i, c, got,
						w.stepEvicts)
				}
				if i < len(pinned) && got != 0 {
					t.Fatalf("trial %d, pinned policy %d at %d: %d step evictions", trial, i, c, got)
				}
				if i >= len(pinned) {
					unpinnedSeen += got
				}
			}
		}
	}
	if unpinnedSeen == 0 {
		t.Fatal("no unpinned policy ever evicted its running step's experts: the count is untested")
	}
}

// TestSLRUProtectedCap: protected never holds more than its fraction of the budget.
func TestSLRUProtectedCap(t *testing.T) {
	r := rand.New(rand.NewPCG(44, 45))
	for range 20 {
		layers, nExpert, k := 1+r.IntN(3), 8+r.IntN(16), 1+r.IntN(4)
		tokens := 20 + r.IntN(100)
		sel := randomSelections(r, layers, tokens, k, nExpert)
		bytes := sizes(layers, 2)
		for _, f := range []float64{0, 0.25, 0.5, 0.75, 1} {
			layer, shared := newSLRUs(t, nExpert, bytes, 1+r.IntN(nExpert), f)
			for _, p := range []*SLRU{layer, shared} {
				w := &capWatch{SLRU: p, t: t}
				if _, err := Replay(sel, TraceOrder(tokens), nExpert, bytes, w); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
}

// capWatch checks the SLRU's segments after every call.
type capWatch struct {
	*SLRU
	t *testing.T
}

func (w *capWatch) check() {
	for g := range w.budget {
		if w.prot[g] > w.protCap[g] || w.cur[g] > w.budget[g] {
			w.t.Fatalf("group %d: %d bytes protected of %d, %d resident of %d", g, w.prot[g], w.protCap[g], w.cur[g],
				w.budget[g])
		}
	}
}

func (w *capWatch) Load(k int) []int      { out := w.SLRU.Load(k); w.check(); return out }
func (w *capWatch) Use(s StepInfo, k int) { w.SLRU.Use(s, k); w.check() }
