package locality

import (
	"math/bits"
	"math/rand/v2"
	"reflect"
	"slices"
	"testing"
)

// evictLog wraps a policy and records the keys it evicts, those it does not keep left out.
type evictLog struct {
	Policy
	evicted []int
}

func (e *evictLog) Load(k int) []int {
	out := e.Policy.Load(k)
	for _, v := range out {
		if v != k {
			e.evicted = append(e.evicted, v)
		}
	}
	return out
}

// useLog is NoCache, recording every access in replay order.
type useLog struct {
	NoCache
	keys []int
}

func (u *useLog) Use(_ StepInfo, k int) { u.keys = append(u.keys, k) }

func accesses(t *testing.T, sel Selections, spans []Span, nExpert int) []int {
	u := &useLog{}
	if _, err := Replay(sel, spans, nExpert, make([]int64, sel.Layers), u); err != nil {
		t.Fatal(err)
	}
	return u.keys
}

// sequence is a one-layer trace of one expert per token.
func sequence(keys ...int32) Selections {
	return Selections{Layers: 1, Tokens: len(keys), K: 1, IDs: keys}
}

func uniform(layers int, b int64) []int64 {
	out := make([]int64, layers)
	for l := range out {
		out[l] = b
	}
	return out
}

func TestPolicyNoCache(t *testing.T) {
	sel := randomSelections(rand.New(rand.NewPCG(1, 2)), 3, 50, 4, 16)
	res, err := Replay(sel, TraceOrder(50), 16, sizes(3, 5), &NoCache{})
	if err != nil {
		t.Fatal(err)
	}
	if d := res.Decode; d.Hits != 0 || d.Misses != 3*50*4 || d.Evictions != 0 || res.PeakBytes != 0 {
		t.Fatalf("NoCache: %+v", res)
	}
}

func TestPolicySharedLRUMatchesHits(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	for trial := range 30 {
		layers, nExpert, k := 1+r.IntN(4), 8+r.IntN(24), 1+r.IntN(6)
		sel := randomSelections(r, layers, 1+r.IntN(200), k, nExpert)
		keys, _ := SharedKeys(sel, nExpert)
		dist := StackDistances(keys, layers*nExpert)
		for _, c := range []int{0, 1, k, layers * nExpert / 3, layers * nExpert} {
			res, err := Replay(sel, TraceOrder(sel.Tokens), nExpert, uniform(layers, 7),
				NewSharedLRU(nExpert, uniform(layers, 7), int64(c)*7))
			if err != nil {
				t.Fatal(err)
			}
			if want := Hits(dist, []int{c})[0]; res.Decode.Hits != want || res.PeakBytes > int64(c)*7 {
				t.Fatalf("trial %d, %d experts: %d hits, want %d; peak %d bytes", trial, c, res.Decode.Hits, want,
					res.PeakBytes)
			}
		}
	}
	// by bytes: a budget of 25 holds one expert of 20 bytes and one of 5, not two of 20
	sel := Selections{Layers: 2, Tokens: 2, K: 1, IDs: []int32{0, 0, 1, 1}} // layer 0 {0} {0}, layer 1 {1} {1}
	res, err := Replay(sel, TraceOrder(2), 2, []int64{20, 5}, NewSharedLRU(2, []int64{20, 5}, 25))
	if err != nil || res.Decode.Hits != 2 || res.PeakBytes != 25 {
		t.Fatalf("20+5 in 25 bytes: %+v, %v", res, err)
	}
	res, err = Replay(sel, TraceOrder(2), 2, []int64{20, 20}, NewSharedLRU(2, []int64{20, 20}, 25))
	if err != nil || res.Decode.Hits != 0 || res.Decode.Evictions != 3 {
		t.Fatalf("20+20 in 25 bytes: %+v, %v", res, err)
	}
}

func TestPolicyLFU(t *testing.T) {
	// two experts fit. Expert 0 is used at tokens 0-3, 1 at 4 and 5; at token 6 expert 2 evicts one
	// of them. Without aging 0 has 4 uses against 1's 2 and stays. Halving every 2 tokens leaves 0 at
	// 1 use before token 4 and at 0 before token 6, 1 at 1: 0 goes.
	sel := sequence(0, 0, 0, 0, 1, 1, 2)
	for _, c := range []struct {
		halfLife int
		evicted  int
	}{{1000, 1}, {2, 0}} {
		p := &evictLog{Policy: NewLFU(3, []int64{4}, 8, c.halfLife)}
		res, err := Replay(sel, TraceOrder(7), 3, []int64{4}, p)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(p.evicted, []int{c.evicted}) || res.Decode.Hits != 4 {
			t.Errorf("half-life %d: evicted %v after %d hits, want [%d] after 4", c.halfLife, p.evicted,
				res.Decode.Hits, c.evicted)
		}
	}
	// equal counts: the least recent goes
	p := &evictLog{Policy: NewLFU(3, []int64{4}, 8, 1000)}
	if _, err := Replay(sequence(0, 1, 2), TraceOrder(3), 3, []int64{4}, p); err != nil ||
		!slices.Equal(p.evicted, []int{0}) {
		t.Errorf("ties: evicted %v, %v", p.evicted, err)
	}
}

func TestPolicyARC(t *testing.T) {
	// c = 2. 0,0 puts 0 in T2; the scan 1,2 replaces only T1's 1, so the final 0 hits where an LRU
	// of 2 misses.
	arc, err := NewARC(3, []int64{4}, 8)
	if err != nil {
		t.Fatal(err)
	}
	p := &evictLog{Policy: arc}
	res, err := Replay(sequence(0, 0, 1, 2, 0), TraceOrder(5), 3, []int64{4}, p)
	if err != nil || res.Decode.Hits != 2 || !slices.Equal(p.evicted, []int{1}) {
		t.Fatalf("ARC on 0,0,1,2,0: %d hits, evicted %v, %v", res.Decode.Hits, p.evicted, err)
	}
	lru, _ := Replay(sequence(0, 0, 1, 2, 0), TraceOrder(5), 3, []int64{4}, NewSharedLRU(3, []int64{4}, 8))
	if lru.Decode.Hits != 1 {
		t.Fatalf("LRU on 0,0,1,2,0: %d hits", lru.Decode.Hits)
	}
	// 0,1,0,2,1,0: 2 replaces T1's 1 (p 0); 1 then hits ghost B1, p grows to 1 and T2's 0 goes; 0
	// hits ghost B2, p falls to 0 and T1's 2 goes.
	arc, _ = NewARC(3, []int64{4}, 8)
	p = &evictLog{Policy: arc}
	res, err = Replay(sequence(0, 1, 0, 2, 1, 0), TraceOrder(6), 3, []int64{4}, p)
	if err != nil || res.Decode.Hits != 1 || !slices.Equal(p.evicted, []int{1, 0, 2}) || arc.p != 0 {
		t.Fatalf("ARC on 0,1,0,2,1,0: %d hits, evicted %v, p %v, %v", res.Decode.Hits, p.evicted, arc.p, err)
	}
	if _, err := NewARC(3, []int64{4, 5}, 8); err == nil {
		t.Error("unequal expert sizes accepted")
	}
	// against the paper's pseudocode, on random traces
	r := rand.New(rand.NewPCG(13, 14))
	for trial := range 40 {
		layers, nExpert, k := 1+r.IntN(3), 6+r.IntN(20), 1+r.IntN(4)
		sel := randomSelections(r, layers, 1+r.IntN(150), k, nExpert)
		keys := accesses(t, sel, TraceOrder(sel.Tokens), nExpert)
		for _, c := range []int{0, 1, 3, layers * nExpert / 2} {
			arc, _ := NewARC(nExpert, uniform(layers, 3), int64(c)*3)
			res, err := Replay(sel, TraceOrder(sel.Tokens), nExpert, uniform(layers, 3), arc)
			if err != nil {
				t.Fatal(err)
			}
			if want := referenceARC(keys, c); res.Decode.Hits != want || res.PeakBytes > int64(c)*3 {
				t.Fatalf("trial %d, c %d: %d hits, the pseudocode %d", trial, c, res.Decode.Hits, want)
			}
		}
	}
}

// referenceARC counts the hits of Megiddo and Modha's ARC(c), written from the paper's figure 4
// with plain slices, most recent first.
func referenceARC(keys []int, c int) int {
	var t1, t2, b1, b2 []int
	p := 0.0
	del := func(s []int, k int) []int { return slices.DeleteFunc(s, func(x int) bool { return x == k }) }
	popLRU := func(s *[]int) int { k := (*s)[len(*s)-1]; *s = (*s)[:len(*s)-1]; return k }
	replace := func(inB2 bool) {
		if len(t1) >= 1 && ((inB2 && float64(len(t1)) == p) || float64(len(t1)) > p) {
			b1 = append([]int{popLRU(&t1)}, b1...)
		} else {
			b2 = append([]int{popLRU(&t2)}, b2...)
		}
	}
	hits := 0
	for _, x := range keys {
		switch {
		case c == 0:
		case slices.Contains(t1, x) || slices.Contains(t2, x):
			hits++
			t1, t2 = del(t1, x), del(t2, x)
			t2 = append([]int{x}, t2...)
		case slices.Contains(b1, x):
			d := 1.0
			if len(b1) < len(b2) {
				d = float64(len(b2)) / float64(len(b1))
			}
			p = min(p+d, float64(c))
			replace(false)
			b1 = del(b1, x)
			t2 = append([]int{x}, t2...)
		case slices.Contains(b2, x):
			d := 1.0
			if len(b2) < len(b1) {
				d = float64(len(b1)) / float64(len(b2))
			}
			p = max(p-d, 0)
			replace(true)
			b2 = del(b2, x)
			t2 = append([]int{x}, t2...)
		default:
			if len(t1)+len(b1) == c {
				if len(t1) < c {
					popLRU(&b1)
					replace(false)
				} else {
					popLRU(&t1)
				}
			} else if total := len(t1) + len(b1) + len(t2) + len(b2); total >= c {
				if total == 2*c {
					popLRU(&b2)
				}
				replace(false)
			}
			t1 = append([]int{x}, t1...)
		}
	}
	return hits
}

// optimalMisses is the fewest misses of any cache of c keys on the access sequence, found by trying
// every choice at every miss: keep nothing new, take a free slot, or replace any resident key.
func optimalMisses(seq []int, c int) int {
	memo := map[[2]int]int{}
	var f func(i int, set uint32) int
	f = func(i int, set uint32) int {
		if i == len(seq) {
			return 0
		}
		if v, ok := memo[[2]int{i, int(set)}]; ok {
			return v
		}
		k := uint32(1) << seq[i]
		var best int
		if set&k != 0 {
			best = f(i+1, set)
		} else {
			best = 1 + f(i+1, set)
			if bits.OnesCount32(set) < c {
				best = min(best, 1+f(i+1, set|k))
			} else {
				for s := set; s != 0; s &= s - 1 {
					best = min(best, 1+f(i+1, set&^(s&-s)|k))
				}
			}
		}
		memo[[2]int{i, int(set)}] = best
		return best
	}
	return f(0, 0)
}

func TestPolicyOracleOptimal(t *testing.T) {
	r := rand.New(rand.NewPCG(15, 16))
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
		keys := accesses(t, sel, spans, nExpert)
		bytes := uniform(layers, 2)
		for c := range layers*nExpert + 1 {
			o := NewOracle(sel, spans, nExpert, bytes, int64(c)*2)
			res, err := Replay(sel, spans, nExpert, bytes, o)
			if err != nil || o.Err() != nil {
				t.Fatal(err, o.Err())
			}
			got := res.Prefill.Misses + res.Decode.Misses
			if want := optimalMisses(keys, c); got != want {
				t.Fatalf("trial %d, %d experts: the oracle misses %d, the optimum %d", trial, c, got, want)
			}
		}
		for c := range nExpert + 1 {
			o := NewLayerOracle(sel, spans, nExpert, bytes, c)
			res, err := Replay(sel, spans, nExpert, bytes, o)
			if err != nil || o.Err() != nil {
				t.Fatal(err, o.Err())
			}
			want := 0
			for l := range layers {
				var layerKeys []int
				for _, x := range keys {
					if x/nExpert == l {
						layerKeys = append(layerKeys, x%nExpert)
					}
				}
				want += optimalMisses(layerKeys, c)
			}
			if got := res.Prefill.Misses + res.Decode.Misses; got != want {
				t.Fatalf("trial %d, %d per layer: the oracle misses %d, the optimum %d", trial, c, got, want)
			}
		}
	}
	// replayed with other spans than it was built from, it says so
	sel := sequence(0, 1, 0)
	o := NewOracle(sel, []Span{{Prefill, 0, 3}}, 2, []int64{1}, 1)
	if _, err := Replay(sel, TraceOrder(3), 2, []int64{1}, o); err != nil || o.Err() == nil {
		t.Errorf("an oracle replayed in another order: %v", o.Err())
	}
}

// policies are every baseline at c experts per layer, the oracles last.
func policies(sel Selections, spans []Span, nExpert int, bytes []int64, c int) []Policy {
	var budget int64
	for _, b := range bytes {
		budget += int64(c) * b
	}
	arc, _ := NewARC(nExpert, bytes, budget)
	return []Policy{&NoCache{}, NewLayerLRU(sel.Layers, nExpert, c), NewSharedLRU(nExpert, bytes, budget),
		NewLFU(nExpert, bytes, budget, 16), arc, NewLayerOracle(sel, spans, nExpert, bytes, c),
		NewOracle(sel, spans, nExpert, bytes, budget)}
}

func TestPolicyOracleBounds(t *testing.T) {
	r := rand.New(rand.NewPCG(17, 18))
	for trial := range 20 {
		layers, nExpert := 1+r.IntN(4), 8+r.IntN(24)
		tokens := 20 + r.IntN(200)
		sel := randomSelections(r, layers, tokens, 1+r.IntN(6), nExpert)
		spans, err := OnlineSpans(tokens, []Answer{{Prompt: tokens / 2, Tokens: make([]int32, tokens/4)}}, false, 8)
		if err != nil {
			t.Fatal(err)
		}
		bytes := uniform(layers, 3)
		for _, c := range []int{0, 1, 4, nExpert / 2, nExpert} {
			loads := []int{}
			for _, p := range policies(sel, spans, nExpert, bytes, c) {
				res, err := Replay(sel, spans, nExpert, bytes, p)
				if err != nil {
					t.Fatal(err)
				}
				loads = append(loads, res.Prefill.Misses+res.Decode.Misses)
			}
			n := len(loads)
			if layerOracle, oracle := loads[n-2], loads[n-1]; oracle > slices.Min(loads) || layerOracle > loads[1] ||
				oracle > layerOracle {
				t.Fatalf("trial %d, %d per layer: misses %v (no cache, layer LRU, shared LRU, LFU, ARC, layer "+
					"oracle, oracle)", trial, c, loads)
			}
		}
	}
}

func TestPolicyDeterministic(t *testing.T) {
	r := rand.New(rand.NewPCG(19, 20))
	sel := randomSelections(r, 3, 200, 4, 24)
	spans, err := OnlineSpans(200, []Answer{{Prompt: 80, Tokens: make([]int32, 40)}}, false, 16)
	if err != nil {
		t.Fatal(err)
	}
	a, b := policies(sel, spans, 24, uniform(3, 5), 6), policies(sel, spans, 24, uniform(3, 5), 6)
	for i := range a {
		x, err := Replay(sel, spans, 24, uniform(3, 5), a[i])
		if err != nil {
			t.Fatal(err)
		}
		y, err := Replay(sel, spans, 24, uniform(3, 5), b[i])
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(x, y) {
			t.Fatalf("policy %d: two replays %+v and %+v", i, x, y)
		}
	}
}
