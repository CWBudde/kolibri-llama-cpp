package locality

import (
	"math/rand/v2"
	"reflect"
	"slices"
	"testing"
)

// randomSelections draws K distinct experts per token and layer, skewed so that small caches hit.
func randomSelections(r *rand.Rand, layers, tokens, k, nExpert int) Selections {
	s := Selections{Layers: layers, Tokens: tokens, K: k, IDs: make([]int32, layers*tokens*k)}
	for i := 0; i < len(s.IDs); i += k {
		for j := 0; j < k; {
			e := int32(min(r.IntN(nExpert), r.IntN(nExpert)))
			if !slices.Contains(s.IDs[i:i+j], e) {
				s.IDs[i+j] = e
				j++
			}
		}
	}
	return s
}

// sizes gives each layer its own expert size, so that bytes are attributed to the right layer.
func sizes(layers int, b int64) []int64 {
	out := make([]int64, layers)
	for l := range out {
		out[l] = b + int64(l)
	}
	return out
}

func TestReplayLayerLRUMatchesHits(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for trial := range 30 {
		layers, nExpert, k := 1+r.IntN(4), 8+r.IntN(24), 1+r.IntN(6)
		sel := randomSelections(r, layers, 1+r.IntN(200), k, nExpert)
		bytes := sizes(layers, 100)
		for _, c := range []int{0, 1, k, nExpert / 2, nExpert} {
			res, err := Replay(sel, TraceOrder(sel.Tokens), nExpert, bytes, NewLayerLRU(layers, nExpert, c))
			if err != nil {
				t.Fatal(err)
			}
			hits := 0
			var missBytes int64
			for l := range layers {
				dist := StackDistances(Flatten(sel.Layer(l)), nExpert)
				h := Hits(dist, []int{c})[0]
				hits += h
				missBytes += int64(len(dist)-h) * bytes[l]
			}
			d := res.Decode
			if d.Hits != hits || d.MissBytes != missBytes || d.Hits+d.Misses != d.Accesses ||
				d.Accesses != layers*sel.Tokens*k || res.Prefill != (Counts{}) {
				t.Fatalf("trial %d, capacity %d: %+v, want %d hits and %d miss bytes of %d accesses", trial, c, res,
					hits, missBytes, layers*sel.Tokens*k)
			}
			if c == 0 && (res.PeakBytes != 0 || d.Evictions != 0) {
				t.Fatalf("trial %d: an empty cache holds %d bytes after %d evictions", trial, res.PeakBytes, d.Evictions)
			}
		}
	}
}

func TestReplayPhases(t *testing.T) {
	// One layer, an LRU of 3 experts; tokens 0-2 a prefill ubatch, 3 and 4 decoded, 5 a prefill
	// again. The ubatch needs its union {0,1,2} once: 3 misses, the LRU then 2,1,0 (most recent
	// first). Token 3 {3,0}: both miss, evicting 0 and 1 (0,3,2). Token 4 {3,4}: 3 hits, 4 misses
	// and evicts 2 (4,3,0). Token 5 {4,1}: 4 hits, 1 misses and evicts 0.
	sel := Selections{Layers: 1, Tokens: 6, K: 2, IDs: []int32{0, 1, 1, 2, 0, 2, 3, 0, 3, 4, 4, 1}}
	spans := []Span{{Prefill, 0, 3}, {Decode, 3, 4}, {Decode, 4, 5}, {Prefill, 5, 6}}
	res, err := Replay(sel, spans, 5, []int64{10}, NewLayerLRU(1, 5, 3))
	if err != nil {
		t.Fatal(err)
	}
	want := Result{
		Prefill:   Counts{Tokens: 4, Steps: 2, Accesses: 5, Hits: 1, Misses: 4, MissBytes: 40, Evictions: 1},
		Decode:    Counts{Tokens: 2, Steps: 2, Accesses: 4, Hits: 1, Misses: 3, MissBytes: 30, Evictions: 3},
		PeakBytes: 30,
	}
	if res != want {
		t.Fatalf("Replay = %+v\nwant     %+v", res, want)
	}
}

// recorder is a policy that keeps, for every step, the keys revealed to it before the step's
// Prefetch; it prefetches what prefetch returns and evicts what evict returns.
type recorder struct {
	seen     map[int]bool
	atFetch  [][]int
	prefetch func(StepInfo) []int
	evict    func(k int) []int
}

func (p *recorder) Prefetch(s StepInfo) []int {
	known := []int{}
	for k := range p.seen {
		known = append(known, k)
	}
	slices.Sort(known)
	p.atFetch = append(p.atFetch, known)
	if p.prefetch != nil {
		return p.prefetch(s)
	}
	return nil
}

func (p *recorder) Load(k int) []int {
	if p.evict != nil {
		return p.evict(k)
	}
	return nil
}

func (p *recorder) Use(_ StepInfo, k int) { p.seen[k] = true }

func TestReplayCausal(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	sel := randomSelections(r, 3, 40, 4, 16)
	answers := []Answer{{Prompt: 9, Tokens: make([]int32, 6)}, {Prompt: 30, Tokens: make([]int32, 3), EOS: true}}
	spans, err := OnlineSpans(40, answers, true, 4) // ends at token 33
	if err != nil {
		t.Fatal(err)
	}
	p := &recorder{seen: map[int]bool{}}
	if _, err := Replay(sel, spans, 16, sizes(3, 1), p); err != nil {
		t.Fatal(err)
	}
	// Before each step's Prefetch the policy knows exactly the experts of the steps before it: all
	// layers of earlier spans and the earlier layers of the step's own span.
	known := map[int]bool{}
	step := 0
	for _, s := range spans {
		for l := range sel.Layers {
			want := []int{}
			for k := range known {
				want = append(want, k)
			}
			slices.Sort(want)
			if step >= len(p.atFetch) || !slices.Equal(p.atFetch[step], want) {
				t.Fatalf("step %d (%v, layer %d): the policy knew other keys before Prefetch than %v", step, s, l, want)
			}
			for tok := s.First; tok < s.End; tok++ {
				for _, e := range sel.IDs[(l*sel.Tokens+tok)*sel.K : (l*sel.Tokens+tok+1)*sel.K] {
					known[l*16+int(e)] = true
				}
			}
			step++
		}
	}
	if step != len(p.atFetch) {
		t.Fatalf("%d steps, %d Prefetch calls", step, len(p.atFetch))
	}
}

func TestReplayPrefetch(t *testing.T) {
	// One layer, two decoded tokens {0,1} and {2,0}; before token 1 the policy prefetches 2, which
	// token 1 uses, and 3, which nobody uses; 0 is resident already and not prefetched again.
	sel := Selections{Layers: 1, Tokens: 2, K: 2, IDs: []int32{0, 1, 2, 0}}
	p := &recorder{seen: map[int]bool{}, prefetch: func(s StepInfo) []int {
		if s.First == 1 {
			return []int{2, 3, 0}
		}
		return nil
	}}
	res, err := Replay(sel, TraceOrder(2), 4, []int64{7}, p)
	if err != nil {
		t.Fatal(err)
	}
	want := Result{Decode: Counts{Tokens: 2, Steps: 2, Accesses: 4, Hits: 2, Misses: 2, MissBytes: 14,
		Prefetches: 2, PrefetchBytes: 14, UsefulPrefetches: 1, UnusedPrefetchBytes: 7}, PeakBytes: 28}
	if res != want {
		t.Fatalf("Replay = %+v\nwant     %+v", res, want)
	}
	// A key prefetched in prefill and evicted before use is unused bytes of the prefill.
	p = &recorder{seen: map[int]bool{}, prefetch: func(s StepInfo) []int {
		if s.First == 0 {
			return []int{3}
		}
		return nil
	}, evict: func(k int) []int {
		if k == 0 {
			return []int{3}
		}
		return nil
	}}
	res, err = Replay(sel, []Span{{Prefill, 0, 1}, {Decode, 1, 2}}, 4, []int64{7}, p)
	if err != nil {
		t.Fatal(err)
	}
	if res.Prefill.UnusedPrefetchBytes != 7 || res.Prefill.Evictions != 1 || res.Decode.UnusedPrefetchBytes != 0 {
		t.Fatalf("Replay = %+v", res)
	}
}

func TestReplayDeterministic(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	sel := randomSelections(r, 4, 300, 6, 32)
	answers := []Answer{{Prompt: 100, Tokens: make([]int32, 50)}, {Prompt: 200, Tokens: make([]int32, 60)}}
	spans, err := OnlineSpans(300, answers, false, 32)
	if err != nil {
		t.Fatal(err)
	}
	a, err := Replay(sel, spans, 32, sizes(4, 9), NewLayerLRU(4, 32, 12))
	if err != nil {
		t.Fatal(err)
	}
	b, err := Replay(sel, spans, 32, sizes(4, 9), NewLayerLRU(4, 32, 12))
	if err != nil {
		t.Fatal(err)
	}
	// both answers stopped at the limit: 49 and 59 tokens decoded
	if !reflect.DeepEqual(a, b) || a.Prefill.Tokens != 300-49-59 || a.Decode.Tokens != 49+59 {
		t.Fatalf("two replays: %+v and %+v", a, b)
	}
}

func TestReplayRejects(t *testing.T) {
	sel := Selections{Layers: 1, Tokens: 3, K: 1, IDs: []int32{0, 1, 0}}
	for _, spans := range [][]Span{
		{{Decode, 0, 1}, {Decode, 2, 3}},  // a gap
		{{Prefill, 0, 2}, {Decode, 1, 3}}, // overlapping
		{{Prefill, 0, 4}},                 // beyond the trace
	} {
		if _, err := Replay(sel, spans, 2, []int64{1}, NewLayerLRU(1, 2, 1)); err == nil {
			t.Errorf("spans %v accepted", spans)
		}
	}
	// an online replay may stop before the trace ends
	if res, err := Replay(sel, []Span{{Prefill, 0, 2}}, 2, []int64{1}, NewLayerLRU(1, 2, 1)); err != nil ||
		res.Prefill.Tokens != 2 || res.Prefill.Accesses != 2 {
		t.Errorf("a prefix of the trace: %+v, %v", res, err)
	}
	bad := []Policy{
		&recorder{seen: map[int]bool{}, prefetch: func(StepInfo) []int { return []int{2} }}, // no such key
		&recorder{seen: map[int]bool{}, evict: func(k int) []int { return []int{1 - k} }},   // not resident
	}
	for i, p := range bad {
		if _, err := Replay(sel, TraceOrder(3), 2, []int64{1}, p); err == nil {
			t.Errorf("policy %d accepted", i)
		}
	}
	if _, err := Replay(sel, TraceOrder(3), 2, []int64{1, 1}, NewLayerLRU(1, 2, 1)); err == nil {
		t.Error("two layers of sizes for one accepted")
	}
}
