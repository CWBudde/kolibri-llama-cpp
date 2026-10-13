package locality

import (
	"math"
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
				d.Accesses != layers*sel.Tokens*k || res.Prefill.Steps != 0 || res.Prefill.Accesses != 0 {
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
		Prefill: Counts{Tokens: 4, Steps: 2, Accesses: 5, Hits: 1, Misses: 4, MissBytes: 40, Evictions: 1,
			Layers: []LayerCounts{{5, 1}}},
		Decode: Counts{Tokens: 2, Steps: 2, Accesses: 4, Hits: 1, Misses: 3, MissBytes: 30, Evictions: 3,
			Layers: []LayerCounts{{4, 1}}},
		PeakBytes: 30,
	}
	if !reflect.DeepEqual(res, want) {
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
	want := Result{Prefill: Counts{Layers: []LayerCounts{{}}}, Decode: Counts{Tokens: 2, Steps: 2, Accesses: 4,
		Hits: 2, Misses: 2, MissBytes: 14, Prefetches: 2, PrefetchBytes: 14, UsefulPrefetches: 1,
		UnusedPrefetchBytes: 7, FullSteps: 1, Layers: []LayerCounts{{4, 2}}}, PeakBytes: 28}
	if !reflect.DeepEqual(res, want) {
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

func TestReplayRecall(t *testing.T) {
	// LRU of 2 per layer. Layer 0 {0,1} {0,1}: token 1 finds its whole set. Layer 1 {2,3} {2,0}:
	// token 1 finds 2 but not 0.
	sel := Selections{Layers: 2, Tokens: 2, K: 2, IDs: []int32{0, 1, 0, 1, 2, 3, 2, 0}}
	res, err := Replay(sel, TraceOrder(2), 4, []int64{1, 1}, NewLayerLRU(2, 4, 2))
	if err != nil {
		t.Fatal(err)
	}
	if d := res.Decode; d.FullSteps != 1 || !reflect.DeepEqual(d.Layers, []LayerCounts{{4, 2}, {4, 1}}) {
		t.Fatalf("recall: %d full steps, layers %v", d.FullSteps, d.Layers)
	}
	// on random traces the layers add up and a step is full exactly when it misses nothing
	r := rand.New(rand.NewPCG(9, 10))
	sel = randomSelections(r, 3, 120, 4, 12)
	spans, err := OnlineSpans(120, []Answer{{Prompt: 50, Tokens: make([]int32, 30)}}, false, 8)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []int{0, 4, 12} {
		res, err := Replay(sel, spans, 12, sizes(3, 1), NewLayerLRU(3, 12, c))
		if err != nil {
			t.Fatal(err)
		}
		for _, p := range []Counts{res.Prefill, res.Decode} {
			var acc, hits int
			for _, l := range p.Layers {
				acc, hits = acc+l.Accesses, hits+l.Hits
			}
			if acc != p.Accesses || hits != p.Hits || p.FullSteps > p.Steps || (c == 0 && p.FullSteps != 0) {
				t.Fatalf("capacity %d: %+v", c, p)
			}
		}
	}
}

func TestReplayTransfer(t *testing.T) {
	// One layer of 10-byte experts over a link of 10 B/s (1 s per expert) and one of 20 B/s, 2 s
	// per decoded token. Tokens {0} {1} {2}; before token 0 the policy prefetches 1, before token 2
	// it prefetches 2.
	//
	// 10 B/s: 1 arrives at 1, 0 (a miss queued behind it) at 2; token 0 stalls 2 and ends at 4.
	// Token 1 finds 1, there since 1: timely, ends at 6. Prefetched at 6, 2 arrives at 7: late,
	// token 2 stalls 1 and ends at 9.
	// 20 B/s: 1 at 0.5, 0 at 1, token 0 ends at 3; token 1, timely again, at 5; 2 arrives at 5.5,
	// late, and token 2 ends at 7.5.
	sel := Selections{Layers: 1, Tokens: 3, K: 1, IDs: []int32{0, 1, 2}}
	p := &recorder{seen: map[int]bool{}, prefetch: func(s StepInfo) []int {
		switch s.First {
		case 0:
			return []int{1}
		case 2:
			return []int{2}
		}
		return nil
	}}
	timing := Timing{Links: []Link{{"slow", 10}, {"fast", 20}}, DecodeStep: 2}
	res, err := ReplayTimed(sel, TraceOrder(3), 4, []int64{10}, p, timing)
	if err != nil {
		t.Fatal(err)
	}
	want := []LinkTime{{"slow", 9, 3, 1, 1}, {"fast", 7.5, 1.5, 1, 1}}
	if !reflect.DeepEqual(res.Decode.Links, want) || res.Decode.UsefulPrefetches != 2 {
		t.Fatalf("decode links %+v\nwant         %+v", res.Decode.Links, want)
	}
	// a prefill ubatch of two tokens at 0.5 s per token: misses 0 and 1 arrive at 1 and 2, then
	// 1 s of compute
	res, err = ReplayTimed(Selections{Layers: 1, Tokens: 2, K: 1, IDs: []int32{0, 1}}, []Span{{Prefill, 0, 2}}, 4,
		[]int64{10}, NewLayerLRU(1, 4, 4), Timing{Links: []Link{{"ssd", 10}}, PrefillToken: 0.5})
	if err != nil || !reflect.DeepEqual(res.Prefill.Links, []LinkTime{{"ssd", 3, 2, 0, 0}}) ||
		!reflect.DeepEqual(res.Decode.Links, []LinkTime{{Link: "ssd"}}) {
		t.Fatalf("prefill: %+v, %v", res, err)
	}
	if res, _ := Replay(sel, TraceOrder(3), 4, []int64{10}, &NoCache{}); res.Decode.Links != nil {
		t.Errorf("an untimed replay has links %v", res.Decode.Links)
	}
	nan, inf := math.NaN(), math.Inf(1)
	for _, bad := range []Timing{
		{Links: []Link{{"x", 0}}}, {Links: []Link{{"x", nan}}}, {Links: []Link{{"x", inf}}},
		{Links: []Link{{"x", 1}}, DecodeStep: nan}, {Links: []Link{{"x", 1}}, PrefillToken: inf},
		{Links: []Link{{"x", 1}}, DecodeStep: -1},
	} {
		if _, err := ReplayTimed(sel, TraceOrder(3), 4, []int64{10}, &NoCache{}, bad); err == nil {
			t.Errorf("timing %+v accepted", bad)
		}
	}
}

func TestReplaySegments(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	sel := randomSelections(r, 3, 200, 4, 16)
	answers := []Answer{{Prompt: 40, Tokens: make([]int32, 30)}, {Prompt: 120, Tokens: make([]int32, 40)}}
	spans, err := OnlineSpans(200, answers, false, 16)
	if err != nil {
		t.Fatal(err)
	}
	timing := Timing{Links: []Link{{"slow", 50}, {"fast", 500}}, DecodeStep: 0.1, PrefillToken: 0.01}
	policy := func() Policy {
		// an LRU that also prefetches the step's previous-token experts, so that prefetches cross cuts
		lru := NewLayerLRU(3, 16, 6)
		return &prefetching{Policy: lru, prefetch: func(s StepInfo) []int {
			if s.First == 0 {
				return nil
			}
			var keys []int
			for _, e := range sel.IDs[(s.Layer*sel.Tokens+s.First-1)*sel.K : (s.Layer*sel.Tokens+s.First)*sel.K] {
				keys = append(keys, s.Layer*16+int(e))
			}
			return keys
		}}
	}
	whole, err := ReplayTimed(sel, spans, 16, sizes(3, 5), policy(), timing)
	if err != nil {
		t.Fatal(err)
	}
	for _, cuts := range [][]int{nil, {1}, {len(spans) - 1}, {3, 7, 20}, {5, 6}} {
		segs, err := ReplaySegments(sel, spans, cuts, 16, sizes(3, 5), policy(), timing)
		if err != nil {
			t.Fatal(err)
		}
		if len(segs) != len(cuts)+1 {
			t.Fatalf("cuts %v: %d segments", cuts, len(segs))
		}
		for i, s := range segs {
			if s.PeakBytes > whole.PeakBytes {
				t.Fatalf("cuts %v: segment %d peaks at %d, the whole replay at %d", cuts, i, s.PeakBytes, whole.PeakBytes)
			}
		}
		got := Merge(segs...)
		if !sameResult(got, whole) {
			t.Fatalf("cuts %v: merged %+v\nwhole        %+v", cuts, got, whole)
		}
	}
	// the cache runs on across a cut: token 1 finds the expert token 0 loaded, and the prefetch
	// made before token 0 counts in the first segment, its use in the second
	sel = Selections{Layers: 1, Tokens: 3, K: 1, IDs: []int32{0, 1, 0}}
	p := &recorder{seen: map[int]bool{}, prefetch: func(s StepInfo) []int {
		if s.First == 0 {
			return []int{1}
		}
		return nil
	}}
	segs, err := ReplaySegments(sel, TraceOrder(3), []int{1}, 2, []int64{7}, p, Timing{})
	if err != nil {
		t.Fatal(err)
	}
	if a, b := segs[0].Decode, segs[1].Decode; a.Prefetches != 1 || a.UsefulPrefetches != 1 || a.Misses != 1 ||
		b.Hits != 2 || b.Misses != 0 || b.Tokens != 2 || segs[1].PeakBytes != 14 {
		t.Fatalf("segments %+v", segs)
	}
	for _, bad := range [][]int{{0}, {3}, {2, 1}, {1, 1}} {
		if _, err := ReplaySegments(sel, TraceOrder(3), bad, 2, []int64{7}, &NoCache{}, Timing{}); err == nil {
			t.Errorf("cuts %v accepted", bad)
		}
	}
}

// prefetching is a Policy whose Prefetch is replaced.
type prefetching struct {
	Policy
	prefetch func(StepInfo) []int
}

func (p *prefetching) Prefetch(s StepInfo) []int { return p.prefetch(s) }

// sameResult compares a and b exactly, their link times up to rounding.
func sameResult(a, b Result) bool {
	near := func(x, y float64) bool { return math.Abs(x-y) <= 1e-9*max(1, math.Abs(x), math.Abs(y)) }
	for _, p := range []Phase{Prefill, Decode} {
		x, y := a.phase(p), b.phase(p)
		if len(x.Links) != len(y.Links) {
			return false
		}
		for i := range x.Links {
			lx, ly := x.Links[i], y.Links[i]
			if !near(lx.Seconds, ly.Seconds) || !near(lx.StallSeconds, ly.StallSeconds) {
				return false
			}
		}
	}
	strip := func(r Result) Result {
		for _, c := range []*Counts{&r.Prefill, &r.Decode} {
			c.Links = slices.Clone(c.Links)
			for i := range c.Links {
				c.Links[i].Seconds, c.Links[i].StallSeconds = 0, 0
			}
		}
		return r
	}
	return reflect.DeepEqual(strip(a), strip(b))
}
