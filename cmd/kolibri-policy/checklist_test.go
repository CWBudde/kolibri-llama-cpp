package main

import (
	"bytes"
	"slices"
	"strings"
	"testing"

	"kolibri-llm/internal/locality"
)

func TestUnionStats(t *testing.T) {
	// one layer: three ubatches of 2 tokens, the last short, with unions 2, 4 and 2
	sel := locality.Selections{Layers: 1, Tokens: 5, K: 2, IDs: []int32{0, 1, 0, 1, 2, 3, 4, 5, 6, 7}}
	spans := []locality.Span{{Phase: locality.Prefill, First: 0, End: 2}, {Phase: locality.Prefill, First: 2, End: 4},
		{Phase: locality.Prefill, First: 4, End: 5}}
	u := unionStats(sel, spans, 8, 2)
	if u.Steps != 3 || u.FullSteps != 2 || u.Median != 2 || u.P95 != 4 || u.Max != 4 {
		t.Fatalf("%+v", u)
	}
	if want := (2.0 + 4 + 2) / 3; u.Mean != want {
		t.Fatalf("mean %v, want %v", u.Mean, want)
	}
}

func TestUnionStatsLayerShare(t *testing.T) {
	// a union above 244 needs that many experts; K is large for the fixture's sake
	const n = 300
	ids := make([]int32, 2*n)
	for i := range n {
		ids[i] = int32(i) // layer 0 has 300 distinct experts
	}
	sel := locality.Selections{Layers: 2, Tokens: 1, K: n, IDs: ids}
	u := unionStats(sel, []locality.Span{{Phase: locality.Prefill, First: 0, End: 1}}, n, 1)
	if u.LayerAbove244 != [2]float64{0, 1} {
		t.Fatalf("per layer above 244: %v", u.LayerAbove244)
	}
	if !slices.Equal(u.Above, []int{1, 1, 1, 1, 1}) {
		t.Fatalf("above: %v", u.Above)
	}
}

// pinnedFixture has a reference with step evictions, LFU, beside references without.
func pinnedFixture() []result {
	run := func(evictions int) Runs {
		var c locality.Counts
		c.StepEvictions = evictions
		return Runs{Runs: []Run{{Replayed: Replayed{Split: Split{Test: locality.Result{Decode: c}}}}}}
	}
	policies := make([]Runs, sharedOracle+1)
	names := map[int]string{sharedLRU: "shared LRU", layerLRU: "layer LRU", lfu: "LFU", arc: "ARC",
		layerLRUPinned: "layer LRU, step-pinned", lfuPinned: "LFU, step-pinned", arcPinned: "ARC, step-pinned"}
	for i := range policies {
		policies[i] = run(0)
		policies[i].Policy = names[i]
	}
	policies[lfu].Runs[0].Split.Test.Decode.StepEvictions = 7
	return []result{{model: "m", workload: "w", w: Workload{Policies: policies, LastN: LastNGrid{N: []int{1},
		Own: []int{1}, Runs: []Runs{run(0)}}}}}
}

func TestCheckPinnedRefs(t *testing.T) {
	results := pinnedFixture()
	if err := checkPinnedRefs(results, "m", pinnedRefNames); err != nil {
		t.Fatal(err)
	}
	if err := checkPinnedRefs(results, "m", []string{"LFU"}); err == nil ||
		!strings.Contains(err.Error(), "7 step evictions") {
		t.Fatalf("an unpinned reference passes: %v", err)
	}
	if err := checkPinnedRefs(results, "m", []string{"nothing"}); err == nil {
		t.Fatal("an unknown reference passes")
	}
}

func TestPinnedRefGateLeavesOthersOut(t *testing.T) {
	entry := func(name string, b int64) GateEntry {
		return GateEntry{Policy: name, Sizes: []GateSize{{ExpertsPerLayer: 8, LoadedBytes: b, Seconds: []float64{1}}}}
	}
	// the unpinned LFU has the fewest bytes, so a gate that keeps it has a stricter reference
	g := Gate{Tokens: 1, Links: []string{"ssd"}, Reference: entry("shared LRU", 100), Policies: []GateEntry{
		entry("LFU", 10), entry("LFU, step-pinned", 50), entry("SLRU, shared", 5)}}
	p := pinnedRefGate(nil, "m", []int{8}, []string{"ssd"}, g)
	if got := p.Reference.Sizes[0].LoadedBytes; got != 50 {
		t.Fatalf("reference of %d bytes, want the pinned LFU's 50", got)
	}
}

func TestCheckPinned(t *testing.T) {
	counts := func(miss int64, evictions int) Runs {
		return Runs{Policy: "p", Runs: []Run{{ExpertsPerLayer: 8, Replayed: Replayed{Online: locality.Result{
			Decode: locality.Counts{MissBytes: miss, StepEvictions: evictions}}}}}}
	}
	runs := make([]Runs, sharedOracle+1)
	for i := range runs {
		runs[i] = counts(100, 0)
	}
	if _, err := checkPinned(runs, []Runs{counts(100, 0), counts(110, 0)}); err != nil {
		t.Fatal(err)
	}
	if _, err := checkPinned(runs, []Runs{counts(100, 0), counts(110, 1)}); err == nil {
		t.Fatal("step evictions pass")
	}
	if _, err := checkPinned(runs, []Runs{counts(90, 0), counts(110, 0)}); err == nil {
		t.Fatal("a pinned oracle that loads less than its oracle passes")
	}
	msg, err := checkPinned(runs, []Runs{counts(120, 0), counts(110, 0)})
	if err != nil || !strings.Contains(msg, "at 1 of 1 sizes") {
		t.Fatalf("%q, %v", msg, err)
	}
}

func TestBestCausalLeavesOraclesOut(t *testing.T) {
	test := func(miss int64, hits int) Runs {
		return Runs{Runs: []Run{{Replayed: Replayed{Split: Split{Test: locality.Result{
			Decode: locality.Counts{MissBytes: miss, Hits: hits, Accesses: 100}}}}}}}
	}
	a, b, o := test(50, 80), test(50, 90), test(1, 99)
	a.Policy, b.Policy, o.Policy, o.Oracle = "a", "b", "o", true
	name, tl := bestCausal([]result{{w: Workload{Policies: []Runs{a, b, o}}}}, 0)
	if name != "b" || tl.loaded != 50 {
		t.Fatalf("%s, %+v", name, tl)
	}
}

func TestChecklistOnlyWithPinnedOracles(t *testing.T) {
	var b bytes.Buffer
	writeChecklist(&b, []int{8}, []result{{model: "m", workload: "w"}}, nil)
	if b.Len() != 0 {
		t.Fatalf("wrote %q", b.String())
	}
}

func TestSharedFits(t *testing.T) {
	model := func(sizes ...int64) Model {
		m := Model{NExpert: 4}
		m.Layers = make([]struct {
			BytesPerExpert int64 `json:"bytes_per_expert"`
		}, len(sizes))
		for l, b := range sizes {
			m.Layers[l].BytesPerExpert = b
		}
		return m
	}
	rs := func(m Model, budget int64, union int) []result {
		return []result{{t: trace{m: m}, w: Workload{Unions: []UnionStats{{Max: union}},
			PinnedOracles: []Runs{{Runs: []Run{{BudgetBytes: budget}}}}}}}
	}
	if n, ok := sharedFits(rs(model(2, 2), 8, 4)); !ok || n != 4 {
		t.Fatalf("a union of 4 experts of 2 bytes in 8 bytes: %d, %v", n, ok)
	}
	if _, ok := sharedFits(rs(model(2, 2), 7, 4)); ok {
		t.Fatal("a union over the budget fits")
	}
	if _, ok := sharedFits(rs(model(2, 3), 80, 4)); ok {
		t.Fatal("experts of two sizes pass")
	}
}
