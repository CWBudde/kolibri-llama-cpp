package main

import (
	"encoding/json"
	"math/rand/v2"
	"slices"
	"testing"

	"kolibri-llm/internal/locality"
)

func TestParseCosts(t *testing.T) {
	if got, err := parseCosts(""); err != nil || got != nil {
		t.Fatalf("empty: %v, %v; want off", got, err)
	}
	if got, err := parseCosts("0,15,30"); err != nil || !slices.Equal(got, []float64{0, 15e-6, 30e-6}) {
		t.Fatalf("0,15,30: %v, %v", got, err)
	}
	if got, _ := parseCosts("15,30"); perfectKind(got[0]) != "perfect (oracle, 15 µs)" ||
		perfectKind(got[1]) != "perfect (oracle, 30 µs)" {
		t.Fatalf("names %q, %q", perfectKind(got[0]), perfectKind(got[1]))
	}
	for _, bad := range []string{"-1", "x", "1,,2", "+Inf"} {
		if _, err := parseCosts(bad); err == nil {
			t.Errorf("%q: no error", bad)
		}
	}
}

func TestPerfectCostReachesGate(t *testing.T) {
	// 4 layers x 400 tokens, Top-2 of 8 experts, skewed; two answers
	r := rand.New(rand.NewPCG(9, 10))
	sel := locality.Selections{Layers: 4, Tokens: 400, K: 2, IDs: make([]int32, 4*400*2)}
	for i := 0; i < len(sel.IDs); i += 2 {
		a := int32(min(r.IntN(8), r.IntN(8)))
		sel.IDs[i], sel.IDs[i+1] = a, (a+1+int32(r.IntN(7)))%8
	}
	answers := []locality.Answer{{Prompt: 100, Tokens: make([]int32, 80)}, {Prompt: 250, Tokens: make([]int32, 80)}}
	spans, err := locality.OnlineSpans(400, answers, true, 64)
	if err != nil {
		t.Fatal(err)
	}
	cut, err := locality.SplitAfter(spans, answers, 1)
	if err != nil {
		t.Fatal(err)
	}
	var m Model
	if err := json.Unmarshal([]byte(`{"n_expert": 8, "layers": [{"bytes_per_expert": 10}, {"bytes_per_expert": 10},
		{"bytes_per_expert": 10}, {"bytes_per_expert": 10}]}`), &m); err != nil {
		t.Fatal(err)
	}
	tm := locality.Timing{Links: []locality.Link{{Name: "ssd", BytesPerSecond: 1000}}, DecodeStep: 0.01,
		PrefillToken: 0.001}
	tr := trace{sel, spans, cut, m, tm}
	sizes := []int{1, 2}
	costs := []float64{0, 1} // 1 s per decoded layer step dwarfs every load
	grids, err := perfectGrids(tr, locality.MarkovParams{HalfLife: 16, Mix: 0.5}, sizes, costs)
	if err != nil {
		t.Fatal(err)
	}
	if len(grids) != 2 || grids[0].CostSeconds != 0 || grids[1].CostSeconds != 1 ||
		grids[1].Horizon != locality.SameToken.String() || grids[1].LeadSteps != 1 {
		t.Fatalf("grids: %d, %+v", len(grids), grids[1].Horizon)
	}
	// at k 2, the trace's Top-2, the prefetches are exactly the true sets of layers 2 and 3
	ranking := locality.RankWith(locality.NewPerfect(sel, locality.NewMarkov(4, 8, grids[1].PadParams)), sel, spans,
		8, locality.SameToken, slices.Max(markovKs))
	if x := ranking.Recall(sel, spans, cut, len(spans), []int{2})[0]; x.Exact != x.Steps/2 || x.Covered != x.Needed/2 {
		t.Fatalf("test suffix recall at k 2: %+v; want layers 2 and 3 of every decoded token", x)
	}
	decoded := 0
	for _, s := range spans[cut:] {
		if s.Phase == locality.Decode {
			decoded++
		}
	}
	steps := float64(decoded * sel.Layers)
	// a reference between the two: faster than the oracle with its cost, slower than without
	ref := GateEntry{Policy: "best", Sizes: []GateSize{{ExpertsPerLayer: 1, LoadedBytes: 1 << 40,
		Seconds: []float64{0.5 * steps}}, {ExpertsPerLayer: 2, LoadedBytes: 0, Seconds: []float64{0.5 * steps}}}}
	a := Gate{Tokens: decoded, Links: []string{"ssd"}, Reference: ref}
	results := []result{{model: "m", workload: "w", w: Workload{Perfect: grids}, t: tr}}
	g := stageBGate(results, "m", sizes, []string{"ssd"}, a)
	if len(g.Policies) != 2 {
		t.Fatalf("%d policies judged, want the oracle at each cost", len(g.Policies))
	}
	for i, e := range g.Policies {
		if want := predictorName(perfectKind(costs[i]), locality.SameToken, "ssd"); e.Policy != want {
			t.Fatalf("policy %q, want %q", e.Policy, want)
		}
		for s := range sizes {
			p := grids[i]
			run := p.Runs[slices.Index(p.K, p.OwnK[s][0])].Runs[s].Split.Test.Decode
			if got := e.Sizes[s].Seconds[0]; got != run.Links[0].Seconds {
				t.Fatalf("cost %g, size %d: gate %g s, the replay with cost %g s", costs[i], sizes[s], got,
					run.Links[0].Seconds)
			}
			if c := e.Sizes[s].Seconds[0]; (c >= steps) != (costs[i] == 1) {
				t.Fatalf("cost %g, size %d: %g s for %g decoded layer steps", costs[i], sizes[s], c, steps)
			}
		}
	}
	// size 1: fewer bytes than the reference at either cost; size 2: faster only without the cost
	if w := g.Policies[0].Sizes[1].Wins; !slices.Equal(w, []string{"ssd"}) {
		t.Fatalf("cost 0, size 2: wins %v", w)
	}
	if w := g.Policies[1].Sizes[1].Wins; w != nil {
		t.Fatalf("cost 1 s, size 2: wins %v", w)
	}
	// without the flag nothing is added: a workload without the oracle judges no oracle row
	results[0].w.Perfect = nil
	if g := stageBGate(results, "m", sizes, []string{"ssd"}, a); len(g.Policies) != 0 {
		t.Fatalf("%d policies without the oracle", len(g.Policies))
	}
}
