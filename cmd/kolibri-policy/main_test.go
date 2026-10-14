package main

import (
	"bytes"
	"encoding/json"
	"math/rand/v2"
	"reflect"
	"slices"
	"strings"
	"testing"

	"kolibri-llm/internal/locality"
)

// cmd/kolibri-cache's tiny example: 2 layers x 3 tokens x 2 of 4 experts, 10 bytes per expert. In
// trace order a per-layer LRU hits 0, 5 and 6 times at 1, 2 and 4 experts per layer.
//
//	layer 0: {0,1} {0,2} {0,1}
//	layer 1: {3,2} {3,2} {1,2}
var (
	tiny = locality.Selections{Layers: 2, Tokens: 3, K: 2, IDs: []int32{
		0, 1, 0, 2, 0, 1,
		3, 2, 3, 2, 1, 2,
	}}
	tinyModel = func() Model {
		var m Model
		if err := json.Unmarshal([]byte(`{"n_expert": 4,
			"layers": [{"bytes_per_expert": 10}, {"bytes_per_expert": 10}]}`), &m); err != nil {
			panic(err)
		}
		return m
	}()
	// token 0 is a prefill, tokens 1 and 2 are decoded
	tinySpans = []locality.Span{{Phase: locality.Prefill, First: 0, End: 1},
		{Phase: locality.Decode, First: 1, End: 2}, {Phase: locality.Decode, First: 2, End: 3}}
	tinyTiming = locality.Timing{Links: []locality.Link{{Name: "ssd", BytesPerSecond: 10}}, DecodeStep: 1,
		PrefillToken: 1}
)

func TestReplay(t *testing.T) {
	sizes := []int{1, 2, 4}
	noCache, runs, err := replay(tiny, tinySpans, 2, tinyModel, sizes, baselines(tiny, tinySpans, tinyModel, 512),
		tinyTiming)
	if err != nil {
		t.Fatal(err)
	}
	if on := noCache.Online; on.Prefill.Misses != 4 || on.Decode.Misses != 8 || len(on.Decode.Links) != 1 {
		t.Fatalf("no cache: %+v", noCache)
	}
	// the test suffix is token 2 alone
	if sp := noCache.Split; sp.Train.Decode.Tokens != 1 || sp.Test.Decode.Tokens != 1 || sp.Test.Decode.Misses != 4 ||
		sp.Test.Prefill.Tokens != 0 {
		t.Fatalf("no cache split: %+v", sp)
	}
	if len(runs) != 6 || runs[sharedOracle].Policy != "oracle, shared" || !runs[sharedOracle].Oracle ||
		!runs[layerOracle].Oracle || runs[arc].Oracle {
		t.Fatalf("policies %+v", runs)
	}
	for i, want := range []int{0, 5, 6} {
		s := runs[layerLRU].Runs[i]
		if s.ExpertsPerLayer != sizes[i] || s.BudgetBytes != int64(sizes[i])*20 {
			t.Fatalf("size %d: %+v", sizes[i], s)
		}
		// online the prefill token's 4 experts are cold; the decoded tokens hit as in trace order
		on := s.Online
		if on.Prefill.Accesses != 4 || on.Prefill.Hits != 0 || on.Prefill.MissBytes != 40 ||
			on.Decode.Accesses != 8 || on.Decode.Hits != want || on.Prefill.Tokens != 1 || on.Decode.Tokens != 2 {
			t.Fatalf("size %d online: %+v", sizes[i], on)
		}
	}
	// token 2 needs {0,1} and {1,2}: with 2 experts per layer it finds 0 and 2, with 4 also 1 in layer 0
	for i, want := range []int{0, 2, 3} {
		if h := runs[layerLRU].Runs[i].Split.Test.Decode.Hits; h != want {
			t.Fatalf("size %d: %d test hits, want %d", sizes[i], h, want)
		}
	}
	if err := checkOracles(noCache.Online, runs); err != nil {
		t.Fatal(err)
	}
	runs[sharedOracle].Runs[1].Online.Decode.MissBytes += 1000
	if checkOracles(noCache.Online, runs) == nil {
		t.Fatal("an oracle that loads more than the layer LRU passes")
	}
}

func TestTraceOrderCheck(t *testing.T) {
	sizes := []int{1, 2, 4}
	tr, err := traceOrder(tiny, tinyModel, sizes)
	if err != nil {
		t.Fatal(err)
	}
	keys, _ := locality.SharedKeys(tiny, 4)
	shared := locality.Hits(locality.StackDistances(keys, 8), []int{2, 4, 8})
	if tr.Accesses != 12 || !slices.Equal(tr.LayerLRU, []int{0, 5, 6}) || !slices.Equal(tr.SharedLRU, shared) {
		t.Fatalf("trace order %+v, shared LRU want %v", tr, shared)
	}
	recorded := []CacheSize{{1, 0}, {2, 5}, {4, 6}}
	if err := check(tr.LayerLRU, sizes, recorded); err != nil {
		t.Fatal(err)
	}
	if check(tr.LayerLRU, sizes, []CacheSize{{1, 0}, {2, 4}, {4, 6}}) == nil {
		t.Fatal("a wrong hit count passes")
	}
	if check(tr.LayerLRU, sizes, []CacheSize{{1, 0}}) == nil {
		t.Fatal("a size kolibri-cache did not simulate passes")
	}
}

func TestParseLinks(t *testing.T) {
	links, err := parseLinks("ssd=5,pcie=25")
	if err != nil || len(links) != 2 || links[1] != (locality.Link{Name: "pcie", BytesPerSecond: 25e9}) {
		t.Fatalf("%v, %v", links, err)
	}
	for _, bad := range []string{"ssd", "ssd=0", "ssd=x", "ssd=NaN", "ssd=Inf", "ssd=-1"} {
		if _, err := parseLinks(bad); err == nil {
			t.Errorf("%q accepted", bad)
		}
	}
}

func TestMarkdownLabelsOracle(t *testing.T) {
	sizes := []int{1, 2}
	noCache, runs, err := replay(tiny, tinySpans, 2, tinyModel, sizes, baselines(tiny, tinySpans, tinyModel, 512),
		tinyTiming)
	if err != nil {
		t.Fatal(err)
	}
	hs := halfLives(512)
	grid, err := lfuGrid(tiny, tinySpans, 2, tinyModel, sizes, hs, tinyTiming)
	if err != nil {
		t.Fatal(err)
	}
	grids := map[string]map[string]Grid{"tiny": {"w": grid}}
	lastN, err := lastNGrid(tiny, tinySpans, 2, tinyModel, sizes, lastNs, tinyTiming)
	if err != nil {
		t.Fatal(err)
	}
	wl := Workload{Tokens: 3, Split: SplitWorkload{2, 1, 2}, NoCache: noCache, Policies: runs, LFUGrid: grid,
		LastN: lastN}
	for s, c := range sizes {
		wl.LFUTuning = append(wl.LFUTuning, tune(grids, "tiny", "w", s, c, 512))
	}
	tr := trace{tiny, tinySpans, 2, tinyModel, tinyTiming}
	var c curves
	if c.train, err = locality.MissCurves(tiny, tinySpans, 0, 2, 4, tinyModel.bytes()); err != nil {
		t.Fatal(err)
	}
	if c.test, err = locality.MissCurves(tiny, tinySpans, 2, 3, 4, tinyModel.bytes()); err != nil {
		t.Fatal(err)
	}
	if wl.Allocation, err = replayAllocations(tr, fits(map[string]map[string]curves{"tiny": {"w": c}}, "tiny", "w"),
		sizes); err != nil {
		t.Fatal(err)
	}
	for _, h := range horizons {
		g, err := markovGrid(tr, h, sizes, 0.5)
		if err != nil {
			t.Fatal(err)
		}
		wl.Markov = append(wl.Markov, g)
	}
	results := []result{{"tiny", "w", wl, tr}}
	g, ok := gateOf(results, "tiny", sizes, []string{"ssd"})
	if !ok {
		t.Fatal("no gate")
	}
	gb := stageBGate(results, "tiny", sizes, []string{"ssd"}, g)
	var b bytes.Buffer
	writeMarkdown(&b, "cmd", Assumptions{Ubatch: 512, DecodeTokensPerSecond: 63.7, PrefillTokensPerSecond: 1347,
		LFUHalfLife: 512, LFUHalfLives: hs, SplitRule: "the rule."}, tinyTiming.Links, sizes, 2, []int{0, 1},
		results, map[string]Gate{"tiny": g}, map[string]Gate{"tiny": gb})
	md := b.String()
	for _, want := range []string{"| oracle, shared † |", "unattainable upper bound", "not a measurement",
		"| ssd ms/token |", "| layer LRU | decode |", "| no cache | prefill |", "Held-out test suffix",
		"| w | 2 | 1 | 2 | 1 | 1 | 0 |", "tuned on the test data", "| tiny | w | 512: ", "| 2 | ",
		"| Hindsight † | Compulsory |", "fitted on the test data", "## Last-N reuse", "| tiny | 1 | ",
		"## Stage A gate", "| layer budgets allocated on the own prefix | ", "| last-N on shared LRU, n tuned",
		"## Stage B: the decayed-count Markov predictor", "| mixed, leave one workload out | – |", "**Cost.**",
		"| tiny | next token | 1 | ", "## Stage B gate", "| next token Markov on shared LRU, k tuned for ssd | ",
		"| same token Markov on shared LRU, k tuned for ssd | "} {
		if !strings.Contains(md, want) {
			t.Errorf("the report lacks %q", want)
		}
	}
}

func TestSplitCheck(t *testing.T) {
	answers := []locality.Answer{{Prompt: 1, Tokens: []int32{5}, EOS: true}, {Prompt: 2, Tokens: []int32{6}, EOS: true}}
	spans, err := locality.OnlineSpans(3, answers, true, 512)
	if err != nil {
		t.Fatal(err)
	}
	// prefill 0, decode 1, decode 2: the test suffix is token 2
	if cut, err := split(SplitWorkload{2, 1, 2}, spans, answers); err != nil || cut != 2 {
		t.Fatalf("split = %d, %v", cut, err)
	}
	for _, bad := range []SplitWorkload{{2, 1, 1}, {3, 1, 2}, {2, 0, 1}} {
		if _, err := split(bad, spans, answers); err == nil {
			t.Errorf("declared split %+v accepted", bad)
		}
	}
}

func TestTune(t *testing.T) {
	// half-lives 64 and 512, one size. A/w1 trains best at 64 and tests best at 512; the other
	// workload A/w2 trains best at 512; the other model's B/w1 at 64.
	g := func(train, test []int64) Grid {
		return Grid{HalfLives: []int{64, 512}, Train: [][]int64{{train[0]}, {train[1]}},
			Test: [][]int64{{test[0]}, {test[1]}}}
	}
	grids := map[string]map[string]Grid{
		"A": {"w1": g([]int64{10, 20}, []int64{5, 3}), "w2": g([]int64{30, 1}, []int64{8, 9})},
		"B": {"w1": g([]int64{7, 9}, []int64{4, 4})},
	}
	got := tune(grids, "A", "w1", 0, 128, 512)
	want := Tuning{ExpertsPerLayer: 128, Default: Choice{512, 3}, Own: Choice{64, 5}, LOWO: &Choice{512, 3},
		OtherModel: &Choice{64, 5}, Hindsight: Choice{512, 3}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tune = %+v\nwant   %+v", got, want)
	}
	// B has one workload and A has no other model's w2: no choice there
	if b, a2 := tune(grids, "B", "w1", 0, 128, 512), tune(grids, "A", "w2", 0, 128, 512); b.LOWO != nil ||
		b.OtherModel == nil || a2.OtherModel != nil || a2.LOWO == nil || a2.LOWO.HalfLife != 64 {
		t.Fatalf("B/w1 %+v, A/w2 %+v", b, a2)
	}
	// a third model leaves no single other model to tune on, and is never pooled with B
	grids["C"] = map[string]Grid{"w1": g([]int64{1, 9}, []int64{4, 4})}
	if c := tune(grids, "A", "w1", 0, 128, 512); c.OtherModel != nil || c.LOWO.HalfLife != 512 {
		t.Fatalf("with three models: %+v", c)
	}
	delete(grids, "C")
	// the test suffix moves only hindsight
	grids["A"]["w1"] = g([]int64{10, 20}, []int64{1, 7})
	got = tune(grids, "A", "w1", 0, 128, 512)
	if got.Own.HalfLife != 64 || got.LOWO.HalfLife != 512 || got.OtherModel.HalfLife != 64 ||
		got.Hindsight.HalfLife != 64 {
		t.Fatalf("after a changed test suffix: %+v", got)
	}
}

func TestFolds(t *testing.T) {
	// leave one workload out never looks at the held-out workload's own prefix
	g := func(train0, train1 int64) Grid {
		return Grid{HalfLives: []int{64, 512}, Train: [][]int64{{train0}, {train1}}, Test: [][]int64{{0}, {0}}}
	}
	grids := map[string]map[string]Grid{"A": {"w1": g(1, 100), "w2": g(5, 1), "w3": g(5, 2)}}
	before := tune(grids, "A", "w1", 0, 1, 512)
	grids["A"]["w1"] = g(100, 1)
	after := tune(grids, "A", "w1", 0, 1, 512)
	if before.LOWO.HalfLife != 512 || after.LOWO.HalfLife != 512 || before.Own.HalfLife != 64 ||
		after.Own.HalfLife != 512 {
		t.Fatalf("before %+v, after %+v", before, after)
	}
	// but it does look at the others': w2 and w3 together prefer 64 now
	grids["A"]["w2"], grids["A"]["w3"] = g(1, 5), g(1, 5)
	if c := tune(grids, "A", "w1", 0, 1, 512); c.LOWO.HalfLife != 64 {
		t.Fatalf("LOWO ignores the other workloads: %+v", c)
	}
}

func TestTuneNoLeak(t *testing.T) {
	// a random trace of 2 layers x 400 tokens, 2 of 8 experts, skewed; two answers
	r := rand.New(rand.NewPCG(5, 6))
	sel := locality.Selections{Layers: 2, Tokens: 400, K: 2, IDs: make([]int32, 2*400*2)}
	draw := func(i int) {
		a := int32(min(r.IntN(8), r.IntN(8)))
		b := (a + 1 + int32(r.IntN(7))) % 8
		sel.IDs[i], sel.IDs[i+1] = a, b
	}
	for i := 0; i < len(sel.IDs); i += 2 {
		draw(i)
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
	if err := json.Unmarshal([]byte(`{"n_expert": 8, "layers": [{"bytes_per_expert": 10}, {"bytes_per_expert": 10}]}`),
		&m); err != nil {
		t.Fatal(err)
	}
	sizes, hs := []int{1, 2, 4}, halfLives(512)
	before, err := lfuGrid(sel, spans, cut, m, sizes, hs, locality.Timing{})
	if err != nil {
		t.Fatal(err)
	}
	// redraw every expert of the test suffix
	for l := range sel.Layers {
		for tok := spans[cut].First; tok < sel.Tokens; tok++ {
			draw((l*sel.Tokens + tok) * sel.K)
		}
	}
	after, err := lfuGrid(sel, spans, cut, m, sizes, hs, locality.Timing{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before.Train, after.Train) {
		t.Fatalf("the training prefix's counts depend on the test suffix:\n%v\n%v", before.Train, after.Train)
	}
	if reflect.DeepEqual(before.Test, after.Test) {
		t.Fatal("a redrawn test suffix scores the same")
	}
	for s := range sizes {
		a := tune(map[string]map[string]Grid{"m": {"w": before}}, "m", "w", s, sizes[s], 512)
		b := tune(map[string]map[string]Grid{"m": {"w": after}}, "m", "w", s, sizes[s], 512)
		if a.Own.HalfLife != b.Own.HalfLife {
			t.Fatalf("size %d: the tuned half-life follows the test suffix: %d, %d", sizes[s], a.Own.HalfLife,
				b.Own.HalfLife)
		}
	}
}

func TestFits(t *testing.T) {
	c := func(v int64) [][]int64 { return [][]int64{{v, v / 2, 0}} }
	all := map[string]map[string]curves{
		"A": {"w1": {c(10), c(100)}, "w2": {c(20), c(200)}, "w3": {c(40), c(400)}},
		"B": {"w1": {c(1000), c(1000)}},
	}
	f := fits(all, "A", "w1")
	// the own prefix and hindsight are w1's; leaving w1 out adds w2's and w3's training curves, never
	// w1's own, any test suffix or the other model's
	if !reflect.DeepEqual(f[ownFit], c(10)) || !reflect.DeepEqual(f[lowoFit], c(60)) ||
		!reflect.DeepEqual(f[hindsightFit], c(100)) {
		t.Fatalf("fits %v", f)
	}
	if f := fits(all, "B", "w1"); f[lowoFit] != nil {
		t.Fatalf("a model with one workload leaves it out onto %v", f[lowoFit])
	}
	if compulsory(c(100)) != 0 || compulsory([][]int64{{9, 4}, {7, 3}}) != 7 {
		t.Fatal("compulsory bytes are not the curves at full capacity")
	}
}

func TestCheckCurves(t *testing.T) {
	sizes := []int{1, 2, 4}
	_, runs, err := replay(tiny, tinySpans, 2, tinyModel, sizes, baselines(tiny, tinySpans, tinyModel, 512), tinyTiming)
	if err != nil {
		t.Fatal(err)
	}
	var c curves
	if c.train, err = locality.MissCurves(tiny, tinySpans, 0, 2, 4, tinyModel.bytes()); err != nil {
		t.Fatal(err)
	}
	if c.test, err = locality.MissCurves(tiny, tinySpans, 2, 3, 4, tinyModel.bytes()); err != nil {
		t.Fatal(err)
	}
	if err := checkCurves(c, runs[layerLRU], tinyModel); err != nil {
		t.Fatal(err)
	}
	c.test[1][2] += 10
	if checkCurves(c, runs[layerLRU], tinyModel) == nil {
		t.Fatal("a wrong test curve passes")
	}
}

// decode is a test suffix result whose decode loads b bytes, missed and prefetched, in s seconds.
func decode(missed, prefetched int64, s float64) locality.Result {
	return locality.Result{Decode: locality.Counts{MissBytes: missed, PrefetchBytes: prefetched,
		Links: []locality.LinkTime{{Link: "ssd", Seconds: s}}}}
}

func TestGate(t *testing.T) {
	sizes := []int{8, 16}
	// two workloads; the reference loads 100+100 bytes in 2+2 s at 8, 50+50 in 1+1 s at 16
	ref := heldOut{"shared LRU", [][]locality.Result{{decode(100, 0, 2), decode(50, 0, 1)},
		{decode(100, 0, 2), decode(50, 0, 1)}}}
	cands := []heldOut{
		// the same in sum: no win anywhere
		{"tie", [][]locality.Result{{decode(150, 0, 3), decode(40, 0, 1)}, {decode(50, 0, 1), decode(60, 0, 1)}}},
		// fewer bytes at 16 only
		{"bytes", [][]locality.Result{{decode(100, 0, 2), decode(40, 0, 1)}, {decode(100, 0, 2), decode(50, 0, 1)}}},
		// prefetching: as many bytes, faster at 8
		{"faster", [][]locality.Result{{decode(60, 40, 1.5), decode(50, 0, 1)}, {decode(100, 0, 2), decode(50, 0, 1)}}},
		// fewer misses but more bytes loaded in all: no win
		{"wasteful", [][]locality.Result{{decode(80, 40, 2), decode(50, 0, 1)}, {decode(100, 0, 2), decode(50, 0, 1)}}},
		// faster by rounding alone: no win
		{"noise", [][]locality.Result{{decode(100, 0, 2-1e-13), decode(50, 0, 1)}, {decode(100, 0, 2), decode(50, 0, 1)}}},
	}
	g := gate(ref, cands, sizes, []string{"ssd"}, 7)
	want := map[string][][]string{"tie": {nil, nil}, "bytes": {nil, {"bytes"}}, "faster": {{"ssd"}, nil},
		"wasteful": {nil, nil}, "noise": {nil, nil}}
	if g.Tokens != 7 || len(g.Policies) != 5 || g.Reference.Sizes[0].LoadedBytes != 200 {
		t.Fatalf("gate %+v", g)
	}
	for _, e := range g.Policies {
		for s, x := range e.Sizes {
			if !slices.Equal(x.Wins, want[e.Policy][s]) {
				t.Errorf("%s at %d: wins %v, want %v", e.Policy, sizes[s], x.Wins, want[e.Policy][s])
			}
		}
		if e.Retained != (e.Policy == "bytes" || e.Policy == "faster") {
			t.Errorf("%s retained: %v", e.Policy, e.Retained)
		}
	}
}

func TestLoadedBytes(t *testing.T) {
	// a useful prefetch loads as many bytes as the miss it saves: 30 missed, 20 prefetched of which
	// 5 unused load 50
	if b := loadedBytes(locality.Counts{MissBytes: 30, PrefetchBytes: 20, UnusedPrefetchBytes: 5}); b != 50 {
		t.Fatalf("loaded %d bytes, want 50", b)
	}
	// an expert prefetched before the test suffix and first used in it counts in the test suffix, as
	// if it had loaded it: 50 and 7 carried load 57
	if b := loadedBytes(locality.Counts{MissBytes: 30, PrefetchBytes: 20, CarriedPrefetchBytes: 7}); b != 57 {
		t.Fatalf("loaded %d bytes with a carried prefetch, want 57", b)
	}
}

func TestChooseMarkov(t *testing.T) {
	// three parameters: the training prefix covers most with the second, the test suffix with the
	// third; the first is the only one of the counts alone
	g := HorizonGrid{Params: []locality.MarkovParams{{HalfLife: 64}, {HalfLife: 64, Mix: 0.5}, {Mix: 1}},
		Train: [][]locality.Recall{{{Covered: 3}}, {{Covered: 9}}, {{Covered: 9}}},
		Test:  [][]locality.Recall{{{Covered: 1}}, {{Covered: 2}}, {{Covered: 8}}}}
	chooseParams(&g)
	if g.Own != 1 || g.CountsOnly != 0 {
		t.Fatalf("own %d, counts only %d, want 1 (the first of equal) and 0", g.Own, g.CountsOnly)
	}
	// leaving one workload out: A/w1 covers most with 0, A/w2 and A/w3 with 1 together, B/w1 with 2
	grid := func(c ...int) HorizonGrid {
		h := HorizonGrid{Params: g.Params}
		for _, x := range c {
			h.Train = append(h.Train, []locality.Recall{{Covered: x}})
		}
		return h
	}
	results := []result{
		{model: "A", workload: "w1", w: Workload{Markov: []HorizonGrid{grid(100, 0, 0)}}},
		{model: "A", workload: "w2", w: Workload{Markov: []HorizonGrid{grid(4, 6, 0)}}},
		{model: "A", workload: "w3", w: Workload{Markov: []HorizonGrid{grid(4, 6, 0)}}},
		{model: "B", workload: "w1", w: Workload{Markov: []HorizonGrid{grid(0, 0, 1000)}}},
	}
	if got := lowoParams(results, "A", "w1", 0); got != 1 {
		t.Fatalf("leave one workload out for A/w1: %d, want 1", got)
	}
	if got := lowoParams(results, "B", "w1", 0); got != -1 {
		t.Fatalf("B has one workload, chose %d", got)
	}
}

// timed is a run whose decode took train seconds in the training prefix and test in the test
// suffix on each link.
func timed(train, test []float64) Run {
	var r Run
	for i := range train {
		r.Split.Train.Decode.Links = append(r.Split.Train.Decode.Links, locality.LinkTime{Seconds: train[i]})
		r.Split.Test.Decode.Links = append(r.Split.Test.Decode.Links, locality.LinkTime{Seconds: test[i]})
	}
	return r
}

func TestChooseK(t *testing.T) {
	// two links, one size, k 6 and 12: on the first link 12 trains faster, on the second they tie;
	// the test suffix would have chosen otherwise on both
	runs := []Runs{{Runs: []Run{timed([]float64{5, 3}, []float64{1, 9})}},
		{Runs: []Run{timed([]float64{4, 3}, []float64{2, 1})}}}
	if got := chooseK(runs, []int{6, 12}, 2); !reflect.DeepEqual(got, [][]int{{12, 6}}) {
		t.Fatalf("chooseK = %v, want [[12 6]]", got)
	}
}

func TestMarkovNoLeak(t *testing.T) {
	// a random trace of 3 layers x 400 tokens, 2 of 8 experts, skewed; two answers
	r := rand.New(rand.NewPCG(7, 8))
	sel := locality.Selections{Layers: 3, Tokens: 400, K: 2, IDs: make([]int32, 3*400*2)}
	draw := func(i int) {
		a := int32(min(r.IntN(8), r.IntN(8)))
		sel.IDs[i], sel.IDs[i+1] = a, (a+1+int32(r.IntN(7)))%8
	}
	for i := 0; i < len(sel.IDs); i += 2 {
		draw(i)
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
		{"bytes_per_expert": 10}]}`), &m); err != nil {
		t.Fatal(err)
	}
	tm := locality.Timing{Links: []locality.Link{{Name: "ssd", BytesPerSecond: 1000}}, DecodeStep: 0.01,
		PrefillToken: 0.001}
	tr := trace{sel, spans, cut, m, tm}
	sizes := []int{1, 2, 4}
	before, err := markovGrid(tr, locality.NextToken, sizes, 0.002)
	if err != nil {
		t.Fatal(err)
	}
	for l := range sel.Layers {
		for tok := spans[cut].First; tok < sel.Tokens; tok++ {
			draw((l*sel.Tokens + tok) * sel.K)
		}
	}
	after, err := markovGrid(tr, locality.NextToken, sizes, 0.002)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before.Train, after.Train) || before.Own != after.Own ||
		before.CountsOnly != after.CountsOnly || !reflect.DeepEqual(before.OwnK, after.OwnK) {
		t.Fatalf("the training prefix's recall or choices depend on the test suffix")
	}
	for k := range before.Runs {
		for s := range sizes {
			if !reflect.DeepEqual(before.Runs[k].Runs[s].Split.Train, after.Runs[k].Runs[s].Split.Train) {
				t.Fatalf("k %d, size %d: the training prefix's replay depends on the test suffix", before.K[k], sizes[s])
			}
		}
	}
	if reflect.DeepEqual(before.Test, after.Test) {
		t.Fatal("a redrawn test suffix scores the same")
	}
	// the cost is on every decoded step: without it the training prefix's decode is faster by it
	x, y := before.Runs[0].Runs[0].Split.Train.Decode, before.Free[0].Runs[0].Split.Train.Decode
	if x.Links[0].Seconds <= y.Links[0].Seconds {
		t.Fatalf("with the cost %g s, without %g s", x.Links[0].Seconds, y.Links[0].Seconds)
	}
	// on the trace as it is now, the online predictor replays as the ranking
	if err := checkOnline(tr, after, locality.NextToken, sizes, 1, 0.002); err != nil {
		t.Fatal(err)
	}
}

func TestStageBGate(t *testing.T) {
	sizes := []int{8}
	links := []string{"ssd", "pcie"}
	// Stage A: the shared LRU loads 100 bytes in 10 s on ssd and 4 s on pcie; ARC 90 bytes in 9 and 5
	entry := func(name string, b int64, ssd, pcie float64) GateEntry {
		return GateEntry{Policy: name, Sizes: []GateSize{{ExpertsPerLayer: 8, LoadedBytes: b,
			Seconds: []float64{ssd, pcie}}}}
	}
	a := Gate{Tokens: 3, Links: links, Reference: entry("shared LRU", 100, 10, 4),
		Policies: []GateEntry{entry("ARC", 90, 9, 5)}}
	best := bestOf(append([]GateEntry{a.Reference}, a.Policies...), "best")
	if b := best.Sizes[0]; b.LoadedBytes != 90 || !slices.Equal(b.Seconds, []float64{9, 4}) {
		t.Fatalf("best of Stage A: %+v", b)
	}
	if a.Reference.Sizes[0].Seconds[0] != 10 {
		t.Fatal("bestOf changed its entries")
	}
	run := func(b int64, ssd, pcie float64) heldOut {
		return heldOut{tests: [][]locality.Result{{{Decode: locality.Counts{MissBytes: b,
			Links: []locality.LinkTime{{Seconds: ssd}, {Seconds: pcie}}}}}}}
	}
	cands := []heldOut{
		run(100, 9.5, 4),    // beats the shared LRU on ssd, not ARC: no win
		run(100, 9, 4),      // ties the best: no win
		run(100, 8, 4),      // beats the best on ssd, judged on ssd: a win
		run(100, 9, 3),      // beats the best on pcie, judged on ssd only: no win
		run(100, 9, 3.9),    // beats the best on pcie, judged on pcie: a win
		run(100, 9, 4-4e-9), // faster on pcie by rounding alone: no win
	}
	only := [][]int{{0}, {0}, {0}, {0}, {1}, {1}}
	g := judge(best, cands, only, sizes, links, 3)
	want := [][]string{nil, nil, {"ssd"}, nil, {"pcie"}, nil}
	for i, e := range g.Policies {
		if !slices.Equal(e.Sizes[0].Wins, want[i]) || e.Retained != (want[i] != nil) {
			t.Errorf("candidate %d: wins %v, retained %v; want %v", i, e.Sizes[0].Wins, e.Retained, want[i])
		}
	}
	// the predictor of both horizons, k tuned per link: the next token's k 6 is faster than the shared
	// LRU on ssd but not than ARC, its k 12 faster than both on pcie
	horizon := func(runs ...Run) HorizonGrid {
		g := HorizonGrid{K: []int{6, 12}, OwnK: [][]int{{6, 12}}}
		for _, r := range runs {
			g.Runs = append(g.Runs, Runs{Runs: []Run{r}})
		}
		return g
	}
	at := func(ssd, pcie float64) Run {
		r := timed([]float64{0, 0}, []float64{ssd, pcie})
		r.Split.Test.Decode.MissBytes = 100
		return r
	}
	results := []result{{model: "m", workload: "w", w: Workload{Markov: []HorizonGrid{
		horizon(at(9.5, 9), at(9, 3.5)), horizon(at(20, 20), at(20, 20))}}}}
	g = stageBGate(results, "m", sizes, links, a)
	if !reflect.DeepEqual(g.Reference.Sizes, best.Sizes) || len(g.Policies) != 4 {
		t.Fatalf("Stage B gate against %+v with %d policies", g.Reference, len(g.Policies))
	}
	for i, e := range g.Policies {
		if e.Retained != (i == 1) || e.Policy != markovName(horizons[i/2], links[i%2]) {
			t.Errorf("%s: retained %v, wins %v", e.Policy, e.Retained, e.Sizes[0].Wins)
		}
	}
}
