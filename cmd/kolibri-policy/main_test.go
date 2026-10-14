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
	grid, err := lfuGrid(tiny, tinySpans, 2, tinyModel, sizes, hs)
	if err != nil {
		t.Fatal(err)
	}
	grids := map[string]map[string]Grid{"tiny": {"w": grid}}
	wl := Workload{Tokens: 3, Split: SplitWorkload{2, 1, 2}, NoCache: noCache, Policies: runs, LFUGrid: grid}
	for s, c := range sizes {
		wl.LFUTuning = append(wl.LFUTuning, tune(grids, "tiny", "w", s, c, 512))
	}
	var b bytes.Buffer
	writeMarkdown(&b, "cmd", Assumptions{Ubatch: 512, DecodeTokensPerSecond: 63.7, PrefillTokensPerSecond: 1347,
		LFUHalfLife: 512, LFUHalfLives: hs, SplitRule: "the rule."}, tinyTiming.Links, sizes, 2, []int{0, 1},
		[]result{{"tiny", "w", wl}})
	md := b.String()
	for _, want := range []string{"| oracle, shared † |", "unattainable upper bound", "not a measurement",
		"| ssd ms/token |", "| layer LRU | decode |", "| no cache | prefill |", "Held-out test suffix",
		"| w | 2 | 1 | 2 | 1 | 1 | 0 |", "tuned on the test data", "| tiny | w | 512: ", "| 2 | "} {
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
	before, err := lfuGrid(sel, spans, cut, m, sizes, hs)
	if err != nil {
		t.Fatal(err)
	}
	// redraw every expert of the test suffix
	for l := range sel.Layers {
		for tok := spans[cut].First; tok < sel.Tokens; tok++ {
			draw((l*sel.Tokens + tok) * sel.K)
		}
	}
	after, err := lfuGrid(sel, spans, cut, m, sizes, hs)
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
