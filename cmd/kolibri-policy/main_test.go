package main

import (
	"bytes"
	"encoding/json"
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
	noCache, runs, err := replay(tiny, tinySpans, tinyModel, sizes, baselines(tiny, tinySpans, tinyModel, 512),
		tinyTiming)
	if err != nil {
		t.Fatal(err)
	}
	if noCache.Prefill.Misses != 4 || noCache.Decode.Misses != 8 || len(noCache.Decode.Links) != 1 {
		t.Fatalf("no cache: %+v", noCache)
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
	if err := checkOracles(noCache, runs); err != nil {
		t.Fatal(err)
	}
	runs[sharedOracle].Runs[1].Online.Decode.MissBytes += 1000
	if checkOracles(noCache, runs) == nil {
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
	noCache, runs, err := replay(tiny, tinySpans, tinyModel, sizes, baselines(tiny, tinySpans, tinyModel, 512),
		tinyTiming)
	if err != nil {
		t.Fatal(err)
	}
	var b bytes.Buffer
	writeMarkdown(&b, "cmd", Assumptions{Ubatch: 512, DecodeTokensPerSecond: 63.7, PrefillTokensPerSecond: 1347,
		LFUHalfLife: 512}, tinyTiming.Links, sizes, 2, []int{0, 1},
		[]result{{"tiny", "w", Workload{Tokens: 3, NoCache: noCache, Policies: runs}}})
	md := b.String()
	for _, want := range []string{"| oracle, shared † |", "unattainable upper bound", "not a measurement",
		"| ssd ms/token |", "| layer LRU | decode |", "| no cache | prefill |"} {
		if !strings.Contains(md, want) {
			t.Errorf("the report lacks %q", want)
		}
	}
}
