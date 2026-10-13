package main

import (
	"encoding/json"
	"testing"

	"kolibri-llm/internal/locality"
)

// cmd/kolibri-cache's tiny example: 2 layers x 3 tokens x 2 of 4 experts, 10 and 20 bytes per
// expert. In trace order a per-layer LRU hits 0, 5 and 6 times at 1, 2 and 4 experts per layer.
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
		if err := json.Unmarshal([]byte(`{"gguf": "tiny.gguf", "n_expert": 4,
			"layers": [{"bytes_per_expert": 10}, {"bytes_per_expert": 20}]}`), &m); err != nil {
			panic(err)
		}
		return m
	}()
)

func cacheSizes(t *testing.T, s string) CacheSizes {
	var c CacheSizes
	if err := json.Unmarshal([]byte(s), &c); err != nil {
		t.Fatal(err)
	}
	return c
}

func TestReplay(t *testing.T) {
	// token 0 is a prefill, tokens 1 and 2 are decoded
	spans := []locality.Span{{Phase: locality.Prefill, First: 0, End: 1}, {Phase: locality.Decode, First: 1, End: 2},
		{Phase: locality.Decode, First: 2, End: 3}}
	lru, err := replay(tiny, spans, tinyModel, []int{1, 2, 4})
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []int{0, 5, 6} {
		s := lru[i]
		if s.TraceOrderHits != want || s.Accesses != 12 || s.CacheBytes != int64(s.ExpertsPerLayer)*30 {
			t.Fatalf("size %d: %+v, want %d trace-order hits of 12", s.ExpertsPerLayer, s, want)
		}
		// online the prefill token's 4 experts are cold; the decoded tokens hit as in trace order
		on := s.Online
		if on.Prefill.Accesses != 4 || on.Prefill.Hits != 0 || on.Prefill.MissBytes != 60 ||
			on.Decode.Accesses != 8 || on.Decode.Hits != want || on.Prefill.Tokens != 1 || on.Decode.Tokens != 2 {
			t.Fatalf("size %d online: %+v", s.ExpertsPerLayer, on)
		}
	}
	if err := check(lru, cacheSizes(t, `{"per_layer": [{"experts_per_layer": 1, "hits": 0},
		{"experts_per_layer": 2, "hits": 5}, {"experts_per_layer": 4, "hits": 6}]}`)); err != nil {
		t.Fatal(err)
	}
	if check(lru, cacheSizes(t, `{"per_layer": [{"experts_per_layer": 1, "hits": 0},
		{"experts_per_layer": 2, "hits": 4}, {"experts_per_layer": 4, "hits": 6}]}`)) == nil {
		t.Fatal("a wrong hit count passes")
	}
	if check(lru, cacheSizes(t, `{"per_layer": [{"experts_per_layer": 1, "hits": 0}]}`)) == nil {
		t.Fatal("a size kolibri-cache did not simulate passes")
	}
}
