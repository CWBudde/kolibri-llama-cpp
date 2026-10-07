package main

import (
	"encoding/json"
	"math"
	"reflect"
	"testing"

	"kolibri-llm/internal/locality"
)

// 2 layers x 3 tokens x 2 of 4 experts; one expert takes 10 bytes in layer 0, 20 in layer 1.
//
//	layer 0: {0,1} {0,2} {0,1}  accesses 0 1 0 2 0 1, stack distances -1 -1 1 -1 1 2
//	layer 1: {3,2} {3,2} {1,2}  accesses 3 2 3 2 1 2, stack distances -1 -1 1 1 -1 1
//
// Shared, token by token and layer by layer (layer 1's expert e is key 4+e):
//
//	0 1 7 6 | 0 2 7 6 | 0 1 5 6  distances -1 -1 -1 -1 | 3 -1 3 3 | 3 4 -1 3
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

func TestSimulate(t *testing.T) {
	w, cold, fullHits := simulate(tiny, tinyModel, []int{1, 2, 4})
	if w.Tokens != 3 || w.Activations != 12 || w.Cold != 6 {
		t.Fatalf("Tokens, Activations, Cold = %d, %d, %d", w.Tokens, w.Activations, w.Cold)
	}
	if !reflect.DeepEqual(cold, []int{3, 3}) || !reflect.DeepEqual(fullHits, []int{3, 3}) {
		t.Fatalf("cold %v, fullHits %v", cold, fullHits)
	}
	// per-layer hits: layer 0 0/2/3, layer 1 0/3/3; shared (capacities 2, 4, 8): 0/2/3 and 0/3/3
	for name, sizes := range map[string][]Size{"per-layer": w.PerLayer, "shared": w.Shared} {
		var hits []int
		for _, s := range sizes {
			hits = append(hits, s.Hits)
			if s.Hits+s.Misses != 12 {
				t.Fatalf("%s %d: %d hits + %d misses", name, s.ExpertsPerLayer, s.Hits, s.Misses)
			}
		}
		if !reflect.DeepEqual(hits, []int{0, 5, 6}) {
			t.Fatalf("%s hits = %v", name, hits)
		}
	}
	// size 2 per layer: misses 4 x 10 + 3 x 20 bytes over 3 tokens; it holds 2 x (10 + 20) bytes
	s := w.PerLayer[1]
	if math.Abs(s.LoadedPerToken-100.0/3) > 1e-9 || s.CacheBytes != 60 || s.HitRate != 5.0/12 {
		t.Fatalf("size 2: loaded %v per token, cache %d bytes, hit rate %v", s.LoadedPerToken, s.CacheBytes, s.HitRate)
	}
	if !reflect.DeepEqual(s.LayerHitRate, []float64{2.0 / 6, 3.0 / 6}) {
		t.Fatalf("size 2: hit rate per layer %v", s.LayerHitRate)
	}
	// the shared cache of 2 x 2 slots spreads the same 5 hits as 2 in layer 0, 3 in layer 1
	if !reflect.DeepEqual(w.Shared[1].LayerHitRate, []float64{2.0 / 6, 3.0 / 6}) {
		t.Fatalf("shared 2: hit rate per layer %v", w.Shared[1].LayerHitRate)
	}
}

func TestCheck(t *testing.T) {
	_, cold, fullHits := simulate(tiny, tinyModel, []int{4})
	stats := []LayerStats{{Counts: []int{3, 2, 1, 0}, Reuses: 3}, {Counts: []int{0, 1, 3, 2}, Reuses: 3}}
	if err := check(stats, cold, fullHits); err != nil {
		t.Fatal(err)
	}
	stats[1].Reuses = 4
	if check(stats, cold, fullHits) == nil {
		t.Fatal("a wrong reuse count passes")
	}
	stats[1].Reuses, stats[0].Counts[3] = 3, 1
	if check(stats, cold, fullHits) == nil {
		t.Fatal("a wrong selected-expert count passes")
	}
	if check(stats[:1], cold, fullHits) == nil {
		t.Fatal("a missing layer passes")
	}
}
