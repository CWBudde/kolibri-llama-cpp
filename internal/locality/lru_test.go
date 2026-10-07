package locality

import (
	"math/rand/v2"
	"reflect"
	"testing"
)

// lruHits simulates a fully associative LRU cache of capacity c over keys, the plain way: a list
// with the most recently used key first.
func lruHits(keys []int, c int) int {
	var cache []int
	hits := 0
	for _, k := range keys {
		i := 0
		for i < len(cache) && cache[i] != k {
			i++
		}
		if i < len(cache) {
			hits++
			cache = append(cache[:i], cache[i+1:]...)
		} else if len(cache) == c {
			cache = cache[:c-1]
		}
		cache = append([]int{k}, cache...)
	}
	return hits
}

func TestStackDistancesByHand(t *testing.T) {
	// a b c a b b d a: a's second access has b, c in between; b's second c, a; b right after
	// itself 0; a's last b, d
	keys := []int{0, 1, 2, 0, 1, 1, 3, 0}
	want := []int{-1, -1, -1, 2, 2, 0, -1, 2}
	if got := StackDistances(keys, 4); !reflect.DeepEqual(got, want) {
		t.Fatalf("StackDistances = %v, want %v", got, want)
	}
}

func TestStackDistancesExample(t *testing.T) {
	// the metrics example, its tokens' experts accessed in order: 0 1 | 0 2 | 1 0 | 0 1
	got := StackDistances(Flatten(example), 4)
	want := []int{-1, -1, 1, -1, 2, 2, 0, 1}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("StackDistances = %v, want %v", got, want)
	}
	// capacity 2 hits the distances 0 and 1, capacity 3 also the 2s
	if h := Hits(got, []int{1, 2, 3, 4}); !reflect.DeepEqual(h, []int{1, 3, 5, 5}) {
		t.Fatalf("Hits = %v", h)
	}
	for c := 1; c <= 4; c++ {
		if h := Hits(got, []int{c})[0]; h != lruHits(Flatten(example), c) {
			t.Fatalf("capacity %d: %d hits, the LRU simulation %d", c, h, lruHits(Flatten(example), c))
		}
	}
}

func TestStackDistancesSingleToken(t *testing.T) {
	if got := StackDistances(Flatten([][]int{{3, 1, 2}}), 4); !reflect.DeepEqual(got, []int{-1, -1, -1}) {
		t.Fatalf("StackDistances = %v", got)
	}
}

func TestStackDistancesAgainstLRU(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for trial := range 200 {
		nKey := 2 + r.IntN(20)
		keys := make([]int, r.IntN(300))
		for i := range keys {
			// skewed, so that small caches hit too
			keys[i] = min(r.IntN(nKey), r.IntN(nKey))
		}
		dist := StackDistances(keys, nKey)
		caps := make([]int, nKey+1)
		for c := range caps {
			caps[c] = c + 1
		}
		hits := Hits(dist, caps)
		for i, c := range caps {
			if want := lruHits(keys, c); hits[i] != want {
				t.Fatalf("trial %d, %d keys, capacity %d: %d hits, the LRU simulation %d", trial, nKey, c, hits[i], want)
			}
			if i > 0 && hits[i] < hits[i-1] {
				t.Fatalf("trial %d: hits fall from %d to %d at capacity %d", trial, hits[i-1], hits[i], c)
			}
		}
		// a cache holding every key misses only first accesses
		cold := 0
		for _, d := range dist {
			if d < 0 {
				cold++
			}
		}
		if hits[nKey-1] != len(keys)-cold {
			t.Fatalf("trial %d: %d hits with every key cached, want %d", trial, hits[nKey-1], len(keys)-cold)
		}
	}
}

func TestSharedKeys(t *testing.T) {
	// two layers, two tokens, two experts each: token-major, layers in order within a token
	s := Selections{Layers: 2, Tokens: 2, K: 2, IDs: []int32{
		0, 1, 2, 0, // layer 0: t0 {0,1}, t1 {2,0}
		1, 3, 1, 2, // layer 1: t0 {1,3}, t1 {1,2}
	}}
	keys, layers := SharedKeys(s, 4)
	if want := []int{0, 1, 5, 7, 2, 0, 5, 6}; !reflect.DeepEqual(keys, want) {
		t.Fatalf("keys = %v, want %v", keys, want)
	}
	if want := []int{0, 0, 1, 1, 0, 0, 1, 1}; !reflect.DeepEqual(layers, want) {
		t.Fatalf("layers = %v, want %v", layers, want)
	}
}
