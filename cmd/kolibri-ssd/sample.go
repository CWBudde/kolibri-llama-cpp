package main

import (
	"fmt"
	"hash/fnv"
	"math/rand/v2"
	"sort"
)

// cellSeed derives a cell's seed from the run's seed and the cell's
// identity, so a cell reads the same offsets in every run, and different
// cells read different ones (the SSD may cache what it read recently).
func cellSeed(seed uint64, parts ...any) uint64 {
	h := fnv.New64a()
	fmt.Fprint(h, seed)
	for _, p := range parts {
		fmt.Fprintf(h, "/%v", p)
	}
	return h.Sum64()
}

// sampleIndices draws n distinct integers from [0, total), uniformly and
// deterministically for a seed: a sparse Fisher-Yates shuffle, O(n) memory.
func sampleIndices(total int64, n int, seed uint64) ([]int64, error) {
	if int64(n) > total {
		return nil, fmt.Errorf("want %d distinct positions, only %d exist", n, total)
	}
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	swapped := map[int64]int64{}
	get := func(i int64) int64 {
		if v, ok := swapped[i]; ok {
			return v
		}
		return i
	}
	out := make([]int64, n)
	for i := range int64(n) {
		j := i + rng.Int64N(total-i)
		vi, vj := get(i), get(j)
		swapped[j] = vi
		out[i] = vj
	}
	return out, nil
}

func alignUp(x, a int64) int64 { return (x + a - 1) / a * a }

// grid lays chunks of a size on positions aligned to align inside extents:
// in each extent the first aligned position, then every stride =
// alignUp(chunk, align) bytes, as long as the chunk ends inside the extent.
// Positions never overlap, so no byte is read twice in a cell.
type grid struct {
	ext    []extent
	first  []int64 // per extent, the first position
	cum    []int64 // per extent, the positions before it
	stride int64
	total  int64
}

func newGrid(ext []extent, chunk, align int64) grid {
	g := grid{ext: ext, stride: alignUp(chunk, align)}
	for _, e := range ext {
		first := alignUp(e.Start, align)
		var n int64
		if e.End-first >= chunk {
			n = (e.End-first-chunk)/g.stride + 1
		}
		g.first = append(g.first, first)
		g.cum = append(g.cum, g.total)
		g.total += n
	}
	return g
}

// at is the file offset of the i-th position.
func (g grid) at(i int64) int64 {
	k := sort.Search(len(g.cum), func(k int) bool { return g.cum[k] > i }) - 1
	return g.first[k] + (i-g.cum[k])*g.stride
}

// randomOffsets draws n distinct, non-overlapping chunk offsets aligned to
// align inside the extents.
func randomOffsets(ext []extent, chunk, align int64, n int, seed uint64) ([]int64, error) {
	if chunk <= 0 || align <= 0 {
		return nil, fmt.Errorf("chunk %d, align %d", chunk, align)
	}
	g := newGrid(ext, chunk, align)
	idx, err := sampleIndices(g.total, n, seed)
	if err != nil {
		return nil, fmt.Errorf("chunk %d: %w", chunk, err)
	}
	out := make([]int64, n)
	for i, j := range idx {
		out[i] = g.at(j)
	}
	return out, nil
}

// expertID is one routed expert.
type expertID struct{ Layer, Expert int }

// randomExperts draws n distinct experts.
func randomExperts(layers, experts, n int, seed uint64) ([]expertID, error) {
	idx, err := sampleIndices(int64(layers)*int64(experts), n, seed)
	if err != nil {
		return nil, fmt.Errorf("experts: %w", err)
	}
	out := make([]expertID, n)
	for i, j := range idx {
		out[i] = expertID{int(j) / experts, int(j) % experts}
	}
	return out, nil
}
