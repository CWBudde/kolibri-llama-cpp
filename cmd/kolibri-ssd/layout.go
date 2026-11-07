package main

import (
	"slices"

	"kolibri-llm/internal/gguf"
)

// kinds are the stacked expert tensors in the order patch 0011 reads them.
var kinds = gguf.ExpertKinds

// expertTensor is one layer's blk.L.ffn_<kind>_exps.weight.
type expertTensor = gguf.ExpertTensor

// layout is where a GGUF keeps its routed experts.
type layout struct {
	Arch    string
	Experts int
	Layers  [][3]expertTensor
	Extents []extent // the expert tensors' bytes, sorted and merged
}

// extent is a byte range [Start, End) of the file.
type extent struct{ Start, End int64 }

func (e extent) len() int64 { return e.End - e.Start }

// expertLayout finds the stacked expert tensors of every layer.
func expertLayout(f *gguf.File) (*layout, error) {
	arch, ne, layers, err := f.ExpertLayers()
	if err != nil {
		return nil, err
	}
	l := &layout{Arch: arch, Experts: ne, Layers: layers}
	var ext []extent
	for _, lt := range layers {
		for _, t := range lt {
			ext = append(ext, extent{t.Offset, t.Offset + int64(ne)*t.Slice})
		}
	}
	l.Extents = mergeExtents(ext)
	return l, nil
}

func mergeExtents(ext []extent) []extent {
	ext = slices.Clone(ext)
	slices.SortFunc(ext, func(a, b extent) int { return int(a.Start - b.Start) })
	var out []extent
	for _, e := range ext {
		if n := len(out); n > 0 && e.Start <= out[n-1].End {
			out[n-1].End = max(out[n-1].End, e.End)
			continue
		}
		out = append(out, e)
	}
	return out
}

// sliceOffset is the file offset of expert e's rows of kind k in layer il.
func (l *layout) sliceOffset(il, k, e int) int64 {
	t := l.Layers[il][k]
	return t.Offset + int64(e)*t.Slice
}

// expertBytes is one expert of layer il: gate, up and down.
func (l *layout) expertBytes(il int) int64 {
	var n int64
	for _, t := range l.Layers[il] {
		n += t.Slice
	}
	return n
}

// maxExpertBytes is the largest expert of any layer.
func (l *layout) maxExpertBytes() int64 {
	var n int64
	for il := range l.Layers {
		n = max(n, l.expertBytes(il))
	}
	return n
}

// sliceSizes are the distinct per-kind slice sizes, ascending.
func (l *layout) sliceSizes() []int64 {
	var s []int64
	for _, lt := range l.Layers {
		for _, t := range lt {
			s = append(s, t.Slice)
		}
	}
	slices.Sort(s)
	return slices.Compact(s)
}

// expertSizes are the distinct per-expert sizes, ascending.
func (l *layout) expertSizes() []int64 {
	var s []int64
	for il := range l.Layers {
		s = append(s, l.expertBytes(il))
	}
	slices.Sort(s)
	return slices.Compact(s)
}

func (l *layout) extentBytes() int64 {
	var n int64
	for _, e := range l.Extents {
		n += e.len()
	}
	return n
}
