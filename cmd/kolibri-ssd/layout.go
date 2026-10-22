package main

import (
	"fmt"
	"slices"

	"kolibri-llm/internal/gguf"
)

// kinds are the stacked expert tensors in the order patch 0011 reads them.
var kinds = [3]string{"gate", "up", "down"}

// expertTensor is one layer's blk.L.ffn_<kind>_exps.weight: expert e's rows
// are the contiguous slice [Offset + e*Slice, Offset + (e+1)*Slice), the
// stacked tensor's nb[2] (llama_expert_cache::load in patch 0011).
type expertTensor struct {
	Type   string `json:"type"`
	Offset int64  `json:"offset"` // absolute file offset of expert 0
	Slice  int64  `json:"slice"`  // bytes per expert
}

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

// expertLayout finds the stacked expert tensors of every layer; each must
// have ne[2] equal to the expert count and split into equal slices.
func expertLayout(f *gguf.File) (*layout, error) {
	arch, ok := f.String("general.architecture")
	if !ok {
		return nil, fmt.Errorf("no general.architecture")
	}
	nl, ok := f.Uint(arch + ".block_count")
	if !ok {
		return nil, fmt.Errorf("no %s.block_count", arch)
	}
	ne, ok := f.Uint(arch + ".expert_count")
	if !ok || ne == 0 {
		return nil, fmt.Errorf("no %s.expert_count", arch)
	}
	l := &layout{Arch: arch, Experts: int(ne)}
	var ext []extent
	for il := range int(nl) {
		var lt [3]expertTensor
		for k, kind := range kinds {
			name := fmt.Sprintf("blk.%d.ffn_%s_exps.weight", il, kind)
			t, ok := f.Tensor(name)
			if !ok {
				return nil, fmt.Errorf("no tensor %s", name)
			}
			if len(t.Dims) != 3 || t.Dims[2] != ne {
				return nil, fmt.Errorf("%s: dims %v, want %d experts on the outermost axis", name, t.Dims, ne)
			}
			n, err := t.Bytes()
			if err != nil {
				return nil, err
			}
			if n%int64(ne) != 0 {
				return nil, fmt.Errorf("%s: %d bytes do not split into %d experts", name, n, ne)
			}
			off := f.FileOffset(t)
			lt[k] = expertTensor{Type: t.Type.String(), Offset: off, Slice: n / int64(ne)}
			ext = append(ext, extent{off, off + n})
		}
		l.Layers = append(l.Layers, lt)
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
