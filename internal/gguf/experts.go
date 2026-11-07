package gguf

import "fmt"

// ExpertKinds are the stacked expert tensors of a layer, in the order the
// expert cache (patch 0011) reads them.
var ExpertKinds = [3]string{"gate", "up", "down"}

// ExpertTensor is one layer's blk.L.ffn_<kind>_exps.weight: expert e's rows
// are the contiguous slice [Offset + e*Slice, Offset + (e+1)*Slice), the
// stacked tensor's nb[2] (llama_expert_cache::load in patch 0011).
type ExpertTensor struct {
	Type   string `json:"type"`
	TypeID Type   `json:"-"`
	Offset int64  `json:"offset"` // absolute file offset of expert 0
	Slice  int64  `json:"slice"`  // bytes per expert
}

// ExpertLayers finds the stacked expert tensors of every layer; each must
// have ne[2] equal to the expert count and split into equal slices. It returns
// the architecture, the expert count and the tensors per layer.
func (f *File) ExpertLayers() (arch string, experts int, layers [][3]ExpertTensor, err error) {
	arch, ok := f.String("general.architecture")
	if !ok {
		return "", 0, nil, fmt.Errorf("no general.architecture")
	}
	nl, ok := f.Uint(arch + ".block_count")
	if !ok {
		return "", 0, nil, fmt.Errorf("no %s.block_count", arch)
	}
	ne, ok := f.Uint(arch + ".expert_count")
	if !ok || ne == 0 {
		return "", 0, nil, fmt.Errorf("no %s.expert_count", arch)
	}
	for il := range int(nl) {
		var lt [3]ExpertTensor
		for k, kind := range ExpertKinds {
			name := fmt.Sprintf("blk.%d.ffn_%s_exps.weight", il, kind)
			t, ok := f.Tensor(name)
			if !ok {
				return "", 0, nil, fmt.Errorf("no tensor %s", name)
			}
			if len(t.Dims) != 3 || t.Dims[2] != ne {
				return "", 0, nil, fmt.Errorf("%s: dims %v, want %d experts on the outermost axis", name, t.Dims, ne)
			}
			n, err := t.Bytes()
			if err != nil {
				return "", 0, nil, err
			}
			if n%int64(ne) != 0 {
				return "", 0, nil, fmt.Errorf("%s: %d bytes do not split into %d experts", name, n, ne)
			}
			lt[k] = ExpertTensor{Type: t.Type.String(), TypeID: t.Type, Offset: f.FileOffset(t), Slice: n / int64(ne)}
		}
		layers = append(layers, lt)
	}
	return arch, int(ne), layers, nil
}
