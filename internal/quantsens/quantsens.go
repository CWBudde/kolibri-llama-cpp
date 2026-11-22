// Package quantsens measures and allocates per-tensor quantization types for
// Kolibri GGUFs, after the "llama.cpp route" of LookAtMyQuant
// (github.com/ShinMegamiBoson/lookatmyquant, gguf_compiler.py and
// compiler.py allocate, MIT): the menu is llama.cpp's own tensor types, the
// cost of a type for a group of tensors is the KL divergence it adds when it
// alone differs from a base file, and the plan is the cheapest set of types
// under a byte budget, assuming the added KLs sum. The result runs in
// stock llama.cpp.
//
// Unlike LookAtMyQuant, which swaps dequantized tensors into a PyTorch
// model, a measurement here is a whole GGUF (gguf.Splice of the base and one
// item from a file that holds every tensor in the candidate type) scored by
// llama-perplexity --kl-divergence against the BF16 logits.
package quantsens

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"kolibri-llm/internal/gguf"
)

// Item is a group of tensors that always gets one type: one tensor kind of
// one layer (Layer >= 0), the same kind across all layers (Layer = -1,
// scope "kind"), or the token embedding or output head.
type Item struct {
	Name    string   `json:"item"`
	Kind    string   `json:"kind"`
	Layer   int      `json:"layer"`
	Tensors []string `json:"tensors"`
}

// Kinds lists the tensor kinds that get a type, in the order items are
// listed per layer. Router (ffn_gate_inp), its bias (exp_probs_b) and every
// norm are not items: they stay as the base has them (F32).
var Kinds = []string{"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "shexp", "attn", "token_embd", "output"}

var blkRe = regexp.MustCompile(`^blk\.(\d+)\.(.+)\.weight$`)

// KindOf returns the kind of a tensor name and its layer (-1 outside the
// blocks), or "" for a tensor that is not quantized by a plan.
func KindOf(name string) (kind string, layer int) {
	switch name {
	case "token_embd.weight":
		return "token_embd", -1
	case "output.weight":
		return "output", -1
	}
	m := blkRe.FindStringSubmatch(name)
	if m == nil {
		return "", -1
	}
	layer, _ = strconv.Atoi(m[1])
	switch m[2] {
	case "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps":
		return m[2], layer
	case "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp":
		return "shexp", layer
	case "attn_q", "attn_k", "attn_v", "attn_output":
		return "attn", layer
	}
	return "", -1
}

// Items groups f's tensors. scope "layer" gives one item per kind and layer
// (blk.12.attn) plus token_embd and output; scope "kind" gives one item per
// kind spanning all layers (attn); scope "band:N" gives one item per kind and
// band of N consecutive layers (blk.10-14.attn; the last band may be
// shorter) plus token_embd and output.
func Items(f *gguf.File, scope string) ([]Item, error) {
	band, err := bandOf(scope)
	if err != nil {
		return nil, err
	}
	var nLayers int
	for _, t := range f.Tensors {
		if _, l := KindOf(t.Name); l+1 > nLayers {
			nLayers = l + 1
		}
	}
	byName := map[string]*Item{}
	var order []string
	for _, t := range f.Tensors {
		kind, layer := KindOf(t.Name)
		if kind == "" {
			continue
		}
		name := kind
		switch {
		case scope == "kind":
			layer = -1
		case layer < 0:
		case band == 1:
			name = fmt.Sprintf("blk.%d.%s", layer, kind)
		default:
			layer -= layer % band
			name = fmt.Sprintf("blk.%d-%d.%s", layer, min(layer+band, nLayers)-1, kind)
		}
		it := byName[name]
		if it == nil {
			it = &Item{Name: name, Kind: kind, Layer: layer}
			byName[name] = it
			order = append(order, name)
		}
		it.Tensors = append(it.Tensors, t.Name)
	}
	rank := map[string]int{}
	for i, k := range Kinds {
		rank[k] = i
	}
	items := make([]Item, 0, len(order))
	for _, n := range order {
		items = append(items, *byName[n])
	}
	sort.SliceStable(items, func(i, j int) bool {
		a, b := items[i], items[j]
		if a.Layer != b.Layer {
			// Blocks first, by layer; token_embd and output (layer -1) last.
			if a.Layer < 0 || b.Layer < 0 {
				return b.Layer < 0 && a.Layer >= 0
			}
			return a.Layer < b.Layer
		}
		return rank[a.Kind] < rank[b.Kind]
	})
	return items, nil
}

// bandOf returns the layers per item of a scope: 1 for "layer", N for
// "band:N", 0 for "kind".
func bandOf(scope string) (int, error) {
	switch scope {
	case "kind":
		return 0, nil
	case "layer":
		return 1, nil
	}
	if n, ok := strings.CutPrefix(scope, "band:"); ok {
		if b, err := strconv.Atoi(n); err == nil && b >= 1 {
			return b, nil
		}
	}
	return 0, fmt.Errorf("scope %q: want layer, kind or band:N", scope)
}

var bandRe = regexp.MustCompile(`^blk\.(\d+)-(\d+)\.(.+)$`)

// ParseBand splits a band item's name (blk.10-14.attn) into its first and
// last layer and its kind.
func ParseBand(name string) (first, last int, kind string, ok bool) {
	m := bandRe.FindStringSubmatch(name)
	if m == nil {
		return 0, 0, "", false
	}
	first, _ = strconv.Atoi(m[1])
	last, _ = strconv.Atoi(m[2])
	return first, last, m[3], true
}

// TypeBytes is the size an item's tensors would have in type typ, from
// their shapes in f (an error if a shape does not fit the type's blocks).
// It lets allocate price a type without the donor file at hand.
func TypeBytes(f *gguf.File, it Item, typ string) (int64, error) {
	ty, ok := gguf.ParseType(typ)
	if !ok {
		return 0, fmt.Errorf("unknown type %q", typ)
	}
	var total int64
	for _, name := range it.Tensors {
		t, ok := f.Tensor(name)
		if !ok {
			return 0, fmt.Errorf("%s: no tensor %s", it.Name, name)
		}
		n, err := gguf.Tensor{Name: name, Dims: t.Dims, Type: ty}.Bytes()
		if err != nil {
			return 0, err
		}
		total += n
	}
	return total, nil
}

// ItemBytes is the size of an item's tensors in f and their type, or
// "mixed" when they differ.
func ItemBytes(f *gguf.File, it Item) (bytes int64, typ string, err error) {
	for i, name := range it.Tensors {
		t, ok := f.Tensor(name)
		if !ok {
			return 0, "", fmt.Errorf("%s: no tensor %s", it.Name, name)
		}
		n, err := t.Bytes()
		if err != nil {
			return 0, "", err
		}
		bytes += n
		if i == 0 {
			typ = t.Type.String()
		} else if typ != t.Type.String() {
			typ = "mixed"
		}
	}
	return bytes, typ, nil
}
