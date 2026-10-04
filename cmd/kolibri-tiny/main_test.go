package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"kolibri-llm/internal/kolibri"
	"kolibri-llm/internal/safetensors"
)

func TestGenerate(t *testing.T) {
	dir := t.TempDir()
	if err := generate(dir, "", 1); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := kolibri.ParseConfig(raw)
	if err != nil {
		t.Fatal(err)
	}
	// The reference fixture: 6 layers, 4 sliding then 2 full.
	if cfg.NumHiddenLayers != 6 || cfg.NumExperts != 8 || !cfg.IsFullAttention(4) || cfg.IsFullAttention(3) {
		t.Errorf("config = %+v", cfg)
	}

	f, err := os.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	h, err := safetensors.ReadHeader(f)
	if err != nil {
		t.Fatal(err)
	}
	if st, _ := f.Stat(); st.Size() != h.FileSize() {
		t.Errorf("file has %d bytes, header implies %d", st.Size(), h.FileSize())
	}
	got := map[string]safetensors.Tensor{}
	for _, tt := range h.Tensors {
		got[tt.Name] = tt
	}
	want := cfg.ExpectedNames()
	if len(got) != len(want) {
		t.Errorf("%d tensors, want %d", len(got), len(want))
	}
	for _, name := range want {
		tt, ok := got[name]
		if !ok {
			t.Errorf("missing %s", name)
			continue
		}
		e, err := kolibri.Classify(name)
		if err != nil {
			t.Fatal(err)
		}
		if tt.DType != "BF16" || !slices.Equal(tt.Shape, cfg.ExpectedShape(e)) {
			t.Errorf("%s: %s%v, want BF16%v", name, tt.DType, tt.Shape, cfg.ExpectedShape(e))
		}
	}

	// The four block norms share a shape; the converter check can only
	// tell them apart if their values differ.
	norms := [][]byte{}
	for _, n := range []string{"input_layernorm", "post_attn_norm", "post_attention_layernorm", "post_ffn_norm"} {
		tt := got["model.layers.0."+n+".weight"]
		buf := make([]byte, tt.NumBytes())
		if _, err := f.ReadAt(buf, 8+h.Size+tt.Offsets[0]); err != nil {
			t.Fatal(err)
		}
		for _, prev := range norms {
			if bytes.Equal(prev, buf) {
				t.Errorf("%s has the same values as another block norm", n)
			}
		}
		norms = append(norms, buf)
	}

	raw, err = os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var manifest []entry
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatal(err)
	}
	// 3 global tensors + 18 per layer.
	if len(manifest) != 3+6*18 {
		t.Errorf("%d manifest entries, want %d", len(manifest), 3+6*18)
	}
	seen := map[string]int{}
	for _, m := range manifest {
		for i, src := range m.Sources {
			seen[src]++
			e, err := kolibri.Classify(src)
			if err != nil {
				t.Fatal(err)
			}
			if e.GGUFName() != m.Name || !slices.Equal(e.Spec.GGUFShape(cfg), m.NE) {
				t.Errorf("%s: source %s maps to %s%v", m.Name, src, e.GGUFName(), e.Spec.GGUFShape(cfg))
			}
			if e.Spec.PerExpert() && e.Expert != i {
				t.Errorf("%s: source %d is expert %d", m.Name, i, e.Expert)
			}
		}
	}
	for _, name := range want {
		if seen[name] != 1 {
			t.Errorf("%s appears %d times in the manifest", name, seen[name])
		}
	}
}

func TestGenerateDeterministic(t *testing.T) {
	a, b := t.TempDir(), t.TempDir()
	if err := generate(a, "", 7); err != nil {
		t.Fatal(err)
	}
	if err := generate(b, "", 7); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"config.json", "model.safetensors", "manifest.json"} {
		x, _ := os.ReadFile(filepath.Join(a, name))
		y, _ := os.ReadFile(filepath.Join(b, name))
		if !bytes.Equal(x, y) {
			t.Errorf("%s differs between two runs with the same seed", name)
		}
	}
}
