package gguf

import (
	"bytes"
	"strings"
	"testing"
)

// TestRoundTrip writes a header with every supported metadata kind, a
// string array the reader must skip and an alignment of 64, and reads it
// back.
func TestRoundTrip(t *testing.T) {
	tensors := []Tensor{
		{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{2560, 512, 384}, Type: IQ3_XXS, Offset: 0},
		{Name: "blk.0.ffn_down_exps.weight", Dims: []uint64{512, 2560, 384}, Type: IQ4_XS, Offset: 192_675_840},
		{Name: "blk.0.attn_norm.weight", Dims: []uint64{2560}, Type: F32, Offset: 192_675_840 + 267_386_880},
	}
	kv := map[string]any{
		"general.architecture":    "kolibri",
		"general.alignment":       uint32(64),
		"kolibri.expert_count":    uint32(384),
		"kolibri.block_count":     uint64(50),
		"kolibri.rope.freq_base":  float32(10000),
		"kolibri.signed":          int32(-3),
		"kolibri.flag":            true,
		"tokenizer.ggml.tokens":   []string{"a", "bb", strings.Repeat("c", 300)},
		"tokenizer.ggml.zzz_last": "after the array",
	}
	var buf bytes.Buffer
	data, err := WriteHeader(&buf, 64, kv, tensors)
	if err != nil {
		t.Fatal(err)
	}
	if data%64 != 0 || int64(buf.Len()) != data {
		t.Fatalf("data offset %d, header %d bytes", data, buf.Len())
	}
	f, err := Read(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if f.Version != 3 || f.Alignment != 64 || f.DataOffset != data {
		t.Fatalf("version %d alignment %d data %d; want 3, 64, %d", f.Version, f.Alignment, f.DataOffset, data)
	}
	if n, ok := f.Uint("kolibri.expert_count"); !ok || n != 384 {
		t.Fatalf("expert_count = %d, %v", n, ok)
	}
	if n, ok := f.Uint("kolibri.block_count"); !ok || n != 50 {
		t.Fatalf("block_count = %d, %v", n, ok)
	}
	if _, ok := f.Uint("kolibri.signed"); ok {
		t.Fatal("a negative int32 read as unsigned")
	}
	if s, _ := f.String("tokenizer.ggml.zzz_last"); s != "after the array" {
		t.Fatalf("value after the skipped array = %q", s)
	}
	if _, ok := f.KV["tokenizer.ggml.tokens"]; ok {
		t.Fatal("array kept")
	}
	if f.KV["kolibri.flag"] != true || f.KV["kolibri.rope.freq_base"] != float64(10000) {
		t.Fatalf("bool/float: %v %v", f.KV["kolibri.flag"], f.KV["kolibri.rope.freq_base"])
	}
	if len(f.Tensors) != 3 {
		t.Fatalf("%d tensors", len(f.Tensors))
	}
	down, ok := f.Tensor("blk.0.ffn_down_exps.weight")
	if !ok || down.Type != IQ4_XS || f.FileOffset(down) != data+192_675_840 {
		t.Fatalf("down = %+v at %d", down, f.FileOffset(down))
	}
}

// TestBytes checks tensor sizes against the per-expert bytes that
// tools/locality/expert_bytes.py recorded from the real checkpoints with
// gguf-py (testdata/locality/expert-bytes.json): 1,699,840 for IQ3_XXS gate
// and up with IQ4_XS down, 4,177,920 for Q8_0.
func TestBytes(t *testing.T) {
	gateUp := []uint64{2560, 512, 384}
	down := []uint64{512, 2560, 384}
	for _, c := range []struct {
		gate, down Type
		want       int64
	}{{IQ3_XXS, IQ4_XS, 1_699_840}, {Q8_0, Q8_0, 4_177_920}} {
		g, err := Tensor{Dims: gateUp, Type: c.gate}.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		d, err := Tensor{Dims: down, Type: c.down}.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		if got := (2*g + d) / 384; got != c.want {
			t.Errorf("%s/%s: %d bytes per expert, want %d", c.gate, c.down, got, c.want)
		}
	}
	if _, err := (Tensor{Name: "x", Dims: []uint64{100}, Type: Q8_0}).Bytes(); err == nil {
		t.Error("100 elements accepted as Q8_0 rows")
	}
	if _, err := (Tensor{Name: "x", Dims: []uint64{32}, Type: Type(99)}).Bytes(); err == nil {
		t.Error("unknown type accepted")
	}
}

func TestReadRejects(t *testing.T) {
	if _, err := Read(strings.NewReader("GGML\x03\x00\x00\x00")); err == nil {
		t.Error("wrong magic accepted")
	}
	var buf bytes.Buffer
	if _, err := WriteHeader(&buf, 32, map[string]any{"general.alignment": uint32(48)}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := Read(&buf); err == nil {
		t.Error("alignment 48 accepted")
	}
}
