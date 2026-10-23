package gguf

import (
	"bytes"
	"maps"
	"slices"
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

// TestMetadata checks the raw metadata pairs: every value kind WriteHeader
// takes, arrays included, comes back in file order with its type id and
// encoded value, and the header's byte positions match EncodeHeader and
// EncodeTensorInfos.
func TestMetadata(t *testing.T) {
	tensors := []Tensor{
		{Name: "token_embd.weight", Dims: []uint64{64, 10}, Type: F32, Offset: 0},
		{Name: "blk.0.attn_norm.weight", Dims: []uint64{64}, Type: F32, Offset: 2560},
	}
	kv := map[string]any{
		"a.u8":     uint8(7),
		"b.u32":    uint32(513),
		"c.i32":    int32(-1),
		"d.u64":    uint64(1) << 40,
		"e.i64":    int64(-5),
		"f.f32":    float32(0.5),
		"g.f64":    float64(1e-6),
		"h.bool":   true,
		"i.str":    "{% chat %}",
		"j.strs":   []string{"", "ab", "c"},
		"k.bools":  []bool{true, false, true},
		"l.i32s":   []int32{1, -2, 3},
		"m.u32s":   []uint32{4, 5},
		"n.f32s":   []float32{0.25, -1},
		"o.string": "last",
	}
	var buf bytes.Buffer
	data, err := WriteHeader(&buf, 32, kv, tensors)
	if err != nil {
		t.Fatal(err)
	}
	raw := slices.Clone(buf.Bytes())
	f, err := Read(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if f.DataOffset != data || len(f.Metadata) != len(kv) {
		t.Fatalf("data %d, %d pairs; want %d, %d", f.DataOffset, len(f.Metadata), data, len(kv))
	}
	keys := slices.Sorted(maps.Keys(kv))
	var want []KV
	for i, k := range keys {
		w, err := NewKV(k, kv[k])
		if err != nil {
			t.Fatal(err)
		}
		want = append(want, w)
		if got := f.Metadata[i]; got.Key != k || got.Type != w.Type || !bytes.Equal(got.Value, w.Value) {
			t.Errorf("pair %d = %s type %d %x; want %s type %d %x", i, got.Key, got.Type, got.Value, k, w.Type, w.Value)
		}
	}
	// spot checks of the encoding: a bool array is type 9, element type 7,
	// count 3, then one byte per element
	b := f.Metadata[slices.Index(keys, "k.bools")]
	if b.Type != tArray || !bytes.Equal(b.Value, []byte{tBool, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1}) {
		t.Errorf("k.bools = type %d %x", b.Type, b.Value)
	}
	if s, ok := f.Metadata[slices.Index(keys, "i.str")].String(); !ok || s != "{% chat %}" {
		t.Errorf("i.str String() = %q, %v", s, ok)
	}
	if _, ok := b.String(); ok {
		t.Error("an array read as a string")
	}
	head := EncodeHeader(3, uint64(len(tensors)), want, EncodeTensorInfos(tensors))
	if f.HeaderEnd != int64(len(head)) || !bytes.Equal(raw[:f.HeaderEnd], head) {
		t.Fatalf("header end %d, EncodeHeader %d bytes, equal %v", f.HeaderEnd, len(head), bytes.Equal(raw[:min(len(raw), len(head))], head))
	}
	if !bytes.Equal(raw[f.InfoOffset:f.HeaderEnd], EncodeTensorInfos(tensors)) {
		t.Error("tensor infos are not at [InfoOffset, HeaderEnd)")
	}
	if !bytes.Equal(raw[f.HeaderEnd:data], make([]byte, data-f.HeaderEnd)) {
		t.Error("padding is not zero")
	}
	if _, err := NewKV("x", struct{}{}); err == nil {
		t.Error("NewKV accepted a struct")
	}
}
