package gguf

import (
	"bytes"
	"strings"
	"testing"
)

// fixture is a tensor and its bytes for buildGGUF.
type fixture struct {
	name string
	dims []uint64
	typ  Type
	fill byte // every byte of the data
}

// buildGGUF writes a whole GGUF the way ggml lays it out: tensors at aligned
// offsets, each padded with zeros.
func buildGGUF(t *testing.T, align uint64, fx []fixture) []byte {
	t.Helper()
	var tensors []Tensor
	var datas [][]byte
	var off uint64
	for _, f := range fx {
		tt := Tensor{Name: f.name, Dims: f.dims, Type: f.typ, Offset: off}
		n, err := tt.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		tensors = append(tensors, tt)
		datas = append(datas, bytes.Repeat([]byte{f.fill}, int(n)))
		off = (off + uint64(n) + align - 1) / align * align
	}
	var buf bytes.Buffer
	kv := map[string]any{"general.architecture": "kolibri1", "general.alignment": uint32(align), "general.file_type": uint32(7)}
	if _, err := WriteHeader(&buf, align, kv, tensors); err != nil {
		t.Fatal(err)
	}
	for _, d := range datas {
		buf.Write(d)
		p := (uint64(len(d)) + align - 1) / align * align
		buf.Write(make([]byte, p-uint64(len(d))))
	}
	return buf.Bytes()
}

func source(t *testing.T, b []byte) Source {
	t.Helper()
	f, err := Read(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	return Source{File: f, Data: bytes.NewReader(b)}
}

// tensorData returns a tensor's info and bytes in a whole GGUF.
func tensorData(t *testing.T, b []byte, name string) (Tensor, []byte) {
	t.Helper()
	s := source(t, b)
	tt, ok := s.File.Tensor(name)
	if !ok {
		t.Fatalf("no tensor %s", name)
	}
	n, _ := tt.Bytes()
	at := s.File.FileOffset(tt)
	return tt, b[at : at+n]
}

var baseFx = []fixture{
	{"token_embd.weight", []uint64{64, 3}, Q8_0, 0x11},
	{"blk.0.attn_norm.weight", []uint64{5}, F32, 0x22},
	{"blk.0.ffn_gate_exps.weight", []uint64{256, 2, 4}, Q8_0, 0x33},
	{"blk.0.ffn_down_exps.weight", []uint64{256, 2, 4}, Q8_0, 0x44},
	{"output.weight", []uint64{64, 3}, Q8_0, 0x55},
}

// TestSpliceEmptyIsIdentity: taking nothing reproduces the base file byte
// for byte.
func TestSpliceEmptyIsIdentity(t *testing.T) {
	for _, align := range []uint64{32, 64} {
		b := buildGGUF(t, align, baseFx)
		var out bytes.Buffer
		n, err := Splice(&out, source(t, b), nil)
		if err != nil {
			t.Fatal(err)
		}
		if n != int64(out.Len()) || !bytes.Equal(out.Bytes(), b) {
			t.Fatalf("align %d: %d bytes written, equal %v", align, n, bytes.Equal(out.Bytes(), b))
		}
	}
}

// TestSpliceTakesTypeAndBytes: a taken tensor carries the donor's type and
// bytes, every other tensor keeps base's, and the offsets stay aligned and
// contiguous, as ggml checks them.
func TestSpliceTakesTypeAndBytes(t *testing.T) {
	b := buildGGUF(t, 32, baseFx)
	donorFx := []fixture{
		{"token_embd.weight", []uint64{64, 3}, Q4_0, 0x66},
		{"blk.0.attn_norm.weight", []uint64{5}, F32, 0x77},
		{"blk.0.ffn_gate_exps.weight", []uint64{256, 2, 4}, IQ3_XXS, 0x88},
		{"blk.0.ffn_down_exps.weight", []uint64{256, 2, 4}, IQ4_XS, 0x99},
		{"output.weight", []uint64{64, 3}, Q4_0, 0xaa},
	}
	d := buildGGUF(t, 32, donorFx)
	ds := source(t, d)
	take := map[string]Source{"blk.0.ffn_gate_exps.weight": ds, "token_embd.weight": ds}
	var out bytes.Buffer
	if _, err := Splice(&out, source(t, b), take); err != nil {
		t.Fatal(err)
	}
	o := out.Bytes()
	of := source(t, o).File
	var want uint64
	for i, tt := range of.Tensors {
		if tt.Name != baseFx[i].name {
			t.Fatalf("tensor %d is %s, want base order %s", i, tt.Name, baseFx[i].name)
		}
		if tt.Offset != want {
			t.Fatalf("%s at %d, want %d", tt.Name, tt.Offset, want)
		}
		n, _ := tt.Bytes()
		want = (want + uint64(n) + 31) / 32 * 32
	}
	if int64(len(o)) != of.DataOffset+int64(want) {
		t.Fatalf("file %d bytes, want %d", len(o), of.DataOffset+int64(want))
	}
	for _, name := range []string{"blk.0.ffn_gate_exps.weight", "token_embd.weight"} {
		got, gb := tensorData(t, o, name)
		dt, db := tensorData(t, d, name)
		if got.Type != dt.Type || !bytes.Equal(gb, db) {
			t.Fatalf("%s: type %s, want the donor's %s; bytes equal %v", name, got.Type, dt.Type, bytes.Equal(gb, db))
		}
	}
	for _, name := range []string{"blk.0.attn_norm.weight", "blk.0.ffn_down_exps.weight", "output.weight"} {
		got, gb := tensorData(t, o, name)
		bt, bb := tensorData(t, b, name)
		if got.Type != bt.Type || !bytes.Equal(gb, bb) {
			t.Fatalf("%s: type %s, want the base's %s; bytes equal %v", name, got.Type, bt.Type, bytes.Equal(gb, bb))
		}
	}
	if ft, _ := of.Uint("general.file_type"); ft != 7 {
		t.Fatalf("general.file_type = %d, want base's 7", ft)
	}
}

// TestSpliceTakeAllMatchesDonor: taking every tensor from a donor with the
// same order and metadata reproduces the donor.
func TestSpliceTakeAllMatchesDonor(t *testing.T) {
	b := buildGGUF(t, 32, baseFx)
	var donorFx []fixture
	for _, f := range baseFx {
		f.fill ^= 0xff
		if f.typ == Q8_0 {
			f.typ = Q4_0
		}
		donorFx = append(donorFx, f)
	}
	d := buildGGUF(t, 32, donorFx)
	ds := source(t, d)
	take := map[string]Source{}
	for _, f := range baseFx {
		take[f.name] = ds
	}
	var out bytes.Buffer
	if _, err := Splice(&out, source(t, b), take); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out.Bytes(), d) {
		t.Fatal("taking every tensor did not reproduce the donor")
	}
}

func TestSpliceRefuses(t *testing.T) {
	b := buildGGUF(t, 32, baseFx)
	other := buildGGUF(t, 32, []fixture{{"blk.0.ffn_gate_exps.weight", []uint64{256, 4, 4}, Q8_0, 1}})
	missing := buildGGUF(t, 32, []fixture{{"output.weight", []uint64{64, 3}, Q8_0, 1}})
	cases := []struct {
		name string
		take map[string]Source
		want string
	}{
		{"shape", map[string]Source{"blk.0.ffn_gate_exps.weight": source(t, other)}, "shape"},
		{"donor lacks it", map[string]Source{"blk.0.ffn_gate_exps.weight": source(t, missing)}, "donor has no tensor"},
		{"not in base", map[string]Source{"blk.9.ffn_gate_exps.weight": source(t, other)}, "not a tensor of the base"},
	}
	for _, c := range cases {
		var out bytes.Buffer
		_, err := Splice(&out, source(t, b), c.take)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Fatalf("%s: err = %v, want %q", c.name, err, c.want)
		}
	}
}
