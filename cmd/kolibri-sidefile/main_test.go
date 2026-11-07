package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"kolibri-llm/internal/gguf"
)

// The fixture has the shape of cmd/kolibri-tiny -router converted to IQ3, as
// far as the side-file cares: a different type and row size for down, the
// tensors in the file's own order (down first), and other tensors between the
// layers. Converting the real tiny checkpoint takes Python and llama.cpp's
// converter: tools/gguf/check_expert_cache.py runs this tool on it, and
// llama.cpp reads the result.
const (
	tLayers  = 3
	tExperts = 8
	tGate    = 2 * 98 * 8 // [512, 8, n]: 512/256 blocks of 98 bytes, 8 rows
	tDown    = 136 * 16   // [256, 16, n]: one block of 136 bytes, 16 rows
)

func writeGGUF(t *testing.T, dir string) (string, *gguf.File) {
	t.Helper()
	var tensors []gguf.Tensor
	var off uint64
	add := func(name string, dims []uint64, typ gguf.Type) {
		tt := gguf.Tensor{Name: name, Dims: dims, Type: typ, Offset: off}
		n, err := tt.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		tensors = append(tensors, tt)
		off += (uint64(n) + 31) / 32 * 32
	}
	for il := range tLayers {
		add(fmt.Sprintf("blk.%d.ffn_down_exps.weight", il), []uint64{256, 16, tExperts}, gguf.IQ4_XS)
		add(fmt.Sprintf("blk.%d.attn_norm.weight", il), []uint64{40}, gguf.F32)
		add(fmt.Sprintf("blk.%d.ffn_gate_exps.weight", il), []uint64{512, 8, tExperts}, gguf.IQ3_XXS)
		add(fmt.Sprintf("blk.%d.ffn_up_exps.weight", il), []uint64{512, 8, tExperts}, gguf.IQ3_XXS)
	}
	kv := map[string]any{
		"general.architecture":  "kolibri1",
		"kolibri1.block_count":  uint32(tLayers),
		"kolibri1.expert_count": uint32(tExperts),
		"tokenizer.ggml.tokens": []string{"x", "y"},
	}
	var buf bytes.Buffer
	data, err := gguf.WriteHeader(&buf, 32, kv, tensors)
	if err != nil {
		t.Fatal(err)
	}
	total := data + int64(off)
	b := make([]byte, total)
	copy(b, buf.Bytes())
	for i := data; i < total; i++ {
		b[i] = byte(i*7%251) | 1 // never zero, so padding cannot pass for data
	}
	path := filepath.Join(dir, "m.gguf")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	return path, f
}

func setup(t *testing.T) (ggufPath, side string, f *gguf.File, m *manifest) {
	t.Helper()
	dir := t.TempDir()
	ggufPath, f = writeGGUF(t, dir)
	side = ggufPath + ".experts"
	m, err := write(ggufPath, side, false)
	if err != nil {
		t.Fatal(err)
	}
	return ggufPath, side, f, m
}

func mustVerify(t *testing.T, ggufPath, side string) {
	t.Helper()
	if _, err := verify(ggufPath, side); err != nil {
		t.Fatalf("verify of a good side-file: %v", err)
	}
}

func wantFault(t *testing.T, ggufPath, side, what string) {
	t.Helper()
	_, err := verify(ggufPath, side)
	if err == nil {
		t.Fatalf("verify passed, want a fault with %q", what)
	}
	if !strings.Contains(err.Error(), what) {
		t.Fatalf("verify: %v, want a fault with %q", err, what)
	}
}

// TestLayout checks the bytes against the GGUF directly, at the offsets the
// expert cache's GGUF path reads, not through the tool's own reader: gate, up
// and down at offs[k] + e*nb[2], then zeros to the alignment.
func TestLayout(t *testing.T) {
	ggufPath, side, f, m := setup(t)
	mustVerify(t, ggufPath, side)
	g, err := os.ReadFile(ggufPath)
	if err != nil {
		t.Fatal(err)
	}
	s, err := os.ReadFile(side)
	if err != nil {
		t.Fatal(err)
	}
	if int64(len(s)) != m.Size || len(s)%alignment != 0 {
		t.Fatalf("%d bytes, manifest says %d, alignment %d", len(s), m.Size, alignment)
	}
	sum := sha256.Sum256(s)
	if hex.EncodeToString(sum[:]) != m.SHA256 {
		t.Error("the manifest's sha256 is not the file's")
	}
	rows := int64(2*tGate + tDown)
	ext := pad(rows)
	start := pad(headerBytes + tLayers*3*tensorBytes + tLayers*tExperts*8)
	for il := range tLayers {
		for e := range tExperts {
			at := start + (int64(il)*tExperts+int64(e))*ext
			for _, kind := range gguf.ExpertKinds {
				tt, _ := f.Tensor(fmt.Sprintf("blk.%d.ffn_%s_exps.weight", il, kind))
				n, _ := tt.Bytes()
				slice := n / tExperts
				off := f.DataOffset + int64(tt.Offset) + int64(e)*slice
				if !bytes.Equal(s[at:at+slice], g[off:off+slice]) {
					t.Fatalf("layer %d expert %d %s: not the GGUF's bytes at %d", il, e, kind, off)
				}
				at += slice
			}
			if end := start + (int64(il)*tExperts+int64(e)+1)*ext; !allZero(s[at:end]) {
				t.Fatalf("layer %d expert %d: padding is not zero", il, e)
			}
		}
	}
	// the sizes differ per tensor: the row size of down is not the gate's
	tt := m.Tensors[0]
	if tt[0].RowBytes != tGate || tt[1].RowBytes != tGate || tt[2].RowBytes != tDown || tt[2].Type != "IQ4_XS" {
		t.Errorf("tensor table %+v", tt)
	}
}

// TestRefuses is every way the data or the manifest can be wrong.
func TestFlippedByte(t *testing.T) {
	for _, tc := range []struct {
		name string
		at   func(p *plan) int64
		want string
	}{
		{"first row", func(p *plan) int64 { return p.DataStart }, "layer 0 expert 0: extent differs"},
		{"down row of a middle expert", func(p *plan) int64 { return p.Offsets[1][3] + 2*tGate + 5 }, "layer 1 expert 3: extent differs"},
		{"padding of the last extent", func(p *plan) int64 { return p.FileSize - 1 }, fmt.Sprintf("layer %d expert %d: extent differs", tLayers-1, tExperts-1)},
		{"the checksum", func(p *plan) int64 { return checksumAt }, "table: differs"},
		{"the tensor table", func(p *plan) int64 { return headerBytes }, "table: differs"},
		{"an extent offset", func(p *plan) int64 { return p.TableBytes - 8 }, "table: differs"},
		{"table padding", func(p *plan) int64 { return p.DataStart - 1 }, "table padding"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ggufPath, side, _, _ := setup(t)
			p, err := planFor(ggufPath)
			if err != nil {
				t.Fatal(err)
			}
			flip(t, side, tc.at(p))
			wantFault(t, ggufPath, side, tc.want)
		})
	}
}

func flip(t *testing.T, path string, at int64) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var b [1]byte
	if _, err := f.ReadAt(b[:], at); err != nil {
		t.Fatal(err)
	}
	b[0] ^= 0x10
	if _, err := f.WriteAt(b[:], at); err != nil {
		t.Fatal(err)
	}
}

func TestTruncated(t *testing.T) {
	ggufPath, side, _, m := setup(t)
	if err := os.Truncate(side, m.Size-alignment); err != nil {
		t.Fatal(err)
	}
	wantFault(t, ggufPath, side, "truncated")
	if err := os.Truncate(side, m.Size+1); err != nil {
		t.Fatal(err)
	}
	wantFault(t, ggufPath, side, "bytes")
}

// TestEditedManifest edits every part of the manifest: verify takes nothing
// from it on trust.
func TestEditedManifest(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(m *manifest)
		want string
	}{
		{"file hash", func(m *manifest) { m.SHA256 = strings.Repeat("0", 64) }, "file sha256"},
		{"table hash", func(m *manifest) { m.TableSHA256 = strings.Repeat("0", 64) }, "table sha256"},
		{"extent hash", func(m *manifest) { m.Extents[1][2] = strings.Repeat("0", 64) }, "layer 1 expert 2: sha256"},
		{"missing extents", func(m *manifest) { m.Extents[2] = m.Extents[2][:tExperts-1] }, "extent hashes"},
		{"size", func(m *manifest) { m.Size++ }, "manifest: size"},
		{"version", func(m *manifest) { m.Version = 2 }, "version 2"},
		{"alignment", func(m *manifest) { m.Alignment = 4096 }, "alignment 4096"},
		{"layers", func(m *manifest) { m.Layers = 2 }, "2 layers"},
		{"experts", func(m *manifest) { m.Experts = 7 }, "x 7 experts"},
		{"GGUF size", func(m *manifest) { m.GGUF.Size++ }, "GGUF identity"},
		{"GGUF header", func(m *manifest) { m.GGUF.HeaderSHA256 = strings.Repeat("0", 64) }, "GGUF identity"},
		{"row bytes", func(m *manifest) { m.Tensors[0][2].RowBytes-- }, "tensor table"},
		{"type", func(m *manifest) { m.Tensors[0][2].Type = "IQ3_XXS" }, "tensor table"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ggufPath, side, _, _ := setup(t)
			m, err := loadManifest(side + manifestName)
			if err != nil {
				t.Fatal(err)
			}
			tc.edit(m)
			data, err := json.Marshal(m)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(side+manifestName, data, 0o644); err != nil {
				t.Fatal(err)
			}
			wantFault(t, ggufPath, side, tc.want)
		})
	}
}

func TestCorruptManifest(t *testing.T) {
	ggufPath, side, _, _ := setup(t)
	if err := os.WriteFile(side+manifestName, []byte(`{"format": "KOLIEXPT", "unknown": 1}`), 0o644); err != nil {
		t.Fatal(err)
	}
	wantFault(t, ggufPath, side, "unknown")
	if err := os.WriteFile(side+manifestName, []byte(`{not json`), 0o644); err != nil {
		t.Fatal(err)
	}
	wantFault(t, ggufPath, side, "manifest")
	if err := os.Remove(side + manifestName); err != nil {
		t.Fatal(err)
	}
	wantFault(t, ggufPath, side, "no such file")
}

// TestOtherGGUF checks the side-file's identity against a GGUF that differs
// only in one byte of its header, then one byte of its data, then in length.
func TestOtherGGUF(t *testing.T) {
	ggufPath, side, f, _ := setup(t)
	flip(t, ggufPath, f.DataOffset/2)
	wantFault(t, ggufPath, side, "GGUF identity")
	flip(t, ggufPath, f.DataOffset/2)
	mustVerify(t, ggufPath, side)

	tt, _ := f.Tensor("blk.2.ffn_up_exps.weight")
	flip(t, ggufPath, f.FileOffset(tt)+7)
	wantFault(t, ggufPath, side, "layer 2 expert 0: extent differs")
	flip(t, ggufPath, f.FileOffset(tt)+7)
	mustVerify(t, ggufPath, side)

	g, err := os.OpenFile(ggufPath, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := g.Write(make([]byte, 32)); err != nil {
		t.Fatal(err)
	}
	g.Close()
	wantFault(t, ggufPath, side, "GGUF identity")
}

func TestWriteRefusesToReplace(t *testing.T) {
	ggufPath, side, _, _ := setup(t)
	if _, err := write(ggufPath, side, false); err == nil || !strings.Contains(err.Error(), "-force") {
		t.Fatalf("second write: %v", err)
	}
	flip(t, side, 100)
	if _, err := write(ggufPath, side, true); err != nil {
		t.Fatal(err)
	}
	mustVerify(t, ggufPath, side)
	if _, err := os.Stat(side + ".tmp"); !os.IsNotExist(err) {
		t.Errorf("temporary file left: %v", err)
	}
}

// TestAlignedRead is what the miss path relies on: every extent and the file
// end on the alignment, so one read of a whole extent is an aligned read
// and never meets the end of the file.
func TestAlignedRead(t *testing.T) {
	ggufPath, _, _, _ := setup(t)
	p, err := planFor(ggufPath)
	if err != nil {
		t.Fatal(err)
	}
	end := p.DataStart
	for il, rows := range p.Offsets {
		for e, off := range rows {
			if off%alignment != 0 || off != end {
				t.Errorf("layer %d expert %d at %d, want %d, aligned to %d", il, e, off, end, alignment)
			}
			end += p.extent(il)
		}
	}
	if end != p.FileSize || p.FileSize%alignment != 0 {
		t.Errorf("file of %d bytes, extents end at %d", p.FileSize, end)
	}
}
