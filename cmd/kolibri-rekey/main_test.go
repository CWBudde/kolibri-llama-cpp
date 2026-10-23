package main

import (
	"bytes"
	"crypto/sha256"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"kolibri-llm/internal/gguf"
)

// The synthetic checkpoint: our converter's keys for a 4-layer model with
// layers 0, 1 and 3 sliding (and so with RoPE), a tokenizer with string and
// int arrays, and three tensors whose data is a byte pattern.
func fixtureKV() map[string]any {
	pattern := []bool{true, true, false, true}
	return map[string]any{
		"general.architecture":                     "kolibri",
		"general.alignment":                        uint32(32),
		"general.name":                             "Kolibri 1 tiny",
		"kolibri.block_count":                      uint32(4),
		"kolibri.context_length":                   uint64(262144),
		"kolibri.rope.freq_base":                   float32(10000),
		"kolibri.attention.layer_norm_rms_epsilon": float32(1e-6),
		"kolibri.attention.sliding_window":         uint32(513),
		"kolibri.attention.sliding_window_pattern": pattern,
		"kolibri.attention.rope_pattern":           pattern,
		"kolibri.expert_weights_norm":              false,
		"kolibri.n_deepstack":                      int32(-1),
		"kolibri.attention.head_count_kv":          []int32{4, 4, 4, 4},
		"tokenizer.ggml.model":                     "gpt2",
		"tokenizer.ggml.pre":                       "kolibri",
		"tokenizer.ggml.tokens":                    []string{"<s>", "a", "b", "ab", strings.Repeat("z", 200)},
		"tokenizer.ggml.token_type":                []int32{3, 1, 1, 1, 1},
		"tokenizer.ggml.merges":                    []string{"a b"},
		"tokenizer.ggml.bos_token_id":              uint32(0),
		"tokenizer.chat_template":                  "{% for m in messages %}{{ m.content }}{% endfor %}",
	}
}

var fixtureTensors = []gguf.Tensor{
	{Name: "token_embd.weight", Dims: []uint64{64, 5}, Type: gguf.F32, Offset: 0},
	{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{256, 2, 3}, Type: gguf.IQ3_XXS, Offset: 1280},
	{Name: "output_norm.weight", Dims: []uint64{64}, Type: gguf.F32, Offset: 1888}, // 1280 + 3*2*98, aligned to 32
}

// writeFixture writes a GGUF with the given metadata and fixtureTensors,
// the data being byte i = i*7 mod 251, and returns its path.
func writeFixture(t *testing.T, kv map[string]any) string {
	t.Helper()
	var buf bytes.Buffer
	data, err := gguf.WriteHeader(&buf, 32, kv, fixtureTensors)
	if err != nil {
		t.Fatal(err)
	}
	last := fixtureTensors[len(fixtureTensors)-1]
	n, err := last.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	b := make([]byte, data+int64(last.Offset)+n)
	copy(b, buf.Bytes())
	for i := data; i < int64(len(b)); i++ {
		b[i] = byte(i * 7 % 251)
	}
	path := filepath.Join(t.TempDir(), "in.gguf")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func sum(t *testing.T, path string, from int64) [32]byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return sha256.Sum256(b[from:])
}

func opts(in, out string) options {
	return options{in: in, out: out, arch: "kolibri1", pre: "qwen2", copy: true}
}

func TestRekey(t *testing.T) {
	in := writeFixture(t, fixtureKV())
	src, err := gguf.Open(in)
	if err != nil {
		t.Fatal(err)
	}
	whole := sum(t, in, 0)
	out := filepath.Join(t.TempDir(), "out.gguf")
	res, err := run(opts(in, out))
	if err != nil {
		t.Fatal(err)
	}
	if sum(t, in, 0) != whole {
		t.Fatal("input modified")
	}
	f, err := gguf.Open(out)
	if err != nil {
		t.Fatal(err)
	}
	if f.DataOffset != src.DataOffset || res.DataOffset != src.DataOffset {
		t.Fatalf("data offset %d (reported %d), input %d", f.DataOffset, res.DataOffset, src.DataOffset)
	}
	if f.HeaderEnd <= f.DataOffset-int64(f.Alignment) || f.HeaderEnd > f.DataOffset {
		t.Fatalf("header ends at %d, data at %d", f.HeaderEnd, f.DataOffset)
	}
	if sum(t, out, f.DataOffset) != sum(t, in, src.DataOffset) {
		t.Fatal("data section changed")
	}
	si, _ := os.Stat(in)
	so, _ := os.Stat(out)
	if si.Size() != so.Size() {
		t.Fatalf("size %d, input %d", so.Size(), si.Size())
	}
	ib, _ := os.ReadFile(in)
	ob, _ := os.ReadFile(out)
	if !bytes.Equal(ib[src.InfoOffset:src.HeaderEnd], ob[f.InfoOffset:f.HeaderEnd]) {
		t.Fatal("tensor infos differ")
	}
	if !bytes.Equal(ob[f.HeaderEnd:f.DataOffset], make([]byte, f.DataOffset-f.HeaderEnd)) {
		t.Fatal("fill before the data section is not zero")
	}

	// every input pair but rope_pattern, in input order, renamed, with the
	// same bytes except for the two rewritten strings; the padding key last
	got := f.Metadata
	if n := len(got); n != len(src.Metadata) || got[n-1].Key != paddingKey {
		t.Fatalf("%d pairs, last %s; want %d with %s last", n, got[n-1].Key, len(src.Metadata), paddingKey)
	}
	if s, ok := got[len(got)-1].String(); !ok || len(s) != res.Padding {
		t.Fatalf("padding value %q, reported %d bytes", s, res.Padding)
	}
	i := 0
	for _, kv := range src.Metadata {
		if kv.Key == "kolibri.attention.rope_pattern" {
			continue
		}
		g := got[i]
		i++
		wantKey := kv.Key
		if rest, ok := strings.CutPrefix(kv.Key, "kolibri."); ok {
			wantKey = "kolibri1." + rest
		}
		if g.Key != wantKey || g.Type != kv.Type {
			t.Errorf("pair %d = %s type %d, want %s type %d", i, g.Key, g.Type, wantKey, kv.Type)
			continue
		}
		switch g.Key {
		case "general.architecture", "tokenizer.ggml.pre":
		default:
			if !bytes.Equal(g.Value, kv.Value) {
				t.Errorf("%s changed", g.Key)
			}
		}
	}
	if a, _ := f.String("general.architecture"); a != "kolibri1" {
		t.Errorf("architecture %q", a)
	}
	if p, _ := f.String("tokenizer.ggml.pre"); p != "qwen2" {
		t.Errorf("pre %q", p)
	}
	if n, _ := f.Uint("kolibri1.attention.sliding_window"); n != 513 {
		t.Errorf("sliding_window %d", n)
	}
	if res.Renamed != 9 || len(res.Dropped) != 1 || res.Dropped[0] != "kolibri.attention.rope_pattern" {
		t.Errorf("renamed %d, dropped %v", res.Renamed, res.Dropped)
	}
	if len(f.Tensors) != len(src.Tensors) {
		t.Fatalf("%d tensors", len(f.Tensors))
	}
	leftovers(t, filepath.Dir(out), 1)
}

// leftovers checks that dir holds n files: no temporary file is left.
func leftovers(t *testing.T, dir string, n int) {
	t.Helper()
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(ents) != n {
		var names []string
		for _, e := range ents {
			names = append(names, e.Name())
		}
		t.Errorf("%s holds %v, want %d files", dir, names, n)
	}
}

func TestRefuses(t *testing.T) {
	for _, c := range []struct {
		name string
		edit func(kv map[string]any)
		want string
	}{
		{"rope pattern differs", func(kv map[string]any) {
			kv["kolibri.attention.rope_pattern"] = []bool{true, true, true, true}
		}, "not element-wise equal"},
		{"rope pattern without window pattern", func(kv map[string]any) {
			delete(kv, "kolibri.attention.sliding_window_pattern")
		}, "sliding_window_pattern"},
		{"not kolibri", func(kv map[string]any) {
			kv["general.architecture"] = "kolibri1"
		}, "architecture"},
		{"no architecture", func(kv map[string]any) {
			delete(kv, "general.architecture")
		}, "architecture"},
		{"renamed key exists", func(kv map[string]any) {
			kv["kolibri1.block_count"] = uint32(4)
		}, "twice"},
		// without rope_pattern to drop, the renamed keys and the padding
		// key outgrow the header's slack
		{"header outgrows its slack", func(kv map[string]any) {
			delete(kv, "kolibri.attention.rope_pattern")
		}, "does not fit"},
	} {
		t.Run(c.name, func(t *testing.T) {
			kv := fixtureKV()
			c.edit(kv)
			in := writeFixture(t, kv)
			dir := t.TempDir()
			_, err := run(opts(in, filepath.Join(dir, "out.gguf")))
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("err = %v, want one mentioning %q", err, c.want)
			}
			t.Log(err)
			leftovers(t, dir, 0)
		})
	}
}

func TestRefusesExistingOutput(t *testing.T) {
	in := writeFixture(t, fixtureKV())
	out := filepath.Join(t.TempDir(), "out.gguf")
	if err := os.WriteFile(out, []byte("keep"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := run(opts(in, out)); err == nil || !strings.Contains(err.Error(), "exists") {
		t.Fatalf("err = %v", err)
	}
	if b, _ := os.ReadFile(out); string(b) != "keep" {
		t.Fatal("existing output overwritten")
	}
	leftovers(t, filepath.Dir(out), 1)
}

// TestPreOnlyKolibri leaves a pre-tokenizer other than kolibri unchanged.
func TestPreOnlyKolibri(t *testing.T) {
	kv := fixtureKV()
	kv["tokenizer.ggml.pre"] = "llama-bpe"
	in := writeFixture(t, kv)
	out := filepath.Join(t.TempDir(), "out.gguf")
	if _, err := run(opts(in, out)); err != nil {
		t.Fatal(err)
	}
	f, err := gguf.Open(out)
	if err != nil {
		t.Fatal(err)
	}
	if p, _ := f.String("tokenizer.ggml.pre"); p != "llama-bpe" {
		t.Errorf("pre %q", p)
	}
}

// TestClone runs the default path, cp -c, on macOS (APFS).
func TestClone(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("cp -c is macOS's")
	}
	in := writeFixture(t, fixtureKV())
	out := filepath.Join(t.TempDir(), "out.gguf")
	o := opts(in, out)
	o.copy = false
	res, err := run(o)
	if err != nil {
		t.Fatal(err)
	}
	if sum(t, out, res.DataOffset) != sum(t, in, res.DataOffset) {
		t.Fatal("data section changed")
	}
}
