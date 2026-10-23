// Command kolibri-rekey rewrites the metadata of one of our Kolibri GGUFs
// for upstream llama.cpp's Kolibri support (#30673), for PLAN Phase 11,
// Stage E0: upstream's architecture is kolibri1, has no rope_pattern key (it
// applies RoPE exactly on the sliding-window layers) and maps the tokenizer
// to the qwen2 pre-tokenizer. The tensors are the same.
//
//	go run ./cmd/kolibri-rekey -in ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf \
//	    -out ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.kolibri1.gguf
//
// The new header: general.architecture = -arch, every kolibri.X key renamed
// <arch>.X, tokenizer.ggml.pre kolibri replaced by -pre, and
// kolibri.attention.rope_pattern dropped, after checking that it equals
// kolibri.attention.sliding_window_pattern element by element (else exit 1).
// Every other pair keeps its order, type and bytes; the tensor infos are
// copied byte for byte.
//
// The original files are 33 to 156 GB and the disk has room for none of
// them, so the output is an APFS clone of the input (cp -c) whose header is
// overwritten in place. The new header must end where the data section
// still starts at the input's offset: a string pair general.rekey.padding is
// sized so that the header ends within the alignment before that offset, and
// zero bytes fill up to it. Only the header's blocks stop being shared with
// the input. -copy copies the file byte by byte instead (any file system;
// the tests use it).
//
// The output is written under a temporary name next to -out, re-read and
// checked (architecture, data offset, tensor infos, every pair), and only
// then linked to -out, so a failure leaves no half file. -out must not exist.
package main

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"

	"kolibri-llm/internal/gguf"
)

const (
	ourArch    = "kolibri"
	ropeKey    = "kolibri.attention.rope_pattern"
	swaKey     = "kolibri.attention.sliding_window_pattern"
	paddingKey = "general.rekey.padding"
)

type options struct {
	in, out   string
	arch, pre string
	copy      bool // plain byte copy instead of an APFS clone
}

// result is what run did, for the summary.
type result struct {
	Renamed    int      // kolibri.* keys renamed
	Dropped    []string // keys dropped
	Pre        string   // tokenizer.ggml.pre in the input
	Padding    int      // bytes in the padding key's string
	Fill       int64    // zero bytes between the header and the data section
	HeaderEnd  int64
	DataOffset int64 // the input's and the output's
}

func main() {
	var o options
	flag.StringVar(&o.in, "in", "", "input GGUF (general.architecture = kolibri)")
	flag.StringVar(&o.out, "out", "", "output GGUF; must not exist")
	flag.StringVar(&o.arch, "arch", "kolibri1", "new general.architecture and key prefix")
	flag.StringVar(&o.pre, "pre", "qwen2", "replaces tokenizer.ggml.pre = kolibri")
	flag.BoolVar(&o.copy, "copy", false, "copy the file byte by byte instead of cloning it (cp -c)")
	flag.Parse()
	if o.in == "" || o.out == "" || o.arch == "" || o.pre == "" || flag.NArg() != 0 {
		flag.Usage()
		os.Exit(2)
	}
	res, err := run(o)
	if err != nil {
		log.Fatal(err)
	}
	how := "clone"
	if o.copy {
		how = "copy"
	}
	fmt.Printf("%s -> %s (%s)\n", o.in, o.out, how)
	fmt.Printf("  general.architecture %s -> %s\n", ourArch, o.arch)
	fmt.Printf("  renamed %d keys %s.* -> %s.*\n", res.Renamed, ourArch, o.arch)
	if res.Pre == ourArch {
		fmt.Printf("  tokenizer.ggml.pre %s -> %s\n", res.Pre, o.pre)
	} else {
		fmt.Printf("  tokenizer.ggml.pre %q kept\n", res.Pre)
	}
	for _, k := range res.Dropped {
		fmt.Printf("  dropped %s (equal to %s)\n", k, swaKey)
	}
	fmt.Printf("  padding: %s of %d bytes, then %d zero bytes\n", paddingKey, res.Padding, res.Fill)
	fmt.Printf("  header ends at %d, data section at %d (unchanged)\n", res.HeaderEnd, res.DataOffset)
}

func run(o options) (*result, error) {
	if _, err := os.Lstat(o.out); err == nil {
		return nil, fmt.Errorf("%s exists", o.out)
	} else if !errors.Is(err, fs.ErrNotExist) {
		return nil, err
	}
	src, err := gguf.Open(o.in)
	if err != nil {
		return nil, err
	}
	if a, _ := src.String("general.architecture"); a != ourArch {
		return nil, fmt.Errorf("%s: general.architecture is %q, want %q", o.in, a, ourArch)
	}
	kvs, res, err := rekey(src.Metadata, o.arch, o.pre)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", o.in, err)
	}
	infos, err := readAt(o.in, src.InfoOffset, src.HeaderEnd-src.InfoOffset)
	if err != nil {
		return nil, err
	}
	kvs, header, err := layout(src, kvs, infos)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", o.in, err)
	}
	res.Padding = len(kvs[len(kvs)-1].Value) - 8
	res.HeaderEnd = int64(len(header))
	res.DataOffset = src.DataOffset
	res.Fill = src.DataOffset - res.HeaderEnd
	header = append(header, make([]byte, res.Fill)...)

	tmp := filepath.Join(filepath.Dir(o.out), "."+filepath.Base(o.out)+".rekey-tmp")
	if _, err := os.Lstat(tmp); err == nil {
		return nil, fmt.Errorf("%s exists (left by an earlier run?)", tmp)
	}
	if o.copy {
		err = copyFile(o.in, tmp)
	} else {
		err = cloneFile(o.in, tmp)
	}
	if err != nil {
		os.Remove(tmp) // the clone or copy may have created it
		return nil, err
	}
	done := false
	defer func() {
		if !done {
			os.Remove(tmp)
		}
	}()
	if err := overwrite(tmp, header); err != nil {
		return nil, err
	}
	if err := verify(o.in, src, infos, tmp, o.arch, kvs); err != nil {
		return nil, fmt.Errorf("%s: %w", tmp, err)
	}
	// a link, unlike a rename, fails if -out appeared meanwhile
	if err := os.Link(tmp, o.out); err != nil {
		return nil, err
	}
	done = true
	return res, os.Remove(tmp)
}

// rekey builds the new metadata from the input's, in its order: the
// architecture and prefix -arch, pre kolibri -> -pre, rope_pattern dropped
// if it equals sliding_window_pattern.
func rekey(meta []gguf.KV, arch, pre string) ([]gguf.KV, *result, error) {
	res := &result{}
	byKey := map[string]gguf.KV{}
	for _, kv := range meta {
		byKey[kv.Key] = kv
	}
	seen := map[string]bool{}
	var out []gguf.KV
	for _, kv := range meta {
		switch {
		case kv.Key == ropeKey:
			swa, ok := byKey[swaKey]
			if !ok {
				return nil, nil, fmt.Errorf("%s present without %s, so it cannot be dropped", ropeKey, swaKey)
			}
			if kv.Type != swa.Type || !bytes.Equal(kv.Value, swa.Value) {
				return nil, nil, fmt.Errorf("%s %s is not element-wise equal to %s %s; upstream applies RoPE exactly on the sliding-window layers", ropeKey, show(kv), swaKey, show(swa))
			}
			res.Dropped = append(res.Dropped, kv.Key)
			continue
		case kv.Key == "general.architecture":
			kv = strKV(kv.Key, arch)
		case kv.Key == "tokenizer.ggml.pre":
			res.Pre, _ = kv.String()
			if res.Pre == ourArch {
				kv = strKV(kv.Key, pre)
			}
		case strings.HasPrefix(kv.Key, ourArch+"."):
			kv.Key = arch + strings.TrimPrefix(kv.Key, ourArch)
			res.Renamed++
		}
		if kv.Key == paddingKey || seen[kv.Key] {
			return nil, nil, fmt.Errorf("key %s twice in the new header", kv.Key)
		}
		seen[kv.Key] = true
		out = append(out, kv)
	}
	return out, res, nil
}

// show prints a bool array as 0s and 1s, any other value as its size.
func show(kv gguf.KV) string {
	const tArray, tBool = 9, 7
	v := kv.Value
	if kv.Type != tArray || len(v) < 12 || v[0] != tBool || v[1] != 0 || v[2] != 0 || v[3] != 0 {
		return fmt.Sprintf("(type %d, %d bytes)", kv.Type, len(v))
	}
	var b strings.Builder
	for _, x := range v[12:] {
		b.WriteByte('0' + min(x, 1))
	}
	return b.String()
}

// strKV encodes a string pair, which cannot fail.
func strKV(key, v string) gguf.KV {
	kv, err := gguf.NewKV(key, v)
	if err != nil {
		panic(err)
	}
	return kv
}

// layout appends the padding pair to kvs and encodes the header (without
// the zero fill): its string is the shortest that ends the header within
// the alignment before the input's data offset, so that the data section
// starts at the same offset.
func layout(src *gguf.File, kvs []gguf.KV, infos []byte) ([]gguf.KV, []byte, error) {
	n := uint64(len(src.Tensors))
	d, a := src.DataOffset, int64(src.Alignment)
	kvs = append(slices.Clip(kvs), strKV(paddingKey, ""))
	base := int64(len(gguf.EncodeHeader(src.Version, n, kvs, infos)))
	if base > d {
		return nil, nil, fmt.Errorf("the new header does not fit: %d bytes with an empty %s, but the data section starts at %d", base, paddingKey, d)
	}
	pad := max(0, d-a+1-base)
	kvs[len(kvs)-1] = strKV(paddingKey, strings.Repeat(" ", int(pad)))
	h := gguf.EncodeHeader(src.Version, n, kvs, infos)
	if end := int64(len(h)); end <= d-a || end > d {
		return nil, nil, fmt.Errorf("header ends at %d, want (%d, %d]", end, d-a, d)
	}
	return kvs, h, nil
}

// cloneFile clones src to dst with APFS clonefile (cp -c): dst shares src's
// blocks until either is written.
func cloneFile(src, dst string) error {
	if out, err := exec.Command("cp", "-c", src, dst).CombinedOutput(); err != nil {
		return fmt.Errorf("cp -c %s %s: %v: %s (no clonefile on this file system? -copy copies)", src, dst, err, bytes.TrimSpace(out))
	}
	return nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	if err := out.Sync(); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

// overwrite writes b at the start of the file at path, in place.
func overwrite(path string, b []byte) error {
	f, err := os.OpenFile(path, os.O_WRONLY, 0)
	if err != nil {
		return err
	}
	if _, err := f.WriteAt(b, 0); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

func readAt(path string, off, n int64) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	b := make([]byte, n)
	if _, err := f.ReadAt(b, off); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return b, nil
}

// verify re-reads the written file at path: same version, alignment, size and data
// offset as the input, the architecture, the tensor infos byte for byte, the
// pairs kvs exactly and zeros up to the data section.
func verify(in string, src *gguf.File, infos []byte, path, arch string, kvs []gguf.KV) error {
	f, err := gguf.Open(path)
	if err != nil {
		return err
	}
	if f.Version != src.Version || f.Alignment != src.Alignment || f.DataOffset != src.DataOffset {
		return fmt.Errorf("version %d, alignment %d, data offset %d; input %d, %d, %d",
			f.Version, f.Alignment, f.DataOffset, src.Version, src.Alignment, src.DataOffset)
	}
	si, err := os.Stat(path)
	if err != nil {
		return err
	}
	ii, err := os.Stat(in)
	if err != nil {
		return err
	}
	if si.Size() != ii.Size() {
		return fmt.Errorf("%d bytes, input %d", si.Size(), ii.Size())
	}
	if a, _ := f.String("general.architecture"); a != arch {
		return fmt.Errorf("general.architecture %q, want %q", a, arch)
	}
	if !slices.EqualFunc(f.Tensors, src.Tensors, func(a, b gguf.Tensor) bool {
		return a.Name == b.Name && a.Type == b.Type && a.Offset == b.Offset && slices.Equal(a.Dims, b.Dims)
	}) {
		return errors.New("tensor infos differ from the input's")
	}
	got, err := readAt(path, f.InfoOffset, f.DataOffset-f.InfoOffset)
	if err != nil {
		return err
	}
	if int64(len(infos)) != f.HeaderEnd-f.InfoOffset || !bytes.Equal(got[:len(infos)], infos) {
		return errors.New("tensor info bytes differ from the input's")
	}
	if slices.ContainsFunc(got[len(infos):], func(b byte) bool { return b != 0 }) {
		return errors.New("non-zero bytes between the header and the data section")
	}
	if !slices.EqualFunc(f.Metadata, kvs, func(a, b gguf.KV) bool {
		return a.Key == b.Key && a.Type == b.Type && bytes.Equal(a.Value, b.Value)
	}) {
		return errors.New("metadata differs from what was written")
	}
	return nil
}
