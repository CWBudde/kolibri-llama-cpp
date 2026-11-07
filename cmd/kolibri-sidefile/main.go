// Command kolibri-sidefile writes and verifies the expert side-file of a GGUF
// (PLAN Phase 11, Stage E3): every expert's gate, up and down rows, contiguous
// in one aligned extent, so that the expert cache reads a miss with one pread
// (--expert-cache-sidefile, patch 0014; docs/expert-streaming.md).
//
//	kolibri-sidefile write  -gguf ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf
//	kolibri-sidefile verify -gguf ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf
//
// The file goes next to the GGUF as <gguf>.experts unless -o names another;
// its manifest is <side-file>.manifest.json: sha256 of every extent and of the
// whole file, and the GGUF's identity (size and a hash of its header).
//
// verify rehashes everything and compares every extent byte for byte with the
// GGUF, through the same offsets the expert cache's GGUF path reads. It is the
// only full check of the data: llama.cpp's loader checks the table, not 31 GB
// of experts.
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"reflect"
	"strings"
	"time"

	"kolibri-llm/internal/gguf"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("kolibri-sidefile: ")
	if len(os.Args) < 2 {
		usage()
	}
	cmd, args := os.Args[1], os.Args[2:]
	fs := flag.NewFlagSet(cmd, flag.ExitOnError)
	ggufPath := fs.String("gguf", "", "the GGUF")
	out := fs.String("o", "", "the side-file (default: <gguf>.experts)")
	force := fs.Bool("force", false, "write: replace an existing side-file")
	if err := fs.Parse(args); err != nil || *ggufPath == "" || fs.NArg() > 0 {
		usage()
	}
	side := *out
	if side == "" {
		side = *ggufPath + ".experts"
	}
	t0 := time.Now()
	switch cmd {
	case "write":
		m, err := write(*ggufPath, side, *force)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("wrote %s: %d bytes, %d layers x %d experts, sha256 %s, %.1f s\n", side, m.Size, m.Layers, m.Experts, m.SHA256, time.Since(t0).Seconds())
		fmt.Printf("manifest %s%s\n", side, manifestName)
	case "verify":
		m, err := verify(*ggufPath, side)
		if err != nil {
			log.Fatalf("%s: FAILED: %v", side, err)
		}
		fmt.Printf("verified %s: %d bytes, %d layers x %d experts, sha256 %s, every extent byte-identical to %s, %.1f s\n",
			side, m.Size, m.Layers, m.Experts, m.SHA256, *ggufPath, time.Since(t0).Seconds())
	default:
		usage()
	}
	report(side)
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: kolibri-sidefile write|verify -gguf FILE [-o SIDE-FILE] [-force]")
	os.Exit(2)
}

// report drops what the tool left in the page cache and says so: a benchmark
// must not read a side-file from memory (kolibri-bench evicts it too).
func report(side string) {
	for _, p := range []string{side, side + manifestName} {
		if err := dropCache(p); err != nil {
			log.Printf("%s: left in the page cache: %v", p, err)
		}
	}
}

// copier reads extents out of a GGUF.
type copier struct {
	p    *plan
	gguf *os.File
}

// readExtent fills buf (an extent of layer il) with expert e's rows from the
// GGUF, at the offsets the cache's GGUF path reads: offs[k] + e*nb[2]. The
// padding stays zero.
func (c copier) readExtent(il, e int, buf []byte) error {
	clear(buf)
	at := 0
	for k, t := range c.p.Layers[il] {
		n := int(t.Slice)
		if _, err := c.gguf.ReadAt(buf[at:at+n], t.Offset+int64(e)*t.Slice); err != nil {
			return fmt.Errorf("layer %d expert %d %s: %w", il, e, gguf.ExpertKinds[k], err)
		}
		at += n
	}
	return nil
}

// write writes the side-file and its manifest through temporary names, so a
// failed run leaves the new ones nowhere and any earlier manifest as it was.
func write(ggufPath, side string, force bool) (*manifest, error) {
	if same, err := isSameFile(ggufPath, side); err != nil {
		return nil, err
	} else if same {
		return nil, fmt.Errorf("%s is the model; name another side-file", side)
	}
	if !force {
		for _, path := range []string{side, side + manifestName} {
			if _, err := os.Stat(path); err == nil {
				return nil, fmt.Errorf("%s exists; -force replaces it", path)
			} else if !errors.Is(err, os.ErrNotExist) {
				return nil, err
			}
		}
	}
	p, err := planFor(ggufPath)
	if err != nil {
		return nil, err
	}
	in, err := openRead(ggufPath)
	if err != nil {
		return nil, err
	}
	defer in.Close()
	tmp := side + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return nil, err
	}
	done := false
	defer func() {
		f.Close()
		if !done {
			os.Remove(tmp)
		}
	}()
	if err := noCache(f); err != nil {
		return nil, err
	}
	m := p.manifestFor()
	whole := sha256.New()
	w := io.MultiWriter(f, whole)
	head := make([]byte, p.DataStart)
	copy(head, p.table())
	if _, err := w.Write(head); err != nil {
		return nil, err
	}
	c := copier{p, in}
	for il := range p.Layers {
		buf := make([]byte, p.extent(il))
		row := make([]string, p.Experts)
		for e := range p.Experts {
			if err := c.readExtent(il, e, buf); err != nil {
				return nil, err
			}
			sum := sha256.Sum256(buf)
			row[e] = hex.EncodeToString(sum[:])
			if _, err := w.Write(buf); err != nil {
				return nil, err
			}
		}
		m.Extents = append(m.Extents, row)
	}
	m.SHA256 = hex.EncodeToString(whole.Sum(nil))
	if err := f.Sync(); err != nil {
		return nil, err
	}
	if err := f.Close(); err != nil {
		return nil, err
	}
	data, err := json.Marshal(m)
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(tmp+manifestName, append(data, '\n'), 0o644); err != nil {
		return nil, err
	}
	if err := install(tmp, side); err != nil {
		return nil, err
	}
	done = true
	return &m, nil
}

// isSameFile reports whether side already names the model, directly or
// through a link. A side that does not exist yet is not the model.
func isSameFile(ggufPath, side string) (bool, error) {
	model, err := os.Stat(ggufPath)
	if err != nil {
		return false, err
	}
	target, err := os.Stat(side)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return os.SameFile(model, target), nil
}

// install moves the finished manifest and side-file to their names. Two
// renames cannot be one atomic step, so an earlier manifest is kept aside until
// the side-file is in place, and put back if it cannot be.
func install(tmp, side string) error {
	newManifest, manifest, kept := tmp+manifestName, side+manifestName, tmp+manifestName+".old"
	hadManifest := true
	if err := os.Rename(manifest, kept); errors.Is(err, os.ErrNotExist) {
		hadManifest = false
	} else if err != nil {
		os.Remove(newManifest)
		return err
	}
	restore := func() {
		os.Remove(newManifest)
		if hadManifest {
			os.Rename(kept, manifest)
		} else {
			os.Remove(manifest)
		}
	}
	if err := os.Rename(newManifest, manifest); err != nil {
		restore()
		return err
	}
	if err := os.Rename(tmp, side); err != nil {
		restore()
		return err
	}
	if hadManifest {
		return os.Remove(kept)
	}
	return nil
}

// loadManifest reads a manifest, refusing fields it does not know.
func loadManifest(path string) (*manifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var m manifest
	if err := dec.Decode(&m); err != nil {
		return nil, fmt.Errorf("manifest %s: %w", path, err)
	}
	return &m, nil
}

// verify recomputes the side-file from the GGUF: the table must be the plan's,
// every extent must hold the GGUF's bytes and zero padding, and the manifest
// must be exactly what the data gives, every field of it. It returns the first
// faults together, as an error.
func verify(ggufPath, side string) (*manifest, error) {
	got, err := loadManifest(side + manifestName)
	if err != nil {
		return nil, err
	}
	p, err := planFor(ggufPath)
	if err != nil {
		return nil, err
	}
	in, err := openRead(ggufPath)
	if err != nil {
		return nil, err
	}
	defer in.Close()
	f, err := openRead(side)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	var faults faultList
	if st.Size() != p.FileSize {
		// the rest would only repeat it
		return nil, fmt.Errorf("file has %d bytes, the GGUF's experts need %d (truncated, or written from another GGUF)", st.Size(), p.FileSize)
	}
	want := p.manifestFor()
	whole := sha256.New()
	head := make([]byte, p.DataStart)
	if _, err := f.ReadAt(head, 0); err != nil {
		return nil, err
	}
	whole.Write(head)
	// the identity first: against another GGUF every later fault would mislead
	const identityAt, identityEnd = 40, checksumAt
	table := p.table()
	if !bytes.Equal(head[identityAt:identityEnd], table[identityAt:identityEnd]) {
		return nil, fmt.Errorf("GGUF identity: the side-file was written from another GGUF (%x, this one is %x: size, header bytes, header sha256)",
			head[identityAt:identityEnd], table[identityAt:identityEnd])
	}
	if !bytes.Equal(head[:len(table)], table) {
		faults.add("table: differs from the GGUF's plan, first at byte %d", firstDifference(head[:len(table)], table))
	}
	if !allZero(head[p.TableBytes:]) {
		faults.add("table padding is not zero")
	}
	c := copier{p, in}
	for il := range p.Layers {
		ext := p.extent(il)
		side, ref := make([]byte, ext), make([]byte, ext)
		row := make([]string, p.Experts)
		for e := range p.Experts {
			if _, err := f.ReadAt(side, p.Offsets[il][e]); err != nil {
				return nil, fmt.Errorf("layer %d expert %d: %w", il, e, err)
			}
			if err := c.readExtent(il, e, ref); err != nil {
				return nil, err
			}
			sum := sha256.Sum256(side)
			row[e] = hex.EncodeToString(sum[:])
			whole.Write(side)
			if at := firstDifference(side, ref); at >= 0 {
				faults.add("layer %d expert %d: extent differs from the GGUF at its byte %d", il, e, at)
			}
		}
		want.Extents = append(want.Extents, row)
	}
	want.SHA256 = hex.EncodeToString(whole.Sum(nil))
	compareManifest(&faults, got, &want)
	if err := faults.err(); err != nil {
		return nil, err
	}
	return &want, nil
}

// compareManifest adds a fault per manifest field that is not what the data
// gives, each extent hash on its own.
func compareManifest(faults *faultList, got, want *manifest) {
	if got.Format != want.Format || got.Version != want.Version {
		faults.add("manifest: format %s version %d, want %s version %d", got.Format, got.Version, want.Format, want.Version)
	}
	if got.Alignment != want.Alignment || got.Layers != want.Layers || got.Experts != want.Experts {
		faults.add("manifest: alignment %d, %d layers x %d experts, want %d, %d x %d", got.Alignment, got.Layers, got.Experts, want.Alignment, want.Layers, want.Experts)
	}
	if got.Size != want.Size {
		faults.add("manifest: size %d, the file has %d", got.Size, want.Size)
	}
	if got.SHA256 != want.SHA256 {
		faults.add("manifest: file sha256 %s, the file's is %s", got.SHA256, want.SHA256)
	}
	if got.TableSHA256 != want.TableSHA256 {
		faults.add("manifest: table sha256 %s, want %s", got.TableSHA256, want.TableSHA256)
	}
	if got.GGUF != want.GGUF {
		faults.add("manifest: GGUF identity %+v, this GGUF is %+v", got.GGUF, want.GGUF)
	}
	if !reflect.DeepEqual(got.Tensors, want.Tensors) {
		faults.add("manifest: tensor table differs from the GGUF's")
	}
	if len(got.Extents) != len(want.Extents) {
		faults.add("manifest: %d layers of extent hashes, want %d", len(got.Extents), len(want.Extents))
		return
	}
	for il, row := range want.Extents {
		if len(got.Extents[il]) != len(row) {
			faults.add("manifest: layer %d has %d extent hashes, want %d", il, len(got.Extents[il]), len(row))
			continue
		}
		for e, sum := range row {
			if got.Extents[il][e] != sum {
				faults.add("layer %d expert %d: sha256 %s, the manifest says %s", il, e, sum, got.Extents[il][e])
			}
		}
	}
}

// faultList keeps the first faults and counts the others.
type faultList struct {
	first []string
	n     int
}

func (f *faultList) add(format string, args ...any) {
	f.n++
	if len(f.first) < 10 {
		f.first = append(f.first, fmt.Sprintf(format, args...))
	}
}

func (f *faultList) err() error {
	if f.n == 0 {
		return nil
	}
	return fmt.Errorf("%d faults: %s", f.n, strings.Join(f.first, "; "))
}

func allZero(b []byte) bool {
	for _, v := range b {
		if v != 0 {
			return false
		}
	}
	return true
}

// firstDifference is the first index where a and b differ, or -1.
func firstDifference(a, b []byte) int {
	if bytes.Equal(a, b) {
		return -1
	}
	for i := range min(len(a), len(b)) {
		if a[i] != b[i] {
			return i
		}
	}
	return min(len(a), len(b))
}

// openRead opens a file to read through, without filling the page cache.
func openRead(path string) (*os.File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	if err := noCache(f); err != nil {
		f.Close()
		return nil, err
	}
	return f, nil
}
