//go:build darwin

package memctl

import (
	"crypto/rand"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestParseWiredPages(t *testing.T) {
	out := "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n" +
		"Pages free:                                   382502.\n" +
		"Pages wired down:                             283161.\n"
	n, err := parseWiredPages(out)
	if err != nil || n != 283161 {
		t.Fatalf("parseWiredPages = %d, %v; want 283161", n, err)
	}
	if _, err := parseWiredPages("Pages free: 1.\n"); err == nil {
		t.Fatal("parseWiredPages accepted output without a wired line")
	}
}

// TestEvict writes a file, reads it so that the page cache holds it, and
// expects Evict to drop nearly all of its pages.
func TestEvict(t *testing.T) {
	path := filepath.Join(t.TempDir(), "evict.bin")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.CopyN(f, rand.Reader, 64<<20); err != nil {
		t.Fatal(err)
	}
	if err := f.Sync(); err != nil {
		t.Fatal(err)
	}
	f.Close()
	data, err := os.ReadFile(path)
	if err != nil || len(data) != 64<<20 {
		t.Fatalf("read back %d bytes: %v", len(data), err)
	}

	before, err := Resident(path)
	if err != nil {
		t.Fatal(err)
	}
	if before.Pages != 64<<20/PageSize() {
		t.Fatalf("%d pages, want %d", before.Pages, 64<<20/PageSize())
	}
	if before.Resident < before.Pages*9/10 {
		t.Fatalf("only %d of %d pages resident after reading the file", before.Resident, before.Pages)
	}
	if err := Evict(path); err != nil {
		t.Fatal(err)
	}
	after, err := Resident(path)
	if err != nil {
		t.Fatal(err)
	}
	if after.Resident > after.Pages/10 {
		t.Fatalf("%d of %d pages still resident after Evict (%d before)", after.Resident, after.Pages, before.Resident)
	}
	t.Logf("resident pages: %d before, %d after Evict, of %d", before.Resident, after.Resident, after.Pages)
}

// TestWire expects 256 MiB of wired memory to raise the system's wired page
// count by about as many pages, and Release to return them. Other processes
// change the count too, so it allows a quarter either way.
func TestWire(t *testing.T) {
	const size = 256 << 20
	want := size / PageSize()
	w0, err := WiredPages()
	if err != nil {
		t.Fatal(err)
	}
	w, err := Wire(size)
	if err != nil {
		t.Fatal(err)
	}
	w1, err := WiredPages()
	if err != nil {
		w.Release()
		t.Fatal(err)
	}
	if err := w.Release(); err != nil {
		t.Fatal(err)
	}
	w2, err := WiredPages()
	if err != nil {
		t.Fatal(err)
	}
	if d := w1 - w0; d < int64(want)*3/4 || d > int64(want)*5/4 {
		t.Fatalf("Wire(256 MiB) changed the wired pages by %d, want about %d", d, want)
	}
	if d := w1 - w2; d < int64(want)*3/4 {
		t.Fatalf("Release returned %d wired pages, want about %d", d, want)
	}
	if err := w.Release(); err != nil {
		t.Fatalf("a second Release: %v", err)
	}
}
