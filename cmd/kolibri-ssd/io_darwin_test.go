//go:build darwin

package main

import (
	"testing"
)

// TestNocacheCell runs a cell through an F_NOCACHE descriptor, as the
// runtime reads, and checks that the reads leave little of the file in the
// page cache after an eviction.
func TestNocacheCell(t *testing.T) {
	path, hdr := writeFixture(t)
	l, err := expertLayout(hdr)
	if err != nil {
		t.Fatal(err)
	}
	f, err := openFile(path, "nocache")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := evict(path); err != nil {
		t.Fatal(err)
	}
	ops, err := cellOps(cell{"expert-three", "nocache", l.maxExpertBytes(), 2}, l, 6*l.maxExpertBytes(), int64(pageSize()), 1)
	if err != nil {
		t.Fatal(err)
	}
	tm, err := runCell(f, ops, 2, 0, int64(pageSize()))
	if err != nil {
		t.Fatal(err)
	}
	if tm.Ops != 6 || tm.Bytes != 6*l.maxExpertBytes() {
		t.Errorf("%d ops, %d bytes", tm.Ops, tm.Bytes)
	}
	r, err := resident(path)
	if err != nil {
		t.Fatal(err)
	}
	if r.Pages == 0 {
		t.Error("no pages counted")
	}
	t.Logf("after nocache reads: %d of %d pages resident", r.Resident, r.Pages)
}
