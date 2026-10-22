package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestPageCacheMiB(t *testing.T) {
	path := filepath.Join(t.TempDir(), "f")
	// 4 MiB just written: in the page cache
	if err := os.WriteFile(path, make([]byte, 4<<20), 0o644); err != nil {
		t.Fatal(err)
	}
	v, err := pageCacheMiB(path)
	if err != nil || v <= 0 || v > 4 {
		t.Fatalf("pageCacheMiB = %v, %v; want (0, 4]", v, err)
	}
	if _, err := pageCacheMiB(filepath.Join(t.TempDir(), "missing")); err == nil {
		t.Fatal("a missing file has a residency")
	}
}
