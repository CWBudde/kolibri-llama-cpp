package tier

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLocate(t *testing.T) {
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(dir, "w.gguf")
	if err := os.WriteFile(file, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link.gguf")
	if err := os.Symlink(file, link); err != nil {
		t.Fatal(err)
	}

	t.Setenv("HOME", dir)
	loc, err := Locate(link)
	if err != nil {
		t.Fatal(err)
	}
	if !loc.Home || loc.Resolved != file {
		t.Errorf("same volume: %+v, want Home and %s", loc, file)
	}
	if !strings.Contains(loc.String(), "home volume") || strings.Contains(loc.String(), "NOT") {
		t.Errorf("same volume: %q", loc)
	}

	// /dev is its own file system, so it stands in for a home on another volume.
	t.Setenv("HOME", "/dev")
	loc, err = Locate(link)
	if err != nil {
		t.Fatal(err)
	}
	if loc.Home {
		t.Errorf("other volume: %+v, want not Home", loc)
	}
	if !strings.Contains(loc.String(), "NOT the home volume") {
		t.Errorf("other volume: %q", loc)
	}
}

func TestLocateDangling(t *testing.T) {
	link := filepath.Join(t.TempDir(), "link.gguf")
	if err := os.Symlink("/nonexistent/w.gguf", link); err != nil {
		t.Fatal(err)
	}
	if _, err := Locate(link); err == nil {
		t.Error("dangling symlink: no error")
	}
}
