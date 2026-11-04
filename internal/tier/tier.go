// Package tier tells whether a file is on the home directory's volume, the
// tier expert streaming benchmarks must read weights from: their I/O timings
// depend on where a GGUF physically lives, and a path under ~/models may be a
// symlink to another volume.
package tier

import (
	"fmt"
	"os"
	"path/filepath"
	"syscall"
)

// Location is where a path ends up once every symlink is followed.
type Location struct {
	Resolved string
	Home     bool // on the same volume as the home directory
}

// Locate follows path's symlinks. A dangling symlink is an error.
func Locate(path string) (Location, error) {
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		return Location{}, err
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return Location{}, err
	}
	a, err := device(resolved)
	if err != nil {
		return Location{}, err
	}
	b, err := device(home)
	if err != nil {
		return Location{}, err
	}
	return Location{Resolved: resolved, Home: a == b}, nil
}

// String is the form that benchmark records carry.
func (l Location) String() string {
	if l.Home {
		return l.Resolved + " (home volume)"
	}
	return l.Resolved + " (NOT the home volume: I/O timings are not comparable)"
}

func device(path string) (uint64, error) {
	var st syscall.Stat_t
	if err := syscall.Stat(path, &st); err != nil {
		return 0, fmt.Errorf("stat %s: %w", path, err)
	}
	return uint64(st.Dev), nil
}
