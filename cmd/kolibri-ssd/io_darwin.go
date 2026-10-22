//go:build darwin

package main

import (
	"fmt"
	"os"
	"syscall"

	"kolibri-llm/internal/memctl"
)

// openFile opens the file read-only; mode nocache sets F_NOCACHE on the
// descriptor, as patch 0011 does, so reads bypass the page cache.
func openFile(path, mode string) (*os.File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	if mode == "nocache" {
		if _, _, errno := syscall.Syscall(syscall.SYS_FCNTL, f.Fd(), syscall.F_NOCACHE, 1); errno != 0 {
			f.Close()
			return nil, fmt.Errorf("fcntl F_NOCACHE %s: %w", path, errno)
		}
	}
	return f, nil
}

// evict drops the file from the page cache (internal/memctl, no root).
func evict(path string) error { return memctl.Evict(path) }

// resident counts the file's pages in the page cache; it reads none.
func resident(path string) (residency, error) {
	r, err := memctl.Resident(path)
	return residency{r.Pages, r.Resident}, err
}

func pageSize() int { return memctl.PageSize() }
