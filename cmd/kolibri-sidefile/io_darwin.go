//go:build darwin

package main

import (
	"fmt"
	"os"
	"syscall"

	"kolibri-llm/internal/memctl"
)

// noCache sets F_NOCACHE, so a file of 31 GB does not fill the page cache.
func noCache(f *os.File) error {
	if _, _, errno := syscall.Syscall(syscall.SYS_FCNTL, f.Fd(), syscall.F_NOCACHE, 1); errno != 0 {
		return fmt.Errorf("fcntl F_NOCACHE %s: %w", f.Name(), errno)
	}
	return nil
}

// dropCache evicts a file from the page cache (internal/memctl, no root).
func dropCache(path string) error { return memctl.Evict(path) }
