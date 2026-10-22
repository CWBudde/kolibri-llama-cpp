//go:build !darwin

package main

import (
	"errors"
	"os"
)

var errDarwin = errors.New("F_NOCACHE, eviction and residency need macOS")

func openFile(path, mode string) (*os.File, error) {
	if mode == "nocache" {
		return nil, errDarwin
	}
	return os.Open(path)
}

func evict(string) error { return errDarwin }

func resident(string) (residency, error) { return residency{}, errDarwin }

func pageSize() int { return os.Getpagesize() }
