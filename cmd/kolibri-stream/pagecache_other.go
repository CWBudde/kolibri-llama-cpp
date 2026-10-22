//go:build !darwin

package main

import "errors"

func pageCacheMiB(string) (float64, error) {
	return 0, errors.New("page-cache residency is measured on macOS only")
}
