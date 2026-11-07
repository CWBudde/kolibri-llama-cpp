//go:build !darwin

package main

import "os"

func noCache(*os.File) error { return nil }

func dropCache(string) error { return nil }
