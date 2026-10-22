package main

import "kolibri-llm/internal/memctl"

// pageCacheMiB is how much of the file at path the page cache holds, in MiB
// (mincore; the file is mapped, not read).
func pageCacheMiB(path string) (float64, error) {
	r, err := memctl.Resident(path)
	if err != nil {
		return 0, err
	}
	return float64(r.Resident) * float64(memctl.PageSize()) / (1 << 20), nil
}
