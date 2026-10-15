package main

import "testing"

func TestParseSize(t *testing.T) {
	for _, c := range []struct {
		in   string
		want int64
	}{{"16GiB", 16 << 30}, {"1.5GiB", 3 << 29}, {"256MiB", 256 << 20}, {"4KiB", 4096}, {"1000", 1000}} {
		got, err := parseSize(c.in)
		if err != nil || got != c.want {
			t.Errorf("parseSize(%q) = %d, %v; want %d", c.in, got, err, c.want)
		}
	}
	for _, bad := range []string{"", "GiB", "-1GiB", "0", "16GB"} {
		if _, err := parseSize(bad); err == nil {
			t.Errorf("parseSize(%q) accepted", bad)
		}
	}
}
