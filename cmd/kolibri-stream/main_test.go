package main

import (
	"math"
	"strings"
	"testing"
	"time"
)

func TestReadStream(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"content":"Ber","tokens":[101],"stop":false}`,
		``,
		`data: {"content":"lin","tokens":[102,103],"stop":false}`,
		`: keep-alive`,
		`data: {"content":"","stop":true,"timings":{"predicted_n":3}}`,
		``,
	}, "\n")
	var counts []int
	var text strings.Builder
	final, err := readStream(strings.NewReader(sse), func(count int, content string) {
		counts = append(counts, count)
		text.WriteString(content)
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := text.String(); got != "Berlin" {
		t.Errorf("text = %q, want Berlin", got)
	}
	if len(counts) != 2 || counts[0] != 1 || counts[1] != 3 {
		t.Errorf("counts = %v, want [1 3]", counts)
	}
	if string(final) != `{"predicted_n":3}` {
		t.Errorf("final timings = %s", final)
	}
}

func TestReadStreamWithoutTokens(t *testing.T) {
	// Without return_tokens each chunk still counts as one token.
	sse := "data: {\"content\":\"a\"}\ndata: {\"content\":\"b\"}\ndata: {\"stop\":true}\n"
	var last int
	if _, err := readStream(strings.NewReader(sse), func(count int, _ string) { last = count }); err != nil {
		t.Fatal(err)
	}
	if last != 2 {
		t.Errorf("last count = %d, want 2", last)
	}
}

func TestReadStreamNoStop(t *testing.T) {
	if _, err := readStream(strings.NewReader("data: {\"content\":\"a\"}\n"), func(int, string) {}); err == nil {
		t.Error("want an error for a stream that ends without stop")
	}
}

func TestWindowRates(t *testing.T) {
	t0 := time.Unix(1000, 0)
	at := func(sec float64) time.Time { return t0.Add(time.Duration(sec * float64(time.Second))) }
	// First token at 0 s; 4 tokens/s up to token 9, then 2 tokens/s.
	stamps := []stamp{{1, at(0)}, {5, at(1)}, {9, at(2)}, {11, at(3)}, {13, at(4)}}
	got := windowRates(stamps, 4)
	want := []window{{1, 5, 4}, {5, 9, 4}, {9, 13, 2}}
	if len(got) != len(want) {
		t.Fatalf("windows = %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i].from != want[i].from || got[i].to != want[i].to || math.Abs(got[i].rate-want[i].rate) > 1e-9 {
			t.Errorf("window %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestParseFootprint(t *testing.T) {
	for _, tc := range []struct {
		out  string
		want float64
	}{
		{"    phys_footprint: 918 MB\n    phys_footprint_peak: 1 GB\n", 918},
		{"    phys_footprint: 1.5 GB\n", 1536},
		{"    phys_footprint: 3184 KB\n", 3184.0 / 1024},
	} {
		got, err := parseFootprint(tc.out)
		if err != nil || math.Abs(got-tc.want) > 1e-9 {
			t.Errorf("parseFootprint(%q) = %v, %v; want %v", tc.out, got, err, tc.want)
		}
	}
	if _, err := parseFootprint("no such line"); err == nil {
		t.Error("want an error without a phys_footprint line")
	}
}

func TestParseSwapUsed(t *testing.T) {
	got, err := parseSwapUsed("total = 16384.00M  used = 14714.50M  free = 1669.50M  (encrypted)")
	if err != nil || got != 14714.50 {
		t.Errorf("parseSwapUsed = %v, %v; want 14714.5", got, err)
	}
}

func TestParsePageins(t *testing.T) {
	out := "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:                               1238098.\nPageins:                                  3410795619.\nPageouts:                                   52063.\n"
	got, err := parsePageins(out)
	if err != nil || got != 3410795619 {
		t.Errorf("parsePageins = %v, %v; want 3410795619", got, err)
	}
	if _, err := parsePageins("Pages free: 1."); err == nil {
		t.Error("want an error without a Pageins line")
	}
}

func TestIDsSHA256(t *testing.T) {
	// tools/locality/workloads.py: sha256 of the decimal IDs joined by commas
	if got := idsSHA256([]int32{1, 22, 333}); got != "8915516745284e69425d3f566025a855f5a5718d6d44f09451fb19d0f61fc185" {
		t.Errorf("idsSHA256 = %s", got)
	}
}

func TestCheckTokens(t *testing.T) {
	manifest := `{"answers": {}, "workloads": {"hr": {"n_tokens": 3, "text_sha256": "x",
		"tokens_sha256": "8915516745284e69425d3f566025a855f5a5718d6d44f09451fb19d0f61fc185"}}}`
	if err := checkTokens("hr", []int32{1, 22, 333}, strings.NewReader(manifest)); err != nil {
		t.Errorf("recorded IDs: %v", err)
	}
	for name, c := range map[string]struct {
		workload string
		ids      []int32
	}{
		"changed ID":   {"hr", []int32{1, 22, 334}},
		"truncated":    {"hr", []int32{1, 22}},
		"not recorded": {"coding", []int32{1, 22, 333}},
	} {
		if err := checkTokens(c.workload, c.ids, strings.NewReader(manifest)); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}
