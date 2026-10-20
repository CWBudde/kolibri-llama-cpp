package main

import (
	"encoding/json"
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
	var counts, ids []int
	var text strings.Builder
	final, err := readStream(strings.NewReader(sse), func(count int, chunkIDs []int, content string) {
		counts = append(counts, count)
		ids = append(ids, chunkIDs...)
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
	if len(ids) != 3 || ids[0] != 101 || ids[2] != 103 {
		t.Errorf("ids = %v, want [101 102 103]", ids)
	}
	if string(final) != `{"predicted_n":3}` {
		t.Errorf("final timings = %s", final)
	}
}

func TestReadStreamWithoutTokens(t *testing.T) {
	// Without return_tokens each chunk still counts as one token.
	sse := "data: {\"content\":\"a\"}\ndata: {\"content\":\"b\"}\ndata: {\"stop\":true}\n"
	var last int
	if _, err := readStream(strings.NewReader(sse), func(count int, _ []int, _ string) { last = count }); err != nil {
		t.Fatal(err)
	}
	if last != 2 {
		t.Errorf("last count = %d, want 2", last)
	}
}

func TestReadStreamNoStop(t *testing.T) {
	if _, err := readStream(strings.NewReader("data: {\"content\":\"a\"}\n"), func(int, []int, string) {}); err == nil {
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

func TestLatencies(t *testing.T) {
	t0 := time.Unix(1000, 0)
	ms := func(v int) time.Time { return t0.Add(time.Duration(v) * time.Millisecond) }
	// The first token at 0; one token 10 ms later, then a chunk of 3 tokens after 30 ms more.
	got := latencies([]stamp{{1, ms(0)}, {2, ms(10)}, {5, ms(40)}})
	want := []time.Duration{10 * time.Millisecond, 10 * time.Millisecond, 10 * time.Millisecond, 10 * time.Millisecond}
	if len(got) != len(want) {
		t.Fatalf("latencies = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("latency %d = %v, want %v", i, got[i], want[i])
		}
	}
	if l := latencies([]stamp{{1, ms(0)}}); len(l) != 0 {
		t.Errorf("one stamp gives %v, want none", l)
	}
}

func TestPercentile(t *testing.T) {
	// 1..20 ms: nearest rank, so p50 is the 10th value, p95 the 19th, p99 and p100 the 20th.
	var d []time.Duration
	for i := 20; i >= 1; i-- {
		d = append(d, time.Duration(i)*time.Millisecond)
	}
	for _, c := range []struct {
		p    float64
		want int
	}{{50, 10}, {95, 19}, {99, 20}, {100, 20}, {1, 1}} {
		if got := percentile(d, c.p); got != time.Duration(c.want)*time.Millisecond {
			t.Errorf("p%v = %v, want %d ms", c.p, got, c.want)
		}
	}
	if got := percentile(nil, 50); got != 0 {
		t.Errorf("p50 of nothing = %v, want 0", got)
	}
	if d[0] != 20*time.Millisecond {
		t.Error("percentile reordered its argument")
	}
}

func TestSummarize(t *testing.T) {
	t0 := time.Unix(1000, 0)
	ms := func(v int) time.Time { return t0.Add(time.Duration(v) * time.Millisecond) }
	stamps := []stamp{{1, ms(500)}, {2, ms(520)}, {3, ms(540)}, {4, ms(600)}}
	s := summarize(t0, stamps, []int{1, 22, 333, 4}, json.RawMessage(`{"prompt_per_second":100}`), 2048, 64)
	if s.Generated != 4 || s.PromptTokens != 2048 || s.Pageins != 64 {
		t.Errorf("counts = %+v", s)
	}
	if s.TTFTMs != 500 {
		t.Errorf("ttft = %v ms, want 500", s.TTFTMs)
	}
	// 3 tokens after the first in 100 ms
	if math.Abs(s.DecodeTokensPerSecond-30) > 1e-9 {
		t.Errorf("decode tokens/s = %v, want 30", s.DecodeTokensPerSecond)
	}
	if s.LatencyMs.P50 != 20 || s.LatencyMs.P95 != 60 || s.LatencyMs.Max != 60 {
		t.Errorf("latency = %+v, want p50 20, p95 60, max 60", s.LatencyMs)
	}
	if s.TokensSHA256 != idsSHA256([]int32{1, 22, 333, 4}) {
		t.Errorf("tokens sha256 = %s", s.TokensSHA256)
	}
}
