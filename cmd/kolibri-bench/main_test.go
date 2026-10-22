package main

import (
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestParseLoadavg(t *testing.T) {
	got, err := parseLoadavg("{ 9.67 11.09 15.90 }")
	if err != nil || got != 9.67 {
		t.Fatalf("parseLoadavg = %v, %v; want 9.67", got, err)
	}
	if _, err := parseLoadavg("{ }"); err == nil {
		t.Fatal("parseLoadavg accepted an empty load")
	}
}

func TestParseSwapUsed(t *testing.T) {
	got, err := parseSwapUsed("total = 10240.00M  used = 9142.00M  free = 1098.00M  (encrypted)")
	if err != nil || got != 9142 {
		t.Fatalf("parseSwapUsed = %v, %v; want 9142", got, err)
	}
}

func TestQuiet(t *testing.T) {
	q := quietRule{MaxLoad: 3, MaxSwapGrowthMiB: 100, Window: 2}
	// samples every 30 s: the window of 2 intervals is 60 s
	for _, c := range []struct {
		name    string
		samples []machine
		want    bool
	}{
		{"too few samples", []machine{{1, 100, true}, {1, 100, true}}, false},
		{"quiet", []machine{{1, 100, true}, {2, 150, true}, {2.9, 199, true}}, true},
		{"load too high", []machine{{1, 100, true}, {1, 100, true}, {3, 100, true}}, false},
		{"swap growing", []machine{{1, 100, true}, {1, 150, true}, {1, 200, true}}, false},
		{"only the last window counts", []machine{{9, 0, true}, {1, 100, true}, {1, 120, true}, {1, 130, true}}, true},
		{"on battery", []machine{{1, 100, true}, {1, 100, true}, {1, 100, false}}, false},
	} {
		if got := q.quiet(c.samples); got != c.want {
			t.Errorf("%s: quiet = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestParseOnAC(t *testing.T) {
	for s, want := range map[string]bool{
		"Now drawing from 'AC Power'\n -InternalBattery-0 (id=7864419)\t80%; charging; present: true\n":         true,
		"Now drawing from 'Battery Power'\n -InternalBattery-0 (id=7864419)\t34%; discharging; present: true\n": false,
	} {
		if got := parseOnAC(s); got != want {
			t.Errorf("parseOnAC(%q) = %v, want %v", s, got, want)
		}
	}
}

func TestExpand(t *testing.T) {
	m := matrix{Configs: []config{
		{Name: "a", Ceilings: []float64{0, 16}, Reps: 2},
		{Name: "b", Ceilings: []float64{0}, Reps: 1},
	}}
	var got []string
	for _, r := range m.expand() {
		got = append(got, r.key())
	}
	// repetitions outermost, so drift spreads over the arms
	want := "a/0g/r1 a/16g/r1 b/0g/r1 a/0g/r2 a/16g/r2"
	if strings.Join(got, " ") != want {
		t.Fatalf("expand = %s, want %s", strings.Join(got, " "), want)
	}
}

func TestServerArgs(t *testing.T) {
	m := matrix{ServerArgs: []string{"-ngl", "99"}}
	c := config{Model: "/m.gguf", Args: []string{"--expert-cache", "128"}}
	got := strings.Join(m.serverArgs(c, 8099, "/run/stats.json"), " ")
	want := "-m /m.gguf --port 8099 -ngl 99 --expert-cache 128 --expert-cache-stats /run/stats.json"
	if got != want {
		t.Fatalf("serverArgs = %s, want %s", got, want)
	}
	// resident arms write the decode timing too (patch 0012)
	c.Args = []string{"-ub", "21"}
	if got := strings.Join(m.serverArgs(c, 8099, "/run/stats.json"), " "); !strings.HasSuffix(got, "-ub 21 --expert-cache-stats /run/stats.json") {
		t.Fatalf("serverArgs without the cache = %s", got)
	}
}

func TestFits(t *testing.T) {
	const gib = 1 << 30
	resident := config{Args: nil}
	// IQ3 is 33.1 GiB; with every weight on Metal it is wired, and 16 GiB more
	// is beyond 48 GiB: the run that hung this Mac on 2026-10-15
	if ok, why := fits(resident, 16, 35556257600, 48*gib); ok || !strings.Contains(why, "33.1 GiB") {
		t.Errorf("resident IQ3 at 16 GiB: fits = %v, %q", ok, why)
	}
	if ok, _ := fits(resident, 0, 35556257600, 48*gib); !ok {
		t.Error("resident IQ3 without a ceiling does not fit")
	}
	for _, args := range [][]string{{"--cpu-moe"}, {"--expert-cache", "128"}} {
		if ok, why := fits(config{Args: args}, 16, 83135393312, 48*gib); !ok {
			t.Errorf("%v keeps the experts pageable, yet does not fit: %s", args, why)
		}
	}
}

func TestMedianRange(t *testing.T) {
	if got := median([]float64{3, 1, 2}); got != 2 {
		t.Errorf("median odd = %v", got)
	}
	if got := median([]float64{4, 1, 2, 3}); got != 2.5 {
		t.Errorf("median even = %v", got)
	}
	if got := median(nil); !math.IsNaN(got) {
		t.Errorf("median of nothing = %v, want NaN", got)
	}
}

func TestFirstDiff(t *testing.T) {
	if got := firstDiff([]int{1, 2, 3}, []int{1, 2, 3}); got != -1 {
		t.Errorf("equal: %d", got)
	}
	if got := firstDiff([]int{1, 2, 3}, []int{1, 5, 3}); got != 1 {
		t.Errorf("differ at 1: %d", got)
	}
	if got := firstDiff([]int{1, 2}, []int{1, 2, 3}); got != 2 {
		t.Errorf("shorter: %d", got)
	}
}

// fakeRun is a finished run with one warm request per decode rate.
func fakeRun(cfg string, ceiling float64, rep int, decode []float64, p95 []float64, ids []int) runResult {
	r := runResult{Config: cfg, CeilingGiB: ceiling, Rep: rep, Status: statusOK}
	r.Requests = append(r.Requests, request{Phase: "cold", Workload: "coding", Summary: &streamSummary{DecodeTokensPerSecond: decode[0] / 2, TokenIDs: ids}})
	for i, d := range decode {
		s := &streamSummary{DecodeTokensPerSecond: d, TokenIDs: ids}
		s.LatencyMs.P95 = p95[i]
		r.Requests = append(r.Requests, request{Phase: "warm", Workload: []string{"coding", "research"}[i], Summary: s})
	}
	return r
}

func TestGate(t *testing.T) {
	m := matrix{Configs: []config{
		{Name: "cache", Ceilings: []float64{16}, Reps: 3},
		{Name: "cpu-moe", Ceilings: []float64{16}, Reps: 3},
		{Name: "resident", Ceilings: []float64{16}, Reps: 1},
	}}
	g := gate{CeilingGiB: 16, Candidate: "cache", Baselines: []string{"cpu-moe", "resident"}, Decides: true}
	var cache, cpuMoe []runResult
	for rep, d := range []float64{20, 21, 22} {
		cache = append(cache, fakeRun("cache", 16, rep+1, []float64{d, d}, []float64{80, 80}, nil))
	}
	for rep, d := range []float64{10, 12, 11} {
		cpuMoe = append(cpuMoe, fakeRun("cpu-moe", 16, rep+1, []float64{d, d}, []float64{60, 60}, nil))
	}
	failed := func(cfg string, rep int) runResult {
		return runResult{Config: cfg, CeilingGiB: 16, Rep: rep, Status: statusLoadFailed}
	}
	join := func(parts ...[]runResult) []runResult { return slices.Concat(parts...) }

	v := evalGate(m, g, join(cache, cpuMoe, []runResult{failed("resident", 1)}))
	if !v.Pass || v.Baseline != "cpu-moe" || !v.DecodeWin || v.TailWin || v.Incomplete != "" {
		t.Fatalf("verdict = %+v, want a decode win over cpu-moe (resident does not run)", v)
	}

	// overlapping repetitions are no win, although the median is higher
	var overlap []runResult
	for rep, d := range []float64{10, 21, 11} {
		overlap = append(overlap, fakeRun("cpu-moe", 16, rep+1, []float64{d, d}, []float64{90, 90}, nil))
	}
	v = evalGate(m, g, join(cache, overlap, []runResult{failed("resident", 1)}))
	if v.DecodeWin {
		t.Fatalf("verdict = %+v: overlapping repetitions must not win on decode", v)
	}
	if !v.TailWin || !v.Pass {
		t.Fatalf("verdict = %+v: every p95 of the cache is below the baseline's best", v)
	}

	// no baseline that runs: no verdict
	v = evalGate(m, g, join(cache, []runResult{failed("cpu-moe", 1), failed("cpu-moe", 2), failed("cpu-moe", 3), failed("resident", 1)}))
	if v.Pass || v.Baseline != "" || v.Incomplete != "" {
		t.Fatalf("verdict = %+v, want none without a running baseline", v)
	}

	// a repetition missing or failed, of the candidate or a baseline: no verdict
	for name, runs := range map[string][]runResult{
		"cpu-moe":  join(cache, cpuMoe[:2], []runResult{failed("resident", 1)}),
		"cache":    join(cache[:2], []runResult{failed("cache", 3)}, cpuMoe, []runResult{failed("resident", 1)}),
		"resident": join(cache, cpuMoe),
	} {
		v = evalGate(m, g, runs)
		if v.Pass || !strings.Contains(v.Incomplete, name) {
			t.Errorf("%s incomplete: verdict = %+v, want none", name, v)
		}
	}
}

func TestReportCountsRequests(t *testing.T) {
	m := matrix{Cold: "coding", Warm: []string{"coding"},
		Configs: []config{
			{Name: "ref", Ceilings: []float64{0}, Reps: 1},
			{Name: "cache", Ceilings: []float64{0}, Reps: 2},
		},
		Equivalence: []equivalence{{Reference: "ref", Arms: []string{"cache"}, Exact: true}}}
	if got := m.expectedRequests([]string{"cache"}); got != 4 {
		t.Fatalf("expectedRequests = %d, want 4", got)
	}
	ids := []int{1, 2, 3}
	ref := fakeRun("ref", 0, 1, []float64{50}, []float64{20}, ids)
	r1 := fakeRun("cache", 0, 1, []float64{20}, []float64{50}, ids)
	r2 := fakeRun("cache", 0, 2, []float64{20}, []float64{50}, ids)
	var out strings.Builder
	if report(&out, m, []runResult{ref, r1}) {
		t.Fatalf("a missing repetition passes:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "FAIL equivalence cache against ref: 2 of 4 requests") {
		t.Errorf("report:\n%s", out.String())
	}
	r2.Status = statusRequestFailed
	if report(io.Discard, m, []runResult{ref, r1, r2}) {
		t.Fatal("a failed repetition passes")
	}
	r2.Status = statusOK
	out.Reset()
	if !report(&out, m, []runResult{ref, r1, r2}) {
		t.Fatalf("all requests equal, yet:\n%s", out.String())
	}
}

func TestEquivalence(t *testing.T) {
	ref := fakeRun("ref", 0, 1, []float64{50, 50}, []float64{20, 20}, []int{1, 2, 3})
	same := fakeRun("cache", 0, 1, []float64{20, 20}, []float64{50, 50}, []int{1, 2, 3})
	other := fakeRun("cache", 16, 2, []float64{20, 20}, []float64{50, 50}, []int{1, 2, 3})
	other.Requests[2].Summary = &streamSummary{TokenIDs: []int{1, 9, 3}}
	res := checkEquivalence(equivalence{Reference: "ref", Arms: []string{"cache"}, Exact: true}, []runResult{ref, same, other})
	if len(res) != 6 {
		t.Fatalf("%d comparisons, want 6", len(res))
	}
	fails := 0
	for _, r := range res {
		if r.Diff >= 0 {
			fails++
			if r.Run != "cache/16g/r2" || r.Workload != "research" || r.Diff != 1 {
				t.Errorf("unexpected difference %+v", r)
			}
		}
	}
	if fails != 1 {
		t.Fatalf("%d differences, want 1", fails)
	}
}

func TestLoadResults(t *testing.T) {
	dir := t.TempDir()
	r := fakeRun("cache", 16, 1, []float64{20, 20}, []float64{50, 50}, []int{1})
	b, _ := json.Marshal(r)
	if err := os.MkdirAll(filepath.Join(dir, r.key()), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, r.key(), "run.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := loadResults(dir)
	if err != nil || len(got) != 1 || got[0].key() != "cache/16g/r1" {
		t.Fatalf("loadResults = %+v, %v", got, err)
	}
}

func TestPeaks(t *testing.T) {
	samples := "time rss_mib footprint_mib swap_used_mib pressure tokens pageins\n" +
		"02:00:00 100 2048 9000 1 0 5\n02:00:02 120 4096 9000 1 10 5\n02:00:04 110 3000 9000 1 20 5\n"
	// a file without the page_cache_mib column (kolibri-stream before E1)
	if p := peaks(strings.NewReader(samples)); p != (samplePeaks{120, 4096, -1}) {
		t.Fatalf("peaks = %+v; want 120, 4096, -1", p)
	}
	samples = "time rss_mib footprint_mib swap_used_mib pressure tokens pageins page_cache_mib\n" +
		"02:00:00 100 2048 9000 1 0 5 300\n02:00:02 120 4096 9000 1 10 5 -\n02:00:04 110 3000 9000 1 20 5 2500\n"
	if p := peaks(strings.NewReader(samples)); p != (samplePeaks{120, 4096, 2500}) {
		t.Fatalf("peaks = %+v; want 120, 4096, 2500", p)
	}
}

func TestDecodeSplit(t *testing.T) {
	// a cache run's stats (patch 0012): 640 decoded tokens
	var cache cacheStats
	if err := json.Unmarshal([]byte(`{"expert_cache": {"buffered": true}, "decode": {"needed": 10, "wait_seconds": 9.2,
		"remap_seconds": 1.6, "tokens": 640, "wall_seconds": 34.56, "splits": 65280, "metal_commits": 66560},
		"process": {"max_phys_footprint": 11811160064}}`), &cache); err != nil {
		t.Fatal(err)
	}
	d, ok := cache.split()
	if !ok || !cache.ExpertCache.Buffered {
		t.Fatalf("split = %+v, %v; buffered %v", d, ok, cache.ExpertCache)
	}
	near := func(a, b float64) bool { return math.Abs(a-b) < 1e-9 }
	if !near(d.Wall, 54) || !near(d.Wait, 14.375) || !near(d.Remap, 2.5) || !near(d.Other, 37.125) || d.Splits != 102 || d.Commits != 104 {
		t.Fatalf("split = %+v", d)
	}
	// a resident run: no cache, only the timing; metal_commits null without Metal
	var res cacheStats
	if err := json.Unmarshal([]byte(`{"expert_cache": null, "decode": {"tokens": 640, "wall_seconds": 10.56, "splits": 1280,
		"metal_commits": null}, "process": {}}`), &res); err != nil {
		t.Fatal(err)
	}
	d, ok = res.split()
	if !ok || res.ExpertCache != nil || !near(d.Wall, 16.5) || !near(d.Other, 16.5) || d.Splits != 2 || !math.IsNaN(d.Commits) {
		t.Fatalf("resident split = %+v, %v", d, ok)
	}
	// Stage D's stats have no timing
	var old cacheStats
	if err := json.Unmarshal([]byte(`{"expert_cache": {}, "decode": {"wait_seconds": 9.2}}`), &old); err != nil {
		t.Fatal(err)
	}
	if _, ok := old.split(); ok {
		t.Fatal("stats without decode tokens give a split")
	}
	if _, ok := (*cacheStats)(nil).split(); ok {
		t.Fatal("no stats give a split")
	}
}

func TestDecodeTable(t *testing.T) {
	m := matrix{Configs: []config{{Name: "res", Ceilings: []float64{0}, Reps: 2}, {Name: "old", Ceilings: []float64{0}, Reps: 1}}}
	commits := int64(1920)
	fp := int64(2 << 30)
	r1 := runResult{Config: "res", Rep: 1, Status: statusOK, Stats: &cacheStats{}}
	r1.Stats.Decode.Tokens, r1.Stats.Decode.WallSeconds, r1.Stats.Decode.Splits, r1.Stats.Decode.MetalCommits = 640, 10.56, 1280, &commits
	r1.Stats.Process.MaxPhysFootprint = &fp
	peak, end := 4096.0, 1024.0
	r1.PeakPageCacheMiB, r1.EndPageCacheMiB = &peak, &end
	r2 := r1
	r2.Rep = 2
	var out strings.Builder
	decodeTable(&out, m, []runResult{r1, r2, {Config: "old", Rep: 1, Status: statusOK}})
	for _, want := range []string{
		"| res | 0 GiB | 16.50 (16.50–16.50) | 0.00 (0.00–0.00) | 0.000 (0.000–0.000) | 16.50 (16.50–16.50) | 2.0 (2.0–2.0) | 3.0 (3.0–3.0) | 2.0 (2.0–2.0) | 4.0 (4.0–4.0) | 1.0 (1.0–1.0) |",
		"| old | 0 GiB | – | – | – | – | – | – | – | – | – |",
	} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("decode table lacks %q:\n%s", want, out.String())
		}
	}
}

func loadMatrix(t *testing.T, path string) matrix {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var m matrix
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

// TestShippedMatrix checks that every config the equivalence checks and gates
// name exists, at the ceilings they compare.
func TestShippedMatrix(t *testing.T) {
	for _, path := range []string{"../../testdata/bench/expert-cache.json", "../../testdata/bench/e1.json"} {
		t.Run(filepath.Base(path), func(t *testing.T) { checkMatrix(t, loadMatrix(t, path)) })
	}
}

// TestE1Matrix keeps E1's settings comparable with Stage D's and its wired
// memory safe: the resident arm and the 384-slot cache (all experts, about
// as much memory as resident) only without a ceiling.
func TestE1Matrix(t *testing.T) {
	d, e := loadMatrix(t, "../../testdata/bench/expert-cache.json"), loadMatrix(t, "../../testdata/bench/e1.json")
	if !slices.Equal(d.ServerArgs, e.ServerArgs) || d.PromptFirst != e.PromptFirst || d.NPredict != e.NPredict ||
		d.Cold != e.Cold || !slices.Equal(d.Warm, e.Warm) {
		t.Errorf("E1's workload settings differ from Stage D's")
	}
	for _, name := range []string{"iq3-resident-ub21", "iq3-cache384-ub21"} {
		c, ok := e.config(name)
		if !ok || !slices.Equal(c.Ceilings, []float64{0}) {
			t.Errorf("%s: ceilings %v, want only 0", name, c.Ceilings)
		}
	}
	const gib = 1 << 30
	for _, c := range e.Configs {
		for _, ceil := range c.Ceilings {
			if ceil > 0 && !usesCache(c.Args) {
				if ok, _ := fits(c, ceil, 35556257600, 48*gib); ok {
					t.Errorf("%s fits at %g GiB", c.Name, ceil)
				}
			}
		}
	}
}

func checkMatrix(t *testing.T, m matrix) {
	if m.PromptFirst <= 0 || m.NPredict <= 0 || m.Cold == "" || len(m.Warm) == 0 || len(m.Configs) == 0 {
		t.Fatalf("incomplete matrix %+v", m)
	}
	has := func(name string, ceiling float64) bool {
		c, ok := m.config(name)
		return ok && slices.Contains(c.Ceilings, ceiling)
	}
	for _, e := range m.Equivalence {
		for _, name := range append([]string{e.Reference}, e.Arms...) {
			if _, ok := m.config(name); !ok {
				t.Errorf("equivalence names %s, which is no config", name)
			}
		}
	}
	for _, g := range m.Gates {
		for _, name := range append([]string{g.Candidate}, g.Baselines...) {
			if !has(name, g.CeilingGiB) {
				t.Errorf("gate at %g GiB names %s, which does not run there", g.CeilingGiB, name)
			}
		}
	}
}
