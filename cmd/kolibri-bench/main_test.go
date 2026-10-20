package main

import (
	"encoding/json"
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
	c.Args = []string{"--cpu-moe"}
	if got := strings.Join(m.serverArgs(c, 8099, "/run/stats.json"), " "); strings.Contains(got, "stats") {
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
	g := gate{CeilingGiB: 16, Candidate: "cache", Baselines: []string{"cpu-moe", "resident"}, Decides: true}
	var runs []runResult
	for rep, d := range []float64{20, 21, 22} {
		runs = append(runs, fakeRun("cache", 16, rep+1, []float64{d, d}, []float64{80, 80}, nil))
	}
	for rep, d := range []float64{10, 12, 11} {
		runs = append(runs, fakeRun("cpu-moe", 16, rep+1, []float64{d, d}, []float64{60, 60}, nil))
	}
	failed := runResult{Config: "resident", CeilingGiB: 16, Rep: 1, Status: statusLoadFailed}
	v := evalGate(g, append(runs, failed))
	if !v.Pass || v.Baseline != "cpu-moe" || !v.DecodeWin || v.TailWin {
		t.Fatalf("verdict = %+v, want a decode win over cpu-moe (resident does not run)", v)
	}

	// overlapping repetitions are no win, although the median is higher
	runs = append(runs[:3:3], fakeRun("cpu-moe", 16, 1, []float64{21, 21}, []float64{90, 90}, nil))
	v = evalGate(g, runs)
	if v.DecodeWin {
		t.Fatalf("verdict = %+v: overlapping repetitions must not win on decode", v)
	}
	if !v.TailWin || !v.Pass {
		t.Fatalf("verdict = %+v: every p95 of the cache is below the baseline's best", v)
	}

	// no baseline that runs: no verdict
	v = evalGate(g, append(runs[:3:3], failed))
	if v.Pass || v.Baseline != "" {
		t.Fatalf("verdict = %+v, want none without a running baseline", v)
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
	if rss, fp := peaks(strings.NewReader(samples)); rss != 120 || fp != 4096 {
		t.Fatalf("peaks = %v, %v; want 120, 4096", rss, fp)
	}
}

// TestShippedMatrix checks that every config the equivalence checks and gates
// name exists, at the ceilings they compare.
func TestShippedMatrix(t *testing.T) {
	data, err := os.ReadFile("../../testdata/bench/expert-cache.json")
	if err != nil {
		t.Fatal(err)
	}
	var m matrix
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
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
