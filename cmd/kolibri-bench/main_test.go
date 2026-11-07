package main

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"kolibri-llm/internal/gguf"
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
	// another build does not know the fork's --expert-cache-stats
	c = config{Model: "/m.gguf", Server: "pr25294", Args: []string{"--moe-stream-cache", "128s"}}
	if got := strings.Join(m.serverArgs(c, 8099, "/run/stats.json"), " "); got != "-m /m.gguf --port 8099 -ngl 99 --moe-stream-cache 128s" {
		t.Fatalf("serverArgs of another build = %s", got)
	}
}

func TestParseServers(t *testing.T) {
	got, err := parseServers("pr25294=/b/llama-server,master=/c/llama-server")
	if err != nil || len(got) != 2 || got["pr25294"] != "/b/llama-server" || got["master"] != "/c/llama-server" {
		t.Fatalf("parseServers = %v, %v", got, err)
	}
	if got, err := parseServers(""); err != nil || len(got) != 0 {
		t.Fatalf(`parseServers("") = %v, %v`, got, err)
	}
	for _, bad := range []string{"pr25294", "=/b", "a=/b,a=/c"} {
		if _, err := parseServers(bad); err == nil {
			t.Errorf("parseServers(%q) accepted", bad)
		}
	}
	m := matrix{Configs: []config{{Name: "fork"}, {Name: "ref", Server: "pr25294"}}}
	if missing := m.missingServers(map[string]string{}, nil); !slices.Equal(missing, []string{"pr25294"}) {
		t.Errorf("missingServers = %v, want [pr25294]", missing)
	}
	if missing := m.missingServers(map[string]string{"pr25294": "/b"}, nil); len(missing) != 0 {
		t.Errorf("missingServers = %v, want none", missing)
	}
	// -only fork: the reference build is not needed
	if missing := m.missingServers(map[string]string{}, []string{"fork"}); len(missing) != 0 {
		t.Errorf("missingServers with -only fork = %v, want none", missing)
	}
}

func TestFits(t *testing.T) {
	const gib = 1 << 30
	// the real files: IQ3 experts take 1,699,840 B, Q8_0 ones 4,177,920 B, 384 per layer in 50 layers
	iq3 := weights{Bytes: 35556257600, ExpertBytes: 384 * 50 * 1699840, SlotBytes: 50 * 1699840}
	q8 := weights{Bytes: 83135393312, ExpertBytes: 384 * 50 * 4177920, SlotBytes: 50 * 4177920}
	resident := config{Args: nil}
	// IQ3 is 33.1 GiB; with every weight on Metal it is wired, and 16 GiB more
	// is beyond 48 GiB: the run that hung this Mac on 2026-10-15
	if ok, why := fits(resident, 16, iq3, 48*gib); ok || !strings.Contains(why, "33.1 GiB") {
		t.Errorf("resident IQ3 at 16 GiB: fits = %v, %q", ok, why)
	}
	if ok, _ := fits(resident, 0, iq3, 48*gib); !ok {
		t.Error("resident IQ3 without a ceiling does not fit")
	}
	if ok, why := fits(config{Args: []string{"--cpu-moe"}}, 16, q8, 48*gib); !ok {
		t.Errorf("--cpu-moe keeps the experts in the page cache, yet does not fit: %s", why)
	}
	// Q8_0's 128 slots (24.9 GiB) beside a 16 GiB hog paged the cache out and
	// produced no token in 20 minutes (2026-10-23); the runs below all completed
	for _, c := range []struct {
		name    string
		args    []string
		ceiling float64
		w       weights
		want    bool
	}{
		{"q8 128 slots at 16 GiB", []string{"--expert-cache", "128"}, 16, q8, false},
		{"q8 128 slots buffered at 16 GiB", []string{"--expert-cache", "128", "--expert-cache-buffered"}, 16, q8, false},
		{"q8 128 slots at 0 GiB", []string{"--expert-cache", "128"}, 0, q8, true},
		{"iq3 128 slots at 16 GiB", []string{"--expert-cache", "128"}, 16, iq3, true},
		{"iq3 384 slots at 0 GiB", []string{"--expert-cache", "384", "-ub", "21"}, 0, iq3, true},
	} {
		ok, why := fits(config{Args: c.args}, c.ceiling, c.w, 48*gib)
		if ok != c.want {
			t.Errorf("%s: fits = %v (%s), want %v", c.name, ok, why, c.want)
		}
		if !ok && !strings.Contains(why, "24.9 GiB") {
			t.Errorf("%s: %q does not name the slots' 24.9 GiB", c.name, why)
		}
	}
	if ok, why := fits(config{Args: []string{"--expert-cache", "many"}}, 0, q8, 48*gib); ok || !strings.Contains(why, "many") {
		t.Errorf("an unparsable slot count: fits = %v, %q", ok, why)
	}
	// #25294's --moe-stream-cache Ns holds N slots per layer like --expert-cache N;
	// its budget form (GiB) is not a slot count
	if ok, why := fits(config{Args: []string{"--moe-stream-cache", "128s"}}, 16, iq3, 48*gib); !ok {
		t.Errorf("iq3 --moe-stream-cache 128s at 16 GiB does not fit: %s", why)
	}
	if ok, why := fits(config{Args: []string{"--moe-stream-cache", "128s"}}, 16, q8, 48*gib); ok || !strings.Contains(why, "24.9 GiB") {
		t.Errorf("q8 --moe-stream-cache 128s at 16 GiB: fits = %v, %q", ok, why)
	}
	if ok, why := fits(config{Args: []string{"--moe-stream-cache", "40"}}, 0, iq3, 48*gib); ok || !strings.Contains(why, `"40"`) {
		t.Errorf("a budget instead of slots: fits = %v, %q", ok, why)
	}
}

func TestWeightsOf(t *testing.T) {
	f := &gguf.File{Tensors: []gguf.Tensor{
		{Name: "token_embd.weight", Dims: []uint64{64, 10}, Type: gguf.F32},
		{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{64, 8, 4}, Type: gguf.F32},
		{Name: "blk.0.ffn_up_exps.weight", Dims: []uint64{64, 8, 4}, Type: gguf.F32},
		{Name: "blk.0.ffn_down_exps.weight", Dims: []uint64{8, 64, 4}, Type: gguf.F32},
		{Name: "blk.1.ffn_gate_exps.weight", Dims: []uint64{64, 8, 4}, Type: gguf.F32},
		{Name: "blk.1.ffn_up_exps.weight", Dims: []uint64{64, 8, 4}, Type: gguf.F32},
		{Name: "blk.1.ffn_down_exps.weight", Dims: []uint64{8, 64, 4}, Type: gguf.F32},
		{Name: "blk.0.ffn_gate_shexp.weight", Dims: []uint64{64, 8}, Type: gguf.F32},
	}}
	w, err := weightsOf(f, 1000000)
	if err != nil {
		t.Fatal(err)
	}
	// one expert is 3 matrices of 64*8 F32 = 6144 B; 2 layers of 4 experts
	if want := (weights{Bytes: 1000000, ExpertBytes: 2 * 4 * 6144, SlotBytes: 2 * 6144}); w != want {
		t.Errorf("weightsOf = %+v, want %+v", w, want)
	}
	f.Tensors = append(f.Tensors, gguf.Tensor{Name: "blk.2.ffn_up_exps.weight", Dims: []uint64{64, 8}, Type: gguf.F32})
	if _, err := weightsOf(f, 1000000); err == nil {
		t.Error("weightsOf accepted a 2-D expert tensor")
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
	noFit := func(cfg string, rep int) runResult {
		return runResult{Config: cfg, CeilingGiB: 16, Rep: rep, Status: statusDoesNotFit}
	}
	join := func(parts ...[]runResult) []runResult { return slices.Concat(parts...) }

	v := evalGate(m, g, join(cache, cpuMoe, []runResult{noFit("resident", 1)}))
	if !v.Pass || v.Baseline != "cpu-moe" || !v.DecodeWin || v.TailWin || v.Incomplete != "" {
		t.Fatalf("verdict = %+v, want a decode win over cpu-moe (resident does not run)", v)
	}

	// overlapping repetitions are no win, although the median is higher
	var overlap []runResult
	for rep, d := range []float64{10, 21, 11} {
		overlap = append(overlap, fakeRun("cpu-moe", 16, rep+1, []float64{d, d}, []float64{90, 90}, nil))
	}
	v = evalGate(m, g, join(cache, overlap, []runResult{noFit("resident", 1)}))
	if v.DecodeWin {
		t.Fatalf("verdict = %+v: overlapping repetitions must not win on decode", v)
	}
	if !v.TailWin || !v.Pass {
		t.Fatalf("verdict = %+v: every p95 of the cache is below the baseline's best", v)
	}

	// no baseline that runs: no verdict
	v = evalGate(m, g, join(cache, []runResult{noFit("cpu-moe", 1), noFit("cpu-moe", 2), noFit("cpu-moe", 3), noFit("resident", 1)}))
	if v.Pass || v.Baseline != "" || v.Incomplete != "" {
		t.Fatalf("verdict = %+v, want none without a running baseline", v)
	}

	// a baseline that failed every repetition did not get beaten: no verdict,
	// although the other baseline loses to the candidate
	for _, runs := range [][]runResult{
		join(cache, cpuMoe, []runResult{failed("resident", 1)}),
		join(cache, cpuMoe, []runResult{{Config: "resident", CeilingGiB: 16, Rep: 1, Status: statusRequestFailed}}),
	} {
		v = evalGate(m, g, runs)
		if v.Pass || v.Baseline != "" || !strings.Contains(v.Incomplete, "resident 0 of 1") {
			t.Errorf("verdict = %+v, want none with a baseline that failed", v)
		}
	}
	// nor does a candidate that failed every repetition count as not running
	v = evalGate(m, g, join([]runResult{failed("cache", 1), failed("cache", 2), failed("cache", 3)}, cpuMoe, []runResult{noFit("resident", 1)}))
	if v.Pass || !strings.Contains(v.Incomplete, "cache 0 of 3") {
		t.Errorf("verdict = %+v, want none with a candidate that failed", v)
	}

	// a repetition missing or failed, of the candidate or a baseline: no verdict
	for name, runs := range map[string][]runResult{
		"cpu-moe":  join(cache, cpuMoe[:2], []runResult{noFit("resident", 1)}),
		"cache":    join(cache[:2], []runResult{failed("cache", 3)}, cpuMoe, []runResult{noFit("resident", 1)}),
		"resident": join(cache, cpuMoe),
	} {
		v = evalGate(m, g, runs)
		if v.Pass || !strings.Contains(v.Incomplete, name) {
			t.Errorf("%s incomplete: verdict = %+v, want none", name, v)
		}
	}
}

// withPrefill sets a run's warm prompt rate and cold time to first token.
func withPrefill(r runResult, prompt, ttftMs float64) runResult {
	for _, q := range r.Requests {
		q.Summary.Timings.PromptPerSecond = prompt
		if q.Phase == "cold" {
			q.Summary.TTFTMs = ttftMs
		}
	}
	return r
}

func TestGateMetrics(t *testing.T) {
	m := matrix{Configs: []config{
		{Name: "waves", Ceilings: []float64{0}, Reps: 2},
		{Name: "capped", Ceilings: []float64{0}, Reps: 2},
		{Name: "stream", Ceilings: []float64{0}, Reps: 2},
	}}
	runs := []runResult{
		withPrefill(fakeRun("waves", 0, 1, []float64{17, 17}, []float64{80, 80}, nil), 300, 9000),
		withPrefill(fakeRun("waves", 0, 2, []float64{18, 18}, []float64{80, 80}, nil), 310, 8800),
		withPrefill(fakeRun("capped", 0, 1, []float64{18, 18}, []float64{70, 70}, nil), 110, 24000),
		withPrefill(fakeRun("capped", 0, 2, []float64{19, 19}, []float64{70, 70}, nil), 112, 23000),
		withPrefill(fakeRun("stream", 0, 1, []float64{20, 20}, []float64{60, 60}, nil), 283, 8600),
		withPrefill(fakeRun("stream", 0, 2, []float64{21, 21}, []float64{60, 60}, nil), 290, 8700),
	}
	for _, tc := range []struct {
		metric    string
		baselines []string
		best      string
		pass      bool
	}{
		// every repetition above the best baseline's fastest: 300 > 290
		{"prefill", []string{"capped", "stream"}, "stream", true},
		// the lowest TTFT is the best baseline, and 8,800 is not below its 8,600
		{"ttft", []string{"capped", "stream"}, "stream", false},
		{"ttft", []string{"capped"}, "capped", true},
		// not slower: the candidate's median 17.5 against capped's slowest 18
		{"decode-not-slower", []string{"capped"}, "capped", false},
		// the default stays the decode rule: no decode or p95 win over stream
		{"", []string{"capped", "stream"}, "stream", false},
	} {
		v := evalGate(m, gate{CeilingGiB: 0, Candidate: "waves", Baselines: tc.baselines, Metric: tc.metric, Decides: true}, runs)
		if v.Baseline != tc.best || v.Pass != tc.pass || v.Incomplete != "" {
			t.Errorf("%q against %v: verdict %+v, want best %s, pass %v", tc.metric, tc.baselines, v, tc.best, tc.pass)
		}
	}
	slower := append(slices.Clone(runs[:2]), withPrefill(fakeRun("capped", 0, 1, []float64{17.2, 17.2}, []float64{70, 70}, nil), 0, 0),
		withPrefill(fakeRun("capped", 0, 2, []float64{19, 19}, []float64{70, 70}, nil), 0, 0))
	if v := evalGate(m, gate{Candidate: "waves", Baselines: []string{"capped"}, Metric: "decode-not-slower"}, slower); !v.Pass {
		t.Errorf("median 17.5 against a slowest repetition of 17.2: verdict %+v, want passed", v)
	}
	if err := (matrix{Gates: []gate{{Metric: "latency"}}}).validate(); err == nil {
		t.Error("an unknown gate metric is accepted")
	}
}

// The decode-not-slower line names its threshold, the slowest repetition, not
// the baseline's median.
func TestReportDecodeNotSlowerThreshold(t *testing.T) {
	m := matrix{Cold: "coding", Warm: []string{"coding", "research"},
		Configs: []config{
			{Name: "waves", Ceilings: []float64{0}, Reps: 1},
			{Name: "capped", Ceilings: []float64{0}, Reps: 3},
		},
		Gates: []gate{{CeilingGiB: 0, Candidate: "waves", Baselines: []string{"capped"}, Metric: "decode-not-slower", Decides: true}}}
	runs := []runResult{
		fakeRun("waves", 0, 1, []float64{18, 18}, []float64{80, 80}, nil),
		fakeRun("capped", 0, 1, []float64{19, 19}, []float64{70, 70}, nil),
		fakeRun("capped", 0, 2, []float64{15, 15}, []float64{70, 70}, nil),
		fakeRun("capped", 0, 3, []float64{17, 17}, []float64{70, 70}, nil),
	}
	var out strings.Builder
	report(&out, m, runs)
	want := "passed: decode median 18.00 against capped's slowest of 15.00 (median 17.00, 15.00–19.00) t/s"
	if !strings.Contains(out.String(), want) {
		t.Errorf("want %q in the report:\n%s", want, out.String())
	}
}

func TestReportCountsRequests(t *testing.T) {
	m := matrix{Cold: "coding", Warm: []string{"coding"},
		Configs: []config{
			{Name: "ref", Ceilings: []float64{0}, Reps: 1},
			{Name: "cache", Ceilings: []float64{0}, Reps: 2},
		},
		Equivalence: []equivalence{{Reference: "ref", Arms: []string{"cache"}, Exact: true}}}
	if got := m.expectedRequests([]string{"cache"}, nil); got != 4 {
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

// A run that fits does not start is a result, not a missing run: the
// equivalence and the load-failure check expect nothing from it.
func TestReportSkipsDoesNotFit(t *testing.T) {
	m := matrix{Cold: "coding", Warm: []string{"coding"},
		Configs: []config{
			{Name: "ref", Ceilings: []float64{0}, Reps: 1},
			{Name: "cache", Args: []string{"--expert-cache", "2"}, Ceilings: []float64{0, 16}, Reps: 1},
		},
		Equivalence: []equivalence{{Reference: "ref", Arms: []string{"cache"}, Exact: true}}}
	ids := []int{1, 2, 3}
	ref := fakeRun("ref", 0, 1, []float64{50}, []float64{20}, ids)
	ok := fakeRun("cache", 0, 1, []float64{20}, []float64{50}, ids)
	ok.Stats = &cacheStats{}
	nofit := runResult{Config: "cache", CeilingGiB: 16, Rep: 1, Status: statusDoesNotFit}
	var out strings.Builder
	if !report(&out, m, []runResult{ref, ok, nofit}) {
		t.Fatalf("a run that does not fit fails the report:\n%s", out.String())
	}
	for _, want := range []string{
		"PASS equivalence cache against ref: 2 of 2 requests",
		"PASS expert load failures: 0 in the 1 of 1 cache runs that completed and wrote their stats (1 did not fit)",
	} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("report lacks %q:\n%s", want, out.String())
		}
	}
	nofit.Status = statusLoadFailed
	if report(io.Discard, m, []runResult{ref, ok, nofit}) {
		t.Error("a run that failed to load passes")
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
	// the stats of a build with waves have more keys (fork branch feat/kolibri1-waves): the same split
	var waves cacheStats
	if err := json.Unmarshal([]byte(`{"expert_cache": {"buffered": true, "waves": true}, "decode": {"needed": 10, "wait_seconds": 9.2,
		"remap_seconds": 1.6, "tokens": 640, "wall_seconds": 34.56, "splits": 65280, "metal_commits": 66560,
		"waves": 0, "waves_built": 0, "wave_parked": 0},
		"prefill": {"waves": 195, "waves_built": 200, "wave_parked": 460800}, "process": {}}`), &waves); err != nil {
		t.Fatal(err)
	}
	if w, ok := waves.split(); !ok || !near(w.Wall, 54) || !near(w.Other, 37.125) || w.Splits != 102 {
		t.Fatalf("split with waves = %+v, %v", w, ok)
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
	for _, path := range []string{"../../testdata/bench/expert-cache.json", "../../testdata/bench/e1.json", "../../testdata/bench/e0.json", "../../testdata/bench/e4.json", "../../testdata/bench/e2.json"} {
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
				if ok, _ := fits(c, ceil, weights{Bytes: 35556257600}, 48*gib); ok {
					t.Errorf("%s fits at %g GiB", c.Name, ceil)
				}
			}
		}
	}
}

// TestE0Matrix keeps E0.2's upstream arms comparable with the fork's cache:
// Stage D's workloads, and the same memory for every cache, 128 IQ3 experts
// per layer (#29887's budget is that in MiB, as it caches bytes, not slots).
func TestE0Matrix(t *testing.T) {
	d, e := loadMatrix(t, "../../testdata/bench/expert-cache.json"), loadMatrix(t, "../../testdata/bench/e0.json")
	if !slices.Equal(d.ServerArgs, e.ServerArgs) || d.PromptFirst != e.PromptFirst || d.NPredict != e.NPredict ||
		d.Cold != e.Cold || !slices.Equal(d.Warm, e.Warm) {
		t.Errorf("E0's workload settings differ from Stage D's")
	}
	const iq3Expert = 1699840 // bytes of one IQ3 expert (gate, up and down), see TestFits
	want := map[string][]string{
		"iq3-cache128":     {"--expert-cache", "128"},
		"iq3-moestream128": {"--moe-stream-cache", "128s"},
		"iq3-moecache":     {"--cpu-moe", "--moe-cache-mib", strconv.Itoa(128 * 50 * iq3Expert >> 20)},
	}
	for name, args := range want {
		c, ok := e.config(name)
		if !ok || !slices.Equal(c.Args, args) {
			t.Errorf("%s: args %v, want %v", name, c.Args, args)
		}
	}
	if c, _ := e.config("iq3-moestream128"); c.Server != "pr25294" {
		t.Errorf("iq3-moestream128 runs on %q, want the pr25294 build", c.Server)
	}
	if missing := e.missingServers(map[string]string{"pr25294": "/b"}, nil); len(missing) != 0 {
		t.Errorf("E0 names builds beyond pr25294: %v", missing)
	}
}

// TestE4Matrix keeps E4's waves comparable with Stage D's cache and E0.2's
// #25294 arm (the same workloads, arms and slots), and fixes its gates as in
// the plan: prefill and TTFT against the best of both, decode against Stage
// D's cache, at both ceilings.
func TestE4Matrix(t *testing.T) {
	z, e := loadMatrix(t, "../../testdata/bench/e0.json"), loadMatrix(t, "../../testdata/bench/e4.json")
	if !slices.Equal(z.ServerArgs, e.ServerArgs) || z.PromptFirst != e.PromptFirst || z.NPredict != e.NPredict ||
		z.Cold != e.Cold || !slices.Equal(z.Warm, e.Warm) {
		t.Errorf("E4's workload settings differ from E0.2's")
	}
	for _, name := range []string{"iq3-cpu-moe", "iq3-cache128", "iq3-moestream128"} {
		a, _ := z.config(name)
		b, ok := e.config(name)
		if !ok || !slices.Equal(a.Args, b.Args) || a.Server != b.Server || a.Model != b.Model {
			t.Errorf("%s differs from E0.2's: %+v against %+v", name, b, a)
		}
	}
	if c, ok := e.config("iq3-cache128-waves"); !ok || !slices.Equal(c.Args, []string{"--expert-cache", "128", "--expert-cache-waves"}) {
		t.Errorf("iq3-cache128-waves: args %v", c.Args)
	}
	// buffered waves tell a read-mode loss against #25294 (which reads buffered)
	// from a loss of the design; they decide nothing. 16 GiB was added after the
	// first 27 runs, to see whether one mode wins at both ceilings.
	if c, ok := e.config("iq3-cache128-waves-buffered"); !ok || !slices.Equal(c.Args, []string{"--expert-cache", "128", "--expert-cache-waves", "--expert-cache-buffered"}) ||
		!slices.Equal(c.Ceilings, []float64{0, 16}) {
		t.Errorf("iq3-cache128-waves-buffered: %+v", c)
	}
	var got []string
	for _, g := range e.Gates {
		if g.Candidate == "iq3-cache128-waves-buffered" {
			if g.Decides || !slices.Equal(g.Baselines, []string{"iq3-cache128", "iq3-moestream128"}) {
				t.Errorf("buffered waves gate %+v", g)
			}
			continue
		}
		if g.Candidate != "iq3-cache128-waves" {
			t.Errorf("gate of %s", g.Candidate)
		}
		got = append(got, fmt.Sprintf("%s %g %v %s", g.Metric, g.CeilingGiB, g.Decides, strings.Join(g.Baselines, "+")))
	}
	want := []string{
		"prefill 0 true iq3-cache128+iq3-moestream128", "prefill 16 true iq3-cache128+iq3-moestream128",
		"ttft 0 true iq3-cache128+iq3-moestream128", "ttft 16 true iq3-cache128+iq3-moestream128",
		"decode-not-slower 0 true iq3-cache128", "decode-not-slower 16 true iq3-cache128",
		"prefill 0 false iq3-cpu-moe", "prefill 16 false iq3-cpu-moe",
	}
	if !slices.Equal(got, want) {
		t.Errorf("gates %q, want %q", got, want)
	}
}

// TestE2Matrix keeps E2's arms comparable with Stage D's and E4's cache (the same
// workloads, model and slots), the GPU arm differing only by the option, and fixes
// what the plan asks for: decode by the Stage D rule at both ceilings, against the
// best other cache arm, and token IDs exactly those of the resident run in the same
// ubatches. Resident at 0 GiB is that reference, not a baseline: Stage D decided at
// 16 GiB, where it does not fit, and E2's bar is Stage D's cache. The comparison
// with the CPU remap stays INFO: its splits round differently from the unsplit
// graph the GPU remap keeps (docs/upstream.md, E0.1).
func TestE2Matrix(t *testing.T) {
	d, e := loadMatrix(t, "../../testdata/bench/e4.json"), loadMatrix(t, "../../testdata/bench/e2.json")
	if !slices.Equal(d.ServerArgs, e.ServerArgs) || d.PromptFirst != e.PromptFirst || d.NPredict != e.NPredict ||
		d.Cold != e.Cold || !slices.Equal(d.Warm, e.Warm) {
		t.Errorf("E2's workload settings differ from E4's")
	}
	base, ok := e.config("iq3-cache128")
	if want, _ := d.config("iq3-cache128"); !ok || !slices.Equal(base.Args, want.Args) || base.Model != want.Model || base.Server != "" {
		t.Errorf("iq3-cache128 differs from E4's: %+v against %+v", base, want)
	}
	arm, ok := e.config("iq3-cache128-gpuremap")
	if !ok || !slices.Equal(arm.Args, append(slices.Clone(base.Args), "--expert-cache-gpu-remap")) || arm.Model != base.Model || arm.Server != base.Server {
		t.Errorf("iq3-cache128-gpuremap differs by more than the option: %+v", arm)
	}
	// the other two as Stage D ran them
	stageD := loadMatrix(t, "../../testdata/bench/expert-cache.json")
	for _, name := range []string{"iq3-cache128-prefetch12", "iq3-resident-ub21"} {
		c, ok := e.config(name)
		want, _ := stageD.config(name)
		if !ok || c.Model != want.Model || c.Server != want.Server || !slices.Equal(c.Args, want.Args) || !slices.Equal(c.Ceilings, want.Ceilings) || c.Reps != want.Reps {
			t.Errorf("%s differs from Stage D's: %+v against %+v", name, c, want)
		}
	}
	for _, c := range []config{base, arm} {
		if !slices.Equal(c.Ceilings, []float64{0, 16}) || c.Reps != 3 {
			t.Errorf("%s: ceilings %v, %d reps", c.Name, c.Ceilings, c.Reps)
		}
	}
	if len(e.Configs) != 4 {
		t.Errorf("%d configs, want the two arms, the prefetch baseline and the resident reference", len(e.Configs))
	}
	if n := len(e.expand()); n != 6+6+6+1 {
		t.Errorf("%d runs, want 19", n)
	}
	var got []string
	for _, g := range e.Gates {
		got = append(got, fmt.Sprintf("%s %s %g %v %s", g.Candidate, g.Metric, g.CeilingGiB, g.Decides, strings.Join(g.Baselines, "+")))
	}
	want := []string{
		"iq3-cache128-gpuremap decode 0 true iq3-cache128+iq3-cache128-prefetch12",
		"iq3-cache128-gpuremap decode 16 true iq3-cache128+iq3-cache128-prefetch12",
	}
	if !slices.Equal(got, want) {
		t.Errorf("gates %q, want %q", got, want)
	}
	var eq []string
	for _, q := range e.Equivalence {
		eq = append(eq, fmt.Sprintf("%s %s %v", q.Reference, strings.Join(q.Arms, "+"), q.Exact))
	}
	wantEq := []string{
		"iq3-resident-ub21 iq3-cache128-gpuremap true",
		"iq3-cache128 iq3-cache128-gpuremap false",
	}
	if !slices.Equal(eq, wantEq) {
		t.Errorf("equivalence %q, want %q", eq, wantEq)
	}
}

func checkMatrix(t *testing.T, m matrix) {
	if err := m.validate(); err != nil {
		t.Error(err)
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
