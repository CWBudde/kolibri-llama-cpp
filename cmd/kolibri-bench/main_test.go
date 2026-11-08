package main

import (
	"encoding/json"
	"errors"
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

// decode-and-p95 (Stage E5's speed gate) needs both wins of the decode rule:
// every repetition faster than the best baseline's fastest one, and every warm
// p95 below that baseline's lowest one.
func TestGateDecodeAndP95(t *testing.T) {
	m := matrix{Configs: []config{
		{Name: "skip", Ceilings: []float64{16}, Reps: 2},
		{Name: "lossless", Ceilings: []float64{16}, Reps: 2},
		{Name: "slow", Ceilings: []float64{16}, Reps: 2},
	}}
	runs := func(cfg string, dec [2]float64, tail [2]float64) []runResult {
		return []runResult{
			fakeRun(cfg, 16, 1, []float64{dec[0], dec[0]}, []float64{tail[0], tail[0]}, nil),
			fakeRun(cfg, 16, 2, []float64{dec[1], dec[1]}, []float64{tail[1], tail[1]}, nil),
		}
	}
	// the baselines: lossless is the best one (median decode 13 against 6), and
	// slow has the lower p95, which must not count: the p95 is lossless's
	baselines := slices.Concat(runs("lossless", [2]float64{12, 14}, [2]float64{90, 100}), runs("slow", [2]float64{5, 7}, [2]float64{40, 40}))
	for _, tc := range []struct {
		name               string
		dec, tail          [2]float64
		decodeWin, tailWin bool
	}{
		// (a) both win: 15 > 14 and 85 < 90
		{"both win", [2]float64{15, 16}, [2]float64{80, 85}, true, true},
		// (b) decode wins, the p95 does not: 90 is not below 90
		{"decode only", [2]float64{15, 16}, [2]float64{80, 90}, true, false},
		// (c) the p95 wins, decode does not: 14 is not faster than 14
		{"p95 only", [2]float64{14, 16}, [2]float64{80, 85}, false, true},
		// (d) neither
		{"neither", [2]float64{10, 11}, [2]float64{95, 110}, false, false},
	} {
		all := slices.Concat(runs("skip", tc.dec, tc.tail), baselines)
		g := gate{CeilingGiB: 16, Candidate: "skip", Baselines: []string{"slow", "lossless"}, Metric: "decode-and-p95", Decides: true}
		v := evalGate(m, g, all)
		pass := tc.decodeWin && tc.tailWin
		if v.Baseline != "lossless" || v.DecodeWin != tc.decodeWin || v.TailWin != tc.tailWin || v.Pass != pass || v.Incomplete != "" {
			t.Errorf("%s: verdict %+v, want lossless, decode win %v, p95 win %v, pass %v", tc.name, v, tc.decodeWin, tc.tailWin, pass)
		}
		// the decode rule on the same runs passes on either win
		g.Metric = "decode"
		if d := evalGate(m, g, all); d.Pass != (tc.decodeWin || tc.tailWin) {
			t.Errorf("%s: decode verdict %+v, want pass %v", tc.name, d, tc.decodeWin || tc.tailWin)
		}
	}
}

// The decode-and-p95 line shows both sub-results with their numbers.
func TestReportDecodeAndP95(t *testing.T) {
	m := matrix{Cold: "coding", Warm: []string{"coding", "research"},
		Configs: []config{
			{Name: "skip", Ceilings: []float64{16}, Reps: 1},
			{Name: "lossless", Ceilings: []float64{16}, Reps: 1},
		},
		Gates: []gate{{CeilingGiB: 16, Candidate: "skip", Baselines: []string{"lossless"}, Metric: "decode-and-p95", Decides: true}}}
	runs := []runResult{
		fakeRun("skip", 16, 1, []float64{15, 15}, []float64{95, 95}, nil),
		fakeRun("lossless", 16, 1, []float64{12, 12}, []float64{90, 90}, nil),
	}
	var out strings.Builder
	report(&out, m, runs)
	want := "- GATE (decode-and-p95) skip at 16 GiB against the best of lossless: not passed, needs both wins: " +
		"decode 15.00 against lossless's 12.00 t/s (win true), p95 95.0 against 90.0 ms (win false)"
	if !strings.Contains(out.String(), want) {
		t.Errorf("want %q in the report:\n%s", want, out.String())
	}
}

func TestGateMetricValidation(t *testing.T) {
	for _, metric := range []string{"", "decode", "prefill", "ttft", "decode-not-slower", "decode-and-p95"} {
		if err := (matrix{Gates: []gate{{Metric: metric}}}).validate(); err != nil {
			t.Errorf("metric %q: %v", metric, err)
		}
	}
	for _, metric := range []string{"decode-or-p95", "p95", "Decode-and-p95"} {
		if err := (matrix{Gates: []gate{{Metric: metric}}}).validate(); err == nil {
			t.Errorf("an unknown gate metric %q is accepted", metric)
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
	if p := peaks(strings.NewReader(samples)); p != (samplePeaks{120, 4096, -1, -1}) {
		t.Fatalf("peaks = %+v; want 120, 4096, -1", p)
	}
	samples = "time rss_mib footprint_mib swap_used_mib pressure tokens pageins page_cache_mib\n" +
		"02:00:00 100 2048 9000 1 0 5 300\n02:00:02 120 4096 9000 1 10 5 -\n02:00:04 110 3000 9000 1 20 5 2500\n"
	if p := peaks(strings.NewReader(samples)); p != (samplePeaks{120, 4096, 2500, -1}) {
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
	// a build that skips experts (Stage E5) counts them per phase, with the
	// weight below which it skips in the header
	var skip cacheStats
	if err := json.Unmarshal([]byte(`{"expert_cache": {"buffered": false, "skip_weight": 0.05, "wait": "spin"},
		"prefill": {"loads": 7, "skipped": 120, "prefetch_skipped": 3},
		"decode": {"needed": 10, "skipped": 254, "prefetch_skipped": 9, "tokens": 635, "wall_seconds": 25.4}, "process": {}}`), &skip); err != nil {
		t.Fatal(err)
	}
	if sw := skip.ExpertCache.SkipWeight; sw == nil || *sw != 0.05 {
		t.Fatalf("skip_weight = %v, want 0.05", sw)
	}
	if s := skip.Decode.Skipped; s == nil || *s != 254 {
		t.Fatalf("decode skipped = %v, want 254", s)
	}
	if s := skip.Prefill.Skipped; s == nil || *s != 120 {
		t.Fatalf("prefill skipped = %v, want 120", s)
	}
	if got, ok := skip.skippedPerToken(); !ok || !near(got, 0.4) {
		t.Fatalf("skipped per token = %v, %v; want 0.4", got, ok)
	}
	// an older build writes neither: absent, not zero
	if cache.ExpertCache.SkipWeight != nil || cache.Decode.Skipped != nil || cache.Prefill.Skipped != nil {
		t.Fatalf("stats without the skip keys: %v, %v, %v", cache.ExpertCache.SkipWeight, cache.Decode.Skipped, cache.Prefill.Skipped)
	}
	if _, ok := cache.skippedPerToken(); ok {
		t.Fatal("stats without skipped give a skip count")
	}
	if _, ok := (*cacheStats)(nil).skippedPerToken(); ok {
		t.Fatal("no stats give a skip count")
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
		"| res | 0 GiB | 16.50 (16.50–16.50) | 0.00 (0.00–0.00) | 0.000 (0.000–0.000) | 16.50 (16.50–16.50) | 2.0 (2.0–2.0) | 3.0 (3.0–3.0) | 2.0 (2.0–2.0) | 4.0 (4.0–4.0) | 1.0 (1.0–1.0) | – |",
		"| old | 0 GiB | – | – | – | – | – | – | – | – | – | – |",
	} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("decode table lacks %q:\n%s", want, out.String())
		}
	}
}

// The skipped experts per decoded token are the mean over a config's runs, and
// blank for a build that does not count them.
func TestDecodeTableSkipped(t *testing.T) {
	m := matrix{Configs: []config{{Name: "skip", Ceilings: []float64{16}, Reps: 2}, {Name: "lossless", Ceilings: []float64{16}, Reps: 1}}}
	run := func(cfg string, rep int, tokens int64, skipped *int64) runResult {
		r := runResult{Config: cfg, CeilingGiB: 16, Rep: rep, Status: statusOK, Stats: &cacheStats{}}
		r.Stats.Decode.Tokens, r.Stats.Decode.WallSeconds, r.Stats.Decode.Skipped = tokens, 1, skipped
		return r
	}
	s1, s2, zero := int64(100), int64(300), int64(0)
	var out strings.Builder
	// 100/400 and 300/500 tokens: a mean of (0.25 + 0.6) / 2
	decodeTable(&out, m, []runResult{run("skip", 1, 400, &s1), run("skip", 2, 500, &s2), run("lossless", 1, 400, nil)})
	if !strings.Contains(out.String(), "| Skipped/token |") {
		t.Errorf("decode table lacks the skip column:\n%s", out.String())
	}
	for _, want := range []string{"| skip | 16 GiB |", "| lossless | 16 GiB |"} {
		i := strings.Index(out.String(), want)
		if i < 0 {
			t.Fatalf("decode table lacks %q:\n%s", want, out.String())
		}
		line, _, _ := strings.Cut(out.String()[i:], "\n")
		cells := strings.Split(strings.Trim(line, "| "), " | ")
		got, wantCell := cells[len(cells)-1], "0.425"
		if want == "| lossless | 16 GiB |" {
			wantCell = "–"
		}
		if got != wantCell {
			t.Errorf("%s skipped/token %q, want %q", want, got, wantCell)
		}
	}
	// a build that counts and skipped nothing shows 0
	out.Reset()
	decodeTable(&out, m, []runResult{run("skip", 1, 400, &zero), run("skip", 2, 500, &zero)})
	if !strings.Contains(out.String(), " | 0.000 |") {
		t.Errorf("no skips shows no zero:\n%s", out.String())
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
	for _, path := range []string{"../../testdata/bench/expert-cache.json", "../../testdata/bench/e1.json", "../../testdata/bench/e0.json", "../../testdata/bench/e4.json", "../../testdata/bench/e3.json", "../../testdata/bench/e2.json",
		"../../testdata/bench/e2e3.json", "../../testdata/bench/e2e3-prefetch.json", "../../testdata/bench/e2e3-prefetch-16g.json",
		"../../testdata/bench/e2e3-spin.json", "../../testdata/bench/e5-skip.json"} {
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

// TestE2E3Matrix fixes the three registered runs of E2 and E3 together: E4's
// workloads, each arm the GGUF arm plus only its options, and the deciding gates
// as they were committed before the first timed run (docs/expert-streaming.md).
func TestE2E3Matrix(t *testing.T) {
	d := loadMatrix(t, "../../testdata/bench/e4.json")
	base, _ := d.config("iq3-cache128")
	sideFile := base.Model + ".experts"
	with := func(extra ...string) []string { return append(slices.Clone(base.Args), extra...) }
	wantArgs := map[string][]string{
		"iq3-cache128":                     base.Args,
		"iq3-cache128-gpuremap":            with("--expert-cache-gpu-remap"),
		"iq3-cache128-sidefile":            with(sidefileFlag, sideFile),
		"iq3-cache128-sidefile-gpuremap":   with(sidefileFlag, sideFile, "--expert-cache-gpu-remap"),
		"iq3-cache128-prefetch12":          with("--expert-prefetch", "12"),
		"iq3-cache128-sidefile-prefetch12": with(sidefileFlag, sideFile, "--expert-prefetch", "12"),
	}
	for _, tc := range []struct {
		path     string
		arms     []string
		ceilings []float64
		runs     int
		deciding []string
	}{
		{"e2e3.json", []string{"iq3-cache128", "iq3-cache128-gpuremap", "iq3-cache128-sidefile", "iq3-cache128-sidefile-gpuremap", "iq3-resident-ub21"},
			[]float64{0, 16}, 4*6 + 1, []string{
				"iq3-cache128-sidefile-gpuremap decode 0 iq3-cache128-sidefile+iq3-cache128-gpuremap",
				"iq3-cache128-sidefile-gpuremap decode 16 iq3-cache128-sidefile+iq3-cache128-gpuremap",
			}},
		{"e2e3-prefetch.json", []string{"iq3-cache128-sidefile-gpuremap", "iq3-cache128-prefetch12", "iq3-cache128-sidefile-prefetch12", "iq3-resident-ub21"},
			[]float64{0, 16}, 3*6 + 1, []string{
				"iq3-cache128-sidefile-gpuremap decode 0 iq3-cache128-sidefile-prefetch12+iq3-cache128-prefetch12",
				"iq3-cache128-sidefile-gpuremap decode 16 iq3-cache128-sidefile-prefetch12+iq3-cache128-prefetch12",
			}},
		{"e2e3-prefetch-16g.json", []string{"iq3-cache128-sidefile-gpuremap", "iq3-cache128-prefetch12", "iq3-cache128-sidefile-prefetch12"},
			[]float64{16}, 3 * 3, []string{
				"iq3-cache128-sidefile-gpuremap decode 16 iq3-cache128-sidefile-prefetch12+iq3-cache128-prefetch12",
			}},
	} {
		t.Run(tc.path, func(t *testing.T) {
			e := loadMatrix(t, "../../testdata/bench/"+tc.path)
			if !slices.Equal(d.ServerArgs, e.ServerArgs) || d.PromptFirst != e.PromptFirst || d.NPredict != e.NPredict ||
				d.Cold != e.Cold || !slices.Equal(d.Warm, e.Warm) {
				t.Errorf("workload settings differ from E4's")
			}
			var names []string
			for _, c := range e.Configs {
				names = append(names, c.Name)
				if c.Name == "iq3-resident-ub21" {
					continue
				}
				if c.Model != base.Model || c.Server != base.Server || !slices.Equal(c.Args, wantArgs[c.Name]) {
					t.Errorf("%s: %+v, want %q", c.Name, c, wantArgs[c.Name])
				}
				if !slices.Equal(c.Ceilings, tc.ceilings) || c.Reps != 3 {
					t.Errorf("%s: ceilings %v, %d reps", c.Name, c.Ceilings, c.Reps)
				}
			}
			if !slices.Equal(names, tc.arms) {
				t.Errorf("arms %q, want %q", names, tc.arms)
			}
			if n := len(e.expand()); n != tc.runs {
				t.Errorf("%d runs, want %d", n, tc.runs)
			}
			var got []string
			for _, g := range e.Gates {
				if g.Decides {
					got = append(got, fmt.Sprintf("%s %s %g %s", g.Candidate, g.Metric, g.CeilingGiB, strings.Join(g.Baselines, "+")))
				}
			}
			if !slices.Equal(got, tc.deciding) {
				t.Errorf("deciding gates %q, want %q", got, tc.deciding)
			}
		})
	}
}

// TestE3Matrix fixes E3's comparison: the side-file against the GGUF path, every
// other flag the same, Stage D's workloads, at both ceilings, 3 repetitions, and
// the plan's gates: decode and prefill at both ceilings decide. The side-file's
// path is exact and next to the GGUF, named as the model is. Bit identity makes
// exact token IDs the rule here.
func TestE3Matrix(t *testing.T) {
	d, e := loadMatrix(t, "../../testdata/bench/e4.json"), loadMatrix(t, "../../testdata/bench/e3.json")
	if !slices.Equal(d.ServerArgs, e.ServerArgs) || d.PromptFirst != e.PromptFirst || d.NPredict != e.NPredict ||
		d.Cold != e.Cold || !slices.Equal(d.Warm, e.Warm) {
		t.Errorf("E3's workload settings differ from E4's")
	}
	if len(e.Configs) != 2 || len(e.expand()) != 12 {
		t.Errorf("%d arms, %d runs, want 2 and 12", len(e.Configs), len(e.expand()))
	}
	a, _ := d.config("iq3-cache128")
	base, ok := e.config("iq3-cache128")
	if !ok || !slices.Equal(a.Args, base.Args) || a.Model != base.Model || a.Server != base.Server {
		t.Errorf("iq3-cache128 differs from E4's: %+v against %+v", base, a)
	}
	side, ok := e.config("iq3-cache128-sidefile")
	sideFile := base.Model + ".experts"
	if !ok || side.Model != base.Model || side.Server != "" ||
		!slices.Equal(side.Args, append(slices.Clone(base.Args), sidefileFlag, sideFile)) {
		t.Errorf("iq3-cache128-sidefile: %+v, want the GGUF arm's flags and %s %s", side, sidefileFlag, sideFile)
	}
	for _, c := range e.Configs {
		if !slices.Equal(c.Ceilings, []float64{0, 16}) || c.Reps != 3 {
			t.Errorf("%s: ceilings %v, %d repetitions", c.Name, c.Ceilings, c.Reps)
		}
	}
	if len(e.Equivalence) != 1 || e.Equivalence[0].Reference != "iq3-cache128" || !e.Equivalence[0].Exact ||
		!slices.Equal(e.Equivalence[0].Arms, []string{"iq3-cache128-sidefile"}) {
		t.Errorf("equivalence %+v", e.Equivalence)
	}
	var got []string
	for _, g := range e.Gates {
		if g.Candidate != "iq3-cache128-sidefile" || !slices.Equal(g.Baselines, []string{"iq3-cache128"}) {
			t.Errorf("gate %+v", g)
		}
		got = append(got, fmt.Sprintf("%s %g %v", g.Metric, g.CeilingGiB, g.Decides))
	}
	want := []string{"decode 0 true", "decode 16 true", "prefill 0 true", "prefill 16 true", "ttft 0 false", "ttft 16 false"}
	if !slices.Equal(got, want) {
		t.Errorf("gates %q, want %q", got, want)
	}
	// the same guard as the arm it is compared with
	const gib = 1 << 30
	w := weights{Bytes: 35556257600, ExpertBytes: 32636928000, SlotBytes: 50 * 1699840}
	for _, ceil := range []float64{0, 16} {
		okA, whyA := fits(base, ceil, w, 48*gib)
		okB, whyB := fits(side, ceil, w, 48*gib)
		if !okA || !okB || whyA != whyB {
			t.Errorf("fits at %g GiB: %v %q against %v %q", ceil, okB, whyB, okA, whyA)
		}
	}
}

func TestSidefileFlags(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	m := matrix{Configs: []config{
		{Name: "a", Model: "~/m.gguf", Args: []string{"--expert-cache", "128"}},
		{Name: "b", Model: "~/m.gguf", Args: []string{"--expert-cache", "128", sidefileFlag, "~/m.gguf.experts"}},
		{Name: "c", Model: "m.gguf", Args: []string{"--expert-cache", "128", sidefileFlag, "rel/m.experts"}},
		{Name: "d", Model: "m.gguf", Args: []string{"--expert-cache", "128", sidefileFlag, "~/m.gguf.experts"}},
	}}
	if err := m.validate(); err != nil {
		t.Fatal(err)
	}
	orig := slices.Clone(m.Configs[1].Args)
	m.expandPaths()
	if m.Configs[1].Model != home+"/m.gguf" {
		t.Errorf("model %s", m.Configs[1].Model)
	}
	got, ok := sidefileArg(m.Configs[1].Args)
	if !ok || got != home+"/m.gguf.experts" || !slices.Equal(orig, []string{"--expert-cache", "128", sidefileFlag, "~/m.gguf.experts"}) {
		t.Errorf("side-file %s", got)
	}
	if got, _ := sidefileArg(m.Configs[2].Args); got != "rel/m.experts" {
		t.Errorf("a relative side-file became %s", got)
	}
	if got := m.sidefiles(); !slices.Equal(got, []string{home + "/m.gguf.experts", "rel/m.experts"}) {
		t.Errorf("side-files %v", got)
	}
	if _, ok := sidefileArg(m.Configs[0].Args); ok {
		t.Error("an arm without the flag has a side-file")
	}
	args := m.serverArgs(m.Configs[1], 1, "s.json")
	if at := slices.Index(args, sidefileFlag); at < 0 || args[at+1] != home+"/m.gguf.experts" {
		t.Errorf("the server gets %v", args)
	}
	for name, bad := range map[string]config{
		"no file":    {Name: "x", Args: []string{"--expert-cache", "128", sidefileFlag}},
		"no cache":   {Name: "x", Args: []string{sidefileFlag, "f"}},
		"other tree": {Name: "x", Server: "pr25294", Args: []string{"--expert-cache", "128", sidefileFlag, "f"}},
	} {
		if err := (matrix{Configs: []config{bad}}).validate(); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// TestMissingSidefile: a missing side-file stops the benchmark before a run.
func TestMissingSidefile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.experts")
	m := matrix{Configs: []config{{Name: "b", Args: []string{"--expert-cache", "128", sidefileFlag, path}}}}
	if err := m.checkSidefiles(); err == nil || !errors.Is(err, os.ErrNotExist) {
		t.Errorf("missing: %v", err)
	}
	if err := os.WriteFile(path, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := m.checkSidefiles(); err == nil {
		t.Error("an empty side-file passed")
	}
	if err := os.WriteFile(path, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := m.checkSidefiles(); err != nil {
		t.Error(err)
	}
	if err := (matrix{Configs: []config{{Name: "a", Args: []string{"--expert-cache", "128"}}}}).checkSidefiles(); err != nil {
		t.Errorf("a matrix without side-files: %v", err)
	}
}

// fakeEvict is a kolibri-evict that logs its arguments and fails on a file
// named fail; it returns the bench and the log's path.
func fakeEvict(t *testing.T, m matrix, cached func(string) (float64, error)) (*bench, string) {
	t.Helper()
	dir := t.TempDir()
	logPath := filepath.Join(dir, "evicted")
	script := fmt.Sprintf(`#!/bin/sh
echo "$1" >> %q
case "$1" in *fail*) echo "$1: no such file" >&2; exit 1;; esac
echo "$1: 5 of 9 pages resident before, 0 after (0 MiB evicted)"
`, logPath)
	if err := os.WriteFile(filepath.Join(dir, "kolibri-evict"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return &bench{m: m, bin: dir, cached: cached}, logPath
}

// TestEvictSidefile: every run of every arm evicts the GGUF and the matrix's
// side-files, records it, and stops if one is missing or stays resident: the
// page cache serves resident pages even with F_NOCACHE, and writing or verifying
// a side-file leaves it resident, as a stray Metal server does for the GGUF.
func TestEvictSidefile(t *testing.T) {
	dir := t.TempDir()
	side := filepath.Join(dir, "m.experts")
	model := filepath.Join(dir, "m.gguf")
	for _, path := range []string{side, model} {
		if err := os.WriteFile(path, make([]byte, 1<<24), 0o644); err != nil { // 16 MiB
			t.Fatal(err)
		}
	}
	gguf := config{Name: "a", Model: model, Args: []string{"--expert-cache", "128"}}
	arm := config{Name: "b", Model: model, Args: []string{"--expert-cache", "128", sidefileFlag, side}}
	m := matrix{Configs: []config{gguf, arm}}
	for _, c := range m.Configs {
		b, logPath := fakeEvict(t, m, func(string) (float64, error) { return 0.1, nil })
		var res runResult
		if err := b.evict(runSpec{Config: c}, &res); err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		got, err := os.ReadFile(logPath)
		if err != nil {
			t.Fatal(err)
		}
		if want := model + "\n" + side + "\n"; string(got) != want {
			t.Errorf("%s evicted %q, want %q", c.Name, got, want)
		}
		if !strings.HasPrefix(res.Evict, model+":") || len(res.EvictSidefiles) != 1 || !strings.HasPrefix(res.EvictSidefiles[0], side+":") {
			t.Errorf("%s recorded %q and %q", c.Name, res.Evict, res.EvictSidefiles)
		}
	}

	// 1 MiB of 16 (6%) resident in one file, the other evicted: both files, both arms
	residentIn := func(path string) func(string) (float64, error) {
		return func(p string) (float64, error) {
			if p == path {
				return 1, nil
			}
			return 0, nil
		}
	}
	var res runResult
	for _, tc := range []struct {
		arm  config
		path string
		want string
	}{
		{arm, side, "side-file " + side},
		{arm, model, "model " + model},
		{gguf, side, "side-file " + side},
		{gguf, model, "model " + model},
	} {
		b, _ := fakeEvict(t, m, residentIn(tc.path))
		if err := b.evict(runSpec{Config: tc.arm}, &res); err == nil || !strings.Contains(err.Error(), "stay in the page cache") || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s with %s resident: %v", tc.arm.Name, tc.path, err)
		}
	}
	// a matrix with no side-file is held to the same limit
	single := matrix{Configs: []config{gguf}}
	b, _ := fakeEvict(t, single, residentIn(model))
	if err := b.evict(runSpec{Config: gguf}, &res); err == nil || !strings.Contains(err.Error(), "model "+model) {
		t.Errorf("a resident GGUF, no side-file: %v", err)
	}
	b, _ = fakeEvict(t, single, func(string) (float64, error) { return 0, errors.New("mincore") })
	if err := b.evict(runSpec{Config: gguf}, &res); err == nil || !strings.Contains(err.Error(), "mincore") {
		t.Errorf("a failed measurement: %v", err)
	}

	gone := filepath.Join(dir, "fail.experts")
	m = matrix{Configs: []config{{Name: "b", Model: model, Args: []string{"--expert-cache", "128", sidefileFlag, gone}}}}
	b, _ = fakeEvict(t, m, func(string) (float64, error) { return 0, nil })
	if err := b.evict(runSpec{Config: m.Configs[0]}, &res); err == nil || !strings.Contains(err.Error(), "no such file") {
		t.Errorf("a missing side-file: %v", err)
	}

	// without side-files only the model, as before
	m = matrix{Configs: []config{gguf}}
	b, logPath := fakeEvict(t, m, func(string) (float64, error) { return 0, nil })
	res = runResult{}
	if err := b.evict(runSpec{Config: gguf}, &res); err != nil || len(res.EvictSidefiles) != 0 {
		t.Errorf("no side-file: %v, %q", err, res.EvictSidefiles)
	}
	if got, _ := os.ReadFile(logPath); string(got) != model+"\n" {
		t.Errorf("evicted %q", got)
	}
}

// statsOf is a side-file run's stats as the fork writes them.
func statsOf(t *testing.T, sidefile string, preads, loads int) *cacheStats {
	t.Helper()
	name := "null"
	if sidefile != "" {
		name = strconv.Quote(sidefile)
	}
	var s cacheStats
	data := fmt.Sprintf(`{"expert_cache": {"buffered": false, "sidefile": %s}, "prefill": {"load_failures": 0, "loads": %d, "preads": %d}, "decode": {"load_failures": 0, "loads": %d, "preads": %d}, "process": {}}`,
		name, loads, preads, loads, preads)
	if err := json.Unmarshal([]byte(data), &s); err != nil {
		t.Fatal(err)
	}
	return &s
}

// TestReadsCheck: the report proves from the stats that an arm read what it
// claims, and fails otherwise.
func TestReadsCheck(t *testing.T) {
	m := matrix{Configs: []config{
		{Name: "a", Args: []string{"--expert-cache", "128"}, Ceilings: []float64{0}, Reps: 1},
		{Name: "b", Args: []string{"--expert-cache", "128", sidefileFlag, "/m.experts"}, Ceilings: []float64{0}, Reps: 1},
	}}
	run := func(cfg string, s *cacheStats) runResult {
		return runResult{Config: cfg, CeilingGiB: 0, Rep: 1, Status: statusOK, Stats: s}
	}
	for _, tc := range []struct {
		name string
		runs []runResult
		ok   bool
		want string
	}{
		{"both right", []runResult{run("a", statsOf(t, "", 30, 10)), run("b", statsOf(t, "/m.experts", 10, 10))}, true, "PASS reads b: 1 runs from /m.experts, 20 preads for 20 experts, want 1 each"},
		{"the arm fell back to the GGUF", []runResult{run("b", statsOf(t, "", 30, 10))}, false, "b/0g/r1 read from none"},
		{"another side-file", []runResult{run("b", statsOf(t, "/other.experts", 10, 10))}, false, "read from /other.experts"},
		{"three reads per expert from the side-file", []runResult{run("b", statsOf(t, "/m.experts", 30, 10))}, false, "took 30 preads for 10 experts"},
		{"the GGUF arm on a side-file", []runResult{run("a", statsOf(t, "/m.experts", 30, 10))}, false, "a/0g/r1 read from /m.experts"},
		{"one read per expert on the GGUF path", []runResult{run("a", statsOf(t, "", 10, 10))}, false, "took 10 preads for 10 experts"},
		{"no stats", []runResult{run("b", nil)}, false, "wrote no stats"},
		// E4's stats: decode hits, but neither loads nor preads
		{"a GGUF baseline from a build without patch 0014", []runResult{run("a", statsOf(t, "", 0, 0))}, false, "a/0g/r1 loaded 0 experts in prefill and 0 in decode"},
		{"no load in decode", []runResult{run("a", func() *cacheStats { s := statsOf(t, "", 30, 10); s.Decode.Loads, s.Decode.Preads = 0, 0; return s }())}, false, "loaded 10 experts in prefill and 0 in decode"},
		{"no load in prefill", []runResult{run("a", func() *cacheStats { s := statsOf(t, "", 30, 10); s.Prefill.Loads, s.Prefill.Preads = 0, 0; return s }())}, false, "loaded 0 experts in prefill and 10 in decode"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var w strings.Builder
			if got := readsCheck(&w, m, tc.runs); got != tc.ok || !strings.Contains(w.String(), tc.want) {
				t.Errorf("readsCheck %v: %s; want %v, %q", got, w.String(), tc.ok, tc.want)
			}
		})
	}
	// a matrix without a side-file is not asked
	m.Configs = m.Configs[:1]
	var w strings.Builder
	if !readsCheck(&w, m, []runResult{run("a", nil)}) || w.Len() != 0 {
		t.Errorf("no side-file: %q", w.String())
	}
}

// TestBuildCheck: one build's runs carry one digest, and a comparison of two
// builds fails the report.
func TestBuildCheck(t *testing.T) {
	side := matrix{Configs: []config{
		{Name: "a", Args: []string{"--expert-cache", "128"}, Ceilings: []float64{0}, Reps: 2},
		{Name: "b", Args: []string{"--expert-cache", "128", sidefileFlag, "/m.experts"}, Ceilings: []float64{0}, Reps: 2},
	}}
	named := matrix{Configs: []config{
		{Name: "fork", Args: []string{"--expert-cache", "128"}, Ceilings: []float64{0}, Reps: 2},
		{Name: "pr", Server: "pr25294", Args: []string{"--moe-stream-cache", "128s"}, Ceilings: []float64{0}, Reps: 2},
	}}
	run := func(cfg string, rep int, digest string) runResult {
		return runResult{Config: cfg, Rep: rep, Status: statusOK, Server: "/srv/" + digest, ServerSHA256: digest}
	}
	for _, tc := range []struct {
		name string
		m    matrix
		runs []runResult
		ok   bool
		want string
	}{
		{"one build", side, []runResult{run("a", 1, "x"), run("b", 1, "x")}, true, "PASS build the fork's -server: 2 runs on x (/srv/x)"},
		{"a baseline from another build", side, []runResult{run("a", 1, "x"), run("b", 1, "y")}, false, "FAIL build the fork's -server: 1 runs on x (/srv/x), 1 runs on y (/srv/y)"},
		{"a rebuild between repetitions of an arm", side, []runResult{run("b", 1, "x"), run("b", 2, "y")}, false, "FAIL build"},
		{"a side-file matrix wants a digest", side, []runResult{run("a", 1, ""), run("b", 1, "")}, false, "FAIL build the fork's -server: 2 runs with no recorded build"},
		{"a resume with the old driver", named, []runResult{run("fork", 1, ""), run("fork", 2, "x")}, false, "1 runs with no recorded build, 1 runs on x"},
		{"named builds differ by design", named, []runResult{run("fork", 1, "x"), run("pr", 1, "y")}, true, "PASS build pr25294: 1 runs on y"},
		{"a named build's own runs differ", named, []runResult{run("pr", 1, "x"), run("pr", 2, "y")}, false, "FAIL build pr25294"},
		{"older results are not asked", named, []runResult{run("fork", 1, ""), run("pr", 1, "")}, true, ""},
		{"a failed run is not compared", side, []runResult{run("a", 1, "x"), {Config: "b", Rep: 1, Status: statusLoadFailed, ServerSHA256: "y"}}, true, "PASS build the fork's -server: 1 runs on x"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var w strings.Builder
			if got := buildCheck(&w, tc.m, tc.runs); got != tc.ok || !strings.Contains(w.String(), tc.want) {
				t.Errorf("buildCheck %v: %q; want %v, %q", got, w.String(), tc.ok, tc.want)
			}
			if tc.want == "" && w.Len() != 0 {
				t.Errorf("wrote %q", w.String())
			}
		})
	}
	// it decides the report
	var w strings.Builder
	if report(&w, side, []runResult{run("a", 1, "x"), run("a", 2, "y")}) || !strings.Contains(w.String(), "FAIL build") {
		t.Errorf("the report passed two builds: %s", w.String())
	}
}

// TestServerBuild: the digest changes with the binary and with a library beside
// it, where the code is, and not with the path or a library's other names.
func TestServerBuild(t *testing.T) {
	dir := t.TempDir()
	write := func(name, text string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), []byte(text), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	write("llama-server", "server")
	write("libllama.0.7.0.dylib", "library")
	write("llama-completion", "another tool")
	if err := os.Symlink("libllama.0.7.0.dylib", filepath.Join(dir, "libllama.dylib")); err != nil {
		t.Fatal(err)
	}
	digest := func(path string) string {
		t.Helper()
		d, err := serverBuild(path)
		if err != nil {
			t.Fatal(err)
		}
		return d
	}
	server := filepath.Join(dir, "llama-server")
	first := digest(server)
	if len(first) != 64 {
		t.Errorf("digest %q", first)
	}
	link := filepath.Join(t.TempDir(), "srv")
	if err := os.Symlink(server, link); err != nil {
		t.Fatal(err)
	}
	if got := digest(link); got != first {
		t.Errorf("through a symlink %s, want %s", got, first)
	}
	if err := os.Symlink("libllama.0.7.0.dylib", filepath.Join(dir, "libllama.0.dylib")); err != nil {
		t.Fatal(err)
	}
	if got := digest(server); got != first {
		t.Errorf("another name of a library changed the digest")
	}
	write("llama-completion", "another tool, rebuilt")
	if got := digest(server); got != first {
		t.Errorf("another tool changed the digest")
	}
	write("libllama.0.7.0.dylib", "library, rebuilt")
	second := digest(server)
	if second == first {
		t.Errorf("a rebuilt library did not change the digest")
	}
	write("llama-server", "server, rebuilt")
	if digest(server) == second {
		t.Errorf("a rebuilt binary did not change the digest")
	}
	if _, err := serverBuild(filepath.Join(dir, "gone")); err == nil {
		t.Errorf("a missing server has a digest")
	}
}

// TestCheckServer: a matrix with a side-file arm stops before its first run on a
// server without the flag; another matrix does not ask.
func TestCheckServer(t *testing.T) {
	dir := t.TempDir()
	fake := func(name, script string) string {
		t.Helper()
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte("#!/bin/sh\n"+script+"\n"), 0o755); err != nil {
			t.Fatal(err)
		}
		return path
	}
	side := matrix{Configs: []config{{Name: "b", Args: []string{"--expert-cache", "128", sidefileFlag, "/m.experts"}}}}
	if err := side.checkServer(fake("with", `[ "$1" = --help ] && echo "--expert-cache-sidefile FNAME   read misses"`)); err != nil {
		t.Errorf("a server with the flag: %v", err)
	}
	if err := side.checkServer(fake("without", `[ "$1" = --help ] && echo "--expert-cache N"`)); err == nil || !strings.Contains(err.Error(), "does not list "+sidefileFlag) {
		t.Errorf("a server without the flag: %v", err)
	}
	if err := side.checkServer(fake("broken", "echo dyld: no library >&2; exit 134")); err == nil || !strings.Contains(err.Error(), "no library") {
		t.Errorf("a server that does not start: %v", err)
	}
	if err := side.checkServer(filepath.Join(dir, "gone")); err == nil {
		t.Errorf("a missing server passed")
	}
	plain := matrix{Configs: []config{{Name: "a", Args: []string{"--expert-cache", "128"}}}}
	if err := plain.checkServer(filepath.Join(dir, "gone")); err != nil {
		t.Errorf("no side-file: %v", err)
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
