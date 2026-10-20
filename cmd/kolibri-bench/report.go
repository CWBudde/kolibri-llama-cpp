package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

// streamSummary is the part of cmd/kolibri-stream's -summary JSON the report uses.
type streamSummary struct {
	Generated             int     `json:"generated"`
	TTFTMs                float64 `json:"ttft_ms"`
	DecodeTokensPerSecond float64 `json:"decode_tokens_per_second"`
	LatencyMs             struct {
		P50 float64 `json:"p50"`
		P95 float64 `json:"p95"`
		P99 float64 `json:"p99"`
		Max float64 `json:"max"`
	} `json:"latency_ms"`
	Timings struct {
		PromptN         int     `json:"prompt_n"`
		PromptPerSecond float64 `json:"prompt_per_second"`
		PredictedPerSec float64 `json:"predicted_per_second"`
	} `json:"timings"`
	Pageins      int64  `json:"pageins"`
	TokenIDs     []int  `json:"token_ids"`
	TokensSHA256 string `json:"tokens_sha256"`
}

// cacheStats is the part of --expert-cache-stats the report uses.
type cacheStats struct {
	Decode struct {
		Needed      int64   `json:"needed"`
		Hits        int64   `json:"hits"`
		DemandBytes int64   `json:"demand_bytes"`
		WaitSeconds float64 `json:"wait_seconds"`
		Failures    int64   `json:"load_failures"`
	} `json:"decode"`
	Prefill struct {
		Failures int64 `json:"load_failures"`
	} `json:"prefill"`
	Process struct {
		DiskBytesRead    *int64 `json:"disk_bytes_read"`
		MaxPhysFootprint *int64 `json:"max_phys_footprint"`
	} `json:"process"`
}

func loadResults(dir string) ([]runResult, error) {
	paths, err := filepath.Glob(filepath.Join(dir, "*", "*g", "r*", "run.json"))
	if err != nil {
		return nil, err
	}
	var out []runResult
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}
		var r runResult
		if err := json.Unmarshal(b, &r); err != nil {
			return nil, fmt.Errorf("%s: %w", p, err)
		}
		out = append(out, r)
	}
	return out, nil
}

// peaks returns the largest rss_mib and footprint_mib of a kolibri-stream
// samples.txt.
func peaks(r io.Reader) (rss, footprint float64) {
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		f := strings.Fields(sc.Text())
		if len(f) < 3 {
			continue
		}
		if v, err := strconv.ParseFloat(f[1], 64); err == nil {
			rss = max(rss, v)
		}
		if v, err := strconv.ParseFloat(f[2], 64); err == nil {
			footprint = max(footprint, v)
		}
	}
	return rss, footprint
}

func median(v []float64) float64 {
	if len(v) == 0 {
		return math.NaN()
	}
	s := slices.Clone(v)
	slices.Sort(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

// firstDiff is the first position where a and b differ (a length counts), or -1.
func firstDiff(a, b []int) int {
	for i := range min(len(a), len(b)) {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return min(len(a), len(b))
	}
	return -1
}

// warm returns a per-request value of a run's warm requests.
func (r runResult) warm(f func(*streamSummary) float64) []float64 {
	var out []float64
	for _, q := range r.Requests {
		if q.Phase == "warm" && q.Summary != nil {
			out = append(out, f(q.Summary))
		}
	}
	return out
}

func (r runResult) cold() *streamSummary {
	for _, q := range r.Requests {
		if q.Phase == "cold" {
			return q.Summary
		}
	}
	return nil
}

func decodeRate(s *streamSummary) float64 { return s.DecodeTokensPerSecond }
func p95(s *streamSummary) float64        { return s.LatencyMs.P95 }

// perRep is a metric per repetition of a config at a ceiling: the median over
// its warm requests, for the runs that completed.
func perRep(runs []runResult, cfg string, ceiling float64, f func(*streamSummary) float64) []float64 {
	var out []float64
	for _, r := range runs {
		if r.Config == cfg && r.CeilingGiB == ceiling && r.Status == statusOK {
			if v := r.warm(f); len(v) > 0 {
				out = append(out, median(v))
			}
		}
	}
	return out
}

type verdict struct {
	Gate       gate
	Incomplete string // the configs with some but not all repetitions completed; no verdict then
	Baseline   string // the best baseline that runs at this ceiling; "" if none does
	Candidate  []float64
	Best       []float64 // the baseline's decode tokens/s per repetition
	CandP95    []float64
	BestP95    []float64
	DecodeWin  bool
	TailWin    bool
	Pass       bool
}

// expectedRequests is the number of requests the arms make over all their
// ceilings and repetitions: the cold one and the warm ones per run.
func (m matrix) expectedRequests(arms []string) int {
	n := 0
	for _, name := range arms {
		if c, ok := m.config(name); ok {
			n += c.Reps * len(c.Ceilings) * (1 + len(m.Warm))
		}
	}
	return n
}

// completed counts a config's completed repetitions at a ceiling against the
// configured ones.
func completed(m matrix, runs []runResult, cfg string, ceiling float64) (ok, want int) {
	if c, found := m.config(cfg); found && slices.Contains(c.Ceilings, ceiling) {
		want = c.Reps
	}
	for _, r := range runs {
		if r.Config == cfg && r.CeilingGiB == ceiling && r.Status == statusOK {
			ok++
		}
	}
	return ok, want
}

// evalGate applies the performance gate fixed in the plan: the candidate wins
// against the baseline with the highest median warm decode tokens/s that runs
// at the same ceiling if every repetition of the candidate is faster than
// that baseline's fastest one, or if every one of its warm p95 latencies is
// below the baseline's lowest one. A config runs at a ceiling if all its
// repetitions completed and does not if all were recorded and none
// completed; anything in between, missing runs included, withholds the
// verdict.
func evalGate(m matrix, g gate, runs []runResult) verdict {
	v := verdict{Gate: g}
	var partial []string
	for _, name := range append([]string{g.Candidate}, g.Baselines...) {
		ok, want := completed(m, runs, name, g.CeilingGiB)
		recorded := 0
		for _, r := range runs {
			if r.Config == name && r.CeilingGiB == g.CeilingGiB {
				recorded++
			}
		}
		if ok != want && !(ok == 0 && recorded == want) {
			partial = append(partial, fmt.Sprintf("%s %d of %d", name, ok, want))
		}
	}
	if len(partial) > 0 {
		v.Incomplete = strings.Join(partial, ", ")
		return v
	}
	v.Candidate = perRep(runs, g.Candidate, g.CeilingGiB, decodeRate)
	v.CandP95 = perRep(runs, g.Candidate, g.CeilingGiB, p95)
	best := math.Inf(-1)
	for _, b := range g.Baselines {
		reps := perRep(runs, b, g.CeilingGiB, decodeRate)
		if len(reps) > 0 && median(reps) > best {
			best, v.Baseline, v.Best = median(reps), b, reps
			v.BestP95 = perRep(runs, b, g.CeilingGiB, p95)
		}
	}
	if v.Baseline == "" || len(v.Candidate) == 0 {
		v.Baseline = ""
		return v
	}
	v.DecodeWin = slices.Min(v.Candidate) > slices.Max(v.Best)
	v.TailWin = len(v.CandP95) > 0 && len(v.BestP95) > 0 && slices.Max(v.CandP95) < slices.Min(v.BestP95)
	v.Pass = v.DecodeWin || v.TailWin
	return v
}

type comparison struct {
	Run      string
	Phase    string
	Workload string
	Diff     int // the first differing token position, -1 if equal
}

// checkEquivalence compares the token IDs of every request of the arms' completed
// runs with the reference's for the same workload.
func checkEquivalence(e equivalence, runs []runResult) []comparison {
	ref := map[string][]int{}
	for _, r := range runs {
		if r.Config != e.Reference || r.Status != statusOK {
			continue
		}
		for _, q := range r.Requests {
			if _, ok := ref[q.Workload]; !ok && q.Summary != nil {
				ref[q.Workload] = q.Summary.TokenIDs
			}
		}
	}
	var out []comparison
	for _, r := range runs {
		if !slices.Contains(e.Arms, r.Config) || r.Status != statusOK {
			continue // a run that did not complete counts as missing
		}
		for _, q := range r.Requests {
			want, ok := ref[q.Workload]
			if !ok || q.Summary == nil {
				continue
			}
			out = append(out, comparison{r.key(), q.Phase, q.Workload, firstDiff(q.Summary.TokenIDs, want)})
		}
	}
	return out
}

func fmtRange(v []float64, prec int) string {
	if len(v) == 0 {
		return "–"
	}
	f := "%." + strconv.Itoa(prec) + "f"
	if len(v) == 1 {
		return fmt.Sprintf(f, v[0])
	}
	return fmt.Sprintf(f+" ("+f+"–"+f+")", median(v), slices.Min(v), slices.Max(v))
}

// report writes the tables and verdicts as Markdown and returns false if an
// exact equivalence failed, or a completed cache run failed a load or wrote no
// stats.
func report(w io.Writer, m matrix, runs []runResult) bool {
	fmt.Fprintln(w, "| Config | Ceiling | Runs | Load s | Cold TTFT s | Cold decode t/s | Warm prompt t/s | Warm decode t/s | Warm p50 ms | Warm p95 ms | Page-ins MiB/token | Peak RSS GiB | Peak footprint GiB | Decode hits | Disk GB read |")
	fmt.Fprintln(w, "|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
	for _, c := range m.Configs {
		for _, ceil := range c.Ceilings {
			var ok, failed []runResult
			for _, r := range runs {
				if r.Config == c.Name && r.CeilingGiB == ceil {
					if r.Status == statusOK {
						ok = append(ok, r)
					} else {
						failed = append(failed, r)
					}
				}
			}
			status := fmt.Sprintf("%d ok", len(ok))
			for _, f := range failed {
				status += fmt.Sprintf(", r%d %s", f.Rep, f.Status)
			}
			var load, ttft, coldDec, prompt, pageins, rss, foot, hits, disk []float64
			for _, r := range ok {
				load = append(load, r.LoadSeconds)
				if s := r.cold(); s != nil {
					ttft = append(ttft, s.TTFTMs/1000)
					coldDec = append(coldDec, s.DecodeTokensPerSecond)
				}
				prompt = append(prompt, median(r.warm(func(s *streamSummary) float64 { return s.Timings.PromptPerSecond })))
				pageins = append(pageins, median(r.warm(func(s *streamSummary) float64 {
					return float64(s.Pageins) * 16 / 1024 / float64(max(s.Generated-1, 1))
				})))
				rss = append(rss, r.PeakRSSMiB/1024)
				foot = append(foot, r.PeakFootprintMiB/1024)
				if r.Stats != nil {
					if r.Stats.Decode.Needed > 0 {
						hits = append(hits, 100*float64(r.Stats.Decode.Hits)/float64(r.Stats.Decode.Needed))
					}
					if d := r.Stats.Process.DiskBytesRead; d != nil {
						disk = append(disk, float64(*d)/1e9)
					}
				}
			}
			dec := perRep(runs, c.Name, ceil, decodeRate)
			p50 := perRep(runs, c.Name, ceil, func(s *streamSummary) float64 { return s.LatencyMs.P50 })
			tail := perRep(runs, c.Name, ceil, p95)
			hitStr := "–"
			if len(hits) > 0 {
				hitStr = fmtRange(hits, 1) + "%"
			}
			fmt.Fprintf(w, "| %s | %g GiB | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n",
				c.Name, ceil, status, fmtRange(load, 1), fmtRange(ttft, 2), fmtRange(coldDec, 2), fmtRange(prompt, 1),
				fmtRange(dec, 2), fmtRange(p50, 1), fmtRange(tail, 1), fmtRange(pageins, 3), fmtRange(rss, 1), fmtRange(foot, 1), hitStr, fmtRange(disk, 2))
		}
	}

	allEqual := true
	fmt.Fprintln(w)
	for _, e := range m.Equivalence {
		res := checkEquivalence(e, runs)
		equal := 0
		var diffs []string
		for _, c := range res {
			if c.Diff < 0 {
				equal++
			} else {
				diffs = append(diffs, fmt.Sprintf("%s %s %s at token %d", c.Run, c.Phase, c.Workload, c.Diff))
			}
		}
		want := m.expectedRequests(e.Arms)
		kind := "INFO"
		if e.Exact {
			kind = "PASS"
			if len(diffs) > 0 || equal != want {
				kind, allEqual = "FAIL", false
			}
		}
		fmt.Fprintf(w, "- %s equivalence %s against %s: %d of %d requests generate the same token IDs", kind, strings.Join(e.Arms, ", "), e.Reference, equal, want)
		if len(diffs) > 0 {
			fmt.Fprintf(w, "; first differences: %s", strings.Join(diffs, "; "))
		}
		fmt.Fprintln(w)
	}

	cacheRuns, withStats, failures := 0, 0, int64(0)
	for _, c := range m.Configs {
		if usesCache(c.Args) {
			cacheRuns += c.Reps * len(c.Ceilings)
		}
	}
	for _, r := range runs {
		if c, ok := m.config(r.Config); ok && usesCache(c.Args) && r.Status == statusOK {
			if r.Stats != nil {
				withStats++
				failures += r.Stats.Decode.Failures + r.Stats.Prefill.Failures
			}
		}
	}
	kind := "PASS"
	if failures > 0 || withStats < cacheRuns {
		kind, allEqual = "FAIL", false
	}
	fmt.Fprintf(w, "- %s expert load failures: %d in the %d of %d cache runs that completed and wrote their stats\n", kind, failures, withStats, cacheRuns)

	fmt.Fprintln(w)
	for _, g := range m.Gates {
		v := evalGate(m, g, runs)
		kind := "INFO"
		if g.Decides {
			kind = "GATE"
		}
		result := "no verdict (no baseline runs at this ceiling, or the candidate does not)"
		if v.Incomplete != "" {
			result = "no verdict, repetitions incomplete: " + v.Incomplete
		} else if v.Baseline != "" {
			result = "not passed"
			if v.Pass {
				result = "passed"
			}
			result = fmt.Sprintf("%s: decode %s against %s's %s t/s (win %v), p95 %s against %s ms (win %v)", result,
				fmtRange(v.Candidate, 2), v.Baseline, fmtRange(v.Best, 2), v.DecodeWin, fmtRange(v.CandP95, 1), fmtRange(v.BestP95, 1), v.TailWin)
		}
		fmt.Fprintf(w, "- %s %s at %g GiB against the best of %s: %s\n", kind, g.Candidate, g.CeilingGiB, strings.Join(g.Baselines, ", "), result)
	}
	return allEqual
}
