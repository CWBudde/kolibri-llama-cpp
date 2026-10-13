// Command kolibri-policy replays the locality workloads' expert selections through expert cache
// policies in the order an online server meets them (internal/locality, replay.go): each prompt's
// new tokens prefilled in ubatches, each answer decoded token by token, with a policy that learns
// a step's experts only when the step runs. It reports prefill and decode apart and is the
// evaluator of PLAN Phase 10, Stage A; it writes docs/expert-policy.md.
//
//	kolibri-policy -models Kolibri-1-IQ3_XXS-IQ4_XS-down-imx,Kolibri-1-Q8_0 -md docs/expert-policy.md
//
// The baselines (policies.go), each at every size C, a budget of C experts of every layer:
//
//   - no cache, once;
//   - layer LRU: an LRU cache of C experts in every layer, cmd/kolibri-cache's per-layer policy;
//   - shared LRU: one LRU cache of the whole budget for all layers, cmd/kolibri-cache's shared one;
//   - LFU: shared, the least used experts evicted, counts halved every -lfu-halflife tokens;
//   - ARC: shared, the Adaptive Replacement Cache;
//   - oracle, per layer and shared: Belady's MIN, which knows the future. It is no baseline but
//     the unattainable bound, marked † wherever it is reported.
//
// Every replay also estimates its time when all loads go over one declared link (-bw, transfer.go),
// with a step's compute time taken from the resident model's measured speed (-decode-tps,
// -prefill-tps). These are assumptions, not measurements.
//
// Every workload is split chronologically after one of its answers (-splits): the training prefix
// is everything up to that answer's decoded tokens, the test suffix the rest. Each replay runs
// straight through and is counted for both apart (locality.ReplaySegments), so the test suffix
// starts warm from the prefix. LFU's half-life is replayed at every value of a grid and chosen
// per size on training prefixes only: the workload's own, the other workloads' (leave one
// workload out) and the other model's; the choice in hindsight on the test suffix is reported as
// the leaky bound of tuning.
//
// For each model and workload it reads <dir>/<workload>.<model>.experts.npy and the token IDs
// <dir>/<workload>.tokens.npy, and the answers' prompt lengths and generated tokens from
// testdata/locality/<workload>.json. A conversation that ends with an answer ends the online
// replay after that answer's decoded tokens: no request sends the rest. These checks are fatal:
//
//   - the token IDs hold every answer's generated tokens after its prompt;
//   - the declared split trains on the first ⌊n/2⌋ of n answers and its test suffix starts at the
//     declared token;
//   - the layer and the shared LRU, replayed token by token in trace order, hit exactly as often as
//     in <dir>/cache-<model>.json of cmd/kolibri-cache at every size;
//   - at every size, the shared oracle loads no more expert bytes than any policy and the per-layer
//     oracle none more than the layer LRU.
//
// It writes <dir>/policy-<model>.json and, given -md, a Markdown report.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"

	"kolibri-llm/internal/locality"
)

// Model are a GGUF's expert sizes as tools/locality/expert_bytes.py records them.
type Model struct {
	NExpert int `json:"n_expert"`
	Layers  []struct {
		BytesPerExpert int64 `json:"bytes_per_expert"`
	} `json:"layers"`
}

func (m Model) bytes() []int64 {
	out := make([]int64, len(m.Layers))
	for l, x := range m.Layers {
		out[l] = x.BytesPerExpert
	}
	return out
}

// budget is the bytes of c experts of every layer.
func (m Model) budget(c int) int64 {
	var all int64
	for _, b := range m.bytes() {
		all += b
	}
	return int64(c) * all
}

// CacheSizes are the fields of a workload in cmd/kolibri-cache's output the check needs.
type CacheSizes struct {
	PerLayer []CacheSize `json:"per_layer"`
	Shared   []CacheSize `json:"shared"`
}

// CacheSize is one size of a cmd/kolibri-cache policy.
type CacheSize struct {
	ExpertsPerLayer int `json:"experts_per_layer"`
	Hits            int `json:"hits"`
}

// Replayed is an online replay, whole and cut into its training prefix and test suffix.
type Replayed struct {
	Online locality.Result `json:"online"`
	Split  Split           `json:"split"`
}

// Run is a policy's online replay at one size.
type Run struct {
	ExpertsPerLayer int   `json:"experts_per_layer"`
	BudgetBytes     int64 `json:"budget_bytes"`
	Replayed
}

// Runs are a policy's replays at every size. An oracle knows the future: an unattainable bound.
type Runs struct {
	Policy string `json:"policy"`
	Oracle bool   `json:"oracle,omitempty"`
	Runs   []Run  `json:"runs"`
}

// TraceOrder are the LRU caches' hits replayed token by token, as cmd/kolibri-cache does.
type TraceOrder struct {
	Accesses  int   `json:"accesses"`
	LayerLRU  []int `json:"layer_lru_hits"`
	SharedLRU []int `json:"shared_lru_hits"`
}

// Workload is one workload's replays.
type Workload struct {
	Tokens     int           `json:"tokens"`
	Answers    int           `json:"answers"`
	Ubatches   int           `json:"ubatches"`
	Split      SplitWorkload `json:"split"`
	NoCache    Replayed      `json:"no_cache"`
	Policies   []Runs        `json:"policies"`
	TraceOrder TraceOrder    `json:"trace_order"`
	LFUGrid    Grid          `json:"lfu_grid"`
	LFUTuning  []Tuning      `json:"lfu_tuning"`
}

// Assumptions are the declared, unmeasured inputs of the replays.
type Assumptions struct {
	Ubatch                 int                `json:"ubatch"`
	LinkGBPerSecond        map[string]float64 `json:"link_gb_per_second"`
	DecodeTokensPerSecond  float64            `json:"decode_tokens_per_second"`
	PrefillTokensPerSecond float64            `json:"prefill_tokens_per_second"`
	LFUHalfLife            int                `json:"lfu_half_life_tokens"`
	LFUHalfLives           []int              `json:"lfu_tuning_half_lives"`
	SplitRule              string             `json:"split_rule"`
}

// Output is <dir>/policy-<model>.json.
type Output struct {
	Assumptions Assumptions         `json:"assumptions"`
	Workloads   map[string]Workload `json:"workloads"`
}

// baseline is a policy to replay at every size c.
type baseline struct {
	name   string
	oracle bool
	build  func(c int) (locality.Policy, error)
}

const (
	layerLRU = iota
	sharedLRU
	lfu
	arc
	layerOracle
	sharedOracle
)

func baselines(sel locality.Selections, spans []locality.Span, m Model, halfLife int) []baseline {
	b := m.bytes()
	return []baseline{
		layerLRU: {"layer LRU", false, func(c int) (locality.Policy, error) {
			return locality.NewLayerLRU(sel.Layers, m.NExpert, c), nil
		}},
		sharedLRU: {"shared LRU", false, func(c int) (locality.Policy, error) {
			return locality.NewSharedLRU(m.NExpert, b, m.budget(c)), nil
		}},
		lfu: {"LFU", false, func(c int) (locality.Policy, error) {
			return locality.NewLFU(m.NExpert, b, m.budget(c), halfLife), nil
		}},
		arc: {"ARC", false, func(c int) (locality.Policy, error) { return locality.NewARC(m.NExpert, b, m.budget(c)) }},
		layerOracle: {"oracle, per layer", true, func(c int) (locality.Policy, error) {
			return locality.NewLayerOracle(sel, spans, m.NExpert, b, c), nil
		}},
		sharedOracle: {"oracle, shared", true, func(c int) (locality.Policy, error) {
			return locality.NewOracle(sel, spans, m.NExpert, b, m.budget(c)), nil
		}},
	}
}

// replay runs no cache once and every baseline at every size online, cut before span cut into the
// training prefix and the test suffix.
func replay(sel locality.Selections, spans []locality.Span, cut int, m Model, sizes []int, bases []baseline,
	timing locality.Timing) (Replayed, []Runs, error) {
	noCache, err := replayed(sel, spans, cut, m, &locality.NoCache{}, timing)
	if err != nil {
		return noCache, nil, err
	}
	runs, err := replayAll(sel, spans, cut, m, sizes, bases, timing)
	return noCache, runs, err
}

// replayed replays p online, whole and split before span cut.
func replayed(sel locality.Selections, spans []locality.Span, cut int, m Model, p locality.Policy,
	timing locality.Timing) (Replayed, error) {
	segs, err := locality.ReplaySegments(sel, spans, []int{cut}, m.NExpert, m.bytes(), p, timing)
	if err != nil {
		return Replayed{}, err
	}
	return Replayed{locality.Merge(segs...), Split{segs[0], segs[1]}}, nil
}

// replayAll replays every baseline at every size, in parallel.
func replayAll(sel locality.Selections, spans []locality.Span, cut int, m Model, sizes []int, bases []baseline,
	timing locality.Timing) ([]Runs, error) {
	out := make([]Runs, len(bases))
	errs := make([]error, len(bases)*len(sizes))
	for i, b := range bases {
		out[i] = Runs{Policy: b.name, Oracle: b.oracle, Runs: make([]Run, len(sizes))}
	}
	jobs := make(chan int)
	var wg sync.WaitGroup
	for range runtime.GOMAXPROCS(0) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				i, s := j/len(sizes), j%len(sizes)
				c := sizes[s]
				p, err := bases[i].build(c)
				if err != nil {
					errs[j] = err
					continue
				}
				res, err := replayed(sel, spans, cut, m, p, timing)
				if o, ok := p.(*locality.Oracle); ok && err == nil {
					err = o.Err()
				}
				errs[j] = err
				out[i].Runs[s] = Run{c, m.budget(c), res}
			}
		}()
	}
	for j := range errs {
		jobs <- j
	}
	close(jobs)
	wg.Wait()
	for j, err := range errs {
		if err != nil {
			return nil, fmt.Errorf("%s at %d experts per layer: %w", bases[j/len(sizes)].name,
				sizes[j%len(sizes)], err)
		}
	}
	return out, nil
}

// traceOrder replays the layer and the shared LRU token by token at every size.
func traceOrder(sel locality.Selections, m Model, sizes []int) (TraceOrder, error) {
	tr := TraceOrder{Accesses: sel.Layers * sel.Tokens * sel.K}
	for _, c := range sizes {
		for _, p := range []locality.Policy{locality.NewLayerLRU(sel.Layers, m.NExpert, c),
			locality.NewSharedLRU(m.NExpert, m.bytes(), m.budget(c))} {
			res, err := locality.Replay(sel, locality.TraceOrder(sel.Tokens), m.NExpert, m.bytes(), p)
			if err != nil {
				return tr, err
			}
			if _, ok := p.(*locality.LayerLRU); ok {
				tr.LayerLRU = append(tr.LayerLRU, res.Decode.Hits)
			} else {
				tr.SharedLRU = append(tr.SharedLRU, res.Decode.Hits)
			}
		}
	}
	return tr, nil
}

// check compares trace-order hits with one of cmd/kolibri-cache's policies at the same sizes.
func check(hits []int, sizes []int, recorded []CacheSize) error {
	byC := map[int]int{}
	for _, s := range recorded {
		byC[s.ExpertsPerLayer] = s.Hits
	}
	for i, c := range sizes {
		h, ok := byC[c]
		if !ok {
			return fmt.Errorf("no size %d (run kolibri-cache with it in -sizes)", c)
		}
		if h != hits[i] {
			return fmt.Errorf("%d experts per layer: %d hits in trace order, kolibri-cache %d", c, hits[i], h)
		}
	}
	return nil
}

// loaded is the expert bytes a replay loads, missed or prefetched.
func loaded(r locality.Result) int64 {
	return r.Prefill.MissBytes + r.Decode.MissBytes + r.Prefill.PrefetchBytes + r.Decode.PrefetchBytes
}

// checkOracles fails unless at every size the shared oracle loads no more than any policy, and the
// per-layer oracle no more than the layer LRU.
func checkOracles(noCache locality.Result, runs []Runs) error {
	for s := range runs[0].Runs {
		o := loaded(runs[sharedOracle].Runs[s].Online)
		if o > loaded(noCache) {
			return fmt.Errorf("the shared oracle loads %d bytes, no cache %d", o, loaded(noCache))
		}
		for _, p := range runs {
			if l := loaded(p.Runs[s].Online); o > l {
				return fmt.Errorf("%d experts per layer: the shared oracle loads %d bytes, %s %d",
					p.Runs[s].ExpertsPerLayer, o, p.Policy, l)
			}
		}
		if lo, l := loaded(runs[layerOracle].Runs[s].Online), loaded(runs[layerLRU].Runs[s].Online); lo > l {
			return fmt.Errorf("%d experts per layer: the per-layer oracle loads %d bytes, the layer LRU %d",
				runs[0].Runs[s].ExpertsPerLayer, lo, l)
		}
	}
	return nil
}

func gb(b int64) string         { return fmt.Sprintf("%.2f", float64(b)/1e9) }
func pct(v float64) string      { return fmt.Sprintf("%.1f%%", 100*v) }
func short(model string) string { return strings.TrimPrefix(model, "Kolibri-1-") }

func rate(c locality.Counts) float64   { return float64(c.Hits) / float64(max(c.Accesses, 1)) }
func recall(c locality.Counts) float64 { return float64(c.FullSteps) / float64(max(c.Steps, 1)) }

// perToken is a phase's miss bytes per token in MB.
func perToken(c locality.Counts) string {
	return fmt.Sprintf("%.0f", float64(c.MissBytes)/1e6/float64(max(c.Tokens, 1)))
}

// msPerToken is a phase's estimated time per token on link i.
func msPerToken(c locality.Counts, i int) string {
	return fmt.Sprintf("%.1f", 1e3*c.Links[i].Seconds/float64(max(c.Tokens, 1)))
}

func label(r Runs) string {
	if r.Oracle {
		return r.Policy + " †"
	}
	return r.Policy
}

type result struct {
	model, workload string
	w               Workload
}

// at is the index of size c, or -1.
func at(sizes []int, c int) int {
	for i, s := range sizes {
		if s == c {
			return i
		}
	}
	return -1
}

func summary(r result, sizes []int, detail int) string {
	i := max(at(sizes, detail), 0)
	parts := []string{"no cache " + perToken(r.w.NoCache.Online.Decode)}
	for _, p := range r.w.Policies {
		parts = append(parts, label(p)+" "+perToken(p.Runs[i].Online.Decode))
	}
	p := r.w.NoCache.Online
	return fmt.Sprintf("%s %s: %d prefill and %d decode tokens; decode MB/token at %d experts per layer: %s", r.model,
		r.workload, p.Prefill.Tokens, p.Decode.Tokens, sizes[i], strings.Join(parts, ", "))
}

func writeMarkdown(w io.Writer, command string, a Assumptions, links []locality.Link, sizes []int, detail int,
	layers []int, results []result) {
	fmt.Fprintf(w, "# Expert cache policies, replayed online\n\n")
	fmt.Fprintf(w, "Generated by `%s` from the selection files of `tools/locality/experts.py`, the answers in "+
		"`testdata/locality/<workload>.json` and the expert sizes in `testdata/locality/expert-bytes.json`; do not "+
		"edit by hand.\n\n", command)
	fmt.Fprintf(w, "The workloads are replayed in the order an online server with a prompt cache meets them: each "+
		"prompt's new tokens prefilled in ubatches of %d, each answer decoded token by token. A decoded token needs "+
		"its six experts per layer; a prefill ubatch the union of its tokens' experts, once. The policy learns a "+
		"step's experts only when the step runs (PLAN Phase 10, Stage A). An answer that stopped at the token "+
		"limit decodes all but its last token, which the next prompt's prefill processes. Every workload ends with "+
		"an answer: no request follows it, so the tokens after its decoded ones are never processed online.\n\n",
		a.Ubatch)
	fmt.Fprintf(w, "Prefill and decode are counted apart: a prefill hit serves a ubatch of up to hundreds of "+
		"teacher-forced tokens at once and is no decode speedup. MB/token are the expert bytes a phase's misses "+
		"load per token of that phase. Set recall is the share of steps that found every expert they need "+
		"resident: a decoded token's exact Top-6 set in a layer, a ubatch's whole union.\n\n")
	fmt.Fprintf(w, "Every policy holds at most the same budget: the bytes of the given number of experts of every "+
		"layer.\n\n")
	fmt.Fprintf(w, "- **layer LRU:** an LRU cache of that many experts in every layer, [expert-cache.md](expert-cache.md)'s "+
		"per-layer policy;\n")
	fmt.Fprintf(w, "- **shared LRU:** one LRU cache of the whole budget for all layers, its shared policy;\n")
	fmt.Fprintf(w, "- **LFU:** shared, the experts used least often evicted (one use per step, counts kept for "+
		"evicted experts too), the least recent of equal counts; all counts are halved every %d tokens. The "+
		"half-life is a declared default, not tuned: a value tuned on these traces would be tuned on the test "+
		"data;\n", a.LFUHalfLife)
	fmt.Fprintf(w, "- **ARC:** shared, the Adaptive Replacement Cache (Megiddo and Modha 2003), which has no "+
		"parameters;\n")
	fmt.Fprintf(w, "- **oracle †:** Belady's MIN, per layer and shared: it knows every future step and drops the "+
		"expert needed furthest ahead, the incoming one included. **It uses future knowledge and is an "+
		"unattainable upper bound, not a policy a server can run.** With equal expert sizes no cache of the "+
		"budget, prefetching or not, loads fewer bytes.\n\n")
	var bw []string
	for _, l := range links {
		bw = append(bw, fmt.Sprintf("%s %.0f GB/s", l.Name, l.BytesPerSecond/1e9))
	}
	fmt.Fprintf(w, "**Estimated time (an assumption, not a measurement).** Each load, missed or prefetched, goes "+
		"over one link at a constant bandwidth, one after another: %s, each estimated on its own. A step "+
		"computes once all its experts have arrived, for the time the fully resident model takes on this Mac "+
		"(`llama-bench` on Metal, [real-checkpoint.md](real-checkpoint.md)): %.1f tokens/s decode, %.0f/s "+
		"prefill, %.1f ms per decoded token. Until its experts arrive, it stalls. None of the baselines "+
		"prefetches, so all their loads stall.\n", strings.Join(bw, ", "), a.DecodeTokensPerSecond,
		a.PrefillTokensPerSecond, 1e3/a.DecodeTokensPerSecond)
	fmt.Fprintf(w, "\n**Held-out splits.** Each workload is split chronologically after one of its answers "+
		"(`testdata/locality/splits.json`): %s A test suffix is replayed "+
		"warm, after its prefix, with whatever state the prefix's causal replay left behind, and only its own steps "+
		"are counted. A policy at test time knows only the steps before it; the answers fix only the order of the "+
		"spans. The oracle † is still built from the whole trace.\n\n", a.SplitRule)
	fmt.Fprintf(w, "| Workload | Answers | Training answers | First test token | Training decode tokens | "+
		"Test decode tokens | Test prefill tokens |\n|---|--:|--:|--:|--:|--:|--:|\n")
	listed := map[string]bool{}
	for _, r := range results {
		if listed[r.workload] {
			continue
		}
		listed[r.workload] = true
		sp := r.w.NoCache.Split
		fmt.Fprintf(w, "| %s | %d | %d | %d | %d | %d | %d |\n", r.workload, r.w.Split.Answers, r.w.Split.TrainAnswers,
			r.w.Split.FirstTestToken, sp.Train.Decode.Tokens, sp.Test.Decode.Tokens, sp.Test.Prefill.Tokens)
	}
	d := max(at(sizes, detail), 0)
	for _, r := range results {
		p := r.w.NoCache.Online
		fmt.Fprintf(w, "\n## %s, %s\n\n%d tokens, %d answers: %d prefill tokens in %d ubatches, %d decoded "+
			"tokens, %d after the last answer not processed. No cache: %s MB per decoded token, %s per "+
			"prefill token.\n\n", r.model, r.workload, r.w.Tokens, r.w.Answers, p.Prefill.Tokens, r.w.Ubatches,
			p.Decode.Tokens, r.w.Tokens-p.Prefill.Tokens-p.Decode.Tokens, perToken(p.Decode), perToken(p.Prefill))
		for _, ph := range []locality.Phase{locality.Decode, locality.Prefill} {
			name := "Decode"
			if ph == locality.Prefill {
				name = "Prefill (ubatches, no decode speedup)"
			}
			fmt.Fprintf(w, "%s, MB/token (hit rate):\n\n| Experts per layer | Budget GB |", name)
			for _, pr := range r.w.Policies {
				fmt.Fprintf(w, " %s |", label(pr))
			}
			fmt.Fprintf(w, "\n|--:|--:|%s\n", strings.Repeat("--:|", len(r.w.Policies)))
			for s, c := range sizes {
				fmt.Fprintf(w, "| %d | %s |", c, gb(r.w.Policies[0].Runs[s].BudgetBytes))
				for _, pr := range r.w.Policies {
					x := phase(pr.Runs[s].Online, ph)
					fmt.Fprintf(w, " %s (%s) |", perToken(x), pct(rate(x)))
				}
				fmt.Fprintln(w)
			}
			fmt.Fprintln(w)
		}
		fmt.Fprintf(w, "At %d experts per layer (%s GB):\n\n", sizes[d], gb(r.w.Policies[0].Runs[d].BudgetBytes))
		rows := []Runs{{Policy: "no cache", Runs: []Run{{Replayed: r.w.NoCache}}}}
		for _, pr := range r.w.Policies {
			rows = append(rows, Runs{pr.Policy, pr.Oracle, []Run{pr.Runs[d]}})
		}
		detailTable(w, rows, links, func(x Replayed) locality.Result { return x.Online })
		fmt.Fprintf(w, "\nHeld-out test suffix at %d experts per layer: answers %d to %d, from token %d on, warm from "+
			"the training prefix:\n\n", sizes[d], r.w.Split.TrainAnswers+1, r.w.Split.Answers, r.w.Split.FirstTestToken)
		detailTable(w, rows, links, func(x Replayed) locality.Result { return x.Split.Test })
		fmt.Fprintf(w, "\nDecode recall of single layers at %d experts per layer (hits per access):\n\n| Layer |",
			sizes[d])
		for _, pr := range r.w.Policies {
			fmt.Fprintf(w, " %s |", label(pr))
		}
		fmt.Fprintf(w, "\n|--:|%s\n", strings.Repeat("--:|", len(r.w.Policies)))
		for _, l := range layers {
			fmt.Fprintf(w, "| %d |", l)
			for _, pr := range r.w.Policies {
				x := pr.Runs[d].Online.Decode.Layers[l]
				fmt.Fprintf(w, " %s |", pct(float64(x.Hits)/float64(max(x.Accesses, 1))))
			}
			fmt.Fprintln(w)
		}
	}
	writeTuning(w, a, sizes, d, results)
}

// detailTable is one row per policy and phase of the result pick selects, at one size.
func detailTable(w io.Writer, rows []Runs, links []locality.Link, pick func(Replayed) locality.Result) {
	fmt.Fprintf(w, "| Policy | Phase | Hit rate | Set recall | MB/token | Evictions | Peak GB |")
	for _, l := range links {
		fmt.Fprintf(w, " %s ms/token |", l.Name)
	}
	fmt.Fprintf(w, "\n|---|---|--:|--:|--:|--:|--:|%s\n", strings.Repeat("--:|", len(links)))
	for _, ph := range []locality.Phase{locality.Decode, locality.Prefill} {
		for _, pr := range rows {
			res := pick(pr.Runs[0].Replayed)
			x := phase(res, ph)
			fmt.Fprintf(w, "| %s | %s | %s | %s | %s | %d | %s |", label(pr), ph, pct(rate(x)), pct(recall(x)),
				perToken(x), x.Evictions, gb(res.PeakBytes))
			for i := range links {
				fmt.Fprintf(w, " %s |", msPerToken(x, i))
			}
			fmt.Fprintln(w)
		}
	}
}

// mbPer is bytes per token in MB.
func mbPer(b int64, tokens int) string {
	return fmt.Sprintf("%.0f", float64(b)/1e6/float64(max(tokens, 1)))
}

// choiceCell is a tuned half-life and the test suffix's decode MB/token with it.
func choiceCell(c *Choice, tokens int) string {
	if c == nil {
		return "–"
	}
	return fmt.Sprintf("%d: %s", c.HalfLife, mbPer(c.TestBytes, tokens))
}

// writeTuning reports LFU's half-life tuned on training prefixes, scored on the test suffixes.
func writeTuning(w io.Writer, a Assumptions, sizes []int, d int, results []result) {
	var hs []string
	for _, h := range a.LFUHalfLives {
		hs = append(hs, strconv.Itoa(h))
	}
	fmt.Fprintf(w, "\n## LFU half-life, tuned on training prefixes\n\n")
	fmt.Fprintf(w, "LFU's half-life is its only parameter. At every budget it is chosen from %s tokens as the one "+
		"whose decode misses load the fewest bytes in a training prefix, the first of equal ones, and scored on the "+
		"test suffix. It is chosen four ways:\n\n", strings.Join(hs, ", "))
	fmt.Fprintf(w, "- **own prefix:** the workload's own training prefix, chronologically before its test suffix;\n")
	fmt.Fprintf(w, "- **leave one workload out:** the other workloads' training prefixes, of the same model;\n")
	fmt.Fprintf(w, "- **other model:** the same workload's training prefix in the other model's trace. The two "+
		"quantizations route the same tokens differently ([expert-locality.md](expert-locality.md) gives the "+
		"share of the same expert sets), and their traces are never pooled;\n")
	fmt.Fprintf(w, "- **hindsight †:** the test suffix itself. **It is tuned on the test data, the bound of tuning "+
		"and not a result.**\n\n")
	fmt.Fprintf(w, "Cells are the half-life and the test suffix's decode MB/token with it; untuned is the declared "+
		"%d.\n\n", a.LFUHalfLife)
	header := "| Untuned | Own prefix | Leave one workload out | Other model | Hindsight † |"
	fmt.Fprintf(w, "At %d experts per layer:\n\n| Model | Workload | %s\n|---|---|--:|--:|--:|--:|--:|\n", sizes[d],
		strings.TrimPrefix(header, "| "))
	for _, r := range results {
		t, n := r.w.LFUTuning[d], r.w.NoCache.Split.Test.Decode.Tokens
		fmt.Fprintf(w, "| %s | %s | %s | %s | %s | %s | %s |\n", r.model, r.workload, choiceCell(&t.Default, n),
			choiceCell(&t.Own, n), choiceCell(t.LOWO, n), choiceCell(t.OtherModel, n), choiceCell(&t.Hindsight, n))
	}
	var models []string
	for _, r := range results {
		if !slices.Contains(models, r.model) {
			models = append(models, r.model)
		}
	}
	for _, m := range models {
		fmt.Fprintf(w, "\n%s, all workloads' test suffixes together, decode MB/token:\n\n| Experts per layer %s\n"+
			"|--:|--:|--:|--:|--:|--:|\n", m, header)
		for s, c := range sizes {
			var tokens int
			var sum [5]int64
			lowo, other := true, true
			for _, r := range results {
				if r.model != m {
					continue
				}
				t := r.w.LFUTuning[s]
				tokens += r.w.NoCache.Split.Test.Decode.Tokens
				sum[0] += t.Default.TestBytes
				sum[1] += t.Own.TestBytes
				if t.LOWO != nil {
					sum[2] += t.LOWO.TestBytes
				} else {
					lowo = false
				}
				if t.OtherModel != nil {
					sum[3] += t.OtherModel.TestBytes
				} else {
					other = false
				}
				sum[4] += t.Hindsight.TestBytes
			}
			cells := make([]string, 5)
			for i, b := range sum {
				cells[i] = mbPer(b, tokens)
			}
			if !lowo {
				cells[2] = "–"
			}
			if !other {
				cells[3] = "–"
			}
			fmt.Fprintf(w, "| %d | %s |\n", c, strings.Join(cells, " | "))
		}
	}
}

func phase(r locality.Result, p locality.Phase) locality.Counts {
	if p == locality.Decode {
		return r.Decode
	}
	return r.Prefill
}

func readJSON(path string, v any) {
	b, err := os.ReadFile(path)
	if err != nil {
		log.Fatal(err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		log.Fatalf("%s: %v", path, err)
	}
}

// positive reports whether v is finite and positive: NaN and infinities are no rates.
func positive(v float64) bool { return v > 0 && !math.IsInf(v, 1) }

// parseLinks reads name=GB/s pairs.
func parseLinks(s string) ([]locality.Link, error) {
	var out []locality.Link
	for f := range strings.SplitSeq(s, ",") {
		name, v, ok := strings.Cut(f, "=")
		g, err := strconv.ParseFloat(v, 64)
		if !ok || err != nil || !positive(g) {
			return nil, fmt.Errorf("%q is not name=GB/s", f)
		}
		out = append(out, locality.Link{Name: name, BytesPerSecond: g * 1e9})
	}
	return out, nil
}

func ints(flagName, s string) []int {
	var out []int
	for f := range strings.SplitSeq(s, ",") {
		c, err := strconv.Atoi(f)
		if err != nil || c < 0 {
			log.Fatalf("-%s: %q is not a count", flagName, f)
		}
		out = append(out, c)
	}
	return out
}

func main() {
	home, _ := os.UserHomeDir()
	dir := flag.String("dir", filepath.Join(home, "models", "eval", "locality"), "selection files, token IDs and output")
	models := flag.String("models", "", "comma-separated GGUF names without .gguf")
	workloads := flag.String("workloads", "coding,research,hr,medtech", "comma-separated workloads")
	traces := flag.String("traces", "testdata/locality", "the workload traces")
	sizesFlag := flag.String("sizes", "8,16,32,48,64,96,128,192,256,384", "cache sizes in experts per layer")
	ubatch := flag.Int("ubatch", 512, "prefill ubatch size (llama-server's default)")
	bytesPath := flag.String("bytes", "testdata/locality/expert-bytes.json", "expert sizes per model")
	bwFlag := flag.String("bw", "ssd=5,pcie=25,ram=100", "declared links, name=GB/s, each estimated on its own")
	decodeTPS := flag.Float64("decode-tps", 63.7, "measured resident decode speed, tokens/s (tg128 on Metal)")
	prefillTPS := flag.Float64("prefill-tps", 1347, "measured resident prefill speed, tokens/s (pp512 on Metal)")
	halfLife := flag.Int("lfu-halflife", 512, "LFU: halve all counts every this many tokens")
	splitsPath := flag.String("splits", "testdata/locality/splits.json", "the workloads' train/test splits")
	detail := flag.Int("detail", 128, "the size the detail tables show")
	layersFlag := flag.String("layers", "0,1,2,10,25,49", "the layers whose recall the report shows")
	md := flag.String("md", "", "write the Markdown report here")
	flag.Parse()
	if *models == "" {
		flag.Usage()
		os.Exit(2)
	}
	sizes := ints("sizes", *sizesFlag)
	layers := ints("layers", *layersFlag)
	links, err := parseLinks(*bwFlag)
	if err != nil {
		log.Fatalf("-bw: %v", err)
	}
	if !positive(*decodeTPS) || !positive(*prefillTPS) || *halfLife <= 0 || at(sizes, *detail) < 0 {
		log.Fatal("-decode-tps and -prefill-tps must be finite and positive, -lfu-halflife positive, -detail one " +
			"of -sizes")
	}
	var splits SplitManifest
	readJSON(*splitsPath, &splits)
	hs := halfLives(*halfLife)
	a := Assumptions{Ubatch: *ubatch, LinkGBPerSecond: map[string]float64{}, DecodeTokensPerSecond: *decodeTPS,
		PrefillTokensPerSecond: *prefillTPS, LFUHalfLife: *halfLife, LFUHalfLives: hs, SplitRule: splits.Rule}
	for _, l := range links {
		a.LinkGBPerSecond[l.Name] = l.BytesPerSecond / 1e9
	}
	var known map[string]Model
	readJSON(*bytesPath, &known)
	var results []result
	grids := map[string]map[string]Grid{}
	names := strings.Split(*models, ",")
	for _, name := range names {
		m, ok := known[name]
		if !ok {
			log.Fatalf("%s: no expert sizes in %s (run tools/locality/expert_bytes.py --record)", name, *bytesPath)
		}
		for _, l := range layers {
			if l >= len(m.Layers) {
				log.Fatalf("-layers: %s has %d layers", name, len(m.Layers))
			}
		}
		var cache map[string]CacheSizes
		readJSON(filepath.Join(*dir, "cache-"+name+".json"), &cache)
		grids[short(name)] = map[string]Grid{}
		for wl := range strings.SplitSeq(*workloads, ",") {
			sel, err := locality.ReadNPYFile(filepath.Join(*dir, fmt.Sprintf("%s.%s.experts.npy", wl, name)))
			if err != nil {
				log.Fatal(err)
			}
			if sel.Layers != len(m.Layers) {
				log.Fatalf("%s %s: %d layers of selections, %d of expert sizes", name, wl, sel.Layers, len(m.Layers))
			}
			f, err := os.Open(filepath.Join(*dir, wl+".tokens.npy"))
			if err != nil {
				log.Fatal(err)
			}
			ids, err := locality.ReadTokens(f)
			f.Close()
			if err != nil {
				log.Fatalf("%s tokens: %v", wl, err)
			}
			f, err = os.Open(filepath.Join(*traces, wl+".json"))
			if err != nil {
				log.Fatal(err)
			}
			answers, ended, err := locality.ReadAnswers(f)
			f.Close()
			if err != nil {
				log.Fatalf("%s trace: %v", wl, err)
			}
			if len(ids) != sel.Tokens {
				log.Fatalf("FAIL %s: %d token IDs, %d tokens of selections", wl, len(ids), sel.Tokens)
			}
			if err := locality.CheckAnswers(ids, answers); err != nil {
				log.Fatalf("FAIL %s: %v", wl, err)
			}
			fmt.Printf("PASS %s: the token IDs hold the %d answers' generated tokens after their prompts\n", wl,
				len(answers))
			spans, err := locality.OnlineSpans(sel.Tokens, answers, ended, *ubatch)
			if err != nil {
				log.Fatalf("%s: %v", wl, err)
			}
			d, ok := splits.Workloads[wl]
			if !ok {
				log.Fatalf("%s: no split in %s", wl, *splitsPath)
			}
			cut, err := split(d, spans, answers)
			if err != nil {
				log.Fatalf("FAIL %s: the split disagrees with %s: %v", wl, *splitsPath, err)
			}
			fmt.Printf("PASS %s: split after %d of %d answers at token %d, as in %s\n", wl, d.TrainAnswers, d.Answers,
				d.FirstTestToken, filepath.Base(*splitsPath))
			tr, err := traceOrder(sel, m, sizes)
			if err != nil {
				log.Fatalf("%s %s: %v", name, wl, err)
			}
			if err := check(tr.LayerLRU, sizes, cache[wl].PerLayer); err != nil {
				log.Fatalf("FAIL %s %s: the layer LRU in trace order disagrees with cache-%s.json: %v", short(name),
					wl, short(name), err)
			}
			fmt.Printf("PASS %s %s: per-layer LRU in trace order hits as in cache-%s.json at %d sizes\n", short(name),
				wl, short(name), len(sizes))
			if err := check(tr.SharedLRU, sizes, cache[wl].Shared); err != nil {
				log.Fatalf("FAIL %s %s: the shared LRU in trace order disagrees with cache-%s.json: %v",
					short(name), wl, short(name), err)
			}
			fmt.Printf("PASS %s %s: shared LRU in trace order hits as in cache-%s.json at %d sizes\n", short(name),
				wl, short(name), len(sizes))
			timing := locality.Timing{Links: links, DecodeStep: 1 / (*decodeTPS * float64(sel.Layers)),
				PrefillToken: 1 / (*prefillTPS * float64(sel.Layers))}
			noCache, runs, err := replay(sel, spans, cut, m, sizes, baselines(sel, spans, m, *halfLife), timing)
			if err != nil {
				log.Fatalf("%s %s: %v", name, wl, err)
			}
			grid, err := lfuGrid(sel, spans, cut, m, sizes, hs)
			if err != nil {
				log.Fatalf("%s %s: %v", name, wl, err)
			}
			grids[short(name)][wl] = grid
			if err := checkOracles(noCache.Online, runs); err != nil {
				log.Fatalf("FAIL %s %s: %v", short(name), wl, err)
			}
			fmt.Printf("PASS %s %s: the shared oracle loads no more than any policy, the per-layer one no more "+
				"than the layer LRU, at %d sizes\n", short(name), wl, len(sizes))
			w := Workload{Tokens: sel.Tokens, Answers: len(answers), Split: d, NoCache: noCache, Policies: runs,
				TraceOrder: tr, LFUGrid: grid}
			for _, s := range spans {
				if s.Phase == locality.Prefill {
					w.Ubatches++
				}
			}
			r := result{short(name), wl, w}
			results = append(results, r)
			fmt.Println(summary(r, sizes, *detail))
		}
	}
	for i := range results {
		r := &results[i]
		for s, c := range sizes {
			r.w.LFUTuning = append(r.w.LFUTuning, tune(grids, r.model, r.workload, s, c, *halfLife))
		}
	}
	for _, name := range names {
		out := Output{Assumptions: a, Workloads: map[string]Workload{}}
		for _, r := range results {
			if r.model == short(name) {
				out.Workloads[r.workload] = r.w
			}
		}
		b, err := json.MarshalIndent(out, "", " ")
		if err != nil {
			log.Fatal(err)
		}
		path := filepath.Join(*dir, "policy-"+name+".json")
		if err := os.WriteFile(path, append(b, '\n'), 0o644); err != nil {
			log.Fatal(err)
		}
		fmt.Println("wrote", path)
	}
	if *md != "" {
		f, err := os.Create(*md)
		if err != nil {
			log.Fatal(err)
		}
		writeMarkdown(f, "go run ./cmd/kolibri-policy -models "+*models+" -md "+*md, a, links, sizes, *detail, layers,
			results)
		if err := f.Close(); err != nil {
			log.Fatal(err)
		}
		fmt.Println("wrote", *md)
	}
}
