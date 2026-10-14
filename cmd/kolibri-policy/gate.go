package main

import (
	"fmt"
	"io"
	"slices"
	"strconv"
	"strings"

	"kolibri-llm/internal/locality"
)

// lastNs are the window lengths the last-N prefetcher is replayed and tuned at, in steps.
var lastNs = []int{1, 2, 4, 8}

// LastNGrid is the last-N prefetcher on the shared LRU, replayed online at every n and size; its
// recall of the test suffix's decoded experts at every n; and at every size the n tuned on the
// workload's own training prefix.
type LastNGrid struct {
	N      []int             `json:"n"`
	Runs   []Runs            `json:"runs"`
	Recall []locality.Recall `json:"test_recall"`
	Own    []int             `json:"own_prefix_n"`
}

// lastNGrid replays last-N at every n of ns and size and tunes n per size on the training prefix:
// the fewest decode bytes loaded, missed or prefetched, as the gate counts them, the first of equal
// ones.
func lastNGrid(sel locality.Selections, spans []locality.Span, cut int, m Model, sizes, ns []int,
	timing locality.Timing) (LastNGrid, error) {
	bases := make([]baseline, len(ns))
	for i, n := range ns {
		bases[i] = baseline{fmt.Sprintf("last-%d on shared LRU", n), false, func(c int) (locality.Policy, error) {
			return locality.NewLastN(locality.NewSharedLRU(m.NExpert, m.bytes(), m.budget(c)), sel.Layers,
				m.NExpert, n), nil
		}}
	}
	runs, err := replayAll(sel, spans, cut, m, sizes, bases, timing)
	if err != nil {
		return LastNGrid{}, err
	}
	g := LastNGrid{N: ns, Runs: runs}
	for _, n := range ns {
		g.Recall = append(g.Recall, locality.LastNRecall(sel, spans, cut, m.NExpert, n))
	}
	for s := range sizes {
		v := make([]int64, len(ns))
		for i := range ns {
			tr := runs[i].Runs[s].Split.Train.Decode
			v[i] = loadedBytes(tr)
		}
		g.Own = append(g.Own, ns[argmin(v)])
	}
	return g, nil
}

// loadedBytes is the expert bytes a phase loads, missed or prefetched, used or not: what the gate
// judges and last-N is tuned on.
func loadedBytes(c locality.Counts) int64 { return c.MissBytes + c.PrefetchBytes }

// GateSize is a policy's decode in all test suffixes of a model together at one size: the expert
// bytes it loads, missed or prefetched, its estimated seconds on every link, and what of these is
// less than the shared LRU's.
type GateSize struct {
	ExpertsPerLayer int       `json:"experts_per_layer"`
	LoadedBytes     int64     `json:"loaded_bytes"`
	Seconds         []float64 `json:"seconds"`
	Wins            []string  `json:"wins,omitempty"`
}

// GateEntry is a policy at the gate: retained if it beats the shared LRU anywhere.
type GateEntry struct {
	Policy   string     `json:"policy"`
	Retained bool       `json:"retained"`
	Sizes    []GateSize `json:"sizes"`
}

// Gate is Stage A's gate for one model: every causal policy against the shared LRU on the decode
// of all test suffixes together, at every size.
type Gate struct {
	Tokens    int         `json:"test_decode_tokens"`
	Links     []string    `json:"links"`
	Reference GateEntry   `json:"reference"`
	Policies  []GateEntry `json:"policies"`
}

// heldOut is a policy's test suffixes, per workload and size.
type heldOut struct {
	policy string
	tests  [][]locality.Result
}

// sumHeld adds a policy's test suffixes up per size.
func sumHeld(h heldOut, sizes []int, links int) GateEntry {
	e := GateEntry{Policy: h.policy}
	for s, c := range sizes {
		g := GateSize{ExpertsPerLayer: c, Seconds: make([]float64, links)}
		for _, t := range h.tests {
			d := t[s].Decode
			g.LoadedBytes += loadedBytes(d)
			for i := range g.Seconds {
				g.Seconds[i] += d.Links[i].Seconds
			}
		}
		e.Sizes = append(e.Sizes, g)
	}
	return e
}

// sameTime is the relative difference below which two estimated times are equal: the replays add
// up their steps' times in floating point, and policies that load the same experts at the same
// times can differ in the last digits.
const sameTime = 1e-9

// gate judges every candidate against ref: a size where it loads fewer bytes, or takes less time on
// a link by more than sameTime, is a win; a policy with a win is retained. Equal is no win.
func gate(ref heldOut, cands []heldOut, sizes []int, links []string, tokens int) Gate {
	g := Gate{Tokens: tokens, Links: links, Reference: sumHeld(ref, sizes, len(links))}
	for _, c := range cands {
		e := sumHeld(c, sizes, len(links))
		for s := range e.Sizes {
			x, r := &e.Sizes[s], g.Reference.Sizes[s]
			if x.LoadedBytes < r.LoadedBytes {
				x.Wins = append(x.Wins, "bytes")
			}
			for i, l := range links {
				if x.Seconds[i] < r.Seconds[i]*(1-sameTime) {
					x.Wins = append(x.Wins, l)
				}
			}
			if len(x.Wins) > 0 {
				e.Retained = true
			}
		}
		g.Policies = append(g.Policies, e)
	}
	return g
}

// tests are the test suffixes of runs, per size.
func tests(runs []Run) []locality.Result {
	out := make([]locality.Result, len(runs))
	for s, r := range runs {
		out[s] = r.Split.Test
	}
	return out
}

// tuned are the test suffixes of the grid run whose parameter of ps is chosen at each size.
func tuned(runs []Runs, ps []int, chosen func(s int) int) []locality.Result {
	out := make([]locality.Result, len(runs[0].Runs))
	for s := range out {
		out[s] = runs[slices.Index(ps, chosen(s))].Runs[s].Split.Test
	}
	return out
}

// gateOf is the gate of model on results; ok is false when the model has no workloads.
func gateOf(results []result, model string, sizes []int, links []string) (Gate, bool) {
	ref := heldOut{policy: "shared LRU"}
	names := []string{"layer LRU", "LFU", "LFU, half-life tuned on the own prefix", "ARC",
		"layer budgets allocated on the own prefix", "layer budgets allocated leaving one workload out",
		"last-N on shared LRU, n tuned on the own prefix"}
	cands := make([]heldOut, len(names))
	for i, n := range names {
		cands[i].policy = n
	}
	tokens := 0
	for _, r := range results {
		if r.model != model {
			continue
		}
		tokens += r.w.NoCache.Split.Test.Decode.Tokens
		w := r.w
		ref.tests = append(ref.tests, tests(w.Policies[sharedLRU].Runs))
		lfuTuned := tuned(w.LFUGrid.runs, w.LFUGrid.HalfLives, func(s int) int { return w.LFUTuning[s].Own.HalfLife })
		lastN := tuned(w.LastN.Runs, w.LastN.N, func(s int) int { return w.LastN.Own[s] })
		per := [][]locality.Result{tests(w.Policies[layerLRU].Runs), tests(w.Policies[lfu].Runs), lfuTuned,
			tests(w.Policies[arc].Runs), nil, nil, lastN}
		for i, f := range []int{ownFit, lowoFit} {
			if a := fitRuns(r, f); a != nil {
				per[4+i] = tests(a.Runs)
			}
		}
		for i, t := range per {
			if t == nil {
				cands[i].policy = "" // missing in a workload: not judged
			}
			cands[i].tests = append(cands[i].tests, t)
		}
	}
	if len(ref.tests) == 0 {
		return Gate{}, false
	}
	var judged []heldOut
	for _, c := range cands {
		if c.policy != "" {
			judged = append(judged, c)
		}
	}
	return gate(ref, judged, sizes, links, tokens), true
}

// writeLastN reports the last-N prefetcher: its recall and its held-out replays.
func writeLastN(w io.Writer, sizes []int, links []locality.Link, results []result) {
	if len(results) == 0 {
		return
	}
	ns := results[0].w.LastN.N
	fmt.Fprintf(w, "\n## Last-N reuse, the first prediction baseline\n\n")
	fmt.Fprintf(w, "The last-N predictor expects a layer to need again the experts it needed in its last n steps. "+
		"As a policy it prefetches them one layer ahead, before the layer before it runs, into a shared LRU of "+
		"the budget, which keeps or evicts them like any other expert. It knows only steps that have run. n is "+
		"chosen per budget from %s on the own training prefix, as the one whose decode loads the fewest "+
		"bytes, missed or prefetched, as the gate counts them.\n\n", joinInts(ns))
	fmt.Fprintf(w, "Recall of the test suffixes' decoded experts, per layer and token:\n\n| Model | n | Candidates | "+
		"Experts covered | Sets covered |\n|---|--:|--:|--:|--:|\n")
	for _, m := range modelsOf(results) {
		for i, n := range ns {
			var sum locality.Recall
			for _, r := range results {
				if r.model == m {
					x := r.w.LastN.Recall[i]
					sum.Steps += x.Steps
					sum.Candidates += x.Candidates
					sum.Needed += x.Needed
					sum.Covered += x.Covered
					sum.Exact += x.Exact
				}
			}
			fmt.Fprintf(w, "| %s | %d | %.1f | %s | %s |\n", m, n, float64(sum.Candidates)/float64(max(sum.Steps, 1)),
				pct(float64(sum.Covered)/float64(max(sum.Needed, 1))), pct(float64(sum.Exact)/float64(max(sum.Steps, 1))))
		}
	}
	for _, m := range modelsOf(results) {
		fmt.Fprintf(w, "\n%s, all workloads' test suffixes together, decode, shared LRU against last-N with the tuned "+
			"n (MB/token loaded, prefetched in vain, ms/token per link):\n\n| Experts per layer | n per workload | "+
			"Shared LRU MB | Last-N MB | Unused MB |", m)
		for _, l := range links {
			fmt.Fprintf(w, " Shared LRU %s | Last-N %s |", l.Name, l.Name)
		}
		fmt.Fprintf(w, "\n|--:|---|--:|--:|--:|%s\n", strings.Repeat("--:|--:|", len(links)))
		for s, c := range sizes {
			var tokens int
			var ref, got, unused int64
			rs, gs := make([]float64, len(links)), make([]float64, len(links))
			var chosen []string
			for _, r := range results {
				if r.model != m {
					continue
				}
				n := r.w.LastN.Own[s]
				chosen = append(chosen, strconv.Itoa(n))
				x := r.w.Policies[sharedLRU].Runs[s].Split.Test.Decode
				y := r.w.LastN.Runs[slices.Index(r.w.LastN.N, n)].Runs[s].Split.Test.Decode
				tokens += x.Tokens
				ref += loadedBytes(x)
				got += loadedBytes(y)
				unused += y.UnusedPrefetchBytes
				for i := range links {
					rs[i] += x.Links[i].Seconds
					gs[i] += y.Links[i].Seconds
				}
			}
			fmt.Fprintf(w, "| %d | %s | %s | %s | %s |", c, strings.Join(chosen, ", "), mbPer(ref, tokens),
				mbPer(got, tokens), mbPer(unused, tokens))
			for i := range links {
				fmt.Fprintf(w, " %.1f | %.1f |", 1e3*rs[i]/float64(max(tokens, 1)), 1e3*gs[i]/float64(max(tokens, 1)))
			}
			fmt.Fprintln(w)
		}
	}
}

// writeGate reports Stage A's gate.
func writeGate(w io.Writer, gates map[string]Gate, models []string) {
	fmt.Fprintf(w, "\n## Stage A gate\n\n")
	fmt.Fprintf(w, "A causal policy is retained if, on the decode of all test suffixes of a model together, it beats "+
		"the shared LRU at one or more budgets: it loads fewer expert bytes, missed or prefetched, or its estimated "+
		"time per token is lower on a declared link. Equal is no win, and times within a part in 10⁹ are equal. "+
		"Tuned and allocated policies use the parameter chosen on training prefixes; the oracle and the hindsight "+
		"fits are bounds and not judged. A size alone under faster means faster on every link.\n")
	for _, m := range models {
		g, ok := gates[m]
		if !ok {
			continue
		}
		fmt.Fprintf(w, "\n%s, %d decoded test tokens:\n\n| Policy | Verdict | Fewer bytes at | Faster at | "+
			"Most bytes saved | Most time saved |\n|---|---|---|---|---|---|\n", m, g.Tokens)
		for _, e := range g.Policies {
			var bytesAt, fasterAt []string
			bestBytes, bestTime := "", ""
			var bytesShare, timeShare float64
			for s, x := range e.Sizes {
				r := g.Reference.Sizes[s]
				var links []string
				for _, win := range x.Wins {
					if win == "bytes" {
						bytesAt = append(bytesAt, strconv.Itoa(x.ExpertsPerLayer))
						if share := 1 - float64(x.LoadedBytes)/float64(r.LoadedBytes); share > bytesShare {
							bytesShare, bestBytes = share, fmt.Sprintf("%s at %d", pct(share), x.ExpertsPerLayer)
						}
						continue
					}
					links = append(links, win)
					i := slices.Index(g.Links, win)
					if share := 1 - x.Seconds[i]/r.Seconds[i]; share > timeShare {
						timeShare, bestTime = share, fmt.Sprintf("%s on %s at %d", pct(share), win, x.ExpertsPerLayer)
					}
				}
				switch {
				case len(links) == len(g.Links):
					fasterAt = append(fasterAt, strconv.Itoa(x.ExpertsPerLayer))
				case len(links) > 0:
					fasterAt = append(fasterAt, fmt.Sprintf("%d (%s)", x.ExpertsPerLayer, strings.Join(links, ", ")))
				}
			}
			verdict := "not retained"
			if e.Retained {
				verdict = "**retained**"
			}
			fmt.Fprintf(w, "| %s | %s | %s | %s | %s | %s |\n", e.Policy, verdict, orDash(strings.Join(bytesAt, ", ")),
				orDash(strings.Join(fasterAt, ", ")), orDash(bestBytes), orDash(bestTime))
		}
	}
}

func orDash(s string) string {
	if s == "" {
		return "–"
	}
	return s
}

func joinInts(v []int) string {
	var s []string
	for _, x := range v {
		s = append(s, strconv.Itoa(x))
	}
	return strings.Join(s, ", ")
}
