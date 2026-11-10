package main

// The perfect same-token predictor (internal/locality/oracle.go), Stage E6's reference: an oracle
// at the same token's horizon that knows the step it predicts, replayed and judged at Stage B's
// gate like the predictors, with a declared cost per decoded layer step (-oracle-predictor-us). No
// predictor of that horizon covers the predicted step better. It is not a bound on the gate's
// bytes or time: beyond the true experts it pads with Markov ranks, and other extras can serve
// later tokens better. Off by default.

import (
	"fmt"
	"io"
	"math"
	"slices"
	"strconv"
	"strings"

	"kolibri-llm/internal/locality"
)

// PerfectGrid is the oracle at one cost on a workload: its prefetches into the shared LRU at every
// k, with that cost and without, k tuned per size and link on the own training prefix. Beyond the
// true experts it ranks as the same token's Markov predictor with PadParams, the own prefix's.
type PerfectGrid struct {
	Horizon     string                `json:"horizon"`
	LeadSteps   int                   `json:"lead_steps"`
	CostSeconds float64               `json:"cost_seconds_per_step"`
	PadParams   locality.MarkovParams `json:"pad_params"`
	Prefetches
}

// parseCosts reads comma-separated costs in µs as seconds; empty is none.
func parseCosts(s string) ([]float64, error) {
	if s == "" {
		return nil, nil
	}
	var out []float64
	for f := range strings.SplitSeq(s, ",") {
		v, err := strconv.ParseFloat(f, 64)
		if err != nil || !(v >= 0) || math.IsInf(v, 1) {
			return nil, fmt.Errorf("%q is not a finite cost in µs, not negative", f)
		}
		out = append(out, v/1e6)
	}
	return out, nil
}

// perfectGrids replays the oracle's prefetches at every k of markovKs, size and cost. Its ranking
// and its replays without cost do not depend on the cost and are made once.
func perfectGrids(t trace, pad locality.MarkovParams, sizes []int, costs []float64) ([]PerfectGrid, error) {
	ranking := locality.RankWith(locality.NewPerfect(t.sel, locality.NewMarkov(t.sel.Layers, t.m.NExpert, pad)),
		t.sel, t.spans, t.m.NExpert, locality.SameToken, slices.Max(markovKs))
	bases := make([]baseline, len(markovKs))
	for i, k := range markovKs {
		bases[i] = baseline{fmt.Sprintf("perfect same token, k %d, on shared LRU", k), true,
			func(c int) (locality.Policy, error) {
				return locality.NewRanked(locality.NewSharedLRU(t.m.NExpert, t.m.bytes(), t.m.budget(c)), ranking, k), nil
			}}
	}
	free, err := replayAll(t.sel, t.spans, t.cut, t.m, sizes, bases, t.timing)
	if err != nil {
		return nil, err
	}
	var out []PerfectGrid
	for _, c := range costs {
		g := PerfectGrid{Horizon: locality.SameToken.String(), LeadSteps: locality.SameToken.Lead(t.sel.Layers),
			CostSeconds: c, PadParams: pad, Prefetches: Prefetches{K: markovKs, Free: free}}
		if g.Runs, err = replayAll(t.sel, t.spans, t.cut, t.m, sizes, bases, withCost(t.timing, c)); err != nil {
			return nil, err
		}
		g.OwnK = chooseK(g.Runs, markovKs, len(t.timing.Links))
		out = append(out, g)
	}
	return out, nil
}

// perfectKind is the predictor kind's name of the oracle at cost seconds per step.
func perfectKind(cost float64) string { return fmt.Sprintf("perfect (oracle, %g µs)", 1e6*cost) }

// stageBKinds are the predictors Stage B's gate judges: predictorKinds, then the oracle at every
// cost the workloads of results have it at, for the same token only, then the hidden-state
// predictor likewise (hidden.go).
func stageBKinds(results []result) []predictorKind {
	kinds := slices.Clone(predictorKinds)
	if len(results) == 0 {
		return kinds
	}
	same := slices.Index(horizons, locality.SameToken)
	for i, g := range results[0].w.Perfect {
		kinds = append(kinds, predictorKind{perfectKind(g.CostSeconds), func(w Workload, hi int) *Prefetches {
			if hi != same || i >= len(w.Perfect) {
				return nil
			}
			return &w.Perfect[i].Prefetches
		}})
	}
	return append(kinds, hiddenKinds(results)...)
}

// perfectChecks are the CHECK lines of the oracle at Stage B's gate g: per cost, its rows retained.
func perfectChecks(model string, g Gate, costs []float64) []string {
	var out []string
	for _, c := range costs {
		kind := perfectKind(c)
		n, of := 0, 0
		var wins []string
		for _, e := range g.Policies {
			if !strings.Contains(e.Policy, kind) {
				continue
			}
			of++
			if e.Retained {
				n++
				for _, x := range e.Sizes {
					for _, w := range x.Wins {
						wins = append(wins, fmt.Sprintf("%s at %d", w, x.ExpertsPerLayer))
					}
				}
			}
		}
		line := fmt.Sprintf("CHECK %s: the perfect same-token predictor at %g µs per decoded layer step is retained "+
			"in %d of %d rows of the Stage B gate", model, 1e6*c, n, of)
		if len(wins) > 0 {
			line += " (" + strings.Join(wins, ", ") + ")"
		}
		out = append(out, line)
	}
	return out
}

// writePerfect reports the oracle, if the workloads have it: what it is and, per cost, its held-out
// decode against the best Stage A policy.
func writePerfect(w io.Writer, links []locality.Link, sizes []int, results []result, gatesB map[string]Gate) {
	if len(results) == 0 || len(results[0].w.Perfect) == 0 {
		return
	}
	layers := results[0].t.sel.Layers
	var costs []string
	for _, g := range results[0].w.Perfect {
		costs = append(costs, fmt.Sprintf("%g", 1e6*g.CostSeconds))
	}
	fmt.Fprintf(w, "\n## Stage E6 bound: a perfect same-token predictor\n\n")
	fmt.Fprintf(w, "An oracle at the same token's horizon, not a predictor: before layer l of a decoded token it "+
		"predicts the token's layer l+1, %d step ahead, as the same-token predictors do, and its first candidates "+
		"are exactly the experts that step uses, read from the selection file. Beyond them, for k above the "+
		"Top-%d, it ranks as the same-token Markov predictor with the own prefix's parameters, without them. It "+
		"predicts where that horizon does, layers 2 to %d of every decoded token. A hidden-state predictor that "+
		"applies layer l+1's router to layer l−1's output predicts the same step at the same time and cannot "+
		"cover that step better than knowing it. This is a reference, not a bound on the gate: a prefetched "+
		"expert stays in the cache, so candidates beyond the true experts, which the oracle takes from the "+
		"Markov predictor, can serve later tokens, and a predictor with better ones can load less or run "+
		"faster. An oracle that is not retained with a cost makes a predictor of this horizon with that cost "+
		"unlikely, not impossible. Its cost per decoded layer step is declared (`-oracle-predictor-us`: %s µs), added to "+
		"every decoded step's compute as the predictors' is. It is replayed and judged at Stage B's gate exactly "+
		"as they are: k from %s tuned per budget and link on the own training prefix, bytes prefetched counted "+
		"as loaded.\n", locality.SameToken.Lead(layers), results[0].t.sel.K, layers-1, strings.Join(costs, ", "),
		joinInts(markovKs))
	same := slices.Index(horizons, locality.SameToken)
	for _, m := range modelsOf(results) {
		gb, ok := gatesB[m]
		if !ok {
			continue
		}
		for ci, cost := range results[0].w.Perfect {
			fmt.Fprintf(w, "\n%s, the oracle at %g µs, all workloads' test suffixes together, decode: MB/token loaded "+
				"and ms/token per link, the best Stage A policy against the oracle with k tuned for the link (k the "+
				"range of the workloads' choices):\n\n| Experts per layer |", m, 1e6*cost.CostSeconds)
			for _, l := range links {
				fmt.Fprintf(w, " k %s | Best Stage A MB | Oracle MB | Best Stage A %s | Oracle %s |", l.Name, l.Name,
					l.Name)
			}
			fmt.Fprintf(w, "\n|--:|%s\n", strings.Repeat("--:|--:|--:|--:|--:|", len(links)))
			for s, c := range sizes {
				fmt.Fprintf(w, "| %d |", c)
				for i := range links {
					var tokens int
					var sec float64
					var b int64
					var ks []int
					for _, r := range results {
						if r.model != m {
							continue
						}
						p := stageBKinds(results)[len(predictorKinds)+ci].pick(r.w, same)
						k := p.OwnK[s][i]
						ks = append(ks, k)
						x := p.Runs[slices.Index(p.K, k)].Runs[s].Split.Test.Decode
						tokens += x.Tokens
						sec += x.Links[i].Seconds
						b += loadedBytes(x)
					}
					kr := fmt.Sprint(slices.Min(ks))
					if slices.Max(ks) != slices.Min(ks) {
						kr += fmt.Sprintf("–%d", slices.Max(ks))
					}
					ref := gb.Reference.Sizes[s]
					per := func(v float64) string { return fmt.Sprintf("%.2f", 1e3*v/float64(max(tokens, 1))) }
					mb := func(v int64) string { return fmt.Sprintf("%.1f", float64(v)/1e6/float64(max(tokens, 1))) }
					fmt.Fprintf(w, " %s | %s | %s | %s | %s |", kr, mb(ref.LoadedBytes), mb(b),
						per(ref.Seconds[i]), per(sec))
				}
				fmt.Fprintln(w)
			}
		}
	}
}
