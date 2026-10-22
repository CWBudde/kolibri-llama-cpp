package main

// The decayed-count Markov predictor (PLAN Phase 10, Stage B, internal/locality/predict.go) on the
// workloads: its recall per horizon, its parameters tuned on training prefixes, its prefetches
// replayed into the shared LRU with and without its own CPU cost, and Stage B's gate against the
// best Stage A policy.

import (
	"fmt"
	"io"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"sync"

	"kolibri-llm/internal/locality"
)

var (
	markovHalfLives = []int{1, 4, 16, 64, 256, 1024, 4096}
	markovMixes     = []float64{0, 0.25, 0.5, 0.75, 1}
	markovKs        = []int{6, 12, 24, 48}
	horizons        = []locality.Horizon{locality.NextToken, locality.SameToken}
)

// markovParams is the predictor's parameter grid: every half-life with every mix below 1, and the
// transitions alone, mix 1, which no half-life changes.
func markovParams() []locality.MarkovParams {
	var out []locality.MarkovParams
	for _, w := range markovMixes {
		if w == 1 {
			out = append(out, locality.MarkovParams{Mix: 1})
			continue
		}
		for _, h := range markovHalfLives {
			out = append(out, locality.MarkovParams{HalfLife: h, Mix: w})
		}
	}
	return out
}

// Prefetches are a predictor's prefetches into the shared LRU at every k of K and size, replayed
// with the predictor's cost and without it, and the k chosen per size and link on the own
// training prefix.
type Prefetches struct {
	K    []int   `json:"k"`
	Runs []Runs  `json:"runs"`
	Free []Runs  `json:"runs_without_cost"`
	OwnK [][]int `json:"own_prefix_k"` // per size and link
}

// HorizonGrid is the predictor of one horizon on a workload: its recall at every parameter and
// candidate count in the training prefix and the test suffix, the parameters chosen, and its
// prefetches with the own prefix's parameters. Indexes are into Params.
type HorizonGrid struct {
	Horizon    string                  `json:"horizon"`
	LeadSteps  int                     `json:"lead_steps"`
	Params     []locality.MarkovParams `json:"params"`
	Train      [][]locality.Recall     `json:"train_recall"`
	Test       [][]locality.Recall     `json:"test_recall"`
	Own        int                     `json:"own_prefix_params"`
	CountsOnly int                     `json:"counts_only_params"`
	LOWO       int                     `json:"leave_one_workload_out_params"` // -1: no other workload
	Prefetches
}

// covered is the needed experts covered summed over the candidate counts: what tuning maximizes.
func covered(rs []locality.Recall) int {
	n := 0
	for _, r := range rs {
		n += r.Covered
	}
	return n
}

// bestParams is the parameters whose score is highest among those keep admits, the first of
// equal ones; -1 if keep admits none.
func bestParams(params []locality.MarkovParams, keep func(locality.MarkovParams) bool, score func(i int) int) int {
	best := -1
	for i, p := range params {
		if keep != nil && !keep(p) {
			continue
		}
		if best < 0 || score(i) > score(best) {
			best = i
		}
	}
	return best
}

// chooseParams chooses g's parameters on its training prefix: the best of all, and the best of
// the counts alone.
func chooseParams(g *HorizonGrid) {
	score := func(i int) int { return covered(g.Train[i]) }
	g.Own = bestParams(g.Params, nil, score)
	g.CountsOnly = bestParams(g.Params, func(p locality.MarkovParams) bool { return p.Mix == 0 }, score)
}

// chooseK is, per size and link, the k whose prefetches take the least decode time in the
// training prefix, the first of equal ones.
func chooseK(runs []Runs, ks []int, links int) [][]int {
	out := make([][]int, len(runs[0].Runs))
	for s := range out {
		out[s] = make([]int, links)
		for i := range links {
			best := 0
			for j := range ks {
				if runs[j].Runs[s].Split.Train.Decode.Links[i].Seconds <
					runs[best].Runs[s].Split.Train.Decode.Links[i].Seconds {
					best = j
				}
			}
			out[s][i] = ks[best]
		}
	}
	return out
}

// withCost is t with the predictor's cost added to every decoded step's compute.
func withCost(t locality.Timing, cost float64) locality.Timing {
	t.DecodeStep += cost
	return t
}

// markovGrid scores the predictor of horizon h at every parameter, tunes it on the own training
// prefix and replays its prefetches at every k and size.
func markovGrid(t trace, h locality.Horizon, sizes []int, cost float64) (HorizonGrid, error) {
	params := markovParams()
	g := HorizonGrid{Horizon: h.String(), LeadSteps: h.Lead(t.sel.Layers), Params: params,
		Train: make([][]locality.Recall, len(params)), Test: make([][]locality.Recall, len(params)), LOWO: -1}
	kmax := slices.Max(markovKs)
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.GOMAXPROCS(0))
	for i, p := range params {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer func() { <-sem; wg.Done() }()
			r := locality.Rank(t.sel, t.spans, t.m.NExpert, h, p, kmax)
			g.Train[i] = r.Recall(t.sel, t.spans, 0, t.cut, markovKs)
			g.Test[i] = r.Recall(t.sel, t.spans, t.cut, len(t.spans), markovKs)
		}()
	}
	wg.Wait()
	chooseParams(&g)
	ranking := locality.Rank(t.sel, t.spans, t.m.NExpert, h, params[g.Own], kmax)
	var err error
	g.Prefetches, err = replayPrefetches(t, fmt.Sprintf("%s Markov", h), ranking, sizes, cost)
	return g, err
}

// replayPrefetches replays ranking's best k into the shared LRU at every k of markovKs and size,
// with the cost per decoded layer step and without it, and chooses k per size and link on the own
// training prefix.
func replayPrefetches(t trace, name string, ranking *locality.Ranking, sizes []int, cost float64) (Prefetches,
	error) {
	p := Prefetches{K: markovKs}
	bases := make([]baseline, len(markovKs))
	for i, k := range markovKs {
		bases[i] = baseline{fmt.Sprintf("%s, k %d, on shared LRU", name, k), false,
			func(c int) (locality.Policy, error) {
				return locality.NewRanked(locality.NewSharedLRU(t.m.NExpert, t.m.bytes(), t.m.budget(c)), ranking, k), nil
			}}
	}
	var err error
	if p.Runs, err = replayAll(t.sel, t.spans, t.cut, t.m, sizes, bases, withCost(t.timing, cost)); err != nil {
		return p, err
	}
	if p.Free, err = replayAll(t.sel, t.spans, t.cut, t.m, sizes, bases, t.timing); err != nil {
		return p, err
	}
	p.OwnK = chooseK(p.Runs, markovKs, len(t.timing.Links))
	return p, nil
}

// checkOnline fails unless pred, a predictor that has seen nothing, as an online policy that
// learns from Use alone replays exactly as p's runs prefetch, with the largest k at size index s.
func checkOnline(t trace, p Prefetches, pred locality.Predictor, h locality.Horizon, sizes []int, s int,
	cost float64) error {
	k := len(p.K) - 1
	q := locality.NewPrefetcher(locality.NewSharedLRU(t.m.NExpert, t.m.bytes(), t.m.budget(sizes[s])), pred,
		t.m.NExpert, h, p.K[k])
	got, err := replayed(t.sel, t.spans, t.cut, t.m, q, withCost(t.timing, cost))
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(got, p.Runs[k].Runs[s].Replayed) {
		return fmt.Errorf("the online predictor's replay differs from its ranking's at k %d, %d experts per layer",
			p.K[k], sizes[s])
	}
	return nil
}

// lowoParams chooses horizon hi's parameters for workload wl of model on the other workloads'
// training prefixes of the model, -1 if there are none.
func lowoParams(results []result, model, wl string, hi int) int {
	var others []HorizonGrid
	for _, r := range results {
		if r.model == model && r.workload != wl {
			others = append(others, r.w.Markov[hi])
		}
	}
	if len(others) == 0 {
		return -1
	}
	return bestParams(others[0].Params, nil, func(i int) int {
		n := 0
		for _, g := range others {
			n += covered(g.Train[i])
		}
		return n
	})
}

// bestOf is, at every size, the fewest bytes and the least time on every link of any entry.
func bestOf(entries []GateEntry, name string) GateEntry {
	out := GateEntry{Policy: name}
	for s := range entries[0].Sizes {
		b := entries[0].Sizes[s]
		b.Seconds, b.Wins = slices.Clone(b.Seconds), nil
		for _, e := range entries[1:] {
			x := e.Sizes[s]
			b.LoadedBytes = min(b.LoadedBytes, x.LoadedBytes)
			for i := range b.Seconds {
				b.Seconds[i] = min(b.Seconds[i], x.Seconds[i])
			}
		}
		out.Sizes = append(out.Sizes, b)
	}
	return out
}

// predictorName is the gate's name of the predictor kind of horizon h tuned for link.
func predictorName(kind string, h locality.Horizon, link string) string {
	return fmt.Sprintf("%s %s on shared LRU, k tuned for %s", h, kind, link)
}

// markovName is the gate's name of horizon h's Markov predictor tuned for link.
func markovName(h locality.Horizon, link string) string { return predictorName("Markov", h, link) }

// predictorKind is a kind of Stage B predictor: its name and its prefetches of horizon hi in a
// workload, nil if the workload has none.
type predictorKind struct {
	name string
	pick func(w Workload, hi int) *Prefetches
}

// predictorKinds are the Stage B predictors in the order the report and the gate list them.
var predictorKinds = []predictorKind{
	{"Markov", func(w Workload, hi int) *Prefetches {
		if hi >= len(w.Markov) {
			return nil
		}
		return &w.Markov[hi].Prefetches
	}},
	{"linear", func(w Workload, hi int) *Prefetches {
		if hi >= len(w.Linear) {
			return nil
		}
		return &w.Linear[hi].Prefetches
	}},
}

// stageBGate judges every predictor of every horizon, k tuned per link, against the best Stage A
// policy of gate a at every size, each on bytes and on its own link. A predictor some workload of
// the model lacks is not judged.
func stageBGate(results []result, model string, sizes []int, links []string, a Gate) Gate {
	ref := bestOf(append([]GateEntry{a.Reference}, a.Policies...), "best Stage A policy")
	var cands []heldOut
	var only [][]int
	for _, kind := range predictorKinds {
		for hi, h := range horizons {
			for li, l := range links {
				c := heldOut{policy: predictorName(kind.name, h, l)}
				for _, r := range results {
					if r.model != model {
						continue
					}
					p := kind.pick(r.w, hi)
					if p == nil {
						c.policy = ""
						break
					}
					c.tests = append(c.tests, tuned(p.Runs, p.K, func(s int) int { return p.OwnK[s][li] }))
				}
				if c.policy != "" {
					cands = append(cands, c)
					only = append(only, []int{li})
				}
			}
		}
	}
	return judge(ref, cands, only, sizes, links, a.Tokens)
}

// stageBIntro is the rule of Stage B's gate as the report states it.
const stageBIntro = "The predictor is retained if, on the decode of all test suffixes of a model together, it " +
	"beats the best Stage A policy at one or more budgets: the fewest bytes and the least time on each link of " +
	"the shared LRU and every Stage A policy judged at Stage A's gate, last-N included and Stage E1's not, at " +
	"that budget. Each horizon's predictor is judged once per link, with k tuned for that link on the own prefix, on bytes and on that link's " +
	"time, its CPU cost included. Equal is no win, and times within a part in 10⁹ are equal. An expert " +
	"prefetched in the training prefix and first used in the test suffix counts in the test suffix's bytes, as " +
	"if it had loaded it (carried across the cut, `Counts.CarriedPrefetchBytes`): else a wrong prediction in " +
	"the training prefix that happens to load an expert the test suffix needs first would count as a saving. No " +
	"Stage A policy carries any; the predictor's time still gains from them, which only flatters it.\n"

// paramsCell is a parameter choice.
func paramsCell(g HorizonGrid, i int) string {
	if i < 0 {
		return "–"
	}
	p := g.Params[i]
	if p.Mix == 1 {
		return "mix 1"
	}
	return fmt.Sprintf("half-life %d, mix %g", p.HalfLife, p.Mix)
}

// sumRecall adds up recalls.
func sumRecall(rs []locality.Recall) locality.Recall {
	var s locality.Recall
	for _, x := range rs {
		s.Steps += x.Steps
		s.Candidates += x.Candidates
		s.Needed += x.Needed
		s.Covered += x.Covered
		s.Exact += x.Exact
	}
	return s
}

// recallCell is experts covered and whole sets covered.
func recallCell(r locality.Recall) string {
	return fmt.Sprintf("%s (%s)", pct(float64(r.Covered)/float64(max(r.Needed, 1))),
		pct(float64(r.Exact)/float64(max(r.Steps, 1))))
}

// writeMarkov reports the predictor: its parameters, recall, lead, cost and held-out replays.
func writeMarkov(w io.Writer, a Assumptions, links []locality.Link, sizes []int, d int, results []result,
	gatesB map[string]Gate) {
	if len(results) == 0 {
		return
	}
	layers := results[0].t.sel.Layers
	timing := results[0].t.timing
	var mixes []string
	for _, x := range markovMixes {
		mixes = append(mixes, fmt.Sprintf("%g", x))
	}
	fmt.Fprintf(w, "\n## Stage B: the decayed-count Markov predictor\n\n")
	fmt.Fprintf(w, "The predictor ranks a layer's experts for a step that has not run yet, from steps that have. Per "+
		"layer it keeps every expert's use count, halved every half-life tokens, and counts how often one decoded "+
		"set of experts followed another: in the same layer from one token to the next, and within a token from a "+
		"layer to the layer after next. An expert's score is mix × its share of the transitions out of the latest "+
		"known set, plus (1 − mix) × its share of the decayed counts; the k best are the candidates, the lower "+
		"expert first of equal scores. It learns a step's experts only after the step has run, prefill ubatches' "+
		"uses included, and predicts decoded steps only, from decoded sets. Two horizons, each predicted on its "+
		"own:\n\n")
	fmt.Fprintf(w, "- **next token:** before layer l of token t it predicts layer l−1 of token t+1, from token t's "+
		"set in layer l−1, which has just run; before layer 0, layer %d of token t from token t−1's. The "+
		"prediction is made %d steps before the step it is for. It uses only earlier tokens' experts;\n", layers-1,
		locality.NextToken.Lead(layers))
	fmt.Fprintf(w, "- **same token:** before layer l of token t it predicts the token's layer l+1, from its set in "+
		"layer l−1, %d step ahead. It also uses the token's own completed layers. Nothing is predicted before "+
		"layer 0, where no layer of the token has run, nor before the last layer, which no layer follows; so layers 0 "+
		"and 1 are never predicted.\n\n", locality.SameToken.Lead(layers))
	fmt.Fprintf(w, "The parameters are chosen per horizon on the own training prefix, as the most needed experts "+
		"covered summed over k = %s: half-lives %s tokens with mixes %s, the transitions alone (mix 1) having no "+
		"half-life. **Counts only** is the best mix 0 on the own prefix, **Markov only** mix 1. **Leaving one "+
		"workload out**, the other workloads' training prefixes of the same model choose them. Choices on the "+
		"own prefix and leaving one workload out:\n\n", joinInts(markovKs), joinInts(markovHalfLives),
		strings.Join(mixes, ", "))
	fmt.Fprintf(w, "| Model | Workload | Next token, own | Next token, leave one out | Same token, own | "+
		"Same token, leave one out |\n|---|---|---|---|---|---|\n")
	for _, r := range results {
		n, s := r.w.Markov[0], r.w.Markov[1]
		fmt.Fprintf(w, "| %s | %s | %s | %s | %s | %s |\n", r.model, r.workload, paramsCell(n, n.Own),
			paramsCell(n, n.LOWO), paramsCell(s, s.Own), paramsCell(s, s.LOWO))
	}
	fmt.Fprintf(w, "\nRecall of the test suffixes' decoded experts per layer and token, all workloads together: the "+
		"needed experts covered (whole Top-6 sets covered). A step its horizon does not predict counts with "+
		"nothing covered, as in last-N's table above, which is the baseline. The prefetch budget is k experts of "+
		"the mean size per predicted step, in MB per decoded token.\n")
	for _, m := range modelsOf(results) {
		var bytes, tokens int64
		for _, r := range results {
			if r.model == m {
				b := r.t.m.bytes()
				var sum int64
				for _, x := range b {
					sum += x
				}
				bytes = sum / int64(len(b))
				break
			}
		}
		for hi, h := range horizons {
			fmt.Fprintf(w, "\n%s, %s:\n\n| Predictor |", m, h)
			for _, k := range markovKs {
				fmt.Fprintf(w, " k %d |", k)
			}
			fmt.Fprintf(w, "\n|---|%s\n", strings.Repeat("--:|", len(markovKs)))
			rows := []struct {
				name string
				pick func(g HorizonGrid) int
			}{
				{"counts only", func(g HorizonGrid) int { return g.CountsOnly }},
				{"Markov only", func(g HorizonGrid) int { return len(g.Params) - 1 }},
				{"mixed, own prefix", func(g HorizonGrid) int { return g.Own }},
				{"mixed, leave one workload out", func(g HorizonGrid) int { return g.LOWO }},
			}
			var budget []locality.Recall
			for _, row := range rows {
				sums := make([]locality.Recall, len(markovKs))
				ok := true
				tokens = 0
				for _, r := range results {
					if r.model != m {
						continue
					}
					g := r.w.Markov[hi]
					i := row.pick(g)
					if i < 0 {
						ok = false
						break
					}
					tokens += int64(r.w.NoCache.Split.Test.Decode.Tokens)
					for k := range markovKs {
						sums[k] = sumRecall([]locality.Recall{sums[k], g.Test[i][k]})
					}
				}
				fmt.Fprintf(w, "| %s |", row.name)
				for k := range markovKs {
					if !ok {
						fmt.Fprintf(w, " – |")
						continue
					}
					fmt.Fprintf(w, " %s |", recallCell(sums[k]))
				}
				fmt.Fprintln(w)
				if row.name == "mixed, own prefix" {
					budget = sums
				}
			}
			fmt.Fprintf(w, "| prefetch budget, MB/token |")
			for k := range markovKs {
				fmt.Fprintf(w, " %s |", mbPer(int64(budget[k].Candidates)*bytes, int(tokens)))
			}
			fmt.Fprintln(w)
		}
	}
	fmt.Fprintf(w, "\n**Lead time.** In the replay's schedule (transfer.go) a layer of a decoded token computes for "+
		"%.3f ms once its experts have arrived (%.1f ms per token over %d layers). A prefetch made before a step has its horizon's lead of steps' "+
		"compute to arrive, if it never waits behind other loads on the link; the experts of the largest size "+
		"that fit through a link in that time:\n\n| Model | Horizon | Lead steps | Lead ms |", 1e3*timing.DecodeStep,
		1e3*timing.DecodeStep*float64(layers), layers)
	for _, l := range links {
		fmt.Fprintf(w, " Experts in time, %s |", l.Name)
	}
	fmt.Fprintf(w, "\n|---|---|--:|--:|%s\n", strings.Repeat("--:|", len(links)))
	for _, m := range modelsOf(results) {
		var big int64
		for _, r := range results {
			if r.model == m {
				big = slices.Max(r.t.m.bytes())
				break
			}
		}
		for _, h := range horizons {
			lead := locality.LeadSeconds(timing, h, layers)
			fmt.Fprintf(w, "| %s | %s | %d | %.2f |", m, h, h.Lead(layers), 1e3*lead)
			for _, l := range links {
				fmt.Fprintf(w, " %d |", int(lead*l.BytesPerSecond/float64(big)))
			}
			fmt.Fprintln(w)
		}
	}
	nExpert := results[0].t.m.NExpert
	fmt.Fprintf(w, "\n**Cost.** Ranking and learning take %.1f µs per decoded layer step on this Mac (`-predict-us`, "+
		"from `BenchmarkMarkovStep` in `internal/locality`: 50 layers of 384 experts, 48 candidates, the slowest of "+
		"six runs on an Apple M5 Pro): %.2f ms of one CPU core per decoded token, %.1f%% of the %.1f ms the "+
		"resident model computes. The replays with cost add it to every decoded step's compute, serially, on the critical path; "+
		"the ones without cost do not. The tables of one predictor take %.0f MB for %d layers of %d experts "+
		"(`locality.MarkovBytes`).\n", 1e6*a.PredictSeconds, 1e3*a.PredictSeconds*float64(layers),
		100*a.PredictSeconds/timing.DecodeStep, 1e3*timing.DecodeStep*float64(layers),
		float64(locality.MarkovBytes(layers, nExpert))/1e6, layers, nExpert)
	fmt.Fprintf(w, "\nThe predictor prefetches its k best candidates before every step into a shared LRU of the "+
		"budget, which keeps or evicts them like any other expert; its parameters are the own prefix's. k is "+
		"chosen per budget and link from %s on the own training prefix, as the one whose decode takes the least "+
		"estimated time with the cost. Prefetching cannot load fewer bytes than the misses it replaces, only hide "+
		"their time.\n", joinInts(markovKs))
	writePrefetches(w, "Markov", links, sizes, d, results, gatesB, func(w Workload, hi int) Prefetches {
		return w.Markov[hi].Prefetches
	})
}

// writePrefetches reports the held-out replays of the prefetches pick selects, the predictor named
// kind in the tables.
func writePrefetches(w io.Writer, kind string, links []locality.Link, sizes []int, d int, results []result,
	gatesB map[string]Gate, pick func(w Workload, hi int) Prefetches) {
	for _, m := range modelsOf(results) {
		gb, ok := gatesB[m]
		if !ok {
			continue
		}
		for hi, h := range horizons {
			fmt.Fprintf(w, "\n%s, %s, all workloads' test suffixes together, decode ms/token: the best Stage A "+
				"policy, the predictor with its cost and without; k the range of the workloads' choices:\n\n"+
				"| Experts per layer |", m, h)
			for _, l := range links {
				fmt.Fprintf(w, " k %s | Best Stage A %s | %s %s | No cost %s |", l.Name, l.Name, kind, l.Name, l.Name)
			}
			fmt.Fprintf(w, "\n|--:|%s\n", strings.Repeat("--:|--:|--:|--:|", len(links)))
			for s, c := range sizes {
				fmt.Fprintf(w, "| %d |", c)
				for i := range links {
					var tokens int
					var with, free float64
					var ks []int
					for _, r := range results {
						if r.model != m {
							continue
						}
						g := pick(r.w, hi)
						k := g.OwnK[s][i]
						ks = append(ks, k)
						j := slices.Index(g.K, k)
						x, y := g.Runs[j].Runs[s].Split.Test.Decode, g.Free[j].Runs[s].Split.Test.Decode
						tokens += x.Tokens
						with += x.Links[i].Seconds
						free += y.Links[i].Seconds
					}
					kr := fmt.Sprint(slices.Min(ks))
					if slices.Max(ks) != slices.Min(ks) {
						kr += fmt.Sprintf("–%d", slices.Max(ks))
					}
					per := func(v float64) string { return fmt.Sprintf("%.1f", 1e3*v/float64(max(tokens, 1))) }
					fmt.Fprintf(w, " %s | %s | %s | %s |", kr, per(gb.Reference.Sizes[s].Seconds[i]), per(with), per(free))
				}
				fmt.Fprintln(w)
			}
		}
	}
	fmt.Fprintf(w, "\nAt %d experts per layer, the test suffixes' decode with the cost and k tuned for each link: the "+
		"used prefetches, those that arrived before their step started (timely) and those it waited for (late), "+
		"and the bytes prefetched in vain:\n\n| Model | Horizon | Link | k per workload | Used prefetches | "+
		"Timely | Late | Unused MB/token |\n|---|---|---|---|--:|--:|--:|--:|\n", sizes[d])
	for _, m := range modelsOf(results) {
		for hi, h := range horizons {
			for i, l := range links {
				var ks []string
				var tokens, used, timely, late int
				var unused int64
				for _, r := range results {
					if r.model != m {
						continue
					}
					g := pick(r.w, hi)
					k := g.OwnK[d][i]
					ks = append(ks, fmt.Sprint(k))
					x := g.Runs[slices.Index(g.K, k)].Runs[d].Split.Test.Decode
					tokens += x.Tokens
					used += x.UsefulPrefetches
					timely += x.Links[i].Timely
					late += x.Links[i].Late
					unused += x.UnusedPrefetchBytes
				}
				fmt.Fprintf(w, "| %s | %s | %s | %s | %d | %s | %d | %s |\n", m, h, l.Name, strings.Join(ks, ", "), used,
					pct(float64(timely)/float64(max(used, 1))), late, mbPer(unused, tokens))
			}
		}
	}
}
