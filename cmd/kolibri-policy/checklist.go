package main

// Stage E7's report (PLAN Phase 11): the measurements Zhang's reporting checklist asks for that the
// other sections leave out. A prefill step's union as a distribution; step-pinned oracles, which hold
// a step's experts together; Stage B's gate against references with no step evictions; the
// predictors' hit rates, bytes and step evictions; held-out oracles at every budget; and a hit rate
// over both phases. It adds sections and changes none of the others: the pinned oracles stay out of
// Workload.Policies, which the other tables list.

import (
	"fmt"
	"io"
	"math"
	"slices"
	"strconv"
	"strings"

	"kolibri-llm/internal/locality"
)

// lastNTuned is the Stage A gate's name of the tuned last-N prefetcher.
const lastNTuned = "last-N on shared LRU, n tuned on the own prefix"

// pinnedRefNames are the Stage A gate's entries that stay in the step-pinned Stage B reference,
// beside the shared LRU. Each must have no step evictions (checkPinnedRefs). Left out, because they
// have no pinned twin: LFU with a tuned half-life and the allocated layer budgets.
var pinnedRefNames = []string{"layer LRU, step-pinned", "LFU, step-pinned", "ARC, step-pinned", lastNTuned}

// unionThresholds are the sizes a step's union is counted against: the budgets in the report, and
// 244 for the claim in docs/expert-streaming.md, "Lossless".
var unionThresholds = []int{64, 128, 192, 244, 256}

// UnionStats is a workload's prefill unions at one ubatch size, over every step: one ubatch at one
// layer. The last ubatch of a prompt is usually short, so FullSteps says how many are full.
type UnionStats struct {
	Ubatch    int     `json:"ubatch"`
	Steps     int     `json:"steps"`
	FullSteps int     `json:"full_steps"`
	Mean      float64 `json:"mean"`
	Median    int     `json:"median"`
	P95       int     `json:"p95"`
	Max       int     `json:"max"`
	// Above counts the steps with a union above each of unionThresholds.
	Above []int `json:"above"`
	// LayerAbove244 is the lowest and highest share of a layer's full ubatches above 244 experts.
	LayerAbove244 [2]float64 `json:"layer_above_244"`
}

// nearestRank is the value at quantile q of ascending v.
func nearestRank(v []int, q float64) int {
	return v[max(int(math.Ceil(q*float64(len(v))))-1, 0)]
}

// unionStats measures the prefill unions of sel when its spans are cut into ubatches of ub tokens.
func unionStats(sel locality.Selections, spans []locality.Span, nExpert, ub int) UnionStats {
	st := UnionStats{Ubatch: ub, Above: make([]int, len(unionThresholds))}
	u := locality.StepUnions(sel, spans, nExpert, locality.Prefill)
	if len(u) == 0 {
		return st
	}
	st.Steps = len(u)
	above, full := make([]int, sel.Layers), make([]int, sel.Layers)
	i := 0
	for _, s := range spans {
		if s.Phase != locality.Prefill {
			continue
		}
		for l := range sel.Layers {
			v := u[i]
			i++
			st.Mean += float64(v)
			for t, c := range unionThresholds {
				if v > c {
					st.Above[t]++
				}
			}
			if s.End-s.First == ub {
				st.FullSteps++
				full[l]++
				if v > 244 {
					above[l]++
				}
			}
		}
	}
	st.Mean /= float64(len(u))
	sorted := slices.Sorted(slices.Values(u))
	st.Median, st.P95, st.Max = nearestRank(sorted, 0.5), nearestRank(sorted, 0.95), sorted[len(sorted)-1]
	st.LayerAbove244 = [2]float64{1, 0}
	for l := range above {
		share := float64(above[l]) / float64(max(full[l], 1))
		st.LayerAbove244[0] = min(st.LayerAbove244[0], share)
		st.LayerAbove244[1] = max(st.LayerAbove244[1], share)
	}
	return st
}

const (
	pinnedLayerOracle = iota
	pinnedSharedOracle
)

// pinnedBaselines are the step-pinned oracles, replayed apart from baselines.
func pinnedBaselines(sel locality.Selections, spans []locality.Span, m Model) []baseline {
	b := m.bytes()
	return []baseline{
		pinnedLayerOracle: {"oracle, per layer, step-pinned", true, func(c int) (locality.Policy, error) {
			return locality.NewPinnedLayerOracle(sel, spans, m.NExpert, b, c), nil
		}},
		pinnedSharedOracle: {"oracle, shared, step-pinned", true, func(c int) (locality.Policy, error) {
			return locality.NewPinnedOracle(sel, spans, m.NExpert, b, m.budget(c)), nil
		}},
	}
}

func allLoaded(r locality.Result) int64 { return loadedBytes(r.Prefill) + loadedBytes(r.Decode) }

// checkPinned fails unless the pinned oracles have no step evictions at any size, and load no less
// than the oracle they pin. It reports whether each per-layer oracle also loads no more than the
// step-pinned layer LRU, which a step that does not fit can break: a schedule is optimal only where
// every step fits.
func checkPinned(runs, pinned []Runs) (string, error) {
	pairs := [][2]int{{layerOracle, pinnedLayerOracle}, {sharedOracle, pinnedSharedOracle}}
	above := 0
	for s := range pinned[0].Runs {
		for _, p := range pairs {
			x := pinned[p[1]].Runs[s]
			if n := x.Online.Prefill.StepEvictions + x.Online.Decode.StepEvictions; n != 0 {
				return "", fmt.Errorf("%d experts per layer: %s has %d step evictions", x.ExpertsPerLayer,
					pinned[p[1]].Policy, n)
			}
			if free, got := allLoaded(runs[p[0]].Runs[s].Online), allLoaded(x.Online); free > got {
				return "", fmt.Errorf("%d experts per layer: %s loads %d bytes, %s %d", x.ExpertsPerLayer,
					runs[p[0]].Policy, free, pinned[p[1]].Policy, got)
			}
		}
		if allLoaded(pinned[pinnedLayerOracle].Runs[s].Online) > allLoaded(runs[layerLRUPinned].Runs[s].Online) {
			above++
		}
	}
	return fmt.Sprintf("the pinned per-layer oracle loads more than the step-pinned layer LRU at %d of %d sizes",
		above, len(pinned[0].Runs)), nil
}

// referenceTests are a workload's test suffixes for the entries the pinned reference may hold.
func referenceTests(w Workload) map[string][]locality.Result {
	out := map[string][]locality.Result{}
	for _, i := range []int{sharedLRU, layerLRU, lfu, arc, layerLRUPinned, lfuPinned, arcPinned} {
		if i < len(w.Policies) {
			out[w.Policies[i].Policy] = tests(w.Policies[i].Runs)
		}
	}
	if len(w.LastN.Runs) > 0 {
		out[lastNTuned] = tuned(w.LastN.Runs, w.LastN.N, func(s int) int { return w.LastN.Own[s] })
	}
	return out
}

// checkPinnedRefs fails if a reference has step evictions in a test suffix of the model, or is
// unknown: a reference that evicts within a step is what the pinned gate exists to leave out.
func checkPinnedRefs(results []result, model string, names []string) error {
	for _, r := range results {
		if r.model != model {
			continue
		}
		refs := referenceTests(r.w)
		for _, n := range append([]string{"shared LRU"}, names...) {
			tests, ok := refs[n]
			if !ok {
				return fmt.Errorf("%s: no reference %q", r.workload, n)
			}
			for _, t := range tests {
				if e := t.Decode.StepEvictions + t.Prefill.StepEvictions; e != 0 {
					return fmt.Errorf("%s: reference %q has %d step evictions", r.workload, n, e)
				}
			}
		}
	}
	return nil
}

// pinnedRefGate is Stage B's gate against the shared LRU and pinnedRefNames only. g is the Stage A
// gate with Stage E1's policies, where the step-pinned ones are judged.
func pinnedRefGate(results []result, model string, sizes []int, links []string, g Gate) Gate {
	a := g
	a.Policies = slices.DeleteFunc(slices.Clone(g.Policies), func(e GateEntry) bool {
		return !slices.Contains(pinnedRefNames, e.Policy)
	})
	return stageBGate(results, model, sizes, links, a)
}

const pinnedGateIntro = "The rule of the Stage B gate above, with one change, fixed before this replay ran: the " +
	"reference holds only policies with no step evictions in any test suffix. These are the shared LRU, the " +
	"step-pinned layer LRU, LFU and ARC, and last-N on the shared LRU with n tuned on the own prefix. LFU with a " +
	"tuned half-life and the allocated layer budgets are left out, since they have no pinned twin, so this " +
	"reference is weaker than the recorded one and a predictor passes it more easily. The recorded gate stands; " +
	"this is a check on whether its verdict depends on references that evict within a step.\n"

// tally adds a phase's counts up over workloads.
type tally struct {
	tokens, accesses, hits int
	loaded                 int64
}

func (t *tally) add(c locality.Counts) {
	t.tokens += c.Tokens
	t.accesses += c.Accesses
	t.hits += c.Hits
	t.loaded += loadedBytes(c)
}

func (t tally) rate() float64 { return float64(t.hits) / float64(max(t.accesses, 1)) }
func (t tally) mb() float64   { return float64(t.loaded) / 1e6 / float64(max(t.tokens, 1)) }

// cell is MB/token and hit rate, the form of the other tables.
func (t tally) cell() string { return fmt.Sprintf("%.0f (%s)", t.mb(), pct(t.rate())) }

// ratio is t's bytes over o's, unrounded, or a dash when o loads nothing.
func (t tally) ratio(o tally) string {
	if o.loaded == 0 {
		return "–"
	}
	return fmt.Sprintf("%.2f", float64(t.loaded)/float64(o.loaded))
}

// bestCausal is the causal policy with the fewest decode bytes in rs together, ties by hit rate. An
// oracle is not causal.
func bestCausal(rs []result, s int) (string, tally) {
	var name string
	var best tally
	for i := range rs[0].w.Policies {
		if rs[0].w.Policies[i].Oracle {
			continue
		}
		var t tally
		for _, r := range rs {
			t.add(r.w.Policies[i].Runs[s].Split.Test.Decode)
		}
		if name == "" || t.loaded < best.loaded || (t.loaded == best.loaded && t.hits > best.hits) {
			name, best = rs[0].w.Policies[i].Policy, t
		}
	}
	return name, best
}

// ofModel are the results of a model.
func ofModel(results []result, model string) []result {
	return slices.DeleteFunc(slices.Clone(results), func(r result) bool { return r.model != model })
}

func heldOutDecode(rs []result, pick func(w Workload) Runs, s int) tally {
	var t tally
	for _, r := range rs {
		t.add(pick(r.w).Runs[s].Split.Test.Decode)
	}
	return t
}

// writeBelady is one table of the gap to both oracles on the held-out decode of rs together.
func writeBelady(w io.Writer, rs []result, sizes []int) {
	fmt.Fprintf(w, "| Experts per layer | Best causal | MB/token (hit) | Oracle, shared † | Pinned, shared † | Best / "+
		"oracle | Best / pinned | Oracle, per layer † | Pinned, per layer † |\n|--:|---|--:|--:|--:|--:|--:|--:|--:|\n")
	policy := func(i int) func(Workload) Runs { return func(x Workload) Runs { return x.Policies[i] } }
	pin := func(i int) func(Workload) Runs { return func(x Workload) Runs { return x.PinnedOracles[i] } }
	for s, c := range sizes {
		name, best := bestCausal(rs, s)
		o, p := heldOutDecode(rs, policy(sharedOracle), s), heldOutDecode(rs, pin(pinnedSharedOracle), s)
		lo, lp := heldOutDecode(rs, policy(layerOracle), s), heldOutDecode(rs, pin(pinnedLayerOracle), s)
		fmt.Fprintf(w, "| %d | %s | %s | %s | %s | %s | %s | %s | %s |\n", c, name, best.cell(), o.cell(), p.cell(),
			best.ratio(o), best.ratio(p), lo.cell(), lp.cell())
	}
}

// writeChecklist is the report's Stage E7 sections.
func writeChecklist(w io.Writer, sizes []int, results []result, gatesP map[string]Gate) {
	if len(results) == 0 || len(results[0].w.PinnedOracles) == 0 {
		return
	}
	models := modelsOf(results)
	fmt.Fprintf(w, "\n## Stage E7: Zhang's checklist\n\nWhat [expert-predictor-result.md](expert-predictor-result.md) "+
		"listed as not measured. Every table is from the same replays as the sections above.\n")

	fmt.Fprintf(w, "\n### Prefill's union per ubatch and layer\n\nA step is one ubatch at one layer, and its union is "+
		"the distinct experts it needs. Every ubatch counts, including a prompt's short last one. Median and p95 are "+
		"nearest-rank. Above N is the share of steps with a union above N, which is what a layer's N slots cannot "+
		"hold at once. Per layer above 244 is the lowest to highest share among layers, on full ubatches only.\n\n")
	fmt.Fprintf(w, "| Model | Workload | Ubatch | Steps | Full | Mean | Median | p95 | Max |")
	for _, c := range unionThresholds {
		fmt.Fprintf(w, " Above %d |", c)
	}
	fmt.Fprintf(w, " Per layer above 244 |\n|---|---|--:|--:|--:|--:|--:|--:|--:|%s--:|\n",
		strings.Repeat("--:|", len(unionThresholds)))
	for _, r := range results {
		for _, u := range r.w.Unions {
			fmt.Fprintf(w, "| %s | %s | %d | %d | %d | %.1f | %d | %d | %d |", r.model, r.workload, u.Ubatch, u.Steps,
				u.FullSteps, u.Mean, u.Median, u.P95, u.Max)
			for _, a := range u.Above {
				fmt.Fprintf(w, " %s |", pct(float64(a)/float64(max(u.Steps, 1))))
			}
			fmt.Fprintf(w, " %s–%s |\n", pct(u.LayerAbove244[0]), pct(u.LayerAbove244[1]))
		}
	}

	fmt.Fprintf(w, "\n### The step-pinned oracles\n\nA pinned oracle is Belady's MIN, except that it never evicts or "+
		"drops an expert of the running step while the step's union fits, so where every step fits it is a schedule "+
		"a server could run. The plain oracle bounds such a schedule's bytes from below and the pinned one from "+
		"above. With equal experts a unit test holds the pinned one to the optimum; a shared budget over layers of "+
		"unequal sizes has no such test, so there the two only bracket it. A step that does not fit has no atomic "+
		"schedule, and what does not fit is used once. So in prefill, where unions pass the budget, the pinned "+
		"columns are not an atomic schedule either.\n\nWhole online replays, all workloads together, MB/token and "+
		"step evictions:\n")
	for _, m := range models {
		rs := ofModel(results, m)
		fmt.Fprintf(w, "\n%s:\n\n| Experts per layer | Decode: oracle, shared † | Pinned † | Prefill: oracle, shared "+
			"† | Pinned † | Oracle's step evictions, decode | Prefill | Pinned's, both phases |\n"+
			"|--:|--:|--:|--:|--:|--:|--:|--:|\n", m)
		for s, c := range sizes {
			var od, pd, op, pp tally
			var osd, osp, psb int
			for _, r := range rs {
				o, p := r.w.Policies[sharedOracle].Runs[s].Online, r.w.PinnedOracles[pinnedSharedOracle].Runs[s].Online
				od.add(o.Decode)
				pd.add(p.Decode)
				op.add(o.Prefill)
				pp.add(p.Prefill)
				osd += o.Decode.StepEvictions
				osp += o.Prefill.StepEvictions
				psb += p.Decode.StepEvictions + p.Prefill.StepEvictions
			}
			fmt.Fprintf(w, "| %d | %s | %s | %s | %s | %d | %d | %d |\n", c, od.cell(), pd.cell(), op.cell(), pp.cell(),
				osd, osp, psb)
		}
	}

	fmt.Fprintf(w, "\n### The Belady gap on held-out decode, at every budget\n\nThe gap to both oracles on the test "+
		"suffixes, warm from the training prefix. An oracle is built per workload from its whole trace, so the "+
		"pooled table sums per-workload oracles: it is not one oracle over the joined traces. Best causal is the "+
		"cache policy with the fewest bytes among those of this report, and in the pooled table it is one policy "+
		"over all workloads. Ratios are from unrounded bytes.\n")
	for _, m := range models {
		rs := ofModel(results, m)
		fmt.Fprintf(w, "\n%s, all test suffixes together:\n\n", m)
		writeBelady(w, rs, sizes)
		for _, r := range rs {
			fmt.Fprintf(w, "\n%s, %s:\n\n", m, r.workload)
			writeBelady(w, []result{r}, sizes)
		}
	}

	fmt.Fprintf(w, "\n### The predictors' hit rates, bytes and step evictions\n\nWhat the gate tables judge and do not "+
		"print, at each predictor's k tuned for that link on the own prefix. Hit rate and MB/token are the test "+
		"suffixes' decode together, and MB/token counts every byte loaded, prefetched or missed, used or not. Step "+
		"evictions are the whole online replays', decode and prefill. The shared LRU the predictors wrap is beside "+
		"them.\n")
	for _, m := range models {
		rs := ofModel(results, m)
		var links []string
		for _, l := range rs[0].t.timing.Links {
			links = append(links, l.Name)
		}
		for _, kind := range predictorKinds {
			for hi, h := range horizons {
				if slices.ContainsFunc(rs, func(r result) bool { return kind.pick(r.w, hi) == nil }) {
					continue
				}
				fmt.Fprintf(w, "\n%s, %s, %s:\n\n| Experts per layer | Shared LRU |", m, kind.name, h)
				for _, l := range links {
					fmt.Fprintf(w, " %s: MB/token (hit) | Step evictions |", l)
				}
				fmt.Fprintf(w, "\n|--:|--:|%s\n", strings.Repeat("--:|--:|", len(links)))
				for s, c := range sizes {
					ref := heldOutDecode(rs, func(x Workload) Runs { return x.Policies[sharedLRU] }, s)
					fmt.Fprintf(w, "| %d | %s |", c, ref.cell())
					for i := range links {
						var t tally
						se := 0
						for _, r := range rs {
							p := kind.pick(r.w, hi)
							run := p.Runs[slices.Index(p.K, p.OwnK[s][i])].Runs[s]
							t.add(run.Split.Test.Decode)
							se += run.Online.Decode.StepEvictions + run.Online.Prefill.StepEvictions
						}
						fmt.Fprintf(w, " %s | %d |", t.cell(), se)
					}
					fmt.Fprintln(w)
				}
			}
		}
	}

	fmt.Fprintf(w, "\n### Lifetime hit rate\n\nHits over accesses in prefill and decode together, whole online replays, "+
		"all workloads together. A prefill step accesses each expert of its union once, so these rates weigh a "+
		"ubatch's experts like a decoded token's, and they say little about either phase alone.\n")
	for _, m := range models {
		rs := ofModel(results, m)
		var rows []Runs
		rows = append(rows, rs[0].w.Policies...)
		rows = append(rows, rs[0].w.PinnedOracles...)
		fmt.Fprintf(w, "\n%s:\n\n| Experts per layer |", m)
		for _, r := range rows {
			fmt.Fprintf(w, " %s |", label(r))
		}
		fmt.Fprintf(w, "\n|--:|%s\n", strings.Repeat("--:|", len(rows)))
		for s, c := range sizes {
			fmt.Fprintf(w, "| %d |", c)
			for i := range rows {
				var t tally
				for _, r := range rs {
					x := r.w.PinnedOracles
					var run Run
					if i < len(r.w.Policies) {
						run = r.w.Policies[i].Runs[s]
					} else {
						run = x[i-len(r.w.Policies)].Runs[s]
					}
					t.add(run.Online.Prefill)
					t.add(run.Online.Decode)
				}
				fmt.Fprintf(w, " %s |", pct(t.rate()))
			}
			fmt.Fprintln(w)
		}
	}
	writeGate(w, "Stage E7: Stage B gate, step-pinned references", pinnedGateIntro, gatesP, models)
}

// retainedCount counts a gate's retained policies.
func retainedCount(g Gate) string {
	n := 0
	for _, e := range g.Policies {
		if e.Retained {
			n++
		}
	}
	return strconv.Itoa(n) + " of " + strconv.Itoa(len(g.Policies))
}
