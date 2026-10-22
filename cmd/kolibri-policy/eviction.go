package main

// Stage E1 (PLAN Phase 11): upstream llama.cpp eviction policies replayed offline, with a step's
// experts as one event (internal/locality, eviction.go).

import (
	"fmt"
	"io"
	"slices"
	"strconv"
	"strings"

	"kolibri-llm/internal/locality"
)

// stageE1 are the gate's names of Stage E1's policies, in the order gateOf lists them after Stage
// A's. They are judged by Stage A's rule, but they are no Stage A policy: Stage B's reference, the
// best Stage A policy, leaves them out.
var stageE1 = []string{"LFU, step-pinned", "ARC, step-pinned", "layer LRU, step-pinned", "SLRU, per layer",
	"SLRU, shared", "hotness, per layer", "hotness, per layer, half-life tuned on the own prefix"}

// stageAOnly is gate g without Stage E1's policies.
func stageAOnly(g Gate) Gate {
	g.Policies = slices.DeleteFunc(slices.Clone(g.Policies), func(e GateEntry) bool {
		return slices.Contains(stageE1, e.Policy)
	})
	return g
}

// hotnessHalfLives are the hotness half-lives tuning chooses from, in tokens, with the declared
// default: down to 1 token, as Stage B's Markov predictor, and up to LFU's longest.
func hotnessHalfLives(def int) []int {
	hs := []int{1, 4, 16, 64, 256, 512, 1024, 4096, 8192}
	if !slices.Contains(hs, def) {
		hs = append(hs, def)
		slices.Sort(hs)
	}
	return hs
}

// hotnessGrid replays the hotness policy at every half-life and size.
func hotnessGrid(sel locality.Selections, spans []locality.Span, cut int, m Model, sizes, hs []int,
	timing locality.Timing) (Grid, error) {
	return halfLifeGrid(sel, spans, cut, m, sizes, hs, timing, "hotness", func(c, h int) (locality.Policy, error) {
		return locality.NewHotness(sel.Layers, m.NExpert, c, float64(h))
	})
}

// writeE1Policies describes Stage E1's policies in the report's list of policies.
func writeE1Policies(w io.Writer, a Assumptions) {
	fmt.Fprintf(w, "- **step-pinned LFU, ARC and layer LRU:** the policy, except that it never evicts an expert "+
		"the running step has used. LFU loads an expert with its count before the step's use, so a later miss of "+
		"the same step can evict it again; ARC's REPLACE may take a list's least recent expert while that list "+
		"holds only the running step's; a layer LRU evicts them when a ubatch needs more experts than a layer "+
		"holds. The step-pinned layer LRU is the per-layer SLRU below without a protected segment (Stage E1);\n")
	fmt.Fprintf(w, "- **SLRU:** the segmented LRU of llama.cpp's `--moe-cache-mib` (PR #29887): a loaded expert "+
		"enters probation, a hit there promotes it to protected, which holds at most %g of the budget and demotes "+
		"its least recent expert to probation when full; victims come from probation's least recent end, from "+
		"protected's only when probation has none. Per layer, of that many experts in every layer, and shared, "+
		"of the whole budget for all layers, as upstream shares one per device (Stage E1);\n", a.SLRUProtected)
	fmt.Fprintf(w, "- **hotness:** the decayed hotness of llama.cpp's `--moe-stream` (PR #25294): every expert "+
		"scores 1 per step that uses it, resident or not, halved every %d tokens (a declared default, also tuned "+
		"below); a load evicts the layer's resident expert of the lowest score, the least recent of equal ones. "+
		"Per layer, as upstream keeps a slot pool per layer (Stage E1);\n", a.HotnessHalfLife)
}

// writeE1Note states how Stage E1's policies treat a step, after the list of policies.
func writeE1Note(w io.Writer) {
	fmt.Fprintf(w, "Stage E1's policies never evict an expert the running step has used, as a server that runs a "+
		"ubatch's or a token's experts of a layer together cannot; an expert that does not fit beside them is "+
		"used once and not kept. **Step evictions** count the evictions of experts the running step has already "+
		"used: a replay that counts them treats a step's experts as separate accesses (Zhang, arXiv 2608.07911).\n\n")
}

// e1Columns are the policies Stage E1's tables compare, by index into a workload's policies.
var e1Columns = []int{layerLRU, layerLRUPinned, sharedLRU, lfu, lfuPinned, arc, arcPinned, layerSLRU,
	sharedSLRU, hotness}

// writeE1 reports Stage E1: step evictions of every policy, the hotness half-life tuned on training
// prefixes, and the test suffixes' hit rates of the upstream policies beside the LRUs.
func writeE1(w io.Writer, a Assumptions, sizes []int, d int, results []result) {
	if len(results) == 0 {
		return
	}
	models := modelsOf(results)
	fmt.Fprintf(w, "\n## Stage E1: upstream eviction policies, a step as one event\n\n")
	fmt.Fprintf(w, "A server runs a decoded token's or a ubatch's experts of a layer together and never evicts "+
		"one it is running; llama.cpp's expert cache pins them. A replay that takes a step's experts one access at "+
		"a time may evict an expert the same step has already used to load the next one, a miss no server has. "+
		"Step evictions, whole online replays of all workloads together:\n")
	for _, m := range models {
		fmt.Fprintf(w, "\n%s:\n\n| Policy | Decode at %d | Prefill at %d | Decode, all sizes | Prefill, all sizes |\n"+
			"|---|--:|--:|--:|--:|\n", m, sizes[d], sizes[d])
		for i, pr := range results[0].w.Policies {
			var cells [4]int
			for _, r := range results {
				if r.model != m {
					continue
				}
				for s, run := range r.w.Policies[i].Runs {
					on := run.Online
					if s == d {
						cells[0] += on.Decode.StepEvictions
						cells[1] += on.Prefill.StepEvictions
					}
					cells[2] += on.Decode.StepEvictions
					cells[3] += on.Prefill.StepEvictions
				}
			}
			fmt.Fprintf(w, "| %s | %d | %d | %d | %d |\n", label(pr), cells[0], cells[1], cells[2], cells[3])
		}
	}
	var hs []string
	for _, h := range a.HotnessHalfLives {
		hs = append(hs, strconv.Itoa(h))
	}
	fmt.Fprintf(w, "\n**Hotness half-life.** At every budget it is chosen from %s tokens as LFU's is: the fewest "+
		"decode miss bytes in a training prefix, the first of equal ones, scored on the test suffix; on the own "+
		"prefix, leaving one workload out, on the other model's prefix, and in hindsight † on the test suffix "+
		"itself. Untuned is the declared %d.\n", strings.Join(hs, ", "), a.HotnessHalfLife)
	header := "| Untuned | Own prefix | Leave one workload out | Other model | Hindsight † |"
	for _, m := range models {
		tuningTotals(w, m, header, sizes, results, func(w Workload) []Tuning { return w.HotnessTuning })
	}
	fmt.Fprintf(w, "\n**Held out.** All workloads' test suffixes together, MB/token (hit rate); hotness tuned "+
		"with the half-life chosen on the own prefix:\n")
	for _, m := range models {
		for _, ph := range []locality.Phase{locality.Decode, locality.Prefill} {
			fmt.Fprintf(w, "\n%s, %s:\n\n| Experts per layer |", m, ph)
			for _, i := range e1Columns {
				fmt.Fprintf(w, " %s |", results[0].w.Policies[i].Policy)
			}
			fmt.Fprintf(w, " hotness, tuned |\n|--:|%s\n", strings.Repeat("--:|", len(e1Columns)+1))
			for s, c := range sizes {
				fmt.Fprintf(w, "| %d |", c)
				cols := make([]locality.Counts, len(e1Columns)+1)
				for _, r := range results {
					if r.model != m {
						continue
					}
					for j, i := range e1Columns {
						addCounts(&cols[j], phase(r.w.Policies[i].Runs[s].Split.Test, ph))
					}
					if g := r.w.HotnessGrid; len(g.runs) > 0 && len(r.w.HotnessTuning) > s {
						h := slices.Index(g.HalfLives, r.w.HotnessTuning[s].Own.HalfLife)
						addCounts(&cols[len(e1Columns)], phase(g.runs[h].Runs[s].Split.Test, ph))
					}
				}
				for _, x := range cols {
					fmt.Fprintf(w, " %s (%s) |", perToken(x), pct(rate(x)))
				}
				fmt.Fprintln(w)
			}
		}
	}
}

// addCounts adds the counts the Stage E1 tables show.
func addCounts(c *locality.Counts, o locality.Counts) {
	c.Tokens += o.Tokens
	c.Accesses += o.Accesses
	c.Hits += o.Hits
	c.MissBytes += o.MissBytes
}
