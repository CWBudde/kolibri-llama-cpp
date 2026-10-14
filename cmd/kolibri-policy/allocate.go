package main

import (
	"fmt"
	"io"
	"slices"
	"strings"

	"kolibri-llm/internal/locality"
)

// curves are a workload's per-layer decode miss curves (locality.MissCurves) in its training
// prefix and in its test suffix.
type curves struct{ train, test [][]int64 }

// Allocation is the layer LRU with capacities per layer fitted one way at every size, frozen and
// replayed online. A leaky fit has seen the test suffix: it is the bound of fitting, not a result.
type Allocation struct {
	Fit        string  `json:"fit"`
	Leaky      bool    `json:"leaky,omitempty"`
	Capacities [][]int `json:"capacities"`
	Runs       []Run   `json:"runs"`
}

// Allocations are a workload's allocated layer LRUs and the decode miss bytes its test suffix's
// first uses of experts load: compulsory misses, which no cache that only loads what a step needs
// avoids.
type Allocations struct {
	CompulsoryTestBytes int64        `json:"compulsory_test_decode_miss_bytes"`
	Fits                []Allocation `json:"fits"`
}

const (
	ownFit = iota
	lowoFit
	hindsightFit
)

var fitNames = [...]string{ownFit: "own prefix", lowoFit: "leave one workload out", hindsightFit: "hindsight †"}

// sumCurves adds curves up, layer by layer and capacity by capacity; nil for none.
func sumCurves(cs [][][]int64) [][]int64 {
	if len(cs) == 0 {
		return nil
	}
	out := make([][]int64, len(cs[0]))
	for l := range out {
		out[l] = make([]int64, len(cs[0][l]))
		for _, c := range cs {
			for i, b := range c[l] {
				out[l][i] += b
			}
		}
	}
	return out
}

// fits are the curves each allocation of workload wl of model is fitted on: its own training
// prefix; the other workloads' training prefixes of the same model, nil if there are none; and in
// hindsight its test suffix.
func fits(all map[string]map[string]curves, model, wl string) [3][][]int64 {
	var others [][][]int64
	var names []string
	for w := range all[model] {
		if w != wl {
			names = append(names, w)
		}
	}
	slices.Sort(names)
	for _, w := range names {
		others = append(others, all[model][w].train)
	}
	own := all[model][wl]
	return [3][][]int64{ownFit: own.train, lowoFit: sumCurves(others), hindsightFit: own.test}
}

// allocate fits capacities on curves at every size's budget.
func allocate(fit [][]int64, m Model, sizes []int) [][]int {
	out := make([][]int, len(sizes))
	for s, c := range sizes {
		out[s] = locality.Allocate(fit, m.bytes(), m.budget(c))
	}
	return out
}

// replayAllocations fits every allocation and replays it online at every size.
func replayAllocations(t trace, fs [3][][]int64, sizes []int) (Allocations, error) {
	out := Allocations{CompulsoryTestBytes: compulsory(fs[hindsightFit])}
	var bases []baseline
	for f, fit := range fs {
		if fit == nil {
			continue
		}
		caps := allocate(fit, t.m, sizes)
		out.Fits = append(out.Fits, Allocation{Fit: fitNames[f], Leaky: f == hindsightFit, Capacities: caps})
		bases = append(bases, baseline{fitNames[f], false, func(c int) (locality.Policy, error) {
			return locality.NewLayerLRUSizes(t.m.NExpert, caps[at(sizes, c)]), nil
		}})
	}
	runs, err := replayAll(t.sel, t.spans, t.cut, t.m, sizes, bases, t.timing)
	if err != nil {
		return out, err
	}
	for i := range out.Fits {
		out.Fits[i].Runs = runs[i].Runs
	}
	return out, nil
}

// compulsory is the miss bytes of curves at full capacity: the first uses alone.
func compulsory(cs [][]int64) int64 {
	var b int64
	for _, c := range cs {
		b += c[len(c)-1]
	}
	return b
}

// checkCurves fails unless at every size the curves are the uniform layer LRU's per-layer decode
// miss bytes, in the training prefix and in the test suffix.
func checkCurves(c curves, layer Runs, m Model) error {
	for _, r := range layer.Runs {
		for i, seg := range []locality.Result{r.Split.Train, r.Split.Test} {
			cs := [][][]int64{c.train, c.test}[i]
			for l, x := range seg.Decode.Layers {
				if r.ExpertsPerLayer >= len(cs[l]) {
					return fmt.Errorf("%d experts per layer, the curves end at %d", r.ExpertsPerLayer, len(cs[l])-1)
				}
				if got, want := cs[l][r.ExpertsPerLayer], int64(x.Accesses-x.Hits)*m.Layers[l].BytesPerExpert; got != want {
					return fmt.Errorf("%s, layer %d, %d experts: the curve gives %d miss bytes, the layer LRU %d",
						[]string{"training prefix", "test suffix"}[i], l, r.ExpertsPerLayer, got, want)
				}
			}
		}
	}
	return nil
}

// fitRuns are workload r's allocation fitted fit, nil if it has none.
func fitRuns(r result, fit int) *Allocation {
	for i := range r.w.Allocation.Fits {
		if r.w.Allocation.Fits[i].Fit == fitNames[fit] {
			return &r.w.Allocation.Fits[i]
		}
	}
	return nil
}

// writeAllocation reports the allocated layer LRUs on the test suffixes.
func writeAllocation(w io.Writer, sizes []int, d int, layers []int, results []result) {
	fmt.Fprintf(w, "\n## Layer budgets, allocated on training prefixes\n\n")
	fmt.Fprintf(w, "A layer LRU need not give every layer the same number of experts. At every budget, the experts "+
		"per layer are allocated from each layer's miss curve, its decode miss bytes at every capacity in a "+
		"training prefix, replayed online. The allocation grows the layer whose next step saves the most bytes "+
		"per byte, along each curve's lower convex hull, until no step saves anything or fits; budget left over "+
		"is spread level. The capacities are then frozen and the layer LRU replayed online; the cells are its "+
		"decode MB/token in the test suffixes. Fitted on:\n\n")
	fmt.Fprintf(w, "- **own prefix:** the workload's own training prefix;\n")
	fmt.Fprintf(w, "- **leave one workload out:** the other workloads' training prefixes of the same model, their "+
		"curves added up;\n")
	fmt.Fprintf(w, "- **hindsight †:** the test suffix itself. **It is fitted on the test data, the bound of "+
		"fitting and not a result.**\n\n")
	fmt.Fprintf(w, "**Compulsory** are the bytes of a test suffix's first uses of experts, which no cache that "+
		"only loads what a step needs avoids. Uniform is the layer LRU, shared the shared LRU, at the same "+
		"budget.\n")
	header := "| Uniform | Shared | Own prefix | Leave one workload out | Hindsight † | Compulsory |"
	for _, m := range modelsOf(results) {
		fmt.Fprintf(w, "\n%s, all workloads' test suffixes together, decode MB/token:\n\n| Experts per layer %s\n"+
			"|--:|--:|--:|--:|--:|--:|--:|\n", m, header)
		for s, c := range sizes {
			var tokens int
			var sum [6]int64
			have := [3]bool{true, true, true}
			for _, r := range results {
				if r.model != m {
					continue
				}
				tokens += r.w.NoCache.Split.Test.Decode.Tokens
				sum[0] += r.w.Policies[layerLRU].Runs[s].Split.Test.Decode.MissBytes
				sum[1] += r.w.Policies[sharedLRU].Runs[s].Split.Test.Decode.MissBytes
				for f := range have {
					if a := fitRuns(r, f); a != nil {
						sum[2+f] += a.Runs[s].Split.Test.Decode.MissBytes
					} else {
						have[f] = false
					}
				}
				sum[5] += r.w.Allocation.CompulsoryTestBytes
			}
			cells := make([]string, len(sum))
			for i, b := range sum {
				cells[i] = mbPer(b, tokens)
			}
			cells[5] = fmt.Sprintf("%.1f", float64(sum[5])/1e6/float64(max(tokens, 1)))
			for f, ok := range have {
				if !ok {
					cells[2+f] = "–"
				}
			}
			fmt.Fprintf(w, "| %d | %s |\n", c, strings.Join(cells, " | "))
		}
	}
	fmt.Fprintf(w, "\nExperts per layer allocated on the own prefix at a budget of %d per layer:\n\n| Model | Workload |",
		sizes[d])
	for _, l := range layers {
		fmt.Fprintf(w, " Layer %d |", l)
	}
	fmt.Fprintf(w, " Fewest | Most |\n|---|---|%s--:|--:|\n", strings.Repeat("--:|", len(layers)))
	for _, r := range results {
		a := fitRuns(r, ownFit)
		if a == nil {
			continue
		}
		caps := a.Capacities[d]
		fmt.Fprintf(w, "| %s | %s |", r.model, r.workload)
		for _, l := range layers {
			fmt.Fprintf(w, " %d |", caps[l])
		}
		fmt.Fprintf(w, " %d | %d |\n", slices.Min(caps), slices.Max(caps))
	}
}

// modelsOf is the models of results, in order.
func modelsOf(results []result) []string {
	var models []string
	for _, r := range results {
		if !slices.Contains(models, r.model) {
			models = append(models, r.model)
		}
	}
	return models
}
