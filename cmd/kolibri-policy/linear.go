package main

// The linear predictor (PLAN Phase 10, Stage B, item 2, internal/locality/linear.go) on the
// workloads: fitted on the training prefix before its last answer's decoded tokens, its L2 penalty,
// epochs and temperature chosen on those decoded tokens, then frozen; scored, replayed and gated
// on the test suffix like the Markov predictor.

import (
	"fmt"
	"io"
	"slices"
	"strings"

	"kolibri-llm/internal/locality"
)

var (
	linearL2s           = []float64{1e-3, 1e-2, 1e-1}
	linearEpochs        = 6
	linearRate          = 0.2
	linearSeed   uint64 = 1
)

// linearAblations are the own-prefix fits of a horizon, the first the one calibrated, replayed and
// gated.
var linearAblations = []struct {
	name   string
	blocks locality.LinearBlocks
}{
	{"A and B", locality.BlockA | locality.BlockB},
	{"A only", locality.BlockA},
	{"B only", locality.BlockB},
}

// lowoFitName is the name of the fit leaving one workload out.
const lowoFitName = "A and B, leave one workload out"

// LinearFit is a linear predictor fitted one way: its configuration as chosen on the validation
// tokens, their recall, the test suffix's recall as replayed, and, for the calibrated fit, its
// temperature.
type LinearFit struct {
	Name        string                `json:"name"`
	Config      locality.LinearConfig `json:"config"`
	Valid       []locality.Recall     `json:"validation_recall"`
	Test        []locality.Recall     `json:"test_recall"`
	Temperature float64               `json:"temperature,omitempty"`
}

// LinearGrid is the linear predictor of one horizon on a workload: its fits (the own prefix's
// with both blocks, with either alone, and leaving one workload out), the first fit's calibration
// on the test suffix before and after its temperature, and its prefetches.
type LinearGrid struct {
	Horizon    string               `json:"horizon"`
	Fits       []LinearFit          `json:"fits"`
	Raw        locality.Calibration `json:"test_calibration_raw"`
	Calibrated locality.Calibration `json:"test_calibration"`
	Prefetches
}

// decodedIn is the tokens of the decoded spans of spans[from:to].
func decodedIn(spans []locality.Span, from, to int) []int {
	var out []int
	for _, s := range spans[from:to] {
		if s.Phase == locality.Decode {
			out = append(out, s.First)
		}
	}
	return out
}

// linearData are a workload's targets: every token before its training prefix's last answer's
// decoded tokens to fit on, those decoded tokens to validate on, and its test suffix's decoded
// tokens to calibrate on.
func linearData(t trace) (fit, valid, test locality.LinearTrace) {
	v := t.cut
	for v > 0 && t.spans[v-1].Phase == locality.Decode {
		v--
	}
	fit = locality.LinearTrace{Sel: t.sel}
	for tok := range t.spans[v].First {
		fit.Targets = append(fit.Targets, tok)
	}
	return fit, locality.LinearTrace{Sel: t.sel, Targets: decodedIn(t.spans, v, t.cut)},
		locality.LinearTrace{Sel: t.sel, Targets: decodedIn(t.spans, t.cut, len(t.spans))}
}

// pickFit is the L2 index and epoch whose validation score is highest, scores[i][e-1] for L2 i and
// epoch e, the smaller L2 and then the fewer epochs of equal ones.
func pickFit(scores [][]int) (int, int) {
	bi, be := 0, 1
	for i, row := range scores {
		for e, v := range row {
			if v > scores[bi][be-1] {
				bi, be = i, e+1
			}
		}
	}
	return bi, be
}

// fitBest fits a linear predictor of horizon h with blocks on fit at every L2 of linearL2s and
// keeps, of every epoch up to linearEpochs, the fit whose recall on valid covers the most needed
// experts summed over markovKs.
func fitBest(fit, valid []locality.LinearTrace, nExpert int, h locality.Horizon,
	blocks locality.LinearBlocks) (*locality.Linear, LinearFit) {
	scores := make([][]int, len(linearL2s))
	models := make([][]*locality.Linear, len(linearL2s))
	recalls := make([][][]locality.Recall, len(linearL2s))
	for i, l2 := range linearL2s {
		cfg := locality.LinearConfig{Blocks: blocks, L2: l2, Rate: linearRate, Epochs: linearEpochs, Seed: linearSeed}
		locality.FitLinear(fit, nExpert, h, cfg, func(_ int, m *locality.Linear) {
			sums := make([]locality.Recall, len(markovKs))
			for _, v := range valid {
				for k, x := range m.Recall(v, markovKs) {
					sums[k] = sumRecall([]locality.Recall{sums[k], x})
				}
			}
			scores[i] = append(scores[i], covered(sums))
			models[i] = append(models[i], m.Clone())
			recalls[i] = append(recalls[i], sums)
		})
	}
	i, e := pickFit(scores)
	return models[i][e-1], LinearFit{Config: locality.LinearConfig{Blocks: blocks, L2: linearL2s[i],
		Rate: linearRate, Epochs: e, Seed: linearSeed}, Valid: recalls[i][e-1]}
}

// testRecall is m's recall on t's test suffix as replayed, at every k of markovKs.
func testRecall(t trace, m *locality.Linear, h locality.Horizon) []locality.Recall {
	r := locality.RankWith(m.Online(), t.sel, t.spans, t.m.NExpert, h, slices.Max(markovKs))
	return r.Recall(t.sel, t.spans, t.cut, len(t.spans), markovKs)
}

// linearGrid fits the linear predictor of horizon h on the own prefix every way, calibrates the
// first fit and replays its prefetches; it returns the first fit's model too.
func linearGrid(t trace, h locality.Horizon, sizes []int, cost float64) (LinearGrid, *locality.Linear, error) {
	fit, valid, test := linearData(t)
	g := LinearGrid{Horizon: h.String()}
	var own *locality.Linear
	for i, a := range linearAblations {
		m, f := fitBest([]locality.LinearTrace{fit}, []locality.LinearTrace{valid}, t.m.NExpert, h, a.blocks)
		f.Name = a.name
		f.Test = testRecall(t, m, h)
		if i == 0 {
			own = m
			f.Temperature = m.Temperature(valid)
			g.Raw = m.Calibration(test, 1, markovKs, slices.Max(markovKs))
			g.Calibrated = m.Calibration(test, f.Temperature, markovKs, slices.Max(markovKs))
		}
		g.Fits = append(g.Fits, f)
	}
	ranking := locality.RankWith(own.Online(), t.sel, t.spans, t.m.NExpert, h, slices.Max(markovKs))
	var err error
	g.Prefetches, err = replayPrefetches(t, fmt.Sprintf("%s linear", h), ranking, sizes, cost)
	return g, own, err
}

// lowoLinearData are the fit and validation targets of the other workloads of model than wl.
func lowoLinearData(results []result, model, wl string) (fit, valid []locality.LinearTrace) {
	for _, r := range results {
		if r.model == model && r.workload != wl {
			f, v, _ := linearData(r.t)
			fit, valid = append(fit, f), append(valid, v)
		}
	}
	return fit, valid
}

// lowoLinear fits horizon hi's linear predictor with both blocks for results[i] on the other
// workloads of its model, and scores it on its test suffix; ok is false if there are none.
func lowoLinear(results []result, i, hi int) (LinearFit, bool) {
	r := results[i]
	fit, valid := lowoLinearData(results, r.model, r.workload)
	if len(fit) == 0 {
		return LinearFit{}, false
	}
	h := horizons[hi]
	m, f := fitBest(fit, valid, r.t.m.NExpert, h, linearAblations[0].blocks)
	f.Name = lowoFitName
	f.Test = testRecall(r.t, m, h)
	return f, true
}

// fitOf is the fit of g named name, nil if it has none.
func fitOf(g LinearGrid, name string) *LinearFit {
	for i := range g.Fits {
		if g.Fits[i].Name == name {
			return &g.Fits[i]
		}
	}
	return nil
}

// fitCell is a fit's choice.
func fitCell(f *LinearFit) string {
	if f == nil {
		return "–"
	}
	s := fmt.Sprintf("L2 %g, %d epochs", f.Config.L2, f.Config.Epochs)
	if f.Temperature > 0 {
		s += fmt.Sprintf(", T %.2f", f.Temperature)
	}
	return s
}

// writeLinear reports the linear predictor: its fits, recall, calibration, cost and held-out
// replays.
func writeLinear(w io.Writer, a Assumptions, links []locality.Link, sizes []int, d int, results []result,
	gatesB map[string]Gate) {
	if len(results) == 0 || len(results[0].w.Linear) == 0 {
		return
	}
	layers, nExpert := results[0].t.sel.Layers, results[0].t.m.NExpert
	timing := results[0].t.timing
	var l2s []string
	for _, x := range linearL2s {
		l2s = append(l2s, fmt.Sprintf("%g", x))
	}
	fmt.Fprintf(w, "\n## Stage B: a trained linear predictor\n\n")
	fmt.Fprintf(w, "The linear predictor is a logistic regression per layer from two multi-hot blocks of known "+
		"experts to the target step's %d experts: a weight per input and output expert and a bias per output, "+
		"fitted by AdaGrad (step %g) on the binary cross-entropy of all outputs (`internal/locality/linear.go`). "+
		"Its targets are the Markov predictor's horizons:\n\n", nExpert, linearRate)
	fmt.Fprintf(w, "- **next token:** block A is token t's set in the layer, block B token t−1's, for token t+1;\n")
	fmt.Fprintf(w, "- **same token:** block A is the token's set two layers below the target, block B the previous "+
		"token's set in the target layer.\n\n")
	fmt.Fprintf(w, "Its inputs are recorded expert sets only: no router logits or hidden states are recorded, so "+
		"none are used. It is fitted on every token of the training prefix before the decoded tokens of its last "+
		"answer, the prompts' recorded sets included. Its L2 penalty (%s) and epochs (1 to %d) are chosen on those "+
		"decoded tokens, as the most needed experts covered summed over k = %s, the smaller penalty and then the "+
		"fewer epochs of equal ones; then it is frozen. Replayed, it knows only decoded steps that have run, and a "+
		"block B whose token was not decoded is empty. **Leaving one workload out**, it is fitted and chosen on "+
		"the other workloads' training prefixes of the same model. The temperature T of the own fit with both "+
		"blocks, p = sigmoid(z/T), is fitted on the validation tokens by cross-entropy. Choices:\n\n",
		strings.Join(l2s, ", "), linearEpochs, joinInts(markovKs))
	fmt.Fprintf(w, "| Model | Workload | Next token, own | Next token, leave one out | Same token, own | "+
		"Same token, leave one out |\n|---|---|---|---|---|---|\n")
	for _, r := range results {
		n, s := r.w.Linear[0], r.w.Linear[1]
		fmt.Fprintf(w, "| %s | %s | %s | %s | %s | %s |\n", r.model, r.workload, fitCell(&n.Fits[0]),
			fitCell(fitOf(n, lowoFitName)), fitCell(&s.Fits[0]), fitCell(fitOf(s, lowoFitName)))
	}
	fmt.Fprintf(w, "\nRecall of the test suffixes' decoded experts per layer and token as replayed, all workloads "+
		"together: the needed experts covered (whole Top-6 sets covered), counted as for the Markov predictor, "+
		"whose own-prefix row is repeated for comparison.\n")
	for _, m := range modelsOf(results) {
		for hi, h := range horizons {
			fmt.Fprintf(w, "\n%s, %s:\n\n| Predictor |", m, h)
			for _, k := range markovKs {
				fmt.Fprintf(w, " k %d |", k)
			}
			fmt.Fprintf(w, "\n|---|%s\n", strings.Repeat("--:|", len(markovKs)))
			rows := []struct {
				name string
				pick func(r result) []locality.Recall
			}{
				{"linear, A and B, own prefix", func(r result) []locality.Recall { return r.w.Linear[hi].Fits[0].Test }},
				{"linear, A only", func(r result) []locality.Recall { return fitOf(r.w.Linear[hi], "A only").Test }},
				{"linear, B only", func(r result) []locality.Recall { return fitOf(r.w.Linear[hi], "B only").Test }},
				{"linear, A and B, leave one workload out", func(r result) []locality.Recall {
					if f := fitOf(r.w.Linear[hi], lowoFitName); f != nil {
						return f.Test
					}
					return nil
				}},
				{"Markov, own prefix", func(r result) []locality.Recall {
					g := r.w.Markov[hi]
					return g.Test[g.Own]
				}},
			}
			for _, row := range rows {
				sums := make([]locality.Recall, len(markovKs))
				ok := true
				for _, r := range results {
					if r.model != m {
						continue
					}
					x := row.pick(r)
					if x == nil {
						ok = false
						break
					}
					for k := range markovKs {
						sums[k] = sumRecall([]locality.Recall{sums[k], x[k]})
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
			}
		}
	}
	fmt.Fprintf(w, "\n**Calibration** of the own fit with both blocks on the test suffixes' decoded tokens, from "+
		"their recorded inputs, all workloads together: the experts its best k are expected to cover, the sum of "+
		"their probabilities, against those they cover, before the temperature (raw) and after it (calibrated); "+
		"and the expected calibration error over its best %d per target in ten bins of probability, the "+
		"workloads' weighted by their candidates:\n\n| Model | Horizon | Temperatures |", slices.Max(markovKs))
	for _, k := range markovKs {
		fmt.Fprintf(w, " k %d covered | expected, raw | expected, calibrated |", k)
	}
	fmt.Fprintf(w, " ECE, raw | ECE, calibrated |\n|---|---|---|%s--:|--:|\n", strings.Repeat("--:|--:|--:|",
		len(markovKs)))
	for _, m := range modelsOf(results) {
		for hi, h := range horizons {
			cov := make([]int, len(markovKs))
			raw, cal := make([]float64, len(markovKs)), make([]float64, len(markovKs))
			var eRaw, eCal float64
			var cands int
			var temps []string
			for _, r := range results {
				if r.model != m {
					continue
				}
				g := r.w.Linear[hi]
				temps = append(temps, fmt.Sprintf("%.2f", g.Fits[0].Temperature))
				for k := range markovKs {
					cov[k] += g.Raw.Covered[k]
					raw[k] += g.Raw.Expected[k]
					cal[k] += g.Calibrated.Expected[k]
				}
				eRaw += g.Raw.ECE * float64(g.Raw.Candidates)
				eCal += g.Calibrated.ECE * float64(g.Calibrated.Candidates)
				cands += g.Raw.Candidates
			}
			fmt.Fprintf(w, "| %s | %s | %s |", m, h, strings.Join(temps, ", "))
			for k := range markovKs {
				fmt.Fprintf(w, " %d | %.0f | %.0f |", cov[k], raw[k], cal[k])
			}
			fmt.Fprintf(w, " %.3f | %.3f |\n", eRaw/float64(max(cands, 1)), eCal/float64(max(cands, 1)))
		}
	}
	fmt.Fprintf(w, "\n**Cost.** A prediction takes %.1f µs per decoded layer step on this Mac (`-linear-us`, from "+
		"`BenchmarkLinearStep` in `internal/locality`: 50 layers of 384 experts, 48 candidates, the slowest of six "+
		"runs on an Apple M5 Pro): %.2f ms of one CPU core per decoded token, %.1f%% of the %.1f ms the resident "+
		"model computes. Its weights take %.0f MB for %d layers of %d experts (`locality.LinearBytes`). It is "+
		"replayed with that cost and without, as the Markov predictor is.\n", 1e6*a.LinearSeconds,
		1e3*a.LinearSeconds*float64(layers), 100*a.LinearSeconds/timing.DecodeStep,
		1e3*timing.DecodeStep*float64(layers), float64(locality.LinearBytes(layers, nExpert))/1e6, layers, nExpert)
	fmt.Fprintf(w, "\nThe own fit with both blocks prefetches its k best candidates before every step into a shared "+
		"LRU of the budget, k chosen per budget and link from %s on the own training prefix by its decode time with "+
		"the cost.\n", joinInts(markovKs))
	writePrefetches(w, "Linear", links, sizes, d, results, gatesB, func(w Workload, hi int) Prefetches {
		return w.Linear[hi].Prefetches
	})
}
