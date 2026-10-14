package main

import (
	"fmt"
	"slices"

	"kolibri-llm/internal/locality"
)

// SplitManifest is testdata/locality/splits.json: the rule and each workload's split.
type SplitManifest struct {
	Rule      string                   `json:"rule"`
	Workloads map[string]SplitWorkload `json:"workloads"`
}

// SplitWorkload is a workload's declared split: the first TrainAnswers of its Answers answers are
// training, the test suffix starts at FirstTestToken.
type SplitWorkload struct {
	Answers        int `json:"answers"`
	TrainAnswers   int `json:"train_answers"`
	FirstTestToken int `json:"first_test_token"`
}

// split checks a declared split against the rule (the first ⌊n/2⌋ answers train) and the trace,
// and returns the index of the first test span.
func split(d SplitWorkload, spans []locality.Span, answers []locality.Answer) (int, error) {
	if d.Answers != len(answers) || d.TrainAnswers != len(answers)/2 {
		return 0, fmt.Errorf("%d of %d answers train, the trace has %d answers and the rule trains on %d",
			d.TrainAnswers, d.Answers, len(answers), len(answers)/2)
	}
	cut, err := locality.SplitAfter(spans, answers, d.TrainAnswers)
	if err != nil {
		return 0, err
	}
	if spans[cut].First != d.FirstTestToken {
		return 0, fmt.Errorf("the test suffix starts at token %d, declared %d", spans[cut].First, d.FirstTestToken)
	}
	return cut, nil
}

// Split is a replay's training prefix and its test suffix, which runs on warm from the prefix.
type Split struct {
	Train locality.Result `json:"train"`
	Test  locality.Result `json:"test"`
}

// halfLives are the LFU half-lives tuning chooses from, in tokens, with the declared default.
func halfLives(def int) []int {
	hs := []int{64, 128, 256, 512, 1024, 2048, 4096, 8192}
	if !slices.Contains(hs, def) {
		hs = append(hs, def)
		slices.Sort(hs)
	}
	return hs
}

// Grid is a workload's LFU decode miss bytes per half-life and size, in its training prefix and in
// its test suffix.
type Grid struct {
	HalfLives []int     `json:"half_lives"`
	Train     [][]int64 `json:"train_decode_miss_bytes"`
	Test      [][]int64 `json:"test_decode_miss_bytes"`
}

// lfuGrid replays LFU at every half-life and size, untimed.
func lfuGrid(sel locality.Selections, spans []locality.Span, cut int, m Model, sizes, hs []int) (Grid, error) {
	bases := make([]baseline, len(hs))
	for i, h := range hs {
		bases[i] = baseline{fmt.Sprintf("LFU %d", h), false, func(c int) (locality.Policy, error) {
			return locality.NewLFU(m.NExpert, m.bytes(), m.budget(c), h), nil
		}}
	}
	runs, err := replayAll(sel, spans, cut, m, sizes, bases, locality.Timing{})
	if err != nil {
		return Grid{}, err
	}
	g := Grid{HalfLives: hs, Train: make([][]int64, len(hs)), Test: make([][]int64, len(hs))}
	for i := range hs {
		for _, r := range runs[i].Runs {
			g.Train[i] = append(g.Train[i], r.Split.Train.Decode.MissBytes)
			g.Test[i] = append(g.Test[i], r.Split.Test.Decode.MissBytes)
		}
	}
	return g, nil
}

// Choice is a half-life and the decode miss bytes it gives the test suffix.
type Choice struct {
	HalfLife  int   `json:"half_life"`
	TestBytes int64 `json:"test_decode_miss_bytes"`
}

// Tuning is LFU's half-life for one workload at one size: the untuned default and the half-life
// tuned on the workload's own training prefix, on the other workloads' training prefixes (leave
// one workload out), on the other model's training prefix of the workload, and in hindsight on
// the test suffix itself. Only the last sees test data: it is the bound of tuning, not a result.
// The other model's choice needs exactly one other model: more are never pooled into one.
type Tuning struct {
	ExpertsPerLayer int     `json:"experts_per_layer"`
	Default         Choice  `json:"default"`
	Own             Choice  `json:"own_prefix"`
	LOWO            *Choice `json:"leave_one_workload_out,omitempty"`
	OtherModel      *Choice `json:"other_model,omitempty"`
	Hindsight       Choice  `json:"hindsight_on_test"`
}

// argmin is the index of the least of v, the first of equal ones.
func argmin(v []int64) int {
	i := 0
	for j, x := range v {
		if x < v[i] {
			i = j
		}
	}
	return i
}

// tune chooses the half-life for grids[model][wl] at size index s; grids holds every model's
// workloads, all on the same half-lives and sizes. The other model's choice is nil unless grids
// holds exactly two models.
func tune(grids map[string]map[string]Grid, model, wl string, s, size, def int) Tuning {
	own := grids[model][wl]
	pick := func(score func(h int) int64) Choice {
		v := make([]int64, len(own.HalfLives))
		for h := range v {
			v[h] = score(h)
		}
		h := argmin(v)
		return Choice{own.HalfLives[h], own.Test[h][s]}
	}
	// sum is a choice on the training prefixes of the grids keep selects, nil if it selects none.
	sum := func(keep func(m, w string) bool) *Choice {
		var gs []Grid
		for m, ws := range grids {
			for w, g := range ws {
				if keep(m, w) {
					gs = append(gs, g)
				}
			}
		}
		if len(gs) == 0 {
			return nil
		}
		c := pick(func(h int) int64 {
			var b int64
			for _, g := range gs {
				b += g.Train[h][s]
			}
			return b
		})
		return &c
	}
	d := slices.Index(own.HalfLives, def)
	return Tuning{
		ExpertsPerLayer: size,
		Default:         Choice{def, own.Test[d][s]},
		Own:             pick(func(h int) int64 { return own.Train[h][s] }),
		LOWO:            sum(func(m, w string) bool { return m == model && w != wl }),
		OtherModel:      sum(func(m, w string) bool { return len(grids) == 2 && m != model && w == wl }),
		Hindsight:       pick(func(h int) int64 { return own.Test[h][s] }),
	}
}
