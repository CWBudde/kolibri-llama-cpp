package main

// Stage E6's hidden-state predictor (internal/locality/hidden.go), replayed from the router's
// top-k that tools/e6/capture_router.py scored on the residual stream: pred-h2, block l+1's router
// on the stream after block l-1, predicts layer l+1 before layer l of a decoded token, the same
// step and lead as the same-token predictors. It is replayed and judged at Stage B's gate like the
// perfect same-token predictor (perfect.go), with a declared cost per decoded layer step
// (-hidden-state-us). pred-h1 (block l's router on the same stream, lead 0) is scored for recall
// only. Off by default (-hidden-state-dir); a model without a capture is not judged. A workload whose
// capture selected other experts than the recorded trace is replayed on the capture's selections,
// every policy with it (captureTrace), as docs/expert-predictor-result.md pre-registers.

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"kolibri-llm/internal/locality"
)

// hiddenRecallKs are the candidate counts the recall of the capture files is reported at.
var hiddenRecallKs = []int{6, 12}

// HiddenGrid is the file-backed predictor at one cost on a workload: its prefetches into the shared
// LRU at every k, with that cost and without, k tuned per size and link on the own training prefix.
type HiddenGrid struct {
	File        string  `json:"file"`
	Horizon     string  `json:"horizon"`
	LeadSteps   int     `json:"lead_steps"`
	CostSeconds float64 `json:"cost_seconds_per_step"`
	Prefetches
}

// HiddenRecall is a capture file's recall of the actual Top-K at every k of K: on the decode of
// the test suffix, on every decoded step and on every token of the trace, each over all layers (the
// layers below Blocks, which no row predicts, counted with nothing covered) and over the predicted
// layers alone. Layer l is predicted by row l-Blocks; Lead is the steps between that prediction,
// made before layer l-Blocks+1, and layer l.
type HiddenRecall struct {
	File                string            `json:"file"`
	Blocks              int               `json:"blocks_ahead"`
	Lead                int               `json:"lead_steps"`
	K                   []int             `json:"k"`
	TestDecode          []locality.Recall `json:"test_decode"`
	Decode              []locality.Recall `json:"decode"`
	Tokens              []locality.Recall `json:"every_token"`
	TestDecodePredicted []locality.Recall `json:"test_decode_predicted_layers"`
	DecodePredicted     []locality.Recall `json:"decode_predicted_layers"`
	TokensPredicted     []locality.Recall `json:"every_token_predicted_layers"`
}

// MappingCheck records the pre-registered mapping check of a workload: the selections themselves,
// fed through the file-backed predictor, ranked as the perfect oracle before every step and
// replayed exactly as it at k K, every size and every cost.
type MappingCheck struct {
	K              int       `json:"k"`
	Sizes          int       `json:"sizes"`
	CostsSeconds   []float64 `json:"costs_seconds_per_step"`
	PredictedSteps int       `json:"predicted_steps"`
}

// predPath is the capture file of horizon h of workload wl and model name in dir.
func predPath(dir, wl, name string, h int) string {
	return filepath.Join(dir, fmt.Sprintf("%s.%s.pred-h%d.npy", wl, name, h))
}

// captureOf reports whether dir holds a capture of model name for every workload: pred-h1, pred-h2
// and the selections the capture made. None of them is no capture; some but not all is an error.
func captureOf(dir, name string, workloads []string) (bool, error) {
	var have, missing []string
	for _, wl := range workloads {
		for _, p := range []string{predPath(dir, wl, name, 1), predPath(dir, wl, name, 2),
			filepath.Join(dir, fmt.Sprintf("%s.%s.experts.npy", wl, name))} {
			if _, err := os.Stat(p); err == nil {
				have = append(have, p)
			} else if errors.Is(err, os.ErrNotExist) {
				missing = append(missing, p)
			} else {
				return false, err
			}
		}
	}
	if len(have) == 0 {
		return false, nil
	}
	if len(missing) > 0 {
		return false, fmt.Errorf("a partial capture of %s in %s: %s missing", name, dir, strings.Join(missing, ", "))
	}
	return true, nil
}

// loadPred reads the capture file of horizon h for the trace of sel, which must have its layers
// and tokens, and checks its rows (locality.CheckPred) and that they are as wide as the widest k
// the replay ranks: locality.HiddenState panics when asked for more than the file holds.
func loadPred(path string, h int, sel locality.Selections, nExpert int) (locality.Selections, error) {
	p, err := locality.ReadNPYFile(path)
	if err != nil {
		return p, err
	}
	if p.Layers != sel.Layers || p.Tokens != sel.Tokens {
		return p, fmt.Errorf("%s: %d layers x %d tokens, the selections %d x %d", path, p.Layers, p.Tokens,
			sel.Layers, sel.Tokens)
	}
	if err := locality.CheckPred(p, h, nExpert); err != nil {
		return p, fmt.Errorf("%s: %w", path, err)
	}
	if need := min(slices.Max(markovKs), nExpert); p.K < need {
		return p, fmt.Errorf("%s: %d experts per row, the replay ranks at least %d (capture_router.py --top-k %d)", path,
			p.K, need, need)
	}
	return p, nil
}

// idsSHA256 is tools/locality/workloads.py's ids_sha256 of token IDs: the SHA-256 of their decimals
// joined by commas.
func idsSHA256(ids []int32) string {
	var b []byte
	for i, id := range ids {
		if i > 0 {
			b = append(b, ',')
		}
		b = strconv.AppendInt(b, int64(id), 10)
	}
	return fmt.Sprintf("%x", sha256.Sum256(b))
}

// captureTokens fails unless the capture in dir was made from ids, the token IDs the replay builds
// workload wl's trace from: manifest.json of tools/e6/capture_router.py must hold their hash as the
// workload's tokens_sha256. Equal selections do not show it, and with the fallback of captureTrace
// they need not be equal: predictions of another token sequence would be read at the wrong tokens.
func captureTokens(dir, wl string, ids []int32) error {
	path := filepath.Join(dir, "manifest.json")
	b, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("the capture's manifest: %w", err)
	}
	var m struct {
		Workloads map[string]struct {
			TokensSHA256 string `json:"tokens_sha256"`
		} `json:"workloads"`
	}
	if err := json.Unmarshal(b, &m); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	w, ok := m.Workloads[wl]
	if !ok || w.TokensSHA256 == "" {
		return fmt.Errorf("%s: no tokens_sha256 of workload %s", path, wl)
	}
	if got := idsSHA256(ids); got != w.TokensSHA256 {
		return fmt.Errorf("%s: workload %s was captured from other token IDs (tokens_sha256 %s) than the replayed ones "+
			"(%s)", path, wl, w.TokensSHA256, got)
	}
	return nil
}

// sameSelections fails unless the capture's selections at path are sel, ID for ID: the predictor
// is never judged on a trace other than the one it was captured on. sel is the trace captureTrace
// chose, so a capture that differs from the recorded selections passes on its own.
func sameSelections(path string, sel locality.Selections) error {
	c, err := locality.ReadNPYFile(path)
	if err != nil {
		return err
	}
	if c.Layers != sel.Layers || c.Tokens != sel.Tokens || c.K != sel.K || !slices.Equal(c.IDs, sel.IDs) {
		return fmt.Errorf("%s: the capture's selections differ from the replayed ones", path)
	}
	return nil
}

// TraceAgreement records the pre-registered fallback of a workload: the capture's selections in File
// differ from the recorded ones, so the workload is replayed on the capture's. SameSets of the Pairs
// (layer, token) pairs of the trace have the same Top-K set in both.
type TraceAgreement struct {
	File     string `json:"file"`
	SameSets int    `json:"same_top_k_sets"`
	Pairs    int    `json:"layer_token_pairs"`
	K        int    `json:"k"`
}

// captureTrace is the trace a workload with a capture is replayed on. The capture's selections at
// path are normally the recorded ones, ID for ID: then it returns recorded and no agreement. If they
// differ (the recorded trace came from another llama.cpp build), it returns the capture's, for every
// policy and the gate to be replayed on, and their agreement with the recorded ones. The capture
// must have the recorded trace's layers, tokens and K: it is the same workload.
func captureTrace(path string, recorded locality.Selections) (locality.Selections, *TraceAgreement, error) {
	c, err := locality.ReadNPYFile(path)
	if err != nil {
		return recorded, nil, err
	}
	if c.Layers != recorded.Layers || c.Tokens != recorded.Tokens || c.K != recorded.K {
		return recorded, nil, fmt.Errorf("%s: %d layers x %d tokens x %d, the recorded selections %d x %d x %d", path,
			c.Layers, c.Tokens, c.K, recorded.Layers, recorded.Tokens, recorded.K)
	}
	if slices.Equal(c.IDs, recorded.IDs) {
		return recorded, nil, nil
	}
	a := TraceAgreement{File: path, Pairs: c.Layers * c.Tokens, K: c.K}
	x, y := make([]int32, c.K), make([]int32, c.K)
	for i := 0; i < len(c.IDs); i += c.K {
		copy(x, c.IDs[i:i+c.K])
		copy(y, recorded.IDs[i:i+c.K])
		slices.Sort(x)
		slices.Sort(y)
		if slices.Equal(x, y) {
			a.SameSets++
		}
	}
	return c, &a, nil
}

// synthPred is a pred-h2 file of the selections themselves: row r of token t holds layer r+2's
// Top-K at t, in the selection file's order, and -1 where layer r+2 does not exist.
func synthPred(sel locality.Selections) locality.Selections {
	p := locality.Selections{Layers: sel.Layers, Tokens: sel.Tokens, K: sel.K, IDs: make([]int32, len(sel.IDs))}
	n := sel.Tokens * sel.K
	for row := range sel.Layers {
		if row+locality.HiddenStateBlocks < sel.Layers {
			copy(p.IDs[row*n:(row+1)*n], sel.IDs[(row+locality.HiddenStateBlocks)*n:(row+locality.HiddenStateBlocks+1)*n])
			continue
		}
		for i := row * n; i < (row+1)*n; i++ {
			p.IDs[i] = -1
		}
	}
	return p
}

// mappingCheck is the pre-registered check of the replay's index mapping: synth, a pred-h2 file of
// the selections themselves (synthPred), fed through the file-backed predictor must rank as the
// perfect same-token oracle before every step at k = the selections' K and replay exactly as it
// at every size and cost. The oracle ranks beyond the Top-K with pad's Markov predictor, the file
// does not, so only k = K is compared.
func mappingCheck(t trace, synth locality.Selections, pad locality.MarkovParams, sizes []int,
	costs []float64) (MappingCheck, error) {
	k := t.sel.K
	mc := MappingCheck{K: k, Sizes: len(sizes), CostsSeconds: costs}
	p, err := locality.NewHiddenState(synth, t.m.NExpert)
	if err != nil {
		return mc, err
	}
	hidden := locality.RankWith(p, t.sel, t.spans, t.m.NExpert, locality.SameToken, k)
	perfect := locality.RankWith(locality.NewPerfect(t.sel, locality.NewMarkov(t.sel.Layers, t.m.NExpert, pad)),
		t.sel, t.spans, t.m.NExpert, locality.SameToken, k)
	for step := range hidden.Steps() {
		hl, ht := hidden.Top(step, k)
		pl, pt := perfect.Top(step, k)
		if hl != pl || !slices.Equal(ht, pt) {
			return mc, fmt.Errorf("step %d: the file-backed predictor ranks layer %d %v, the oracle layer %d %v", step,
				hl, ht, pl, pt)
		}
		if ht != nil {
			mc.PredictedSteps++
		}
	}
	bases := []baseline{}
	for _, r := range []*locality.Ranking{hidden, perfect} {
		bases = append(bases, baseline{"", false, func(c int) (locality.Policy, error) {
			return locality.NewRanked(locality.NewSharedLRU(t.m.NExpert, t.m.bytes(), t.m.budget(c)), r, k), nil
		}})
	}
	for _, c := range costs {
		runs, err := replayAll(t.sel, t.spans, t.cut, t.m, sizes, bases, withCost(t.timing, c))
		if err != nil {
			return mc, err
		}
		for s := range sizes {
			if !reflect.DeepEqual(runs[0].Runs[s].Replayed, runs[1].Runs[s].Replayed) {
				return mc, fmt.Errorf("at %g µs and %d experts per layer the file-backed replay differs from the "+
					"oracle's", 1e6*c, sizes[s])
			}
		}
	}
	return mc, nil
}

// hiddenRecall scores the capture file pred of horizon h on the trace.
func hiddenRecall(t trace, pred locality.Selections, h int, file string) HiddenRecall {
	x := HiddenRecall{File: file, Blocks: h, Lead: h - 1, K: hiddenRecallKs}
	x.TestDecode, x.TestDecodePredicted = locality.HiddenRecall(pred, h, t.sel, t.spans, t.cut, len(t.spans),
		hiddenRecallKs)
	x.Decode, x.DecodePredicted = locality.HiddenRecall(pred, h, t.sel, t.spans, 0, len(t.spans), hiddenRecallKs)
	every := locality.TraceOrder(t.sel.Tokens)
	x.Tokens, x.TokensPredicted = locality.HiddenRecall(pred, h, t.sel, every, 0, len(every), hiddenRecallKs)
	return x
}

// hiddenRanking is the file-backed predictor's ranking of the trace at the largest k of markovKs.
func hiddenRanking(t trace, pred locality.Selections) (*locality.Ranking, error) {
	p, err := locality.NewHiddenState(pred, t.m.NExpert)
	if err != nil {
		return nil, err
	}
	return locality.RankWith(p, t.sel, t.spans, t.m.NExpert, locality.SameToken, slices.Max(markovKs)), nil
}

// hiddenGrids replays the file-backed predictor's prefetches at every k of markovKs, size and cost.
// Its ranking and its replays without cost do not depend on the cost and are made once.
func hiddenGrids(t trace, pred locality.Selections, file string, sizes []int, costs []float64) ([]HiddenGrid,
	error) {
	ranking, err := hiddenRanking(t, pred)
	if err != nil {
		return nil, err
	}
	bases := make([]baseline, len(markovKs))
	for i, k := range markovKs {
		bases[i] = baseline{fmt.Sprintf("hidden state pred-h2 same token, k %d, on shared LRU", k), false,
			func(c int) (locality.Policy, error) {
				return locality.NewRanked(locality.NewSharedLRU(t.m.NExpert, t.m.bytes(), t.m.budget(c)), ranking, k), nil
			}}
	}
	free, err := replayAll(t.sel, t.spans, t.cut, t.m, sizes, bases, t.timing)
	if err != nil {
		return nil, err
	}
	var out []HiddenGrid
	for _, c := range costs {
		g := HiddenGrid{File: file, Horizon: locality.SameToken.String(), LeadSteps: locality.SameToken.Lead(t.sel.Layers),
			CostSeconds: c, Prefetches: Prefetches{K: markovKs, Free: free}}
		if g.Runs, err = replayAll(t.sel, t.spans, t.cut, t.m, sizes, bases, withCost(t.timing, c)); err != nil {
			return nil, err
		}
		g.OwnK = chooseK(g.Runs, markovKs, len(t.timing.Links))
		out = append(out, g)
	}
	return out, nil
}

// hiddenKind is the predictor kind's name of the file-backed predictor at cost seconds per step.
func hiddenKind(cost float64) string { return fmt.Sprintf("hidden state (pred-h2, %g µs)", 1e6*cost) }

// hiddenKinds are the file-backed predictor's kinds at every cost the first workload with it has,
// for the same token only; a workload without it has none.
func hiddenKinds(results []result) []predictorKind {
	same := slices.Index(horizons, locality.SameToken)
	var kinds []predictorKind
	for _, r := range results {
		for i, g := range r.w.Hidden {
			kinds = append(kinds, predictorKind{hiddenKind(g.CostSeconds), func(w Workload, hi int) *Prefetches {
				if hi != same || i >= len(w.Hidden) {
					return nil
				}
				return &w.Hidden[i].Prefetches
			}})
		}
		if len(kinds) > 0 {
			break
		}
	}
	return kinds
}

// hiddenChecks are the CHECK lines of the file-backed predictor at Stage B's gate g: per cost, its
// rows retained and where.
func hiddenChecks(model string, g Gate, costs []float64) []string {
	var out []string
	for _, c := range costs {
		kind := hiddenKind(c)
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
		line := fmt.Sprintf("CHECK %s: the hidden-state predictor (pred-h2) at %g µs per decoded layer step is "+
			"retained in %d of %d rows of the Stage B gate", model, 1e6*c, n, of)
		if len(wins) > 0 {
			line += " (" + strings.Join(wins, ", ") + ")"
		}
		out = append(out, line)
	}
	return out
}

// writeHidden reports the file-backed predictor, if -hidden-state-dir is given: what it is, the
// models without a capture, the mapping check, the capture files' recall and, per cost, its held-out
// decode against the best Stage A policy. Its verdicts are in the Stage B gate's table.
func writeHidden(w io.Writer, a Assumptions, links []locality.Link, sizes []int, results []result,
	gatesB map[string]Gate) {
	if a.HiddenStateDir == "" || len(results) == 0 {
		return
	}
	layers := results[0].t.sel.Layers
	var costs []string
	for _, c := range a.HiddenStateSeconds {
		costs = append(costs, fmt.Sprintf("%g", 1e6*c))
	}
	fmt.Fprintf(w, "\n## Stage E6: a hidden-state predictor, replayed from the capture\n\n")
	fmt.Fprintf(w, "The router's top-48 scored on the residual stream by `tools/e6/capture_router.py`, read from "+
		"`%s`. Row l of `pred-hH` holds the experts block l+H's router ranks first on the residual stream after "+
		"block l, best first. The judged predictor reads `pred-h2`: before layer l of a decoded token it predicts "+
		"the token's layer l+1, %d step ahead, as the same-token predictors and the perfect oracle do, with the "+
		"first k experts of row l−1, block l+1's router on the stream after block l−1, which has run. It reads no "+
		"row at or after l and no later token. It predicts layers 2 to %d of every decoded token. Its cost per "+
		"decoded layer step is declared (`-hidden-state-us`: %s µs), added to every decoded step's compute as the "+
		"predictors' is. It is replayed and judged at Stage B's gate exactly as they are: k from %s tuned per "+
		"budget and link on the own training prefix, on the shared LRU, bytes prefetched counted as loaded; its "+
		"verdicts are the rows \"same token hidden state (pred-h2, … µs)\" of the Stage B gate below. The "+
		"capture's selections are checked to be the replayed ones, ID for ID.\n", a.HiddenStateDir,
		locality.SameToken.Lead(layers), layers-1, strings.Join(costs, ", "), joinInts(markovKs))
	var with []string
	for _, m := range modelsOf(results) {
		has := false
		for _, r := range ofModel(results, m) {
			has = has || len(r.w.Hidden) > 0
		}
		if !has {
			fmt.Fprintf(w, "\n%s has no capture in `%s` and is not judged.\n", m, a.HiddenStateDir)
			continue
		}
		with = append(with, m)
	}
	for _, r := range results {
		if x := r.w.HiddenTrace; x != nil {
			fmt.Fprintf(w, "\n**%s %s is replayed on the capture's own selections.** `%s` differs from the recorded "+
				"trace, which came from another llama.cpp build: %d of %d (layer, token) pairs (%.2f%%) have the same "+
				"Top-%d set. As pre-registered, every policy of this workload, the shared LRU and the gate's reference "+
				"among them, is replayed on the capture's selections and not on the recorded ones; the recorded hits "+
				"of `kolibri-cache` are not compared.\n", r.model, r.workload, x.File, x.SameSets, x.Pairs,
				100*float64(x.SameSets)/float64(max(x.Pairs, 1)), x.K)
		}
	}
	for _, m := range with {
		var ks []string
		steps := 0
		all := true
		for _, r := range ofModel(results, m) {
			if mc := r.w.HiddenMapping; mc != nil {
				ks = append(ks, fmt.Sprint(mc.K))
				steps += mc.PredictedSteps
			} else {
				all = false
			}
		}
		if all && len(ks) > 0 {
			fmt.Fprintf(w, "\n**The mapping check passes for %s.** In every workload the replayed selections "+
				"themselves, written as a `pred-h2` file (row l−1 holding layer l+1's Top-%s) and fed through the "+
				"file-backed predictor, rank as the perfect same-token oracle before each of %d predicted steps and "+
				"replay exactly as it, at k = %s, every budget and every cost. The oracle ranks beyond the Top-%s by "+
				"the Markov predictor and the file does not, so only k = %s is compared.\n", m, ks[0], steps, ks[0],
				ks[0], ks[0])
		}
	}
	fmt.Fprintf(w, "\nRecall of the actual Top-%d per layer and token, needed experts covered (whole sets covered), "+
		"for `pred-h1` (layer l from block l's router on the stream after block l−1, lead 0, not judged) and "+
		"`pred-h2` (layer l+1 from block l+1's router on the same stream, lead 1, the judged predictor). Test "+
		"decode is the decoded tokens of the test suffixes, all decode every decoded token, every token every "+
		"position of the trace, prefill included.\n", results[0].t.sel.K)
	for _, only := range []bool{false, true} {
		if only {
			fmt.Fprintf(w, "\nThe predicted layers alone (layers 1 to %d for `pred-h1`, 2 to %d for `pred-h2`):\n\n",
				layers-1, layers-1)
		} else {
			fmt.Fprintf(w, "\nAll layers, those no row predicts (layer 0 for `pred-h1`, 0 and 1 for `pred-h2`) "+
				"counted with nothing covered, as in Stage B's recall tables:\n\n")
		}
		fmt.Fprintf(w, "| Model | Workload | File | Lead |")
		for _, part := range []string{"Test decode", "All decode", "Every token"} {
			for _, k := range hiddenRecallKs {
				fmt.Fprintf(w, " %s, k %d |", part, k)
			}
		}
		fmt.Fprintf(w, "\n|---|---|---|--:|%s\n", strings.Repeat("--:|", 3*len(hiddenRecallKs)))
		for _, m := range with {
			pooled := map[int][3][]locality.Recall{}
			var blocks []int
			row := func(wl string, x HiddenRecall, parts [3][]locality.Recall) {
				fmt.Fprintf(w, "| %s | %s | pred-h%d | %d |", m, wl, x.Blocks, x.Lead)
				for _, p := range parts {
					for _, r := range p {
						fmt.Fprintf(w, " %s |", recallCell(r))
					}
				}
				fmt.Fprintln(w)
			}
			var leads = map[int]HiddenRecall{}
			for _, r := range ofModel(results, m) {
				for _, x := range r.w.HiddenRecall {
					parts := [3][]locality.Recall{x.TestDecode, x.Decode, x.Tokens}
					if only {
						parts = [3][]locality.Recall{x.TestDecodePredicted, x.DecodePredicted, x.TokensPredicted}
					}
					row(r.workload, x, parts)
					if _, ok := leads[x.Blocks]; !ok {
						blocks = append(blocks, x.Blocks)
						leads[x.Blocks] = x
						pooled[x.Blocks] = [3][]locality.Recall{make([]locality.Recall, len(x.K)),
							make([]locality.Recall, len(x.K)), make([]locality.Recall, len(x.K))}
					}
					sum := pooled[x.Blocks]
					for i := range sum {
						for j := range sum[i] {
							sum[i][j] = sumRecall([]locality.Recall{sum[i][j], parts[i][j]})
						}
					}
				}
			}
			for _, b := range blocks {
				row("all", leads[b], pooled[b])
			}
		}
	}
	same := slices.Index(horizons, locality.SameToken)
	kinds := hiddenKinds(results)
	for _, m := range with {
		gb, ok := gatesB[m]
		if !ok {
			continue
		}
		for _, kind := range kinds {
			fmt.Fprintf(w, "\n%s, the hidden-state predictor at %s, all workloads' test suffixes together, decode: "+
				"MB/token loaded and ms/token per link, the best Stage A policy against the predictor with k tuned for "+
				"the link (k the range of the workloads' choices):\n\n| Experts per layer |", m,
				strings.TrimSuffix(strings.TrimPrefix(kind.name, "hidden state (pred-h2, "), ")"))
			for _, l := range links {
				fmt.Fprintf(w, " k %s | Best Stage A MB | Predictor MB | Best Stage A %s | Predictor %s |", l.Name,
					l.Name, l.Name)
			}
			fmt.Fprintf(w, "\n|--:|%s\n", strings.Repeat("--:|--:|--:|--:|--:|", len(links)))
			for s, c := range sizes {
				fmt.Fprintf(w, "| %d |", c)
				for i := range links {
					var tokens int
					var sec float64
					var b int64
					var ks []int
					for _, r := range ofModel(results, m) {
						p := kind.pick(r.w, same)
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
					fmt.Fprintf(w, " %s | %s | %s | %s | %s |", kr, mb(ref.LoadedBytes), mb(b), per(ref.Seconds[i]),
						per(sec))
				}
				fmt.Fprintln(w)
			}
		}
	}
}

// hiddenWorkload adds the hidden-state predictor of model name's capture in dir to workload wl's
// w: the capture's selections checked against the replayed ones, both capture files' recall, the
// mapping check against the perfect oracle (pad its Markov padding, never used at k = K), and the
// predictor's prefetches at every cost, checked to replay online as its ranking at size index d.
func hiddenWorkload(w *Workload, t trace, dir, name, wl string, pad locality.MarkovParams, sizes []int, d int,
	costs []float64) error {
	m := short(name)
	selPath := filepath.Join(dir, fmt.Sprintf("%s.%s.experts.npy", wl, name))
	if err := sameSelections(selPath, t.sel); err != nil {
		return err
	}
	fmt.Printf("PASS %s %s: the capture's selections are the replayed ones, ID for ID (%s)\n", m, wl,
		filepath.Base(selPath))
	for _, h := range []int{1, locality.HiddenStateBlocks} {
		path := predPath(dir, wl, name, h)
		pred, err := loadPred(path, h, t.sel, t.m.NExpert)
		if err != nil {
			return err
		}
		fmt.Printf("PASS %s %s: pred-h%d rows 0 to %d hold %d distinct experts each, rows %d to %d only -1\n", m, wl,
			h, pred.Layers-h-1, pred.K, pred.Layers-h, pred.Layers-1)
		w.HiddenRecall = append(w.HiddenRecall, hiddenRecall(t, pred, h, path))
		if h != locality.HiddenStateBlocks {
			continue
		}
		mc, err := mappingCheck(t, synthPred(t.sel), pad, sizes, costs)
		if err != nil {
			return fmt.Errorf("the mapping check: %w", err)
		}
		w.HiddenMapping = &mc
		var us []string
		for _, c := range costs {
			us = append(us, fmt.Sprintf("%g", 1e6*c))
		}
		fmt.Printf("PASS %s %s: mapping check: the selections themselves as a pred-h2 file rank as the perfect "+
			"same-token oracle before all %d predicted steps and replay exactly as it at k %d, %d sizes and %s µs "+
			"(compared at k %d only: the oracle pads beyond the Top-%d with Markov ranks)\n", m, wl, mc.PredictedSteps,
			mc.K, len(sizes), strings.Join(us, ", "), mc.K, mc.K)
		if w.Hidden, err = hiddenGrids(t, pred, path, sizes, costs); err != nil {
			return err
		}
		p, err := locality.NewHiddenState(pred, t.m.NExpert)
		if err != nil {
			return err
		}
		if err := checkOnline(t, w.Hidden[0].Prefetches, p, locality.SameToken, sizes, d, costs[0]); err != nil {
			return fmt.Errorf("the hidden-state predictor: %w", err)
		}
		fmt.Printf("PASS %s %s: the hidden-state predictor as an online policy replays as its ranking at k %d and %d "+
			"experts per layer\n", m, wl, slices.Max(markovKs), sizes[d])
	}
	return nil
}
