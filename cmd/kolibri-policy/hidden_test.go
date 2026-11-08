package main

import (
	"bytes"
	"encoding/json"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"kolibri-llm/internal/locality"
)

// hiddenTestTrace is 4 layers x 400 tokens, Top-2 of 8 experts, skewed; two answers.
func hiddenTestTrace(t *testing.T) trace {
	t.Helper()
	r := rand.New(rand.NewPCG(11, 12))
	sel := locality.Selections{Layers: 4, Tokens: 400, K: 2, IDs: make([]int32, 4*400*2)}
	for i := 0; i < len(sel.IDs); i += 2 {
		a := int32(min(r.IntN(8), r.IntN(8)))
		sel.IDs[i], sel.IDs[i+1] = a, (a+1+int32(r.IntN(7)))%8
	}
	answers := []locality.Answer{{Prompt: 100, Tokens: make([]int32, 80)}, {Prompt: 250, Tokens: make([]int32, 80)}}
	spans, err := locality.OnlineSpans(400, answers, true, 64)
	if err != nil {
		t.Fatal(err)
	}
	cut, err := locality.SplitAfter(spans, answers, 1)
	if err != nil {
		t.Fatal(err)
	}
	var m Model
	if err := json.Unmarshal([]byte(`{"n_expert": 8, "layers": [{"bytes_per_expert": 10}, {"bytes_per_expert": 10},
		{"bytes_per_expert": 10}, {"bytes_per_expert": 10}]}`), &m); err != nil {
		t.Fatal(err)
	}
	tm := locality.Timing{Links: []locality.Link{{Name: "ssd", BytesPerSecond: 1000}}, DecodeStep: 0.01,
		PrefillToken: 0.001}
	return trace{sel, spans, cut, m, tm}
}

// widen pads every target row of a pred file of width sel.K to all nExpert experts, the missing
// ones in ascending order, as a top-k of every expert would hold them.
func widen(p locality.Selections, nExpert int) locality.Selections {
	out := locality.Selections{Layers: p.Layers, Tokens: p.Tokens, K: nExpert, IDs: make([]int32, 0,
		p.Layers*p.Tokens*nExpert)}
	for i := 0; i < len(p.IDs); i += p.K {
		row := p.IDs[i : i+p.K]
		if row[0] < 0 {
			for range nExpert {
				out.IDs = append(out.IDs, -1)
			}
			continue
		}
		out.IDs = append(out.IDs, row...)
		for e := range int32(nExpert) {
			if !slices.Contains(row, e) {
				out.IDs = append(out.IDs, e)
			}
		}
	}
	return out
}

func TestSynthPred(t *testing.T) {
	tr := hiddenTestTrace(t)
	p := synthPred(tr.sel)
	if p.Layers != 4 || p.Tokens != 400 || p.K != 2 {
		t.Fatalf("shape %d x %d x %d", p.Layers, p.Tokens, p.K)
	}
	if err := locality.CheckPred(p, 2, 8); err != nil {
		t.Fatal(err)
	}
	// row 1 of token 7 is layer 3's Top-2 at token 7; rows 2 and 3 have no target
	if got, want := p.IDs[(1*400+7)*2:(1*400+8)*2], tr.sel.IDs[(3*400+7)*2:(3*400+8)*2]; !slices.Equal(got, want) {
		t.Fatalf("row 1, token 7: %v, want %v", got, want)
	}
}

func TestHiddenMappingCheck(t *testing.T) {
	tr := hiddenTestTrace(t)
	pad := locality.MarkovParams{HalfLife: 16, Mix: 0.5}
	sizes, costs := []int{1, 2}, []float64{0, 1e-3}
	if _, err := mappingCheck(tr, synthPred(tr.sel), pad, sizes, costs); err != nil {
		t.Fatal(err)
	}
	// the selections one token late: row r of token t holds layer r+2 of token t-1
	late := synthPred(tr.sel)
	w := late.K
	for row := range 2 {
		for tok := 1; tok < 400; tok++ {
			copy(late.IDs[(row*400+tok)*w:(row*400+tok+1)*w], tr.sel.IDs[((row+2)*400+tok-1)*w:((row+2)*400+tok)*w])
		}
	}
	if _, err := mappingCheck(tr, late, pad, sizes, costs); err == nil {
		t.Fatal("the selections a token late pass the mapping check")
	}
}

func TestHiddenCostReachesGate(t *testing.T) {
	tr := hiddenTestTrace(t)
	pred := widen(synthPred(tr.sel), 8)
	sizes := []int{1, 2}
	costs := []float64{0, 1} // 1 s per decoded layer step dwarfs every load
	grids, err := hiddenGrids(tr, pred, "f.pred-h2.npy", sizes, costs)
	if err != nil {
		t.Fatal(err)
	}
	if len(grids) != 2 || grids[0].CostSeconds != 0 || grids[1].CostSeconds != 1 ||
		grids[1].Horizon != locality.SameToken.String() || grids[1].LeadSteps != 1 || grids[1].File != "f.pred-h2.npy" ||
		!slices.Equal(grids[1].K, markovKs) {
		t.Fatalf("grids: %d, %+v", len(grids), grids[1])
	}
	decoded := 0
	for _, s := range tr.spans[tr.cut:] {
		if s.Phase == locality.Decode {
			decoded++
		}
	}
	steps := float64(decoded * tr.sel.Layers)
	ref := GateEntry{Policy: "best", Sizes: []GateSize{{ExpertsPerLayer: 1, LoadedBytes: 1 << 40,
		Seconds: []float64{0.5 * steps}}, {ExpertsPerLayer: 2, LoadedBytes: 0, Seconds: []float64{0.5 * steps}}}}
	a := Gate{Tokens: decoded, Links: []string{"ssd"}, Reference: ref}
	// a model without a capture listed first: it judges no hidden-state row, the other does
	results := []result{{model: "q8", workload: "w", w: Workload{}, t: tr},
		{model: "m", workload: "w", w: Workload{Hidden: grids}, t: tr}}
	if g := stageBGate(results, "q8", sizes, []string{"ssd"}, a); len(g.Policies) != 0 {
		t.Fatalf("the model without a capture judges %d policies", len(g.Policies))
	}
	g := stageBGate(results, "m", sizes, []string{"ssd"}, a)
	if len(g.Policies) != 2 {
		t.Fatalf("%d policies judged, want the predictor at each cost", len(g.Policies))
	}
	for i, e := range g.Policies {
		want := predictorName(hiddenKind(costs[i]), locality.SameToken, "ssd")
		if e.Policy != want || !strings.HasPrefix(want, "same token hidden state (pred-h2, ") {
			t.Fatalf("policy %q, want %q", e.Policy, want)
		}
		for s := range sizes {
			p := grids[i]
			run := p.Runs[slices.Index(p.K, p.OwnK[s][0])].Runs[s].Split.Test.Decode
			if got := e.Sizes[s].Seconds[0]; got != run.Links[0].Seconds {
				t.Fatalf("cost %g, size %d: gate %g s, the replay with cost %g s", costs[i], sizes[s], got,
					run.Links[0].Seconds)
			}
			if got := e.Sizes[s].LoadedBytes; got != loadedBytes(run) {
				t.Fatalf("cost %g, size %d: gate %d bytes, the replay %d", costs[i], sizes[s], got, loadedBytes(run))
			}
			if c := e.Sizes[s].Seconds[0]; (c >= steps) != (costs[i] == 1) {
				t.Fatalf("cost %g, size %d: %g s for %g decoded layer steps", costs[i], sizes[s], c, steps)
			}
		}
	}
	if w := g.Policies[0].Sizes[1].Wins; !slices.Equal(w, []string{"ssd"}) {
		t.Fatalf("cost 0, size 2: wins %v", w)
	}
	if w := g.Policies[1].Sizes[1].Wins; w != nil {
		t.Fatalf("cost 1 s, size 2: wins %v", w)
	}
	lines := hiddenChecks("m", g, costs)
	if len(lines) != 2 || !strings.HasPrefix(lines[0], "CHECK m: the hidden-state predictor (pred-h2) at 0 µs per "+
		"decoded layer step is retained in 1 of 1 rows of the Stage B gate (") ||
		lines[1] != "CHECK m: the hidden-state predictor (pred-h2) at 1e+06 µs per decoded layer step is retained in "+
			"1 of 1 rows of the Stage B gate (bytes at 1)" {
		t.Fatalf("checks %q", lines)
	}
	// without the flag nothing is added
	results[1].w.Hidden = nil
	if g := stageBGate(results, "m", sizes, []string{"ssd"}, a); len(g.Policies) != 0 {
		t.Fatalf("%d policies without the predictor", len(g.Policies))
	}
}

func TestHiddenRecallRows(t *testing.T) {
	tr := hiddenTestTrace(t)
	pred := widen(synthPred(tr.sel), 8)
	x := hiddenRecall(tr, pred, 2, "f")
	if x.Blocks != 2 || x.Lead != 1 || x.File != "f" || !slices.Equal(x.K, hiddenRecallKs) {
		t.Fatalf("%+v", x)
	}
	// built from the selections: every predicted layer covered at k 6 on every step it predicts
	for _, r := range [][]locality.Recall{x.TestDecodePredicted, x.DecodePredicted, x.TokensPredicted} {
		if r[0].Covered != r[0].Needed || r[0].Exact != r[0].Steps || r[0].Steps == 0 {
			t.Fatalf("predicted layers at k 6: %+v", r[0])
		}
	}
	if x.Tokens[0].Steps != tr.sel.Tokens*tr.sel.Layers || x.TokensPredicted[0].Steps != tr.sel.Tokens*2 {
		t.Fatalf("every token: %+v, predicted %+v", x.Tokens[0], x.TokensPredicted[0])
	}
	h1 := hiddenRecall(tr, pred, 1, "g")
	if h1.Lead != 0 || h1.TokensPredicted[0].Steps != tr.sel.Tokens*3 {
		t.Fatalf("pred-h1: %+v", h1)
	}
}

func TestCaptureOf(t *testing.T) {
	dir := t.TempDir()
	wls := []string{"a", "b"}
	if ok, err := captureOf(dir, "M", wls); ok || err != nil {
		t.Fatalf("no files: %v, %v", ok, err)
	}
	touch := func(name string) {
		if err := os.WriteFile(filepath.Join(dir, name), nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	touch("a.M.pred-h1.npy")
	touch("a.M.pred-h2.npy")
	if _, err := captureOf(dir, "M", wls); err == nil {
		t.Fatal("a capture of one workload of two: no error")
	}
	touch("b.M.pred-h1.npy")
	touch("b.M.pred-h2.npy")
	touch("a.M.experts.npy")
	if _, err := captureOf(dir, "M", wls); err == nil {
		t.Fatal("a capture without b's selections: no error")
	}
	touch("b.M.experts.npy")
	if ok, err := captureOf(dir, "M", wls); !ok || err != nil {
		t.Fatalf("complete: %v, %v", ok, err)
	}
}

func TestWriteHidden(t *testing.T) {
	tr := hiddenTestTrace(t)
	pred := widen(synthPred(tr.sel), 8)
	sizes := []int{1, 2}
	grids, err := hiddenGrids(tr, pred, "f", sizes, []float64{23e-6})
	if err != nil {
		t.Fatal(err)
	}
	w := Workload{Hidden: grids, HiddenRecall: []HiddenRecall{hiddenRecall(tr, pred, 1, "f1"),
		hiddenRecall(tr, pred, 2, "f2")}, HiddenMapping: &MappingCheck{K: 2, Sizes: 2, CostsSeconds: []float64{23e-6}}}
	results := []result{{model: "q8", workload: "w", w: Workload{}, t: tr},
		{model: "m", workload: "w", w: w, t: tr}}
	ref := GateEntry{Policy: "best", Sizes: []GateSize{{ExpertsPerLayer: 1, Seconds: []float64{1}},
		{ExpertsPerLayer: 2, Seconds: []float64{1}}}}
	a := Gate{Tokens: 1, Links: []string{"ssd"}, Reference: ref}
	gatesB := map[string]Gate{"m": stageBGate(results, "m", sizes, []string{"ssd"}, a),
		"q8": stageBGate(results, "q8", sizes, []string{"ssd"}, a)}
	var b bytes.Buffer
	writeHidden(&b, Assumptions{HiddenStateDir: "/cap", HiddenStateSeconds: []float64{23e-6}}, tr.timing.Links, sizes,
		results, gatesB)
	md := b.String()
	for _, want := range []string{"## Stage E6: a hidden-state predictor", "q8 has no capture in `/cap`",
		"| m | w | pred-h1 | 0 |", "| m | w | pred-h2 | 1 |", "| m | all | pred-h2 | 1 |",
		"m, the hidden-state predictor at 23 µs", "| Predictor MB |", "mapping check"} {
		if !strings.Contains(md, want) {
			t.Errorf("the report lacks %q:\n%s", want, md)
		}
	}
	// off: nothing
	b.Reset()
	writeHidden(&b, Assumptions{}, tr.timing.Links, sizes, []result{{model: "q8", workload: "w", t: tr}}, gatesB)
	if b.Len() != 0 {
		t.Fatalf("written without the flag: %q", b.String())
	}
}
