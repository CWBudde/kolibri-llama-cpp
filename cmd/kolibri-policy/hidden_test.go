package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
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

// writeSelections writes sel as numpy.save writes an int16 [layers, tokens, k] array.
func writeSelections(t *testing.T, path string, sel locality.Selections) {
	t.Helper()
	header := fmt.Sprintf("{'descr': '<i2', 'fortran_order': False, 'shape': (%d, %d, %d), }", sel.Layers, sel.Tokens,
		sel.K)
	header += strings.Repeat(" ", (64-(10+len(header)+1)%64)%64) + "\n"
	var b bytes.Buffer
	b.WriteString("\x93NUMPY\x01\x00")
	_ = binary.Write(&b, binary.LittleEndian, uint16(len(header)))
	b.WriteString(header)
	ids := make([]int16, len(sel.IDs))
	for i, e := range sel.IDs {
		ids[i] = int16(e)
	}
	_ = binary.Write(&b, binary.LittleEndian, ids)
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

// The pre-registered fallback: a capture whose selections differ from the recorded ones becomes the
// trace, and the share of (layer, token) pairs with the same Top-K set is reported. A capture that
// matches changes nothing, and one of another shape is an error.
func TestCaptureTrace(t *testing.T) {
	rec := hiddenTestTrace(t).sel
	path := filepath.Join(t.TempDir(), "w.M.experts.npy")

	writeSelections(t, path, rec)
	got, agree, err := captureTrace(path, rec)
	if err != nil || agree != nil || !slices.Equal(got.IDs, rec.IDs) {
		t.Fatalf("a matching capture: agreement %v, err %v", agree, err)
	}

	// three pairs change: one to another set, two keep their set in another order
	other := rec
	other.IDs = slices.Clone(rec.IDs)
	pair := func(l, tok int) []int32 { return other.IDs[(l*rec.Tokens+tok)*rec.K:][:rec.K] }
	p := pair(1, 7)
	for e := range int32(8) {
		if e != p[0] && e != p[1] {
			p[0] = e
			break
		}
	}
	for _, lt := range [][2]int{{0, 0}, {3, 399}} {
		p := pair(lt[0], lt[1])
		p[0], p[1] = p[1], p[0]
	}
	writeSelections(t, path, other)
	got, agree, err = captureTrace(path, rec)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got.IDs, other.IDs) || slices.Equal(got.IDs, rec.IDs) {
		t.Fatal("a differing capture: the trace is not the capture's selections")
	}
	if want := (TraceAgreement{File: path, SameSets: 4*400 - 1, Pairs: 4 * 400, K: 2}); agree == nil || *agree != want {
		t.Fatalf("agreement %+v, want %+v", agree, want)
	}
	if err := sameSelections(path, got); err != nil {
		t.Fatalf("the capture against the trace it became: %v", err)
	}
	if err := sameSelections(path, rec); err == nil {
		t.Fatal("the capture against the recorded trace: no error")
	}

	writeSelections(t, path, locality.Selections{Layers: 4, Tokens: 399, K: 2, IDs: rec.IDs[:4*399*2]})
	if _, _, err := captureTrace(path, rec); err == nil {
		t.Fatal("a capture of another token count: no error")
	}
}

// The report names a workload replayed on the capture's own selections, and says nothing new
// without one.
func TestWriteHiddenAgreement(t *testing.T) {
	tr := hiddenTestTrace(t)
	pred := widen(synthPred(tr.sel), 8)
	sizes := []int{1, 2}
	grids, err := hiddenGrids(tr, pred, "f", sizes, []float64{23e-6})
	if err != nil {
		t.Fatal(err)
	}
	report := func(agree *TraceAgreement) string {
		w := Workload{Hidden: grids, HiddenRecall: []HiddenRecall{hiddenRecall(tr, pred, 2, "f2")},
			HiddenMapping: &MappingCheck{K: 2, Sizes: 2, CostsSeconds: []float64{23e-6}}, HiddenTrace: agree}
		results := []result{{model: "m", workload: "w", w: w, t: tr}}
		ref := GateEntry{Policy: "best", Sizes: []GateSize{{ExpertsPerLayer: 1, Seconds: []float64{1}},
			{ExpertsPerLayer: 2, Seconds: []float64{1}}}}
		a := Gate{Tokens: 1, Links: []string{"ssd"}, Reference: ref}
		var b bytes.Buffer
		writeHidden(&b, Assumptions{HiddenStateDir: "/cap", HiddenStateSeconds: []float64{23e-6}}, tr.timing.Links,
			sizes, results, map[string]Gate{"m": stageBGate(results, "m", sizes, []string{"ssd"}, a)})
		return b.String()
	}
	plain := report(nil)
	if strings.Contains(plain, "own selections") {
		t.Fatalf("a matching capture is reported as a fallback:\n%s", plain)
	}
	md := report(&TraceAgreement{File: "/cap/w.M.experts.npy", SameSets: 1599, Pairs: 1600, K: 2})
	for _, want := range []string{"m w is replayed on the capture's own selections", "1599 of 1600", "99.94%",
		"same Top-2 set"} {
		if !strings.Contains(md, want) {
			t.Errorf("the report lacks %q", want)
		}
	}
	// nothing else changes
	i := strings.Index(md, "\n\n**m w is replayed")
	j := i + 2 + strings.Index(md[i+2:], "\n")
	if i < 0 || md[:i]+md[j:] != plain {
		t.Errorf("the fallback changes more of the report than its own paragraph")
	}
}

// The capture's manifest must name the token IDs the replay uses: the hash tools/locality/workloads.py's
// ids_sha256 gives them. Another token sequence, a workload the manifest lacks or no manifest is an
// error.
func TestCaptureTokens(t *testing.T) {
	ids := []int32{1, 22, 333}
	// sha256 of "1,22,333", as ids_sha256([1, 22, 333])
	if got := idsSHA256(ids); got != "8915516745284e69425d3f566025a855f5a5718d6d44f09451fb19d0f61fc185" {
		t.Fatalf("idsSHA256 = %q", got)
	}
	dir := t.TempDir()
	if err := captureTokens(dir, "w", ids); err == nil {
		t.Fatal("no manifest.json: no error")
	}
	manifest := func(body string) {
		if err := os.WriteFile(filepath.Join(dir, "manifest.json"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	entry := func(sum string, n int, full bool) string {
		return fmt.Sprintf(`{"tool": "e6-capture-router-1", "workloads": {"w": {"n_tokens": %d, "tokens_sha256": %q, `+
			`"full": %v}}}`, n, sum, full)
	}
	manifest(entry(idsSHA256(ids), 3, true))
	if err := captureTokens(dir, "w", ids); err != nil {
		t.Fatalf("the capture's own tokens: %v", err)
	}
	if err := captureTokens(dir, "other", ids); err == nil || !strings.Contains(err.Error(), "other") {
		t.Fatalf("a workload the manifest lacks: %v", err)
	}
	// the same count of other tokens
	if err := captureTokens(dir, "w", []int32{1, 22, 334}); err == nil || !strings.Contains(err.Error(), "token IDs") {
		t.Fatalf("other token IDs: %v", err)
	}
	manifest(entry("", 3, true))
	if err := captureTokens(dir, "w", ids); err == nil {
		t.Fatal("an entry without tokens_sha256: no error")
	}
	manifest("{")
	if err := captureTokens(dir, "w", ids); err == nil {
		t.Fatal("a broken manifest: no error")
	}
}

// A capture narrower than the widest k the replay ranks (capture_router.py --top-k 6) would make
// the predictor panic; loadPred rejects it as an input error that names the width needed.
func TestLoadPredWidth(t *testing.T) {
	tr := hiddenTestTrace(t)
	narrow := synthPred(tr.sel) // 2 experts per row, of 8
	func() {
		defer func() {
			if recover() == nil {
				t.Error("ranking a pred file of 2 at k 8 does not panic: the width check guards nothing")
			}
		}()
		_, _ = hiddenRanking(tr, narrow)
	}()
	dir := t.TempDir()
	for _, h := range []int{1, locality.HiddenStateBlocks} {
		p := narrow
		if h == 1 { // row r holds layer r+1
			p = locality.Selections{Layers: 4, Tokens: 400, K: 2, IDs: slices.Clone(tr.sel.IDs)}
			copy(p.IDs, tr.sel.IDs[400*2:])
			for i := 3 * 400 * 2; i < len(p.IDs); i++ {
				p.IDs[i] = -1
			}
		}
		path := filepath.Join(dir, fmt.Sprintf("w.M.pred-h%d.npy", h))
		writeSelections(t, path, p)
		_, err := loadPred(path, h, tr.sel, 8)
		if err == nil || !strings.Contains(err.Error(), "2 experts per row") || !strings.Contains(err.Error(), "at least 8") {
			t.Errorf("pred-h%d of width 2: %v", h, err)
		}
		writeSelections(t, path, widen(p, 8))
		if _, err := loadPred(path, h, tr.sel, 8); err != nil {
			t.Errorf("pred-h%d of width 8: %v", h, err)
		}
	}
}
