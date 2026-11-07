package probe

import (
	"bytes"
	"errors"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
)

func smokeRows() []Item {
	var out []Item
	for _, it := range Generate(Seed) {
		if it.Split == SplitSmoke {
			out = append(out, it)
		}
	}
	return out
}

// signalActivations plants a direction per label in noise: label(row) moves it along a direction,
// with a group-level noise too, as paraphrases of one group sit close together.
func signalActivations(items []Item, dim int, label func(Item) int, classes int, strength float64, seed uint64) Matrix {
	r := rand.New(rand.NewPCG(seed, 1))
	dirs := make([][]float64, classes)
	for c := range dirs {
		dirs[c] = make([]float64, dim)
		for d := range dirs[c] {
			dirs[c][d] = r.NormFloat64()
		}
	}
	groupNoise := map[string][]float64{}
	m := Matrix{Rows: len(items), Dim: dim, Data: make([]float32, len(items)*dim)}
	for i, it := range items {
		gn, ok := groupNoise[it.ParaphraseGroup]
		if !ok {
			gn = make([]float64, dim)
			for d := range gn {
				gn[d] = 0.5 * r.NormFloat64()
			}
			groupNoise[it.ParaphraseGroup] = gn
		}
		row := m.Row(i)
		for d := range row {
			row[d] = float32(r.NormFloat64() + gn[d] + strength*dirs[label(it)][d])
		}
	}
	return m
}

// The solver reaches the stationary point: at the optimum l2 a_ic = (y_ic - p_ic)/n and the bias's
// gradient vanishes, for two and for three classes.
func TestSolverStationary(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 1))
	for _, k := range []int{2, 3} {
		const n, d = 12, 30
		y := make([]int, n)
		x := make([][]float64, n)
		for i := range x {
			y[i] = i % k
			x[i] = make([]float64, d)
			for j := range x[i] {
				x[i][j] = r.NormFloat64() / math.Sqrt(d)
			}
			x[i][y[i]] += 0.3
		}
		const l2 = 0.01
		kern := kernel(x, x)
		m := fitSoftmax(kern, y, k, l2)
		if !m.converged {
			t.Fatalf("k=%d: no convergence in %d iterations", k, m.iters)
		}
		gA, gB := make([]float64, n*k), make([]float64, k)
		if worst := gradient(kern, y, n, k, l2, m.alpha, m.bias, gA, gB); worst > gradTol {
			t.Fatalf("k=%d: gradient %g", k, worst)
		}
		// well separated training rows: the fit classifies them
		p := m.predict(kern, n, n, k)
		for i := range n {
			if argmax(p[i*k:(i+1)*k]) != y[i] {
				t.Fatalf("k=%d: training row %d misclassified", k, i)
			}
		}
	}
}

func TestSolverIsDeterministic(t *testing.T) {
	rows := smokeRows()
	m := signalActivations(rows, 64, TaskExecute.Label, 2, 0.5, 3)
	folds, err := GroupFolds(rows, 5, Seed)
	if err != nil {
		t.Fatal(err)
	}
	a, err := evaluateFolds(rows, []NamedMatrix{{"x", m}}, folds, DefaultOptions)
	if err != nil {
		t.Fatal(err)
	}
	b, err := evaluateFolds(rows, []NamedMatrix{{"x", m}}, folds, DefaultOptions)
	if err != nil {
		t.Fatal(err)
	}
	ja, _ := a.JSON()
	jb, _ := b.JSON()
	if !bytes.Equal(ja, jb) {
		t.Fatal("two runs give different reports")
	}
}

func TestGroupFolds(t *testing.T) {
	rows := smokeRows()
	folds, err := GroupFolds(rows, 5, Seed)
	if err != nil {
		t.Fatal(err)
	}
	if err := CheckFolds(rows, folds); err != nil {
		t.Fatal(err)
	}
	for f, test := range folds {
		if len(test) < 6 || len(test) > 10 {
			t.Errorf("fold %d has %d rows of 40: unbalanced", f, len(test))
		}
		counts := map[string]int{}
		for _, r := range trainRows(len(rows), test) {
			counts[rows[r].OperationFamily]++
		}
		for _, fam := range Families {
			if counts[fam] == 0 {
				t.Errorf("fold %d: no training row of %s", f, fam)
			}
		}
	}
	again, _ := GroupFolds(rows, 5, Seed)
	for f := range folds {
		if !slices.Equal(folds[f], again[f]) {
			t.Fatal("GroupFolds is not deterministic")
		}
	}
}

// Probes find a planted direction, at a level the baselines' table can be compared with.
func TestProbeFindsSignal(t *testing.T) {
	rows := smokeRows()
	for _, task := range []Task{TaskExecute, TaskFamily} {
		m := signalActivations(rows, 256, task.Label, len(task.Classes), 0.5, 7)
		folds, _ := GroupFolds(rows, 5, Seed)
		rep, err := evaluateFolds(rows, []NamedMatrix{{"layer-x", m}}, folds, Options{Folds: 5, L2: 0.01, Shuffles: 5, Seed: Seed})
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range rep.Results {
			if r.Name == "layer-x" && r.Task == task.Name {
				if r.Metrics.BalancedAccuracy < 0.9 {
					t.Errorf("%s: balanced accuracy %.3f on a strong signal", task.Name, r.Metrics.BalancedAccuracy)
				}
				if r.Unconverged > 0 {
					t.Errorf("%s: %d folds did not converge", task.Name, r.Unconverged)
				}
			}
		}
	}
}

// Control M75: on shuffled labels the same probe on the same activations scores near chance.
func TestM75ShuffledLabelsNearChance(t *testing.T) {
	rows := smokeRows()
	for _, task := range []Task{TaskExecute, TaskFamily} {
		m := signalActivations(rows, 256, task.Label, len(task.Classes), 0.5, 11)
		folds, _ := GroupFolds(rows, 5, Seed)
		rep, err := evaluateFolds(rows, []NamedMatrix{{"layer-x", m}}, folds, Options{Folds: 5, L2: 0.01, Shuffles: 200, Seed: Seed})
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range rep.Results {
			if r.Name != "layer-x" || r.Task != task.Name {
				continue
			}
			s := r.Shuffled
			if math.Abs(s.MeanBalanced-s.ChanceBalanced) > 0.05 {
				t.Errorf("%s: shuffled balanced accuracy %.3f, chance %.3f", task.Name, s.MeanBalanced, s.ChanceBalanced)
			}
			if r.Metrics.BalancedAccuracy <= s.MaxBalanced {
				t.Errorf("%s: the real labels score %.3f, a shuffle reached %.3f", task.Name, r.Metrics.BalancedAccuracy, s.MaxBalanced)
			}
			if s.ShareAtOrAbove != 0 {
				t.Errorf("%s: %.3f of the shuffles reach the real score", task.Name, s.ShareAtOrAbove)
			}
		}
	}
}

// naiveFolds is the split the probe must refuse: every k-th row, whatever its group.
func naiveFolds(n, k int) Folds {
	folds := make(Folds, k)
	for i := range n {
		folds[i%k] = append(folds[i%k], i)
	}
	return folds
}

// Control M76: a split that breaks the groups is refused, and what it would have reported is
// exactly the leak the refusal is for: activations that carry only a group's fingerprint, labels
// constant per group. Grouped folds score that near chance, the broken split near 1.
func TestM76LeakingSplitRefused(t *testing.T) {
	rows := smokeRows()
	naive := naiveFolds(len(rows), 5)
	m := signalActivations(rows, 64, TaskExecute.Label, 2, 0, 5)
	_, err := evaluateFolds(rows, []NamedMatrix{{"x", m}}, naive, DefaultOptions)
	if !errors.Is(err, ErrGroupLeak) {
		t.Fatalf("a split across groups: %v, want ErrGroupLeak", err)
	}
	if err := CheckFolds(rows, Folds{{0}, {0}}); err == nil {
		t.Fatal("a row tested twice passed")
	}
	if err := CheckFolds(rows, Folds{{0}}); err == nil {
		t.Fatal("rows that are never tested passed")
	}

	// the leak itself: 40 groups of 4 rows, label constant per group, activation = group noise
	const groups, perGroup, dim = 40, 4, 128
	r := rand.New(rand.NewPCG(9, 9))
	var items []Item
	fp := Matrix{Rows: groups * perGroup, Dim: dim, Data: make([]float32, groups*perGroup*dim)}
	for g := range groups {
		noise := make([]float64, dim)
		for d := range noise {
			noise[d] = r.NormFloat64()
		}
		for p := range perGroup {
			name := "g" + strconv.Itoa(g)
			items = append(items, Item{ID: name + "-" + strconv.Itoa(p), ParaphraseGroup: name, TemplateID: name, ExecuteExpected: g%2 == 0, OperationFamily: FamilyNone, Prompt: name})
			row := fp.Row(g*perGroup + p)
			for d := range row {
				row[d] = float32(noise[d] + 0.1*r.NormFloat64())
			}
		}
	}
	labels := make([]int, len(items))
	for i, it := range items {
		labels[i] = TaskExecute.Label(it)
	}
	score := func(folds Folds) float64 {
		oof, err := crossValidate(kernelsOf(fp, folds, len(items)), labels, 2, 0.01)
		if err != nil {
			t.Fatal(err)
		}
		return scores(items, labels, oof.pred, nil, TaskExecute).BalancedAccuracy
	}
	if v := score(naiveFolds(len(items), 5)); v < 0.95 {
		t.Errorf("the leaking split scores %.3f on group fingerprints; the control lost its point", v)
	}
	grouped, err := GroupFolds(items, 5, Seed)
	if err != nil {
		t.Fatal(err)
	}
	if err := CheckFolds(items, grouped); err != nil {
		t.Fatal(err)
	}
	if v := score(grouped); math.Abs(v-0.5) > 0.2 {
		t.Errorf("grouped folds score %.3f on group fingerprints, want near chance", v)
	}
	if err := CheckFolds(items, naiveFolds(len(items), 5)); !errors.Is(err, ErrGroupLeak) {
		t.Errorf("fingerprint data, split across groups: %v, want ErrGroupLeak", err)
	}
}

func TestBaselines(t *testing.T) {
	rows := smokeRows()
	folds, _ := GroupFolds(rows, 5, Seed)
	m := signalActivations(rows, 32, TaskExecute.Label, 2, 0, 13)
	rep, err := evaluateFolds(rows, []NamedMatrix{{"layer-x", m}}, folds, Options{Folds: 5, L2: 0.01, Seed: Seed})
	if err != nil {
		t.Fatal(err)
	}
	get := func(task, name string) Row {
		for _, r := range rep.Results {
			if r.Task == task && r.Name == name {
				return r
			}
		}
		t.Fatalf("no row %s %s", task, name)
		return Row{}
	}
	// the labels are balanced 20/20: the majority of a training fold is one class per fold, so chance
	if v := get("execute", "majority").Metrics.BalancedAccuracy; v > 0.6 {
		t.Errorf("majority baseline: balanced accuracy %.3f", v)
	}
	// the rule fires on the tempting keywords of category D, on purpose
	rule := get("execute", "keyword rule").Metrics
	if rule.Recall == nil || *rule.Recall < 0.8 {
		t.Errorf("keyword rule: recall %v, it should find the computations", rule.Recall)
	}
	if rule.HardNegativeCorrect*2 > rule.HardNegatives {
		t.Errorf("keyword rule: %d/%d hard negatives right; they are built to tempt it", rule.HardNegativeCorrect, rule.HardNegatives)
	}
	if rule.Precision == nil || *rule.Precision > 0.8 {
		t.Errorf("keyword rule: precision %v", rule.Precision)
	}
	// the learned words have only the training groups to learn from, so all this asks is that they
	// see something; how they compare with the rule is a result, not a property
	bow := get("execute", "bag of words").Metrics
	if bow.BalancedAccuracy <= 0.5 {
		t.Errorf("bag of words: balanced accuracy %.3f", bow.BalancedAccuracy)
	}
	// two surface cues give nearly the whole execute label, so they are the bar, above the words
	if surface := get("execute", "surface rule").Metrics; surface.Accuracy != 39.0/40 || surface.Accuracy <= bow.Accuracy {
		t.Errorf("surface rule: accuracy %.3f, bag of words %.3f; docs/layer-probe.md says 39 of 40", surface.Accuracy, bow.Accuracy)
	}
	for _, task := range []string{"execute", "operation_family"} {
		if m := get(task, "prompt length").Metrics; m.Brier == nil || m.BalancedAccuracy == 0 {
			t.Errorf("%s: prompt length row: %+v", task, m)
		}
	}
}

// The cue itself: the refusal words are on every refusal and on no other row, and the rule's one
// error is the digit that needs no execution.
func TestSurfaceCues(t *testing.T) {
	for _, it := range smokeRows() {
		if got, want := surfaceRefusalRe.MatchString(it.Prompt), it.Category == CategoryNonExecution; got != want {
			t.Errorf("%s: refusal word %v, want %v: %q", it.ID, got, want, it.Prompt)
		}
		if surfaceExecute(it.Prompt) != it.ExecuteExpected && it.ID != "A-011" {
			t.Errorf("%s: the surface rule errs on %q", it.ID, it.Prompt)
		}
	}
	// a refusal is longer than every computation it pairs with
	rows := smokeRows()
	for _, d := range rows {
		if d.Category != CategoryNonExecution {
			continue
		}
		for _, c := range rows {
			if c.ParaphraseGroup == d.ParaphraseGroup && c.ExecuteExpected && c.PromptLength >= d.PromptLength {
				t.Errorf("%s (%d characters) is not longer than %s (%d)", d.ID, d.PromptLength, c.ID, c.PromptLength)
			}
		}
	}
}

func TestDeviationsAndReportName(t *testing.T) {
	if d := DefaultOptions.Deviations(); len(d) != 0 || DefaultOptions.ReportName() != "report" {
		t.Errorf("the committed options: %v, %s", d, DefaultOptions.ReportName())
	}
	o := DefaultOptions
	o.L2 = 0.1
	if d := o.Deviations(); len(d) != 1 || !strings.Contains(d[0], "L2 0.1 (committed 0.01)") {
		t.Errorf("L2 moved: %v", d)
	}
	if o.ReportName() == "report" {
		t.Error("a moved option gets the committed report's name")
	}
	for _, edit := range []func(*Options){func(o *Options) { o.Folds = 4 }, func(o *Options) { o.Shuffles = 1 }, func(o *Options) { o.Seed = 1 }} {
		o := DefaultOptions
		edit(&o)
		if len(o.Deviations()) != 1 || o.ReportName() == "report" {
			t.Errorf("%+v passes as committed", o)
		}
	}
	rows := smokeRows()
	folds, _ := GroupFolds(rows, 5, Seed)
	m := signalActivations(rows, 8, TaskExecute.Label, 2, 0, 1)
	o = DefaultOptions
	o.L2, o.Shuffles = 0.1, 1
	rep, err := evaluateFolds(rows, []NamedMatrix{{"x", m}}, folds, o)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(rep.Markdown(), "These are not the committed definitions: L2 0.1 (committed 0.01), shuffles 1 (committed 20).") {
		t.Errorf("the report does not say it deviates:\n%s", rep.Markdown())
	}
}

func writeCapture(t *testing.T, dir string, items []Item, dataset []byte, layers []int, dim int) {
	t.Helper()
	const nLayer = 50
	var ls, gs []string
	for _, n := range layers {
		m := signalActivations(items, dim, TaskExecute.Label, 2, 0.5, uint64(n))
		var b bytes.Buffer
		if err := WriteNpy(&b, m); err != nil {
			t.Fatal(err)
		}
		file := "layer-" + strconv.Itoa(n) + ".npy"
		if err := os.WriteFile(filepath.Join(dir, file), b.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
		ls = append(ls, `{"one_based":`+strconv.Itoa(n)+`,"zero_based":`+strconv.Itoa(n-1)+`,"node":"l_out-`+strconv.Itoa(n-1)+
			`","file":"`+file+`","sha256":"`+sha256Hex(b.Bytes())+`","dtype":"float32","shape":[`+strconv.Itoa(len(items))+`,`+strconv.Itoa(dim)+`]}`)
		il := strconv.Itoa(n - 1)
		readBy := "norm-" + strconv.Itoa(n)
		if n == nLayer {
			readBy = "norm"
		}
		gs = append(gs, `"l_out-`+il+`":{"sources":["node_9","ffn_inp-`+il+`"],"post_ffn":{"ops":["MUL","RMS_NORM","ADD"],"weight":"blk.`+il+
			`.post_ffw_norm.weight","norm":"norm-`+il+`","experts":["ffn_moe_out-`+il+`","ffn_shexp_out-`+il+`"]},"read_by":"`+readBy+`"}`)
	}
	var rows []string
	for _, it := range items {
		rows = append(rows, `{"id":"`+it.ID+`","n_tokens":10,"last_token":108}`)
	}
	js := `{"schema":"` + CaptureSchema + `","dataset_sha256":"` + sha256Hex(dataset) + `","logits_check":"equal","repeatable":true,"model_sha256":"abc",` +
		`"n_layer":` + strconv.Itoa(nLayer) + `,"last_tokens":[108],"template_is_released":true,"reasoning":"none","device":"CPU","n_ubatch_effective":21,"expert_cache_slots":128,` +
		`"graph_check":{` + strings.Join(gs, ",") + `},"rows":[` + strings.Join(rows, ",") + `],"layers":[` + strings.Join(ls, ",") + `]}`
	if err := os.WriteFile(filepath.Join(dir, "capture.json"), []byte(js), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestCaptureRoundTripAndReport(t *testing.T) {
	items := Generate(Seed)
	dataset, _ := MarshalJSONL(items)
	dir := t.TempDir()
	writeCapture(t, dir, items, dataset, []int{10, 25, 40, 50}, 64)
	c, ls, err := ReadCapture(dir, dataset, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(ls) != 4 || c.Raw["model_sha256"] != "abc" {
		t.Fatalf("%d layers, raw %v", len(ls), c.Raw["model_sha256"])
	}
	var named []NamedMatrix
	for _, l := range ls {
		named = append(named, NamedMatrix{"layer-" + strconv.Itoa(l.Layer.OneBased), l.Matrix})
	}
	rep, err := Evaluate(items, named, Options{Folds: 5, L2: 0.01, Shuffles: 2, Seed: Seed})
	if err != nil {
		t.Fatal(err)
	}
	rep.CaptureFacts = c.Facts()
	md := rep.Markdown()
	for _, want := range []string{"Small sample", "Decodability is not causal usefulness", "Surface cues", "no confidence interval", "It selects no layer", "| layer-10 | probe |", "| keyword rule | baseline |", "| surface rule | baseline |", "| prompt length | baseline |", "| majority | baseline |", "| bag of words | baseline |", "Pilot prompts", "| layer-50 | D-001 | execute | no |",
		"Logits against a context without a callback: equal.", "Repeated decode: equal.", "l_out-9, l_out-24, l_out-39, l_out-49 are each the residual add", "Last prompt token, the same for every prompt: 108.", "Chat template is the released one: yes.", "expert cache slots 128"} {
		if !strings.Contains(md, want) {
			t.Errorf("the report lacks %q:\n%s", want, md)
		}
	}
	if rep.NRows != 40 || rep.NGroups != 25 {
		t.Errorf("%d rows in %d groups", rep.NRows, rep.NGroups)
	}
}

// The report tells a template that is not the released one, or that it could not compare.
func TestCaptureFactsTemplate(t *testing.T) {
	no := false
	for released, want := range map[*bool]string{&no: "released one: **no**.", nil: "released one: not known"} {
		c := Capture{LastTokens: []int{1}, TemplateIsReleased: released}
		if got := strings.Join(c.Facts(), "\n"); !strings.Contains(got, want) {
			t.Errorf("%v: %q lacks %q", released, got, want)
		}
	}
}

func TestReadCaptureRefuses(t *testing.T) {
	items := Generate(Seed)
	dataset, _ := MarshalJSONL(items)
	replace := func(old, new string) func(string) {
		return func(dir string) {
			p := filepath.Join(dir, "capture.json")
			b, _ := os.ReadFile(p)
			if !bytes.Contains(b, []byte(old)) {
				t.Errorf("capture.json lacks %s", old)
			}
			os.WriteFile(p, bytes.ReplaceAll(b, []byte(old), []byte(new)), 0o644)
		}
	}
	for name, c := range map[string]struct {
		edit    func(dir string)
		dataset []byte
	}{
		"other dataset": {func(string) {}, append(bytes.Clone(dataset), ' ')},
		"file changed": {func(dir string) {
			p := filepath.Join(dir, "layer-10.npy")
			b, _ := os.ReadFile(p)
			b[len(b)-1] ^= 1
			os.WriteFile(p, b, 0o644)
		}, dataset},
		"logits changed": {func(dir string) {
			p := filepath.Join(dir, "capture.json")
			b, _ := os.ReadFile(p)
			os.WriteFile(p, bytes.Replace(b, []byte(`"logits_check":"equal"`), []byte(`"logits_check":"differ"`), 1), 0o644)
		}, dataset},
		"not repeatable": {func(dir string) {
			p := filepath.Join(dir, "capture.json")
			b, _ := os.ReadFile(p)
			os.WriteFile(p, bytes.Replace(b, []byte(`"repeatable":true`), []byte(`"repeatable":false`), 1), 0o644)
		}, dataset},
		// control M88: a capture that was not checked gets no report, whatever it did to the graph
		"logits not checked": {replace(`"logits_check":"equal"`, `"logits_check":"skipped"`), dataset},
		"no repeat":          {replace(`"repeatable":true`, `"repeatable":null`), dataset},
		// control M87: what the graph check found is read again, and a norm that is not there refuses
		"no graph check":      {replace(`"graph_check":{`, `"graph_check_gone":{`), dataset},
		"no post-FFN norm":    {replace(`"ops":["MUL","RMS_NORM","ADD"]`, `"ops":["ADD","ADD","MUL"]`), dataset},
		"other weight":        {replace(`"weight":"blk.9.post_ffw_norm.weight"`, `"weight":"blk.9.ffn_norm.weight"`), dataset},
		"other experts":       {replace(`"experts":["ffn_moe_out-9","ffn_shexp_out-9"]`, `"experts":["ffn_moe_out-9"]`), dataset},
		"other reader":        {replace(`"read_by":"norm-10"`, `"read_by":"norm"`), dataset},
		"several last tokens": {replace(`"last_tokens":[108]`, `"last_tokens":[108,109]`), dataset},
		"no last tokens":      {replace(`"last_tokens":[108]`, `"last_tokens":[]`), dataset},
		"a row's last token":  {replace(`"id":"A-001","n_tokens":10,"last_token":108`, `"id":"A-001","n_tokens":10,"last_token":109`), dataset},
		"numbering": {func(dir string) {
			p := filepath.Join(dir, "capture.json")
			b, _ := os.ReadFile(p)
			os.WriteFile(p, bytes.Replace(b, []byte(`"node":"l_out-9"`), []byte(`"node":"l_out-10"`), 1), 0o644)
		}, dataset},
		"row order": {func(dir string) {
			p := filepath.Join(dir, "capture.json")
			b, _ := os.ReadFile(p)
			os.WriteFile(p, bytes.Replace(b, []byte(`"id":"A-001"`), []byte(`"id":"A-002"`), 1), 0o644)
		}, dataset},
	} {
		dir := t.TempDir()
		writeCapture(t, dir, items, dataset, []int{10}, 8)
		c.edit(dir)
		if _, _, err := ReadCapture(dir, c.dataset, items); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

func TestNpyRefuses(t *testing.T) {
	m := Matrix{Rows: 1, Dim: 2, Data: []float32{1, float32(math.Inf(1))}}
	var b bytes.Buffer
	if err := WriteNpy(&b, m); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadNpy(&b); err == nil {
		t.Error("a non-finite activation passed")
	}
	if _, err := ReadNpy(strings.NewReader("not npy at all")); err == nil {
		t.Error("garbage passed")
	}
	m.Data[1] = 2
	b.Reset()
	WriteNpy(&b, m)
	got, err := ReadNpy(bytes.NewReader(b.Bytes()))
	if err != nil || got.Rows != 1 || got.Dim != 2 || got.Data[1] != 2 {
		t.Fatalf("round trip: %+v, %v", got, err)
	}
	if _, err := ReadNpy(bytes.NewReader(append(b.Bytes(), 0))); err == nil {
		t.Error("trailing data passed")
	}
}

// fitSoftmax's momentum needs a positive L2, or it is NaN and the report looks fine with invalid scores.
func TestEvaluateRefusesAnL2ThatIsNotPositive(t *testing.T) {
	for _, l2 := range []float64{0, -0.01, math.NaN(), math.Inf(1)} {
		_, err := Evaluate(nil, nil, Options{Folds: 5, L2: l2, Shuffles: 2, Seed: Seed})
		if err == nil || !strings.Contains(err.Error(), "L2 must be positive") {
			t.Errorf("L2 %g: %v", l2, err)
		}
	}
}

// The gradient in alpha's coordinates is the primal gradient in w: an alpha step with r/n + l2 a is a
// step on w = X'a with X'(r/n) + l2 w. So at the fit's optimum the primal objective's gradient
// vanishes in the weights, and l2 regularizes |w|^2 as documented.
func TestFitIsThePrimalOptimum(t *testing.T) {
	x := [][]float64{{1, 0, 3}, {0, 2, 0}, {1, 1, 0}, {4, 0, 1}, {0, 1, 2}, {2, 2, 2}}
	y := []int{0, 1, 2, 0, 1, 2}
	const k, l2 = 3, 0.01
	n, d := len(x), len(x[0])
	kern := kernel(x, x)
	m := fitSoftmax(kern, y, k, l2)
	if !m.converged {
		t.Fatal("not converged")
	}
	probs := m.predict(kern, n, n, k)
	for c := range k {
		for f := range d {
			var grad, w float64
			for i := range n {
				r := probs[i*k+c]
				if y[i] == c {
					r--
				}
				grad += x[i][f] * r / float64(n)
				w += x[i][f] * m.alpha[i*k+c]
			}
			if g := grad + l2*w; math.Abs(g) > 1e-6 {
				t.Fatalf("class %d feature %d: primal gradient %g", c, f, g)
			}
		}
	}
}
