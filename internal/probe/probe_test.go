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
	if bow := get("execute", "bag of words").Metrics; bow.BalancedAccuracy <= 0.5 {
		t.Errorf("bag of words: balanced accuracy %.3f", bow.BalancedAccuracy)
	}
}

func writeCapture(t *testing.T, dir string, items []Item, dataset []byte, layers []int, dim int) {
	t.Helper()
	var ls []string
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
	}
	var rows []string
	for _, it := range items {
		rows = append(rows, `{"id":"`+it.ID+`","n_tokens":10}`)
	}
	js := `{"schema":"` + CaptureSchema + `","dataset_sha256":"` + sha256Hex(dataset) + `","logits_check":"equal","repeatable":true,"model_sha256":"abc","rows":[` +
		strings.Join(rows, ",") + `],"layers":[` + strings.Join(ls, ",") + `]}`
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
	md := rep.Markdown()
	for _, want := range []string{"Small sample", "Decodability is not causal usefulness", "no confidence interval", "It selects no layer", "| layer-10 | probe |", "| keyword rule | baseline |", "| majority | baseline |", "| bag of words | baseline |", "Pilot prompts", "| layer-50 | D-001 | execute | no |"} {
		if !strings.Contains(md, want) {
			t.Errorf("the report lacks %q:\n%s", want, md)
		}
	}
	if rep.NRows != 40 || rep.NGroups != 25 {
		t.Errorf("%d rows in %d groups", rep.NRows, rep.NGroups)
	}
}

func TestReadCaptureRefuses(t *testing.T) {
	items := Generate(Seed)
	dataset, _ := MarshalJSONL(items)
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
