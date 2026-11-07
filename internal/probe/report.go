package probe

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
)

// Task is one probe target.
type Task struct {
	Name    string
	Classes []string
	Label   func(Item) int
	Rule    func(Item) int // the fixed keyword rule's prediction
	Surface func(Item) int // the surface rule's prediction
}

// TaskExecute is the binary gate: execute or not.
var TaskExecute = Task{
	Name: "execute", Classes: []string{"no", "execute"},
	Label: func(it Item) int {
		if it.ExecuteExpected {
			return 1
		}
		return 0
	},
	Rule: func(it Item) int {
		if keywordFamily(it.Prompt) != FamilyNone {
			return 1
		}
		return 0
	},
	Surface: func(it Item) int {
		if surfaceExecute(it.Prompt) {
			return 1
		}
		return 0
	},
}

// TaskFamily is the multiclass controller: which class of operation is needed.
var TaskFamily = Task{
	Name: "operation_family", Classes: Families,
	Label:   func(it Item) int { return slices.Index(Families, it.OperationFamily) },
	Rule:    func(it Item) int { return slices.Index(Families, keywordFamily(it.Prompt)) },
	Surface: func(it Item) int { return slices.Index(Families, surfaceFamily(it.Prompt)) },
}

// foldKernels are one fold's precomputed inner products.
type foldKernels struct {
	train, test   []int
	ktrain, ktest []float64
}

func kernelsOf(f Features, folds Folds, n int) []foldKernels {
	out := make([]foldKernels, len(folds))
	for i, test := range folds {
		train := trainRows(n, test)
		ktrain, ktest := f.Kernels(train, test)
		out[i] = foldKernels{train, test, ktrain, ktest}
	}
	return out
}

// outOfFold is every row's prediction by the fold that held it out.
type outOfFold struct {
	prob       []float64 // rows x classes
	pred       []int
	maxIters   int
	unconverge int
}

func crossValidate(fk []foldKernels, labels []int, k int, l2 float64) (outOfFold, error) {
	out := outOfFold{prob: make([]float64, len(labels)*k), pred: make([]int, len(labels))}
	for f, fold := range fk {
		y := make([]int, len(fold.train))
		seen := make([]bool, k)
		for i, r := range fold.train {
			y[i] = labels[r]
			seen[y[i]] = true
		}
		if slices.Contains(seen, false) {
			return out, fmt.Errorf("fold %d: a class has no training row", f)
		}
		m := fitSoftmax(fold.ktrain, y, k, l2)
		out.maxIters = max(out.maxIters, m.iters)
		if !m.converged {
			out.unconverge++
		}
		p := m.predict(fold.ktest, len(fold.test), len(fold.train), k)
		for i, r := range fold.test {
			copy(out.prob[r*k:(r+1)*k], p[i*k:(i+1)*k])
			out.pred[r] = argmax(p[i*k : (i+1)*k])
		}
	}
	return out, nil
}

func argmax(p []float64) int {
	best := 0
	for i, v := range p {
		if v > p[best] {
			best = i
		}
	}
	return best
}

// Metrics are held-out scores over all folds. Pointers are null where a baseline has no such score.
type Metrics struct {
	Accuracy            float64  `json:"accuracy"`
	BalancedAccuracy    float64  `json:"balanced_accuracy"`
	MacroF1             float64  `json:"macro_f1"`
	Precision           *float64 `json:"execute_precision,omitempty"`
	Recall              *float64 `json:"execute_recall,omitempty"`
	HardNegativeCorrect int      `json:"hard_negative_correct"`
	HardNegatives       int      `json:"hard_negatives"`
	// ParaphraseAgree is how many same-label row pairs of one group get one prediction.
	ParaphraseAgree int      `json:"paraphrase_agree"`
	ParaphrasePairs int      `json:"paraphrase_pairs"`
	Brier           *float64 `json:"brier,omitempty"`
	ECE             *float64 `json:"ece,omitempty"`
}

func scores(items []Item, labels, pred []int, prob []float64, task Task) Metrics {
	k := len(task.Classes)
	var m Metrics
	tp, fp, fn := make([]int, k), make([]int, k), make([]int, k)
	correct := 0
	for i, y := range labels {
		if pred[i] == y {
			correct++
			tp[y]++
		} else {
			fp[pred[i]]++
			fn[y]++
		}
		if items[i].HardNegative {
			m.HardNegatives++
			if pred[i] == y {
				m.HardNegativeCorrect++
			}
		}
	}
	m.Accuracy = float64(correct) / float64(len(labels))
	present := 0
	for c := range k {
		if tp[c]+fn[c] == 0 {
			continue
		}
		present++
		recall := float64(tp[c]) / float64(tp[c]+fn[c])
		m.BalancedAccuracy += recall
		if tp[c] > 0 {
			precision := float64(tp[c]) / float64(tp[c]+fp[c])
			m.MacroF1 += 2 * precision * recall / (precision + recall)
		}
	}
	m.BalancedAccuracy /= float64(present)
	m.MacroF1 /= float64(present)
	if task.Name == TaskExecute.Name {
		var r, p float64
		if tp[1]+fn[1] > 0 {
			r = float64(tp[1]) / float64(tp[1]+fn[1])
		}
		if tp[1]+fp[1] > 0 {
			p = float64(tp[1]) / float64(tp[1]+fp[1])
		}
		m.Precision, m.Recall = &p, &r
	}
	for i := range items {
		for j := i + 1; j < len(items); j++ {
			if items[i].ParaphraseGroup == items[j].ParaphraseGroup && labels[i] == labels[j] {
				m.ParaphrasePairs++
				if pred[i] == pred[j] {
					m.ParaphraseAgree++
				}
			}
		}
	}
	if prob != nil {
		var brier float64
		const bins = 5
		var binConf, binHit [bins]float64
		var binN [bins]int
		for i, y := range labels {
			p := prob[i*k : (i+1)*k]
			for c, v := range p {
				e := v
				if c == y {
					e = v - 1
				}
				brier += e * e
			}
			top := p[argmax(p)]
			b := min(int(top*bins), bins-1)
			binConf[b] += top
			if argmax(p) == y {
				binHit[b]++
			}
			binN[b]++
		}
		brier /= float64(len(labels))
		var ece float64
		for b := range bins {
			if binN[b] > 0 {
				ece += math.Abs(binConf[b]-binHit[b]) / float64(len(labels))
			}
		}
		m.Brier, m.ECE = &brier, &ece
	}
	return m
}

// Shuffled is the control of one probe: the same probe on permuted labels.
type Shuffled struct {
	Runs           int     `json:"runs"`
	MeanBalanced   float64 `json:"mean_balanced_accuracy"`
	MaxBalanced    float64 `json:"max_balanced_accuracy"`
	ChanceBalanced float64 `json:"chance_balanced_accuracy"`
	ShareAtOrAbove float64 `json:"share_of_runs_at_or_above_real"`
}

// shuffled runs the probe on permuted labels. Labels that are permuted carry no information about
// the activations, so a real gap between the probe and this control is the least a score must show.
func shuffled(fk []foldKernels, items []Item, labels []int, task Task, l2 float64, runs int, seed uint64, real float64) (Shuffled, error) {
	k := len(task.Classes)
	s := Shuffled{Runs: runs, ChanceBalanced: 1 / float64(k)}
	r := rand.New(rand.NewPCG(seed, 0x5eed))
	perm := slices.Clone(labels)
	atOrAbove := 0
	for run, tries := 0, 0; run < runs; tries++ {
		if tries > 100*runs {
			return s, fmt.Errorf("%s: %d label permutations left a class without training rows", task.Name, tries)
		}
		r.Shuffle(len(perm), func(i, j int) { perm[i], perm[j] = perm[j], perm[i] })
		oof, err := crossValidate(fk, perm, k, l2)
		if err != nil { // a permutation can empty a class of one fold's training rows
			continue
		}
		run++
		b := scores(items, perm, oof.pred, nil, task).BalancedAccuracy
		s.MeanBalanced += b
		s.MaxBalanced = max(s.MaxBalanced, b)
		if b >= real {
			atOrAbove++
		}
	}
	if runs > 0 {
		s.MeanBalanced /= float64(runs)
		s.ShareAtOrAbove = float64(atOrAbove) / float64(runs)
	}
	return s, nil
}

// Options fix the readout. The values in DefaultOptions are definitions: they are committed before
// a real capture and must not move afterwards.
type Options struct {
	Folds    int     `json:"folds"`
	L2       float64 `json:"l2"`
	Shuffles int     `json:"shuffles"`
	Seed     uint64  `json:"seed"`
}

// DefaultOptions are the definitions of the smoke readout.
var DefaultOptions = Options{Folds: 5, L2: 0.01, Shuffles: 20, Seed: Seed}

// Deviations says how these options differ from the committed definitions; empty if they do not.
func (o Options) Deviations() []string {
	var out []string
	d := DefaultOptions
	if o.Folds != d.Folds {
		out = append(out, fmt.Sprintf("folds %d (committed %d)", o.Folds, d.Folds))
	}
	if o.L2 != d.L2 {
		out = append(out, fmt.Sprintf("L2 %g (committed %g)", o.L2, d.L2))
	}
	if o.Shuffles != d.Shuffles {
		out = append(out, fmt.Sprintf("shuffles %d (committed %d)", o.Shuffles, d.Shuffles))
	}
	if o.Seed != d.Seed {
		out = append(out, fmt.Sprintf("seed %d (committed %d)", o.Seed, d.Seed))
	}
	return out
}

// ReportName is the report's file name without its extension. Only the committed definitions give
// plain "report", so that a report under that name is the committed one.
func (o Options) ReportName() string {
	if len(o.Deviations()) == 0 {
		return "report"
	}
	return fmt.Sprintf("report-deviation-folds%d-l2%g-shuffles%d-seed%d", o.Folds, o.L2, o.Shuffles, o.Seed)
}

// Row is one line of a task's table.
type Row struct {
	Name     string    `json:"name"`
	Kind     string    `json:"kind"` // probe or baseline
	Task     string    `json:"task"`
	Metrics  Metrics   `json:"metrics"`
	Shuffled *Shuffled `json:"shuffled,omitempty"`
	// Unconverged counts the folds whose solver stopped at its iteration limit.
	Unconverged int `json:"unconverged_folds"`
}

// PilotRow is one layer's prediction for a pilot prompt: trained on the whole smoke split, whose
// groups the pilot does not share. One prompt per category, so this checks the pipeline only.
type PilotRow struct {
	Layer   string `json:"layer"`
	ID      string `json:"id"`
	Task    string `json:"task"`
	Label   string `json:"label"`
	Predict string `json:"predict"`
}

// Report is the readout of one capture.
type Report struct {
	NRows   int        `json:"n_rows"`
	NGroups int        `json:"n_groups"`
	Options Options    `json:"options"`
	Folds   Folds      `json:"folds"`
	Results []Row      `json:"results"`
	Pilot   []PilotRow `json:"pilot"`
	Caveats []string   `json:"caveats"`
	// Deviations are the options that differ from the committed definitions.
	Deviations []string `json:"deviations,omitempty"`
	// CaptureFacts are what the capture was checked against, for reading; Capture is all of it.
	CaptureFacts []string `json:"capture_facts,omitempty"`
	Capture      any      `json:"capture,omitempty"`
}

// Caveats every report carries. They are part of the result: nothing here supports a layer choice.
func caveatSmallSample(rows, groups int) string {
	return fmt.Sprintf("Small sample: this is pipeline validation, not evidence. With %d rows in %d groups a difference between layers is noise until shown otherwise, so no layer is selected here and no confidence interval is given: the sample does not support one. A choice needs hundreds to thousands of prompts with a held-out, template-disjoint test set.", rows, groups)
}

var (
	CaveatDecodable = "Decodability is not causal usefulness: a linear probe that reads a label from an activation shows that the information is there at that layer, not that the model uses it there, nor that injecting a result at that layer would change its behavior correctly. Only an intervention can show that, and this report runs none."
	CaveatSurface   = "Surface cues: on these rows a digit and the absence of the refusal words (aber, ohne, but, without) give the execute label on 39 of 40 smoke rows, and a refusal is longer than its partner. The surface rule and the prompt length are in the tables, and a probe that does not beat them has shown nothing the text's surface does not show. Beating the bag of words is not enough, and the rule was written after seeing the rows. These rows cannot separate a probe that reads what a prompt asks from one that reads those cues."
	CaveatPolicy    = "The labels are a policy (" + LabelPolicy + "): execute_expected says execution is helpful, not that it is strictly necessary; execution_need keeps the two apart."
)

// Evaluate runs the probes of every layer and the baselines over the rows, with the same folds.
// layers maps names to one matrix per dataset row of items; rows outside split smoke are the
// pilot, held out of the folds.
func Evaluate(items []Item, layers []NamedMatrix, opt Options) (*Report, error) {
	var smoke, pilot []int
	for i, it := range items {
		if it.Split == SplitSmoke {
			smoke = append(smoke, i)
		} else {
			pilot = append(pilot, i)
		}
	}
	rows := make([]Item, len(smoke))
	for i, r := range smoke {
		rows[i] = items[r]
	}
	folds, err := GroupFolds(rows, opt.Folds, opt.Seed)
	if err != nil {
		return nil, err
	}
	rep, err := evaluateFolds(rows, selectAll(layers, smoke), folds, opt)
	if err != nil {
		return nil, err
	}
	if len(pilot) > 0 {
		if rep.Pilot, err = pilotReadout(items, smoke, pilot, layers, opt); err != nil {
			return nil, err
		}
	}
	return rep, nil
}

// NamedMatrix is a layer's activations under its report name.
type NamedMatrix struct {
	Name   string
	Matrix Matrix
}

func selectAll(layers []NamedMatrix, rows []int) []NamedMatrix {
	out := make([]NamedMatrix, len(layers))
	for i, l := range layers {
		out[i] = NamedMatrix{l.Name, l.Matrix.Select(rows)}
	}
	return out
}

// evaluateFolds refuses a split that leaks, whoever made it.
func evaluateFolds(rows []Item, layers []NamedMatrix, folds Folds, opt Options) (*Report, error) {
	if err := CheckFolds(rows, folds); err != nil {
		return nil, err
	}
	groups := map[string]bool{}
	for _, it := range rows {
		groups[it.ParaphraseGroup] = true
	}
	rep := &Report{NRows: len(rows), NGroups: len(groups), Options: opt, Folds: folds, Deviations: opt.Deviations(),
		Caveats: []string{caveatSmallSample(len(rows), len(groups)), CaveatDecodable, CaveatSurface, CaveatPolicy}}
	length := kernelsOf(lengthFeature(rows), folds, len(rows))
	bow := kernelsOf(BagOfWords{rows}, folds, len(rows))
	for _, task := range []Task{TaskExecute, TaskFamily} {
		k := len(task.Classes)
		labels := make([]int, len(rows))
		rule, surface := make([]int, len(rows)), make([]int, len(rows))
		for i, it := range rows {
			labels[i], rule[i], surface[i] = task.Label(it), task.Rule(it), task.Surface(it)
		}
		maj := make([]int, len(rows))
		for _, test := range folds {
			counts := make([]int, k)
			for _, r := range trainRows(len(rows), test) {
				counts[labels[r]]++
			}
			top := 0
			for c, n := range counts {
				if n > counts[top] {
					top = c
				}
			}
			for _, r := range test {
				maj[r] = top
			}
		}
		rep.Results = append(rep.Results,
			Row{Name: "majority", Kind: "baseline", Task: task.Name, Metrics: scores(rows, labels, maj, nil, task)},
			Row{Name: "keyword rule", Kind: "baseline", Task: task.Name, Metrics: scores(rows, labels, rule, nil, task)},
			Row{Name: "surface rule", Kind: "baseline", Task: task.Name, Metrics: scores(rows, labels, surface, nil, task)})
		byLength, err := crossValidate(length, labels, k, opt.L2)
		if err != nil {
			return nil, fmt.Errorf("prompt length, %s: %w", task.Name, err)
		}
		rep.Results = append(rep.Results, Row{Name: "prompt length", Kind: "baseline", Task: task.Name,
			Metrics: scores(rows, labels, byLength.pred, byLength.prob, task), Unconverged: byLength.unconverge})
		oof, err := crossValidate(bow, labels, k, opt.L2)
		if err != nil {
			return nil, fmt.Errorf("bag of words, %s: %w", task.Name, err)
		}
		rep.Results = append(rep.Results, Row{Name: "bag of words", Kind: "baseline", Task: task.Name,
			Metrics: scores(rows, labels, oof.pred, oof.prob, task), Unconverged: oof.unconverge})
		for _, l := range layers {
			fk := kernelsOf(l.Matrix, folds, len(rows))
			oof, err := crossValidate(fk, labels, k, opt.L2)
			if err != nil {
				return nil, fmt.Errorf("%s, %s: %w", l.Name, task.Name, err)
			}
			m := scores(rows, labels, oof.pred, oof.prob, task)
			s, err := shuffled(fk, rows, labels, task, opt.L2, opt.Shuffles, opt.Seed, m.BalancedAccuracy)
			if err != nil {
				return nil, err
			}
			rep.Results = append(rep.Results, Row{Name: l.Name, Kind: "probe", Task: task.Name, Metrics: m, Shuffled: &s, Unconverged: oof.unconverge})
		}
	}
	return rep, nil
}

func pilotReadout(items []Item, smoke, pilot []int, layers []NamedMatrix, opt Options) ([]PilotRow, error) {
	var out []PilotRow
	for _, l := range layers {
		ktrain, ktest := l.Matrix.Kernels(smoke, pilot)
		for _, task := range []Task{TaskExecute, TaskFamily} {
			y := make([]int, len(smoke))
			for i, r := range smoke {
				y[i] = task.Label(items[r])
			}
			k := len(task.Classes)
			p := fitSoftmax(ktrain, y, k, opt.L2).predict(ktest, len(pilot), len(smoke), k)
			for i, r := range pilot {
				out = append(out, PilotRow{Layer: l.Name, ID: items[r].ID, Task: task.Name,
					Label: task.Classes[task.Label(items[r])], Predict: task.Classes[argmax(p[i*k:(i+1)*k])]})
			}
		}
	}
	return out, nil
}

// JSON is the report as indented JSON.
func (r *Report) JSON() ([]byte, error) {
	b, err := json.MarshalIndent(r, "", "  ")
	return append(b, '\n'), err
}

func opt3(p *float64) string {
	if p == nil {
		return "n/a"
	}
	return fmt.Sprintf("%.3f", *p)
}

// Markdown is the report for reading. It lists every layer and baseline and ranks none.
func (r *Report) Markdown() string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Layer probe readout\n\n**Result: pipeline validation on %d rows in %d paraphrase groups, %d grouped folds. It selects no layer.**\n\n", r.NRows, r.NGroups, r.Options.Folds)
	if len(r.Deviations) > 0 {
		fmt.Fprintf(&b, "**These are not the committed definitions: %s.**\n\n", strings.Join(r.Deviations, ", "))
	}
	for _, c := range r.Caveats {
		fmt.Fprintf(&b, "- %s\n", c)
	}
	fmt.Fprintf(&b, "\nDefinitions: L2 %g on standardized activations scaled to unit expected squared norm, %d label shuffles per probe, seed %d. All scores are held out by grouped cross-validation.\n", r.Options.L2, r.Options.Shuffles, r.Options.Seed)
	for _, task := range []Task{TaskExecute, TaskFamily} {
		fmt.Fprintf(&b, "\n## %s (%s)\n\n", task.Name, strings.Join(task.Classes, ", "))
		fmt.Fprintf(&b, "| Row | Kind | Accuracy | Balanced | Macro F1 | Execute P | Execute R | Hard negatives | Paraphrase agreement | Brier | ECE | Shuffled balanced, mean (max) |\n")
		fmt.Fprintf(&b, "|---|---|---|---|---|---|---|---|---|---|---|---|\n")
		for _, row := range r.Results {
			if row.Task != task.Name {
				continue
			}
			m := row.Metrics
			shuf := "n/a"
			if row.Shuffled != nil {
				shuf = fmt.Sprintf("%.3f (%.3f), chance %.3f", row.Shuffled.MeanBalanced, row.Shuffled.MaxBalanced, row.Shuffled.ChanceBalanced)
			}
			fmt.Fprintf(&b, "| %s | %s | %.3f | %.3f | %.3f | %s | %s | %d/%d | %d/%d | %s | %s | %s |\n",
				row.Name, row.Kind, m.Accuracy, m.BalancedAccuracy, m.MacroF1, opt3(m.Precision), opt3(m.Recall),
				m.HardNegativeCorrect, m.HardNegatives, m.ParaphraseAgree, m.ParaphrasePairs, opt3(m.Brier), opt3(m.ECE), shuf)
			if row.Unconverged > 0 {
				fmt.Fprintf(&b, "| %s: %d folds did not converge | | | | | | | | | | | |\n", row.Name, row.Unconverged)
			}
		}
	}
	if len(r.CaptureFacts) > 0 {
		b.WriteString("\n## Capture\n\n")
		for _, f := range r.CaptureFacts {
			fmt.Fprintf(&b, "- %s\n", f)
		}
	}
	if len(r.Pilot) > 0 {
		b.WriteString("\n## Pilot prompts\n\nTrained on the smoke split, whose groups the pilot does not share. One prompt per category: this checks the pipeline.\n\n| Layer | Prompt | Task | Label | Prediction |\n|---|---|---|---|---|\n")
		for _, p := range r.Pilot {
			fmt.Fprintf(&b, "| %s | %s | %s | %s | %s |\n", p.Layer, p.ID, p.Task, p.Label, p.Predict)
		}
	}
	return b.String()
}
