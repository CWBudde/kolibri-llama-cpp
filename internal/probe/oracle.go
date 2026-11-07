package probe

import (
	"fmt"
	"math/big"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"unicode"
)

// The oracle checks every expected answer without the generator: it reads a row's numbers from the
// prompt's text and computes the answer with other algorithms (math/big's Mul, Exp, MulRange, GCD
// and ProbablyPrime, fast doubling, an insertion sort). It shares no code with generate.go.

var (
	groupedNumRe = regexp.MustCompile(`\d{1,3}(?:[., ]\d{3})+|\d+`)
	plainNumRe   = regexp.MustCompile(`\d+`)
	fibRe        = regexp.MustCompile(`F\((\d+)\)`)
	thresholdRe  = regexp.MustCompile(`(?:greater than|größer als) (\d+)`)
	// forbidRe is how a category D prompt forbids execution.
	forbidRe = regexp.MustCompile(`(?i)\b(?:ohne|nicht|nichts|without|do not)\b`)
)

// facts are the oracle's own answers for the factual prompts, by prompt.
var facts = map[string]string{
	"Was ist die Hauptstadt von Frankreich?":     "Paris",
	"Was ist die Hauptstadt von Spanien?":        "Madrid",
	"What is the capital of Spain?":              "Madrid",
	"In welchem Jahr fiel die Berliner Mauer?":   "1989",
	"Wie viele Planeten hat unser Sonnensystem?": "8",
}

func bigFrom(s string) (*big.Int, error) {
	digits := strings.Map(func(r rune) rune {
		if unicode.IsDigit(r) {
			return r
		}
		return -1
	}, s)
	n, ok := new(big.Int).SetString(digits, 10)
	if !ok {
		return nil, fmt.Errorf("number %q", s)
	}
	return n, nil
}

// operands reads the n numbers of an arithmetic prompt, which may have thousands separators.
func operands(prompt string, n int) ([]*big.Int, error) {
	found := groupedNumRe.FindAllString(prompt, -1)
	if len(found) != n {
		return nil, fmt.Errorf("%d numbers in %q, want %d", len(found), prompt, n)
	}
	out := make([]*big.Int, n)
	for i, s := range found {
		var err error
		if out[i], err = bigFrom(s); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func listInts(prompt string) ([]int, error) {
	var out []int
	for _, s := range plainNumRe.FindAllString(prompt, -1) {
		v, err := strconv.Atoi(s)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

func intsText(l []int) string {
	parts := make([]string, len(l))
	for i, v := range l {
		parts[i] = strconv.Itoa(v)
	}
	return strings.Join(parts, ", ")
}

// fibFastDoubling returns F(n) with F(1) = F(2) = 1.
func fibFastDoubling(n int) *big.Int {
	a, b := big.NewInt(0), big.NewInt(1)
	for i := strconv.IntSize - 1; i >= 0; i-- {
		// F(2k) = F(k)(2F(k+1) - F(k)), F(2k+1) = F(k)^2 + F(k+1)^2
		twoB := new(big.Int).Lsh(b, 1)
		c := new(big.Int).Mul(a, twoB.Sub(twoB, a))
		d := new(big.Int).Add(new(big.Int).Mul(a, a), new(big.Int).Mul(b, b))
		if n>>i&1 == 1 {
			a, b = d, new(big.Int).Add(c, d)
		} else {
			a, b = c, d
		}
	}
	return a
}

// Oracle computes a compute prompt's answer from its text and operation alone.
func Oracle(operation, prompt string) (string, error) {
	switch operation {
	case "multiplication":
		n, err := operands(prompt, 2)
		if err != nil {
			return "", err
		}
		return new(big.Int).Mul(n[0], n[1]).String(), nil
	case "exponentiation":
		n, err := operands(prompt, 2)
		if err != nil {
			return "", err
		}
		return new(big.Int).Exp(n[0], n[1], nil).String(), nil
	case "factorial":
		n, err := operands(prompt, 1)
		if err != nil {
			return "", err
		}
		return new(big.Int).MulRange(1, n[0].Int64()).String(), nil
	case "gcd":
		n, err := operands(prompt, 2)
		if err != nil {
			return "", err
		}
		return new(big.Int).GCD(nil, nil, n[0], n[1]).String(), nil
	case "primality":
		n, err := operands(prompt, 1)
		if err != nil {
			return "", err
		}
		return strconv.FormatBool(n[0].ProbablyPrime(32)), nil
	case "fibonacci":
		m := fibRe.FindStringSubmatch(prompt)
		if m == nil {
			return "", fmt.Errorf("no F(n) in %q", prompt)
		}
		n, err := strconv.Atoi(m[1])
		if err != nil {
			return "", err
		}
		return fibFastDoubling(n).String(), nil
	case "sorting":
		l, err := listInts(prompt)
		if err != nil {
			return "", err
		}
		for i := 1; i < len(l); i++ { // insertion sort, ascending
			for j := i; j > 0 && l[j] < l[j-1]; j-- {
				l[j], l[j-1] = l[j-1], l[j]
			}
		}
		lower := strings.ToLower(prompt)
		switch {
		case strings.Contains(lower, "absteigend"), strings.Contains(lower, "descending"):
			slices.Reverse(l)
		case strings.Contains(lower, "aufsteigend"), strings.Contains(lower, "ascending"):
		default:
			return "", fmt.Errorf("no sort direction in %q", prompt)
		}
		return intsText(l), nil
	case "filtering_even":
		l, err := listInts(prompt)
		if err != nil {
			return "", err
		}
		return intsText(slices.DeleteFunc(l, func(v int) bool { return v&1 == 1 })), nil
	case "filtering_greater":
		m := thresholdRe.FindStringSubmatch(prompt)
		if m == nil {
			return "", fmt.Errorf("no threshold in %q", prompt)
		}
		t, err := strconv.Atoi(m[1])
		if err != nil {
			return "", err
		}
		l, err := listInts(strings.Replace(prompt, m[0], "", 1))
		if err != nil {
			return "", err
		}
		return intsText(slices.DeleteFunc(l, func(v int) bool { return v <= t })), nil
	}
	return "", fmt.Errorf("oracle: operation %q", operation)
}

// VerifyAnswers checks every row against the oracle and the facts: an exact answer must be what
// the oracle computes from the prompt, and a row without one must have no answer to compute.
func VerifyAnswers(items []Item) []error {
	var errs []error
	fail := func(it Item, format string, a ...any) {
		errs = append(errs, fmt.Errorf("%s: %s", it.ID, fmt.Sprintf(format, a...)))
	}
	for _, it := range items {
		switch it.Category {
		case CategoryArithmetic, CategoryAlgorithm:
			want, err := Oracle(it.Operation, it.Prompt)
			if err != nil {
				fail(it, "%v", err)
			} else if it.ExpectedAnswerOrOracle != want {
				fail(it, "expected answer %q, the oracle computes %q", it.ExpectedAnswerOrOracle, want)
			}
		case CategoryNoExecution:
			if want, ok := facts[it.Prompt]; ok {
				if it.ExpectedAnswerOrOracle != want {
					fail(it, "expected answer %q, the oracle's fact is %q", it.ExpectedAnswerOrOracle, want)
				}
			} else if it.ExpectedAnswerOrOracle != NoExactAnswer {
				fail(it, "expected answer %q, no fact in the oracle", it.ExpectedAnswerOrOracle)
			}
		case CategoryNonExecution:
			if it.ExpectedAnswerOrOracle != NoExactAnswer {
				fail(it, "a prompt that forbids execution has answer %q", it.ExpectedAnswerOrOracle)
			}
			if !forbidRe.MatchString(it.Prompt) {
				fail(it, "no words that forbid execution in %q", it.Prompt)
			}
		default:
			fail(it, "category %q", it.Category)
		}
	}
	return errs
}

// nearDuplicateAt is the masked word overlap at which prompts of two groups count as near-identical.
const nearDuplicateAt = 0.8

// maskedWords are a prompt's lowercased words with every number masked, so that two prompts from
// one number template look the same whatever their numbers.
func maskedWords(prompt string) map[string]bool {
	out := map[string]bool{}
	for w := range strings.FieldsFuncSeq(groupedNumRe.ReplaceAllString(strings.ToLower(prompt), "#"), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '#'
	}) {
		out[w] = true
	}
	return out
}

func overlap(a, b map[string]bool) float64 {
	both := 0
	for w := range a {
		if b[w] {
			both++
		}
	}
	return float64(both) / float64(len(a)+len(b)-both)
}

// VerifyStructure checks what the dataset promises beyond its answers: the counts, balance and
// labels, unique prompts, and that nothing shared between two groups can leak across a fold: no
// template, no numeric seed and no near-identical prompt.
func VerifyStructure(items []Item) []error {
	var errs []error
	failf := func(format string, a ...any) { errs = append(errs, fmt.Errorf(format, a...)) }
	type splitCat struct{ split, category string }
	counts := map[splitCat]int{}
	families := map[string]int{}
	positives := 0
	ids, prompts := map[string]bool{}, map[string]bool{}
	templateGroup, seedGroup := map[string]string{}, map[uint32]string{}
	pilotGroups, smokeGroups := map[string]bool{}, map[string]bool{}
	pairedOps := map[string]map[string]bool{}
	for _, it := range items {
		counts[splitCat{it.Split, it.Category}]++
		if ids[it.ID] {
			failf("%s: duplicate id", it.ID)
		}
		ids[it.ID] = true
		if prompts[it.Prompt] {
			failf("%s: duplicate prompt %q", it.ID, it.Prompt)
		}
		prompts[it.Prompt] = true
		if it.PromptLength != len([]rune(it.Prompt)) {
			failf("%s: prompt_length %d", it.ID, it.PromptLength)
		}
		if g, ok := templateGroup[it.TemplateID]; ok && g != it.ParaphraseGroup {
			failf("template %s in groups %s and %s", it.TemplateID, g, it.ParaphraseGroup)
		}
		templateGroup[it.TemplateID] = it.ParaphraseGroup
		if it.NumericSeed != nil {
			if g, ok := seedGroup[*it.NumericSeed]; ok && g != it.ParaphraseGroup {
				failf("numeric seed %d in groups %s and %s", *it.NumericSeed, g, it.ParaphraseGroup)
			}
			seedGroup[*it.NumericSeed] = it.ParaphraseGroup
		}
		execute, family, need := false, FamilyNone, NeedNone
		switch it.Category {
		case CategoryArithmetic:
			execute, family = true, FamilyArithmetic
		case CategoryAlgorithm:
			execute, family = true, FamilyAlgorithm
		case CategoryNonExecution:
			if it.ExplicitExecutionConstraint != ConstraintForbid || !it.HardNegative {
				failf("%s: category D is a hard negative that forbids execution", it.ID)
			}
		}
		if it.ExecuteExpected != execute || it.OperationFamily != family {
			failf("%s: execute %v family %s in category %s", it.ID, it.ExecuteExpected, it.OperationFamily, it.Category)
		}
		if execute {
			need = NeedHelpful
			if it.ExecutionNeed == NeedNecessary {
				need = NeedNecessary
			}
		}
		if it.ExecutionNeed != need {
			failf("%s: execution_need %s", it.ID, it.ExecutionNeed)
		}
		if it.LabelPolicy != LabelPolicy {
			failf("%s: label_policy %s", it.ID, it.LabelPolicy)
		}
		if it.Language != "de" && it.Language != "en" {
			failf("%s: language %s", it.ID, it.Language)
		}
		if it.Split == SplitSmoke {
			smokeGroups[it.ParaphraseGroup] = true
			families[it.OperationFamily]++
			if it.ExecuteExpected {
				positives++
			}
			if it.Operation != "" && it.Category != CategoryNoExecution {
				if pairedOps[it.ParaphraseGroup] == nil {
					pairedOps[it.ParaphraseGroup] = map[string]bool{}
				}
				pairedOps[it.ParaphraseGroup][it.Category] = true
			}
		} else {
			pilotGroups[it.ParaphraseGroup] = true
		}
	}
	for _, cat := range []string{CategoryNoExecution, CategoryArithmetic, CategoryAlgorithm, CategoryNonExecution} {
		if n := counts[splitCat{SplitSmoke, cat}]; n != 10 {
			failf("%s: %d smoke prompts, want 10", cat, n)
		}
		if n := counts[splitCat{SplitPilot, cat}]; n != 1 {
			failf("%s: %d pilot prompts, want 1", cat, n)
		}
	}
	if positives != 20 || families[FamilyNone] != 20 || families[FamilyArithmetic] != 10 || families[FamilyAlgorithm] != 10 {
		failf("smoke balance: %d positives, families %v", positives, families)
	}
	for g := range pilotGroups {
		if smokeGroups[g] {
			failf("group %s is in the pilot and in the smoke split", g)
		}
	}
	// the same operation, once to compute and once forbidden, in one group: a keyword cannot tell them
	paired := 0
	for _, cats := range pairedOps {
		if cats[CategoryNonExecution] && (cats[CategoryArithmetic] || cats[CategoryAlgorithm]) {
			paired++
		}
	}
	if paired != 10 {
		failf("%d groups pair a computation with its refusal, want 10", paired)
	}
	for i, a := range items {
		wa := maskedWords(a.Prompt)
		for _, b := range items[i+1:] {
			if a.ParaphraseGroup == b.ParaphraseGroup {
				continue
			}
			if v := overlap(wa, maskedWords(b.Prompt)); v >= nearDuplicateAt {
				failf("%s and %s are near-identical across groups (%.2f): %q, %q", a.ID, b.ID, v, a.Prompt, b.Prompt)
			}
		}
	}
	return errs
}
