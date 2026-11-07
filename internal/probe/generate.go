package probe

import (
	"fmt"
	"hash/fnv"
	"math/big"
	"math/rand/v2"
	"slices"
	"strconv"
	"strings"
)

// Seed is the generator's fixed seed: the number of issue #63.
const Seed = 63

// values are a group's drawn numbers. All rows of a group share them, so that a row differs from
// the group's first row in one axis only.
type values struct {
	A, B uint64   // operands
	List []uint64 // list inputs
	T    uint64   // a filter's threshold
}

// row is one prompt of a group, for one category.
type row struct {
	category, template, lang, axis string
	constraint                     string
	text                           func(values) string
	answer                         string // a fixed answer; "" derives it or uses NoExactAnswer
	hard                           bool
}

// group is one paraphrase group: its rows never span a train and a test fold.
type group struct {
	name, op string
	split    string
	draw     func(*rand.Rand) values // nil for a prompt without drawn numbers
	rows     []row
}

func inRange(r *rand.Rand, lo, hi uint64) uint64 { return lo + r.Uint64N(hi-lo+1) }

// distinctList draws n distinct values in [lo, hi], in drawn order.
func distinctList(r *rand.Rand, n int, lo, hi uint64) []uint64 {
	var out []uint64
	for len(out) < n {
		if v := inRange(r, lo, hi); !slices.Contains(out, v) {
			out = append(out, v)
		}
	}
	return out
}

func isPrimeTrial(n uint64) bool {
	if n < 2 {
		return false
	}
	for d := uint64(2); d*d <= n; d++ {
		if n%d == 0 {
			return false
		}
	}
	return true
}

func nextPrime(n uint64) uint64 {
	for !isPrimeTrial(n) {
		n++
	}
	return n
}

// grouped writes n with a separator every three digits, the formatting axis of the operands.
func grouped(n uint64, sep string) string {
	s := strconv.FormatUint(n, 10)
	var parts []string
	for len(s) > 3 {
		parts = append([]string{s[len(s)-3:]}, parts...)
		s = s[:len(s)-3]
	}
	return strings.Join(append([]string{s}, parts...), sep)
}

func plain(n uint64) string { return strconv.FormatUint(n, 10) }

func listText(l []uint64) string {
	parts := make([]string, len(l))
	for i, v := range l {
		parts[i] = plain(v)
	}
	return strings.Join(parts, ", ")
}

// answerOf computes a row's answer with the generator's own code. The oracle (oracle.go) derives
// every answer again from the prompt's text, with other algorithms.
func answerOf(op string, v values, text string) string {
	switch op {
	case "multiplication":
		return plain(v.A * v.B)
	case "exponentiation":
		out, base := big.NewInt(1), new(big.Int).SetUint64(v.A)
		for range v.B {
			out.Mul(out, base)
		}
		return out.String()
	case "factorial":
		out := big.NewInt(1)
		for i := uint64(2); i <= v.A; i++ {
			out.Mul(out, new(big.Int).SetUint64(i))
		}
		return out.String()
	case "gcd":
		a, b := v.A, v.B
		for b != 0 {
			a, b = b, a%b
		}
		return plain(a)
	case "sorting":
		l := slices.Clone(v.List)
		slices.Sort(l)
		if strings.Contains(text, "absteigend") {
			slices.Reverse(l)
		}
		return listText(l)
	case "primality":
		return strconv.FormatBool(isPrimeTrial(v.A))
	case "fibonacci":
		var a, b uint64 = 0, 1
		for range v.A {
			a, b = b, a+b
		}
		return plain(a)
	case "filtering_even":
		var l []uint64
		for _, x := range v.List {
			if x%2 == 0 {
				l = append(l, x)
			}
		}
		return listText(l)
	case "filtering_greater":
		var l []uint64
		for _, x := range v.List {
			if x > v.T {
				l = append(l, x)
			}
		}
		return listText(l)
	}
	panic("answerOf: operation " + op)
}

func fixed(s string) func(values) string { return func(values) string { return s } }

// noExec is a category A row.
func noExec(template, lang, axis, text, answer string, hard bool) row {
	return row{category: CategoryNoExecution, template: template, lang: lang, axis: axis, constraint: ConstraintNone,
		text: fixed(text), answer: answer, hard: hard}
}

// compute is a category B or C row.
func compute(category, template, lang, axis, constraint string, text func(values) string) row {
	return row{category: category, template: template, lang: lang, axis: axis, constraint: constraint, text: text}
}

// abstain is a category D row: same numbers as its group's compute row, the instruction forbids.
func abstain(template, lang string, text func(values) string) row {
	return row{category: CategoryNonExecution, template: template, lang: lang, axis: "instruction",
		constraint: ConstraintForbid, text: text, hard: true}
}

func pair(a, b func(*rand.Rand) uint64) func(*rand.Rand) values {
	return func(r *rand.Rand) values { return values{A: a(r), B: b(r)} }
}

func between(lo, hi uint64) func(*rand.Rand) uint64 {
	return func(r *rand.Rand) uint64 { return inRange(r, lo, hi) }
}

func drawGCD(glo, ghi, xlo, xhi uint64) func(*rand.Rand) values {
	return func(r *rand.Rand) values {
		g, x := inRange(r, glo, ghi), inRange(r, xlo, xhi)
		y := inRange(r, xlo, xhi)
		for y == x {
			y = inRange(r, xlo, xhi)
		}
		return values{A: g * x, B: g * y}
	}
}

func oneNumber(lo, hi uint64) func(*rand.Rand) values {
	return func(r *rand.Rand) values { return values{A: inRange(r, lo, hi)} }
}

func listOf(n int, lo, hi uint64) func(*rand.Rand) values {
	return func(r *rand.Rand) values { return values{List: distinctList(r, n, lo, hi)} }
}

// groups is the dataset. It is data: changing a prompt, its order or the seed changes
// testdata/probe/smoke.jsonl, which must not move once a real capture has read it.
var groups = []group{
	// The issue's four pilot prompts, word for word, each in a group of its own that no smoke row
	// shares, so they can never leak into the smoke folds.
	{name: "capital-01", split: SplitPilot, rows: []row{
		noExec("pilot-capital", "de", "base", "Was ist die Hauptstadt von Frankreich?", "Paris", false)}},
	{name: "multiplication-01", op: "multiplication", split: SplitPilot,
		draw: func(*rand.Rand) values { return values{A: 17893, B: 24761} },
		rows: []row{compute(CategoryArithmetic, "pilot-multiplication", "de", "base", ConstraintDemand,
			func(v values) string { return fmt.Sprintf("Berechne %d mal %d exakt.", v.A, v.B) })}},
	{name: "sorting-01", op: "sorting", split: SplitPilot,
		draw: func(*rand.Rand) values { return values{List: []uint64{17, 3, 29, 8, 11}} },
		rows: []row{compute(CategoryAlgorithm, "pilot-sorting", "de", "base", ConstraintNone,
			func(v values) string { return "Sortiere die Zahlen " + listText(v.List) + " aufsteigend." })}},
	{name: "euclid-01", op: "gcd", split: SplitPilot, rows: []row{
		abstain("pilot-euclid", "de", fixed("Erkläre den euklidischen Algorithmus, aber rechne nichts aus."))}},

	// Category A. Hard negatives ask about a number or mention an operation, and need no execution.
	{name: "capital-02", split: SplitSmoke, rows: []row{
		noExec("capital-de", "de", "base", "Wie heißt die Hauptstadt von Spanien?", "Madrid", false),
		noExec("capital-en", "en", "language", "What is the capital of Spain?", "Madrid", false)}},
	{name: "history-01", split: SplitSmoke, rows: []row{
		noExec("history-year", "de", "base", "In welchem Jahr fiel die Berliner Mauer?", "1989", true)}},
	{name: "astronomy-01", split: SplitSmoke, rows: []row{
		noExec("astronomy-count", "de", "base", "Wie viele Planeten hat unser Sonnensystem?", "8", true)}},
	{name: "explain-sky-01", split: SplitSmoke, rows: []row{
		noExec("explain-sky-de", "de", "base", "Erkläre in zwei Sätzen, warum der Himmel blau ist.", NoExactAnswer, false),
		noExec("explain-sky-en", "en", "language", "Why is the sky blue? Answer in two sentences.", NoExactAnswer, false)}},
	{name: "explain-commutative-01", op: "multiplication", split: SplitSmoke, rows: []row{
		noExec("explain-commutative", "de", "base", "Warum ist die Multiplikation kommutativ? Erkläre es anschaulich.", NoExactAnswer, true)}},
	{name: "writing-01", split: SplitSmoke, rows: []row{
		noExec("writing-poem", "de", "base", "Schreibe ein kurzes Gedicht über den Herbst.", NoExactAnswer, false)}},
	{name: "writing-02", split: SplitSmoke, rows: []row{
		noExec("writing-note", "en", "base", "Write a short thank-you note to a colleague.", NoExactAnswer, false)}},
	{name: "writing-number-01", split: SplitSmoke, rows: []row{
		noExec("writing-number", "de", "base", "Schreibe einen Satz darüber, warum die 7 als Glückszahl gilt.", NoExactAnswer, true)}},

	// Category B, with category D partners on the same numbers. Magnitude and formatting vary.
	{name: "multiplication-02", op: "multiplication", split: SplitSmoke, draw: pair(between(12, 99), between(12, 99)), rows: []row{
		compute(CategoryArithmetic, "mul-small-de", "de", "base", ConstraintNone,
			func(v values) string { return fmt.Sprintf("Was ist %d mal %d?", v.A, v.B) }),
		abstain("mul-small-explain-de", "de",
			func(v values) string {
				return fmt.Sprintf("Erkläre das schriftliche Multiplizieren am Beispiel %d mal %d, aber rechne das Ergebnis nicht aus.", v.A, v.B)
			})}},
	{name: "multiplication-03", op: "multiplication", split: SplitSmoke, draw: pair(between(1000000, 9999999), between(1000, 9999)), rows: []row{
		compute(CategoryArithmetic, "mul-large-dot-de", "de", "base", ConstraintDemand,
			func(v values) string {
				return fmt.Sprintf("Berechne das exakte Produkt %s × %s.", grouped(v.A, "."), grouped(v.B, "."))
			}),
		compute(CategoryArithmetic, "mul-large-comma-en", "en", "language", ConstraintDemand,
			func(v values) string {
				return fmt.Sprintf("Compute the exact product %s × %s.", grouped(v.A, ","), grouped(v.B, ","))
			})}},
	{name: "multiplication-04", op: "multiplication", split: SplitSmoke, draw: pair(between(10000, 99999), between(10000, 99999)), rows: []row{
		compute(CategoryArithmetic, "mul-space-de", "de", "base", ConstraintNone,
			func(v values) string {
				return fmt.Sprintf("Multipliziere %s mit %s.", grouped(v.A, " "), grouped(v.B, " "))
			})}},
	{name: "exponentiation-01", op: "exponentiation", split: SplitSmoke, draw: pair(between(3, 9), between(15, 25)), rows: []row{
		compute(CategoryArithmetic, "pow-word-de", "de", "base", ConstraintNone,
			func(v values) string { return fmt.Sprintf("Berechne %d hoch %d.", v.A, v.B) }),
		abstain("pow-explain-de", "de",
			func(v values) string {
				return fmt.Sprintf("Erkläre, wie man %d hoch %d durch wiederholtes Quadrieren bestimmt, ohne den Wert zu berechnen.", v.A, v.B)
			})}},
	{name: "exponentiation-02", op: "exponentiation", split: SplitSmoke, draw: pair(between(2, 3), between(40, 61)), rows: []row{
		compute(CategoryArithmetic, "pow-caret-de", "de", "base", ConstraintDemand,
			func(v values) string { return fmt.Sprintf("Wie lautet der exakte Wert von %d^%d?", v.A, v.B) })}},
	{name: "factorial-01", op: "factorial", split: SplitSmoke, draw: oneNumber(15, 25), rows: []row{
		compute(CategoryArithmetic, "fact-bang-de", "de", "base", ConstraintDemand,
			func(v values) string { return fmt.Sprintf("Wie groß ist %d! genau?", v.A) }),
		abstain("fact-explain-de", "de",
			func(v values) string {
				return fmt.Sprintf("Was bedeutet %d!? Erkläre die Definition, aber berechne den Wert nicht.", v.A)
			})}},
	{name: "factorial-02", op: "factorial", split: SplitSmoke, draw: oneNumber(8, 14), rows: []row{
		compute(CategoryArithmetic, "fact-word-en", "en", "base", ConstraintNone,
			func(v values) string { return fmt.Sprintf("What is the factorial of %d?", v.A) })}},
	{name: "gcd-01", op: "gcd", split: SplitSmoke, draw: drawGCD(3, 40, 20, 99), rows: []row{
		compute(CategoryArithmetic, "gcd-de", "de", "base", ConstraintNone,
			func(v values) string { return fmt.Sprintf("Bestimme den ggT von %d und %d.", v.A, v.B) }),
		abstain("gcd-explain-de", "de",
			func(v values) string {
				return fmt.Sprintf("Erkläre, wie der euklidische Algorithmus den ggT von %d und %d findet, aber rechne nichts aus.", v.A, v.B)
			})}},
	{name: "gcd-02", op: "gcd", split: SplitSmoke, draw: drawGCD(1000, 99999, 1000, 9999), rows: []row{
		compute(CategoryArithmetic, "gcd-en", "en", "base", ConstraintNone,
			func(v values) string {
				return fmt.Sprintf("What is the greatest common divisor of %d and %d?", v.A, v.B)
			}),
		abstain("gcd-pseudocode-en", "en",
			func(v values) string {
				return fmt.Sprintf("Write pseudocode for the greatest common divisor of %d and %d, but do not run it or compute the result.", v.A, v.B)
			})}},

	// Category C, likewise. Input length, order and phrasing vary.
	{name: "sorting-02", op: "sorting", split: SplitSmoke, draw: listOf(4, 2, 99), rows: []row{
		compute(CategoryAlgorithm, "sort-short-de", "de", "base", ConstraintNone,
			func(v values) string { return "Sortiere absteigend: " + listText(v.List) + "." }),
		abstain("sort-explain-de", "de",
			func(v values) string {
				return "Erkläre Bubblesort an den Zahlen " + listText(v.List) + ", aber sortiere sie nicht."
			})}},
	{name: "sorting-03", op: "sorting", split: SplitSmoke, draw: listOf(9, 2, 999), rows: []row{
		compute(CategoryAlgorithm, "sort-long-en", "en", "base", ConstraintNone,
			func(v values) string { return "Sort these numbers in ascending order: " + listText(v.List) + "." }),
		compute(CategoryAlgorithm, "sort-long-de", "de", "language", ConstraintNone,
			func(v values) string {
				return "Bringe diese Zahlen in aufsteigende Reihenfolge: " + listText(v.List) + "."
			})}},
	{name: "primality-01", op: "primality", split: SplitSmoke,
		draw: func(r *rand.Rand) values { return values{A: nextPrime(inRange(r, 1000, 9000))} }, rows: []row{
			compute(CategoryAlgorithm, "prime-yes-de", "de", "base", ConstraintNone,
				func(v values) string { return fmt.Sprintf("Ist %d eine Primzahl?", v.A) }),
			abstain("prime-explain-de", "de",
				func(v values) string {
					return fmt.Sprintf("Beschreibe, wie man prüft, ob %d eine Primzahl ist, ohne die Prüfung durchzuführen.", v.A)
				})}},
	{name: "primality-02", op: "primality", split: SplitSmoke,
		draw: func(r *rand.Rand) values {
			return values{A: nextPrime(inRange(r, 300, 900)) * nextPrime(inRange(r, 300, 900))}
		}, rows: []row{
			compute(CategoryAlgorithm, "prime-no-en", "en", "base", ConstraintNone,
				func(v values) string { return fmt.Sprintf("Is %d a prime number?", v.A) }),
			abstain("prime-explain-en", "en",
				func(v values) string {
					return fmt.Sprintf("Explain how trial division would test whether %d is a prime number, without actually carrying it out.", v.A)
				})}},
	{name: "fibonacci-01", op: "fibonacci", split: SplitSmoke, draw: oneNumber(20, 40), rows: []row{
		compute(CategoryAlgorithm, "fib-de", "de", "base", ConstraintNone,
			func(v values) string {
				return fmt.Sprintf("Berechne die Fibonacci-Zahl F(%d) mit F(1) = F(2) = 1.", v.A)
			}),
		abstain("fib-pseudocode-de", "de",
			func(v values) string {
				return fmt.Sprintf("Schreibe Pseudocode für die Fibonacci-Zahl F(%d) mit F(1) = F(2) = 1, aber führe ihn nicht aus.", v.A)
			})}},
	{name: "fibonacci-02", op: "fibonacci", split: SplitSmoke, draw: oneNumber(60, 90), rows: []row{
		compute(CategoryAlgorithm, "fib-en", "en", "base", ConstraintDemand,
			func(v values) string {
				return fmt.Sprintf("Give the exact Fibonacci number F(%d), with F(1) = F(2) = 1.", v.A)
			})}},
	{name: "filtering-01", op: "filtering_even", split: SplitSmoke, draw: listOf(6, 2, 99), rows: []row{
		compute(CategoryAlgorithm, "filter-even-de", "de", "base", ConstraintNone,
			func(v values) string { return "Welche Zahlen der Liste " + listText(v.List) + " sind gerade?" }),
		abstain("filter-explain-de", "de",
			func(v values) string {
				return "Beschreibe, wie man aus der Liste " + listText(v.List) + " die geraden Zahlen herausfiltert, ohne das Ergebnis zu nennen."
			})}},
	{name: "filtering-02", op: "filtering_greater", split: SplitSmoke,
		draw: func(r *rand.Rand) values { return values{List: distinctList(r, 8, 2, 200), T: inRange(r, 50, 150)} }, rows: []row{
			compute(CategoryAlgorithm, "filter-gt-en", "en", "base", ConstraintNone,
				func(v values) string {
					return fmt.Sprintf("From the list %s, keep only the numbers greater than %d.", listText(v.List), v.T)
				}),
			compute(CategoryAlgorithm, "filter-gt-de", "de", "language", ConstraintNone,
				func(v values) string {
					return fmt.Sprintf("Behalte aus der Liste %s nur die Zahlen größer als %d.", listText(v.List), v.T)
				})}},
}

// largest is the largest number a computation involves.
func largest(v values, answer string) *big.Int {
	out := new(big.Int)
	nums := append([]uint64{v.A, v.B, v.T}, v.List...)
	for _, n := range nums {
		if b := new(big.Int).SetUint64(n); b.Cmp(out) > 0 {
			out = b
		}
	}
	if b, ok := new(big.Int).SetString(answer, 10); ok && b.Cmp(out) > 0 {
		out = b
	}
	return out
}

// necessaryFrom is the size at which exact execution counts as necessary, not just helpful.
var necessaryFrom = big.NewInt(100000)

func difficultyOf(top *big.Int, n int) string {
	switch {
	case n >= 8 || top.Cmp(big.NewInt(10000000000)) >= 0:
		return "hard"
	case n >= 6 || top.Cmp(big.NewInt(10000)) >= 0:
		return "medium"
	}
	return "easy"
}

func letter(category string) string {
	switch category {
	case CategoryNoExecution:
		return "A"
	case CategoryArithmetic:
		return "B"
	case CategoryAlgorithm:
		return "C"
	case CategoryNonExecution:
		return "D"
	}
	panic("letter: category " + category)
}

// groupSeed derives a group's numeric seed from its name, so that a group's numbers do not depend
// on the other groups.
func groupSeed(name string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(name))
	return h.Sum32()
}

// Generate builds the dataset from the fixed seed. The same seed gives the same bytes.
func Generate(seed uint64) []Item {
	counters := map[string]int{}
	var out []Item
	for _, g := range groups {
		var v values
		var numSeed *uint32
		if g.draw != nil {
			s := groupSeed(g.name)
			numSeed = &s
			v = g.draw(rand.New(rand.NewPCG(seed, uint64(s))))
		}
		for _, r := range g.rows {
			text := r.text(v)
			it := Item{
				Category: r.category, Prompt: text, ParaphraseGroup: g.name, Split: g.split,
				Language: r.lang, TemplateID: r.template, NumericSeed: numSeed,
				PromptLength: len([]rune(text)), ExplicitExecutionConstraint: r.constraint,
				Operation: g.op, LabelPolicy: LabelPolicy, HardNegative: r.hard, VariedAxis: r.axis,
			}
			counters[it.Category]++
			it.ID = fmt.Sprintf("%s-%03d", letter(it.Category), counters[it.Category])
			switch r.category {
			case CategoryArithmetic, CategoryAlgorithm:
				it.ExecuteExpected = true
				it.OperationFamily = FamilyArithmetic
				if r.category == CategoryAlgorithm {
					it.OperationFamily = FamilyAlgorithm
				}
				it.ExpectedAnswerOrOracle = answerOf(g.op, v, text)
				top := largest(v, it.ExpectedAnswerOrOracle)
				it.Difficulty = difficultyOf(top, len(v.List))
				it.ExecutionNeed = NeedHelpful
				if top.Cmp(necessaryFrom) >= 0 {
					it.ExecutionNeed = NeedNecessary
				}
			default:
				it.OperationFamily = FamilyNone
				it.ExecutionNeed = NeedNone
				it.ExpectedAnswerOrOracle = r.answer
				if it.ExpectedAnswerOrOracle == "" {
					it.ExpectedAnswerOrOracle = NoExactAnswer
				}
				it.Difficulty = "easy"
				if r.hard {
					it.Difficulty = "hard"
				}
			}
			out = append(out, it)
		}
	}
	return out
}
