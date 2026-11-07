package probe

import (
	"math"
	"regexp"
	"strings"
	"unicode"
)

// Features gives the inner products a linear probe needs for one fold. Whatever it standardizes or
// counts, it learns from the training rows only: a test row must not shape its own features.
type Features interface {
	// Kernels returns the train-by-train and test-by-train kernels of a fold.
	Kernels(train, test []int) (ktrain, ktest []float64)
}

// Matrix is one layer's activations: Rows x Dim, row major.
type Matrix struct {
	Rows, Dim int
	Data      []float32
}

// Row is one row's activation.
func (m Matrix) Row(i int) []float32 { return m.Data[i*m.Dim : (i+1)*m.Dim] }

// Select returns the given rows, in order.
func (m Matrix) Select(rows []int) Matrix {
	out := Matrix{Rows: len(rows), Dim: m.Dim, Data: make([]float32, 0, len(rows)*m.Dim)}
	for _, i := range rows {
		out.Data = append(out.Data, m.Row(i)...)
	}
	return out
}

// standardized gives the rows, standardized per feature with the training rows' mean and standard
// deviation and divided by the square root of the dimension, so that a training row's expected
// squared norm is 1 and l2 means the same at every layer. A constant feature becomes 0.
func standardized(row func(int) []float32, dim int, train, test []int) (xtrain, xtest [][]float64) {
	mean, std := make([]float64, dim), make([]float64, dim)
	for _, i := range train {
		for d, v := range row(i) {
			mean[d] += float64(v)
		}
	}
	for d := range mean {
		mean[d] /= float64(len(train))
	}
	for _, i := range train {
		for d, v := range row(i) {
			e := float64(v) - mean[d]
			std[d] += e * e
		}
	}
	scale := math.Sqrt(float64(dim))
	for d := range std {
		std[d] = math.Sqrt(std[d] / float64(len(train)))
	}
	apply := func(rows []int) [][]float64 {
		out := make([][]float64, len(rows))
		for r, i := range rows {
			out[r] = make([]float64, dim)
			for d, v := range row(i) {
				if std[d] > 0 {
					out[r][d] = (float64(v) - mean[d]) / std[d] / scale
				}
			}
		}
		return out
	}
	return apply(train), apply(test)
}

func kernel(a, b [][]float64) []float64 {
	out := make([]float64, len(a)*len(b))
	for i, x := range a {
		for j, y := range b {
			var s float64
			for d, v := range x {
				s += v * y[d]
			}
			out[i*len(b)+j] = s
		}
	}
	return out
}

// Kernels implements Features.
func (m Matrix) Kernels(train, test []int) (ktrain, ktest []float64) {
	xtrain, xtest := standardized(m.Row, m.Dim, train, test)
	return kernel(xtrain, xtrain), kernel(xtest, xtrain)
}

// promptWords are a prompt's lowercased words with every number as one word.
func promptWords(prompt string) []string {
	return strings.FieldsFunc(groupedNumRe.ReplaceAllString(strings.ToLower(prompt), "#"), func(r rune) bool {
		return !unicode.IsLetter(r) && r != '#'
	})
}

// BagOfWords is the learned keyword baseline: which of the training rows' words a prompt holds. It
// sees the words, the probe does not, so a probe that does not beat it has shown nothing the text
// does not show.
type BagOfWords struct{ Items []Item }

// Kernels implements Features.
func (b BagOfWords) Kernels(train, test []int) (ktrain, ktest []float64) {
	vocab := map[string]int{}
	for _, i := range train {
		for _, w := range promptWords(b.Items[i].Prompt) {
			if _, ok := vocab[w]; !ok {
				vocab[w] = len(vocab)
			}
		}
	}
	presence := make([][]float32, len(b.Items))
	for i, it := range b.Items {
		presence[i] = make([]float32, len(vocab))
		for _, w := range promptWords(it.Prompt) {
			if d, ok := vocab[w]; ok {
				presence[i][d] = 1
			}
		}
	}
	xtrain, xtest := standardized(func(i int) []float32 { return presence[i] }, len(vocab), train, test)
	return kernel(xtrain, xtrain), kernel(xtest, xtrain)
}

// The fixed keyword rule, the simplest baseline: words that ask for a computation or name an
// operation, as a naive router would match them. They read no label and were fixed before any
// capture, but by the hand that wrote the prompts, so the rule knows the dataset's vocabulary.
var (
	keywordsArithmetic = []string{"berechne", "compute", "mal", "hoch", "multipliziere", "multiplikation", "multiplizieren", "ggt", "fakultät", "factorial", "divisor", "×", "^", "!"}
	keywordsAlgorithm  = []string{"sortiere", "sort", "bubblesort", "reihenfolge", "primzahl", "prime", "fibonacci", "filtere", "behalte", "keep", "liste", "list"}
)

func hasKeyword(prompt string, keywords []string) bool {
	lower := strings.ToLower(prompt)
	words := map[string]bool{}
	for _, w := range strings.FieldsFunc(lower, func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) }) {
		words[w] = true
	}
	for _, k := range keywords {
		if words[k] || (len([]rune(k)) == 1 && strings.Contains(lower, k)) {
			return true
		}
	}
	return false
}

// keywordFamily is the family the fixed rule assigns; arithmetic wins over algorithm.
func keywordFamily(prompt string) string {
	switch {
	case hasKeyword(prompt, keywordsArithmetic):
		return FamilyArithmetic
	case hasKeyword(prompt, keywordsAlgorithm):
		return FamilyAlgorithm
	}
	return FamilyNone
}

// The surface rule, the bar a probe must clear on this dataset: a digit, and none of the words the
// refusals turn on. Written after seeing the rows, and it gets 39 of the 40 smoke rows, so most of
// the execute label can be read off two cues that have nothing to do with what a prompt means.
var (
	surfaceDigitRe   = regexp.MustCompile(`\d`)
	surfaceRefusalRe = regexp.MustCompile(`(?i)\b(?:aber|ohne|but|without)\b`)
)

func surfaceExecute(prompt string) bool {
	return surfaceDigitRe.MatchString(prompt) && !surfaceRefusalRe.MatchString(prompt)
}

// surfaceFamily is the fixed keyword rule behind the surface rule's gate.
func surfaceFamily(prompt string) string {
	if !surfaceExecute(prompt) {
		return FamilyNone
	}
	return keywordFamily(prompt)
}

// lengthFeature is each row's prompt length, the other surface cue: a refusal says more than the
// computation it pairs with. The same classifier reads it, as one feature.
func lengthFeature(items []Item) Matrix {
	m := Matrix{Rows: len(items), Dim: 1, Data: make([]float32, len(items))}
	for i, it := range items {
		m.Data[i] = float32(len([]rune(it.Prompt)))
	}
	return m
}
