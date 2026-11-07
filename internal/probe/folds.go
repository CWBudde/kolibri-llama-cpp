package probe

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"slices"
)

// Folds are the test rows of each fold; a fold's training rows are all the others.
type Folds [][]int

// ErrGroupLeak is the refusal of a split in which a paraphrase group spans train and test.
var ErrGroupLeak = errors.New("paraphrase group spans train and test")

// GroupFolds splits rows into k folds along their paraphrase groups, deterministically for a seed:
// groups in shuffled order, largest first, each into the fold it unbalances least across the
// families. A group is one unit, so the classes can only be balanced as far as the groups allow.
func GroupFolds(items []Item, k int, seed uint64) (Folds, error) {
	if k < 2 {
		return nil, fmt.Errorf("%d folds, want at least 2", k)
	}
	var names []string
	rows := map[string][]int{}
	for i, it := range items {
		if _, ok := rows[it.ParaphraseGroup]; !ok {
			names = append(names, it.ParaphraseGroup)
		}
		rows[it.ParaphraseGroup] = append(rows[it.ParaphraseGroup], i)
	}
	if len(names) < k {
		return nil, fmt.Errorf("%d groups for %d folds", len(names), k)
	}
	r := rand.New(rand.NewPCG(seed, uint64(k)))
	r.Shuffle(len(names), func(i, j int) { names[i], names[j] = names[j], names[i] })
	slices.SortStableFunc(names, func(a, b string) int { return len(rows[b]) - len(rows[a]) })

	folds := make(Folds, k)
	counts := make([]map[string]int, k)
	for i := range counts {
		counts[i] = map[string]int{}
	}
	for _, g := range names {
		best, bestCost := 0, -1
		for f := range folds {
			cost := len(folds[f]) * len(folds[f])
			add := map[string]int{}
			for _, i := range rows[g] {
				add[items[i].OperationFamily]++
			}
			for _, fam := range Families {
				c := counts[f][fam] + add[fam]
				cost += 2 * c * c
			}
			if bestCost < 0 || cost < bestCost {
				best, bestCost = f, cost
			}
		}
		for _, i := range rows[g] {
			counts[best][items[i].OperationFamily]++
		}
		folds[best] = append(folds[best], rows[g]...)
	}
	for f := range folds {
		slices.Sort(folds[f])
	}
	return folds, nil
}

// CheckFolds refuses a split that could leak: a row tested twice or never, a paraphrase group, a
// template or a numeric seed in train and test of one fold.
func CheckFolds(items []Item, folds Folds) error {
	seen := make([]bool, len(items))
	for f, test := range folds {
		if len(test) == 0 {
			return fmt.Errorf("fold %d: no test rows", f)
		}
		inTest := make([]bool, len(items))
		groups, templates, seeds := map[string]bool{}, map[string]bool{}, map[uint32]bool{}
		for _, i := range test {
			if i < 0 || i >= len(items) {
				return fmt.Errorf("fold %d: row %d of %d", f, i, len(items))
			}
			if seen[i] {
				return fmt.Errorf("fold %d: row %s is tested twice", f, items[i].ID)
			}
			seen[i], inTest[i] = true, true
			groups[items[i].ParaphraseGroup], templates[items[i].TemplateID] = true, true
			if s := items[i].NumericSeed; s != nil {
				seeds[*s] = true
			}
		}
		for i, it := range items {
			if inTest[i] {
				continue
			}
			if groups[it.ParaphraseGroup] {
				return fmt.Errorf("fold %d: %w: %s", f, ErrGroupLeak, it.ParaphraseGroup)
			}
			if templates[it.TemplateID] {
				return fmt.Errorf("fold %d: %w: template %s", f, ErrGroupLeak, it.TemplateID)
			}
			if it.NumericSeed != nil && seeds[*it.NumericSeed] {
				return fmt.Errorf("fold %d: %w: numeric seed %d", f, ErrGroupLeak, *it.NumericSeed)
			}
		}
	}
	for i, ok := range seen {
		if !ok {
			return fmt.Errorf("row %s is never tested", items[i].ID)
		}
	}
	return nil
}

// trainRows are the rows outside a fold's test rows, in order.
func trainRows(n int, test []int) []int {
	inTest := make([]bool, n)
	for _, i := range test {
		inTest[i] = true
	}
	out := make([]int, 0, n-len(test))
	for i := range n {
		if !inTest[i] {
			out = append(out, i)
		}
	}
	return out
}
