package locality

// A hidden-state predictor replayed from a capture file (PLAN Stage E6): the router of a later
// block applied to the residual stream after an earlier one, scored offline by
// tools/e6/capture_router.py. A file pred-hH holds, for every layer row r and token t, the
// experts block r+H's router ranks first on l_out-r, the residual stream after block r of token
// t, best first, -1 where block r+H does not exist. Before layer l of a token runs, blocks 0 to
// l-1 have run, so rows 0 to l-1 of the token are known and row l is not.

import "fmt"

// HiddenStateBlocks is the horizon in blocks of the file HiddenState reads: pred-h2.
const HiddenStateBlocks = 2

// HiddenState predicts at the same token's horizon from a pred-h2 file: before layer l of decoded
// token t it predicts layer l+1, as SameToken and Perfect do, with the first k experts of row l-1
// of token t, block l+1's router on the residual stream after block l-1, which has run. It never
// reads a row of the token at or after l, nor a later token. It predicts where Perfect does:
// before decoded steps of layers 1 to L-2, so layers 2 to L-1 of every decoded token. It learns
// nothing from the steps.
type HiddenState struct {
	pred Selections
}

// NewHiddenState returns the predictor of the pred-h2 file pred, for layers of nExpert experts.
// Every row with a target block must hold distinct experts below nExpert, every row without one
// (row+2 >= pred.Layers) only -1: a file of another horizon or layout fails here.
func NewHiddenState(pred Selections, nExpert int) (*HiddenState, error) {
	if err := CheckPred(pred, HiddenStateBlocks, nExpert); err != nil {
		return nil, err
	}
	return &HiddenState{pred: pred}, nil
}

// CheckPred checks a pred-hH file of horizon h for layers of nExpert experts: every row r with
// r+h below pred.Layers holds pred.K distinct experts in [0, nExpert), every other row only -1.
func CheckPred(pred Selections, h, nExpert int) error {
	if pred.Layers <= 0 || pred.Tokens <= 0 || pred.K <= 0 || len(pred.IDs) != pred.Layers*pred.Tokens*pred.K {
		return fmt.Errorf("pred-h%d: shape %d x %d x %d with %d IDs", h, pred.Layers, pred.Tokens, pred.K,
			len(pred.IDs))
	}
	seen := make([]int, nExpert) // per expert, the row it was last seen in, plus one
	for row := range pred.Layers {
		for t := range pred.Tokens {
			i := row*pred.Tokens + t
			for j, e := range pred.IDs[i*pred.K : (i+1)*pred.K] {
				if row+h >= pred.Layers {
					if e != -1 {
						return fmt.Errorf("pred-h%d: row %d, token %d holds %d, want -1: block %d does not exist", h,
							row, t, e, row+h)
					}
					continue
				}
				if e < 0 || int(e) >= nExpert {
					return fmt.Errorf("pred-h%d: row %d, token %d, rank %d: expert %d of %d", h, row, t, j, e, nExpert)
				}
				if seen[e] == i+1 {
					return fmt.Errorf("pred-h%d: row %d, token %d: expert %d twice", h, row, t, e)
				}
				seen[e] = i + 1
			}
		}
	}
	return nil
}

// Observe does nothing: the file holds every prediction.
func (p *HiddenState) Observe(StepInfo, []int) {}

// Predict returns, before step s, layer s.Layer+1 and the first k experts of row s.Layer-1 of the
// step's token, in out's storage; nil for any horizon but SameToken and where SameToken predicts
// nothing. k must not exceed the file's width.
func (p *HiddenState) Predict(h Horizon, s StepInfo, k int, out []int) (int, []int) {
	target := s.Layer + 1
	if h != SameToken || s.Phase != Decode || k <= 0 || s.Layer < 1 || target >= p.pred.Layers {
		return 0, nil
	}
	if s.End-s.First != 1 {
		panic("locality: a decoded span of more than one token")
	}
	if k > p.pred.K { // a Ranking stores k candidates per step: fewer would read as expert 0
		panic(fmt.Sprintf("locality: %d candidates asked of a pred file of %d", k, p.pred.K))
	}
	return target, p.row(s.Layer-1, s.First, k, out[:0])
}

// row appends the first k experts of row r, token t, skipping -1, to out.
func (p *HiddenState) row(r, t, k int, out []int) []int {
	i := (r*p.pred.Tokens + t) * p.pred.K
	for _, e := range p.pred.IDs[i : i+p.pred.K] {
		if len(out) == k {
			break
		}
		if e >= 0 {
			out = append(out, int(e))
		}
	}
	return out
}

// HiddenRecall scores a pred-hH file of horizon h against the decoded steps of spans[from:to]:
// layer l of a decoded token t against the first k experts of row l-h of token t, block l's
// router on the residual stream after block l-h. all counts every layer, the layers below h,
// which no row predicts, with no candidates and nothing covered, as Ranking.Recall does; predicted
// counts the predicted layers alone. With h = 2 all is the Recall of HiddenState's ranking.
func HiddenRecall(pred Selections, h int, sel Selections, spans []Span, from, to int, ks []int) (all,
	predicted []Recall) {
	all, predicted = make([]Recall, len(ks)), make([]Recall, len(ks))
	p := HiddenState{pred: pred}
	var need, cand []int
	for _, s := range spans[from:to] {
		if s.Phase != Decode {
			continue
		}
		if s.End-s.First != 1 {
			panic("locality: a decoded span of more than one token")
		}
		for l := range sel.Layers {
			need = stepExperts(sel, s, l, need[:0])
			for i, k := range ks {
				x := &all[i]
				x.Steps++
				x.Needed += len(need)
				if l < h {
					continue
				}
				cand = p.row(l-h, s.First, k, cand[:0])
				covered := 0
				for _, e := range need {
					for _, c := range cand {
						if c == e {
							covered++
							break
						}
					}
				}
				y := &predicted[i]
				y.Steps++
				y.Needed += len(need)
				for _, r := range []*Recall{x, y} {
					r.Candidates += len(cand)
					r.Covered += covered
					if covered == len(need) {
						r.Exact++
					}
				}
			}
		}
	}
	return all, predicted
}
