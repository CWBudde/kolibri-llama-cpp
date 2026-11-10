package locality

import "slices"

// Perfect is an oracle at the same token's horizon (PLAN Stage E6's reference): before step (t, l) it
// predicts layer l+1 of token t, one step ahead, as SameToken does, and its best candidates are
// exactly the experts that step will use, read from the selections. It is no predictor but the
// best recall of the predicted step any predictor of that horizon can have: a hidden-state
// predictor applying layer l+1's router to layer l-1's output predicts the same step at the same
// time, and cannot cover it better than knowing it. It does not bound what a predictor's other
// candidates do for later steps. Beyond the true experts, for k above them, it ranks as the predictor it pads with, without
// them. It predicts where SameToken does: before decoded steps of layers 1 to L-2.
type Perfect struct {
	sel Selections
	pad Predictor
	buf []int
}

// NewPerfect returns the oracle for sel, padded with pad's ranks; pad has seen nothing and must
// rank, wherever SameToken predicts, k experts, as Markov does.
func NewPerfect(sel Selections, pad Predictor) *Perfect { return &Perfect{sel: sel, pad: pad} }

// Observe lets the padding predictor learn step s.
func (p *Perfect) Observe(s StepInfo, ids []int) { p.pad.Observe(s, ids) }

// Predict returns, before step s, layer s.Layer+1 and that step's true experts in the order Replay
// meets them, then the padding predictor's best others, k in all, in out's storage; nil for any
// horizon but SameToken and where SameToken predicts nothing.
func (p *Perfect) Predict(h Horizon, s StepInfo, k int, out []int) (int, []int) {
	target := s.Layer + 1
	if h != SameToken || s.Phase != Decode || k <= 0 || s.Layer < 1 || target >= p.sel.Layers {
		return 0, nil
	}
	out = stepExperts(p.sel, s.Span, target, out[:0])
	out = out[:min(len(out), k)]
	if len(out) == k {
		return target, out
	}
	l, ranked := p.pad.Predict(h, s, k, p.buf)
	if ranked != nil {
		p.buf = ranked
	}
	if l != target {
		ranked = nil
	}
	for _, e := range ranked {
		if len(out) == k {
			break
		}
		if !slices.Contains(out, e) {
			out = append(out, e)
		}
	}
	if len(out) < k { // a Ranking stores k candidates per step: fewer would read as expert 0
		panic("locality: the perfect predictor's padding ranked too few experts")
	}
	return target, out
}
