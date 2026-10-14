package locality

// The decayed-count Markov predictor (PLAN Phase 10, Stage B): a CPU-cheap ranking of a layer's
// experts for a step that has not run yet, from steps that have. It learns, per layer, every
// expert's exponentially decayed use count and how often one decoded set of experts followed
// another: in the same layer from one token to the next, and within a token from a layer to the
// layer after next. A score mixes the two; the best scores are the candidates.
//
// Every prediction has a horizon, which fixes what it is for and what it may know. Before a step
// runs, Replay's policy knows the earlier steps (replay.go), so before step (t, l):
//
//   - next token: predicts layer l-1 of token t+1 from token t's set in layer l-1, the step just
//     completed; before layer 0, layer L-1 of token t from token t-1's. It is made L-1 steps
//     before the step it is for.
//   - same token: predicts layer l+1 of token t from its set in layer l-1, one step before the
//     step it is for. Layer 0 has no completed layer of its token to predict from, and nothing
//     follows the last layer.
//
// Only decoded steps are predicted and only decoded sets predict: a prefill ubatch's union of
// experts is no token's set. Its uses still count.

import (
	"math"
	"slices"
)

// Horizon is what a prediction made before a step is for.
type Horizon int

const (
	// NextToken predicts the next token's experts in the layer that just completed.
	NextToken Horizon = iota
	// SameToken predicts the running token's experts in the layer after the next.
	SameToken
)

func (h Horizon) String() string {
	if h == SameToken {
		return "same token"
	}
	return "next token"
}

// Lead is the number of steps that run between a prediction of horizon h, made before a step,
// and the step it is for, in a model of layers layers.
func (h Horizon) Lead(layers int) int {
	if h == SameToken {
		return 1
	}
	return layers - 1
}

// LeadSeconds is the compute time of a horizon's lead between decoded steps: the time a prefetch
// has to arrive before the step it is for starts, if it never waits behind other loads.
func LeadSeconds(t Timing, h Horizon, layers int) float64 {
	return float64(h.Lead(layers)) * t.DecodeStep
}

// MarkovParams are the predictor's parameters: the half-life of its use counts in tokens, and the
// weight of the transitions in a score, Mix 0 for the counts alone and 1 for the transitions
// alone.
type MarkovParams struct {
	HalfLife int     `json:"half_life_tokens"`
	Mix      float64 `json:"mix"`
}

// Markov is the predictor's state for layers of nExpert experts.
type Markov struct {
	layers, nExpert int
	p               MarkovParams
	freq            []float64 // per key: its uses, decayed to the token in seen
	sum             []float64 // per layer: the sum of its freq
	seen            []int     // per layer: the token its counts are decayed to
	temporal        []float32 // per layer, from expert and to expert: token t-1's set to token t's
	tRow            []float32 // per layer and from expert: the transitions out of it
	cross           []float32 // per layer l, from expert of l-2 and to expert of l, within a token
	xRow            []float32
	last            [][]int // per layer: its latest decoded set
	lastTok         []int   // per layer: that set's token, -1 for none
	score           []float64
}

// NewMarkov returns a predictor that has seen nothing; p.HalfLife must be positive.
func NewMarkov(layers, nExpert int, p MarkovParams) *Markov {
	p.HalfLife = max(p.HalfLife, 1)
	n := layers * nExpert
	m := &Markov{layers: layers, nExpert: nExpert, p: p, freq: make([]float64, n), sum: make([]float64, layers),
		seen: make([]int, layers), temporal: make([]float32, n*nExpert), tRow: make([]float32, n),
		cross: make([]float32, n*nExpert), xRow: make([]float32, n), last: make([][]int, layers),
		lastTok: make([]int, layers), score: make([]float64, nExpert)}
	for l := range m.lastTok {
		m.lastTok[l] = -1
	}
	return m
}

// Bytes is the size of the predictor's tables, MarkovBytes.
func (m *Markov) Bytes() int64 { return MarkovBytes(m.layers, m.nExpert) }

// MarkovBytes is the size of a predictor's tables for layers of nExpert experts: two transition
// tables of float32 counts with their row sums, and the float64 decayed counts with their sums
// per layer.
func MarkovBytes(layers, nExpert int) int64 {
	l, n := int64(layers), int64(nExpert)
	return 2*l*(n*n+n)*4 + l*(n+1)*8
}

// Observe learns that step s, now run, used the experts ids of its layer, each once.
func (m *Markov) Observe(s StepInfo, ids []int) {
	l, n := s.Layer, m.nExpert
	if d := s.First - m.seen[l]; d > 0 {
		f := math.Pow(0.5, float64(d)/float64(m.p.HalfLife))
		for e := range n {
			m.freq[l*n+e] *= f
		}
		m.sum[l] *= f
		m.seen[l] = s.First
	}
	for _, e := range ids {
		m.freq[l*n+e]++
	}
	m.sum[l] += float64(len(ids))
	if s.Phase != Decode {
		m.lastTok[l] = -1
		return
	}
	if s.First > 0 && m.lastTok[l] == s.First-1 {
		m.add(m.temporal, m.tRow, l, m.last[l], ids)
	}
	if l >= 2 && m.lastTok[l-2] == s.First {
		m.add(m.cross, m.xRow, l, m.last[l-2], ids)
	}
	m.last[l] = append(m.last[l][:0], ids...)
	m.lastTok[l] = s.First
}

// add counts a transition from every expert of from to every expert of to in layer l's table.
func (m *Markov) add(table, row []float32, l int, from, to []int) {
	n := m.nExpert
	for _, a := range from {
		r := table[(l*n+a)*n : (l*n+a+1)*n]
		for _, b := range to {
			r[b]++
		}
		row[l*n+a] += float32(len(to))
	}
}

// Predict ranks, before step s, the experts of the step horizon h predicts: it returns that
// step's layer and its k best experts, best first, the lower expert first of equal scores, in
// out's storage. It returns nil when h predicts nothing before s.
func (m *Markov) Predict(h Horizon, s StepInfo, k int, out []int) (int, []int) {
	if s.Phase != Decode || k <= 0 {
		return 0, nil
	}
	var target, from int
	var table, row []float32
	switch h {
	case NextToken:
		from, tok := s.Layer-1, s.First
		if s.Layer == 0 {
			from, tok = m.layers-1, s.First-1
		}
		if tok < 0 || m.lastTok[from] != tok {
			return 0, nil
		}
		target, table, row = from, m.temporal, m.tRow
	case SameToken:
		target, from = s.Layer+1, s.Layer-1
		if s.Layer < 1 || target >= m.layers || m.lastTok[from] != s.First {
			return 0, nil
		}
		table, row = m.cross, m.xRow
	}
	n, w := m.nExpert, m.p.Mix
	clear(m.score)
	var den float64
	for _, a := range m.last[from] {
		den += float64(row[target*n+a])
	}
	if den > 0 && w > 0 {
		for _, a := range m.last[from] {
			for b, c := range table[(target*n+a)*n : (target*n+a+1)*n] {
				m.score[b] += float64(c)
			}
		}
		for b := range m.score {
			m.score[b] *= w / den
		}
	}
	if sum := m.sum[target]; sum > 0 && w < 1 {
		for b, f := range m.freq[target*n : (target+1)*n] {
			m.score[b] += (1 - w) * f / sum
		}
	}
	k = min(k, n)
	out = out[:0]
	for b, v := range m.score {
		if len(out) < k {
			out = append(out, b)
		} else if !(v > m.score[out[k-1]]) {
			continue
		}
		i := len(out) - 1
		for i > 0 && m.score[out[i-1]] < v {
			out[i] = out[i-1]
			i--
		}
		out[i] = b
	}
	return target, out
}

// stepExperts is the experts of layer l that span s uses, in the order Replay meets them, once
// each, appended to out.
func stepExperts(sel Selections, s Span, l int, out []int) []int {
	start := len(out)
	var seen map[int32]bool
	if s.End-s.First > 1 {
		seen = map[int32]bool{}
	}
	for tok := s.First; tok < s.End; tok++ {
		for _, e := range sel.IDs[(l*sel.Tokens+tok)*sel.K : (l*sel.Tokens+tok+1)*sel.K] {
			if seen != nil {
				if seen[e] {
					continue
				}
				seen[e] = true
			} else if slices.Contains(out[start:], int(e)) {
				continue
			}
			out = append(out, int(e))
		}
	}
	return out
}

// Ranking is a horizon's predictions before every step of a replay, the k best experts of each.
// What a predictor learns depends on the steps' experts only, never on a cache, so one ranking
// serves every cache and size it is replayed with.
type Ranking struct {
	h                  Horizon
	layers, nExpert, k int
	has                []bool  // per step: whether a prediction was made before it
	ids                []int16 // per step, its k candidates
}

// Rank runs a predictor with p through spans in Replay's order, predicting before every step with
// horizon h and learning each step's experts after it.
func Rank(sel Selections, spans []Span, nExpert int, h Horizon, p MarkovParams, k int) *Ranking {
	k = min(k, nExpert)
	m := NewMarkov(sel.Layers, nExpert, p)
	steps := len(spans) * sel.Layers
	r := &Ranking{h: h, layers: sel.Layers, nExpert: nExpert, k: k, has: make([]bool, steps),
		ids: make([]int16, steps*k)}
	var out, set []int
	step := 0
	for _, s := range spans {
		for l := range sel.Layers {
			info := StepInfo{step, s, l}
			var ids []int
			if _, ids = m.Predict(h, info, k, out); ids != nil {
				r.has[step] = true
				for i, e := range ids {
					r.ids[step*k+i] = int16(e)
				}
				out = ids
			}
			set = stepExperts(sel, s, l, set[:0])
			m.Observe(info, set)
			step++
		}
	}
	return r
}

// Steps is the number of steps the ranking covers.
func (r *Ranking) Steps() int { return len(r.has) }

// target is the layer the prediction before step s is for.
func (r *Ranking) target(s int) int {
	if r.h == SameToken {
		return s%r.layers + 1
	}
	return (s%r.layers + r.layers - 1) % r.layers
}

// Top is the layer and the k best experts predicted before step s, nil if none was; k is at most
// the ranking's.
func (r *Ranking) Top(s, k int) (int, []int) {
	if !r.has[s] {
		return 0, nil
	}
	out := make([]int, min(k, r.k))
	for i := range out {
		out[i] = int(r.ids[s*r.k+i])
	}
	return r.target(s), out
}

// Recall scores the ranking at every candidate count of ks, at most the ranking's, against the
// decoded steps of spans[from:to], which must be the spans it was ranked on: each against the
// prediction its horizon's lead before it. A decoded step that nothing predicts counts with no
// candidates and nothing covered.
func (r *Ranking) Recall(sel Selections, spans []Span, from, to int, ks []int) []Recall {
	out := make([]Recall, len(ks))
	lead := r.h.Lead(r.layers)
	var need []int
	pos := make([]int, 0, sel.K)
	for si := from; si < to; si++ {
		s := spans[si]
		if s.Phase != Decode {
			continue
		}
		for l := range sel.Layers {
			need = stepExperts(sel, s, l, need[:0])
			pos = pos[:0]
			j := si*r.layers + l - lead
			predicted := j >= 0 && r.has[j]
			for _, e := range need {
				p := r.k // not among the candidates
				if predicted {
					if i := slices.Index(r.ids[j*r.k:(j+1)*r.k], int16(e)); i >= 0 {
						p = i
					}
				}
				pos = append(pos, p)
			}
			for i, k := range ks {
				k = min(k, r.k)
				x := &out[i]
				x.Steps++
				x.Needed += len(need)
				if !predicted {
					continue
				}
				x.Candidates += k
				covered := 0
				for _, p := range pos {
					if p < k {
						covered++
					}
				}
				x.Covered += covered
				if covered == len(need) {
					x.Exact++
				}
			}
		}
	}
	return out
}

// Ranked prefetches, before every step, the k best experts a Ranking predicted before it, into
// the cache it wraps, which keeps and evicts them within its budget like any other expert. It
// must be replayed on the spans the ranking was ranked on.
type Ranked struct {
	Policy
	r   *Ranking
	k   int
	out []int
}

// NewRanked wraps base with r's k best candidates, k at most r's.
func NewRanked(base Policy, r *Ranking, k int) *Ranked {
	return &Ranked{Policy: base, r: r, k: min(k, r.k)}
}

func (p *Ranked) Prefetch(s StepInfo) []int {
	p.out = append(p.out[:0], p.Policy.Prefetch(s)...)
	if s.Index < len(p.r.has) && p.r.has[s.Index] {
		l := p.r.target(s.Index)
		for _, e := range p.r.ids[s.Index*p.r.k : s.Index*p.r.k+p.k] {
			p.out = append(p.out, l*p.r.nExpert+int(e))
		}
	}
	return p.out
}

// Prefetcher is the predictor as a policy: it learns a step's experts from Use and prefetches,
// before every step, its k best candidates for horizon h into the cache it wraps. Replayed on the
// same spans it prefetches what Rank ranks.
type Prefetcher struct {
	Policy
	m       *Markov
	h       Horizon
	k       int
	step    StepInfo
	set     []int
	pending bool
	ids     []int
	out     []int
}

// NewPrefetcher wraps base with m's predictions for horizon h, k of them before every step.
func NewPrefetcher(base Policy, m *Markov, h Horizon, k int) *Prefetcher {
	return &Prefetcher{Policy: base, m: m, h: h, k: k}
}

// flush lets the predictor learn the last step's experts.
func (p *Prefetcher) flush() {
	if p.pending {
		p.m.Observe(p.step, p.set)
		p.set, p.pending = p.set[:0], false
	}
}

func (p *Prefetcher) Prefetch(s StepInfo) []int {
	p.flush()
	p.out = append(p.out[:0], p.Policy.Prefetch(s)...)
	l, ids := p.m.Predict(p.h, s, p.k, p.ids)
	for _, e := range ids {
		p.out = append(p.out, l*p.m.nExpert+e)
	}
	if ids != nil {
		p.ids = ids
	}
	return p.out
}

func (p *Prefetcher) Use(s StepInfo, k int) {
	if !p.pending || s.Index != p.step.Index {
		p.flush()
		p.step, p.pending = s, true
	}
	p.set = append(p.set, k%p.m.nExpert)
	p.Policy.Use(s, k)
}
