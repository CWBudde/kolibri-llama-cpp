package locality

// The linear predictor (PLAN Phase 10, Stage B, item 2): a small multi-label model trained on the
// recorded expert sets of training prefixes, then frozen. Per layer it is a logistic regression
// from two multi-hot blocks of known experts to the target step's experts, with one bias per
// expert:
//
//   - next token: token t's experts in layer m from token t-1's (block A) and t-2's (block B) in m;
//   - same token: token t's experts in layer m from its own in layer m-2 (A) and token t-1's in
//     layer m (B), which is the target layer's recency.
//
// The targets and inputs are those of the Markov predictor's horizons (predict.go). Trained, it
// reads recorded sets of every token, a prompt's included; replayed, it knows only decoded steps
// that have run, as every policy does. It never sees router logits or hidden states, which are
// not recorded.

import (
	"math"
	"math/rand/v2"
	"slices"
	"sync"
)

// LinearBlocks are the input blocks a linear predictor uses.
type LinearBlocks int

const (
	// BlockA is the nearest known set: the token before in the layer, or the token's layer two
	// below.
	BlockA LinearBlocks = 1 << iota
	// BlockB is the second: the token two before in the layer, or the token before in the target
	// layer.
	BlockB
)

// LinearConfig is how a linear predictor is fitted: its blocks, an L2 penalty on the weights of a
// sample's inputs, AdaGrad's step, the epochs over the samples and the seed of their order.
type LinearConfig struct {
	Blocks LinearBlocks `json:"blocks"`
	L2     float64      `json:"l2"`
	Rate   float64      `json:"rate"`
	Epochs int          `json:"epochs"`
	Seed   uint64       `json:"seed"`
}

// LinearTrace is a recorded trace and the tokens whose experts a model fits or is scored on, in
// every layer that has the horizon's inputs.
type LinearTrace struct {
	Sel     Selections
	Targets []int
}

// Linear is a fitted linear predictor for one horizon.
type Linear struct {
	h               Horizon
	blocks          LinearBlocks
	layers, nExpert int
	w               []float32 // per layer, 2·nExpert input rows (A's experts, then B's) of nExpert
	b               []float32 // per layer, nExpert biases
}

func newLinear(layers, nExpert int, h Horizon, blocks LinearBlocks) *Linear {
	return &Linear{h: h, blocks: blocks, layers: layers, nExpert: nExpert,
		w: make([]float32, layers*2*nExpert*nExpert), b: make([]float32, layers*nExpert)}
}

// Clone is a copy of m that later training of m does not change.
func (m *Linear) Clone() *Linear {
	c := *m
	c.w, c.b = slices.Clone(m.w), slices.Clone(m.b)
	return &c
}

// LinearBytes is the size of a linear predictor's weights and biases for layers of nExpert
// experts, in float32.
func LinearBytes(layers, nExpert int) int64 {
	l, n := int64(layers), int64(nExpert)
	return l * (2*n*n + n) * 4
}

// linearInputs is where the inputs of target token t in layer m of horizon h are: block A's token
// and layer, and block B's, bTok -1 for none. ok is false when the target has no block A.
func linearInputs(h Horizon, t, m int) (aTok, aLayer, bTok, bLayer int, ok bool) {
	if h == SameToken {
		return t, m - 2, t - 1, m, m >= 2
	}
	return t - 1, m, max(t-2, -1), m, t >= 1
}

// recorded is the experts of token t in layer l.
func recorded(sel Selections, t, l int) []int32 {
	return sel.IDs[(l*sel.Tokens+t)*sel.K : (l*sel.Tokens+t+1)*sel.K]
}

// logits are layer l's scores for the inputs a and b, b nil for none, in z.
func logits[T int | int32](m *Linear, l int, a, b []T, z []float32) {
	n := m.nExpert
	copy(z, m.b[l*n:(l+1)*n])
	rows := m.w[l*2*n*n : (l+1)*2*n*n]
	if m.blocks&BlockA != 0 {
		for _, e := range a {
			for o, v := range rows[int(e)*n : (int(e)+1)*n] {
				z[o] += v
			}
		}
	}
	if m.blocks&BlockB != 0 {
		for _, e := range b {
			for o, v := range rows[(n+int(e))*n : (n+int(e)+1)*n] {
				z[o] += v
			}
		}
	}
}

func sigmoid(z float64) float64 { return 1 / (1 + math.Exp(-z)) }

// sample is a target token of a trace.
type sample struct{ tr, t int32 }

// FitLinear fits a linear predictor for horizon h on the targets of data, every layer apart: the
// binary cross-entropy of all nExpert outputs, AdaGrad, the samples in an order shuffled per epoch
// and layer from cfg.Seed. After every epoch, epoch, if not nil, gets the model so far, which it
// must Clone to keep. A target's inputs are read from tokens before it, or from its own token's
// layer two below, so the fit reads no token after the last target.
func FitLinear(data []LinearTrace, nExpert int, h Horizon, cfg LinearConfig, epoch func(e int, m *Linear)) *Linear {
	layers := data[0].Sel.Layers
	m := newLinear(layers, nExpert, h, cfg.Blocks)
	n := nExpert
	k := data[0].Sel.K
	prior := float32(math.Log(float64(k) / float64(n-k)))
	for i := range m.b {
		m.b[i] = prior
	}
	type state struct {
		samples []sample
		gw, gb  []float32
		rng     *rand.Rand
		z       []float32
	}
	st := make([]state, layers)
	for l := range st {
		s := &st[l]
		for i, d := range data {
			for _, t := range d.Targets {
				if _, _, _, _, ok := linearInputs(h, t, l); ok && t < d.Sel.Tokens {
					s.samples = append(s.samples, sample{int32(i), int32(t)})
				}
			}
		}
		s.gw, s.gb = make([]float32, 2*n*n), make([]float32, n)
		s.rng = rand.New(rand.NewPCG(cfg.Seed, uint64(l)))
		s.z = make([]float32, n)
	}
	rate, l2 := float32(cfg.Rate), float32(cfg.L2)
	const eps = 1e-8
	step := func(l int, s *state, x sample) {
		sel := data[x.tr].Sel
		aTok, aLayer, bTok, bLayer, _ := linearInputs(h, int(x.t), l)
		a := recorded(sel, aTok, aLayer)
		var b []int32
		if bTok >= 0 {
			b = recorded(sel, bTok, bLayer)
		}
		logits(m, l, a, b, s.z)
		g := s.z // the loss's gradient in the logits, in place
		for o, z := range g {
			g[o] = float32(sigmoid(float64(z)))
		}
		for _, e := range recorded(sel, int(x.t), l) {
			g[e]--
		}
		bias := m.b[l*n : (l+1)*n]
		for o, v := range g {
			s.gb[o] += v * v
			bias[o] -= rate * v / (float32(math.Sqrt(float64(s.gb[o]))) + eps)
		}
		update := func(row int) {
			w := m.w[(l*2*n+row)*n : (l*2*n+row+1)*n]
			gw := s.gw[row*n : (row+1)*n]
			for o, v := range g {
				d := v + l2*w[o]
				gw[o] += d * d
				w[o] -= rate * d / (float32(math.Sqrt(float64(gw[o]))) + eps)
			}
		}
		if cfg.Blocks&BlockA != 0 {
			for _, e := range a {
				update(int(e))
			}
		}
		if cfg.Blocks&BlockB != 0 {
			for _, e := range b {
				update(n + int(e))
			}
		}
	}
	for e := 1; e <= cfg.Epochs; e++ {
		var wg sync.WaitGroup
		for l := range st {
			wg.Add(1)
			go func() {
				defer wg.Done()
				s := &st[l]
				s.rng.Shuffle(len(s.samples), func(i, j int) { s.samples[i], s.samples[j] = s.samples[j], s.samples[i] })
				for _, x := range s.samples {
					step(l, s, x)
				}
			}()
		}
		wg.Wait()
		if epoch != nil {
			epoch(e, m)
		}
	}
	return m
}

// targets calls f for every target of tr that has the horizon's inputs, with its layer and its
// inputs' and own recorded sets.
func (m *Linear) targets(tr LinearTrace, f func(l int, a, b, want []int32)) {
	for _, t := range tr.Targets {
		for l := range m.layers {
			aTok, aLayer, bTok, bLayer, ok := linearInputs(m.h, t, l)
			if !ok {
				continue
			}
			var b []int32
			if bTok >= 0 {
				b = recorded(tr.Sel, bTok, bLayer)
			}
			f(l, recorded(tr.Sel, aTok, aLayer), b, recorded(tr.Sel, t, l))
		}
	}
}

// Recall scores the model's best k experts of every k of ks on tr's targets, from their recorded
// inputs.
func (m *Linear) Recall(tr LinearTrace, ks []int) []Recall {
	out := make([]Recall, len(ks))
	z := make([]float32, m.nExpert)
	var top []int
	kmax := min(slices.Max(ks), m.nExpert)
	m.targets(tr, func(l int, a, b, want []int32) {
		logits(m, l, a, b, z)
		top = topK(z, kmax, top)
		for i, k := range ks {
			k = min(k, kmax)
			x := &out[i]
			x.Steps++
			x.Needed += len(want)
			x.Candidates += k
			covered := 0
			for _, e := range want {
				if slices.Contains(top[:k], int(e)) {
					covered++
				}
			}
			x.Covered += covered
			if covered == len(want) {
				x.Exact++
			}
		}
	})
	return out
}

// Temperature is the temperature T of the model's logits, p = sigmoid(z/T), that fits tr's
// targets best: the least binary cross-entropy of all outputs.
func (m *Linear) Temperature(tr LinearTrace) float64 {
	var z []float32
	var y []bool
	buf := make([]float32, m.nExpert)
	m.targets(tr, func(l int, a, b, want []int32) {
		logits(m, l, a, b, buf)
		z = append(z, buf...)
		at := len(y)
		y = append(y, make([]bool, m.nExpert)...)
		for _, e := range want {
			y[at+int(e)] = true
		}
	})
	return fitTemperature(z, y)
}

// fitTemperature is the T in [0.05, 20] with the least mean binary cross-entropy of sigmoid(z/T)
// against y, by golden-section search on log T.
func fitTemperature(z []float32, y []bool) float64 {
	loss := func(u float64) float64 {
		t := math.Exp(u)
		var s float64
		for i, v := range z {
			x := float64(v) / t
			// log(1+e^x) - y·x, stably
			s += math.Max(x, 0) + math.Log1p(math.Exp(-math.Abs(x)))
			if y[i] {
				s -= x
			}
		}
		return s
	}
	lo, hi := math.Log(0.05), math.Log(20)
	g := (math.Sqrt(5) - 1) / 2
	a, b := hi-g*(hi-lo), lo+g*(hi-lo)
	fa, fb := loss(a), loss(b)
	for hi-lo > 1e-4 {
		if fa < fb {
			hi, b, fb = b, a, fa
			a = hi - g*(hi-lo)
			fa = loss(a)
		} else {
			lo, a, fa = a, b, fb
			b = lo + g*(hi-lo)
			fb = loss(b)
		}
	}
	return math.Exp((lo + hi) / 2)
}

// Calibration is how well a model's probabilities match: at every k of K the probabilities of its
// best k experts summed over the targets (the experts it expects to cover) against those it
// covers, and the expected calibration error over its Candidates best experts of all targets, in
// ten bins of probability.
type Calibration struct {
	K          []int     `json:"k"`
	Expected   []float64 `json:"expected_covered"`
	Covered    []int     `json:"covered"`
	ECE        float64   `json:"ece"`
	Candidates int       `json:"candidates"`
}

// Calibration is the model's calibration on tr's targets at temperature temp, the ECE over its
// best top experts of each.
func (m *Linear) Calibration(tr LinearTrace, temp float64, ks []int, top int) Calibration {
	var ps [][]float64
	var hits [][]bool
	z := make([]float32, m.nExpert)
	var best []int
	m.targets(tr, func(l int, a, b, want []int32) {
		logits(m, l, a, b, z)
		best = topK(z, top, best)
		p, h := make([]float64, len(best)), make([]bool, len(best))
		for i, e := range best {
			p[i] = sigmoid(float64(z[e]) / temp)
			h[i] = slices.Contains(want, int32(e))
		}
		ps, hits = append(ps, p), append(hits, h)
	})
	return calibrate(ps, hits, ks, top)
}

// calibrate is the Calibration of targets whose best candidates have probabilities ps, best
// first, and are hits or not.
func calibrate(ps [][]float64, hits [][]bool, ks []int, top int) Calibration {
	c := Calibration{K: ks, Expected: make([]float64, len(ks)), Covered: make([]int, len(ks))}
	var sum, hit [10]float64
	var count [10]int
	for i, p := range ps {
		for j, k := range ks {
			for r := range min(k, len(p)) {
				c.Expected[j] += p[r]
				if hits[i][r] {
					c.Covered[j]++
				}
			}
		}
		for r := range min(top, len(p)) {
			bin := min(int(p[r]*10), 9)
			sum[bin] += p[r]
			count[bin]++
			if hits[i][r] {
				hit[bin]++
			}
			c.Candidates++
		}
	}
	for b := range sum {
		if count[b] > 0 {
			c.ECE += math.Abs(sum[b]-hit[b]) / float64(c.Candidates)
		}
	}
	return c
}

// noToken marks a layer that has no decoded set yet.
const noToken = math.MinInt32

// LinearState is a fitted model as an online predictor: it keeps every layer's last two decoded
// sets and learns nothing.
type LinearState struct {
	m                *Linear
	last, prev       [][]int
	lastTok, prevTok []int
	score            []float32
}

// Online is a predictor of the frozen model m that has seen nothing.
func (m *Linear) Online() *LinearState {
	p := &LinearState{m: m, last: make([][]int, m.layers), prev: make([][]int, m.layers),
		lastTok: make([]int, m.layers), prevTok: make([]int, m.layers), score: make([]float32, m.nExpert)}
	for l := range p.lastTok {
		p.lastTok[l], p.prevTok[l] = noToken, noToken
	}
	return p
}

// Observe keeps a decoded step's experts; a ubatch's union is no token's set.
func (p *LinearState) Observe(s StepInfo, ids []int) {
	if s.Phase != Decode {
		return
	}
	l := s.Layer
	p.prev[l], p.last[l] = p.last[l], p.prev[l]
	p.prevTok[l] = p.lastTok[l]
	p.last[l] = append(p.last[l][:0], ids...)
	p.lastTok[l] = s.First
}

// Predict ranks the target of horizon h before step s, as Markov.Predict does; a model predicts
// only the horizon it was fitted for, and only when its block A has been decoded.
func (p *LinearState) Predict(h Horizon, s StepInfo, k int, out []int) (int, []int) {
	m := p.m
	if h != m.h || s.Phase != Decode || k <= 0 {
		return 0, nil
	}
	var target int
	var a, b []int
	switch h {
	case NextToken:
		from, tok := s.Layer-1, s.First
		if s.Layer == 0 {
			from, tok = m.layers-1, s.First-1
		}
		if p.lastTok[from] != tok {
			return 0, nil
		}
		target, a = from, p.last[from]
		if p.prevTok[from] == tok-1 {
			b = p.prev[from]
		}
	case SameToken:
		target = s.Layer + 1
		from := s.Layer - 1
		if s.Layer < 1 || target >= m.layers || p.lastTok[from] != s.First {
			return 0, nil
		}
		a = p.last[from]
		if p.lastTok[target] == s.First-1 {
			b = p.last[target]
		}
	}
	logits(m, target, a, b, p.score)
	return target, topK(p.score, min(k, m.nExpert), out)
}
