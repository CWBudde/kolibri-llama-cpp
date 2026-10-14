package locality

// The last-N reuse predictor (PLAN Phase 10, Stage A): the first prediction baseline. It predicts
// that a layer needs again the experts it needed in its last n steps, and prefetches them one
// layer ahead, while the layer before runs. It knows only steps that have run: what Use reported.

// LastN prefetches, before layer l of a step, the experts layer l+1 used in its last n steps;
// before the last layer, those layer 0 used in its last n steps, the running one included, for
// the next step. Everything else is the cache it wraps: what that keeps and evicts within its
// budget, the prefetched experts included.
type LastN struct {
	Policy
	n, layers int
	recent    [][][]int // per layer, the keys of its last n steps, a ring
	at        []int     // per layer, the ring's slot of its latest step
	index     []int     // per layer, the StepInfo.Index of its latest step
	mark      []int     // per key, the Prefetch call that listed it last
	calls     int
	out       []int
}

// NewLastN wraps base, for layers of nExpert experts, with prefetches of the last n steps.
func NewLastN(base Policy, layers, nExpert, n int) *LastN {
	p := &LastN{Policy: base, n: n, layers: layers, recent: make([][][]int, layers), at: make([]int, layers),
		index: make([]int, layers), mark: make([]int, layers*nExpert)}
	for l := range layers {
		p.recent[l] = make([][]int, n)
		p.index[l] = -1
	}
	return p
}

func (p *LastN) Prefetch(s StepInfo) []int {
	p.calls++
	p.out = append(p.out[:0], p.Policy.Prefetch(s)...)
	for _, set := range p.recent[(s.Layer+1)%p.layers] {
		for _, k := range set {
			if p.mark[k] != p.calls {
				p.mark[k] = p.calls
				p.out = append(p.out, k)
			}
		}
	}
	return p.out
}

func (p *LastN) Use(s StepInfo, k int) {
	l := s.Layer
	if p.index[l] != s.Index {
		p.index[l] = s.Index
		p.at[l] = (p.at[l] + 1) % p.n
		p.recent[l][p.at[l]] = p.recent[l][p.at[l]][:0]
	}
	p.recent[l][p.at[l]] = append(p.recent[l][p.at[l]], k)
	p.Policy.Use(s, k)
}

// Recall is how well the last n steps of a layer cover a decoded token's experts in it, summed
// over the decoded tokens and layers: Candidates are the experts predicted, Needed those the token
// selects, Covered those of them predicted, Exact the tokens and layers predicted completely.
type Recall struct {
	Steps      int `json:"steps"`
	Candidates int `json:"candidates"`
	Needed     int `json:"needed"`
	Covered    int `json:"covered"`
	Exact      int `json:"exact"`
}

// LastNRecall is LastN's Recall over the decoded spans from spans[from] on: each predicted from
// the layer's experts in the n spans before it, which may lie before from.
func LastNRecall(sel Selections, spans []Span, from, nExpert, n int) Recall {
	var r Recall
	last := make([]int, nExpert) // per expert, the last span that used it in the layer, plus one
	for l := range sel.Layers {
		clear(last)
		for i, s := range spans {
			if i >= from && s.Phase == Decode {
				r.Steps++
				for _, u := range last {
					if u > 0 && u >= i+1-n {
						r.Candidates++
					}
				}
				exact := true
				for _, e := range sel.IDs[(l*sel.Tokens+s.First)*sel.K : (l*sel.Tokens+s.First+1)*sel.K] {
					r.Needed++
					if u := last[e]; u > 0 && u >= i+1-n {
						r.Covered++
					} else {
						exact = false
					}
				}
				if exact {
					r.Exact++
				}
			}
			for tok := s.First; tok < s.End; tok++ {
				for _, e := range sel.IDs[(l*sel.Tokens+tok)*sel.K : (l*sel.Tokens+tok+1)*sel.K] {
					last[e] = i + 1
				}
			}
		}
	}
	return r
}
