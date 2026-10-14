package locality

import (
	"math/rand/v2"
	"slices"
	"testing"
)

// decoded is one decoded token per span, all of them.
func decoded(tokens int) []Span {
	out := make([]Span, tokens)
	for i := range out {
		out[i] = Span{Decode, i, i + 1}
	}
	return out
}

// observeAll feeds m every step of spans, a step's experts in replay order and once each.
func observeAll(m *Markov, sel Selections, spans []Span) {
	step := 0
	for _, s := range spans {
		for l := range sel.Layers {
			m.Observe(StepInfo{step, s, l}, stepExperts(sel, s, l, nil))
			step++
		}
	}
}

func TestMarkovScores(t *testing.T) {
	// one layer of 5 experts, decoded {0,1} {1,2} {1,3}, counts halved every token. Then the counts
	// are 0.25, 1.75, 0.5, 1 and 0 (sum 3.5), and the transitions out of {1,3}, from 1 only, go to
	// 1 twice, to 2 and to 3 once each, of 4.
	sel := Selections{Layers: 1, Tokens: 4, K: 2, IDs: []int32{0, 1, 1, 2, 1, 3, 0, 0}}
	next := StepInfo{3, Span{Decode, 3, 4}, 0}
	for _, c := range []struct {
		mix  float64
		want []int
	}{
		{0, []int{1, 3, 2, 0, 4}},   // counts: 1.75, 1, 0.5, 0.25, 0
		{1, []int{1, 2, 3, 0, 4}},   // transitions: 0.5, 0.25, 0.25 (ties to the lower), 0, 0
		{0.9, []int{1, 3, 2, 0, 4}}, // 3: 0.9·0.25 + 0.1·1/3.5 = 0.254 > 2: 0.225 + 0.1·0.5/3.5 = 0.239
	} {
		m := NewMarkov(1, 5, MarkovParams{HalfLife: 1, Mix: c.mix})
		observeAll(m, sel, decoded(3))
		layer, got := m.Predict(NextToken, next, 5, nil)
		if layer != 0 || !slices.Equal(got, c.want) {
			t.Fatalf("mix %g: layer %d, ranked %v, want 0, %v", c.mix, layer, got, c.want)
		}
		if _, top := m.Predict(NextToken, next, 2, nil); !slices.Equal(top, c.want[:2]) {
			t.Fatalf("mix %g: top 2 %v, want %v", c.mix, top, c.want[:2])
		}
	}
	// three layers of 3 experts, one each. Layer 2 followed layer 0's 0 by 2 and its 1 by 1, so the
	// same token's layer 2 after a 0 in layer 0 is 2. Only layer 1's step predicts it: layer 0's has
	// no completed layer of the token, and nothing follows layer 2.
	sel = Selections{Layers: 3, Tokens: 3, K: 1, IDs: []int32{0, 1, 0, 1, 2, 0, 2, 1, 0}}
	m := NewMarkov(3, 3, MarkovParams{HalfLife: 64, Mix: 1})
	observeAll(m, sel, decoded(2))
	m.Observe(StepInfo{6, Span{Decode, 2, 3}, 0}, []int{0})
	if layer, got := m.Predict(SameToken, StepInfo{7, Span{Decode, 2, 3}, 1}, 1, nil); layer != 2 ||
		!slices.Equal(got, []int{2}) {
		t.Fatalf("same token: layer %d, %v, want 2, [2]", layer, got)
	}
	for _, l := range []int{0, 2} {
		if _, got := m.Predict(SameToken, StepInfo{6 + l, Span{Decode, 2, 3}, l}, 1, nil); got != nil {
			t.Fatalf("same token before layer %d: %v, want nothing", l, got)
		}
	}
	// the next token in layer 0, predicted before the running token's layer 1
	if layer, got := m.Predict(NextToken, StepInfo{7, Span{Decode, 2, 3}, 1}, 3, nil); layer != 0 || len(got) != 3 {
		t.Fatalf("next token: layer %d, %v", layer, got)
	}
	// nothing before the first step of a trace decoded from token 0: no set has run yet
	if _, got := NewMarkov(3, 3, MarkovParams{HalfLife: 64}).Predict(NextToken, StepInfo{0, Span{Decode, 0, 1}, 0}, 3,
		nil); got != nil {
		t.Fatalf("before any step: %v", got)
	}
	// nothing before a prefill step, and nothing from a ubatch
	if _, got := m.Predict(NextToken, StepInfo{7, Span{Prefill, 2, 3}, 1}, 3, nil); got != nil {
		t.Fatalf("before a prefill step: %v", got)
	}
	m.Observe(StepInfo{9, Span{Prefill, 3, 5}, 0}, []int{0, 1})
	if _, got := m.Predict(NextToken, StepInfo{10, Span{Decode, 5, 6}, 1}, 3, nil); got != nil {
		t.Fatalf("after a ubatch: %v", got)
	}
}

func TestMarkovCausal(t *testing.T) {
	r := rand.New(rand.NewPCG(21, 22))
	sel, spans, _ := onlineSplit(r)
	for _, h := range []Horizon{NextToken, SameToken} {
		p := MarkovParams{HalfLife: 16, Mix: 0.5}
		base := Rank(sel, spans, 12, h, p, 6)
		changed := false
		for range 20 {
			// redraw every expert from step (i, l) on: the predictions before it and before every
			// earlier step stay as they were
			i, l := r.IntN(len(spans)), r.IntN(sel.Layers)
			re := Selections{Layers: sel.Layers, Tokens: sel.Tokens, K: sel.K, IDs: slices.Clone(sel.IDs)}
			other := randomSelections(r, sel.Layers, sel.Tokens, sel.K, 12)
			for ll := range sel.Layers {
				from := spans[i].First
				if ll < l {
					from = spans[i].End
				}
				copy(re.IDs[(ll*sel.Tokens+from)*sel.K:(ll+1)*sel.Tokens*sel.K],
					other.IDs[(ll*sel.Tokens+from)*sel.K:(ll+1)*sel.Tokens*sel.K])
			}
			got := Rank(re, spans, 12, h, p, 6)
			cut := i*sel.Layers + l
			for s := range base.Steps() {
				la, a := base.Top(s, 6)
				lb, b := got.Top(s, 6)
				if s <= cut && (la != lb || !slices.Equal(a, b)) {
					t.Fatalf("%v: before step %d the prediction changed with steps from %d on: %v, %v", h, s, cut, a, b)
				}
				if s > cut && !slices.Equal(a, b) {
					changed = true
				}
			}
		}
		if !changed {
			t.Fatalf("%v: redrawing the experts never changed a later prediction", h)
		}
	}
}

// keysAt is a logged prefetch set as the experts of one layer; ok is false if it spans layers.
func keysAt(keys []int, nExpert int) (layer int, ids []int, ok bool) {
	layer = -1
	for _, k := range keys {
		if layer >= 0 && k/nExpert != layer {
			return 0, nil, false
		}
		layer = k / nExpert
		ids = append(ids, k%nExpert)
	}
	return layer, ids, true
}

func TestPrefetcher(t *testing.T) {
	r := rand.New(rand.NewPCG(23, 24))
	sel, spans, _ := onlineSplit(r)
	bytes := sizes(sel.Layers, 10)
	p := MarkovParams{HalfLife: 32, Mix: 0.75}
	for _, h := range []Horizon{NextToken, SameToken} {
		ranking := Rank(sel, spans, 12, h, p, 12)
		for _, k := range []int{1, 3, 12} {
			// online, learning from Use only, it prefetches the ranking's top k
			online := &prefetchLog{Policy: NewPrefetcher(&NoCache{}, NewMarkov(sel.Layers, 12, p), 12, h, k)}
			ranked := &prefetchLog{Policy: NewRanked(&NoCache{}, ranking, k)}
			for _, q := range []*prefetchLog{online, ranked} {
				if _, err := Replay(sel, spans, 12, bytes, q); err != nil {
					t.Fatal(err)
				}
			}
			lead := h.Lead(sel.Layers)
			step := 0
			for i, s := range spans {
				for l := range sel.Layers {
					// a prediction before a decoded step whose last step was decoded: the next token's
					// layer just completed, or the running token's layer after next
					want := s.Phase == Decode && (h == NextToken && (l > 0 || i > 0 && spans[i-1].Phase == Decode) ||
						h == SameToken && l >= 1 && l+1 < sel.Layers)
					target := (l - 1 + sel.Layers) % sel.Layers
					if h == SameToken {
						target = l + 1
					}
					ol, oids, ok := keysAt(online.sets[step], 12)
					if !ok || want != (len(oids) > 0) || want && (ol != target || len(oids) != k) {
						t.Fatalf("%v, k %d, step %d (span %d, layer %d): prefetches %v", h, k, step, i, l,
							online.sets[step])
					}
					if !slices.Equal(online.sets[step], ranked.sets[step]) {
						t.Fatalf("%v, k %d, step %d: online %v, ranked %v", h, k, step, online.sets[step],
							ranked.sets[step])
					}
					if want && (step+lead)%sel.Layers != target {
						t.Fatalf("%v: lead %d from layer %d misses layer %d", h, lead, l, target)
					}
					step++
				}
			}
			// within a shared LRU's budget
			for _, budget := range []int64{0, 30, 120} {
				res, err := Replay(sel, spans, 12, bytes, NewRanked(NewSharedLRU(12, bytes, budget), ranking, k))
				if err != nil {
					t.Fatal(err)
				}
				if res.PeakBytes > budget {
					t.Fatalf("budget %d: peak %d", budget, res.PeakBytes)
				}
			}
		}
	}
}

func TestPredictRecall(t *testing.T) {
	r := rand.New(rand.NewPCG(25, 26))
	sel, spans, cut := onlineSplit(r)
	ks := []int{1, 4, 12}
	for _, h := range []Horizon{NextToken, SameToken} {
		ranking := Rank(sel, spans, 12, h, MarkovParams{HalfLife: 8, Mix: 0.5}, 12)
		for _, rg := range [][2]int{{0, cut}, {cut, len(spans)}} {
			got := ranking.Recall(sel, spans, rg[0], rg[1], ks)
			for i, k := range ks {
				// by brute force: what the ranked prefetcher prefetched the lead before each decoded
				// step, scored against the step; an unpredicted step counts with nothing covered
				log := &prefetchLog{Policy: NewRanked(&NoCache{}, ranking, k)}
				if _, err := Replay(sel, spans, 12, sizes(sel.Layers, 1), log); err != nil {
					t.Fatal(err)
				}
				var want Recall
				for si := rg[0]; si < rg[1]; si++ {
					s := spans[si]
					if s.Phase != Decode {
						continue
					}
					for l := range sel.Layers {
						need := stepExperts(sel, s, l, nil)
						want.Steps++
						want.Needed += len(need)
						j := si*sel.Layers + l - h.Lead(sel.Layers)
						if j < 0 {
							continue
						}
						layer, ids, ok := keysAt(log.sets[j], 12)
						if !ok || len(ids) > 0 && layer != l {
							t.Fatalf("%v: the prefetch before step %d is for layer %d, not %d", h, j, layer, l)
						}
						want.Candidates += len(ids)
						covered := 0
						for _, e := range need {
							if slices.Contains(ids, e) {
								covered++
							}
						}
						want.Covered += covered
						if len(ids) > 0 && covered == len(need) {
							want.Exact++
						}
					}
				}
				if got[i] != want {
					t.Fatalf("%v, spans %v, k %d: %+v, want %+v", h, rg, k, got[i], want)
				}
			}
		}
	}
}

func TestOverlap(t *testing.T) {
	if NextToken.Lead(50) != 49 || SameToken.Lead(50) != 1 {
		t.Fatalf("leads %d and %d at 50 layers", NextToken.Lead(50), SameToken.Lead(50))
	}
	if got := LeadSeconds(Timing{DecodeStep: 2e-3}, NextToken, 50); got != 49*2e-3 {
		t.Fatalf("next token: %g s", got)
	}
	// three layers: a token's layer 0 expert is its layer 2 expert, layer 1's is always 0. The
	// prefetch for layer 2, issued before layer 1, which hits, has one step's compute to arrive;
	// only token 0's has longer, while its layer 1 waits for its own miss.
	const tokens = 40
	sel := Selections{Layers: 3, Tokens: tokens, K: 1, IDs: make([]int32, 3*tokens)}
	for tok := range tokens {
		e := int32(tok % 4)
		sel.IDs[tok], sel.IDs[2*tokens+tok] = e, e
	}
	bytes := []int64{1e6, 1e6, 1e6}
	ranking := Rank(sel, decoded(tokens), 4, SameToken, MarkovParams{HalfLife: 64, Mix: 1}, 1)
	for _, c := range []struct {
		bw   float64
		late bool
	}{{1e6 / 1e-3, false}, {1e6 / 3e-3, true}} { // an expert takes 1 or 3 ms, a step 2 ms
		res, err := ReplayTimed(sel, decoded(tokens), 4, bytes, NewRanked(NewSharedLRU(4, bytes, 4e6), ranking, 1),
			Timing{Links: []Link{{"link", c.bw}}, DecodeStep: 2e-3})
		if err != nil {
			t.Fatal(err)
		}
		lt := res.Decode.Links[0]
		timely := res.Decode.UsefulPrefetches
		if c.late {
			timely = 1
		}
		if res.Decode.UsefulPrefetches < tokens/2 || lt.Timely != timely {
			t.Fatalf("%g B/s: %d useful prefetches, %d timely, %d late", c.bw, res.Decode.UsefulPrefetches, lt.Timely,
				lt.Late)
		}
	}
}

func TestMarkovBytes(t *testing.T) {
	// two transition tables of float32 counts and their row sums, decayed float64 counts and sums
	m := NewMarkov(2, 3, MarkovParams{HalfLife: 1, Mix: 0.5})
	if got, want := m.Bytes(), int64(2*2*(3*3+3)*4+2*(3+1)*8); got != want {
		t.Fatalf("%d bytes, want %d", got, want)
	}
}

func BenchmarkMarkovStep(b *testing.B) {
	// the real models' shape: 50 layers of 384 experts, 6 a token, 48 candidates
	const layers, nExpert, k, tokens = 50, 384, 6, 256
	r := rand.New(rand.NewPCG(27, 28))
	sel := randomSelections(r, layers, tokens, k, nExpert)
	sets := make([][]int, layers*tokens)
	for tok := range tokens {
		for l := range layers {
			sets[tok*layers+l] = stepExperts(sel, Span{Decode, tok, tok + 1}, l, nil)
		}
	}
	info := func(i int) StepInfo {
		tok := i / layers
		return StepInfo{i, Span{Decode, tok, tok + 1}, i % layers}
	}
	for _, h := range []Horizon{NextToken, SameToken} {
		b.Run(h.String(), func(b *testing.B) {
			m := NewMarkov(layers, nExpert, MarkovParams{HalfLife: 256, Mix: 0.5})
			out := make([]int, 0, 48)
			step := 0
			for range layers * tokens {
				m.Observe(info(step), sets[step%len(sets)])
				step++
			}
			b.ResetTimer()
			for range b.N {
				_, out = m.Predict(h, info(step), 48, out[:0])
				m.Observe(info(step), sets[step%len(sets)])
				step++
			}
		})
	}
}
