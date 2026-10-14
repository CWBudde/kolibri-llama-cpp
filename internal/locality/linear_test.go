package locality

import (
	"math"
	"math/rand/v2"
	"reflect"
	"slices"
	"testing"
)

func TestLinearInputs(t *testing.T) {
	// next token: token t in layer m from t-1 and t-2 in m; same token: from the token's layer
	// m-2 and token t-1 in m
	for _, c := range []struct {
		h          Horizon
		t, m       int
		aTok, aLay int
		bTok, bLay int
		ok         bool
	}{
		{NextToken, 5, 3, 4, 3, 3, 3, true},
		{NextToken, 1, 0, 0, 0, -1, 0, true}, // no token before 0: no block B
		{NextToken, 0, 0, 0, 0, 0, 0, false}, // nothing before token 0
		{SameToken, 5, 3, 5, 1, 4, 3, true},
		{SameToken, 0, 2, 0, 0, -1, 2, true},
		{SameToken, 5, 1, 0, 0, 0, 0, false}, // layers 0 and 1 have no layer two below
	} {
		aTok, aLay, bTok, bLay, ok := linearInputs(c.h, c.t, c.m)
		if ok != c.ok || ok && (aTok != c.aTok || aLay != c.aLay || bTok != c.bTok || bTok >= 0 && bLay != c.bLay) {
			t.Errorf("%v, token %d, layer %d: %d %d %d %d %v", c.h, c.t, c.m, aTok, aLay, bTok, bLay, ok)
		}
	}
}

// mapped is a trace of 3 layers of 8 experts, 2 a token: layer 0 is random, layer 1 noise, and
// layer 2 the token's layer 0 shifted by 3 experts.
func mapped(r *rand.Rand, tokens int) Selections {
	sel := randomSelections(r, 3, tokens, 2, 8)
	for tok := range tokens {
		for j := range 2 {
			sel.IDs[(2*tokens+tok)*2+j] = (sel.IDs[tok*2+j] + 3) % 8
		}
	}
	return sel
}

// tokensFrom is the tokens [from, to).
func tokensFrom(from, to int) []int {
	out := make([]int, 0, to-from)
	for t := from; t < to; t++ {
		out = append(out, t)
	}
	return out
}

func TestLinearLearns(t *testing.T) {
	r := rand.New(rand.NewPCG(31, 32))
	sel := mapped(r, 1000)
	fit := []LinearTrace{{sel, tokensFrom(0, 800)}}
	test := LinearTrace{sel, tokensFrom(800, 1000)}
	cfg := LinearConfig{Blocks: BlockA | BlockB, Rate: 0.2, Epochs: 3, Seed: 1}
	m := FitLinear(fit, 8, SameToken, cfg, nil)
	// layer 2 from layer 0, the only same-token target here: every set at k 2
	if got := m.Recall(test, []int{2})[0]; got.Covered != got.Needed || got.Exact != got.Steps || got.Steps != 200 {
		t.Fatalf("with block A: %+v", got)
	}
	// without block A only the previous token's layer 2 is left, which says nothing
	cfg.Blocks = BlockB
	if got := FitLinear(fit, 8, SameToken, cfg, nil).Recall(test, []int{2})[0]; got.Covered*2 > got.Needed {
		t.Fatalf("block B alone covers %d of %d", got.Covered, got.Needed)
	}
	// the epoch callback sees every epoch, in order
	var epochs []int
	FitLinear(fit, 8, SameToken, cfg, func(e int, _ *Linear) { epochs = append(epochs, e) })
	if !slices.Equal(epochs, []int{1, 2, 3}) {
		t.Fatalf("epochs %v", epochs)
	}
}

func TestLinearNoLeak(t *testing.T) {
	// fitting on the tokens before 500 reads no token from 500 on
	r := rand.New(rand.NewPCG(33, 34))
	sel := randomSelections(r, 4, 700, 3, 12)
	cfg := LinearConfig{Blocks: BlockA | BlockB, L2: 1e-4, Rate: 0.2, Epochs: 2, Seed: 7}
	for _, h := range []Horizon{NextToken, SameToken} {
		before := FitLinear([]LinearTrace{{sel, tokensFrom(0, 500)}}, 12, h, cfg, nil)
		re := Selections{Layers: sel.Layers, Tokens: sel.Tokens, K: sel.K, IDs: slices.Clone(sel.IDs)}
		other := randomSelections(r, sel.Layers, sel.Tokens, sel.K, 12)
		for l := range sel.Layers {
			copy(re.IDs[(l*sel.Tokens+500)*sel.K:(l+1)*sel.Tokens*sel.K], other.IDs[(l*sel.Tokens+500)*sel.K:(l+1)*sel.Tokens*sel.K])
		}
		after := FitLinear([]LinearTrace{{re, tokensFrom(0, 500)}}, 12, h, cfg, nil)
		if !reflect.DeepEqual(before, after) {
			t.Fatalf("%v: the weights depend on tokens after the fitted ones", h)
		}
		// and it is deterministic
		if again := FitLinear([]LinearTrace{{sel, tokensFrom(0, 500)}}, 12, h, cfg, nil); !reflect.DeepEqual(before, again) {
			t.Fatalf("%v: two fits differ", h)
		}
	}
}

func TestLinearCausal(t *testing.T) {
	r := rand.New(rand.NewPCG(35, 36))
	sel, spans, cut := onlineSplit(r)
	for _, h := range []Horizon{NextToken, SameToken} {
		m := FitLinear([]LinearTrace{{sel, tokensFrom(0, spans[cut].First)}}, 12, h,
			LinearConfig{Blocks: BlockA | BlockB, Rate: 0.2, Epochs: 2, Seed: 3}, nil)
		checkCausal(t, r, sel, spans, h, func(sel Selections) *Ranking {
			return RankWith(m.Online(), sel, spans, 12, h, 6)
		})
	}
}

func TestLinearOnline(t *testing.T) {
	// the frozen model as a policy prefetches what its ranking ranks, at the targets of its
	// horizon, and only for the horizon it was fitted for
	r := rand.New(rand.NewPCG(37, 38))
	sel, spans, cut := onlineSplit(r)
	for _, h := range []Horizon{NextToken, SameToken} {
		m := FitLinear([]LinearTrace{{sel, tokensFrom(0, spans[cut].First)}}, 12, h,
			LinearConfig{Blocks: BlockA | BlockB, Rate: 0.2, Epochs: 1, Seed: 5}, nil)
		ranking := RankWith(m.Online(), sel, spans, 12, h, 12)
		online := &prefetchLog{Policy: NewPrefetcher(&NoCache{}, m.Online(), 12, h, 4)}
		ranked := &prefetchLog{Policy: NewRanked(&NoCache{}, ranking, 4)}
		for _, q := range []*prefetchLog{online, ranked} {
			if _, err := Replay(sel, spans, 12, sizes(sel.Layers, 1), q); err != nil {
				t.Fatal(err)
			}
		}
		predicted := 0
		for s := range online.sets {
			if !slices.Equal(online.sets[s], ranked.sets[s]) {
				t.Fatalf("%v, step %d: online %v, ranked %v", h, s, online.sets[s], ranked.sets[s])
			}
			l, ids, ok := keysAt(online.sets[s], 12)
			want := (s%sel.Layers + sel.Layers - 1) % sel.Layers
			if h == SameToken {
				want = s%sel.Layers + 1
			}
			if !ok || len(ids) > 0 && l != want {
				t.Fatalf("%v, step %d: prefetches %v", h, s, online.sets[s])
			}
			if len(ids) > 0 {
				predicted++
			}
		}
		if predicted == 0 {
			t.Fatalf("%v: nothing predicted", h)
		}
		other := NextToken
		if h == NextToken {
			other = SameToken
		}
		if _, ids := m.Online().Predict(other, StepInfo{1, Span{Decode, 200, 201}, 1}, 4, nil); ids != nil {
			t.Fatalf("a %v model predicts %v", h, other)
		}
	}
}

func TestTemperature(t *testing.T) {
	// labels drawn with probability sigmoid(z/2): the fitted temperature is near 2
	r := rand.New(rand.NewPCG(39, 40))
	var z []float32
	var y []bool
	for range 200000 {
		v := float32(r.NormFloat64() * 4)
		z = append(z, v)
		y = append(y, r.Float64() < 1/(1+math.Exp(-float64(v)/2)))
	}
	if got := fitTemperature(z, y); math.Abs(got-2) > 0.1 {
		t.Fatalf("temperature %g, want 2", got)
	}
}

func TestCalibration(t *testing.T) {
	// two targets of one expert each, ranked by hand: target 0's best is right, target 1's second
	c := calibrate([][]float64{{0.9, 0.3}, {0.6, 0.2}}, [][]bool{{true, false}, {false, true}}, []int{1, 2}, 2)
	// expected covered: 0.9+0.6 at k 1, plus 0.3+0.2 at k 2; covered 1 and 2
	if math.Abs(c.Expected[0]-1.5) > 1e-12 || math.Abs(c.Expected[1]-2) > 1e-12 || !slices.Equal(c.Covered, []int{1, 2}) {
		t.Fatalf("calibration %+v", c)
	}
	// ten bins: 0.9 (hit) and 0.6 (miss) alone in theirs, 0.3 (miss) and 0.2 (hit) in theirs:
	// ECE = (0.1 + 0.6 + 0.3 + 0.8) / 4
	if math.Abs(c.ECE-0.45) > 1e-12 || c.Candidates != 4 {
		t.Fatalf("ECE %g over %d, want 0.45 over 4", c.ECE, c.Candidates)
	}
}

func BenchmarkLinearStep(b *testing.B) {
	// the real models' shape: 50 layers of 384 experts, 6 a token, 48 candidates
	const layers, nExpert, k, tokens = 50, 384, 6, 256
	r := rand.New(rand.NewPCG(41, 42))
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
			m := newLinear(layers, nExpert, h, BlockA|BlockB)
			for i := range m.w {
				m.w[i] = float32(r.NormFloat64())
			}
			p := m.Online()
			out := make([]int, 0, 48)
			step := 0
			for range 2 * layers {
				p.Observe(info(step), sets[step%len(sets)])
				step++
			}
			b.ResetTimer()
			for range b.N {
				_, out = p.Predict(h, info(step), 48, out[:0])
				p.Observe(info(step), sets[step%len(sets)])
				step++
			}
		})
	}
}
