package locality

import (
	"math/rand"
	"slices"
	"testing"
)

// admissionSel builds one-layer selections of k experts per token, repeated in every layer.
func admissionSel(layers, k int, tokens [][]int32) Selections {
	sel := Selections{Layers: layers, Tokens: len(tokens), K: k}
	for range layers {
		for _, t := range tokens {
			if len(t) != k {
				panic("admissionSel: a token without k experts")
			}
			sel.IDs = append(sel.IDs, t...)
		}
	}
	return sel
}

func decodeSpans(first, end int) []Span {
	var out []Span
	for t := first; t < end; t++ {
		out = append(out, Span{Decode, t, t + 1})
	}
	return out
}

// TestWaveSlotsHandComputed replays a waves ubatch after three decoded tokens and compares the
// slots with what llama_expert_slots::plan_waves and stage_wave leave, computed by hand.
func TestWaveSlotsHandComputed(t *testing.T) {
	cases := []struct {
		name         string
		warm, ubatch [][]int32
		experts      []int   // per slot
		lastUse      []int64 // per slot
		needed, hits int
		lru          []int // least recent first
	}{
		{
			// uses 0:3 1:2 2:1 3:1 4:1, two waves built of size 3: [2 3 4] [1 0]. Stage 0 keeps
			// expert 0 (resident, next wave), loads 2 3 4 over 5 6 7 and expert 1 ahead over 8.
			name:    "next wave's resident stays and its missing loads ahead",
			warm:    [][]int32{{5, 6}, {7, 8}, {9, 0}},
			ubatch:  [][]int32{{0, 1}, {0, 2}, {0, 3}, {1, 4}},
			experts: []int{2, 3, 4, 1, 9, 0},
			lastUse: []int64{4, 4, 4, 5, 3, 5},
			needed:  5, hits: 1,
			lru: []int{9, 2, 3, 4, 1, 0},
		},
		{
			// three waves [4 5] [2 3] [0 1]: experts 0 and 1 are the least recently used when
			// stage 0 picks its victims and are not pinned by it, but a later wave needs them.
			name:    "experts of later waves are evicted last",
			warm:    [][]int32{{0, 1}, {6, 7}, {8, 9}},
			ubatch:  [][]int32{{0, 1}, {0, 1}, {0, 1}, {2, 3}, {2, 3}, {4, 5}},
			experts: []int{0, 1, 4, 5, 2, 3},
			lastUse: []int64{6, 6, 4, 4, 5, 5},
			needed:  6, hits: 2,
			lru: []int{4, 5, 2, 3, 0, 1},
		},
		{
			// waves [2 1] [0]: the last wave has fewer than k experts and parks on a ready slot,
			// which changes no slot.
			name:    "a wave of fewer than k experts",
			warm:    [][]int32{{5, 6}, {7, 8}, {9, 10}},
			ubatch:  [][]int32{{0, 1}, {0, 1}, {0, 1}, {0, 2}},
			experts: []int{2, 1, 0, 8, 9, 10},
			lastUse: []int64{4, 4, 5, 2, 3, 3},
			needed:  3, hits: 0,
			lru: []int{8, 9, 10, 2, 1, 0},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := newWaveSlots(1, 12, 6, true)
			for _, tok := range c.warm {
				if _, _, err := s.ubatch(0, tok, 2); err != nil {
					t.Fatal(err)
				}
			}
			var ids []int32
			for _, tok := range c.ubatch {
				ids = append(ids, tok...)
			}
			needed, hits, err := s.ubatch(0, ids, 2)
			if err != nil {
				t.Fatal(err)
			}
			L := &s.layers[0]
			if !slices.Equal(L.expert, c.experts) || !slices.Equal(L.lastUse, c.lastUse) {
				t.Errorf("slots %v last used %v, want %v %v", L.expert, L.lastUse, c.experts, c.lastUse)
			}
			if needed != c.needed || hits != c.hits {
				t.Errorf("needed %d hits %d, want %d %d", needed, hits, c.needed, c.hits)
			}
			if got := s.lru(0); !slices.Equal(got, c.lru) {
				t.Errorf("LRU order %v, want %v", got, c.lru)
			}
		})
	}
}

// TestWaveSlotsDecodeAfterWaves: a decoded token after the first case evicts the least recently
// used slots: expert 10 goes over 9, expert 11 over 2, the lowest of the equally old slots.
func TestWaveSlotsDecodeAfterWaves(t *testing.T) {
	s := newWaveSlots(1, 12, 6, true)
	for _, ids := range [][]int32{{5, 6}, {7, 8}, {9, 0}, {0, 1, 0, 2, 0, 3, 1, 4}, {10, 11}} {
		if _, _, err := s.ubatch(0, ids, 2); err != nil {
			t.Fatal(err)
		}
	}
	if want := []int{11, 3, 4, 1, 10, 0}; !slices.Equal(s.layers[0].expert, want) {
		t.Errorf("slots %v, want %v", s.layers[0].expert, want)
	}
}

func TestWaveCounts(t *testing.T) {
	for _, c := range []struct{ slots, k, cap int }{{128, 6, 61}, {18, 6, 6}, {17, 6, 0}, {6, 2, 2}, {64, 6, 29}} {
		if got := waveCap(c.slots, c.k); got != c.cap {
			t.Errorf("waveCap(%d, %d) = %d, want %d", c.slots, c.k, got, c.cap)
		}
	}
	for _, c := range []struct{ touch, slots, k, n int }{{384, 128, 6, 4}, {132, 128, 6, 2}, {8, 6, 2, 2}, {12, 6, 2, 3}} {
		if got := waveCount(c.touch, c.slots, c.k); got != c.n {
			t.Errorf("waveCount(%d, %d, %d) = %d, want %d", c.touch, c.slots, c.k, got, c.n)
		}
	}
}

// TestWaveSlotsUnionTooLarge: without waves a ubatch whose union exceeds the slots fails, as the
// runtime aborts.
func TestWaveSlotsUnionTooLarge(t *testing.T) {
	s := newWaveSlots(1, 12, 6, false)
	if _, _, err := s.ubatch(0, []int32{0, 1, 2, 3, 4, 5, 6, 7}, 2); err == nil {
		t.Error("a union of 8 experts in 6 slots without waves: no error")
	}
}

// TestDecodeStepPinned: the experts of a decoded token never evict each other, at any slot count
// down to k, and all of them are resident after the step.
func TestDecodeStepPinned(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const nExpert, k = 16, 3
	for _, slots := range []int{3, 4, 7} {
		s := newWaveSlots(1, nExpert, slots, true)
		for step := range 500 {
			ids := make([]int32, 0, k)
			for _, e := range rng.Perm(nExpert)[:k] {
				ids = append(ids, int32(e))
			}
			needed, hits, err := s.ubatch(0, ids, k)
			if err != nil {
				t.Fatal(err)
			}
			if needed != k || hits > k {
				t.Fatalf("slots %d step %d: needed %d hits %d", slots, step, needed, hits)
			}
			for _, e := range ids {
				if s.layers[0].where[e] < 0 {
					t.Fatalf("slots %d step %d: expert %d of the step is not resident after it", slots, step, e)
				}
			}
			n := 0
			for _, e := range s.layers[0].expert {
				if e >= 0 {
					n++
				}
			}
			if n > slots {
				t.Fatalf("slots %d: %d residents", slots, n)
			}
		}
	}
}

// randomConversation is a trace of prompts and answers: prefill phases of the given lengths, each
// followed by decoded tokens.
func randomConversation(rng *rand.Rand, layers, nExpert, k int, prompts, decodes []int, ubatch int) (Selections, []Span) {
	var tokens [][]int32
	var spans []Span
	for i, p := range prompts {
		for first := len(tokens); first < len(tokens)+p; first += ubatch {
			spans = append(spans, Span{Prefill, first, min(first+ubatch, len(tokens)+p)})
		}
		spans = append(spans, decodeSpans(len(tokens)+p, len(tokens)+p+decodes[i])...)
		for range p + decodes[i] {
			tok := make([]int32, 0, k)
			for _, e := range rng.Perm(nExpert)[:k] {
				tok = append(tok, int32(e))
			}
			tokens = append(tokens, tok)
		}
	}
	sel := Selections{Layers: layers, Tokens: len(tokens), K: k}
	for range layers {
		for _, i := range rng.Perm(len(tokens)) { // a different order per layer
			sel.IDs = append(sel.IDs, tokens[i]...)
		}
	}
	return sel, spans
}

// TestTodayIsPlainLRU: when no ubatch can exceed the slots, the waves model is the plain remap,
// and with one expert per token, where a step has no ties, that is LayerLRU.
func TestTodayIsPlainLRU(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	t.Run("waves on and off agree", func(t *testing.T) {
		// ubatches of 4 tokens of 3 experts touch at most 12 experts: never more than 12 slots
		sel, spans := randomConversation(rng, 3, 40, 3, []int{9, 6, 11}, []int{20, 30, 25}, 4)
		var got [2]AdmissionResult
		for i, waves := range []bool{true, false} {
			cfg := AdmissionConfig{Slots: 12, NExpert: 40, Waves: waves, First: []int{4, 8}}
			r, err := ReplayAdmission(sel, spans, 0, cfg, AdmitToday)
			if err != nil {
				t.Fatal(err)
			}
			if waves && r.WaveSteps != 0 {
				t.Errorf("%d steps ran in waves, want none", r.WaveSteps)
			}
			r.WaveSteps = 0
			got[i] = r
		}
		if !admissionEqual(got[0], got[1]) {
			t.Errorf("waves on %+v, off %+v", got[0], got[1])
		}
	})
	t.Run("k 1 equals LayerLRU", func(t *testing.T) {
		const layers, nExpert, slots = 2, 30, 8
		sel, spans := randomConversation(rng, layers, nExpert, 1, []int{1, 1}, []int{200, 150}, 1)
		for i := range spans { // one token per span: every span as a decoded token
			spans[i].Phase = Decode
		}
		cfg := AdmissionConfig{Slots: slots, NExpert: nExpert, Waves: true, First: []int{16}}
		r, err := ReplayAdmission(sel, spans, 0, cfg, AdmitToday)
		if err != nil {
			t.Fatal(err)
		}
		ref, err := Replay(sel, spans, nExpert, make([]int64, layers), NewLayerLRU(layers, nExpert, slots))
		if err != nil {
			t.Fatal(err)
		}
		if r.All.Hits != ref.Decode.Hits || r.All.Needed != ref.Decode.Accesses {
			t.Errorf("hits %d of %d, LayerLRU %d of %d", r.All.Hits, r.All.Needed, ref.Decode.Hits, ref.Decode.Accesses)
		}
	})
}

func admissionEqual(a, b AdmissionResult) bool {
	return slices.Equal(a.First, b.First) && a.All == b.All && a.Steady == b.Steady && a.Prompts == b.Prompts &&
		a.SeedLoads == b.SeedLoads && a.Seeded == b.Seeded && a.PrefillNeeded == b.PrefillNeeded &&
		a.PrefillLoads == b.PrefillLoads && a.WaveSteps == b.WaveSteps
}

// seedTrace is one layer, 12 experts, 6 slots, k 2: three decoded tokens, a prompt of four tokens
// in one waves ubatch (the first case of TestWaveSlotsHandComputed), then four decoded tokens.
func seedTrace(layers int) (Selections, []Span) {
	tokens := [][]int32{{5, 6}, {7, 8}, {9, 0}, // decoded: slots [5 6 7 8 9 0]
		{0, 1}, {0, 2}, {0, 3}, {1, 4}, // prompt: today leaves [2 3 4 1 9 0]
		{10, 11}, {10, 5}, {10, 11}, {9, 5}} // decoded
	spans := append(decodeSpans(0, 3), Span{Prefill, 3, 7})
	return admissionSel(layers, 2, tokens), append(spans, decodeSpans(7, 11)...)
}

// TestAdmissionSeeding checks every variant's slots and LRU order right after the prompt of
// seedTrace, and the experts it had to load for them.
func TestAdmissionSeeding(t *testing.T) {
	sel, spans := seedTrace(1)
	cases := []struct {
		v     AdmissionVariant
		lru   []int // least recent first
		loads int
	}{
		// as the waves leave them
		{AdmitToday, []int{9, 2, 3, 4, 1, 0}, 0},
		// counts 0:3 1:2 2:1 3:1 4:1, five distinct experts: all seeded, and the sixth slot keeps
		// what it held; ties by lower expert ID, the most used last
		{AdmitLFU, []int{9, 4, 3, 2, 1, 0}, 0},
		// backwards from the last token: {1 4} {0 3} {0 2}, five distinct experts; a token's experts
		// are equally recent and go by slot
		{AdmitRecency, []int{9, 2, 3, 0, 4, 1}, 0},
		// what the slots held before the prompt, in that order; 5 6 7 8 are loaded again
		{AdmitDecodeSet, []int{5, 6, 7, 8, 9, 0}, 4},
		// the decode run uses 10:3 5:2 11:2 9:1; the two slots left keep the most recent residents
		{AdmitOracle, []int{1, 0, 9, 11, 5, 10}, 3},
	}
	for _, c := range cases {
		t.Run(c.v.String(), func(t *testing.T) {
			cfg := AdmissionConfig{Slots: 6, NExpert: 12, Waves: true, First: []int{2}}
			// the replay up to the prompt's last span, where the variant seeds
			r, s, err := replayAdmission(sel, spans, 4, 0, cfg, c.v)
			if err != nil {
				t.Fatal(err)
			}
			if got := s.lru(0); !slices.Equal(got, c.lru) {
				t.Errorf("LRU order %v, want %v", got, c.lru)
			}
			if r.SeedLoads != c.loads || r.Prompts != 1 {
				t.Errorf("%d seed loads over %d prompts, want %d over 1", r.SeedLoads, r.Prompts, c.loads)
			}
		})
	}
}

// TestAdmissionRecencyFull: recency stops in the middle of a token once the slots are full, and
// LFU keeps only the most used.
func TestAdmissionRecencyFull(t *testing.T) {
	// prompt of 4 tokens of 2 experts in 6 slots: 8 distinct experts
	tokens := [][]int32{{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 1}}
	sel := admissionSel(1, 2, tokens)
	spans := []Span{{Prefill, 0, 4}, {Decode, 4, 5}}
	cfg := AdmissionConfig{Slots: 6, NExpert: 12, Waves: true, First: []int{1}}
	_, s, err := replayAdmission(sel, spans, 1, 0, cfg, AdmitRecency)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := s.lru(0), []int{2, 3, 4, 5, 6, 7}; !slices.Equal(got, want) {
		t.Errorf("recency LRU order %v, want %v", got, want)
	}
	// every expert is used once: LFU keeps the six lowest IDs, the lowest most recent
	_, s, err = replayAdmission(sel, spans, 1, 0, cfg, AdmitLFU)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := s.lru(0), []int{5, 4, 3, 2, 1, 0}; !slices.Equal(got, want) {
		t.Errorf("LFU LRU order %v, want %v", got, want)
	}
}

// TestAdmissionDecode replays seedTrace's decode after each variant by hand.
func TestAdmissionDecode(t *testing.T) {
	sel, spans := seedTrace(1)
	// decoded after the prompt: {10 11} {10 5} {10 11} {9 5}, 8 experts needed
	cases := []struct {
		v            AdmissionVariant
		first2, hits int // hits in the first two tokens, and in all four
	}{
		// today [2 3 4 1 9 0], LRU 9 2 3 4 1 0: {10 11} misses twice over 9 and 2, {10 5} hits 10,
		// 5 over 3; {10 11} hits twice; {9 5}: 9 was evicted, 5 hits
		{AdmitToday, 1, 4},
		// decode-set [5 6 7 8 9 0]: {10 11} over 5 and 6, {10 5} hits 10, 5 over 7; {10 11} hits
		// twice; {9 5} hits twice
		{AdmitDecodeSet, 1, 5},
		// oracle holds 10 5 11 9: every expert hits
		{AdmitOracle, 4, 8},
	}
	for _, c := range cases {
		t.Run(c.v.String(), func(t *testing.T) {
			cfg := AdmissionConfig{Slots: 6, NExpert: 12, Waves: true, First: []int{2}}
			// scored from the prompt on: the three decoded tokens before it are the prefix
			r, err := ReplayAdmission(sel, spans, 3, cfg, c.v)
			if err != nil {
				t.Fatal(err)
			}
			want := AdmissionWindow{Tokens: 2, Needed: 4, Hits: c.first2}
			if r.First[0] != want {
				t.Errorf("first 2: %+v, want %+v", r.First[0], want)
			}
			if want := (AdmissionWindow{Tokens: 4, Needed: 8, Hits: c.hits}); r.All != want {
				t.Errorf("all: %+v, want %+v", r.All, want)
			}
			if want := (AdmissionWindow{Tokens: 2, Needed: 4, Hits: c.hits - c.first2}); r.Steady != want {
				t.Errorf("steady: %+v, want %+v", r.Steady, want)
			}
		})
	}
}

// TestAdmissionWindows: the first-N windows count each prompt's first N decoded tokens, the
// steady window the tokens after the largest N, and only the test suffix is scored.
func TestAdmissionWindows(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	const layers, k = 3, 2
	sel, spans := randomConversation(rng, layers, 30, k, []int{5, 7, 3, 6}, []int{10, 100, 3, 70}, 4)
	cut := slices.IndexFunc(spans, func(s Span) bool { return s.First == 15 }) // the second prompt
	if cut < 0 || spans[cut].Phase != Prefill {
		t.Fatalf("no prefill span at token 15: %v", spans)
	}
	cfg := AdmissionConfig{Slots: 8, NExpert: 30, Waves: true, First: []int{16, 64}}
	for _, v := range AdmissionVariants {
		r, err := ReplayAdmission(sel, spans, cut, cfg, v)
		if err != nil {
			t.Fatal(err)
		}
		// test answers decode 100, 3 and 70 tokens
		tokens := []int{16 + 3 + 16, 64 + 3 + 64, 100 + 3 + 70, 36 + 6}
		got := []AdmissionWindow{r.First[0], r.First[1], r.All, r.Steady}
		for i, w := range got {
			if w.Tokens != tokens[i] || w.Needed != tokens[i]*layers*k || w.Hits < 0 || w.Hits > w.Needed {
				t.Errorf("%v window %d: %+v, want %d tokens", v, i, w, tokens[i])
			}
		}
		if r.First[1].Hits+r.Steady.Hits != r.All.Hits || r.First[0].Hits > r.First[1].Hits {
			t.Errorf("%v: hits %d in the first 16, %d in the first 64, %d after, %d in all", v, r.First[0].Hits,
				r.First[1].Hits, r.Steady.Hits, r.All.Hits)
		}
		if r.Prompts != 3 {
			t.Errorf("%v: %d test prompts, want 3", v, r.Prompts)
		}
		if v == AdmitToday && (r.SeedLoads != 0 || r.Seeded != 0) {
			t.Errorf("today seeds %d experts with %d loads", r.Seeded, r.SeedLoads)
		}
	}
}

// TestAdmissionOracleBound: on a trace whose decode keeps to few experts, the oracle's first
// tokens miss nothing, and no variant beats it there.
func TestAdmissionOracleBound(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	const k, nExpert, slots = 2, 40, 8
	var tokens [][]int32
	var spans []Span
	for range 3 {
		first := len(tokens)
		for range 12 { // a prompt over all experts
			p := rng.Perm(nExpert)
			tokens = append(tokens, []int32{int32(p[0]), int32(p[1])})
		}
		spans = append(spans, Span{Prefill, first, first + 12})
		for range 20 { // an answer over experts 0..7
			p := rng.Perm(slots)
			tokens = append(tokens, []int32{int32(p[0]), int32(p[1])})
		}
		spans = append(spans, decodeSpans(first+12, first+32)...)
	}
	sel := admissionSel(2, k, tokens)
	cfg := AdmissionConfig{Slots: slots, NExpert: nExpert, Waves: true, First: []int{4}}
	oracle, err := ReplayAdmission(sel, spans, 0, cfg, AdmitOracle)
	if err != nil {
		t.Fatal(err)
	}
	if oracle.All.Hits != oracle.All.Needed {
		t.Errorf("oracle hits %d of %d", oracle.All.Hits, oracle.All.Needed)
	}
	for _, v := range AdmissionVariants {
		r, err := ReplayAdmission(sel, spans, 0, cfg, v)
		if err != nil {
			t.Fatal(err)
		}
		if r.First[0].Hits > oracle.First[0].Hits {
			t.Errorf("%v hits %d in the first tokens, the oracle %d", v, r.First[0].Hits, oracle.First[0].Hits)
		}
	}
}

// TestAdmissionLastLayerOutputs: with LastLayerOutputs the last layer of a prompt sees only the
// prompt's last token, so its slots keep what the decode before left.
func TestAdmissionLastLayerOutputs(t *testing.T) {
	sel, spans := seedTrace(2)
	cfg := AdmissionConfig{Slots: 6, NExpert: 12, Waves: true, LastLayerOutputs: true, First: []int{2}}
	_, s, err := replayAdmission(sel, spans, 4, 0, cfg, AdmitToday)
	if err != nil {
		t.Fatal(err)
	}
	if want := []int{2, 3, 4, 1, 9, 0}; !slices.Equal(s.layers[0].expert, want) {
		t.Errorf("layer 0 slots %v, want %v", s.layers[0].expert, want)
	}
	// the last layer runs token {1 4} alone, over the least recently used 5 and 6
	if want := []int{1, 4, 7, 8, 9, 0}; !slices.Equal(s.layers[1].expert, want) {
		t.Errorf("last layer slots %v, want %v", s.layers[1].expert, want)
	}
}

func TestAdmissionConfigErrors(t *testing.T) {
	sel, spans := seedTrace(1)
	for _, cfg := range []AdmissionConfig{
		{Slots: 1, NExpert: 12, First: []int{2}},             // fewer slots than k
		{Slots: 6, NExpert: 12, First: []int{4, 2}},          // not ascending
		{Slots: 6, NExpert: 12},                              // no window
		{Slots: 6, NExpert: 4, Waves: true, First: []int{2}}, // expert IDs out of range
	} {
		if _, err := ReplayAdmission(sel, spans, 0, cfg, AdmitToday); err == nil {
			t.Errorf("%+v: no error", cfg)
		}
	}
}

func TestServerSpans(t *testing.T) {
	sizes := func(spans []Span) []int {
		var out []int
		next := spans[0].First
		for _, s := range spans {
			if s.Phase != Prefill || s.First != next {
				t.Fatalf("spans %v", spans)
			}
			next = s.End
			out = append(out, s.End-s.First)
		}
		return out
	}
	for _, c := range []struct {
		first, end, batch, ubatch int
		want                      []int
	}{
		{0, 2048, 2048, 512, []int{512, 512, 508, 512, 4}}, // batches end at 1532 and 2044
		{0, 5000, 2048, 512, []int{512, 512, 512, 512, 512, 512, 512, 512, 388, 512, 4}},
		{100, 110, 2048, 512, []int{6, 4}},
		{0, 3, 2048, 512, []int{3}},
		{0, 520, 2048, 512, []int{4, 512, 4}},
		{0, 516, 2048, 512, []int{512, 4}},
		{0, 1024, 512, 512, []int{512, 508, 4}},
	} {
		if got := sizes(ServerSpans(c.first, c.end, c.batch, c.ubatch)); !slices.Equal(got, c.want) {
			t.Errorf("ServerSpans(%d, %d, %d, %d) = %v, want %v", c.first, c.end, c.batch, c.ubatch, got, c.want)
		}
	}
	spans := []Span{{Prefill, 0, 512}, {Prefill, 512, 600}, {Decode, 600, 601}, {Decode, 601, 602}, {Prefill, 602, 610}}
	want := []Span{{Prefill, 0, 84}, {Prefill, 84, 596}, {Prefill, 596, 600}, {Decode, 600, 601}, {Decode, 601, 602},
		{Prefill, 602, 606}, {Prefill, 606, 610}}
	if got := ServerPrefill(spans, 2048, 512); !slices.Equal(got, want) {
		t.Errorf("ServerPrefill = %v, want %v", got, want)
	}
}

// TestReplayPrefillCounts replays the first case of TestWaveSlotsHandComputed as a prompt: the
// counters are the fork's, by hand.
func TestReplayPrefillCounts(t *testing.T) {
	tokens := [][]int32{{0, 1}, {0, 2}, {0, 3}, {1, 4}, {10, 11}, {10, 5}}
	sel := admissionSel(2, 2, tokens)
	spans := []Span{{Prefill, 0, 4}, {Decode, 4, 5}, {Decode, 5, 6}}
	cfg := AdmissionConfig{Slots: 6, NExpert: 12, Waves: true, LastLayerOutputs: true, First: []int{1}}
	// layer 0: one plan of two waves [2 3 4] [1 0], five loads into empty slots; the last layer:
	// token {1 4} alone, a remap of two loads
	prefill, decode, err := ReplayPrefill(sel, spans, Selections{}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if want := (PrefillCounts{Steps: 2, Needed: 7, Hits: 0, Misses: 7, Evictions: 0, Waves: 2, WavesBuilt: 2}); prefill != want {
		t.Errorf("prefill %+v, want %+v", prefill, want)
	}
	// layer 0 holds [2 3 4 1 0 -]: {10 11} takes the empty slot and evicts 2, {10 5} hits 10 and
	// evicts 3; the last layer holds [1 4]: three loads into empty slots, one hit
	if want := (PrefillCounts{Steps: 4, Needed: 8, Hits: 2, Misses: 6, Evictions: 2}); decode != want {
		t.Errorf("decode %+v, want %+v", decode, want)
	}
	// the startup decode {0 1} {0 2} before it: 3 loads in layer 0, {0 2} alone in the last layer;
	// the prompt then finds 0 1 2 resident in layer 0 and nothing in the last
	startup := admissionSel(2, 2, tokens[:2])
	prefill, _, err = ReplayPrefill(sel, spans, startup, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if want := (PrefillCounts{Steps: 4, Needed: 12, Hits: 3, Misses: 9, Evictions: 0, Waves: 2, WavesBuilt: 2}); prefill != want {
		t.Errorf("prefill after a startup decode %+v, want %+v", prefill, want)
	}
	if _, _, err := ReplayPrefill(sel, []Span{{Decode, 0, 1}, {Prefill, 1, 4}}, Selections{}, cfg); err == nil {
		t.Error("a decoded token before the prompt: no error")
	}
}
