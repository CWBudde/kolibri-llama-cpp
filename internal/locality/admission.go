package locality

// Admission replayed offline (PLAN Phase 11, Stage E4): what a layer's expert slots hold when a
// prompt's prefill ends, how well that serves the decode that follows, and whether seeding the
// slots differently at that point would serve it better.
//
// The slots are modelled after the fork's llama_expert_slots (src/llama-expert-slots.cpp on
// feat/kolibri1-waves2), one set per layer, each slot with its expert and the tick of its last use:
//
//   - remap: a decoded token, or a ubatch that cannot touch more experts than the layer has slots.
//     The step's distinct experts are pinned; those resident are touched, the others loaded, each
//     into an empty slot or else over the least recently used unpinned one. All of a step's
//     experts get the same tick, and among equally old slots the lowest goes first (pick_victim).
//   - waves (--expert-cache-waves): a ubatch of n tokens with min(n_expert, n*k) > slots, whatever
//     its union turns out to be (llama_expert_cache::n_waves). plan_waves sorts the union stably
//     by use count, least used first, and cuts it into waves; stage_wave runs them in order. A
//     stage pins its wave and what of the next wave is resident, touches and loads its wave,
//     evicting experts of later waves last, and loads the next wave ahead into what is left.
//
// Loads are synchronous here. In the fork they are asynchronous, but a stage and a remap wait
// for the experts they need, a slot assigned during a stage is loading and no victim until the
// stage ends, and what a stage loads ahead is the next stage's own wave: without a prefetcher the
// slots after a step are the same. The Markov prefetcher (--expert-prefetch) is not modelled.
//
// At the end of a prefill phase, the last prefill span before a run of decoded tokens, a variant
// may replace what the waves left (AdmissionVariant). The decode then runs through the plain
// remap, and its hits are scored in windows of each answer's first decoded tokens.

import (
	"cmp"
	"fmt"
	"slices"
)

// waveCap is llama_expert_wave_cap: the experts per wave, 0 when the slots are too few for waves.
func waveCap(slots, k int) int {
	if k > 0 && slots >= 3*k {
		return (slots - k) / 2
	}
	return 0
}

// waveCount is llama_expert_wave_count: the waves a graph is built with for a ubatch that can
// touch nTouch experts.
func waveCount(nTouch, slots, k int) int {
	if slots > k {
		return (nTouch + (slots - k) - 1) / (slots - k)
	}
	return 0
}

// waveLayer is one layer's slots.
type waveLayer struct {
	expert  []int   // per slot, -1 when empty
	lastUse []int64 // per slot, the tick of its last use
	where   []int   // per expert, its slot or -1
	wave    []int   // per expert, its wave in the layer's last plan or -1
	plan    [][]int
}

// waveSlots are the slots of every layer, as llama_expert_slots keeps them.
type waveSlots struct {
	slots, nExpert int
	waves          bool
	tick           int64
	layers         []waveLayer
	keep           []bool // per expert: pinned by the running step
	loading        []bool // per slot: assigned by the running step
	preloaded      []bool // per expert: loaded ahead by the previous stage
	uses           []int
	counts         PrefillCounts // of every remap and ubatch, as llama_expert_cache_counts counts them
}

// PrefillCounts are the counters the fork keeps for a phase (llama_expert_cache_counts), summed
// over the layers: Steps are remaps and wave plans, Needed their distinct experts, Hits those
// resident, Misses the loads, Evictions the loads over a resident expert, WavesBuilt the wave
// stages run and Waves those of them that had a wave.
type PrefillCounts struct {
	Steps      int `json:"steps"`
	Needed     int `json:"needed"`
	Hits       int `json:"hits"`
	Misses     int `json:"misses"`
	Evictions  int `json:"evictions"`
	Waves      int `json:"waves"`
	WavesBuilt int `json:"waves_built"`
}

func newWaveSlots(layers, nExpert, slots int, waves bool) *waveSlots {
	c := &waveSlots{slots: slots, nExpert: nExpert, waves: waves, layers: make([]waveLayer, layers),
		keep: make([]bool, nExpert), loading: make([]bool, slots), preloaded: make([]bool, nExpert),
		uses: make([]int, nExpert)}
	for l := range c.layers {
		L := &c.layers[l]
		L.expert, L.lastUse = make([]int, slots), make([]int64, slots)
		L.where, L.wave = make([]int, nExpert), make([]int, nExpert)
		for s := range L.expert {
			L.expert[s] = -1
		}
		for e := range L.where {
			L.where[e], L.wave[e] = -1, -1
		}
	}
	return c
}

// victim is pick_victim: an empty slot, else the least recently used slot that the running step
// neither pins nor loads, the lowest of equals; while wave wNow stages (wNow >= 0), slots of
// later waves' experts go last. It returns -1 if there is none.
func (c *waveSlots) victim(l, wNow int) int {
	L := &c.layers[l]
	best, bestLater := -1, false
	for s, e := range L.expert {
		if e < 0 {
			return s
		}
		if c.loading[s] || c.keep[e] {
			continue
		}
		later := wNow >= 0 && L.wave[e] > wNow
		if best < 0 || (!later && bestLater) || (later == bestLater && L.lastUse[s] < L.lastUse[best]) {
			best, bestLater = s, later
		}
	}
	return best
}

// assign loads expert e into slot s of layer l, over what the slot held.
func (c *waveSlots) assign(l, s, e int) {
	L := &c.layers[l]
	if old := L.expert[s]; old >= 0 {
		L.where[old] = -1
	}
	L.expert[s], L.lastUse[s], L.where[e] = e, c.tick, s
	c.loading[s] = true
}

// load is assign for a step's miss, counted.
func (c *waveSlots) load(l, s, e int) {
	c.counts.Misses++
	if c.layers[l].expert[s] >= 0 {
		c.counts.Evictions++
	}
	c.assign(l, s, e)
}

func (c *waveSlots) check(ids []int32) error {
	for _, e := range ids {
		if e < 0 || int(e) >= c.nExpert {
			return fmt.Errorf("expert %d of %d", e, c.nExpert)
		}
	}
	return nil
}

// remap is llama_expert_slots::remap: one step through the plain LRU. It returns the step's
// distinct experts and how many of them were resident.
func (c *waveSlots) remap(l int, ids []int32) (needed, hits int, err error) {
	if err := c.check(ids); err != nil {
		return 0, 0, err
	}
	L := &c.layers[l]
	var uniq []int
	for _, e := range ids {
		if !c.keep[e] {
			c.keep[e] = true
			uniq = append(uniq, int(e))
		}
	}
	defer func() {
		for _, e := range uniq {
			c.keep[e] = false
		}
		clear(c.loading)
	}()
	if len(uniq) > c.slots {
		return 0, 0, fmt.Errorf("layer %d needs %d distinct experts in one step, more than its %d slots", l,
			len(uniq), c.slots)
	}
	c.tick++
	for _, e := range uniq {
		if s := L.where[e]; s >= 0 {
			hits++
			L.lastUse[s] = c.tick
		}
	}
	for _, e := range uniq {
		if L.where[e] < 0 {
			c.load(l, c.victim(l, -1), e) // a victim exists: fewer experts are pinned than slots
		}
	}
	c.counts.Steps++
	c.counts.Needed += len(uniq)
	c.counts.Hits += hits
	return len(uniq), hits, nil
}

// waved reports whether a ubatch of nTokens tokens of k experts runs in waves
// (llama_expert_cache::n_waves > 1).
func (c *waveSlots) waved(nTokens, k int) bool {
	return c.waves && waveCap(c.slots, k) > 0 && min(c.nExpert, nTokens*k) > c.slots
}

// ubatch runs one step of layer l: ids are its tokens' experts, k per token. It goes through the
// waves when the ubatch could exceed the slots, else through remap, and returns the step's
// distinct experts and how many of them needed no load.
func (c *waveSlots) ubatch(l int, ids []int32, k int) (needed, hits int, err error) {
	if !c.waved(len(ids)/k, k) {
		return c.remap(l, ids)
	}
	if err := c.check(ids); err != nil {
		return 0, 0, err
	}
	L := &c.layers[l]

	// plan_waves: the union in first-use order, stably sorted by use count, in waves of the cap,
	// larger only if the union would otherwise take more waves than the graph was built with
	nBuilt := waveCount(min(c.nExpert, len(ids)), c.slots, k)
	clear(c.uses)
	var uniq []int
	for _, e := range ids {
		if c.uses[e] == 0 {
			uniq = append(uniq, int(e))
		}
		c.uses[e]++
	}
	size := max(waveCap(c.slots, k), (len(uniq)+nBuilt-1)/nBuilt)
	if size+k > c.slots {
		return 0, 0, fmt.Errorf("layer %d: waves of %d experts and %d per token do not fit %d slots", l, size, k, c.slots)
	}
	for _, w := range L.plan {
		for _, e := range w {
			L.wave[e] = -1
		}
	}
	L.plan = L.plan[:0]
	slices.SortStableFunc(uniq, func(a, b int) int { return cmp.Compare(c.uses[a], c.uses[b]) })
	for i, e := range uniq {
		if i%size == 0 {
			L.plan = append(L.plan, nil)
		}
		L.plan[len(L.plan)-1] = append(L.plan[len(L.plan)-1], e)
		L.wave[e] = len(L.plan) - 1
	}

	// stage_wave, once per wave the graph holds; a stage beyond the plan has no wave and changes
	// no slot
	var pinned []int
	pin := func(e int) {
		if !c.keep[e] {
			c.keep[e] = true
			pinned = append(pinned, e)
		}
	}
	c.counts.Steps++
	c.counts.Needed += len(uniq)
	for w := range nBuilt {
		c.tick++
		c.counts.WavesBuilt++
		if w >= len(L.plan) {
			continue
		}
		c.counts.Waves++
		cur := L.plan[w]
		var nxt []int
		if w+1 < len(L.plan) {
			nxt = L.plan[w+1]
		}
		pinned = pinned[:0]
		for _, e := range cur {
			pin(e)
		}
		// a wave of fewer than k experts parks on ready slots of others, the most recent first
		if len(cur) < k {
			var ready []int
			for s, e := range L.expert {
				if e >= 0 && !c.keep[e] {
					ready = append(ready, s)
				}
			}
			slices.SortStableFunc(ready, func(a, b int) int { return cmp.Compare(L.lastUse[b], L.lastUse[a]) })
			if len(ready) < k-len(cur) {
				return 0, 0, fmt.Errorf("layer %d wave %d: %d ready slots to park %d experts on", l, w, len(ready),
					k-len(cur))
			}
			for _, s := range ready[:k-len(cur)] {
				pin(L.expert[s])
			}
		}
		// what of the next wave is resident stays, as far as this wave still fits beside it
		room := c.slots - len(pinned)
		for _, e := range nxt {
			if room > 0 && L.where[e] >= 0 && !c.keep[e] {
				pin(e)
				room--
			}
		}
		for _, e := range cur {
			s := L.where[e]
			if c.preloaded[e] {
				c.preloaded[e] = false // counted as a load by the stage before
			} else if s >= 0 {
				hits++
				c.counts.Hits++
			}
			if s >= 0 {
				L.lastUse[s] = c.tick
			}
		}
		for _, e := range cur {
			if L.where[e] >= 0 {
				continue
			}
			s := c.victim(l, w)
			if s < 0 {
				return 0, 0, fmt.Errorf("layer %d wave %d: no slot to load expert %d into", l, w, e)
			}
			c.load(l, s, e)
		}
		// the next wave loads behind this one, into slots this stage does not pin and not over
		// its own experts; what finds no slot loads at its own stage
		var held []int
		for _, e := range nxt {
			if L.where[e] >= 0 && !c.keep[e] {
				c.keep[e] = true
				held = append(held, e)
			}
		}
		for _, e := range nxt {
			if L.where[e] >= 0 {
				continue
			}
			s := c.victim(l, w)
			if s < 0 {
				break
			}
			c.load(l, s, e)
			c.preloaded[e] = true
		}
		for _, e := range held {
			c.keep[e] = false
		}
		for _, e := range pinned {
			c.keep[e] = false
		}
		clear(c.loading)
	}
	return len(uniq), hits, nil
}

// lru lists layer l's residents, least recently used first, the lower slot first among equals:
// the order in which remap would evict them.
func (c *waveSlots) lru(l int) []int {
	L := &c.layers[l]
	var slots []int
	for s, e := range L.expert {
		if e >= 0 {
			slots = append(slots, s)
		}
	}
	slices.SortStableFunc(slots, func(a, b int) int { return cmp.Compare(L.lastUse[a], L.lastUse[b]) })
	for i, s := range slots {
		slots[i] = L.expert[s]
	}
	return slots
}

// seed makes seeds resident in layer l and the most recently used of its slots. seeds are
// distinct, at most the slots, and most recent first; ranks ascend with them, and seeds of one
// rank are equally recent. A seed that is resident keeps its slot; one that is not is loaded into
// an empty slot or over the least recently used resident that is no seed. Residents that are not
// evicted stay, older than every seed. It returns the seeds it had to load.
func (c *waveSlots) seed(l int, seeds, ranks []int) (loads int) {
	if len(seeds) == 0 {
		return 0
	}
	L := &c.layers[l]
	for _, e := range seeds {
		c.keep[e] = true
	}
	for _, e := range seeds {
		if L.where[e] < 0 {
			c.assign(l, c.victim(l, -1), e)
			loads++
		}
	}
	last := ranks[len(ranks)-1]
	for i, e := range seeds {
		L.lastUse[L.where[e]] = c.tick + int64(last-ranks[i]) + 1
		c.keep[e] = false
	}
	c.tick += int64(last) + 1
	clear(c.loading)
	return loads
}

// restore puts layer l's slots back to a snapshot (expert and lastUse per slot) and returns how
// many of its experts were not resident.
func (c *waveSlots) restore(l int, expert []int, lastUse []int64) (loads int) {
	L := &c.layers[l]
	for _, e := range expert {
		if e >= 0 && L.where[e] < 0 {
			loads++
		}
	}
	for _, e := range L.expert {
		if e >= 0 {
			L.where[e] = -1
		}
	}
	copy(L.expert, expert)
	copy(L.lastUse, lastUse)
	for s, e := range expert {
		if e >= 0 {
			L.where[e] = s
		}
	}
	return loads
}

// AdmissionVariant is what the slots hold when a prompt's prefill ends and its decode begins.
type AdmissionVariant int

const (
	// AdmitToday leaves the slots as the prefill's waves and remaps left them.
	AdmitToday AdmissionVariant = iota
	// AdmitLFU seeds the experts the prompt's prefill used most, counted per token over the whole
	// prefill phase, ties to the lower expert ID; the most used is the most recent.
	AdmitLFU
	// AdmitRecency seeds the experts of the prompt's last tokens, walking back from the last one
	// until the slots are full; the last token's are the most recent.
	AdmitRecency
	// AdmitDecodeSet restores what the slots held before the prefill phase began: the prefill
	// bypasses the slots.
	AdmitDecodeSet
	// AdmitOracle seeds the experts the decode run that follows uses most. It reads the future, so
	// it is no policy. It is not an upper bound either: with two slots and the decode 0,0,0,1,2,2
	// it seeds 0 and 2 for four hits, while 1 and 0 give five.
	AdmitOracle
)

// AdmissionVariants are all variants, in report order.
var AdmissionVariants = []AdmissionVariant{AdmitToday, AdmitLFU, AdmitRecency, AdmitDecodeSet, AdmitOracle}

func (v AdmissionVariant) String() string {
	switch v {
	case AdmitToday:
		return "today"
	case AdmitLFU:
		return "lfu"
	case AdmitRecency:
		return "recency"
	case AdmitDecodeSet:
		return "decode-set"
	case AdmitOracle:
		return "future-lfu"
	}
	return fmt.Sprintf("AdmissionVariant(%d)", int(v))
}

// AdmissionConfig is the cache an admission replay runs.
type AdmissionConfig struct {
	Slots   int // per layer
	NExpert int
	// Waves is --expert-cache-waves. Without it a ubatch whose union exceeds the slots fails.
	Waves bool
	// LastLayerOutputs models that the last layer runs a ubatch's output rows only
	// (models/kolibri1.cpp): of a prefill phase it sees the last token before the decode, alone,
	// and nothing else.
	LastLayerOutputs bool
	// First are the windows scored, ascending: an answer's first First[i] decoded tokens.
	First []int
}

// AdmissionWindow are the decoded tokens of a window, the experts they needed, summed over the
// layers, and how many of those were resident.
type AdmissionWindow struct {
	Tokens int `json:"tokens"`
	Needed int `json:"needed"`
	Hits   int `json:"hits"`
}

// HitRate is the share of needed experts found resident.
func (w AdmissionWindow) HitRate() float64 { return float64(w.Hits) / float64(max(w.Needed, 1)) }

// MissesPerToken is the experts loaded per decoded token, summed over the layers.
func (w AdmissionWindow) MissesPerToken() float64 {
	return float64(w.Needed-w.Hits) / float64(max(w.Tokens, 1))
}

// AdmissionResult are a replay's scores over its test suffix, the spans from the cut on.
type AdmissionResult struct {
	// First[i] is every answer's first AdmissionConfig.First[i] decoded tokens, All every decoded
	// token and Steady those after the largest window.
	First  []AdmissionWindow `json:"first"`
	All    AdmissionWindow   `json:"all"`
	Steady AdmissionWindow   `json:"steady"`
	// Prompts are the prefill phases that end before a decode.
	Prompts int `json:"prompts"`
	// Seeded are the experts the variant put into the slots at the ends of those phases, summed
	// over the layers, and SeedLoads those of them that were not resident after the prefill: what
	// the variant loads on top of the prefill.
	Seeded    int `json:"seeded"`
	SeedLoads int `json:"seed_loads"`
	// PrefillNeeded are the distinct experts of the prefill steps, PrefillLoads those loaded.
	PrefillNeeded int `json:"prefill_needed"`
	PrefillLoads  int `json:"prefill_loads"`
	// WaveSteps are the prefill steps, one per layer of a ubatch, that ran in waves.
	WaveSteps int `json:"wave_steps"`
}

// ReplayAdmission replays spans through per-layer slots: prefill ubatches through the waves model,
// decoded tokens through the plain remap. At the end of every prefill phase that a decode follows
// it applies v. The whole trace is replayed, so that the test suffix starts warm, and the spans
// from index cut on are scored.
func ReplayAdmission(sel Selections, spans []Span, cut int, cfg AdmissionConfig, v AdmissionVariant) (AdmissionResult, error) {
	r, _, err := replayAdmission(sel, spans, len(spans), cut, cfg, v)
	return r, err
}

// replayAdmission replays the first n spans and also returns the slots they leave.
func replayAdmission(sel Selections, spans []Span, n, cut int, cfg AdmissionConfig, v AdmissionVariant) (AdmissionResult,
	*waveSlots, error) {
	var r AdmissionResult
	if sel.K < 1 || cfg.Slots < sel.K || cfg.NExpert < 1 {
		return r, nil, fmt.Errorf("admission: %d slots for %d experts per token of %d", cfg.Slots, sel.K, cfg.NExpert)
	}
	if len(cfg.First) == 0 || cfg.First[0] < 1 || !slices.IsSorted(cfg.First) {
		return r, nil, fmt.Errorf("admission: windows %v, want ascending positive token counts", cfg.First)
	}
	next := 0
	for i, s := range spans {
		if s.First != next || s.End <= s.First || s.End > sel.Tokens {
			return r, nil, fmt.Errorf("admission: span %d is [%d, %d), want one from %d within %d tokens", i, s.First,
				s.End, next, sel.Tokens)
		}
		next = s.End
	}
	ids := func(l, first, end int) []int32 { return sel.IDs[(l*sel.Tokens+first)*sel.K : (l*sel.Tokens+end)*sel.K] }
	c := newWaveSlots(sel.Layers, cfg.NExpert, cfg.Slots, cfg.Waves)
	r.First = make([]AdmissionWindow, len(cfg.First))
	steadyFrom := cfg.First[len(cfg.First)-1]

	type snapshot struct {
		expert  []int
		lastUse []int64
	}
	before := make([]snapshot, sel.Layers) // the slots when the running prefill phase began
	phaseFirst := -1                       // the first token of the running prefill phase, or -1
	decoded := 0                           // tokens decoded since the last prefill phase
	count := make([]int, cfg.NExpert)
	taken := make([]bool, cfg.NExpert)
	var seeds, ranks []int

	// mostUsed sets seeds to the experts the tokens [first, end) use most at layer l
	mostUsed := func(l, first, end int) {
		clear(count)
		seeds, ranks = seeds[:0], ranks[:0]
		for _, e := range ids(l, first, end) {
			if count[e] == 0 {
				seeds = append(seeds, int(e))
			}
			count[e]++
		}
		slices.SortFunc(seeds, func(a, b int) int { return cmp.Or(cmp.Compare(count[b], count[a]), cmp.Compare(a, b)) })
		seeds = seeds[:min(len(seeds), cfg.Slots)]
		for i := range seeds {
			ranks = append(ranks, i)
		}
	}

	for i, s := range spans[:n] {
		scored := i >= cut
		if s.Phase == Decode {
			phaseFirst = -1
			var w AdmissionWindow
			for l := range sel.Layers {
				needed, hits, err := c.remap(l, ids(l, s.First, s.End))
				if err != nil {
					return r, nil, fmt.Errorf("admission: span %d: %w", i, err)
				}
				w.Needed += needed
				w.Hits += hits
			}
			w.Tokens = s.End - s.First
			if scored {
				add := func(to *AdmissionWindow) {
					to.Tokens += w.Tokens
					to.Needed += w.Needed
					to.Hits += w.Hits
				}
				add(&r.All)
				for j, first := range cfg.First {
					if decoded < first {
						add(&r.First[j])
					}
				}
				if decoded >= steadyFrom {
					add(&r.Steady)
				}
			}
			decoded += s.End - s.First
			continue
		}

		if phaseFirst < 0 {
			phaseFirst = s.First
			for l := range before {
				before[l].expert = append(before[l].expert[:0], c.layers[l].expert...)
				before[l].lastUse = append(before[l].lastUse[:0], c.layers[l].lastUse...)
			}
		}
		// the phase ends here if a decoded token follows
		ends := i+1 < len(spans) && spans[i+1].Phase == Decode
		for l := range sel.Layers {
			first := s.First
			if cfg.LastLayerOutputs && l == sel.Layers-1 {
				if !ends {
					continue
				}
				first = s.End - 1
			}
			step := ids(l, first, s.End)
			waved := c.waved(s.End-first, sel.K)
			needed, hits, err := c.ubatch(l, step, sel.K)
			if err != nil {
				return r, nil, fmt.Errorf("admission: span %d: %w", i, err)
			}
			if scored {
				r.PrefillNeeded += needed
				r.PrefillLoads += needed - hits
				if waved {
					r.WaveSteps++
				}
			}
		}
		if !ends {
			continue
		}
		decoded = 0
		if scored {
			r.Prompts++
		}
		if v == AdmitToday {
			continue
		}
		runEnd := s.End // the end of the decode run that follows
		for _, d := range spans[i+1:] {
			if d.Phase != Decode {
				break
			}
			runEnd = d.End
		}
		for l := range sel.Layers {
			first := phaseFirst // the phase's tokens this layer saw
			if cfg.LastLayerOutputs && l == sel.Layers-1 {
				first = s.End - 1
			}
			loads, seeded := 0, 0
			switch v {
			case AdmitLFU:
				mostUsed(l, first, s.End)
			case AdmitOracle:
				mostUsed(l, s.End, runEnd)
			case AdmitRecency:
				clear(taken)
				seeds, ranks = seeds[:0], ranks[:0]
				for t := s.End - 1; t >= first && len(seeds) < cfg.Slots; t-- {
					for _, e := range ids(l, t, t+1) {
						if !taken[e] && len(seeds) < cfg.Slots {
							taken[e] = true
							seeds = append(seeds, int(e))
							ranks = append(ranks, s.End-1-t)
						}
					}
				}
			case AdmitDecodeSet:
				loads = c.restore(l, before[l].expert, before[l].lastUse)
				for _, e := range before[l].expert {
					if e >= 0 {
						seeded++
					}
				}
			default:
				return r, nil, fmt.Errorf("admission: unknown variant %d", int(v))
			}
			if v != AdmitDecodeSet {
				loads, seeded = c.seed(l, seeds, ranks), len(seeds)
			}
			if scored {
				r.Seeded += seeded
				r.SeedLoads += loads
			}
		}
	}
	return r, c, nil
}

// ServerSpans are the ubatches in which llama-server prefills the prompt tokens [first, end) it has
// not processed yet (tools/server/server-context.cpp, update_slots): batches of up to nBatch
// tokens, which also end 4+ubatch and 4 tokens before the prompt does, where the server makes its
// context checkpoints, each batch cut into ubatches of ubatch tokens. The last 4 tokens of a
// prompt are therefore a ubatch of their own. It does not know the message boundaries, at which
// the server may end a batch as well.
func ServerSpans(first, end, nBatch, ubatch int) []Span {
	var out []Span
	for first < end {
		stop := min(first+nBatch, end)
		for _, off := range []int{min(nBatch, 4+ubatch), min(nBatch, 4)} {
			if at := end - off; at > first && at < stop {
				stop = at
			}
		}
		for t := first; t < stop; t += ubatch {
			out = append(out, Span{Prefill, t, min(t+ubatch, stop)})
		}
		first = stop
	}
	return out
}

// ReplayPrefill replays one prompt from empty slots, its prefill spans and then the decoded tokens
// that follow, if any, and returns what the fork's prefill and decode counters would read: the
// check of the waves model against a run of the server. The last prefill span holds the prompt's
// output; with cfg.LastLayerOutputs the last layer runs that span only.
//
// A fresh llama-server decodes two tokens twice before its first prompt, the warmup (BOS, EOS) and
// the probe of common_context_can_seq_rm (token 0 twice), and the cache counts them as prefill. A
// trace does not hold their selections. startup gives them, or stands in for them: its tokens are
// replayed two at a time, each pair a decode of its own with an output, run and counted before
// the prompt. It may be empty.
func ReplayPrefill(sel Selections, spans []Span, startup Selections, cfg AdmissionConfig) (prefill, decode PrefillCounts,
	err error) {
	if sel.K < 1 || cfg.Slots < sel.K || cfg.NExpert < 1 {
		return prefill, decode, fmt.Errorf("prefill: %d slots for %d experts per token of %d", cfg.Slots, sel.K, cfg.NExpert)
	}
	c := newWaveSlots(sel.Layers, cfg.NExpert, cfg.Slots, cfg.Waves)
	last := len(spans) - 1 // the last prefill span
	for last >= 0 && spans[last].Phase == Decode {
		last--
	}
	if startup.Tokens%2 != 0 || (startup.Tokens > 0 && (startup.Layers != sel.Layers || startup.K != sel.K)) {
		return prefill, decode, fmt.Errorf("prefill: startup selections of %d layers, %d tokens and %d experts per token, "+
			"want pairs of tokens of %d layers and %d experts", startup.Layers, startup.Tokens, startup.K, sel.Layers, sel.K)
	}
	for t := 0; t < startup.Tokens; t += 2 {
		for l := range sel.Layers {
			first := t
			if cfg.LastLayerOutputs && l == sel.Layers-1 {
				first = t + 1
			}
			ids := startup.IDs[(l*startup.Tokens+first)*sel.K : (l*startup.Tokens+t+2)*sel.K]
			if _, _, err := c.ubatch(l, ids, sel.K); err != nil {
				return prefill, decode, fmt.Errorf("prefill: startup: %w", err)
			}
		}
	}
	next := 0
	for i, s := range spans {
		if (s.Phase == Decode) != (i > last) || s.First != next || s.End <= s.First || s.End > sel.Tokens ||
			(s.Phase == Decode && s.End != s.First+1) {
			return prefill, decode, fmt.Errorf("prefill: span %d is %v [%d, %d), want prefill spans, then decoded "+
				"tokens, from %d within %d tokens", i, s.Phase, s.First, s.End, next, sel.Tokens)
		}
		next = s.End
		for l := range sel.Layers {
			first := s.First
			if cfg.LastLayerOutputs && l == sel.Layers-1 && s.Phase == Prefill {
				if i < last {
					continue
				}
				first = s.End - 1
			}
			if _, _, err := c.ubatch(l, sel.IDs[(l*sel.Tokens+first)*sel.K:(l*sel.Tokens+s.End)*sel.K], sel.K); err != nil {
				return prefill, decode, fmt.Errorf("prefill: span %d: %w", i, err)
			}
		}
		if i == last {
			prefill = c.counts
		}
	}
	decode = PrefillCounts{Steps: c.counts.Steps - prefill.Steps, Needed: c.counts.Needed - prefill.Needed,
		Hits: c.counts.Hits - prefill.Hits, Misses: c.counts.Misses - prefill.Misses,
		Evictions: c.counts.Evictions - prefill.Evictions}
	return prefill, decode, nil
}

// ServerPrefill cuts every prefill phase of spans, its consecutive prefill spans, into the ubatches
// of ServerSpans, and leaves the decoded tokens as they are.
func ServerPrefill(spans []Span, nBatch, ubatch int) []Span {
	var out []Span
	for i := 0; i < len(spans); {
		if spans[i].Phase != Prefill {
			out = append(out, spans[i])
			i++
			continue
		}
		j := i
		for j+1 < len(spans) && spans[j+1].Phase == Prefill {
			j++
		}
		out = append(out, ServerSpans(spans[i].First, spans[j].End, nBatch, ubatch)...)
		i = j + 1
	}
	return out
}
