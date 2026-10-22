package locality

// Upstream eviction policies replayed offline (PLAN Phase 11, Stage E1): the segmented LRU of
// llama.cpp's --moe-cache-mib (PR #29887) and the decayed hotness of --moe-stream (PR #25294),
// and a step-pinned LFU.
//
// A step's experts are one event (Zhang, arXiv 2608.07911: per-access replays inflate LRU misses):
// a server runs a ubatch's or a token's experts of a layer together and never evicts one it is
// running. So none of these policies evicts a key the running step has used already. When a
// step needs more experts than the cache holds, the excess is used once and not kept (Load
// returns the key itself) instead. Replay counts the evictions of keys the running step has used
// as Counts.StepEvictions; for these policies they are zero by construction.

import (
	"container/heap"
	"fmt"
	"math"
)

// SLRU is a segmented LRU: a probation and a protected segment, each in LRU order. A loaded
// expert enters probation; a hit in probation promotes it to protected, a hit in protected
// moves it to protected's front. Protected holds at most a fraction of the budget: what
// overflows it is demoted, protected's least recent first, to probation's front. Victims come
// from probation's least recent end, from protected's once probation holds nothing evictable,
// the experts the running step has used skipped. With fraction 0 nothing stays protected, and it
// is an LRU that does not evict the running step's experts.
//
// The shared SLRU holds budget bytes for all layers, as llama.cpp shares one per device; the
// per-layer one capacity experts in every layer, as LayerLRU.
type SLRU struct {
	nExpert  int
	bytes    []int64
	perLayer bool
	budget   []int64 // per group: one for all layers, or one per layer
	protCap  []int64 // per group, the most bytes protected may hold
	cur      []int64 // per group, bytes resident
	prot     []int64 // per group, bytes protected
	l        *lists  // group g's probation is list 2g, its protected list 2g+1
	pinned   []int   // per list, its keys the running step has used
	used     []int   // per key, the last step that used it, plus one
	step     int     // the running step, plus one
	loaded   int     // the key the last Load fetched, whose Use is part of that miss
	victims  []int
	evict    []int
}

// NewSLRU returns an empty SLRU of budget bytes shared by all layers of nExpert experts of
// bytes[l] each, protected holding at most fraction of it; fraction must be in [0, 1].
func NewSLRU(nExpert int, bytes []int64, budget int64, fraction float64) (*SLRU, error) {
	return newSLRU(nExpert, bytes, []int64{budget}, false, fraction)
}

// NewLayerSLRU returns an empty SLRU of capacity experts in every layer, protected holding at
// most fraction of each layer's.
func NewLayerSLRU(nExpert int, bytes []int64, capacity int, fraction float64) (*SLRU, error) {
	budget := make([]int64, len(bytes))
	for l, b := range bytes {
		budget[l] = int64(capacity) * b
	}
	return newSLRU(nExpert, bytes, budget, true, fraction)
}

func newSLRU(nExpert int, bytes []int64, budget []int64, perLayer bool, fraction float64) (*SLRU, error) {
	if !(fraction >= 0 && fraction <= 1) {
		return nil, fmt.Errorf("SLRU: protected fraction %v, want one in [0, 1]", fraction)
	}
	nKey := len(bytes) * nExpert
	c := &SLRU{nExpert: nExpert, bytes: bytes, perLayer: perLayer, budget: budget,
		protCap: make([]int64, len(budget)), cur: make([]int64, len(budget)), prot: make([]int64, len(budget)),
		l: newLists(nKey, 2*len(budget)), pinned: make([]int, 2*len(budget)), used: make([]int, nKey), loaded: -1}
	for g, b := range budget {
		c.protCap[g] = int64(fraction * float64(b))
	}
	return c, nil
}

func (c *SLRU) size(k int) int64 { return c.bytes[k/c.nExpert] }

func (c *SLRU) group(k int) int {
	if c.perLayer {
		return k / c.nExpert
	}
	return 0
}

// put makes k list i's most recent key.
func (c *SLRU) put(i, k int) {
	c.l.pushFront(i, k)
	if c.used[k] == c.step {
		c.pinned[i]++
	}
}

// take removes k from its list.
func (c *SLRU) take(k int) {
	if c.used[k] == c.step {
		c.pinned[c.l.in[k]]--
	}
	c.l.remove(k)
}

// Prefetch notes the running step; it prefetches nothing.
func (c *SLRU) Prefetch(s StepInfo) []int {
	if s.Index+1 != c.step {
		c.step = s.Index + 1
		clear(c.pinned)
	}
	return nil
}

func (c *SLRU) Load(k int) []int {
	c.evict = c.evict[:0]
	c.loaded = k
	g, sz := c.group(k), c.size(k)
	need := c.cur[g] + sz - c.budget[g]
	if sz > c.budget[g] {
		return append(c.evict, k)
	}
	// the victims, probation's least recent first, then protected's, the running step's skipped
	c.victims = c.victims[:0]
	var freed int64
	for _, i := range []int{2 * g, 2*g + 1} {
		if freed >= need {
			break
		}
		if c.pinned[i] == c.l.size[i] {
			continue
		}
		h := c.l.nKey + i
		for v := c.l.prev[h]; v != h && freed < need; v = c.l.prev[v] {
			if c.used[v] != c.step {
				c.victims = append(c.victims, v)
				freed += c.size(v)
			}
		}
	}
	if freed < need { // the running step holds the cache: use k once
		return append(c.evict, k)
	}
	for _, v := range c.victims {
		if c.l.in[v] == 2*g+1 {
			c.prot[g] -= c.size(v)
		}
		c.take(v)
		c.cur[g] -= c.size(v)
		c.evict = append(c.evict, v)
	}
	c.put(2*g, k)
	c.cur[g] += sz
	return c.evict
}

// Use promotes a hit in probation and refreshes one in protected; the Use of the expert just
// loaded belongs to its miss.
func (c *SLRU) Use(_ StepInfo, k int) {
	in := c.l.in[k]
	if in < 0 { // used once, not kept
		c.used[k] = c.step
		c.loaded = -1
		return
	}
	c.take(k)
	c.used[k] = c.step
	if k == c.loaded {
		c.loaded = -1
		c.put(in, k)
		return
	}
	g := c.group(k)
	if in == 2*g {
		c.prot[g] += c.size(k)
	}
	c.put(2*g+1, k)
	for c.prot[g] > c.protCap[g] {
		v := c.l.back(2*g + 1)
		c.take(v)
		c.prot[g] -= c.size(v)
		c.put(2*g, v)
	}
}

// Hotness keeps capacity experts in every layer, as llama.cpp's --moe-stream keeps a slot pool
// per layer, and evicts the coldest: every expert has a score that decays by half every halfLife
// tokens and gains 1 for every step that uses it, resident or not, and a load evicts the resident
// expert of the layer with the lowest score, the least recently used of equal ones. The decay is
// applied lazily, as of the running span's first token.
type Hotness struct {
	nExpert    int
	capacities []int
	halfLife   float64
	// lv is every key's score as log2(score) + (tokens at its last use) / halfLife: the decay is
	// common to all keys, so lv orders them as their scores at any later token do. -Inf for none.
	lv       []float64
	last     []int
	clock    int
	t        float64 // the running span's first token / halfLife
	size     []int
	resident []bool
	h        []keyHeap // per layer, its resident keys the running step has not used, coldest first
	pins     []int     // resident keys the running step has used
	step     int
	evict    []int
}

// NewHotness returns an empty Hotness of capacity experts in each of layers of nExpert experts,
// scores halved every halfLife tokens; halfLife must be positive.
func NewHotness(layers, nExpert, capacity int, halfLife float64) (*Hotness, error) {
	if !(halfLife > 0) || math.IsInf(halfLife, 1) {
		return nil, fmt.Errorf("hotness: half-life %v, want a positive one", halfLife)
	}
	n := layers * nExpert
	c := &Hotness{nExpert: nExpert, capacities: make([]int, layers), halfLife: halfLife, lv: make([]float64, n),
		last: make([]int, n), size: make([]int, layers), resident: make([]bool, n), h: make([]keyHeap, layers)}
	for l := range c.capacities {
		c.capacities[l] = capacity
	}
	for k := range c.lv {
		c.lv[k] = math.Inf(-1)
	}
	pos := make([]int, n)
	for k := range pos {
		pos[k] = -1
	}
	less := func(a, b int) bool {
		if c.lv[a] != c.lv[b] {
			return c.lv[a] < c.lv[b]
		}
		return c.last[a] < c.last[b]
	}
	for l := range c.h {
		c.h[l] = keyHeap{pos: pos, less: less}
	}
	return c, nil
}

// Prefetch unpins the last step's experts and advances the decay; it prefetches nothing.
func (c *Hotness) Prefetch(s StepInfo) []int {
	if s.Index+1 != c.step {
		c.step = s.Index + 1
		for _, k := range c.pins {
			heap.Push(&c.h[k/c.nExpert], k)
		}
		c.pins = c.pins[:0]
	}
	c.t = float64(s.First) / c.halfLife
	return nil
}

func (c *Hotness) Load(k int) []int {
	c.evict = c.evict[:0]
	l := k / c.nExpert
	if c.size[l] == c.capacities[l] {
		if c.h[l].Len() == 0 { // nothing evictable: the running step holds the layer
			return append(c.evict, k)
		}
		v := heap.Pop(&c.h[l]).(int)
		c.resident[v] = false
		c.size[l]--
		c.evict = append(c.evict, v)
	}
	c.resident[k] = true
	c.size[l]++
	heap.Push(&c.h[l], k)
	return c.evict
}

// Use adds 1 to k's decayed score and pins k for the rest of the step.
func (c *Hotness) Use(_ StepInfo, k int) {
	if math.IsInf(c.lv[k], -1) {
		c.lv[k] = c.t
	} else {
		c.lv[k] = c.t + math.Log2(1+math.Exp2(c.lv[k]-c.t))
	}
	c.last[k] = c.clock
	c.clock++
	if p := c.h[k/c.nExpert].pos[k]; p >= 0 {
		heap.Remove(&c.h[k/c.nExpert], p)
		c.pins = append(c.pins, k)
	}
}

// Score is k's score as of token t: its uses, each halved every halfLife tokens since.
func (c *Hotness) Score(k, t int) float64 {
	return math.Exp2(c.lv[k] - float64(t)/c.halfLife)
}
