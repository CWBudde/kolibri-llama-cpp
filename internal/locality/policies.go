package locality

// Baseline cache policies (PLAN Phase 10, Stage A) for Replay, besides replay.go's LayerLRU. None
// prefetches: they decide only what a cache keeps of the experts the steps have needed so far.
// A budget is a number of expert bytes the cache may hold; an expert of layer l takes bytes[l].
//
// The Oracle is no policy a server could run. It is built from the whole trace and knows every
// future step, outside Replay's causality; it is the unattainable bound the others are measured
// against, and every report labels it so.

import (
	"container/heap"
	"fmt"
	"math"
)

// lists are intrusive doubly linked lists of keys in [0, nKey), a key in at most one of them. List
// i's head is node nKey+i; a list runs from its most recent key (the head's next) to its least
// recent (the head's prev).
type lists struct {
	nKey       int
	prev, next []int
	in         []int // the list a key is in, -1 for none
	size       []int
}

func newLists(nKey, n int) *lists {
	l := &lists{nKey: nKey, prev: make([]int, nKey+n), next: make([]int, nKey+n), in: make([]int, nKey),
		size: make([]int, n)}
	for i := range l.in {
		l.in[i] = -1
	}
	for i := range n {
		h := nKey + i
		l.prev[h], l.next[h] = h, h
	}
	return l
}

// pushFront makes k list i's most recent key; k must be in no list.
func (l *lists) pushFront(i, k int) {
	h := l.nKey + i
	l.prev[k], l.next[k] = h, l.next[h]
	l.prev[l.next[h]] = k
	l.next[h] = k
	l.in[k] = i
	l.size[i]++
}

func (l *lists) remove(k int) {
	l.next[l.prev[k]], l.prev[l.next[k]] = l.next[k], l.prev[k]
	l.size[l.in[k]]--
	l.in[k] = -1
}

// back is list i's least recent key; the list must not be empty.
func (l *lists) back(i int) int { return l.prev[l.nKey+i] }

// keyHeap is a heap of keys ordered by less, with each key's index in pos (-1 when absent). Several
// heaps of disjoint keys may share pos.
type keyHeap struct {
	keys []int
	pos  []int
	less func(a, b int) bool
}

func (h *keyHeap) Len() int           { return len(h.keys) }
func (h *keyHeap) Less(i, j int) bool { return h.less(h.keys[i], h.keys[j]) }
func (h *keyHeap) Swap(i, j int) {
	h.keys[i], h.keys[j] = h.keys[j], h.keys[i]
	h.pos[h.keys[i]], h.pos[h.keys[j]] = i, j
}
func (h *keyHeap) Push(x any) { k := x.(int); h.pos[k] = len(h.keys); h.keys = append(h.keys, k) }
func (h *keyHeap) Pop() any {
	k := h.keys[len(h.keys)-1]
	h.keys = h.keys[:len(h.keys)-1]
	h.pos[k] = -1
	return k
}

// NoCache keeps nothing: every access misses. It is the floor every cache is measured from.
type NoCache struct{ evict [1]int }

func (c *NoCache) Prefetch(StepInfo) []int { return nil }
func (c *NoCache) Load(k int) []int        { c.evict[0] = k; return c.evict[:] }
func (c *NoCache) Use(StepInfo, int)       {}

// SharedLRU is one LRU cache for the experts of all layers together, holding at most budget bytes:
// a load evicts the least recently used experts until the new one fits. An expert larger than the
// whole budget is used once and not kept. With equal expert sizes it is cmd/kolibri-cache's shared
// policy at budget / size experts.
type SharedLRU struct {
	nExpert     int
	bytes       []int64
	budget, cur int64
	l           *lists
	evict       []int
}

// NewSharedLRU returns an empty SharedLRU for layers of nExpert experts of bytes[l] each.
func NewSharedLRU(nExpert int, bytes []int64, budget int64) *SharedLRU {
	return &SharedLRU{nExpert: nExpert, bytes: bytes, budget: budget, l: newLists(len(bytes)*nExpert, 1)}
}

func (c *SharedLRU) size(k int) int64 { return c.bytes[k/c.nExpert] }

func (c *SharedLRU) Prefetch(StepInfo) []int { return nil }

func (c *SharedLRU) Load(k int) []int {
	c.evict = c.evict[:0]
	if c.size(k) > c.budget {
		return append(c.evict, k)
	}
	for c.cur+c.size(k) > c.budget {
		v := c.l.back(0)
		c.l.remove(v)
		c.cur -= c.size(v)
		c.evict = append(c.evict, v)
	}
	c.l.pushFront(0, k)
	c.cur += c.size(k)
	return c.evict
}

func (c *SharedLRU) Use(_ StepInfo, k int) {
	if c.l.in[k] == 0 {
		c.l.remove(k)
		c.l.pushFront(0, k)
	}
}

// LFU is a frequency cache for the experts of all layers together, holding at most budget bytes. It
// counts every expert's uses, one per step that needs it, resident or not, and halves all counts
// whenever the replay reaches another multiple of halfLife tokens, so that old use fades. A load
// evicts the resident experts used least often, the least recently used of equal counts, until the
// new one fits; the new expert is always kept unless it exceeds the whole budget.
type LFU struct {
	nExpert     int
	bytes       []int64
	budget, cur int64
	halfLife    int
	aged        int // the halvings so far
	count, last []int
	clock       int
	h           keyHeap
	evict       []int
	// pinned (NewPinnedLFU): the keys the running step has used are out of h, pins, and never
	// evicted; pinBytes are their bytes and step the running step, plus one.
	pinned   bool
	pins     []int
	pinBytes int64
	step     int
}

// NewPinnedLFU is NewLFU that never evicts an expert the running step has used (eviction.go): a
// load that does not fit beside them is used once and not kept.
func NewPinnedLFU(nExpert int, bytes []int64, budget int64, halfLife int) *LFU {
	c := NewLFU(nExpert, bytes, budget, halfLife)
	c.pinned = true
	return c
}

// NewLFU returns an empty LFU; halfLife is in tokens and must be positive.
func NewLFU(nExpert int, bytes []int64, budget int64, halfLife int) *LFU {
	n := len(bytes) * nExpert
	c := &LFU{nExpert: nExpert, bytes: bytes, budget: budget, halfLife: max(halfLife, 1), count: make([]int, n),
		last: make([]int, n)}
	c.h = keyHeap{pos: make([]int, n), less: func(a, b int) bool {
		if c.count[a] != c.count[b] {
			return c.count[a] < c.count[b]
		}
		return c.last[a] < c.last[b]
	}}
	for k := range c.h.pos {
		c.h.pos[k] = -1
	}
	return c
}

func (c *LFU) size(k int) int64 { return c.bytes[k/c.nExpert] }

// Prefetch ages the counts once per halfLife tokens reached; it prefetches nothing.
func (c *LFU) Prefetch(s StepInfo) []int {
	if c.pinned && s.Index+1 != c.step {
		c.step = s.Index + 1
		for _, k := range c.pins {
			heap.Push(&c.h, k)
		}
		c.pins, c.pinBytes = c.pins[:0], 0
	}
	if s.First/c.halfLife > c.aged {
		for ; c.aged < s.First/c.halfLife; c.aged++ {
			for k := range c.count {
				c.count[k] >>= 1
			}
		}
		heap.Init(&c.h)
	}
	return nil
}

func (c *LFU) Load(k int) []int {
	c.evict = c.evict[:0]
	if c.size(k) > c.budget || (c.pinned && c.pinBytes+c.size(k) > c.budget) {
		return append(c.evict, k)
	}
	for c.cur+c.size(k) > c.budget {
		v := heap.Pop(&c.h).(int)
		c.cur -= c.size(v)
		c.evict = append(c.evict, v)
	}
	heap.Push(&c.h, k)
	c.cur += c.size(k)
	return c.evict
}

func (c *LFU) Use(_ StepInfo, k int) {
	c.count[k]++
	c.last[k] = c.clock
	c.clock++
	if p := c.h.pos[k]; p >= 0 {
		if c.pinned {
			heap.Remove(&c.h, p)
			c.pins = append(c.pins, k)
			c.pinBytes += c.size(k)
		} else {
			heap.Fix(&c.h, p)
		}
	}
}

// ARC is the Adaptive Replacement Cache (Megiddo and Modha, FAST 2003) for the experts of all
// layers together: a recency list T1 and a frequency list T2 of resident experts, ghost lists B1 and
// B2 of experts recently evicted from them, and a target size p for T1 that a ghost hit moves
// towards the list it came from. It has no parameters. Its capacity is in experts, so all experts
// must have the same size.
type ARC struct {
	c      int
	p      float64
	l      *lists // t1, t2, b1, b2
	loaded int    // the key the last Load fetched, whose Use is part of that miss
	evict  []int
	// pinned (NewPinnedARC): the experts the running step has used are never evicted; used is
	// every key's last step that used it, step the running step, both plus one, and pins the
	// resident experts the running step has used.
	pinned bool
	used   []int
	step   int
	pins   int
}

// NewPinnedARC is NewARC that never evicts an expert the running step has used (eviction.go):
// REPLACE takes the least recent other expert of the list it chooses, of the other list if that
// has none, and a load that finds every resident expert used by the running step is used once
// and not kept, the lists and p unchanged.
func NewPinnedARC(nExpert int, bytes []int64, budget int64) (*ARC, error) {
	a, err := NewARC(nExpert, bytes, budget)
	if err != nil {
		return nil, err
	}
	a.pinned, a.used = true, make([]int, len(bytes)*nExpert)
	return a, nil
}

const (
	arcT1 = iota
	arcT2
	arcB1
	arcB2
)

// NewARC returns an empty ARC holding budget / size experts; it fails unless every layer's experts
// have the same size.
func NewARC(nExpert int, bytes []int64, budget int64) (*ARC, error) {
	for _, b := range bytes {
		if b != bytes[0] || b <= 0 {
			return nil, fmt.Errorf("ARC: expert sizes %d and %d differ; its capacity is in experts", bytes[0], b)
		}
	}
	var c int
	if len(bytes) > 0 {
		c = int(budget / bytes[0])
	}
	return &ARC{c: c, l: newLists(len(bytes)*nExpert, 4), loaded: -1}, nil
}

// Prefetch notes the running step; it prefetches nothing.
func (a *ARC) Prefetch(s StepInfo) []int {
	if a.pinned && s.Index+1 != a.step {
		a.step, a.pins = s.Index+1, 0
	}
	return nil
}

// victim is list i's least recent expert, the running step's skipped when pinned; -1 for none.
func (a *ARC) victim(i int) int {
	h := a.l.nKey + i
	for v := a.l.prev[h]; v != h; v = a.l.prev[v] {
		if !a.pinned || a.used[v] != a.step {
			return v
		}
	}
	return -1
}

// replace evicts T1's or T2's least recent expert into its ghost list, as the paper's REPLACE.
func (a *ARC) replace(inB2 bool) {
	t1, t2 := a.l.size[arcT1], a.l.size[arcT2]
	if t1+t2 < a.c {
		return
	}
	from, to := arcT2, arcB2
	if t1 >= 1 && ((inB2 && float64(t1) == a.p) || float64(t1) > a.p) || t2 == 0 {
		from, to = arcT1, arcB1
	}
	v := a.victim(from)
	if v < 0 { // pinned: every expert of the list is the running step's
		from, to = arcT1+arcT2-from, arcB1+arcB2-to
		v = a.victim(from)
	}
	a.l.remove(v)
	a.l.pushFront(to, v)
	a.evict = append(a.evict, v)
}

func (a *ARC) Load(k int) []int {
	a.evict = a.evict[:0]
	a.loaded = k
	if a.c == 0 {
		return append(a.evict, k)
	}
	s := a.l.size
	if a.pinned && s[arcT1]+s[arcT2] == a.c && a.pins == a.c {
		return append(a.evict, k)
	}
	switch a.l.in[k] {
	case arcB1:
		d := 1.0
		if s[arcB1] < s[arcB2] {
			d = float64(s[arcB2]) / float64(s[arcB1])
		}
		a.p = math.Min(a.p+d, float64(a.c))
		a.replace(false)
		a.l.remove(k)
		a.l.pushFront(arcT2, k)
	case arcB2:
		d := 1.0
		if s[arcB2] < s[arcB1] {
			d = float64(s[arcB1]) / float64(s[arcB2])
		}
		a.p = math.Max(a.p-d, 0)
		a.replace(true)
		a.l.remove(k)
		a.l.pushFront(arcT2, k)
	default:
		if l1 := s[arcT1] + s[arcB1]; l1 == a.c {
			if s[arcT1] < a.c {
				a.l.remove(a.l.back(arcB1))
				a.replace(false)
			} else {
				v := a.victim(arcT1)
				a.l.remove(v)
				a.evict = append(a.evict, v)
			}
		} else if total := l1 + s[arcT2] + s[arcB2]; total >= a.c {
			if total == 2*a.c {
				a.l.remove(a.l.back(arcB2))
			}
			a.replace(false)
		}
		a.l.pushFront(arcT1, k)
	}
	return a.evict
}

// Use moves a hit to T2's front; the Use of the expert just loaded belongs to its miss.
func (a *ARC) Use(_ StepInfo, k int) {
	in := a.l.in[k]
	if a.pinned && (in == arcT1 || in == arcT2) && a.used[k] != a.step {
		a.pins++
	}
	if a.pinned {
		a.used[k] = a.step
	}
	if k == a.loaded {
		a.loaded = -1
		return
	}
	if in == arcT1 || in == arcT2 {
		a.l.remove(k)
		a.l.pushFront(arcT2, k)
	}
}

// Oracle is Belady's MIN with bypass: built from the whole trace, it knows when every expert is
// next used and, to make room, drops the resident expert or the incoming one whose next use lies
// furthest ahead. With equal expert sizes this loads the fewest experts any cache of the budget
// can, prefetching or not (Belady 1966): no policy is cheaper, and none a server can run is as
// cheap. ORACLE: it uses future knowledge and is an unattainable upper bound, not a baseline to
// deploy. With unequal sizes dropping the furthest first is a heuristic, not a bound.
//
// That bound is on the bytes of the whole replay. A slice of it, a phase or a test suffix, is what
// one schedule loads there and bounds nothing: a schedule may load more elsewhere to load less there.
// NewHeldOutOracle bounds a test suffix's decode.
//
// The global Oracle holds budget bytes for all layers; the per-layer one capacity experts in every
// layer, as LayerLRU. It must be replayed with the selections and spans it was built from.
type Oracle struct {
	nExpert int
	bytes   []int64
	budget  []int64 // per group: one for all layers, or one per layer
	cur     []int64
	perL    bool
	uses    [][]int32 // every key's accesses in replay order, numbered
	ptr     []int     // a key's next access in uses
	access  int
	h       []keyHeap // per group, resident keys, furthest next use first
	evict   []int
	err     error
	// pinned (NewPinnedOracle): the keys the running step has used are out of h, pins, and never
	// evicted; pinBytes are their bytes per group and step the running step, plus one.
	pinned   bool
	pins     []int
	pinBytes []int64
	step     int
	// held out (NewHeldOutOracle): costs says which accesses cost anything, by number. next ranks a key
	// by its next access only if that costs, so a step's own experts need not rank first, and a
	// pinned one pins stepKeys, every step's keys, when the step starts.
	costs    []bool
	stepKeys [][]int32
}

// NewOracle returns an Oracle of budget bytes shared by all layers, for Replay(sel, spans, ...).
func NewOracle(sel Selections, spans []Span, nExpert int, bytes []int64, budget int64) *Oracle {
	return newOracle(sel, spans, nExpert, bytes, []int64{budget}, false)
}

// NewLayerOracle returns an Oracle of capacity experts in every layer.
func NewLayerOracle(sel Selections, spans []Span, nExpert int, bytes []int64, capacity int) *Oracle {
	return newOracle(sel, spans, nExpert, bytes, layerBudgets(bytes, capacity), true)
}

func layerBudgets(bytes []int64, capacity int) []int64 {
	budget := make([]int64, len(bytes))
	for l, b := range bytes {
		budget[l] = int64(capacity) * b
	}
	return budget
}

// NewPinnedOracle is NewOracle that holds a step's experts together, as a server does (eviction.go):
// it never evicts an expert the running step has used, and it keeps an expert it loads even when
// that expert's next use is the furthest, as long as the step's experts fit. A step whose union does
// not fit has no such schedule, and what does not fit is used once, as in the pinned LFU.
//
// So where every step fits this is a valid event-atomic schedule, and its bytes bound the
// event-atomic optimum from above, as NewOracle's bound it from below. With experts of equal size
// TestPinnedOracleOptimal holds it to that optimum; with a shared budget over layers of unequal
// sizes it is only a bound.
func NewPinnedOracle(sel Selections, spans []Span, nExpert int, bytes []int64, budget int64) *Oracle {
	return pinnedOracle(newOracle(sel, spans, nExpert, bytes, []int64{budget}, false))
}

// NewPinnedLayerOracle is NewLayerOracle with NewPinnedOracle's pins.
func NewPinnedLayerOracle(sel Selections, spans []Span, nExpert int, bytes []int64, capacity int) *Oracle {
	return pinnedOracle(newOracle(sel, spans, nExpert, bytes, layerBudgets(bytes, capacity), true))
}

func pinnedOracle(o *Oracle) *Oracle {
	o.pinned, o.pinBytes = true, make([]int64, len(o.budget))
	return o
}

// NewHeldOutOracle is the Oracle of one objective: the decode misses of spans[cut:]. Every other
// access costs nothing, so the training prefix and every prefill serve only to fill the cache. An
// expert whose next access is free can be dropped for nothing, since that access brings it back, so
// it ranks with those never used again; the others rank by next access, as in MIN. With equal expert
// sizes no demand cache of the budget has fewer decode misses in the suffix, however it spent its
// prefix and prefills. TestHeldOutOracleOptimal holds it to that by exhaustive search. A policy that
// prefetches is not bound by it, since a prefetch in a prefill costs decode nothing.
// ORACLE: an unattainable bound, looser than any schedule that also pays for what it loads elsewhere.
func NewHeldOutOracle(sel Selections, spans []Span, cut, nExpert int, bytes []int64, budget int64) *Oracle {
	o := newOracle(sel, spans, nExpert, bytes, []int64{budget}, false)
	return o.heldOut(sel, spans, cut)
}

// NewHeldOutLayerOracle is NewHeldOutOracle with capacity experts in every layer.
func NewHeldOutLayerOracle(sel Selections, spans []Span, cut, nExpert int, bytes []int64, capacity int) *Oracle {
	o := newOracle(sel, spans, nExpert, bytes, layerBudgets(bytes, capacity), true)
	return o.heldOut(sel, spans, cut)
}

// NewPinnedHeldOutOracle is NewHeldOutOracle that holds a step's experts together, as
// NewPinnedOracle does. Where every step fits it is a valid event-atomic schedule, so its decode
// misses in the suffix bound the event-atomic optimum from above, as NewHeldOutOracle's bound it from
// below.
func NewPinnedHeldOutOracle(sel Selections, spans []Span, cut, nExpert int, bytes []int64, budget int64) *Oracle {
	return pinnedOracle(NewHeldOutOracle(sel, spans, cut, nExpert, bytes, budget))
}

// NewPinnedHeldOutLayerOracle is NewHeldOutLayerOracle with NewPinnedOracle's pins.
func NewPinnedHeldOutLayerOracle(sel Selections, spans []Span, cut, nExpert int, bytes []int64, capacity int) *Oracle {
	return pinnedOracle(NewHeldOutLayerOracle(sel, spans, cut, nExpert, bytes, capacity))
}

// heldOut narrows the objective to the decode of spans[cut:], numbering accesses as newOracle does.
func (o *Oracle) heldOut(sel Selections, spans []Span, cut int) *Oracle {
	needed := make([]int, sel.Layers*o.nExpert)
	step := 0
	for i, s := range spans {
		for l := range sel.Layers {
			step++
			var keys []int32
			for t := s.First; t < s.End && t < sel.Tokens; t++ {
				for _, e := range sel.IDs[(l*sel.Tokens+t)*sel.K : (l*sel.Tokens+t+1)*sel.K] {
					k := l*o.nExpert + int(e)
					if needed[k] != step {
						needed[k] = step
						keys = append(keys, int32(k))
						o.costs = append(o.costs, i >= cut && s.Phase == Decode)
					}
				}
			}
			o.stepKeys = append(o.stepKeys, keys)
		}
	}
	return o
}

func newOracle(sel Selections, spans []Span, nExpert int, bytes []int64, budget []int64, perL bool) *Oracle {
	nKey := sel.Layers * nExpert
	o := &Oracle{nExpert: nExpert, bytes: bytes, budget: budget, cur: make([]int64, len(budget)), perL: perL,
		uses: make([][]int32, nKey), ptr: make([]int, nKey), h: make([]keyHeap, len(budget))}
	needed := make([]int, nKey)
	step, access := 0, 0
	for _, s := range spans {
		for l := range sel.Layers {
			step++
			for t := s.First; t < s.End && t < sel.Tokens; t++ {
				for _, e := range sel.IDs[(l*sel.Tokens+t)*sel.K : (l*sel.Tokens+t+1)*sel.K] {
					k := l*nExpert + int(e)
					if needed[k] != step {
						needed[k] = step
						o.uses[k] = append(o.uses[k], int32(access))
						access++
					}
				}
			}
		}
	}
	pos := make([]int, nKey)
	for k := range pos {
		pos[k] = -1
	}
	for g := range o.h {
		o.h[g] = keyHeap{pos: pos, less: func(a, b int) bool { return o.next(a) > o.next(b) }}
	}
	return o
}

// next is what key k's next access is worth: its number, or math.MaxInt for none or for a free one.
func (o *Oracle) next(k int) int { return o.rank(k, o.ptr[k]) }

// rank is next(k) with key k's accesses before its i-th already run.
func (o *Oracle) rank(k, i int) int {
	if i >= len(o.uses[k]) {
		return math.MaxInt
	}
	a := int(o.uses[k][i])
	if o.costs != nil && !o.costs[a] {
		return math.MaxInt
	}
	return a
}

// Err reports a replay that did not follow the trace the Oracle was built from.
func (o *Oracle) Err() error { return o.err }

func (o *Oracle) group(k int) int {
	if o.perL {
		return k / o.nExpert
	}
	return 0
}

func (o *Oracle) size(k int) int64 { return o.bytes[k/o.nExpert] }

// Prefetch releases the last step's pins; it prefetches nothing.
func (o *Oracle) Prefetch(s StepInfo) []int {
	if o.pinned && s.Index+1 != o.step {
		o.step = s.Index + 1
		for _, k := range o.pins {
			heap.Push(&o.h[o.group(k)], k)
		}
		o.pins = o.pins[:0]
		clear(o.pinBytes)
		if o.costs != nil {
			// next does not rank a free step's experts first, so hold them before anything loads
			for _, k := range o.stepKeys[s.Index] {
				o.pin(int(k))
			}
		}
	}
	return nil
}

// pin moves resident key k out of h, so that nothing evicts it in this step.
func (o *Oracle) pin(k int) {
	g := o.group(k)
	if p := o.h[g].pos[k]; p >= 0 {
		heap.Remove(&o.h[g], p)
		o.pins = append(o.pins, k)
		o.pinBytes[g] += o.size(k)
	}
}

func (o *Oracle) Load(k int) []int {
	o.evict = o.evict[:0]
	g := o.group(k)
	if o.pinned {
		if o.pinBytes[g]+o.size(k) > o.budget[g] {
			return append(o.evict, k) // the step's union does not fit: no atomic schedule exists
		}
		h := &o.h[g]
		for o.cur[g]+o.size(k) > o.budget[g] {
			v := heap.Pop(h).(int)
			o.cur[g] -= o.size(v)
			o.evict = append(o.evict, v)
		}
		heap.Push(h, k)
		o.cur[g] += o.size(k)
		return o.evict
	}
	after := o.rank(k, o.ptr[k]+1) // k's next access after the one it is loaded for
	h := &o.h[g]
	for o.cur[g]+o.size(k) > o.budget[g] {
		if h.Len() == 0 || o.next(h.keys[0]) <= after {
			return append(o.evict, k) // k itself is needed last: use it once, keep the rest
		}
		v := heap.Pop(h).(int)
		o.cur[g] -= o.size(v)
		o.evict = append(o.evict, v)
	}
	heap.Push(h, k)
	o.cur[g] += o.size(k)
	return o.evict
}

func (o *Oracle) Use(_ StepInfo, k int) {
	if (o.ptr[k] >= len(o.uses[k]) || int(o.uses[k][o.ptr[k]]) != o.access) && o.err == nil {
		o.err = fmt.Errorf("oracle: access %d is key %d, not the one the trace it was built from has there", o.access, k)
	}
	g := o.group(k)
	if o.pinned {
		o.pin(k)
	}
	o.access++
	o.ptr[k]++
	if p := o.h[g].pos[k]; p >= 0 {
		heap.Fix(&o.h[g], p)
	}
}
