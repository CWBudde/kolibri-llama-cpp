package locality

// Causal trace replay (PLAN Phase 10, Stage A): a workload's expert selections replayed through a
// cache policy in the order an online server meets them, with the policy learning a step's experts
// only once the step runs.
//
// A trace is one teacher-forced prefill, but a server meets a conversation turn by turn: it
// prefills each prompt's new tokens in ubatches and then decodes the answer token by token
// (OnlineSpans, answers.go). A span is such a ubatch or decoded token; a step is one layer of a
// span and needs the experts its tokens select at that layer, for a ubatch their union. Steps run
// span by span, within a span layer by layer.
//
// Causality is the evaluator's, not the policy's: before a step Replay tells the policy only the
// step's position (StepInfo) and asks what to prefetch; only then does it reveal the step's
// experts, one Use at a time. At prediction time a policy therefore knows the experts of every
// earlier step, which for a layer of a span are those of all earlier spans and of the span's
// earlier layers, and nothing of the step itself or any later one.
//
// Counts are kept per phase. A prefill step serves a teacher-forced ubatch of up to hundreds of
// tokens at once; its hits and bytes say nothing about the latency of a decoded token and are not
// to be reported as a decode speedup.

import "fmt"

// Phase is how a server processes a span: a prompt's ubatch at once, or one generated token.
type Phase int

const (
	Prefill Phase = iota
	Decode
)

func (p Phase) String() string {
	if p == Decode {
		return "decode"
	}
	return "prefill"
}

// Span is tokens [First, End) that pass through the model together: a prefill ubatch or one
// decoded token.
type Span struct {
	Phase      Phase
	First, End int
}

// StepInfo is what a policy knows of a step before the step runs: its number in the replay, its
// span and its layer, not its experts.
type StepInfo struct {
	Index int
	Span
	Layer int
}

// A Policy decides which experts a cache holds. An expert e of layer l is key l*nExpert+e. For
// every step Replay calls, in this order: Prefetch once; Load for every key it loads, whether
// prefetched or missed; and Use for every expert the step needs, in the order its tokens select
// them, after a hit or after the expert's Load on a miss.
type Policy interface {
	// Prefetch returns keys to load before step s, which it knows only by its StepInfo.
	Prefetch(s StepInfo) []int
	// Load reports that key k is now resident and returns resident keys to evict for it. The
	// slice may be reused by the next call. Returning k itself uses k once without keeping it.
	Load(k int) []int
	// Use reports that the running step s uses key k: resident, or just loaded and dropped.
	Use(s StepInfo, k int)
}

// Counts are one phase's totals of a replay. Prefetches, their bytes, the useful ones (used by a
// later step before eviction) and the unused bytes (evicted or still unused at the end) count in
// the phase of the step they were prefetched for; evictions in the phase of the step that caused
// them.
type Counts struct {
	Tokens              int   `json:"tokens"`
	Steps               int   `json:"steps"`
	Accesses            int   `json:"accesses"`
	Hits                int   `json:"hits"`
	Misses              int   `json:"misses"`
	MissBytes           int64 `json:"miss_bytes"`
	Prefetches          int   `json:"prefetches"`
	PrefetchBytes       int64 `json:"prefetch_bytes"`
	UsefulPrefetches    int   `json:"useful_prefetches"`
	UnusedPrefetchBytes int64 `json:"unused_prefetch_bytes"`
	Evictions           int   `json:"evictions"`
}

// Result is a replay's counts per phase and the most expert bytes resident at once.
type Result struct {
	Prefill   Counts `json:"prefill"`
	Decode    Counts `json:"decode"`
	PeakBytes int64  `json:"peak_bytes"`
}

func (r *Result) phase(p Phase) *Counts {
	if p == Decode {
		return &r.Decode
	}
	return &r.Prefill
}

// TraceOrder is every token as its own decode span: the order of cmd/kolibri-cache's simulation,
// not that of an online server.
func TraceOrder(tokens int) []Span {
	out := make([]Span, tokens)
	for t := range out {
		out[t] = Span{Decode, t, t + 1}
	}
	return out
}

// Replay runs sel through p, span by span. The spans must cover the tokens [0, sel.Tokens) in
// order, each exactly once. bytes[l] is the size of one expert of layer l; expert IDs are in
// [0, nExpert). It fails on a span out of order and on a policy that prefetches an unknown key or
// evicts a key that is not resident.
func Replay(sel Selections, spans []Span, nExpert int, bytes []int64, p Policy) (Result, error) {
	var r Result
	if len(bytes) != sel.Layers {
		return r, fmt.Errorf("replay: %d layers of expert sizes, %d of selections", len(bytes), sel.Layers)
	}
	next := 0
	for i, s := range spans {
		if s.First != next || s.End <= s.First || s.End > sel.Tokens {
			return r, fmt.Errorf("replay: span %d is [%d, %d), want one from %d within %d tokens", i, s.First, s.End,
				next, sel.Tokens)
		}
		next = s.End
	}
	if next != sel.Tokens {
		return r, fmt.Errorf("replay: the spans end at token %d of %d", next, sel.Tokens)
	}
	nKey := sel.Layers * nExpert
	resident := make([]bool, nKey)
	prefetched := make([]Phase, nKey) // the phase a resident key was prefetched in, while unused
	pending := make([]bool, nKey)     // resident, prefetched and not used yet
	needed := make([]int, nKey)       // the last step that needed the key, plus one
	size := func(k int) int64 { return bytes[k/nExpert] }
	var cur int64
	load := func(k int, c *Counts) error {
		resident[k] = true
		cur += size(k)
		for _, e := range p.Load(k) {
			if e < 0 || e >= nKey || !resident[e] {
				return fmt.Errorf("replay: the policy evicts key %d, which is not resident", e)
			}
			resident[e] = false
			cur -= size(e)
			if pending[e] {
				pending[e] = false
				r.phase(prefetched[e]).UnusedPrefetchBytes += size(e)
			}
			if e != k {
				c.Evictions++
			}
		}
		r.PeakBytes = max(r.PeakBytes, cur)
		return nil
	}
	step := 0
	for _, s := range spans {
		c := r.phase(s.Phase)
		c.Tokens += s.End - s.First
		for l := range sel.Layers {
			info := StepInfo{step, s, l}
			step++
			c.Steps++
			for _, k := range p.Prefetch(info) {
				if k < 0 || k >= nKey {
					return r, fmt.Errorf("replay: step %d prefetches key %d of %d", info.Index, k, nKey)
				}
				if resident[k] {
					continue
				}
				c.Prefetches++
				c.PrefetchBytes += size(k)
				prefetched[k], pending[k] = s.Phase, true
				if err := load(k, c); err != nil {
					return r, err
				}
			}
			for t := s.First; t < s.End; t++ {
				for _, e := range sel.IDs[(l*sel.Tokens+t)*sel.K : (l*sel.Tokens+t+1)*sel.K] {
					k := l*nExpert + int(e)
					if needed[k] == step { // already used by this step: a ubatch's union
						continue
					}
					needed[k] = step
					c.Accesses++
					if resident[k] {
						c.Hits++
						if pending[k] {
							pending[k] = false
							r.phase(prefetched[k]).UsefulPrefetches++
						}
					} else {
						c.Misses++
						c.MissBytes += size(k)
						if err := load(k, c); err != nil {
							return r, err
						}
					}
					p.Use(info, k)
				}
			}
		}
	}
	for k, ok := range pending {
		if ok {
			r.phase(prefetched[k]).UnusedPrefetchBytes += size(k)
		}
	}
	return r, nil
}

// LayerLRU is an LRU cache of the same number of experts in every layer, cmd/kolibri-cache's
// per-layer policy; it prefetches nothing. It is the evaluator's reference: replayed in
// TraceOrder, its hits are those Hits counts on each layer's stack distances. A capacity of 0
// keeps nothing.
type LayerLRU struct {
	nExpert, capacity int
	prev, next        []int // a list per layer, most recent first; key nKey+l is layer l's head
	size              []int
	evict             []int
}

// NewLayerLRU returns an empty LayerLRU for layers of nExpert experts.
func NewLayerLRU(layers, nExpert, capacity int) *LayerLRU {
	n := layers*nExpert + layers
	c := &LayerLRU{nExpert: nExpert, capacity: capacity, prev: make([]int, n), next: make([]int, n),
		size: make([]int, layers)}
	for l := range layers {
		h := layers*nExpert + l
		c.prev[h], c.next[h] = h, h
	}
	return c
}

func (c *LayerLRU) head(k int) int { return len(c.size)*c.nExpert + k/c.nExpert }

func (c *LayerLRU) unlink(k int) {
	c.next[c.prev[k]], c.prev[c.next[k]] = c.next[k], c.prev[k]
}

func (c *LayerLRU) pushFront(k int) {
	h := c.head(k)
	c.prev[k], c.next[k] = h, c.next[h]
	c.prev[c.next[h]] = k
	c.next[h] = k
}

func (c *LayerLRU) Prefetch(StepInfo) []int { return nil }

func (c *LayerLRU) Load(k int) []int {
	c.evict = c.evict[:0]
	if c.capacity == 0 {
		return append(c.evict, k)
	}
	l := k / c.nExpert
	if c.size[l] == c.capacity {
		v := c.prev[c.head(k)]
		c.unlink(v)
		c.size[l]--
		c.evict = append(c.evict, v)
	}
	c.pushFront(k)
	c.size[l]++
	return c.evict
}

func (c *LayerLRU) Use(_ StepInfo, k int) {
	if c.capacity == 0 {
		return
	}
	c.unlink(k)
	c.pushFront(k)
}
