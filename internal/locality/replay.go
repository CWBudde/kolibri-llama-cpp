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

import (
	"fmt"
	"slices"
)

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
	// FullSteps are the steps that found every expert they need resident: for a decoded token
	// its exact Top-K set, for a ubatch its whole union.
	FullSteps int           `json:"full_steps"`
	Layers    []LayerCounts `json:"layers"`
	// Links is the transfer model's estimate per declared link (transfer.go), when timed.
	Links []LinkTime `json:"links,omitempty"`
}

// LayerCounts are one layer's accesses and hits in a phase: its share of hits is the layer's recall.
type LayerCounts struct {
	Accesses int `json:"accesses"`
	Hits     int `json:"hits"`
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

// Replay runs sel through p, span by span. The spans must cover tokens [0, n) in order, each
// exactly once, for some n up to sel.Tokens: an online replay may end before the trace does.
// bytes[l] is the size of one expert of layer l; expert IDs are in [0, nExpert). It fails on a
// span out of order and on a policy that prefetches an unknown key or evicts a key that is not
// resident.
func Replay(sel Selections, spans []Span, nExpert int, bytes []int64, p Policy) (Result, error) {
	return ReplayTimed(sel, spans, nExpert, bytes, p, Timing{})
}

// ReplayTimed is Replay that also estimates, for every link of t, the time the replay takes when
// each load goes over that link (transfer.go).
func ReplayTimed(sel Selections, spans []Span, nExpert int, bytes []int64, p Policy, t Timing) (Result, error) {
	rs, err := ReplaySegments(sel, spans, nil, nExpert, bytes, p, t)
	if err != nil {
		return Result{}, err
	}
	return rs[0], nil
}

// ReplaySegments is ReplayTimed cut into segments: cuts are ascending span indexes, and segment i
// holds the spans from cut i-1 (0 for the first) to cut i (len(spans) for the last). The policy,
// the cache and the links run on across a cut, so a later segment starts warm from the earlier
// ones, as a server's would. A segment counts what its steps cause and its peak is the most bytes
// resident during its steps; prefetches count in the segment of the step they were prefetched
// for. A segment's counts never depend on later steps: at a cut, its prefetches not used yet
// count as unused, as at the end of a replay, and a later use of them is a plain hit. The first
// segment is therefore the replay of its spans alone. Merge adds the segments up to the
// unsegmented replay, except for prefetches used across a cut.
func ReplaySegments(sel Selections, spans []Span, cuts []int, nExpert int, bytes []int64, p Policy,
	t Timing) ([]Result, error) {
	if len(bytes) != sel.Layers {
		return nil, fmt.Errorf("replay: %d layers of expert sizes, %d of selections", len(bytes), sel.Layers)
	}
	if err := t.check(); err != nil {
		return nil, err
	}
	next := 0
	for i, s := range spans {
		if s.First != next || s.End <= s.First || s.End > sel.Tokens {
			return nil, fmt.Errorf("replay: span %d is [%d, %d), want one from %d within %d tokens", i, s.First, s.End,
				next, sel.Tokens)
		}
		next = s.End
	}
	for i, c := range cuts {
		if c <= 0 || c >= len(spans) || (i > 0 && c <= cuts[i-1]) {
			return nil, fmt.Errorf("replay: cuts %v, want ascending span indexes within (0, %d)", cuts, len(spans))
		}
	}
	rs := make([]Result, len(cuts)+1)
	nKey := sel.Layers * nExpert
	for j := range rs {
		for _, c := range []*Counts{&rs[j].Prefill, &rs[j].Decode} {
			c.Layers = make([]LayerCounts, sel.Layers)
			for _, l := range t.Links {
				c.Links = append(c.Links, LinkTime{Link: l.Name})
			}
		}
	}
	r := &rs[0] // the running span's segment
	qs := make([]queue, len(t.Links))
	for i, l := range t.Links {
		qs[i] = queue{bw: l.BytesPerSecond, arrival: make([]float64, nKey)}
	}
	resident := make([]bool, nKey)
	prefetched := make([]*Counts, nKey) // the counts a resident key was prefetched in, while unused
	pending := make([]bool, nKey)       // resident, prefetched and not used yet
	needed := make([]int, nKey)         // the last step that needed the key, plus one
	size := func(k int) int64 { return bytes[k/nExpert] }
	var cur int64
	load := func(k int, c *Counts) error {
		for i := range qs {
			qs[i].load(k, size(k))
		}
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
				prefetched[e].UnusedPrefetchBytes += size(e)
			}
			if e != k {
				c.Evictions++
			}
		}
		r.PeakBytes = max(r.PeakBytes, cur)
		return nil
	}
	step, seg := 0, 0
	for i, s := range spans {
		if seg < len(cuts) && i == cuts[seg] {
			for k, ok := range pending {
				if ok {
					pending[k] = false
					prefetched[k].UnusedPrefetchBytes += size(k)
				}
			}
			seg++
			r = &rs[seg]
			r.PeakBytes = cur
		}
		c := r.phase(s.Phase)
		c.Tokens += s.End - s.First
		compute := t.compute(s)
		for l := range sel.Layers {
			info := StepInfo{step, s, l}
			step++
			c.Steps++
			for _, k := range p.Prefetch(info) {
				if k < 0 || k >= nKey {
					return nil, fmt.Errorf("replay: step %d prefetches key %d of %d", info.Index, k, nKey)
				}
				if resident[k] {
					continue
				}
				c.Prefetches++
				c.PrefetchBytes += size(k)
				prefetched[k], pending[k] = c, true
				if err := load(k, c); err != nil {
					return nil, err
				}
			}
			for i := range qs {
				qs[i].ready = qs[i].now
			}
			full := true
			for tok := s.First; tok < s.End; tok++ {
				for _, e := range sel.IDs[(l*sel.Tokens+tok)*sel.K : (l*sel.Tokens+tok+1)*sel.K] {
					k := l*nExpert + int(e)
					if needed[k] == step { // already used by this step: a ubatch's union
						continue
					}
					needed[k] = step
					c.Accesses++
					c.Layers[l].Accesses++
					if resident[k] {
						c.Hits++
						c.Layers[l].Hits++
						if pending[k] {
							pending[k] = false
							pc := prefetched[k]
							pc.UsefulPrefetches++
							for i := range qs {
								if qs[i].arrival[k] <= qs[i].now {
									pc.Links[i].Timely++
								} else {
									pc.Links[i].Late++
								}
							}
						}
					} else {
						full = false
						c.Misses++
						c.MissBytes += size(k)
						if err := load(k, c); err != nil {
							return nil, err
						}
					}
					for i := range qs {
						qs[i].ready = max(qs[i].ready, qs[i].arrival[k])
					}
					p.Use(info, k)
				}
			}
			if full {
				c.FullSteps++
			}
			for i := range qs {
				q := &qs[i]
				stall := q.ready - q.now
				c.Links[i].StallSeconds += stall
				c.Links[i].Seconds += stall + compute
				q.now = q.ready + compute
			}
		}
	}
	for k, ok := range pending {
		if ok {
			prefetched[k].UnusedPrefetchBytes += size(k)
		}
	}
	return rs, nil
}

// Merge adds the segments of a replay up: their counts and link times, and the largest peak. It
// is the unsegmented replay unless a prefetch made before a cut is used after it.
func Merge(rs ...Result) Result {
	var m Result
	for i, r := range rs {
		if i == 0 {
			m = Result{Prefill: r.Prefill.clone(), Decode: r.Decode.clone(), PeakBytes: r.PeakBytes}
			continue
		}
		m.Prefill.add(r.Prefill)
		m.Decode.add(r.Decode)
		m.PeakBytes = max(m.PeakBytes, r.PeakBytes)
	}
	return m
}

func (c Counts) clone() Counts {
	c.Layers, c.Links = slices.Clone(c.Layers), slices.Clone(c.Links)
	return c
}

// add adds o's counts to c's; both have the same layers and links.
func (c *Counts) add(o Counts) {
	c.Tokens += o.Tokens
	c.Steps += o.Steps
	c.Accesses += o.Accesses
	c.Hits += o.Hits
	c.Misses += o.Misses
	c.MissBytes += o.MissBytes
	c.Prefetches += o.Prefetches
	c.PrefetchBytes += o.PrefetchBytes
	c.UsefulPrefetches += o.UsefulPrefetches
	c.UnusedPrefetchBytes += o.UnusedPrefetchBytes
	c.Evictions += o.Evictions
	c.FullSteps += o.FullSteps
	for l, x := range o.Layers {
		c.Layers[l].Accesses += x.Accesses
		c.Layers[l].Hits += x.Hits
	}
	for i, x := range o.Links {
		c.Links[i].Seconds += x.Seconds
		c.Links[i].StallSeconds += x.StallSeconds
		c.Links[i].Timely += x.Timely
		c.Links[i].Late += x.Late
	}
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
