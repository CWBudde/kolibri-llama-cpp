package locality

// Transfer model (PLAN Phase 10, Stage A): what the loads of a replay would cost in time, under
// declared assumptions. It is an estimate, never a measurement.
//
// Every load, prefetched or missed, goes over one link of constant bandwidth, one load after
// another. A step takes its compute time once all the experts it needs have arrived; until then
// it stalls. A prefetch enters the link when Prefetch returns, before the step it is called for;
// a miss when the step reaches it, which is the step's earliest start. A used prefetch is timely
// if it arrived by the earliest start of the step that first used it, which therefore did not wait
// for it, and late if not: it was accurate, but too late to hide its transfer.
//
// Each link of a Timing is replayed on its own, as the only link: SSD, PCIe and RAM are
// alternatives for where experts come from, not stages of one path.

import (
	"fmt"
	"math"
)

// Link is a declared path for expert bytes at a constant bandwidth, e.g. an SSD or PCIe.
type Link struct {
	Name           string
	BytesPerSecond float64
}

// Timing are the transfer model's assumptions: the links and the compute time of a step that
// waits for nothing, DecodeStep seconds for one layer of a decoded token and PrefillToken seconds
// per token for one layer of a prefill ubatch. No links: no estimate.
type Timing struct {
	Links        []Link
	DecodeStep   float64
	PrefillToken float64
}

func (t Timing) check() error {
	for _, l := range t.Links {
		if !(l.BytesPerSecond > 0) || math.IsInf(l.BytesPerSecond, 1) {
			return fmt.Errorf("replay: link %s has no finite positive bandwidth: %g", l.Name, l.BytesPerSecond)
		}
	}
	if !finite(t.DecodeStep) || !finite(t.PrefillToken) {
		return fmt.Errorf("replay: compute times %g and %g are not finite and non-negative", t.DecodeStep,
			t.PrefillToken)
	}
	return nil
}

// finite reports whether v is a finite, non-negative time.
func finite(v float64) bool { return v >= 0 && !math.IsInf(v, 1) }

// compute is the time one layer of span s takes once its experts are there.
func (t Timing) compute(s Span) float64 {
	if s.Phase == Decode {
		return t.DecodeStep
	}
	return t.PrefillToken * float64(s.End-s.First)
}

// LinkTime is a phase's estimate on one link: the time of its steps, compute and stalls, the stalls
// alone, and its used prefetches that arrived in time and those that did not (Timely + Late =
// UsefulPrefetches).
type LinkTime struct {
	Link         string  `json:"link"`
	Seconds      float64 `json:"seconds"`
	StallSeconds float64 `json:"stall_seconds"`
	Timely       int     `json:"timely_prefetches"`
	Late         int     `json:"late_prefetches"`
}

// queue is one link during a replay.
type queue struct {
	bw      float64
	now     float64   // the running step's earliest start: when the step before it ended
	free    float64   // when the link finishes the loads queued so far
	ready   float64   // when the running step's experts have all arrived, now at the earliest
	arrival []float64 // when each key's latest load arrives
}

func (q *queue) load(k int, b int64) {
	q.free = max(q.now, q.free) + float64(b)/q.bw
	q.arrival[k] = q.free
}
