package locality

// Layer budgets (PLAN Phase 10, Stage A): a cache of one LRU per layer whose capacities differ by
// layer, fitted to a training prefix. A layer's miss curve is its decode miss bytes at every
// capacity, from one pass over its stack distances; Allocate spends a byte budget where the
// curves fall fastest. The capacities are frozen before the test suffix is replayed.

import "fmt"

// MissCurves are, for every layer l and capacity c in [0, nExpert], the bytes the decode misses
// of spans[from:to] load in an LRU cache of c experts of layer l, replayed online over spans[:to]:
// a ubatch's union once, in Replay's order. The spans before from warm the caches and are not
// counted; spans from to on are not looked at. At capacity nExpert only an expert's first use in
// the replay misses: the compulsory misses no cache avoids.
func MissCurves(sel Selections, spans []Span, from, to, nExpert int, bytes []int64) ([][]int64, error) {
	if from < 0 || from > to || to > len(spans) || len(bytes) != sel.Layers {
		return nil, fmt.Errorf("miss curves: spans [%d, %d) of %d, %d layers of expert sizes for %d", from, to,
			len(spans), len(bytes), sel.Layers)
	}
	out := make([][]int64, sel.Layers)
	for l := range sel.Layers {
		var keys []int
		var counted []bool
		used := make([]int, nExpert) // the last span that used the expert, plus one
		for i, s := range spans[:to] {
			for tok := s.First; tok < s.End; tok++ {
				for _, e := range sel.IDs[(l*sel.Tokens+tok)*sel.K : (l*sel.Tokens+tok+1)*sel.K] {
					if used[e] == i+1 {
						continue
					}
					used[e] = i + 1
					keys = append(keys, int(e))
					counted = append(counted, i >= from && s.Phase == Decode)
				}
			}
		}
		// an access of stack distance d misses in an LRU of up to d experts, a first one in all: a
		// difference array over the capacities
		diff := make([]int64, nExpert+2)
		for i, d := range StackDistances(keys, nExpert) {
			if !counted[i] {
				continue
			}
			if d < 0 {
				d = nExpert
			}
			diff[0]++
			diff[d+1]--
		}
		out[l] = make([]int64, nExpert+1)
		var m int64
		for c := range out[l] {
			m += diff[c]
			out[l][c] = m * bytes[l]
		}
	}
	return out, nil
}

// Allocate gives every layer l a capacity in [0, len(curves[l])-1] experts of bytes[l] each, at
// most budget bytes in all, for the fewest miss bytes on curves. It grows, one step at a time, the
// layer whose next step saves the most bytes per byte spent, the first of equal ones. The steps
// follow each curve's lower convex hull, so a layer may grow by several experts at once where a
// single one saves less than the next ones do. A layer stops growing where its next step saves
// nothing or no longer fits; budget left over is then spread level, the smallest capacity first.
// With equal expert sizes and convex curves the result is optimal.
func Allocate(curves [][]int64, bytes []int64, budget int64) []int {
	hulls := make([][]int, len(curves))
	for l, c := range curves {
		hulls[l] = lowerHull(c)
	}
	caps := make([]int, len(curves))
	next := make([]int, len(curves)) // the index of the layer's next hull point
	for l := range next {
		next[l] = 1
	}
	left := budget
	for {
		best, gain := -1, 0.0
		for l, h := range hulls {
			if next[l] >= len(h) {
				continue
			}
			c := h[next[l]]
			cost := int64(c-caps[l]) * bytes[l]
			saved := curves[l][caps[l]] - curves[l][c]
			if saved <= 0 || cost > left {
				next[l] = len(h)
				continue
			}
			if g := float64(saved) / float64(cost); best < 0 || g > gain {
				best, gain = l, g
			}
		}
		if best < 0 {
			break
		}
		c := hulls[best][next[best]]
		left -= int64(c-caps[best]) * bytes[best]
		caps[best] = c
		next[best]++
	}
	for {
		l := -1
		for i := range caps {
			if caps[i] < len(curves[i])-1 && bytes[i] <= left && (l < 0 || caps[i] < caps[l]) {
				l = i
			}
		}
		if l < 0 {
			return caps
		}
		caps[l]++
		left -= bytes[l]
	}
}

// lowerHull is the capacities on the lower convex hull of curve, from 0 to its last one. Points on
// a straight piece of it stay in, so that each piece of equal slope is taken one step at a time.
func lowerHull(curve []int64) []int {
	var h []int
	for c := range curve {
		for len(h) >= 2 {
			a, b := h[len(h)-2], h[len(h)-1]
			// b lies above the line from a to c: drop it
			if float64(curve[b]-curve[a])*float64(c-a) > float64(curve[c]-curve[a])*float64(b-a) {
				h = h[:len(h)-1]
				continue
			}
			break
		}
		h = append(h, c)
	}
	return h
}
