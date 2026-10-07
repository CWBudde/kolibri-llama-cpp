// Package locality measures how a workload's expert selections concentrate, per layer: the
// question behind streaming or caching experts (PLAN Phase 8).
//
// For one layer, a workload is a sequence of tokens, each with the K experts the router selected.
// Measure reports, over that sequence:
//
//   - Coverage: the share of all experts selected at least once in tokens [0, t), at fixed t and at
//     the end;
//   - Counts: how often each expert was selected; MaxShare (the top expert's share of all
//     activations) and MedianCount summarize them;
//   - TopShare(N): the share of activations taken by the N most selected experts, and
//     ExpertsFor(f): how many experts (most selected first) take at least the share f;
//   - Entropy: the Shannon entropy of the selection frequencies in bits, NormEntropy the same over
//     log2 of the expert count (1 for uniform use), and Gini, the Gini coefficient of the counts
//     over all experts, those never selected included (0 for uniform use);
//   - reuse: for every selection of an expert selected before (Reuses of them), TokenDistance is
//     the number of tokens since its previous selection and StackDistance the number of other
//     distinct experts selected strictly between the two, which is what an LRU cache must hold
//     besides it to hit; both as nearest-rank median, 90th and 99th percentiles. ColdShare is
//     the share of activations that are an expert's first;
//   - NeverSelected: the share of experts the layer never selects.
package locality

import (
	"math"
	"sort"
)

// Point is a coverage read-out: the share of experts selected at least once in the first Tokens.
type Point struct {
	Tokens int     `json:"tokens"`
	Share  float64 `json:"share"`
}

// Quantiles are nearest-rank percentiles of a distance; all zero when nothing was reused.
type Quantiles struct {
	Median int `json:"median"`
	P90    int `json:"p90"`
	P99    int `json:"p99"`
}

// Stats are one layer's measures over one workload.
type Stats struct {
	Tokens        int       `json:"tokens"`
	Activations   int       `json:"activations"`
	Coverage      []Point   `json:"coverage"`
	Counts        []int     `json:"counts"`
	MaxShare      float64   `json:"max_share"`
	MedianCount   float64   `json:"median_count"`
	Entropy       float64   `json:"entropy_bits"`
	NormEntropy   float64   `json:"norm_entropy"`
	Gini          float64   `json:"gini"`
	Reuses        int       `json:"reuses"`
	ColdShare     float64   `json:"cold_share"`
	TokenDistance Quantiles `json:"token_distance"`
	StackDistance Quantiles `json:"stack_distance"`
	NeverSelected float64   `json:"never_selected"`
}

// Measure computes a layer's Stats from its selections (sel[t] are token t's experts, each in
// [0, nExpert)). Coverage is read out after each count in at that does not exceed the token count,
// and at the end.
func Measure(sel [][]int, nExpert int, at []int) Stats {
	s := Stats{Tokens: len(sel), Counts: make([]int, nExpert)}
	last := make([]int, nExpert) // each expert's last token, -1 before its first selection
	for e := range last {
		last[e] = -1
	}
	reads := append([]int(nil), at...)
	sort.Ints(reads)
	var tokenDist, stackDist []int
	seen, cold := 0, 0
	for t, experts := range sel {
		for len(reads) > 0 && reads[0] <= t {
			if reads[0] > 0 && (len(s.Coverage) == 0 || s.Coverage[len(s.Coverage)-1].Tokens != reads[0]) {
				s.Coverage = append(s.Coverage, Point{reads[0], float64(seen) / float64(nExpert)})
			}
			reads = reads[1:]
		}
		// every distance of this token before any of its updates: the token's other experts
		// are not between, but their earlier selections are
		for _, e := range experts {
			if last[e] < 0 {
				continue
			}
			between := 0
			for x, lx := range last {
				if x != e && lx > last[e] {
					between++
				}
			}
			tokenDist = append(tokenDist, t-last[e])
			stackDist = append(stackDist, between)
		}
		for _, e := range experts {
			if last[e] < 0 {
				seen++
				cold++
			}
			last[e] = t
			s.Counts[e]++
			s.Activations++
		}
	}
	if len(s.Coverage) == 0 || s.Coverage[len(s.Coverage)-1].Tokens != len(sel) {
		s.Coverage = append(s.Coverage, Point{len(sel), float64(seen) / float64(nExpert)})
	}
	s.Reuses = len(tokenDist)
	s.TokenDistance, s.StackDistance = quantiles(tokenDist), quantiles(stackDist)
	if s.Activations == 0 {
		return s
	}
	total := float64(s.Activations)
	s.ColdShare = float64(cold) / total
	never := 0
	for _, c := range s.Counts {
		if c == 0 {
			never++
			continue
		}
		p := float64(c) / total
		s.Entropy -= p * math.Log2(p)
		s.MaxShare = math.Max(s.MaxShare, p)
	}
	s.NeverSelected = float64(never) / float64(nExpert)
	s.NormEntropy = s.Entropy / math.Log2(float64(nExpert))
	asc := append([]int(nil), s.Counts...)
	sort.Ints(asc)
	if n := len(asc); n%2 == 1 {
		s.MedianCount = float64(asc[n/2])
	} else {
		s.MedianCount = float64(asc[n/2-1]+asc[n/2]) / 2
	}
	// Gini = 2 sum_i i x_(i) / (n sum x) - (n+1)/n, with x ascending and i from 1
	weighted := 0.0
	for i, c := range asc {
		weighted += float64(i+1) * float64(c)
	}
	n := float64(len(asc))
	s.Gini = 2*weighted/(n*total) - (n+1)/n
	return s
}

// TopShare is the share of activations taken by the n most selected experts.
func (s Stats) TopShare(n int) float64 {
	if s.Activations == 0 {
		return 0
	}
	sum := 0
	for i, c := range s.descending() {
		if i == n {
			break
		}
		sum += c
	}
	return float64(sum) / float64(s.Activations)
}

// ExpertsFor is how many experts, most selected first, take at least the share f of activations.
func (s Stats) ExpertsFor(f float64) int {
	sum := 0
	for i, c := range s.descending() {
		sum += c
		if float64(sum) >= f*float64(s.Activations)-1e-9 {
			return i + 1
		}
	}
	return len(s.Counts)
}

func (s Stats) descending() []int {
	d := append([]int(nil), s.Counts...)
	sort.Sort(sort.Reverse(sort.IntSlice(d)))
	return d
}

// SameSet reports whether two tokens selected the same experts, in any order.
func SameSet(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	x, y := append([]int(nil), a...), append([]int(nil), b...)
	sort.Ints(x)
	sort.Ints(y)
	for i := range x {
		if x[i] != y[i] {
			return false
		}
	}
	return true
}

func quantiles(d []int) Quantiles {
	if len(d) == 0 {
		return Quantiles{}
	}
	sort.Ints(d)
	rank := func(q float64) int {
		return d[max(int(math.Ceil(q*float64(len(d))))-1, 0)]
	}
	return Quantiles{Median: rank(0.5), P90: rank(0.9), P99: rank(0.99)}
}
