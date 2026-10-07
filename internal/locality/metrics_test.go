package locality

import (
	"math"
	"reflect"
	"testing"
)

// Four experts, two per token:
//
//	t0 {0,1}  t1 {0,2}  t2 {1,0}  t3 {0,1}
//
// Counts 4, 3, 1, 0 of 8 activations; expert 3 is never selected.
var example = [][]int{{0, 1}, {0, 2}, {1, 0}, {0, 1}}

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestMeasureFrequencies(t *testing.T) {
	s := Measure(example, 4, []int{1, 2, 3, 8})
	if !reflect.DeepEqual(s.Counts, []int{4, 3, 1, 0}) {
		t.Fatalf("Counts = %v", s.Counts)
	}
	if s.Tokens != 4 || s.Activations != 8 {
		t.Fatalf("Tokens, Activations = %d, %d", s.Tokens, s.Activations)
	}
	if !near(s.MaxShare, 0.5) || !near(s.MedianCount, 2) {
		t.Fatalf("MaxShare, MedianCount = %v, %v", s.MaxShare, s.MedianCount)
	}
	if !near(s.NeverSelected, 0.25) {
		t.Fatalf("NeverSelected = %v", s.NeverSelected)
	}
}

func TestMeasureCoverage(t *testing.T) {
	// distinct experts in tokens [0, t): {0,1}, {0,1,2}, {0,1,2}, and at the end; 8 is past the end
	s := Measure(example, 4, []int{1, 2, 3, 8})
	want := []Point{{1, 0.5}, {2, 0.75}, {3, 0.75}, {4, 0.75}}
	if !reflect.DeepEqual(s.Coverage, want) {
		t.Fatalf("Coverage = %v, want %v", s.Coverage, want)
	}
}

func TestMeasureConcentration(t *testing.T) {
	s := Measure(example, 4, nil)
	if !near(s.TopShare(1), 4.0/8) || !near(s.TopShare(2), 7.0/8) || !near(s.TopShare(9), 1) {
		t.Fatalf("TopShare(1, 2, 9) = %v, %v, %v", s.TopShare(1), s.TopShare(2), s.TopShare(9))
	}
	// the top expert covers 50%, the top two 87.5%, all three used ones 100%
	got := []int{s.ExpertsFor(0.5), s.ExpertsFor(0.8), s.ExpertsFor(0.9), s.ExpertsFor(1)}
	if !reflect.DeepEqual(got, []int{1, 2, 3, 3}) {
		t.Fatalf("ExpertsFor(0.5, 0.8, 0.9, 1) = %v", got)
	}
	// p = .5, .375, .125, 0: -(.5 log2 .5 + .375 log2 .375 + .125 log2 .125)
	entropy := 0.5 + 0.375*math.Log2(1/0.375) + 0.375
	if !near(s.Entropy, entropy) || !near(s.NormEntropy, entropy/2) {
		t.Fatalf("Entropy, NormEntropy = %v, %v, want %v, %v", s.Entropy, s.NormEntropy, entropy, entropy/2)
	}
	// the sum over ordered pairs of |x_i - x_j| is 28, divided by 2 n^2 mean = 2 * 16 * 2
	if !near(s.Gini, 28.0/64) {
		t.Fatalf("Gini = %v, want %v", s.Gini, 28.0/64)
	}
}

func TestMeasureReuse(t *testing.T) {
	// repeat uses (token: expert, token distance, other experts selected strictly between):
	// t1: 0, 1, 0;  t2: 1, 2, 2 (0 and 2 at t1);  t2: 0, 1, 0;  t3: 0, 1, 0;  t3: 1, 1, 0
	s := Measure(example, 4, nil)
	if s.Reuses != 5 || !near(s.ColdShare, 3.0/8) {
		t.Fatalf("Reuses, ColdShare = %d, %v", s.Reuses, s.ColdShare)
	}
	// nearest rank of [1 1 1 1 2] and [0 0 0 0 2]
	if want := (Quantiles{Median: 1, P90: 2, P99: 2}); s.TokenDistance != want {
		t.Fatalf("TokenDistance = %+v, want %+v", s.TokenDistance, want)
	}
	if want := (Quantiles{Median: 0, P90: 2, P99: 2}); s.StackDistance != want {
		t.Fatalf("StackDistance = %+v, want %+v", s.StackDistance, want)
	}
}

func TestMeasureStackDistanceWithinOneToken(t *testing.T) {
	// The other experts of the reusing token are not "between", but their earlier uses are.
	for _, c := range []struct {
		sel  [][]int
		n    int
		want Quantiles
	}{
		// expert 1 returns at t2 after 2 and 0 at t1; 2 is listed first at t2 and also returns there.
		// Stack distances: t1 0 -> 0, t2 2 -> 0, t2 1 -> 2.
		{[][]int{{1, 0}, {2, 0}, {2, 1}}, 3, Quantiles{Median: 0, P90: 2, P99: 2}},
		// expert 1 returns at t2 after 0 and 3 at t1; 2 is new at t2 and so not between.
		// Stack distances: t1 0 -> 0, t2 1 -> 2.
		{[][]int{{1, 0}, {0, 3}, {2, 1}}, 4, Quantiles{Median: 0, P90: 2, P99: 2}},
	} {
		if s := Measure(c.sel, c.n, nil); s.StackDistance != c.want {
			t.Errorf("%v: StackDistance = %+v, want %+v", c.sel, s.StackDistance, c.want)
		}
	}
}

func TestMeasureSingleToken(t *testing.T) {
	s := Measure([][]int{{2, 3}}, 4, []int{1, 1024})
	if want := []Point{{1, 0.5}}; !reflect.DeepEqual(s.Coverage, want) {
		t.Fatalf("Coverage = %v, want %v", s.Coverage, want)
	}
	if s.Reuses != 0 || s.TokenDistance != (Quantiles{}) || !near(s.ColdShare, 1) || !near(s.NeverSelected, 0.5) {
		t.Fatalf("Reuses %d, TokenDistance %+v, ColdShare %v, NeverSelected %v", s.Reuses, s.TokenDistance,
			s.ColdShare, s.NeverSelected)
	}
	// two equally used experts out of four: 1 bit, half of log2 4; the Gini of [0 0 1 1] is 0.5
	if !near(s.Entropy, 1) || !near(s.NormEntropy, 0.5) || !near(s.Gini, 0.5) {
		t.Fatalf("Entropy, NormEntropy, Gini = %v, %v, %v", s.Entropy, s.NormEntropy, s.Gini)
	}
}

func TestSameSet(t *testing.T) {
	if !SameSet([]int{3, 1, 2}, []int{1, 2, 3}) || SameSet([]int{1, 2, 3}, []int{1, 2, 4}) {
		t.Fatal("SameSet compares sets, regardless of order")
	}
}
