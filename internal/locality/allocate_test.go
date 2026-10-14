package locality

import (
	"math/rand/v2"
	"reflect"
	"slices"
	"testing"
)

// onlineSplit is a random trace of 3 layers x 300 tokens, 3 of 12 experts, served as two answers
// and split after the first.
func onlineSplit(r *rand.Rand) (Selections, []Span, int) {
	sel := randomSelections(r, 3, 300, 3, 12)
	answers := []Answer{{Prompt: 70, Tokens: make([]int32, 60)}, {Prompt: 180, Tokens: make([]int32, 100)}}
	spans, err := OnlineSpans(300, answers, true, 32)
	if err != nil {
		panic(err)
	}
	cut, err := SplitAfter(spans, answers, 1)
	if err != nil {
		panic(err)
	}
	return sel, spans, cut
}

func TestMissCurves(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	for trial := range 5 {
		sel, spans, cut := onlineSplit(r)
		bytes := sizes(sel.Layers, 100)
		train, err := MissCurves(sel, spans, 0, cut, 12, bytes)
		if err != nil {
			t.Fatal(err)
		}
		test, err := MissCurves(sel, spans, cut, len(spans), 12, bytes)
		if err != nil {
			t.Fatal(err)
		}
		// at every capacity, the curves are the per-layer decode miss bytes of a layer LRU replayed
		// online, in the training prefix and in the test suffix
		for c := 0; c <= 12; c++ {
			segs, err := ReplaySegments(sel, spans, []int{cut}, 12, bytes, NewLayerLRU(sel.Layers, 12, c), Timing{})
			if err != nil {
				t.Fatal(err)
			}
			for l := range sel.Layers {
				for i, curve := range [][][]int64{train, test} {
					x := segs[i].Decode.Layers[l]
					if want := int64(x.Accesses-x.Hits) * bytes[l]; curve[l][c] != want {
						t.Fatalf("trial %d, segment %d, layer %d, capacity %d: curve %d, replay %d", trial, i, l, c,
							curve[l][c], want)
					}
				}
			}
		}
		// at nExpert only first uses miss: the compulsory misses no cache avoids
		for l := range sel.Layers {
			seen := map[int32]bool{}
			var cold int64
			for i, s := range spans {
				for tok := s.First; tok < s.End; tok++ {
					for _, e := range sel.IDs[(l*sel.Tokens+tok)*sel.K : (l*sel.Tokens+tok+1)*sel.K] {
						if !seen[e] && i >= cut && s.Phase == Decode {
							cold += bytes[l]
						}
						seen[e] = true
					}
				}
			}
			if test[l][12] != cold {
				t.Fatalf("trial %d, layer %d: %d bytes at full capacity, %d of first uses", trial, l, test[l][12], cold)
			}
		}
		// the training curves never look at the test suffix
		for l := range sel.Layers {
			for tok := spans[cut].First; tok < sel.Tokens; tok++ {
				i := (l*sel.Tokens + tok) * sel.K
				for j := range sel.K {
					sel.IDs[i+j] = int32((int(sel.IDs[i+j]) + 5) % 12)
				}
			}
		}
		again, err := MissCurves(sel, spans, 0, cut, 12, bytes)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(train, again) {
			t.Fatalf("trial %d: the training curves follow the test suffix", trial)
		}
	}
	if _, err := MissCurves(Selections{Layers: 1, Tokens: 2, K: 1, IDs: []int32{0, 1}}, TraceOrder(2), 1, 3, 2,
		[]int64{1}); err == nil {
		t.Fatal("spans beyond the replay accepted")
	}
}

// misses is the miss bytes of capacities caps on curves.
func misses(curves [][]int64, caps []int) int64 {
	var m int64
	for l, c := range caps {
		m += curves[l][c]
	}
	return m
}

// spent is the bytes capacities caps take.
func spent(caps []int, bytes []int64) int64 {
	var b int64
	for l, c := range caps {
		b += int64(c) * bytes[l]
	}
	return b
}

// bestMisses is the fewest miss bytes of any capacities within budget, by brute force.
func bestMisses(curves [][]int64, bytes []int64, budget int64) int64 {
	best := int64(-1)
	caps := make([]int, len(curves))
	var walk func(l int)
	walk = func(l int) {
		if l == len(curves) {
			if m := misses(curves, caps); spent(caps, bytes) <= budget && (best < 0 || m < best) {
				best = m
			}
			return
		}
		for c := range curves[l] {
			caps[l] = c
			walk(l + 1)
		}
	}
	walk(0)
	return best
}

func TestAllocate(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 10))
	// convex curves of equal expert sizes: greedy is optimal at every budget
	for trial := range 200 {
		curves := make([][]int64, 3)
		for l := range curves {
			drops := make([]int64, 5)
			for i := range drops {
				drops[i] = int64(r.IntN(20))
			}
			slices.Sort(drops)
			slices.Reverse(drops) // decreasing gains: convex
			curves[l] = []int64{100}
			for _, d := range drops {
				curves[l] = append(curves[l], curves[l][len(curves[l])-1]-d)
			}
		}
		bytes := []int64{3, 3, 3}
		for budget := int64(0); budget <= 48; budget++ {
			caps := Allocate(curves, bytes, budget)
			if spent(caps, bytes) > budget {
				t.Fatalf("trial %d, budget %d: %v spends %d", trial, budget, caps, spent(caps, bytes))
			}
			if m, best := misses(curves, caps), bestMisses(curves, bytes, budget); m != best {
				t.Fatalf("trial %d, budget %d: %v misses %d bytes, the optimum %d (%v)", trial, budget, caps, m, best,
					curves)
			}
		}
	}
	// not convex: layer 0 gains nothing from one expert but everything from two, which beats layer
	// 1's two smaller steps
	curves := [][]int64{{10, 10, 0}, {10, 6, 4}}
	if caps := Allocate(curves, []int64{1, 1}, 2); !slices.Equal(caps, []int{2, 0}) {
		t.Fatalf("over a step: %v, want [2 0]", caps)
	}
	// a flat layer gets nothing while another gains; budget left over is filled level
	curves = [][]int64{{5, 5, 5}, {9, 4, 0}}
	if caps := Allocate(curves, []int64{1, 1}, 2); !slices.Equal(caps, []int{0, 2}) {
		t.Fatalf("a flat layer: %v, want [0 2]", caps)
	}
	if caps := Allocate(curves, []int64{1, 1}, 3); !slices.Equal(caps, []int{1, 2}) {
		t.Fatalf("budget left over: %v, want [1 2]", caps)
	}
	// sizes in bytes: one expert of layer 0 costs 3 bytes, of layer 1 one byte
	curves = [][]int64{{9, 0}, {8, 4, 2, 1}}
	bytes := []int64{3, 1}
	for budget := int64(0); budget <= 8; budget++ {
		caps := Allocate(curves, bytes, budget)
		if spent(caps, bytes) > budget || caps[0] > 1 || caps[1] > 3 {
			t.Fatalf("budget %d: %v spends %d", budget, caps, spent(caps, bytes))
		}
	}
	if caps := Allocate(curves, bytes, 4); !slices.Equal(caps, []int{1, 1}) {
		t.Fatalf("mixed sizes: %v, want [1 1]", caps)
	}
	// more budget than experts
	if caps := Allocate(curves, bytes, 100); !slices.Equal(caps, []int{1, 3}) {
		t.Fatalf("all of it: %v, want [1 3]", caps)
	}
}

func TestLayerLRUSizes(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	sel, spans, _ := onlineSplit(r)
	bytes := sizes(sel.Layers, 10)
	// equal capacities are NewLayerLRU
	for _, c := range []int{0, 1, 4, 12} {
		a, err := Replay(sel, spans, 12, bytes, NewLayerLRU(sel.Layers, 12, c))
		if err != nil {
			t.Fatal(err)
		}
		b, err := Replay(sel, spans, 12, bytes, NewLayerLRUSizes(12, []int{c, c, c}))
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(a, b) {
			t.Fatalf("capacity %d: %+v\nwant %+v", c, b, a)
		}
	}
	// each layer is its own cache: layer 1 keeps nothing, layer 2 everything
	res, err := Replay(sel, spans, 12, bytes, NewLayerLRUSizes(12, []int{4, 0, 12}))
	if err != nil {
		t.Fatal(err)
	}
	one, err := Replay(sel, spans, 12, bytes, NewLayerLRU(sel.Layers, 12, 4))
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []Counts{res.Prefill, res.Decode} {
		if c.Layers[1].Hits != 0 {
			t.Fatalf("layer 1 of capacity 0 hits %d times", c.Layers[1].Hits)
		}
	}
	if res.Decode.Layers[0] != one.Decode.Layers[0] {
		t.Fatalf("layer 0 depends on the others: %+v, %+v", res.Decode.Layers[0], one.Decode.Layers[0])
	}
	if res.PeakBytes > 4*bytes[0]+12*bytes[2] {
		t.Fatalf("peak %d bytes over the capacities", res.PeakBytes)
	}
}
