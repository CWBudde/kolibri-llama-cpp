package main

import (
	"reflect"
	"testing"

	"kolibri-llm/internal/locality"
)

func TestAgreement(t *testing.T) {
	// 2 layers x 2 tokens x 2 experts; layer 0 agrees on both tokens (one in another order),
	// layer 1 on neither
	a := locality.Selections{Layers: 2, Tokens: 2, K: 2, IDs: []int32{0, 1, 2, 3, 4, 5, 6, 7}}
	b := locality.Selections{Layers: 2, Tokens: 2, K: 2, IDs: []int32{1, 0, 2, 3, 4, 9, 6, 8}}
	total, per := agreement(a, b)
	if total != 0.5 || !reflect.DeepEqual(per, []float64{1, 0}) {
		t.Fatalf("agreement = %v, %v", total, per)
	}
}

func TestMeasureReadsOutCoverage(t *testing.T) {
	// 1 layer x 3 tokens: coverage is read out only at the end, since 1024 is past it
	sel := locality.Selections{Layers: 1, Tokens: 3, K: 2, IDs: []int32{0, 1, 0, 2, 0, 1}}
	ls := measure(sel, 4)
	if len(ls) != 1 || !reflect.DeepEqual(points(ls), []int{3}) || coverage(ls[0], 3) != 0.75 {
		t.Fatalf("points %v, coverage at 3 %v", points(ls), coverage(ls[0], 3))
	}
	if ls[0].Top6 != 1 || ls[0].For50 != 1 || ls[0].For99 != 3 {
		t.Fatalf("Top6 %v, For50 %d, For99 %d", ls[0].Top6, ls[0].For50, ls[0].For99)
	}
}
