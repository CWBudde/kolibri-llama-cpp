package main

import (
	"math"
	"testing"
)

func TestE4M3(t *testing.T) {
	tests := map[byte]float64{
		0x00: 0, 0x38: 1, 0xb8: -1, 0x40: 2, 0x3c: 1.5,
		0x7e: 448,               // largest finite
		0x01: math.Ldexp(1, -9), // smallest subnormal
		0x08: math.Ldexp(1, -6), // smallest normal
	}
	for in, want := range tests {
		if got := e4m3(in); got != want {
			t.Errorf("e4m3(%#02x) = %g, want %g", in, got, want)
		}
	}
	if !math.IsNaN(e4m3(0x7f)) || !math.IsNaN(e4m3(0xff)) {
		t.Error("0x7f/0xff must be NaN")
	}
}

func TestDecodeBF16(t *testing.T) {
	// 1.0 = 0x3f80, -2.0 = 0xc000 (little endian)
	got, err := decode("BF16", []byte{0x80, 0x3f, 0x00, 0xc0})
	if err != nil || len(got) != 2 || got[0] != 1 || got[1] != -2 {
		t.Errorf("decode = %v, %v", got, err)
	}
}

func TestDequantBlocks(t *testing.T) {
	// 3x3 matrix, 2x2 blocks -> 2x2 scales; last block row/col is partial.
	w := []float64{1, 1, 1, 1, 1, 1, 1, 1, 1}
	if err := dequantBlocks(w, []int64{3, 3}, []float64{2, 3, 5, 7}, []int64{2, 2}); err != nil {
		t.Fatal(err)
	}
	want := []float64{2, 2, 3, 2, 2, 3, 5, 5, 7}
	for i := range want {
		if w[i] != want[i] {
			t.Fatalf("got %v, want %v", w, want)
		}
	}
}
