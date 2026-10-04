package safetensors

import (
	"bytes"
	"slices"
	"strings"
	"testing"
)

func TestWriteRoundTrip(t *testing.T) {
	in := []WriteTensor{
		{Name: "b", DType: "F32", Shape: []int64{3}, Data: bytes.Repeat([]byte{2}, 12)},
		{Name: "a", DType: "BF16", Shape: []int64{2, 2}, Data: []byte{1, 2, 3, 4, 5, 6, 7, 8}},
	}
	var buf bytes.Buffer
	if err := Write(&buf, in, map[string]string{"format": "pt"}); err != nil {
		t.Fatal(err)
	}
	raw := buf.Bytes()
	h, err := ReadHeader(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	if h.Metadata["format"] != "pt" {
		t.Errorf("metadata = %v", h.Metadata)
	}
	if h.FileSize() != int64(len(raw)) {
		t.Errorf("FileSize = %d, file has %d bytes", h.FileSize(), len(raw))
	}
	if (8+h.Size)%8 != 0 {
		t.Errorf("data section starts at %d, not 8-byte aligned", 8+h.Size)
	}
	// Written in name order, so "a" comes first in the data section.
	want := map[string]WriteTensor{"a": in[1], "b": in[0]}
	for i, got := range h.Tensors {
		w := want[got.Name]
		if got.Name != []string{"a", "b"}[i] || got.DType != w.DType || !slices.Equal(got.Shape, w.Shape) {
			t.Errorf("tensor %d = %+v, want %s %s%v", i, got, w.Name, w.DType, w.Shape)
		}
		data := raw[8+h.Size+got.Offsets[0] : 8+h.Size+got.Offsets[1]]
		if !bytes.Equal(data, w.Data) {
			t.Errorf("%s: data %v, want %v", got.Name, data, w.Data)
		}
	}
}

func TestWriteRejects(t *testing.T) {
	tests := []struct {
		name string
		in   []WriteTensor
		want string
	}{
		{"size", []WriteTensor{{Name: "a", DType: "BF16", Shape: []int64{3}, Data: make([]byte, 4)}}, "want 6"},
		// [-1, -1] has a positive element count, so the size check alone passes.
		{"negative", []WriteTensor{{Name: "a", DType: "U8", Shape: []int64{-1, -1}, Data: []byte{0}}}, "negative dimension"},
		{"dtype", []WriteTensor{{Name: "a", DType: "Q4", Shape: []int64{1}, Data: make([]byte, 1)}}, "unknown dtype"},
		{"duplicate", []WriteTensor{{Name: "a", DType: "U8", Shape: []int64{1}, Data: []byte{0}}, {Name: "a", DType: "U8", Shape: []int64{1}, Data: []byte{0}}}, "duplicate"},
		{"reserved", []WriteTensor{{Name: "__metadata__", DType: "U8", Shape: []int64{1}, Data: []byte{0}}}, "reserved"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := Write(&bytes.Buffer{}, tc.in, nil)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want containing %q", err, tc.want)
			}
		})
	}
}
