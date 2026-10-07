package locality

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"
)

// npy builds a version 1.0 .npy file as numpy.save writes it: magic, version, header length,
// the header dict padded with spaces to a multiple of 64 bytes and ended by a newline, then the data.
func npy(header string, data any) []byte {
	var b bytes.Buffer
	b.WriteString("\x93NUMPY\x01\x00")
	pad := 64 - (10+len(header)+1)%64
	header += strings.Repeat(" ", pad%64) + "\n"
	_ = binary.Write(&b, binary.LittleEndian, uint16(len(header)))
	b.WriteString(header)
	_ = binary.Write(&b, binary.LittleEndian, data)
	return b.Bytes()
}

func TestReadNPYInt16(t *testing.T) {
	// 2 layers x 3 tokens x 2 experts
	ids := []int16{0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 383}
	s, err := ReadNPY(bytes.NewReader(npy("{'descr': '<i2', 'fortran_order': False, 'shape': (2, 3, 2), }", ids)))
	if err != nil {
		t.Fatal(err)
	}
	if s.Layers != 2 || s.Tokens != 3 || s.K != 2 {
		t.Fatalf("shape %d x %d x %d", s.Layers, s.Tokens, s.K)
	}
	layer := s.Layer(1)
	if len(layer) != 3 || layer[0][0] != 10 || layer[2][1] != 383 {
		t.Fatalf("Layer(1) = %v", layer)
	}
}

func TestReadNPYInt32(t *testing.T) {
	ids := []int32{7, 8}
	s, err := ReadNPY(bytes.NewReader(npy("{'descr': '<i4', 'fortran_order': False, 'shape': (1, 1, 2), }", ids)))
	if err != nil {
		t.Fatal(err)
	}
	if got := s.Layer(0); len(got) != 1 || got[0][0] != 7 || got[0][1] != 8 {
		t.Fatalf("Layer(0) = %v", got)
	}
}

func TestReadTokens(t *testing.T) {
	ids, err := ReadTokens(bytes.NewReader(npy("{'descr': '<i4', 'fortran_order': False, 'shape': (3,), }", []int32{1, 100278, 7})))
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 3 || ids[0] != 1 || ids[1] != 100278 || ids[2] != 7 {
		t.Fatalf("ReadTokens = %v", ids)
	}
}

func TestReadTokensRejects(t *testing.T) {
	for name, file := range map[string][]byte{
		"3-D":                npy("{'descr': '<i4', 'fortran_order': False, 'shape': (1, 1, 2), }", []int32{1, 2}),
		"float data":         npy("{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }", []float32{1}),
		"short data":         npy("{'descr': '<i4', 'fortran_order': False, 'shape': (3,), }", []int32{1, 2}),
		"negative dimension": npy("{'descr': '<i4', 'fortran_order': False, 'shape': (-2,), }", []int32{1, 2}),
		"oversized":          npy("{'descr': '<i4', 'fortran_order': False, 'shape': (4294967296,), }", []int32{1}),
		"not npy":            []byte("PK\x03\x04 a zip file"),
	} {
		if _, err := ReadTokens(bytes.NewReader(file)); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

func TestReadNPYRejects(t *testing.T) {
	for name, file := range map[string][]byte{
		"fortran order":      npy("{'descr': '<i2', 'fortran_order': True, 'shape': (1, 1, 2), }", []int16{1, 2}),
		"float data":         npy("{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1, 1), }", []float32{1}),
		"2-D":                npy("{'descr': '<i2', 'fortran_order': False, 'shape': (1, 2), }", []int16{1, 2}),
		"short data":         npy("{'descr': '<i2', 'fortran_order': False, 'shape': (1, 2, 2), }", []int16{1, 2}),
		"not npy":            []byte("PK\x03\x04 a zip file"),
		"negative dimension": npy("{'descr': '<i2', 'fortran_order': False, 'shape': (1, -2, 2), }", []int16{1, 2}),
		"zero dimension":     npy("{'descr': '<i2', 'fortran_order': False, 'shape': (1, 0, 2), }", []int16{}),
		"overflowing size": npy("{'descr': '<i2', 'fortran_order': False, 'shape': (4294967296, 4294967296, 6), }",
			[]int16{1}),
		"oversized": npy("{'descr': '<i2', 'fortran_order': False, 'shape': (1048576, 1048576, 6), }", []int16{1}),
	} {
		if _, err := ReadNPY(bytes.NewReader(file)); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}
