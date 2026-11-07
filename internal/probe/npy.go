package probe

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"regexp"
	"strconv"
)

// The capture stores each layer as a NumPy .npy file: numpy reads it directly, and this reader is
// all Go needs. Only what the capture writes is accepted: little-endian float32, C order, 2-D.

var (
	npyDescrRe   = regexp.MustCompile(`'descr':\s*'([^']+)'`)
	npyFortranRe = regexp.MustCompile(`'fortran_order':\s*(True|False)`)
	npyShapeRe   = regexp.MustCompile(`'shape':\s*\((\d+),\s*(\d+),?\s*\)`)
)

// ReadNpy reads a float32 matrix.
func ReadNpy(r io.Reader) (Matrix, error) {
	var m Matrix
	magic := make([]byte, 8)
	if _, err := io.ReadFull(r, magic); err != nil {
		return m, err
	}
	if string(magic[:6]) != "\x93NUMPY" {
		return m, errors.New("npy: bad magic")
	}
	var headerLen int
	switch magic[6] {
	case 1:
		var n uint16
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return m, err
		}
		headerLen = int(n)
	case 2, 3:
		var n uint32
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return m, err
		}
		headerLen = int(n)
	default:
		return m, fmt.Errorf("npy: version %d", magic[6])
	}
	header := make([]byte, headerLen)
	if _, err := io.ReadFull(r, header); err != nil {
		return m, err
	}
	descr := npyDescrRe.FindSubmatch(header)
	if descr == nil || string(descr[1]) != "<f4" {
		return m, fmt.Errorf("npy: dtype in %q, want <f4", header)
	}
	if f := npyFortranRe.FindSubmatch(header); f == nil || string(f[1]) != "False" {
		return m, fmt.Errorf("npy: order in %q, want C", header)
	}
	shape := npyShapeRe.FindSubmatch(header)
	if shape == nil {
		return m, fmt.Errorf("npy: shape in %q, want 2-D", header)
	}
	var err error
	if m.Rows, err = strconv.Atoi(string(shape[1])); err != nil {
		return m, err
	}
	if m.Dim, err = strconv.Atoi(string(shape[2])); err != nil {
		return m, err
	}
	raw := make([]byte, 4*m.Rows*m.Dim)
	if _, err := io.ReadFull(r, raw); err != nil {
		return m, fmt.Errorf("npy: data: %w", err)
	}
	if n, _ := r.Read(make([]byte, 1)); n != 0 {
		return m, errors.New("npy: data after the matrix")
	}
	m.Data = make([]float32, m.Rows*m.Dim)
	for i := range m.Data {
		v := math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return m, fmt.Errorf("npy: non-finite value at row %d", i/m.Dim)
		}
		m.Data[i] = v
	}
	return m, nil
}

// WriteNpy writes a float32 matrix in the same format, for tests and synthetic activations.
func WriteNpy(w io.Writer, m Matrix) error {
	header := fmt.Sprintf("{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", m.Rows, m.Dim)
	for (10+len(header)+1)%64 != 0 {
		header += " "
	}
	header += "\n"
	if _, err := w.Write(append([]byte("\x93NUMPY\x01\x00"), byte(len(header)), byte(len(header)>>8))); err != nil {
		return err
	}
	if _, err := io.WriteString(w, header); err != nil {
		return err
	}
	raw := make([]byte, 4*len(m.Data))
	for i, v := range m.Data {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
	}
	_, err := w.Write(raw)
	return err
}
