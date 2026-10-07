package locality

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// Selections are the experts a workload selects, as tools/locality/experts.py writes them:
// [Layers][Tokens][K] expert IDs in C order.
type Selections struct {
	Layers, Tokens, K int
	IDs               []int32
}

// Layer returns one layer's selections, one slice of K expert IDs per token.
func (s Selections) Layer(l int) [][]int {
	out := make([][]int, s.Tokens)
	flat := make([]int, s.Tokens*s.K)
	for i := range flat {
		flat[i] = int(s.IDs[(l*s.Tokens)*s.K+i])
	}
	for t := range out {
		out[t] = flat[t*s.K : (t+1)*s.K]
	}
	return out
}

// maxIDs bounds the IDs a file may declare (8 GiB as int32); the workloads hold under 10 million.
const maxIDs = 1 << 31

var (
	descrRe   = regexp.MustCompile(`'descr':\s*'([^']*)'`)
	fortranRe = regexp.MustCompile(`'fortran_order':\s*(True|False)`)
	shapeRe   = regexp.MustCompile(`'shape':\s*\(([^)]*)\)`)
)

// ReadNPY reads a 3-D little-endian int16 or int32 array in C order from a .npy file
// (format versions 1 to 3).
func ReadNPY(r io.Reader) (Selections, error) {
	var s Selections
	dims, ids, err := readInts(r, 3)
	if err != nil {
		return s, err
	}
	s.Layers, s.Tokens, s.K = dims[0], dims[1], dims[2]
	s.IDs = ids
	return s, nil
}

// ReadTokens reads a 1-D little-endian int16 or int32 array, such as a workload's token IDs
// (<workload>.tokens.npy of tools/locality/workloads.py), from a .npy file.
func ReadTokens(r io.Reader) ([]int32, error) {
	_, ids, err := readInts(r, 1)
	return ids, err
}

// readInts reads a .npy file's integer array of rank dimensions, in C order.
func readInts(r io.Reader, rank int) ([]int, []int32, error) {
	br := bufio.NewReader(r)
	magic := make([]byte, 8)
	if _, err := io.ReadFull(br, magic); err != nil {
		return nil, nil, fmt.Errorf("npy: %w", err)
	}
	if string(magic[:6]) != "\x93NUMPY" {
		return nil, nil, errors.New("npy: not a .npy file")
	}
	var n int
	switch magic[6] {
	case 1:
		var n16 uint16
		if err := binary.Read(br, binary.LittleEndian, &n16); err != nil {
			return nil, nil, fmt.Errorf("npy: %w", err)
		}
		n = int(n16)
	case 2, 3:
		var n32 uint32
		if err := binary.Read(br, binary.LittleEndian, &n32); err != nil {
			return nil, nil, fmt.Errorf("npy: %w", err)
		}
		n = int(n32)
	default:
		return nil, nil, fmt.Errorf("npy: format version %d", magic[6])
	}
	hdr := make([]byte, n)
	if _, err := io.ReadFull(br, hdr); err != nil {
		return nil, nil, fmt.Errorf("npy header: %w", err)
	}
	h := string(hdr)
	descr, fortran, shape := descrRe.FindStringSubmatch(h), fortranRe.FindStringSubmatch(h), shapeRe.FindStringSubmatch(h)
	if descr == nil || fortran == nil || shape == nil {
		return nil, nil, fmt.Errorf("npy: header %q", h)
	}
	if fortran[1] != "False" {
		return nil, nil, errors.New("npy: Fortran order")
	}
	var dims []int
	for f := range strings.SplitSeq(shape[1], ",") {
		if f = strings.TrimSpace(f); f == "" {
			continue
		}
		d, err := strconv.Atoi(f)
		if err != nil {
			return nil, nil, fmt.Errorf("npy: shape %q", shape[1])
		}
		dims = append(dims, d)
	}
	if len(dims) != rank {
		return nil, nil, fmt.Errorf("npy: shape (%s), want %d dimensions", shape[1], rank)
	}
	// the shape comes from the file: check it before it sizes an allocation
	size := 1
	for _, d := range dims {
		if d <= 0 || size > maxIDs/d {
			return nil, nil, fmt.Errorf("npy: shape (%s), want positive dimensions of at most %d IDs", shape[1], maxIDs)
		}
		size *= d
	}
	ids := make([]int32, size)
	switch descr[1] {
	case "<i2":
		ids16 := make([]int16, len(ids))
		if err := binary.Read(br, binary.LittleEndian, ids16); err != nil {
			return nil, nil, fmt.Errorf("npy data: %w", err)
		}
		for i, id := range ids16 {
			ids[i] = int32(id)
		}
	case "<i4":
		if err := binary.Read(br, binary.LittleEndian, ids); err != nil {
			return nil, nil, fmt.Errorf("npy data: %w", err)
		}
	default:
		return nil, nil, fmt.Errorf("npy: dtype %s, want <i2 or <i4", descr[1])
	}
	return dims, ids, nil
}

// ReadNPYFile reads a .npy file with ReadNPY.
func ReadNPYFile(path string) (Selections, error) {
	f, err := os.Open(path)
	if err != nil {
		return Selections{}, err
	}
	defer f.Close()
	s, err := ReadNPY(f)
	if err != nil {
		return s, fmt.Errorf("%s: %w", path, err)
	}
	return s, nil
}
