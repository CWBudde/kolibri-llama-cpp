// Package safetensors parses and validates safetensors file headers without
// touching tensor payloads.
//
// File layout: 8-byte little-endian header length N, N bytes of JSON header,
// then the raw tensor data. Offsets in the header are relative to the start
// of the data section (byte 8+N of the file).
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
)

// MaxHeaderSize guards against corrupt length prefixes. The spec caps headers
// at 100 MB.
const MaxHeaderSize = 100 << 20

// dtypeBits maps safetensors dtype names to their element width in bits.
var dtypeBits = map[string]int{
	"BOOL": 8, "U8": 8, "I8": 8, "F8_E4M3": 8, "F8_E5M2": 8, "F8_E8M0": 8,
	"I16": 16, "U16": 16, "F16": 16, "BF16": 16,
	"I32": 32, "U32": 32, "F32": 32,
	"I64": 64, "U64": 64, "F64": 64,
	"F4": 4, "F6_E2M3": 6, "F6_E3M2": 6,
}

// Tensor describes one tensor entry of a safetensors header.
type Tensor struct {
	Name    string   `json:"name"`
	DType   string   `json:"dtype"`
	Shape   []int64  `json:"shape"`
	Offsets [2]int64 `json:"data_offsets"`
}

// NumElements returns the product of the shape (1 for scalars).
func (t Tensor) NumElements() int64 {
	n := int64(1)
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// NumBytes returns the payload length according to the data offsets.
func (t Tensor) NumBytes() int64 { return t.Offsets[1] - t.Offsets[0] }

// Header is a parsed safetensors header.
type Header struct {
	// Size is the JSON header length N from the 8-byte prefix.
	Size     int64
	Metadata map[string]string
	// Tensors is sorted by data offset.
	Tensors []Tensor
}

// DataSize returns the length of the data section implied by the header.
func (h *Header) DataSize() int64 {
	if len(h.Tensors) == 0 {
		return 0
	}
	return h.Tensors[len(h.Tensors)-1].Offsets[1]
}

// FileSize returns the total file size implied by the header.
func (h *Header) FileSize() int64 { return 8 + h.Size + h.DataSize() }

// ReadHeaderSize decodes the 8-byte length prefix.
func ReadHeaderSize(prefix []byte) (int64, error) {
	if len(prefix) < 8 {
		return 0, fmt.Errorf("safetensors: prefix too short (%d bytes)", len(prefix))
	}
	n := binary.LittleEndian.Uint64(prefix[:8])
	if n == 0 || n > MaxHeaderSize {
		return 0, fmt.Errorf("safetensors: implausible header size %d", n)
	}
	return int64(n), nil
}

// ParseHeader decodes the JSON header (without the length prefix) and
// validates it.
func ParseHeader(raw []byte) (*Header, error) {
	var entries map[string]json.RawMessage
	if err := json.Unmarshal(raw, &entries); err != nil {
		return nil, fmt.Errorf("safetensors: decode header: %w", err)
	}
	h := &Header{Size: int64(len(raw))}
	for name, msg := range entries {
		if name == "__metadata__" {
			if err := json.Unmarshal(msg, &h.Metadata); err != nil {
				return nil, fmt.Errorf("safetensors: decode __metadata__: %w", err)
			}
			continue
		}
		t := Tensor{Name: name}
		if err := json.Unmarshal(msg, &t); err != nil {
			return nil, fmt.Errorf("safetensors: decode tensor %q: %w", name, err)
		}
		h.Tensors = append(h.Tensors, t)
	}
	sort.Slice(h.Tensors, func(i, j int) bool {
		a, b := h.Tensors[i].Offsets, h.Tensors[j].Offsets
		if a[0] != b[0] {
			return a[0] < b[0]
		}
		return a[1] < b[1]
	})
	if err := h.validate(); err != nil {
		return nil, err
	}
	return h, nil
}

// validate checks dtypes, payload lengths, and that the tensors tile the data
// section without gaps or overlaps.
func (h *Header) validate() error {
	var errs []error
	var next int64
	for _, t := range h.Tensors {
		bits, ok := dtypeBits[t.DType]
		if !ok {
			errs = append(errs, fmt.Errorf("%s: unknown dtype %q", t.Name, t.DType))
			continue
		}
		for _, d := range t.Shape {
			if d < 0 {
				errs = append(errs, fmt.Errorf("%s: negative dimension in %v", t.Name, t.Shape))
			}
		}
		if want := (t.NumElements()*int64(bits) + 7) / 8; t.NumBytes() != want {
			errs = append(errs, fmt.Errorf("%s: %d bytes for %s%v, want %d", t.Name, t.NumBytes(), t.DType, t.Shape, want))
		}
		if t.Offsets[0] != next {
			errs = append(errs, fmt.Errorf("%s: starts at %d, previous tensor ends at %d", t.Name, t.Offsets[0], next))
		}
		next = t.Offsets[1]
	}
	return errors.Join(errs...)
}

// ReadHeader reads and parses the header from r.
func ReadHeader(r io.ReaderAt) (*Header, error) {
	prefix := make([]byte, 8)
	if _, err := r.ReadAt(prefix, 0); err != nil {
		return nil, fmt.Errorf("safetensors: read prefix: %w", err)
	}
	n, err := ReadHeaderSize(prefix)
	if err != nil {
		return nil, err
	}
	raw := make([]byte, n)
	if _, err := r.ReadAt(raw, 8); err != nil {
		return nil, fmt.Errorf("safetensors: read header: %w", err)
	}
	return ParseHeader(raw)
}
