// Package gguf reads the header of a GGUF file (version 2 or 3): its
// metadata scalars and its tensor infos, with each tensor's type, shape,
// byte size and absolute offset in the file. It reads no tensor data, so a
// 33 GB checkpoint costs only its header (a few MB, mostly the tokenizer).
// KV holds the scalar values decoded; Metadata keeps every pair, arrays (the
// vocabulary) included, as stored, so that a header can be re-encoded
// (write.go).
package gguf

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// Magic is "GGUF" read as a little-endian uint32.
const Magic = 0x46554747

// DefaultAlignment applies when general.alignment is absent.
const DefaultAlignment = 32

// Type is a ggml tensor type (ggml_type).
type Type uint32

// The types whose block sizes this package knows; the ids and sizes are
// gguf-py's GGMLQuantizationType and GGML_QUANT_SIZES.
const (
	F32     Type = 0
	F16     Type = 1
	Q4_0    Type = 2
	Q4_1    Type = 3
	Q5_0    Type = 6
	Q5_1    Type = 7
	Q8_0    Type = 8
	Q8_1    Type = 9
	Q2_K    Type = 10
	Q3_K    Type = 11
	Q4_K    Type = 12
	Q5_K    Type = 13
	Q6_K    Type = 14
	Q8_K    Type = 15
	IQ2_XXS Type = 16
	IQ2_XS  Type = 17
	IQ3_XXS Type = 18
	IQ1_S   Type = 19
	IQ4_NL  Type = 20
	IQ3_S   Type = 21
	IQ2_S   Type = 22
	IQ4_XS  Type = 23
	I8      Type = 24
	I16     Type = 25
	I32     Type = 26
	I64     Type = 27
	F64     Type = 28
	IQ1_M   Type = 29
	BF16    Type = 30
	TQ1_0   Type = 34
	TQ2_0   Type = 35
	MXFP4   Type = 39
)

type typeInfo struct {
	name      string
	blockSize int64 // elements per block
	typeSize  int64 // bytes per block
}

const qk = 256 // QK_K

var types = map[Type]typeInfo{
	F32:     {"F32", 1, 4},
	F16:     {"F16", 1, 2},
	Q4_0:    {"Q4_0", 32, 2 + 16},
	Q4_1:    {"Q4_1", 32, 2 + 2 + 16},
	Q5_0:    {"Q5_0", 32, 2 + 4 + 16},
	Q5_1:    {"Q5_1", 32, 2 + 2 + 4 + 16},
	Q8_0:    {"Q8_0", 32, 2 + 32},
	Q8_1:    {"Q8_1", 32, 2 + 2 + 32},
	Q2_K:    {"Q2_K", qk, 2 + 2 + qk/16 + qk/4},
	Q3_K:    {"Q3_K", qk, 2 + qk/4 + qk/8 + 12},
	Q4_K:    {"Q4_K", qk, 2 + 2 + qk/2 + 12},
	Q5_K:    {"Q5_K", qk, 2 + 2 + qk/2 + qk/8 + 12},
	Q6_K:    {"Q6_K", qk, 2 + qk/2 + qk/4 + qk/16},
	Q8_K:    {"Q8_K", qk, 4 + qk + qk/8},
	IQ2_XXS: {"IQ2_XXS", qk, 2 + qk/4},
	IQ2_XS:  {"IQ2_XS", qk, 2 + qk/4 + qk/32},
	IQ3_XXS: {"IQ3_XXS", qk, 2 + qk/4 + qk/8},
	IQ1_S:   {"IQ1_S", qk, 2 + qk/8 + qk/16},
	IQ4_NL:  {"IQ4_NL", 32, 2 + 16},
	IQ3_S:   {"IQ3_S", qk, 2 + qk/4 + qk/8 + qk/32 + 4},
	IQ2_S:   {"IQ2_S", qk, 2 + qk/4 + qk/16},
	IQ4_XS:  {"IQ4_XS", qk, 2 + 2 + qk/2 + qk/64},
	I8:      {"I8", 1, 1},
	I16:     {"I16", 1, 2},
	I32:     {"I32", 1, 4},
	I64:     {"I64", 1, 8},
	F64:     {"F64", 1, 8},
	IQ1_M:   {"IQ1_M", qk, qk/8 + qk/16 + qk/32},
	BF16:    {"BF16", 1, 2},
	TQ1_0:   {"TQ1_0", qk, 2 + 4*13},
	TQ2_0:   {"TQ2_0", qk, 2 + 64},
	MXFP4:   {"MXFP4", 32, 1 + 16},
}

func (t Type) String() string {
	if ti, ok := types[t]; ok {
		return ti.name
	}
	return fmt.Sprintf("type%d", uint32(t))
}

// Tensor is one tensor info. Dims are ggml's ne, innermost first.
type Tensor struct {
	Name   string
	Dims   []uint64
	Type   Type
	Offset uint64 // relative to the data section
}

// Bytes is the tensor's size in the file: rows of ne[0] elements packed in
// blocks, times the other dims.
func (t Tensor) Bytes() (int64, error) {
	ti, ok := types[t.Type]
	if !ok {
		return 0, fmt.Errorf("%s: unknown tensor type %d", t.Name, uint32(t.Type))
	}
	if len(t.Dims) == 0 {
		return 0, fmt.Errorf("%s: no dims", t.Name)
	}
	if t.Dims[0]%uint64(ti.blockSize) != 0 {
		return 0, fmt.Errorf("%s: ne[0] = %d is not a multiple of %s's block of %d", t.Name, t.Dims[0], ti.name, ti.blockSize)
	}
	n := int64(t.Dims[0]) / ti.blockSize * ti.typeSize
	for _, d := range t.Dims[1:] {
		n *= int64(d)
	}
	return n, nil
}

// File is a GGUF header.
type File struct {
	Version    uint32
	Alignment  uint64
	DataOffset int64          // absolute offset of the data section
	KV         map[string]any // scalars only: uint64, int64, float64, bool, string
	Metadata   []KV           // every pair in file order, as stored
	Tensors    []Tensor
	InfoOffset int64 // absolute offset of the first tensor info
	HeaderEnd  int64 // end of the tensor infos, where the padding to DataOffset starts
}

// KV is one metadata pair as stored: Type is the value's type id (gguf_type)
// and Value its encoding after the type id; for an array that is the element
// type, the count and the elements.
type KV struct {
	Key   string
	Type  uint32
	Value []byte
}

// String returns a string value.
func (kv KV) String() (string, bool) {
	if kv.Type != tString || len(kv.Value) < 8 || binary.LittleEndian.Uint64(kv.Value) != uint64(len(kv.Value)-8) {
		return "", false
	}
	return string(kv.Value[8:]), true
}

// FileOffset is the absolute offset of a tensor's data, as llama.cpp's
// loader computes it (gguf_get_data_offset + gguf_get_tensor_offset).
func (f *File) FileOffset(t Tensor) int64 { return f.DataOffset + int64(t.Offset) }

// Tensor finds a tensor by name.
func (f *File) Tensor(name string) (Tensor, bool) {
	for _, t := range f.Tensors {
		if t.Name == name {
			return t, true
		}
	}
	return Tensor{}, false
}

// Uint returns an unsigned or non-negative signed integer value.
func (f *File) Uint(key string) (uint64, bool) {
	switch v := f.KV[key].(type) {
	case uint64:
		return v, true
	case int64:
		if v >= 0 {
			return uint64(v), true
		}
	}
	return 0, false
}

// String returns a string value.
func (f *File) String(key string) (string, bool) {
	s, ok := f.KV[key].(string)
	return s, ok
}

// Open reads the header of the GGUF file at path.
func Open(path string) (*File, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fh.Close()
	f, err := Read(fh)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return f, nil
}

// reader tracks its position and, while rec is non-nil, keeps the bytes it
// reads (a metadata value).
type reader struct {
	r   *bufio.Reader
	pos int64
	rec []byte
}

func (r *reader) read(p []byte) error {
	n, err := io.ReadFull(r.r, p)
	r.pos += int64(n)
	if r.rec != nil {
		r.rec = append(r.rec, p[:n]...)
	}
	return err
}

// maxRecord bounds a recorded array's elements, so a corrupt count fails
// instead of allocating gigabytes.
const maxRecord = 1 << 30

func (r *reader) skip(n uint64) error {
	if r.rec != nil {
		if n > maxRecord {
			return fmt.Errorf("array of %d bytes at %d", n, r.pos)
		}
		return r.read(make([]byte, n))
	}
	for n > 0 {
		k := int(min(n, 1<<30))
		d, err := r.r.Discard(k)
		r.pos += int64(d)
		if err != nil {
			return err
		}
		n -= uint64(k)
	}
	return nil
}

func (r *reader) u32() (uint32, error) {
	var b [4]byte
	err := r.read(b[:])
	return binary.LittleEndian.Uint32(b[:]), err
}

func (r *reader) u64() (uint64, error) {
	var b [8]byte
	err := r.read(b[:])
	return binary.LittleEndian.Uint64(b[:]), err
}

// maxString bounds a string's length, so a corrupt header fails instead of
// allocating gigabytes.
const maxString = 1 << 20

func (r *reader) str() (string, error) {
	n, err := r.u64()
	if err != nil {
		return "", err
	}
	if n > maxString {
		return "", fmt.Errorf("string of %d bytes at %d", n, r.pos-8)
	}
	b := make([]byte, n)
	err = r.read(b)
	return string(b), err
}

// value type ids (gguf_type)
const (
	tU8 = iota
	tI8
	tU16
	tI16
	tU32
	tI32
	tF32
	tBool
	tString
	tArray
	tU64
	tI64
	tF64
)

var scalarSize = map[uint32]int{tU8: 1, tI8: 1, tU16: 2, tI16: 2, tU32: 4, tI32: 4, tF32: 4, tBool: 1, tU64: 8, tI64: 8, tF64: 8}

// value reads one value of type vt; arrays are skipped and give nil.
func (r *reader) value(vt uint32) (any, error) {
	switch vt {
	case tString:
		return r.str()
	case tArray:
		et, err := r.u32()
		if err != nil {
			return nil, err
		}
		n, err := r.u64()
		if err != nil {
			return nil, err
		}
		if sz, ok := scalarSize[et]; ok {
			if n > maxRecord {
				return nil, fmt.Errorf("array of %d elements at %d", n, r.pos)
			}
			return nil, r.skip(n * uint64(sz))
		}
		for range n {
			if _, err := r.value(et); err != nil {
				return nil, err
			}
		}
		return nil, nil
	}
	sz, ok := scalarSize[vt]
	if !ok {
		return nil, fmt.Errorf("unknown value type %d at %d", vt, r.pos)
	}
	var b [8]byte
	if err := r.read(b[:sz]); err != nil {
		return nil, err
	}
	switch vt {
	case tU8:
		return uint64(b[0]), nil
	case tI8:
		return int64(int8(b[0])), nil
	case tU16:
		return uint64(binary.LittleEndian.Uint16(b[:])), nil
	case tI16:
		return int64(int16(binary.LittleEndian.Uint16(b[:]))), nil
	case tU32:
		return uint64(binary.LittleEndian.Uint32(b[:])), nil
	case tI32:
		return int64(int32(binary.LittleEndian.Uint32(b[:]))), nil
	case tF32:
		return float64(math.Float32frombits(binary.LittleEndian.Uint32(b[:]))), nil
	case tBool:
		return b[0] != 0, nil
	case tU64:
		return binary.LittleEndian.Uint64(b[:]), nil
	case tI64:
		return int64(binary.LittleEndian.Uint64(b[:])), nil
	default: // tF64
		return math.Float64frombits(binary.LittleEndian.Uint64(b[:])), nil
	}
}

// Read parses a GGUF header from the start of r.
func Read(rd io.Reader) (*File, error) {
	r := &reader{r: bufio.NewReaderSize(rd, 1<<20)}
	magic, err := r.u32()
	if err != nil {
		return nil, err
	}
	if magic != Magic {
		return nil, fmt.Errorf("not a GGUF file (magic %#x)", magic)
	}
	f := &File{KV: map[string]any{}, Alignment: DefaultAlignment}
	if f.Version, err = r.u32(); err != nil {
		return nil, err
	}
	if f.Version < 2 || f.Version > 3 {
		return nil, fmt.Errorf("GGUF version %d, want 2 or 3", f.Version)
	}
	nTensors, err := r.u64()
	if err != nil {
		return nil, err
	}
	nKV, err := r.u64()
	if err != nil {
		return nil, err
	}
	for range nKV {
		key, err := r.str()
		if err != nil {
			return nil, fmt.Errorf("metadata key: %w", err)
		}
		vt, err := r.u32()
		if err != nil {
			return nil, err
		}
		r.rec = []byte{}
		v, err := r.value(vt)
		raw := r.rec
		r.rec = nil
		if err != nil {
			return nil, fmt.Errorf("metadata %s: %w", key, err)
		}
		if v != nil {
			f.KV[key] = v
		}
		f.Metadata = append(f.Metadata, KV{Key: key, Type: vt, Value: raw})
	}
	if a, ok := f.Uint("general.alignment"); ok {
		if a == 0 || a&(a-1) != 0 {
			return nil, fmt.Errorf("general.alignment %d is not a power of 2", a)
		}
		f.Alignment = a
	}
	f.InfoOffset = r.pos
	for range nTensors {
		var t Tensor
		if t.Name, err = r.str(); err != nil {
			return nil, fmt.Errorf("tensor name: %w", err)
		}
		nd, err := r.u32()
		if err != nil {
			return nil, err
		}
		if nd == 0 || nd > 4 {
			return nil, fmt.Errorf("tensor %s: %d dims", t.Name, nd)
		}
		t.Dims = make([]uint64, nd)
		for i := range t.Dims {
			if t.Dims[i], err = r.u64(); err != nil {
				return nil, err
			}
		}
		tt, err := r.u32()
		if err != nil {
			return nil, err
		}
		t.Type = Type(tt)
		if t.Offset, err = r.u64(); err != nil {
			return nil, err
		}
		if t.Offset%f.Alignment != 0 {
			return nil, fmt.Errorf("tensor %s: offset %d is not aligned to %d", t.Name, t.Offset, f.Alignment)
		}
		f.Tensors = append(f.Tensors, t)
	}
	f.HeaderEnd = r.pos
	a := int64(f.Alignment)
	f.DataOffset = (r.pos + a - 1) / a * a
	return f, nil
}
