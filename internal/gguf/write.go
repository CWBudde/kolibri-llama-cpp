package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"maps"
	"math"
	"slices"
)

// enc appends little-endian values to a byte slice.
type enc []byte

func (e *enc) u8(v uint8)   { *e = append(*e, v) }
func (e *enc) u32(v uint32) { *e = binary.LittleEndian.AppendUint32(*e, v) }
func (e *enc) u64(v uint64) { *e = binary.LittleEndian.AppendUint64(*e, v) }
func (e *enc) str(s string) { e.u64(uint64(len(s))); *e = append(*e, s...) }

func (e *enc) bool(v bool) {
	if v {
		e.u8(1)
	} else {
		e.u8(0)
	}
}

// arr appends an array's element type and count.
func (e *enc) arr(et uint32, n int) { e.u32(et); e.u64(uint64(n)) }

// NewKV encodes a metadata pair from a Go value: uint8, uint32, int32,
// uint64, int64, float32, float64, bool, string, or a slice of string, bool,
// int32, uint32 or float32.
func NewKV(key string, v any) (KV, error) {
	var e enc
	var vt uint32
	switch v := v.(type) {
	case uint8:
		vt = tU8
		e.u8(v)
	case uint32:
		vt = tU32
		e.u32(v)
	case int32:
		vt = tI32
		e.u32(uint32(v))
	case uint64:
		vt = tU64
		e.u64(v)
	case int64:
		vt = tI64
		e.u64(uint64(v))
	case float32:
		vt = tF32
		e.u32(math.Float32bits(v))
	case float64:
		vt = tF64
		e.u64(math.Float64bits(v))
	case bool:
		vt = tBool
		e.bool(v)
	case string:
		vt = tString
		e.str(v)
	case []string:
		vt = tArray
		e.arr(tString, len(v))
		for _, s := range v {
			e.str(s)
		}
	case []bool:
		vt = tArray
		e.arr(tBool, len(v))
		for _, b := range v {
			e.bool(b)
		}
	case []int32:
		vt = tArray
		e.arr(tI32, len(v))
		for _, x := range v {
			e.u32(uint32(x))
		}
	case []uint32:
		vt = tArray
		e.arr(tU32, len(v))
		for _, x := range v {
			e.u32(x)
		}
	case []float32:
		vt = tArray
		e.arr(tF32, len(v))
		for _, x := range v {
			e.u32(math.Float32bits(x))
		}
	default:
		return KV{}, fmt.Errorf("metadata %s: unsupported value %T", key, v)
	}
	return KV{Key: key, Type: vt, Value: e}, nil
}

// EncodeTensorInfos encodes tensor infos as a GGUF header stores them.
func EncodeTensorInfos(tensors []Tensor) []byte {
	var e enc
	for _, t := range tensors {
		e.str(t.Name)
		e.u32(uint32(len(t.Dims)))
		for _, d := range t.Dims {
			e.u64(d)
		}
		e.u32(uint32(t.Type))
		e.u64(t.Offset)
	}
	return e
}

// EncodeHeader encodes a GGUF header up to the end of its tensor infos,
// without the padding to the data section: magic, version, the counts, the
// metadata pairs in the given order and nTensors tensor infos already
// encoded (EncodeTensorInfos, or bytes copied from another header).
func EncodeHeader(version uint32, nTensors uint64, kvs []KV, infos []byte) []byte {
	var e enc
	e.u32(Magic)
	e.u32(version)
	e.u64(nTensors)
	e.u64(uint64(len(kvs)))
	for _, kv := range kvs {
		e.str(kv.Key)
		e.u32(kv.Type)
		e = append(e, kv.Value...)
	}
	return append(e, infos...)
}

// WriteHeader writes a GGUF v3 header for tests and fixtures: kv in sorted
// key order (any value NewKV takes), then the tensor infos, then zero
// padding up to the data section, whose offset it returns. The tensors'
// data is the caller's.
func WriteHeader(w io.Writer, alignment uint64, kv map[string]any, tensors []Tensor) (int64, error) {
	var kvs []KV
	for _, k := range slices.Sorted(maps.Keys(kv)) {
		p, err := NewKV(k, kv[k])
		if err != nil {
			return 0, err
		}
		kvs = append(kvs, p)
	}
	h := EncodeHeader(3, uint64(len(tensors)), kvs, EncodeTensorInfos(tensors))
	a := int64(alignment)
	data := (int64(len(h)) + a - 1) / a * a
	if _, err := w.Write(append(h, make([]byte, data-int64(len(h)))...)); err != nil {
		return 0, err
	}
	return data, nil
}
