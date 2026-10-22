package gguf

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"slices"
)

// WriteHeader writes a GGUF v3 header for tests and fixtures: kv in sorted
// key order (uint32, uint64, int32, float32, bool, string or []string
// values), then the tensor infos, then zero padding up to the data section,
// whose offset it returns. The tensors' data is the caller's.
func WriteHeader(w io.Writer, alignment uint64, kv map[string]any, tensors []Tensor) (int64, error) {
	bw := bufio.NewWriter(w)
	var pos int64
	put := func(v any) {
		_ = binary.Write(bw, binary.LittleEndian, v)
		pos += int64(binary.Size(v))
	}
	str := func(s string) {
		put(uint64(len(s)))
		_, _ = bw.WriteString(s)
		pos += int64(len(s))
	}
	put(uint32(Magic))
	put(uint32(3))
	put(uint64(len(tensors)))
	put(uint64(len(kv)))
	keys := make([]string, 0, len(kv))
	for k := range kv {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	for _, k := range keys {
		str(k)
		switch v := kv[k].(type) {
		case uint32:
			put(uint32(tU32))
			put(v)
		case uint64:
			put(uint32(tU64))
			put(v)
		case int32:
			put(uint32(tI32))
			put(v)
		case float32:
			put(uint32(tF32))
			put(math.Float32bits(v))
		case bool:
			put(uint32(tBool))
			b := uint8(0)
			if v {
				b = 1
			}
			put(b)
		case string:
			put(uint32(tString))
			str(v)
		case []string:
			put(uint32(tArray))
			put(uint32(tString))
			put(uint64(len(v)))
			for _, s := range v {
				str(s)
			}
		default:
			return 0, fmt.Errorf("metadata %s: unsupported value %T", k, v)
		}
	}
	for _, t := range tensors {
		str(t.Name)
		put(uint32(len(t.Dims)))
		for _, d := range t.Dims {
			put(d)
		}
		put(uint32(t.Type))
		put(t.Offset)
	}
	a := int64(alignment)
	data := (pos + a - 1) / a * a
	if _, err := bw.Write(make([]byte, data-pos)); err != nil {
		return 0, err
	}
	return data, bw.Flush()
}
