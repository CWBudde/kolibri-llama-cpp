package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
)

// WriteTensor is one tensor to serialize: its header entry plus the raw
// little-endian payload.
type WriteTensor struct {
	Name  string
	DType string
	Shape []int64
	Data  []byte
}

// Write serializes tensors as a safetensors file. Tensors are laid out in
// name order; the header is padded with spaces so the data section starts on
// an 8-byte boundary, as the reference implementation does.
func Write(w io.Writer, tensors []WriteTensor, metadata map[string]string) error {
	sorted := slices.Clone(tensors)
	slices.SortFunc(sorted, func(a, b WriteTensor) int { return strings.Compare(a.Name, b.Name) })

	entries := make(map[string]any, len(sorted)+1)
	if metadata != nil {
		entries["__metadata__"] = metadata
	}
	var errs []error
	var off int64
	for i, t := range sorted {
		if t.Name == "__metadata__" {
			errs = append(errs, fmt.Errorf("%s: reserved name", t.Name))
			continue
		}
		if i > 0 && sorted[i-1].Name == t.Name {
			errs = append(errs, fmt.Errorf("%s: duplicate name", t.Name))
			continue
		}
		if slices.ContainsFunc(t.Shape, func(d int64) bool { return d < 0 }) {
			errs = append(errs, fmt.Errorf("%s: negative dimension in %v", t.Name, t.Shape))
			continue
		}
		hdr := Tensor{DType: t.DType, Shape: t.Shape, Offsets: [2]int64{off, off + int64(len(t.Data))}}
		bits, ok := dtypeBits[t.DType]
		if !ok {
			errs = append(errs, fmt.Errorf("%s: unknown dtype %q", t.Name, t.DType))
			continue
		}
		if want := (hdr.NumElements()*int64(bits) + 7) / 8; int64(len(t.Data)) != want {
			errs = append(errs, fmt.Errorf("%s: %d bytes for %s%v, want %d", t.Name, len(t.Data), t.DType, t.Shape, want))
			continue
		}
		entries[t.Name] = map[string]any{"dtype": hdr.DType, "shape": hdr.Shape, "data_offsets": hdr.Offsets}
		off = hdr.Offsets[1]
	}
	if err := errors.Join(errs...); err != nil {
		return fmt.Errorf("safetensors: %w", err)
	}

	raw, err := json.Marshal(entries)
	if err != nil {
		return fmt.Errorf("safetensors: encode header: %w", err)
	}
	if pad := (8 - len(raw)%8) % 8; pad > 0 {
		raw = append(raw, bytes.Repeat([]byte{' '}, pad)...)
	}
	if err := binary.Write(w, binary.LittleEndian, uint64(len(raw))); err != nil {
		return err
	}
	if _, err := w.Write(raw); err != nil {
		return err
	}
	for _, t := range sorted {
		if _, err := w.Write(t.Data); err != nil {
			return err
		}
	}
	return nil
}
