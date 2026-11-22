package gguf

import (
	"fmt"
	"io"
	"maps"
	"slices"
)

// Source is a GGUF whose tensor data Splice reads: its header and the whole
// file.
type Source struct {
	File *File
	Data io.ReaderAt
}

// Splice writes to w a GGUF made of base with some tensors taken from other
// files: the metadata pairs and the tensor order are base's, and each tensor
// named in take comes with its type and bytes from take[name]. Its shape
// must match base's. The layout is ggml's: the data section starts at the
// header padded to base's alignment, every tensor starts at an aligned
// offset and is followed by zero padding up to the next one, so with an
// empty take a file written by llama.cpp comes out byte for byte the same.
// Nothing is requantized: building a mixed file from files that each hold
// the whole model in one type costs one copy of the bytes. It returns the
// bytes written.
//
// The metadata is not changed, so general.file_type still names base's
// type.
func Splice(w io.Writer, base Source, take map[string]Source) (int64, error) {
	bf := base.File
	for _, name := range slices.Sorted(maps.Keys(take)) {
		if _, ok := bf.Tensor(name); !ok {
			return 0, fmt.Errorf("splice: %s is not a tensor of the base file", name)
		}
	}
	type part struct {
		src  Source
		at   int64 // absolute offset in src
		size int64
	}
	a := int64(bf.Alignment)
	if a <= 0 {
		return 0, fmt.Errorf("splice: base alignment %d", bf.Alignment)
	}
	pad := func(n int64) int64 { return (n + a - 1) / a * a }
	infos := make([]Tensor, len(bf.Tensors))
	parts := make([]part, len(bf.Tensors))
	var off int64
	for i, t := range bf.Tensors {
		src, from := base, t
		if s, ok := take[t.Name]; ok {
			dt, ok := s.File.Tensor(t.Name)
			if !ok {
				return 0, fmt.Errorf("splice: the donor has no tensor %s", t.Name)
			}
			if !slices.Equal(dt.Dims, t.Dims) {
				return 0, fmt.Errorf("splice: %s has shape %v in the donor, %v in the base", t.Name, dt.Dims, t.Dims)
			}
			src, from = s, dt
		}
		size, err := from.Bytes()
		if err != nil {
			return 0, fmt.Errorf("splice: %w", err)
		}
		infos[i] = Tensor{Name: t.Name, Dims: t.Dims, Type: from.Type, Offset: uint64(off)}
		parts[i] = part{src: src, at: src.File.FileOffset(from), size: size}
		off = pad(off + size)
	}
	h := EncodeHeader(bf.Version, uint64(len(infos)), bf.Metadata, EncodeTensorInfos(infos))
	h = append(h, make([]byte, pad(int64(len(h)))-int64(len(h)))...)
	n, err := w.Write(h)
	written := int64(n)
	if err != nil {
		return written, err
	}
	zeros := make([]byte, a)
	for i, p := range parts {
		c, err := io.Copy(w, io.NewSectionReader(p.src.Data, p.at, p.size))
		written += c
		if err != nil {
			return written, fmt.Errorf("splice: %s: %w", infos[i].Name, err)
		}
		if c != p.size {
			return written, fmt.Errorf("splice: %s: copied %d of %d bytes", infos[i].Name, c, p.size)
		}
		n, err := w.Write(zeros[:pad(p.size)-p.size])
		written += int64(n)
		if err != nil {
			return written, err
		}
	}
	return written, nil
}
