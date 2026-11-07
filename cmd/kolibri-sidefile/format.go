package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"os"

	"kolibri-llm/internal/gguf"
)

// The side-file, little endian. llama_expert_sidefile (patch 0014) reads the
// same layout; a change here is a new version on both sides.
//
//	 0  char[8]   magic "KOLIEXPT"
//	 8  u32       version
//	12  u32       alignment of the extents
//	16  u32       layers
//	20  u32       experts per layer
//	24  u64       table bytes: this header, the tensor entries and the extent offsets
//	32  u64       file size
//	40  u64       GGUF size
//	48  u64       GGUF header bytes: everything before its data section
//	56  byte[32]  sha256 of those GGUF header bytes
//	88  byte[32]  sha256 of the table, with these 32 bytes zero
//	120 byte[8]   zero
//	128 layers*3 tensor entries, gate, up, down of each layer:
//	      u64 absolute offset of expert 0 in the GGUF, u64 row bytes (one
//	      expert), u32 ggml type, u32 zero
//	then layers*experts extent offsets, u64, absolute in the side-file
//
// The table is padded with zeros to the alignment. An expert's extent holds
// its gate, up and down rows, contiguous, as the GGUF has them, and zeros to
// the alignment. Reading the padded extent keeps one miss one aligned read.
const (
	magic        = "KOLIEXPT"
	version      = 1
	headerBytes  = 128
	tensorBytes  = 24
	checksumAt   = 88
	checksumEnd  = checksumAt + sha256.Size
	kindsPerRow  = len(gguf.ExpertKinds)
	manifestName = ".manifest.json"
)

// alignment of the extents, 16 KiB:
//   - An aligned read is what F_NOCACHE's direct path wants. As far as XNU's
//     cluster layer goes, a read whose file offset or destination is not aligned
//     to the device block is copied through the cache. That is from its source
//     as remembered, not measured here; Stage E3's benchmark measures the result.
//   - 16 KiB is the VM page on Apple silicon and a multiple of the SSD's 4 KiB
//     block. So a boundary page never holds two experts: with buffered reads an
//     expert would otherwise drag in its neighbour's page, and mincore's residency
//     stays per expert.
//   - The slots cannot be the read's target. Their rows are no multiples of the
//     block (IQ3's gate rows are 122.5 blocks of 4 KiB), so a slot's address is
//     usually misaligned, and an expert's three rows sit in three tensors. So the
//     read goes to a page-aligned staging buffer per I/O thread, and three copies
//     take the rows out of it.
//
// The cost in file size is under 16 KiB per expert: 0.24% at IQ3.
const alignment = 16 << 10

func pad(n int64) int64 { return (n + alignment - 1) / alignment * alignment }

// plan is where a GGUF's experts go in the side-file.
type plan struct {
	Layers       [][3]gguf.ExpertTensor
	Experts      int
	GGUFSize     int64
	GGUFHeader   int64 // bytes before the data section
	GGUFHeaderSH [sha256.Size]byte
	TableBytes   int64
	DataStart    int64
	FileSize     int64
	Offsets      [][]int64 // per layer and expert: its extent's offset
}

// rows is one expert's bytes in layer il: gate, up and down.
func (p *plan) rows(il int) int64 {
	var n int64
	for _, t := range p.Layers[il] {
		n += t.Slice
	}
	return n
}

// extent is the bytes one expert of layer il takes in the file, padding included.
func (p *plan) extent(il int) int64 { return pad(p.rows(il)) }

// makePlan lays the experts out in layer and expert order behind the table.
func makePlan(f *gguf.File, size int64, header []byte) (*plan, error) {
	if int64(len(header)) != f.DataOffset {
		return nil, fmt.Errorf("GGUF header of %d bytes, data section at %d", len(header), f.DataOffset)
	}
	_, ne, layers, err := f.ExpertLayers()
	if err != nil {
		return nil, err
	}
	p := &plan{Layers: layers, Experts: ne, GGUFSize: size, GGUFHeader: f.DataOffset, GGUFHeaderSH: sha256.Sum256(header)}
	p.TableBytes = headerBytes + int64(len(layers))*int64(kindsPerRow)*tensorBytes + int64(len(layers))*int64(ne)*8
	p.DataStart = pad(p.TableBytes)
	off := p.DataStart
	for il, lt := range layers {
		for k, t := range lt {
			if t.Offset+int64(ne)*t.Slice > size {
				return nil, fmt.Errorf("layer %d %s: ends at %d, past the GGUF's %d bytes", il, gguf.ExpertKinds[k], t.Offset+int64(ne)*t.Slice, size)
			}
		}
		rows := make([]int64, ne)
		for e := range ne {
			rows[e] = off
			off += p.extent(il)
		}
		p.Offsets = append(p.Offsets, rows)
	}
	p.FileSize = off
	return p, nil
}

// table encodes the header and the tables, with the checksum set and without
// the padding.
func (p *plan) table() []byte {
	b := make([]byte, 0, p.TableBytes)
	b = append(b, magic...)
	b = binary.LittleEndian.AppendUint32(b, version)
	b = binary.LittleEndian.AppendUint32(b, alignment)
	b = binary.LittleEndian.AppendUint32(b, uint32(len(p.Layers)))
	b = binary.LittleEndian.AppendUint32(b, uint32(p.Experts))
	b = binary.LittleEndian.AppendUint64(b, uint64(p.TableBytes))
	b = binary.LittleEndian.AppendUint64(b, uint64(p.FileSize))
	b = binary.LittleEndian.AppendUint64(b, uint64(p.GGUFSize))
	b = binary.LittleEndian.AppendUint64(b, uint64(p.GGUFHeader))
	b = append(b, p.GGUFHeaderSH[:]...)
	b = append(b, make([]byte, sha256.Size)...) // the checksum, below
	b = append(b, make([]byte, 8)...)
	for _, lt := range p.Layers {
		for _, t := range lt {
			b = binary.LittleEndian.AppendUint64(b, uint64(t.Offset))
			b = binary.LittleEndian.AppendUint64(b, uint64(t.Slice))
			b = binary.LittleEndian.AppendUint32(b, uint32(t.TypeID))
			b = binary.LittleEndian.AppendUint32(b, 0)
		}
	}
	for _, rows := range p.Offsets {
		for _, off := range rows {
			b = binary.LittleEndian.AppendUint64(b, uint64(off))
		}
	}
	sum := sha256.Sum256(b)
	copy(b[checksumAt:checksumEnd], sum[:])
	return b
}

// manifestTensor is one row of the manifest's tensor table.
type manifestTensor struct {
	Kind       string `json:"kind"`
	Type       string `json:"type"`
	GGUFOffset int64  `json:"gguf_offset"`
	RowBytes   int64  `json:"row_bytes"`
}

// manifest is the JSON file next to the side-file: what kolibri-sidefile
// verify checks every byte against. The loader does not read it; it checks the
// table in the side-file, which holds no data hashes.
type manifest struct {
	Format      string             `json:"format"`
	Version     int                `json:"version"`
	Alignment   int                `json:"alignment"`
	Layers      int                `json:"layers"`
	Experts     int                `json:"experts"`
	Size        int64              `json:"size"`
	SHA256      string             `json:"sha256"`
	TableSHA256 string             `json:"table_sha256"`
	GGUF        manifestGGUF       `json:"gguf"`
	Tensors     [][]manifestTensor `json:"tensors"`
	Extents     [][]string         `json:"extent_sha256"`
}

// manifestGGUF is the GGUF's identity.
type manifestGGUF struct {
	Size         int64  `json:"size"`
	HeaderBytes  int64  `json:"header_bytes"`
	HeaderSHA256 string `json:"header_sha256"`
}

// manifestFor is the part of the manifest a plan fixes; the file's and the
// extents' hashes come from the data.
func (p *plan) manifestFor() manifest {
	m := manifest{Format: magic, Version: version, Alignment: alignment, Layers: len(p.Layers), Experts: p.Experts,
		Size: p.FileSize, GGUF: manifestGGUF{p.GGUFSize, p.GGUFHeader, hex.EncodeToString(p.GGUFHeaderSH[:])}}
	sum := sha256.Sum256(p.table())
	m.TableSHA256 = hex.EncodeToString(sum[:])
	for _, lt := range p.Layers {
		var row []manifestTensor
		for k, t := range lt {
			row = append(row, manifestTensor{gguf.ExpertKinds[k], t.Type, t.Offset, t.Slice})
		}
		m.Tensors = append(m.Tensors, row)
	}
	return m
}

// planFor loads a GGUF's header and plans its side-file.
func planFor(path string) (*plan, error) {
	f, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	in, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer in.Close()
	st, err := in.Stat()
	if err != nil {
		return nil, err
	}
	if st.Size() < f.DataOffset {
		return nil, errors.New("GGUF shorter than its header")
	}
	header := make([]byte, f.DataOffset)
	if _, err := in.ReadAt(header, 0); err != nil {
		return nil, err
	}
	return makePlan(f, st.Size(), header)
}
