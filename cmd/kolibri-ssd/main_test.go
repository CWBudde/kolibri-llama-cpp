package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"

	"kolibri-llm/internal/gguf"
)

// The synthetic checkpoint: 3 layers of 4 experts, gate and up IQ3_XXS
// [512, 8, 4], down IQ4_XS [256, 16, 4], with a small F32 tensor between
// the layers so that each layer's experts are one extent.
const (
	tLayers  = 3
	tExperts = 4
	// gate/up: 512/256 blocks * 98 bytes * 8 rows = 1568 per expert
	tGateSlice = 512 / 256 * 98 * 8
	// down: 256/256 * 136 * 16 = 2176 per expert
	tDownSlice = 136 * 16
)

// writeFixture writes the synthetic GGUF, its data filled with a pattern
// (byte i of the file is i*7 mod 251), and returns its path and header.
func writeFixture(t *testing.T) (string, *gguf.File) {
	t.Helper()
	var tensors []gguf.Tensor
	var off uint64
	add := func(name string, dims []uint64, typ gguf.Type) {
		tt := gguf.Tensor{Name: name, Dims: dims, Type: typ, Offset: off}
		n, err := tt.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		tensors = append(tensors, tt)
		off += (uint64(n) + 31) / 32 * 32
	}
	for il := range tLayers {
		// down first, as in the real file
		add(fmt.Sprintf("blk.%d.ffn_down_exps.weight", il), []uint64{256, 16, tExperts}, gguf.IQ4_XS)
		add(fmt.Sprintf("blk.%d.ffn_gate_exps.weight", il), []uint64{512, 8, tExperts}, gguf.IQ3_XXS)
		add(fmt.Sprintf("blk.%d.ffn_up_exps.weight", il), []uint64{512, 8, tExperts}, gguf.IQ3_XXS)
		add(fmt.Sprintf("blk.%d.attn_norm.weight", il), []uint64{40}, gguf.F32)
	}
	kv := map[string]any{
		"general.architecture":  "kolibri",
		"kolibri.block_count":   uint32(tLayers),
		"kolibri.expert_count":  uint32(tExperts),
		"tokenizer.ggml.tokens": []string{"x", "y"},
	}
	var buf bytes.Buffer
	data, err := gguf.WriteHeader(&buf, 32, kv, tensors)
	if err != nil {
		t.Fatal(err)
	}
	total := data + int64(off)
	b := make([]byte, total)
	copy(b, buf.Bytes())
	for i := data; i < total; i++ {
		b[i] = byte(i * 7 % 251)
	}
	path := filepath.Join(t.TempDir(), "fixture.gguf")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	return path, f
}

// TestExpertLayout checks expert e's slice offsets against the tensors'
// offsets as llama.cpp's loader computes them (data section + tensor
// offset) plus e times nb[2], as patch 0011's load does.
func TestExpertLayout(t *testing.T) {
	_, f := writeFixture(t)
	l, err := expertLayout(f)
	if err != nil {
		t.Fatal(err)
	}
	if len(l.Layers) != tLayers || l.Experts != tExperts {
		t.Fatalf("%d layers x %d experts", len(l.Layers), l.Experts)
	}
	for il := range tLayers {
		for k, kind := range kinds {
			tt, _ := f.Tensor(fmt.Sprintf("blk.%d.ffn_%s_exps.weight", il, kind))
			n, _ := tt.Bytes()
			for e := range tExperts {
				want := f.DataOffset + int64(tt.Offset) + int64(e)*n/tExperts
				if got := l.sliceOffset(il, k, e); got != want {
					t.Errorf("layer %d %s expert %d at %d, want %d", il, kind, e, got, want)
				}
			}
		}
		if got := l.expertBytes(il); got != 2*tGateSlice+tDownSlice {
			t.Errorf("expert bytes %d", got)
		}
	}
	if got := l.sliceSizes(); !slices.Equal(got, []int64{tGateSlice, tDownSlice}) {
		t.Errorf("slice sizes %v", got)
	}
	// down, gate and up are adjacent in each layer, the norm splits layers
	if len(l.Extents) != tLayers {
		t.Errorf("%d extents, want %d: %v", len(l.Extents), tLayers, l.Extents)
	}
	if l.extentBytes() != tLayers*tExperts*(2*tGateSlice+tDownSlice) {
		t.Errorf("extent bytes %d", l.extentBytes())
	}

	f.Tensors[1].Dims = []uint64{512, 8, 3}
	if _, err := expertLayout(f); err == nil {
		t.Error("a stacked tensor with 3 of 4 experts accepted")
	}
}

func TestSampleIndices(t *testing.T) {
	a, err := sampleIndices(1000, 1000, 7)
	if err != nil {
		t.Fatal(err)
	}
	s := slices.Clone(a)
	slices.Sort(s)
	for i, v := range s {
		if v != int64(i) {
			t.Fatalf("a full draw is not a permutation: %v...", s[:10])
		}
	}
	b, _ := sampleIndices(1000, 1000, 7)
	if !slices.Equal(a, b) {
		t.Error("same seed, different draw")
	}
	c, _ := sampleIndices(1000, 1000, 8)
	if slices.Equal(a, c) {
		t.Error("different seeds, same draw")
	}
	if _, err := sampleIndices(10, 11, 1); err == nil {
		t.Error("11 distinct of 10 accepted")
	}
}

// TestRandomOffsets: aligned, inside one extent, distinct and
// non-overlapping, deterministic per seed.
func TestRandomOffsets(t *testing.T) {
	ext := []extent{{100, 100 + 10<<20}, {20 << 20, 20<<20 + 1<<20}, {30<<20 + 5, 30<<20 + 5 + 3<<20}}
	const align = 16 << 10
	for _, chunk := range []int64{64 << 10, 501_760, 1 << 20} {
		g := newGrid(ext, chunk, align)
		n := int(g.total)
		offs, err := randomOffsets(ext, chunk, align, n, 42)
		if err != nil {
			t.Fatal(err)
		}
		again, _ := randomOffsets(ext, chunk, align, n, 42)
		if !slices.Equal(offs, again) {
			t.Fatal("not deterministic")
		}
		s := slices.Clone(offs)
		slices.Sort(s)
		for i, o := range s {
			if o%align != 0 {
				t.Fatalf("chunk %d: offset %d not aligned", chunk, o)
			}
			if !slices.ContainsFunc(ext, func(e extent) bool { return o >= e.Start && o+chunk <= e.End }) {
				t.Fatalf("chunk %d: [%d, %d) outside the extents", chunk, o, o+chunk)
			}
			if i > 0 && s[i-1]+chunk > o {
				t.Fatalf("chunk %d: %d and %d overlap", chunk, s[i-1], o)
			}
		}
		if _, err := randomOffsets(ext, chunk, align, n+1, 42); err == nil {
			t.Errorf("chunk %d: %d offsets of %d positions accepted", chunk, n+1, n)
		}
	}
	// 1 MiB chunks: the first extent holds 9 (it starts at 100, so the first
	// aligned position is 16 KiB), the second 1, the third 2.
	if g := newGrid(ext, 1<<20, align); g.total != 9+1+2 {
		t.Errorf("1 MiB grid: %d positions, want 12", g.total)
	}
}

func TestRandomExperts(t *testing.T) {
	ids, err := randomExperts(50, 384, 50*384, 3)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[expertID]bool{}
	for _, id := range ids {
		if id.Layer < 0 || id.Layer >= 50 || id.Expert < 0 || id.Expert >= 384 || seen[id] {
			t.Fatalf("bad or repeated expert %+v", id)
		}
		seen[id] = true
	}
}

func TestPercentile(t *testing.T) {
	var d []time.Duration
	for i := 100; i >= 1; i-- { // unsorted on purpose
		d = append(d, time.Duration(i)*time.Microsecond)
	}
	for _, c := range []struct {
		p    float64
		want time.Duration
	}{{50, 50}, {99, 99}, {100, 100}, {0, 1}, {1, 1}, {1.5, 2}} {
		if got := percentile(d, c.p); got != c.want*time.Microsecond {
			t.Errorf("p%g = %v, want %v", c.p, got, c.want*time.Microsecond)
		}
	}
	if d[0] != 100*time.Microsecond {
		t.Error("percentile sorted its input")
	}
	s := summarize(timing{Ops: 100, Bytes: 200e6, Seconds: 2, Latencies: d})
	want := stats{MBps: 100, IOPS: 50, P50Us: 50, P99Us: 99, MaxUs: 100, MeanUs: 50.5}
	if s != want {
		t.Errorf("summarize = %+v, want %+v", s, want)
	}
}

func addr(b []byte) uintptr { return uintptr(unsafe.Pointer(&b[0])) }

// recorder wraps the file and records every read.
type recorder struct {
	f     *os.File
	mu    sync.Mutex
	reads [][2]int64
}

func (r *recorder) ReadAt(p []byte, off int64) (int, error) {
	r.mu.Lock()
	r.reads = append(r.reads, [2]int64{off, int64(len(p))})
	r.mu.Unlock()
	return r.f.ReadAt(p, off)
}

// TestCellSmoke runs raw and expert cells on the fixture, buffered and
// without eviction (as -no-evict does), and checks what was read.
func TestCellSmoke(t *testing.T) {
	path, hdr := writeFixture(t)
	l, err := expertLayout(hdr)
	if err != nil {
		t.Fatal(err)
	}
	f, err := openFile(path, "buffered")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	const align = 512
	volume := int64(2 * tLayers * tExperts * (2*tGateSlice + tDownSlice) / 3)
	for _, c := range []cell{
		{"raw", "buffered", 1024, 1},
		{"raw", "buffered", 1024, 4},
		{"expert-three", "buffered", l.maxExpertBytes(), 3},
		{"expert-contiguous", "buffered", l.maxExpertBytes(), 2},
	} {
		ops, err := cellOps(c, l, volume, align, 1)
		if err != nil {
			t.Fatalf("%+v: %v", c, err)
		}
		rec := &recorder{f: f}
		tm, err := runCell(rec, ops, c.QD, 0, align)
		if err != nil {
			t.Fatalf("%+v: %v", c, err)
		}
		if tm.Ops != len(ops) || len(tm.Latencies) != len(ops) || tm.TimeCap {
			t.Errorf("%+v: %d ops timed of %d", c, tm.Ops, len(ops))
		}
		var want int64
		for _, o := range ops {
			for _, r := range o {
				want += r.N
			}
		}
		if tm.Bytes != want {
			t.Errorf("%+v: %d bytes, want %d", c, tm.Bytes, want)
		}
		for _, r := range rec.reads {
			if !slices.ContainsFunc(l.Extents, func(e extent) bool { return r[0] >= e.Start && r[0]+r[1] <= e.End }) {
				t.Errorf("%+v: read [%d, %d) outside the experts", c, r[0], r[0]+r[1])
			}
		}
		if c.Test == "expert-three" {
			if len(rec.reads) != 3*len(ops) {
				t.Errorf("expert-three: %d reads for %d experts", len(rec.reads), len(ops))
			}
			// every read is one of an expert's three slices
			for _, r := range rec.reads {
				ok := false
				for il := range tLayers {
					for k := range kinds {
						for e := range tExperts {
							ok = ok || (r[0] == l.sliceOffset(il, k, e) && r[1] == l.Layers[il][k].Slice)
						}
					}
				}
				if !ok {
					t.Errorf("expert-three: read [%d, +%d) is no expert slice", r[0], r[1])
				}
			}
		}
	}

	// the data arrives: one raw read, compared with the pattern
	ops := rawOps([]int64{l.Extents[0].Start}, 4096)
	buf := alignedBuf(4096, 4096)
	if uintptr(len(buf)) != 4096 || addr(buf)%4096 != 0 {
		t.Fatal("alignedBuf not aligned")
	}
	if err := readFull(f, buf, ops[0][0].Off); err != nil {
		t.Fatal(err)
	}
	for i, v := range buf {
		if want := byte((ops[0][0].Off + int64(i)) * 7 % 251); v != want {
			t.Fatalf("byte %d = %d, want %d", i, v, want)
		}
	}
	if err := readFull(f, buf, 1<<40); err == nil {
		t.Error("a read past the end succeeded")
	}
}

func TestTimeCap(t *testing.T) {
	path, hdr := writeFixture(t)
	l, _ := expertLayout(hdr)
	f, _ := os.Open(path)
	defer f.Close()
	ops, err := cellOps(cell{"raw", "buffered", 512, 1}, l, 512*100, 512, 1)
	if err != nil {
		t.Fatal(err)
	}
	tm, err := runCell(f, ops, 1, time.Nanosecond, 512)
	if err != nil {
		t.Fatal(err)
	}
	if !tm.TimeCap || tm.Ops >= len(ops) {
		t.Errorf("1 ns cap: %d of %d ops, capped %v", tm.Ops, len(ops), tm.TimeCap)
	}
}

func TestParseChunks(t *testing.T) {
	_, hdr := writeFixture(t)
	l, _ := expertLayout(hdr)
	got, err := parseChunks("64KiB,slice,1MiB,expert,1024,64KiB", l)
	if err != nil {
		t.Fatal(err)
	}
	want := []int64{1024, tGateSlice, tDownSlice, 2*tGateSlice + tDownSlice, 64 << 10, 1 << 20}
	if !slices.Equal(got, want) {
		t.Errorf("chunks %v, want %v", got, want)
	}
	if _, err := parseChunks("fast", l); err == nil {
		t.Error("chunk fast accepted")
	}
	cs := planCells(false, []string{"nocache", "buffered"}, want, []int{1, 4}, l)
	if len(cs) != 2*len(want)*2 {
		t.Errorf("%d raw cells", len(cs))
	}
	if cs := planCells(true, []string{"nocache"}, want, []int{1, 4}, l); len(cs) != 4 || cs[0].Test != "expert-three" || cs[2].Test != "expert-contiguous" {
		t.Errorf("expert cells %+v", cs)
	}
}

func TestParseOnAC(t *testing.T) {
	if !parseOnAC("Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t80%; charging;") {
		t.Error("AC not recognized")
	}
	if parseOnAC("Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t80%; discharging;") {
		t.Error("battery taken for AC")
	}
}

// TestM41 is the control's rule: a warm buffered raw cell of up to 256 KiB
// must be 3x the evicted one.
func TestM41(t *testing.T) {
	cell := func(mode string, warm bool, mbps float64) cellRecord {
		return cellRecord{Kind: "cell", Test: "raw", Mode: mode, Chunk: 64 << 10, QD: 4, Warm: warm, stats: stats{MBps: mbps}}
	}
	if _, ok := m41([]cellRecord{cell("buffered", false, 2000), cell("buffered", true, 9000)}); !ok {
		t.Error("4.5x failed")
	}
	if lines, ok := m41([]cellRecord{cell("buffered", false, 2000), cell("buffered", true, 2500)}); ok || !strings.HasPrefix(lines[0], "FAIL") {
		t.Errorf("1.25x passed: %v", lines)
	}
	// nocache pairs are INFO only, and no buffered pair is no pass
	if lines, ok := m41([]cellRecord{cell("nocache", false, 2000), cell("nocache", true, 2100)}); ok || !strings.HasPrefix(lines[0], "INFO") {
		t.Errorf("nocache only: %v %v", lines, ok)
	}
	// so are chunks above 256 KiB: bandwidth-bound, page cache and SSD close
	big := []cellRecord{cell("buffered", false, 2000), cell("buffered", true, 9000), cell("buffered", false, 8000), cell("buffered", true, 16000)}
	big[2].Chunk, big[3].Chunk = 8<<20, 8<<20
	if lines, ok := m41(big); !ok || !strings.HasPrefix(lines[1], "INFO") {
		t.Errorf("8 MiB at 2x decided: %v %v", lines, ok)
	}
}

func TestReportRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run.jsonl")
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	run := runRecord{Kind: "run", Date: "2026-10-22T10:00:00Z", Command: "kolibri-ssd -expert", File: "x.gguf", FileBytes: 1234567,
		Layer0:  [3]expertTensor{{"IQ3_XXS", 1, 501760}, {"IQ3_XXS", 2, 501760}, {"IQ4_XS", 3, 696320}},
		Machine: map[string]string{"cpu": "Apple M5 Pro", "ssd": "APPLE SSD AP1024Z"}, Power: "Now drawing from 'AC Power'"}
	c := cellRecord{Kind: "cell", Test: "expert-three", Mode: "nocache", Chunk: 1_699_840, QD: 4, Ops: 10, Bytes: 16_998_400, Seconds: 0.01,
		stats: stats{MBps: 1699.84, IOPS: 1000, P50Us: 1500, P99Us: 4000, MaxUs: 5000, MeanUs: 1600}, Pages: 100, ResidentAfter: 3}
	_ = enc.Encode(run)
	_ = enc.Encode(c)
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), `"mb_per_s":1699.84`) || !strings.Contains(buf.String(), `"p99_us":4000`) {
		t.Fatalf("stats not flat in %s", buf.String())
	}
	runs, cells, err := readRecords([]string{path})
	if err != nil {
		t.Fatal(err)
	}
	if len(runs) != 1 || len(cells) != 1 || cells[0] != c {
		t.Fatalf("read back %+v %+v", runs, cells)
	}
	var md bytes.Buffer
	writeMarkdown(&md, runs, cells)
	for _, want := range []string{"## expert-three, nocache, evicted (cold)", "| 1,699,840 B |", "1700 (1000)<br>1.50 / 4.00 / 5.00", "APPLE SSD AP1024Z", "gate IQ3_XXS 501,760"} {
		if !strings.Contains(md.String(), want) {
			t.Errorf("markdown lacks %q:\n%s", want, md.String())
		}
	}
}
