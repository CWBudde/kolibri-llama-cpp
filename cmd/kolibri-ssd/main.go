// Command kolibri-ssd measures the internal SSD's random-read curve the way
// the runtime expert cache (patch 0011) reads it, for PLAN Phase 11, Stage E1:
// is one IQ3 expert (1.7 MB) latency- or bandwidth-bound, how many I/O
// threads pay off, and how much one contiguous read per expert would gain
// over today's three (gate, up, down). It only reads the GGUF.
//
//	go run ./cmd/kolibri-ssd -json ~/models/eval/ssd/sweep.jsonl
//	go run ./cmd/kolibri-ssd -expert -json ~/models/eval/ssd/expert.jsonl
//	go run ./cmd/kolibri-ssd -report -md docs/ssd-curve.md ~/models/eval/ssd/*.jsonl
//
// The defaults (chunks of 64 KiB, 256 KiB, the expert slices and 8 MiB; QD
// 1, 4 and 16; both modes; 512 MiB or 20 s per cell) fit a run of about 20
// minutes; -full sweeps 7 chunk sizes, QD 1 to 16 and 2 GiB per cell.
//
// The raw sweep (default) reads, per cell, a fixed volume (-bytes, or until
// -max-seconds) in chunks of one size (-chunks; "slice" stands for the
// GGUF's per-tensor expert slice sizes and "expert" for its per-expert
// sizes) at distinct, non-overlapping random offsets aligned to -align
// inside the routed-expert tensors, drawn with a seed derived from -seed and
// the cell. QD (-qd) is the number of goroutines, each issuing blocking
// preads on one shared descriptor, as the cache's I/O threads share one fd.
// Mode nocache sets F_NOCACHE on that descriptor, as the runtime does;
// buffered reads through the page cache.
//
// -expert reads whole experts instead: distinct random (layer, expert)
// pairs, each either with three preads of its gate, up and down slices one
// after another, as patch 0011 does (expert-three), or with one pread of the
// same size at a random aligned offset in the expert region (expert-
// contiguous: what a side file with the experts repacked would cost). QD is
// the number of concurrent workers; latency is per expert.
//
// Before every cell the file is evicted from the page cache
// (internal/memctl) and its residency checked: more than -max-resident left
// aborts the run (another process maps the file). -no-evict skips both and
// instead reads the cell's ops once, untimed and buffered, so the timed pass
// finds them in the page cache. That is control M41: the same command with
// -no-evict must report page-cache speed far above the evicted SSD speed,
// or eviction does not work and the curve measures memory. Cells take their
// offsets from -seed and their test, mode, chunk and QD, so the warm run
// reads exactly the cold run's offsets. -report checks M41 on the combined
// JSON lines and exits 1 unless every buffered raw cell of up to 256 KiB is
// at least 3x faster warm than evicted:
//
//	go run ./cmd/kolibri-ssd -modes buffered -chunks 64KiB,256KiB -json m41-cold.jsonl
//	go run ./cmd/kolibri-ssd -modes buffered -chunks 64KiB,256KiB -json m41-warm.jsonl -no-evict
//	go run ./cmd/kolibri-ssd -report m41-cold.jsonl m41-warm.jsonl
//
// Output: one JSON line per run and per cell (stdout, or -json), progress on
// stderr, and with -md a markdown report. The run refuses battery power
// unless -allow-battery: on battery a Mac sleeps and timings are wrong.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"kolibri-llm/internal/gguf"
)

// residency is a file's pages in the page cache.
type residency struct{ Pages, Resident int }

// parseSize reads a byte count with an optional KiB, MiB or GiB suffix.
func parseSize(s string) (int64, error) {
	mult := int64(1)
	for _, u := range []struct {
		suffix string
		mult   int64
	}{{"GiB", 1 << 30}, {"MiB", 1 << 20}, {"KiB", 1 << 10}} {
		if rest, ok := strings.CutSuffix(s, u.suffix); ok {
			s, mult = rest, u.mult
			break
		}
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil || v <= 0 {
		return 0, fmt.Errorf("size %q: want a positive number with an optional KiB, MiB or GiB", s)
	}
	return int64(v * float64(mult)), nil
}

// parseChunks reads -chunks: sizes and the keywords slice and expert.
func parseChunks(s string, l *layout) ([]int64, error) {
	var out []int64
	for f := range strings.SplitSeq(s, ",") {
		switch f = strings.TrimSpace(f); f {
		case "slice":
			out = append(out, l.sliceSizes()...)
		case "expert":
			out = append(out, l.expertSizes()...)
		default:
			n, err := parseSize(f)
			if err != nil {
				return nil, err
			}
			out = append(out, n)
		}
	}
	slices.Sort(out)
	return slices.Compact(out), nil
}

func parseInts(s string) ([]int, error) {
	var out []int
	for f := range strings.SplitSeq(s, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil || n < 1 {
			return nil, fmt.Errorf("%q: want positive integers", s)
		}
		out = append(out, n)
	}
	return out, nil
}

func parseModes(s string) ([]string, error) {
	var out []string
	for f := range strings.SplitSeq(s, ",") {
		if f = strings.TrimSpace(f); f != "nocache" && f != "buffered" {
			return nil, fmt.Errorf("mode %q: want nocache or buffered", f)
		}
		out = append(out, f)
	}
	return out, nil
}

// cell is one measurement to make.
type cell struct {
	Test  string
	Mode  string
	Chunk int64 // bytes per op: a chunk, or the largest expert
	QD    int
}

// planCells lists the cells: per mode, per chunk (or expert layout), per QD.
func planCells(expert bool, modes []string, chunks []int64, qds []int, l *layout) []cell {
	var cs []cell
	for _, m := range modes {
		if expert {
			for _, t := range []string{"expert-three", "expert-contiguous"} {
				for _, q := range qds {
					cs = append(cs, cell{t, m, l.maxExpertBytes(), q})
				}
			}
			continue
		}
		for _, c := range chunks {
			for _, q := range qds {
				cs = append(cs, cell{"raw", m, c, q})
			}
		}
	}
	return cs
}

// cellOps makes a cell's ops: volume/chunk of them, at least one.
func cellOps(c cell, l *layout, volume, align int64, seed uint64) ([]op, error) {
	n := int(max(1, volume/c.Chunk))
	s := cellSeed(seed, c.Test, c.Mode, c.Chunk, c.QD)
	switch c.Test {
	case "raw":
		offs, err := randomOffsets(l.Extents, c.Chunk, align, n, s)
		if err != nil {
			return nil, err
		}
		return rawOps(offs, c.Chunk), nil
	case "expert-three":
		ids, err := randomExperts(len(l.Layers), l.Experts, n, s)
		if err != nil {
			return nil, err
		}
		return expertThreeOps(l, ids), nil
	default: // expert-contiguous
		ids, err := randomExperts(len(l.Layers), l.Experts, n, s)
		if err != nil {
			return nil, err
		}
		offs, err := randomOffsets(l.Extents, c.Chunk, align, n, s+1)
		if err != nil {
			return nil, err
		}
		return expertContiguousOps(l, ids, offs), nil
	}
}

func expandHome(p string) string {
	if rest, ok := strings.CutPrefix(p, "~/"); ok {
		if h, err := os.UserHomeDir(); err == nil {
			return filepath.Join(h, rest)
		}
	}
	return p
}

func output(name string, args ...string) string {
	out, err := exec.Command(name, args...).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// machine describes the Mac; missing tools leave fields empty.
func machine() map[string]string {
	m := map[string]string{
		"model": output("sysctl", "-n", "hw.model"),
		"cpu":   output("sysctl", "-n", "machdep.cpu.brand_string"),
		"os":    output("sw_vers", "-productVersion"),
	}
	if b, err := strconv.ParseInt(output("sysctl", "-n", "hw.memsize"), 10, 64); err == nil {
		m["memory"] = fmt.Sprintf("%d GiB", b>>30)
	}
	for line := range strings.SplitSeq(output("system_profiler", "SPNVMeDataType"), "\n") {
		if v, ok := strings.CutPrefix(strings.TrimSpace(line), "Model: "); ok {
			m["ssd"] = v
			break
		}
	}
	return m
}

// fullGrid is -full: the finer sweep the defaults trim to fit a 20-minute
// run.
var fullGrid = map[string]string{
	"chunks":      "64KiB,256KiB,slice,1MiB,expert,4MiB,8MiB",
	"qd":          "1,2,4,8,16",
	"bytes":       "2GiB",
	"max-seconds": "30",
}

// parseOnAC reads pmset -g batt (or ps), as cmd/kolibri-bench does.
func parseOnAC(s string) bool { return strings.Contains(s, "'AC Power'") }

func main() {
	path := flag.String("gguf", "~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf", "GGUF to read (only read)")
	expert := flag.Bool("expert", false, "read whole experts (three preads, or one contiguous) instead of raw chunks")
	modesF := flag.String("modes", "nocache,buffered", "comma-separated: nocache (F_NOCACHE), buffered")
	chunksF := flag.String("chunks", "64KiB,256KiB,slice,8MiB", "chunk sizes of the raw sweep; slice and expert are the GGUF's per-tensor expert slice and per-expert sizes")
	qdF := flag.String("qd", "1,4,16", "queue depths: concurrent readers on one descriptor")
	volumeF := flag.String("bytes", "512MiB", "bytes to read per cell")
	maxSec := flag.Float64("max-seconds", 20, "stop a cell after this many seconds (0: no cap)")
	full := flag.Bool("full", false, "the full grid: -chunks "+fullGrid["chunks"]+" -qd "+fullGrid["qd"]+" -bytes "+fullGrid["bytes"]+" -max-seconds "+fullGrid["max-seconds"]+" (explicit flags win)")
	alignF := flag.String("align", "", "offset alignment (default: the page size)")
	seed := flag.Uint64("seed", 1, "seed of the random offsets and experts")
	noEvict := flag.Bool("no-evict", false, "control M41: skip the eviction and its check, and warm each cell's offsets before timing")
	maxResF := flag.String("max-resident", "64MiB", "abort if more of the file stays in the page cache after eviction")
	jsonPath := flag.String("json", "", "write the JSON lines here instead of stdout")
	mdPath := flag.String("md", "", "write a markdown report")
	report := flag.Bool("report", false, "only render -md (or stdout) from the JSON-lines files given as arguments, and check M41")
	layoutOnly := flag.Bool("layout", false, "print the GGUF's expert layout and exit")
	allowBattery := flag.Bool("allow-battery", false, "run on battery power")
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: kolibri-ssd [flags]\n       kolibri-ssd -report [-md FILE] RUN.jsonl...")
		flag.PrintDefaults()
	}
	flag.Parse()
	log.SetFlags(0)
	log.SetPrefix("kolibri-ssd: ")
	if *full {
		set := map[string]bool{}
		flag.Visit(func(f *flag.Flag) { set[f.Name] = true })
		for name, v := range fullGrid {
			if !set[name] {
				if err := flag.Set(name, v); err != nil {
					log.Fatal(err)
				}
			}
		}
	}

	if *report {
		os.Exit(runReport(flag.Args(), *mdPath))
	}

	file := expandHome(*path)
	hdr, err := gguf.Open(file)
	if err != nil {
		log.Fatal(err)
	}
	l, err := expertLayout(hdr)
	if err != nil {
		log.Fatalf("%s: %v", file, err)
	}
	if *layoutOnly {
		printLayout(os.Stdout, file, l)
		return
	}

	modes, err := parseModes(*modesF)
	if err != nil {
		log.Fatal(err)
	}
	chunks, err := parseChunks(*chunksF, l)
	if err != nil {
		log.Fatal(err)
	}
	qds, err := parseInts(*qdF)
	if err != nil {
		log.Fatal(err)
	}
	volume, err := parseSize(*volumeF)
	if err != nil {
		log.Fatal(err)
	}
	maxRes, err := parseSize(*maxResF)
	if err != nil {
		log.Fatal(err)
	}
	align := int64(pageSize())
	if *alignF != "" {
		if align, err = parseSize(*alignF); err != nil {
			log.Fatal(err)
		}
	}

	power := output("pmset", "-g", "batt")
	first, _, _ := strings.Cut(power, "\n")
	log.Printf("power: %s", first)
	if !parseOnAC(power) && !*allowBattery {
		log.Fatal("not on AC power: timings on battery are not valid (-allow-battery to run anyway)")
	}

	st, err := os.Stat(file)
	if err != nil {
		log.Fatal(err)
	}
	out := io.Writer(os.Stdout)
	if *jsonPath != "" {
		jf, err := os.Create(expandHome(*jsonPath))
		if err != nil {
			log.Fatal(err)
		}
		defer jf.Close()
		out = jf
	}
	enc := json.NewEncoder(out)

	run := runRecord{
		Kind: "run", Date: time.Now().Format(time.RFC3339), Command: strings.Join(os.Args, " "),
		File: file, FileBytes: st.Size(), ExpertBytes: l.extentBytes(), Layers: len(l.Layers), Experts: l.Experts,
		Layer0: l.Layers[0], Machine: machine(), Power: first, PageSize: pageSize(), Seed: *seed, NoEvict: *noEvict,
	}
	if err := enc.Encode(run); err != nil {
		log.Fatal(err)
	}

	cells := planCells(*expert, modes, chunks, qds, l)
	log.Printf("%s: %d cells, up to %s each, %s bytes of experts", filepath.Base(file), len(cells), size(volume), group(l.extentBytes()))
	maxDur := time.Duration(*maxSec * float64(time.Second))
	fds := map[string]*os.File{}
	for _, m := range append([]string{"buffered"}, modes...) {
		if fds[m] != nil {
			continue
		}
		if fds[m], err = openFile(file, m); err != nil {
			log.Fatal(err)
		}
		defer fds[m].Close()
	}

	var recs []cellRecord
	for i, c := range cells {
		ops, err := cellOps(c, l, volume, align, *seed)
		if err != nil {
			log.Fatal(err)
		}
		rec := cellRecord{Kind: "cell", Test: c.Test, Mode: c.Mode, Chunk: c.Chunk, QD: c.QD, Warm: *noEvict}
		if *noEvict {
			if _, err := runCell(fds["buffered"], ops, 8, 0, align); err != nil {
				log.Fatal(err)
			}
		} else if err := evict(file); err != nil {
			log.Fatal(err)
		}
		before, err := resident(file)
		if err != nil {
			log.Fatal(err)
		}
		if !*noEvict && int64(before.Resident)*int64(pageSize()) > maxRes {
			log.Fatalf("%d pages (%.0f MiB) of %s stay in the page cache after eviction, over -max-resident %s: does another process map it?",
				before.Resident, float64(before.Resident)*float64(pageSize())/(1<<20), file, *maxResF)
		}
		t, err := runCell(fds[c.Mode], ops, c.QD, maxDur, align)
		if err != nil {
			log.Fatal(err)
		}
		after, err := resident(file)
		if err != nil {
			log.Fatal(err)
		}
		rec.Ops, rec.Bytes, rec.Seconds, rec.TimeCap = t.Ops, t.Bytes, t.Seconds, t.TimeCap
		rec.stats = summarize(t)
		rec.Pages, rec.ResidentBefore, rec.ResidentAfter = before.Pages, before.Resident, after.Resident
		if err := enc.Encode(rec); err != nil {
			log.Fatal(err)
		}
		recs = append(recs, rec)
		log.Printf("[%d/%d] %s %s %s QD %d: %.0f MB/s, %.0f ops/s, p50 %s p99 %s ms, %d ops in %.1f s, resident %d -> %d",
			i+1, len(cells), c.Test, c.Mode, size(c.Chunk), c.QD, rec.MBps, rec.IOPS, ms(rec.P50Us), ms(rec.P99Us), rec.Ops, rec.Seconds, before.Resident, after.Resident)
	}
	if *mdPath != "" {
		if err := writeMarkdownFile(*mdPath, []runRecord{run}, recs); err != nil {
			log.Fatal(err)
		}
	}
}

func writeMarkdownFile(path string, runs []runRecord, cells []cellRecord) error {
	f, err := os.Create(expandHome(path))
	if err != nil {
		return err
	}
	writeMarkdown(f, runs, cells)
	return f.Close()
}

// runReport renders the report of JSON-lines files and checks M41 when
// they hold warm cells; it returns the exit code.
func runReport(paths []string, md string) int {
	if len(paths) == 0 {
		log.Print("-report needs JSON-lines files")
		return 2
	}
	runs, cells, err := readRecords(paths)
	if err != nil {
		log.Print(err)
		return 1
	}
	if md != "" {
		err = writeMarkdownFile(md, runs, cells)
	} else {
		writeMarkdown(os.Stdout, runs, cells)
	}
	if err != nil {
		log.Print(err)
		return 1
	}
	if !slices.ContainsFunc(cells, func(c cellRecord) bool { return c.Warm }) {
		return 0
	}
	lines, ok := m41(cells)
	for _, l := range lines {
		fmt.Fprintln(os.Stderr, l)
	}
	if !ok {
		fmt.Fprintf(os.Stderr, "FAIL M41: a buffered raw warm cell up to %s is not %.0fx its evicted cell, or there is no such pair\n", size(m41MaxChunk), m41Ratio)
		return 1
	}
	fmt.Fprintf(os.Stderr, "PASS M41: every buffered raw warm cell up to %s is at least %.0fx its evicted cell\n", size(m41MaxChunk), m41Ratio)
	return 0
}

// printLayout prints where the experts are, to check against gguf-py.
func printLayout(w io.Writer, file string, l *layout) {
	fmt.Fprintf(w, "%s: %s, %d layers x %d experts, %s bytes of expert tensors in %d extents\n",
		file, l.Arch, len(l.Layers), l.Experts, group(l.extentBytes()), len(l.Extents))
	for il, lt := range l.Layers {
		fmt.Fprintf(w, "blk.%d:", il)
		for k, t := range lt {
			fmt.Fprintf(w, " %s %s @%d slice %d;", kinds[k], t.Type, t.Offset, t.Slice)
		}
		fmt.Fprintf(w, " expert %d\n", l.expertBytes(il))
	}
}
