package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"
)

// runRecord is the first JSON line of a run.
type runRecord struct {
	Kind        string            `json:"kind"` // "run"
	Date        string            `json:"date"`
	Command     string            `json:"command"`
	File        string            `json:"file"`
	FileBytes   int64             `json:"file_bytes"`
	ExpertBytes int64             `json:"expert_region_bytes"`
	Layers      int               `json:"layers"`
	Experts     int               `json:"experts"`
	Layer0      [3]expertTensor   `json:"layer0"` // gate, up, down of layer 0
	Machine     map[string]string `json:"machine"`
	Power       string            `json:"power"`
	PageSize    int               `json:"page_size"`
	Seed        uint64            `json:"seed"`
	NoEvict     bool              `json:"no_evict"`
}

// cellRecord is one cell's JSON line.
type cellRecord struct {
	Kind    string  `json:"kind"` // "cell"
	Test    string  `json:"test"` // raw, expert-three, expert-contiguous
	Mode    string  `json:"mode"` // nocache, buffered
	Chunk   int64   `json:"chunk"`
	QD      int     `json:"qd"`
	Warm    bool    `json:"warm"` // -no-evict: read once untimed before
	Ops     int     `json:"ops"`
	Bytes   int64   `json:"bytes"`
	Seconds float64 `json:"seconds"`
	TimeCap bool    `json:"time_cap"`
	stats
	// pages of the file in the page cache before the timed pass and after
	// it, of Pages
	Pages          int `json:"pages"`
	ResidentBefore int `json:"resident_before"`
	ResidentAfter  int `json:"resident_after"`
}

// readRecords reads the JSON lines of one or more runs.
func readRecords(paths []string) ([]runRecord, []cellRecord, error) {
	var runs []runRecord
	var cells []cellRecord
	for _, p := range paths {
		f, err := os.Open(p)
		if err != nil {
			return nil, nil, err
		}
		sc := bufio.NewScanner(f)
		sc.Buffer(nil, 1<<20)
		for n := 1; sc.Scan(); n++ {
			line := sc.Bytes()
			var k struct {
				Kind string `json:"kind"`
			}
			if err := json.Unmarshal(line, &k); err != nil {
				f.Close()
				return nil, nil, fmt.Errorf("%s:%d: %w", p, n, err)
			}
			switch k.Kind {
			case "run":
				var r runRecord
				err = json.Unmarshal(line, &r)
				runs = append(runs, r)
			case "cell":
				var c cellRecord
				err = json.Unmarshal(line, &c)
				cells = append(cells, c)
			}
			if err != nil {
				f.Close()
				return nil, nil, fmt.Errorf("%s:%d: %w", p, n, err)
			}
		}
		f.Close()
		if err := sc.Err(); err != nil {
			return nil, nil, err
		}
	}
	return runs, cells, nil
}

// size names a chunk: MiB when whole, KiB for a power of 2, else bytes (the
// expert slices).
func size(n int64) string {
	switch {
	case n >= 1<<20 && n%(1<<20) == 0:
		return fmt.Sprintf("%d MiB", n>>20)
	case n >= 1<<10 && n&(n-1) == 0:
		return fmt.Sprintf("%d KiB", n>>10)
	default:
		return fmt.Sprintf("%s B", group(n))
	}
}

// group writes n with thousands separators.
func group(n int64) string {
	s := fmt.Sprint(n)
	var b strings.Builder
	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteRune(c)
	}
	return b.String()
}

func ms(us float64) string { return fmt.Sprintf("%.2f", us/1000) }

// groupKey is one table: a test and a mode, evicted or warm.
type groupKey struct {
	Test, Mode string
	Warm       bool
}

func (g groupKey) String() string {
	state := "evicted (cold)"
	if g.Warm {
		state = "not evicted, warmed (-no-evict)"
	}
	return fmt.Sprintf("%s, %s, %s", g.Test, g.Mode, state)
}

// writeMarkdown renders runs and cells: a header per run, then per test,
// mode and state a matrix of chunk (or expert layout) by queue depth.
func writeMarkdown(w io.Writer, runs []runRecord, cells []cellRecord) {
	fmt.Fprintln(w, "# SSD random-read curve")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Written by `cmd/kolibri-ssd` (PLAN Phase 11, Stage E1). Each cell reads a fixed volume at distinct,")
	fmt.Fprintln(w, "non-overlapping random offsets inside the GGUF's routed-expert tensors; QD is the number of goroutines")
	fmt.Fprintln(w, "issuing blocking `pread`s on one shared descriptor, as the expert cache's I/O threads do. `nocache` sets")
	fmt.Fprintln(w, "`F_NOCACHE` like patch 0011; `buffered` goes through the page cache. MB/s is 10^6 bytes per second of")
	fmt.Fprintln(w, "wall time; latencies are per op (one chunk, or one expert) in ms.")
	for _, r := range runs {
		fmt.Fprintln(w)
		fmt.Fprintf(w, "## Run %s\n\n", r.Date)
		fmt.Fprintf(w, "- Command: `%s`\n", r.Command)
		fmt.Fprintf(w, "- File: `%s` (%s bytes; experts %s bytes, %d layers x %d)\n", r.File, group(r.FileBytes), group(r.ExpertBytes), r.Layers, r.Experts)
		fmt.Fprintf(w, "- Expert slices (layer 0): gate %s %s, up %s %s, down %s %s bytes\n",
			r.Layer0[0].Type, group(r.Layer0[0].Slice), r.Layer0[1].Type, group(r.Layer0[1].Slice), r.Layer0[2].Type, group(r.Layer0[2].Slice))
		var m []string
		for _, k := range []string{"model", "cpu", "memory", "os", "ssd"} {
			if v := r.Machine[k]; v != "" {
				m = append(m, k+" "+v)
			}
		}
		fmt.Fprintf(w, "- Machine: %s\n", strings.Join(m, "; "))
		fmt.Fprintf(w, "- Power: %s; page size %d; seed %d; evict before each cell: %v\n", r.Power, r.PageSize, r.Seed, !r.NoEvict)
	}

	var groups []groupKey
	for _, c := range cells {
		g := groupKey{c.Test, c.Mode, c.Warm}
		if !slices.Contains(groups, g) {
			groups = append(groups, g)
		}
	}
	for _, g := range groups {
		var gc []cellRecord
		for _, c := range cells {
			if (groupKey{c.Test, c.Mode, c.Warm}) == g {
				gc = append(gc, c)
			}
		}
		writeGroup(w, g, gc)
	}
	if lines, _ := m41(cells); len(lines) > 0 {
		fmt.Fprintln(w)
		fmt.Fprintln(w, "## Control M41: eviction matters")
		fmt.Fprintln(w)
		for _, l := range lines {
			fmt.Fprintln(w, "- "+l)
		}
	}
}

func writeGroup(w io.Writer, g groupKey, cells []cellRecord) {
	var rows []int64
	var qds []int
	for _, c := range cells {
		if !slices.Contains(rows, c.Chunk) {
			rows = append(rows, c.Chunk)
		}
		if !slices.Contains(qds, c.QD) {
			qds = append(qds, c.QD)
		}
	}
	slices.Sort(rows)
	slices.Sort(qds)
	unit := "chunk"
	if strings.HasPrefix(g.Test, "expert") {
		unit = "expert"
	}
	fmt.Fprintln(w)
	fmt.Fprintf(w, "## %s\n\n", g)
	fmt.Fprintf(w, "MB/s (ops/s), then p50 / p99 / max latency in ms per %s.\n\n", unit)
	fmt.Fprintf(w, "| %s |", unit)
	for _, q := range qds {
		fmt.Fprintf(w, " QD %d |", q)
	}
	fmt.Fprintln(w)
	fmt.Fprint(w, "|---:|")
	for range qds {
		fmt.Fprint(w, "---:|")
	}
	fmt.Fprintln(w)
	for _, r := range rows {
		fmt.Fprintf(w, "| %s |", size(r))
		for _, q := range qds {
			i := slices.IndexFunc(cells, func(c cellRecord) bool { return c.Chunk == r && c.QD == q })
			if i < 0 {
				fmt.Fprint(w, " |")
				continue
			}
			c := cells[i]
			capped := ""
			if c.TimeCap {
				capped = "*"
			}
			fmt.Fprintf(w, " %.0f (%.0f)%s<br>%s / %s / %s |", c.MBps, c.IOPS, capped, ms(c.P50Us), ms(c.P99Us), ms(c.MaxUs))
		}
		fmt.Fprintln(w)
	}
	var before, after, pages int
	capped := false
	for _, c := range cells {
		before, after, pages = max(before, c.ResidentBefore), max(after, c.ResidentAfter), max(pages, c.Pages)
		capped = capped || c.TimeCap
	}
	fmt.Fprintf(w, "\nResident pages of %s: at most %s before a timed pass, %s after one.", group(int64(pages)), group(int64(before)), group(int64(after)))
	if capped {
		fmt.Fprint(w, " * stopped by the time cap before the full volume.")
	}
	fmt.Fprintln(w)
}

// m41Ratio is how much faster a warm buffered cell must be than the same
// evicted cell for control M41 to pass, at chunks up to m41MaxChunk: there
// a read is latency-bound, and the page cache answers in microseconds where
// the SSD takes a tenth of a millisecond. Larger chunks approach the SSD's
// bandwidth, which on Apple silicon is within a small factor of a memcpy.
const (
	m41Ratio    = 3.0
	m41MaxChunk = 256 << 10
)

// m41 compares every warm cell (-no-evict) with the evicted cell of the same
// test, mode, chunk and queue depth. It passes if every buffered raw pair
// up to m41MaxChunk is at least m41Ratio times faster warm. Other pairs are
// INFO: larger chunks, experts, and nocache (a resident page is served from
// the page cache even through F_NOCACHE). ok is false without a deciding
// pair.
func m41(cells []cellRecord) (lines []string, ok bool) {
	type key struct {
		Test, Mode string
		Chunk      int64
		QD         int
	}
	cold := map[key]cellRecord{}
	for _, c := range cells {
		if !c.Warm {
			cold[key{c.Test, c.Mode, c.Chunk, c.QD}] = c
		}
	}
	ok = true
	pairs := 0
	for _, c := range cells {
		if !c.Warm {
			continue
		}
		e, found := cold[key{c.Test, c.Mode, c.Chunk, c.QD}]
		if !found || e.MBps <= 0 {
			continue
		}
		ratio := c.MBps / e.MBps
		verdict := "INFO"
		if c.Mode == "buffered" && c.Test == "raw" && c.Chunk <= m41MaxChunk {
			pairs++
			verdict = "PASS"
			if ratio < m41Ratio {
				verdict, ok = "FAIL", false
			}
		}
		lines = append(lines, fmt.Sprintf("%s %s %s %s QD %d: warm %.0f MB/s, evicted %.0f MB/s, %.1fx",
			verdict, c.Test, c.Mode, size(c.Chunk), c.QD, c.MBps, e.MBps, ratio))
	}
	if pairs == 0 {
		return lines, false
	}
	return lines, ok
}
