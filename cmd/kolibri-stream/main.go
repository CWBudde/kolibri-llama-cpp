// Command kolibri-stream runs one long streamed /completion against a running
// llama-server, timestamps every generated token and, given the server's PID,
// samples its memory and the system's memory pressure while it generates. It
// backs "Sustained generation" in docs/real-checkpoint.md.
//
//	kolibri-stream -n 32000 -pid $(pgrep -n llama-server) -out run/ \
//	    -p "Write a detailed, multi-chapter history of the city of Berlin, from its founding to the present day."
//
// With -tokens the prompt is a locality workload's token IDs instead
// (<workload>.tokens.npy of tools/locality/workloads.py), checked against
// testdata/locality/manifest.json and cut to the first -first tokens:
//
//	kolibri-stream -tokens ~/models/eval/locality/hr.tokens.npy -first 8000 -n 128 -pid $! -out run/
//
// It writes run/tokens.log (one "unix_ms count" line per streamed chunk),
// run/text.txt (the generated text) and run/samples.txt (every -every: time,
// RSS and phys_footprint in MiB, system swap used in MiB, the
// kern.memorystatus_vm_pressure_level, the token count, the system's
// page-ins so far and, with -model, the MiB of that file in the page cache). At the end it prints the server's timings, the tokens/s per
// -window tokens and the system's page-ins from the first generated token to
// the last, the pages read from disk while generating. With -summary FILE it
// also writes the request as JSON for cmd/kolibri-bench: the time to the first
// token, decode tokens/s, the nearest-rank p50, p95 and p99 of the per-token
// latency, the server's timings and the generated token IDs. The sampler uses ps,
// footprint, sysctl and vm_stat, so it is macOS only.
package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"kolibri-llm/internal/locality"
)

type chunk struct {
	Content string          `json:"content"`
	Tokens  []int           `json:"tokens"`
	Stop    bool            `json:"stop"`
	Timings json.RawMessage `json:"timings"`
}

// readStream parses llama-server's SSE stream, calls onToken with the running
// token count, the chunk's token IDs and its text, and returns the final
// chunk's timings. A chunk without "tokens" (no return_tokens) counts as one
// token.
func readStream(r io.Reader, onToken func(count int, ids []int, content string)) (json.RawMessage, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	count := 0
	for sc.Scan() {
		line, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok {
			continue
		}
		var c chunk
		if err := json.Unmarshal([]byte(line), &c); err != nil {
			return nil, fmt.Errorf("chunk %q: %w", line, err)
		}
		if c.Stop {
			return c.Timings, nil
		}
		count += max(len(c.Tokens), 1)
		onToken(count, c.Tokens, c.Content)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return nil, errors.New("stream ended without a stop chunk")
}

type stamp struct {
	count int
	at    time.Time
}

type window struct {
	from, to int
	rate     float64 // tokens/s
}

// windowRates splits the stamps at the first count reaching each multiple of
// size and returns the tokens/s between consecutive boundaries, starting at
// the first stamp and ending with a partial window at the last one.
func windowRates(stamps []stamp, size int) []window {
	if len(stamps) < 2 {
		return nil
	}
	var out []window
	prev := stamps[0]
	next := size
	for i, s := range stamps[1:] {
		last := i == len(stamps)-2
		if s.count < next && !last {
			continue
		}
		if dt := s.at.Sub(prev.at).Seconds(); dt > 0 {
			out = append(out, window{prev.count, s.count, float64(s.count-prev.count) / dt})
		}
		prev = s
		for next <= s.count {
			next += size
		}
	}
	return out
}

// latencies returns one decode latency per token after the first: the time
// between consecutive stamps, split evenly over the tokens of a chunk.
func latencies(stamps []stamp) []time.Duration {
	var out []time.Duration
	for i := 1; i < len(stamps); i++ {
		k := stamps[i].count - stamps[i-1].count
		if k <= 0 {
			continue
		}
		dt := stamps[i].at.Sub(stamps[i-1].at) / time.Duration(k)
		for range k {
			out = append(out, dt)
		}
	}
	return out
}

// percentile is the nearest-rank p-th percentile of d (0 for none); d is not
// changed.
func percentile(d []time.Duration, p float64) time.Duration {
	if len(d) == 0 {
		return 0
	}
	s := slices.Clone(d)
	slices.Sort(s)
	rank := int(math.Ceil(p / 100 * float64(len(s))))
	return s[min(max(rank, 1), len(s))-1]
}

// summary is one request's result, written by -summary for cmd/kolibri-bench.
type summary struct {
	PromptTokens          int     `json:"prompt_tokens"` // 0 for a text prompt
	Generated             int     `json:"generated"`
	TTFTMs                float64 `json:"ttft_ms"` // from sending the request to the first token
	DecodeTokensPerSecond float64 `json:"decode_tokens_per_second"`
	LatencyMs             struct {
		P50 float64 `json:"p50"`
		P95 float64 `json:"p95"`
		P99 float64 `json:"p99"`
		Max float64 `json:"max"`
	} `json:"latency_ms"` // per decoded token after the first
	Timings      json.RawMessage `json:"timings"` // the server's
	Pageins      int64           `json:"pageins"` // the system's while generating; -1 unknown
	TokenIDs     []int           `json:"token_ids"`
	TokensSHA256 string          `json:"tokens_sha256"`
}

func summarize(sent time.Time, stamps []stamp, ids []int, timings json.RawMessage, promptTokens int, pageins int64) summary {
	s := summary{PromptTokens: promptTokens, Timings: timings, Pageins: pageins, TokenIDs: ids}
	if len(stamps) > 0 {
		last := stamps[len(stamps)-1]
		s.Generated = last.count
		s.TTFTMs = msec(stamps[0].at.Sub(sent))
		if dt := last.at.Sub(stamps[0].at).Seconds(); dt > 0 {
			s.DecodeTokensPerSecond = float64(last.count-stamps[0].count) / dt
		}
	}
	lat := latencies(stamps)
	s.LatencyMs.P50 = msec(percentile(lat, 50))
	s.LatencyMs.P95 = msec(percentile(lat, 95))
	s.LatencyMs.P99 = msec(percentile(lat, 99))
	s.LatencyMs.Max = msec(percentile(lat, 100))
	ids32 := make([]int32, len(ids))
	for i, id := range ids {
		ids32[i] = int32(id)
	}
	s.TokensSHA256 = idsSHA256(ids32)
	return s
}

func msec(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

var footprintRe = regexp.MustCompile(`phys_footprint:\s+([0-9.]+)\s+(KB|MB|GB)`)

// parseFootprint returns phys_footprint from `footprint -p PID` in MiB.
func parseFootprint(out string) (float64, error) {
	m := footprintRe.FindStringSubmatch(out)
	if m == nil {
		return 0, errors.New("no phys_footprint line")
	}
	v, err := strconv.ParseFloat(m[1], 64)
	if err != nil {
		return 0, err
	}
	switch m[2] {
	case "KB":
		v /= 1024
	case "GB":
		v *= 1024
	}
	return v, nil
}

var swapRe = regexp.MustCompile(`used = ([0-9.]+)M`)

// parseSwapUsed returns the used swap from `sysctl -n vm.swapusage` in MiB.
func parseSwapUsed(out string) (float64, error) {
	m := swapRe.FindStringSubmatch(out)
	if m == nil {
		return 0, fmt.Errorf("no used swap in %q", out)
	}
	return strconv.ParseFloat(m[1], 64)
}

var pageinsRe = regexp.MustCompile(`(?m)^Pageins:\s+(\d+)\.`)

// parsePageins returns the system's page-ins so far (pages read from disk) from vm_stat.
func parsePageins(out string) (int64, error) {
	m := pageinsRe.FindStringSubmatch(out)
	if m == nil {
		return 0, errors.New("no Pageins line")
	}
	return strconv.ParseInt(m[1], 10, 64)
}

// pageins runs vm_stat; -1 if that fails.
func pageins() int64 {
	out, err := command("vm_stat")
	if err != nil {
		return -1
	}
	n, err := parsePageins(out)
	if err != nil {
		return -1
	}
	return n
}

// idsSHA256 hashes token IDs as tools/locality/workloads.py does: the decimal IDs joined by commas.
func idsSHA256(ids []int32) string {
	parts := make([]string, len(ids))
	for i, id := range ids {
		parts[i] = strconv.Itoa(int(id))
	}
	sum := sha256.Sum256([]byte(strings.Join(parts, ",")))
	return hex.EncodeToString(sum[:])
}

// checkTokens compares a workload's token IDs with its entry in testdata/locality/manifest.json.
func checkTokens(workload string, ids []int32, manifest io.Reader) error {
	var m struct {
		Workloads map[string]struct {
			NTokens      int    `json:"n_tokens"`
			TokensSHA256 string `json:"tokens_sha256"`
		} `json:"workloads"`
	}
	if err := json.NewDecoder(manifest).Decode(&m); err != nil {
		return fmt.Errorf("manifest: %w", err)
	}
	w, ok := m.Workloads[workload]
	if !ok {
		return fmt.Errorf("%s: no such workload in the manifest", workload)
	}
	if got := idsSHA256(ids); len(ids) != w.NTokens || got != w.TokensSHA256 {
		return fmt.Errorf("%s: %d token IDs with sha256 %s, the manifest records %d with %s",
			workload, len(ids), got, w.NTokens, w.TokensSHA256)
	}
	return nil
}

func command(name string, args ...string) (string, error) {
	out, err := exec.Command(name, args...).Output()
	return strings.TrimSpace(string(out)), err
}

// sample appends one line for the process pid to w; it returns false once the
// process is gone. The last column is the page cache's share of the model
// file in MiB, "-" without a model or off macOS.
func sample(w io.Writer, pid int, tokens int, model string) bool {
	p := strconv.Itoa(pid)
	rss, err := command("ps", "-o", "rss=", "-p", p)
	if err != nil {
		return false
	}
	rssKiB, _ := strconv.ParseFloat(rss, 64)
	fpOut, _ := command("footprint", "-p", p)
	fp, _ := parseFootprint(fpOut)
	swOut, _ := command("sysctl", "-n", "vm.swapusage")
	sw, _ := parseSwapUsed(swOut)
	level, _ := command("sysctl", "-n", "kern.memorystatus_vm_pressure_level")
	cached := "-"
	if model != "" {
		if v, err := pageCacheMiB(model); err == nil {
			cached = fmt.Sprintf("%.0f", v)
		}
	}
	fmt.Fprintf(w, "%s %.0f %.0f %.0f %s %d %d %s\n", time.Now().Format("15:04:05"), rssKiB/1024, fp, sw, level, tokens, pageins(), cached)
	return true
}

func main() {
	url := flag.String("url", "http://127.0.0.1:8099/completion", "llama-server /completion endpoint")
	prompt := flag.String("p", "", "prompt")
	tokens := flag.String("tokens", "", "a workload's <name>.tokens.npy, sent as the prompt instead of -p")
	first := flag.Int("first", 0, "with -tokens: send only the first tokens (0: all)")
	manifestPath := flag.String("manifest", "testdata/locality/manifest.json", "with -tokens: the workloads' recorded token IDs")
	n := flag.Int("n", 32000, "tokens to generate (n_predict, with ignore_eos)")
	pid := flag.Int("pid", 0, "llama-server PID to sample; 0 disables sampling")
	every := flag.Duration("every", 10*time.Second, "sampling interval")
	model := flag.String("model", "", "with -pid: also sample how much of this file (the GGUF) the page cache holds")
	size := flag.Int("window", 2048, "tokens per reported window")
	dir := flag.String("out", ".", "output directory")
	summaryPath := flag.String("summary", "", "also write the request's summary as JSON: latency percentiles, tokens/s, token IDs")
	flag.Parse()
	if (*prompt == "") == (*tokens == "") {
		flag.Usage()
		os.Exit(2)
	}
	var promptValue any = *prompt
	promptTokens := 0
	if *tokens != "" {
		f, err := os.Open(*tokens)
		if err != nil {
			log.Fatal(err)
		}
		ids, err := locality.ReadTokens(f)
		f.Close()
		if err != nil {
			log.Fatalf("%s: %v", *tokens, err)
		}
		mf, err := os.Open(*manifestPath)
		if err != nil {
			log.Fatal(err)
		}
		err = checkTokens(strings.TrimSuffix(filepath.Base(*tokens), ".tokens.npy"), ids, mf)
		mf.Close()
		if err != nil {
			log.Fatalf("FAIL %v", err)
		}
		if *first > 0 && *first < len(ids) {
			ids = ids[:*first]
		}
		promptValue = ids
		promptTokens = len(ids)
		fmt.Printf("prompt: %d token IDs of %s, as recorded\n", len(ids), *tokens)
	}
	if err := os.MkdirAll(*dir, 0o755); err != nil {
		log.Fatal(err)
	}
	create := func(name string) *os.File {
		f, err := os.Create(filepath.Join(*dir, name))
		if err != nil {
			log.Fatal(err)
		}
		return f
	}
	tokLog, text := create("tokens.log"), create("text.txt")
	defer tokLog.Close()
	defer text.Close()

	var count atomic.Int64
	done := make(chan struct{})
	sampled := make(chan struct{})
	if *pid > 0 {
		samples := create("samples.txt")
		defer samples.Close()
		fmt.Fprintln(samples, "time rss_mib footprint_mib swap_used_mib pressure tokens pageins page_cache_mib")
		go func() {
			defer close(sampled)
			tick := time.NewTicker(*every)
			defer tick.Stop()
			for sample(samples, *pid, int(count.Load()), *model) {
				select {
				case <-done:
					return
				case <-tick.C:
				}
			}
		}()
	} else {
		close(sampled)
	}

	body, err := json.Marshal(map[string]any{
		"prompt": promptValue, "n_predict": *n, "ignore_eos": true, "temperature": 0,
		"stream": true, "return_tokens": true, "cache_prompt": false,
	})
	if err != nil {
		log.Fatal(err)
	}
	sent := time.Now()
	resp, err := http.Post(*url, "application/json", bytes.NewReader(body))
	if err != nil {
		log.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(resp.Body)
		log.Fatalf("%s: %s", resp.Status, msg)
	}

	var stamps []stamp
	var ids []int
	firstPageins := int64(-1)
	timings, err := readStream(resp.Body, func(c int, chunkIDs []int, content string) {
		if len(stamps) == 0 {
			firstPageins = pageins()
		}
		now := time.Now()
		count.Store(int64(c))
		stamps = append(stamps, stamp{c, now})
		ids = append(ids, chunkIDs...)
		fmt.Fprintf(tokLog, "%d %d\n", now.UnixMilli(), c)
		text.WriteString(content)
	})
	close(done)
	<-sampled
	if err != nil {
		log.Fatal(err)
	}
	lastPageins := pageins()
	fmt.Printf("tokens %d, timings %s\n", count.Load(), timings)
	pages := int64(-1)
	if firstPageins >= 0 && lastPageins >= 0 && len(stamps) > 1 {
		gen := stamps[len(stamps)-1].count - stamps[0].count
		pages = lastPageins - firstPageins
		fmt.Printf("system page-ins while generating: %d pages of 16 KiB (%.1f MiB) over %d tokens, %.3f MiB per token\n",
			pages, float64(pages)/64, gen, float64(pages)/64/float64(gen))
	}
	if *summaryPath != "" {
		s := summarize(sent, stamps, ids, timings, promptTokens, pages)
		fmt.Printf("time to first token %.1f ms; decode %.2f tokens/s, latency p50 %.2f ms, p95 %.2f ms, max %.2f ms\n",
			s.TTFTMs, s.DecodeTokensPerSecond, s.LatencyMs.P50, s.LatencyMs.P95, s.LatencyMs.Max)
		out, err := json.MarshalIndent(s, "", "  ")
		if err != nil {
			log.Fatal(err)
		}
		if err := os.WriteFile(*summaryPath, append(out, '\n'), 0o644); err != nil {
			log.Fatal(err)
		}
	}
	for _, w := range windowRates(stamps, *size) {
		fmt.Printf("%6d-%6d %.1f tokens/s\n", w.from, w.to, w.rate)
	}
}
