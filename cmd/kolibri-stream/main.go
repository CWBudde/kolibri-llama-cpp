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
// kern.memorystatus_vm_pressure_level, the token count and the system's
// page-ins so far). At the end it prints the server's timings, the tokens/s per
// -window tokens and the system's page-ins from the first generated token to
// the last, the pages read from disk while generating. The sampler uses ps,
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
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
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
// token count and the chunk's text, and returns the final chunk's timings.
// A chunk without "tokens" (no return_tokens) counts as one token.
func readStream(r io.Reader, onToken func(count int, content string)) (json.RawMessage, error) {
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
		onToken(count, c.Content)
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
// process is gone.
func sample(w io.Writer, pid int, tokens int) bool {
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
	fmt.Fprintf(w, "%s %.0f %.0f %.0f %s %d %d\n", time.Now().Format("15:04:05"), rssKiB/1024, fp, sw, level, tokens, pageins())
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
	size := flag.Int("window", 2048, "tokens per reported window")
	dir := flag.String("out", ".", "output directory")
	flag.Parse()
	if (*prompt == "") == (*tokens == "") {
		flag.Usage()
		os.Exit(2)
	}
	var promptValue any = *prompt
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
		fmt.Fprintln(samples, "time rss_mib footprint_mib swap_used_mib pressure tokens pageins")
		go func() {
			defer close(sampled)
			tick := time.NewTicker(*every)
			defer tick.Stop()
			for sample(samples, *pid, int(count.Load())) {
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
	firstPageins := int64(-1)
	timings, err := readStream(resp.Body, func(c int, content string) {
		if len(stamps) == 0 {
			firstPageins = pageins()
		}
		now := time.Now()
		count.Store(int64(c))
		stamps = append(stamps, stamp{c, now})
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
	if firstPageins >= 0 && lastPageins >= 0 && len(stamps) > 1 {
		pages, gen := lastPageins-firstPageins, stamps[len(stamps)-1].count-stamps[0].count
		fmt.Printf("system page-ins while generating: %d pages of 16 KiB (%.1f MiB) over %d tokens, %.3f MiB per token\n",
			pages, float64(pages)/64, gen, float64(pages)/64/float64(gen))
	}
	for _, w := range windowRates(stamps, *size) {
		fmt.Printf("%6d-%6d %.1f tokens/s\n", w.from, w.to, w.rate)
	}
}
