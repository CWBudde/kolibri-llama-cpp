// Command kolibri-stream runs one long streamed /completion against a running
// llama-server, timestamps every generated token and, given the server's PID,
// samples its memory and the system's memory pressure while it generates. It
// backs "Sustained generation" in docs/real-checkpoint.md.
//
//	kolibri-stream -n 32000 -pid $(pgrep -n llama-server) -out run/ \
//	    -p "Write a detailed, multi-chapter history of the city of Berlin, from its founding to the present day."
//
// It writes run/tokens.log (one "unix_ms count" line per streamed chunk),
// run/text.txt (the generated text) and run/samples.txt (every -every: time,
// RSS and phys_footprint in MiB, system swap used in MiB, the
// kern.memorystatus_vm_pressure_level and the token count). At the end it
// prints the server's timings and the tokens/s per -window tokens. The
// sampler uses ps, footprint and sysctl, so it is macOS only.
package main

import (
	"bufio"
	"bytes"
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
	fmt.Fprintf(w, "%s %.0f %.0f %.0f %s %d\n", time.Now().Format("15:04:05"), rssKiB/1024, fp, sw, level, tokens)
	return true
}

func main() {
	url := flag.String("url", "http://127.0.0.1:8099/completion", "llama-server /completion endpoint")
	prompt := flag.String("p", "", "prompt")
	n := flag.Int("n", 32000, "tokens to generate (n_predict, with ignore_eos)")
	pid := flag.Int("pid", 0, "llama-server PID to sample; 0 disables sampling")
	every := flag.Duration("every", 10*time.Second, "sampling interval")
	size := flag.Int("window", 2048, "tokens per reported window")
	dir := flag.String("out", ".", "output directory")
	flag.Parse()
	if *prompt == "" {
		flag.Usage()
		os.Exit(2)
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
		fmt.Fprintln(samples, "time rss_mib footprint_mib swap_used_mib pressure tokens")
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
		"prompt": *prompt, "n_predict": *n, "ignore_eos": true, "temperature": 0,
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
	timings, err := readStream(resp.Body, func(c int, content string) {
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
	fmt.Printf("tokens %d, timings %s\n", count.Load(), timings)
	for _, w := range windowRates(stamps, *size) {
		fmt.Printf("%6d-%6d %.1f tokens/s\n", w.from, w.to, w.rate)
	}
}
