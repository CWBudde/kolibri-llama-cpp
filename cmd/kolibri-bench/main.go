// Command kolibri-bench runs the expert cache benchmark of PLAN Phase 10,
// Stage D (docs/expert-streaming.md, "Benchmark"): every arm of a matrix
// (testdata/bench/expert-cache.json) at every memory ceiling, warm and cold,
// with llama-server, and then the report with the output equivalence and the
// performance gates.
//
//	go run ./cmd/kolibri-bench -matrix testdata/bench/expert-cache.json -out ~/models/eval/expert-bench
//	go run ./cmd/kolibri-bench -matrix testdata/bench/expert-cache.json -out ~/models/eval/expert-bench -report
//
// A config runs the fork's -server unless it names another build ("server"),
// which -servers maps to a binary: pr25294=<llama-server> for E0.2's matrix
// (testdata/bench/e0.json, docs/upstream.md). Only the fork's server is asked
// for --expert-cache-stats.
//
// Every run of a config, a ceiling and a repetition:
//  1. waits until the machine is quiet: a 1-minute load below -quiet-load,
//     swap growing by less than -quiet-swap MiB over the last minute, and
//     AC power;
//  2. evicts the GGUF from the page cache (cmd/kolibri-evict), for a cold start;
//  3. at a ceiling above 0, wires that much memory (cmd/kolibri-memhog);
//  4. starts llama-server and times it until /health answers;
//  5. sends the cold request, then the warm ones (cmd/kolibri-stream -tokens
//     -summary): the first prompt_first tokens of a workload, n_predict tokens
//     generated at temperature 0;
//  6. stops the server with SIGINT, so it writes --expert-cache-stats, and
//     the hog.
//
// A run starts only on AC power, and the machine is kept from sleep
// (caffeinate -i -s) while the driver runs.
//
// Each run writes <out>/<config>/<ceiling>g/r<rep>/run.json; a run that has
// one is skipped, so an interrupted benchmark resumes. A server that fails to
// load or a request that fails is recorded, not fatal: at a ceiling it is a
// result. A resident run whose wired weights and hog exceed the memory, or a
// cache run whose slots do not fit beside the hog, is recorded as "does not
// fit" without starting it (see fits). The report goes to stdout and
// <out>/report.md; it exits 1 if an exact equivalence fails or a cache run
// failed to load an expert.
package main

import (
	"bufio"
	"context"
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
	"slices"
	"strconv"
	"strings"
	"syscall"
	"time"

	"kolibri-llm/internal/gguf"
)

type config struct {
	Name     string    `json:"name"`
	Model    string    `json:"model"`
	Server   string    `json:"server,omitempty"` // a build named in -servers; empty: the fork's -server
	Args     []string  `json:"args"`
	Ceilings []float64 `json:"ceilings_gib"`
	Reps     int       `json:"reps"`
}

// equivalence compares the arms' token IDs with the reference's; Exact makes
// a difference a failure, otherwise it is INFO.
type equivalence struct {
	Reference string   `json:"reference"`
	Arms      []string `json:"arms"`
	Exact     bool     `json:"exact"`
}

// gate is one performance comparison; Decides marks the plan's gate, the
// others are INFO.
type gate struct {
	CeilingGiB float64  `json:"ceiling_gib"`
	Candidate  string   `json:"candidate"`
	Baselines  []string `json:"baselines"`
	Decides    bool     `json:"decides"`
}

type matrix struct {
	ServerArgs  []string      `json:"server_args"`
	PromptFirst int           `json:"prompt_first"`
	NPredict    int           `json:"n_predict"`
	Cold        string        `json:"cold"`
	Warm        []string      `json:"warm"`
	Configs     []config      `json:"configs"`
	Equivalence []equivalence `json:"equivalence"`
	Gates       []gate        `json:"gates"`
}

type runSpec struct {
	Config  config
	Ceiling float64
	Rep     int
}

func runKey(cfg string, ceiling float64, rep int) string {
	return fmt.Sprintf("%s/%gg/r%d", cfg, ceiling, rep)
}

func (r runSpec) key() string { return runKey(r.Config.Name, r.Ceiling, r.Rep) }

// expand lists the runs with the repetitions outermost, so that drift of the
// machine spreads over the arms.
func (m matrix) expand() []runSpec {
	var out []runSpec
	for rep := 1; ; rep++ {
		n := len(out)
		for _, c := range m.Configs {
			if rep > c.Reps {
				continue
			}
			for _, ceil := range c.Ceilings {
				out = append(out, runSpec{c, ceil, rep})
			}
		}
		if len(out) == n {
			return out
		}
	}
}

func (m matrix) config(name string) (config, bool) {
	for _, c := range m.Configs {
		if c.Name == name {
			return c, true
		}
	}
	return config{}, false
}

func usesCache(args []string) bool { return slices.Contains(args, "--expert-cache") }

// slotArg finds an arm's slot cache: this fork's --expert-cache N or #25294's
// --moe-stream-cache Ns, with its value ("" if the flag ends the args).
func slotArg(args []string) (flag, arg string, ok bool) {
	for i, a := range args {
		if a == "--expert-cache" || a == "--moe-stream-cache" {
			if i+1 < len(args) {
				arg = args[i+1]
			}
			return a, arg, true
		}
	}
	return "", "", false
}

// weights is what fits needs from a GGUF.
type weights struct {
	Bytes       int64 // the file
	ExpertBytes int64 // the routed experts' tensors (ffn_{gate,up,down}_exps) of all layers
	SlotBytes   int64 // one expert of every layer: what one cache slot per layer holds
}

// weightsOf sums a GGUF's expert tensors; size is the file's size.
func weightsOf(f *gguf.File, size int64) (weights, error) {
	w := weights{Bytes: size}
	for _, t := range f.Tensors {
		if !expertTensor.MatchString(t.Name) {
			continue
		}
		if len(t.Dims) != 3 || t.Dims[2] == 0 {
			return w, fmt.Errorf("%s: dims %v, want [n_in, n_out, n_expert]", t.Name, t.Dims)
		}
		n, err := t.Bytes()
		if err != nil {
			return w, err
		}
		w.ExpertBytes += n
		w.SlotBytes += n / int64(t.Dims[2])
	}
	return w, nil
}

var expertTensor = regexp.MustCompile(`^blk\.\d+\.ffn_(gate|up|down)_exps\.weight$`)

// osReserveGiB is the memory left to macOS and everything else beside a
// cache run's slots, its non-expert weights and the hog. Q8_0's 128 slots
// (24.9 GiB), its non-expert weights (2.7 GiB) and a 16 GiB hog left
// 4.4 GiB: the slots were paged out and the cold request produced no token
// in 20 minutes (2026-10-23). IQ3's cache runs, at 33.1 GiB or less, completed.
const osReserveGiB = 8

// fits reports whether a run can start: with neither --cpu-moe nor
// --expert-cache, -ngl 99 puts every weight into Metal buffers, which are
// wired like the hog's memory, and the two together must stay below the
// physical memory. Beyond it macOS does not fail the load, it hangs: on
// 2026-10-15 IQ3 resident beside a 16 GiB hog ended in a watchdog panic.
// With --cpu-moe the experts stay in the page cache and always fit. The
// cache's slots are anonymous memory that macOS can page out, which keeps
// the machine up but turns hits into page-ins; they fit only with the
// non-expert weights, the hog and osReserveGiB beside them. #25294's
// --moe-stream-cache Ns slots are Metal buffers, wired, and count the same.
func fits(c config, ceilingGiB float64, w weights, memBytes int64) (bool, string) {
	const gib = 1 << 30
	mem := float64(memBytes) / gib
	if slices.Contains(c.Args, "--cpu-moe") {
		return true, ""
	}
	flag, arg, ok := slotArg(c.Args)
	if !ok {
		need := float64(w.Bytes)/gib + ceilingGiB
		if need < mem {
			return true, ""
		}
		return false, fmt.Sprintf("the resident weights (%.1f GiB) and the %g GiB hog are %.1f GiB wired, beyond the %.0f GiB of memory",
			float64(w.Bytes)/gib, ceilingGiB, need, mem)
	}
	n := arg
	if flag == "--moe-stream-cache" {
		n = strings.TrimSuffix(arg, "s")
		if n == arg {
			n = "" // a budget in GiB, not slots
		}
	}
	slots, err := strconv.Atoi(n)
	if err != nil || slots < 0 {
		return false, fmt.Sprintf("%s %q is not a slot count", flag, arg)
	}
	cache := float64(int64(slots)*w.SlotBytes) / gib
	rest := float64(w.Bytes-w.ExpertBytes) / gib
	need := rest + cache + ceilingGiB + osReserveGiB
	if need < mem {
		return true, ""
	}
	return false, fmt.Sprintf("the non-expert weights (%.1f GiB), %d cache slots (%.1f GiB), the %g GiB hog and %d GiB for the system are %.1f GiB, beyond the %.0f GiB of memory",
		rest, slots, cache, ceilingGiB, osReserveGiB, need, mem)
}

// serverArgs asks every arm for --expert-cache-stats: with the cache the
// cache's counters and the decode timing, without it (patch 0012) the decode
// timing alone, so that resident and cache arms split their ms per token alike.
// Another build (c.Server) has no such option and writes no stats.
func (m matrix) serverArgs(c config, port int, stats string) []string {
	args := append([]string{"-m", c.Model, "--port", strconv.Itoa(port)}, m.ServerArgs...)
	args = append(args, c.Args...)
	if c.Server != "" {
		return args
	}
	return append(args, "--expert-cache-stats", stats)
}

// parseServers reads -servers: comma-separated name=path pairs of the other
// llama-server builds that configs name.
func parseServers(s string) (map[string]string, error) {
	servers := map[string]string{}
	if s == "" {
		return servers, nil
	}
	for _, kv := range strings.Split(s, ",") {
		name, path, ok := strings.Cut(kv, "=")
		if !ok || name == "" || path == "" {
			return nil, fmt.Errorf("-servers: %q is not name=path", kv)
		}
		if _, dup := servers[name]; dup {
			return nil, fmt.Errorf("-servers: %s given twice", name)
		}
		servers[name] = path
	}
	return servers, nil
}

// missingServers lists the builds that configs name but servers lacks, of
// the configs in only (-only), or of all if only is nil.
func (m matrix) missingServers(servers map[string]string, only []string) []string {
	var missing []string
	for _, c := range m.Configs {
		if only != nil && !slices.Contains(only, c.Name) {
			continue
		}
		if _, ok := servers[c.Server]; c.Server != "" && !ok && !slices.Contains(missing, c.Server) {
			missing = append(missing, c.Server)
		}
	}
	return missing
}

const (
	statusOK            = "ok"
	statusLoadFailed    = "load failed"
	statusRequestFailed = "request failed"
	statusDoesNotFit    = "does not fit" // not started, see fits
)

type request struct {
	Phase    string         `json:"phase"`
	Workload string         `json:"workload"`
	Summary  *streamSummary `json:"summary,omitempty"`
	Error    string         `json:"error,omitempty"`
}

type runResult struct {
	Config      string    `json:"config"`
	CeilingGiB  float64   `json:"ceiling_gib"`
	Rep         int       `json:"rep"`
	Started     time.Time `json:"started"`
	Status      string    `json:"status"`
	Error       string    `json:"error,omitempty"`
	LogTail     string    `json:"log_tail,omitempty"`
	QuietWaitS  float64   `json:"quiet_wait_seconds"`
	Load1       float64   `json:"load1_at_start"`
	SwapMiB     float64   `json:"swap_used_mib_at_start"`
	Evict       string    `json:"evict"`
	Hog         string    `json:"hog,omitempty"`
	LoadSeconds float64   `json:"load_seconds"`
	OthersCPU   struct {
		Mean float64 `json:"mean"`
		Max  float64 `json:"max"`
	} `json:"others_cpu_percent"` // the other processes' CPU while the server ran, in % of one core
	PeakRSSMiB       float64 `json:"peak_rss_mib"`
	PeakFootprintMiB float64 `json:"peak_footprint_mib"`
	// the model file's pages in the page cache: the largest sample during the
	// requests, and once after the server stopped (nil: not measured)
	PeakPageCacheMiB *float64    `json:"peak_page_cache_mib,omitempty"`
	EndPageCacheMiB  *float64    `json:"end_page_cache_mib,omitempty"`
	Requests         []request   `json:"requests"`
	Stats            *cacheStats `json:"stats,omitempty"`
}

func (r runResult) key() string { return runKey(r.Config, r.CeilingGiB, r.Rep) }

// machine is one sample of the machine's state.
type machine struct {
	Load1   float64
	SwapMiB float64
	OnAC    bool
}

type quietRule struct {
	MaxLoad          float64
	MaxSwapGrowthMiB float64
	Window           int // samples back to compare the swap with
}

// quiet holds if the last sample's load is below the limit, swap grew by
// less than the limit over the window, and the machine runs on AC power: on
// battery a closed Mac sleeps despite caffeinate, and the valid runs were all
// on AC.
func (q quietRule) quiet(s []machine) bool {
	if len(s) <= q.Window {
		return false
	}
	last, then := s[len(s)-1], s[len(s)-1-q.Window]
	return last.OnAC && last.Load1 < q.MaxLoad && last.SwapMiB-then.SwapMiB < q.MaxSwapGrowthMiB
}

// parseOnAC reads pmset -g ps.
func parseOnAC(s string) bool { return strings.Contains(s, "'AC Power'") }

func parseLoadavg(s string) (float64, error) {
	f := strings.Fields(strings.Trim(strings.TrimSpace(s), "{}"))
	if len(f) == 0 {
		return 0, fmt.Errorf("vm.loadavg %q", s)
	}
	return strconv.ParseFloat(f[0], 64)
}

var swapRe = regexp.MustCompile(`used = ([0-9.]+)M`)

func parseSwapUsed(s string) (float64, error) {
	m := swapRe.FindStringSubmatch(s)
	if m == nil {
		return 0, fmt.Errorf("no used swap in %q", s)
	}
	return strconv.ParseFloat(m[1], 64)
}

func sample() (machine, error) {
	la, err := exec.Command("sysctl", "-n", "vm.loadavg").Output()
	if err != nil {
		return machine{}, err
	}
	sw, err := exec.Command("sysctl", "-n", "vm.swapusage").Output()
	if err != nil {
		return machine{}, err
	}
	var m machine
	if m.Load1, err = parseLoadavg(string(la)); err != nil {
		return m, err
	}
	if m.SwapMiB, err = parseSwapUsed(string(sw)); err != nil {
		return m, err
	}
	ps, err := exec.Command("pmset", "-g", "ps").Output()
	m.OnAC = parseOnAC(string(ps))
	return m, err
}

// waitQuiet samples every interval until the rule holds and returns the wait.
func waitQuiet(q quietRule, every time.Duration, logf func(string, ...any)) (time.Duration, machine, error) {
	start := time.Now()
	var hist []machine
	for {
		m, err := sample()
		if err != nil {
			return 0, m, err
		}
		hist = append(hist, m)
		if q.quiet(hist) {
			return time.Since(start), m, nil
		}
		if len(hist)%10 == 0 {
			logf("waiting for a quiet machine for %s: load %.2f, swap %.0f MiB, on AC %v", time.Since(start).Round(time.Second), m.Load1, m.SwapMiB, m.OnAC)
		}
		time.Sleep(every)
	}
}

// othersCPU sums the %CPU of all processes except the excluded ones.
func othersCPU(exclude []int) (float64, error) {
	out, err := exec.Command("ps", "-A", "-o", "pid=,%cpu=").Output()
	if err != nil {
		return 0, err
	}
	sum := 0.0
	for line := range strings.SplitSeq(string(out), "\n") {
		f := strings.Fields(line)
		if len(f) != 2 {
			continue
		}
		pid, _ := strconv.Atoi(f[0])
		if slices.Contains(exclude, pid) {
			continue
		}
		v, _ := strconv.ParseFloat(f[1], 64)
		sum += v
	}
	return sum, nil
}

type bench struct {
	m         matrix
	out       string
	bin       string            // the built kolibri-stream, -evict and -memhog
	servers   map[string]string // the llama-server builds by config.Server; "" is -server
	manifest  string
	workloads string
	port      int
	quiet     quietRule
	every     time.Duration
	reqTO     time.Duration
	loadTO    time.Duration
	memBytes  int64 // hw.memsize
}

func (b *bench) logf(format string, args ...any) {
	log.Printf(format, args...)
}

// tail returns the last n lines of a file.
func tail(path string, n int) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	return strings.Join(lines[max(0, len(lines)-n):], "\n")
}

// startHog wires memory and waits for its "wired" line.
func (b *bench) startHog(gib float64, dir string) (*exec.Cmd, string, error) {
	cmd := exec.Command(filepath.Join(b.bin, "kolibri-memhog"), "-wire", fmt.Sprintf("%gGiB", gib))
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, "", err
	}
	logf, err := os.Create(filepath.Join(dir, "hog.txt"))
	if err != nil {
		return nil, "", err
	}
	cmd.Stderr = logf
	if err := cmd.Start(); err != nil {
		logf.Close()
		return nil, "", err
	}
	line, err := bufio.NewReader(stdout).ReadString('\n')
	fmt.Fprint(logf, line)
	logf.Close()
	if err != nil || !strings.HasPrefix(line, "wired ") {
		cmd.Process.Kill()
		cmd.Wait()
		return nil, "", fmt.Errorf("kolibri-memhog: %q: %v; %s", strings.TrimSpace(line), err, tail(filepath.Join(dir, "hog.txt"), 5))
	}
	go io.Copy(io.Discard, stdout)
	return cmd, strings.TrimSpace(line), nil
}

func stop(cmd *exec.Cmd, done <-chan error, timeout time.Duration) {
	cmd.Process.Signal(syscall.SIGINT)
	select {
	case <-done:
	case <-time.After(timeout):
		cmd.Process.Kill()
		<-done
	}
}

// run runs one spec and writes its run.json. An error is the driver's own,
// not a result: it is returned only after the server and the hog are
// stopped, and no run.json is written, so a resume repeats the run.
func (b *bench) run(spec runSpec) (res runResult, err error) {
	dir := filepath.Join(b.out, spec.key())
	res = runResult{Config: spec.Config.Name, CeilingGiB: spec.Ceiling, Rep: spec.Rep, Status: statusOK}
	defer func() {
		if err != nil {
			return
		}
		data, merr := json.MarshalIndent(res, "", "  ")
		if merr == nil {
			merr = os.WriteFile(filepath.Join(dir, "run.json"), append(data, '\n'), 0o644)
		}
		err = merr
	}()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return res, err
	}
	st, err := os.Stat(spec.Config.Model)
	if err != nil {
		return res, err
	}
	f, err := gguf.Open(spec.Config.Model)
	if err != nil {
		return res, err
	}
	w, err := weightsOf(f, st.Size())
	if err != nil {
		return res, err
	}
	if ok, why := fits(spec.Config, spec.Ceiling, w, b.memBytes); !ok {
		res.Status, res.Error = statusDoesNotFit, why
		return res, nil
	}

	wait, m, err := waitQuiet(b.quiet, b.every, b.logf)
	if err != nil {
		return res, err
	}
	res.QuietWaitS, res.Load1, res.SwapMiB = wait.Seconds(), m.Load1, m.SwapMiB
	res.Started = time.Now()

	ev, err := exec.Command(filepath.Join(b.bin, "kolibri-evict"), spec.Config.Model).CombinedOutput()
	if err != nil {
		return res, fmt.Errorf("kolibri-evict: %v: %s", err, ev)
	}
	res.Evict = strings.TrimSpace(string(ev))

	var hogPids []int
	if spec.Ceiling > 0 {
		hog, line, err := b.startHog(spec.Ceiling, dir)
		if err != nil {
			return res, err
		}
		res.Hog = line
		hogPids = append(hogPids, hog.Process.Pid)
		defer func() {
			hog.Process.Signal(syscall.SIGINT)
			hog.Wait()
		}()
	}

	serverLog := filepath.Join(dir, "server.log")
	lf, err := os.Create(serverLog)
	if err != nil {
		return res, err
	}
	defer lf.Close()
	srv := exec.Command(b.servers[spec.Config.Server], b.m.serverArgs(spec.Config, b.port, filepath.Join(dir, "stats.json"))...)
	srv.Stdout, srv.Stderr = lf, lf
	t0 := time.Now()
	if err := srv.Start(); err != nil {
		return res, err
	}
	done := make(chan error, 1)
	go func() { done <- srv.Wait() }()
	exited := false
	defer func() {
		if !exited {
			stop(srv, done, 3*time.Minute)
		}
		// what the page cache keeps of the file once the server is gone: a
		// hidden tier that phys_footprint does not show
		if v, err := pageCacheMiB(spec.Config.Model); err == nil {
			res.EndPageCacheMiB = &v
		}
		if data, err := os.ReadFile(filepath.Join(dir, "stats.json")); err == nil {
			var st cacheStats
			if json.Unmarshal(data, &st) == nil {
				res.Stats = &st
			}
		}
	}()

	cpuDone := make(chan struct{})
	cpuStopped := make(chan struct{})
	go func() {
		defer close(cpuStopped)
		exclude := append([]int{os.Getpid(), srv.Process.Pid}, hogPids...)
		var sum float64
		n := 0
		for {
			if v, err := othersCPU(exclude); err == nil {
				sum += v
				n++
				res.OthersCPU.Max = max(res.OthersCPU.Max, v)
				res.OthersCPU.Mean = sum / float64(n)
			}
			select {
			case <-cpuDone:
				return
			case <-time.After(5 * time.Second):
			}
		}
	}()
	defer func() {
		close(cpuDone)
		<-cpuStopped
	}()

	health := fmt.Sprintf("http://127.0.0.1:%d/health", b.port)
	deadline := time.Now().Add(b.loadTO)
	for {
		select {
		case err := <-done:
			exited = true
			res.Status, res.Error, res.LogTail = statusLoadFailed, fmt.Sprintf("llama-server exited while loading: %v", err), tail(serverLog, 15)
			return res, nil
		default:
		}
		if resp, err := http.Get(health); err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				break
			}
		}
		if time.Now().After(deadline) {
			res.Status, res.Error, res.LogTail = statusLoadFailed, fmt.Sprintf("no /health after %s", b.loadTO), tail(serverLog, 15)
			return res, nil
		}
		time.Sleep(250 * time.Millisecond)
	}
	res.LoadSeconds = time.Since(t0).Seconds()
	b.logf("%s: loaded in %.1f s", spec.key(), res.LoadSeconds)

	phases := []request{{Phase: "cold", Workload: b.m.Cold}}
	for _, w := range b.m.Warm {
		phases = append(phases, request{Phase: "warm", Workload: w})
	}
	for _, q := range phases {
		rdir := filepath.Join(dir, q.Phase+"-"+q.Workload)
		ctx, cancel := context.WithTimeout(context.Background(), b.reqTO)
		cmd := exec.CommandContext(ctx, filepath.Join(b.bin, "kolibri-stream"),
			"-url", fmt.Sprintf("http://127.0.0.1:%d/completion", b.port),
			"-tokens", filepath.Join(b.workloads, q.Workload+".tokens.npy"), "-first", strconv.Itoa(b.m.PromptFirst),
			"-manifest", b.manifest, "-n", strconv.Itoa(b.m.NPredict),
			"-pid", strconv.Itoa(srv.Process.Pid), "-model", spec.Config.Model, "-every", "2s", "-window", "1000000",
			"-out", rdir, "-summary", filepath.Join(rdir, "summary.json"))
		out, err := cmd.CombinedOutput()
		cancel()
		if werr := os.WriteFile(filepath.Join(rdir+".txt"), out, 0o644); werr != nil {
			return res, werr
		}
		if err != nil {
			q.Error = fmt.Sprintf("%v: %s", err, tail(rdir+".txt", 3))
			res.Requests = append(res.Requests, q)
			res.Status, res.Error, res.LogTail = statusRequestFailed, q.Error, tail(serverLog, 15)
			return res, nil
		}
		data, err := os.ReadFile(filepath.Join(rdir, "summary.json"))
		if err != nil {
			return res, err
		}
		var s streamSummary
		if err := json.Unmarshal(data, &s); err != nil {
			return res, fmt.Errorf("%s: %v", filepath.Join(rdir, "summary.json"), err)
		}
		q.Summary = &s
		res.Requests = append(res.Requests, q)
		if f, err := os.Open(filepath.Join(rdir, "samples.txt")); err == nil {
			pk := peaks(f)
			res.PeakRSSMiB, res.PeakFootprintMiB = max(res.PeakRSSMiB, pk.RSS), max(res.PeakFootprintMiB, pk.Footprint)
			if pk.PageCache >= 0 && (res.PeakPageCacheMiB == nil || pk.PageCache > *res.PeakPageCacheMiB) {
				res.PeakPageCacheMiB = &pk.PageCache
			}
			f.Close()
		}
		b.logf("%s: %s %s: prompt %.1f t/s, decode %.2f t/s, p95 %.1f ms", spec.key(), q.Phase, q.Workload,
			s.Timings.PromptPerSecond, s.DecodeTokensPerSecond, s.LatencyMs.P95)
	}
	return res, nil
}

func expandHome(p string) string {
	if rest, ok := strings.CutPrefix(p, "~/"); ok {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, rest)
		}
	}
	return p
}

func main() {
	matrixPath := flag.String("matrix", "testdata/bench/expert-cache.json", "the benchmark matrix")
	out := flag.String("out", "", "results directory")
	server := flag.String("server", "third_party/llama.cpp/build/bin/llama-server", "llama-server")
	otherServers := flag.String("servers", "", "the other llama-server builds that configs name, as name=path,...")
	workloads := flag.String("workloads", "~/models/eval/locality", "the workloads' <name>.tokens.npy")
	manifest := flag.String("manifest", "testdata/locality/manifest.json", "the workloads' recorded token IDs")
	port := flag.Int("port", 8099, "llama-server port")
	quietLoad := flag.Float64("quiet-load", 3, "start a run only below this 1-minute load")
	quietSwap := flag.Float64("quiet-swap", 100, "and only if swap grew by less than this many MiB in the last minute")
	only := flag.String("only", "", "run only the configs with these comma-separated names")
	reportOnly := flag.Bool("report", false, "only write the report of the runs in -out")
	flag.Parse()
	if *out == "" {
		flag.Usage()
		os.Exit(2)
	}
	data, err := os.ReadFile(*matrixPath)
	if err != nil {
		log.Fatal(err)
	}
	var m matrix
	if err := json.Unmarshal(data, &m); err != nil {
		log.Fatalf("%s: %v", *matrixPath, err)
	}
	for i := range m.Configs {
		m.Configs[i].Model = expandHome(m.Configs[i].Model)
	}
	dir := expandHome(*out)

	if !*reportOnly {
		abs := func(p string) string {
			a, err := filepath.Abs(expandHome(p))
			if err != nil {
				log.Fatal(err)
			}
			return a
		}
		var names []string
		if *only != "" {
			names = strings.Split(*only, ",")
		}
		servers, err := parseServers(*otherServers)
		if err != nil {
			log.Fatal(err)
		}
		if missing := m.missingServers(servers, names); len(missing) > 0 {
			log.Fatalf("%s: configs name the builds %v, which -servers does not give", *matrixPath, missing)
		}
		for name, path := range servers {
			servers[name] = abs(path)
		}
		servers[""] = abs(*server)
		b := &bench{m: m, out: dir, bin: filepath.Join(dir, "bin"), servers: servers, manifest: abs(*manifest),
			workloads: abs(*workloads), port: *port, quiet: quietRule{*quietLoad, *quietSwap, 2},
			every: 30 * time.Second, reqTO: 30 * time.Minute, loadTO: 20 * time.Minute}
		// rates timed across a sleep are meaningless; on 2026-10-16 idle
		// sleep spread four runs over hours
		awake := exec.Command("caffeinate", "-i", "-s", "-w", strconv.Itoa(os.Getpid()))
		if err := awake.Start(); err != nil {
			log.Fatalf("caffeinate: %v", err)
		}
		go awake.Wait()
		mem, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
		if err != nil {
			log.Fatalf("sysctl hw.memsize: %v", err)
		}
		if b.memBytes, err = strconv.ParseInt(strings.TrimSpace(string(mem)), 10, 64); err != nil {
			log.Fatalf("hw.memsize %q: %v", mem, err)
		}
		build := exec.Command("go", "build", "-o", b.bin+"/", "./cmd/kolibri-stream", "./cmd/kolibri-evict", "./cmd/kolibri-memhog")
		build.Stdout, build.Stderr = os.Stderr, os.Stderr
		if err := build.Run(); err != nil {
			log.Fatalf("building the tools: %v", err)
		}
		specs := m.expand()
		for i, spec := range specs {
			if names != nil && !slices.Contains(names, spec.Config.Name) {
				continue
			}
			if _, err := os.Stat(filepath.Join(dir, spec.key(), "run.json")); err == nil {
				continue
			} else if !errors.Is(err, os.ErrNotExist) {
				log.Fatal(err)
			}
			b.logf("run %d of %d: %s", i+1, len(specs), spec.key())
			r, err := b.run(spec)
			if err != nil {
				log.Fatalf("%s: %v", spec.key(), err)
			}
			b.logf("%s: %s %s (others' CPU %.0f%% mean, %.0f%% max)", spec.key(), r.Status, r.Error, r.OthersCPU.Mean, r.OthersCPU.Max)
		}
	}

	runs, err := loadResults(dir)
	if err != nil {
		log.Fatal(err)
	}
	f, err := os.Create(filepath.Join(dir, "report.md"))
	if err != nil {
		log.Fatal(err)
	}
	ok := report(io.MultiWriter(os.Stdout, f), m, runs)
	if err := f.Close(); err != nil {
		log.Fatal(err)
	}
	if !ok {
		os.Exit(1)
	}
}
