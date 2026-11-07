package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const recordSchema = 2

// Fixed settings of a baseline run. n_predict is longer with thinking on,
// because the reasoning has to close before an answer can be scored.
const (
	defaultNPredict      = 256
	defaultNPredictThink = 768
	defaultSeed          = 42
	defaultNProbs        = 10
	// lossNProbs asks for every candidate: the server caps it at the vocabulary.
	lossNProbs = 1 << 24
	// maxLossBans bounds the retries at one position of a text.
	maxLossBans = 32
	// zeroLogprob: llama-server writes the lowest float for a probability of 0.
	zeroLogprob = -1e30
)

// Settings are the request settings a run pins.
type Settings struct {
	NPredict      int  `json:"n_predict"`
	NPredictThink int  `json:"n_predict_think"`
	Seed          int  `json:"seed"`
	NProbs        int  `json:"n_probs"`
	Rule          Rule `json:"rule"`
}

// Pins identify a run. Every record carries them, and a resume refuses a
// directory whose pins differ in any field.
type Pins struct {
	Schema         int    `json:"schema"`
	CorpusSHA256   string `json:"corpus_sha256"`
	CorpusItems    int    `json:"corpus_items"`
	WeightPath     string `json:"weight_path"`
	WeightRealpath string `json:"weight_realpath"`
	WeightSize     int64  `json:"weight_size"`
	WeightSHA256   string `json:"weight_sha256"`
	ServerRealpath string `json:"server_realpath"`
	ServerSHA256   string `json:"server_sha256"`
	// ServerLibs maps each shared library next to the binary to its SHA-256. The
	// llama-server binary is a small stub: the code is in these (serverLibs).
	ServerLibs map[string]string `json:"server_libs"`
	ServerArgs []string          `json:"server_args"`
	// ServerEnv is what a started server inherits and reads as settings: the
	// LLAMA_ and GGML_ variables, for example LLAMA_ARG_UBATCH.
	ServerEnv []string `json:"server_env"`
	// Attached means the server was already running: its binary is what -server
	// named, and its flags and environment are unknown. The tool cannot verify
	// either, and the server's API does not report the ubatch.
	Attached bool     `json:"attached"`
	Settings Settings `json:"settings"`

	// The rest comes from the server, so a resume checks it after the start.
	ServerBuildInfo    string `json:"server_build_info"`
	ChatTemplate       string `json:"chat_template"`
	ChatTemplateSHA256 string `json:"chat_template_sha256"`
	NVocab             int    `json:"n_vocab"`
}

// staticPart drops the fields that only a running server can tell.
func (p Pins) staticPart() Pins {
	p.ServerBuildInfo, p.ChatTemplate, p.ChatTemplateSHA256, p.NVocab = "", "", "", 0
	return p
}

// diffPins names the JSON fields in which two pins differ.
func diffPins(a, b Pins) ([]string, error) {
	flat := func(p Pins) (map[string]json.RawMessage, error) {
		data, err := json.Marshal(p)
		if err != nil {
			return nil, err
		}
		var m map[string]json.RawMessage
		return m, json.Unmarshal(data, &m)
	}
	ma, err := flat(a)
	if err != nil {
		return nil, err
	}
	mb, err := flat(b)
	if err != nil {
		return nil, err
	}
	var out []string
	for k, v := range ma {
		if string(v) != string(mb[k]) {
			out = append(out, k)
		}
	}
	slices.Sort(out)
	return out, nil
}

// LossPosition is one predicted token of a text.
type LossPosition struct {
	// Target is the text's token at this position.
	Target  int     `json:"target"`
	Logprob float64 `json:"logprob"`
	TopID   int     `json:"top_id"`
	// Candidates is how many the server listed: the whole vocabulary, or no loss.
	Candidates int `json:"candidates"`
	// Bans is how many tokens had to be banned before the server listed a
	// distribution (measureLoss).
	Bans int `json:"bans,omitempty"`
}

func (p *LossPosition) UnmarshalJSON(data []byte) error {
	type plain LossPosition
	a := struct {
		plain
		Logprob *float64 `json:"logprob"`
	}{}
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	lp, err := decodeLogprob(a.Logprob)
	if err != nil {
		return fmt.Errorf("target %d: %w", a.Target, err)
	}
	*p = LossPosition(a.plain)
	p.Logprob = lp
	return nil
}

// Record is one prompt's result. Everything a report states is computed from
// these fields; no derived number is stored.
type Record struct {
	ID   string `json:"id"`
	Item Item   `json:"item"`
	Pins Pins   `json:"pins"`

	TemplateRequest json.RawMessage `json:"template_request,omitempty"`
	RenderedPrompt  string          `json:"rendered_prompt,omitempty"`
	TokenizeRequest json.RawMessage `json:"tokenize_request"`
	PromptTokens    []int           `json:"prompt_tokens"`
	// Request is the exact body sent to /completion. For a text it is the last
	// position's; the others differ in a shorter prompt and in their bans.
	Request json.RawMessage `json:"request"`

	ResolvedSettings json.RawMessage `json:"resolved_settings"`
	GeneratedTokens  []int           `json:"generated_tokens,omitempty"`
	Content          string          `json:"content,omitempty"`
	StopType         string          `json:"stop_type,omitempty"`
	Truncated        bool            `json:"truncated"`
	TokensEvaluated  int             `json:"tokens_evaluated,omitempty"`
	TokensPredicted  int             `json:"tokens_predicted,omitempty"`
	// Probs has one entry per generated token, null where the server listed none.
	Probs     []*tokenProb   `json:"probs,omitempty"`
	Positions []LossPosition `json:"positions,omitempty"`
}

// runConfig is one invocation of the run mode.
type runConfig struct {
	CorpusPath string
	Out        string
	Model      string
	Server     string
	ServerArgs []string
	// URL is a running server's address; empty starts Server.
	URL string
	// AttachedResume continues a -url run although the tool cannot compare the
	// server's flags with the first part's.
	AttachedResume bool
	Port           int
	Settings       Settings
	Limit          int // stop after this many new records; 0 is no limit
	LoadTimeout    time.Duration
	RequestTimeout time.Duration
	Logf           func(string, ...any)
}

func hashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.CopyBuffer(h, f, make([]byte, 1<<24)); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// realpath is the absolute path with every symlink resolved.
func realpath(path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	return filepath.EvalSymlinks(abs)
}

func sha256String(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}

// isServerLib says whether a file next to the binary holds code the server can
// load: a shared library, or the Metal library of a build that does not embed it.
func isServerLib(name string) bool {
	for _, suffix := range []string{".dylib", ".so", ".dll", ".metallib"} {
		if strings.HasSuffix(name, suffix) {
			return true
		}
	}
	return strings.Contains(name, ".so.")
}

// serverLibs hashes the code next to the server binary, where a llama.cpp build
// keeps its libraries and its rpath points. It takes every library there, because
// a build's stub links most of them and a backend can load others at run time; a
// rebuild that changes a library the server does not load therefore refuses a
// resume too, which errs the right way. A link to a file in the same directory is
// a build's versioned name, and its target is hashed under its own name. A link
// that leaves the directory is hashed under the link's name, or that code would
// not be pinned at all. A library loaded from any other path is not covered.
func serverLibs(binary string) (map[string]string, error) {
	dir, err := filepath.EvalSymlinks(filepath.Dir(binary))
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	libs := map[string]string{}
	for _, e := range entries {
		name := e.Name()
		if !isServerLib(name) {
			continue
		}
		path := filepath.Join(dir, name)
		switch {
		case e.Type()&os.ModeSymlink != 0:
			target, err := filepath.EvalSymlinks(path)
			if err != nil {
				return nil, fmt.Errorf("library link %s: %v", name, err)
			}
			if filepath.Dir(target) == dir {
				continue
			}
		case !e.Type().IsRegular():
			continue
		}
		if libs[name], err = hashFile(path); err != nil {
			return nil, err
		}
	}
	return libs, nil
}

// serverEnv lists the variables a started server reads as settings.
func serverEnv() []string {
	env := []string{}
	for _, kv := range os.Environ() {
		if strings.HasPrefix(kv, "LLAMA_") || strings.HasPrefix(kv, "GGML_") {
			env = append(env, kv)
		}
	}
	slices.Sort(env)
	return env
}

// staticPins hashes the weight and the server's code, so a mismatch is refused
// before any server starts.
func staticPins(cfg runConfig, c *Corpus) (Pins, error) {
	p := Pins{Schema: recordSchema, CorpusSHA256: c.SHA256, CorpusItems: len(c.Items), WeightPath: cfg.Model, ServerArgs: cfg.ServerArgs, Attached: cfg.URL != "", Settings: cfg.Settings}
	if p.ServerArgs == nil {
		p.ServerArgs = []string{}
	}
	var err error
	if p.WeightRealpath, err = realpath(cfg.Model); err != nil {
		return p, fmt.Errorf("weight: %v", err)
	}
	st, err := os.Stat(p.WeightRealpath)
	if err != nil {
		return p, err
	}
	p.WeightSize = st.Size()
	cfg.Logf("hashing %s (%.1f GiB)", p.WeightRealpath, float64(st.Size())/(1<<30))
	if p.WeightSHA256, err = hashFile(p.WeightRealpath); err != nil {
		return p, err
	}
	if p.ServerRealpath, err = realpath(cfg.Server); err != nil {
		return p, fmt.Errorf("server: %v", err)
	}
	if p.ServerSHA256, err = hashFile(p.ServerRealpath); err != nil {
		return p, err
	}
	if p.ServerLibs, err = serverLibs(p.ServerRealpath); err != nil {
		return p, err
	}
	p.ServerEnv = []string{}
	if !p.Attached {
		p.ServerEnv = serverEnv()
	}
	return p, nil
}

func writeJSONAtomic(path string, v any) error {
	// not indented: that would reformat the exact request bodies a record holds
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".tmp-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(append(data, '\n')); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

func runPath(out string) string        { return filepath.Join(out, "run.json") }
func recordsDir(out string) string     { return filepath.Join(out, "records") }
func recordPath(out, id string) string { return filepath.Join(recordsDir(out), id+".json") }

func readRecord(path string) (Record, error) {
	var r Record
	data, err := os.ReadFile(path)
	if err != nil {
		return r, err
	}
	if err := json.Unmarshal(data, &r); err != nil {
		return r, fmt.Errorf("%s: %v", path, err)
	}
	return r, nil
}

// refuseMismatch fails if have differs from want in any field but those that
// only a server knows, when static is set.
func refuseMismatch(what string, have, want Pins, static bool) error {
	if static {
		have, want = have.staticPart(), want.staticPart()
	}
	diff, err := diffPins(have, want)
	if err != nil {
		return err
	}
	if len(diff) > 0 {
		return fmt.Errorf("refusing to resume: %s differs in %s; use a new -out", what, strings.Join(diff, ", "))
	}
	return nil
}

// doneRecords checks the records already in the directory against the pins and
// the corpus, and returns their IDs.
func doneRecords(out string, c *Corpus, pins Pins) (map[string]bool, error) {
	entries, err := os.ReadDir(recordsDir(out))
	if errors.Is(err, os.ErrNotExist) {
		return map[string]bool{}, nil
	}
	if err != nil {
		return nil, err
	}
	items := map[string]Item{}
	for _, it := range c.Items {
		items[it.ID] = it
	}
	done := map[string]bool{}
	for _, e := range entries {
		id, ok := strings.CutSuffix(e.Name(), ".json")
		if !ok || strings.HasPrefix(e.Name(), ".") {
			continue // the temp file of an interrupted write
		}
		r, err := readRecord(filepath.Join(recordsDir(out), e.Name()))
		if err != nil {
			return nil, err
		}
		if err := refuseMismatch("record "+e.Name(), r.Pins, pins, false); err != nil {
			return nil, err
		}
		it, ok := items[id]
		if !ok || r.ID != id {
			return nil, fmt.Errorf("refusing to resume: record %s is not an item of the corpus", e.Name())
		}
		have, _ := json.Marshal(r.Item)
		want, _ := json.Marshal(it)
		if string(have) != string(want) {
			return nil, fmt.Errorf("refusing to resume: record %s holds another version of its corpus item", e.Name())
		}
		done[id] = true
	}
	return done, nil
}

// startServer starts llama-server on a local port and waits for /health.
func startServer(ctx context.Context, cfg runConfig, port int, log io.Writer) (*exec.Cmd, <-chan error, error) {
	args := append([]string{"-m", cfg.Model, "--host", "127.0.0.1", "--port", strconv.Itoa(port)}, cfg.ServerArgs...)
	cmd := exec.Command(cfg.Server, args...)
	cmd.Stdout, cmd.Stderr = log, log
	if err := cmd.Start(); err != nil {
		return nil, nil, err
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	health := fmt.Sprintf("http://127.0.0.1:%d/health", port)
	deadline := time.Now().Add(cfg.LoadTimeout)
	for {
		select {
		case err := <-done:
			return nil, nil, fmt.Errorf("llama-server exited while loading: %v (see server.log)", err)
		case <-ctx.Done():
			stopServer(cmd, done)
			return nil, nil, ctx.Err()
		default:
		}
		if healthy(ctx, health, deadline) {
			return cmd, done, nil
		}
		if time.Now().After(deadline) {
			stopServer(cmd, done)
			return nil, nil, fmt.Errorf("no /health after %s", cfg.LoadTimeout)
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// healthProbeTimeout bounds one /health request: a server that accepts the
// connection and never answers must not outlast the load timeout.
const healthProbeTimeout = 5 * time.Second

// healthy reports whether url answers 200, within the probe's own bound and the
// deadline, whichever comes first.
func healthy(ctx context.Context, url string, deadline time.Time) bool {
	ctx, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()
	ctx, cancelProbe := context.WithTimeout(ctx, healthProbeTimeout)
	defer cancelProbe()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return false
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func stopServer(cmd *exec.Cmd, done <-chan error) {
	_ = cmd.Process.Signal(syscall.SIGTERM)
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		_ = cmd.Process.Kill()
		<-done
	}
}

func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

// forbiddenServerArgs are the flags the tool sets itself.
var forbiddenServerArgs = []string{"-m", "--model", "--host", "--port"}

// serverFlagName is a pass-through argument's flag, also when it is written
// --name=value. llama-server refuses that form today, which the guard should not
// depend on, since these arguments come after the tool's own and would win.
func serverFlagName(arg string) string {
	name, _, _ := strings.Cut(arg, "=")
	return name
}

// run drives the server through the corpus, one prompt at a time, and leaves
// one record per finished prompt. It returns ctx's error when interrupted.
func run(ctx context.Context, cfg runConfig) error {
	for _, a := range cfg.ServerArgs {
		if slices.Contains(forbiddenServerArgs, serverFlagName(a)) {
			return fmt.Errorf("server flag %s is set by the tool", a)
		}
	}
	if cfg.URL != "" && len(cfg.ServerArgs) > 0 {
		return fmt.Errorf("-url attaches to a running server; pass-through flags have no effect there")
	}
	var absErr error
	if cfg.Model, absErr = filepath.Abs(cfg.Model); absErr != nil {
		return absErr
	}
	if cfg.Server, absErr = filepath.Abs(cfg.Server); absErr != nil {
		return absErr
	}
	corpus, err := loadCorpus(cfg.CorpusPath)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(recordsDir(cfg.Out), 0o755); err != nil {
		return err
	}
	pins, err := staticPins(cfg, corpus)
	if err != nil {
		return err
	}

	// The first check needs no server: starting one on 33 GiB takes a while.
	var previous *Pins
	if data, err := os.ReadFile(runPath(cfg.Out)); err == nil {
		previous = new(Pins)
		if err := json.Unmarshal(data, previous); err != nil {
			return fmt.Errorf("%s: %v", runPath(cfg.Out), err)
		}
		if err := refuseMismatch(runPath(cfg.Out), *previous, pins, true); err != nil {
			return err
		}
		if cfg.URL != "" && !cfg.AttachedResume {
			return fmt.Errorf("refusing to resume: a server attached with -url does not report its flags, so a restarted one with another ubatch would pass; start the server with -server, or pass -attached-resume if this is the same server process")
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}

	base := cfg.URL
	if base == "" {
		port := cfg.Port
		if port == 0 {
			if port, err = freePort(); err != nil {
				return err
			}
		}
		log, err := os.OpenFile(filepath.Join(cfg.Out, "server.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			return err
		}
		defer log.Close()
		cmd, done, err := startServer(ctx, cfg, port, log)
		if err != nil {
			return err
		}
		defer stopServer(cmd, done)
		base = fmt.Sprintf("http://127.0.0.1:%d", port)
	}
	c := newClient(base, cfg.RequestTimeout)

	props, err := c.props(ctx)
	if err != nil {
		return err
	}
	servedPath, err := realpath(props.ModelPath)
	if err != nil {
		return fmt.Errorf("the server's model_path %q: %v", props.ModelPath, err)
	}
	if servedPath != pins.WeightRealpath {
		return fmt.Errorf("the server serves %s, -model is %s", servedPath, pins.WeightRealpath)
	}
	pins.ServerBuildInfo, pins.ChatTemplate, pins.ChatTemplateSHA256 = props.BuildInfo, props.ChatTemplate, sha256String(props.ChatTemplate)
	if pins.NVocab, err = c.nVocab(ctx); err != nil {
		return err
	}
	if previous != nil {
		if err := refuseMismatch(runPath(cfg.Out), *previous, pins, false); err != nil {
			return err
		}
	}
	done, err := doneRecords(cfg.Out, corpus, pins)
	if err != nil {
		return err
	}
	if err := writeJSONAtomic(runPath(cfg.Out), pins); err != nil {
		return err
	}

	written := 0
	for i, it := range corpus.Items {
		if done[it.ID] {
			continue
		}
		if cfg.Limit > 0 && written >= cfg.Limit {
			break
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		r, err := runItem(ctx, c, it, pins)
		if err != nil {
			return fmt.Errorf("%s: %w", it.ID, err)
		}
		if err := writeJSONAtomic(recordPath(cfg.Out, it.ID), r); err != nil {
			return err
		}
		written++
		cfg.Logf("[%d/%d] %s: %d tokens, stop %s", i+1, len(corpus.Items), it.ID, len(r.GeneratedTokens)+len(r.Positions), r.StopType)
	}
	cfg.Logf("%d new records, %d before, %d items", written, len(done), len(corpus.Items))
	return nil
}

// promptTokens renders a chat item and tokenizes the prompt, without BOS.
func promptTokens(ctx context.Context, c *client, it Item, r *Record) error {
	text := it.Prompt
	if it.Form == formChat || it.Form == formChatThink {
		tr := templateRequest{Messages: []chatMessage{{Role: "user", Content: it.Prompt}}}
		if it.Form == formChat {
			tr.ReasoningEffort = "none"
		}
		var err error
		if r.TemplateRequest, err = json.Marshal(tr); err != nil {
			return err
		}
		if text, err = c.applyTemplate(ctx, r.TemplateRequest); err != nil {
			return err
		}
		// The reference's property (docs/chat.md): the prompt ends in the prefilled
		// empty think block if and only if thinking is off.
		prefilled := strings.HasSuffix(text, thinkEnd+"\n\n")
		if prefilled != (it.Form == formChat) {
			return fmt.Errorf("the rendered prompt's think block does not match form %s: %q", it.Form, text)
		}
		r.RenderedPrompt = text
	}
	var err error
	if r.TokenizeRequest, err = json.Marshal(tokenizeRequest{Content: text, AddSpecial: false, ParseSpecial: true}); err != nil {
		return err
	}
	if r.PromptTokens, err = c.tokenize(ctx, r.TokenizeRequest); err != nil {
		return err
	}
	if it.ExpectTokens != nil && !slices.Equal(r.PromptTokens, it.ExpectTokens) {
		return fmt.Errorf("the server tokenizes the prompt into %v, the docs record %v", r.PromptTokens, it.ExpectTokens)
	}
	return nil
}

func runItem(ctx context.Context, c *client, it Item, pins Pins) (*Record, error) {
	r := &Record{ID: it.ID, Item: it, Pins: pins}
	if err := promptTokens(ctx, c, it, r); err != nil {
		return nil, err
	}
	if it.Form == formText {
		return r, measureLoss(ctx, c, r)
	}
	n := pins.Settings.NPredict
	if it.Form == formChatThink {
		n = pins.Settings.NPredictThink
	}
	req := greedyRequest(r.PromptTokens, n, pins.Settings.Seed, pins.Settings.NProbs)
	var err error
	if r.Request, err = json.Marshal(req); err != nil {
		return nil, err
	}
	resp, err := c.completion(ctx, r.Request)
	if err != nil {
		return nil, err
	}
	aligned, err := checkCompletion(req, resp)
	if err != nil {
		return nil, err
	}
	r.ResolvedSettings, r.GeneratedTokens, r.Content, r.StopType = resp.GenerationSettings, resp.Tokens, resp.Content, resp.StopType
	r.Truncated, r.TokensEvaluated, r.TokensPredicted, r.Probs = resp.Truncated, resp.TokensEvaluated, resp.TokensPredicted, aligned
	return r, nil
}

// checkResolved refuses a response that did not run what was asked for. It cannot
// hold cache_prompt: llama-server does not report it in generation_settings.
func checkResolved(req completionRequest, resp completionResponse) error {
	var s struct {
		Seed              int      `json:"seed"`
		Temperature       float64  `json:"temperature"`
		RepeatPenalty     float64  `json:"repeat_penalty"`
		NProbs            int      `json:"n_probs"`
		PostSamplingProbs *bool    `json:"post_sampling_probs"`
		Samplers          []string `json:"samplers"`
	}
	if err := json.Unmarshal(resp.GenerationSettings, &s); err != nil {
		return fmt.Errorf("generation_settings: %v", err)
	}
	if s.Seed != req.Seed || s.Temperature != req.Temperature || s.RepeatPenalty != req.RepeatPenalty || s.NProbs != req.NProbs || !slices.Equal(s.Samplers, req.Samplers) {
		return fmt.Errorf("the server resolved other settings than requested: %s", resp.GenerationSettings)
	}
	if s.PostSamplingProbs == nil || *s.PostSamplingProbs != req.PostSamplingProbs {
		return fmt.Errorf("the server resolved other settings than requested: post_sampling_probs in %s", resp.GenerationSettings)
	}
	if resp.Truncated {
		return fmt.Errorf("the server truncated the prompt")
	}
	if resp.TokensEvaluated != len(req.Prompt) {
		return fmt.Errorf("the server evaluated %d prompt tokens, sent %d", resp.TokensEvaluated, len(req.Prompt))
	}
	return nil
}

// alignProbs gives each token its entry. The server's entries must be a
// subsequence of the tokens: it leaves out only tokens that end in an incomplete
// UTF-8 sequence.
func alignProbs(tokens []int, probs []tokenProb) ([]*tokenProb, error) {
	aligned := make([]*tokenProb, len(tokens))
	j := 0
	for i, t := range tokens {
		if j < len(probs) && probs[j].ID == t {
			aligned[i] = &probs[j]
			j++
		}
	}
	if j != len(probs) {
		return nil, fmt.Errorf("the %d probabilities are not a subsequence of the %d tokens", len(probs), len(tokens))
	}
	return aligned, nil
}

// checkCompletion refuses a response that is not what was asked for, so no
// such run ever becomes a record. It returns the probabilities per token.
func checkCompletion(req completionRequest, resp completionResponse) ([]*tokenProb, error) {
	if err := checkResolved(req, resp); err != nil {
		return nil, err
	}
	if len(resp.Tokens) != resp.TokensPredicted {
		return nil, fmt.Errorf("%d tokens, tokens_predicted %d", len(resp.Tokens), resp.TokensPredicted)
	}
	switch resp.StopType {
	case "limit":
		if len(resp.Tokens) != req.NPredict {
			return nil, fmt.Errorf("stop limit after %d of %d tokens", len(resp.Tokens), req.NPredict)
		}
	case "eos":
	default:
		return nil, fmt.Errorf("stop type %q", resp.StopType)
	}
	if len(resp.Tokens) > 0 && len(resp.Probs) == 0 {
		return nil, fmt.Errorf("%d tokens and no probabilities", len(resp.Tokens))
	}
	aligned, err := alignProbs(resp.Tokens, resp.Probs)
	if err != nil {
		return nil, err
	}
	for i, p := range aligned {
		if p == nil {
			continue
		}
		if len(p.Top) == 0 {
			return nil, fmt.Errorf("token %d: no candidates", i)
		}
		if err := checkLogprob(p.Logprob); err != nil {
			return nil, fmt.Errorf("token %d: %w", i, err)
		}
		for _, e := range p.Top {
			if err := checkLogprob(e.Logprob); err != nil {
				return nil, fmt.Errorf("token %d, candidate %d: %w", i, e.ID, err)
			}
		}
		// A tie may list another token first, but then lists the generated one too.
		tied := slices.ContainsFunc(p.Top, func(e topEntry) bool { return e.ID == p.ID && e.Logprob == p.Top[0].Logprob })
		if t := p.Top[0]; t.ID != p.ID && !tied {
			return nil, fmt.Errorf("token %d is not greedy: generated %d (logprob %g), most likely %d (logprob %g)", i, p.ID, p.Logprob, t.ID, t.Logprob)
		}
	}
	return aligned, nil
}

// measureLoss asks, for each token of the text after the first, for the whole
// distribution given the tokens before it. A position whose target the server
// does not list is an error, not a bound.
//
// With n_predict 1 the server lists nothing if the one generated token ends in
// an incomplete UTF-8 sequence. The position is then asked again with that token
// banned from sampling. The ban does not touch the distribution, which comes
// from the raw logits; TestLossBanKeepsTheDistribution and the tiny run in
// docs/loop-baseline.md check that.
func measureLoss(ctx context.Context, c *client, r *Record) error {
	if len(r.PromptTokens) < 2 {
		return fmt.Errorf("a text needs at least 2 tokens")
	}
	for i := 1; i < len(r.PromptTokens); i++ {
		req := greedyRequest(r.PromptTokens[:i], 1, r.Pins.Settings.Seed, lossNProbs)
		var resp completionResponse
		for {
			if err := ctx.Err(); err != nil {
				return err
			}
			body, err := json.Marshal(req)
			if err != nil {
				return err
			}
			if resp, err = c.completion(ctx, body); err != nil {
				return err
			}
			if err := checkResolved(req, resp); err != nil {
				return fmt.Errorf("position %d: %w", i, err)
			}
			r.Request, r.ResolvedSettings, r.Truncated, r.TokensEvaluated = body, resp.GenerationSettings, resp.Truncated, resp.TokensEvaluated
			if len(resp.Probs) > 0 {
				break
			}
			if len(resp.Tokens) != 1 || len(req.LogitBias) >= maxLossBans {
				return fmt.Errorf("position %d: no distribution after %d bans", i, len(req.LogitBias))
			}
			req.LogitBias = append(req.LogitBias, [2]any{resp.Tokens[0], false})
		}
		top := resp.Probs[0].Top
		if len(top) != r.Pins.NVocab {
			return fmt.Errorf("position %d: %d candidates, the vocabulary has %d", i, len(top), r.Pins.NVocab)
		}
		target := r.PromptTokens[i]
		k := slices.IndexFunc(top, func(t topEntry) bool { return t.ID == target })
		if k < 0 {
			return fmt.Errorf("position %d: target %d is not among the candidates", i, target)
		}
		r.Positions = append(r.Positions, LossPosition{Target: target, Logprob: top[k].Logprob, TopID: top[0].ID, Candidates: len(top), Bans: len(req.LogitBias)})
	}
	return nil
}
