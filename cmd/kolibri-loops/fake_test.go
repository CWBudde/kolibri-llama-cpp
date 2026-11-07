package main

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
)

const fakeTemplate = "{{ fake chat template }}"

// documentedTokens are the IDs the docs record, so the corpus's expect_tokens
// pass against the fake tokenizer.
var documentedTokens = map[string][]int{
	"Die Hauptstadt von Deutschland ist": {452, 22090, 493, 1678, 2459},
	"The capital of Germany is":          {325, 9255, 279, 6905, 735},
}

// fakeServer speaks the part of llama-server's API the tool uses. Its shapes
// were read from a real llama-server on the tiny checkpoint (docs/loop-baseline.md).
type fakeServer struct {
	mu       sync.Mutex
	requests map[string]int
	bodies   [][]byte // the /completion bodies, as received

	modelPath string
	template  string
	buildInfo string
	nVocab    int

	// gen makes the generated tokens and the stop type for a prompt.
	gen func(prompt []int, nPredict int) ([]int, string)
	// failAfter answers 500 to /completion after this many requests, if positive.
	failAfter int
	// fragments are tokens the server lists no probability for, as llama-server
	// does for a token that ends in an incomplete UTF-8 sequence.
	fragments map[int]bool
	// The mutations of the negative controls.
	notGreedy       bool // the most likely candidate is not the generated token
	truncateVocab   int  // the server lists only this many candidates for a text
	tokenizeOffset  int  // the tokenizer drifts
	ignoreSeed      bool // the server resolves another seed
	wrongThinkBlock bool // the template ignores reasoning_effort
	nanLogits       bool // every log probability is null, as the server writes NaN
}

func newFake(modelPath string) *fakeServer {
	return &fakeServer{requests: map[string]int{}, modelPath: modelPath, template: fakeTemplate, buildInfo: "fake-1", nVocab: 512, gen: defaultGen}
}

// defaultGen loops on the documented German prompt and counts upwards otherwise.
func defaultGen(prompt []int, n int) ([]int, string) {
	out := make([]int, n)
	for i := range out {
		if prompt[0] == 452 {
			out[i] = 1678
		} else {
			out[i] = 1000 + i
		}
	}
	return out, "limit"
}

func (f *fakeServer) count(path string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.requests[path]
}

func (f *fakeServer) total() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	n := 0
	for _, v := range f.requests {
		n += v
	}
	return n
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func (f *fakeServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	f.requests[r.URL.Path]++
	seen := f.requests[r.URL.Path]
	f.mu.Unlock()
	switch r.URL.Path {
	case "/health":
		writeJSON(w, map[string]string{"status": "ok"})
	case "/props":
		writeJSON(w, map[string]any{"model_path": f.modelPath, "build_info": f.buildInfo, "chat_template": f.template})
	case "/v1/models":
		writeJSON(w, map[string]any{"data": []any{map[string]any{"meta": map[string]int{"n_vocab": f.nVocab}}}})
	case "/tokenize":
		var q tokenizeRequest
		_ = json.NewDecoder(r.Body).Decode(&q)
		if q.AddSpecial || !q.ParseSpecial {
			http.Error(w, "the tool must not add BOS and must parse special tokens", http.StatusBadRequest)
			return
		}
		writeJSON(w, map[string]any{"tokens": f.tokenize(q.Content)})
	case "/apply-template":
		var q templateRequest
		_ = json.NewDecoder(r.Body).Decode(&q)
		prompt := "<|im_start|>user\n" + q.Messages[0].Content + "<|im_end|>\n<|im_start|>assistant\n"
		if q.ReasoningEffort == "none" && !f.wrongThinkBlock {
			prompt += "<think>\n\n</think>\n\n"
		}
		writeJSON(w, map[string]string{"prompt": prompt})
	case "/completion":
		if f.failAfter > 0 && seen > f.failAfter {
			http.Error(w, "the slot is gone", http.StatusInternalServerError)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		f.mu.Lock()
		f.bodies = append(f.bodies, body)
		f.mu.Unlock()
		var q completionRequest
		if err := json.Unmarshal(body, &q); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		writeJSON(w, f.complete(q))
	default:
		http.NotFound(w, r)
	}
}

func (f *fakeServer) tokenize(s string) []int {
	if t, ok := documentedTokens[s]; ok {
		if f.tokenizeOffset != 0 {
			t = []int{t[0] + f.tokenizeOffset, t[1], t[2], t[3], t[4]}
		}
		return t
	}
	var out []int
	for _, b := range []byte(s) {
		out = append(out, int(b)+f.tokenizeOffset)
	}
	return out
}

func (f *fakeServer) complete(q completionRequest) map[string]any {
	seed := q.Seed
	if f.ignoreSeed {
		seed++
	}
	lpValue := func(v float64) any {
		if f.nanLogits {
			return nil
		}
		return v
	}
	settings := map[string]any{"seed": seed, "temperature": q.Temperature, "repeat_penalty": q.RepeatPenalty, "n_probs": q.NProbs, "post_sampling_probs": q.PostSamplingProbs, "samplers": q.Samplers}
	var tokens []int
	var probs []map[string]any
	stop := "limit"
	if q.NProbs >= f.nVocab {
		// a text's position: the whole distribution, uniform
		n := f.nVocab
		if f.truncateVocab > 0 {
			n = f.truncateVocab
		}
		lp := -math.Log(float64(f.nVocab))
		top := make([]map[string]any, n)
		for i := range top {
			top[i] = map[string]any{"id": i, "token": "t", "bytes": []int{1}, "logprob": lpValue(lp)}
		}
		banned := map[int]bool{}
		for _, b := range q.LogitBias {
			banned[int(b[0].(float64))] = true
		}
		first := 0
		for banned[first] {
			first++
		}
		tokens = []int{first}
		if !f.fragments[first] {
			probs = []map[string]any{{"id": first, "token": "t", "bytes": []int{1}, "logprob": lpValue(lp), "top_logprobs": top}}
		}
	} else {
		tokens, stop = f.gen(q.Prompt, q.NPredict)
		for _, t := range tokens {
			if f.fragments[t] {
				continue
			}
			top := []map[string]any{{"id": t, "logprob": lpValue(-0.1)}, {"id": t + 1, "logprob": lpValue(-3.0)}}
			lp := -0.1
			if f.notGreedy {
				top = []map[string]any{{"id": t + 1, "logprob": -0.1}, {"id": t, "logprob": -3.0}}
				lp = -3.0
			}
			probs = append(probs, map[string]any{"id": t, "token": "t", "logprob": lpValue(lp), "top_logprobs": top[:min(q.NProbs, 2)]})
		}
	}
	var content strings.Builder
	for _, t := range tokens {
		fmt.Fprintf(&content, " %d", t)
	}
	return map[string]any{
		"content": content.String(), "tokens": tokens, "stop_type": stop, "truncated": false,
		"tokens_evaluated": len(q.Prompt), "tokens_predicted": len(tokens),
		"completion_probabilities": probs, "generation_settings": settings,
		"timings": map[string]float64{"predicted_per_second": 1},
	}
}

// startFake serves f over HTTP and sets the tool's -url.
func startFake(t *testing.T, f *fakeServer) string {
	t.Helper()
	s := httptest.NewServer(f)
	t.Cleanup(s.Close)
	return s.URL
}

// fakeServerMain makes the test binary a llama-server: TestStartsServer runs it
// through -server.
func fakeServerMain() {
	args := os.Args[1:]
	flagValue := func(name string) string {
		i := slices.Index(args, name)
		if i < 0 || i+1 >= len(args) {
			fmt.Fprintln(os.Stderr, "missing", name)
			os.Exit(3)
		}
		return args[i+1]
	}
	if err := os.WriteFile(os.Getenv("KOLIBRI_LOOPS_FAKE_ARGS"), []byte(strings.Join(args, "\n")), 0o644); err != nil {
		os.Exit(3)
	}
	port, err := strconv.Atoi(flagValue("--port"))
	if err != nil {
		os.Exit(3)
	}
	l, err := net.Listen("tcp", net.JoinHostPort(flagValue("--host"), strconv.Itoa(port)))
	if err != nil {
		os.Exit(3)
	}
	_ = http.Serve(l, newFake(flagValue("-m")))
}
