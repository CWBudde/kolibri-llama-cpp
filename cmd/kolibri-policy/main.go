// Command kolibri-policy replays the locality workloads' expert selections through expert cache
// policies in the order an online server meets them (internal/locality, replay.go): each prompt's
// new tokens prefilled in ubatches, each answer decoded token by token, with a policy that learns
// a step's experts only when the step runs. It reports prefill and decode apart and is the
// evaluator of PLAN Phase 10, Stage A; it writes docs/expert-policy.md.
//
//	kolibri-policy -models Kolibri-1-IQ3_XXS-IQ4_XS-down-imx,Kolibri-1-Q8_0 -md docs/expert-policy.md
//
// The only policy so far is the evaluator's reference, a per-layer LRU cache of C experts in every
// layer (cmd/kolibri-cache's per-layer policy), at every size C.
//
// For each model and workload it reads <dir>/<workload>.<model>.experts.npy and the token IDs
// <dir>/<workload>.tokens.npy, and the answers' prompt lengths and generated tokens from
// testdata/locality/<workload>.json. Two checks are fatal: the token IDs must hold every answer's
// generated tokens after its prompt, and the per-layer LRU, replayed token by token in trace order,
// must hit exactly as often as in <dir>/cache-<model>.json of cmd/kolibri-cache at every size.
// It writes <dir>/policy-<model>.json and, given -md, a Markdown report.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"kolibri-llm/internal/locality"
)

// Model are a GGUF's expert sizes as tools/locality/expert_bytes.py records them.
type Model struct {
	NExpert int `json:"n_expert"`
	Layers  []struct {
		BytesPerExpert int64 `json:"bytes_per_expert"`
	} `json:"layers"`
}

func (m Model) bytes() []int64 {
	out := make([]int64, len(m.Layers))
	for l, x := range m.Layers {
		out[l] = x.BytesPerExpert
	}
	return out
}

// CacheSizes are the fields of a workload in cmd/kolibri-cache's output the check needs.
type CacheSizes struct {
	PerLayer []struct {
		ExpertsPerLayer int `json:"experts_per_layer"`
		Hits            int `json:"hits"`
	} `json:"per_layer"`
}

// Size is the per-layer LRU at one size.
type Size struct {
	ExpertsPerLayer int             `json:"experts_per_layer"`
	CacheBytes      int64           `json:"cache_bytes"`
	Online          locality.Result `json:"online"`
	TraceOrderHits  int             `json:"trace_order_hits"`
	Accesses        int             `json:"trace_order_accesses"`
}

// Workload is one workload's replay.
type Workload struct {
	Tokens   int    `json:"tokens"`
	Answers  int    `json:"answers"`
	Ubatch   int    `json:"ubatch"`
	Ubatches int    `json:"ubatches"`
	LRU      []Size `json:"layer_lru"`
}

// replay runs the per-layer LRU at every size, online and in trace order.
func replay(sel locality.Selections, spans []locality.Span, m Model, sizes []int) ([]Size, error) {
	var all int64
	for _, b := range m.bytes() {
		all += b
	}
	out := make([]Size, len(sizes))
	for i, c := range sizes {
		online, err := locality.Replay(sel, spans, m.NExpert, m.bytes(), locality.NewLayerLRU(sel.Layers, m.NExpert, c))
		if err != nil {
			return nil, err
		}
		trace, err := locality.Replay(sel, locality.TraceOrder(sel.Tokens), m.NExpert, m.bytes(),
			locality.NewLayerLRU(sel.Layers, m.NExpert, c))
		if err != nil {
			return nil, err
		}
		out[i] = Size{c, int64(c) * all, online, trace.Decode.Hits, trace.Decode.Accesses}
	}
	return out, nil
}

// check compares the trace-order hits with cmd/kolibri-cache's per-layer hits at the same sizes.
func check(lru []Size, cache CacheSizes) error {
	recorded := map[int]int{}
	for _, s := range cache.PerLayer {
		recorded[s.ExpertsPerLayer] = s.Hits
	}
	for _, s := range lru {
		h, ok := recorded[s.ExpertsPerLayer]
		if !ok {
			return fmt.Errorf("no size %d (run kolibri-cache with it in -sizes)", s.ExpertsPerLayer)
		}
		if h != s.TraceOrderHits {
			return fmt.Errorf("%d experts per layer: %d hits in trace order, kolibri-cache %d", s.ExpertsPerLayer,
				s.TraceOrderHits, h)
		}
	}
	return nil
}

func gb(b int64) string         { return fmt.Sprintf("%.2f", float64(b)/1e9) }
func pct(v float64) string      { return fmt.Sprintf("%.1f%%", 100*v) }
func short(model string) string { return strings.TrimPrefix(model, "Kolibri-1-") }

func rate(c locality.Counts) float64 { return float64(c.Hits) / float64(max(c.Accesses, 1)) }

// perToken is a phase's miss bytes per token in MB.
func perToken(c locality.Counts) string {
	return fmt.Sprintf("%.0f", float64(c.MissBytes)/1e6/float64(max(c.Tokens, 1)))
}

type result struct {
	model, workload string
	w               Workload
}

func summary(r result) string {
	var parts []string
	for _, s := range r.w.LRU {
		parts = append(parts, fmt.Sprintf("%d %s/%s", s.ExpertsPerLayer, pct(rate(s.Online.Prefill)),
			pct(rate(s.Online.Decode))))
	}
	p := r.w.LRU[0].Online
	return fmt.Sprintf("%s %s: %d prefill and %d decode tokens; per-layer LRU hit rate prefill/decode %s", r.model,
		r.workload, p.Prefill.Tokens, p.Decode.Tokens, strings.Join(parts, ", "))
}

func writeMarkdown(w io.Writer, command string, results []result) {
	fmt.Fprintf(w, "# Expert cache policies, replayed online\n\n")
	fmt.Fprintf(w, "Generated by `%s` from the selection files of `tools/locality/experts.py`, the answers in "+
		"`testdata/locality/<workload>.json` and the expert sizes in `testdata/locality/expert-bytes.json`; do not "+
		"edit by hand.\n\n", command)
	fmt.Fprintf(w, "The workloads are replayed in the order an online server with a prompt cache meets them: each "+
		"prompt's new tokens prefilled in ubatches, each answer decoded token by token. A decoded token needs its "+
		"six experts per layer; a prefill ubatch the union of its tokens' experts, once. The policy learns a "+
		"step's experts only when the step runs (PLAN Phase 10, Stage A). An answer that stopped at the token "+
		"limit decodes all but its last token, which the next prompt's prefill processes.\n\n")
	fmt.Fprintf(w, "Prefill and decode are counted apart: a prefill hit serves a ubatch of up to hundreds of "+
		"teacher-forced tokens at once and is no decode speedup. MB/token are the expert bytes a phase's misses "+
		"load per token of that phase. The trace-order hit rate replays every token as one decode step, the "+
		"order of [expert-cache.md](expert-cache.md), whose hits it reproduces exactly.\n\n")
	fmt.Fprintf(w, "Policy: an LRU cache of the given number of experts in every layer, the only policy so far. "+
		"Cache GB counts the expert bytes it holds when full.\n")
	for _, r := range results {
		p := r.w.LRU[0].Online
		fmt.Fprintf(w, "\n## %s, %s\n\n%d tokens, %d answers: %d prefill tokens in %d ubatches of up to %d, "+
			"%d decoded tokens.\n\n", r.model, r.workload, r.w.Tokens, r.w.Answers, p.Prefill.Tokens, r.w.Ubatches,
			r.w.Ubatch, p.Decode.Tokens)
		fmt.Fprintf(w, "| Experts per layer | Cache GB | Prefill hit rate | Prefill MB/token | Decode hit rate "+
			"| Decode MB/token | Trace-order hit rate |\n|--:|--:|--:|--:|--:|--:|--:|\n")
		for _, s := range r.w.LRU {
			fmt.Fprintf(w, "| %d | %s | %s | %s | %s | %s | %s |\n", s.ExpertsPerLayer, gb(s.CacheBytes),
				pct(rate(s.Online.Prefill)), perToken(s.Online.Prefill), pct(rate(s.Online.Decode)),
				perToken(s.Online.Decode), pct(float64(s.TraceOrderHits)/float64(s.Accesses)))
		}
	}
}

func readJSON(path string, v any) {
	b, err := os.ReadFile(path)
	if err != nil {
		log.Fatal(err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		log.Fatalf("%s: %v", path, err)
	}
}

func main() {
	home, _ := os.UserHomeDir()
	dir := flag.String("dir", filepath.Join(home, "models", "eval", "locality"), "selection files, token IDs and output")
	models := flag.String("models", "", "comma-separated GGUF names without .gguf")
	workloads := flag.String("workloads", "coding,research,hr,medtech", "comma-separated workloads")
	traces := flag.String("traces", "testdata/locality", "the workload traces")
	sizesFlag := flag.String("sizes", "8,16,32,48,64,96,128,192,256,384", "cache sizes in experts per layer")
	ubatch := flag.Int("ubatch", 512, "prefill ubatch size (llama-server's default)")
	bytesPath := flag.String("bytes", "testdata/locality/expert-bytes.json", "expert sizes per model")
	md := flag.String("md", "", "write the Markdown report here")
	flag.Parse()
	if *models == "" {
		flag.Usage()
		os.Exit(2)
	}
	var sizes []int
	for s := range strings.SplitSeq(*sizesFlag, ",") {
		c, err := strconv.Atoi(s)
		if err != nil || c < 0 {
			log.Fatalf("-sizes: %q is not a count", s)
		}
		sizes = append(sizes, c)
	}
	var known map[string]Model
	readJSON(*bytesPath, &known)
	var results []result
	for name := range strings.SplitSeq(*models, ",") {
		m, ok := known[name]
		if !ok {
			log.Fatalf("%s: no expert sizes in %s (run tools/locality/expert_bytes.py --record)", name, *bytesPath)
		}
		var cache map[string]CacheSizes
		readJSON(filepath.Join(*dir, "cache-"+name+".json"), &cache)
		out := map[string]Workload{}
		for wl := range strings.SplitSeq(*workloads, ",") {
			sel, err := locality.ReadNPYFile(filepath.Join(*dir, fmt.Sprintf("%s.%s.experts.npy", wl, name)))
			if err != nil {
				log.Fatal(err)
			}
			if sel.Layers != len(m.Layers) {
				log.Fatalf("%s %s: %d layers of selections, %d of expert sizes", name, wl, sel.Layers, len(m.Layers))
			}
			f, err := os.Open(filepath.Join(*dir, wl+".tokens.npy"))
			if err != nil {
				log.Fatal(err)
			}
			ids, err := locality.ReadTokens(f)
			f.Close()
			if err != nil {
				log.Fatalf("%s tokens: %v", wl, err)
			}
			f, err = os.Open(filepath.Join(*traces, wl+".json"))
			if err != nil {
				log.Fatal(err)
			}
			answers, err := locality.ReadAnswers(f)
			f.Close()
			if err != nil {
				log.Fatalf("%s trace: %v", wl, err)
			}
			if len(ids) != sel.Tokens {
				log.Fatalf("FAIL %s: %d token IDs, %d tokens of selections", wl, len(ids), sel.Tokens)
			}
			if err := locality.CheckAnswers(ids, answers); err != nil {
				log.Fatalf("FAIL %s: %v", wl, err)
			}
			fmt.Printf("PASS %s: the token IDs hold the %d answers' generated tokens after their prompts\n", wl,
				len(answers))
			spans, err := locality.OnlineSpans(sel.Tokens, answers, *ubatch)
			if err != nil {
				log.Fatalf("%s: %v", wl, err)
			}
			lru, err := replay(sel, spans, m, sizes)
			if err != nil {
				log.Fatalf("%s %s: %v", name, wl, err)
			}
			if err := check(lru, cache[wl]); err != nil {
				log.Fatalf("FAIL %s %s: the trace-order replay disagrees with cache-%s.json: %v", short(name), wl,
					short(name), err)
			}
			fmt.Printf("PASS %s %s: per-layer LRU in trace order hits as in cache-%s.json at %d sizes\n", short(name),
				wl, short(name), len(sizes))
			w := Workload{Tokens: sel.Tokens, Answers: len(answers), Ubatch: *ubatch, LRU: lru}
			for _, s := range spans {
				if s.Phase == locality.Prefill {
					w.Ubatches++
				}
			}
			out[wl] = w
			r := result{short(name), wl, w}
			results = append(results, r)
			fmt.Println(summary(r))
		}
		b, err := json.MarshalIndent(out, "", " ")
		if err != nil {
			log.Fatal(err)
		}
		path := filepath.Join(*dir, "policy-"+name+".json")
		if err := os.WriteFile(path, append(b, '\n'), 0o644); err != nil {
			log.Fatal(err)
		}
		fmt.Println("wrote", path)
	}
	if *md != "" {
		f, err := os.Create(*md)
		if err != nil {
			log.Fatal(err)
		}
		writeMarkdown(f, "go run ./cmd/kolibri-policy -models "+*models+" -md "+*md, results)
		if err := f.Close(); err != nil {
			log.Fatal(err)
		}
		fmt.Println("wrote", *md)
	}
}
