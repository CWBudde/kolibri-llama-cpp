// Command kolibri-cache replays the locality workloads' expert selections through simulated LRU
// expert caches of several sizes (internal/locality) and reports hit rate, miss rate and the
// expert bytes a cache would load per token. It is a simulation, kept apart from any streaming
// implementation; it backs "Expert cache simulation" in docs/real-checkpoint.md.
//
//	kolibri-cache -models Kolibri-1-IQ3_XXS-IQ4_XS-down-imx,Kolibri-1-Q8_0 -md docs/expert-cache.md
//
// Two policies, each at every size C (experts per layer):
//
//   - per-layer: every layer has its own LRU cache of C experts;
//   - shared: one LRU cache of C x layers experts for all layers together.
//
// The accesses are in decode order: token by token, within a token layer by layer, and within a
// layer the experts in the router's order. A miss loads the expert's gate, up and down rows, whose
// bytes come from testdata/locality/expert-bytes.json (tools/locality/expert_bytes.py).
//
// For each model and workload it reads <dir>/<workload>.<model>.experts.npy and checks the
// simulation against <dir>/stats-<model>.json of cmd/kolibri-locality: a cache holding every
// expert of a layer misses exactly the experts the layer selects at all, and hits every reuse.
// A mismatch is fatal. It writes <dir>/cache-<model>.json and, given -md, a Markdown report.
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
	GGUF    string `json:"gguf"`
	NExpert int    `json:"n_expert"`
	Layers  []struct {
		BytesPerExpert int64 `json:"bytes_per_expert"`
	} `json:"layers"`
}

// expertBytes is the bytes of one expert of every layer together.
func (m Model) expertBytes() int64 {
	var all int64
	for _, l := range m.Layers {
		all += l.BytesPerExpert
	}
	return all
}

// LayerStats are the fields of a layer in cmd/kolibri-locality's stats the check needs.
type LayerStats struct {
	Counts []int `json:"counts"`
	Reuses int   `json:"reuses"`
}

// Size is one cache size under one policy.
type Size struct {
	ExpertsPerLayer int       `json:"experts_per_layer"`
	CacheBytes      int64     `json:"cache_bytes"`
	Hits            int       `json:"hits"`
	Misses          int       `json:"misses"`
	HitRate         float64   `json:"hit_rate"`
	LoadedPerToken  float64   `json:"loaded_bytes_per_token"`
	LayerHitRate    []float64 `json:"layer_hit_rate"`
}

// Workload is one workload's simulation.
type Workload struct {
	Tokens      int    `json:"tokens"`
	Activations int    `json:"activations"`
	Cold        int    `json:"cold"`
	PerLayer    []Size `json:"per_layer"`
	Shared      []Size `json:"shared"`
}

// sizesOf turns the hits per layer at each size (hits[l][i] at sizes[i]) into the reported sizes.
// A cache of C experts per layer holds C experts' bytes of every layer: the per-layer cache
// exactly, the shared one, which may hold any layers' experts, at the mean expert size.
func sizesOf(hits [][]int, acts []int, tokens int, sizes []int, m Model) []Size {
	out := make([]Size, len(sizes))
	for i, c := range sizes {
		s := Size{ExpertsPerLayer: c, CacheBytes: int64(c) * m.expertBytes(), LayerHitRate: make([]float64, len(hits))}
		var loaded int64
		for l := range hits {
			miss := acts[l] - hits[l][i]
			s.Hits += hits[l][i]
			s.Misses += miss
			s.LayerHitRate[l] = float64(hits[l][i]) / float64(acts[l])
			loaded += int64(miss) * m.Layers[l].BytesPerExpert
		}
		s.HitRate = float64(s.Hits) / float64(s.Hits+s.Misses)
		s.LoadedPerToken = float64(loaded) / float64(tokens)
		out[i] = s
	}
	return out
}

// simulate replays sel through both policies at the given sizes (experts per layer). It also
// returns, per layer, the cold accesses and the hits of a per-layer cache holding every expert.
func simulate(sel locality.Selections, m Model, sizes []int) (w Workload, cold, fullHits []int) {
	n := m.NExpert
	acts := make([]int, sel.Layers)
	hitsPL := make([][]int, sel.Layers)
	cold, fullHits = make([]int, sel.Layers), make([]int, sel.Layers)
	for l := range sel.Layers {
		dist := locality.StackDistances(locality.Flatten(sel.Layer(l)), n)
		acts[l] = len(dist)
		hitsPL[l] = locality.Hits(dist, sizes)
		fullHits[l] = locality.Hits(dist, []int{n})[0]
		for _, d := range dist {
			if d < 0 {
				cold[l]++
			}
		}
	}
	keys, layerOf := locality.SharedKeys(sel, n)
	byLayer := make([][]int, sel.Layers)
	for i, d := range locality.StackDistances(keys, sel.Layers*n) {
		byLayer[layerOf[i]] = append(byLayer[layerOf[i]], d)
	}
	sharedCaps := make([]int, len(sizes))
	for i, c := range sizes {
		sharedCaps[i] = c * sel.Layers
	}
	hitsSh := make([][]int, sel.Layers)
	for l := range byLayer {
		hitsSh[l] = locality.Hits(byLayer[l], sharedCaps)
	}
	w = Workload{Tokens: sel.Tokens, Activations: sel.Layers * sel.Tokens * sel.K}
	for _, c := range cold {
		w.Cold += c
	}
	w.PerLayer = sizesOf(hitsPL, acts, sel.Tokens, sizes, m)
	w.Shared = sizesOf(hitsSh, acts, sel.Tokens, sizes, m)
	return w, cold, fullHits
}

// check compares the simulation with cmd/kolibri-locality's stats of the same selections: the
// experts a layer selects at all (nonzero counts) are its cold misses, its reuses the hits of a
// cache holding every expert.
func check(stats []LayerStats, cold, fullHits []int) error {
	if len(stats) != len(cold) {
		return fmt.Errorf("%d layers in the stats, %d simulated", len(stats), len(cold))
	}
	for l, s := range stats {
		selected := 0
		for _, c := range s.Counts {
			if c > 0 {
				selected++
			}
		}
		if selected != cold[l] || s.Reuses != fullHits[l] {
			return fmt.Errorf("layer %d: the stats count %d experts selected and %d reuses, the simulation "+
				"%d cold misses and %d hits with every expert cached", l, selected, s.Reuses, cold[l], fullHits[l])
		}
	}
	return nil
}

func gb(b int64) string         { return fmt.Sprintf("%.2f", float64(b)/1e9) }
func mb(b float64) string       { return fmt.Sprintf("%.0f", b/1e6) }
func pct(v float64) string      { return fmt.Sprintf("%.1f%%", 100*v) }
func short(model string) string { return strings.TrimPrefix(model, "Kolibri-1-") }

type result struct {
	model, workload string
	m               Model
	w               Workload
}

func summary(r result) string {
	var pl, sh []string
	for i, s := range r.w.PerLayer {
		pl = append(pl, fmt.Sprintf("%d %s", s.ExpertsPerLayer, pct(s.HitRate)))
		sh = append(sh, fmt.Sprintf("%d %s", s.ExpertsPerLayer, pct(r.w.Shared[i].HitRate)))
	}
	return fmt.Sprintf("%s %s: %d tokens, cold %s; hit rate per-layer %s; shared %s", r.model, r.workload,
		r.w.Tokens, pct(float64(r.w.Cold)/float64(r.w.Activations)), strings.Join(pl, ", "), strings.Join(sh, ", "))
}

// layerSizes are the sizes of the per-layer tables, those of the report's sizes among them.
var layerSizes = []int{32, 64, 128}

func writeMarkdown(w io.Writer, command string, results []result) {
	fmt.Fprintf(w, "# Expert cache simulation\n\n")
	fmt.Fprintf(w, "Generated by `%s` from the selection files of `tools/locality/experts.py` and the expert sizes "+
		"in `testdata/locality/expert-bytes.json`; do not edit by hand. The simulation is explained in "+
		"[real-checkpoint.md](real-checkpoint.md#expert-cache-simulation).\n\n", command)
	fmt.Fprintf(w, "A size is experts per layer: the per-layer cache holds that many experts of every layer, the "+
		"shared cache that many times the layer count of any layers. Cache GB counts the expert bytes it "+
		"holds; MB/token are the expert bytes its misses load per token. Without a cache every token loads "+
		"all its experts.\n")
	for _, r := range results {
		all := r.m.expertBytes()
		noCache := float64(r.w.Activations) / float64(r.w.Tokens) * float64(all) / float64(len(r.m.Layers))
		fmt.Fprintf(w, "\n## %s, %s\n\n%d tokens; all experts %s GB; no cache %s MB/token; cold misses %s of "+
			"%d activations.\n\n", r.model, r.workload, r.w.Tokens, gb(int64(r.m.NExpert)*all), mb(noCache),
			pct(float64(r.w.Cold)/float64(r.w.Activations)), r.w.Activations)
		fmt.Fprintf(w, "| Experts per layer | Cache GB | Per-layer hit rate | Per-layer miss rate | Per-layer MB/token "+
			"| Shared hit rate | Shared miss rate | Shared MB/token |\n|--:|--:|--:|--:|--:|--:|--:|--:|\n")
		for i, p := range r.w.PerLayer {
			s := r.w.Shared[i]
			fmt.Fprintf(w, "| %d | %s | %s | %s | %s | %s | %s | %s |\n", p.ExpertsPerLayer, gb(p.CacheBytes),
				pct(p.HitRate), pct(1-p.HitRate), mb(p.LoadedPerToken), pct(s.HitRate), pct(1-s.HitRate),
				mb(s.LoadedPerToken))
		}
	}
	for _, r := range results {
		var idx []int
		for _, c := range layerSizes {
			for i, p := range r.w.PerLayer {
				if p.ExpertsPerLayer == c {
					idx = append(idx, i)
				}
			}
		}
		if len(idx) == 0 {
			continue
		}
		fmt.Fprintf(w, "\n## Hit rate per layer: %s, %s\n\n| Layer |", r.model, r.workload)
		for _, i := range idx {
			fmt.Fprintf(w, " Per-layer %d |", r.w.PerLayer[i].ExpertsPerLayer)
		}
		for _, i := range idx {
			fmt.Fprintf(w, " Shared %d |", r.w.Shared[i].ExpertsPerLayer)
		}
		fmt.Fprintf(w, "\n|--:|%s\n", strings.Repeat("--:|", 2*len(idx)))
		for l := range r.m.Layers {
			fmt.Fprintf(w, "| %d |", l)
			for _, i := range idx {
				fmt.Fprintf(w, " %s |", pct(r.w.PerLayer[i].LayerHitRate[l]))
			}
			for _, i := range idx {
				fmt.Fprintf(w, " %s |", pct(r.w.Shared[i].LayerHitRate[l]))
			}
			fmt.Fprintln(w)
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
	dir := flag.String("dir", filepath.Join(home, "models", "eval", "locality"), "selection files, stats and output")
	models := flag.String("models", "", "comma-separated GGUF names without .gguf")
	workloads := flag.String("workloads", "coding,research,hr,medtech", "comma-separated workloads")
	sizesFlag := flag.String("sizes", "8,16,32,48,64,96,128,192,256,384", "cache sizes in experts per layer")
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
		if err != nil || c < 1 {
			log.Fatalf("-sizes: %q is not a positive count", s)
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
		var stats map[string][]LayerStats
		readJSON(filepath.Join(*dir, "stats-"+name+".json"), &stats)
		out := map[string]Workload{}
		for wl := range strings.SplitSeq(*workloads, ",") {
			sel, err := locality.ReadNPYFile(filepath.Join(*dir, fmt.Sprintf("%s.%s.experts.npy", wl, name)))
			if err != nil {
				log.Fatal(err)
			}
			if sel.Layers != len(m.Layers) {
				log.Fatalf("%s %s: %d layers of selections, %d of expert sizes", name, wl, sel.Layers, len(m.Layers))
			}
			w, cold, fullHits := simulate(sel, m, sizes)
			if err := check(stats[wl], cold, fullHits); err != nil {
				log.Fatalf("FAIL %s %s: the simulation disagrees with stats-%s.json: %v", short(name), wl, short(name), err)
			}
			fmt.Printf("PASS %s %s: %d layers, cold misses and full-cache hits as in stats-%s.json\n",
				short(name), wl, sel.Layers, short(name))
			out[wl] = w
			r := result{short(name), wl, m, w}
			results = append(results, r)
			fmt.Println(summary(r))
		}
		b, err := json.MarshalIndent(out, "", " ")
		if err != nil {
			log.Fatal(err)
		}
		path := filepath.Join(*dir, "cache-"+name+".json")
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
		writeMarkdown(f, "go run ./cmd/kolibri-cache -models "+*models+" -md "+*md, results)
		if err := f.Close(); err != nil {
			log.Fatal(err)
		}
		fmt.Println("wrote", *md)
	}
}
