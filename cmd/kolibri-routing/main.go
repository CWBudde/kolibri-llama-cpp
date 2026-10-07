// Command kolibri-routing measures the expert-routing overhead of a GGUF on one backend from
// per-op timings with the model's real graph shapes. It backs "Routing overhead" in
// docs/real-checkpoint.md.
//
//	test-export-graph-ops -m model.gguf -c 4096 -ub 512 -o ops.txt
//	test-backend-ops perf -b MTL0 --test-file ops.txt > perf.txt
//	llama-eval-callback -m model.gguf -ngl 99 -p Berlin > nodes.txt   # a one-token prompt
//	llama-bench -m model.gguf -ngl 99 -t 10 -p 512 -n 128 -r 5 -o json > bench.json
//	kolibri-routing -perf perf.txt -nodes nodes.txt -bench bench.json -layers 50
//
// The router ops are the nodes build_moe_ffn computes before the expert matmuls (routerNames):
// the router logits, their sigmoid or softmax, the selection bias, the Top-k sort and the
// gathered and normalized weights. Their time per token is the sum, over every such node of a
// one-token graph (the eval-callback listing), of test-backend-ops' time for an op of the same
// shapes; per ubatch, the same with the ubatch-shaped timing of each router op. Both are also
// given as a share of llama-bench's measured tg and pp time. ffn_moe_weighted, which scales the
// expert outputs by the weights, is reported apart, and the sum over all timed nodes is a
// cross-check of the isolated timings against the measured token time. Attention has no timing
// of its decode shape (the exported graph reads the whole cache) and is counted as untimed.
//
// Ops are timed in isolation, so fusion and the graph's dispatch overhead are not modeled. It
// fails when a router op has no timing or does not occur once per layer.
package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// routerNames are the routing nodes build_moe_ffn names (src/llama-graph.cpp), in graph order;
// ffn_moe_topk is a view and costs nothing.
var routerNames = []string{
	"ffn_moe_logits", "ffn_moe_logits_biased", "ffn_moe_probs", "ffn_moe_probs_biased",
	"ffn_moe_group_topk", "ffn_moe_probs_masked", "ffn_moe_argsort", "ffn_moe_topk",
	"ffn_moe_weights", "ffn_moe_weights_softmax", "ffn_moe_weights_sum",
	"ffn_moe_weights_sum_clamped", "ffn_moe_weights_norm", "ffn_moe_weights_scaled",
}

const combineName = "ffn_moe_weighted"

// viewOps are skipped, as test-export-graph-ops skips them: they compute nothing.
var viewOps = map[string]bool{"NONE": true, "VIEW": true, "RESHAPE": true, "PERMUTE": true, "TRANSPOSE": true}

// Perf is one op timing from test-backend-ops perf (console output).
type Perf struct {
	Op, Name string
	NE       [4]int
	Src      [][4]int
	US       float64 // microseconds per run
}

// Node is one graph node from llama-eval-callback, with at most its first two sources.
type Node struct {
	Name, Op string
	NE       [4]int
	Src      [][4]int
}

// Bench are llama-bench's measured speeds.
type Bench struct {
	PP, TG  float64 // tokens/s
	NPrompt int
	NUbatch int
}

// OpTime is one router op: its time per node for one token and per ubatch, and its node count.
type OpTime struct {
	Name  string  `json:"name"`
	Op    string  `json:"op"`
	NE    [4]int  `json:"ne"`
	PPNE  [4]int  `json:"pp_ne"`
	Count int     `json:"count"`
	TG    float64 `json:"tg_us"`
	PP    float64 `json:"pp_us"`
}

// Report is the routing overhead of one model on one backend.
type Report struct {
	Layers   int      `json:"layers"`
	Router   []OpTime `json:"router"`
	RouterTG float64  `json:"router_tg_us"` // per token
	RouterPP float64  `json:"router_pp_us"` // per ubatch
	TokenUS  float64  `json:"token_us"`     // measured, tg
	UbatchUS float64  `json:"ubatch_us"`    // measured, pp
	TGShare  float64  `json:"tg_share"`
	PPShare  float64  `json:"pp_share"`
	Combine  OpTime   `json:"combine"`
	Nodes    int      `json:"nodes"`
	AllTG    float64  `json:"all_tg_us"` // sum over the timed nodes of a one-token graph
	// Untimed counts, per op, the nodes without a timing of their shapes (other than router ops)
	Untimed map[string]int `json:"untimed"`
	Ubatch  int            `json:"ubatch"`
}

var (
	perfRe    = regexp.MustCompile(`^\s*([A-Z_0-9]+)\((.*)\):\s+\d+ runs -\s+([0-9.]+) us/run`)
	nameRe    = regexp.MustCompile(`(?:^|,)name=([^,]*)`)
	neRe      = regexp.MustCompile(`(?:^|,)ne=\[([-0-9,]+)\]`)
	perfSrcRe = regexp.MustCompile(`[a-z0-9_]+\[([-0-9,]+)\](?:nb\[[0-9,]+\])?`)
	nodeRe    = regexp.MustCompile(`^common_debug_cb_eval:\s*(.+?) = \(\w+\)\s*([A-Z_0-9]+)\((.*)\) = \{([^}]*)\}\s*$`)
	nodeSrcRe = regexp.MustCompile(`\{(-?\d+), (-?\d+), (-?\d+), (-?\d+)\}`)
)

func shape(fields []string) ([4]int, error) {
	var ne [4]int
	if len(fields) != 4 {
		return ne, fmt.Errorf("shape %v, want 4 dimensions", fields)
	}
	for i, f := range fields {
		v, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil {
			return ne, err
		}
		ne[i] = v
	}
	return ne, nil
}

// parsePerf reads the timed ops of test-backend-ops perf's console output.
func parsePerf(r io.Reader) ([]Perf, error) {
	var out []Perf
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<16), 1<<20)
	for sc.Scan() {
		m := perfRe.FindStringSubmatch(sc.Text())
		if m == nil {
			continue
		}
		params := m[2]
		p := Perf{Op: m[1]}
		if n := nameRe.FindStringSubmatch(params); n != nil {
			p.Name = n[1]
		}
		ne := neRe.FindStringSubmatch(params)
		if ne == nil {
			return nil, fmt.Errorf("perf: no ne in %q", sc.Text())
		}
		var err error
		if p.NE, err = shape(strings.Split(ne[1], ",")); err != nil {
			return nil, fmt.Errorf("perf %q: %w", sc.Text(), err)
		}
		if _, src, ok := strings.Cut(params, "sources="); ok {
			for _, s := range perfSrcRe.FindAllStringSubmatch(src, -1) {
				sh, err := shape(strings.Split(s[1], ","))
				if err != nil {
					return nil, fmt.Errorf("perf %q: %w", sc.Text(), err)
				}
				p.Src = append(p.Src, sh)
			}
		}
		if p.US, err = strconv.ParseFloat(m[3], 64); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, sc.Err()
}

// parseNodes reads the computing nodes of llama-eval-callback's output, in graph order.
func parseNodes(r io.Reader) ([]Node, error) {
	var out []Node
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<16), 1<<20)
	for sc.Scan() {
		m := nodeRe.FindStringSubmatch(sc.Text())
		if m == nil || viewOps[m[2]] {
			continue
		}
		n := Node{Name: m[1], Op: m[2]}
		var err error
		if n.NE, err = shape(strings.Split(m[4], ",")); err != nil {
			return nil, fmt.Errorf("node %q: %w", sc.Text(), err)
		}
		for _, s := range nodeSrcRe.FindAllStringSubmatch(m[3], -1) {
			sh, _ := shape(s[1:])
			n.Src = append(n.Src, sh)
		}
		out = append(out, n)
	}
	return out, sc.Err()
}

// parseBench reads llama-bench's JSON output: one pp and one tg test.
func parseBench(r io.Reader) (Bench, error) {
	var rows []struct {
		NPrompt int     `json:"n_prompt"`
		NGen    int     `json:"n_gen"`
		NUbatch int     `json:"n_ubatch"`
		AvgTS   float64 `json:"avg_ts"`
	}
	var b Bench
	if err := json.NewDecoder(r).Decode(&rows); err != nil {
		return b, fmt.Errorf("bench: %w", err)
	}
	for _, row := range rows {
		switch {
		case row.NPrompt > 0 && row.NGen == 0:
			b.PP, b.NPrompt, b.NUbatch = row.AvgTS, row.NPrompt, row.NUbatch
		case row.NPrompt == 0 && row.NGen > 0:
			b.TG = row.AvgTS
		}
	}
	if b.PP <= 0 || b.TG <= 0 {
		return b, errors.New("bench: want one pp and one tg test")
	}
	return b, nil
}

// base strips a node name's layer suffix: ffn_moe_logits-12 -> ffn_moe_logits.
func base(name string) string {
	if i := strings.LastIndexByte(name, '-'); i > 0 {
		if _, err := strconv.Atoi(name[i+1:]); err == nil {
			return name[:i]
		}
	}
	return name
}

// sameSrc reports whether a timed op's sources start with a node's: llama-eval-callback prints
// only the first two.
func sameSrc(timed, node [][4]int) bool {
	if len(timed) < len(node) || len(node) < min(len(timed), 2) {
		return false
	}
	for i := range node {
		if timed[i] != node[i] {
			return false
		}
	}
	return true
}

// timings finds the timed ops with a node's op and shapes; among several, those of the same name.
// Several left differ in strides only, which the node listing does not show.
func timings(perf []Perf, n Node) []Perf {
	var cands []Perf
	for _, p := range perf {
		if p.Op == n.Op && p.NE == n.NE && sameSrc(p.Src, n.Src) {
			cands = append(cands, p)
		}
	}
	if len(cands) > 1 {
		var named []Perf
		for _, p := range cands {
			if base(p.Name) == base(n.Name) {
				named = append(named, p)
			}
		}
		if len(named) > 0 {
			cands = named
		}
	}
	return cands
}

// ubatchTiming finds the timing of a router op's other shape: the same op and name, the ubatch.
func ubatchTiming(perf []Perf, name, op string, tgNE [4]int) (Perf, bool) {
	var cands []Perf
	for _, p := range perf {
		if base(p.Name) == name && p.Op == op && p.NE != tgNE {
			cands = append(cands, p)
		}
	}
	if len(cands) != 1 {
		return Perf{}, false
	}
	return cands[0], true
}

func analyze(perf []Perf, nodes []Node, b Bench, layers int) (Report, error) {
	r := Report{Layers: layers, Nodes: len(nodes), Ubatch: min(b.NPrompt, b.NUbatch)}
	isRouter := map[string]bool{}
	for _, n := range routerNames {
		isRouter[n] = true
	}
	index := map[string]int{}
	for _, n := range nodes {
		cands := timings(perf, n)
		name := base(n.Name)
		if (isRouter[name] || name == combineName) && len(cands) != 1 {
			return r, fmt.Errorf("%s (%s %v): %d matching timings, want 1", n.Name, n.Op, n.NE, len(cands))
		}
		if len(cands) == 0 {
			// attention reads the KV cells in use, the exported graph the whole cache
			if r.Untimed == nil {
				r.Untimed = map[string]int{}
			}
			r.Untimed[n.Op]++
			continue
		}
		for _, c := range cands {
			r.AllTG += c.US / float64(len(cands))
		}
		p := cands[0]
		if !isRouter[name] && name != combineName {
			continue
		}
		i, seen := index[name]
		if !seen {
			pp, ok := ubatchTiming(perf, name, n.Op, n.NE)
			if !ok && isRouter[name] {
				return r, fmt.Errorf("%s (%s): no single ubatch-shaped timing", name, n.Op)
			}
			t := OpTime{Name: name, Op: n.Op, NE: n.NE, PPNE: pp.NE, TG: p.US, PP: pp.US}
			if name == combineName {
				r.Combine = t
				index[name] = -1
			} else {
				index[name] = len(r.Router)
				r.Router = append(r.Router, t)
			}
			i = index[name]
		} else if i >= 0 && r.Router[i].NE != n.NE {
			return r, fmt.Errorf("%s: shapes %v and %v in one graph", n.Name, r.Router[i].NE, n.NE)
		}
		if i < 0 {
			r.Combine.Count++
		} else {
			r.Router[i].Count++
		}
	}
	if len(r.Router) == 0 {
		return r, errors.New("no router op among the nodes")
	}
	for _, t := range r.Router {
		if t.Count != layers {
			return r, fmt.Errorf("%s: %d nodes, want one per layer (%d)", t.Name, t.Count, layers)
		}
		r.RouterTG += float64(t.Count) * t.TG
		r.RouterPP += float64(t.Count) * t.PP
	}
	if logits := r.Router[0]; logits.Name == "ffn_moe_logits" && logits.PPNE[1] != r.Ubatch {
		return r, fmt.Errorf("the timings are for a ubatch of %d tokens, llama-bench ran %d", logits.PPNE[1], r.Ubatch)
	}
	r.TokenUS = 1e6 / b.TG
	r.UbatchUS = float64(r.Ubatch) * 1e6 / b.PP
	r.TGShare = r.RouterTG / r.TokenUS
	r.PPShare = r.RouterPP / r.UbatchUS
	return r, nil
}

func open(path string) *os.File {
	f, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	return f
}

func main() {
	perfPath := flag.String("perf", "", "test-backend-ops perf output (console) for the exported graph ops")
	nodesPath := flag.String("nodes", "", "llama-eval-callback output for a one-token prompt")
	benchPath := flag.String("bench", "", "llama-bench -o json output with one pp and one tg test")
	layers := flag.Int("layers", 50, "layers; every router op must occur once per layer")
	out := flag.String("out", "", "write the report as JSON here")
	flag.Parse()
	if *perfPath == "" || *nodesPath == "" || *benchPath == "" {
		flag.Usage()
		os.Exit(2)
	}
	perf, err := parsePerf(open(*perfPath))
	if err != nil {
		log.Fatal(err)
	}
	nodes, err := parseNodes(open(*nodesPath))
	if err != nil {
		log.Fatal(err)
	}
	b, err := parseBench(open(*benchPath))
	if err != nil {
		log.Fatal(err)
	}
	r, err := analyze(perf, nodes, b, *layers)
	if err != nil {
		log.Fatalf("FAIL routing overhead: %v", err)
	}
	fmt.Printf("| Router op | Op | Shape (token) | us per node, token | Shape (ubatch) | us per node, ubatch | Nodes |\n")
	fmt.Printf("|---|---|---|---|---|---|---|\n")
	for _, t := range append(r.Router, r.Combine) {
		fmt.Printf("| %s | %s | %v | %.2f | %v | %.2f | %d |\n", t.Name, t.Op, t.NE, t.TG, t.PPNE, t.PP, t.Count)
	}
	fmt.Printf("router per token: %.3f ms of %.3f ms measured (tg %.2f tokens/s): %.1f%%\n",
		r.RouterTG/1e3, r.TokenUS/1e3, b.TG, 100*r.TGShare)
	fmt.Printf("router per %d-token ubatch: %.3f ms of %.3f ms measured (pp %.2f tokens/s): %.1f%%\n",
		r.Ubatch, r.RouterPP/1e3, r.UbatchUS/1e3, b.PP, 100*r.PPShare)
	fmt.Printf("combine (%s) per token: %.3f ms\n", combineName, float64(r.Combine.Count)*r.Combine.TG/1e3)
	untimed := 0
	for _, c := range r.Untimed {
		untimed += c
	}
	fmt.Printf("%d of %d nodes per token, timed in isolation: %.3f ms (%.0f%% of measured); untimed %v\n",
		r.Nodes-untimed, r.Nodes, r.AllTG/1e3, 100*r.AllTG/r.TokenUS, r.Untimed)
	if *out != "" {
		data, err := json.MarshalIndent(r, "", "  ")
		if err != nil {
			log.Fatal(err)
		}
		if err := os.WriteFile(*out, append(data, '\n'), 0o644); err != nil {
			log.Fatal(err)
		}
	}
	fmt.Printf("PASS routing overhead: %d router ops x %d layers, %.1f%% of a token, %.1f%% of a ubatch\n",
		len(r.Router), r.Layers, 100*r.TGShare, 100*r.PPShare)
}
