package main

import (
	"math"
	"strings"
	"testing"
)

// A two-layer graph in the formats the real tools print: one router op (two in layer 1 would be a
// count error), an expert matmul and the combine, each timed for one token and for a 4-token
// ubatch by test-backend-ops perf.
const tinyPerf = `Testing 3 devices
Backend 1/3: MTL0
  Device description: Apple M5 Pro
  MUL_MAT(name=ffn_moe_logits-0,type=f32,ne=[8,1,1,1],op_params=[0:10],sources=f32[4,8,1,1],f32[4,1,1,1]):        1000 runs -     6.00 us/run -        1 kB/run - ` + "\x1b[1;34m" + `   1.00 GB/s` + "\x1b[0m" + `
  MUL_MAT(name=ffn_moe_logits-0,type=f32,ne=[8,4,1,1],op_params=[0:10],sources=f32[4,8,1,1],f32[4,4,1,1]):        1000 runs -    40.00 us/run -        1 kB/run -    1.00 GB/s
  MUL_MAT(name=ffn_moe_logits-0,type=f32,ne=[8,2,1,1],op_params=[0:10],sources=f32[4,8,1,1],f32[4,2,1,1]):        1000 runs -    20.00 us/run -        1 kB/run -    1.00 GB/s
  ARGSORT(name=ffn_moe_argsort-0,type=i32,ne=[8,1,1,1],op_params=[0:1],sources=f32[8,1,1,1]):        1000 runs -     4.00 us/run -        1 kB/run -    1.00 GB/s
  ARGSORT(name=ffn_moe_argsort-0,type=i32,ne=[8,4,1,1],op_params=[0:1],sources=f32[8,4,1,1]):        1000 runs -    60.00 us/run -        1 kB/run -    1.00 GB/s
  MUL_MAT_ID(name=ffn_moe_gate-0,type=f32,ne=[2,2,1,1],op_params=[],sources=iq3_xxs[4,2,8,1],f32[4,1,1,1],i32[2,1,1,1]nb[4,8,8,8]):        1000 runs -    30.00 us/run -        1 kB/run -    1.00 GB/s
  SET_ROWS(name=cache_k_l0,type=f16,ne=[4,16,1,1],op_params=[],sources=f32[4,1,1,1],i64[1,1,1,1],f16[4,16,1,1]):        1000 runs -     1.00 us/run -        1 kB/run -    1.00 GB/s
  ADD(name=ffn_inp-0,type=f32,ne=[4,1,1,1],op_params=[],sources=f32[4,1,1,1],f32[4,1,1,1]):        1000 runs -     2.00 us/run -        1 kB/run -    1.00 GB/s
  ADD(name=node_7,type=f32,ne=[4,1,1,1],op_params=[],sources=f32[4,1,1,1],f32[4,1,1,1]nb[4,32,32,32]):        1000 runs -     4.00 us/run -        1 kB/run -    1.00 GB/s
  MUL(name=ffn_moe_weighted-0,type=f32,ne=[4,2,1,1],op_params=[],sources=f32[4,2,1,1],f32[1,2,1,1]):        1000 runs -     2.00 us/run -        1 kB/run -    1.00 GB/s
  Backend MTL0: OK
`

const tinyNodes = `some log line
common_debug_cb_eval:         ffn_moe_logits-0 = (f32)    MUL_MAT(blk.0.ffn_gate_inp.weight{4, 8, 1, 1}, norm-0{4, 1, 1, 1}}) = {8, 1, 1, 1}
                                     [ 0.1, 0.2 ]
common_debug_cb_eval:        ffn_moe_argsort-0 = (i32)    ARGSORT(ffn_moe_logits-0{8, 1, 1, 1}, }) = {8, 1, 1, 1}
common_debug_cb_eval:           ffn_moe_topk-0 = (i32)       VIEW(ffn_moe_argsort-0{8, 1, 1, 1}, }) = {2, 1, 1, 1}
common_debug_cb_eval:           ffn_moe_gate-0 = (f32) MUL_MAT_ID(blk.0.ffn_gate_exps.weight{4, 2, 8, 1}, norm-0{4, 1, 1, 1}, ffn_moe_topk-0{2, 1, 1, 1}}) = {2, 2, 1, 1}
common_debug_cb_eval:       ffn_moe_weighted-0 = (f32)        MUL(ffn_moe_down-0{4, 2, 1, 1}, ffn_moe_weights-0{1, 2, 1, 1}}) = {4, 2, 1, 1}
common_debug_cb_eval: ffn_moe_weighted (view) = (f32)       VIEW(ffn_moe_weighted-0{4, 2, 1, 1}, }) = {4, 1, 1, 1}
common_debug_cb_eval:        cache_k_l0 (view) = (f16)   SET_ROWS(Kcur-0 (view){4, 1, 1, 1}, MTL0#attn_inp_k_idxs#0{1, 1, 1, 1}}) = {4, 16, 1, 1}
common_debug_cb_eval:                   node_9 = (f32) FLASH_ATTN_EXT(Qcur-0 (view){4, 1, 2, 1}, cache_k_l0 (view){4, 3, 1, 1}}) = {4, 2, 1, 1}
common_debug_cb_eval:                    l_out-0 = (f32)        ADD(ffn_out-0{4, 1, 1, 1}, ffn_inp-0{4, 1, 1, 1}}) = {4, 1, 1, 1}
common_debug_cb_eval:         ffn_moe_logits-1 = (f32)    MUL_MAT(blk.1.ffn_gate_inp.weight{4, 8, 1, 1}, norm-1{4, 1, 1, 1}}) = {8, 1, 1, 1}
common_debug_cb_eval:        ffn_moe_argsort-1 = (i32)    ARGSORT(ffn_moe_logits-1{8, 1, 1, 1}, }) = {8, 1, 1, 1}
common_debug_cb_eval:           ffn_moe_gate-1 = (f32) MUL_MAT_ID(blk.1.ffn_gate_exps.weight{4, 2, 8, 1}, norm-1{4, 1, 1, 1}, ffn_moe_topk-1{2, 1, 1, 1}}) = {2, 2, 1, 1}
common_debug_cb_eval:       ffn_moe_weighted-1 = (f32)        MUL(ffn_moe_down-1{4, 2, 1, 1}, ffn_moe_weights-1{1, 2, 1, 1}}) = {4, 2, 1, 1}
`

const tinyBench = `[
  {"n_prompt": 4, "n_gen": 0, "n_ubatch": 4, "avg_ts": 1000.0},
  {"n_prompt": 0, "n_gen": 8, "n_ubatch": 4, "avg_ts": 10000.0}
]`

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

// tinyRouter are the tiny graph's router stages.
var tinyRouter = []string{"ffn_moe_logits", "ffn_moe_argsort"}

func tinyReport(t *testing.T, perf, nodes string, layers int, required []string) (Report, error) {
	t.Helper()
	p, err := parsePerf(strings.NewReader(perf))
	if err != nil {
		t.Fatal(err)
	}
	n, err := parseNodes(strings.NewReader(nodes))
	if err != nil {
		t.Fatal(err)
	}
	b, err := parseBench(strings.NewReader(tinyBench))
	if err != nil {
		t.Fatal(err)
	}
	return analyze(p, n, b, layers, required)
}

func TestParsePerf(t *testing.T) {
	p, err := parsePerf(strings.NewReader(tinyPerf))
	if err != nil {
		t.Fatal(err)
	}
	if len(p) != 10 {
		t.Fatalf("%d timings, want 10", len(p))
	}
	g := p[5]
	if g.Op != "MUL_MAT_ID" || g.Name != "ffn_moe_gate-0" || g.NE != [4]int{2, 2, 1, 1} || g.US != 30 {
		t.Fatalf("p[5] = %+v", g)
	}
	if len(g.Src) != 3 || g.Src[0] != [4]int{4, 2, 8, 1} || g.Src[2] != [4]int{2, 1, 1, 1} {
		t.Fatalf("p[5].Src = %v", g.Src)
	}
}

func TestParseNodes(t *testing.T) {
	n, err := parseNodes(strings.NewReader(tinyNodes))
	if err != nil {
		t.Fatal(err)
	}
	// views are skipped, as test-export-graph-ops skips them
	if len(n) != 11 {
		t.Fatalf("%d nodes, want 11: %+v", len(n), n)
	}
	g := n[2]
	if g.Name != "ffn_moe_gate-0" || g.Op != "MUL_MAT_ID" || g.NE != [4]int{2, 2, 1, 1} || len(g.Src) != 3 {
		t.Fatalf("n[2] = %+v", g)
	}
}

func TestAnalyze(t *testing.T) {
	r, err := tinyReport(t, tinyPerf, tinyNodes, 2, tinyRouter)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Router) != 2 || r.Router[0].Name != "ffn_moe_logits" || r.Router[1].Name != "ffn_moe_argsort" {
		t.Fatalf("Router = %+v", r.Router)
	}
	// per token: 2 layers x (6 + 4) us; per 4-token ubatch: 2 x (40 + 60) us, not the 2-token logits
	if r.Router[0].PPNE != [4]int{8, 4, 1, 1} {
		t.Fatalf("logits ubatch shape %v, want [8 4 1 1]", r.Router[0].PPNE)
	}
	if !near(r.RouterTG, 20) || !near(r.RouterPP, 200) {
		t.Fatalf("RouterTG, RouterPP = %v, %v", r.RouterTG, r.RouterPP)
	}
	// tg at 10000 tokens/s is 100 us per token; pp at 1000 tokens/s is 4000 us per 4-token ubatch
	if !near(r.TokenUS, 100) || !near(r.UbatchUS, 4000) || !near(r.TGShare, 0.2) || !near(r.PPShare, 0.05) {
		t.Fatalf("TokenUS %v UbatchUS %v TGShare %v PPShare %v", r.TokenUS, r.UbatchUS, r.TGShare, r.PPShare)
	}
	if r.Combine.Name != "ffn_moe_weighted" || !near(r.Combine.TG, 2) || r.Combine.Count != 2 {
		t.Fatalf("Combine = %+v", r.Combine)
	}
	// every node: 2 x (6 + 4 + 30 + 2) us, the cache write (three sources, two listed) and l_out,
	// the mean of two ADDs that differ in strides only; the attention over 3 cells has no timing
	if r.Nodes != 11 || !near(r.AllTG, 88) || len(r.Untimed) != 1 || r.Untimed["FLASH_ATTN_EXT"] != 1 {
		t.Fatalf("Nodes, AllTG, Untimed = %d, %v, %v", r.Nodes, r.AllTG, r.Untimed)
	}
}

func TestAnalyzeRejects(t *testing.T) {
	noArgsort := strings.ReplaceAll(tinyPerf, "ARGSORT(", "XARGSORT(")
	layer0 := tinyNodes[:strings.Index(tinyNodes, "common_debug_cb_eval:         ffn_moe_logits-1")]
	var noArgsortNodes []string
	for line := range strings.Lines(tinyNodes) {
		if !strings.Contains(line, "ffn_moe_argsort-0 =") && !strings.Contains(line, "ffn_moe_argsort-1 =") {
			noArgsortNodes = append(noArgsortNodes, line)
		}
	}
	// the argsort's prompt timing is for 2 tokens, not the 4-token ubatch
	wrongUbatch := strings.Replace(tinyPerf, "ARGSORT(name=ffn_moe_argsort-0,type=i32,ne=[8,4,1,1],op_params=[0:1],sources=f32[8,4,1,1])",
		"ARGSORT(name=ffn_moe_argsort-0,type=i32,ne=[8,2,1,1],op_params=[0:1],sources=f32[8,2,1,1])", 1)
	for name, c := range map[string]struct {
		perf, nodes string
		layers      int
	}{
		"router op without timing": {noArgsort, tinyNodes, 2},
		"layers missing":           {tinyPerf, layer0, 2},
		"layer count":              {tinyPerf, tinyNodes, 3},
		"no router":                {tinyPerf, strings.ReplaceAll(tinyNodes, "ffn_moe_", "ffn_x_"), 2},
		"router stage missing":     {tinyPerf, strings.Join(noArgsortNodes, ""), 2},
		"no ubatch-shaped timing":  {wrongUbatch, tinyNodes, 2},
	} {
		if _, err := tinyReport(t, c.perf, c.nodes, c.layers, tinyRouter); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}
