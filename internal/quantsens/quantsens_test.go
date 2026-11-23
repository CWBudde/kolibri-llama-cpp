package quantsens

import (
	"bytes"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"kolibri-llm/internal/gguf"
)

func header(t *testing.T, tensors []gguf.Tensor) *gguf.File {
	t.Helper()
	var buf bytes.Buffer
	if _, err := gguf.WriteHeader(&buf, 32, map[string]any{"general.architecture": "kolibri1"}, tensors); err != nil {
		t.Fatal(err)
	}
	f, err := gguf.Read(&buf)
	if err != nil {
		t.Fatal(err)
	}
	return f
}

func twoLayers() []gguf.Tensor {
	var ts []gguf.Tensor
	add := func(name string, typ gguf.Type, dims ...uint64) {
		ts = append(ts, gguf.Tensor{Name: name, Dims: dims, Type: typ})
	}
	add("token_embd.weight", gguf.Q8_0, 64, 10)
	for _, l := range []string{"0", "1"} {
		p := "blk." + l + "."
		add(p+"attn_norm.weight", gguf.F32, 64)
		add(p+"attn_q.weight", gguf.Q8_0, 64, 64)
		add(p+"attn_k.weight", gguf.Q8_0, 64, 32)
		add(p+"attn_v.weight", gguf.Q8_0, 64, 32)
		add(p+"attn_output.weight", gguf.Q8_0, 64, 64)
		add(p+"ffn_gate_inp.weight", gguf.F32, 64, 8)
		add(p+"exp_probs_b.bias", gguf.F32, 8)
		add(p+"ffn_gate_exps.weight", gguf.IQ3_XXS, 256, 32, 8)
		add(p+"ffn_up_exps.weight", gguf.IQ3_XXS, 256, 32, 8)
		add(p+"ffn_down_exps.weight", gguf.IQ4_XS, 256, 256, 8)
		add(p+"ffn_gate_shexp.weight", gguf.Q8_0, 64, 32)
		add(p+"ffn_up_shexp.weight", gguf.Q8_0, 64, 32)
		add(p+"ffn_down_shexp.weight", gguf.Q8_0, 32, 64)
	}
	add("output_norm.weight", gguf.F32, 64)
	add("output.weight", gguf.Q8_0, 64, 10)
	return ts
}

func TestItems(t *testing.T) {
	f := header(t, twoLayers())
	items, err := Items(f, "layer")
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, it := range items {
		names = append(names, it.Name)
	}
	want := []string{
		"blk.0.ffn_gate_exps", "blk.0.ffn_up_exps", "blk.0.ffn_down_exps", "blk.0.shexp", "blk.0.attn",
		"blk.1.ffn_gate_exps", "blk.1.ffn_up_exps", "blk.1.ffn_down_exps", "blk.1.shexp", "blk.1.attn",
		"token_embd", "output",
	}
	if !slices.Equal(names, want) {
		t.Fatalf("items %v\nwant %v", names, want)
	}
	attn := items[4]
	if attn.Layer != 0 || attn.Kind != "attn" || len(attn.Tensors) != 4 {
		t.Fatalf("blk.0.attn = %+v", attn)
	}
	for _, it := range items {
		for _, n := range it.Tensors {
			if strings.Contains(n, "norm") || strings.Contains(n, "gate_inp") || strings.Contains(n, "probs") {
				t.Fatalf("%s holds %s, which no plan may change", it.Name, n)
			}
		}
	}
	kinds, err := Items(f, "kind")
	if err != nil {
		t.Fatal(err)
	}
	if len(kinds) != len(Kinds) {
		t.Fatalf("%d kind items, want %d", len(kinds), len(Kinds))
	}
	for _, it := range kinds {
		if it.Layer != -1 {
			t.Fatalf("%s: layer %d in kind scope", it.Name, it.Layer)
		}
		if it.Kind == "attn" && len(it.Tensors) != 8 {
			t.Fatalf("kind attn has %d tensors, want 8", len(it.Tensors))
		}
	}
	b, typ, err := ItemBytes(f, items[4])
	if err != nil || typ != "Q8_0" || b != (64*64*2+64*32*2)/32*34 {
		t.Fatalf("blk.0.attn: %d bytes, %s, %v", b, typ, err)
	}
	if n, err := TypeBytes(f, items[4], "q8_0"); err != nil || n != b {
		t.Fatalf("TypeBytes(blk.0.attn, q8_0) = %d, %v; header says %d", n, err, b)
	}
	if _, err := TypeBytes(f, items[4], "NOPE"); err == nil {
		t.Fatal("unknown type priced")
	}
	gate := kinds[slices.IndexFunc(kinds, func(i Item) bool { return i.Kind == "ffn_gate_exps" })]
	if _, typ, _ := ItemBytes(f, gate); typ != "IQ3_XXS" {
		t.Fatalf("kind ffn_gate_exps type %s", typ)
	}
}

// The tail of a real run (format of tools/perplexity/perplexity.cpp).
const kldTail = `
====== Perplexity statistics ======
Mean PPL(Q)                   :  27.701223 ±   1.348882
Mean PPL(base)                :  27.882732 ±   1.360090
Cor(ln(PPL(Q)), ln(PPL(base))):  98.31%
Mean ln(PPL(Q)/PPL(base))     :  -0.006532 ±   0.006883
Mean PPL(Q)/PPL(base)         :   0.993489 ±   0.006838

====== KL divergence statistics ======
Mean    KLD:   0.095291 ±   0.006905
Maximum KLD:  11.307846
99.9%   KLD:   5.081012

====== Token probability statistics ======
Mean    Δp:  0.051 ± 0.104 %
Same top p: 88.540 ±  0.446 %
`

func TestParseKLD(t *testing.T) {
	s, err := ParseKLD(kldTail)
	if err != nil {
		t.Fatal(err)
	}
	if s.KLD != 0.095291 || s.KLDErr != 0.006905 || s.SameTop != 88.54 || s.SameTopErr != 0.446 || s.LnPPLRatio != -0.006532 {
		t.Fatalf("%+v", s)
	}
	if s.RouteTokens != 0 || s.RouteLayers != nil {
		t.Fatalf("route stats without a route run: %+v", s)
	}
	r, err := ParseKLD(kldTail + `
moe_route: router agreement with the reference ids (own top-6 vs file)
moe_route: layer  same set  mean overlap
moe_route:     1   99.512%   5.995
moe_route:    12   96.000%   5.950
moe_route: all     97.756%   5.972  (20480 token-layers)
`)
	if err != nil {
		t.Fatal(err)
	}
	if r.RouteSame != 97.756 || r.RouteOverlap != 5.972 || r.RouteTokens != 20480 || len(r.RouteLayers) != 2 || r.RouteLayers[12] != 96 {
		t.Fatalf("%+v", r)
	}
	if _, err := ParseKLD("llama_model_load: error loading model"); err == nil {
		t.Fatal("parsed a failed run")
	}
}

// brute finds the optimum by enumeration.
func brute(ch []Choice, capB int64) float64 {
	best := math.Inf(1)
	var rec func(i int, bytes int64, cost float64)
	rec = func(i int, bytes int64, cost float64) {
		if bytes > capB {
			return
		}
		if i == len(ch) {
			best = min(best, cost)
			return
		}
		for _, o := range ch[i].Options {
			rec(i+1, bytes+o.Bytes, cost+o.DKL)
		}
	}
	rec(0, 0, 0)
	return best
}

// TestAllocateMatchesBruteForce: at unit 1 the DP is exact.
func TestAllocateMatchesBruteForce(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for trial := range 300 {
		n := 1 + r.IntN(6)
		var ch []Choice
		var minB int64
		for i := range n {
			c := Choice{Item: string(rune('a' + i))}
			k := 1 + r.IntN(4)
			lo := int64(1 << 30)
			for range k {
				b := int64(1 + r.IntN(20))
				lo = min(lo, b)
				c.Options = append(c.Options, Option{Type: "t", Bytes: b, DKL: r.Float64()*2 - 0.5})
			}
			minB += lo
			ch = append(ch, c)
		}
		fixed := int64(r.IntN(5))
		budget := fixed + minB + int64(r.IntN(40))
		p, err := Allocate(ch, fixed, budget, 1)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		want := brute(ch, budget-fixed)
		if math.Abs(p.DKL-want) > 1e-9 {
			t.Fatalf("trial %d: DKL %g, brute force %g", trial, p.DKL, want)
		}
		if p.Total > budget {
			t.Fatalf("trial %d: total %d over budget %d", trial, p.Total, budget)
		}
		var sum float64
		for i, pk := range p.Picks {
			if pk.Item != ch[i].Item {
				t.Fatalf("trial %d: pick %d is %s", trial, i, pk.Item)
			}
			sum += pk.DKL
		}
		if math.Abs(sum-p.DKL) > 1e-12 {
			t.Fatalf("trial %d: picks sum to %g, plan says %g", trial, sum, p.DKL)
		}
	}
	if _, err := Allocate([]Choice{{Item: "a", Options: []Option{{Bytes: 10}}}}, 5, 12, 1); err == nil {
		t.Fatal("allocated past the budget")
	}
}

// TestAllocateRoundsUp: sizes round up to the unit, so a coarse unit stays
// under the budget at the price of missing a plan that fits only within the
// rounding (here small+big, 1600 of 1650 bytes, is 3+4 = 7 units of 256
// against 6); a fine unit finds it.
func TestAllocateRoundsUp(t *testing.T) {
	ch := []Choice{
		{Item: "a", Options: []Option{{Type: "big", Bytes: 1000}, {Type: "small", Bytes: 600, DKL: 1}}},
		{Item: "b", Options: []Option{{Type: "big", Bytes: 1000}, {Type: "small", Bytes: 600, DKL: 2}}},
	}
	p, err := Allocate(ch, 0, 1650, 256)
	if err != nil {
		t.Fatal(err)
	}
	if p.Total > 1650 || p.Picks[0].Type != "small" || p.Picks[1].Type != "small" {
		t.Fatalf("unit 256: %+v", p)
	}
	p, err = Allocate(ch, 0, 1650, 50)
	if err != nil {
		t.Fatal(err)
	}
	if p.Total != 1600 || p.Picks[0].Type != "small" || p.Picks[1].Type != "big" {
		t.Fatalf("unit 50: %+v", p)
	}
}

func TestBuildChoices(t *testing.T) {
	infos := []ItemInfo{
		{Item: Item{Name: "blk.0.attn", Kind: "attn", Layer: 0}, BaseType: "Q8_0", BaseBytes: 100, Donor: map[string]int64{"Q4_K": 50, "Q8_0": 100, "IQ4_XS": 45}},
		{Item: Item{Name: "blk.1.attn", Kind: "attn", Layer: 1}, BaseType: "Q8_0", BaseBytes: 100, Donor: map[string]int64{"Q4_K": 50, "Q8_0": 100, "IQ4_XS": 45}},
	}
	recs := []Record{
		{Scope: "base", KLD: 0.1},
		{Scope: "kind", Item: "attn", Type: "Q4_K", DKL: 0.02},
		{Scope: "layer", Item: "blk.1.attn", Type: "Q4_K", DKL: 0.015},
	}
	ch := BuildChoices(infos, recs)
	if len(ch) != 2 {
		t.Fatalf("%d choices", len(ch))
	}
	// blk.0: base + derived Q4_K (0.02 / 2); IQ4_XS has no measurement.
	if len(ch[0].Options) != 2 || ch[0].Options[0].Type != "Q8_0" || ch[0].Options[0].DKL != 0 ||
		ch[0].Options[1] != (Option{Type: "Q4_K", Bytes: 50, DKL: 0.01, Derived: true}) {
		t.Fatalf("blk.0 options %+v", ch[0].Options)
	}
	if len(ch[1].Options) != 2 || ch[1].Options[1] != (Option{Type: "Q4_K", Bytes: 50, DKL: 0.015}) {
		t.Fatalf("blk.1 options %+v", ch[1].Options)
	}
}

func TestTensorTypes(t *testing.T) {
	f := header(t, twoLayers())
	items, _ := Items(f, "layer")
	var p Plan
	for _, it := range items {
		p.Picks = append(p.Picks, Pick{Item: it.Name, Option: Option{Type: "IQ4_XS"}})
	}
	s, err := TensorTypes(p, items)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(s, "^blk\\.1\\.ffn_down_shexp\\.weight$=iq4_xs\n") || !strings.Contains(s, "^token_embd\\.weight$=iq4_xs\n") {
		t.Fatalf("tensor types:\n%s", s)
	}
	if n := strings.Count(s, "\n"); n != 2*(3+3+4)+2 {
		t.Fatalf("%d lines", n)
	}
	if _, err := TensorTypes(Plan{Picks: []Pick{{Item: "blk.9.attn"}}}, items); err == nil {
		t.Fatal("unknown item accepted")
	}
}

func TestItemsBand(t *testing.T) {
	f := header(t, twoLayers())
	items, err := Items(f, "band:5")
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, it := range items {
		names = append(names, it.Name)
	}
	// Two layers in a band of 5: one band, cut at the last layer.
	want := []string{"blk.0-1.ffn_gate_exps", "blk.0-1.ffn_up_exps", "blk.0-1.ffn_down_exps", "blk.0-1.shexp", "blk.0-1.attn", "token_embd", "output"}
	if !slices.Equal(names, want) {
		t.Fatalf("items %v\nwant %v", names, want)
	}
	if it := items[4]; it.Layer != 0 || len(it.Tensors) != 8 {
		t.Fatalf("band attn = %+v", it)
	}
	one, err := Items(f, "band:1")
	if err != nil || len(one) != 12 || one[0].Name != "blk.0.ffn_gate_exps" {
		t.Fatalf("band:1 gives %d items (%v), want the layer scope's names", len(one), err)
	}
	for _, s := range []string{"band:0", "band:x", "bands"} {
		if _, err := Items(f, s); err == nil {
			t.Fatalf("scope %q accepted", s)
		}
	}
	if a, b, k, ok := ParseBand("blk.10-14.ffn_down_exps"); !ok || a != 10 || b != 14 || k != "ffn_down_exps" {
		t.Fatalf("ParseBand = %d %d %s %v", a, b, k, ok)
	}
}

func TestBuildChoicesBand(t *testing.T) {
	info := func(l int) ItemInfo {
		return ItemInfo{Item: Item{Name: fmt.Sprintf("blk.%d.attn", l), Kind: "attn", Layer: l}, BaseType: "Q8_0", BaseBytes: 100, Donor: map[string]int64{"Q4_K": 50}}
	}
	infos := []ItemInfo{info(0), info(1), info(2), info(3),
		{Item: Item{Name: "output", Kind: "output", Layer: -1}, BaseType: "Q8_0", BaseBytes: 100, Donor: map[string]int64{"Q4_K": 50}}}
	recs := []Record{
		{Scope: "kind", Item: "attn", Type: "Q4_K", DKL: 0.4},
		{Scope: "band:2", Item: "blk.0-1.attn", Type: "Q4_K", DKL: 0.02},
		{Scope: "layer", Item: "blk.1.attn", Type: "Q4_K", DKL: 0.015},
		{Scope: "band:2", Item: "output", Type: "Q4_K", DKL: 0.03},
	}
	ch := BuildChoices(infos, recs)
	got := []float64{}
	for _, c := range ch {
		got = append(got, c.Options[1].DKL)
	}
	// blk.0 from the band (0.02/2), blk.1 its own record, blk.2-3 the kind (0.4/4), output whole.
	if want := []float64{0.01, 0.015, 0.1, 0.1, 0.03}; !slices.Equal(got, want) {
		t.Fatalf("ΔKL %v, want %v", got, want)
	}
	if !ch[0].Options[1].Derived || ch[1].Options[1].Derived || ch[4].Options[1].Derived {
		t.Fatalf("derived flags %+v", ch)
	}
}
