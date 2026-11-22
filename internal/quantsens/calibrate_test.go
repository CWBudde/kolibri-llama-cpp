package quantsens

import (
	"math"
	"testing"
)

func TestRecordKind(t *testing.T) {
	for in, want := range map[string]string{"blk.12.attn": "attn", "blk.10-14.ffn_down_exps": "ffn_down_exps", "attn": "attn", "token_embd": "token_embd"} {
		if got := RecordKind(in); got != want {
			t.Errorf("RecordKind(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestCalibrate(t *testing.T) {
	proxy := []Record{
		{Scope: "base", KLD: 0},
		{Scope: "kind", Item: "attn", Type: "Q4_K", DKL: 0.02},
		{Scope: "kind", Item: "attn", Type: "BF16", DKL: 0.004},
		{Scope: "band:5", Item: "blk.0-4.attn", Type: "Q4_K", DKL: 0.005},
		{Scope: "band:5", Item: "blk.0-4.attn", Type: "BF16", DKL: 0.001},
		{Scope: "band:5", Item: "blk.0-4.shexp", Type: "Q4_K", DKL: 0.001}, // no kind records
	}
	target := []Record{
		{Scope: "base", KLD: 0.1},
		{Scope: "kind", Item: "attn", Type: "Q4_K", DKL: 0.04},
		{Scope: "kind", Item: "attn", Type: "BF16", DKL: -0.002},
	}
	out, ratios, dropped := Calibrate(proxy, target)
	if dropped != 1 || len(ratios) != 2 || len(out) != 4 {
		t.Fatalf("out %d, ratios %v, dropped %d", len(out), ratios, dropped)
	}
	want := map[[2]string]float64{
		{"attn", "Q4_K"}: 0.04, {"attn", "BF16"}: -0.002,
		{"blk.0-4.attn", "Q4_K"}: 0.01, {"blk.0-4.attn", "BF16"}: -0.0005,
	}
	for _, r := range out {
		if w := want[[2]string{r.Item, r.Type}]; math.Abs(r.DKL-w) > 1e-12 {
			t.Errorf("%s %s: ΔKL %g, want %g", r.Item, r.Type, r.DKL, w)
		}
	}
}
