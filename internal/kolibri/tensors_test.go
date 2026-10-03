package kolibri

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// tinyConfig mirrors KOLIBRI1_CONFIG from the reference repo's
// tests/checkpoints.py (aleph-alpha-inference@049a6a7).
const tinyConfig = `{
  "model_type": "kolibri1", "hidden_size": 256, "num_hidden_layers": 6,
  "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 32,
  "vocab_size": 96000, "num_experts": 8, "num_experts_per_tok": 2,
  "moe_intermediate_size": 256, "shared_expert_intermediate_size": 256,
  "sliding_window": 65,
  "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                  "sliding_attention", "full_attention", "full_attention"]
}`

func loadConfig(t *testing.T, path string) *Config {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	c, err := ParseConfig(raw)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestClassify(t *testing.T) {
	tests := []struct {
		name, class string
		role        Role
		layer, exp  int
		gguf        string
	}{
		{"model.embed_tokens.weight", "token_embd", RoleWeight, -1, -1, "token_embd.weight"},
		{"lm_head.weight", "output", RoleWeight, -1, -1, "output.weight"},
		{"model.layers.7.input_layernorm.weight", "attn_norm", RoleWeight, 7, -1, "blk.7.attn_norm.weight"},
		{"model.layers.7.post_attn_norm.weight", "attn_post_norm", RoleWeight, 7, -1, "blk.7.post_attention_norm.weight"},
		{"model.layers.7.post_attention_layernorm.weight", "ffn_norm", RoleWeight, 7, -1, "blk.7.ffn_norm.weight"},
		{"model.layers.7.post_ffn_norm.weight", "ffn_post_norm", RoleWeight, 7, -1, "blk.7.post_ffw_norm.weight"},
		{"model.layers.49.moe.router.expert_bias", "exp_probs_b", RoleWeight, 49, -1, "blk.49.exp_probs_b.bias"},
		{"model.layers.3.mlp.experts.383.down_proj.weight", "ffn_down_exps", RoleWeight, 3, 383, "blk.3.ffn_down_exps.weight"},
		{"model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv", "ffn_gate_exps", RoleScaleInv, 3, 0, ""},
		{"model.layers.0.mlp.shared_experts.up_proj.weight", "ffn_up_shexp", RoleWeight, 0, -1, "blk.0.ffn_up_shexp.weight"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			e, err := Classify(tc.name)
			if err != nil {
				t.Fatal(err)
			}
			if e.Spec.Class != tc.class || e.Role != tc.role || e.Layer != tc.layer || e.Expert != tc.exp || e.GGUFName() != tc.gguf {
				t.Errorf("got %s/%s L%d E%d %q", e.Spec.Class, e.Role, e.Layer, e.Expert, e.GGUFName())
			}
		})
	}
	for _, bad := range []string{
		"model.layers.1.mlp.shared_expert.gate_proj.weight", // Qwen2-MoE spelling
		"model.layers.01.input_layernorm.weight",
		"model.layers.1.input_layernorm.weight_scale_inv", // norms are never FP8
		"model.layers.1.mlp.gate.e_score_correction_bias",
	} {
		if _, err := Classify(bad); err == nil {
			t.Errorf("Classify(%q) succeeded", bad)
		}
	}
}

func TestGGUFNamesUnique(t *testing.T) {
	seen := map[string]string{}
	for _, s := range Specs {
		if prev, ok := seen[s.GGUF]; ok {
			t.Errorf("%s and %s both map to %s", prev, s.Class, s.GGUF)
		}
		seen[s.GGUF] = s.Class
	}
}

// TestTinyCheckpointNames checks ExpectedNames against the tensor set that
// write_kolibri1_checkpoint() produces in the reference tests.
func TestTinyCheckpointNames(t *testing.T) {
	c, err := ParseConfig([]byte(tinyConfig))
	if err != nil {
		t.Fatal(err)
	}
	names := c.ExpectedNames()
	// 3 global + 6 layers * (6 attn + 4 norms + gate + bias + 3 shared + 8*3 experts)
	if want := 3 + 6*(6+4+1+1+3+8*3); len(names) != want {
		t.Errorf("%d names, want %d", len(names), want)
	}
	if c.SWAPeriod() != 0 {
		t.Errorf("tiny config (4 SWA + 2 full) reported period %d", c.SWAPeriod())
	}
	for _, n := range names {
		if _, err := Classify(n); err != nil {
			t.Error(err)
		}
	}
}

func TestReleasedConfigs(t *testing.T) {
	bf16 := loadConfig(t, "../../inventory/bf16/config.json")
	fp8 := loadConfig(t, "../../inventory/fp8/config.json")

	if p := bf16.SWAPeriod(); p != 5 {
		t.Errorf("SWAPeriod = %d, want 5", p)
	}
	if bf16.IsFullAttention(3) || !bf16.IsFullAttention(4) || !bf16.IsFullAttention(49) {
		t.Error("full-attention layers are not 4, 9, ..., 49")
	}
	if n := len(bf16.ExpectedNames()); n != 58353 {
		t.Errorf("BF16 expects %d tensors, want 58353", n)
	}
	if n := len(fp8.ExpectedNames()); n != 116303 {
		t.Errorf("FP8 expects %d tensors, want 116303", n)
	}

	router, _ := Classify("model.layers.12.mlp.gate.weight")
	expert, _ := Classify("model.layers.12.mlp.experts.5.down_proj.weight")
	if fp8.Quantized(router.Spec, 12) || !fp8.Quantized(expert.Spec, 12) || bf16.Quantized(expert.Spec, 12) {
		t.Error("router must stay BF16 and experts FP8 only in the FP8 checkpoint")
	}
	expert.Role = RoleScaleInv
	if got := fp8.ExpectedShape(expert); !slices.Equal(got, []int64{20, 4}) {
		t.Errorf("down_proj scale shape %v, want [20 4]", got)
	}

	if got := bf16.ExpectedShape(expert); got != nil {
		t.Errorf("BF16 config accepted a scale tensor with shape %v", got)
	}

	exps, _ := Classify("model.layers.0.mlp.experts.0.gate_proj.weight")
	if got := exps.Spec.GGUFShape(bf16); !slices.Equal(got, []int64{2560, 512, 384}) {
		t.Errorf("ffn_gate_exps GGUF shape %v", got)
	}
}

// TestCommittedInventory re-checks the committed tensors.jsonl.gz files
// against the current classification, so the code and the inventory cannot
// drift apart silently.
func TestCommittedInventory(t *testing.T) {
	for _, variant := range []string{"bf16", "fp8"} {
		t.Run(variant, func(t *testing.T) {
			dir := filepath.Join("../../inventory", variant)
			c := loadConfig(t, filepath.Join(dir, "config.json"))
			f, err := os.Open(filepath.Join(dir, "tensors.jsonl.gz"))
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			zr, err := gzip.NewReader(f)
			if err != nil {
				t.Fatal(err)
			}
			want := map[string]bool{}
			for _, n := range c.ExpectedNames() {
				want[n] = true
			}
			sc := bufio.NewScanner(zr)
			sc.Buffer(nil, 1<<20)
			n := 0
			for sc.Scan() {
				var r struct {
					Name  string  `json:"name"`
					Shape []int64 `json:"shape"`
					Class string  `json:"class"`
					GGUF  string  `json:"gguf"`
				}
				if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
					t.Fatal(err)
				}
				n++
				e, err := Classify(r.Name)
				if err != nil {
					t.Error(err)
					continue
				}
				if !want[r.Name] {
					t.Errorf("%s: unexpected", r.Name)
				}
				delete(want, r.Name)
				if e.Spec.Class != r.Class || e.GGUFName() != r.GGUF || !slices.Equal(c.ExpectedShape(e), r.Shape) {
					t.Errorf("%s: inventory says %s %q %v", r.Name, r.Class, r.GGUF, r.Shape)
				}
			}
			if err := sc.Err(); err != nil {
				t.Fatal(err)
			}
			if len(want) > 0 {
				t.Errorf("%d expected tensors missing from inventory (of %d)", len(want), n)
			}
		})
	}
}
