package kolibri

import (
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// Group is the coarse tensor category of the HF -> GGUF mapping (docs/checkpoint.md).
type Group string

const (
	GroupEmbedding    Group = "embedding/lm_head"
	GroupAttention    Group = "attention_qkvo"
	GroupQKNorm       Group = "qk_norm"
	GroupBlockNorm    Group = "block_norm"
	GroupRouter       Group = "router"
	GroupRoutedExpert Group = "routed_expert"
	GroupSharedExpert Group = "shared_expert"
)

// Role distinguishes a weight from its FP8 block scale.
type Role string

const (
	RoleWeight   Role = "weight"
	RoleScaleInv Role = "weight_scale_inv"
)

// Spec describes one logical tensor kind of the checkpoint.
type Spec struct {
	// Class is a stable identifier used in the inventory output.
	Class string
	Group Group
	// HF is the safetensors name with {L} (layer) and {E} (expert)
	// placeholders.
	HF string
	// GGUF is the target name. Per-expert HF tensors are stacked into one
	// GGUF tensor per layer, so GGUF never contains {E}.
	GGUF string
	// Linear marks matmul weights; only these get FP8 block scales.
	Linear bool
	// Shape returns the HF (PyTorch, row-major [out, in]) shape of one
	// tensor of this kind.
	Shape func(c *Config) []int64
	// Note documents semantics that the name alone does not convey.
	Note string

	re *regexp.Regexp
}

// PerLayer reports whether the tensor exists once per decoder layer.
func (s *Spec) PerLayer() bool { return strings.Contains(s.HF, "{L}") }

// PerExpert reports whether the HF checkpoint stores one tensor per expert.
func (s *Spec) PerExpert() bool { return strings.Contains(s.HF, "{E}") }

// GGUFShape returns the GGUF/ggml shape (ne[0] innermost) of the converted
// tensor. Per-expert tensors are stacked along a new outermost axis.
func (s *Spec) GGUFShape(c *Config) []int64 {
	ne := slices.Clone(s.Shape(c))
	slices.Reverse(ne)
	if s.PerExpert() {
		ne = append(ne, c.NumExperts)
	}
	return ne
}

func vec(f func(c *Config) int64) func(c *Config) []int64 {
	return func(c *Config) []int64 { return []int64{f(c)} }
}

func mat(out, in func(c *Config) int64) func(c *Config) []int64 {
	return func(c *Config) []int64 { return []int64{out(c), in(c)} }
}

var (
	hidden  = func(c *Config) int64 { return c.HiddenSize }
	vocab   = func(c *Config) int64 { return c.VocabSize }
	headDim = func(c *Config) int64 { return c.HeadDim }
	qDim    = func(c *Config) int64 { return c.NumAttentionHeads * c.HeadDim }
	kvDim   = func(c *Config) int64 { return c.NumKeyValueHeads * c.HeadDim }
	experts = func(c *Config) int64 { return c.NumExperts }
	expFF   = func(c *Config) int64 { return c.MoEIntermediateSize }
	shFF    = func(c *Config) int64 { return c.SharedExpertIntermediateSize }
)

// Specs lists every tensor kind a Kolibri-1 forward pass needs, in forward
// order. The GGUF names follow gguf-py's MODEL_TENSOR naming.
var Specs = []*Spec{
	{Class: "token_embd", Group: GroupEmbedding, HF: "model.embed_tokens.weight", GGUF: "token_embd.weight", Shape: mat(vocab, hidden)},

	{Class: "attn_norm", Group: GroupBlockNorm, HF: "model.layers.{L}.input_layernorm.weight", GGUF: "blk.{L}.attn_norm.weight", Shape: vec(hidden),
		Note: "pre-attention RMSNorm"},
	{Class: "attn_q", Group: GroupAttention, HF: "model.layers.{L}.self_attn.q_proj.weight", GGUF: "blk.{L}.attn_q.weight", Linear: true, Shape: mat(qDim, hidden)},
	{Class: "attn_k", Group: GroupAttention, HF: "model.layers.{L}.self_attn.k_proj.weight", GGUF: "blk.{L}.attn_k.weight", Linear: true, Shape: mat(kvDim, hidden)},
	{Class: "attn_v", Group: GroupAttention, HF: "model.layers.{L}.self_attn.v_proj.weight", GGUF: "blk.{L}.attn_v.weight", Linear: true, Shape: mat(kvDim, hidden)},
	{Class: "attn_q_norm", Group: GroupQKNorm, HF: "model.layers.{L}.self_attn.q_norm.weight", GGUF: "blk.{L}.attn_q_norm.weight", Shape: vec(headDim),
		Note: "per-head RMSNorm over head_dim, shared by all 48 query heads, applied before RoPE"},
	{Class: "attn_k_norm", Group: GroupQKNorm, HF: "model.layers.{L}.self_attn.k_norm.weight", GGUF: "blk.{L}.attn_k_norm.weight", Shape: vec(headDim),
		Note: "per-head RMSNorm over head_dim, shared by all 4 KV heads, applied before RoPE"},
	{Class: "attn_output", Group: GroupAttention, HF: "model.layers.{L}.self_attn.o_proj.weight", GGUF: "blk.{L}.attn_output.weight", Linear: true, Shape: mat(hidden, qDim)},
	{Class: "attn_post_norm", Group: GroupBlockNorm, HF: "model.layers.{L}.post_attn_norm.weight", GGUF: "blk.{L}.post_attention_norm.weight", Shape: vec(hidden),
		Note: "sandwich norm on the attention output, before the residual add (MODEL_TENSOR.ATTN_POST_NORM; the generic map sends this HF name to ATTN_OUT_NORM, which is wrong here)"},
	{Class: "ffn_norm", Group: GroupBlockNorm, HF: "model.layers.{L}.post_attention_layernorm.weight", GGUF: "blk.{L}.ffn_norm.weight", Shape: vec(hidden),
		Note: "pre-FFN RMSNorm despite the HF name (Qwen convention, MODEL_TENSOR.FFN_NORM; Gemma2/OLMo2 map the same HF name to ATTN_POST_NORM)"},

	{Class: "ffn_gate_inp", Group: GroupRouter, HF: "model.layers.{L}.mlp.gate.weight", GGUF: "blk.{L}.ffn_gate_inp.weight", Shape: mat(experts, hidden),
		Note: "router; logits computed in FP32; listed in FP8 modules_to_not_convert"},
	{Class: "exp_probs_b", Group: GroupRouter, HF: "model.layers.{L}.moe.router.expert_bias", GGUF: "blk.{L}.exp_probs_b.bias", Shape: vec(experts),
		Note: "torchtitan routing correction bias; added to the raw logits for top-k selection only, not to sigmoid(logits) as llama.cpp's DeepSeek-V3 path does"},

	{Class: "ffn_gate_exps", Group: GroupRoutedExpert, HF: "model.layers.{L}.mlp.experts.{E}.gate_proj.weight", GGUF: "blk.{L}.ffn_gate_exps.weight", Linear: true, Shape: mat(expFF, hidden)},
	{Class: "ffn_up_exps", Group: GroupRoutedExpert, HF: "model.layers.{L}.mlp.experts.{E}.up_proj.weight", GGUF: "blk.{L}.ffn_up_exps.weight", Linear: true, Shape: mat(expFF, hidden)},
	{Class: "ffn_down_exps", Group: GroupRoutedExpert, HF: "model.layers.{L}.mlp.experts.{E}.down_proj.weight", GGUF: "blk.{L}.ffn_down_exps.weight", Linear: true, Shape: mat(hidden, expFF)},

	{Class: "ffn_gate_shexp", Group: GroupSharedExpert, HF: "model.layers.{L}.mlp.shared_experts.gate_proj.weight", GGUF: "blk.{L}.ffn_gate_shexp.weight", Linear: true, Shape: mat(shFF, hidden),
		Note: "ungated shared expert (no shared_expert_gate); output added to the routed sum with weight 1"},
	{Class: "ffn_up_shexp", Group: GroupSharedExpert, HF: "model.layers.{L}.mlp.shared_experts.up_proj.weight", GGUF: "blk.{L}.ffn_up_shexp.weight", Linear: true, Shape: mat(shFF, hidden)},
	{Class: "ffn_down_shexp", Group: GroupSharedExpert, HF: "model.layers.{L}.mlp.shared_experts.down_proj.weight", GGUF: "blk.{L}.ffn_down_shexp.weight", Linear: true, Shape: mat(hidden, shFF)},

	{Class: "ffn_post_norm", Group: GroupBlockNorm, HF: "model.layers.{L}.post_ffn_norm.weight", GGUF: "blk.{L}.post_ffw_norm.weight", Shape: vec(hidden),
		Note: "sandwich norm on the MoE output (routed + shared), before the residual add (MODEL_TENSOR.FFN_POST_NORM)"},

	{Class: "output_norm", Group: GroupBlockNorm, HF: "model.norm.weight", GGUF: "output_norm.weight", Shape: vec(hidden)},
	{Class: "output", Group: GroupEmbedding, HF: "lm_head.weight", GGUF: "output.weight", Shape: mat(vocab, hidden),
		Note: "untied (tie_word_embeddings=false)"},
}

func init() {
	for _, s := range Specs {
		p := regexp.QuoteMeta(s.HF)
		p = strings.ReplaceAll(p, `\{L\}`, `(?P<L>0|[1-9][0-9]*)`)
		p = strings.ReplaceAll(p, `\{E\}`, `(?P<E>0|[1-9][0-9]*)`)
		s.re = regexp.MustCompile("^" + p + "$")
	}
}

// Entry is a classified checkpoint tensor name.
type Entry struct {
	Spec   *Spec
	Role   Role
	Layer  int // -1 if not per layer
	Expert int // -1 if not per expert
}

// GGUFName returns the target GGUF tensor name. FP8 scales have no GGUF
// counterpart: the converter folds them into the dequantized weight.
func (e Entry) GGUFName() string {
	if e.Role == RoleScaleInv {
		return ""
	}
	return strings.ReplaceAll(e.Spec.GGUF, "{L}", strconv.Itoa(e.Layer))
}

// Classify maps a safetensors tensor name to its Spec.
func Classify(name string) (Entry, error) {
	role, base := RoleWeight, name
	if b, ok := strings.CutSuffix(name, "."+string(RoleScaleInv)); ok {
		role, base = RoleScaleInv, b+".weight"
	}
	for _, s := range Specs {
		m := s.re.FindStringSubmatch(base)
		if m == nil {
			continue
		}
		e := Entry{Spec: s, Role: role, Layer: -1, Expert: -1}
		if i := s.re.SubexpIndex("L"); i >= 0 {
			e.Layer, _ = strconv.Atoi(m[i])
		}
		if i := s.re.SubexpIndex("E"); i >= 0 {
			e.Expert, _ = strconv.Atoi(m[i])
		}
		if role == RoleScaleInv && !s.Linear {
			return Entry{}, fmt.Errorf("kolibri: %s: FP8 scale for non-linear tensor", name)
		}
		return e, nil
	}
	return Entry{}, fmt.Errorf("kolibri: unrecognized tensor %q", name)
}

// Quantized reports whether the FP8 checkpoint should store this spec as
// block-quantized FP8 with a scale tensor.
func (c *Config) Quantized(s *Spec, layer int) bool {
	q := c.QuantizationConfig
	if q == nil || !s.Linear {
		return false
	}
	module := strings.TrimSuffix(strings.ReplaceAll(s.HF, "{L}", strconv.Itoa(layer)), ".weight")
	return !slices.ContainsFunc(q.ModulesToNotConvert, func(m string) bool {
		return m == module || strings.HasPrefix(module, m+".")
	})
}

// ExpectedShape returns the HF shape of the tensor described by e.
func (c *Config) ExpectedShape(e Entry) []int64 {
	shape := e.Spec.Shape(c)
	if e.Role != RoleScaleInv {
		return shape
	}
	if c.QuantizationConfig == nil {
		return nil // a scale tensor in an unquantized checkpoint is never valid
	}
	bs := c.QuantizationConfig.WeightBlockSize
	return []int64{ceilDiv(shape[0], bs[0]), ceilDiv(shape[1], bs[1])}
}

func ceilDiv(a, b int64) int64 { return (a + b - 1) / b }

// ExpectedNames enumerates every tensor name the checkpoint must contain.
func (c *Config) ExpectedNames() []string {
	var names []string
	add := func(s *Spec, layer int, name string) {
		names = append(names, name)
		if c.Quantized(s, layer) {
			names = append(names, strings.TrimSuffix(name, ".weight")+"."+string(RoleScaleInv))
		}
	}
	for _, s := range Specs {
		if !s.PerLayer() {
			add(s, -1, s.HF)
			continue
		}
		for l := range c.NumHiddenLayers {
			n := strings.ReplaceAll(s.HF, "{L}", strconv.Itoa(l))
			if !s.PerExpert() {
				add(s, l, n)
				continue
			}
			for e := range c.NumExperts {
				add(s, l, strings.ReplaceAll(n, "{E}", strconv.FormatInt(e, 10)))
			}
		}
	}
	return names
}
