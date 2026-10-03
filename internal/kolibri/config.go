// Package kolibri holds the Kolibri-1 checkpoint layout: config fields,
// tensor classification, expected shapes, and the HF -> GGUF name mapping.
package kolibri

import (
	"encoding/json"
	"fmt"
)

// Config is the subset of config.json that determines tensor shapes and
// forward-pass semantics.
type Config struct {
	Architectures                []string            `json:"architectures"`
	ModelType                    string              `json:"model_type"`
	HiddenSize                   int64               `json:"hidden_size"`
	NumHiddenLayers              int                 `json:"num_hidden_layers"`
	NumAttentionHeads            int64               `json:"num_attention_heads"`
	NumKeyValueHeads             int64               `json:"num_key_value_heads"`
	HeadDim                      int64               `json:"head_dim"`
	HiddenAct                    string              `json:"hidden_act"`
	MaxPositionEmbeddings        int64               `json:"max_position_embeddings"`
	RMSNormEps                   float64             `json:"rms_norm_eps"`
	VocabSize                    int64               `json:"vocab_size"`
	RopeTheta                    float64             `json:"rope_theta"`
	NumExperts                   int64               `json:"num_experts"`
	NumExpertsPerTok             int64               `json:"num_experts_per_tok"`
	MoEIntermediateSize          int64               `json:"moe_intermediate_size"`
	SharedExpertIntermediateSize int64               `json:"shared_expert_intermediate_size"`
	NormTopKProb                 bool                `json:"norm_topk_prob"`
	AttentionBias                bool                `json:"attention_bias"`
	TieWordEmbeddings            bool                `json:"tie_word_embeddings"`
	UseSlidingWindow             bool                `json:"use_sliding_window"`
	SlidingWindow                int64               `json:"sliding_window"`
	LayerTypes                   []string            `json:"layer_types"`
	BOSTokenID                   *int64              `json:"bos_token_id"`
	EOSTokenID                   *int64              `json:"eos_token_id"`
	PadTokenID                   *int64              `json:"pad_token_id"`
	DType                        string              `json:"dtype"`
	HeadDType                    string              `json:"head_dtype"`
	QuantizationConfig           *QuantizationConfig `json:"quantization_config"`
}

// QuantizationConfig describes the FP8 checkpoint's block quantization.
type QuantizationConfig struct {
	QuantMethod         string   `json:"quant_method"`
	ActivationScheme    string   `json:"activation_scheme"`
	WeightBlockSize     []int64  `json:"weight_block_size"`
	ModulesToNotConvert []string `json:"modules_to_not_convert"`
}

// ParseConfig decodes config.json and checks the invariants the rest of this
// package relies on.
func ParseConfig(raw []byte) (*Config, error) {
	var c Config
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("kolibri: decode config: %w", err)
	}
	if c.ModelType != "kolibri1" {
		return nil, fmt.Errorf("kolibri: model_type %q, want kolibri1", c.ModelType)
	}
	if len(c.LayerTypes) != c.NumHiddenLayers {
		return nil, fmt.Errorf("kolibri: %d layer_types for %d layers", len(c.LayerTypes), c.NumHiddenLayers)
	}
	for i, lt := range c.LayerTypes {
		if lt != "sliding_attention" && lt != "full_attention" {
			return nil, fmt.Errorf("kolibri: layer %d has unknown type %q", i, lt)
		}
	}
	if q := c.QuantizationConfig; q != nil && len(q.WeightBlockSize) != 2 {
		return nil, fmt.Errorf("kolibri: weight_block_size %v, want 2 dims", q.WeightBlockSize)
	}
	return &c, nil
}

// IsFullAttention reports whether layer il is a full-attention (NoPE) layer.
func (c *Config) IsFullAttention(il int) bool { return c.LayerTypes[il] == "full_attention" }

// SWAPeriod returns the period n of a repeating "n-1 sliding, 1 full"
// pattern, or 0 if layer_types does not follow one.
func (c *Config) SWAPeriod() int {
	first := -1
	for i := range c.LayerTypes {
		if c.IsFullAttention(i) {
			first = i
			break
		}
	}
	if first < 0 {
		return 0
	}
	n := first + 1
	for i := range c.LayerTypes {
		if c.IsFullAttention(i) != ((i+1)%n == 0) {
			return 0
		}
	}
	return n
}
