// Command kolibri-tiny writes a tiny random-weight Kolibri-1 checkpoint for
// converter tests, so they need no 156 GB download.
//
//	kolibri-tiny -out DIR -tokenizer-dir DIR [-seed N]
//
// The shape follows the reference repo's tests/checkpoints.py
// (Aleph-Alpha/aleph-alpha-inference@049a6a7): 6 layers (4 sliding, 2 full),
// hidden 256, 8 experts. Differences:
//   - vocab_size and the special-token IDs are the real ones, because the
//     converter reads the real 128k tokenizer (copied from -tokenizer-dir);
//   - weights are BF16 like the released checkpoint, not F32;
//   - norms are random instead of ones, so a converter that swaps two block
//     norms (they share a shape) produces different data;
//   - matrices have standard deviation 0.02 instead of 1 (see initScale), so
//     a forward pass is well-conditioned.
//
// Tensor names and shapes come from internal/kolibri, the same source of
// truth as the Phase 1 inventory. manifest.json lists every expected GGUF
// tensor with its ggml shape and its HF sources in stacking order.
package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"kolibri-llm/internal/kolibri"
	"kolibri-llm/internal/safetensors"
)

// tinyConfig is tests/checkpoints.py KOLIBRI1_CONFIG with the real vocab_size,
// special-token IDs and head_dtype.
const tinyConfig = `{
  "architectures": ["Kolibri1ForCausalLM"],
  "model_type": "kolibri1",
  "hidden_size": 256,
  "num_hidden_layers": 6,
  "num_attention_heads": 8,
  "num_key_value_heads": 2,
  "head_dim": 32,
  "hidden_act": "silu",
  "max_position_embeddings": 2048,
  "rms_norm_eps": 1e-06,
  "vocab_size": 128000,
  "rope_theta": 10000.0,
  "num_experts": 8,
  "num_experts_per_tok": 2,
  "moe_intermediate_size": 256,
  "shared_expert_intermediate_size": 256,
  "norm_topk_prob": false,
  "attention_bias": false,
  "attention_dropout": 0.0,
  "use_sliding_window": true,
  "sliding_window": 65,
  "layer_types": [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "full_attention"
  ],
  "bos_token_id": null,
  "eos_token_id": 127906,
  "pad_token_id": 127901,
  "tie_word_embeddings": false,
  "dtype": "bfloat16",
  "head_dtype": "float32",
  "use_cache": true
}
`

// tokenizerFiles are copied next to the weights for convert_hf_to_gguf.py.
var tokenizerFiles = []string{"tokenizer.json", "tokenizer_config.json"}

// entry is one expected GGUF tensor.
type entry struct {
	Name string `json:"name"`
	// NE is the ggml shape, innermost dimension first.
	NE []int64 `json:"ne"`
	// Sources are the HF tensors, in stacking order for per-expert tensors.
	Sources []string `json:"sources"`
}

func main() {
	out := flag.String("out", "", "output directory")
	tokDir := flag.String("tokenizer-dir", "", "directory with tokenizer.json and tokenizer_config.json (optional)")
	seed := flag.Uint64("seed", 1, "random seed")
	flag.Parse()
	if *out == "" || flag.NArg() != 0 {
		flag.Usage()
		os.Exit(2)
	}
	if err := generate(*out, *tokDir, *seed); err != nil {
		log.Fatal(err)
	}
}

func generate(dir, tokDir string, seed uint64) error {
	cfg, err := kolibri.ParseConfig([]byte(tinyConfig))
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(tinyConfig), 0o644); err != nil {
		return err
	}
	for _, name := range tokenizerFiles {
		if tokDir == "" {
			break
		}
		raw, err := os.ReadFile(filepath.Join(tokDir, name))
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(dir, name), raw, 0o644); err != nil {
			return err
		}
	}

	rng := rand.New(rand.NewPCG(seed, 0))
	var tensors []safetensors.WriteTensor
	for _, name := range cfg.ExpectedNames() {
		e, err := kolibri.Classify(name)
		if err != nil {
			return err
		}
		shape := cfg.ExpectedShape(e)
		mean, std := initScale(e.Spec)
		tensors = append(tensors, safetensors.WriteTensor{Name: name, DType: "BF16", Shape: shape, Data: randBF16(rng, shape, mean, std)})
	}
	if err := writeFile(filepath.Join(dir, "model.safetensors"), func(w *bufio.Writer) error {
		return safetensors.Write(w, tensors, map[string]string{"format": "pt"})
	}); err != nil {
		return err
	}

	raw, err := json.MarshalIndent(manifest(cfg), "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "manifest.json"), append(raw, '\n'), 0o644)
}

// manifest lists the GGUF tensors the converter must produce.
func manifest(cfg *kolibri.Config) []entry {
	var m []entry
	add := func(s *kolibri.Spec, layer int) {
		hf := strings.ReplaceAll(s.HF, "{L}", strconv.Itoa(layer))
		e := entry{Name: strings.ReplaceAll(s.GGUF, "{L}", strconv.Itoa(layer)), NE: s.GGUFShape(cfg)}
		if !s.PerExpert() {
			e.Sources = []string{hf}
		}
		for x := range cfg.NumExperts {
			if s.PerExpert() {
				e.Sources = append(e.Sources, strings.ReplaceAll(hf, "{E}", strconv.FormatInt(x, 10)))
			}
		}
		m = append(m, e)
	}
	for _, s := range kolibri.Specs {
		if !s.PerLayer() {
			add(s, -1)
		}
	}
	for l := range cfg.NumHiddenLayers {
		for _, s := range kolibri.Specs {
			if s.PerLayer() {
				add(s, l)
			}
		}
	}
	return m
}

// initScale returns the mean and standard deviation of a tensor kind.
// Matrices use HF's default initializer_range (0.02) and norms scatter
// around 1, so a forward pass stays well-conditioned; with unit-scale
// matrices the activations grow and tiny rounding differences flip the
// top-k routing. The correction bias keeps unit scale, like the real one
// (sampled ranges about [-7.6, 3.0]), so it decides the routing.
func initScale(s *kolibri.Spec) (mean, std float64) {
	switch {
	case s.Group == kolibri.GroupBlockNorm || s.Group == kolibri.GroupQKNorm:
		return 1, 0.1
	case s.Class == "exp_probs_b":
		return 0, 1
	default:
		return 0, 0.02
	}
}

// randBF16 draws normal values with the given mean and standard deviation,
// rounded to BF16 (nearest even).
func randBF16(rng *rand.Rand, shape []int64, mean, std float64) []byte {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	buf := make([]byte, 2*n)
	for i := range n {
		bits := math.Float32bits(float32(mean + std*rng.NormFloat64()))
		bits += 0x7fff + (bits>>16)&1
		binary.LittleEndian.PutUint16(buf[2*i:], uint16(bits>>16))
	}
	return buf
}

func writeFile(path string, write func(*bufio.Writer) error) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	if err := write(w); err != nil {
		f.Close()
		return fmt.Errorf("%s: %w", path, err)
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
