# Kolibri GGUF architecture and conversion

This document covers how the `kolibri` architecture is registered in gguf-py
and libllama, which GGUF metadata the converter writes, and how it maps the
Hugging Face tensors. The model class that loads the result is described in
[model.md](model.md); the tokenizer in [tokenizer.md](tokenizer.md).

The changes live in five patches. They apply in order on top of
`0001-kolibri-tokenizer.patch`, at llama.cpp `1537a0a8`:

| Patch | Content |
|---|---|
| `0002-kolibri-arch.patch` | architecture registration and tensor list |
| `0003-kolibri-converter.patch` | converter class `KolibriModel`, base metadata |
| `0004-kolibri-moe-swa-metadata.patch` | MoE and hybrid-attention metadata |
| `0005-kolibri-router-gating.patch` | router gating metadata |
| `0006-kolibri-tensor-conversion.patch` | tensor mapping and expert stacking |

Applied to the pinned commit, patches 0001–0006 give exactly the tree of the
`feat/kolibri` branch of [CWBudde/llama.cpp](https://github.com/CWBudde/llama.cpp)
at that point. The one exception is `models/ggml-vocab-kolibri.gguf`, which the
fork commits and the patches do not.

## Architecture registration

Each new entry sits next to the matching `laguna` entry.

| File | Change |
|---|---|
| `gguf-py/gguf/constants.py` | `MODEL_ARCH.KOLIBRI`; `MODEL_ARCH_NAMES` → `"kolibri"`; `MODEL_TENSORS[KOLIBRI]` |
| `src/llama-arch.h` | `LLM_ARCH_KOLIBRI` |
| `src/llama-arch.cpp` | `LLM_ARCH_NAMES` → `"kolibri"` |
| `src/llama-model.cpp` | `llama_model_rope_type`: `LLAMA_ROPE_TYPE_NEOX`, because the sliding layers use NEOX-style RoPE ([checkpoint.md](checkpoint.md)). Without this case, `-Wswitch` warns. |

`llama_model_create` needs a model class for every architecture, even for a
`vocab_only` load. The class `llama_model_kolibri` and its `test-llama-archs`
entry are described in [model.md](model.md).

### Tensor list

`MODEL_TENSORS[KOLIBRI]` holds exactly the 21 tensor kinds of the
`HF → GGUF` mapping in [checkpoint.md](checkpoint.md):

- embedding, head and output norm: `TOKEN_EMBD`, `OUTPUT_NORM`, `OUTPUT`;
- attention: `ATTN_NORM`, `ATTN_Q`, `ATTN_K`, `ATTN_V`, `ATTN_OUT`,
  `ATTN_Q_NORM`, `ATTN_K_NORM`, `ATTN_POST_NORM`;
- MoE block: `FFN_NORM`, `FFN_GATE_INP`, `FFN_EXP_PROBS_B`,
  `FFN_{GATE,UP,DOWN}_EXP`, `FFN_{GATE,UP,DOWN}_SHEXP`, `FFN_POST_NORM`.

The sandwich norms are `ATTN_POST_NORM` (`post_attention_norm`) and
`FFN_POST_NORM` (`post_ffw_norm`). Kolibri has no dense FFN and no attention
gate.

`tensor_mapping.py` is not changed. Its generic entries map Kolibri's norm
names to the wrong tensors ([checkpoint.md, pitfall 1](checkpoint.md#pitfalls)),
so the converter class maps all four block norms explicitly.

## Converter class

`conversion/kolibri.py` defines `KolibriModel`, registered for
`Kolibri1ForCausalLM`; `conversion/__init__.py` adds the `TEXT_MODEL_MAP`
entry `"Kolibri1ForCausalLM": "kolibri"`. The class has three parts:

- **`set_vocab`:** calls `_set_vocab_gpt2()`. The pre-tokenizer hash resolves
  to `kolibri` ([tokenizer.md](tokenizer.md)).
- **`set_gguf_parameters`:** calls `super()`, then writes the keys that
  `TextModel` does not write or would write differently (see
  [Metadata](#metadata)).
- **`modify_tensors`:** maps every tensor; see
  [Tensor conversion](#tensor-conversion).

All keys use existing gguf-py writers; Kolibri needs no new GGUF keys.

## Metadata

| GGUF key | Value | Written by | Source |
|---|---|---|---|
| `kolibri.block_count` | 50 | `TextModel` | `config.json` |
| `kolibri.context_length` | 262144 | `TextModel` | `config.json` |
| `kolibri.embedding_length` | 2560 | `TextModel` | `config.json` |
| `kolibri.attention.head_count`, `.head_count_kv` | 48, 4 | `TextModel` | `config.json` |
| `kolibri.attention.key_length`, `.value_length` | 128, 128 | `TextModel` | `head_dim` |
| `kolibri.rope.dimension_count` | 128 | `KolibriModel` | `head_dim` |
| `kolibri.attention.layer_norm_rms_epsilon` | 1e-6 (as float32) | `TextModel` | `config.json` |
| `kolibri.expert_count`, `.expert_used_count` | 384, 6 | `TextModel` | `config.json` |
| `kolibri.expert_feed_forward_length` | 512 | `KolibriModel` | `moe_intermediate_size` |
| `kolibri.expert_shared_count` | 1 | `KolibriModel` | checkpoint |
| `kolibri.expert_shared_feed_forward_length` | 512 | `KolibriModel` | checkpoint |
| `kolibri.attention.sliding_window` | 513 | `KolibriModel` | `config.json` |
| `kolibri.attention.sliding_window_pattern` | bool[50], true = SWA | `KolibriModel` | `layer_types` |
| `kolibri.rope.freq_base` | 10000 | `TextModel` | `config.json` |
| `kolibri.attention.rope_pattern` | bool[50], true = RoPE | `KolibriModel` | `layer_types` |
| `kolibri.expert_gating_func` | 2 (`LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`) | `KolibriModel` | router semantics |
| `kolibri.expert_weights_norm` | false | `KolibriModel` | `norm_topk_prob` |
| `kolibri.expert_weights_scale` | 1.0 | `KolibriModel` | vLLM `routed_scaling_factor` |

- **RoPE dimension.** The sliding layers rotate the full head, so
  `rope.dimension_count = head_dim`.
- **Shared expert.** The checkpoint has one ungated shared expert per layer,
  with `gate_proj` shape [512, 2560]. Qwen2/3-MoE write only the shared FFN
  length. Kolibri also writes `expert_shared_count = 1`, so the count does not
  depend on a model-class default.
- **Sliding window 513.** `config.json` stores `sliding_window = 513`, and vLLM
  passes it to FlashAttention as `window = (512, 0)`: the 512 preceding tokens
  plus the current one. llama.cpp `LLAMA_SWA_TYPE_STANDARD` masks a key when
  `p1 - p0 >= n_swa` (`src/llama-hparams.h`). With `n_swa = 513` that is the
  same window, so the value is stored unchanged. [attention.md](attention.md)
  tests the 511, 512 and 513 boundaries numerically.
- **Patterns.** Both arrays come from `layer_types`:
  - `sliding_window_pattern` is read by a required `ml.get_arr` into
    `hparams.is_swa_impl` in `llama_model_kolibri::load_arch_hparams`, not by
    `load_swa_pattern`; see the evaluation in [attention.md](attention.md).
  - `rope_pattern` is read into `hparams.rope_pattern` and used by
    `llama_hparams::has_rope`. It equals the SWA pattern, because the
    full-attention layers 4, 9, …, 49 use no positional encoding (RNoPE).
- **Gating function.** Kolibri weights each selected expert with the sigmoid
  of its raw router logit (`sigmoid_logit_add_routing`, see
  [checkpoint.md, router semantics](checkpoint.md#router-semantics)).
  `TextModel` writes `expert_gating_func` only when the config names a scoring
  function, and Kolibri's `config.json` names none. So `KolibriModel` writes it,
  as the GLM, Laguna and Hunyuan converters do.
- **No renormalization, no scaling.** The six selected weights are used as
  they are. `expert_weights_scale = 1.0` is written explicitly, so the value
  does not depend on a loader default.
- **Correction bias.** These keys do not describe it. Kolibri adds
  `expert_bias` to the raw logits for the selection only, while
  `build_moe_ffn` adds `exp_probs_b` to the sigmoid output. The model class
  therefore has an arch-specific selection branch
  ([checkpoint.md, pitfall 2](checkpoint.md#pitfalls); [model.md](model.md)).

## Tensor conversion

`KolibriModel.modify_tensors` maps the tensors as follows:

| HF tensor (per layer) | GGUF tensor | How |
|---|---|---|
| `input_layernorm` | `attn_norm` | explicit |
| `post_attn_norm` | `post_attention_norm` | explicit (sandwich norm) |
| `post_attention_layernorm` | `ffn_norm` | explicit (pre-FFN norm, Qwen convention) |
| `post_ffn_norm` | `post_ffw_norm` | explicit (sandwich norm) |
| `moe.router.expert_bias` | `exp_probs_b.bias` | explicit |
| `mlp.experts.{E}.{gate,up,down}_proj` | `ffn_{gate,up,down}_exps` | stacked per layer, expert order |
| everything else | as in the [checkpoint.md](checkpoint.md) table | generic `TensorNameMap` |

- **Explicit names.** The generic map sends `post_attention_layernorm` to
  `ffn_norm` or to `post_attention_norm`, depending on the model family, and
  `post_attn_norm` to `attn_output_norm`
  ([checkpoint.md, pitfall 1](checkpoint.md#pitfalls)). The four block norms
  therefore never go through it.
- **Experts.** The checkpoint has one tensor per expert. They are buffered per
  layer, as for Qwen2MoE. Once all 3 × `num_experts` are in, they are stacked to
  a PyTorch `[n_expert, out, in]` tensor, which is ggml
  `ne = {in, out, n_expert}`. Only names under `mlp.experts.` take this path.
  `mlp.shared_experts` contains "experts" too, but takes the generic path to
  `ffn_*_shexp`. `prepare_tensors` fails if an expert tensor is left over.
- **Dtypes.** No override is needed:
  - the base class keeps 1D tensors (norms, `exp_probs_b`) and the router
    `ffn_gate_inp` in F32;
  - `--outtype bf16` writes all other matrices as BF16, which is lossless for
    the BF16 checkpoint.
- **Loading.** The base class loads tensors lazily. On the real 156 GB
  checkpoint this keeps the conversion at a peak footprint of 7.5 GiB, with
  no streaming code of its own. `check_real.py` confirms every tensor is
  bit-exact: [real-checkpoint.md](real-checkpoint.md).
- **`--fuse-gate-up-exps`** is not supported: it needs `ffn_gate_up_exps` in
  `MODEL_TENSORS[KOLIBRI]`, which is not listed.

## Tiny checkpoint (`cmd/kolibri-tiny`)

The checks need a checkpoint, and the real one is 156 GB. `cmd/kolibri-tiny`
writes a tiny random one with the shape of the reference repo's fixture
(`tests/checkpoints.py` at `049a6a7`):

- 6 layers: 4 sliding, then 2 full;
- hidden 256, 8/2 heads, head_dim 32;
- 8 experts, top 2, expert and shared FFN 256.

It differs from the reference fixture in four points:

- **Vocabulary.** `vocab_size` and the special-token IDs are the real ones:
  128000, no BOS, EOS 127906. The converter reads the real tokenizer, which
  `check_tensors.py` copies in from the pinned, sha256-checked files.
- **BF16 weights,** like the released checkpoint, where the reference fixture
  uses F32.
- **Random norms** instead of `ones`. The four block norms share a shape, so
  only different values reveal a converter that swaps two of them.
- **Scaled weights.** Matrices have standard deviation 0.02, HF's default
  `initializer_range`; norms scatter around 1 (std 0.1); the correction bias
  keeps std 1, like the real one. With the reference's unit-scale `randn`
  matrices, the activations grow over the layers and rounding noise flips the
  top-k routing, which makes a forward-pass check ([model.md](model.md))
  useless. The converter check is bit-exact and does not depend on the scale.

Further variants (`-router`, `-attn`, `-pattern`) are described in
[model.md](model.md) and [attention.md](attention.md).

Names and shapes come from `internal/kolibri`, the Go source of truth behind
the checkpoint inventory. The generator also writes `manifest.json`, which
lists for every expected GGUF tensor:

- its ggml shape;
- its HF sources, in stacking order.

Go tests check:

- the file against `ExpectedNames` and `ExpectedShape`;
- the manifest: 111 tensors, every source exactly once, experts in index order;
- that the block norms differ;
- that two runs with the same seed are byte-identical.

The safetensors writer lives in `internal/safetensors`, with a round-trip test
against the header parser.

## Checks

### Architecture (`tools/gguf/check_arch.py`)

`check_arch.py --llama-cpp third_party/llama.cpp` exits non-zero on any
mismatch. It checks two things:

- **gguf-py.** The tensor names of `MODEL_TENSORS[KOLIBRI]` equal the `gguf`
  names in `inventory/bf16/summary.json`: 21 registered, 21 in the mapping.
  Missing and extra names are both errors. The comparison uses names because
  some kinds are aliases (`FFN_PRE_NORM` and `FFN_NORM` are both
  `blk.N.ffn_norm`).
- **libllama.** The script loads a GGUF that holds only
  `general.architecture = kolibri`, in-process with a log callback. The probe
  must get past the architecture and fail on its missing vocabulary:

  ```
  libllama: 'kolibri' is a known architecture with a model class (the empty probe fails on its vocabulary)
  ```

  Unpatched libllama logs `unknown model architecture: 'kolibri'`; without a
  model class it logs `unsupported model architecture: 'kolibri'`. Both fail
  the check, as does an empty GGUF that loads.

  The probe does not use `llama-tokenize`. Its asynchronous logger lost the
  error line in one of three runs whose output was a pipe.

Without patch 0002, the script fails with `gguf.MODEL_ARCH has no KOLIBRI` and
`libllama does not know the architecture 'kolibri'`.

### Metadata (`tools/gguf/check_metadata.py`)

`check_metadata.py --llama-cpp third_party/llama.cpp` checks the metadata
without a single tensor:

1. It runs `convert_hf_to_gguf.py --vocab-only` on the pinned config and
   tokenizer files. This works because `prepare_metadata(vocab_only=True)` still
   calls `set_gguf_parameters()`.
2. It reads the result with `gguf.GGUFReader`.
3. It compares each key with an independent source, not with converter code.
   Each line is labeled with the property it checks.

The expected values come from:

- `inventory/bf16/config.json` for the config-derived keys;
- the inventory summary for the shared expert (count = `ffn_gate_shexp`
  tensors / layers, FFN size = its `hf_shape[0]`), the sliding window, and the
  pattern (built from `attention.full_attention_layers`, and checked against
  `swa_period`: period 5, full layer last);
- fixed reference values for the gating function and the scale, which have no
  config key; a comment in the checker names the source.

Per-layer arrays print as 0/1 strings. Result (21/21):

```
PASS architecture: general.architecture = 'kolibri'
PASS 50 blocks: kolibri.block_count = 50
PASS embedding size 2,560: kolibri.embedding_length = 2560
PASS 48/4 Q/KV heads: kolibri.attention.head_count = 48
PASS 48/4 Q/KV heads: kolibri.attention.head_count_kv = 4
PASS head dimension / RoPE dimension: kolibri.attention.key_length = 128
PASS head dimension / RoPE dimension: kolibri.attention.value_length = 128
PASS head dimension / RoPE dimension: kolibri.rope.dimension_count = 128
PASS RMSNorm epsilon: kolibri.attention.layer_norm_rms_epsilon = 9.999999974752427e-07
PASS 384 experts / 6 active: kolibri.expert_count = 384
PASS 384 experts / 6 active: kolibri.expert_used_count = 6
PASS expert FFN size 512: kolibri.expert_feed_forward_length = 512
PASS shared expert: kolibri.expert_shared_count = 1
PASS shared expert: kolibri.expert_shared_feed_forward_length = 512
PASS SWA size 512: kolibri.attention.sliding_window = 513
PASS repeating SWA/full pattern (period 5, full last): kolibri.attention.sliding_window_pattern = 11110111101111011110111101111011110111101111011110
PASS RoPE base 10,000 and SWA-only RoPE: kolibri.rope.freq_base = 10000.0
PASS RoPE base 10,000 and SWA-only RoPE: kolibri.attention.rope_pattern = 11110111101111011110111101111011110111101111011110
PASS router gating: kolibri.expert_gating_func = 2
PASS router gating: kolibri.expert_weights_norm = False
PASS router gating: kolibri.expert_weights_scale = 1.0
```

Negative controls:

- Without patch 0003, the converter fails with `Model Kolibri1ForCausalLM is
  not supported`.
- Without patch 0004, exactly the six keys that only `KolibriModel` writes
  fail as missing: expert FFN size, both shared-expert keys, sliding window,
  and the two patterns.
- Without patch 0005, exactly the three router gating lines fail as missing.

Mutation tests (config copies passed via `--model-dir`, or a modified
converter):

| Mutation | Fails exactly |
|---|---|
| 49 layers, `head_dim = 64`, `rms_norm_eps = 1e-5` | block count, key/value length, RoPE dimension, epsilon |
| `sliding_window = 512`, `layer_types` shifted by one layer, `moe_intermediate_size = 256` | expert FFN, sliding window, SWA pattern, RoPE pattern |
| `norm_topk_prob = true` | `expert_weights_norm` |
| converter with `SOFTMAX` and scale 0.5 | gating function, scale |

### Tensors (`tools/gguf/check_tensors.py`)

`check_tensors.py`:

1. generates the tiny checkpoint;
2. converts it with `convert_hf_to_gguf.py --outtype bf16`;
3. compares the GGUF with the manifest:
   - the tensor set;
   - the shape of every tensor;
   - its dtype;
   - its data, bit-exact against the BF16 source and stacked in expert
     order for `*_exps`;
4. checks four metadata keys of the full (not `--vocab-only`) conversion:
   the architecture, block count, expert count and per-layer SWA pattern.

Result:

```
PASS tensor set: 111 tensors, manifest 111
PASS token_embd (1/1): ne [256, 128000], BF16, data bit-exact
PASS output_norm (1/1): ne [256], F32, data bit-exact
PASS output (1/1): ne [256, 128000], BF16, data bit-exact
PASS attn_norm (6/6): ne [256], F32, data bit-exact
PASS attn_q (6/6): ne [256, 256], BF16, data bit-exact
PASS attn_k (6/6): ne [256, 64], BF16, data bit-exact
PASS attn_v (6/6): ne [256, 64], BF16, data bit-exact
PASS attn_q_norm (6/6): ne [32], F32, data bit-exact
PASS attn_k_norm (6/6): ne [32], F32, data bit-exact
PASS attn_output (6/6): ne [256, 256], BF16, data bit-exact
PASS post_attention_norm (6/6): ne [256], F32, data bit-exact
PASS ffn_norm (6/6): ne [256], F32, data bit-exact
PASS ffn_gate_inp (6/6): ne [256, 8], F32, data bit-exact
PASS exp_probs_b (6/6): ne [8], F32, data bit-exact
PASS ffn_gate_exps (6/6): ne [256, 256, 8], BF16, 8 experts stacked, data bit-exact
PASS ffn_up_exps (6/6): ne [256, 256, 8], BF16, 8 experts stacked, data bit-exact
PASS ffn_down_exps (6/6): ne [256, 256, 8], BF16, 8 experts stacked, data bit-exact
PASS ffn_gate_shexp (6/6): ne [256, 256], BF16, data bit-exact
PASS ffn_up_shexp (6/6): ne [256, 256], BF16, data bit-exact
PASS ffn_down_shexp (6/6): ne [256, 256], BF16, data bit-exact
PASS post_ffw_norm (6/6): ne [256], F32, data bit-exact
PASS metadata: general.architecture = 'kolibri'
PASS metadata: kolibri.block_count = 6
PASS metadata: kolibri.expert_count = 8
PASS metadata: kolibri.attention.sliding_window_pattern = [True, True, True, True, False, False]
```

In the reference shape, hidden size, expert FFN and `8 × 32` are all 256, so
many matrices are square. The shape lines therefore cannot tell `gate`
from `down`, or a transposed matrix from the right one. The data comparison
can, because it compares the row-major bytes.

Without patch 0006, the conversion stops at the first tensor with a
`NotImplementedError`. Mutation tests on the converter:

- **Norm swap.** Swapping the `post_attn_norm` and `post_attention_layernorm`
  targets fails exactly the 12 `post_attention_norm` and `ffn_norm` tensors
  (data). Remapping only one of them aborts the conversion with a duplicate
  tensor name.
- **Reversed expert order** fails all 18 `ffn_*_exps` tensors (data).
- **Naive expert match.** Matching `"experts"` anywhere in the name, as
  Qwen2MoE does, also catches `mlp.shared_experts`. The conversion then
  aborts: "Unprocessed experts: ['model.layers.0.mlp.shared_experts…".
