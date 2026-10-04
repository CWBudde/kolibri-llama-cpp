# Phase 3: GGUF architecture registration and converter

This report covers five steps:

1. The first three Phase 3 items register the `kolibri` architecture in gguf-py
   and in libllama, and list its tensors (patch 0002).
2. The converter class writes the first five GGUF metadata items: block count,
   embedding size, head counts, head/RoPE dimension, and RMSNorm epsilon
   (patch 0003). See [Converter and base metadata](#converter-and-base-metadata).
3. The converter writes the MoE and hybrid-attention metadata: experts, expert
   FFN size, shared expert, sliding window, SWA/full pattern, and SWA-only RoPE
   (patch 0004). See [MoE and hybrid-attention metadata](#moe-and-hybrid-attention-metadata).
4. The converter writes the router gating keys: sigmoid gating, no weight
   renormalization, scale 1.0 (patch 0005). See [Router gating metadata](#router-gating-metadata).
5. The converter maps and writes every tensor: explicit block norms and
   correction bias, stacked experts, BF16 output (patch 0006). See
   [Tensor conversion](#tensor-conversion).

With that, every GGUF metadata item of Phase 3 is written, and a conversion
writes every tensor. Only a tiny synthetic checkpoint has been converted so
far; the real 156 GB checkpoint is still to come.

The patches are also kept as commits on the `feat/kolibri` branch of
[CWBudde/llama.cpp](https://github.com/CWBudde/llama.cpp):

- 0001–0002 were committed there directly.
- 0003 came in through `feat/kolibri-converter` (fork PR #2).
- 0004 came in through `feat/kolibri-moe-swa-metadata` (fork PR #3).
- 0005 came in through `feat/kolibri-router-gating` (fork PR #5).
- 0006 comes in through `feat/kolibri-tensor-conversion` (fork PR #6).

Patches 0001–0006 applied to the pinned commit give exactly the tree of
`feat/kolibri`. The one exception is `models/ggml-vocab-kolibri.gguf`, which the
fork commits and the patches do not.

## Changes (`patches/llama.cpp/0002-kolibri-arch.patch`)

The patch applies on top of `0001-kolibri-tokenizer.patch`, at llama.cpp
`1537a0a8`. Each new entry sits next to the matching `laguna` entry.

| File | Change |
|---|---|
| `gguf-py/gguf/constants.py` | `MODEL_ARCH.KOLIBRI`; `MODEL_ARCH_NAMES` → `"kolibri"`; `MODEL_TENSORS[KOLIBRI]` |
| `src/llama-arch.h` | `LLM_ARCH_KOLIBRI` |
| `src/llama-arch.cpp` | `LLM_ARCH_NAMES` → `"kolibri"` |
| `src/llama-model.cpp` | `llama_model_rope_type`: `LLAMA_ROPE_TYPE_NEOX`. Phase 1 found that the sliding layers use NEOX-style RoPE. Without this case, `-Wswitch` warns. |
| `tests/test-llama-archs.cpp` | `arch_supported()` returns false for `kolibri` (FIXME), because there is no `llama_model_kolibri` yet |

## Tensor list

`MODEL_TENSORS[KOLIBRI]` holds exactly the 21 tensor kinds of the Phase 1
`HF → GGUF` mapping:

- embedding, head and output norm: `TOKEN_EMBD`, `OUTPUT_NORM`, `OUTPUT`;
- attention: `ATTN_NORM`, `ATTN_Q`, `ATTN_K`, `ATTN_V`, `ATTN_OUT`,
  `ATTN_Q_NORM`, `ATTN_K_NORM`, `ATTN_POST_NORM`;
- MoE block: `FFN_NORM`, `FFN_GATE_INP`, `FFN_EXP_PROBS_B`,
  `FFN_{GATE,UP,DOWN}_EXP`, `FFN_{GATE,UP,DOWN}_SHEXP`, `FFN_POST_NORM`.

The sandwich norms are `ATTN_POST_NORM` (`post_attention_norm`) and
`FFN_POST_NORM` (`post_ffw_norm`). Kolibri has no dense FFN and no attention
gate.

`tensor_mapping.py` is not changed. Its generic entries map Kolibri's norm
names to the wrong tensors (Phase 1, pitfall 1), so the converter class maps
all four block norms explicitly.

## Why `test-llama-archs` skips `kolibri`

That test builds a random model for every architecture that `llama_model_saver`
supports. `llama_model_create` throws `unsupported model architecture` for an
architecture without a model class. Phase 4 adds `llama_model_kolibri` and
removes the skip.

> Update: patch 0007 did both; see [Phase 4](phase4-model.md).

For the same reason, `ggml-vocab-kolibri.gguf` keeps the `qwen3moe`
placeholder architecture. `llama_model_create` runs even with `vocab_only`,
so a vocab GGUF that declares `kolibri` does not load before Phase 4.

> Update: the vocab GGUF is now written under `kolibri`; see
> [Phase 4: router](phase4-router.md#vocab-gguf-under-kolibri-item-11).

## Check

`tools/gguf/check_arch.py --llama-cpp third_party/llama.cpp` exits non-zero on
any mismatch. It checks two things:

- **gguf-py.** The tensor names of `MODEL_TENSORS[KOLIBRI]` equal the `gguf`
  names in `inventory/bf16/summary.json`. Missing and extra names are both
  errors. The comparison uses names because some kinds are aliases
  (`FFN_PRE_NORM` and `FFN_NORM` are both `blk.N.ffn_norm`).
- **libllama.** The script loads a GGUF that holds only
  `general.architecture = kolibri`, in-process with a log callback.
  - Unpatched libllama logs `unknown model architecture: 'kolibri'`.
  - Patched libllama logs `unsupported model architecture: 'kolibri'`; this is
    the expected result until Phase 4.

  If the empty GGUF ever loads, the check fails and asks to be updated.

  > Update: since patch 0007 the expected result is that the empty probe gets
  > past the architecture and fails on its missing vocabulary; see
  > [Phase 4](phase4-model.md).

  The probe does not use `llama-tokenize`. Its asynchronous logger lost the
  error line in one of three runs whose output was a pipe.

Results on 2026-10-03:

```
gguf-py: 21 tensor names registered, 21 in the Phase 1 mapping
libllama: 'kolibri' is a known architecture without a model class (expected before Phase 4)
```

Before the patch, the script failed with `gguf.MODEL_ARCH has no KOLIBRI` and
`libllama does not know the architecture 'kolibri'`. With the patch, the
llama.cpp test suites still pass:

- `test-llama-archs`: `all 133 test(s) passed`; `kolibri` is listed as SKIP;
- `ctest -R 'test-tokenizer-0|test-generate-models'`: 17/17.

## Converter and base metadata

### Changes (`patches/llama.cpp/0003-kolibri-converter.patch`)

The patch applies on top of 0002. In the fork it is the commit on
`feat/kolibri-converter`.

| File | Change |
|---|---|
| `conversion/kolibri.py` | `KolibriModel`, registered for `Kolibri1ForCausalLM` |
| `conversion/__init__.py` | `TEXT_MODEL_MAP` entry `"Kolibri1ForCausalLM": "kolibri"` |

`KolibriModel` does three things:

- **`set_vocab`:** calls `_set_vocab_gpt2()`. The Phase 2 pre-tokenizer hash
  resolves to `kolibri`.
- **`set_gguf_parameters`:** calls `super()`, then writes
  `rope.dimension_count = head_dim`, because the sliding layers rotate the full
  head (Phase 1).
- **`modify_tensors`:** raises `NotImplementedError`. Without an override, the
  generic `TensorNameMap` would map the sandwich norms to the wrong tensors
  (Phase 1, pitfall 1) and would not stack the per-expert tensors. A full
  conversion therefore fails until the tensor mapping exists. Patch 0006 adds
  it; see [Tensor conversion](#tensor-conversion).

### Metadata written

| PLAN.md item | GGUF key | Value | Written by |
|---|---|---|---|
| 50 blocks | `kolibri.block_count` | 50 | `TextModel` |
| embedding size | `kolibri.embedding_length` | 2560 | `TextModel` |
| 48/4 Q/KV heads | `kolibri.attention.head_count`, `.head_count_kv` | 48, 4 | `TextModel` |
| head / RoPE dimension | `kolibri.attention.key_length`, `.value_length` | 128, 128 | `TextModel` (from `head_dim`) |
| head / RoPE dimension | `kolibri.rope.dimension_count` | 128 | `KolibriModel` |
| RMSNorm epsilon | `kolibri.attention.layer_norm_rms_epsilon` | 1e-6 (as float32) | `TextModel` |

`TextModel` also writes `context_length` (262144), `expert_count` (384),
`expert_used_count` (6) and `rope.freq_base` (10000). The next step checks the
last three of these (see below).

### Check

`tools/gguf/check_metadata.py --llama-cpp third_party/llama.cpp` checks the
metadata without a single tensor:

1. It runs `convert_hf_to_gguf.py --vocab-only` on the pinned config and
   tokenizer files. This works because `prepare_metadata(vocab_only=True)` still
   calls `set_gguf_parameters()`.
2. It reads the result with `gguf.GGUFReader`.
3. It compares each key with `inventory/bf16/config.json`, not with converter
   code. Each line names the PLAN.md item it covers.

Results on 2026-10-03:

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
```

Before the patch, the converter failed with `Model Kolibri1ForCausalLM is not
supported`. As a mutation test, a config copy with 49 layers, `head_dim = 64`
and `rms_norm_eps = 1e-5` (passed via `--model-dir`) fails exactly the
block-count, key/value length, RoPE dimension, and epsilon lines.

## MoE and hybrid-attention metadata

### Changes (`patches/llama.cpp/0004-kolibri-moe-swa-metadata.patch`)

The patch applies on top of 0003. It only extends
`KolibriModel.set_gguf_parameters` in `conversion/kolibri.py`, using existing
gguf-py writers. No new GGUF keys are needed.

Before this patch, the 0003 comments were reflowed to one sentence per line.
That addressed a Codex review finding: llama.cpp `AGENTS.md` asks for no
hard-wrapped comments.

### Metadata written

| PLAN.md item | GGUF key | Value | Written by |
|---|---|---|---|
| 384 experts / 6 active | `kolibri.expert_count`, `.expert_used_count` | 384, 6 | `TextModel` |
| expert FFN size 512 | `kolibri.expert_feed_forward_length` | 512 | `KolibriModel` |
| shared expert | `kolibri.expert_shared_count` | 1 | `KolibriModel` |
| shared expert | `kolibri.expert_shared_feed_forward_length` | 512 | `KolibriModel` |
| SWA size 512 | `kolibri.attention.sliding_window` | 513 | `KolibriModel` |
| repeating SWA/full pattern | `kolibri.attention.sliding_window_pattern` | bool[50], true = SWA | `KolibriModel` |
| RoPE base 10,000 | `kolibri.rope.freq_base` | 10000 | `TextModel` |
| SWA-only RoPE | `kolibri.attention.rope_pattern` | bool[50], true = RoPE | `KolibriModel` |

- **Shared expert.** The checkpoint has one ungated shared expert per layer,
  with `gate_proj` shape [512, 2560]. Qwen2/3-MoE write only the shared FFN
  length. Kolibri also writes `expert_shared_count = 1`, so the count does not
  depend on a model-class default.
- **Sliding window 513.** `config.json` stores `sliding_window = 513`, and vLLM
  passes it to FlashAttention as `window = (512, 0)`: the 512 preceding tokens
  plus the current one. llama.cpp `LLAMA_SWA_TYPE_STANDARD` masks a key when
  `p1 - p0 >= n_swa` (`src/llama-hparams.h`). With `n_swa = 513` that is the
  same window, so the value is stored unchanged. Phase 5 tests the 511, 512 and
  513 boundaries numerically.
- **Patterns.** Both arrays come from `layer_types`:
  - `sliding_window_pattern` is read by a required `ml.get_arr` into
    `hparams.is_swa_impl` in `llama_model_kolibri::load_arch_hparams`, not
    by `load_swa_pattern`; see the evaluation in
    [phase5-attention.md](phase5-attention.md).
  - `rope_pattern` is read into `hparams.rope_pattern` and used by
    `llama_hparams::has_rope`. It equals the SWA pattern, because the
    full-attention layers 4, 9, …, 49 use no positional encoding (RNoPE,
    Phase 1).

  The Phase 4/5 loader decides how `llama_model_kolibri` reads these keys.

### Check

`check_metadata.py` takes the expected values for these keys from the Phase 1
inventory summary, not from `config.json` alone:

- **shared expert:** count = `ffn_gate_shexp` tensors / layers, and FFN size =
  its `hf_shape[0]`;
- **sliding window:** `attention.sliding_window`;
- **pattern:** built from `attention.full_attention_layers`, and checked against
  `swa_period` (period 5, full layer last).

The per-layer arrays print as 0/1 strings. Results on 2026-10-03:

```
PASS 384 experts / 6 active: kolibri.expert_count = 384
PASS 384 experts / 6 active: kolibri.expert_used_count = 6
PASS expert FFN size 512: kolibri.expert_feed_forward_length = 512
PASS shared expert: kolibri.expert_shared_count = 1
PASS shared expert: kolibri.expert_shared_feed_forward_length = 512
PASS SWA size 512: kolibri.attention.sliding_window = 513
PASS repeating SWA/full pattern (period 5, full last): kolibri.attention.sliding_window_pattern = 11110111101111011110111101111011110111101111011110
PASS RoPE base 10,000 and SWA-only RoPE: kolibri.rope.freq_base = 10000.0
PASS RoPE base 10,000 and SWA-only RoPE: kolibri.attention.rope_pattern = 11110111101111011110111101111011110111101111011110
```

The batch 1 lines still pass. Before the patch, the six keys that only
`KolibriModel` writes (expert FFN size, both shared-expert keys, sliding window,
and the two patterns) were missing, and exactly those six lines failed.

As a mutation test, a config copy with `sliding_window = 512`, `layer_types`
shifted by one layer, and `moe_intermediate_size = 256` fails exactly the
expert FFN, sliding window, SWA pattern, and RoPE pattern lines.

## Router gating metadata

### Changes (`patches/llama.cpp/0005-kolibri-router-gating.patch`)

The patch applies on top of 0004. It adds three writer calls to
`KolibriModel.set_gguf_parameters`, again with existing gguf-py writers.

### Metadata written

| GGUF key | Value | Source |
|---|---|---|
| `kolibri.expert_gating_func` | 2 (`LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`) | Phase 1 router semantics |
| `kolibri.expert_weights_norm` | false | `config.json` `norm_topk_prob` |
| `kolibri.expert_weights_scale` | 1.0 | vLLM `routed_scaling_factor = 1.0` |

- **Gating function.** Kolibri weights each selected expert with the sigmoid
  of its raw router logit (Phase 1, from `sigmoid_logit_add_routing`).
  `TextModel` writes `expert_gating_func` only when the config names a scoring
  function, and Kolibri's `config.json` names none. So `KolibriModel` writes it,
  as the GLM, Laguna and Hunyuan converters do.
- **No renormalization, no scaling.** The six selected weights are used as
  they are. `expert_weights_scale = 1.0` is written explicitly, so the value
  does not depend on a loader default.
- **What this does not cover.** These keys do not describe the correction
  bias. Kolibri adds `expert_bias` to the raw logits for the selection only,
  while `build_moe_ffn` adds `exp_probs_b` to the sigmoid output. Phase 4
  needs an arch-specific selection branch for that (Phase 1, pitfall 2).
  Patch 0007 adds it; see [Phase 4](phase4-model.md).

### Check

`check_metadata.py` takes `expert_weights_norm` from `config.json`. The
gating function and the scale have no config key, so the checker holds them
to the Phase 1 reference values, with a comment naming the source. Results on
2026-10-04, after the 18 earlier lines:

```
PASS router gating: kolibri.expert_gating_func = 2
PASS router gating: kolibri.expert_weights_norm = False
PASS router gating: kolibri.expert_weights_scale = 1.0
```

Before the patch, exactly these three lines failed (key missing), and the
other 18 passed. As mutation tests:

- a config copy with `norm_topk_prob = true` fails only the
  `expert_weights_norm` line;
- a converter with `SOFTMAX` and scale 0.5 fails exactly the gating function
  and scale lines.

## Tensor conversion

### Changes (`patches/llama.cpp/0006-kolibri-tensor-conversion.patch`)

The patch applies on top of 0005. It replaces the `NotImplementedError` in
`KolibriModel.modify_tensors` with the tensor mapping:

| HF tensor (per layer) | GGUF tensor | How |
|---|---|---|
| `input_layernorm` | `attn_norm` | explicit |
| `post_attn_norm` | `post_attention_norm` | explicit (sandwich norm) |
| `post_attention_layernorm` | `ffn_norm` | explicit (pre-FFN norm, Qwen convention) |
| `post_ffn_norm` | `post_ffw_norm` | explicit (sandwich norm) |
| `moe.router.expert_bias` | `exp_probs_b.bias` | explicit |
| `mlp.experts.{E}.{gate,up,down}_proj` | `ffn_{gate,up,down}_exps` | stacked per layer, expert order |
| everything else | as in the Phase 1 table | generic `TensorNameMap` |

- **Explicit names.** The generic map sends `post_attention_layernorm` to
  `ffn_norm` or to `post_attention_norm`, depending on the model family, and
  `post_attn_norm` to `attn_output_norm` (Phase 1, pitfall 1). The four block
  norms therefore never go through it.
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
- **Not covered.**
  - Streaming: the base class loads tensors lazily, but no measurement on the
    real checkpoint exists yet.
  - FP8: a follow-up.
  - `--fuse-gate-up-exps`: needs `ffn_gate_up_exps` in
    `MODEL_TENSORS[KOLIBRI]`, which it is not.

### Tiny checkpoint (`cmd/kolibri-tiny`)

The check needs a checkpoint, and the real one is 156 GB. `cmd/kolibri-tiny`
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
- **Scaled weights** (added in Phase 4). Matrices have standard deviation
  0.02, HF's default `initializer_range`; norms scatter around 1 (std 0.1);
  the correction bias keeps std 1, like the real one. With the reference's
  unit-scale `randn` matrices, the activations grow over the layers and
  rounding noise flips the top-k routing, which makes a forward-pass check
  useless. The converter check is bit-exact and does not depend on the scale.

Names and shapes come from `internal/kolibri`, the Go source of truth behind
the Phase 1 inventory. The generator also writes `manifest.json`, which lists
for every expected GGUF tensor:
- its ggml shape;
- its HF sources, in stacking order.

Go tests check:
- the file against `ExpectedNames` and `ExpectedShape`;
- the manifest: 111 tensors, every source exactly once, experts in index order;
- that the block norms differ;
- that two runs with the same seed are byte-identical.

The safetensors writer is new in `internal/safetensors`, with a round-trip test
against the existing header parser.

### Check

`tools/gguf/check_tensors.py`:

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

Results on 2026-10-04:

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

Before the patch, the conversion stopped at the first tensor with the
`NotImplementedError`. As mutation tests on the converter:

- **Norm swap.** Swapping the `post_attn_norm` and `post_attention_layernorm`
  targets fails exactly the 12 `post_attention_norm` and `ffn_norm` tensors
  (data). Remapping only one of them aborts the conversion with a duplicate
  tensor name.
- **Reversed expert order** fails all 18 `ffn_*_exps` tensors (data).
- **Naive expert match.** Matching `"experts"` anywhere in the name, as
  Qwen2MoE does, also catches `mlp.shared_experts`. The conversion then
  aborts: "Unprocessed experts: ['model.layers.0.mlp.shared_experts…".
