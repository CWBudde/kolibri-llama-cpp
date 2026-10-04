# Kolibri model class and MoE graph

`llama_model_kolibri` loads a converted Kolibri GGUF and builds its forward
graph. Two checks cover it:

- `tools/gguf/check_model.py` loads and decodes the tiny synthetic checkpoint
  on CPU and Metal and checks the graph wiring;
- `tools/gguf/check_moe.py` compares every step of the MoE block with the
  reference implementation, layer by layer, on a fixture with the reference
  router test's shape (384 experts, top 6).

The attention half of the graph (window, RoPE, KV cache, GQA, QK norm) is
checked against the reference in [attention.md](attention.md).

On the real weights, the complete 50-layer graph runs unquantized on the CPU,
and a Q3_K/Q8_0 quantization runs fully on Metal. See
[real-checkpoint.md](real-checkpoint.md). Those runs are structural only:
without the vLLM reference they say nothing about numerical agreement.

## Changes (`patches/llama.cpp/0007-kolibri-model.patch`)

| File | Change |
|---|---|
| `src/models/kolibri.cpp` (new) | `load_arch_hparams`, `load_arch_tensors`, graph |
| `src/models/models.h`, `src/llama-model.cpp` | class declaration, factory case |
| `src/llama-graph.cpp` | Kolibri selection branch in `build_moe_ffn`; F32 router logits |
| `tests/test-llama-archs.cpp` | `kolibri` is MoE-only; SWA and RoPE pattern arrays in the fixture |

The template is `laguna.cpp`: sigmoid MoE with `exp_probs_b`, one shared
expert, QK norm, hybrid SWA. Kolibri drops Laguna's attention gate, dense
lead layers and per-layer-type YaRN, and adds the sandwich norms. The SWA
RoPE fields are initialized from the model's RoPE values, and the graph's
RoPE parameters come from `get_rope_freq_base/scale`, so the K-shift uses the
same per-layer frequencies as the graph.

The patch is `git diff 6b81ee010 2e3588bf4` in
[CWBudde/llama.cpp](https://github.com/CWBudde/llama.cpp). Patches 0001–0007
applied to the pinned commit give exactly `feat/kolibri` at `2e3588bf4`. The
one exception is `models/ggml-vocab-kolibri.gguf`, which the fork commits and
the patches do not.

## Hyperparameters

- **Required keys:**
  - RMS epsilon;
  - expert FFN length and shared-expert FFN length;
  - gating function (`expert_gating_func`, 2 = sigmoid);
  - `sliding_window`;
  - the two per-layer arrays `attention.sliding_window_pattern` and
    `attention.rope_pattern`.

  A GGUF without them fails to load.
- **Optional keys:**
  - `expert_weights_norm` and `expert_weights_scale`. The defaults (no
    renormalization, scale 1.0) are Kolibri's values. The converter writes
    both anyway, because libllama does read them: a GGUF with a wrong value
    would compute wrongly without any error (see [Mutation tests](#mutation-tests)).
  - the shared-expert count, default 1.
- `expert_used_count` sets the number of selected experts. The converter takes
  it from `num_experts_per_tok` (6 for the real checkpoint; checked by
  `check_metadata.py`, see [gguf-conversion.md](gguf-conversion.md)). Nothing
  else in the graph hard-codes a count.
- `swa_type = LLAMA_SWA_TYPE_STANDARD`, with `n_swa = 513` unchanged (see
  [checkpoint.md](checkpoint.md)). The graph uses the iSWA KV cache.
- `LLM_TYPE_UNKNOWN`: no existing type fits 50 layers / 78B, and adding one
  needs entries in the type name table.

## Graph

```
x  = embed(tokens)
per layer il:
  h   = RMSNorm_attn_norm(x)
  q,k,v = h Wq, h Wk, h Wv           # 48/4 heads, head_dim 128 (GQA)
  q,k = RMSNorm_q(q), RMSNorm_k(k)   # per head, before RoPE
  if rope_pattern[il]: q,k = RoPE_neox(q,k, base 10000)   # SWA layers only
  a   = Attn_iswa(q,k,v) Wo          # SWA mask on sliding layers, scale 1/sqrt(128)
  x   = x + RMSNorm_post_attention_norm(a)
  h   = RMSNorm_ffn_norm(x)
  m   = MoE_routed(h) + FFN_shexp(h) # shexp ungated, same input
  x   = x + RMSNorm_post_ffw_norm(m)
logits = RMSNorm_output_norm(x) W_output        # F32 accumulation
```

This is the block structure from the checkpoint inventory
([checkpoint.md](checkpoint.md), "Block structure and residual order"). Each
step has a `cb()` name: `attn_norm`, `Qcur_normed`, `Qcur_rope`, `attn_out`,
`attn_post_norm`, `ffn_inp`, `ffn_norm`, `ffn_moe_out`, `ffn_shexp`,
`ffn_out`, `ffn_post_norm`, `l_out`. `check_moe.py` uses these names to
localize every MoE step per layer, and `check_attn.py` does the same for the
attention steps.

**Routed + shared order.** The graph computes `ffn_out = ffn_moe_out +
ffn_shexp`; the reference computes `shared_output + fused_output`. IEEE
addition is commutative, so the operand order does not change a single bit.
What matters is the order of the operations: the sum comes before
`post_ffw_norm`, and the norm before the residual add. `check_moe.py`
confirms both numerically in every layer.

## Router

### The reference

[`Aleph-Alpha/aleph-alpha-inference@049a6a7`](https://github.com/Aleph-Alpha/aleph-alpha-inference/tree/049a6a7bd2405b27d6d280d256bd3d585191c7ae),
the pinned reference:

- **`aleph_alpha_inference/kolibri1.py`, `sigmoid_logit_add_routing`:** the
  router. `check_moe.py` carries a verbatim copy, without its `torch.compile`
  decorator:

  ```python
  logits = gating_output.float()
  topk_ids = torch.topk(logits + e_score_correction_bias, k=topk, dim=-1)[1]
  topk_weights = torch.sigmoid(logits.gather(1, topk_ids))
  if renormalize:
      topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
  ```

  `renormalize` is `config.norm_topk_prob`, which is `false`.
- **`Kolibri1SparseMoeBlock`:** FP32 router logits and an ungated shared expert.
  vLLM's MoE runner returns `shared_output + fused_output`.
- **`Kolibri1DecoderLayer.forward`:**
  `post_ffn_norm(mlp(post_attention_layernorm(h)))`, then the residual add.
- **`tests/test_kolibri1.py`, `test_routing_semantics`:**
  - 384 experts, top 6;
  - logits drawn with std 3, the bias with std 5;
  - an assert that the test is not vacuous: selecting on
    `sigmoid(logits) + bias` must pick different experts.

### The Kolibri branch in `build_moe_ffn`

`build_moe_ffn` adds `exp_probs_b` to the probabilities `sigmoid(logits)`,
following DeepSeek-V3. Kolibri adds it to the raw logits (see the router
pitfall in [checkpoint.md](checkpoint.md#pitfalls)). The patch adds a branch
next to the existing LLAMA4 and GROVEMOE branches:

```cpp
// kolibri adds the selection bias to the raw logits, not to the sigmoid probs
// the expert weights stay sigmoid(logits) without the bias
if (arch == LLM_ARCH_KOLIBRI && exp_probs_b != nullptr) {
    selection_probs = ggml_add(ctx0, logits, exp_probs_b);
}
```

The weights are still taken from `probs = sigmoid(logits)`. With
`expert_weights_norm = false` and `expert_weights_scale = 1.0`, they are used
as they are. The router logits accumulate in F32, as vLLM computes them.

The two bias placements select different experts at 592 of 600 token-layers
on the `-router` fixture. A router that put the bias on `sigmoid(logits)`, as
llama.cpp's default branch does, would therefore fail almost everywhere.

### Router sensitivity on the real weights (open)

On the real checkpoint, Q8_0 against this port's BF16 GGUF reaches a mean KL
divergence of 0.050, with the same top token at 92.5% of positions. Both ran on
the CPU, over wikitext-2, 20 × 512 tokens. For dense models, Q8_0 usually lands
near 0.001 and over 99%.

The suspect is the top-6 selection. A small change to a router logit near the
selection boundary swaps a whole expert, and the change then carries through
the remaining layers.

This is not measured yet. The PLAN Phase 7 Top-6 agreement item and the
Phase 6 Metal near-tie item cover it, after the BF16 port matches vLLM.
Numbers are in [real-checkpoint.md](real-checkpoint.md).

## Fixtures (`cmd/kolibri-tiny`)

Both fixtures are tiny synthetic checkpoints from `cmd/kolibri-tiny` (see
[gguf-conversion.md](gguf-conversion.md)): 6 layers SSSSFF, hidden 256, 8/2
heads, BF16.

| | default | `-router` |
|---|---|---|
| experts, top-k | 8, 2 | 384, 6 (the reference router test's shape) |
| `moe_intermediate_size` | 256 | 16 (keeps 384 experts small) |
| router weight std | 0.02 | 3/16, so logits have std ≈ 3 on the unit-RMS `ffn_norm` output |
| `expert_bias` std | 1 | 5 |
| sliding window | 65 | 65 |

The default is used by `check_tensors.py` and `check_model.py`, `-router` by
`check_moe.py`.

Matrices have std 0.02 and norms 1 ± 0.1. The reference fixture draws its
matrices with unit standard deviation (`torch.randn`); the activations then
grow over the layers, and an error spreads from one position to all later
ones (see [below](#why-the-strict-comparison-takes-the-routing-out)).

## Check: loading and wiring (`tools/gguf/check_model.py`)

The check converts the default checkpoint to BF16 and to F32. Then it decodes
the same 100 tokens through the libllama C API, with the model and all
computation pinned to one device. 100 tokens are more than the window, so the
sliding-window mask cuts in.

1. **Load and decode.** Both GGUFs load on CPU and Metal. libllama fails on any
   missing, wrongly shaped or unused tensor. The logits of all positions are
   finite.
2. **No routing**, strict. The F32 GGUF with `expert_used_count` overridden to
   all 8 experts:
   - CPU in ubatches of 16 vs one ubatch of 100, so the KV cache carries the
     context across ubatches;
   - every device vs the CPU.

   Both must stay at NMSE ≤ 1e-4, the `test-llama-archs` threshold.
3. **Top-2 routing**, as converted. The same comparisons must:
   - pick the same greedy token at every position;
   - keep NMSE ≤ 1e-4 at all but at most 2 positions.
4. **Graph wiring.** libllama's `cb_eval` callback records every graph node
   with the names of its inputs, from one CPU decode with top-2 routing. In
   every layer:
   - `ffn_moe_probs_biased` = `ffn_moe_logits` + `exp_probs_b`, so the bias
     sits on the raw logits;
   - `ffn_moe_weights` comes from the unbiased `ffn_moe_probs`;
   - `ffn_moe_down` uses the stacked `ffn_down_exps`;
   - `ffn_out` = `ffn_moe_out` + `ffn_shexp`;
   - `post_attention_norm` feeds `ffn_inp`, and `post_ffw_norm` feeds `l_out`;
   - `Qcur_rope`/`Kcur_rope` exist exactly on the sliding layers.

   Unnamed intermediates are followed back to their source, for example the
   last layer's `get_rows` of the output rows. This checks the structure;
   the numerics against the reference are `check_moe.py`'s job.

Result on 2026-10-04 (M5 Pro):

```
PASS bf16 load and decode on CPU: logits (100, 128000), all finite
PASS bf16 load and decode on MTL0: logits (100, 128000), all finite
PASS f32 load and decode on CPU: logits (100, 128000), all finite
PASS f32 load and decode on MTL0: logits (100, 128000), all finite
PASS f32 no routing (top-8 of 8), CPU ubatch 16 vs 100: NMSE 2.67e-06
PASS f32 no routing (top-8 of 8), MTL0 vs CPU: NMSE 1.72e-06
PASS f32 top-2 routing, CPU ubatch 16 vs 100: 100/100 positions with NMSE <= 0.0001, greedy token equal at 100/100
PASS f32 top-2 routing, MTL0 vs CPU: 99/100 positions with NMSE <= 0.0001, greedy token equal at 100/100
PASS graph wiring: 6 layers x 9 node checks; RoPE on the 4 sliding layers only (272 nodes)
```

### Why the strict comparison takes the routing out

With unit-std matrices, CPU vs Metal on the BF16 GGUF gave NMSE 8.4e-3. The
cause, step by step:

1. **The fixture's scale.** Unit-std matrices let the activations grow over
   the layers, and the error spread from one position to all later ones. With
   std 0.02 for matrices and 1 ± 0.1 for norms, NMSE was 3.7e-4, and exactly
   one position was off.
2. **BF16 rounding.** For BF16 matmuls the CPU rounds the activations to BF16,
   Metal and the BLAS path do not. With the F32 GGUF the error stays on one
   position (35); the median per-position NMSE is 1.7e-6.
3. **A top-k near-tie.** With `expert_used_count` overridden to 8, there is no
   selection. That one position then drops to 2.9e-6 as well, which confirms
   that two experts score almost the same there and rounding picks the other
   one. `test-llama-archs` never sees this, because its fixture activates
   both of its 2 experts.

## Check: MoE block against the reference (`tools/gguf/check_moe.py`)

1. **Convert to F32.** The `-router` checkpoint is converted to F32;
   BF16 → F32 is exact, so the GGUF weights equal the safetensors values.
2. **Decode and capture.** It decodes 100 tokens on every device. libllama's
   `cb_eval` captures ten named nodes per layer:
   - `ffn_inp`, `ffn_norm`;
   - `ffn_moe_logits`, `ffn_moe_topk`, `ffn_moe_weights`;
   - `ffn_moe_out`, `ffn_shexp`, `ffn_out`, `ffn_post_norm`, `l_out`.

   A view is read from its root tensor with its own strides; `ffn_moe_topk` is
   a view of the argsort.
3. **Recompute in float64.** Each step is recomputed in float64 from
   libllama's own input to that step:

| Check | Reference |
|---|---|
| Top-6 | `ffn_moe_topk` is `[n_tokens, 6]`; the 6 comes from the GGUF's `expert_used_count` |
| selection | `sigmoid_logit_add_routing` on libllama's own `ffn_moe_logits`: the selected sets must match exactly |
| non-vacuity | `top6(sigmoid(logits) + bias)` must differ from the selection at ≥ 1 token in every layer |
| weights | `ffn_moe_weights` = the reference's `topk_weights`, compared per expert id |
| weights applied | the experts are multiplied with the `ffn_moe_weights` node itself, not `ffn_moe_weights_norm` or `_scaled` |
| `ffn_norm` | `RMSNorm(ffn_inp) · w` |
| `ffn_moe_logits` | `ffn_norm @ W_gate^T` |
| `ffn_moe_out` | `Σ_k w_k · down_k(silu(gate_k h) · up_k h)` over the reference selection |
| `ffn_shexp` | `down(silu(gate h) · up h)` |
| `ffn_out` | `shared + routed` |
| `ffn_post_norm` | `RMSNorm(ffn_out) · w` |
| `l_out` | `ffn_post_norm + ffn_inp` |

**Why the selection uses libllama's logits.** A logit recomputed in another
precision can flip a near-tie between two experts. Selecting on the same
logits cannot flip anything, so the selection check is exact. The logits
themselves are compared separately.

**Thresholds:**
- **CPU:** NMSE ≤ 1e-10 against float64. The observed maximum is 3.3e-14.
- **Other devices:** NMSE ≤ 1e-4, the backend tolerance of `test-llama-archs`.
  - The strict bound fails on Metal: `ffn_moe_logits` 1.8e-7, `ffn_shexp`
    1.3e-6.
  - The cause is the backend, not Kolibri:
    `ggml/src/ggml-metal/kernels/mul_mm.metal:851` instantiates
    `kernel_mul_mm_f32_f32` with `half` / `simdgroup_half8x8` tiles. Metal
    multiplies F32 matrices with half-precision inputs. On the fixture the
    router logits have a relative RMS error of about 4e-4; for the real model,
    a near-tie in the top 6 could resolve differently than in vLLM.
- **Selection and weights:** exact on every device.

Result on 2026-10-04 (M5 Pro):

```
PASS CPU Top-6: ffn_moe_topk is [(100, 6)] (n_tokens, n_expert_used)
PASS CPU selection = top6(logits + bias) at 600/600 token-layers
PASS CPU non-vacuous: top6(sigmoid(logits) + bias) selects differently at 592/600 token-layers, at least 97 per layer
PASS CPU weights = sigmoid(logits[selected]): max abs error 1.2e-07
PASS CPU experts weighted with ffn_moe_weights itself (no norm, no scale)
PASS CPU ffn_norm vs reference: max NMSE over 6 layers 3.0e-15 (<= 1e-10)
PASS CPU ffn_moe_logits vs reference: max NMSE over 6 layers 8.9e-15 (<= 1e-10)
PASS CPU ffn_moe_out vs reference: max NMSE over 6 layers 2.8e-14 (<= 1e-10)
PASS CPU ffn_shexp vs reference: max NMSE over 6 layers 2.9e-14 (<= 1e-10)
PASS CPU ffn_out vs reference: max NMSE over 6 layers 3.0e-14 (<= 1e-10)
PASS CPU ffn_post_norm vs reference: max NMSE over 6 layers 3.3e-14 (<= 1e-10)
PASS CPU l_out vs reference: max NMSE over 6 layers 1.7e-14 (<= 1e-10)
PASS MTL0 Top-6: ffn_moe_topk is [(100, 6)] (n_tokens, n_expert_used)
PASS MTL0 selection = top6(logits + bias) at 600/600 token-layers
PASS MTL0 non-vacuous: top6(sigmoid(logits) + bias) selects differently at 592/600 token-layers, at least 97 per layer
PASS MTL0 weights = sigmoid(logits[selected]): max abs error 1.2e-07
PASS MTL0 experts weighted with ffn_moe_weights itself (no norm, no scale)
PASS MTL0 ffn_norm vs reference: max NMSE over 6 layers 3.0e-15 (<= 0.0001)
PASS MTL0 ffn_moe_logits vs reference: max NMSE over 6 layers 1.8e-07 (<= 0.0001)
PASS MTL0 ffn_moe_out vs reference: max NMSE over 6 layers 9.9e-08 (<= 0.0001)
PASS MTL0 ffn_shexp vs reference: max NMSE over 6 layers 1.3e-06 (<= 0.0001)
PASS MTL0 ffn_out vs reference: max NMSE over 6 layers 1.1e-06 (<= 0.0001)
PASS MTL0 ffn_post_norm vs reference: max NMSE over 6 layers 3.4e-07 (<= 0.0001)
PASS MTL0 l_out vs reference: max NMSE over 6 layers 1.7e-07 (<= 0.0001)
```

## `test-llama-archs`

`test-llama-archs -a '^kolibri$'` runs its own random fixture: 2 layers, one
full and one sliding, and 2 experts with both active.

```
|         kolibri|Apple M5 Pro|   MoE|  OK (4.71e-07)|       OK|
|         kolibri|  Accelerate|   MoE|  OK (3.01e-13)|       OK|
|         kolibri|Apple M5 Pro|   MoE|  OK (2.45e-13)|       OK|
|         kolibri|        Meta|   MoE|  OK (4.71e-07)|     SKIP|
```

It builds its tensors from the class's own list, so only a real GGUF reveals a
missing load. Its consistency comparison cannot see a wrongly wired graph,
because CPU and Metal run the same wrong graph. Without the `moe_mandatory`
entry for `kolibri`, its dense fixture lacks
`kolibri.expert_feed_forward_length` and the row fails.

## Mutation tests

Each mutation was run, then reverted. The kv overrides go through the C API;
the code changes are rebuilt libllama variants.

| Mutation | Fails with |
|---|---|
| `post_ffw_norm` not loaded | `check_model.py`: `wrong number of tensors; expected 111, got 105` |
| shared expert not loaded or used | `check_model.py`: `expected 111, got 93` |
| shared expert loaded, not added | `check_model.py` wiring: `ffn_out-0 <- ['ffn_moe_weighted-0 (view)', …]`; `check_moe.py`: `nodes not captured: ['ffn_moe_out-0', 'ffn_shexp-0', …]` (the unused shared expert is pruned from the graph, and the routed output takes the name `ffn_out`) |
| selection branch removed (bias on `sigmoid`) | `check_model.py` wiring: `ffn_moe_probs_biased-0 <- ['ffn_moe_probs-0', …]`; `check_moe.py`: `selection = top6(logits + bias) at 8/600 token-layers`, and weights, `ffn_moe_out`, `ffn_out`, `ffn_post_norm`, `l_out` |
| RoPE on every layer | `check_model.py` wiring: `layer 4 (full_attention): RoPE nodes present` |
| `kolibri` removed from `moe_mandatory` | `test-llama-archs`: the dense fixture lacks `kolibri.expert_feed_forward_length` |
| kv override `expert_weights_norm = true` | `check_moe.py`: experts weighted with `ffn_moe_weights_norm-0 (reshaped)`; `ffn_moe_out` NMSE 6.7e-1 |
| kv override `expert_weights_scale = 2.5` | `check_moe.py`: experts weighted with `ffn_moe_weights_scaled-0`; `ffn_moe_out` NMSE 2.3 |
| kv override `expert_gating_func = 1` (softmax) | `check_moe.py`: weights max abs error 1.0; `ffn_moe_out` NMSE 9.6e-1 |
| kv override `expert_used_count = 5` | `check_moe.py`: `Top-6: ffn_moe_topk is [(100, 5)]` |

`test-llama-archs` passes every mutation except the `moe_mandatory` one.

## Vocab GGUF under `kolibri`

`models/ggml-vocab-kolibri.gguf` is the vocab-only GGUF that
`test-tokenizer-0-kolibri` loads (see [tokenizer.md](tokenizer.md)).
It is written under the `kolibri` architecture, which needs the model class:
`llama_model_create` rejects an architecture without one even with
`vocab_only`.

- **Source:** `tools/tokenizer/vocab_gguf.py` writes the file with the
  converter's own Kolibri class (`get_model_class("Kolibri1ForCausalLM")`,
  then `write_vocab()`), which is `convert_hf_to_gguf.py --vocab-only` plus a
  fixed `general.name`. Run on the HF cache, the CLI would record the
  revision SHA as `general.name` and `general.finetune`.
- **Contents:**
  - `general.architecture = kolibri`;
  - 21 `kolibri.*` hyperparameter keys;
  - `general.file_type = 0`;
  - tokens, merges, token types, special IDs and the chat template.
- **Checks:**
  - `test-tokenizer-0-kolibri` passes with the existing `.inp`/`.out`;
  - `compare.py` matches on all fuzz sets;
  - `check_arch.py` passes;
  - regenerating gives a byte-identical file.
- **Fork:** commit `1331f3f1b`. The binary is not part of the patch series.
