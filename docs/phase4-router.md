# Phase 4: router and MoE block against the reference

This report covers the router half of Phase 4:
- Top-6 selection;
- sigmoid gating and its flags;
- the order of routed and shared output;
- the vocab GGUF under the `kolibri` architecture.

`tools/gguf/check_moe.py` compares every step of the MoE block in libllama
with the reference implementation, layer by layer. It runs on a tiny
checkpoint with the reference router test's shape: 384 experts, top 6.

The model class itself is described in [phase4-model.md](phase4-model.md).
This batch needed no change to it.

## The reference

[`Aleph-Alpha/aleph-alpha-inference@049a6a7`](https://github.com/Aleph-Alpha/aleph-alpha-inference/tree/049a6a7bd2405b27d6d280d256bd3d585191c7ae),
the pin from Phase 1:

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

## Fixture: `kolibri-tiny -router`

The default tiny checkpoint has 8 experts and top 2. That is too few to test a
selection out of 384. `-router` writes the reference router test's shape and
magnitudes instead:

| | default | `-router` |
|---|---|---|
| experts, top-k | 8, 2 | 384, 6 |
| `moe_intermediate_size` | 256 | 16 (keeps 384 experts small) |
| router weight std | 0.02 | 3/16, so logits have std ≈ 3 on the unit-RMS `ffn_norm` output |
| `expert_bias` std | 1 | 5 |

Everything else is the same: 6 layers SSSSFF, hidden 256, BF16. The default
output is unchanged, so `check_tensors.py` and `check_model.py` see the same
checkpoint as before.

## Check (`tools/gguf/check_moe.py`)

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
  - The first run failed on Metal with the strict bound: `ffn_moe_logits`
    1.8e-7, `ffn_shexp` 1.3e-6.
  - The cause is the backend, not Kolibri:
    `ggml/src/ggml-metal/kernels/mul_mm.metal:851` instantiates
    `kernel_mul_mm_f32_f32` with `half` / `simdgroup_half8x8` tiles. Metal
    multiplies F32 matrices with half-precision inputs.
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

The two bias placements select different experts at 592 of 600 token-layers.
A router that put the bias on `sigmoid(logits)`, as llama.cpp's default
branch does, would therefore fail almost everywhere. Mutation M5 below shows
exactly that.

## Answers to the plan items

**Top-6.**
- `build_moe_ffn` selects `n_expert_used` experts. The value is the GGUF's
  `kolibri.expert_used_count`, which the converter takes from
  `num_experts_per_tok` (6 for the real checkpoint; `check_metadata.py`).
- Nothing else in the graph hard-codes a count.
- Overriding the key to 5 makes the Top-6 check fail.

**Sigmoid gating and its flags.**
- `expert_gating_func` is a **required** key in `load_arch_hparams`, written
  as 2 (sigmoid). Overriding it to softmax (1) fails the weights check.
- `expert_weights_norm` and `expert_weights_scale` are **optional**. Their
  loader defaults (no renormalization, scale 1.0) are Kolibri's values, and
  the converter writes them anyway.
- libllama does read both flags. With `norm = true` or `scale = 2.5`, the
  experts are multiplied with `ffn_moe_weights_norm` or
  `ffn_moe_weights_scaled`, and the outputs leave the reference. A GGUF with
  a wrong value would therefore compute wrongly without any error, which is
  why the converter writes them explicitly.

**Routed + shared order.**
- The graph computes `ffn_out = ffn_moe_out + ffn_shexp`. The reference
  computes `shared_output + fused_output`.
- IEEE addition is commutative, so the operand order does not change a
  single bit.
- What matters is the order of the operations: the sum comes before
  `post_ffn_norm`, and the norm before the residual add. The check confirms
  both numerically in every layer.

**Graph callbacks.** The `cb()` names localize every step of the MoE half
per layer; the check above uses them. The attention internals (`Qcur_normed`,
`Qcur_rope`, `attn_out`) have names too, but nothing compares them with a
reference yet. That needs the Phase 5/6 reference.

## Mutation tests

Each mutation was run, then reverted. M1–M4 are kv overrides through the C
API; M5 and M6 are rebuilt libllama variants.

| Mutation | `check_moe.py` (CPU) |
|---|---|
| M1 `expert_weights_norm = true` | FAIL: experts weighted with `ffn_moe_weights_norm-0 (reshaped)`; `ffn_moe_out` NMSE 6.7e-1 |
| M2 `expert_weights_scale = 2.5` | FAIL: experts weighted with `ffn_moe_weights_scaled-0`; `ffn_moe_out` NMSE 2.3 |
| M3 `expert_gating_func = 1` (softmax) | FAIL: weights max abs error 1.0; `ffn_moe_out` NMSE 9.6e-1 |
| M4 `expert_used_count = 5` | FAIL: `Top-6: ffn_moe_topk is [(100, 5)]` |
| M5 Kolibri selection branch removed (bias on `sigmoid`) | FAIL: `selection = top6(logits + bias) at 8/600 token-layers`; weights, `ffn_moe_out`, `ffn_out`, `ffn_post_norm`, `l_out` |
| M6 shared expert not added (`ffn_out = ffn_moe_out`) | FAIL: `nodes not captured: ['ffn_moe_out-0', 'ffn_shexp-0', …]`. The unused shared expert is pruned from the graph, and the routed output takes the name `ffn_out` |

## Vocab GGUF under `kolibri` (item 11)

`ggml-vocab-kolibri.gguf` was a Phase 2 shim under the placeholder
architecture `qwen3moe`. Without a model class, `llama_model_create` rejected
`kolibri` even with `vocab_only`.

- **New source:** `tools/tokenizer/vocab_gguf.py` now writes the file with
  the converter's own Kolibri class (`get_model_class("Kolibri1ForCausalLM")`,
  then `write_vocab()`), which is `convert_hf_to_gguf.py --vocab-only` plus a
  fixed `general.name`. Run on the HF cache, the CLI would record the
  revision SHA as `general.name` and `general.finetune`.
- **What changed in the file:**
  - `general.architecture = kolibri`;
  - 21 `kolibri.*` hyperparameter keys;
  - `general.file_type = 0`.

  Tokens, merges, token types, special IDs and the chat template are
  unchanged.
- **Checks:**
  - `test-tokenizer-0-kolibri` passes with the existing `.inp`/`.out`;
  - `compare.py` matches on all fuzz sets;
  - `check_arch.py` passes;
  - regenerating gives a byte-identical file.
- **Fork:** commit `1331f3f1b` on `feat/kolibri-vocab-arch`. As before, the
  binary is not part of the patch series.

## Patch 0007 re-exported

Fork PR #7 was merged with a follow-up commit `d5237ac15`, "model : kolibri
uses the per-layer RoPE frequencies of the K-shift". Main's patch 0007
predated it. `patches/llama.cpp/0007-kolibri-model.patch` is now
`git diff 6b81ee010 2e3588bf4`. Patches 0001–0007 applied to the pinned
commit give exactly `feat/kolibri` at 2e3588bf4, except
`models/ggml-vocab-kolibri.gguf`.

## What stays open

- **Metal precision for the real router.**
  - Metal computes the router logits through half-precision tiles, with a
    relative RMS error of about 4e-4 on the fixture.
  - On the fixture this only matters against the reference, not for
    libllama's own selection.
  - For the real model, a near-tie in the top 6 could resolve differently
    than in vLLM.
  - Phase 6 compares the selected expert ids, and Phase 8 decides whether the
    router logits need an F32 path on Metal.
- **Attention-side callbacks** (item 9) and all attention semantics: Phases 5
  and 6.
- **The Phase 4 Definition of Done:** the 50-layer graph needs a full-size
  GGUF.
