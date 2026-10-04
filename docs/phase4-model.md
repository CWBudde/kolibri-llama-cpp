# Phase 4: model class and MoE graph

This report covers the first six Phase 4 items. `llama_model_kolibri` loads a
converted Kolibri GGUF and builds its forward graph (patch
`0007-kolibri-model.patch`). The graph runs on the tiny synthetic checkpoint,
on CPU and Metal. Nothing has been compared with the reference implementation
yet. The router semantics come in the next step, and the attention semantics
in Phase 5.

The patch is also commit `e091df2c8` on `feat/kolibri-model` of
[CWBudde/llama.cpp](https://github.com/CWBudde/llama.cpp), fork PR #7 into
`feat/kolibri`. Patches 0001–0007 applied to the pinned commit give exactly
that tree. The one exception is `models/ggml-vocab-kolibri.gguf`, which the
fork commits and the patches do not.

## Changes (`patches/llama.cpp/0007-kolibri-model.patch`)

| File | Change |
|---|---|
| `src/models/kolibri.cpp` (new) | `load_arch_hparams`, `load_arch_tensors`, graph |
| `src/models/models.h`, `src/llama-model.cpp` | class declaration, factory case |
| `src/llama-graph.cpp` | Kolibri selection branch in `build_moe_ffn`; F32 router logits |
| `tests/test-llama-archs.cpp` | `kolibri` skip removed, MoE-only, SWA and RoPE pattern arrays in the fixture |

The template is `laguna.cpp`: sigmoid MoE with `exp_probs_b`, one shared
expert, QK norm, hybrid SWA. Kolibri drops Laguna's attention gate, dense
lead layers and per-layer-type YaRN, and adds the sandwich norms.

### Hyperparameters

- **Required keys:**
  - RMS epsilon;
  - expert FFN length and shared-expert FFN length;
  - gating function;
  - `sliding_window`;
  - the two per-layer arrays `attention.sliding_window_pattern` and
    `attention.rope_pattern`.

  A GGUF without them fails to load.
- **Optional keys:**
  - `expert_weights_norm` and `expert_weights_scale`. The defaults (no
    renormalization, no scale) are Kolibri's values. The converter writes both
    anyway.
  - the shared-expert count, default 1.
- `swa_type = LLAMA_SWA_TYPE_STANDARD`, with `n_swa = 513` unchanged
  (Phase 1). The graph uses the iSWA KV cache.
- `LLM_TYPE_UNKNOWN`: no existing type fits 50 layers / 78B, and adding one
  needs entries in the type name table.

### Graph

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

This is the block structure from the Phase 1 inventory ("Block structure and
residual order"). Each step has a `cb()` name: `attn_norm`, `Qcur_normed`,
`Qcur_rope`, `attn_out`, `attn_post_norm`, `ffn_inp`, `ffn_norm`,
`ffn_moe_out`, `ffn_shexp`, `ffn_out`, `ffn_post_norm`, `l_out`.

### Router (Phase 1, pitfall 2)

`build_moe_ffn` adds `exp_probs_b` to the probabilities `sigmoid(logits)`,
following DeepSeek-V3. Kolibri adds it to the raw logits. The patch adds a
branch next to the existing LLAMA4 and GROVEMOE branches:

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

## Check (`tools/gguf/check_model.py`)

The check reuses the tiny checkpoint of Phase 3 (`cmd/kolibri-tiny`):
- 6 layers, SSSSFF;
- hidden 256, 8/2 heads;
- 8 experts, top 2;
- sliding window 65.

It converts the checkpoint to BF16 and to F32. Then it decodes the same 100
tokens through the libllama C API, with the model and all computation pinned
to one device. 100 tokens are more than the window, so the sliding-window
mask cuts in.

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
   last layer's `get_rows` of the output rows. This checks the structure, not
   the numerics against the reference.

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

`test-llama-archs` runs its own random fixture: 2 layers, one full and one
sliding, and 2 experts with both active.

```
|         kolibri|Apple M5 Pro|   MoE|  OK (4.71e-07)|       OK|
|         kolibri|  Accelerate|   MoE|  OK (3.01e-13)|       OK|
|         kolibri|Apple M5 Pro|   MoE|  OK (2.45e-13)|       OK|
|         kolibri|        Meta|   MoE|  OK (4.71e-07)|     SKIP|
```

### Why the strict comparison takes the routing out

The first runs failed: CPU vs Metal on the BF16 GGUF gave NMSE 8.4e-3. The
cause was established step by step.

1. **The fixture's scale.** The reference fixture draws its matrices with unit
   standard deviation (`torch.randn`). The activations then grow over the
   layers, and the error spread from one position to all later ones.
   `cmd/kolibri-tiny` now uses std 0.02 for matrices and 1 ± 0.1 for norms;
   see the Phase 3 doc. After that, NMSE was 3.7e-4, and exactly one position
   was off.
2. **BF16 rounding.** For BF16 matmuls the CPU rounds the activations to BF16,
   Metal and the BLAS path do not. With the F32 GGUF the error stays on one
   position (35); the median per-position NMSE is 1.7e-6.
3. **A top-k near-tie.** With `expert_used_count` overridden to 8, there is no
   selection. That one position then drops to 2.9e-6 as well, which confirms
   that two experts score almost the same there and rounding picks the other
   one. `test-llama-archs` never sees this, because its fixture activates
   both of its 2 experts.

### Mutation tests

Each mutation was built and run, then reverted.

| Mutation | `check_model.py` | `test-llama-archs` |
|---|---|---|
| `post_ffw_norm` not loaded | FAIL: `wrong number of tensors; expected 111, got 105` | OK |
| shared expert not loaded or used | FAIL: `expected 111, got 93` | OK |
| shared expert loaded, not added | FAIL wiring: `ffn_out-0 <- ['ffn_moe_weighted-0 (view)', …]` | OK |
| selection branch removed (bias on `sigmoid`) | FAIL wiring: `ffn_moe_probs_biased-0 <- ['ffn_moe_probs-0', …]` | OK |
| RoPE on every layer | FAIL wiring: `layer 4 (full_attention): RoPE nodes present` | OK |
| `kolibri` removed from `moe_mandatory` | — | FAIL: the dense fixture lacks `kolibri.expert_feed_forward_length` |

`test-llama-archs` builds its tensors from the class's own list, so only a
real GGUF reveals a missing load. Its consistency comparison cannot see a
wrongly wired graph, because CPU and Metal run the same wrong graph.

The wiring check proves where the bias, the shared expert and RoPE sit in
the graph. It does not prove that the numbers match the reference. That
needs a reference:
- for the router, the next batch dumps `ffn_moe_logits`, `exp_probs_b` and
  the selected experts, and checks `top_k(logits + bias)` and
  `sigmoid(logits)` numerically;
- for the attention, Phase 5 tests masks and boundaries.

## What stays open

- **Phase 4 items 5, 6, 8 and 9:**
  - Top-6;
  - the gating flags;
  - the routed + shared order against the reference;
  - callbacks proven useful for debugging.
- **Item 11:** the vocab GGUF regeneration. `check_arch.py` already expects the
  model class: its empty probe now gets past the architecture and fails on the
  missing vocabulary.
- **The Phase 4 Definition of Done:** the 50-layer graph. Only the 6-layer
  fixture and the `test-llama-archs` model have run.
- **Phase 5.** The graph already contains iSWA, RoPE on the SWA layers only and
  GQA, but none of it is checked against the reference.
