# Phase 5: hybrid attention masks and RoPE against the reference

This report covers the mask and RoPE half of Phase 5, plus the attention half
of Phase 4 item 9 (graph callbacks for layer-by-layer debugging):

- `LLAMA_SWA_TYPE_STANDARD` and Kolibri's 512-preceding-token window;
- RoPE with base 10,000 on the sliding-window layers only;
- no positional rotation on the full-attention layers;
- the exact off-by-one, and the boundaries 511, 512 and 513 and beyond, also
  across KV-cache reads and SWA-cache eviction.

`tools/gguf/check_attn.py` compares every attention step in libllama with the
reference implementation, layer by layer. It runs on a tiny checkpoint with
the real attention heads and sliding window.

The model class is described in [phase4-model.md](phase4-model.md). This
batch needed no change to it: the fork already had the window, the RoPE
pattern and the iSWA cache; they had not been checked against the reference.

## The reference

- **[`Aleph-Alpha/aleph-alpha-inference@049a6a7`](https://github.com/Aleph-Alpha/aleph-alpha-inference/tree/049a6a7bd2405b27d6d280d256bd3d585191c7ae),
  `aleph_alpha_inference/kolibri1.py`, `Kolibri1Attention` (lines 81–121):**
  - full-attention layers pass `sliding_window=None` to vLLM's `Attention` and
    have no rotary embedding;
  - sliding layers pass `per_layer_sliding_window=config.sliding_window` and
    use `get_rope(head_dim, max_position, rope_parameters)`;
  - q and k are RMS-normalized per head, then rotated; the scale is
    `head_dim**-0.5`.
- **vLLM v0.29.0** (the reference pins `>=0.29.0,<0.30.0`):
  - `vllm/v1/attention/backends/flash_attn.py:862`:
    `self.sliding_window = (sliding_window - 1, 0)`. FlashAttention's window
    `(left, right)` lets query `i` see key `j` for `i - left <= j <= i + right`,
    so with `sliding_window = 513` a token sees the 512 preceding tokens and
    itself.
  - `vllm/model_executor/layers/rotary_embedding/__init__.py:36`:
    `get_rope(..., is_neox_style=True)`, and `rotary_dim = head_size *
    partial_rotary_factor` with the default 1.0, so all 128 dimensions rotate.
  - `rotary_embedding/base.py` and `common.py`: `_compute_cos_sin_cache`,
    `RotaryEmbedding.forward_static` and `ApplyRotaryEmb.forward_static`.
    `check_attn.py` ports these three, computing the cos/sin cache in float64
    where vLLM uses float32.
- **llama.cpp** (`src/llama-hparams.h:489`): `LLAMA_SWA_TYPE_STANDARD` masks
  key `p0` for query `p1` when `p1 - p0 >= n_swa`. With `n_swa = 513` that is
  the same mask, as Phase 1 already concluded from the source alone.

## Fixture: `kolibri-tiny -attn`

| Parameter | Value | Default fixture | Real model |
|---|---|---|---|
| query heads | 48 | 8 | 48 |
| KV heads | 4 | 2 | 4 |
| head_dim | 128 | 32 | 128 |
| sliding_window | 513 | 65 | 513 |
| layers | 6: 4 sliding, 2 full | same | 50 |
| hidden, experts | 256, 8 (top 2) | same | 2560, 384 (top 6) |

`TestGenerateAttn` checks the config and the attention tensor shapes. The
default fixture and its determinism test are unchanged. `-router` and `-attn`
are mutually exclusive.

## Check (`tools/gguf/check_attn.py`)

The fixture is converted to F32, so the weights in libllama equal the
safetensors values. 1100 tokens, more than two windows, are decoded in four
ways:

| Run | `llama_decode` calls | SWA cache |
|---|---|---|
| batch | one call with 1100 tokens | 2048 cells (full) |
| chunks of 64, SWA cache 768 | 17 × 64 + 12; each chunk reads the cache of the earlier ones | 768 cells, so cells are evicted and reused (libllama log: "creating SWA KV cache, size = 768 cells") |
| chunks of 64, full SWA cache | the same | 2048 cells |
| single tokens 500–529 | 500, then 30 single tokens across 511/512/513, then 570 | 1280 cells |

The four runs are crossed with flash attention off and on, on every device.
The CPU also runs with an F32 KV cache, with flash attention off.

`cb_eval` captures each layer's attention nodes, one part per `llama_decode`
call. Every step is recomputed in float64 from libllama's own input to that
step, so each failure points at one node:

| Node | Reference |
|---|---|
| `attn_norm` | RMSNorm of the layer input (the embedding row for layer 0, else the previous `l_out`) |
| `Qcur_normed`, `Kcur_normed`, `Vcur` | projections of `attn_norm`, then the per-head RMSNorm for q and k |
| `Qcur_rope`, `Kcur_rope` | the vLLM NeoX RoPE of `Qcur/Kcur_normed`, sliding layers only |
| `kqv_out` | attention of libllama's own q, k and v: window `(512, 0)` on the sliding layers, causal on the full layers; GQA groups of 12 query heads per KV head |
| `attn_out` | `kqv_out @ Wo` |
| `attn_post_norm`, `ffn_inp` | the sandwich norm and the residual add |

Further checks:
- **RoPE nodes:** they exist in layers 0–3 and in no full layer.
- **Boundaries:** the `kqv_out` error at positions 511, 512, 513 and 514.
- **Window (off-by-one):** from the first position where the masks differ,
  libllama's deviation from the window-513 reference is projected onto the
  step from that reference to the window-512 one (and to the window-514 one).
  The coefficient `c` is about 0 for window 513 and about 1 for the other
  window, whatever the backend's precision. Bound: `|c| <= 0.1`, set by that
  meaning, not by a run.
- **No RoPE on the full layers:** a reference with RoPE applied on the full
  layers is at least 100× further off than the unrotated one.

### Thresholds

| Configuration | Bound | Observed |
|---|---|---|
| CPU, flash attention off, F32 KV | 1e-10 | at most 1.5e-13 (`attn_out`) |
| the same, `Qcur_rope`/`Kcur_rope` only | 1e-8 | 6.8e-10 |
| everything else (F16 KV, CPU flash attention, Metal) | 1e-4 (`test-llama-archs`) | at most 1.1e-6 over all positions, 3.6e-6 at a single position (CPU flash attention, single-token steps) |

The three non-strict cases each have a cause found in the source:

- **RoPE angles in float32.** `ggml_rope_cache_init` (`ggml/src/ggml-cpu/ops.cpp:6207`)
  starts at `theta = p` and multiplies by `theta_scale` once per frequency, in
  float32, so the rounding accumulates over 64 steps. At positions up to 1100
  this gives NMSE 6.8e-10. vLLM's own float32 cos/sin cache is off from
  float64 by 2.3e-11. Metal's RoPE gives 6e-11.
- **Flash attention always runs on an F16 KV.** `build_attn_mha`
  (`src/llama-graph.cpp:2663-2669`) casts an F32 K and V to F16 before
  `ggml_flash_attn_ext`. An F32 KV cache therefore changes nothing with flash
  attention, and the check skips that combination.
- **F16 KV cache:** `kqv_out` NMSE about 5e-7.

On F16 alone the window bound is weak: an off-by-one gives `kqv_out` NMSE
3.2e-4 against the 1e-4 bound, a margin of 3×. The projection test holds
every configuration to `|c| <= 4.2e-4` on the CPU and `<= 5.8e-3` on Metal.

### Result

```
.venv/bin/python tools/gguf/check_attn.py --llama-cpp third_party/llama.cpp
```

rc 0, 300 PASS lines (12 CPU and 8 Metal configurations, 15 lines each).
The strict CPU configuration and one Metal configuration:

```
PASS CPU [batch, flash attn off, KV f32] RoPE nodes in layers [0, 1, 2, 3] (sliding layers [0, 1, 2, 3])
PASS CPU [batch, flash attn off, KV f32] attn_norm vs reference: max NMSE over layers 3.0e-15 (<= 1e-10)
PASS CPU [batch, flash attn off, KV f32] Qcur_normed vs reference: max NMSE over layers 1.1e-14 (<= 1e-10)
PASS CPU [batch, flash attn off, KV f32] Kcur_normed vs reference: max NMSE over layers 1.1e-14 (<= 1e-10)
PASS CPU [batch, flash attn off, KV f32] Vcur vs reference: max NMSE over layers 8.6e-15 (<= 1e-10)
PASS CPU [batch, flash attn off, KV f32] Qcur_rope vs reference: max NMSE over layers 6.8e-10 (<= 1e-08)
PASS CPU [batch, flash attn off, KV f32] Kcur_rope vs reference: max NMSE over layers 6.8e-10 (<= 1e-08)
PASS CPU [batch, flash attn off, KV f32] kqv_out vs reference: max NMSE over layers 1.8e-14 (<= 1e-10)
PASS CPU [batch, flash attn off, KV f32] attn_out vs reference: max NMSE over layers 1.5e-13 (<= 1e-10)
PASS CPU [batch, flash attn off, KV f32] attn_post_norm vs reference: max NMSE over layers 3.1e-15 (<= 1e-10)
PASS CPU [batch, flash attn off, KV f32] ffn_inp vs reference: max NMSE over layers 7.5e-16 (<= 1e-10)
PASS CPU [batch, flash attn off, KV f32] kqv_out at positions 511: 2.7e-14, 512: 2.7e-14, 513: 2.7e-14, 514: 3.1e-14 (sliding layers, <= 1e-10)
PASS CPU [batch, flash attn off, KV f32] window 513, not 512: from position 512 on, libllama moves +1.4e-08 of the way to the window-512 reference (|c| <= 0.1)
PASS CPU [batch, flash attn off, KV f32] window 513, not 514: from position 513 on, libllama moves +1.6e-08 of the way to the window-514 reference (|c| <= 0.1)
PASS CPU [batch, flash attn off, KV f32] no RoPE on the full layers: a rotated reference is off by NMSE 2.8e-02, the unrotated one by 2.9e-15 (>= 100x)
PASS MTL0 [chunks of 64, SWA cache 768, flash attn on, KV f16] kqv_out vs reference: max NMSE over layers 1.2e-07 (<= 0.0001)
PASS MTL0 [chunks of 64, SWA cache 768, flash attn on, KV f16] kqv_out at positions 511: 1.4e-07, 512: 1.4e-07, 513: 1.3e-07, 514: 1.3e-07 (sliding layers, <= 0.0001)
PASS MTL0 [chunks of 64, SWA cache 768, flash attn on, KV f16] window 513, not 512: from position 512 on, libllama moves -3.3e-05 of the way to the window-512 reference (|c| <= 0.1)
PASS MTL0 [chunks of 64, SWA cache 768, flash attn on, KV f16] window 513, not 514: from position 513 on, libllama moves +3.4e-05 of the way to the window-514 reference (|c| <= 0.1)
PASS MTL0 [chunks of 64, SWA cache 768, flash attn on, KV f16] no RoPE on the full layers: a rotated reference is off by NMSE 2.8e-02, the unrotated one by 3.5e-09 (>= 100x)
```

With an F32 KV cache and flash attention off, the chunked and single-token
runs give exactly the batch run's numbers. In the other configurations they
stay within the same bounds and window coefficients. So the KV cache,
including the evicting 768-cell SWA cache, hands each query exactly the keys
of its window.

## Answers to the plan items

- **`LLAMA_SWA_TYPE_STANDARD` (5.3):** it matches. `kqv_out` on the sliding
  layers equals the reference with FlashAttention window `(512, 0)` in every
  run, and `CHUNKED` fails (M4).
- **RoPE base 10,000 on the sliding layers only (5.7):** `Qcur/Kcur_rope` equal
  vLLM's NeoX RoPE with `rope_theta` from `config.json`, in layers 0–3. A
  different base (M3) or the GPT-J rotation (M6) fails.
- **No rotation on the full layers (5.8):** layers 4 and 5 have no RoPE node,
  and their `kqv_out` matches the unrotated reference to 2.9e-15, against
  2.8e-2 for a rotated one. RoPE on every layer (M5) fails.
- **Off-by-one (5.9):** 512 preceding tokens plus the current one. The window
  coefficient is at most 4.2e-4 (CPU) and 5.8e-3 (Metal) toward window 512 or
  514. Both overrides (M1, M2) move it to 1.0.
- **Boundaries 511, 512, 513 and beyond (5.10):** 1100 tokens. The errors at
  511–514 are those of the other positions, in one batch, in chunks with
  cache reads and eviction, and in single-token steps across the boundary.
  M1 leaves position 511 unchanged and fails from 512 on; M2 fails from 513 on.
- **Graph callbacks, attention half (4.9):** the existing names (`attn_norm`,
  `Qcur/Kcur_normed`, `Vcur`, `Qcur/Kcur_rope`, `kqv_out`, `attn_out`,
  `attn_post_norm`, `ffn_inp`, `l_out`) suffice to localize each step. Each
  mutation fails at the node it changes: M3 and M6 only at the RoPE nodes,
  M1, M2 and M4 at `kqv_out`, M5 at the RoPE-node list and the full layers'
  `kqv_out`. One fix was needed in the check harness, not in libllama:
  `Vcur` names two nodes, the matmul and its reshape, so the Runner keeps the
  last node of a name per computation.

## Mutation tests

Each mutation was run, then reverted. M1–M3 are kv overrides through the C
API; M4–M6 are libllama variants built from an edited `src/models/kolibri.cpp`.
The table shows the strict CPU configuration (batch, flash attention off, F32
KV); the F16 configurations fail in the same checks.

| Mutation | `check_attn.py` (CPU, strict) |
|---|---|
| M1 `attention.sliding_window = 512` | FAIL: `kqv_out at positions 511: 2.7e-14, 512: 2.0e-03, 513: 1.1e-03`; window coefficient `+1.0e+00` toward 512; `kqv_out` NMSE 3.2e-4 |
| M2 `attention.sliding_window = 514` | FAIL: `511: 2.7e-14, 512: 2.7e-14, 513: 2.2e-03`; coefficient `+1.0e+00` toward 514 |
| M3 `rope.freq_base = 20000` | FAIL: `Qcur_rope`, `Kcur_rope` NMSE 1.2; nothing else |
| M4 `swa_type = LLAMA_SWA_TYPE_CHUNKED` | FAIL: `kqv_out at positions 511: 2.7e-14, 512: 2.7e-14, 513: 1.6e+02`; `kqv_out` NMSE 1.0 |
| M5 RoPE on every layer (`has_rope` ignored) | FAIL: `RoPE nodes in layers [0, 1, 2, 3, 4, 5]`; the full layers match a rotated reference (1.6e-12), not the unrotated one (3.4e-2) |
| M6 RoPE type NORM instead of NEOX | FAIL: `Qcur_rope` NMSE 1.7, `Kcur_rope` 1.8 |

## What stays open

- **Phase 5 items 1, 2, 4, 5 and 6:**
  - 1: the 4:1 pattern for 50 layers;
  - 2: evaluate `load_swa_pattern`;
  - 4: evaluate `llama_memory_hybrid_iswa`;
  - 5: GQA 48/4;
  - 6: Q/K RMSNorm.

  The check already runs 48/4 heads and compares `Qcur/Kcur_normed`, but these
  items were not part of this batch.
- **Phase 5 item 11:** contexts of 8k, 16k, 64k and 262k. A float64 O(n²)
  reference at 64k needs a different approach.
- **Phase 6:** the same comparison against a vLLM run of the real checkpoint.
