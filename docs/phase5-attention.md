# Phase 5: hybrid attention, KV cache and RoPE against the reference

This report covers Phase 5, plus the attention half of Phase 4 item 9 (graph
callbacks for layer-by-layer debugging):

- `LLAMA_SWA_TYPE_STANDARD` and Kolibri's 512-preceding-token window;
- RoPE with base 10,000 on the sliding-window layers only;
- no positional rotation on the full-attention layers;
- the exact off-by-one, and the boundaries 511, 512 and 513 and beyond, also
  across KV-cache reads and SWA-cache eviction;
- the 4:1 SWA/full pattern over the real 50 layers, and the per-layer split of
  the iSWA KV cache;
- GQA with 48 query and 4 KV heads, and the per-head Q/K RMSNorm;
- two evaluations: `load_swa_pattern()` and `llama_memory_hybrid_iswa`;
- long contexts: 8k, 16k and 64k tokens, and the real 262k.

`tools/gguf/check_attn.py` compares every attention step in libllama with the
reference implementation, layer by layer. It runs on two tiny checkpoints:
one with the real attention heads and sliding window, one with the real
50-layer SWA/full pattern. `tools/gguf/check_long.py` compares the attention
at long contexts on the first one.

The report covers three batches: #10 (window, off-by-one, RoPE, KV-cache
reads), #11 (layer pattern, iSWA cache split, GQA, QK norm) and the
long-context batch. None needed a change to the model class described in
[phase4-model.md](phase4-model.md): the fork already had the window, the RoPE
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
default fixture and its determinism test are unchanged.

## Fixture: `kolibri-tiny -pattern`

The real layer pattern on the default fixture's small heads: 50 layers, full
attention where `il % 5 == 4` (four sliding, then one full), and
`sliding_window` 513. `TestGeneratePattern` checks that `layer_types` equals
the released `inventory/bf16/config.json`. The small heads (8 Q, 2 KV,
head_dim 32) keep the capture of all 50 layers at about 0.6 GB.

`-router`, `-attn` and `-pattern` are mutually exclusive.

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

The `-pattern` fixture runs the batch and the evicting chunked run, in the
same configurations. Its SWA cache holds 40 layers in 768 cells (libllama log:
"creating SWA KV cache, size = 768 cells" and "768 cells, 40 layers"). The
single-token and full-cache runs are left to `-attn`.

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
- **KV cache:** from the libllama log, `llama_kv_cache_iswa` creates a non-SWA
  cache with exactly the full layers and a SWA cache with exactly the sliding
  layers, and no recurrent memory is created.
- **RoPE nodes:** they exist in exactly the sliding layers (0–3; for
  `-pattern`, the 40 layers with `il % 5 != 4`) and in no full layer.
- **Boundaries:** the `kqv_out` error at positions 511, 512, 513 and 514.
- **Window (off-by-one):** from the first position where the masks differ,
  libllama's deviation from the window-513 reference is projected onto the
  step from that reference to the window-512 one (and to the window-514 one).
  The coefficient `c` is about 0 for window 513 and about 1 for the other
  window, whatever the backend's precision. Bound: `|c| <= 0.1`, set by that
  meaning, not by a run.
- **No RoPE on the full layers:** a reference with RoPE applied on the full
  layers is at least 100× further off than the unrotated one.
- **Per-head Q/K RMSNorm:** one RMSNorm over the whole projection (all heads,
  the weight repeated per head) is at least 100× further off `Qcur/Kcur_normed`
  than the per-head one.
- **GQA mapping:** query head h reads KV head h // 12, as FlashAttention and
  ggml's `mul_mat` broadcast group them. The strided mapping h % 4 is at least
  100× further off `kqv_out`.

The 100× separations are the same factor as the full-layer RoPE check. The
smallest observed margin is 6.7e3 times that (no RoPE on the full layers,
`-pattern`, CPU flash attention).

### Thresholds

| Configuration | Bound | Observed |
|---|---|---|
| CPU, flash attention off, F32 KV | 1e-10 | at most 1.5e-13 (`attn_out`) |
| the same, `Qcur_rope`/`Kcur_rope` only | 1e-8 | 6.8e-10 |
| everything else (F16 KV, CPU flash attention, Metal) | 1e-4 (`test-llama-archs`) | at most 1.1e-6 over all positions, 3.6e-6 at a single position (CPU flash attention, single-token steps); `-pattern` at most 1.7e-6 (`attn_out`) |

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
every configuration to `|c| <= 4.2e-4` on the CPU and `<= 5.8e-3` on Metal
for `-attn`, and `<= 1.4e-2` for `-pattern`. The strict CPU configuration
gives at most 1.5e-7.

### Result

```
.venv/bin/python tools/gguf/check_attn.py --llama-cpp third_party/llama.cpp
```

rc 0, 570 PASS lines in 173 s: 380 for `-attn` (12 CPU and 8 Metal
configurations, 19 lines each) and 190 for `-pattern` (6 CPU and 4 Metal
configurations). The strict CPU configuration for `-attn`, then `-pattern`
with the evicting SWA cache, then the new lines on Metal:

```
PASS CPU attn [batch, flash attn off, KV f32] KV cache layers {'non-SWA': 2, 'SWA': 4} (llama_kv_cache_iswa, no recurrent memory: 2 full, 4 sliding)
PASS CPU attn [batch, flash attn off, KV f32] RoPE nodes in layers [0, 1, 2, 3] (sliding layers [0, 1, 2, 3])
PASS CPU attn [batch, flash attn off, KV f32] attn_norm vs reference: max NMSE over layers 3.0e-15 (<= 1e-10)
PASS CPU attn [batch, flash attn off, KV f32] Qcur_normed vs reference: max NMSE over layers 1.1e-14 (<= 1e-10)
PASS CPU attn [batch, flash attn off, KV f32] Kcur_normed vs reference: max NMSE over layers 1.1e-14 (<= 1e-10)
PASS CPU attn [batch, flash attn off, KV f32] Vcur vs reference: max NMSE over layers 8.6e-15 (<= 1e-10)
PASS CPU attn [batch, flash attn off, KV f32] Qcur_rope vs reference: max NMSE over layers 6.8e-10 (<= 1e-08)
PASS CPU attn [batch, flash attn off, KV f32] Kcur_rope vs reference: max NMSE over layers 6.8e-10 (<= 1e-08)
PASS CPU attn [batch, flash attn off, KV f32] kqv_out vs reference: max NMSE over layers 1.8e-14 (<= 1e-10)
PASS CPU attn [batch, flash attn off, KV f32] attn_out vs reference: max NMSE over layers 1.5e-13 (<= 1e-10)
PASS CPU attn [batch, flash attn off, KV f32] attn_post_norm vs reference: max NMSE over layers 3.1e-15 (<= 1e-10)
PASS CPU attn [batch, flash attn off, KV f32] ffn_inp vs reference: max NMSE over layers 7.5e-16 (<= 1e-10)
PASS CPU attn [batch, flash attn off, KV f32] kqv_out at positions 511: 2.7e-14, 512: 2.7e-14, 513: 2.7e-14, 514: 3.1e-14 (sliding layers, <= 1e-10)
PASS CPU attn [batch, flash attn off, KV f32] window 513, not 512: from position 512 on, libllama moves +1.4e-08 of the way to the window-512 reference (|c| <= 0.1)
PASS CPU attn [batch, flash attn off, KV f32] window 513, not 514: from position 513 on, libllama moves +1.6e-08 of the way to the window-514 reference (|c| <= 0.1)
PASS CPU attn [batch, flash attn off, KV f32] no RoPE on the full layers: a rotated reference is off by NMSE 2.8e-02, the unrotated one by 2.9e-15 (>= 100x)
PASS CPU attn [batch, flash attn off, KV f32] Qcur_normed per head: one RMSNorm over all 48 heads is off by NMSE 3.7e-03, the per-head one by 1.1e-14 (>= 100x)
PASS CPU attn [batch, flash attn off, KV f32] Kcur_normed per head: one RMSNorm over all 4 heads is off by NMSE 2.9e-03, the per-head one by 1.1e-14 (>= 100x)
PASS CPU attn [batch, flash attn off, KV f32] GQA 48/4: query head h reads KV head h // 12; with h % 4 kqv_out is off by NMSE 1.5e+00, with h // 12 by 1.8e-14 (>= 100x)
PASS CPU pattern [chunks of 64, SWA cache 768, flash attn off, KV f32] KV cache layers {'non-SWA': 10, 'SWA': 40} (llama_kv_cache_iswa, no recurrent memory: 10 full, 40 sliding)
PASS CPU pattern [chunks of 64, SWA cache 768, flash attn off, KV f32] RoPE nodes in layers 11110111101111011110111101111011110111101111011110 (sliding layers 11110111101111011110111101111011110111101111011110)
PASS CPU pattern [chunks of 64, SWA cache 768, flash attn off, KV f32] kqv_out vs reference: max NMSE over layers 1.6e-14 (<= 1e-10)
PASS CPU pattern [chunks of 64, SWA cache 768, flash attn off, KV f32] kqv_out at positions 511: 2.4e-14, 512: 2.4e-14, 513: 2.9e-14, 514: 2.4e-14 (sliding layers, <= 1e-10)
PASS CPU pattern [chunks of 64, SWA cache 768, flash attn off, KV f32] window 513, not 512: from position 512 on, libllama moves -1.5e-07 of the way to the window-512 reference (|c| <= 0.1)
PASS CPU pattern [chunks of 64, SWA cache 768, flash attn off, KV f32] window 513, not 514: from position 513 on, libllama moves +1.5e-07 of the way to the window-514 reference (|c| <= 0.1)
PASS CPU pattern [chunks of 64, SWA cache 768, flash attn off, KV f32] no RoPE on the full layers: a rotated reference is off by NMSE 1.0e-02, the unrotated one by 2.9e-15 (>= 100x)
PASS CPU pattern [chunks of 64, SWA cache 768, flash attn off, KV f32] GQA 8/2: query head h reads KV head h // 4; with h % 2 kqv_out is off by NMSE 7.5e-01, with h // 4 by 1.6e-14 (>= 100x)
PASS MTL0 attn [batch, flash attn on, KV f16] KV cache layers {'non-SWA': 2, 'SWA': 4} (llama_kv_cache_iswa, no recurrent memory: 2 full, 4 sliding)
PASS MTL0 attn [batch, flash attn on, KV f16] Qcur_normed per head: one RMSNorm over all 48 heads is off by NMSE 3.7e-03, the per-head one by 4.8e-08 (>= 100x)
PASS MTL0 attn [batch, flash attn on, KV f16] GQA 48/4: query head h reads KV head h // 12; with h % 4 kqv_out is off by NMSE 1.5e+00, with h // 12 by 1.2e-07 (>= 100x)
```

The `-pattern` fixture adds 62 s and the GQA contrast about 35 s to the 76 s
of the first version. The check keeps every configuration: the evicting run
on the 40-layer SWA cache is the one that exercises the real layer split.

With an F32 KV cache and flash attention off, the chunked and single-token
runs give exactly the batch run's numbers. In the other configurations they
stay within the same bounds and window coefficients. So the KV cache,
including the evicting 768-cell SWA cache, hands each query exactly the keys
of its window.

## Long contexts (`tools/gguf/check_long.py`)

The `-attn` fixture, converted to F32, with the GGUF `context_length`
overridden to the real 262144. The tokens are decoded as one sequence in
ubatches of 256 (flash attention off) or 512 (on), each reading the KV cache of
the earlier ones; `n_ctx` is the sequence length, and the SWA cache holds the
window plus one ubatch (1024 or 1280 cells, so it evicts all the way).

A float64 reference over all positions costs O(n²), so the check compares 66
query rows per length: the last 32 positions, 32 random ones from the second
half, and the two around the ubatch boundary at n/2. Two optional `Runner`
parameters keep this affordable:
- `outputs`: logits only at the last position, where 262144 rows would take
  134 GB;
- `rows`: `cb_eval` keeps only the positions a node is needed at. That is q and
  `kqv_out` at the checked rows; k and v at every position on the full layers;
  and k and v on the sliding layers within the window before a checked row.

The checks, per run:
- **KV cache:** `llama_kv_cache_iswa` with n cells for the 2 full layers, the 4
  sliding layers in the SWA cache.
- **`kqv_out`:** at the checked rows, against the attention recomputed in
  float64 from libllama's own q, k and v. On the full layers it covers all
  earlier keys, up to 262143 of them; on the sliding layers the window of 513.
  The bounds are those of `check_attn.py`: 1e-10 for the CPU with an F32 KV
  cache, 1e-4 otherwise. The reference uses libllama's rotated q and k, so the
  RoPE angles do not enter it.
- **`Qcur_rope`, `Kcur_rope`:** against float64 RoPE of libllama's own
  `Qcur/Kcur_normed`. They must be within 100× (`MAX_ROPE_VS_VLLM`, fixed
  before the first run) of the error of vLLM's float32 cos/sin cache at the
  same positions. That error comes from the `check_attn.py` port with a
  float32 cache, applied in float64 so that only the cache differs.

| Configuration | 8192 | 16384 | 65536 | 262144 |
|---|---|---|---|---|
| CPU, flash attention off, F32 KV (strict) | 11 s | 40 s | 617 s, run once | — |
| CPU, flash attention on, F16 KV | 5 s | 20 s | 181 s | — |
| Metal, flash attention off, F16 KV | 1 s | 3 s | 33 s | — |
| Metal, flash attention on, F16 KV | 1 s | 2 s | 16 s | 222 s |

The CPU runs on all cores (`n_threads = os.cpu_count()`); libllama's default
of 4 threads takes twice as long. Without flash attention the KQ matrix is
n_kv × n_ubatch × 48 heads in float32, so the ubatch stays at 256. The strict
CPU run at 65536 took 617 s, so it is not part of the check; its result is
below.

### Result

```
.venv/bin/python tools/gguf/check_long.py --llama-cpp third_party/llama.cpp
```

rc 0, 60 PASS lines in 566 s (12 configurations, 5 lines each). The longest
length on each device:

```
PASS CPU attn [65536 tokens, ubatch 512, flash attn on, KV f16] KV cache layers {'non-SWA': 2, 'SWA': 4}, cells {'non-SWA': 65536, 'SWA': 1280} (llama_kv_cache_iswa: 65536 cells for the 2 full layers, the 4 sliding layers in the SWA cache; decoded in 181 s)
PASS CPU attn [65536 tokens, ubatch 512, flash attn on, KV f16] kqv_out vs reference on the full layers (all earlier keys) at 66 positions in [32767, 65535]: max NMSE 1.8e-10 (<= 0.0001)
PASS CPU attn [65536 tokens, ubatch 512, flash attn on, KV f16] kqv_out vs reference on the sliding layers (window 513) at 66 positions in [32767, 65535]: max NMSE 6.7e-08 (<= 0.0001)
PASS CPU attn [65536 tokens, ubatch 512, flash attn on, KV f16] Qcur_rope vs float64 RoPE at 66 positions in [32767, 65535]: NMSE 5.2e-06, vLLM's float32 cos/sin cache 2.1e-07 (<= 100x)
PASS MTL0 attn [262144 tokens, ubatch 512, flash attn on, KV f16] KV cache layers {'non-SWA': 2, 'SWA': 4}, cells {'non-SWA': 262144, 'SWA': 1280} (llama_kv_cache_iswa: 262144 cells for the 2 full layers, the 4 sliding layers in the SWA cache; decoded in 222 s)
PASS MTL0 attn [262144 tokens, ubatch 512, flash attn on, KV f16] kqv_out vs reference on the full layers (all earlier keys) at 66 positions in [131071, 262143]: max NMSE 1.2e-08 (<= 0.0001)
PASS MTL0 attn [262144 tokens, ubatch 512, flash attn on, KV f16] kqv_out vs reference on the sliding layers (window 513) at 66 positions in [131071, 262143]: max NMSE 1.4e-07 (<= 0.0001)
PASS MTL0 attn [262144 tokens, ubatch 512, flash attn on, KV f16] Qcur_rope vs float64 RoPE at 66 positions in [131071, 262143]: NMSE 6.8e-06, vLLM's float32 cos/sin cache 2.9e-06 (<= 100x)
```

The strict CPU configuration, run once at 65536 (617 s to decode):

```
PASS CPU [65536 tokens, ubatch 256, flash attn off, KV f32] kqv_out vs reference on the full layers (all earlier keys) at 66 positions in [32767, 65535]: max NMSE 5.6e-14 (<= 1e-10)
PASS CPU [65536 tokens, ubatch 256, flash attn off, KV f32] kqv_out vs reference on the sliding layers (window 513) at 66 positions in [32767, 65535]: max NMSE 2.6e-14 (<= 1e-10)
```

`kqv_out` does not grow with the context: in the strict configuration it is
1.1e-14 at 8192 and 5.6e-14 at 65536 on the full layers, 2.6e-14 on the
sliding layers at every length. With flash attention it stays at 1.2e-8 on
Metal up to 262144.

### RoPE at large positions

`Qcur_rope` against float64, the largest NMSE over the sliding layers
(`Kcur_rope` is the same within 10%):

| Positions | CPU (`ggml_rope_cache_init`) | Metal | vLLM float32 cache | CPU / vLLM | Metal / vLLM |
|---|---|---|---|---|---|
| up to 1100 (`check_attn.py`) | 6.8e-10 | 6e-11 | 2.3e-11 | 30 | 2.6 |
| 4095–8191 | 9.9e-8 | 8.5e-9 | 3.8e-9 | 26 | 2.2 |
| 8191–16383 | 3.3e-7 | 2.9e-8 | 1.2e-8 | 28 | 2.4 |
| 32767–65535 | 5.2e-6 | 4.8e-7 | 2.1e-7 | 25 | 2.3 |
| 131071–262143 | not run | 6.8e-6 | 2.9e-6 | — | 2.3 |

The error grows with the square of the position for all three, as float32
rounding of an angle of about p radians predicts. The ratio to vLLM's own
float32 cache stays flat: about 27 on the CPU, which multiplies the angle once
per frequency, and about 2.3 on Metal. Both are inside the 100× bound at every
length.

At 262144 the reference's own float32 cache is off from exact RoPE by NMSE
2.9e-6 in q and k, so exact parity with float64 is not the target there.

## Answers to the plan items

- **The 4:1 pattern for 50 layers (5.1):** the converter writes
  `attention.sliding_window_pattern` and `attention.rope_pattern` from
  `layer_types` (Phase 3, `check_metadata.py`). On the 50-layer `-pattern`
  fixture, libllama has RoPE nodes in exactly the 40 sliding layers, a KV
  cache split of 10 non-SWA and 40 SWA layers, and `kqv_out` matching the
  window-513 reference on the sliding layers and the causal one on the full
  layers, also with an evicting SWA cache.
- **`load_swa_pattern()` (5.2):** not used; the required explicit array stays.
  - `kolibri.cpp` reads `attention.sliding_window_pattern` with a required
    `ml.get_arr` into `hparams.is_swa_impl`, and `rope_pattern` the same way.
    There is no custom per-layer logic: the graph asks `hparams.is_swa(il)`
    and `has_rope(il)`, and `build_attn_inp_kv_iswa` does the rest.
  - `llama_model_base::load_swa_pattern` (`src/llama-model.cpp:3454`) runs
    the same `get_arr`, but optional, and falls back to a period through
    `set_swa_pattern` when the array is missing. The fallback serves
    architectures whose older GGUFs stored only a period (gemma2/3, cohere2,
    exaone4, afmoe, laguna). Recent architectures that always write the array
    use the required `get_arr`, like Kolibri: gemma4, step35, granite-swa,
    maple, dots3note, spark2-5, dflash.
  - A period cannot express the reference's own fixture (SSSSFF): with
    `set_swa_pattern(5)` (M7) the 50-layer model still passes, but the
    6-layer fixture fails at layer 5. A Kolibri GGUF always carries the
    array, and a missing one fails the load instead of silently using a
    period.
- **`llama_memory_hybrid_iswa` (5.4):** it does not apply. It pairs a
  recurrent memory with an iSWA attention cache
  (`src/llama-memory-hybrid-iswa.h`), for models whose layers are attention
  or recurrent. Kolibri has no recurrent layers, `llm_arch_is_hybrid` is
  false for it, and `llama_model::create_memory` gives it
  `llama_kv_cache_iswa`: one cache for the full layers and one, window-sized,
  for the sliding layers. The check asserts that split from the libllama log
  in every run, with no recurrent memory, and the chunked runs show that the
  evicting SWA cache hands each query exactly its window.
- **GQA 48/4 (5.5):** the existing path (`build_qkv` with `n_head_kv`, then
  `build_attn`) groups query heads contiguously: head h reads KV head h // 12,
  as in FlashAttention. `kqv_out` matches that reference to 1.8e-14; the
  strided mapping h % 4 is off by 1.5.
- **Per-head Q/K RMSNorm (5.6):** `build_norm` on the `[head_dim, n_head,
  n_tokens]` view with the `[head_dim]` weight is the reference's
  `RMSNorm(head_dim)` on `q.view(..., n_heads, head_dim)`: `Qcur/Kcur_normed`
  match to 1.1e-14. One RMSNorm over all heads is off by 3.7e-3 (Q) and
  2.9e-3 (K). Without the Q or K norm (M8, M9) exactly that node fails.
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
- **Short contexts, then 8k/16k/64k, then 262k (5.11):** the short case is
  `check_attn.py` (1100 tokens). `check_long.py` decodes 8192, 16384 and 65536
  tokens on the CPU and Metal, and 262144 on Metal with flash attention. In
  every run the full layers hold n cells, and `kqv_out` matches the float64
  attention over all earlier keys. It does not degrade with length: 5.6e-14 at
  65536 in the strict configuration, 1.2e-8 at 262144 on Metal. Only the RoPE
  angles lose precision with the position, as float32 does in vLLM's own
  cache; libllama stays within 30× (CPU) and 2.6× (Metal) of that.
- **Graph callbacks, attention half (4.9):** the existing names (`attn_norm`,
  `Qcur/Kcur_normed`, `Vcur`, `Qcur/Kcur_rope`, `kqv_out`, `attn_out`,
  `attn_post_norm`, `ffn_inp`, `l_out`) suffice to localize each step. Each
  mutation fails at the node it changes: M3 and M6 only at the RoPE nodes,
  M1, M2 and M4 at `kqv_out`, M5 at the RoPE-node list and the full layers'
  `kqv_out`. One fix was needed in the check harness, not in libllama:
  `Vcur` names two nodes, the matmul and its reshape, so the Runner keeps the
  last node of a name per computation.

## Mutation tests

Each mutation was run, then reverted. M1–M3 and M10 are kv overrides through
the C API; M4–M9 and M11 are libllama variants built from an edited
`src/models/kolibri.cpp`. M1–M6 ran on `-attn`, M7 on both fixtures, M10 and
M11 through `check_long.py`. The table shows the strict CPU configuration
(batch, flash attention off, F32 KV; for M10 and M11, 8192 tokens); the F16
configurations fail in the same checks.

| Mutation | `check_attn.py` (CPU, strict) |
|---|---|
| M1 `attention.sliding_window = 512` | FAIL: `kqv_out at positions 511: 2.7e-14, 512: 2.0e-03, 513: 1.1e-03`; window coefficient `+1.0e+00` toward 512; `kqv_out` NMSE 3.2e-4 |
| M2 `attention.sliding_window = 514` | FAIL: `511: 2.7e-14, 512: 2.7e-14, 513: 2.2e-03`; coefficient `+1.0e+00` toward 514 |
| M3 `rope.freq_base = 20000` | FAIL: `Qcur_rope`, `Kcur_rope` NMSE 1.2; nothing else |
| M4 `swa_type = LLAMA_SWA_TYPE_CHUNKED` | FAIL: `kqv_out at positions 511: 2.7e-14, 512: 2.7e-14, 513: 1.6e+02`; `kqv_out` NMSE 1.0 |
| M5 RoPE on every layer (`has_rope` ignored) | FAIL: `RoPE nodes in layers [0, 1, 2, 3, 4, 5]`; the full layers match a rotated reference (1.6e-12), not the unrotated one (3.4e-2) |
| M6 RoPE type NORM instead of NEOX | FAIL: `Qcur_rope` NMSE 1.7, `Kcur_rope` 1.8 |
| M7 `hparams.set_swa_pattern(5)` instead of the pattern array | `-attn` FAIL: `KV cache layers {'non-SWA': 1, 'SWA': 5}`, `kqv_out` NMSE 5.9e-2 (layer 5 gets the window), and as a consequence the full-layer RoPE and GQA contrasts; `-pattern` PASS (0 failures) |
| M8 no `attn_q_norm` | FAIL: `Qcur_normed` NMSE 4.7e-1 and its per-head contrast; nothing else |
| M9 no `attn_k_norm` | FAIL: `Kcur_normed` NMSE 4.7e-1 and its per-head contrast; nothing else |
| M10 `rope.freq_base = 10001` | FAIL: `Qcur_rope`, `Kcur_rope` NMSE 1.7e-4 at 8192 against a bound of 3.8e-7 (100 × vLLM's 3.8e-9); 1.0e-2 at 65536 on Metal; `kqv_out` and the KV cache pass |
| M11 RoPE on every layer (`has_rope` ignored) | FAIL: `kqv_out` on the full layers, NMSE 2.6e-1, since the reference attends with the unrotated `Qcur/Kcur_normed` there; the sliding layers and RoPE pass |

M10 changes each angle by up to 1e-4 relative, so its error grows with the
position: 60× from 8192 to 65536 tokens.

## What stays open

- **Phase 6:** the same comparison against a vLLM run of the real checkpoint.
