# Phase 3: GGUF architecture registration and converter

This report covers two steps:

1. The first three Phase 3 items register the `kolibri` architecture in gguf-py
   and in libllama, and list its tensors (patch 0002).
2. The converter class writes the first five GGUF metadata items: block count,
   embedding size, head counts, head/RoPE dimension, and RMSNorm epsilon
   (patch 0003). See [Converter and base metadata](#converter-and-base-metadata).

The remaining metadata, expert packing and tensor conversion come next.

The patches are also kept as commits on the `feat/kolibri` branch of
[CWBudde/llama.cpp](https://github.com/CWBudde/llama.cpp). Patches 0001 and
0002 applied to the pinned commit give exactly that branch's tree. The one
exception is `models/ggml-vocab-kolibri.gguf`, which the fork commits and the
patches do not.

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

For the same reason, `ggml-vocab-kolibri.gguf` keeps the `qwen3moe`
placeholder architecture. `llama_model_create` runs even with `vocab_only`,
so a vocab GGUF that declares `kolibri` does not load before Phase 4.

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
  conversion therefore fails until the tensor mapping exists.

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
`expert_used_count` (6) and `rope.freq_base` (10000). The later metadata items
own those keys and will check them together with the rest of the MoE and
SWA/RoPE metadata.

The remaining items can use existing keys, so no new GGUF keys are needed:

- **SWA/full pattern:** `attention.sliding_window_pattern`, as a per-layer bool
  array (as granite-swa does).
- **SWA-only RoPE:** `attention.rope_pattern`, a per-layer bool array where
  1 = RoPE. `llama_hparams::has_rope` reads it.

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
