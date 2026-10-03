# Phase 3: GGUF architecture registration

Scope of this step: the first three Phase 3 items. These register the `kolibri`
architecture in gguf-py and in libllama, and list its tensors. The converter
class, the GGUF metadata, and expert packing come next.

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
