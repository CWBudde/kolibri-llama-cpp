# Kolibri support for llama.cpp --- Implementation Plan

## Goal

Add native Aleph Alpha Kolibri-1 support to `llama.cpp`, validate it
against Aleph Alpha's vLLM reference implementation, and produce a
quantized Metal build that is practical on a Mac with 48 GB unified
memory.

## Repository audit baseline

This plan was cross-checked against the public reference implementation:

- Repository: `Aleph-Alpha/aleph-alpha-inference`
- Branch: `main`
- Commit: `049a6a7bd2405b27d6d280d256bd3d585191c7ae` (release 1.0.0, 2026-10-03)
- Reference vLLM: `>=0.29.0,<0.30.0`
- Reference Transformers: `>=5.5.3`
- Key files: `aleph_alpha_inference/kolibri1.py`, `config.py`, `reasoning.py`, `tests/checkpoints.py`, `tests/test_kolibri1.py`, and the copied released chat template in `tests/kolibri1_chat_template.jinja`.

Important consequence: Kolibri is deliberately implemented as a relatively small delta on top of `Qwen3Moe*` classes in vLLM. The llama.cpp port should mirror that strategy: reuse Qwen3-MoE infrastructure aggressively and implement only the verified Kolibri deltas.

## Confirmed architecture facts

These are no longer assumptions and should be treated as implementation
requirements:

-   50 MoE layers; all 50 are MoE.
-   Hidden size: 2,560.
-   48 query heads, 4 KV heads, head size 128.
-   Per-head Q/K RMSNorm.
-   384 routed experts per layer, Top-6 selected per token.
-   1 shared expert per layer.
-   Expert hidden size: 512; MLP: SwiGLU.
-   Routing: token-choice, Top-6 sigmoid routing.
-   Attention pattern: 4 sliding-window layers followed by 1
    full-attention layer.
-   Sliding window: 512 preceding tokens plus current token.
-   RoPE base: 10,000; RoPE is applied only to sliding-window layers.
-   Native trained context: 262,144; validated extrapolation to
    1,048,576.
-   Vocabulary: 128,000; tokenizer was trained with UniBPE.
-   Reference serving path: Aleph Alpha's `aleph-alpha-inference` vLLM
    plugin.
-   BF16 checkpoint footprint: about 156 GB; FP8 checkpoint footprint:
    about 78 GB.

The most useful llama.cpp references are not just Qwen3-MoE: -
`src/models/qwen3moe.cpp` --- compact routed-MoE reference. -
`src/models/openai-moe.cpp` --- existing MoE + sliding-window
infrastructure. - `src/models/laguna.cpp` --- especially relevant
because it already implements a repeating full/SWA per-layer pattern and
per-layer-type RoPE behavior. - Existing hybrid SWA memory support
(`llama_memory_hybrid_iswa`) should be evaluated before adding any new
KV-cache mechanism.

## Phase 0 --- Reproducible reference environment

-   [ ] Record exact Kolibri model revision/commit.
-   [ ] Record exact `aleph-alpha-inference` and vLLM versions.
-   [ ] Run the official BF16 or FP8 reference implementation on
    suitable hardware.
-   [ ] Save a small deterministic reference corpus: token IDs, selected
    logits, generated tokens, and sampling settings.
-   [ ] Use deterministic/greedy decoding for numerical validation
    before testing recommended sampling (`temperature=1.0`,
    `top_p=0.97`, `top_k=128`).
-   [ ] Record reasoning/tool-call templates separately from base text
    generation.

**Definition of Done:** We have immutable reference outputs against
which the llama.cpp port can be tested.

## Phase 1 --- Checkpoint and tensor inventory

-   [x] Inspect `config.json`, safetensors index, tokenizer files,
    generation config, and chat template.
-   [x] Enumerate every tensor name, shape, dtype, and shard.
-   [x] Classify tensors into:
    -   [x] token embeddings / LM head;
    -   [x] attention Q/K/V/O;
    -   [x] per-head Q/K RMSNorm;
    -   [x] block RMSNorms;
    -   [x] router;
    -   [x] routed expert gate/up/down;
    -   [x] shared expert gate/up/down.
-   [x] Confirm whether routed expert tensors are stored packed or
    per-expert.
-   [x] Confirm exact shared-expert placement and residual/addition
    order from the reference implementation.
-   [x] Confirm router normalization/scaling details beyond the
    documented Top-6 sigmoid selection.
-   [x] Document BF16 and FP8 checkpoint tensor differences.
-   [x] Produce an explicit `HF tensor -> GGUF tensor` mapping table.

**Definition of Done:** No tensor required for a forward pass remains
unidentified.

**Result:** done; see `docs/phase1-tensor-inventory.md` and `inventory/`.

## Phase 2 --- Tokenizer compatibility

Kolibri's tokenizer vocabulary was *trained* with UniBPE. That does not
automatically imply that llama.cpp needs a new runtime tokenizer
algorithm: first test whether the exported tokenizer can be represented
by existing GGUF tokenizer metadata and llama.cpp's BPE implementation.

-   [x] Inspect `tokenizer.json` model type, pre-tokenizer, normalizer,
    decoder, merge table, added tokens, and byte fallback behavior.
-   [x] Attempt conversion using existing GGUF tokenizer machinery
    before implementing anything new.
-   [x] Preserve all special tokens and chat/reasoning/tool-call tokens.
-   [x] Build golden tests for:
    -   [x] German compounds (`Bundessozialgerichtes`,
        `Protokolldaten`);
    -   [x] umlauts and ß;
    -   [x] English;
    -   [x] whitespace/newlines;
    -   [x] arbitrary Unicode;
    -   [x] source code;
    -   [x] special/chat tokens;
    -   [x] invalid/edge UTF-8 behavior where applicable.
-   [x] Compare exact token ID sequences against Hugging
    Face/tokenizers.
-   [x] Compare detokenization byte-for-byte where meaningful.

**Decision gate:** Only add tokenizer runtime code if the shipped
tokenizer cannot be represented faithfully by existing llama.cpp/GGUF
BPE support.

**Definition of Done:** Golden tests produce identical token IDs to the
reference tokenizer.

**Result:** done. No new runtime tokenizer code was needed. The existing `gpt2`
vocab type with the `QWEN2` pre-tokenizer represents the tokenizer exactly; the
patch adds only a `kolibri` name registration (with "no BOS"). All 85 golden
cases and about 1.35 million fuzz strings match the reference. See
`docs/phase2-tokenizer.md`, `testdata/tokenizer/`, `tools/tokenizer/` and
`patches/llama.cpp/`.

## Phase 3 --- GGUF architecture and HF → GGUF conversion

-   [ ] Add `MODEL_ARCH.KOLIBRI` in `gguf-py/gguf/constants.py`.
-   [ ] Add matching `LLM_ARCH_KOLIBRI` in `src/llama-arch.h`.
-   [ ] Add architecture-name/key/tensor mappings in the normal
    llama.cpp architecture tables.
-   [ ] Define GGUF metadata for:
    -   [ ] 50 blocks;
    -   [ ] embedding size 2,560;
    -   [ ] 48/4 Q/KV heads;
    -   [ ] head dimension / RoPE dimension;
    -   [ ] RMSNorm epsilon;
    -   [ ] 384 experts / 6 active;
    -   [ ] expert FFN size 512;
    -   [ ] shared expert;
    -   [ ] SWA size 512;
    -   [ ] repeating SWA/full pattern;
    -   [ ] RoPE base 10,000 and SWA-only RoPE behavior.
-   [ ] Implement Kolibri converter class.
-   [ ] Map/pack expert tensors in the layout expected by llama.cpp.
-   [ ] Start with BF16 as the correctness reference.
-   [ ] Treat direct FP8 conversion as a follow-up optimization rather
    than blocking initial correctness.
-   [ ] Stream tensors/shards during conversion so the full 156 GB
    checkpoint never needs to reside in RAM.
-   [ ] Verify GGUF metadata, tensor count, shapes, and dtypes with
    automated assertions.

**Definition of Done:** A structurally correct unquantized GGUF is
produced reproducibly.

## Phase 4 --- Native model loading and MoE graph

Start from existing infrastructure rather than treating Kolibri as a
novel MoE implementation.

-   [ ] Add `llama_model_kolibri` and register it in the model factory.
-   [ ] Load embeddings, output norm/head, block norms, Q/K norms, and
    attention projections.
-   [ ] Reuse existing expert tensor representation where possible.
-   [ ] Reuse `build_moe_ffn()` or the closest current helper for the
    384 routed experts.
-   [ ] Configure Top-6 selection.
-   [ ] Configure sigmoid expert gating; verify whether
    normalization/scaling flags are required.
-   [ ] Add the shared expert path using existing shared-expert
    primitives if compatible.
-   [ ] Add routed + shared outputs in exactly the reference order.
-   [ ] Add graph callbacks/names useful for layer-by-layer debugging.

**Definition of Done:** The complete 50-layer unquantized graph builds
and executes without tensor-shape or unsupported-op errors.

## Phase 5 --- Hybrid attention and KV cache

This phase is now lower risk than originally assumed because llama.cpp
already contains reusable hybrid-SWA machinery.

-   [ ] Model the Kolibri pattern as `SWA, SWA, SWA, SWA, FULL`,
    repeating for 50 layers.
-   [ ] Evaluate `load_swa_pattern()` instead of introducing custom
    per-layer attention logic.
-   [ ] Use `LLAMA_SWA_TYPE_STANDARD` if its mask semantics match
    Kolibri's 512-preceding-token window.
-   [ ] Evaluate `llama_memory_hybrid_iswa` for mixed SWA/full KV
    storage.
-   [ ] Reuse existing GQA path for 48 Q heads / 4 KV heads.
-   [ ] Reuse existing per-head Q/K RMSNorm support.
-   [ ] Apply RoPE with base 10,000 on SWA layers only.
-   [ ] Ensure full-attention layers receive no positional rotation,
    matching Kolibri.
-   [ ] Validate the exact off-by-one semantics: 512 preceding tokens +
    current token.
-   [ ] Test at context boundaries: 511, 512, 513, and larger.
-   [ ] Test short contexts first, then 8k/16k/64k before attempting
    262k.

**Definition of Done:** Attention masks, RoPE behavior, and KV-cache
semantics match the reference implementation.

## Phase 6 --- Numerical validation

-   [ ] Feed identical token IDs to vLLM/reference and llama.cpp.
-   [ ] Compare embedding output.
-   [ ] Compare Q/K after QK RMSNorm.
-   [ ] Compare attention output for one SWA layer.
-   [ ] Compare attention output for one full layer.
-   [ ] Compare router logits/probabilities and selected expert IDs.
-   [ ] Compare routed expert output.
-   [ ] Compare shared expert output.
-   [ ] Compare complete layer outputs.
-   [ ] Compare final logits.
-   [ ] Compare greedy next-token sequences.
-   [ ] Establish tolerances separately for BF16 and any FP8 reference
    run.

**Hard gate:** Do not diagnose quantization quality until the
unquantized implementation passes this phase.

**Definition of Done:** llama.cpp BF16/F16 inference is numerically
consistent with the reference within documented tolerances.

## Phase 7 --- Quantization strategy

The 48 GB target makes mixed quantization more important than simply
producing a generic Q4.

-   [ ] Produce Q8 first as a quantizer sanity check.
-   [ ] Produce Q6/Q5 baselines if useful.
-   [ ] Produce Q4 variants.
-   [ ] Produce IQ3/Q3 variants if Q4 lacks memory headroom.
-   [ ] Keep router tensors at high precision initially.
-   [ ] Keep normalization tensors at high precision.
-   [ ] Evaluate Q/K projections and QK norms conservatively because
    routing/attention errors can amplify across 50 layers.
-   [ ] Evaluate whether shared experts deserve higher precision than
    routed experts.
-   [ ] Build an importance matrix if supported/useful for the selected
    quantization scheme.
-   [ ] Compare perplexity/task outputs and router expert-selection
    agreement against the unquantized model.
-   [ ] Specifically measure how often quantization changes Top-6 expert
    selection.

**Definition of Done:** At least one quantization retains acceptable
behavior and runs stably through Metal.

## Phase 8 --- 48 GB Apple Silicon target

A raw 4-bit estimate for 78.1B parameters is about 39.1 GB before
quantization metadata, higher-precision tensors, runtime buffers, KV
cache, and macOS. Q4 is therefore a boundary case, not a guaranteed fit.

For each candidate:

-   [ ] Measure GGUF file size.
-   [ ] Measure actual unified-memory use after load.
-   [ ] Measure Metal buffers/runtime overhead.
-   [ ] Measure SWA and full-attention KV-cache memory separately.
-   [ ] Measure prompt-processing tokens/s.
-   [ ] Measure generation tokens/s.
-   [ ] Measure expert-routing overhead.
-   [ ] Test 8k, 16k, and 32k contexts first.
-   [ ] Expand context only if memory headroom permits.
-   [ ] Test sustained generation and memory pressure.
-   [ ] Compare quality and Top-6 routing agreement with BF16.

**Target:** Prefer a configuration that leaves several GB of unified
memory headroom rather than merely loading successfully.

**Expected decision:** Q4 may fit only narrowly; an IQ3/Q3-style
quantization is likely to provide substantially safer headroom on a 48
GB Mac.

## Phase 9 --- User-facing inference behavior

Base inference correctness and chat/tool behavior should be tested
separately.

-   [ ] Port/preserve the Kolibri chat template.
-   [ ] Validate reasoning modes: none / low / medium / high.
-   [ ] Validate tool-call formatting.
-   [ ] Compare the official Kolibri reasoning/tool parsers with what
    can be represented through llama.cpp templates/grammars.
-   [ ] Confirm stop tokens and EOS behavior.
-   [ ] Only after base greedy inference is correct, validate Aleph
    Alpha's recommended sampling parameters.

**Definition of Done:** `llama-cli`/`llama-server` can reproduce
ordinary chat behavior and, where feasible, Kolibri reasoning/tool-call
conventions.

## Suggested PR split

### PR 1 --- Converter, metadata, tokenizer tests

-   [ ] `MODEL_ARCH` / `LLM_ARCH`.
-   [ ] GGUF metadata.
-   [ ] HF tensor conversion.
-   [ ] Tokenizer compatibility and golden tests.

### PR 2 --- Kolibri model + MoE

-   [ ] Model registration/loading.
-   [ ] Q/K norms and projections.
-   [ ] Routed Top-6 sigmoid MoE.
-   [ ] Shared expert.
-   [ ] Basic forward graph.

### PR 3 --- Hybrid SWA/full attention + reference tests

-   [ ] Repeating 4:1 pattern.
-   [ ] SWA-only RoPE.
-   [ ] Hybrid KV cache.
-   [ ] Numerical reference tests.

### PR 4 --- Quantization/Metal improvements, only if required

-   [ ] Quantization-specific tensor policies.
-   [ ] Metal fixes/performance tuning.
-   [ ] 48 GB memory/performance measurements.

## Immediate implementation tasks derived from the repository audit

- [x] Pin the reference to commit `049a6a7bd2405b27d6d280d256bd3d585191c7ae` in project notes/tests.
- [ ] Port the tiny synthetic checkpoint shape from `tests/checkpoints.py` into a llama.cpp conversion/inference fixture. This allows architecture work without downloading 156 GB.
- [ ] Add a standalone router unit test using 384 experts / Top-6 and a non-zero correction bias, matching `test_routing_semantics()` from the official repo.
- [ ] Add a tiny 6-layer hybrid-attention fixture so both SWA and full/RNoPE layers execute in one fast test.
- [x] Make sandwich norms a first-class mapping requirement before any full-checkpoint conversion.
- [ ] Treat BF16 as the initial source of truth; postpone direct FP8-source support until BF16 logits match.
- [ ] Copy the official chat-template compatibility cases into later llama.cpp template tests, especially the `reasoning_effort` precedence and tool-loop cases.

## First actionable milestone

Before downloading/processing the entire BF16 checkpoint, build a small
inspection script that reads model metadata and safetensors headers
without loading tensor payloads. Its output should be committed as a
machine-readable tensor inventory.

Then implement tokenizer golden tests and the GGUF metadata/converter
skeleton.

The first major milestone remains:

> Produce an unquantized Kolibri GGUF that runs in llama.cpp and matches
> the reference implementation at tokenizer, router/expert selection,
> layer-output, logits, and greedy next-token levels.

Only after that gate should the project optimize Q4/IQ3 for the 48 GB
Mac.
