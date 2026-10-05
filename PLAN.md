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

## Done so far

Evidence, mutation tests and reproduction steps live in `docs/`. Everything
below except the last bullet is checked on tiny synthetic checkpoints
(`cmd/kolibri-tiny`) on CPU and Metal.

-   **Checkpoint and tensor inventory.** Every BF16 and FP8 tensor is
    identified and classified; the HF → GGUF mapping, expert storage,
    residual order, router semantics and BF16/FP8 differences are recorded.
    `cmd/kolibri-inventory`, `inventory/`;
    [docs/checkpoint.md](docs/checkpoint.md).
-   **Tokenizer.** No new runtime code: the `gpt2` vocab with the `QWEN2`
    pre-tokenizer represents it exactly, and the patch only registers
    `kolibri` (no BOS). 85 golden cases and about 1.35 million fuzz strings
    match the reference. `tools/tokenizer/`;
    [docs/tokenizer.md](docs/tokenizer.md).
-   **GGUF architecture and converter.** `kolibri` is registered in gguf-py
    and libllama. The converter writes all hyperparameters (incl. window 513,
    the per-layer SWA and RoPE patterns, sigmoid gating without weight norm,
    scale 1.0) and all 21 tensor kinds, experts stacked into `ffn_*_exps`,
    BF16 matrices with F32 norms and router, bit-exact. `check_arch.py`,
    `check_metadata.py`, `check_tensors.py`; patches 0002–0006;
    [docs/gguf-conversion.md](docs/gguf-conversion.md).
-   **Model class and MoE graph.** `llama_model_kolibri` (patch 0007) loads
    every tensor incl. both sandwich norms. Top-6 selection on `logits +
    bias` with `sigmoid(logits)` weights, the ungated shared expert, and
    `ffn_moe_out + ffn_shexp` → `post_ffw_norm` → residual match the
    reference (NMSE ≤ 3.3e-14). `test-llama-archs` passes for `kolibri`;
    the vocab GGUF is regenerated under `kolibri`. `check_model.py`,
    `check_moe.py`; [docs/model.md](docs/model.md).
-   **Hybrid attention and KV cache.** The 4:1 SWA/full pattern from the
    per-layer array, `llama_kv_cache_iswa` (no hybrid memory needed),
    `LLAMA_SWA_TYPE_STANDARD` with window 513 (512 preceding + current), GQA
    48/4, per-head Q/K RMSNorm, NeoX RoPE base 10,000 on sliding layers only.
    Exact at positions 511–514 and holding at 8k, 16k, 64k and 262k tokens.
    `check_attn.py`, `check_long.py`; [docs/attention.md](docs/attention.md).
-   **Chat template and inference behavior.** The template ships in every
    GGUF and in the fork (patch 0008). llama-server renders the reference's
    prompts byte-identically for every reasoning mode and tool-call shape
    (after a `reasoning_effort: "none"` fix); `test-chat` parses reasoning
    and tool calls, including the reference's `reasoning_effort` precedence
    and tool-loop cases; generation stops on the reference's two eos tokens.
    Two parser output-splitting differences are documented. `check_chat.py`;
    [docs/chat.md](docs/chat.md).
-   **Fixtures and audit tasks.** The reference is pinned to commit
    `049a6a7bd2405b27d6d280d256bd3d585191c7ae`. `cmd/kolibri-tiny` writes
    the reference fixture shape (6 layers SSSSFF, 8 experts) and the
    `-router` (384 experts, top 6, non-zero correction bias), `-attn` and
    `-pattern` variants; `-attn` (6 layers SSSSFF, real heads and window)
    runs sliding and full/NoPE layers in one fast `check_attn.py` test.
    Sandwich norms are a first-class mapping
    requirement.
-   **Real checkpoint (structural, not numerical).**
    -   The pinned BF16 checkpoint converts in 8 minutes with a peak
        footprint of 7.5 GiB. All 903 GGUF tensors are bit-exact against the
        58,353 safetensors tensors, and all 21 metadata assertions pass.
    -   The 50-layer graph runs unquantized on the CPU.
    -   The first quantized candidate fits Metal on the 48 GB Mac with 32k
        context: Q3_K routed experts, Q8_0 elsewhere, F32 router and norms,
        33.7 GiB.
    -   `check_real.py`, `check_metadata.py --gguf`;
        [docs/real-checkpoint.md](docs/real-checkpoint.md).

## Phase 0 --- Reproducible reference environment

-   [ ] Record exact Kolibri model revision/commit.
-   [ ] Record exact `aleph-alpha-inference` and vLLM versions.
-   [ ] Run the official BF16 or FP8 reference implementation on
    suitable hardware.
-   [ ] Save a small deterministic reference corpus: token IDs, selected
    logits, generated tokens, and sampling settings.
-   [ ] Extend the deterministic corpus with tokenizer edge cases (special
    tokens, Unicode, whitespace, code, German/English text) and keep exact
    token-ID/round-trip agreement as a regression gate.
-   [ ] Use deterministic/greedy decoding for numerical validation
    before testing recommended sampling (`temperature=1.0`,
    `top_p=0.97`, `top_k=128`).
-   [ ] Record reasoning/tool-call templates separately from base text
    generation.

**Definition of Done:** We have immutable reference outputs against
which the llama.cpp port can be tested.

## Phases 3/4 --- Remaining: the real checkpoint

The converter and model class are complete on synthetic checkpoints; what
is left needs the 156 GB BF16 checkpoint.

-   [ ] Treat direct FP8 conversion as a follow-up optimization rather
    than blocking initial correctness.
-   [x] Stream tensors/shards during conversion so the full 156 GB
    checkpoint never needs to reside in RAM. (2026-10-04) — no converter
    change needed. The lazy base class and the per-layer expert buffer keep
    the conversion at "8025841608 peak memory footprint" (`/usr/bin/time -l`)
    for the 156 GB checkpoint.
-   [x] Convert the real BF16 checkpoint and run the metadata, tensor
    count, shape and dtype assertions on it. (2026-10-04) — new
    `check_real.py`: "PASS tensor set: 903 tensors, inventory 903", every
    class "data bit-exact", "PASS parameters: 78,103,074,560 in the GGUF".
    `check_metadata.py --gguf` on the real GGUF: 21 PASS, 0 FAIL.
-   [x] Build and execute the complete 50-layer unquantized graph.
    (2026-10-04) — BF16 GGUF, CPU-only (`-dev none`), greedy: "The capital of
    Germany is Berlin.", "1.45 tokens per second". Metal with `--cpu-moe`
    crashed with SIGBUS for files above the Metal working set; patch 0009
    (2026-10-05) maps only the file ranges of each backend's tensors, and
    Q8_0 and BF16 now run with `-ngl 99 --cpu-moe`, though no faster than
    CPU-only ([docs/real-checkpoint.md](docs/real-checkpoint.md)).

**Definition of Done:** A structurally correct unquantized GGUF is
produced reproducibly, and the complete 50-layer unquantized graph builds
and executes without tensor-shape or unsupported-op errors.

## Phase 6 --- Numerical validation

-   [ ] Feed identical token IDs to vLLM/reference and llama.cpp.
-   [ ] Compare embedding output.
-   [ ] Compare Q/K after QK RMSNorm.
-   [ ] Compare attention output for one SWA layer.
-   [ ] Compare attention output for one full layer.
-   [ ] Compare router logits/probabilities and selected expert IDs.
-   [ ] Report router agreement explicitly: Top-1 agreement, Top-6 set
    agreement/overlap, and the rate of selection changes at near-ties.
-   [ ] Report KL divergence for comparable output distributions alongside
    NMSE/top-token agreement, so small numerical drift can be separated from
    behavior-changing drift.
-   [ ] Repeat the expert-ID comparison on Metal: its F32 matmul uses
    half-precision tiles (router logits off by a relative RMS of about
    4e-4), so Top-6 near-ties may pick different experts than vLLM. Decide
    in Phase 8 whether the router logits need an F32 path on Metal.
-   [ ] Compare routed expert output.
-   [ ] Compare shared expert output.
-   [ ] Compare complete layer outputs. (2026-10-05) — partial: against
    `tools/ref/kolibri_ref.py`, a torch port of the vLLM model code, not vLLM.
    On wikitext-2 chunk 1 the BF16 GGUF's `l_out` NMSE rises smoothly from
    1e-7 (layer 0) to at most 0.08 as the router picks other experts at
    near-ties (same experts for 99.8% of tokens in layer 0, 64% in layer 49).
-   [ ] Compare final logits. (2026-10-05) — partial: against the torch port,
    chunk 1: "PPL 15.1787 vs 15.1244, KLD 0.033408, same top token 96.5%".
-   [ ] Compare greedy next-token sequences. (2026-10-05) — partial: against
    the torch port, "libllama greedy token along the reference continuation:
    equal at 16/16" for the German prompt.
-   [ ] Include raw German continuations in that comparison. In this
    port's BF16 (CPU), "Die Hauptstadt von Deutschland ist" continues
    greedily with " Deutschland Deutschland Deutschland …", while English
    raw prompts and German through the chat template stay coherent.
    Tokenization matches the reference (2026-10-05). (2026-10-05) — partial:
    the torch port of the vLLM model code continues the same way, in float32
    and bfloat16, and an F32 KV cache or flash attention changes nothing. Only
    vLLM itself can still contradict it.
-   [ ] Establish tolerances separately for BF16 and any FP8 reference
    run. (2026-10-05) — partial: the torch port rounded to BF16 (vLLM's
    precision) against itself in float32: "PPL 15.3272 vs 15.1244, KLD
    0.032722, same top token 93.7%". 22.3% of (token, layer) pairs have the
    6th and 7th router scores within 0.01, so BF16 tolerances must allow
    expert flips.

### Cross-implementation validation ideas

Use independent community work only as a source of test ideas, not implementation code. All pass/fail decisions remain anchored to Aleph Alpha's pinned reference.

-   [ ] Reproduce a larger fixed tokenizer case set independently and record exact token-ID agreement.
-   [ ] Add a compact router probe that records logits, Top-1, Top-6 set overlap and near-tie margins per token/layer.
-   [ ] Keep a small end-to-end corpus whose BF16 reference artifacts can be rerun after upstream llama.cpp changes.
-   [ ] For every retained quantization, emit the same summary row: perplexity, KLD, same-top-token rate, Top-1 router agreement and Top-6 overlap.

**Hard gate:** Do not diagnose quantization quality until the
unquantized implementation passes this phase.

-   [ ] Turn a compact subset of tokenizer, attention, router/expert,
    layer-output and final-logit reference cases into an automated regression
    suite that can be rerun after llama.cpp/upstream changes.

**Definition of Done:** llama.cpp BF16/F16 inference is numerically
consistent with the reference within documented tolerances, with the key
agreement metrics captured by repeatable regression tests.

## Phase 7 --- Quantization strategy

The 48 GB target makes mixed quantization more important than simply
producing a generic Q4.

-   [x] Produce Q8 first as a quantizer sanity check. (2026-10-04) —
    79,279 MiB in 40 s. On the CPU (`--no-repack`) it answers "Berlin". Against
    this port's BF16 on wikitext-2 (20 × 512): "Mean KLD: 0.049704 ± 0.004569",
    "Same top p: 92.490 ± 0.369 %".
-   [ ] Produce Q6/Q5 baselines if useful.
-   [ ] Produce Q4 variants.
-   [x] Produce IQ3/Q3 variants if Q4 lacks memory headroom. (2026-10-05) —
    Q4 lacks headroom: dry runs give Q3_K_M 35,781 MiB and IQ4_XS
    40,286 MiB against a Metal working set of 38,338 MiB. Five routed-expert
    variants with Q8_0 elsewhere: Q3_K with and without the imatrix and IQ3_S
    ("quant size = 33716.93 MiB"), IQ3_XXS ("quant size = 30341.93 MiB") and
    IQ3_XXS gate/up with IQ4_XS down ("quant size = 33904.43 MiB").
-   [x] Keep router tensors at high precision initially. (2026-10-04) —
    `ffn_gate_inp` and `exp_probs_b` stay F32 in both quantized files.
    `llama-quantize` dry run: "blk.0.ffn_gate_inp.weight ... type = f32".
-   [x] Keep normalization tensors at high precision. (2026-10-04) — all
    1D norms stay F32. `llama-quantize` dry run: "blk.0.attn_norm.weight ...
    type = f32".
-   [ ] Evaluate Q/K projections and QK norms conservatively because
    routing/attention errors can amplify across 50 layers.
-   [ ] Evaluate whether shared experts deserve higher precision than
    routed experts.
-   [x] Build an importance matrix if supported/useful for the selected
    quantization scheme. (2026-10-05) — built from Q8_0 on a 45/35/20
    English/German/code text (`tools/quant/calibration.py`), "loaded 550
    importance matrix entries ... computed on 250 chunks". It does not help
    Q3_K ("Mean KLD: 0.108158" with, 0.108174 without) but enables the IQ
    variants; the best is IQ3_XXS/IQ4_XS at "Mean KLD: 0.095291 ± 0.006905".
-   [ ] Compare perplexity/task outputs and router expert-selection
    agreement against the unquantized model. (2026-10-04) — partial:
    perplexity and KLD are measured against this port's BF16 for all five
    3-bit variants (0.095 to 0.108 mean KLD, 88.0 to 88.5% same top token; the
    table is in docs/real-checkpoint.md). For the Q3 mix on Metal: "Mean KLD:
    0.108174 ± 0.007597", "Same top p: 88.431 ± 0.448 %".
    Router agreement, task outputs and any judgment of quality wait for the
    Phase 6 gate.
-   [ ] Specifically measure how often quantization changes Top-6 expert
    selection, using the same Top-1/Top-6 agreement metrics as Phase 6.
-   [ ] Run the same fixed evaluation corpus across BF16 and each retained
    Q8/Q6/Q5/Q4/Q3 candidate and record KLD, top-token agreement, router
    agreement and perplexity/task outputs in one comparable matrix.
-   [ ] Explain why Q8_0 against BF16 reaches KLD 0.050 with 92.5% same top
    token on the same backend (CPU), about 50 times a dense model's Q8_0.
    The suspect is Top-6 flips at router near-ties. Investigate after
    Phase 6.

**Definition of Done:** At least one quantization retains acceptable
behavior and runs stably through Metal.

## Phase 8 --- 48 GB Apple Silicon target

A raw 4-bit estimate for 78.1B parameters is about 39.1 GB before
quantization metadata, higher-precision tensors, runtime buffers, KV
cache, and macOS. Q4 is therefore a boundary case, not a guaranteed fit.

For each candidate:

-   [x] Measure GGUF file size. (2026-10-05) — `llama-quantize` "quant
    size" for every candidate: Q3_K and IQ3_S 33,717 MiB, IQ3_XXS 30,342 MiB,
    IQ3_XXS/IQ4_XS 33,904 MiB.
-   [ ] Measure actual unified-memory use after load. (2026-10-04) —
    partial, Q3 mix at 32k: "MTL0 ... 34384 = 33384 + 740 + 260", host 418 MiB.
    IQ3_XXS/IQ4_XS at 32k (2026-10-05): "MTL0 ... 34904 = 33904 + 740 + 260",
    host 375 MiB. IQ3_S and IQ3_XXS remain.
-   [ ] Measure Metal buffers/runtime overhead. (2026-10-04) — partial,
    Q3 mix: compute buffer 260 MiB on MTL0 and 43 MiB on CPU.
-   [ ] Measure SWA and full-attention KV-cache memory separately.
    (2026-10-04) — partial, Q3 mix at 32k: full attention "640.00 MiB (32768
    cells, 10 layers)", SWA "100.00 MiB (1280 cells, 40 layers)".
-   [x] Measure prompt-processing tokens/s. (2026-10-05) — `llama-bench
    -ngl 99 -p 512 -r 3` on Metal: Q3 mix "pp512 | 1223.02 ± 10.95",
    IQ3_S 1331.99, IQ3_XXS 1367.29, IQ3_XXS/IQ4_XS "pp512 | 1350.55 ± 7.73".
-   [x] Measure generation tokens/s. (2026-10-05) — `llama-bench -ngl 99
    -n 128 -r 3` on Metal: Q3 mix "tg128 | 60.67 ± 0.08", IQ3_S 61.78,
    IQ3_XXS 63.03, IQ3_XXS/IQ4_XS "tg128 | 64.00 ± 0.07". `llama-server` chat
    on the Q3 mix: 58.7 tokens/s.
-   [ ] Measure expert-routing overhead.
-   [ ] Test 8k, 16k, and 32k contexts first.
-   [ ] Expand context only if memory headroom permits.
-   [ ] Test sustained generation and memory pressure.
-   [ ] Compare quality and Top-6 routing agreement with BF16.
-   [ ] Benchmark resident Metal inference against streamed/offloaded expert
    execution using the same quantization, prompt corpus and context lengths.
    Record prompt-processing and generation tokens/s, peak unified memory,
    host/Metal buffer use, bytes transferred per generated token and, where
    measurable, expert-cache hit rate. This should distinguish the cost of
    keeping the quantized experts resident from exploiting Kolibri's sparse
    Top-6-of-384 expert activation.


**Target:** Prefer a configuration that leaves several GB of unified
memory headroom rather than merely loading successfully.

**Expected decision:** Q4 may fit only narrowly; an IQ3/Q3-style
quantization is likely to provide substantially safer headroom on a 48
GB Mac.

## Phase 9 --- User-facing inference behavior

Template, reasoning modes, tool calls, parsers and stop tokens are done
(see *Done so far*).

-   [ ] Only after base greedy inference is correct, validate Aleph
    Alpha's recommended sampling parameters.
-   [ ] Decide whether to propose the `reasoning_effort: "none"` fix
    (one line in llama-server's `server-common.cpp`) upstream.
-   [ ] Test `continue_final_message` with thinking; the reference parser
    itself has a known gap there.

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

- [ ] Treat BF16 as the initial source of truth; postpone direct FP8-source support until BF16 logits match.
- [ ] Report the upstream llama.cpp issues found along the way (invalid UTF-8 aborting `llama_tokenize`, overlong decoding, two test-tooling bugs, Metal `--cpu-moe` SIGBUS above the working set, fixed here by patch 0009); see README "Known upstream issues", [docs/tokenizer.md](docs/tokenizer.md) and [docs/real-checkpoint.md](docs/real-checkpoint.md).

## First actionable milestone

The first major milestone remains:

> Produce an unquantized Kolibri GGUF that runs in llama.cpp and matches
> the reference implementation at tokenizer, router/expert selection,
> layer-output, logits, and greedy next-token levels.

Only after that gate should the project optimize Q4/IQ3 for the 48 GB
Mac.
