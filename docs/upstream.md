# Upstream llama.cpp issues

Three issues this port ran into belong to ggml-org/llama.cpp, not to the
Kolibri patches. This page checks that each still exists on current upstream
master, links the upstream reports that already exist, and keeps the repro
runnable. `tools/upstream/repro.py` exits 1 as soon as a check no longer
reproduces, so an upstream fix shows up as a red run.

The last section compares upstream's own Kolibri model (#30673) with this
port's, for Phase 11, Stage E0.

Upstream's CONTRIBUTING.md forbids AI-written posts ("It is strictly
prohibited to use AI to write your posts for you (bug reports, feature
requests, pull request descriptions, …)"), and its AGENTS.md forbids agents
from opening PRs. This page therefore collects evidence only. Any upstream
report, comment or PR is still to be written by hand.

## Setup

Upstream master `988190680d5a89fce97de3c20df2c2813731fd61` (2026-10-07,
97 commits after the pinned `1537a0a8`), in a worktree outside the repo, with
the same build options as the fork:

```sh
git -C third_party/llama.cpp fetch https://github.com/ggml-org/llama.cpp master
git -C third_party/llama.cpp worktree add --detach /tmp/upstream FETCH_HEAD
cmake -S /tmp/upstream -B /tmp/upstream/build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_METAL=ON -DLLAMA_OPENSSL=OFF
cmake --build /tmp/upstream/build --target llama llama-completion -j
```

The first two checks run on plain master with upstream's own vocab
`models/ggml-vocab-qwen2.gguf`, so they need nothing from Kolibri. The third
needs a GGUF larger than the Metal working set and a tree that loads it. For
Kolibri that is master plus the runtime parts of patches 0001–0008. All
runtime files apply; only the converter's Python files and
`tests/test-llama-archs.cpp` conflict, and loading a GGUF uses neither:

```sh
git -C /tmp/upstream apply '--exclude=conversion/*' --exclude=convert_hf_to_gguf_update.py \
  --exclude=tests/test-llama-archs.cpp "$PWD"/patches/llama.cpp/000[1-8]-*.patch
```

## 1. Invalid UTF-8 in `llama_tokenize`

```sh
.venv/bin/python tools/upstream/repro.py utf8 --llama-cpp /tmp/upstream
```

```text
libc++abi: terminating due to uncaught exception of type std::invalid_argument: invalid codepoint
REPRODUCED utf8 abort: F4 90 80 80 (U+110000) killed the process with signal 6
REPRODUCED utf8 overlong: C0 AF gives [14], '/' gives [14]
REPRODUCED utf8 surrogate: ED A0 80 gives [169, 63219], detokenized b'\xed\xa0\x80'
```

The cause is unchanged on master:
- **No validation:** `unicode_cpt_from_utf8` (`src/unicode.cpp`) checks
  neither overlong forms nor surrogates nor code points above U+10FFFF.
- **The abort:** `unicode_cpt_to_utf8` throws
  `std::invalid_argument("invalid codepoint")` for such a code point, and
  nothing between it and the C API catches it.

The behavior with the Kolibri vocab, compared with the reference, is in
[tokenizer.md](tokenizer.md), "Invalid UTF-8".

**Upstream:** the abort is
[ggml-org/llama.cpp#29713](https://github.com/ggml-org/llama.cpp/issues/29713)
(input `F6 9F A0 A0 A0 A0`, Linux), closed as not planned on 2026-09-30
without a comment. Overlong forms and surrogates have no report.

**Partly fixed since (2026-10-23).** On master `184f6536`, the abort and the
overlong form no longer reproduce; [#30485](https://github.com/ggml-org/llama.cpp/pull/30485)
("unicode : reject overlong UTF-8 and out-of-range codepoints", 2026-10-15)
fixed both. Surrogates still pass through:

```text
NOT REPRODUCED utf8 abort: F4 90 80 80 (U+110000) returned [23479]
NOT REPRODUCED utf8 overlong: C0 AF gives [9973], '/' gives [14]
REPRODUCED utf8 surrogate: ED A0 80 gives [169, 63219], detokenized b'\xed\xa0\x80'
```

`repro.py utf8` therefore exits 1 on current master, as intended for a fix.

## 2. `tests/test-tokenizer-random.py`

```sh
.venv/bin/python tools/upstream/repro.py tokenizer-random --llama-cpp /tmp/upstream
```

```text
REPRODUCED test-tokenizer-random LibLlamaModel: tokenize('Hello') raised TypeError: initializer for ctype 'struct llama_vocab *' must be a pointer to same type, not cdata 'struct llama_model *'
REPRODUCED test-tokenizer-random TokenizerGroundtruth vocab: transformers 5.18.0: 1 entry for 127998 tokens
REPRODUCED test-tokenizer-random TokenizerGroundtruth added_tokens: transformers 5.18.0: 1 entry for 98 added tokens
```

- **`LibLlamaModel`:** its `tokenize` and `detokenize` pass `self.model`
  where `llama_tokenize`/`llama_detokenize` take a `llama_vocab *`. The
  script therefore fails at its first call.
- **`TokenizerGroundtruth`:** builds `vocab` and `added_tokens` with
  `batch_decode` of a flat ID list. Under transformers 5 that returns a
  single string, so the vocab and added-token fuzzers each test one
  concatenated string. The check uses the pinned Kolibri tokenizer; the
  cause is the transformers call, not the model. `vocab` and
  `added_tokens` are separate checks, so a fix of only one of them shows.

**Upstream:** no report. The open
[#10276](https://github.com/ggml-org/llama.cpp/issues/10276) is about the
same script but asks for something else.

## 3. Metal with `--cpu-moe` above the working set

```sh
.venv/bin/python tools/upstream/repro.py cpu-moe --llama-cpp /tmp/upstream \
  --gguf ~/models/Kolibri-1-Q8_0.gguf
```

It runs `llama-completion -ngl 99 --cpu-moe --no-repack -c 512 -n 1 -v`.
Four runs on master plus 0001–0008, all rc 0:

```text
REPRODUCED cpu-moe: GGUF 79,284 MiB, Metal maps 79,279 MiB of it: killed by signal 10
REPRODUCED cpu-moe: GGUF 79,284 MiB, Metal maps 79,279 MiB of it: killed by signal 10, Metal out of memory
REPRODUCED cpu-moe: GGUF 79,284 MiB, Metal maps 79,279 MiB of it: exit 1, Metal out of memory
REPRODUCED cpu-moe: GGUF 79,284 MiB, Metal maps 79,279 MiB of it: killed by signal 10, Metal out of memory
```

Earlier runs with a first version of the check gave the same split: two
exits after the Metal OOM and one SIGBUS in three runs.

**The cause is unchanged.** `llama_model_loader::get_mapping_range` still
takes one range per file and backend, from the first offloaded tensor to the
last. With `--cpu-moe` the routed experts lie between them, so Metal maps the
whole file ("MTL0_Mapped model buffer size = 79279.43 MiB"). The recommended
working set is 40,200.90 MB.

**How it ends varies between runs.** Either the warmup decode fails after
"Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
and the process exits 1, or the CPU expert matmul dies with SIGBUS. The
check counts both as reproduced. At the pinned commit the documented runs
ended in SIGBUS ([real-checkpoint.md](real-checkpoint.md), "Streaming the
routed experts with `--cpu-moe`").

**Master's own hint.** Master now warns "tensor overrides to CPU are used
with mmap enabled - consider using --load-mode none for better performance".
`--help` lists `mmap` ("memory-map model") as a separate mode, so `none`
does not stream the experts from the mapped file. It was not measured here.

**Patch 0009 still fixes it on master.** It applies alone to unmodified
master (`git apply --check`). On master plus 0001–0009, Metal maps only the
ranges that hold its tensors:

```text
RUNS cpu-moe: GGUF 79,284 MiB, Metal maps 2,447 MiB of it: exit 0
```

**Upstream:**
- [#24510](https://github.com/ggml-org/llama.cpp/issues/24510) describes the
  first-to-last mapping (`-ngl` maps the whole file when the offloaded
  tensors are not contiguous).
- [#27822](https://github.com/ggml-org/llama.cpp/issues/27822) reports the
  crash in `ggml_compute_forward_mul_mat_id` after the Metal OOM, with
  `--cpu-moe`.

Both are closed as not planned. Patch 0009 has not been proposed upstream.
On the fork it is [CWBudde/llama.cpp#10](https://github.com/CWBudde/llama.cpp/pull/10).

**Master `184f6536` fixes it without 0009 (2026-10-23).** Master now merges
a context's mapping ranges (`merge_ranges`, `llama-model-loader.cpp:1460`),
and on unmodified master with the rewritten Q8_0 file Metal maps only its own
tensors:

```sh
.venv/bin/python tools/upstream/repro.py cpu-moe --llama-cpp UPSTREAM \
  --gguf ~/models/Kolibri-1-Q8_0.gguf --expect ok
```

```text
RUNS cpu-moe: GGUF 79,284 MiB, Metal maps 2,448 MiB of it: exit 0
```

0009 is retired with the rebase onto this commit
([The rebase](#the-rebase-stage-e01)).

## Negative controls

Each was applied to the upstream worktree, run, and reverted.

| Mutation | Result |
|---|---|
| M23: `unicode_cpt_to_utf8` returns U+FFFD instead of throwing | `utf8`: "NOT REPRODUCED utf8 abort: F4 90 80 80 (U+110000) returned [5691]", rc 1 |
| M24: `test-tokenizer-random.py` passes `llama_model_get_vocab(self.model)` | `tokenizer-random`: "NOT REPRODUCED test-tokenizer-random LibLlamaModel: tokenize('Hello') returned [9707]", rc 1 |
| M25: `cpu-moe` without `--expect ok` on master plus 0001–0009 | "NOT REPRODUCED cpu-moe: GGUF 79,284 MiB, Metal maps 2,447 MiB of it: exit 0", rc 1 |
| M26: `test-tokenizer-random.py` decodes the added tokens one by one (only `vocab` still collapses) | `tokenizer-random`: "NOT REPRODUCED test-tokenizer-random TokenizerGroundtruth added_tokens: transformers 5.18.0: 98 entries for 98 added tokens", rc 1 |

## Upstream's Kolibri model (Phase 11, Stage E0)

Upstream added Kolibri-1 in
[#30673](https://github.com/ggml-org/llama.cpp/pull/30673) (merge
`62318cd5`, 2026-10-17). Stage E0 asks whether Phase 11 can build on it
instead of this port's model patches. It was compared with the fork on master
`184f6536` (2026-10-23), built unmodified as in [Setup](#setup), plus the
targets `llama-perplexity` and `test-tokenizer-0`.

**Verdict: upstream's model matches ours.** The tiny-checkpoint gates pass
with logits bit-identical to the fork's. On the CPU, the real BF16 GGUF gives
the fork's recorded bits at every captured node of all six e2e cases, and
IQ3 gives the same KLD lines. Only Metal differs, by a backend change between
our pinned commit and master, not by the model. Phase 11 can build on
upstream's Kolibri; retiring the covered patches is the rest of E0's first
item.

### Differences in the GGUF

Upstream names the architecture `kolibri1`, so every arch key is
`kolibri1.*`. Converting the tiny checkpoint with both converters gives the
same 111 tensors with bit-identical data. The metadata differs as follows:

| Key | Upstream | This port |
|---|---|---|
| `general.architecture` | `kolibri1` | `kolibri` |
| `tokenizer.ggml.pre` | `qwen2` | `kolibri` (patch 0001 maps it to the same `QWEN2` pre-type) |
| `<arch>.attention.rope_pattern` | absent; RoPE on exactly the sliding layers (`is_swa`) | written, equal to `sliding_window_pattern` in every GGUF here |
| `<arch>.expert_weights_scale` | absent (fixed 1.0) | 1.0 |
| `<arch>.feed_forward_length`, `<arch>.vocab_size` | written | absent; neither is read by either model |
| `tokenizer.ggml.add_space_prefix`, `tokenizer.ggml.clean_spaces` | `false`, `false` | absent; the pre-type's defaults are the same |

Our GGUFs therefore do not load in upstream as they are. Reconverting is not
possible here: the safetensors were deleted, and the disk has 18 GiB free.
`cmd/kolibri-rekey` rewrites only the header instead:
- the architecture and key prefix become `kolibri1`, and the pre-tokenizer
  becomes `qwen2`;
- `rope_pattern` is dropped, after the tool checks that it equals the
  sliding-window pattern;
- a padding string keeps the header's length.

The output is an APFS clone (`cp -c`) whose header is overwritten in place,
so the tensor data stays shared with the original. Both real files together
cost 9.5 MB:

```sh
go run ./cmd/kolibri-rekey -in ~/models/Kolibri-1-BF16.gguf -out ~/models/Kolibri-1-BF16-kolibri1.gguf
go run ./cmd/kolibri-rekey -in ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf \
  -out ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx-kolibri1.gguf
```

```text
padding: general.rekey.padding of 28 bytes, then 31 zero bytes
header ends at 4887041, data section at 4887072 (unchanged)
padding: general.rekey.padding of 39 bytes, then 31 zero bytes
header ends at 4887329, data section at 4887360 (unchanged)
```

GGUFs from upstream's own converter carry `qwen2` and the sigmoid gating key
(`expert_gating_func` 2). They need no `--override-kv`, contrary to HF
discussion #3 about Hub files; Hub files made with other converters were not
checked.

### The gate: tiny checkpoints

Gate: the existing tolerances.
- `compare_real.py --tiny`: NMSE 1e-6 against the float64 torch reference,
  with at least 98% of (token, layer) pairs routed alike.
- `check_model.py`: NMSE 1e-4 between ubatches and devices, at most 2 of 100
  positions above it with routing.

Each fixture runs twice on upstream:
- (a) converted by upstream's converter;
- (b) converted by ours and rekeyed (`--convert-llama-cpp third_party/llama.cpp --rekey`).

`--save-logits` keeps the logits for the bitwise comparison with the fork.

```sh
.venv/bin/python tools/ref/compare_real.py --llama-cpp /tmp/upstream --tiny [--preset router|attn|pattern] \
  [--convert-llama-cpp third_party/llama.cpp --rekey] --save-logits DIR
.venv/bin/python tools/gguf/check_model.py --llama-cpp /tmp/upstream --no-wiring \
  [--convert-llama-cpp third_party/llama.cpp --rekey]
```

| Fixture | Fork | Upstream (a) | Upstream (b) | Logits against the fork |
|---|---|---|---|---|
| default: 8 experts, top 2, layers SSSSFF | PASS, logits NMSE 4.75e-13 | PASS, 4.75e-13 | PASS | bit-identical (all experts and top-2) |
| `attn`: real heads, window | PASS, 3.45e-12 | PASS, 3.45e-12 | PASS | bit-identical (both runs) |
| `pattern`: real 50-layer SWA/full pattern | PASS, 9.18e-13 | PASS, 9.18e-13 | PASS | bit-identical (both runs) |
| `router`: 384 experts, top 6, real bias scale | PASS, 100.0% same experts, router NMSE 4.29e-13 | PASS, the same | PASS | bit-identical (top-6) |
| `check_model.py`: CPU, Metal, ubatch 16 vs 100, BF16 and F32 | PASS | PASS, same values (MTL0 vs CPU top-2: 99/100) | PASS, same values | — |

- **The router preset runs only the top-6 pass.** With all 384 experts
  active, the graph outgrows libllama's graph context, in the fork as well.
  `check_moe.py` checks the routing semantics on that fixture against the
  reference routing function, and the fork passes it. Upstream's logits are
  bit-identical to the fork's on the same fixture, so upstream gets the same
  router:
  - the bias is added to the raw logits before top-6;
  - the weights are the unbiased sigmoid;
  - there is no renormalization and no scale.
- **`check_model.py`'s graph-wiring check reads this port's node names, so
  `--no-wiring` skips it.** Upstream names the shared expert output
  `ffn_shexp_out` and leaves the routed-plus-shared sum unnamed.
  `compare_real.py` reads `ffn_shexp_out` as `ffn_shexp`. Since the rebase
  the wiring check reads upstream's names, and `--no-wiring` is gone.
- **The RoPE/NoPE split and the window of 513 are covered by the same
  bit-identical runs.** The `pattern` fixture has the real 50-layer pattern,
  and `attn` has the real heads and window; the real model is covered by the
  e2e case `de-long`, 1,024 tokens past the window.

### Tokenizer

`tools/tokenizer/compare.py` on upstream gives identical results with the
GGUF from upstream's converter and with ours rekeyed (`pre = qwen2`):
- golden 85/85;
- every fuzz generator, for example `ascii_lr_strip` 442,368/442,368 and
  `vocab_words` 127,193/127,193;
- the corpora `wiki.test.raw` 4,212/4,212 and `kolibri-calibration.txt`
  6,461/6,461, which match the pinned sha256 of the reference IDs.

Invalid UTF-8 (report only): upstream matches the reference on 10 of 14
cases, the fork on 8 of 14, and the fork still aborts on U+110000. That is
issue 1's partial fix.

### The real checkpoint (INFO)

**BF16, CPU, `e2e.py`, 13 min.**
- Upstream on the rekeyed BF16 reports "unchanged" for all six cases: every
  captured node and the logits have the sha256 of the fork's recorded run.
- The lines against the float32 reference are therefore the fork's.
- For example, de-long: "PPL 26.3648 vs 26.5145, logits NMSE 1.65e-03, KLD
  0.046278, same top token 95.1%; router Top-1 same 97.32%, Top-6 set same
  85.55%".

```sh
.venv/bin/python tools/ref/e2e.py --llama-cpp /tmp/upstream --gguf ~/models/Kolibri-1-BF16-kolibri1.gguf
```

```text
INFO de-long, not gated (recorded with Kolibri-1-BF16.gguf, this run Kolibri-1-BF16-kolibri1.gguf): unchanged
```

**IQ3_XXS/IQ4_XS, KLD against our BF16** (`wiki.test.raw`, 20 × 512,
`llama-perplexity --kl-divergence-base ~/models/eval/kld-bf16-c512-n20.bin --kl-divergence -c 512 --chunks 20`):

| Build | Device | Mean KLD | Same top token | PPL(Q)/PPL(base) |
|---|---|--:|--:|--:|
| fork `128787f8` | Metal `-ngl 99` | 0.0953 ± 0.0069 | 88.47 ± 0.45% | 1.0161 |
| upstream `184f6536` | Metal `-ngl 99` | 0.0998 ± 0.0073 | 88.65 ± 0.44% | 1.0208 |
| fork | CPU `-dev none -t 10` | 0.1014 ± 0.0071 | 88.04 ± 0.45% | — |
| upstream | CPU `-dev none -t 10` | 0.1014 ± 0.0071 | 88.04 ± 0.45% | — |

- **On the CPU, the per-chunk lines are identical.** On Metal they differ.
- **The model code is the same.** The difference comes with the backend:
  master's Metal kernels and fusions are 20 days newer than our pinned commit.
- **All four results lie within one standard error of each other.** The
  fork's Metal value reproduces the recorded 0.095 of
  [real-checkpoint.md](real-checkpoint.md).

`compare_vllm.py` did not run: its captures were under `/tmp` and are gone.
The tiny gates above go against the torch reference, which matched vLLM on
the tiny checkpoint and on real-weight slices
([vllm-reference.md](vllm-reference.md)). The slices were not rerun: the full
BF16 checkpoint already gives the fork's bits at every captured node.

### Which fork patches upstream covers

| Patch | Covered | Evidence or what remains |
|---|---|---|
| 0001 tokenizer | yes | `qwen2` pre-type; the tokenizer harness is identical. Our `kolibri` pre name and `ggml-vocab-kolibri.gguf.{inp,out}` go. |
| 0002 arch | yes, as `kolibri1` | every GGUF and tool here that says `kolibri` must move to `kolibri1` (`cmd/kolibri-rekey` for the files) |
| 0003–0006 converter | yes | identical tensors; the key differences above |
| 0007 model | yes | bit-identical logits, tiny and real, on the CPU |
| 0008 chat template | no | upstream has neither the template file and tests nor the `server-common.cpp` line that passes `enable_thinking=false` for `reasoning_effort: none` (master `server-common.cpp:1495`) |
| 0009 mmap ranges | yes | master merges mapping ranges (`merge_ranges`, `llama-model-loader.cpp:1460`): "RUNS cpu-moe: GGUF 79,284 MiB, Metal maps 2,448 MiB of it: exit 0" on unmodified master |
| 0010 continuation tests | no | tests on top of 0008 |
| 0011, 0012 expert cache | no | the fork's own work, rebased (below) |

### Controls

| Mutation | Result |
|---|---|
| M43a: upstream with `kolibri1.expert_weights_norm=true` for libllama only (`compare_real.py --tiny --override-kv`) | "FAIL tiny, all 8 experts: logits NMSE 5.08e-01", "FAIL tiny, top-2 routing: same experts for 97.0% of (token, layer) pairs", "FAIL 12 problem(s)", rc 1 |
| M43b: a rekeyed GGUF with `tokenizer.ggml.pre = llama-bpe` (`kolibri-rekey -pre llama-bpe`) | "golden: 82/85 ok", for example "FAIL english/apostrophes", rc 1 |

### The rebase (Stage E0.1)

The fork moved onto master `184f6536`, the commit the comparison above ran
on, so its verdict still applies. Fork branch `feat/kolibri1` holds 0008,
0010, 0011 and 0012 as 9 commits; the older `feat/kolibri*` branches stay as
they were. Patches 0001–0007 and 0009 are retired, and the regenerated series
applied to a clean `184f6536` gives exactly the fork's tree (`7ade40f0`).

**What moved:**
- The model hooks of 0011 moved from `src/models/kolibri.cpp` to upstream's
  `src/models/kolibri1.cpp` (lazy expert tensors, the cache in the graph).
  `build_moe_ffn` now also has upstream's MoE cache (#29887); the fork's
  expert cache maps the router's ids to its slots before it, and the two
  cannot be on together (an assert).
- 0008's tests follow master's chat API: prompts are `common_chat_prompt`,
  and `oaicompat_chat_params_parse` takes a vocab and a session.
- The local GGUFs were rewritten with `cmd/kolibri-rekey` under their old
  names; the old-architecture files keep a `-kolibri` suffix and share
  their data blocks. Nothing records a file's hash, and the expert sizes
  in `testdata/locality/expert-bytes.json` still verify.
- The checks expect upstream's GGUF keys and graph: `check_arch.py` and
  `check_metadata.py` check `kolibri1` (with `tokenizer.ggml.pre = qwen2`,
  and no `expert_weights_scale` or `rope_pattern`); `check_model.py` and
  `check_moe.py` read `ffn_shexp_out`, and find the unnamed routed-plus-shared
  sum and post-FFN norm through `l_out`.

**Checks on the rebased build:**

| Check | Result |
|---|---|
| tokenizer `compare.py` | "golden: 85/85 ok", every fuzz and corpus case equal, UTF-8 report 10/14 |
| `check_arch`, `check_metadata`, `check_tensors`, `check_model`, `check_moe`, `check_attn` | all PASS; `check_model` "graph wiring: 6 layers x 9 node checks", MTL0 vs CPU top-2 99/100 as in E0 |
| `compare_real.py --tiny`, all four fixtures | PASS with E0's values (logits NMSE 4.75e-13, 3.45e-12, 9.18e-13; router 4.29e-13); the saved logits byte-identical to unmodified master |
| `check_chat.py` | all PASS |
| ctest `test-expert-cache`, `test-chat`, `test-chat-template`, `test-chat-peg-parser`, `test-chat-auto-parser`, `test-llama-archs` | 6/6; the Markov fixture "as the Go predictor" |
| `check_expert_cache.py`, tiny | 164 PASS, rc 0 |
| `check_expert_cache.py --real` IQ3, 128 and 64 slots | "PASS: 0 failures" |
| `repro.py cpu-moe --expect ok`, Q8_0 | "RUNS cpu-moe: GGUF 79,284 MiB, Metal maps 2,448 MiB of it: exit 0" |

**The expert cache on Metal: same partition, same bits.** The cache's CPU
remap splits the Metal graph after each layer's `ffn_moe_topk`. Master's Metal
backend fuses ops differently with those splits, so against an unsplit
resident run the cache's logits are no longer bit-identical on Metal: up to
6e-4 on the tiny F32 fixture, 0.134 on Q8_0, and 3.23 on the real IQ3 file
(160 positions; a near-tie routed differently compounds over 50 layers). On
the CPU they still are.
- `GGML_METAL_FUSION_DISABLE=1` removes the difference, and so does skipping
  only the `RMS_NORM + MUL + ADD` fusion. Both runs fuse the same number of
  ops.
- A resident run that cb_eval splits at the same points is bit-identical to
  the cache on every fixture, and the cache is deterministic.
- So `check_expert_cache.py` compares the GPU against a resident run split
  after each `ffn_moe_topk` (`Runner.logits(split_after=...)`) and reports the
  difference to the unsplit run as info. The CPU still compares against the
  plain resident run.
- E2 removes the CPU split. Its gate can then require bit identity to the
  unsplit run again. Until then, `kolibri-bench`'s exact equivalence of
  `iq3-cache128` against `iq3-resident-ub21` compares differently fused
  graphs on master.

**Controls:**

| Mutation | Result |
|---|---|
| M44: the rebased build loads an un-rewritten `kolibri` GGUF | "error loading model: unknown model architecture: 'kolibri'", rc 1 |
| M45: the cache reads each odd expert's down rows from its neighbour (`e ^ 1`) | every bit-identity line fails, CPU and Metal (the split reference included); rc 1 |

## Upstream's expert caches on this Mac (Phase 11, Stage E0.2)

Two upstream paths cache experts. E0.2 asks whether they run on this Mac with
Kolibri-1 and how they compare with the fork's cache. #25294's result also
decides Phase 11's stop rule ("if #25294 matches or beats the fork's cache on
Kolibri, move the remaining work onto its design").

### #29887: `--moe-cache-mib` on Metal

Merged on master `184f6536` (`src/llama-moe-cache.cpp`). It accepts GPU and
IGPU devices; Metal (`MTL0`) reports itself as a GPU. It caches only layers
whose experts sit in a host buffer, so on Metal it needs `--cpu-moe`:

- With `--cpu-moe --moe-cache-mib 10375` (the bytes of 128 IQ3 experts per
  layer, as much memory as `--expert-cache 128`) it is on: "MTL0 MoE cache
  size = 10375.00 MiB for 31125.00 MiB of host experts". Over 48 decode tokens:
  "hits = 12472, misses = 3986, hit rate = 75.78%, uploaded = 6461.68 MiB",
  and the same 48 tokens as `--cpu-moe`. The server logs these lines only
  with `-v`.
- **Control M47:** without `--cpu-moe` it logs "no layer has all of its experts
  in host memory, MoE cache is disabled".

On unified memory the upload is a copy from the page cache into another
buffer of the same RAM, and the benchmark below shows it slower than plain
`--cpu-moe`.

### #25294: `--moe-stream` on Kolibri

The PR (head `1248fd8fa`, July 2026) predates kolibri1 (#30673) and no longer
merges. It was merged onto `184f6536` as a reference build, fork branch
[`ref/pr25294-on-184f6536`](https://github.com/CWBudde/llama.cpp/tree/ref/pr25294-on-184f6536)
(`8a8ddf827`). It is not a fork patch and not for upstream.

**The port.** The PR's logic is unchanged. Only master's APIs are adapted
(listed in the merge commit):
- `load_mode` replaced `use_mmap`, so `--moe-stream` now turns an mmap load
  mode into `NONE`;
- `n_expert_used` is per layer now, so the PR reads `n_expert_used_max()`;
- `TENSOR_STREAMED` moved to `1 << 6`;
- `build_lora_mm_id` takes both #29887's slots and the PR's `ids_scale`, with
  an assert that the two caches are never on together.

The PR's hooks are generic (the `ffn_*_exps` tensors, `build_moe_ffn`), so
kolibri1 needs no model code.

Smoke on the tiny 384-expert top-6 fixture: on the CPU the logits are
bit-identical to plain master, both with waves (64 slots, a 32-token ubatch)
and in a single wave (200 slots). On Metal, streaming differs by 4.2e-4: its
CPU remap splits the graph, as the fork's does (see "The rebase").

**Auto-sizing aborts on Kolibri, as reported.**
- Without a slot count, `llama_moe_stream_resolve_slots` takes
  `clamp(2*n_expert_used, 16, n_expert)`, which is 16 slots.
- A ubatch that touches more experts than that runs in waves, and waves need
  `3*n_expert_used` = 18 slots.
- So plain `--moe-stream` aborts while reserving the graph, before any request:
  "src/llama-graph.cpp:2394: MoE expert streaming: multi-pass expert GEMMs need
  an expert cache of at least 3*n_expert_used slots (have 16, need 18);
  increase --moe-stream-cache or reduce -ub", rc 134.
- Top-5 or fewer fits in 16 slots; Kolibri's top-6 is the first that does not.
- **Control M46:** `--moe-stream-cache 17s` aborts the same way ("have 17,
  need 18"), and `18s` starts.

With `--moe-stream-cache 128s`, 48 greedy tokens equal those of the fork's
`--cpu-moe`.

### The benchmark

`testdata/bench/e0.json`:
- Stage D's workloads, 2 repetitions, IQ3, 2026-10-23, on AC power.
- The fork's arms run the rebased fork; `iq3-moestream128` runs the port
  (`kolibri-bench -servers pr25294=<port>/build/bin/llama-server`).
- Every cache holds the same memory, 128 experts per layer.
- The gates were fixed before the runs.

| Arm | Ceiling | Warm decode t/s | Warm p95 ms | Warm prompt t/s | Cold TTFT s | Footprint GiB | Page cache peak GiB |
|---|--:|--:|--:|--:|--:|--:|--:|
| `iq3-cpu-moe` | 0 | 16.63 (15.87–17.40) | 79.3 | 218.7 | 25.70 | 1.0 | 30.8 |
| `iq3-cache128` (fork, `F_NOCACHE`) | 0 | 17.71 (17.19–18.23) | 87.3 | 81.5 | 35.05 | 11.0 | 2.7 |
| `iq3-cache128-buffered` (fork) | 0 | 20.52 (19.81–21.22) | 69.7 | 110.0 | 23.77 | 11.0 | 18.5 |
| `iq3-moecache` (#29887) | 0 | 11.99 (11.22–12.75) | 174.7 | 71.8 | 97.54 | 11.0 | 21.5 |
| `iq3-moestream128` (#25294) | 0 | **20.86** (20.42–21.30) | **65.4** | **283.2** | **8.62** | 14.0 | 17.9 |
| `iq3-cpu-moe` | 16 | 5.89 (5.33–6.46) | 649.5 | 11.4 | 217.07 | 1.1 | 16.8 |
| `iq3-cache128` (fork, `F_NOCACHE`) | 16 | **17.96** (17.60–18.33) | **80.7** | 79.0 | 35.50 | 11.0 | 2.7 |
| `iq3-cache128-buffered` (fork) | 16 | 13.47 (12.15–14.79) | 117.6 | 53.8 | 46.30 | 11.0 | 6.2 |
| `iq3-moestream128` (#25294) | 16 | 12.69 (11.93–13.45) | 124.2 | **165.9** | **15.43** | 14.0 | 3.5 |

**Gates (the fork's cache must beat #25294):**
- 0 GiB, `F_NOCACHE`: **not passed** (17.71 against 20.86 t/s, p95 87.3 against 65.4 ms).
- 0 GiB, buffered: **not passed** (20.52 against 20.86 t/s, the ranges overlap; p95 69.7 against 65.4 ms).
- 16 GiB, `F_NOCACHE`: **passed** (17.96 against 12.69 t/s, p95 80.7 against 124.2 ms).
- 16 GiB, buffered: not passed (13.47 against 12.69, the ranges overlap).

**Phase 11's stop rule fires at 0 GiB.** There #25294 matches the fork's best
decode, and its prefill is 2.6× faster (283 against 110 t/s warm, cold TTFT
8.6 against 23.8 s). This is the wave prefill that E4 planned to build: the
port runs the default 512-token ubatch, while the fork's cache caps it at 21.

**At the 16 GiB ceiling the fork still wins decode.** Its `F_NOCACHE` slots
hold, while #25294's buffered reads lose their page cache to the hog
(macOS has no `O_DIRECT`). #25294 still leads prefill there (166 against
79 t/s). It also needs more memory: a 14.0 GiB footprint against 11.0, likely because
it turns mmap off, so the non-expert weights count in its own memory (not
measured separately).

**#29887 loses to plain `--cpu-moe`** (11.99 against 16.63 t/s, p95 175
against 79 ms): 285 command-buffer commits per token against 103. It answers
a host-to-VRAM problem that unified memory does not have.

**Info and checks:**
- Equivalence (INFO, against `iq3-cpu-moe`): no arm generates `--cpu-moe`'s
  tokens to the end. #29887 and #25294 both first differ at tokens 15, 13, 14
  and 15 (the GPU's expert matmuls against the CPU's); the fork's cache at 72,
  12, 27 and 15.
- PASS exact equivalence of buffered against `F_NOCACHE`: 20 of 20 requests.
- PASS no expert load failures in the fork's 8 cache runs.
- The port writes no `--expert-cache-stats`, so its decode split and hit rate
  are missing.

**Drift from E1.** Two fork arms moved beyond E1's range on the same matrix
settings:
- `iq3-cache128` at 16 GiB: 17.96 against E1's 16.94 (16.58–17.31);
- `iq3-cache128-buffered` at 16 GiB: 13.47 against 15.22 (15.10–15.34).

The 0 GiB arms are within E1's ranges. E1 ran on the old base; this run uses
master's Metal backend. The cause is not established, and neither change
alters a gate: with E1's numbers, the fork's buffered decode at 0 GiB
(20.96–21.31) still does not beat #25294 (20.42–21.30), and `F_NOCACHE` at
16 GiB still wins.

**For #25294's author** (evidence for E7, filed by hand):
- The top-6 abort and its threshold (M46).
- The port onto current master works unchanged in logic, bit-identical on the CPU.
- On Apple unified memory its buffered reads lose to `F_NOCACHE` slots under
  memory pressure.

### Reproduce

```sh
git -C third_party/llama.cpp fetch origin ref/pr25294-on-184f6536
git -C third_party/llama.cpp worktree add /tmp/pr25294 FETCH_HEAD
cmake -S /tmp/pr25294 -B /tmp/pr25294/build -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build /tmp/pr25294/build --target llama-server -j
go run ./cmd/kolibri-bench -matrix testdata/bench/e0.json -out ~/models/eval/expert-bench-e0 \
  -servers pr25294=/tmp/pr25294/build/bin/llama-server
```
