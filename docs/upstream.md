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
  `compare_real.py` reads `ffn_shexp_out` as `ffn_shexp`.
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
| 0009 mmap ranges | open | master now merges mapping ranges (`merge_ranges`, `llama-model-loader.cpp:1460`); `repro.py cpu-moe` decides whether issue 3 is fixed |
| 0010 continuation tests | no | tests on top of 0008 |
| 0011, 0012 expert cache | no | the fork's own work; it has to be rebased |

### Controls

| Mutation | Result |
|---|---|
| M43a: upstream with `kolibri1.expert_weights_norm=true` for libllama only (`compare_real.py --tiny --override-kv`) | "FAIL tiny, all 8 experts: logits NMSE 5.08e-01", "FAIL tiny, top-2 routing: same experts for 97.0% of (token, layer) pairs", "FAIL 12 problem(s)", rc 1 |
| M43b: a rekeyed GGUF with `tokenizer.ggml.pre = llama-bpe` (`kolibri-rekey -pre llama-bpe`) | "golden: 82/85 ok", for example "FAIL english/apostrophes", rc 1 |
