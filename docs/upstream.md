# Upstream llama.cpp issues

Three issues this port ran into belong to ggml-org/llama.cpp, not to the
Kolibri patches. This page checks that each still exists on current upstream
master, links the upstream reports that already exist, and keeps the repro
runnable. `tools/upstream/repro.py` exits 1 as soon as a check no longer
reproduces, so an upstream fix shows up as a red run.

The sections after them compare upstream's own Kolibri model (#30673) and its
two expert caches with this port's, for Phase 11, Stage E0. The last section
collects what upstream could use from #25294 and #29887, for Stage E7.

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
| 0013 expert cache waves | no | retired 2026-11-06, the fork's own work: Stage E4's gates failed at 0 GiB ([expert-streaming.md](expert-streaming.md#stage-e4-waves-for-prefill)) |
| 0014 expert cache side-file | no | the fork's own work, Stage E3; its four deciding gates passed on 2026-11-07, merged by fork PR #15 ([expert-streaming.md](expert-streaming.md#stage-e3-one-read-per-expert)) |
| 0015 expert cache GPU remap | no | retired 2026-11-07, the fork's own work: Stage E2's gate failed at 16 GiB and passed at 0 GiB ([expert-streaming.md](expert-streaming.md#stage-e2-gpu-remap)) |
| 0016 expert cache GPU remap on the side-file | no | the fork's own work, 2026-11-08: E2 and E3 together passed their registered gates against prefetch, through fork PR #16 ([expert-streaming.md](expert-streaming.md#stage-e2-and-e3-together)) |

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

**Patch 0013 is retired (Phase 11, Stage E4, 2026-11-06).** Its gates failed at
0 GiB, so the number is kept and the code stays on fork branch
`feat/kolibri1-waves` (`90311e5a2`, fork PR #14, open and not merged). It never
reached `feat/kolibri1`. At that date the next patch was 0014, and the series
0008, 0010, 0011 and 0012 applied to a clean `184f6536` equalled
`feat/kolibri1`'s tree (`7ade40f0`, `git diff --quiet` rc 0). The current tree
is in the next paragraph.

**Patch 0014 (Phase 11, Stage E3, 2026-11-07)** is on `feat/kolibri1` as
`c0b2de9be`, the squash of fork PR #15. The patch is that commit's plain diff. The
series 0008, 0010, 0011, 0012 and 0014 applied to a clean `184f6536` equals its
tree (`db0c6d55`), and without 0014 it equals its parent's, `ba4bd5bbc` (`7e901295`,
both by `git write-tree`). Its gates passed, so the number stays in the series. The
benchmark ran on `dd2040eaf`, whose tree (`53739515`) is what the same series
gave at `0b19754`, before the review fixes below.

**Merge order: fork PR #15 into `feat/kolibri1`, then PR #72.** The README's option
A (the patches by glob) and option B (the fork branch) give the same tree.

**Review fixes (fork PRs #13 and #15, 2026-11-07).** Two findings were real; the
fixes are `634e34bb4` on `feat/kolibri1`, merged into the side-file branch, and
`1f6ce476c` there. The first is in 0011, the second in 0014.
- **A LoRA on the routed experts was left out silently.** The matmuls read the
  slots, and an adapter's weight is looked up by tensor name. Such an adapter now
  fails to load, by name.
- **The side-file's table sized an allocation.** Its GGUF header length was
  checked only against the GGUF's size. The hash now reads in pieces of 1 MiB, and
  a length past the first expert row is refused.

Both act at load, and neither changes the code that reads a miss. E3's benchmark
ran on `dd2040eaf` before these commits, so they were not timed, and nothing here
asks to repeat it.

Three findings were declined, with the reasons on the PRs: the Kolibri-only
options in `common/arg.cpp` (twice, on #13 and #14), since this checklist is
upstream's and a slot count is the machine's memory budget, not model metadata;
and experts spread over several GGUF files, a stated limit that fails with an
error.

| Mutation | Result |
|---|---|
| M80: the LoRA refusal off (`if (false && ...)`) | `check_expert_cache.py`: "FAIL LoRA on ffn_gate_exps with the cache: exits 0 without generating, refused by name", the same for up and down, "FAIL: 3 failures", rc 1 |
| M81: the bound on the GGUF header off (`if (false && gguf_head > first_row)`) | `test-expert-cache`: "FAIL a GGUF header as long as the GGUF", "FAIL a GGUF header one byte into the experts", "2 checks failed", rc 1. Rerun on an export of `1f6ce476c`: `~/models/eval/expert-bench-e3.runner/review-controls.log` |

Commit `da2f581`'s message calls these two M70 and M71. PR #77 uses those numbers
for the loop baseline, so they moved to M80 and M81.

**Second review (fork PRs #13 to #16, 2026-11-07).** Four findings on `feat/kolibri1`
were real, all in 0011: `4a4661220`, merged into the side-file branch as `fcd9271a5`.
- **A draft model inherited the target's cache.** It would fail the architecture
  check at load. `common_base_params_to_speculative` now clears the cache's params.
- **`--expert-cache` with `--moe-cache-mib` aborted.** The graph's `GGML_ASSERT` fired
  on the tiny checkpoint. The context now refuses the pair by name.
- **A reopened model could be another file.** macOS reopens by path, and the offsets
  would read wrong weights without an error. A dup would share `F_NOCACHE` with the
  loader, so the cache compares device and inode and refuses a replaced file. Nothing
  tests the race itself.
- **A remap could evict a slot the last graph still read.** `graph_compute` is
  async. Metal waits for the last graph at its next split, which comes first when
  layer 0 is on the GPU, but a prefetch on a layer before that split could still
  evict. `process_ubatch` now synchronizes first when a cache is present. Nothing
  tests the race itself, and no benchmark ran with this: a decode is already
  synchronized before its next call, a prefill's wait moves ahead of its first split.

**Third review (PR #72, 2026-11-07).** `19001f2e0` on the side-file branch, in 0014.
A draft model inherited the target's side-file, and it would fail to load.
`common_base_params_to_speculative` now clears that option too, and `test-arg-parser`
checks all four cache options on the draft params. The test failed before the fix.
It acts only when params are derived for a draft, so nothing in the timed path
changes.

`ca32334bb` adds a test that an alignment of zero is refused. A review read the
division as coming before the check, and the check comes first. The other findings
were declined, with the reasons on the PRs: `llama.h` and the CLI options (upstream's
rules), `gguf_head`'s lower bound ("Limits" in
[expert-streaming.md](expert-streaming.md)), and everything on the waves and GPU remap
branches, which do not merge.

| Mutation | Result |
|---|---|
| M89: the MoE cache refusal off (`if (false && params.moe_cache_size > 0)`) | `check_expert_cache.py`: "FAIL llama-completion with --expert-cache and --moe-cache-mib exits -6", "FAIL: 1 failures", rc 1 |
| M90: the draft's cache params not cleared | `test-expert-cache`: "FAIL draft.expert_cache == 0", the same for the layers and the stats, "3 checks failed" |

**iOS, tvOS and visionOS did not build (fork CI, 2026-11-07).** `llama-expert-cache.cpp`
included `libproc.h`, which those SDKs do not ship, so `CI (apple)` failed on every
branch with the cache. `ba4bd5bbc` (0011) limits it to macOS; the three counters that
need `proc_pid_rusage` stay `null` elsewhere, as on Linux. It is merged into the
side-file branch as `bf7bf16b6`. This machine has no iOS, tvOS or visionOS SDK. Locally
the other path was only compiled against the macOS SDK with `TARGET_OS_OSX` forced to 0.

**Patch 0015 is retired (Phase 11, Stage E2, 2026-11-07).** Its deciding gate
failed at 16 GiB and passed at 0 GiB, so the number is kept and the code stays
on fork branch `feat/kolibri1-gpu-remap` (`0bb0f5444`, fork PR #16, draft, open
and not merged). It never reached `feat/kolibri1`. The series' glob matched it only on Stage E2's
branch, from `35b6ee2` until `5c40670` moved it to `pending/`, and the patch is not
in this PR's merge. The next patch was 0016.

**Patch 0016 is the GPU remap on top of the side-file (2026-11-08).** E2 and E3
together passed their registered gates against prefetch, so the remap enters the
series after all, now as `git diff c0b2de9be 714f6bb42`. Fork PR #16's branch was
fast-forwarded to the benchmarked merge `b001e946f`, then got one fix from review:
a draft model's params clear the remap with the other cache settings. A squash merge
gives the series' tree. The next patch is 0017.

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
- E2 removes the CPU split, behind `--expert-cache-gpu-remap`: patch 0016, on
  top of the side-file (E2 alone failed its decode gate at 16 GiB). It
  requires bit identity to the unsplit run (`check_expert_cache.py`, tiny and
  real IQ3, "PASS: 0 failures") and the same token IDs as `iq3-resident-ub21`
  on the real workload (exact in `testdata/bench/e2.json`: 30 of 30 requests).
  Without the option, `kolibri-bench`'s exact equivalence of
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
  memory pressure. This is E0's decode result, and E4 does not confirm it; see
  [What upstream could use](#what-upstream-could-use-25294-and-29887-phase-11-stage-e7).

### Reproduce

```sh
git -C third_party/llama.cpp fetch origin ref/pr25294-on-184f6536
git -C third_party/llama.cpp worktree add /tmp/pr25294 FETCH_HEAD
cmake -S /tmp/pr25294 -B /tmp/pr25294/build -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build /tmp/pr25294/build --target llama-server -j
go run ./cmd/kolibri-bench -matrix testdata/bench/e0.json -out ~/models/eval/expert-bench-e0 \
  -servers pr25294=/tmp/pr25294/build/bin/llama-server
```

## What upstream could use: #25294 and #29887 (Phase 11, Stage E7)

Collected 2026-11-07 from what is recorded on `main`, for the user to write from by hand. Nothing here has been posted anywhere, and this section holds facts, numbers and file references, not text to paste. Upstream's rule at the top of this page applies to whatever gets filed.

**In short:**
- **#25294:** its auto-sizing aborts on Top-6, confirmed with a threshold. It runs on Kolibri once sized by hand. At 0 GiB it beat the fork's `F_NOCACHE` arms in both benchmarks; only the fork's buffered waves passed it, in E4, in a non-deciding arm.
- **#29887:** on Metal it needs `--cpu-moe`, and in its only arm (E0, 0 GiB) it loses to plain `--cpu-moe`.
- **Both:** a Metal slot path, a ceiling and equivalence method, and a benchmark matrix they could run.

**State of the evidence:**
- **E0 and E1:** recorded.
- **E4 (waves):** recorded. Its gates failed at 0 GiB, so its code is not in the patch series.
- **E3 (one read per expert):** recorded. Its four deciding gates passed, and patch 0014 is in the series.
- **E2 (no per-layer CPU sync):** recorded. Its gate failed at 16 GiB and passed at 0 GiB, so its code is not in the patch series. See ["The slot path after E3 and E2"](#the-slot-path-after-e3-and-e2).
- **Upstream's side:** the PR states are as of this repo's checks of 2026-10-22 and 2026-10-23. They were not rechecked for this section. Check them, and both threads, before filing.

### Machine and builds

| Item | Value |
|---|---|
| Machine | M5 Pro, 48 GiB, APPLE SSD AP1024Z, internal drive ([expert-diagnostics.md](expert-diagnostics.md)) |
| macOS | changed between E0 and E4 (27.2 installed on 2026-10-29, [expert-streaming.md](expert-streaming.md#observations)) |
| Master used | `184f6536` (2026-10-23) |
| #25294 | PR head `1248fd8fa` did not merge on master as of 2026-10-23. All numbers come from the port on `184f6536`, fork branch `ref/pr25294-on-184f6536` (`8a8ddf827`), with the PR's logic unchanged and only API changes, listed in its merge commit. **No number here comes from the PR head itself.** |
| #29887 | as merged in `184f6536` |
| Fork | `feat/kolibri1`: series 0008, 0010, 0011, 0012 for E0 to E2's baselines; 0014 was added by E3 (`patches/llama.cpp/`) |
| Waves | `feat/kolibri1-waves` (`90311e5a2`), fork PR #14, open, not in the series |
| GPU remap | `feat/kolibri1-gpu-remap` (`0bb0f5444`), fork PR #16, draft, not in the series |
| GGUF | `Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf`, 35.6 GB, this repo's own quantization mix, header rewritten by `cmd/kolibri-rekey` |

This repo (`CWBudde/kolibri-llama-cpp`) is public, so its files can be linked. The benchmark's raw `run.json` and `report.md` are under `~/models/eval/`, outside the repo, and would need attaching.

### #25294: the Top-6 auto-size abort

Confirmed by E0.2 ("#25294: `--moe-stream` on Kolibri" above).

| Fact | Value |
|---|---|
| Sizing without a slot count | `llama_moe_stream_resolve_slots` takes `clamp(2*n_expert_used, 16, n_expert)` = 16 slots |
| Waves need | `3*n_expert_used` = 18 slots |
| Outcome | abort while reserving the graph, before any request, rc 134 |
| Message | "src/llama-graph.cpp:2394: MoE expert streaming: multi-pass expert GEMMs need an expert cache of at least 3*n_expert_used slots (have 16, need 18); increase --moe-stream-cache or reduce -ub" |
| Threshold (M46) | `--moe-stream-cache 17s` aborts ("have 17, need 18"), `18s` starts |
| Which models | Top-5 or fewer fits in 16 slots, and Top-6 is the first that does not (arithmetic from the two rules, not run on other models) |
| Trigger | a ubatch that touches more experts than the slots |
| Line number | from the port, not the PR head |
| Workaround used | `--moe-stream-cache 128s` |
| Reported before | The research notes list it among the PR's known issues: "Auto-sizing aborts when `n_expert_used >= 6` (suggested floor: 3×n_expert_used)" (`research_notes/AirLLM style offloading for Kolibri/apple_ssd_streaming.md`). The repo does not link that report, so find it in the thread first. What this repo adds is the run on Kolibri, the message and M46 |

**Missing:** the repo records no command line for the abort or for M46, and `tools/upstream/repro.py` has no check for it. The file the runs used is not named beside them either; E0.2's matrix uses the IQ3 GGUF. Whether the tiny router fixture (`cmd/kolibri-tiny -router`, 384 experts, top 6) reproduces it was not tried. That would matter, since it would need no real weight.

### #25294 and #29887: the benchmark numbers

Warm rates are medians over repetitions, each repetition being the median of 4 warm requests. Cold TTFT is the median over repetitions of the one cold request, including a 2,048-token prefill from an evicted page cache. Every cache holds 128 IQ3 experts per layer. E0 ran 2 repetitions and E4 ran 3. Sources, with ranges: "The benchmark" above and [E4's table](expert-streaming.md#results-1).

| Arm | Ceiling GiB | Stage | Warm prompt t/s | Cold TTFT s | Warm decode t/s | Footprint GiB |
|---|--:|---|--:|--:|--:|--:|
| #25294, `128s` | 0 | E0 | 283.2 | 8.62 | 20.86 | 14.0 |
| #25294, `128s` | 0 | E4 | 351.3 | 7.44 | 23.98 | 14.0 |
| #25294, `128s` | 16 | E0 | 165.9 | 15.43 | 12.69 | 14.0 |
| #25294, `128s` | 16 | E4 | 226.7 | 10.95 | 18.52 | 14.0 |
| #29887, `--cpu-moe --moe-cache-mib 10375` | 0 | E0 | 71.8 | 97.54 | 11.99 | 11.0 |
| `--cpu-moe` | 0 | E0 | 218.7 | 25.70 | 16.63 | 1.0 |
| `--cpu-moe` | 0 | E4 | 281.2 | 12.16 | 15.34 | 1.1 |
| `--cpu-moe` | 16 | E0 | 11.4 | 217.07 | 5.89 | 1.1 |
| `--cpu-moe` | 16 | E4 | 26.6 | 129.87 | 8.68 | 1.1 |
| fork, `F_NOCACHE`, ubatch capped at 21 | 0 | E0 | 81.5 | 35.05 | 17.71 | 11.0 |
| fork, `F_NOCACHE`, ubatch capped at 21 | 0 | E4 | 86.0 | 32.53 | 17.39 | 11.0 |
| fork, `F_NOCACHE`, ubatch capped at 21 | 16 | E0 | 79.0 | 35.50 | 17.96 | 11.0 |
| fork, `F_NOCACHE`, ubatch capped at 21 | 16 | E4 | 85.9 | 32.84 | 18.07 | 11.0 |
| fork, buffered, capped | 0 | E0 | 110.0 | 23.77 | 20.52 | 11.0 |
| fork, buffered, capped | 16 | E0 | 53.8 | 46.30 | 13.47 | 11.0 |
| fork, waves, `F_NOCACHE` | 0 | E4 | 301.4 | 8.91 | 18.58 | 11.0 |
| fork, waves, `F_NOCACHE` | 16 | E4 | 301.8 | 8.93 | 18.86 | 11.0 |
| fork, waves, buffered | 0 | E4 | 476.1 | 6.61 | 24.10 | 11.0 |
| fork, waves, buffered | 16 | E4 | 217.2 | 12.31 | 18.30 | 11.0 |

**Do not read more into this than it holds:**
- **E0's "F_NOCACHE beats #25294's decode at 16 GiB" does not repeat in E4.** E0: 17.96 against 12.69 t/s, a passed gate. E4: 18.07 (16.90–18.93) against 18.52 (8.14–19.39), overlapping ranges, and E4 had no gate on it. Its 16 GiB r1 is in E4's disturbed block, which may be the 8.14; the recorded text does not tie that value to the run. So buffered reads losing decode under memory pressure is an E0 result that E4 does not confirm. In prefill the fall remains: #25294 runs 351.3 t/s at 0 GiB and 226.7 at 16, while `F_NOCACHE` waves hold 301.4 and 301.8.
- **Sessions differ.** Across E0 and E4, #25294's warm prompt went from 283.2 to 351.3 t/s and `--cpu-moe`'s from 218.7 to 281.2. E4's Observations section shows #25294's repetitions rising with the page cache peak, n = 5, cause not established. Compare only within one stage.
- **Waves are not an upstream candidate from this data.** `F_NOCACHE` waves lost to #25294 at 0 GiB and buffered waves at 16 GiB, so no single mode won at both ceilings. They show that the design ports onto a zero-copy slot pool, not that it is faster.
- **The machine was shared.** Six consecutive E4 runs had others' CPU at or above 100%. Recomputing the gates without them flips none. See [Validity](expert-streaming.md#validity).

### #25294: further facts

| Fact | Value | Source |
|---|---|---|
| Port is sound on the CPU | tiny 384-expert top-6 fixture: logits bit-identical to plain master, with waves (64 slots, 32-token ubatch) and in one wave (200 slots) | "#25294: `--moe-stream` on Kolibri" above |
| Metal is not bit-identical to the unsplit run | on the same fixture it differs by 4.2e-4. The research notes record that the PR's byte-identical-output claim was retracted on 2026-09-06 | same; `apple_ssd_streaming.md` |
| Likely reason on Metal | not isolated for #25294. For the fork: the CPU remap splits the graph and master's fusion (`RMS_NORM + MUL + ADD`) then rounds differently; `GGML_METAL_FUSION_DISABLE=1` removes the difference, and a resident run split at the same points is bit-identical | "The rebase" above |
| Tokens on real weights | #25294, waves and buffered waves generate identical tokens, 128 of 128 on all 5 request kinds (90 requests) | [expert-streaming.md](expert-streaming.md#lossless) |
| Tokens against `--cpu-moe` | first differ at tokens 15, 13, 14 and 15. No run isolates the cause; GPU against CPU matmuls is the known candidate | [expert-streaming.md](expert-streaming.md#observations) |
| Footprint | 14.0 against the fork's 11.0 GiB. Likely mmap off, so the non-expert weights count; not measured separately | "The benchmark" above |
| Pass count | #25294 uses `n_waves = ceil(touch / cap)` (`src/llama-graph.cpp:2396`), 7 passes at 128 slots. The fork's waves build `ceil(n_touch / (slots - k))`, 4 passes, and grow waves when a union needs it | [expert-streaming.md](expert-streaming.md#deviations-and-known-limits) |
| Empty passes | in the fork's waves a static graph runs every built pass: 3,606 of 3,920 used over 12 runs; the time they cost is not measured. Not measured for #25294 | same |
| Its eviction, replayed | decayed hotness at 128 per layer, held out, IQ3: decode 92.1% against layer LRU 91.4%, prefill 36.1% against 20.7%. A trace replay of the rule, not its code | [expert-diagnostics.md](expert-diagnostics.md#eviction-policies-offline) |

**Not recorded for #25294:**
- its decode split, hit rate, disk bytes and load failures: it writes no `--expert-cache-stats`;
- any arm on Q8_0;
- bit identity on real weights: tokens only;
- what its miss path's copy costs on Metal. From the port's source, not measured: `src/llama-moe-stream.cpp` at `8a8ddf827` reads an expert into a staging buffer and then calls `ggml_backend_tensor_set` on the slot, so a load copies once after the read. Its slots are Metal buffers, which `cmd/kolibri-bench` treats as wired (`fits`, `cmd/kolibri-bench/main.go`).

### #29887: facts on Metal

| Fact | Value | Source |
|---|---|---|
| Activation | it accepts GPU and IGPU devices, and Metal reports a GPU; it caches only layers whose experts sit in a host buffer, so on Metal it needs `--cpu-moe` | "#29887: `--moe-cache-mib` on Metal" above |
| On | "MTL0 MoE cache size = 10375.00 MiB for 31125.00 MiB of host experts" | same |
| Off (M47) | without `--cpu-moe`: "no layer has all of its experts in host memory, MoE cache is disabled" | same |
| Counters over 48 decode tokens | "hits = 12472, misses = 3986, hit rate = 75.78%, uploaded = 6461.68 MiB"; same tokens as `--cpu-moe` | same |
| Logging | the server logs these lines only with `-v` | same |
| Against plain `--cpu-moe`, 0 GiB | decode 11.99 against 16.63 t/s, p95 174.7 against 79.3 ms, prompt 71.8 against 218.7 t/s, cold TTFT 97.54 against 25.70 s | "The benchmark" above |
| Commits per token | 285 against 103 | same |
| With the fork's cache | `build_moe_ffn` has both; they cannot be on together (an assert) | "The rebase" above |
| SLRU, replayed | at 128 per layer, held out, IQ3: shared SLRU decode 93.1% against shared LRU 92.5%, prefill 30.3% against 7.8% | [expert-diagnostics.md](expert-diagnostics.md#eviction-policies-offline) |
| Half split at small budgets, replayed | when the entry segment holds less than one token's experts across all layers: shared SLRU at 8 per layer 11.5% decode hits against 34% for LRU. A replay of the rule with one cache for all layers, as upstream shares one per device | same |

**Not recorded for #29887:** any arm at 16 GiB (the matrix has it at 0 only), Q8_0, a discrete GPU, and its internal split of time. The survey records its upstream benchmarks as RTX 4090/5090, where a host-to-VRAM copy has a purpose, so this result is about unified memory only. On unified memory the upload copies from the page cache into another buffer of the same RAM.

### The zero-copy Metal slot path

Code: `patches/llama.cpp/0011-kolibri-expert-cache.patch`, mainly `src/llama-expert-slots.*`. Design: [expert-streaming.md](expert-streaming.md#how-it-works).

| Part | Fact |
|---|---|
| Slots | slot tensors per layer, in one anonymous host mapping per device that Metal uses in place (`ggml_backend_dev_buffer_from_host_ptr`); slots never written cost no memory |
| Miss | `pread` writes straight into the slot, with no host-to-device copy; three ranges per expert (gate, up, down), from a second descriptor with `F_NOCACHE` |
| Remap | a CPU custom op maps the router's ids to slots before the three `mul_mat_id`; the expert weights keep the router's ids |
| Safety | a remap pins its ubatch's experts; a loading slot is never a victim; the scheduler's sync before each CPU split guards the slots of the previous layer |
| Footprint | `max_phys_footprint` 10.90 GiB at 128 IQ3 slots; page cache peak 2.7 GiB; holds at a 16 GiB hog |
| Correctness | `check_expert_cache.py`: logits bit-identical to a resident run at the same ubatch and, on Metal, the same split (tiny 164 checks, rc 0; real IQ3 "PASS: 0 failures") |
| Controls | M32: slot id off by one, all 32 logit checks fail. M33: victims chosen without pins, `test-expert-cache` fails. M45: the neighbour's down rows, every bit-identity line fails |

**What it costs, measured.** Decode, IQ3, 0 GiB, ms/token ([expert-diagnostics.md](expert-diagnostics.md#overhead-control-and-decode-split)):

| Config | Decode | Load wait | Remap | Splits/token | Metal commits/token |
|---|--:|--:|--:|--:|--:|
| resident `-ub 21` | 16.82 | 0.00 | 0.000 | 2 | 3 |
| 384 slots `-ub 21` (no misses after warmup) | 35.64 | 0.71 | 0.530 | 102 | 103 |
| 128 slots | 57.81 | 15.27 | 0.629 | 102 | 103 |

So 17.6 ms/token of the slot path is neither waiting nor remapping and comes with the 100 extra splits, about 0.18 ms per split. Zero-copy does not make interception free. The survey cites 19–21% interception overhead for the Metal slot-pool PoC (Discussion #23324), in another setup; here 35.64 is more than twice 16.82.

**Limits** ([expert-streaming.md](expert-streaming.md#limits)): Kolibri only, no LoRA on the routed experts, experts in one file, one context per model, devices without `buffer_from_host_ptr` rejected at load, `-fit` does not know the slot buffer, and the ubatch is capped at slots / 6 without waves. On macOS, cached pages are served from the page cache despite `F_NOCACHE`. On Linux, reads go through the page cache.

**Not measured:** zero-copy against a copying slot pool in isolation. No arm has the same cache with a copy, so the saving is not a number here. The comparison with #29887 mixes it with a different problem (experts in host RAM, not on SSD). #25294 copies once on a load, but it differs in ubatch, eviction and waves, so it does not isolate zero-copy either.

### The slot path after E3 and E2

Both stages aim at the cost in the table above. Every number is from the fork's own cache at 128 IQ3 slots, as medians over 3 repetitions. Each stage has its own matrix and session, so compare only within a stage. **Neither matrix has a #25294 or #29887 arm, so no gate compares them with upstream, and no run has E2 and E3 together.** Sources, with ranges: [E3](expert-streaming.md#stage-e3-one-read-per-expert) and [E2](expert-streaming.md#stage-e2-gpu-remap).

**E3: one `pread` per expert from a side-file, against three from the GGUF.** Gates passed, including on 30 of 30 identical token ID sequences.

| Metric | GGUF 0 GiB | Side-file 0 GiB | GGUF 16 GiB | Side-file 16 GiB |
|---|--:|--:|--:|--:|
| Warm decode t/s | 17.30 | 20.30 | 17.08 | 19.75 |
| Warm p95 ms | 86.1 | 70.6 | 79.7 | 71.4 |
| Warm prompt t/s | 88.4 | 106.5 | 87.0 | 106.8 |
| Cold TTFT s | 33.16 | 26.68 | 33.13 | 26.75 |
| Load wait ms/token | 14.48 | 8.85 | 14.63 | 8.86 |

The 16 GiB p95 did not win by the rule, so that decode gate passes on tokens/s alone. The side-file takes another 30.5 GiB. The benchmark ran on `dd2040eaf`, and the review fixes in the series after it were not timed.

**E2: a GPU remap with no scheduler split at a layer that hits, against the CPU remap.** The gate failed at 16 GiB and passed at 0 GiB, by small margins that three repetitions cannot resolve. Lossless: logits bit-identical to the unsplit resident run on real IQ3, and 30 of 30 token ID sequences equal.

| Metric | CPU remap 0 GiB | GPU remap 0 GiB | CPU remap 16 GiB | GPU remap 16 GiB |
|---|--:|--:|--:|--:|
| Warm decode t/s | 17.78 | 19.61 | 18.46 | 19.96 |
| Decode ms/token | 56.82 | 52.10 | 54.31 | 52.10 |
| Splits per token | 102 | 2 | 102 | 2 |
| Commits per token | 103 | 33.97 | 103 | 33.97 |

At 16 GiB the gate's baseline was the CPU remap with `--expert-prefetch 12`, not this column: slowest 18.94 against fastest 19.00 t/s, highest p95 79.33 against lowest 78.58 ms. So removing 100 splits saved 4.72 and 2.21 ms/token here, well under the 17.6 ms that came with those splits at all hits.

**Where the rest of the gap goes** ([expert-decode-gap.md](expert-decode-gap.md)). This is a diagnosis on a replay pattern with private timers. It passes no gate, and its arms have one to three runs. On the GPU remap a token took 49.31 ms against 16.46 resident, including 13.33 ms of load wait. One ordering of the arms splits the remaining 19.5 ms as follows, and another order would change the sizes:

| Cause | ms/token | How it was separated |
|---|--:|---|
| The thread waiting for loads sleeps | 9.45 | same misses, with that thread spinning instead |
| Extra command buffers and stops | 7.38 | forced stops with no load and no idle |
| The gated path at all hits | 2.91 | tokens with no miss against resident |

Spinning that thread took 49.31 to 38.90 ms/token. It is an experiment, not a design: energy, heat and sustained runs were not measured. No clock or power figure was observed, so the explanation that the chip drops to a lower performance state is an inference.

**Not recorded:** whether any of this carries over to #25294 or #29887, whose slot paths differ.

### The equivalence and ceiling method

Code: `cmd/kolibri-bench`, `cmd/kolibri-evict`, `cmd/kolibri-memhog`, `internal/memctl`, `cmd/kolibri-stream`, `tools/gguf/check_expert_cache.py`. Written up in [expert-streaming.md](expert-streaming.md#method).

| Part | What it does | Why, or what went wrong without it |
|---|---|---|
| Ceiling | `kolibri-memhog` wires 0 or 16 GiB with `mlock` before the server starts | compression and swap cannot undo it |
| Fit check | `fits` checks against `hw.memsize` and records "does not fit". Resident: all weights and the hog. Cache: the non-expert weights, the slots, the hog and 8 GiB for the system. `--cpu-moe` always fits | resident IQ3 beside the hog wired 49.1 GiB of 48 and hung the Mac until a watchdog panic (2026-10-15). Q8_0 at 128 slots paged out and gave no token in 20 minutes |
| Cold start | `kolibri-evict` drops the GGUF with `msync(MS_INVALIDATE)`, no root | 576,976 resident pages to 0; M41 and M35 check it |
| Quiet machine | 1-minute load below 3, swap growth below 100 MiB per minute, AC power, `caffeinate` | runs on battery slept and were repeated. It holds only at a run's start, so others' CPU is recorded per run |
| Order | repetitions outermost | spreads drift over the arms |
| Gates | in the matrix, fixed before the first timed run; every repetition must beat the best repetition of the best baseline (E4's decode-not-slower: the median must reach the baseline's slowest repetition); no hit rate counts | E4's failed gates stayed failed. A later added arm is labelled. The order is what each stage's text records: on `main`, the matrices of D, E0 and E4 first appear in the commit that holds their results |
| Equivalence, exact | same token IDs, every request of every run of an arm | M37: one changed ID fails with the first difference |
| Bit identity | `np.array_equal` on logits against a resident run at the same ubatch, on Metal also the same split | master's fusion differs across CPU splits, up to 3.23 in IQ3 logits against an unsplit run |
| Reference choice | at a different ubatch or device, token equivalence is INFO only | the ubatch changes the order of the sums: first differences at tokens 12 to 22 |
| Footprint | `phys_footprint` beside the GGUF's page-cache residency from `mincore` | buffered reads hold up to 20.8 GiB that a 10.9 GiB footprint does not show |
| Negative controls | checks are shown failing on a mutation (global M-numbers) | M49's plain form failed only the unit test, which is why a strong form was added |
| Trace replay | causal, phases apart, a step as one event, a Belady bound labelled unattainable | [expert-predictor-result.md](expert-predictor-result.md) |

### The Kolibri benchmark matrix

All matrices run `llama-server -ngl 99 -t 10 -c 4096 -np 1 --fit off`, one cold request (coding) and four warm ones (coding, research, hr, medtech), each 2,048 prompt tokens and 128 generated at temperature 0 with `ignore_eos`.

| Matrix | Stage | Reps | Arms | Ceilings GiB | Fork build | Results in the repo |
|---|---|--:|---|---|---|---|
| `testdata/bench/expert-cache.json` | D | 3 (the control 1) | IQ3 resident, `--cpu-moe`, cache 128, cache with prefetch 12, resident `-ub 21`; Q8_0 cache, cache with prefetch, `--cpu-moe --no-repack` | 0, 16 (Q8_0 and the control at 0) | `2da6ec661` | [expert-streaming.md](expert-streaming.md#results) |
| `testdata/bench/e1.json` | E1 | 2 | IQ3 resident `-ub 21`, cache 384 `-ub 21`, cache 128, cache 128 buffered; Q8_0 cache 128, buffered | 0, 16 (the two `-ub 21` controls at 0) | `746be1eae` | [expert-diagnostics.md](expert-diagnostics.md) |
| `testdata/bench/e0.json` | E0.2 | 2 | IQ3 `--cpu-moe`, cache 128, buffered, #29887, #25294 | 0, 16 (#29887 at 0) | rebased fork | "The benchmark" above |
| `testdata/bench/e4.json` | E4 | 3 | IQ3 `--cpu-moe`, cache 128, waves, waves buffered, #25294 | 0, 16 | `90311e5a2` | [expert-streaming.md](expert-streaming.md#results-1) |

To add an upstream build as an arm: give the config a `server` name, and map it with `-servers name=<llama-server>` ("Reproduce" above).

**For a reader who wants to rerun it:**
- `go run ./cmd/kolibri-models check testdata/bench/<matrix>.json` confirms the weight is on the internal drive first ([AGENTS.md](../AGENTS.md), "Benchmarks").
- `iq3` is a custom mix, and the matrix hard-codes `~/models/` names. This repo records no public download or hash for the file, and the safetensors it came from were deleted here, so a reader must convert their own ([real-checkpoint.md](real-checkpoint.md)).
- Runs take hours and wire much of the machine. The fit limits are for 48 GiB.
- The workloads come from `tools/locality/workloads.py` and are written to `~/models/eval/`; `testdata/locality/manifest.json` holds their hashes.
