# Upstream llama.cpp issues

Three issues this port ran into belong to ggml-org/llama.cpp, not to the
Kolibri patches. This page checks that each still exists on current upstream
master, links the upstream reports that already exist, and keeps the repro
runnable. `tools/upstream/repro.py` exits 1 as soon as a check no longer
reproduces, so an upstream fix shows up as a red run.

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
  --exclude=tests/test-llama-archs.cpp patches/llama.cpp/000[1-8]-*.patch
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

## 2. `tests/test-tokenizer-random.py`

```sh
.venv/bin/python tools/upstream/repro.py tokenizer-random --llama-cpp /tmp/upstream
```

```text
REPRODUCED test-tokenizer-random LibLlamaModel: tokenize('Hello') raised TypeError: initializer for ctype 'struct llama_vocab *' must be a pointer to same type, not cdata 'struct llama_model *'
REPRODUCED test-tokenizer-random TokenizerGroundtruth: transformers 5.18.0: 1 vocab entry for 127998 tokens, 1 added-token entry for 98 added tokens
```

- **`LibLlamaModel`:** its `tokenize` and `detokenize` pass `self.model`
  where `llama_tokenize`/`llama_detokenize` take a `llama_vocab *`. The
  script therefore fails at its first call.
- **`TokenizerGroundtruth`:** builds `vocab` and `added_tokens` with
  `batch_decode` of a flat ID list. Under transformers 5 that returns a
  single string, so the vocab and added-token fuzzers each test one
  concatenated string. The check uses the pinned Kolibri tokenizer; the
  cause is the transformers call, not the model.

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
