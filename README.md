# kolibri-llama-cpp

This repo adds native support for [Aleph Alpha Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16)
to [llama.cpp](https://github.com/ggml-org/llama.cpp). The goal is an
unquantized GGUF that matches Aleph Alpha's vLLM reference implementation,
followed by a quantized Metal build that runs on a Mac with 48 GB unified
memory.

Kolibri-1 is a 50-layer MoE model:
- 384 routed experts per layer, Top-6 sigmoid routing, plus one shared expert;
- hybrid attention: four sliding-window layers, then one full-attention layer;
- sandwich norms;
- a 128k byte-level BPE vocabulary.

The port reuses llama.cpp's Qwen3-MoE infrastructure as far as possible, the
same strategy the reference vLLM plugin follows.

This is an independent project, not affiliated with Aleph Alpha. The llama.cpp
changes are kept as patches against a pinned upstream commit, not as a fork.

## Status

See [`PLAN.md`](PLAN.md) for the full plan.

| Topic | State |
|---|---|
| Reference outputs from the official vLLM implementation | open; needs hardware for the 78–156 GB checkpoint |
| Checkpoint and tensor inventory | done: [docs/checkpoint.md](docs/checkpoint.md) |
| Tokenizer compatibility | done: [docs/tokenizer.md](docs/tokenizer.md) |
| GGUF architecture and HF → GGUF converter | arch registration, GGUF metadata and tensor conversion done, checked on a tiny synthetic checkpoint: [docs/gguf-conversion.md](docs/gguf-conversion.md). The real BF16 checkpoint converts with a peak footprint of 7.5 GiB and is bit-exact in every tensor: [docs/real-checkpoint.md](docs/real-checkpoint.md). FP8 input is open |
| Model class and MoE graph | done on tiny synthetic checkpoints (CPU and Metal); router and MoE block match the reference function: [docs/model.md](docs/model.md). The full 50-layer graph runs on the real BF16 weights (CPU) |
| Hybrid attention and KV cache | done; the 4:1 pattern over 50 layers, the iSWA cache split, sliding-window mask, off-by-one, RoPE on the sliding layers only, GQA 48/4 and per-head QK norm match the reference on tiny checkpoints; the attention and KV cache also hold at 8k, 16k, 64k and 262k tokens: [docs/attention.md](docs/attention.md) |
| Numerical validation | open; needs the vLLM reference outputs. Against a torch port of the vLLM model code on the real weights, the BF16 GGUF is as close as BF16 rounding of that port itself (KLD 0.033 each, wikitext-2 chunk 1), and both continue raw German prompts with the same repetition: [docs/real-checkpoint.md](docs/real-checkpoint.md) |
| Quantization, Apple Silicon | candidates run fully on Metal on a 48 GB Mac, with 32k context within the default Metal limit. The best so far: IQ3_XXS gate/up and IQ4_XS down for the routed experts with an importance matrix, Q8_0 elsewhere, 33.1 GiB, 64 tokens/s. Quality is only measured against this port's own BF16 (KLD 0.095; 0.108 for the plain Q3_K mix) until the numerical validation passes: [docs/real-checkpoint.md](docs/real-checkpoint.md) |
| Chat template and inference behavior | in progress; llama-server renders the chat template exactly as the reference does for every reasoning mode and tool-call shape, the chat parser splits reasoning, content and tool calls, and generation stops on the reference's two eos tokens: [docs/chat.md](docs/chat.md). The recommended sampling waits for the numerical validation |

### Results so far

- **Every tensor is identified.** All 19,200 per-expert tensors of each kind,
  the router with its correction bias, the shared expert and the four block
  norms map to GGUF names. The mapping table is in
  [docs/checkpoint.md](docs/checkpoint.md).
- **The tokenizer needs no new runtime code.** llama.cpp's existing `gpt2` BPE
  vocabulary with the `QWEN2` pre-tokenizer reproduces it exactly. Kolibri
  needs only a name registration with "no BOS token":
  - 85 golden cases match the HF `tokenizers` reference;
  - about 1.35 million differential fuzz strings match as well, both token
    IDs and detokenized bytes;
  - so does a fixed corpus of 10,673 English, German and code cases
    (1.27 million tokens).
- **The `kolibri` architecture is registered.** It is known to gguf-py and
  libllama.
- **The converter writes the GGUF metadata.** `convert_hf_to_gguf.py` knows
  `Kolibri1ForCausalLM`. It writes:
  - the dimensions, head counts and RMSNorm epsilon;
  - the MoE layout: 384 experts, Top-6, expert FFN 512, one shared expert;
  - the hybrid attention: sliding window 513 (512 preceding + current), a
    per-layer SWA/full pattern, and RoPE on the sliding layers only;
  - the router gating: sigmoid weights, no renormalization, scale 1.0.
- **The converter writes every tensor.** The block norms and the correction
  bias map explicitly, and the per-expert tensors are stacked per layer.
  - A tiny random BF16 checkpoint with the reference fixture's shape converts
    to a GGUF whose 111 tensors match the expected names, shapes and dtypes.
  - Their data is bit-exact against the source.
  - The real 156 GB checkpoint has not been converted yet.
- **libllama loads and runs a Kolibri GGUF.** `llama_model_kolibri` builds
  the sandwich-norm block, the iSWA attention with RoPE on the sliding layers
  only, and the routed + shared MoE. Its router selects experts by
  `logits + bias` and weights them by `sigmoid(logits)`.
  - The tiny checkpoint decodes 100 tokens on CPU and Metal.
  - With all experts active, the two devices and two ubatch sizes agree to
    NMSE ≤ 3e-6; with top-2 routing the greedy tokens agree at every
    position.
  - The graph's wiring is checked node by node: bias on the raw logits,
    routed + shared sum, sandwich norms, RoPE only on the sliding layers.
- **The router and MoE block match the reference.** A tiny checkpoint with
  384 experts and top 6 runs through libllama on CPU and Metal. Every MoE step
  of every layer matches a float64 recomputation built on the reference's own
  routing function:
  - router logits and the Top-6 selection;
  - the sigmoid weights, without renormalization or scale;
  - routed and shared output, their sum, the post-FFN norm and the residual.

  On the CPU every step stays at NMSE ≤ 4e-14; on Metal, selection and weights
  are exact.
- **The hybrid attention matches the reference.** A tiny checkpoint with the
  real heads (48 query, 4 KV, head_dim 128) and the real window (513) decodes
  1100 tokens on CPU and Metal: in one batch, in chunks of 64 that read an
  evicting SWA cache, and one token at a time across positions 511–513. Every
  attention step of every layer matches a float64 recomputation:
  - per-head QK norm, then vLLM's NeoX RoPE on the sliding layers only;
  - the attention with FlashAttention window `(512, 0)`, which is 512
    preceding tokens plus the current one, and causal attention without
    RoPE on the full layers;
  - output projection, sandwich norm and residual.

  A second tiny checkpoint has the real 50 layers (four sliding, one full,
  repeated): RoPE runs in exactly the 40 sliding layers, and the iSWA cache
  holds 10 full and 40 sliding layers. The GQA head mapping and the per-head
  QK norm are each at least 100× closer than the wrong variant.

  On the CPU with an F32 KV cache, the attention stays at NMSE ≤ 2e-13. An
  off-by-one in either direction moves the output all the way to the wrong
  window's reference, in every configuration.

  At long contexts (8192, 16384 and 65536 tokens on CPU and Metal, the real
  262144 on Metal), the full layers' KV cache holds every position, and the
  attention at sampled positions matches a float64 recomputation over all
  earlier keys without degrading: NMSE 5.6e-14 at 65536 on the CPU with an F32
  KV cache. Only the RoPE angles lose float32 precision with the position, and
  vLLM's own float32 cos/sin cache loses it too; libllama stays within 30× of
  that error.
- **llama-server renders the chat template exactly as the reference does.**
  For 7 message sets (tool loops, parallel calls, reasoning in the history)
  and 31 ways to choose the reasoning mode, `/apply-template` gives the same
  prompt, byte for byte, as transformers with the arguments vLLM derives. One
  request shape (`reasoning_effort: "none"` next to `enable_thinking: true`)
  needed a one-line llama-server fix. llama.cpp's chat parser splits
  `<think>` reasoning and `<tool_call>` JSON calls like the official parsers,
  apart from whitespace and a tool call written inside the reasoning.
  Generation stops on exactly `<|im_end|>` and `<|endoftext|>`.

## Layout

| Path | Contents |
|---|---|
| `cmd/kolibri-inventory` | Builds the tensor inventory from Hugging Face. It reads only the safetensors headers, through HTTP range requests, so no payload is downloaded. |
| `cmd/kolibri-peek` | Fetches individual small tensors through range requests and prints value statistics. |
| `cmd/kolibri-tiny` | Writes a tiny random-weight checkpoint with the reference fixture's shape, plus a manifest of the expected GGUF tensors, for converter and model tests. `-router` writes the reference router test's shape instead (384 experts, top 6); `-attn` writes the real attention heads and sliding window (48/4 heads, head_dim 128, window 513); `-pattern` writes the real 50-layer SWA/full pattern. |
| `internal/` | Hugging Face client, safetensors header parser and writer, and the Kolibri tensor specs. `internal/kolibri/tensors.go` is the source of truth for the HF → GGUF mapping. |
| `inventory/{bf16,fp8}/` | Committed inventories: `summary.json` and `tensors.jsonl.gz`. They pin the model revisions and the sha256 of every file. |
| `testdata/tokenizer/golden.jsonl` | Tokenizer golden cases: IDs and decoded text from the reference tokenizer. |
| `testdata/tokenizer/corpus.json` | Manifest of the fixed tokenizer corpus: sha256 of each text file, number of cases, token count and sha256 of the reference IDs. |
| `testdata/e2e/` | The end-to-end corpus: six fixed cases with their token IDs (`corpus.json`), and the sha256 of every stored reference array plus the recorded libllama run with the sha256 of each of its nodes (`manifest.json`). The artifacts themselves live in `~/models/eval/e2e`. |
| `tools/tokenizer/` | Golden-file generator, llama.cpp ↔ reference comparison (golden, fuzz, invalid UTF-8, fixed corpus), vocab-only GGUF writer, pinned Python requirements. |
| `tools/gguf/` | `check_arch.py` checks the `kolibri` architecture registration in gguf-py and libllama. `check_metadata.py` checks the converter's GGUF metadata against `config.json`. `check_tensors.py` converts the `cmd/kolibri-tiny` checkpoint and checks every tensor's name, shape, dtype and data. `check_model.py` loads that checkpoint's GGUF in libllama, compares the logits across devices and ubatch sizes, and checks the graph's wiring. `check_moe.py` compares every MoE step per layer with the reference router, on the 384-expert variant. `check_attn.py` compares every attention step per layer with the reference attention (window, RoPE, KV cache, GQA, QK norm), on the `-attn` and `-pattern` variants. `check_long.py` does the same for the attention at 8k, 16k, 64k and 262k tokens, on sampled positions. `check_real.py` checks a GGUF converted from the real BF16 checkpoint against the inventory: shard hashes, tensor set, shapes, dtypes and bit-exact data. |
| `tools/ref/` | `kolibri_ref.py` is a whole-model forward in torch, ported from the reference's vLLM model code, that reads a Kolibri GGUF. `compare_real.py` validates it against libllama on the tiny fixture, then compares libllama with it on the real BF16 GGUF, layer by layer, on the CPU and with `--metal` on the GPU. `router_probe.py` records router logits, Top-1, Top-6 overlap and near-tie margins per token and layer. `e2e.py` stores the reference for the end-to-end corpus once and checks a libllama build against it, every node bit-identical to a recorded run. `regress.py` runs the regression suite: the tokenizer corpus, `compare_real.py --tiny` and `e2e.py`. |
| `tools/quant/` | `calibration.py` builds the imatrix calibration text from English wikitext, German Wikipedia and source code, so the imatrix reaches the experts that English text alone leaves without data. |
| `tools/chat/` | `check_chat.py` checks the chat template in the GGUFs and in llama-server, compares llama-server's rendered prompts with the reference renderer for every reasoning mode and tool-call shape, and checks the stop tokens. |
| `patches/llama.cpp/` | The llama.cpp changes, applied in order. The same changes are commits on [CWBudde/llama.cpp](https://github.com/CWBudde/llama.cpp) `feat/kolibri`. 0009 is a generic llama.cpp fix, not Kolibri code: it lets Metal run with `--cpu-moe` on files above the Metal working set. |
| `docs/` | Reference docs per topic (checkpoint, tokenizer, conversion, model, attention, chat, real checkpoint), with findings, pitfalls and reproduction steps. |

## Setup

Requires Go (see `go.mod`), Python 3.12 with [uv](https://github.com/astral-sh/uv),
CMake, Ninja and a C++ compiler.

```sh
# Go tools and tests (no network needed for the tests)
go test ./...

# llama.cpp source, option A: the pinned commit with the patches applied
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git -C third_party/llama.cpp checkout -b kolibri 1537a0a8b2f8711d840878b0a0677ab2213c882c
for p in patches/llama.cpp/*.patch; do git -C third_party/llama.cpp apply "$PWD/$p"; done

# option B, instead of A: the fork, which has the same patches as commits
git clone --branch feat/kolibri https://github.com/CWBudde/llama.cpp third_party/llama.cpp

# build (either option)
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=ON
cmake --build third_party/llama.cpp/build --target llama llama-tokenize llama-server test-tokenizer-0 test-llama-archs \
    test-chat test-chat-peg-parser test-chat-auto-parser test-chat-template

# Python environment for the tokenizer and GGUF tools
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install --index-strategy unsafe-best-match -r tools/tokenizer/requirements.txt
```

`third_party/` and `.venv/` are gitignored. Without Ninja, `-G "Unix Makefiles"`
works too. On macOS, the Python tools load `libllama.dylib` instead of `libllama.so`.

## Checks

```sh
go vet ./... && go test ./...
.venv/bin/python tools/tokenizer/golden.py --check            # golden file vs. installed reference
.venv/bin/python tools/tokenizer/vocab_gguf.py --llama-cpp third_party/llama.cpp \
    --out third_party/llama.cpp/models/ggml-vocab-kolibri.gguf
.venv/bin/python tools/tokenizer/compare.py --llama-cpp third_party/llama.cpp \
    --vocab third_party/llama.cpp/models/ggml-vocab-kolibri.gguf  # golden + fuzz + UTF-8 report + corpus (skipped without ~/models/eval)
.venv/bin/python tools/gguf/check_arch.py --llama-cpp third_party/llama.cpp
.venv/bin/python tools/gguf/check_metadata.py --llama-cpp third_party/llama.cpp
.venv/bin/python tools/gguf/check_tensors.py --llama-cpp third_party/llama.cpp
.venv/bin/python tools/gguf/check_model.py --llama-cpp third_party/llama.cpp
.venv/bin/python tools/gguf/check_moe.py --llama-cpp third_party/llama.cpp
.venv/bin/python tools/gguf/check_attn.py --llama-cpp third_party/llama.cpp
.venv/bin/python tools/gguf/check_long.py --llama-cpp third_party/llama.cpp  # about 10 min
.venv/bin/python tools/chat/check_chat.py --llama-cpp third_party/llama.cpp
.venv/bin/python tools/ref/compare_real.py --llama-cpp third_party/llama.cpp --tiny
ctest --test-dir third_party/llama.cpp/build -R 'test-tokenizer-0|test-generate-models|test-chat'
third_party/llama.cpp/build/bin/test-llama-archs
```

On the real checkpoint (156 GB download, another 156 GB for the GGUF; see
[docs/real-checkpoint.md](docs/real-checkpoint.md)):

```sh
.venv/bin/python tools/gguf/check_real.py --llama-cpp third_party/llama.cpp \
    --model-dir ~/models/Kolibri-1-BF16 --gguf ~/models/Kolibri-1-BF16.gguf  # about 12 min
.venv/bin/python tools/gguf/check_metadata.py --llama-cpp third_party/llama.cpp \
    --gguf ~/models/Kolibri-1-BF16.gguf
.venv/bin/python tools/ref/compare_real.py --llama-cpp third_party/llama.cpp \
    --gguf ~/models/Kolibri-1-BF16.gguf --kld-base ~/models/eval/kld-bf16-c512-n20.bin \
    --metal --probe-out /tmp/kolibri-router  # about 15 min, plus 3 min for --metal
.venv/bin/python tools/ref/regress.py --llama-cpp third_party/llama.cpp \
    --gguf ~/models/Kolibri-1-BF16.gguf  # regression suite, end-to-end corpus included, about 27 min
```

Network access:
- The tokenizer tools download only `config.json`, `tokenizer.json` and
  `tokenizer_config.json`. They use the pinned revision and verify each file's
  sha256 against `inventory/bf16/summary.json`.
- Regenerating the inventory needs network access but downloads no tensor
  payloads; see [docs/checkpoint.md](docs/checkpoint.md).
- `tools/quant/calibration.py` downloads 225 German Wikipedia articles from
  the Hugging Face datasets server once and caches them next to its output.

## Known upstream issues

These belong upstream. Only the last has a fix here, as a generic patch.
Details are in [docs/tokenizer.md](docs/tokenizer.md), except the last,
which is in [docs/real-checkpoint.md](docs/real-checkpoint.md).

- **Invalid UTF-8 can crash `llama_tokenize`.** Some inputs abort the whole
  process, for example the 4 bytes `F4 90 80 80` (U+110000). This affects
  every model that uses llama.cpp's byte-level BPE tokenizer.
- **`tests/test-tokenizer-random.py` is broken:**
  - its `LibLlamaModel` uses the pre-`llama_vocab` API;
  - under transformers 5 its word lists collapse into one string.
- **Metal with `--cpu-moe` crashes on files larger than the Metal working set.**
  Upstream wraps one mmap range per backend, from its first tensor to its
  last. With `--cpu-moe` that range covers nearly the whole file, so Metal
  maps and requests residency for the routed experts too. The CPU expert
  matmul (`ggml_compute_forward_mul_mat_id`) then dies with SIGBUS
  (`KERN_PROTECTION_FAILURE`); the 79 GiB Q8_0 and 149 GiB BF16 files crash.
  The generic fix is patch 0009 (`0009-mmap-buffer-ranges.patch`), which
  maps only the ranges that hold a backend's tensors. On the fork it is
  [CWBudde/llama.cpp#10](https://github.com/CWBudde/llama.cpp/pull/10); it is
  not yet proposed to ggml-org/llama.cpp.
