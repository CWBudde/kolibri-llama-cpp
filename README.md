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

| Phase | Topic | State |
|---|---|---|
| 0 | Reference outputs from the official vLLM implementation | open; needs hardware for the 78–156 GB checkpoint |
| 1 | Checkpoint and tensor inventory | done: [docs/phase1-tensor-inventory.md](docs/phase1-tensor-inventory.md) |
| 2 | Tokenizer compatibility | done: [docs/phase2-tokenizer.md](docs/phase2-tokenizer.md) |
| 3 | GGUF architecture and HF → GGUF converter | in progress; arch registration, GGUF metadata and tensor conversion done, checked on a tiny synthetic checkpoint: [docs/phase3-gguf-arch.md](docs/phase3-gguf-arch.md) |
| 4 | Native model loading and MoE graph | in progress; model class loads and runs the tiny synthetic checkpoint on CPU and Metal: [docs/phase4-model.md](docs/phase4-model.md). Router and MoE block match the reference function: [docs/phase4-router.md](docs/phase4-router.md). The 50-layer graph needs the full checkpoint |
| 5 | Hybrid attention and KV cache | in progress; sliding-window mask, off-by-one, RoPE on the sliding layers only and SWA-cache reads match the reference on a tiny checkpoint with the real heads and window: [docs/phase5-attention.md](docs/phase5-attention.md) |
| 6–9 | Numerical validation, quantization, Apple Silicon, chat behavior | open |

### Results so far

- **Every tensor is identified.** All 19,200 per-expert tensors of each kind,
  the router with its correction bias, the shared expert and the four block
  norms map to GGUF names. The mapping table is in the Phase 1 doc.
- **The tokenizer needs no new runtime code.** llama.cpp's existing `gpt2` BPE
  vocabulary with the `QWEN2` pre-tokenizer reproduces it exactly. Kolibri
  needs only a name registration with "no BOS token":
  - 85 golden cases match the HF `tokenizers` reference;
  - about 1.35 million differential fuzz strings match as well, both token
    IDs and detokenized bytes.
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

  On the CPU with an F32 KV cache, the attention stays at NMSE ≤ 2e-13. An
  off-by-one in either direction moves the output all the way to the wrong
  window's reference, in every configuration.

## Layout

| Path | Contents |
|---|---|
| `cmd/kolibri-inventory` | Builds the tensor inventory from Hugging Face. It reads only the safetensors headers, through HTTP range requests, so no payload is downloaded. |
| `cmd/kolibri-peek` | Fetches individual small tensors through range requests and prints value statistics. |
| `cmd/kolibri-tiny` | Writes a tiny random-weight checkpoint with the reference fixture's shape, plus a manifest of the expected GGUF tensors, for converter and model tests. `-router` writes the reference router test's shape instead (384 experts, top 6); `-attn` writes the real attention heads and sliding window (48/4 heads, head_dim 128, window 513). |
| `internal/` | Hugging Face client, safetensors header parser and writer, and the Kolibri tensor specs. `internal/kolibri/tensors.go` is the source of truth for the HF → GGUF mapping. |
| `inventory/{bf16,fp8}/` | Committed inventories: `summary.json` and `tensors.jsonl.gz`. They pin the model revisions and the sha256 of every file. |
| `testdata/tokenizer/golden.jsonl` | Tokenizer golden cases: IDs and decoded text from the reference tokenizer. |
| `tools/tokenizer/` | Golden-file generator, llama.cpp ↔ reference comparison (golden, fuzz, invalid UTF-8), vocab-only GGUF writer, pinned Python requirements. |
| `tools/gguf/` | `check_arch.py` checks the `kolibri` architecture registration in gguf-py and libllama. `check_metadata.py` checks the converter's GGUF metadata against `config.json`. `check_tensors.py` converts the `cmd/kolibri-tiny` checkpoint and checks every tensor's name, shape, dtype and data. `check_model.py` loads that checkpoint's GGUF in libllama, compares the logits across devices and ubatch sizes, and checks the graph's wiring. `check_moe.py` compares every MoE step per layer with the reference router, on the 384-expert variant. `check_attn.py` compares every attention step per layer with the reference attention (window, RoPE, KV cache), on the `-attn` variant. |
| `patches/llama.cpp/` | The llama.cpp changes, applied in order. The same changes are commits on [CWBudde/llama.cpp](https://github.com/CWBudde/llama.cpp) `feat/kolibri`. |
| `docs/` | One report per phase, with findings, pitfalls and reproduction steps. |

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
cmake --build third_party/llama.cpp/build --target llama llama-tokenize test-tokenizer-0 test-llama-archs

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
    --vocab third_party/llama.cpp/models/ggml-vocab-kolibri.gguf  # golden + fuzz + UTF-8 report
.venv/bin/python tools/gguf/check_arch.py --llama-cpp third_party/llama.cpp
.venv/bin/python tools/gguf/check_metadata.py --llama-cpp third_party/llama.cpp
.venv/bin/python tools/gguf/check_tensors.py --llama-cpp third_party/llama.cpp
.venv/bin/python tools/gguf/check_model.py --llama-cpp third_party/llama.cpp
.venv/bin/python tools/gguf/check_moe.py --llama-cpp third_party/llama.cpp
.venv/bin/python tools/gguf/check_attn.py --llama-cpp third_party/llama.cpp
ctest --test-dir third_party/llama.cpp/build -R 'test-tokenizer-0|test-generate-models'
third_party/llama.cpp/build/bin/test-llama-archs
```

Network access:
- The tokenizer tools download only `config.json`, `tokenizer.json` and
  `tokenizer_config.json`. They use the pinned revision and verify each file's
  sha256 against `inventory/bf16/summary.json`.
- Regenerating the inventory needs network access but downloads no tensor
  payloads; see the Phase 1 doc.

## Known upstream issues

These belong upstream, not in the Kolibri patches. Details are in
`docs/phase2-tokenizer.md`.

- **Invalid UTF-8 can crash `llama_tokenize`.** Some inputs abort the whole
  process, for example the 4 bytes `F4 90 80 80` (U+110000). This affects
  every model that uses llama.cpp's byte-level BPE tokenizer.
- **`tests/test-tokenizer-random.py` is broken:**
  - its `LibLlamaModel` uses the pre-`llama_vocab` API;
  - under transformers 5 its word lists collapse into one string.
