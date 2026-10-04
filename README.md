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
| 3 | GGUF architecture and HF → GGUF converter | in progress; arch registration and base metadata done: [docs/phase3-gguf-arch.md](docs/phase3-gguf-arch.md) |
| 4–9 | Model graph, hybrid KV cache, numerical validation, quantization, Apple Silicon, chat behavior | open |

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
  libllama; the model class itself is Phase 4.
- **The converter writes the base metadata.** `convert_hf_to_gguf.py` knows
  `Kolibri1ForCausalLM`. It writes block count, embedding size, head counts,
  head/RoPE dimension and RMSNorm epsilon. Tensor conversion is still to come.

## Layout

| Path | Contents |
|---|---|
| `cmd/kolibri-inventory` | Builds the tensor inventory from Hugging Face. It reads only the safetensors headers, through HTTP range requests, so no payload is downloaded. |
| `cmd/kolibri-peek` | Fetches individual small tensors through range requests and prints value statistics. |
| `internal/` | Hugging Face client, safetensors header parser, and the Kolibri tensor specs. `internal/kolibri/tensors.go` is the source of truth for the HF → GGUF mapping. |
| `inventory/{bf16,fp8}/` | Committed inventories: `summary.json` and `tensors.jsonl.gz`. They pin the model revisions and the sha256 of every file. |
| `testdata/tokenizer/golden.jsonl` | Tokenizer golden cases: IDs and decoded text from the reference tokenizer. |
| `tools/tokenizer/` | Golden-file generator, llama.cpp ↔ reference comparison (golden, fuzz, invalid UTF-8), vocab-only GGUF writer, pinned Python requirements. |
| `tools/gguf/` | `check_arch.py` checks the `kolibri` architecture registration in gguf-py and libllama. `check_metadata.py` checks the converter's GGUF metadata against `config.json`. |
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
