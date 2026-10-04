# Kolibri-1 tokenizer in llama.cpp

The shipped Kolibri-1 tokenizer is an ordinary byte-level BPE tokenizer.
llama.cpp's existing `gpt2` vocab type represents it exactly, together with
the existing `QWEN2` pre-tokenizer. No tokenizer runtime code is needed.
Kolibri needs only a name registration: the converter maps Kolibri's pre-tokenizer checksum to `kolibri`,
and `llama-vocab.cpp` maps `kolibri` to `LLAMA_VOCAB_PRE_TYPE_QWEN2` and declares
that the model has no BOS token.

Token IDs match the reference tokenizer on:

- all 85 golden cases;
- the 46 upstream test strings;
- every vocab entry;
- about 1.35 million differential fuzz strings.

Detokenization reproduces the input byte for byte.

The UniBPE training algorithm does not matter at inference time. The released
`tokenizer.json` is a standard BPE merge table, and both libraries apply it the
same way.

## Pinned sources and versions

| What | Version |
|---|---|
| Tokenizer | `Aleph-Alpha/Kolibri-1-BF16@7a8f290e7858825c3cf5e4c447ba68345de9f1d3`, `tokenizer.json` sha256 `5d4798f2…4c13` (identical in the FP8 repo) |
| Reference stack | `tokenizers` 0.23.2 and `transformers` 5.18.0; the reference repo requires transformers ≥ 5.5.3, and vLLM tokenizes through it |
| llama.cpp | `1537a0a8b2f8711d840878b0a0677ab2213c882c` plus `patches/llama.cpp/0001-kolibri-tokenizer.patch` |

`tools/tokenizer/requirements.txt` pins the Python environment.

## `tokenizer.json` and how llama.cpp represents it

| Component | Kolibri | llama.cpp / GGUF | Same behavior? |
|---|---|---|---|
| Model | `BPE`; 127,900 vocab entries; 127,644 merges (pairs); `ignore_merges: false` | `tokenizer.ggml.model = gpt2`; merges ranked in file order. The `QWEN2` pre-type leaves `ignore_merges` off. | yes |
| `byte_fallback` | `true` | not represented | Never triggers: the vocab contains all 256 byte-level symbols and no `<0xNN>` tokens. |
| Normalizer | none | none | yes |
| Pre-tokenizer | `Split(Isolated)` with the regex below, then `ByteLevel(add_prefix_space=false, use_regex=false)` | `QWEN2` regex through the native `unicode_regex_split_custom_qwen2` fast path, then byte-level mapping | yes; the regexes are identical, see below |
| Post-processor | `ByteLevel(add_prefix_space=true, trim_offsets=false)` | n/a | It only adjusts offsets and never changes IDs. |
| Decoder | `ByteLevel` | byte-level `token_to_piece` | Yes. `add_prefix_space` does not apply when decoding; every golden case round-trips. |
| BOS/EOS | no BOS (`bos_token: null`, `add_bos_token: false`); EOS `<\|im_end\|>` 127906; pad `<\|endoftext\|>` 127901; `generation_config` stops on 127906 and 127901 | `add_bos_token = false`, `eos_token_id = 127906`, `padding_token_id = 127901`. The `kolibri` branch sets `special_bos_id = LLAMA_TOKEN_NULL`, following the `chatglm`/`glm4` precedent; without it, llama.cpp's BPE default would make token 11 (`ċ`, byte `\x0b`) the BOS. llama.cpp detects both 127901 and 127906 as end-of-generation tokens. | yes |
| Clean-up | `clean_up_tokenization_spaces: false` | `QWEN2` sets `clean_spaces = false` | yes |

**Regex.** Kolibri's split regex is

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

`\p{N}{1}` is `\p{N}`, so this is exactly the original Qwen2 regex. llama.cpp
already carries that regex in the case-expanded form
`(?:'[sS]|'[tT]|…)`. llama.cpp uses the same regex for the `qwen2`,
`stablelm2`, `hunyuan`, `solar-open` and `grok-2` pre-types. `laguna` uses it
too, after an extra newline split.

**Checksum.** `get_vocab_base_pre` produces the new checksum
`6e040dfe72e4b85855588c53acf4909ac4e98e55bc3a33cd7db499180dc42a78`. The patch
maps it to the name `kolibri`, following upstream practice of naming even
pre-tokenizers that reuse an existing type. That keeps the door open if the
regex ever diverges.

## Added tokens: the IDs in the file are not the runtime IDs

The file lists 98 added tokens with IDs 127900…127999 and skips 127923 and
127924. **The `tokenizers` library ignores the `id` field.** In file order, it
gives each added token that is not in the base vocab the next free ID after the
vocab. So:

| Token | `id` in `tokenizer.json` | runtime ID (reference) |
|---|---|---|
| `<\|text\|>` … `<\|pii_9\|>` | 127900–127922 | unchanged |
| `<\|reserved-token-2\|>` … `<\|reserved-token-76\|>` | 127925–127999 | **127923–127997** |
| (unassigned) | 127923, 127924 | **127998, 127999** |

The reference serving stack therefore encodes `<|reserved-token-2|>` as 127923.
llama.cpp's converter already does the same: it builds the token list from
`AutoTokenizer.vocab`, which reflects the runtime IDs, and writes
`[PAD127998]` and `[PAD127999]` as `UNUSED`. The golden case
`special/reserved-shifted` pins this behavior.

All chat, reasoning, tool and PII tokens come before the gap, so the shift only
affects reserved tokens, which the model never uses. The checkpoint inventory
(`summary.json`, see [checkpoint.md](checkpoint.md)) records both views: `runtime_id` per added token,
`file_id_gaps` and `runtime_unassigned`. A Go test (`cmd/kolibri-inventory`)
cross-checks them against the golden file.

**GGUF token types:**

- 127,900 `NORMAL`;
- 92 `CONTROL`: the `special: true` tokens;
- 6 `USER_DEFINED`: `<think>`, `</think>`, `<tool_call>`, `</tool_call>`,
  `<tool_response>`, `</tool_response>` (`special: false`);
- 2 `UNUSED`: the padding.

**Special-token matching.** The reference always matches added tokens in the
input text (`split_special_tokens: false`). llama.cpp matches `USER_DEFINED`
tokens always, but `CONTROL` tokens only with `parse_special = true`. That is
the mode llama.cpp uses for chat-templated prompts, and the comparison runs use
it. So user-supplied text containing `<|im_end|>` is tokenized as a control
token by both stacks (relevant for chat, see [chat.md](chat.md)).

**False-positive warning.** transformers ≥ 5 logs "incorrect regex pattern …
set `fix_mistral_regex=True`" when it loads this tokenizer from a local
directory. The Mistral heuristic fires for any `config.json` without a
`transformers_version` key, and Kolibri's config has none. Unless the flag is
passed explicitly, nothing changes, and vLLM does not pass it. The shipped regex
is the reference. The tools pass `fix_mistral_regex=False` to silence the
warning.

## Golden tests

`testdata/tokenizer/golden.jsonl` holds 85 cases. `tools/tokenizer/golden.py`
generates them with `tokenizers`, and every case is cross-checked against
`AutoTokenizer.encode`. Each record stores the text, the reference IDs, and the
reference decode. For every case the decode equals the input.

| Group | Cases | Covers |
|---|---|---|
| `german` | 9 | `Bundessozialgerichtes`, `Protokolldaten`, lower- and upper-case variants, long compounds, hyphenated compounds, sentences |
| `umlauts` | 6 | ä/ö/ü/ß/ẞ, `Straße`/`STRASSE`, decomposed (NFD) umlauts (no normalizer!), other Latin diacritics |
| `english` | 7 | contractions in mixed case, apostrophes, numbers (single-digit split), punctuation, URL/email |
| `whitespace` | 19 | empty, spaces, tabs, `\n`, `\r\n`, `\r`, trailing/leading/inner runs, NBSP and other Unicode spaces, U+2028/2029/0085, a 600-space run |
| `unicode` | 19 | emoji with ZWJ, flags and skin tones; CJK; Arabic and Hebrew; Indic; Thai and Khmer; Cyrillic and Greek; combining marks; math; invisible and bidi controls; U+FFFD; private use; astral planes; noncharacters; C0 controls; non-ASCII digits |
| `code` | 10 | Python, Go, C++, JSON, HTML, shell, SQL, regex, operators, deep indentation |
| `special` | 13 | `<\|im_start\|>`/`<\|im_end\|>`, adjacent specials, `<think>` inline and inside words, tool call/response, role and PII tokens, shifted reserved tokens, nonexistent reserved tokens, specials surrounded by whitespace, partial and look-alike specials |
| `chat` | 2 | the released chat template rendered by `apply_chat_template`, simple and with tools, a reasoning trace, a tool call and a tool result |

`tools/tokenizer/compare.py` loads `libllama` through cffi with
`vocab_only=true`. Results with the patched llama.cpp:

| Check | Result |
|---|---|
| golden: token IDs | 85/85 identical |
| golden: `llama_detokenize` bytes equal the reference decode | 85/85 |
| upstream `test-tokenizer-0-kolibri` (46 standard strings, `.inp`/`.out` written as `convert_hf_to_gguf_update.py` would) | passed |
| fuzz: llama.cpp's `custom_text` and `custom_text_edge_cases` | 40 + 26 identical |
| fuzz: `ascii_lr_strip`, `apostrophe` | 442,368 + 442,368 identical |
| fuzz: every Unicode code point < U+30000 (assigned, non-private), one per string | 139,783 identical |
| fuzz: every vocab entry, decoded, as input text | 127,193 identical |
| fuzz: added tokens with whitespace around them / random sequences of 500 added tokens | 25,088 / 10,000 identical |
| fuzz: random texts from `random_chars`, `random_unicodes` and `random_vocab_chars` (1,024 vocab characters each) | 10,000 + 10,000 + 10,000 identical |
| fuzz: `random_vocab_words`: every stripped vocab word, plus 5,000 texts of 300–400 random word groups | 132,193 identical |

Each count is a number of strings. A string passes only if llama.cpp produces
the reference IDs **and** detokenizing those IDs gives back the input bytes. As
a negative control, the same runner on another BPE vocab (`ggml-vocab-qwen2.gguf`)
fails on the first golden case and exits with status 1.

The patch also leaves the other 15 `test-tokenizer-0-*` tests passing.

### Invalid UTF-8

The reference cannot receive invalid UTF-8: vLLM tokenizes Python `str`, and
JSON requests are valid Unicode. Exact equivalence is therefore not defined.
`compare.py` reports llama.cpp's behavior instead, comparing it with the
reference applied to Python's `errors="replace"` decoding:

| Input | llama.cpp | Reference on replaced text |
|---|---|---|
| truncated sequences (`\xc3`, `abc\xe2\x82`, `\xf0\x9f\x9a`) | one U+FFFD **per byte** | one U+FFFD per maximal subpart; IDs differ |
| stray continuation bytes, `\xff`, Latin-1 text | U+FFFD per byte | same; IDs identical |
| overlong `\xc0\xaf` | **decoded as `/`** | U+FFFD U+FFFD |
| surrogate `\xed\xa0\x80` | encoded as three raw bytes | U+FFFD ×3 |
| `\xf4\x90\x80\x80` (U+110000) | **process aborts**: uncaught `std::invalid_argument("invalid codepoint")` through `llama_tokenize` | U+FFFD |

This is a generic llama.cpp bug, not a Kolibri incompatibility.
`unicode_cpt_from_utf8` does not reject overlong forms, surrogates or code
points above U+10FFFF, and `unicode_cpt_to_utf8` later throws. It affects every
BPE model and belongs in a separate upstream issue or PR, not in the Kolibri
patch. llama-server is only exposed if invalid UTF-8 can reach the tokenizer
without passing through its JSON parser.

## Issues in llama.cpp's own test tooling

Both were worked around in `compare.py`; neither affects the Kolibri patch.

- `tests/test-tokenizer-random.py`: `LibLlamaModel.tokenize` passes a
  `llama_model *` where `llama_tokenize` now takes a `llama_vocab *`, so the
  script fails at the first call.
- `TokenizerGroundtruth`: under transformers 5, `batch_decode` of a flat ID list
  returns one string. As a result, `vocab` and `added_tokens` each collapse into
  a single concatenated entry. The vocab and added-token fuzzers silently test
  one huge string instead of 128k words.

## The llama.cpp patch

The tokenizer part is `patches/llama.cpp/0001-kolibri-tokenizer.patch`, the
first of the series against `1537a0a8`:

- `conversion/base.py`: checksum `6e040dfe…` → `res = "kolibri"`;
- `convert_hf_to_gguf_update.py`: registers `kolibri` with
  `https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16`;
- `src/llama-vocab.cpp`: a `kolibri` branch: `LLAMA_VOCAB_PRE_TYPE_QWEN2`,
  `clean_spaces = false`, `special_bos_id = LLAMA_TOKEN_NULL`;
- `tests/CMakeLists.txt`: `test-tokenizer-0-kolibri`;
- `models/ggml-vocab-kolibri.gguf.{inp,out}`: the upstream test strings and the
  reference IDs.

## Vocab GGUF

The patches leave out the 4.8 MB `models/ggml-vocab-kolibri.gguf`.
`tools/tokenizer/vocab_gguf.py` regenerates it with the converter's own Kolibri
class (`get_model_class("Kolibri1ForCausalLM")`, then `write_vocab()`). That is
`convert_hf_to_gguf.py --vocab-only` plus a fixed `general.name`: run on the HF
cache, the CLI would record the revision SHA as `general.name` and
`general.finetune`.

The file has `general.architecture = kolibri`, the 21 `kolibri.*`
hyperparameter keys, `general.file_type = 0`, the tokens, merges, token types,
special IDs and the chat template. libllama skips the hyperparameters with
`vocab_only=true`, but needs the `kolibri` model class to accept the file at
all. `test-tokenizer-0-kolibri` passes with the existing `.inp`/`.out`,
`compare.py` matches on all fuzz sets, and regenerating gives a byte-identical
file.

## Reproducing

```sh
# llama.cpp at the pinned commit, with the patches
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git -C third_party/llama.cpp checkout -b kolibri 1537a0a8b2f8711d840878b0a0677ab2213c882c
for p in patches/llama.cpp/*.patch; do git -C third_party/llama.cpp apply "$PWD/$p"; done
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=ON
cmake --build third_party/llama.cpp/build --target llama test-tokenizer-0

# Python environment
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install --index-strategy unsafe-best-match -r tools/tokenizer/requirements.txt

.venv/bin/python tools/tokenizer/golden.py --check          # golden file vs. installed reference
.venv/bin/python tools/tokenizer/vocab_gguf.py --llama-cpp third_party/llama.cpp \
    --out third_party/llama.cpp/models/ggml-vocab-kolibri.gguf
third_party/llama.cpp/build/bin/test-tokenizer-0 third_party/llama.cpp/models/ggml-vocab-kolibri.gguf
.venv/bin/python tools/tokenizer/compare.py --llama-cpp third_party/llama.cpp \
    --vocab third_party/llama.cpp/models/ggml-vocab-kolibri.gguf --iterations 10000
```

The tools download only `config.json`, `tokenizer.json` and
`tokenizer_config.json`, at the pinned revision, and verify their sha256 against
`inventory/bf16/summary.json`.
