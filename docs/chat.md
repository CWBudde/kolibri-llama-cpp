# Kolibri chat template, reasoning and tool calls

This document covers what llama.cpp does with Kolibri's chat conventions:

- the Kolibri chat template, preserved byte for byte;
- the reasoning modes none / low / medium / high;
- the tool-call format;
- the official reasoning and tool parsers compared with llama.cpp's;
- the stop tokens and EOS.

None of it needs the checkpoint. Aleph Alpha's recommended sampling
(`temperature=1.0`, `top_p=0.97`, `top_k=128`) is not covered here: it is
validated only once base greedy inference matches the reference.

Two checks cover it:

- **`tools/chat/check_chat.py`** checks what reaches the model: the template,
  the prompt llama-server renders for each request, and the stop tokens.
- **The fork's `tests/test-chat.cpp`** (`ctest -R test-chat`) checks what
  comes back: how llama.cpp splits generated text into reasoning, content and
  tool calls.

Kolibri needed one line of shared llama-server code (see
[The `reasoning_effort: "none"` fix](#the-reasoning_effort-none-fix)), plus the
template file and tests. Everything else already worked.

## The reference

- **[`Aleph-Alpha/aleph-alpha-inference@049a6a7`](https://github.com/Aleph-Alpha/aleph-alpha-inference/tree/049a6a7bd2405b27d6d280d256bd3d585191c7ae):**
  - `README.md` serves the model with `--reasoning-parser kolibri1
    --tool-call-parser kolibri1 --enable-auto-tool-choice`. Thinking is on by
    default and is turned off with `reasoning_effort: "none"` or
    `enable_thinking: false`.
  - `aleph_alpha_inference/reasoning.py`:
    - `thinking_enabled(kwargs)`: a non-null `reasoning_effort` decides
      (`!= "none"`); otherwise thinking is on unless `enable_thinking is False`.
    - `Kolibri1Parser` is vLLM's `Qwen3Parser`, starting in the reasoning state
      if and only if `thinking_enabled`.
  - `aleph_alpha_inference/__init__.py` registers the `kolibri1` tool parser as
    vLLM's `Hermes2ProToolParser`: JSON in `<tool_call>…</tool_call>`.
  - `tests/test_reasoning.py`: 10 template-argument sets
    (`THINKING_OFF_KWARGS`, 4, and `THINKING_ON_KWARGS`, 6), the property "the
    prompt ends in the prefilled empty think block if and only if thinking is
    off", streaming and non-streaming splits, and the single-turn and
    tool-loop message sets.
  - `tests/kolibri1_chat_template.jinja`: the released template. Without its
    3-line header comment it equals `chat_template` in `tokenizer_config.json`.
- **vLLM v0.29.0:**
  - `vllm/entrypoints/openai/chat_completion/protocol.py`,
    `build_chat_params`: the OpenAI `reasoning_effort` field overrides
    `chat_template_kwargs` (`renderers/params.py`, `merge_kwargs`). Unless the
    request sets `enable_thinking`, it also sets `enable_thinking = effort !=
    "none"`.
  - `chat_utils._postprocess_messages`: tool-call arguments reach the template
    as objects.
- **The template renderer:** Hugging Face transformers 5.18
  `render_jinja_template`, the renderer `tests/test_reasoning.py` uses.

## The template

sha256 `9ba35d4b…7b0f97`, 6236 characters. What it does:

- **System block:** always emitted, with a `# Reasoning effort` sentence:
  - **none:** "Reasoning is disabled."
  - **low:** `minimal`, `low`.
  - **medium:** `medium`.
  - **high:** `high`, `xhigh`, `max`, unset, and any other value.
  - The system message and the `# Tools` block go into the same block.
- **Thinking off:** `reasoning_effort == "none"`, or, only when
  `reasoning_effort` is unset, `enable_thinking is false`. The generation
  prompt is then `<|im_start|>assistant\n<think>\n\n</think>\n\n`. With
  thinking on it stops at `<|im_start|>assistant\n`, and the model opens
  `<think>` itself.
- **Earlier assistant turns:**
  - An assistant turn after the last user query (that is, in a tool loop) gets
    a `<think>` block, empty if there is no reasoning. With
    `preserve_thinking=true`, every assistant turn gets one.
  - Reasoning comes from `message.reasoning`, else `reasoning_content`, else
    the text before `</think>` in the content.
- **Tool calls:** `<tool_call>\n{"name": "…", "arguments": …}\n</tool_call>`,
  with a newline between content and the first call and between calls.
- **Tool results:** consecutive tool results are grouped into one
  `<|im_start|>user` turn of `<tool_response>` blocks.

Where llama.cpp keeps it:

- The converter's Kolibri class writes it into `tokenizer.chat_template` from
  `tokenizer_config.json` (`gguf.SpecialVocab`). The vocab GGUF and every
  converted GGUF carry it.
- The fork also has it as `models/templates/Aleph-Alpha-Kolibri-1.jinja`,
  where llama.cpp's chat tests read templates from.

### How llama.cpp handles it

- **Engine:** llama.cpp renders with its own Jinja engine (`common/jinja`).
  `chat_template_kwargs` become template variables, and `enable_thinking` is
  always set, from `--reasoning` or the request.
- **Format detection:** no template-specific handler matches, so the
  differential autoparser analyses the template (`common/chat.cpp:1341`).
  `test-chat` prints `peg-native` for it, with:
  - tag-based reasoning on `<think>`/`</think>`;
  - JSON tool calls in `<tool_call>` markers;
  - a lazy grammar triggered on `<tool_call>`.
- **Capabilities:** the caps probe reports `supports_reasoning_effort`.
- **llama-server request handling** (`tools/server/server-common.cpp`,
  `oaicompat_chat_params_parse`):
  - request `chat_template_kwargs` are merged over `--chat-template-kwargs`;
  - an `enable_thinking` kwarg sets the parser's thinking flag;
  - the OpenAI `reasoning_effort` field becomes the `reasoning_effort`
    variable, except `"none"`, which turns thinking off.
- **`--reasoning-preserve` is on by default.** It sets `preserve_thinking=true`,
  so earlier assistant turns keep their reasoning. vLLM's default renders
  without it.

## Check (`tools/chat/check_chat.py`)

### 1. Template preserved

The sha256 of the template, compared across:

- `tokenizer_config.json` of the pinned revision;
- the vocab GGUF;
- the GGUF `convert_hf_to_gguf.py` writes for the `cmd/kolibri-tiny`
  checkpoint;
- the fork's template file;
- llama-server's `/props`.

### 2. Render parity

- **Setup:** llama-server serves the converted tiny GGUF. Each request goes to
  `/apply-template`, which runs the same request handling as
  `/v1/chat/completions` without inference.
- **The reference:** the template rendered by transformers with the
  arguments vLLM 0.29 derives from the same request (`vllm_template_kwargs`,
  a port of `build_chat_params`). Tool-call arguments are passed as objects.
- **The comparison:** the two prompts must be byte-identical. The check also
  asserts the reference property on llama-server's prompt: it ends in the
  prefilled empty think block if and only if `thinking_enabled()` is false.

**Message sets (7):**
- the reference's `single` and `tool_loop`;
- a system message;
- a `tools` list;
- content plus two parallel tool calls and two tool results, with non-ASCII
  arguments;
- `reasoning_content` in the history;
- three turns with a system message.

**Requests per message set:**

| Server flags | Requests | Count |
|---|---|---|
| default (`--reasoning-preserve`) | the reference's 10 `chat_template_kwargs` sets; `reasoning_effort` = none, minimal, low, medium, high, xhigh, max; each of those 7 levels with `enable_thinking` true and false | 31 |
| `--no-reasoning-preserve` | the same | 31 |
| `--reasoning off` | default request, reference `enable_thinking=false` | 1 |
| `--reasoning-effort low` | default request, reference `reasoning_effort="low"` | 1 |

The default server is compared with `preserve_thinking=true` on the reference
side, and `--no-reasoning-preserve` without it. The `history_reasoning` set
renders differently between the two, so both settings are checked.

### 3. Stop tokens

Both GGUFs are loaded vocab-only in libllama and checked for:
- the EOG set over all 128000 ids;
- the eos, eot and pad ids;
- how the tag and control tokens render.

### Result

Full check: `check_chat.py` rc 0, **37 PASS**. The template line:

```
PASS chat template: tokenizer_config.json 9ba35d4bd6ba, vocab GGUF 9ba35d4bd6ba, converted GGUF 9ba35d4bd6ba, models/templates/Aleph-Alpha-Kolibri-1.jinja 9ba35d4bd6ba, llama-server /props 9ba35d4bd6ba (released 9ba35d4bd6ba, 6236 chars)
```

Two of the 28 render lines (the rest have the same shape):

```
PASS chat render [default] tool_loop: 31/31 prompts equal the reference, prefilled think block iff thinking off 31/31
PASS chat render [--no-reasoning-preserve] parallel_calls: 31/31 prompts equal the reference, prefilled think block iff thinking off 31/31
```

The stop-token lines (each also for the converted GGUF):

```
PASS chat stop tokens [vocab GGUF]: libllama EOG tokens over all 128000 ids [127901, 127906] (generation_config.json eos_token_id [127901, 127906])
PASS chat stop tokens [vocab GGUF]: eos 127906, pad 127901 (tokenizer_config.json: 127906, 127901), eot 127906 (one of [127901, 127906]; the GGUF has no eot key, libllama picks it by token text)
PASS chat stop tokens [vocab GGUF]: tag tokens render as text ['<think>', '</think>', '<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>'], none is EOG or control
PASS chat stop tokens [vocab GGUF]: control tokens render as '' without special, as their text with special (127901: '<|endoftext|>', 127904: '<|im_start|>', 127906: '<|im_end|>')
```

## Parser tests (fork `tests/test-chat.cpp`)

**A Kolibri `peg_tester` block** (in the style of the Reka-Edge block). Each
case streams the input one prefix at a time and checks:
- the accumulated diffs;
- the final message;
- the grammar and its lazy trigger.

The cases:

- thinking on: `<think>\nI'm\nthinking\n</think>\n\n…` gives reasoning
  `I'm\nthinking` and the content; also after a tool loop;
- `reasoning_format none` keeps the think block in the content;
- output cut off inside the think block (`<think>\nstill`, partial) is
  reasoning;
- thinking off: the output is content; also after a tool loop;
- a tool call after the reasoning; a tool call with thinking off; content plus
  two parallel calls; a partial call while streaming;
- a `<tool_call>` inside the reasoning stays reasoning.

**`test_kolibri_reasoning_effort`:**
- **The reference's 10 kwarg sets,** for the single-turn and tool-loop message
  sets:
  - `enable_thinking` is set the way llama-server derives it from the kwargs;
  - the prompt ends in the prefilled block if and only if thinking is off;
  - with thinking on, `<think>\nthinking\n</think>\n\nThe answer.` parses to
    reasoning `thinking` plus content `The answer.`; with thinking off,
    `The answer.` parses to content.
- **The system sentence** for each of the 7 effort levels.
- **Through `oaicompat_chat_params_parse`:** the `reasoning_effort` field with
  an `enable_thinking` kwarg (`none`+true, `none`+false, `low`+false,
  `high`+true).

**Elsewhere:**
- Kolibri expects `true` in `test_reasoning_effort_caps`.
- Kolibri has a row in `test-chat-auto-parser`'s role-marker table.

Result in the full check: `ctest -R 'test-tokenizer-0|test-generate-models|test-chat'`
gives "100% tests passed out of 21": the 17 tokenizer tests and the 4 chat
tests (`test-chat`, `test-chat-peg-parser`, `test-chat-auto-parser`,
`test-chat-template`).

## The `reasoning_effort: "none"` fix

**The mismatch:** without the fix, `check_chat.py` fails 1/31 requests on
every message set:

```
FAIL chat render [default] single: 30/31 prompts equal the reference, prefilled think block iff thinking off 30/31; {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": true}}: at char 49: llama.cpp 't|>system\n# Reasoning effort\n\nReasoning effort is set to high. Think carefully t', reference 't|>system\n# Reasoning effort\n\nReasoning is disabled. Proceed straight to answeri'
```

**Why:**
- For `"none"`, `oaicompat_chat_params_parse` set the parser's
  `enable_thinking = false` and removed the `reasoning_effort` kwarg.
- An explicit `enable_thinking: true` kwarg still reached the template as a
  variable (`common/chat.cpp:925`, the extra context overrides
  `enable_thinking`).
- So Kolibri rendered a thinking prompt. vLLM lets the field win and renders
  thinking off.

**The fix** (shared code): `"none"` also sets the
`enable_thinking` kwarg to false, one line in `server-common.cpp`. For
templates that read only `enable_thinking`, nothing changes. For the
contradicting request, the template now sees what the server already assumed.

**Verification:**
- The new server cases in `test_kolibri_reasoning_effort` fail without the
  line ("Expected: 1", `test-chat` rc 134) and pass with it.
- With it, `check_chat.py` gives 31/31 on every message set.

## Official parsers vs llama.cpp

| Behaviour | vLLM (`kolibri1` parsers) | llama.cpp | Covered by |
|---|---|---|---|
| Thinking mode | `thinking_enabled(chat_template_kwargs)`: `reasoning_effort` wins over `enable_thinking` | the template decides, and the parser follows the rendered prompt, also when the request's `enable_thinking` is false but `reasoning_effort` is `low` | `test_kolibri_reasoning_effort` (10 kwarg sets × 2 message sets) |
| Thinking on | starts in REASONING; `<think>` is dropped; `</think>` switches to content | the same split | peg block, effort test |
| Thinking off | starts in CONTENT after the prefilled block | the whole output is content | peg block, effort test |
| Whitespace | returns `"\nlet me see\n"`-style text; the reference test compares after `.strip()`, and truncated output is `"\nstill"` | strips the template's newlines: reasoning `let me see`, truncated `still` | peg block (`<think>\nstill` → `still`) |
| `<tool_call>` before `</think>` | `Qwen3Parser` ends the reasoning at `<tool_call>` (`vllm/parser/qwen3.py:139`, "Tool call directly from reasoning (implicit end)") | stays reasoning until `</think>` | peg block (fake call in the reasoning) |
| Tool calls | `Hermes2ProToolParser`: one or more `<tool_call>` JSON blocks, content before the first call | the same format, parallel calls, content before them, streaming partial arguments | peg block |
| Structured output | the grammar applies once the reasoning has ended (`is_reasoning_end`, current turn only) | lazy grammar, triggered on `<tool_call>` | `test_peg_parser` builds the grammar and checks its triggers |
| `continue_final_message` | known gap, documented in `reasoning.py` | not tested | — |

Two differences remain, both in how output is split, not in the prompt:

- **Whitespace.** llama.cpp returns the reasoning without the template's
  newlines. The reference tests strip it as well.
- **A `<tool_call>` the model writes inside its reasoning.** vLLM ends the
  reasoning there; llama.cpp keeps it as reasoning until `</think>`. A
  well-behaved generation closes `</think>` before calling a tool, as the
  template renders it.

Neither is Kolibri code; both are documented and left as they are.

## Stop tokens and EOS

| Token | id | GGUF | libllama |
|---|---|---|---|
| `<\|im_end\|>` | 127906 | `eos_token_id` | EOG, also EOT |
| `<\|endoftext\|>` | 127901 | `padding_token_id` | EOG, detected by its text |
| `<\|im_start\|>` | 127904 | control | not EOG |
| `<think>` … `</tool_response>` | 127907–127912 | user-defined | text, not EOG |

- **Stopping matches HF.** `generation_config.json` stops on `[127906,
  127901]`, and libllama's EOG set is exactly those two. The converter writes
  only one eos id, but libllama adds `<|endoftext|>` by its text
  (`src/llama-vocab.cpp:2940`).
- **No EOT key.** libllama chooses the EOT id by token text from an unordered
  map, so it is either of the two. Both are EOG, so stopping does not depend
  on which. Writing an explicit eot id would pin it; that is a converter
  change and not needed for correct stopping.
- **Tag tokens are text.** The six tag tokens are not special in
  `tokenizer_config.json`, so they detokenize as text and reach the chat
  parser even with special tokens hidden. The control tokens render as
  nothing unless special rendering is on.

## Server flags

What the HF/vLLM default (`tests/test_reasoning.py`, `vllm serve`) means in
llama-server terms:

- `--jinja` (the default) renders the embedded template.
- **`--no-reasoning-preserve`:** renders earlier assistant turns exactly as
  vLLM's default does. llama-server's default `--reasoning-preserve` keeps the
  reasoning of every earlier turn (`preserve_thinking=true`). That is valid,
  checked against the reference with `preserve_thinking=true`, but it differs
  from vLLM's default prompt once the history has reasoning.
- **Reasoning mode per request:** `reasoning_effort` (`none` … `max`) or
  `chat_template_kwargs`, both checked above.
- **Server-wide reasoning mode:** `--reasoning off` or `--reasoning-effort
  LEVEL`.

## Mutation tests

Each was run, then reverted.

| Mutation | Result |
|---|---|
| M12: llama-server renders a copy of the template that checks `enable_thinking` before `reasoning_effort` (`--chat-template-file`) | FAIL on every message set: 24/31 prompts equal the reference. The 7 failing requests are the reference's `{"reasoning_effort": "low", "enable_thinking": false}` and the 6 levels other than none with `enable_thinking` false. The template sha from `/props` fails too. The CLI-default cases pass, because nothing contradicts there. |
| M13: vocab GGUF with eos moved to `<think>` (`gguf_new_metadata.py --special-token eos '<think>'`) | FAIL: EOG set `[127901, 127906, 127907]`, eos 127907, and `<think>` flagged as EOG. The unchanged converted GGUF stays PASS. |
| M14: the fork's template file without the thinking-off prefill | FAIL: `test-chat` in `test_kolibri_reasoning_effort` ("Expected: 1 Actual: 0", rc 134), and `check_chat.py`'s template sha. The peg block alone still passes, because without a prefill, content still parses as content. The prefill belongs to the effort test. |

The `reasoning_effort: "none"` fix was also checked in reverse: without the
line, its new server cases fail (see above).

## Verified behavior

- **Chat template preserved:**
  - The converter copies it into every GGUF. The fork adds it as a template
    file for llama.cpp's tests.
  - All six copies (including llama-server's `/props`) share sha256
    `9ba35d4b…`.
- **Reasoning modes:**
  - llama-server renders every reasoning mode exactly as the reference does.
    The modes come from the reference's 10 argument sets, 7 effort levels with
    and without `enable_thinking`, and 2 CLI defaults, on 7 message sets, with
    and without preserved reasoning.
  - The parser splits reasoning and content per mode.
  - One request shape needed the server fix above.
- **Tool-call formatting:** tools, tool loops, parallel calls and grouped tool
  results render byte-identically. llama.cpp parses the Hermes JSON calls
  (single, parallel, after reasoning, partial).
- **Official parsers:** the table above. Behaviour is the same except the two
  documented output-splitting differences. `continue_final_message` with
  thinking is a known gap in the reference parser itself and is not tested.
- **Stop tokens and EOS:** libllama stops on exactly the two eos ids of
  `generation_config.json`, and the tag tokens reach the parser as text.
- **Reference compatibility cases:** `test_kolibri_reasoning_effort` carries the
  reference's `THINKING_OFF_KWARGS`/`THINKING_ON_KWARGS` precedence cases and
  the tool-loop message set into `test-chat`.
