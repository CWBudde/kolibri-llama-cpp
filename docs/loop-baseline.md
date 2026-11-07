# Loop baseline

The reproducible baseline of Kolibri-1's repetition loops, step 1 of issue #66. It is a measurement: no weight is edited, and nothing here suggests a fix. Steps 2 to 5 of the issue (activation patching, weight edits) are not started.

**Status: the corpus, the tool and the definitions are committed. Results are pending; see [Results](#results-pending).**

| Issue metric | State |
|---|---|
| Repetition rate, time to loop, unit length | done, from the generated token IDs |
| Answer accuracy | done, on prompts with a known answer |
| Perplexity on held-out text | done, from the server's full distribution |
| Reference logits | not done; see [Not done](#not-done) |

## The real run

Only one process may hold the real weight at a time. Run this from the worktree, or from any checkout that has the tool, when nothing else holds it:

```sh
cd /Users/christian/Code/kolibri-llama-cpp/third_party/wt-loops
go run ./cmd/kolibri-loops run \
  -out ~/models/eval/loop-baseline/iq3-cache128 \
  -server /Users/christian/Code/kolibri-llama-cpp/third_party/llama.cpp/build/bin/llama-server \
  -model ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf \
  -- -ngl 99 -t 10 -c 4096 -np 1 --fit off --jinja --expert-cache 128

go run ./cmd/kolibri-loops report -out ~/models/eval/loop-baseline/iq3-cache128 \
  -json ~/models/eval/loop-baseline/iq3-cache128/report.json
```

`run` hashes the weight first, so it reads the whole file before the server starts. The same command continues an interrupted run.

### What a run on this weight does not replace

**A run on `Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf` does not replace BF16. The BF16 run remains a follow-up.** The file is quantized, and issue #66 asks to compare against BF16 and the reference before quantizing.

`--expert-cache 128` keeps the 33 GiB from being resident, and it is lossless in the sense of [expert-streaming.md](expert-streaming.md): the logits are bit-identical to resident Metal **at the same ubatch**. The cache caps the ubatch at 21, while the documented runs of the prompts below used the default ubatch, and tokens from the two settings differ: "0 of 65 requests are the same" against resident IQ3 at the default ubatch, with a first difference at token 12 to 22. So a documented control can come out differently here without anything being wrong. That is why [the control table](#corpus) informs and does not gate.

## Method

`cmd/kolibri-loops run` starts one `llama-server` (`-server`, `-model`, and the flags after `--`) or attaches to a running one (`-url`), and sends the corpus one prompt at a time. `cmd/kolibri-loops report` recomputes every number from the records. It reads no corpus, no weight and no server.

### Decoding

Every sampler setting is explicit, so nothing depends on the server's defaults: `samplers: ["temperature"]` at temperature 0, `seed` 42, every penalty neutral.
- **`n_predict`:** 256, and 768 with thinking on, because the reasoning has to close before an answer can be scored.
- **`cache_prompt` off:** batch sizes can change logits, so each prompt is decoded the same way whatever ran before it.
- **`post_sampling_probs` off:** the log probabilities are the softmax of the raw logits.
- **`n_probs`:** 10 per generated token.

### Prompts

- **Raw:** the text goes through `/tokenize` with `add_special: false, parse_special: true`, as in the documented runs: no BOS, and `<|endoftext|>` is a token.
- **Chat:** the user message goes through `/apply-template`, then the same `/tokenize`. `chat` sends `reasoning_effort: "none"`; `chat-think` leaves thinking on. The tool checks the property of the reference: the prompt ends in the prefilled empty think block if and only if thinking is off.
- **Raw, chat and chat-think are never pooled.** Every table has them in separate rows, per split and language.

The request itself sends token IDs. The two prompts with documented IDs carry `expect_tokens`, and a run stops if the server tokenizes them differently.

### Records

One JSON file per prompt, `records/<id>.json`, written to a temp file and renamed, so a record is complete or absent. Each holds:
- `pins`: the realpath, size and SHA-256 of the weight; the realpath and SHA-256 of the server binary and its arguments; the server's `build_info`; the chat template in full and its SHA-256; the vocabulary size; the corpus's SHA-256; the settings and the loop rule;
- the corpus item, the exact `/apply-template`, `/tokenize` and `/completion` request bodies, and the rendered prompt;
- the settings the server says it resolved, the prompt and generated token IDs, the text, the stop type, and the top candidates per token.

No derived number is stored, so a report cannot disagree with its records.

### Resume and refusals

A second run with the same `-out` continues after the last finished record. It refuses, naming the field, if anything in the pins moved. Weight, binary, corpus and settings are compared before a server starts; template, build and vocabulary right after it answers. It also refuses a record whose pins or corpus item differ, and a record it cannot parse.

The tool refuses to write a record if the response is not what was asked: other resolved settings, a truncated prompt, a different prompt length, a token that is not the most likely one, or a stop type other than limit or EOS. The report applies the same checks again. With `-url`, the binary hashed is the one `-server` names, and the tool cannot verify that it is the one serving; the served model path is checked. Prefer starting the server.

**One server behavior the records have to carry.** llama-server lists no probability for a token that ends in an incomplete UTF-8 sequence, while `tokens` has it. A record therefore has a `null` in `probs` at that position, and the report counts these. This was found on the tiny checkpoint, where the fake server had not shown it.

## The loop rule

Fixed in `FixedRule` (`detect.go`) before any real run. It must not move afterwards; a resume refuses a changed rule.

A generation **loops** if and only if all of these hold:
1. It stopped at `n_predict`. A generation the model ended itself escaped, whatever it repeated before.
2. A region up to its **last token** holds a period `p`, that is `g[i] == g[i-p]` for every `i` from `start+p` to the end, with `p` at most 64.
3. The region has at least 16 tokens and at least 3 whole units.

Among the periods that qualify, the earliest start wins, then the shortest period.

- **Repetition rate:** looping generations over generations, per split, form and language.
- **Time to loop:** `start`, the index in the generated tokens, 0 for the first one. The prompt does not count.
- **Unit length:** `p`, in tokens.

It compares token IDs, so the same word in two tokenizations does not match. What it does not count, each with a test:
- a repeat that stops before the last token, even a long one;
- a template with changing slots, as in "The capital of Spain is Madrid. The capital of Portugal is Lisbon.";
- one token short of the region, or under 3 whole units;
- a unit longer than 64 tokens;
- a loop the last token breaks.

## Answer accuracy

Only items with a known answer are scored. A term matches as a whole word, ignoring case, so "Berlin" does not match "Berliner", and "12" does not match "112".
- **correct:** the first accepted or rejected term in the text is an accepted one;
- **wrong:** a rejected term comes first, or no term occurs;
- **unanswered:** nothing was generated; with thinking on, also when `</think>` never comes. With thinking on, only the text after the last `</think>` counts.

A generation can be correct and loop afterwards. The two are reported apart.

## Held-out loss

The `text` items are short passages written for this corpus. For each token after the first, the tool asks for the distribution given the tokens before it (`n_predict` 1, `n_probs` above the vocabulary) and stores the target's log probability.

- **Honest because:** the server returns the softmax over the whole vocabulary, and the tool refuses a position with fewer candidates than `n_vocab`. It never computes a bound from a partial list. A target with probability 0 leaves the perplexity empty, and the report says how many.
- **The ban:** at `n_predict` 1 the server lists nothing if the generated token is a UTF-8 fragment. The tool then asks again with that token banned. The distribution comes from the raw logits, so the ban does not change it. On the tiny checkpoint the 128000 candidates were identical with and without a ban. The report counts the bans.
- **Perplexity:** `exp(mean negative log probability)`, per split and language.
- **Not comparable with the PPL in [real-checkpoint.md](real-checkpoint.md):** other texts, other lengths, another tool. It is meant for before and after comparisons on this corpus.

## Corpus

`testdata/loops/corpus.jsonl`, 112 items in 32 pairs. A pair is one fact or topic, and it holds one item per language and form:

| Stratum | Pairs | Items per pair | Content |
|---|---:|---:|---|
| `documented-loop` | 5 | 4 (raw and chat, de and en) | contains a prompt documented as looping |
| `documented-non-loop` | 3 | 4 | contains a prompt documented as not looping |
| `answer-known` | 12 | 4 | a fact with a known answer |
| `open` | 4 | 4 | no known answer, nothing documented |
| `thinking` | 4 | 2 (`chat-think`, de and en) | thinking on |
| `text` | 4 | 2 (`text`, de and en) | held-out loss |

**Documented controls.** `control` and `source` mark the prompts [real-checkpoint.md](real-checkpoint.md) and `testdata/e2e/corpus.json` document, and a test checks that each prompt is verbatim in its source. They are the sections "Raw German completions repeat, BF16 included" and "Raw prompts past the first sentence".
- **Looping:** four German raw prompts, plus both `<|endoftext|>` prompts, which also give the one documented English loop.
- **English looping controls are scarce.** Only that one is documented, so the English counterparts of the German prompts count as balance, not as controls.
- **Other weights:** the IQ3 column applies to most; `think-sky.de.chat-think` was documented on the Q3_K mix, and `moe-explain.de.chat` is the e2e corpus's `de-chat`.
- **Short documented runs:** the table's runs stopped at 40 tokens, and a run here is 256. A prompt documented as not looping can still loop later.

### The split

Fixed now, so later intervention work cannot pick its test set after the fact. `split` is on every line, and `validateCorpus` recomputes it:

> Within a stratum, order the pairs by the SHA-256 of `kolibri-loops/split/v1/` + name. They take turns, tuning first. A whole pair goes to one split, so a fact never sits on both sides.

Looping controls land in both splits. The corpus's hash is pinned in `main_test.go`, so a later edit to a prompt or a split changes that line in plain sight.

## Negative controls

Each mutation must make a check fail. All of them run in `go test ./cmd/kolibri-loops`, against a fake server whose shapes were read from a real llama-server.

| Control | Mutation | What catches it |
|---|---|---|
| M72 | The detector knows only period 1 | It misses the period-2 fixture the fixed rule finds |
| M73 | A looser rule (2 units, 4 tokens) | It counts near-misses the fixed rule rejects |
| M74a | The server's generated token is not the most likely one | "not greedy"; no record |
| M74b | The server lists only part of the distribution for a text | "the vocabulary has"; no text record, and no bound passed off as a loss |
| M74c | The tokenizer drifts | "the docs record"; no record |
| M74d | The server resolves another seed | "other settings"; no record |
| M74e | The template ignores the thinking mode | "think block"; no record |

The numbers start at M72 because M70 and M71 are taken by PR #72's controls in [upstream.md](upstream.md).

## What has been checked

- **`go test ./cmd/kolibri-loops`:** the detector's edge cases, answer scoring, the corpus, resume, refusals and the controls, against a fake server.
- **A real llama-server on tiny random weights** (`cmd/kolibri-tiny`, the real tokenizer and chat template):
  - all 112 prompts through to a report;
  - the documented token IDs matched;
  - a changed setting was refused;
  - a finished run sent no generation requests;
  - a second run gave identical tokens and identical loss positions in every record;
  - `--expert-cache 128`, `--fit off` and `-ngl 99` on the 384-expert fixture ran 3 prompts.

  This shows the tool works, and nothing about loops: random weights say nothing about Kolibri-1.
- **Not checked:** the real weight. No run on it has happened yet.

## Not done

- **Reference logits.** `llama-server` reports the model under test; the reference is `tools/ref/kolibri_ref.py` or vLLM on BF16, which do not run through its API. The records keep the top 10 log probabilities per generated token for a later comparison, but those are this run's, not the reference's, and they are not reported as such. The report's "p(first)" and "p(in loop)" are also this run's.
- **BF16.** See [above](#what-a-run-on-this-weight-does-not-replace).
- **Tool use.** The issue asks for it at validation, step 4; the corpus has reasoning but no tools.

## Results (pending)

**No result exists yet.** This section gets its tables from `kolibri-loops report` once the real run has happened, with the run's pins. The definitions above do not change for it.
