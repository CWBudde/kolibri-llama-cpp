# Loop baseline

The reproducible baseline of Kolibri-1's repetition loops, step 1 of issue #66. It is a measurement: no weight is edited, and nothing here suggests a fix. Steps 2 to 5 of the issue (activation patching, weight edits) are not started.

**Status: the corpus, the tool and the definitions are committed, and two runs on the quantized IQ3 weight, one per build, are reported under [Results](#results). BF16 and reference logits remain open.**

| Issue metric | State |
|---|---|
| Repetition rate, time to loop, unit length | done, from the generated token IDs |
| Answer accuracy | done, on the first sentence, with the occurrence rate beside it |
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

**Which build.** `-server` is whatever `build/bin` holds when the run starts. On 2026-11-07 that is `b11839-90311e5a2`, fork branch `feat/kolibri1-waves`: `feat/kolibri1` without its last commit, plus the Stage E4 waves code that left the patch series ([expert-streaming.md](expert-streaming.md#stage-e4-waves-for-prefill)). Waves need `--expert-cache-waves`, which the command does not pass, and on this build `--expert-cache 128` still logs "n_ubatch 512 capped to 21". A build without that code gave the same tokens in every record: see [On the patch series](#on-the-patch-series). The pins record which build a run used, and a resume on another one is refused.

### What a run on this weight does not replace

**A run on `Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf` does not replace BF16. The BF16 run remains a follow-up.** The file is quantized, and issue #66 asks to compare against BF16 and the reference before quantizing.

`--expert-cache 128` keeps the 33 GiB from being resident, and it is lossless in the sense of [expert-streaming.md](expert-streaming.md): the logits are bit-identical to resident Metal **at the same ubatch**. The cache caps the ubatch at 21, while the documented runs of the prompts below used the default ubatch, and tokens from the two settings differ: "0 of 65 requests are the same" against resident IQ3 at the default ubatch, with a first difference at token 12 to 22. So a documented control can come out differently here without anything being wrong. That is why [the control table](#corpus) informs and does not gate.

## Method

`cmd/kolibri-loops run` starts one `llama-server` (`-server`, `-model`, and the flags after `--`) and sends the corpus one prompt at a time. It can attach to a running server with `-url` instead, which pins less; see [Resume and refusals](#resume-and-refusals). `cmd/kolibri-loops report` recomputes every number from the records. It reads no corpus, no weight and no server.

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
- `pins`: the realpath, size and SHA-256 of the weight; the realpath and SHA-256 of the server binary, the SHA-256 of each shared library next to it, its arguments, and the `LLAMA_` and `GGML_` variables it inherits; the server's `build_info`; the chat template in full and its SHA-256; the vocabulary size; the corpus's SHA-256; the settings and the loop rule;
- the corpus item, the exact `/apply-template`, `/tokenize` and `/completion` request bodies, and the rendered prompt;
- the settings the server says it resolved, whether it truncated, how many tokens it says it evaluated and predicted, the prompt and generated token IDs, the text, the stop type, and the top candidates per token.

No derived number is stored. The report prints the rule from the pins and computes with `FixedRule`, and it refuses records whose pinned rule differs, so a rule that moved later cannot give new numbers under the old label.

### Resume and refusals

A second run with the same `-out` continues after the last finished record. It refuses, naming the field, if anything in the pins moved. Weight, server code, flags, environment, corpus and settings are compared before a server starts; template, build and vocabulary right after it answers. It also refuses a record whose pins or corpus item differ, and a record it cannot parse.

- **The server's code is in its libraries.** `build/bin/llama-server` is a 49,984-byte stub, and `otool -L` shows the code in `libllama-server-impl`, `libllama`, `libggml-metal` and others. A rebuild of uncommitted work can change those and leave both the stub's hash and the commit in `build_info` alone. So `server_libs` holds the hash of every library in the binary's directory, where a build keeps them and its rpath points, and of a `.metallib` if the build does not embed one. A link that leaves the directory is hashed too. A rebuild that changes a library the server does not load also refuses a resume, which errs the right way. A library loaded from any other path is not pinned, such as Homebrew's OpenSSL or the system's Metal.
- **Flags and environment change logits.** The ubatch changes tokens ([above](#what-a-run-on-this-weight-does-not-replace)), and `common/arg.cpp` reads `LLAMA_ARG_UBATCH` as well as the flag. `server_args` and `server_env` pin both for a server the tool starts.
- **A server attached with `-url` pins less.** Its API does not report the ubatch, so a restarted server on other flags passes every other pin. Such a run therefore refuses to resume unless `-attached-resume` says it is the same server process, which the tool cannot verify. The binary hashed is the one `-server` names, also unverified; the served model path is checked. The report says the flags and environment are unknown. Prefer starting the server.

The tool refuses to write a record if the response is not what was asked: other resolved settings, a truncated prompt, a different prompt length, a token that is not the most likely one, a stop type other than limit or EOS, or a log probability that is null or above 0. llama-server's JSON library writes NaN as null, and Go would read that null as 0, which is probability 1, so a NaN decode would otherwise pass as a confident loop. A small tolerance above 0 allows for float32 rounding at probability 1.

The report re-checks what a record stores:
- **Every record:** its file name matches its ID, and its log probabilities can be log probabilities.
- **Generations:** the request is the pinned `n_predict`, seed and `n_probs`; the resolved settings, truncation and token counts match; each token is greedy.
- **Texts:** the last request belongs to the text and the pinned seed; its resolved settings and truncation match; each position targets the text's token and listed the whole vocabulary.

It reads no server and no tokenizer, and records carry no checksum. An edit that stays plausible gets through: another plausible log probability, or a `content` that does not match the tokens. Earlier positions of a text are checked during the run only, because a record keeps one request per text.

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
- a template with changing slots, as in "The capital of Spain is Madrid. The capital of Portugal is Lisbon.", or a numbered list such as "1. Berlin 2. Berlin 3. Berlin", where the counter breaks the period;
- one token short of the region, or under 3 whole units;
- a unit longer than 64 tokens;
- a loop the last token breaks.

**What it counts that is not degenerate.** Legitimate periodic output the limit cuts off is a loop by this rule, because token IDs cannot tell the two apart: 40 distinct tokens and then 16 of period 2 count, which is the shape of a Markdown separator such as `|---|---|---` if it tokenizes to a repeating pair. No corpus prompt asks for a table, so this should be rare; whether real text tokenizes that way was not checked. A test holds this behavior.

### What the token limit hides

**The limit censors the rule.** A loop of period `p` counts only if it starts by `n_predict - max(16, 3p)`:

| Period | Latest start at 256 tokens | At 768 tokens |
|---:|---:|---:|
| 1 to 5 | 240 | 752 |
| 30 | 166 | 678 |
| 64 | 64 | 576 |

- **The repetition rate is a lower bound.** A generation can start looping later than this and read as no loop.
- **Time to loop covers detected loops only,** so it is biased toward early loops.
- **Forms have different windows,** because `chat-think` runs to 768 tokens. That is one more reason not to pool them.
- **A delay can read as a cure.** A later intervention that only moves a loop's start from token 100 to token 170 at period 30 reads as "no loop". The table's "Limit, no loop" column counts these censored generations; compare it before and after, not only the rate, and only at equal `n_predict`.

A cell or control without a loop has no start: the report prints `-` and `report.json` holds `null`, because start 0 means the first token.

## Answer accuracy

Only items with a known answer are scored. A term matches as a whole word, ignoring case, so "Berlin" does not match "Berliner", and "12" does not match "112".

**Accuracy scores the first sentence of the answer:** the text up to the first line break, or the first `.`, `!` or `?` that a space or the end follows, at most 200 characters. With thinking on, the answer is the text after the last `</think>`.
- **correct:** the first accepted or rejected term in that sentence is an accepted one;
- **wrong:** a rejected term comes first, or no term occurs;
- **unanswered:** nothing was generated; with thinking on, also when `</think>` never comes.

**The report's "Mentioned" column is the same terms over the whole answer.** It is an occurrence rate and not an accuracy. Only `capital-australia` has rejected terms, so " 1961. Sie wurde 1961 gebaut und erst 1989 geöffnet." counts as mentioned though the first answer is wrong, and it gets likelier the longer a generation runs. Reporting both shows that gap, and an intervention that only changes length moves "Mentioned" and not accuracy, as long as the first sentence stays the same.

**Accuracy errs the other way.** An answer after a preface or in a list counts as wrong: "Natürlich! Die Hauptstadt ist Berlin." or "1. Berlin". So does "ca. 100 Grad", where the abbreviation ends the sentence. Read the two columns together, and read `per_record` before quoting a low number.

The formula accepts its plain and LaTeX spellings (`H2O`, `H₂O`, `H_2O`, `\text{H}_2\text{O}`). A generation can be correct and loop afterwards. The two are reported apart.

## Held-out loss

The `text` items are short passages written for this corpus. For each token after the first, the tool asks for the distribution given the tokens before it (`n_predict` 1, `n_probs` above the vocabulary) and stores the target's log probability.

- **Honest because:** the server returns the softmax over the whole vocabulary, and the tool refuses a position with fewer candidates than `n_vocab`. It never computes a bound from a partial list. A target with probability 0 leaves the perplexity empty, and the report says how many.
- **The ban:** at `n_predict` 1 the server lists nothing if the generated token is a UTF-8 fragment. The tool then asks again with that token banned. The distribution comes from the raw logits, so the ban does not change it. On the tiny checkpoint the 128000 candidates were identical with and without a ban. The report counts the bans.
- **Perplexity:** `exp(mean negative log probability)`, per split and language.
- **Not comparable with the PPL in [real-checkpoint.md](real-checkpoint.md):** other texts, other lengths, another tool. It is meant for before and after comparisons on this corpus.

## Corpus

`testdata/loops/corpus.jsonl`, 112 items in 32 pairs. A pair is one fact or topic, and it holds one item per language and form. Every pair belongs to a group; see [The split](#the-split).

| Stratum | Pairs | Items per pair | Content |
|---|---:|---:|---|
| `documented-loop` | 5 | 4 (raw and chat, de and en) | contains a prompt documented as looping |
| `documented-non-loop` | 3 | 4 | contains a prompt documented as not looping |
| `answer-known` | 12 | 4 | a fact with a known answer |
| `open` | 4 | 4 | no known answer, nothing documented |
| `thinking` | 4 | 2 (`chat-think`, de and en) | thinking on |
| `text` | 4 | 2 (`text`, de and en) | held-out loss |

**Documented controls.** `control` and `source` mark the prompts [real-checkpoint.md](real-checkpoint.md) and `testdata/e2e/corpus.json` document. A test checks that each prompt is verbatim in its source, and that its label agrees with what the source records for the run, on the line that holds the prompt. They are the sections "Raw German completions repeat, BF16 included" and "Raw prompts past the first sentence".
- **Looping:** four German raw prompts, plus both `<|endoftext|>` prompts, which also give the one documented English loop.
- **The controls are not balanced.** Every looping control is raw, and none is chat: chat is documented as coherent. English looping controls are scarce too, since only that one is documented, so the English counterparts of the German prompts count as balance, not as controls. The balance in the table above is of items, not of controls.
- **Other weights:** the IQ3 column applies to most; `think-sky.de.chat-think` was documented on the Q3_K mix, and `moe-explain.de.chat` is the e2e corpus's `de-chat`.
- **Short documented runs:** a run here is 256 tokens, and most documented runs are shorter. The table's runs stopped at 40 tokens, and `moe-explain.de.chat` rests on 16 greedy tokens and the doc's word "coherent". The others stop earlier or run longer (`think-sky`: 528). A prompt documented as not looping can still loop later.

### The split

Fixed now, so later intervention work cannot pick its test set after the fact. `split` is on every line, and `validateCorpus` recomputes it:

> A group's stratum is the first of its pairs' strata, in the order of the table above. Within a stratum, order the groups by the SHA-256 of `kolibri-loops/split/v2/` + name. They take turns, tuning first. A whole group goes to one split, so a fact, a prompt or an answer never sits on both sides.

Pairs alone would not do this: 12 pairs state or ask the same few facts in other words.
- **`capital-germany-france`** holds the 8 pairs on Germany's capital and on France's, since `capitals-context` states the second. `wall-year` is in it because its English prompt holds "Berlin".
- **`rhine`** holds `rhine-clause` and `rhine-context`.
- **`planet`** holds `planet-largest` and `think-planet`, which share their prompts.
- **Every other pair** is a group of its own.

`validateCorpus` refuses items of different groups where one prompt contains the other, leaving out `<|endoftext|>`; that accept the same term; or where one prompt holds the other's accepted term. It cannot see a fact restated in other words, which the groups above handle by hand. A shared word can still cross, such as "Central Europe" in `text-lake` and `rhine-context`, or "water" in `boiling-water` and `formula-water`.

**What this costs.**
- **The sizes differ:** tuning has 68 items and held-out 44, because the one large group went to tuning.
- **Looping controls are lopsided, though each side has one.** Tuning has 5 and held-out has 1 (`rhine-clause.de.raw`), because two groups hold them and five of the six are about Berlin. The held-out rate rests on every raw German prompt there, not on that control alone.
- **Held-out has no capital prompt at all.** The rule's turns put `capital-australia`, `capital-austria` and `capital-japan` in tuning too, so the template "The capital of X is" is tuning only, and so are the only rejected terms. A rate on held-out says nothing about capitals, and an intervention that only learned capitals shows no gain there. Picking another salt to even this out would choose the split by its content, so it stays.
- **The English looping control is in tuning only,** so an English intervention has nothing documented to validate on. Only one is documented, and it cannot be on both sides.

The corpus's hash is pinned in `main_test.go`, so a later edit to a prompt or a split changes that line in plain sight.

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
| M74f | Every log probability is null, as the server writes NaN | "log probability is null"; no record |

The numbers start at M72 because M70 and M71 are taken by PR #72's controls in [upstream.md](upstream.md).

## What has been checked

- **`go test ./cmd/kolibri-loops`:** the detector's edge cases, answer scoring, the corpus, resume, refusals and the controls, against a fake server.
- **A real llama-server on tiny random weights** (`cmd/kolibri-tiny`, the real tokenizer and chat template):
  - all 112 prompts through to a report;
  - the documented token IDs matched;
  - a changed setting, changed flags, `LLAMA_ARG_UBATCH` and `-url` were each refused before anything was written;
  - `server_libs` held 16 libraries, among them all 9 that `otool -L` reaches from `llama-server` over `@rpath`, and their hashes matched the files;
  - a finished run wrote 0 new records;
  - a second run gave identical tokens and identical loss positions in every record;
  - `--expert-cache 128`, `--fit off` and `-ngl 99` on the router fixture ran 3 prompts, with the server's log showing the ubatch capped to 21.

  This shows the tool works, and nothing about loops: random weights say nothing about Kolibri-1.
- **The real weight:** two runs, one per build, with identical tokens in all 112 records; see [On the patch series](#on-the-patch-series). They differ in the build, so no run has been repeated on one build.

## Not done

- **Reference logits.** `llama-server` reports the model under test; the reference is `tools/ref/kolibri_ref.py` or vLLM on BF16, which do not run through its API. The records keep the top 10 log probabilities per generated token for a later comparison, but those are this run's, not the reference's, and they are not reported as such. The report's "p(first)" and "p(in loop)" are also this run's.
- **BF16.** See [above](#what-a-run-on-this-weight-does-not-replace).
- **Tool use.** The issue asks for it at validation, step 4; the corpus has reasoning but no tools.

Each report lists these three too, so they stay with a `report.json` read without this page.

## Results

One run on 2026-11-07, 112 of 112 records in a single pass, with no resume. It used [the command above](#the-real-run), with a binary built from commit `65d1be4` in place of `go run`. The tables are the output of `kolibri-loops report`, unedited; the notes under them were checked against `report.json` and the records. Records, `report.json` and `server.log` are in `~/models/eval/loop-baseline/iq3-cache128`, on this machine only.

**What this run is, and is not:**
- **One quantized weight.** It does not replace BF16, and [the BF16 run remains a follow-up](#what-a-run-on-this-weight-does-not-replace).
- **`--expert-cache 128` at ubatch 21.** `server.log` shows "n_ubatch 512 capped to 21". The documented runs used the default ubatch, and tokens from the two settings differ, so the control table informs and does not gate.
- **The waves build.** `b11839-90311e5a2` is fork branch `feat/kolibri1-waves`, without passing `--expert-cache-waves`. A run on `feat/kolibri1` [gave the same records](#on-the-patch-series) at these settings.
- **No reference logits.** "p(first)" and "p(in loop)" are this run's own probabilities.
- **Run once on this build.** The second run on this weight used [another build](#on-the-patch-series). Before it, repeatability had been checked on tiny random weights only.
- **No timing.** The run took no timing, and none is published.
- **Not done:** reference logits, BF16 and tool use; see [Not done](#not-done).
- **A baseline only.** It says nothing about any intervention.

**Pins:**
- **Weight:** `Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf`, a real local file, 35556257600 bytes, SHA-256 `93720b94a5a383f5bad07677ca05014bf0681a662e646ea3b1b196cda33c88bc`.
- **Server:** `llama-server` SHA-256 `a3e31e656f5f540b4dc9b7583d9d06399632901cdaeffe0726edbcb400a8b798`, 16 libraries next to it with digest `822af4fc6b434ce96e82ae6b025bdf30dcbf97730f12206608e52aca8df3427c`, flags `-ngl 99 -t 10 -c 4096 -np 1 --fit off --jinja --expert-cache 128`, no `LLAMA_` or `GGML_` variables.
- **Inputs:** chat template SHA-256 `9ba35d4bd6baa26b66aa75d03a922dfee98b16bb1fa37481b195d247267b0f97`, corpus SHA-256 `722992046f85c1ac63394e954dd776239d08ad7a283b4117cdd1140bf36b2ddc`, vocabulary 128000.
- **Decoding:** greedy, seed 42, `n_predict` 256 (768 with thinking on), `n_probs` 10.
- **Counts:** 0 generated tokens without a listed probability, 0 bans while measuring the loss.

### On the patch series

The same command ran a second time, on `feat/kolibri1` at `b11839-634e34bb4`, which has no Stage E4 code, into a new directory: 112 of 112 records in a single pass from commit `6b04d16`, whose only changes after `65d1be4` are docs. Records, `report.json` and `server.log` are in `~/models/eval/loop-baseline/iq3-cache128-series`.

**There is no difference. Every record's token IDs are the same, so there is no first differing token.**

| Compared record by record, all 112 | Result |
|---|---|
| Generated token IDs, 16,862 tokens over 104 generations | equal |
| Prompt token IDs, rendered prompts and all request bodies | equal |
| `content`, stop type, resolved settings, truncation and token counts | equal |
| Top 10 log probabilities per generated token | equal, as JSON numbers |
| Log probabilities of the 8 texts' 273 loss positions | equal, as JSON numbers |
| Pins | differ only in `server_realpath`, `server_sha256`, `server_libs` and `server_build_info` |
| `kolibri-loops report`, text | differs only in the line that names the server |

So all tables in this section hold for both builds. The series build's directory holds 9 libraries rather than 16, because only `llama-server` and `libllama` were built there. Its `server.log` shows the same "n_ubatch 512 capped to 21".

**What this reaches:**
- **The two builds at these settings:** this weight, `--expert-cache 128`, ubatch 21 and these 112 prompts. It says nothing about them resident or at another ubatch.
- **Repeatability, in part.** These are two separate server processes, and any variation between them that changed a token or a stored log probability would show here. No such variation came up. They ran on two builds, and two runs are not a measured rate.

The comparison is `~/models/eval/loop-baseline/iq3-cache128-series.compare.py`, and the report's text is next to it.

### Repetition

| Split | Form | Lang | Prompts | Loops | Rate | Median start | Latest start | Periods | Ended by EOS | Limit, no loop (censored) |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| tuning | raw | de | 15 | 13 | 0.867 | 0.0 | 2 | [1 1 2 2 2 2 2 2 2 5 6 13 21] | 1 | 1 |
| tuning | raw | en | 15 | 10 | 0.667 | 0.5 | 216 | [2 2 3 4 5 6 6 7 15 17] | 1 | 4 |
| tuning | chat | de | 15 | 0 | 0.000 | - | - | [] | 14 | 1 |
| tuning | chat | en | 15 | 0 | 0.000 | - | - | [] | 14 | 1 |
| tuning | chat-think | de | 2 | 0 | 0.000 | - | - | [] | 2 | 0 |
| tuning | chat-think | en | 2 | 0 | 0.000 | - | - | [] | 2 | 0 |
| held-out | raw | de | 9 | 8 | 0.889 | 0.0 | 8 | [1 1 2 2 4 5 7 9] | 0 | 1 |
| held-out | raw | en | 9 | 2 | 0.222 | 210.5 | 238 | [3 7] | 0 | 7 |
| held-out | chat | de | 9 | 0 | 0.000 | - | - | [] | 9 | 0 |
| held-out | chat | en | 9 | 0 | 0.000 | - | - | [] | 9 | 0 |
| held-out | chat-think | de | 2 | 0 | 0.000 | - | - | [] | 2 | 0 |
| held-out | chat-think | en | 2 | 0 | 0.000 | - | - | [] | 2 | 0 |

The rate is a lower bound and the starts cover detected loops only; see [What the token limit hides](#what-the-token-limit-hides). Cells hold 2 to 15 prompts, and the tuning and held-out rows hold different prompts, with no capital prompt on the held-out side. The two chat generations at the limit are `story.de.chat` and `story.en.chat`. Every `chat-think` generation ended before its limit.

### Answer accuracy

| Split | Form | Lang | Scored | Correct | Wrong | Unanswered | Accuracy | Mentioned | Mention rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| tuning | raw | de | 11 | 3 | 8 | 0 | 0.273 | 3 | 0.273 |
| tuning | raw | en | 11 | 10 | 1 | 0 | 0.909 | 10 | 0.909 |
| tuning | chat | de | 11 | 11 | 0 | 0 | 1.000 | 11 | 1.000 |
| tuning | chat | en | 11 | 11 | 0 | 0 | 1.000 | 11 | 1.000 |
| tuning | chat-think | de | 1 | 1 | 0 | 0 | 1.000 | 1 | 1.000 |
| tuning | chat-think | en | 1 | 1 | 0 | 0 | 1.000 | 1 | 1.000 |
| held-out | raw | de | 5 | 0 | 5 | 0 | 0.000 | 0 | 0.000 |
| held-out | raw | en | 5 | 4 | 1 | 0 | 0.800 | 4 | 0.800 |
| held-out | chat | de | 5 | 5 | 0 | 0 | 1.000 | 5 | 1.000 |
| held-out | chat | en | 5 | 5 | 0 | 0 | 1.000 | 5 | 1.000 |
| held-out | chat-think | de | 2 | 2 | 0 | 0 | 1.000 | 2 | 1.000 |
| held-out | chat-think | en | 2 | 2 | 0 | 0 | 1.000 | 2 | 1.000 |

The low raw German numbers, from `per_record`: 13 of the 16 scored prompts are wrong because no accepted term occurs, and each of the 13 loops from token 0, as in " Deutschland Deutschland". The other three are correct: `question-capital.de.raw` does not loop, while `capitals-context.de.raw` and `wall-year.de.raw` give the answer and also loop from token 0. "Mentioned" equals "Correct" in every row of this run.

### Held-out loss

| Split | Lang | Texts | Positions | Zero probability | Mean NLL (nats) | Perplexity |
|---|---|---:|---:|---:|---:|---:|
| tuning | de | 2 | 75 | 0 | 3.5593 | 35.138 |
| tuning | en | 2 | 71 | 0 | 3.1641 | 23.666 |
| held-out | de | 2 | 64 | 0 | 3.5902 | 36.240 |
| held-out | en | 2 | 63 | 0 | 3.9640 | 52.668 |

Each cell holds 2 texts. These numbers are for before and after comparisons on this corpus, and [are not comparable with the PPL in real-checkpoint.md](#held-out-loss).

### Documented controls

| Prompt | Split | Documented | Loops here | Start | Period | p(first) | p(in loop) | Agrees |
|---|---|---|---|---:|---:|---:|---:|---|
| berlin-lead.de.raw | tuning | loop | true | 0 | 13 | 0.931 | 0.999 | true |
| capital-germany.de.raw | tuning | loop | true | 0 | 1 | 0.843 | 0.999 | true |
| capital-germany.en.raw | tuning | non-loop | true | 64 | 15 | 0.519 | 0.997 | false |
| capitals-context.de.raw | tuning | loop | true | 0 | 21 | 0.980 | 1.000 | true |
| capitals-context.en.raw | tuning | non-loop | true | 143 | 6 | 0.777 | 0.986 | false |
| eot-capital.de.raw | tuning | loop | true | 0 | 5 | 0.422 | 0.995 | true |
| eot-capital.en.raw | tuning | loop | true | 0 | 3 | 0.336 | 0.996 | true |
| moe-explain.de.chat | tuning | non-loop | false | - | - | - | - | true |
| question-capital.de.raw | tuning | non-loop | false | - | - | - | - | true |
| rhine-clause.de.raw | held-out | loop | true | 0 | 2 | 0.395 | 0.998 | true |
| rhine-context.en.raw | held-out | non-loop | true | 238 | 3 | 0.856 | 0.951 | false |
| think-sky.de.chat-think | tuning | non-loop | false | - | - | - | - | true |

9 of the 12 agree. All 6 documented loops loop here. The 3 that differ are English prompts documented as not looping, and they loop here from token 64, 143 and 238 of 256. For all three, the documented continuation is the beginning of this run's text as well. Two of those documented runs stopped at 40 tokens; the third, `capital-germany.en.raw`, documents one sentence without stating its length. So those documents do not show the tokens where these loops start, and this run does not contradict them. It also ran at another ubatch.
