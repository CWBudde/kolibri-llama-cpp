# Layer probe (issue #63)

**Result: the pipeline ran on the real weight and its checks held, but no probe shows more than the text's surface. On execute, layers 10, 25 and 40 tie the surface rule at 0.975 balanced accuracy, and layer 50 is one row of 40 above it at 1.000. On operation family all four are below it, 0.733 to 0.900 against 0.967. These are scores from 40 prompts: they validate the pipeline, select no layer, and show what can be decoded, not what the model uses.** The probe observes only: no base weight changes and no coprocessor runs.

| Part | Where | State |
|---|---|---|
| 44 prompts with oracle validation | `testdata/probe/smoke.jsonl`, `internal/probe`, `go run ./cmd/kolibri-probe generate` or `verify` | done |
| Activation capture | `tools/probe/capture.py`, through libllama's `cb_eval` | done on tiny; one real run, Metal with the cache |
| Probes, baselines, report | `go run ./cmd/kolibri-probe report`, `internal/probe` | done on synthetic and tiny activations and on that real run |
| Readout of the real model | [Results](#results) | run once, pipeline validation only |

## What a layer number means

The issue's layers 10, 25, 40 and 50 are **one-based**: they count the blocks, 1 to 50, and 50 can't be read any other way, because the graph's blocks are numbered 0 to 49 (`il` in `src/models/kolibri1.cpp`). Layer N is node **`l_out-(N-1)`**:

| Issue's layer | Node | What follows it |
|---|---|---|
| 10 | `l_out-9` | block 11's input norm |
| 25 | `l_out-24` | block 26's input norm |
| 40 | `l_out-39` | block 41's input norm |
| 50 | `l_out-49` | `output_norm`, then the output head |

**Normalization.** `l_out-il` is the block's residual sum, `post_ffw_norm(routed + shared experts) + ffn_inp`, and `build_cvec` passes it through with no control vector loaded. So the post-FFN norm sits inside the sum, and nothing normalizes the sum itself. It is read **before** the next block's input norm and, at layer 50, **before `output_norm`** (`result_norm`). Layer 50 is therefore not the embedding that `llama_get_embeddings` would return.

Reading this from the source doesn't prove it for a given build, so the capture reads it from the model's graph on its first decode (`Capture.check_graph`) and refuses to go on otherwise. `l_out-il` must have the sources `(post-FFN norm, ffn_inp-il)`, and the first one is checked by walking down from it, since those nodes have no names of their own: a `MUL` by `blk.il.post_ffw_norm.weight`, over the `RMS_NORM` `norm-il`, over the `ADD` of `ffn_moe_out-il` and `ffn_shexp_out-il`. `norm-(il+1)` must read `l_out-il`, and the last one must feed `norm`. `capture.json` keeps what it found under `graph_check`, and `kolibri-probe report` checks that again. So the `normalization` string each layer records is written only after this check passed. The check reads what the graph is wired from, not what the numbers mean: a wrong RMS epsilon, say, would pass.

## The prompts

`testdata/probe/smoke.jsonl` holds 44 rows: the issue's 4 pilot prompts word for word (`split: pilot`) and 40 smoke prompts (`split: smoke`), 10 per category.

| Category | Execute | Family | What varies |
|---|---|---|---|
| A `no_execution` | no | `none` | facts, explanations, writing; hard negatives ask for a recalled number or mention an operation |
| B `exact_arithmetic` | yes | `arithmetic` | multiplication, powers, factorial, GCD; magnitude, thousands separators, language |
| C `small_algorithm` | yes | `algorithm` | sorting, primality, Fibonacci, filtering; input length, order, phrasing, language |
| D `explicit_non_execution` | no | `none` | the numbers of a B or C partner in the same group, with the instruction not to compute |

- **Schema.** The issue's eight fields come first, in its order, then `language`, `template_id`, `numeric_seed`, `prompt_length`, `explicit_execution_constraint` and `expected_answer_or_oracle`. Added are `operation`, `execution_need`, `label_policy`, `hard_negative` and `varied_axis`. The metadata of a capture (model and template revision, layer index, dtype) is in its `capture.json`, since it isn't a property of a prompt.
- **Labels are a policy.** `execute_expected` follows `label_policy` (`execute-when-exact-result-asked-v1`) and means execution is helpful. `execution_need` separates `helpful` from `necessary`: an exact result is necessary when a number of the prompt or of the answer reaches 100,000, so that 5 numbers to sort stay `helpful`. The oracle works this out again from the prompt and its own answer. A D prompt has family `none`, because it needs nothing; `operation` keeps what it mentions.
- **Pairs.** 10 groups hold a computation and its refusal on the same operation and the same numbers (`verifyPairs` compares them), so that a word for the operation can't separate them. **Other surface cues can**, and the pairs don't remove them: see [What the text alone gives](#what-the-text-alone-gives). A refusal is also a longer sentence with other wording: B-002 is `Was ist 36 mal 49?` (18 characters), D-002 is `Erkläre das schriftliche Multiplizieren am Beispiel 36 mal 49, aber rechne das Ergebnis nicht aus.` (98). `varied_axis` names the axis the template varies, not the only difference in characters. Other rows of a group differ in language or wording.
- **Determinism.** `kolibri-probe generate` draws from PCG with a fixed seed (63) and a per-group stream. `TestRegenerateBytes` regenerates the file and compares bytes.
- **Oracle.** `internal/probe/oracle.go` shares no code with the generator. It reads a row's operation and numbers from the prompt's text, so a row's `operation` field can't choose what it computes and must agree with the text, and computes the answer by other algorithms: `math/big`'s `Mul`, `Exp`, `MulRange`, `GCD` and `ProbablyPrime`, fast doubling, an insertion sort. Facts come from its own table. A D prompt must carry words that forbid execution.
- **Leak checks.** `VerifyStructure` refuses a template or a numeric seed in two groups, and two prompts of different groups with a masked word overlap of 0.6 or more, which is what a sentence frame shared by two groups reaches: the frames that differ in a word or two scored 0.67 to 0.75, and 5 smoke prompts were reworded for it. The highest pair left is 0.50 (`What is the capital of Spain?`, `What is the factorial of 12?`). Folds keep groups whole, so this holds for every fold. It compares words, not topics: A-001 and A-002 both still ask for a capital, in other words (0.33). The pilot's groups are in no smoke row.

## Capture

The issue's activations are read through `cb_eval`, the callback route `cmd/kolibri-routing` and `tools/gguf/check_*.py` already use. No fork change was needed. The tool is Python because the repo's libllama binding (`tools/tokenizer/common.py`, `check_model.Runner`) is; everything after it reads NumPy files from Go.

- **The row.** The last prompt token's row. Each prompt is decoded on a cleared memory with an output at its last position only. Below the last block a node has one row per token of the ubatch; the last block's rows are cut to the outputs by the graph. A prompt longer than the ubatch runs over several graph computations, and the capture keeps the last one's last row and checks that the rows add up to the prompt.
- **The prompt.** The model's own `tokenizer.chat_template`, rendered by the reference's renderer (`render_jinja_template`) with `add_generation_prompt` and `reasoning_effort: "none"`. That prefills the empty think block, so the last prompt token is `\n\n`, the same token for every prompt, right where the answer starts. A definition: `--reasoning default` captures the other position, as a separate run.
- **Determinism.** No sampling, one decode per prompt, fixed threads and ubatch. `--repeat` decodes the pilot prompts again and compares activations and logits exactly.
- **Stored.** `layer-<N>.npy`, `<f4` `[44, n_embd]` in dataset order, and `capture.json`: the sha256 of the model, of libllama (`server_sha256`: libllama does the server's work, no server process is involved), of the ggml libraries and of the dataset; the fork's head; the template's sha256 and pinned revision, and whether it is the released one; dtype, shape, one- and zero-based index and node per layer; device, threads, requested and effective ubatch, cache slots; each prompt's tokens. The output directory must be new or empty, because `np.save` and the cache's stats file replace what they find and results may exist nowhere else.

### Non-mutating

The callback only reads, and the check is exact: `np.array_equal`, no tolerance. All prompts are first decoded by a model and context that never had a callback, and the capturing decode must give the same logits. A capture run does this itself and records `logits_check`. `kolibri-probe report` refuses a capture whose logits differ, and also one where the check didn't run, since nothing then shows that it left the logits alone: `--skip-logits-check` and `--no-repeat` are for debugging the capture, and their output gets no report. The report also refuses a capture without a `graph_check` that holds, or whose prompts don't all end in one token. It shows these facts and whether the template is the released one, under "Capture".

`tools/probe/check_capture.py` shows the same on tiny checkpoints, with F32 and BF16 weights, on the CPU and on Metal, reading 4 layers and all layers. `--layers` scales the indices: the 6-layer tiny model reads 2,3,5,6, and the 50-layer `-pattern` model reads the real 10,25,40,50.

### Metal, CPU and the expert cache

| Path | On tiny (`check_capture.py`) | On the real model |
|---|---|---|
| CPU | passes | not run |
| Metal, resident | passes | not run |
| Expert cache, CPU and Metal, ubatch capped at 3 on tiny, 21 on the real model | passes: exactly the logits of the cached model without a callback, and logits and activations exactly a resident run's at the same ubatch | Metal at 128 slots passes: exactly the logits of the cached model without a callback, and a repeat is exact. CPU and the comparison with a resident run were not run |

- **The cache serves one context per model** ("the expert cache serves one context per model"), so the plain pass and the capture load the model apart, one after the other.
- **The cache caps the ubatch** at the fewest slots divided by 6: 21 at `--expert-cache 128`. So a prompt spans several computations, and the numbers belong to that ubatch: the ubatch changes the order of the sums ([expert-streaming.md](expert-streaming.md)), and on Metal so does the number of outputs in the last block. `check_capture.py` shows the second on tiny: there the last block's row isn't bit-identical between a decode with one output and one with all outputs, while the CPU's is, and every layer below it is exact on both. A capture's numbers are reproducible for its recorded settings, and two captures compare only at equal settings.
- **Overhead.** The structure is known: each read node ends a graph split, so 4 more splits per ubatch, a copy of `n_tokens x n_embd` floats per read node, and a Python call for each node the scheduler asks about. The real cost is unmeasured: `capture.json` records `plain_decode_seconds` and `capture_decode_seconds`, but the run wasn't set up as a timing run (nothing evicts the weight, and the plain pass always goes first), so this doc takes no figure from them.

## Probes and report

`kolibri-probe report -capture DIR` trains on activations alone and scores only held-out rows. The options below are **definitions, committed before any real capture, and they must not move afterwards**: `DefaultOptions` in `internal/probe/report.go`, the dataset, and the capture defaults in `tools/probe/capture.py`. **One moved all the same, after the real capture: the label shuffle, [disclosed with the results](#the-label-shuffle-moved-after-the-capture).**

The report enforces this where it can. It never replaces a report, and only `DefaultOptions` write `report.md`. `-l2`, `-folds` or `-shuffles` at another value write `report-deviation-*`, which says at its top what moved, so trying values leaves every try on disk under a name that says so. A `-dataset` whose bytes aren't the committed file's does the same, whatever its path and however well it passes the checks: its sha256 is compared with `CommittedDatasetSHA256`, and the report's name carries the start of it.

- **Probes.** One multinomial logistic regression per layer and target: execute or not, and operation family (`none`, `arithmetic`, `algorithm`). L2 strength 0.01, features standardized with the training rows' mean and deviation and scaled to unit expected squared norm, no feature selection. The solver is deterministic (Nesterov's method from zero) and stays in the span of the training rows, since there are far more dimensions than rows.
- **Folds.** 5 folds by `paraphrase_group`, over the smoke split. `CheckFolds` runs before any training and refuses a group, template or numeric seed in train and test of a fold, whoever made the split.
- **Baselines**, on the same folds: majority of the training rows, a fixed keyword rule, a fixed surface rule, the prompt's length through the same classifier, and a learned bag of words with the same classifier and regularization. The two rules' lists were written by the hand that wrote the prompts, which favors them.
- **Scores.** Accuracy, balanced accuracy, macro F1, precision and recall of execute, hard negatives right, agreement of same-label rows within a group, Brier score and a 5-bin calibration error.
- **Shuffled labels.** 20 runs per probe, same folds. A probe must beat them by more than their spread. A run moves the unit the folds hold out: each paraphrase group takes the block of labels of a group of its own size, in an order drawn anew inside the block. So every group keeps a set of labels that some group of its size really has, the class counts stay, and labels move only among the 10 single rows and among the 15 pairs. `report.json` names the scheme, `whole-groups-among-equal-sizes-v2`; a report without that field shuffled single rows.
- **No confidence intervals.** The issue asks for them only where the sample supports them, and 40 rows in 25 groups doesn't. The report says so and gives none.
- **Pilot prompts.** Trained on all smoke rows and read on the 4 pilot prompts, whose groups the smoke split doesn't share. One prompt per category: a pipeline check.

### What the text alone gives

**On the committed rows the text's surface gives the execute label on 39 of 40 smoke rows, so the bag of words is far below what the text shows.** The words `aber`, `ohne`, `but` and `without` are in all 10 refusals and in none of the other 30 rows, and of those 30 only the 20 computations and A-011 hold a digit. On execute the surface rule is "a digit and none of those words", and it errs only on A-011 (`warum die 7 als Glückszahl gilt`). On family a digit and a conjunction can't assign anything, so the rule is that gate followed by the fixed keyword list (`surfaceFamily`), 26 words and symbols that name an operation. Its one error there is C-007 (`Berechne die Fibonacci-Zahl F(38) ...`), labelled `algorithm` and read as `arithmetic`, because `berechne` is on the arithmetic list and that list goes first. It gets A-011 right. The rule was written after seeing the rows. These scores need no weight, and on execute `TestBaselines` and `TestSurfaceCues` hold them:

| Baseline | Execute, balanced | Family, balanced |
|---|---|---|
| majority | 0.350 | 0.333 |
| keyword rule | 0.725 | 0.783 |
| surface rule | **0.975** | **0.967** |
| prompt length | 0.625 | 0.483 |
| bag of words | 0.650 | 0.450 |

So a probe at 0.90 beats the bag of words, and the keyword rule, and still reads less than the surface rule. And on execute a probe at or above 0.975 doesn't show more than these two cues either: for that target these rows can't separate reading what a prompt asks from reading a digit and a conjunction. On family the bar is those cues plus the keyword list. The dataset keeps this on purpose rather than hide it by rewording a few rows: 40 prompts validate the pipeline, and the larger set below is where the cues have to be broken.

### What a result would and wouldn't mean

- **Small sample.** 40 prompts validate the pipeline. They don't select a layer, and the report ranks none.
- **Decodable isn't useful.** A probe that reads a label from an activation shows that the information is there at that layer. It doesn't show that the model uses it there, or that injecting a result there would change behavior correctly. Only an intervention can show that, and that comes later, with a frozen base model and a trainable adapter (#61).
- **Random weights decode too.** A random-weight tiny model's activations score above chance on these labels, probably because prompts that share words stay close through random projections too. A score on the real model therefore means little by itself, whatever the shuffled-label control says: that control excludes chance, not the text. The surface rule is the bar, and clearing it wouldn't rule those cues out.
- **Labels are a policy.** A different policy gives different labels.

### Controls

| Control | Mutation | Must happen | Where |
|---|---|---|---|
| M75 | shuffled labels | the same probe on the same activations scores near chance, and no shuffle reaches the real score | `TestM75ShuffledLabelsNearChance` |
| M76 | a split that breaks the groups | refused with `ErrGroupLeak`; on group-fingerprint activations it would score near 1, grouped folds near chance | `TestM76LeakingSplitRefused` |
| M77 | a capture that writes into a node | logits differ, the capture exits 1, the report refuses it | `check_capture.py` |
| M78 | a wrong answer, a changed operand, a wrong operation or `execution_need`, a refusal with other numbers | the oracle fails the row | `TestM78OracleCatchesCorruption`, `TestOracleReadsTheOperation`, `TestPairsAndNeed` |
| M79 | the issue's numbers taken as zero-based | refused by the graph check and, at the last layer, by the missing node | `check_capture.py` |
| M87 | a graph whose `l_out` lacks the post-FFN norm, has another block's or has only the routed experts | refused by the graph check, and by the report from `capture.json`. On the real tiny graph's record with those parts changed, since no fork build has that wiring | `check_capture.py`, `TestReadCaptureRefuses` |
| M88 | a capture that wasn't checked: `--skip-logits-check`, also with a write into a node, or `--no-repeat` | the capture says UNCHECKED, the report refuses it | `check_capture.py`, `TestReadCaptureRefuses` |
| M91 | labels shuffled row by row, as first committed, or blocks that keep their row order | a shuffle keeps each group's labels together among groups of one size, and the higher label of a pair comes first in half of the draws. The row-level shuffle fails the first, a kept row order the second, and on labels constant per group `shuffled` itself must reach the spread only whole groups give | `TestM91ShufflesMoveWholeGroups` |

## Reproduce

```sh
go test ./internal/probe ./cmd/kolibri-probe
go run ./cmd/kolibri-probe verify
.venv/bin/python tools/probe/check_capture.py --llama-cpp third_party/llama.cpp   # tiny only

# the real capture: one holder of the weight at a time
go run ./cmd/kolibri-models status                # the weight must be hot
.venv/bin/python tools/probe/capture.py --llama-cpp third_party/llama.cpp \
    --model ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf \
    --layers 10,25,40,50 --expert-cache 128 --device MTL0 --threads 10 \
    --out ~/models/eval/layer-probe-<date>
go run ./cmd/kolibri-probe report -capture ~/models/eval/layer-probe-<date>
```

## Results

**One capture, 2026-11-07, from commit `4e35417`, with the definitions committed then.** Its report wrote `report.md` and no `report-deviation-*`, and both are in `~/models/eval/layer-probe-20261107`. **The tables below are from a second report on the same activations, from `aa70025`, whose label shuffle [replaced the first's](#the-label-shuffle-moved-after-the-capture).** Only the shuffle columns differ. It is in `~/models/eval/layer-probe-20261107-groupshuffle`.

- **Weight.** `/Users/christian/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf`, a real file on the internal drive (`model_realpath`), sha256 `93720b94a5a383f5bad07677ca05014bf0681a662e646ea3b1b196cda33c88bc`.
- **Build.** `third_party/llama.cpp` at `90311e5a2`, on fork branch `feat/kolibri1-waves`, with waves off (the cache's stats file says `waves: false`). That branch is outside the patch series: `feat/kolibri1` without its last commit, plus the Stage E4 waves code ([expert-streaming.md](expert-streaming.md#stage-e4-waves-for-prefill)), which changes `src/llama-graph.cpp`, the file that builds the graph this capture reads. Waves off is a runtime flag. A build without that code gave byte-identical activations and logits at these settings: see [On the patch series](#on-the-patch-series). `check_capture.py`'s tiny runs used this build and have not been repeated on the series.
- **Settings.** `MTL0`, 10 threads, 128 cache slots, effective ubatch 21, `n_ctx` 512, reasoning mode `none`. Two captures compare only at equal settings.
- **Dataset.** sha256 `e4681b761c1542876ab72c4db1f3d44e1de6c7b1190e2c532116496c00c2cc90`, as committed.

### On the patch series

The same capture was repeated on `feat/kolibri1` at `634e34bb4` (build 11839), which has no Stage E4 code. It was built Release with Metal in a worktree of its own, with the same `LLAMA_` and `GGML_` options as the waves build, and run with the commands in [Reproduce](#reproduce) from `ef27468`, whose only changes after `4e35417` are docs. Same weight, dataset, device, threads, slots and effective ubatch. Records: `~/models/eval/layer-probe-20261107-series`, and the second report on its activations in `~/models/eval/layer-probe-20261107-series-groupshuffle`.

| Compared with the run above | Result |
|---|---|
| `layer-10`, `-25`, `-40`, `-50` `.npy` | byte-identical, and `np.array_equal` holds, on all 44 rows and 2560 dimensions |
| Logits at each prompt's last token | byte-identical on all 44 prompts, `[44, 128000]` `float32`, sha256 `78f0bdb1…` on both builds |
| Prompt tokens per row, template, graph check, logits check, repeat | equal in `capture.json` |
| `report.md` | byte-identical, the first report's and the second's |
| `report.json` | differs only in the capture's provenance, the first report's and the second's: `server`, `server_sha256`, `backend_sha256`, `llama_cpp_head` and three timing fields |

`capture.json` keeps no logits, only the verdict of each run's own check, so the waves run's logits weren't on disk. Both builds were therefore run once more with `dump_logits.py`, which repeats the capture's plain pass (no callback, same settings) and writes the logits. The waves dump loaded `libllama` `f5801cb0…`, the same file the capture above records. Script, arrays and logs: `~/models/eval/layer-probe-20261107-logits`.

**This reaches these settings only: this weight, Metal, 128 slots, ubatch 21, the last row of 44 prompts, the four layers and the logits.** It doesn't say that the two builds agree on the CPU, resident, or at another ubatch.

### Capture checks

| Check | Result |
|---|---|
| Logits against a context without a callback | equal, `np.array_equal`, on all 44 prompts |
| Repeated decode | equal on the 4 pilot prompts, logits and activations |
| Graph | `l_out-9`, `l_out-24`, `l_out-39`, `l_out-49` are each the residual add of `ffn_inp` and the post-FFN norm, read by `norm-10`, `norm-25`, `norm-40` and `norm`. The graph check refused nothing |
| Last prompt token | 263, `\n\n`, on all 44 prompts |
| Chat template | the released one, revision `7a8f290` |
| Activations | `float32`, `[44, 2560]` per layer, all finite |
| Solver | 0 unconverged folds in every row of the tables below |

### Execute (no, execute)

Held out by grouped cross-validation: 40 rows, 25 groups, 5 folds, 20 label shuffles per probe. The tables are the second report's.

| Row | Kind | Accuracy | Balanced | Macro F1 | Execute P | Execute R | Hard negatives | Paraphrase agreement | Brier | ECE | Shuffled balanced, mean (max) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| majority | baseline | 0.350 | 0.350 | 0.335 | 0.286 | 0.200 | 9/14 | 5/5 | n/a | n/a | n/a |
| keyword rule | baseline | 0.725 | 0.725 | 0.703 | 0.645 | 1.000 | 3/14 | 5/5 | n/a | n/a | n/a |
| surface rule | baseline | 0.975 | **0.975** | 0.975 | 0.952 | 1.000 | 13/14 | 5/5 | n/a | n/a | n/a |
| prompt length | baseline | 0.625 | 0.625 | 0.623 | 0.609 | 0.700 | 11/14 | 5/5 | 0.459 | 0.115 | n/a |
| bag of words | baseline | 0.650 | 0.650 | 0.615 | 0.594 | 0.950 | 7/14 | 4/5 | 0.400 | 0.096 | n/a |
| layer-10 | probe | 0.975 | 0.975 | 0.975 | 0.952 | 1.000 | 14/14 | 4/5 | 0.084 | 0.153 | 0.463 (0.675), chance 0.500 |
| layer-25 | probe | 0.975 | 0.975 | 0.975 | 1.000 | 0.950 | 14/14 | 5/5 | 0.074 | 0.139 | 0.423 (0.575), chance 0.500 |
| layer-40 | probe | 0.975 | 0.975 | 0.975 | 0.952 | 1.000 | 14/14 | 4/5 | 0.113 | 0.188 | 0.469 (0.625), chance 0.500 |
| layer-50 | probe | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 14/14 | 5/5 | 0.141 | 0.242 | 0.474 (0.650), chance 0.500 |

### Operation family (none, arithmetic, algorithm)

| Row | Kind | Accuracy | Balanced | Macro F1 | Hard negatives | Paraphrase agreement | Brier | ECE | Shuffled balanced, mean (max) |
|---|---|---|---|---|---|---|---|---|---|
| majority | baseline | 0.500 | 0.333 | 0.222 | 14/14 | 5/5 | n/a | n/a | n/a |
| keyword rule | baseline | 0.700 | 0.783 | 0.704 | 3/14 | 5/5 | n/a | n/a | n/a |
| surface rule | baseline | 0.975 | **0.967** | 0.967 | 14/14 | 5/5 | n/a | n/a | n/a |
| prompt length | baseline | 0.575 | 0.483 | 0.418 | 14/14 | 5/5 | 0.580 | 0.048 | n/a |
| bag of words | baseline | 0.550 | 0.450 | 0.448 | 13/14 | 5/5 | 0.547 | 0.167 | n/a |
| layer-10 | probe | 0.825 | 0.767 | 0.769 | 14/14 | 4/5 | 0.270 | 0.197 | 0.287 (0.433), chance 0.333 |
| layer-25 | probe | 0.800 | 0.733 | 0.736 | 14/14 | 4/5 | 0.267 | 0.138 | 0.297 (0.383), chance 0.333 |
| layer-40 | probe | 0.925 | 0.900 | 0.914 | 14/14 | 5/5 | 0.224 | 0.276 | 0.312 (0.417), chance 0.333 |
| layer-50 | probe | 0.825 | 0.767 | 0.779 | 14/14 | 5/5 | 0.292 | 0.228 | 0.310 (0.417), chance 0.333 |

### The label shuffle moved after the capture

**The rule is that definitions are committed before a real capture and don't move. This one moved after the capture and after its first report, and its new form was written with the old numbers in view. The conclusion didn't change, and the numbers below show why the lower highest values at 20 runs are no stronger support.**

- **Used first.** The capture's report permuted the labels of single rows (`r.Shuffle` over all rows, up to `fc01ef1`). That report is kept as it was in `~/models/eval/layer-probe-20261107`.
- **Why it was replaced.** The folds hold out a paraphrase group because its rows depend on each other. A row-level shuffle takes a group's labels apart while its activations stay close together, so it isn't a null for the task the probes are scored on. A review of this PR found this in the code, not in the scores.
- **What changed.** Only which permutations run: whole groups' label blocks, among groups of one size. Blocks of two sizes can't swap, and the 25 groups here have sizes 1 and 2. Runs, seed, folds, L2, dataset and capture are the same, and between the two reports only the definitions sentence and the 8 shuffle cells differ.
- **Why a block's order is drawn anew.** The dataset writes a computation before its refusal, so a block that keeps its row order lands on another such pair with the real labels. That was measured and rejected: its mean over 1000 runs is 0.532 to 0.601 on execute, above chance, which is the real signal leaking into the control.

Balanced accuracy on shuffled labels, mean (highest), on the capture's activations. The 1000-run reports are deviations and are no definition of anything. Records: `~/models/eval/layer-probe-20261107-groupshuffle`, with the row-level runs from `fc01ef1` in `row-level-1000/` and the kept row order in `row-order-kept-1000/`, whose `built-from.txt` says how it was built.

| Target | Layer | Real | Row by row, 20 runs (first report) | Whole groups, 20 runs (the tables above) | Row by row, 1000 runs | Whole groups, 1000 runs | Row order kept, 1000 runs |
|---|---|---|---|---|---|---|---|
| execute | 10 | 0.975 | 0.461 (0.675) | 0.463 (0.675) | 0.478 (0.750) | 0.483 (0.775) | 0.601 (0.875) |
| execute | 25 | 0.975 | 0.456 (0.700) | 0.423 (0.575) | 0.474 (0.825) | 0.476 (0.775) | 0.595 (0.850) |
| execute | 40 | 0.975 | 0.467 (0.725) | 0.469 (0.625) | 0.469 (0.850) | 0.475 (0.825) | 0.555 (0.850) |
| execute | 50 | 1.000 | 0.454 (0.725) | 0.474 (0.650) | 0.465 (0.800) | 0.475 (0.800) | 0.532 (0.850) |
| family | 10 | 0.767 | 0.301 (0.567) | 0.287 (0.433) | 0.308 (0.633) | 0.306 (0.550) | 0.360 (0.683) |
| family | 25 | 0.733 | 0.290 (0.500) | 0.297 (0.383) | 0.308 (0.650) | 0.308 (0.600) | 0.355 (0.650) |
| family | 40 | 0.900 | 0.305 (0.467) | 0.312 (0.417) | 0.307 (0.550) | 0.313 (0.533) | 0.343 (0.617) |
| family | 50 | 0.767 | 0.300 (0.483) | 0.310 (0.417) | 0.305 (0.517) | 0.313 (0.550) | 0.324 (0.567) |

- **The lower highest values at 20 runs are the luck of 20 draws.** Seven of the eight fell and one stayed. At 1000 runs the two schemes reach about the same: 0.750 to 0.850 against 0.775 to 0.825 on execute, 0.517 to 0.650 against 0.533 to 0.600 on family.
- **The old control was barely harder on these rows.** Its means over 1000 runs are at most about 0.01 lower: 0.465 to 0.478 against 0.475 to 0.483 on execute, and on family 0.305 to 0.308 against 0.306 to 0.313. A run's spread didn't grow either on activations with a planted direction (M75's), about 0.10 on execute under both, probably because 10 of the 15 pairs hold both labels and their order is drawn anew. M75's bound therefore stayed at 0.05.
- **The conclusion holds under all three.** No run in any column of this table reaches its real score. That still excludes chance only, not the text.

### What these scores say

- **Against the surface rule, the bar.** No probe clears it in a way that counts. On execute, three layers tie it and layer 50 is above it by one row of 40. On operation family, where the bar is the gate plus the keyword list, all four are below it. The rule was written after seeing the rows, which favors it, and [clearing it wouldn't rule its cues out](#what-a-result-would-and-wouldnt-mean).
- **Against the other baselines.** On execute all four probes are above the keyword rule, prompt length and bag of words. On operation family all four are above prompt length and bag of words, but by balanced accuracy only layer 40 is above the keyword rule's 0.783: layers 10 and 50 (0.767) and 25 (0.733) are below it. By plain accuracy all four are above its 0.700.
- **Single rows.** All probes get the 14 execute hard negatives right, where the surface rule errs on A-011. Each of layers 10, 25 and 40 has one execute error of its own: layers 10 and 40 a false alarm outside the hard negatives, layer 25 a missed computation. `report.json` doesn't record which rows, and one row is no finding either way.
- **Shuffled labels.** Each real score is above the highest of its 20 shuffles, and no shuffle reaches it, under the scheme that [replaced the first](#the-label-shuffle-moved-after-the-capture) and under the first. That excludes chance, not the text: random weights decode too.
- **Calibration.** Brier and ECE are as the report gives them. Layer 50, with the highest execute accuracy, has the worst Brier and ECE of the four execute probes. 40 rows don't make this meaningful.
- **Pilot prompts.** All 32 predictions (4 prompts, 2 tasks, 4 layers) equal the label. This checks the pipeline.
- **Nothing here compares layers.** The differences between them, layer 50's 1.000 and layer 40's 0.900 among them, are a few rows of 40. The report ranks no layer, and neither does this section. Nothing here tests whether the model uses this information, or whether injecting a result at any of these layers would change its behavior: no intervention ran.

### The report's caveats

- Small sample: this is pipeline validation, not evidence. With 40 rows in 25 groups a difference between layers is noise until shown otherwise, so no layer is selected here and no confidence interval is given: the sample does not support one. A choice needs hundreds to thousands of prompts with a held-out, template-disjoint test set.
- Decodability is not causal usefulness: a linear probe that reads a label from an activation shows that the information is there at that layer, not that the model uses it there, nor that injecting a result at that layer would change its behavior correctly. Only an intervention can show that, and this report runs none.
- Surface cues: on these rows a digit and the absence of the refusal words (aber, ohne, but, without) give the execute label on 39 of 40 smoke rows, and a refusal is longer than its partner. The surface rule and the prompt length are in the tables, and a probe that does not beat them has shown nothing the text's surface does not show. Beating the bag of words is not enough, and the rule was written after seeing the rows. These rows cannot separate a probe that reads what a prompt asks from one that reads those cues.
- The labels are a policy (execute-when-exact-result-asked-v1): execute_expected says execution is helpful, not that it is strictly necessary; execution_need keeps the two apart.

### Not run or not reported

- The real weight on the CPU, resident on Metal, and compared with a resident run.
- A build without the Stage E4 code, on the CPU, resident or on tiny. On Metal with the cache it [gave the same bytes](#on-the-patch-series).
- The other position, `--reasoning default`.
- The capture's overhead: [no figure is taken](#metal-cpu-and-the-expert-cache) from these decodes, and this was not a timed run.

## Next

- **A larger, leakage-resistant evaluation.** Hundreds to thousands of prompts from more templates, with a held-out test set disjoint in templates, paraphrase groups and number templates, and confidence intervals over groups. This is where a layer choice can start. Then calibration and robustness to paraphrase become meaningful. It must break the surface cues: computations that say `aber` or `ohne` and long ones, refusals without those words and short ones, and many numbers that need no execution.
- **The two-stage gate hypothesis.** An early execute gate and a later family controller: compare where each target's score rises, on that larger set.
- **Intervention.** Only after the readout: a frozen base model, a trainable adapter, and whether an injected result changes downstream behavior correctly.
