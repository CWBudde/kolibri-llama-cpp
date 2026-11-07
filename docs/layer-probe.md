# Layer probe (issue #63)

**Result: the instrumentation for issue #63 exists and passes on tiny checkpoints. It has not read a real Kolibri weight, so nothing is known yet about any layer.** It observes only: no base weight changes and no coprocessor runs.

| Part | Where | State |
|---|---|---|
| 44 prompts with oracle validation | `testdata/probe/smoke.jsonl`, `internal/probe`, `go run ./cmd/kolibri-probe generate` or `verify` | done |
| Activation capture | `tools/probe/capture.py`, through libllama's `cb_eval` | done on tiny; real run pending |
| Probes, baselines, report | `go run ./cmd/kolibri-probe report`, `internal/probe` | done on synthetic and tiny activations; real run pending |
| Readout of the real model | [Results pending](#results-pending) | **not run** |

## What a layer number means

The issue's layers 10, 25, 40 and 50 are **one-based**: they count the blocks, 1 to 50, and 50 can't be read any other way, because the graph's blocks are numbered 0 to 49 (`il` in `src/models/kolibri1.cpp`). Layer N is node **`l_out-(N-1)`**:

| Issue's layer | Node | What follows it |
|---|---|---|
| 10 | `l_out-9` | block 11's input norm |
| 25 | `l_out-24` | block 26's input norm |
| 40 | `l_out-39` | block 41's input norm |
| 50 | `l_out-49` | `output_norm`, then the output head |

**Normalization.** `l_out-il` is the block's residual sum, `post_ffw_norm(routed + shared experts) + ffn_inp`, and `build_cvec` passes it through with no control vector loaded. So the post-FFN norm sits inside the sum, and nothing normalizes the sum itself. It is read **before** the next block's input norm and, at layer 50, **before `output_norm`** (`result_norm`). Layer 50 is therefore not the embedding that `llama_get_embeddings` would return.

Reading this from the source doesn't prove it for a given build, so the capture reads it from the model's graph on its first decode (`Capture.check_graph`) and refuses to go on otherwise: `l_out-il` must have the sources `(post-FFN norm, ffn_inp-il)`, `norm-(il+1)` must read it, and the last one must feed `norm`. `capture.json` keeps what it found under `graph_check`.

## The prompts

`testdata/probe/smoke.jsonl` holds 44 rows: the issue's 4 pilot prompts word for word (`split: pilot`) and 40 smoke prompts (`split: smoke`), 10 per category.

| Category | Execute | Family | What varies |
|---|---|---|---|
| A `no_execution` | no | `none` | facts, explanations, writing; hard negatives ask for a recalled number or mention an operation |
| B `exact_arithmetic` | yes | `arithmetic` | multiplication, powers, factorial, GCD; magnitude, thousands separators, language |
| C `small_algorithm` | yes | `algorithm` | sorting, primality, Fibonacci, filtering; input length, order, phrasing, language |
| D `explicit_non_execution` | no | `none` | the numbers of a B or C partner in the same group, with the instruction not to compute |

- **Schema.** The issue's eight fields come first, in its order, then `language`, `template_id`, `numeric_seed`, `prompt_length`, `explicit_execution_constraint` and `expected_answer_or_oracle`. Added are `operation`, `execution_need`, `label_policy`, `hard_negative` and `varied_axis`. The metadata of a capture (model and template revision, layer index, dtype) is in its `capture.json`, since it isn't a property of a prompt.
- **Labels are a policy.** `execute_expected` follows `label_policy` (`execute-when-exact-result-asked-v1`) and means execution is helpful. `execution_need` separates `helpful` from `necessary`: an exact result is necessary when a number involved reaches 100,000, so that 5 numbers to sort stay `helpful`. A D prompt has family `none`, because it needs nothing; `operation` keeps what it mentions.
- **Pairs.** 10 groups hold a computation and its refusal on the same numbers, so that a keyword can't separate them. They differ in the instruction only (`varied_axis`), and other rows differ in language or wording only.
- **Determinism.** `kolibri-probe generate` draws from PCG with a fixed seed (63) and a per-group stream. `TestRegenerateBytes` regenerates the file and compares bytes.
- **Oracle.** `internal/probe/oracle.go` shares no code with the generator. It reads a row's numbers from the prompt's text and computes the answer by other algorithms: `math/big`'s `Mul`, `Exp`, `MulRange`, `GCD` and `ProbablyPrime`, fast doubling, an insertion sort. Facts come from its own table. A D prompt must carry words that forbid execution.
- **Leak checks.** `VerifyStructure` refuses a template or a numeric seed in two groups, and two prompts of different groups with a masked word overlap of 0.8 or more. The pilot's groups are in no smoke row.

## Capture

The issue's activations are read through `cb_eval`, the callback route `cmd/kolibri-routing` and `tools/gguf/check_*.py` already use. No fork change was needed. The tool is Python because the repo's libllama binding (`tools/tokenizer/common.py`, `check_model.Runner`) is; everything after it reads NumPy files from Go.

- **The row.** The last prompt token's row. Each prompt is decoded on a cleared memory with an output at its last position only. Below the last block a node has one row per token of the ubatch; the last block's rows are cut to the outputs by the graph. A prompt longer than the ubatch runs over several graph computations, and the capture keeps the last one's last row and checks that the rows add up to the prompt.
- **The prompt.** The model's own `tokenizer.chat_template`, rendered by the reference's renderer (`render_jinja_template`) with `add_generation_prompt` and `reasoning_effort: "none"`. That prefills the empty think block, so the last prompt token is `\n\n`, the same token for every prompt, right where the answer starts. A definition: `--reasoning default` captures the other position, as a separate run.
- **Determinism.** No sampling, one decode per prompt, fixed threads and ubatch. `--repeat` decodes the pilot prompts again and compares activations and logits exactly.
- **Stored.** `layer-<N>.npy`, `<f4` `[44, n_embd]` in dataset order, and `capture.json`: the sha256 of the model, of libllama (`server_sha256`: libllama does the server's work, no server process is involved), of the ggml libraries and of the dataset; the fork's head; the template's sha256 and pinned revision, and whether it is the released one; dtype, shape, one- and zero-based index and node per layer; device, threads, requested and effective ubatch, cache slots; each prompt's tokens. A directory that holds a capture is refused, because results may exist nowhere else.

### Non-mutating

The callback only reads, and the check is exact: `np.array_equal`, no tolerance. All prompts are first decoded by a model and context that never had a callback, and the capturing decode must give the same logits. A capture run does this itself and records `logits_check`, and `kolibri-probe report` refuses a capture whose logits differ.

`tools/probe/check_capture.py` shows the same on tiny checkpoints, with F32 and BF16 weights, on the CPU and on Metal, reading 4 layers and all layers. `--layers` scales the indices: the 6-layer tiny model reads 2,3,5,6, and the 50-layer `-pattern` model reads the real 10,25,40,50.

### Metal, CPU and the expert cache

| Path | On tiny (`check_capture.py`) | On the real model |
|---|---|---|
| CPU | passes | not run |
| Metal | passes | not run |
| Expert cache, CPU and Metal, ubatch capped at 3 | passes: exactly the logits of the cached model without a callback, and logits and activations exactly a resident run's at the same ubatch | not run |

- **The cache serves one context per model** ("the expert cache serves one context per model"), so the plain pass and the capture load the model apart, one after the other.
- **The cache caps the ubatch** at the fewest slots divided by 6: 21 at `--expert-cache 128`. So a prompt spans several computations, and the numbers belong to that ubatch: the ubatch changes the order of the sums ([expert-streaming.md](expert-streaming.md)), and on Metal so does the number of outputs in the last block. `check_capture.py` shows the second on tiny: there the last block's row isn't bit-identical between a decode with one output and one with all outputs, while the CPU's is, and every layer below it is exact on both. A capture's numbers are reproducible for its recorded settings, and two captures compare only at equal settings.
- **Overhead.** The structure is known: each read node ends a graph split, so 4 more splits per ubatch, a copy of `n_tokens x n_embd` floats per read node, and a Python call for each node the scheduler asks about. The real cost is pending: `capture.json` records `plain_decode_seconds` and `capture_decode_seconds`, and they aren't a benchmark.

## Probes and report

`kolibri-probe report -capture DIR` trains on activations alone and scores only held-out rows. The options below are **definitions, committed before any real capture, and they must not move afterwards**: `DefaultOptions` in `internal/probe/report.go`, the dataset, and the capture defaults in `tools/probe/capture.py`.

- **Probes.** One multinomial logistic regression per layer and target: execute or not, and operation family (`none`, `arithmetic`, `algorithm`). L2 strength 0.01, features standardized with the training rows' mean and deviation and scaled to unit expected squared norm, no feature selection. The solver is deterministic (Nesterov's method from zero) and stays in the span of the training rows, since there are far more dimensions than rows.
- **Folds.** 5 folds by `paraphrase_group`, over the smoke split. `CheckFolds` runs before any training and refuses a group, template or numeric seed in train and test of a fold, whoever made the split.
- **Baselines**, on the same folds: majority of the training rows, a fixed keyword rule, and a learned bag of words with the same classifier and regularization. The rule's lists were written by the hand that wrote the prompts, which favors it. Words see what the probe does not, so a probe that doesn't beat them has shown nothing the text doesn't show.
- **Scores.** Accuracy, balanced accuracy, macro F1, precision and recall of execute, hard negatives right, agreement of same-label rows within a group, Brier score and a 5-bin calibration error.
- **Shuffled labels.** 20 runs per probe, same folds. A probe must beat them by more than their spread.
- **No confidence intervals.** The issue asks for them only where the sample supports them, and 40 rows in 25 groups doesn't. The report says so and gives none.
- **Pilot prompts.** Trained on all smoke rows and read on the 4 pilot prompts, whose groups the smoke split doesn't share. One prompt per category: a pipeline check.

### What a result would and wouldn't mean

- **Small sample.** 40 prompts validate the pipeline. They don't select a layer, and the report ranks none.
- **Decodable isn't useful.** A probe that reads a label from an activation shows that the information is there at that layer. It doesn't show that the model uses it there, or that injecting a result there would change behavior correctly. Only an intervention can show that, and that comes later, with a frozen base model and a trainable adapter (#61).
- **Random weights decode too.** A random-weight tiny model's activations score above chance on these labels, probably because prompts that share words stay close through random projections too. A score on the real model therefore means little by itself, whatever the shuffled-label control says: that control excludes chance, not the text. The bag of words is the bar.
- **Labels are a policy.** A different policy gives different labels.

### Controls

| Control | Mutation | Must happen | Where |
|---|---|---|---|
| M75 | shuffled labels | the same probe on the same activations scores near chance, and no shuffle reaches the real score | `TestM75ShuffledLabelsNearChance` |
| M76 | a split that breaks the groups | refused with `ErrGroupLeak`; on group-fingerprint activations it would score near 1, grouped folds near chance | `TestM76LeakingSplitRefused` |
| M77 | a capture that writes into a node | logits differ, the capture exits 1, the report refuses it | `check_capture.py` |
| M78 | a wrong answer or a changed operand | the oracle fails the row | `TestM78OracleCatchesCorruption` |
| M79 | the issue's numbers taken as zero-based | refused by the graph check and, at the last layer, by the missing node | `check_capture.py` |

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

## Results pending

**No real capture has run, so this section has no numbers.** The real run fills in:

- [ ] the capture's `logits_check` and `repeatable` on the real model, Metal, with the cache;
- [ ] the overhead from `plain_decode_seconds` and `capture_decode_seconds`;
- [ ] the report's tables for layers 10, 25, 40 and 50 against the three baselines and the shuffled labels.

## Next, once the pipeline has run

- **A larger, leakage-resistant evaluation.** Hundreds to thousands of prompts from more templates, with a held-out test set disjoint in templates, paraphrase groups and number templates, and confidence intervals over groups. This is where a layer choice can start. Then calibration and robustness to paraphrase become meaningful.
- **The two-stage gate hypothesis.** An early execute gate and a later family controller: compare where each target's score rises, on that larger set.
- **Intervention.** Only after the readout: a frozen base model, a trainable adapter, and whether an injected result changes downstream behavior correctly.
