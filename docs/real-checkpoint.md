# The real checkpoint on a 48 GB Mac

This document covers the full Kolibri-1 BF16 checkpoint:

- download and conversion to GGUF;
- the structural checks on the converted file;
- the first unquantized runs;
- the quantized variants that fit a 48 GB Apple Silicon Mac.

The tiny synthetic checkpoints of [gguf-conversion.md](gguf-conversion.md),
[model.md](model.md) and [attention.md](attention.md) cover the converter and
the graph in detail. This document only shows that those results carry over
to the real weights.

**What this is not:** a numerical validation. Without the official vLLM
reference (PLAN Phase 0 and Phase 6), a coherent answer and a low KL divergence
cannot tell a port bug from quantization damage. Every quality number below
compares llama.cpp with llama.cpp: a quantized GGUF against this port's own BF16
GGUF.

Measured on a Mac17,9 (Apple M5 Pro, 15 cores, 48 GB) at fork commit
`e1a553f5f` (`feat/kolibri` after the chat-template merge).

## Download

```sh
.venv/bin/hf download Aleph-Alpha/Kolibri-1-BF16 \
    --revision 7a8f290e7858825c3cf5e4c447ba68345de9f1d3 \
    --local-dir ~/models/Kolibri-1-BF16
```

32 shards, 156,206,149,120 bytes, about 70 minutes at 37 MiB/s. Before the
conversion, `check_real.py` hashes every shard against the inventory's pinned
size and sha256:

```text
$ .venv/bin/python tools/gguf/check_real.py --llama-cpp third_party/llama.cpp \
      --model-dir ~/models/Kolibri-1-BF16
PASS shards: 32/32 match the pinned size and sha256 (58 s)
```

## Conversion

```sh
/usr/bin/time -l .venv/bin/python third_party/llama.cpp/convert_hf_to_gguf.py \
    ~/models/Kolibri-1-BF16 --outtype bf16 --outfile ~/models/Kolibri-1-BF16.gguf
```

```text
INFO:gguf.gguf_writer:/Users/christian/models/Kolibri-1-BF16.gguf: n_tensors = 903, total_size = 156.3G
      466.60 real       172.76 user       234.77 sys
          8524414976  maximum resident set size
          8025841608  peak memory footprint
```

- **Streaming.** No converter change was needed. The base class loads tensors
  lazily, and `KolibriModel` buffers the 1,152 expert tensors of one layer as
  lazy references until it stacks them. The peak footprint stays at 7.5 GiB
  for a 146 GiB checkpoint.
- **Shards in layer order.** Each layer spans at most two consecutive shards
  ([checkpoint.md](checkpoint.md)), so a layer's experts are complete soon
  after its first shard is read.
- **Disk.** The GGUF is 156,310,433,312 bytes, the same as the safetensors plus
  the tokenizer and metadata. Converting needs about 312 GB free for both.

## Structural checks on the real GGUF

`tools/gguf/check_real.py` runs the comparisons of `check_tensors.py` on the
full checkpoint. Its expectations come from the inventory
(`inventory/bf16/summary.json`), not from the converter. It expands the 21
tensor classes over 50 layers into 903 GGUF tensors and 58,353 HF sources.

It reads raw BF16 bytes from the safetensors headers, without torch. Both files
are memory-mapped, so a full run reads 312 GB and peaks at about 1 GB of
footprint.

```text
$ .venv/bin/python tools/gguf/check_real.py --llama-cpp third_party/llama.cpp \
      --model-dir ~/models/Kolibri-1-BF16 --gguf ~/models/Kolibri-1-BF16.gguf --no-hash
PASS tensor set: 903 tensors, inventory 903
PASS token_embd (1/1): ne [2560, 128000], BF16, data bit-exact
PASS attn_norm (50/50): ne [2560], F32, data bit-exact
...
PASS ffn_gate_inp (50/50): ne [2560, 384], F32, data bit-exact
PASS exp_probs_b (50/50): ne [384], F32, data bit-exact
PASS ffn_gate_exps (50/50): ne [2560, 512, 384], BF16, 384 experts stacked, data bit-exact
PASS ffn_up_exps (50/50): ne [2560, 512, 384], BF16, 384 experts stacked, data bit-exact
PASS ffn_down_exps (50/50): ne [512, 2560, 384], BF16, 384 experts stacked, data bit-exact
...
PASS output (1/1): ne [2560, 128000], BF16, data bit-exact
info data comparison took 637 s
PASS metadata: general.architecture
PASS metadata: kolibri.block_count
PASS metadata: kolibri.expert_count
PASS metadata: kolibri.expert_used_count
PASS metadata: kolibri.attention.sliding_window_pattern
PASS parameters: 78,103,074,560 in the GGUF, inventory 78,103,074,560
```

`check_metadata.py --gguf` reads the hyperparameters from the converted file,
instead of from a `--vocab-only` conversion, and compares them with
`config.json` and the inventory. All 21 assertions pass on the real GGUF.

**Mutation.** I ran the check on the tiny fixture with the manifest in place of
the inventory, then flipped one byte of expert 3's `gate_proj` in layer 0
after conversion. The data comparison catches it:

```text
FAIL blk.0.ffn_gate_exps.weight: data differs from model.layers.0.mlp.experts.3.gate_proj.weight
```

## Unquantized run

```sh
build/bin/llama-completion -m ~/models/Kolibri-1-BF16.gguf -dev none -ngl 0 \
    -c 512 -n 32 --temp 0 --seed 1 -no-cnv -p "The capital of Germany is"
```

```text
The capital of Germany is Berlin.

We are going to Berlin.

We are going to Berlin.

We
prompt eval time =    7598.19 ms /     5 tokens ( 1519.64 ms per token,     0.66 tokens per second)
       eval time =   21363.46 ms /    31 runs   (  689.14 ms per token,     1.45 tokens per second)
```

- The complete 50-layer graph builds and runs on the real weights without a
  shape error or an unsupported op.
- The greedy repetition is normal for a raw text prompt without the chat
  template.
- Generation runs on the CPU from the memory-mapped file. Each token touches
  6 experts per layer, about 2.4 GB of expert weights. The OS page cache serves
  the experts it holds, and the SSD the rest.

### Streaming the routed experts with `--cpu-moe`

`-ngl 99 --cpu-moe` puts attention, the router and the shared expert on
Metal. The routed experts stay on the CPU and are read on demand from the
memory-mapped file. Without patch 0009 this dies in the warmup decode for
every file above the Metal working set. The numbers with the patch are from
fork commit `6d51eaf70`.

```text
EXC_BAD_ACCESS (SIGBUS) KERN_PROTECTION_FAILURE
libggml-cpu   ggml_compute_forward_mul_mat_id
libggml-cpu   ggml_graph_compute_thread
libllama      llama_context::decode
libllama-common common_init_from_params
```

| GGUF | Size | `-ngl 99 --cpu-moe` without 0009 |
|---|---|---|
| Q3_K experts, rest Q8_0 | 33,717 MiB | runs, 17.6 tokens/s |
| Q8_0 (with `--no-repack`) | 79,279 MiB | SIGBUS (exit 138) |
| BF16 | 149,065 MiB | SIGBUS (exit 138) |

**Cause.** With mmap, llama.cpp gave each backend one buffer over the file
range from its first tensor to its last. The Metal tensors sit in all 50
layers, so the Metal range covered nearly the whole file, routed experts
included. Two steps follow:

- `ggml-metal-device.m` wraps that range in no-copy shared buffers
  (`newBufferWithBytesNoCopy`).
- A background thread requests residency for those buffers every 5 ms.

Above the working set of 37 GiB, the CPU then faults on expert pages of the
same mapping. With `GGML_METAL_NO_RESIDENCY=1` the Q8_0 run survives, at
57 seconds per token.

**Fix (patch 0009).** `get_mapping_ranges` splits a context's tensors into
runs of nearby tensors, merging gaps up to 16 MiB, and each run gets its own
buffer. The expert blocks between the runs stay outside every Metal buffer:

| GGUF, flags | `MTL0_Mapped` without 0009 | with 0009 |
|---|---|---|
| Q8_0, `-ngl 99 --cpu-moe --no-repack` | nearly the whole 79 GiB file | 2,447 MiB in 152 buffers |
| BF16, `-ngl 99 --cpu-moe` | nearly the whole 149 GiB file | 4,440 MiB in 52 buffers |
| IQ3_XXS/IQ4_XS, `-ngl 99` | 33,904 MiB | 33,572 MiB in 2 buffers |

- Both large files now run. Q8_0 and BF16 answer "Berlin".
- The model that fits keeps its speed: `llama-bench` measures 1,347 / 63.7
  tokens/s (pp512 / tg128), against 1,351 / 64.0 before.
- Its 332 MiB token embeddings, which stay on the CPU, are no longer wrapped
  by Metal as well.
- The tiny-fixture checks (`check_model.py`, `check_moe.py`, `check_attn.py`)
  and llama.cpp's 43 `ctest` tests pass.

**Speed: streaming works, but Metal adds little.** Q8_0 with `--no-repack`,
10 threads, a 21-token prompt and 128 greedy tokens:

| Setup | Prompt tokens/s | Generation tokens/s |
|---|---|---|
| CPU only (`-dev none`) | 4.81 | 10.36 |
| Metal, `--n-cpu-moe 50` (= `--cpu-moe`) | 4.82 | 9.49 |
| Metal, `--n-cpu-moe 40` (10 layers of experts on Metal) | 4.73 | 11.25 |
| Metal, `--n-cpu-moe 35` (15 layers) | 4.06 | 11.29 |

- **Why so little:** the routed experts are 96.7% of the weights. The CPU
  reads and multiplies them either way. Moving the rest to Metal saves
  little and adds two CPU/GPU handoffs per layer.
- **`llama-bench` agrees:** pp512 is 52.8 tokens/s CPU-only and 28.0 with
  `--n-cpu-moe 50`, both with `-t 10`.
- **BF16:** 2.07 tokens/s with `-ngl 99 --cpu-moe` and 2.50 CPU-only, both
  with `-t 10` and 32 tokens.
- **Partial offload:** experts on Metal for 10 to 15 layers gain about 9%.
  That is within the run-to-run spread of the page cache, so these are single
  runs, not a ranking.
- **Thread count:** use `-t 10`. The default of 5 threads (the performance
  cores) gives CPU-only Q8_0 15.0 instead of 20.9 tokens/s in `llama-bench`
  tg32.
- **`--no-repack` is required** for Q8_0 with `--cpu-moe`. Otherwise the CPU
  experts land in the repack buffer, a 77 GB copy in RAM.

`llama-server -m Kolibri-1-Q8_0.gguf -ngl 99 --cpu-moe --no-repack -t 10
--jinja` answers "What is 17 * 23?" (`reasoning_effort: "none"`) with "391".
It answers the German sky question in two sentences close to the Q3_K mix's
answer below ("Da die Atmosphäre das blaue Licht stärker streut als andere
Farben, erscheint der Himmel blau."), at 10.1 tokens/s over 479 tokens.

The split across backends changes no result materially. KLD against this
port's BF16, on the same 20 × 512 tokens as in "Quantization loss" below:

| Q8_0 run | Mean KLD | 99% KLD | Same top token | PPL(Q)/PPL(base) |
|---|---|---|---|---|
| CPU, `--no-repack` | 0.050 ± 0.005 | 0.76 | 92.5 ± 0.4% | 1.003 ± 0.005 |
| Metal, `--cpu-moe --no-repack -t 10 -ub 2048` | 0.055 ± 0.005 | 1.06 | 92.4 ± 0.4% | 0.999 ± 0.005 |

The small rise fits the Metal near-tie note in PLAN Phase 6: the router runs
on Metal here.

For a 48 GB Mac, `--cpu-moe` streaming therefore makes Q8_0 and BF16 usable
next to the CPU-only runs, but not faster. The quantized model that fits on
Metal stays about six times faster.

## Quantization

`llama-quantize --dry-run` sizes on the BF16 GGUF:

| Type | Size | BPW |
|---|---|---|
| Q8_0 | 79,279 MiB | 8.51 |
| IQ4_XS | 40,286 MiB | 4.33 |
| Q3_K_M | 35,781 MiB | 3.84 |
| Q8_0, routed experts Q3_K | 33,717 MiB | 3.62 |
| Q3_K_S | 32,297 MiB | 3.47 |
| IQ3_XXS | 30,302 MiB | 3.25 |

The first candidate keeps everything except the routed experts at Q8_0:

```sh
build/bin/llama-quantize --tensor-type ffn_gate_exps=q3_k --tensor-type ffn_up_exps=q3_k \
    --tensor-type ffn_down_exps=q3_k ~/models/Kolibri-1-BF16.gguf \
    ~/models/Kolibri-1-Q3_K-exps-Q8_0.gguf Q8_0
```

- **Routed experts:** Q3_K. They are 96.7% of the parameters.
- **Q8_0:** attention, the shared expert, the token embeddings and the output.
- **F32:** the router (`ffn_gate_inp`), the correction bias (`exp_probs_b`)
  and every norm. `llama-quantize` never quantizes `ffn_gate_inp`, and it
  leaves 1D tensors alone.

`llama-quantize` streams tensor by tensor:

| Output | Time | Peak RSS |
|---|---|---|
| Q8_0 | 40 s | 5.4 GB |
| Q3_K experts | 70 s | 5.4 GB |

## The quantized model on Metal

```sh
build/bin/llama-completion -m ~/models/Kolibri-1-Q3_K-exps-Q8_0.gguf -ngl 99 \
    -c 512 -n 32 --temp 0 --seed 1 -no-cnv -p "The capital of Germany is"
```

```text
The capital of Germany is Berlin.
The capital of France is Paris.
The capital of Italy is Rome.
The capital of Spain is Madrid.
The capital of Portugal is Lisbon.
       load time =   13228.82 ms
prompt eval time =      70.31 ms /     5 tokens (   14.06 ms per token,    71.12 tokens per second)
       eval time =     517.49 ms /    31 runs   (   16.69 ms per token,    59.90 tokens per second)
```

Memory at 32k context (`-c 32768 -v`), with the default Metal limit (no
`iogpu.wired_limit_mb` change):

```text
| memory breakdown [MiB]  | total    free     self   model   context   compute    unaccounted |
|   - MTL0 (Apple M5 Pro) | 38338 = 38338 + (34384 = 33384 +     740 +     260) +      -34384 |
|   - Host                |                    418 =   332 +       0 +      86                |
llama_kv_cache: size =  640.00 MiB ( 32768 cells,  10 layers,  1/1 seqs), K (f16):  320.00 MiB, V (f16):  320.00 MiB
llama_kv_cache: size =  100.00 MiB (  1280 cells,  40 layers,  1/1 seqs), K (f16):   50.00 MiB, V (f16):   50.00 MiB
```

- **Full-attention KV cache:** only the 10 full-attention layers grow with
  context, at 20 KiB per token, or 640 MiB at 32k.
- **Sliding-window KV cache:** the 40 sliding layers hold 1,280 cells
  regardless of context.
- **Headroom:** about 3.9 GiB below the default Metal working set of
  38,338 MiB.
- **Token embeddings:** they stay in host memory, at 332 MiB.

## Quantization loss against this port's BF16

The reference is `llama-perplexity` on the BF16 GGUF, CPU-only, over
wikitext-2 `wiki.test.raw`:

- 20 chunks of 512 tokens;
- logits saved with `--kl-divergence-base`;
- 50 minutes, at 570 s per pass of 4 chunks;
- result: PPL = 27.8827 ± 1.36009.

| Model | Backend | Mean KLD | 99% KLD | Same top token | PPL(Q)/PPL(base) |
|---|---|---|---|---|---|
| Q3_K experts, rest Q8_0 | Metal | 0.108 ± 0.008 | 1.81 | 88.4 ± 0.4% | 0.993 ± 0.007 |
| Q8_0 | CPU, `--no-repack` | 0.050 ± 0.005 | 0.76 | 92.5 ± 0.4% | 1.003 ± 0.005 |

The Q3 row includes backend differences, because BF16 ran on the CPU and Q3
on Metal. The Q8_0 row compares like with like on the CPU, at 213 s per pass.

**Q8_0 KLD is high.** Here Q8_0 reaches 0.050 with 92.5% same top token, on
the same backend as its reference. For dense models, Q8_0 usually lands near
KLD 0.001 with over 99% same top token. A likely amplifier is the router: a small error in a
router logit that sits near the top-6 boundary swaps an expert. That fits the
PLAN Phase 6 note on Metal near-ties and the Phase 7 item "measure how often
quantization changes Top-6 expert selection". Per the Phase 6 hard gate, this
stays undiagnosed until the BF16 port matches vLLM.

**Q8_0 on the CPU: use `--no-repack`.** The first Q8_0 run used the CPU
backend's default weight repacking. macOS killed it (exit 137) after 8 chunks,
with 11.7 of 13.3 GB swap in use. Other processes also used memory at the
time, so repacking is the suspect, not a confirmed cause. With `--no-repack`
the weights stay memory-mapped, and the footprint stays at about 2.5 GB.

## Importance matrix and IQ3 variants

### Calibration text

`tools/quant/calibration.py` builds the calibration text:

- 45% English: the start of wikitext-2 `wiki.train.raw`;
- 35% German: Wikipedia lead sections from fixed offsets of
  `wikimedia/wikipedia` 20231101.de;
- 20% code: C++, Python and Go files of this repo and of llama.cpp
  `e1a553f5f`, each checked against a pinned sha256;
- 522,320 bytes, 128,098 tokens, sha256 `e6fc52c9…`.

English wikitext alone does not reach enough experts. A test run over 8 chunks
of `wiki.train.raw` left 12 to 28% of the experts in layers 46 to 49 without a
token. `llama-quantize` gives an expert without data uniform weights, as if
there were no imatrix.

```sh
.venv/bin/python tools/quant/calibration.py \
    --wikitext ~/models/eval/wikitext-2-raw/wiki.train.raw -o ~/models/eval/kolibri-calibration.txt
build/bin/llama-imatrix -m ~/models/Kolibri-1-Q8_0.gguf -f ~/models/eval/kolibri-calibration.txt \
    -dev none -ngl 0 --no-repack -c 512 -b 2048 -ub 2048 -o ~/models/eval/kolibri-imatrix.gguf
```

- **Source model:** Q8_0 on the CPU. BF16 would take about 3 times as long
  (570 s per pass of 4 chunks in the KLD base run).
- **`-ub 2048`:** each pass reads the memory-mapped weights once for 4 chunks
  instead of 4 times, so a pass takes 80 s instead of 213 s.
- **Run:** 250 chunks in 87 minutes, peak footprint 4.9 GB.

### Expert coverage

Coverage is read from the imatrix file's per-expert counts:

| Layers | Experts with data | Median tokens per expert |
|---|---|---|
| 0, 1, 2 | 41.1%, 30.7%, 66.1% | 0, 0, 476 |
| 3 to 49 | 84.9% to 99.7% | 616 to 1,551 |

1,888 of the 19,200 (layer, expert) pairs have fewer than 32 tokens. Layers
0 and 1 route most tokens to a minority of their experts, so more calibration
text would probably not fill them.

### Variants

Each variant:

- quantizes only the routed experts, from the BF16 GGUF;
- keeps Q8_0 elsewhere and F32 for the router and the norms.

KLD is against this port's BF16 on `wiki.test.raw`, 20 × 512 tokens. Speed
is `llama-bench -ngl 99 -p 512 -n 128 -r 3` on Metal.

| Experts (gate, up / down) | imatrix | Size | Mean KLD | 99% KLD | Same top token | PPL(Q)/PPL(base) | RMS Δp | pp512 tokens/s | tg128 tokens/s | Quantize time |
|---|---|---|---|---|---|---|---|---|---|---|
| Q3_K / Q3_K | no | 33,717 MiB | 0.108 ± 0.008 | 1.81 | 88.4 ± 0.4% | 0.993 ± 0.007 | 7.38% | 1,223 | 60.7 | 70 s |
| Q3_K / Q3_K | yes | 33,717 MiB | 0.108 ± 0.008 | 2.03 | 88.4 ± 0.4% | 1.025 ± 0.008 | 7.51% | 1,219 | 59.8 | 134 s |
| IQ3_S / IQ3_S | yes | 33,717 MiB | 0.100 ± 0.007 | 1.90 | 88.5 ± 0.4% | 0.990 ± 0.007 | 7.15% | 1,332 | 61.8 | 846 s |
| IQ3_XXS / IQ3_XXS | yes | 30,342 MiB | 0.107 ± 0.007 | 1.86 | 88.0 ± 0.5% | 1.031 ± 0.008 | 7.21% | 1,367 | 63.0 | 1,306 s |
| IQ3_XXS / IQ4_XS | yes | 33,904 MiB | 0.095 ± 0.007 | 1.82 | 88.5 ± 0.4% | 1.016 ± 0.007 | 6.70% | 1,351 | 64.0 | 1,151 s |

- **The imatrix does not help Q3_K here.** The mean KLD is unchanged.
- **The best KLD is IQ3_XXS gate/up with IQ4_XS down.** It is 12% below the
  Q3_K mix, 187 MiB larger, and generates 5% faster. The gap is about 1.3
  standard errors of the separate runs. The runs share their tokens, so the
  paired difference is probably tighter, but `llama-perplexity` does not
  report it.
- **IQ3_XXS for all experts** matches the Q3_K mix at 3.3 GiB less, which
  leaves room for longer contexts.
- **The IQ types are faster on Metal than Q3_K here**, by 9 to 12% in prompt
  processing and 2 to 5% in generation.
- **A floor:** Q8_0 alone reaches KLD 0.050, about half of every 3-bit
  variant's KLD. Better expert quantization cannot remove that part.
- **Caveats:**
  - The KLD text is English wikitext, and the calibration includes wikitext's
    train split.
  - As above, this measures quantization loss against this port's own BF16,
    not quality.

The IQ3_XXS/IQ4_XS file at 32k context (`-c 32768 -v`):

```text
| memory breakdown [MiB]  | total    free     self   model   context   compute    unaccounted |
|   - MTL0 (Apple M5 Pro) | 38338 = 3101 + (34904 = 33904 +     740 +     260) +         332 |
|   - Host                |                   375 =   332 +       0 +      43                |
```

That leaves 3.4 GiB below the default Metal working set.

```sh
build/bin/llama-quantize --imatrix ~/models/eval/kolibri-imatrix.gguf \
    --tensor-type ffn_gate_exps=iq3_xxs --tensor-type ffn_up_exps=iq3_xxs \
    --tensor-type ffn_down_exps=iq4_xs ~/models/Kolibri-1-BF16.gguf \
    ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf Q8_0
```

## Raw German completions repeat, BF16 included

Greedy raw completions (`llama-completion -no-cnv --temp 0`, no chat
template) of short German prompts fall into repetition on every file,
including the unquantized BF16 on the CPU:

| Prompt | BF16 (CPU) | Q8_0 (CPU) | IQ3_XXS/IQ4_XS (Metal) |
|---|---|---|---|
| "Die Hauptstadt von Deutschland ist" | " Deutschland Deutschland Deutschland …" | ", dass, dass, dass …" | " Deutschland Deutschland Deutschland …" |
| "Der Rhein ist ein Fluss, der" | | | ", der, der, der …" |
| "Frage: Was ist die Hauptstadt von Deutschland?\nAntwort:" | | | " Berlin\n\nDie Hauptstadt von Deutschland ist Berlin. Berlin ist die größte Stadt Deutschlands und liegt im Osten des Landes." |
| "The capital of Germany is" | " Berlin." (see above) | " Berlin." | " Berlin. Berlin is the largest city in Germany and the capital." |

What is ruled out:

- **Tokenization:** llama.cpp and the reference tokenizer give the same
  five ids, `452 22090 493 1678 2459`.
- **A missing BOS:** neither side adds one. The reference has
  `add_bos_token: False` and no `bos_token`, and the GGUF has
  `tokenizer.ggml.add_bos_token = False`.

German through the chat template works: the 528-token German answer below
is coherent. Kolibri is a post-trained reasoning model, so raw continuation
may be its genuine behavior, but only the reference can tell. Per the
Phase 6 gate, this goes to the greedy-sequence comparison against vLLM,
not into a diagnosis here.

## Chat on the quantized model

```sh
build/bin/llama-server -m ~/models/Kolibri-1-Q3_K-exps-Q8_0.gguf -ngl 99 -c 16384 --jinja --port 8099
```

Three OpenAI-style requests, greedy (`temperature: 0`):

| Request | Result | Speed |
|---|---|---|
| "Erkläre in zwei Sätzen, warum der Himmel blau ist." (default reasoning) | 528 tokens; `reasoning_content` holds the plan; `content`: "Das Sonnenlicht besteht aus allen Farben des Regenbogens. Da die Erdatmosphäre das blaue Licht stärker streut als die anderen Farben, erscheint der Himmel blau."; `finish_reason: stop` | 58.7 tokens/s |
| "What is 17 * 23? Answer with the number only." with `reasoning_effort: "none"` | empty reasoning, `content`: "391", 4 tokens | 58.2 tokens/s |
| "What is the weather in Berlin right now?" with a `get_weather` tool and `reasoning_effort: "low"` | `tool_calls`: `get_weather({"city": "Berlin"})`, `finish_reason: tool_calls` | 59.3 tokens/s |

On the real weights, as on the fixtures of [chat.md](chat.md):

- the reasoning split works;
- the thinking-off prefill works;
- the tool-call parser works;
- generation stops on the reference's EOS.

## Reproduce

All files live outside the repo, in `~/models`.

1. Download the checkpoint (156 GB) and run `check_real.py` without `--gguf`
   to hash the shards.
2. Convert with `--outtype bf16` (156 GB more), then run `check_real.py` with
   `--gguf` and `check_metadata.py --gguf`.
3. Delete the safetensors. Re-download them from the pinned revision if
   needed.
4. Run `llama-quantize` as above.
5. Get wikitext-2 with `third_party/llama.cpp/scripts/get-wikitext-2.sh`.
6. Build the calibration text and the imatrix, then quantize the IQ variants,
   as in "Importance matrix and IQ3 variants". The imatrix takes 87 minutes;
   each IQ variant takes 14 to 22 minutes to quantize.

Disk peaks at about 312 GB during the conversion. After that it is the BF16
GGUF plus the quantized files.
