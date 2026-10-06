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
GGUF. The exception is "Reference forward on the real weights", which compares
the BF16 GGUF with a torch port of the vLLM reference code, not with vLLM.

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
quantization changes Top-6 expert selection".

The reference forward (below) makes that amplification plausible:

- On the reference's own routing, the 6th and 7th expert scores lie within
  0.01 of each other for 22% of the (token, layer) pairs.
- Rounding the reference itself to BF16 already costs KLD 0.033 against its
  float32 run. Q8_0's 0.050 on top of BF16 is in that range.

It does not run the Q8_0 GGUF or its quantization path, though, so it cannot
rule out an additional Q8_0 defect. That stays open until Q8_0's expert
selections are measured against BF16 (the Phase 7 Top-6 item).

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

| Prompt | BF16 (CPU) | Q8_0 (CPU) | Q8_0 (Metal, `--cpu-moe`) | IQ3_XXS/IQ4_XS (Metal) |
|---|---|---|---|---|
| "Die Hauptstadt von Deutschland ist" | " Deutschland Deutschland Deutschland …" | ", dass, dass, dass …" | " Deutschland Deutschland Deutschland …" | " Deutschland Deutschland Deutschland …" |
| "Der Rhein ist ein Fluss, der" | | | " ist ein Fluss, der ist ein Fluss, der …" | ", der, der, der …" |
| "Frage: Was ist die Hauptstadt von Deutschland?\nAntwort:" | | | | " Berlin\n\nDie Hauptstadt von Deutschland ist Berlin. Berlin ist die größte Stadt Deutschlands und liegt im Osten des Landes." |
| "The capital of Germany is" | " Berlin." (see above) | " Berlin." | " Berlin.\n\nWe need to answer the question: \"What is the capital of Germany?\" …" | " Berlin. Berlin is the largest city in Germany and the capital." |

What is ruled out:

- **Tokenization:** llama.cpp and the reference tokenizer give the same
  five ids, `452 22090 493 1678 2459`.
- **A missing BOS:** neither side adds one. The reference has
  `add_bos_token: False` and no `bos_token`, and the GGUF has
  `tokenizer.ggml.add_bos_token = False`.
- **The KV cache and flash attention:** on the IQ3_XXS/IQ4_XS file,
  `-fa off`, `-fa on`, and `-ctk f32 -ctv f32` with flash attention off and on,
  all give the same two repeating continuations.
- **Patch 0009 and the split across backends:** Q8_0 with
  `-ngl 99 --cpu-moe --no-repack -t 10` repeats as well. Its first row
  matches BF16 rather than Q8_0 on the CPU, probably because the router runs
  on Metal there and flips experts at the near-tie router scores described
  in "Reference forward on the real weights".
- **llama.cpp itself:** the torch reference below continues
  " Deutschland Deutschland …" too, in float32 and in bfloat16.

German through the chat template works: the 528-token German answer below
is coherent, and so is the Q8_0 `--cpu-moe` answer in "Streaming the
routed experts with `--cpu-moe`". Kolibri is a post-trained reasoning model,
and on these weights the vLLM model code computes the same raw continuation.
Only a run of vLLM itself (PLAN Phase 6) can still contradict that.

## Reference forward on the real weights

`tools/ref/kolibri_ref.py` is a whole-model forward of Kolibri-1 in torch. It
ports `kolibri1.py` at the pinned commit and the vLLM v0.29.0 Qwen3Moe pieces
it inherits:

- embedding, then 50 decoder layers, then the final norm;
- per layer: input norm, attention, `post_attn_norm`, the residual add,
  `post_attention_layernorm`, the MoE block, `post_ffn_norm`, the residual add;
- attention: per-head Q/K RMSNorm, RoPE on the sliding layers only, window
  513, GQA 48/4;
- MoE: F32 router logits, `sigmoid_logit_add_routing`, SwiGLU experts plus
  the ungated shared expert;
- LM head in F32.

It reads the weights from the BF16 GGUF, whose tensors are bit-exact to the
safetensors, and touches only the experts a token is routed to. It reuses the
RoPE, attention and routing ports of `check_attn.py` and `check_moe.py`.
`--dtype bfloat16` emulates vLLM's precision: BF16 matmul inputs and outputs,
norm outputs and residual stream; F32 router logits and LM head.

`tools/ref/compare_real.py --tiny` first validates the reference against
libllama on the tiny fixture (F32, CPU, F32 KV cache):

```text
PASS tiny, all 8 experts: logits NMSE 4.75e-13, worst layer node NMSE 7.49e-13 (bound 1e-06)
PASS tiny, all 8 experts, router probe: worst layer router-logit NMSE 5.44e-13 where every earlier layer picks the same experts (bound 1e-06), Top-1 same 100.0%, Top-8 set same 100.0%
PASS tiny, top-2 routing: same experts for 100.0% of (token, layer) pairs, worst layer node NMSE 7.19e-13 where they agree (bound 1e-06)
PASS tiny, top-2 routing, router probe: worst layer router-logit NMSE 4.74e-13 where every earlier layer picks the same experts (bound 1e-06), Top-1 same 100.0%, Top-2 set same 100.0%
```

**Mutations of the reference**, each reverted:

| Mutation | Result |
|---|---|
| M15: no RoPE on the sliding layers | `FAIL tiny, all 8 experts: logits NMSE 5.86e-01` |
| M16: `post_attn_norm` and `post_attention_layernorm` swapped | `FAIL tiny, all 8 experts: logits NMSE 7.93e-02` |
| M17: reference router logits scaled by 1.01 | `FAIL tiny, top-2 routing, router probe: worst layer router-logit NMSE 1.06e-04 …` (12 of 16 lines fail: the four above, and every node line from layer 0's expert output on) |
| M18: no Q norm in the reference | `FAIL tiny, all 8 experts, layer 0 (sliding attention), 100 of 100 tokens on the same experts before it: NMSE Q after QK norm 4.91e+00, K after QK norm 1.35e-14, …` |
| M19: reference embedding rows shifted by one token ID | `FAIL tiny, all 8 experts, token embeddings (embd): NMSE 1.99e+00` |

`compare_real.py --gguf` then runs the real BF16 GGUF on the CPU, with
libllama's default KV cache and flash-attention settings, against the
reference in float32. It took 15 minutes in its first run, with a peak
footprint of 11 GB. With `--metal` it also runs chunk 1 on the GPU with the
routed experts on the CPU (152 s). The run with `--metal` took 46 minutes,
partly in Low Power Mode on battery, with a peak footprint of 11.3 GB.

**German prompt** "Die Hauptstadt von Deutschland ist" (5 tokens):

- layers 0 to 12: `l_out` NMSE at most 2.7e-6, the same experts for every
  token;
- layer 13: one token picks a different expert set, and from there the
  differences grow with each further flip, up to `l_out` NMSE 0.52 (layer 37);
- over all 5 positions: logits NMSE 1.88e-02, KLD 0.2446, same top token
  at 5/5;
- router: Top-1 same for 84.4% and the Top-6 set same for 61.2% of the 250
  (token, layer) pairs, almost all after layer 13;
- next token: both rank " Deutschland" first, and both top 5 hold
  " ein", " nicht" and " die";
- the reference's greedy continuation, float32 and bfloat16 alike:
  " Deutschland Deutschland Deutschland …";
- libllama's greedy token along that continuation: equal at 16/16.

**wikitext-2 test, chunk 1** (512 tokens, from the KLD base file):

| Comparison | PPL (test vs base) | Logits NMSE | KLD | Same top token |
|---|---|---|---|---|
| libllama BF16 (CPU) vs reference float32 | 15.1787 vs 15.1244 | 2.02e-03 | 0.0334 | 96.5% |
| reference bfloat16 vs reference float32 | 15.3272 vs 15.1244 | 2.56e-03 | 0.0327 | 93.7% |
| libllama BF16 (Metal, experts on CPU) vs reference float32 | 14.9382 vs 15.1244 | 1.35e-03 | 0.0178 | 96.9% |
| libllama BF16 (Metal) vs libllama BF16 (CPU) | 14.9382 vs 15.1787 | 1.98e-03 | 0.0487 | 93.3% |

- The per-layer NMSE rises smoothly, from 1e-7 in layer 0 to at most 0.08
  (layer 31), with the share of tokens on the same experts falling from 99.8%
  to 64%. No single node or layer jumps.
- Router margin on the reference: the 6th and 7th scores (logits + bias)
  differ by a median of 0.033, by less than 0.01 for 22.3% and by less than
  0.1 for 83.3% of the (token, layer) pairs, against a median score spread
  of 0.51 over the 384 experts.
- Activation maxima: |k| 38, |v| 0.9 and |`l_out`| 10,113. Nothing comes
  near the F16 limit of 65,504, and nothing is non-finite.

### Per-node comparison

`compare_real.py` also compares the nodes of PLAN Phase 6 one by one:
- the token embeddings (libllama's `embd`, the `get_rows` of `token_embd`);
- in the first sliding layer (0) and the first full layer (4):
  - Q and K after the QK norm, before RoPE (`Qcur_normed`, `Kcur_normed`);
  - the attention output before and after its sandwich norm (`attn_out`,
    `attn_post_norm`);
  - the routed and shared expert output (`ffn_moe_out`, `ffn_shexp`);
- the worst layer of the routed and shared expert output.

Each node counts only the tokens whose experts agree in every layer it
depends on. For Q, K, the attention output and the shared expert, that is
the earlier layers: a layer computes them before its router runs. For the
routed expert output it is this layer too. That way the NMSE measures each
node's own arithmetic, not the effect of an expert flip.

On the tiny fixture every node line is within NMSE 1e-6 of the float64
reference, for example:

```text
PASS tiny, top-2 routing, token embeddings (embd): NMSE 0.00e+00 (bound 1e-06)
PASS tiny, top-2 routing, layer 4 (full attention), 100 of 100 tokens on the same experts before it: NMSE Q after QK norm 3.68e-13, K after QK norm 3.17e-13, attention output attn_out 1.06e-13, attn_post_norm 1.01e-13 (bound 1e-06)
PASS tiny, top-2 routing, worst layer with tokens on the same experts: NMSE routed expert output ffn_moe_out 6.57e-13 (layer 5), shared expert output ffn_shexp 7.19e-13 (layer 5) (bound 1e-06)
```

M18 and M19 in the mutation table above make them fail.

The real BF16 GGUF on the CPU, against the reference in float32, wikitext-2
chunk 1:

| Node | libllama vs reference float32 | reference bfloat16 vs float32 |
|---|---|---|
| token embeddings | 0 | 0 |
| layer 0 (sliding): Q / K after QK norm | 2.01e-06 / 1.76e-06 | 1.24e-05 / 1.18e-05 |
| layer 0: `attn_out` / `attn_post_norm` | 1.71e-07 / 1.14e-07 | 3.06e-06 / 8.22e-06 |
| layer 0: routed / shared expert output | 6.15e-06 / 1.06e-06 | 3.41e-05 / 7.57e-06 |
| layer 4 (full): Q / K after QK norm | 1.93e-06 / 2.78e-06 | 2.75e-05 / 3.55e-05 |
| layer 4: `attn_out` / `attn_post_norm` | 2.04e-06 / 3.87e-06 | 2.55e-05 / 5.68e-05 |
| layer 4: routed / shared expert output | 3.28e-06 / 4.73e-07 | 2.72e-05 / 5.44e-06 |
| worst layer: routed expert output | 2.71e-03 (layer 33) | 2.77e-02 (layer 45) |
| worst layer: shared expert output | 1.79e-03 (layer 49) | 8.08e-03 (layer 49) |
| tokens on the same experts before / through layer 4 | 505 / 503 of 512 | 462 / 449 of 512 |

- **Embeddings.** Both sides widen the same BF16 rows to F32, exactly.
- **Layers 0 and 4.** Every node is within 6.2e-6, 5 to 72 times below the
  reference's own BF16 rounding.
  - Q and K sit near 2e-6. That is consistent with libllama's CPU matmul,
    which converts the F32 activations to BF16 for BF16 weights
    (`vec_dot_type` BF16); the float32 reference does not round them.
- **Deep layers.** The error grows through the layers but stays below the
  BF16 baseline: on the tokens still on the same experts, the worst routed
  expert output is 2.7e-3, against 2.8e-2 for the reference's BF16 rounding.
- **German prompt** (5 tokens, every token on the same experts through layer 4):
  - layer 0: Q / K after QK norm 1.75e-06 / 1.39e-06, `attn_out` 3.21e-07;
  - layer 4: routed expert output 4.49e-06, shared 3.92e-07;
  - worst layer: routed 4.16e-03 (layer 20), shared 1.48e-03 (layer 21).

As everywhere in this file, the reference is the torch port of the vLLM model
code, not vLLM itself.

### Router agreement

`tools/ref/router_probe.py` compares two routers per (token, layer):
- the F32 router logits and their sigmoid;
- the Top-1 expert (largest logits + bias);
- the Top-6 set and its overlap (0 to 6 experts in common);
- the second router's margin, its 6th minus 7th score of logits + bias.

`--probe-out DIR` saves every probe as `router-<name>.npz`, with all per
(token, layer) arrays and both sides' logits.

Chunk 1, 25,600 (token, layer) pairs:

| Comparison | Top-1 same | Top-6 set same | Mean overlap | Layer 0: logit NMSE, set same |
|---|---|---|---|---|
| libllama CPU vs reference float32 | 97.09% | 84.05% | 5.817 | 2.05e-09, 99.8% |
| reference bfloat16 vs reference float32 | 95.14% | 75.88% | 5.711 | 2.07e-07, 97.9% |
| libllama Metal vs reference float32 | 97.66% | 87.17% | 5.861 | 1.25e-07, 98.8% |
| libllama Metal vs libllama CPU | 97.02% | 83.88% | 5.815 | 1.36e-07, 98.6% |

Share of pairs whose Top-6 set changes, by the second router's margin:

| Comparison | < 0.001 | 0.001 to 0.01 | 0.01 to 0.1 | ≥ 0.1 |
|---|---|---|---|---|
| (pairs, reference float32) | 870 | 4,838 | 15,610 | 4,282 |
| libllama CPU vs reference float32 | 29.8% | 26.5% | 15.5% | 3.1% |
| reference bfloat16 vs reference float32 | 36.4% | 37.7% | 23.4% | 8.9% |
| libllama Metal vs reference float32 | 30.7% | 23.2% | 11.6% | 1.9% |

- **CPU against the float32 reference.** libllama's experts match the float32 reference more often than the reference's own BF16 rounding does: 84.1% against 75.9% same sets.
- **Where the flips happen.** They concentrate at near-ties.
  - In layer 0, 99.8% of the sets agree and the router logits differ by NMSE 2e-9.
  - 97% of the 4,084 flipped sets have a margin below 0.1.
  - The 131 flips at margins of 0.1 and above start in layer 24, and each follows an earlier flip on the same token, which moved its hidden state.
- **Metal against the CPU.** Metal's router logits differ from the CPU's by NMSE 1.36e-7 in layer 0.
  - That is a relative RMS of about 3.7e-4, consistent with the half-precision tiles noted in PLAN Phase 6. It includes the difference of layer 0's attention on Metal.
  - Layer 0's sets differ for 1.4% of the tokens.
  - That is fewer than BF16 rounding of the reference changes (2.1%).
  - For scale: over the chunk, the same BF16 file on Metal against the CPU gives KLD 0.0487. Q8_0 against BF16 on the CPU gives 0.050, though over 20 chunks rather than one (see "Quantization loss against this port's BF16").
- **Metal against the reference.** Over the whole chunk, Metal sits closer to the float32 reference than the CPU does: KLD 0.0178 against 0.0334, and 87.2% against 84.1% same sets. This run does not show why.
- **Phase 8 decision.** Whether the router needs an F32 path on Metal stays open. On these numbers, Metal's router error is smaller than the BF16 tolerance.

**Conclusion.**

- libllama BF16 sits as close to the float32 reference as BF16 rounding of the
  reference itself does (KLD 0.033 each).
- The hidden states drift apart through expert flips at near-tie router scores,
  not through a faulty step. 97% of libllama's flipped sets have a margin
  below 0.1, every larger-margin flip follows an earlier one on the same
  token, and libllama on the CPU and on Metal flips fewer sets than BF16
  rounding of the reference does.
- The German repetition is therefore what the vLLM model code computes on
  these weights, not a llama.cpp defect.
- The BF16 sensitivity makes router amplification a plausible cause of the
  high Q8_0 KLD. Q8_0 itself was not run here, so that stays open.
- What this cannot rule out is a misreading of the vLLM code shared by the port
  and this reference. Only running vLLM itself covers that (PLAN Phases 0 and 6).

## End-to-end corpus

`compare_real.py --gguf` recomputes the reference on every run, which takes
about 15 minutes, and covers only two inputs, both shorter than the sliding
window. `tools/ref/e2e.py` stores the reference once for a small fixed corpus,
so that a new llama.cpp build can be checked against it in minutes.

The corpus, `testdata/e2e/corpus.json`, holds six cases with their token IDs
from the pinned tokenizer:

| Case | Input | Tokens |
|---|---|---|
| `de-raw` | "Die Hauptstadt von Deutschland ist" + 16 greedy tokens | 5 + 16 |
| `en-raw` | "The capital of Germany is" + 16 greedy tokens | 5 + 16 |
| `code` | a Python function header with its docstring + 16 greedy tokens | 13 + 16 |
| `de-chat` | one German user turn through the released chat template, thinking off + 16 greedy tokens | 52 + 16 |
| `wiki-c1` | wikitext-2 test chunk 1, from the KLD base file | 512 |
| `de-long` | the start of the German part of the calibration text | 1024 |

`de-long` is the only case longer than the 513-token window, so it is the only
one in which the sliding layers actually drop keys.

`--write` runs the reference in float32 and in bfloat16 and writes one `.npz`
per case to `~/models/eval/e2e`:
- the logits of both runs;
- the float32 run's router logits and selected experts in every layer;
- the router bias.

For the prompts, the sequence is the prompt plus the float32 greedy
continuation. The artifacts take 1.85 GB, of which `de-long` is 1.13 GB.
`testdata/e2e/manifest.json` records:
- the sha256 of every array;
- the greedy continuations;
- the reference's bfloat16-vs-float32 metrics;
- the libllama run that `--write` then does, with a sha256 of its logits.

The write took 2 h 23 min, with a peak footprint of 21.1 GB:

| Case | `de-raw` | `en-raw` | `code` | `de-chat` | `wiki-c1` | `de-long` |
|---|---|---|---|---|---|---|
| Reference, both dtypes | 123 s | 1038 s | 1130 s | 4613 s | 357 s | 494 s |

The prompts are slow because each greedy token recomputes the whole sequence,
since the reference has no KV cache. The run is disk-bound: with 52 tokens or
more, every forward reads about half of the experts from the 156 GB file. A
second write of `en-raw` gave the same sha256 for every array, so the
reference is deterministic.

The default mode checks:
- the corpus against its sources, wherever those are present;
- every artifact against the manifest.

Any difference is a `FAIL`, with exit 1. It then runs libllama on the CPU on
each case and prints `INFO` lines, against the reference in float32 and in
bfloat16:
- logits NMSE, KLD and same top token, over the second half for the chunks, with
  PPL as in `llama-perplexity`;
- router Top-1 and Top-6 set agreement;
- for the prompts, libllama's greedy token along the reference continuation.

Last, per case, it states whether libllama's logits are bit-identical to the
recorded run, or how its metrics moved. The metrics are not gated, since the
real model has no tolerance yet. The bit-identity itself is gated since
"Regression suite" below.

Results, libllama BF16 on the CPU. The default mode took 13 minutes, with a
peak footprint of 10.8 GB, and every case came out bit-identical to the run
that `--write` recorded:

| Case | Reference bfloat16 vs float32: KLD | libllama vs float32: NMSE, KLD, same top | libllama vs bfloat16: KLD | Router Top-1 / Top-6 set same | Greedy along the reference |
|---|---|---|---|---|---|
| `de-raw` | 0.1508 | 9.56e-03, 0.1125, 100.0% | 0.0910 | 85.43% / 56.48% | 16/16 |
| `en-raw` | 0.0079 | 1.59e-04, 0.0015, 95.2% | 0.0091 | 97.71% / 90.29% | 15/16 |
| `code` | 0.2868 | 9.61e-04, 0.0026, 100.0% | 0.2675 | 96.62% / 85.24% | 16/16 |
| `de-chat` | 0.0023 | 2.51e-04, 0.0021, 97.1% | 0.0012 | 99.38% / 94.71% | 16/16 |
| `wiki-c1` | 0.0327 (PPL 15.3272 vs 15.1244) | 2.02e-03, 0.0334, 96.5% (PPL 15.1787) | 0.0334 | 97.09% / 84.05% | |
| `de-long` | 0.1025 (PPL 26.1070 vs 26.5145) | 1.65e-03, 0.0463, 95.1% (PPL 26.3648) | 0.0725 | 97.32% / 85.55% | |

What the results show:
- **`wiki-c1` matches the earlier comparison.** It gives exactly the numbers of
  `compare_real.py --gguf` on the same chunk (see "Reference forward on the real
  weights" and "Router agreement").
- **`de-long` is past the window.** libllama sits closer to the float32
  reference (KLD 0.046) than the reference's own bfloat16 run does (0.102).
  Nothing in it points at the sliding-window mask.
- **`de-raw` repeats as before.** It is the known repetition case. Its router
  diverges about as much as in "Reference forward on the real weights": the
  Top-6 set is the same for 56.5% of the pairs over 21 tokens here, against
  61.2% over the 5 prompt tokens there.
- **`code`.** libllama sits much closer to float32 (KLD 0.0026) than the
  bfloat16 reference does (0.287).

Negative controls, each on a scratch copy:

| Change | Result |
|---|---|
| one array of `de-raw.npz` re-saved with one router logit changed by 1e-3 | `FAIL de-raw: router_logits in de-raw.npz differ from the manifest's sha256`, exit 1 |
| one byte of `de-raw.npz` flipped | `FAIL de-raw: de-raw.npz is unreadable (Bad CRC-32 for file 'ref_float32.npy'); recreate it with --write`, exit 1 |
| one token ID of `en-raw` changed in the corpus | `FAIL corpus/en-raw: token IDs differ from its source at position 2 (5 vs 5 tokens)` and `FAIL en-raw: the artifact's tokens differ from the corpus`, exit 1 |
| the IQ3_XXS/IQ4_XS chat candidate, not Q8_0, which needs `--no-repack` on this machine while the tool keeps libllama's defaults, (`Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf`) in place of the BF16 GGUF, on `en-raw` and `wiki-c1` | `INFO wiki-c1: CHANGED since the recorded libllama run; recorded with Kolibri-1-BF16.gguf, this run Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf: vs_float32 kld 0.090176 (recorded 0.033408), …`, router Top-6 set same 60.85% (recorded 84.05%); `en-raw` also CHANGED, KLD 0.0097 (recorded 0.0015), greedy 13/16 (recorded 15/16) |

The reference is the torch port, not vLLM. Logits from a vLLM BF16 run can
replace these artifacts later, in the same `.npz` layout.

Rerun after an upstream llama.cpp change:

```sh
.venv/bin/python tools/ref/e2e.py --llama-cpp third_party/llama.cpp --gguf ~/models/Kolibri-1-BF16.gguf
```

The results above are those of the first version of the tool. The current
one gates the comparison with the recorded run; see "Regression suite".

## Regression suite

`tools/ref/regress.py` reruns a compact subset of the reference comparisons
with one command, after a llama.cpp or upstream change. It fails if any of its
three parts fails:

| Part | What | Gate |
|---|---|---|
| tokenizer | `compare.py --only golden --only corpus` on a fresh vocab-only GGUF | token IDs and detokenized bytes identical to the pinned reference tokenizer: 85 golden cases, 4,212 + 6,461 corpus cases |
| tiny reference | `compare_real.py --tiny` | token embeddings, Q and K after the QK norm, attention output, router logits and experts, routed and shared expert output, `l_out` and logits of the F32 tiny model within NMSE 1e-6 of the torch reference |
| real weights | `e2e.py` on the BF16 GGUF, all six corpus cases | every captured libllama node and the logits bit-identical to the recorded libllama run |

The first two parts compare with the reference itself, the third with a
reviewed earlier libllama run. The real model has no tolerance yet, so a
change there fails until someone has looked at it.

`e2e.py` captures 451 nodes per case:
- the token embeddings;
- per layer, Q and K after the QK norm, the attention output before and after
  its sandwich norm, the router logits and selected experts, the routed and
  shared expert output, and `l_out`.

The manifest records the sha256 of each, next to the logits'. With the
recorded GGUF and thread count, a case passes only if all of them are
identical. Otherwise it fails and names the first changed node in graph order,
with the metrics' movement against the recorded run. A GGUF or thread count
other than the recorded one is reported as `INFO` and not gated. That is the
way to look at a quantized GGUF. `regress.py` runs `e2e.py` with `--strict`,
where such a case fails, and so does a case without a recorded run.

`--record` accepts a reviewed change. Capturing every node leaves libllama's
logits as they were: the six logits hashes equal those recorded without the
node capture.

For two cases, `--write` also stores a compact set of the float32
reference's nodes, so the per-node comparison of "Per-node comparison" reruns
without the torch pass:

| Case | Stored nodes | Artifact |
|---|---|---|
| `wiki-c1` | token embeddings; Q, K, attention output and expert outputs in layers 0 and 4; expert outputs and `l_out` in all 50 layers | 1.40 GB (564 MB before) |
| `de-long` | Q, K, attention output and expert outputs in layers 0 and 4, past the 513-token window; `l_out` in all 50 layers | 1.79 GB (1.13 GB before) |

Writing both took 46 minutes, with a peak footprint of 17.3 GB. The other
arrays came out with the same sha256 as before.

libllama against these nodes (`INFO`):

| Node | `wiki-c1` | `de-long` |
|---|---|---|
| token embeddings | 0.00e+00 | |
| layer 0 (sliding): Q, K after QK norm | 2.01e-06, 1.76e-06 | 2.03e-06, 1.75e-06 |
| layer 0: attn_out, attn_post_norm | 1.71e-07, 1.14e-07 | 1.59e-07, 1.12e-07 |
| layer 0: ffn_moe_out, ffn_shexp | 6.15e-06, 1.06e-06 | 6.03e-06, 1.11e-06 |
| layer 4 (full): Q, K after QK norm | 1.93e-06, 2.78e-06 | 1.86e-06, 2.66e-06 |
| layer 4: attn_out, attn_post_norm | 2.04e-06, 3.87e-06 | 1.97e-06, 2.34e-06 |
| layer 4: ffn_moe_out, ffn_shexp | 3.28e-06, 4.73e-07 | 2.33e-06, 4.64e-07 |
| worst layer: ffn_moe_out, ffn_shexp | 2.71e-03 (33), 1.79e-03 (49) | |
| `l_out`: layer 0, 4, worst, 49 | 7.63e-08, 2.74e-06, 7.60e-02 (31), 1.17e-02 | 8.12e-08, 2.09e-06, 2.81e-02 (43), 1.18e-02 |

These numbers are the NMSE over the tokens whose experts agree in the earlier
layers (all tokens for `l_out`). `wiki-c1` repeats the numbers of
"Per-node comparison" exactly. In `de-long`, the sliding layer 0 drops keys
past the window, and it stays as close to the reference as on the shorter
chunk.

The suite took 27 minutes:
- tokenizer: 8 s;
- tiny reference: 10 s;
- real weights: 26.5 minutes.

The real-weights time depends on how much of the 156 GB GGUF is in the page
cache. libllama took between 205 s and 792 s on `wiki-c1` across runs.

Negative controls:

| Change | Result |
|---|---|
| a copy of the manifest with the recorded sha256 of `wiki-c1`'s `l_out-20` changed, through `regress.py --cases wiki-c1 --manifest COPY` | `FAIL wiki-c1: changed since the recorded libllama run, first changed node l_out-20; …`, `FAIL regression suite: real weights`, exit 1 |
| `e2e.py --cases wiki-c1 --override kolibri.attention.layer_norm_rms_epsilon=1e-4` (1e-6 in the GGUF) | `FAIL wiki-c1: changed since the recorded libllama run, first changed node Qcur_normed-0; vs_float32 kld 1.934355 (recorded 0.033408), …`, exit 1. The token embeddings stay identical; Q after the QK norm is the first captured node after layer 0's attention norm, the first RMS norm |

Run after an upstream llama.cpp change:

```sh
.venv/bin/python tools/ref/regress.py --llama-cpp third_party/llama.cpp [--gguf ~/models/Kolibri-1-BF16.gguf]
```

After reviewing a failed real-weights case, accept the change with
`e2e.py … --record`.

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
7. Run `tools/ref/compare_real.py --tiny`, then with `--gguf` and the KLD base
   file from "Quantization loss against this port's BF16" (15 minutes), plus
   `--metal --probe-out DIR` for the Metal pass and the router probes.
8. Run `tools/ref/e2e.py --write` once to store the end-to-end corpus's
   reference (2 h 23 min), as in "End-to-end corpus". After each llama.cpp
   change, run `tools/ref/regress.py`, about 27 minutes, as in "Regression
   suite".

Disk peaks at about 312 GB during the conversion. After that it is the BF16
GGUF plus the quantized files.
