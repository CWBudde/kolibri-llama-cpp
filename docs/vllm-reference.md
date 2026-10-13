# vLLM reference run

Phases 0 and 6 compare llama.cpp with vLLM itself, running Aleph Alpha's
`aleph-alpha-inference` plugin. Until now every comparison went against
`tools/ref/kolibri_ref.py`, a torch port of that code. A misreading shared by
the port and libllama would pass every check. Only vLLM itself can rule that
out.

Two tools cover the run:
- **`tools/ref/vllm_capture.py`** runs vLLM with the plugin and records, per
  case:
  - the tokens;
  - the logits for every position;
  - the router logits and the selected experts per layer;
  - the per-layer activations under libllama's node names.
- **`tools/ref/compare_vllm.py`** compares the torch port and libllama with that
  capture.

On the tiny synthetic checkpoint the whole chain runs on a Mac, with vLLM's
CPU backend. The real checkpoint needs a GPU host (see "On the real checkpoint" below).

## Environment

vLLM pins its own torch, so it gets an environment separate from `.venv`. The
plugin 1.0.0 requires `vllm>=0.29.0,<0.30.0`; `tools/ref/requirements-vllm.txt`
pins vLLM 0.29.0 and the plugin 1.0.0.

```sh
# CUDA host
python3.12 -m venv vllm-venv && vllm-venv/bin/pip install -r tools/ref/requirements-vllm.txt

# Apple Silicon: the CPU wheel from the vLLM v0.29.0 release
gh release download v0.29.0 -R vllm-project/vllm -p 'vllm-0.29.0+cpu-cp312-cp312-macosx_11_0_arm64.whl'
uv venv -p 3.12 vllm-venv
VIRTUAL_ENV=$PWD/vllm-venv uv pip install vllm-0.29.0+cpu-cp312-cp312-macosx_11_0_arm64.whl \
    aleph-alpha-inference==1.0.0
```

On the Mac this installed vllm 0.29.0+cpu, torch 2.13.0 and transformers
5.19.0. `capture.json` records the versions of every run.

## What is captured

`vllm_capture.py` imports nothing from this repo. A GPU host needs only the
script, the checkpoint and, for the corpus, `testdata/e2e/corpus.json`.

**Runs per case:**
1. **Greedy run (prompt cases only):** vLLM generates the continuation at
   temperature 0 and returns the top-20 logprobs of every generated token.
2. **Prefill over the whole sequence:** prompt plus continuation, or the chunk.
   The prefix cache is off, and the sequence fits in one batch.

**Forward hooks during the prefill** record these nodes:

| Node (libllama name) | vLLM module |
|---|---|
| `embd` | `model.embed_tokens` |
| `Qcur_normed-il`, `Kcur_normed-il` | `self_attn.q_norm`, `k_norm`: after the QK norm, before RoPE |
| `attn_out-il` | `self_attn.o_proj` |
| `attn_post_norm-il` | `post_attn_norm` |
| `ffn_moe_logits-il` | `mlp.gate`: F32 router logits, before the bias |
| `ffn_moe_topk-il` | Top-k of those logits plus `gate.e_score_correction_bias`, as `sigmoid_logit_add_routing` selects |
| `ffn_shexp-il` | `mlp.shared_experts` |
| `ffn_moe_out-il` | the routed output as the MoE runner adds it to the shared output (`shared_output + fused_output`), taken from its `apply_routed_output_transform`, an identity for Kolibri |
| `l_out-il` | the decoder layer's hidden state plus residual: vLLM adds them in the next layer's norm |
| logits | `model.compute_logits` on the output of `model.norm`: the plugin's `LogitsProcessor`, in the config's `head_dtype` |

**Why `ffn_moe_out` is the summand:** the block's output is that sum,
already rounded in the model dtype. In bfloat16, subtracting the shared output
from it would not recover the routed branch. On the tiny corpus capture in
bfloat16, that difference gave an NMSE of about 1e-5 per layer against the
summand, and its values were not even BF16-representable. In float32 the two
agree (NMSE 2.5e-15).

Hooks copy their tensor, because the rotary embedding rotates q and k in
place after `q_norm` and `k_norm` return. Without the copy, the sliding layers'
Q and K were recorded after RoPE: NMSE 9.9e-01 against the port. The full
layers, which have no RoPE, still matched at 1e-12.

**A check of the hooked logits:** each greedy token's top-20 logprobs from
the decode steps are compared with the log-softmax of the prefill logits at the
same position. `capture.json` records the result as `logprobs_max_abs_diff`.

**Output per case:** `<case>.npz` in the layout of `e2e.py`'s artifacts:
- `tokens`
- `logits` (T, n_vocab)
- `router_logits` (L, T, E)
- `router_topk`
- `router_bias`
- `is_swa`
- `node.<name>`

`capture.json` adds:
- the versions, the platform and the GPUs;
- the `config.json` and corpus sha256;
- dtype, experts per token, pipeline-parallel size and attention layers;
- per case the greedy tokens and the sha256 of every array (as `e2e.py`).

`--attn-layers 0,4` keeps the attention nodes of those layers only, as
`e2e.py`'s compact set does. The set must include the first sliding and the
first full layer, which `compare_vllm.py`'s node lines read, and the capture
checks that before it loads the model. From the shapes, de-long's capture is still about
2.3 GB on the real checkpoint:
- 1.57 GB for `l_out`, `ffn_moe_out` and `ffn_shexp` in all 50 layers;
- 0.52 GB of logits;
- 0.08 GB of router logits;
- 0.1 GB of attention nodes.

## Tiny checkpoint

`cmd/kolibri-tiny -seed 1` (6 layers, 8 experts, top-2), 100 random tokens with
seed 1 (as `compare_real.py --tiny`) and 4 greedy tokens, captured in float32:

```sh
go run ./cmd/kolibri-tiny -out /tmp/kt/kolibri-tiny -tokenizer-dir <tokenizer dir> -seed 1
vllm-venv/bin/python tools/ref/vllm_capture.py --model /tmp/kt/kolibri-tiny --random 100 --seed 1 \
    --greedy 4 --dtype float32 --gpu-memory-utilization 0.1 --out /tmp/kt/cap-top2
.venv/bin/python tools/ref/compare_vllm.py --llama-cpp third_party/llama.cpp --capture /tmp/kt/cap-top2 \
    --model /tmp/kt/kolibri-tiny --gate --libllama
```

**`--gpu-memory-utilization`:** on the CPU backend it is the share of RAM vLLM
reserves. The default 0.9 fails on a 48 GB Mac with other processes running.

**Capture:** "INFO tiny: 104 tokens (4 greedy), 49 arrays, decode logprobs vs
prefill logits max abs diff 3.20e-06".

**Comparison:** the torch port runs in float64. libllama runs on the F32 GGUF,
with an F32 KV cache and without flash attention. The bounds are those of
`compare_real.py --tiny`.

```text
PASS tiny, torch port vs vLLM: logits NMSE 1.52e-12, KLD 0.000000, same top token 100.0%; same experts for 100.0% of (token, layer) pairs, worst layer node NMSE 2.84e-12 where they agree
PASS tiny, torch port vs vLLM, router probe: worst layer router-logit NMSE 1.60e-12 where every earlier layer picks the same experts, Top-1 same 100.0%, Top-2 set same 100.0%
PASS tiny, libllama vs vLLM: logits NMSE 1.70e-12, KLD 0.000000, same top token 100.0%; same experts for 100.0% of (token, layer) pairs, worst layer node NMSE 3.23e-12 where they agree
PASS tiny, libllama vs vLLM, router probe: worst layer router-logit NMSE 1.72e-12 where every earlier layer picks the same experts, Top-1 same 100.0%, Top-2 set same 100.0%
PASS vLLM capture, 2 of 8 experts: torch port and libllama within the tiny bounds
```

With all eight experts (`--experts-used 8`, through `hf_overrides`), routing
has no top-k discontinuity, so every node counts every token:

```text
PASS tiny, torch port vs vLLM: logits NMSE 1.53e-12, KLD 0.000000, same top token 100.0%, worst layer node NMSE 2.86e-12
PASS tiny, libllama vs vLLM: logits NMSE 1.68e-12, KLD 0.000000, same top token 100.0%, worst layer node NMSE 3.28e-12
PASS vLLM capture, 8 of 8 experts: torch port and libllama within the tiny bounds
```

In both runs, the argmax of the port and of libllama follows vLLM's 4 greedy
tokens. Both runs also pass the node lines: embeddings, layer 0 (sliding) and
layer 4 (full), and the worst expert layer. Each run exits 0.

**What this shows:** on the tiny shape, vLLM with the plugin computes what the
port and libllama compute, node by node, at float32 rounding level.

**What it does not show:** anything about the real weights, BF16 rounding, or
the 50-layer, 384-expert shape.

**The corpus path:** the e2e corpus, captured from the tiny checkpoint in
bfloat16 with `--attn-layers 0,4` (no `--gate`, INFO only), runs through all
six cases. Example line: "INFO de-chat, libllama vs vLLM: logits NMSE 5.62e-04,
KLD 0.000029, same top token 95.6%; same experts for 99.8% of (token, layer)
pairs". These numbers measure bf16 rounding on random weights, not the model.

**Negative control M27:** the captured router logits of layer 3 scaled by
1.01 (as M17) fail, exit 1:

```text
FAIL tiny, torch port vs vLLM, router probe: worst layer router-logit NMSE 9.80e-05 where every earlier layer picks the same experts, Top-1 same 99.8%, Top-2 set same 100.0%
FAIL vLLM capture, 2 of 8 experts: 1 check(s) failed
```

## Slices of the real weights

The full model needs a GPU host, but a slice of it does not. `tools/ref/slice.py`
writes chosen layers of the real checkpoint as a smaller HF checkpoint, taken
from the BF16 GGUF:
- the inventory's name mapping, which `check_real.py` verified bit-exact;
- the experts unstacked, the F32 norms and router written back as the BF16
  they hold;
- the real config, with `num_hidden_layers` and `layer_types` cut to the
  chosen layers.

A slice is a smaller, well-defined Kolibri model with the real attention shape
(48/4 heads, window 513), the real 384-expert top-6 routing and the real
weights. vLLM, the port and libllama have to agree on it. It does not compute
what the full model computes at those depths. Two layers take 7.5 GB in BF16
and fit vLLM's CPU backend in float32.

```sh
.venv/bin/python tools/ref/slice.py --gguf ~/models/Kolibri-1-BF16.gguf \
    --model-dir ~/models/Kolibri-1-BF16 --layers 0,4 --out /tmp/slice-0-4      # 22 s
vllm-venv/bin/python tools/ref/vllm_capture.py --model /tmp/slice-0-4 \
    --corpus testdata/e2e/corpus.json --dtype float32 --gpu-memory-utilization 0.45 --out /tmp/cap-0-4
.venv/bin/python tools/ref/compare_vllm.py --llama-cpp third_party/llama.cpp --capture /tmp/cap-0-4 \
    --model /tmp/slice-0-4 --gate --max-nmse 1e-8 --libllama  # converts the slice to an F32 GGUF (15 GB)
```

**Slices run:**
- layers 0 and 4, the first sliding and the first full layer;
- layers 48 and 49, the last ones, before the final norm and the head.

Each slice took under half a minute to capture and as long to compare, on all
six cases of the e2e corpus. `de-long` runs 1024 tokens, past the window.

**Result:** every node, the router and the logits of the port (float64) and of
libllama agree with vLLM at 1e-12 to 2e-10 NMSE. Every case picks the same
experts in every (token, layer) pair. Each slice gives 105 PASS lines and exit
0, for example:

```text
PASS de-raw, libllama vs vLLM: logits NMSE 1.20e-12, KLD 0.000000, same top token 100.0%; same experts for 100.0% of (token, layer) pairs, worst layer node NMSE 1.06e-11 where they agree
PASS de-long, libllama vs vLLM, layer 0 (sliding attention), 1024 of 1024 tokens on the same experts before it: NMSE Q after QK norm 6.92e-13, K after QK norm 6.83e-13, attention output attn_out 1.67e-12, attn_post_norm 1.59e-12
PASS vLLM capture, 6 of 384 experts: torch port and libllama within NMSE 1e-08
PASS de-long, libllama vs vLLM: PPL 212415.5290 vs 212415.3448, logits NMSE 1.92e-10, KLD 0.000000, same top token 100.0%; same experts for 100.0% of (token, layer) pairs, worst layer node NMSE 1.50e-10 where they agree
PASS vLLM capture, 6 of 384 experts: torch port and libllama within NMSE 1e-08
```

The first three lines are slice [0,4], the last two slice [48,49]. A slice's
PPL is meaningless as a language model number; only the agreement counts.

**Why `--max-nmse 1e-8`:** the default bound, 1e-6 from `compare_real.py
--tiny`, is too loose for the real weights. Control M28 sets
`sliding_window` 512 in vLLM's copy of the config only, a window one key
short:
- that moves `de-long`'s layer-0 attention output from 1.5e-12 to 1.7e-07, and
  its logits to 7.6e-07, still inside 1e-6;
- at 1e-8 it fails, on `de-long` only, the one case past the window, with exit
  1: "FAIL de-long, torch port vs vLLM, layer 0 (sliding attention), … attention
  output attn_out 1.67e-07 …", "FAIL vLLM capture, 6 of 384 experts: 6
  check(s) failed".

Without the mutation, the slices stay at least 50 times below 1e-8.

With top-k routing the logits are gated over the tokens whose experts agree in
every layer, as the nodes are. An expert flip changes them legitimately. Control
M29 scales `de-raw`'s captured logits by 1.01 and leaves every node alone, a
fault confined to the head. It fails, with exit 1: "FAIL de-raw, torch port vs
vLLM: logits NMSE 9.80e-05, … worst layer node NMSE 4.66e-12 and logits NMSE
9.80e-05 (21 of 21 tokens) where they agree".

## Full model as chained slices

A slice fed the previous slice's output computes what the full model computes
in its layers. vLLM's decoder layer hands on the hidden state and the residual
separately, but the next layer uses only their sum, in its fused add and
RMSNorm. The capture records that sum as `l_out`. `vllm_capture.py --embeds`
passes it to the next slice as vLLM's `prompt_embeds`. There it becomes the
first layer's residual and the input of its norm, just as in the full model.
In float32 nothing is rounded at the handover. `tools/ref/vllm_chain.py run`
does this for the whole model:
1. Slice layers 0–1, 2–3, … 48–49 from the BF16 GGUF.
2. Capture each slice in vLLM's environment, fed the previous one's `l_out`,
   then delete the slice.
3. Stitch the 25 captures into one capture of all 50 layers, renumbered to the
   global layers.

`compare_vllm.py` reads the result as a capture of the full checkpoint.

```sh
.venv/bin/python tools/ref/vllm_chain.py run --gguf ~/models/Kolibri-1-BF16.gguf \
    --model-dir ~/models/Kolibri-1-BF16 --vllm-python vllm-venv/bin/python \
    --work /tmp/chain --out /tmp/chain/capture      # 13 min, 3.5 GB
.venv/bin/python tools/ref/compare_vllm.py --llama-cpp third_party/llama.cpp --capture /tmp/chain/capture \
    --gguf ~/models/Kolibri-1-BF16.gguf --port-artifacts ~/models/eval/e2e --libllama   # 13 min
```

**Inputs:**
- **The same tokens in every slice:** `vllm_capture.py --sequence` runs `e2e.py`'s
  sequences. That is each prompt plus the torch port's float32 greedy
  continuation, or the chunk, with no greedy run of its own.
- **vLLM's greedy continuation:** `vllm_chain.py` reads it off vLLM's argmax
  along that sequence. Where the argmax is the next token at every position,
  vLLM's greedy decoding takes exactly those tokens, and the stitched capture
  records them as vLLM's greedy tokens. This assumes decode steps give the same
  argmax as the prefill; on the tiny checkpoint, decode logprobs and prefill
  logits differ by 3.2e-06.
- **Attention nodes:** kept for layers 0 and 4, as in `e2e.py`'s compact set.
- **KV cache size:** `--kv-cache-gib` (1 GiB) fixes it. Otherwise the CPU
  backend subtracts the process's resident memory from its share of RAM, and
  that fails once a float32 slice is loaded.

**The torch port's side:** `--port-artifacts` takes it from `e2e.py`'s stored
float32 artifacts. These are the logits and the router of every case, plus the
compact node set of wiki-c1 and de-long. This replaces a float64 forward over
the full checkpoint.

**Chaining check:** one slice of layers 0–2 against the chain (0–1, then 2),
on all six cases. It is bit-identical:

```text
PASS de-long: 20 arrays in both, 20 bit-identical, worst NMSE 0.00e+00
PASS diff …/full3 vs …/chain3: every common array within NMSE 0
```

**Controls:**
- **M30:** the handed-over `l_out-1` scaled by 1+1e-6. Stitched with its own
  part 0–1, the diff fails on every case with exit 1: "FAIL de-raw: 20 arrays in
  both, 14 bit-identical, worst NMSE 1.79e-12 (node.ffn_moe_out-2); 6 over the
  bound: logits NMSE 3.81e-13; …". Stitched with the real part 0–1, the stitch
  fails on the sha256: "FAIL stitch de-raw: m30-part-2-2 was fed {'node':
  'l_out-1', 'sha256': 'c4e07da1…'}, not the l_out-1 of part-0-1", exit 1.
- **M31:** layer 2 fed `l_out-0` instead of `l_out-1`. The stitch fails: "…
  was fed {'node': 'l_out-0', …}, not the l_out-1 of part-0-1", exit 1.

**Result, full model:**
- **vLLM's greedy continuation:** every prompt case's argmax follows the
  sequence at 16 of 16 positions, so vLLM's greedy continuation is the torch
  port's. Raw German repeats:

  ```text
  INFO de-raw: vLLM's argmax along the sequence is its next token at 16 of 16 positions, so vLLM's greedy continuation is the sequence's: ' Deutschland Deutschland Deutschland Deutschland Deutschland …'
  INFO en-raw: … ' Berlin.\n\nWe need to answer the question: "What is the capital of Germany'
  INFO de-chat: … 'Ein Mixture-of-Experts-Modell ist ein neuronales Netz, das aus mehreren spezialisierten'
  ```
- **Torch port (float32) against vLLM:** the four prompt cases agree, with
  logits NMSE 6e-13 to 7e-12 and the same experts in every (token, layer) pair.
  Over the long cases, a few experts flip at near-ties: 99.85% of Top-6 sets
  agree on wiki-c1 and 98.24% on de-long. Both sides are float32 with a
  different summation order.

  ```text
  INFO wiki-c1, torch port float32 vs vLLM: PPL 15.1244 vs 15.1267, logits NMSE 6.13e-06, KLD 0.000055, same top token 100.0%; same experts for 99.9% of (token, layer) pairs, …
  INFO de-long, torch port float32 vs vLLM: PPL 26.5145 vs 26.2731, logits NMSE 4.60e-04, KLD 0.006585, same top token 97.7%; same experts for 98.2% of (token, layer) pairs, …
  INFO de-long, torch port float32 vs vLLM, layer 0 (sliding attention), 1024 of 1024 tokens on the same experts before it: NMSE Q after QK norm 0.00e+00, K after QK norm 0.00e+00, attention output attn_out 6.38e-14, attn_post_norm 5.89e-14
  ```
- **libllama against vLLM:** almost the numbers libllama shows against the
  torch port (PLAN Phase 6). The difference between the two references is far
  below libllama's distance to either:

  ```text
  INFO wiki-c1, libllama vs vLLM: PPL 15.1787 vs 15.1267, logits NMSE 2.02e-03, KLD 0.033433, same top token 96.5%; same experts for 84.1% of (token, layer) pairs, …
  INFO router wiki-c1, libllama vs vLLM: Top-1 same 97.09%, Top-6 set same 84.07%, mean overlap 5.817/6 over 25600 (token, layer) pairs; …
  INFO de-raw, libllama vs vLLM: logits NMSE 9.56e-03, KLD 0.112537, same top token 100.0%; same experts for 56.5% of (token, layer) pairs, …
  INFO de-raw, libllama vs vLLM: argmax follows vLLM's greedy tokens at 16 of 16 positions
  ```

**Metal:** `--metal --case wiki-c1 --case de-long` adds libllama on the GPU,
with the routed experts on the CPU, as `compare_real.py --metal`. It adds
2–3 min per case. Against vLLM, Metal picks the same experts slightly more
often than libllama's CPU path:

| case | CPU vs vLLM, Top-6 set same | Metal vs vLLM | Metal vs CPU |
|---|---|---|---|
| wiki-c1 | 84.07% | 87.17% | 83.88% |
| de-long | 85.50% | 86.63% | 86.39% |

```text
INFO router wiki-c1, libllama MTL0 vs vLLM: Top-1 same 97.68%, Top-6 set same 87.17%, mean overlap 5.861/6 over 25600 (token, layer) pairs; …
INFO wiki-c1, libllama MTL0 vs vLLM: PPL 14.9382 vs 15.1267, logits NMSE 1.36e-03, KLD 0.017919, same top token 96.9%; same experts for 87.2% of (token, layer) pairs, …
INFO de-long, libllama MTL0 vs vLLM: PPL 26.8693 vs 26.2731, logits NMSE 1.60e-03, KLD 0.041976, same top token 93.2%; same experts for 86.6% of (token, layer) pairs, …
```

The half-precision tiles of Metal's F32 matmul do not cost agreement with
vLLM. Metal and the CPU differ from each other about as much as each differs
from vLLM. Both flip at the same near-ties: a third of the sets with a margin
below 0.001 change in every pairing. That is the input PLAN Phase 8 needs for
its decision on an F32 router path on Metal.

All of these lines are INFO, as on the full checkpoint below: the tolerances
are still open. The chain runs in float32, not in vLLM's BF16. A BF16 chain
would also add a BF16 rounding at every handover, because the next layer
normalizes the unrounded float32 sum. So it cannot stand in for the BF16 run
the tolerances need.

## On the real checkpoint

```sh
vllm-venv/bin/python tools/ref/vllm_capture.py --model Kolibri-1-BF16 \
    --corpus testdata/e2e/corpus.json --dtype bfloat16 --attn-layers 0,4 --out cap-bf16
# back on the Mac, with the capture copied over:
.venv/bin/python tools/ref/compare_vllm.py --llama-cpp third_party/llama.cpp --capture cap-bf16 \
    --gguf ~/models/Kolibri-1-BF16.gguf --libllama
```

`--model` is the checkpoint directory at the pinned revision `7a8f290e`. Every
line is INFO, since the tolerances (PLAN Phase 6) are to be set from this
comparison. The torch port runs in its bfloat16 emulation here.

**Not measured yet, estimates from the file sizes:**
- **BF16:** the weights alone are 156.2 GB, so it needs one GPU with more than
  that, a 192 GB-class GPU such as a B200.
- **FP8:** 78.9 GB fits one 141 GB H200 or larger. It is a different
  reference than BF16. Against the BF16 GGUF, its numbers include the FP8
  quantization.
- **The corpus is short:** 1,024 tokens at most, so the KV cache is small.

**`--pipeline-parallel N`** splits the layers across GPUs. Every layer still
lives whole on one worker, and `collective_rpc` merges the workers' captures.
It is implemented, but unverified. vLLM's CPU backend fails with pipeline
parallel 2 even without any hooks: a plain `LLM(pipeline_parallel_size=2)`
dies with a `KeyError` in the scheduler's `update_from_output`, with threads
bound by `VLLM_CPU_OMP_THREADS_BIND='0-3|4-7'`. A single GPU is the verified
path. Tensor parallelism would split heads and experts across workers and is
not offered.

**What is open in PLAN:**
- **Done:**
  - the capture tooling, verified on the tiny checkpoint;
  - the full model in float32 as chained slices (above), which closes the
    Phase 6 "vLLM" leaves it covers.
- **Still open, waiting for the real run:**
  - the vLLM version of the reference run (128/132);
  - the run itself (134) and its corpus (136);
  - the BF16 tolerances.
- **Once the run exists:** its output and the comparison above are the
  evidence those items need.
