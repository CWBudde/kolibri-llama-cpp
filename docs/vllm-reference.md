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
| `ffn_moe_out-il` | `mlp` output minus `ffn_shexp`: vLLM's MoE runner returns their sum |
| `l_out-il` | the decoder layer's hidden state plus residual: vLLM adds them in the next layer's norm |
| logits | `model.compute_logits` on the output of `model.norm`: the plugin's `LogitsProcessor`, in the config's `head_dtype` |

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
`e2e.py`'s compact set does. From the shapes, de-long's capture is still about
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
    --model /tmp/kt/kolibri-tiny --tiny --libllama
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
bfloat16 with `--attn-layers 0,4` (no `--tiny`, INFO only), runs through all
six cases. Example line: "INFO de-chat, libllama vs vLLM: logits NMSE 5.62e-04,
KLD 0.000029, same top token 95.6%; same experts for 99.8% of (token, layer)
pairs". These numbers measure bf16 rounding on random weights, not the model.

**Negative control M27:** the captured router logits of layer 3 scaled by
1.01 (as M17) fail, exit 1:

```text
FAIL tiny, torch port vs vLLM, router probe: worst layer router-logit NMSE 9.80e-05 where every earlier layer picks the same experts, Top-1 same 99.8%, Top-2 set same 100.0%
FAIL vLLM capture, 2 of 8 experts: 1 check(s) failed
```

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

**What the run closes in PLAN:**
- **Phase 0:** the vLLM version (128/132), the reference run (134) and the
  corpus (136);
- **Phase 6:** the "vLLM" leaves;
- **The tolerances** (285).
