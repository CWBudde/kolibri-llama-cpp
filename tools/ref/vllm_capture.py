#!/usr/bin/env python3
"""Capture Kolibri-1 reference outputs from vLLM itself, with Aleph Alpha's
aleph-alpha-inference plugin (PLAN Phases 0 and 6).

Runs in its own environment with vllm and the plugin (requirements-vllm.txt),
not in the repo's .venv, and imports nothing from this repo: a GPU host needs
this file, the checkpoint and, for --corpus, testdata/e2e/corpus.json.
compare_vllm.py then compares the capture with the torch port and libllama.

Each case is run twice:

- prompt cases: a greedy run (temperature 0) gives vLLM's continuation and the
  top-20 logprobs of every generated token;
- a single prefill over the whole sequence (prompt plus that continuation, or
  the chunk) runs with forward hooks on the plugin's modules and records, under
  libllama's node names:
  - embd: model.embed_tokens;
  - Qcur_normed-il, Kcur_normed-il: self_attn.q_norm, k_norm (before RoPE);
  - attn_out-il: self_attn.o_proj; attn_post_norm-il: post_attn_norm;
  - ffn_moe_logits-il: mlp.gate, the F32 router logits before the bias;
    ffn_moe_topk-il: the top k of those logits plus gate.e_score_correction_bias,
    as sigmoid_logit_add_routing selects them;
  - ffn_shexp-il: mlp.shared_experts; ffn_moe_out-il: the routed output as the
    MoE runner adds it to the shared output (its apply_routed_output_transform),
    so a bfloat16 run keeps it apart from the rounding of that sum;
  - l_out-il: the sum of the hidden state and residual the decoder layer returns
    (vLLM adds them in the next layer's norm);
  - logits for every position: model.compute_logits (the plugin's LogitsProcessor,
    in the config's head_dtype) on the output of model.norm.

The prefix cache is off and the whole sequence fits in one batch, so every
position is computed in that prefill. The hooks run in the workers; with
pipeline parallelism each worker holds whole layers, and the parts are merged
by layer. Tensor parallelism would split heads and experts across workers and
is not offered.

As a check of the hooked logits, every greedy token's top-20 logprobs from the
decode steps are compared with the log-softmax of the prefill logits at the
same position ("logprobs_max_abs_diff" in capture.json).

Output, per case <name>.npz (the layout of tools/ref/e2e.py's artifacts):
tokens (T,) int32, logits (T, n_vocab) float32, router_logits (L, T, E) float32,
router_topk (L, T, k) int32, router_bias (L, E) float32, is_swa (L,) bool and
node.<name> (T, width) float32; plus capture.json with the versions, the
device, the settings, the greedy tokens and the sha256 of every array.

    vllm_capture.py --model DIR --random 100 --seed 1 --greedy 4 --dtype float32 --out DIR
    vllm_capture.py --model DIR --corpus testdata/e2e/corpus.json --dtype bfloat16 \\
        --pipeline-parallel 2 --attn-layers 0,4 --out DIR
"""

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")  # collective_rpc with these functions

N_LOGPROBS = 20


def array_sha256(a) -> str:
    """As tools/ref/e2e.py: dtype and shape, then the raw bytes."""
    import numpy as np

    a = np.ascontiguousarray(a)
    return hashlib.sha256(f"{a.dtype.str}{a.shape}".encode() + a.tobytes()).hexdigest()


def _install(worker, attn_layers):
    """In each worker: forward hooks that append their outputs to model._capture while it is a dict."""
    import torch

    model = worker.get_model()
    inner = model.model
    model._capture = None

    def keep(name, t):
        if model._capture is not None:
            # a copy: the rotary embedding rotates q and k in place after q_norm and k_norm return
            model._capture.setdefault(name, []).append(
                t.detach().to("cpu", torch.float32, copy=True).reshape(t.shape[0], -1))

    def first(out):
        return out[0] if isinstance(out, tuple) else out

    hooks = []

    def hook(module, name, fn=first):
        hooks.append(module.register_forward_hook(lambda m, args, out: keep(name, fn(out))))

    def routed(runner, name):
        # The MoE runner returns shared_output + fused_output, rounded in the model dtype; the routed
        # output is taken as the summand, from apply_routed_output_transform (an identity for Kolibri),
        # the last step before that addition.
        transform = runner.apply_routed_output_transform

        def keep_routed(x):
            y = transform(x)
            keep(name, y)
            return y

        runner.apply_routed_output_transform = keep_routed

    from vllm.model_executor.models.utils import PPMissingLayer

    if not isinstance(inner.embed_tokens, PPMissingLayer):
        hook(inner.embed_tokens, "embd")
    layers = []
    for il in range(inner.start_layer, inner.end_layer):
        layer = inner.layers[il]
        attn, mlp = layer.self_attn, layer.mlp
        if attn_layers is None or il in attn_layers:
            hook(attn.q_norm, f"Qcur_normed-{il}")
            hook(attn.k_norm, f"Kcur_normed-{il}")
            hook(attn.o_proj, f"attn_out-{il}")
            hook(layer.post_attn_norm, f"attn_post_norm-{il}")
        hook(mlp.gate, f"ffn_moe_logits-{il}")
        hook(mlp.shared_experts, f"ffn_shexp-{il}")
        routed(mlp.experts, f"ffn_moe_out-{il}")
        hook(layer, f"l_out-{il}", fn=lambda out: out[0] + out[1])
        layers.append({"il": il, "swa": attn.rotary_emb is not None,
                       "bias": mlp.gate.e_score_correction_bias.detach().to("cpu", torch.float32).numpy()})
    if not isinstance(inner.norm, PPMissingLayer):
        hook(inner.norm, "final_norm")
    model._hooks = hooks
    return layers


def _start(worker):
    worker.get_model()._capture = {}


def _collect(worker):
    """The worker's captured nodes, each [n_tokens, width] as float32 numpy, and the logits."""
    import torch

    model = worker.get_model()
    cap, model._capture = model._capture, None
    out = {name: torch.cat(parts).numpy() for name, parts in cap.items() if name != "final_norm"}
    if "final_norm" in cap:
        w = model.lm_head.weight
        with torch.no_grad():
            h = torch.cat(cap["final_norm"]).to(w.device, w.dtype)
            out["logits"] = model.compute_logits(h).to("cpu", torch.float32).numpy()
    return out


def versions() -> dict:
    from importlib.metadata import version

    import torch

    v = {p: version(p) for p in ("vllm", "aleph-alpha-inference", "torch", "transformers")}
    v["python"] = platform.python_version()
    v["platform"] = platform.platform()
    if torch.cuda.is_available():
        v["cuda"] = torch.version.cuda
        v["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return v


def load_cases(args) -> dict:
    if args.corpus:
        corpus = json.loads(args.corpus.read_text())
        return {name: {"kind": c["kind"], "tokens": c["tokens"], "n_greedy": c.get("n_greedy", 0)}
                for name, c in corpus["cases"].items() if not args.case or name in args.case}
    rng = random.Random(args.seed)
    tokens = [rng.randrange(args.vocab) for _ in range(args.random)]
    return {"tiny": {"kind": "prompt", "tokens": tokens, "n_greedy": args.greedy}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="HF checkpoint directory")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--corpus", type=Path, help="testdata/e2e/corpus.json")
    src.add_argument("--random", type=int, help="N random token IDs (compare_real.py --tiny uses 100, seed 1)")
    ap.add_argument("--case", action="append", help="--corpus: only these cases")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--vocab", type=int, default=128000, help="--random: the ID range")
    ap.add_argument("--greedy", type=int, default=0, help="--random: greedy tokens after the random prompt")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--experts-used", type=int, help="override num_experts_per_tok (all experts: no top-k cut)")
    ap.add_argument("--pipeline-parallel", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                    help="vLLM's memory share; on the CPU backend the share of RAM it reserves")
    ap.add_argument("--attn-layers", help="comma-separated layers whose attention nodes are kept (default: all)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import numpy as np
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    cases = load_cases(args)
    longest = max(len(c["tokens"]) + c["n_greedy"] for c in cases.values())
    config = json.loads((args.model / "config.json").read_text())
    overrides = {"num_experts_per_tok": args.experts_used} if args.experts_used else None
    attn_layers = [int(x) for x in args.attn_layers.split(",")] if args.attn_layers else None
    if attn_layers is not None:
        # compare_vllm.py's node lines need the first sliding and the first full layer
        types = config["layer_types"]
        need = {types.index("sliding_attention"), types.index("full_attention")}
        if not need <= set(attn_layers):
            raise SystemExit(f"--attn-layers must include the first sliding and the first full layer, "
                             f"{sorted(need)}")
    llm = LLM(model=str(args.model), dtype=args.dtype, seed=0, enforce_eager=True, enable_prefix_caching=False,
              max_model_len=longest + 16, max_num_batched_tokens=max(2048, longest + 16), max_num_seqs=1,
              pipeline_parallel_size=args.pipeline_parallel, tensor_parallel_size=1, hf_overrides=overrides,
              compilation_config={"mode": 0}, gpu_memory_utilization=args.gpu_memory_utilization)
    layers = sorted((x for part in llm.collective_rpc(_install, args=(attn_layers,)) for x in part),
                    key=lambda x: x["il"])
    n_layer = len(layers)
    assert [x["il"] for x in layers] == list(range(config["num_hidden_layers"]))
    k = args.experts_used or config["num_experts_per_tok"]

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": str(args.model),
        "config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
        "corpus": str(args.corpus) if args.corpus else None,
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest() if args.corpus else None,
        "random": {"n": args.random, "seed": args.seed, "vocab": args.vocab} if args.random else None,
        "dtype": args.dtype,
        "experts_used": k,
        "pipeline_parallel": args.pipeline_parallel,
        "attn_layers": attn_layers,
        "sampling": {"greedy": {"temperature": 0.0, "logprobs": N_LOGPROBS}},
        "versions": versions(),
        "cases": {},
    }
    for name, case in cases.items():
        t0 = time.time()
        prompt = case["tokens"]
        greedy, logprobs = [], []
        if case["n_greedy"]:
            sp = SamplingParams(temperature=0.0, max_tokens=case["n_greedy"], logprobs=N_LOGPROBS, ignore_eos=True)
            out = llm.generate(TokensPrompt(prompt_token_ids=prompt), sp, use_tqdm=False)[0].outputs[0]
            greedy = list(out.token_ids)
            logprobs = [{int(t): lp.logprob for t, lp in step.items()} for step in out.logprobs]
        tokens = prompt + greedy

        llm.collective_rpc(_start)
        llm.generate(TokensPrompt(prompt_token_ids=tokens), SamplingParams(temperature=0.0, max_tokens=1),
                     use_tqdm=False)
        cap = {}
        for part in llm.collective_rpc(_collect):
            cap.update(part)
        n = len(tokens)
        for key, v in cap.items():
            if len(v) != n:
                raise SystemExit(f"FAIL {name}: {key} has {len(v)} rows for {n} tokens (prefill split or padded)")

        logits = cap.pop("logits")
        bias = np.stack([x["bias"] for x in layers])
        router_logits = np.stack([cap.pop(f"ffn_moe_logits-{il}") for il in range(n_layer)])
        router_topk = np.argsort(-(router_logits + bias[:, None, :]), axis=-1, kind="stable")[..., :k]
        arrays = {
            "tokens": np.asarray(tokens, dtype=np.int32),
            "logits": logits.astype(np.float32),
            "router_logits": router_logits.astype(np.float32),
            "router_topk": router_topk.astype(np.int32),
            "router_bias": bias.astype(np.float32),
            "is_swa": np.array([x["swa"] for x in layers]),
            **{f"node.{key}": v.astype(np.float32) for key, v in cap.items()},
        }
        np.savez(args.out / f"{name}.npz", **arrays)

        # the decode-step logprobs against the prefill logits at the same position
        diff = 0.0
        if logprobs:
            x = logits[len(prompt) - 1:len(prompt) - 1 + len(logprobs)].astype(np.float64)
            x = x - x.max(axis=1, keepdims=True)
            x = x - np.log(np.exp(x).sum(axis=1, keepdims=True))
            diff = max(abs(x[i, t] - lp) for i, step in enumerate(logprobs) for t, lp in step.items())
        top = logits[len(prompt) - 1:-1].argmax(axis=1).tolist() if greedy else []
        meta["cases"][name] = {
            "kind": case["kind"],
            "n_tokens": n,
            "n_prompt": len(prompt),
            "greedy": greedy,
            "greedy_matches_prefill_argmax": top == greedy,
            "logprobs_max_abs_diff": float(diff),
            "seconds": round(time.time() - t0, 1),
            "arrays": {key: array_sha256(v) for key, v in arrays.items()},
        }
        print(f"INFO {name}: {n} tokens ({len(greedy)} greedy), {len(arrays)} arrays, decode logprobs vs prefill "
              f"logits max abs diff {diff:.2e}, {meta['cases'][name]['seconds']} s", flush=True)
    (args.out / "capture.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"INFO wrote {len(cases)} case(s) to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
