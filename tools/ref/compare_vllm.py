#!/usr/bin/env python3
"""Compare a vLLM capture (vllm_capture.py) with the torch port and libllama.

vLLM is the reference: every line compares the torch port (kolibri_ref.py)
and, with --libllama, libllama on the CPU against the captured vLLM arrays,
on the captured tokens, with the experts per token the capture ran with
(--metal adds libllama on the GPU with the routed experts on the CPU, as
compare_real.py --metal, and its router against libllama's on the CPU):

- per layer, the NMSE of every captured node and the share of tokens with the
  same experts (where not all experts are active, a node counts only the tokens
  whose experts agree in that layer and every earlier one);
- the node lines of compare_real.py (embeddings, the first sliding and the
  first full layer, the worst expert layer);
- the router probe (router_probe.py) and logits NMSE, KLD and same top token
  (a chunk: PPL and the rest over its second half, as e2e.py);
- whether the torch port's or libllama's argmax follows vLLM's greedy tokens.

--gate (alias --tiny) gates a float32 capture with compare_real.py --tiny's
bounds: NMSE 1e-6 for the logits (with top-k routing over the tokens whose
experts agree in every layer), every layer's nodes, the node lines and the
masked router logits, and the same experts for 98% of the
(token, layer) pairs. It exits 1 on a FAIL. It applies to the cmd/kolibri-tiny
checkpoint and to slices of the real weights (slice.py), both captured in
float32. --max-nmse tightens the NMSE bound: on a real-weight slice the three
agree to about 1e-11, and a window one key short (docs/vllm-reference.md, M28)
moves the attention output only to about 1e-7, inside 1e-6. On the full real
checkpoint in bfloat16 the lines are INFO: its tolerances (PLAN Phase 6) are
still to be set from this comparison.

The torch port runs in float64 against a float32 capture and in its bfloat16
emulation against a bfloat16 one. On the full checkpoint that forward takes
hours; --port-artifacts takes its float32 side from e2e.py's stored artifacts
instead, for the nodes they hold (the logits and the router in every case,
e2e.py's compact node set in wiki-c1 and de-long), as for the float32 capture
of the full model from vllm_chain.py. libllama runs with an F32 KV cache and
without flash attention under --gate, as compare_real.py --tiny, and with its
defaults otherwise, as e2e.py.

    compare_vllm.py --llama-cpp third_party/llama.cpp --capture DIR --model DIR --gate --libllama
    compare_vllm.py --llama-cpp third_party/llama.cpp --capture DIR --gguf ~/models/Kolibri-1-BF16.gguf --libllama
    compare_vllm.py --llama-cpp third_party/llama.cpp --capture CHAIN --gguf ~/models/Kolibri-1-BF16.gguf \\
        --port-artifacts ~/models/eval/e2e --libllama [--metal --case wiki-c1 --case de-long]
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "gguf"))
sys.path.insert(0, str(ROOT / "tools" / "ref"))

from check_model import Runner, nmse  # noqa: E402
from compare_real import (EMBD, MAX_NMSE_TINY, MIN_SAME_EXPERTS_TINY, NODES, capture_names,  # noqa: E402
                          chunk_stats, flat, logit_stats, masked_router_nmse, node_lines, router, stats_text)
from kolibri_ref import Weights, forward  # noqa: E402
from router_probe import same_experts, summary  # noqa: E402
from tiny import arch  # noqa: E402

N_CTX = 1024


def load(path: Path) -> dict:
    """The capture as a node dict in kolibri_ref's naming, plus tokens and logits."""
    import numpy as np

    with np.load(path) as z:
        a = {key: z[key] for key in z.files}
    d = {key.removeprefix("node."): v for key, v in a.items() if key.startswith("node.")}
    for il in range(len(a["router_logits"])):
        d[f"ffn_moe_logits-{il}"] = a["router_logits"][il]
        d[f"ffn_moe_topk-{il}"] = a["router_topk"][il]
    return {"nodes": d, "tokens": a["tokens"].tolist(), "logits": a["logits"], "is_swa": a["is_swa"]}


def artifacts(path: Path, tokens: list[int]) -> tuple[dict, object]:
    """e2e.py's float32 reference arrays of a case as a torch port dump, and its logits."""
    import numpy as np

    with np.load(path) as z:
        a = {key: z[key] for key in z.files}
    if a["tokens"].tolist() != tokens:
        raise SystemExit(f"FAIL {path}: other tokens than the capture")
    d = {key.removeprefix("node."): v for key, v in a.items() if key.startswith("node.")}
    for il in range(len(a["router_logits"])):
        d[f"ffn_moe_logits-{il}"] = a["router_logits"][il]
        d[f"ffn_moe_topk-{il}"] = a["router_topk"][il]
    return d, a["ref_float32"]


def layer_table(got: dict, ref: dict, n_layer: int, mask_rows: bool) -> list[tuple]:
    """compare_real.layer_table for a ref that may lack nodes (--attn-layers): NaN where it does."""
    import numpy as np

    rows, ok = [], None
    for il in range(n_layer):
        same = same_experts(got[f"ffn_moe_topk-{il}"], ref[f"ffn_moe_topk-{il}"])
        ok = same if ok is None else ok & same
        keep = ok if mask_rows else np.ones_like(ok)
        vals = [nmse(ref[key][keep], got[key][keep]) if key in ref and key in got and keep.any() else float("nan")
                for key in (f"{node}-{il}" for node in NODES)]
        rows.append((il, *vals, float(same.mean())))
    return rows


def compare(label: str, got: dict, got_logits, cap: dict, entry: dict, W: Weights, all_experts: bool,
            report, bound: float) -> None:
    import numpy as np

    ref, tokens = cap["nodes"], cap["tokens"]
    rows = layer_table(got, ref, W.n_layer, mask_rows=not all_experts)
    print(f"INFO {label}: layer, NMSE " + ", ".join(NODES) + ", same experts")
    for il, *vals, same in rows:
        print(f"INFO   {il:2d}  " + "  ".join(f"{v:.2e}" for v in vals) + f"  {same:6.1%}")
    worst = max((v for row in rows for v in row[1:-1] if not np.isnan(v)), default=float("nan"))
    same = float(np.mean([row[-1] for row in rows]))
    if entry["kind"] == "chunk":
        s = chunk_stats(cap["logits"], got_logits, tokens, len(tokens) // 2)
        text = f"PPL {s['ppl_test']:.4f} vs {s['ppl_base']:.4f}, " + stats_text(s)
    else:
        s = logit_stats(cap["logits"], got_logits)
        text = stats_text(s)
    if all_experts:
        report(s["nmse"] <= bound and worst <= bound,
               f"{label}: {text}, worst layer node NMSE {worst:.2e}")
    else:
        # an expert flip changes the logits legitimately, so they are gated over the tokens whose experts agree
        # in every layer, as the nodes are
        ok = np.all([same_experts(got[f"ffn_moe_topk-{il}"], ref[f"ffn_moe_topk-{il}"])
                     for il in range(W.n_layer)], axis=0)
        v = nmse(cap["logits"][ok], got_logits[ok]) if ok.any() else float("nan")
        report(same >= MIN_SAME_EXPERTS_TINY and worst <= bound and v <= bound,
               f"{label}: {text}; same experts for {same:.1%} of (token, layer) pairs, worst layer node NMSE "
               f"{worst:.2e} and logits NMSE {v:.2e} ({int(ok.sum())} of {len(ok)} tokens) where they agree")
    # node_lines reads the attention nodes of the first sliding and the first full layer, from both sides
    first = {list(W.is_swa).index(True), list(W.is_swa).index(False)}
    missing = sorted(il for il in first if f"Qcur_normed-{il}" not in ref or f"Qcur_normed-{il}" not in got)
    if missing:
        print(f"INFO {label}: node lines skipped, the attention nodes of layer(s) {missing} are not on both sides")
    else:
        for v, line in node_lines(got, {key: v for key, v in ref.items() if key in got}, W):
            report(v <= bound, f"{label}, {line}")
    p = router(got, ref, W)
    v = masked_router_nmse(p)
    report(v <= bound and p["set_same"].mean() >= MIN_SAME_EXPERTS_TINY,
           f"{label}, router probe: worst layer router-logit NMSE {v:.2e} where every earlier layer picks the "
           f"same experts, Top-1 same {p['top1_same'].mean():.1%}, Top-{W.n_expert_used} set same "
           f"{p['set_same'].mean():.1%}")
    for line in summary(p, label):
        print("INFO " + line)
    if entry["greedy"]:
        n0 = entry["n_prompt"]
        top = got_logits[n0 - 1:-1].argmax(axis=1).tolist()
        agree = sum(a == b for a, b in zip(top, entry["greedy"]))
        report(top == entry["greedy"], f"{label}: argmax follows vLLM's greedy tokens at {agree} of "
                                       f"{len(entry['greedy'])} positions")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True)
    ap.add_argument("--capture", type=Path, required=True, help="a vllm_capture.py output directory")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gguf", type=Path, help="the GGUF of the captured checkpoint")
    src.add_argument("--model", type=Path, help="the captured checkpoint, converted here to an F32 GGUF")
    ap.add_argument("--libllama", action="store_true", help="also compare libllama on the CPU")
    ap.add_argument("--metal", action="store_true",
                    help="--libllama: also on the GPU, routed experts on the CPU (as compare_real.py --metal)")
    ap.add_argument("--case", action="append", help="only these cases")
    ap.add_argument("--gate", "--tiny", dest="gate", action="store_true",
                    help="gate a float32 capture with compare_real.py --tiny's bounds")
    ap.add_argument("--max-nmse", type=float, default=MAX_NMSE_TINY,
                    help=f"--gate: the NMSE bound (default {MAX_NMSE_TINY:g}, compare_real.py --tiny's)")
    ap.add_argument("--port-artifacts", type=Path, help="e2e.py's artifact directory: the torch port's float32 "
                                                       "arrays instead of its forward")
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()
    if args.metal and not args.libllama:
        raise SystemExit("--metal needs --libllama")

    meta = json.loads((args.capture / "capture.json").read_text())
    unknown = sorted(set(args.case or []) - set(meta["cases"]))
    if unknown:
        raise SystemExit(f"unknown cases: {', '.join(unknown)} (the capture holds {', '.join(meta['cases'])})")
    gguf = args.gguf
    if args.model:
        from tiny import convert

        gguf = convert(args.model, args.llama_cpp, "f32")
    W = Weights(gguf, args.llama_cpp)
    prefix = arch(gguf, args.llama_cpp)  # kolibri, or kolibri1 for upstream's converter and kolibri-rekey
    converted, k = W.n_expert_used, meta["experts_used"]
    W.n_expert_used = k
    n_expert = W.get("blk.0.exp_probs_b.bias").numel()
    all_experts = k == n_expert
    dtype = {"float32": "float64", "bfloat16": "bfloat16"}[meta["dtype"]]
    v = meta["versions"]
    port_from = f"float32 artifacts in {args.port_artifacts}" if args.port_artifacts else dtype
    chain = f", stitched from {len(meta['chain'])} slices" if meta.get("chain") else ""
    print(f"INFO capture: vllm {v['vllm']}, aleph-alpha-inference {v['aleph-alpha-inference']}, torch {v['torch']}, "
          f"{meta['dtype']}, {k} of {n_expert} experts per token{chain}; torch port in {port_from}; {gguf.name}")

    errs = []

    def report(ok: bool, what: str) -> None:
        if args.gate:
            print(("PASS " if ok else "FAIL ") + what)
            if not ok:
                errs.append(what)
        else:
            print("INFO " + what)

    runner = Runner(args.llama_cpp) if args.libllama else None
    devs = runner.devices() if runner else []
    if args.metal and len(devs) < 2:
        raise SystemExit("--metal: no GPU device")
    for name, entry in meta["cases"].items():
        if args.case and name not in args.case:
            continue
        cap = load(args.capture / f"{name}.npz")
        tokens = cap["tokens"]
        if list(map(bool, cap["is_swa"])) != list(W.is_swa):
            raise SystemExit(f"FAIL {name}: the capture's sliding-window pattern differs from {gguf.name}")
        diff = entry["logprobs_max_abs_diff"]
        print(f"INFO {name}: {len(tokens)} tokens, vLLM decode logprobs vs its prefill logits max abs diff "
              + (f"{diff:.2e}" if diff is not None else "n/a (no decode run)"))
        if args.port_artifacts:
            port, port_logits = artifacts(args.port_artifacts / f"{name}.npz", tokens)
        else:
            port = {}
            port_logits = forward(W, tokens, dtype, port).numpy()
        compare(f"{name}, torch port{' float32' if args.port_artifacts else ''} vs vLLM", port, port_logits, cap,
                entry, W, all_experts, report, args.max_nmse)
        if runner:
            lib = runner.lib
            ctx = dict(flash_attn_type=lib.LLAMA_FLASH_ATTN_TYPE_DISABLED, type_k=lib.GGML_TYPE_F32,
                       type_v=lib.GGML_TYPE_F32) if args.gate else {}
            overrides = {f"{prefix}.expert_used_count": k} if k != converted else None

            def run(dev, **kw):
                capture = capture_names(W.n_layer)
                t0 = time.time()
                logits = runner.logits(gguf, dev, tokens, len(tokens), overrides, capture=capture,
                                       n_ctx=max(N_CTX, len(tokens)), n_threads=args.threads,
                                       n_threads_batch=args.threads, **ctx, **kw)
                return logits, flat(capture, len(tokens)), time.time() - t0

            got_logits, got, _ = run(devs[0][1])
            assert EMBD in got
            compare(f"{name}, libllama vs vLLM", got, got_logits, cap, entry, W, all_experts, report,
                    args.max_nmse)
            if args.metal:
                gpu_name, gpu = devs[1]
                mtl_logits, mtl, seconds = run(gpu, cpu_moe=True)
                print(f"INFO {name} on {gpu_name}, routed experts on the CPU: libllama {seconds:.0f} s")
                compare(f"{name}, libllama {gpu_name} vs vLLM", mtl, mtl_logits, cap, entry, W, all_experts,
                        report, args.max_nmse)
                # the GPU's router against the CPU's, both libllama: what the backend alone changes
                for line in summary(router(mtl, got, W), f"{name}, libllama {gpu_name} vs libllama CPU"):
                    print("INFO " + line)
                del mtl

    if args.gate:
        print(("FAIL" if errs else "PASS") + f" vLLM capture, {k} of {n_expert} experts: "
              + (f"{len(errs)} check(s) failed" if errs else f"torch port and libllama within NMSE {args.max_nmse:g}"))
        sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
