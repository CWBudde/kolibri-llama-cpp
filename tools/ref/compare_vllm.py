#!/usr/bin/env python3
"""Compare a vLLM capture (vllm_capture.py) with the torch port and libllama.

vLLM is the reference: every line compares the torch port (kolibri_ref.py)
and, with --libllama, libllama on the CPU against the captured vLLM arrays,
on the captured tokens, with the experts per token the capture ran with:

- per layer, the NMSE of every captured node and the share of tokens with the
  same experts (where not all experts are active, a node counts only the tokens
  whose experts agree in that layer and every earlier one);
- the node lines of compare_real.py (embeddings, the first sliding and the
  first full layer, the worst expert layer);
- the router probe (router_probe.py) and logits NMSE, KLD and same top token
  (a chunk: PPL and the rest over its second half, as e2e.py);
- whether the torch port's or libllama's argmax follows vLLM's greedy tokens.

--tiny gates the cmd/kolibri-tiny checkpoint, captured in float32, with
compare_real.py's bounds: NMSE 1e-6 for the logits (all experts), every
layer's nodes, the node lines and the masked router logits, and the same
experts for 98% of the (token, layer) pairs. It exits 1 on a FAIL. On the
real checkpoint the lines are INFO: its tolerances (PLAN Phase 6) are still to
be set from this comparison.

The torch port runs in float64 against a float32 capture and in its bfloat16
emulation against a bfloat16 one. libllama runs with an F32 KV cache and
without flash attention under --tiny, as compare_real.py --tiny, and with its
defaults otherwise, as e2e.py.

    compare_vllm.py --llama-cpp third_party/llama.cpp --capture DIR --model DIR --tiny --libllama
    compare_vllm.py --llama-cpp third_party/llama.cpp --capture DIR --gguf ~/models/Kolibri-1-BF16.gguf --libllama
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "gguf"))
sys.path.insert(0, str(ROOT / "tools" / "ref"))

from check_model import Runner, nmse  # noqa: E402
from compare_real import (ARCH, EMBD, MAX_NMSE_TINY, MIN_SAME_EXPERTS_TINY, NODES, capture_names,  # noqa: E402
                          chunk_stats, flat, logit_stats, masked_router_nmse, node_lines, router, stats_text)
from kolibri_ref import Weights, forward  # noqa: E402
from router_probe import same_experts, summary  # noqa: E402

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


def layer_table(got: dict, ref: dict, n_layer: int, mask_rows: bool) -> list[tuple]:
    """compare_real.layer_table for a ref that may lack nodes (--attn-layers): NaN where it does."""
    import numpy as np

    rows, ok = [], None
    for il in range(n_layer):
        same = same_experts(got[f"ffn_moe_topk-{il}"], ref[f"ffn_moe_topk-{il}"])
        ok = same if ok is None else ok & same
        keep = ok if mask_rows else np.ones_like(ok)
        vals = [nmse(ref[key][keep], got[key][keep]) if key in ref and keep.any() else float("nan")
                for key in (f"{node}-{il}" for node in NODES)]
        rows.append((il, *vals, float(same.mean())))
    return rows


def compare(label: str, got: dict, got_logits, cap: dict, entry: dict, W: Weights, all_experts: bool,
            report) -> None:
    import numpy as np

    ref, tokens = cap["nodes"], cap["tokens"]
    rows = layer_table(got, ref, W.n_layer, mask_rows=not all_experts)
    print(f"INFO {label}: layer, NMSE " + ", ".join(NODES) + ", same experts")
    for il, *vals, same in rows:
        print(f"INFO   {il:2d}  " + "  ".join(f"{v:.2e}" for v in vals) + f"  {same:6.1%}")
    worst = max(v for row in rows for v in row[1:-1] if not np.isnan(v))
    same = float(np.mean([row[-1] for row in rows]))
    if entry["kind"] == "chunk":
        s = chunk_stats(cap["logits"], got_logits, tokens, len(tokens) // 2)
        text = f"PPL {s['ppl_test']:.4f} vs {s['ppl_base']:.4f}, " + stats_text(s)
    else:
        s = logit_stats(cap["logits"], got_logits)
        text = stats_text(s)
    if all_experts:
        report(s["nmse"] <= MAX_NMSE_TINY and worst <= MAX_NMSE_TINY,
               f"{label}: {text}, worst layer node NMSE {worst:.2e}")
    else:
        report(same >= MIN_SAME_EXPERTS_TINY and worst <= MAX_NMSE_TINY,
               f"{label}: {text}; same experts for {same:.1%} of (token, layer) pairs, worst layer node NMSE "
               f"{worst:.2e} where they agree")
    for v, line in node_lines(got, ref, W):
        report(v <= MAX_NMSE_TINY, f"{label}, {line}")
    p = router(got, ref, W)
    v = masked_router_nmse(p)
    report(v <= MAX_NMSE_TINY and p["set_same"].mean() >= MIN_SAME_EXPERTS_TINY,
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
    ap.add_argument("--tiny", action="store_true", help="gate with compare_real.py --tiny's bounds")
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()

    meta = json.loads((args.capture / "capture.json").read_text())
    gguf = args.gguf
    if args.model:
        from tiny import convert

        gguf = convert(args.model, args.llama_cpp, "f32")
    W = Weights(gguf, args.llama_cpp)
    converted, k = W.n_expert_used, meta["experts_used"]
    W.n_expert_used = k
    n_expert = W.get("blk.0.exp_probs_b.bias").numel()
    all_experts = k == n_expert
    dtype = {"float32": "float64", "bfloat16": "bfloat16"}[meta["dtype"]]
    v = meta["versions"]
    print(f"INFO capture: vllm {v['vllm']}, aleph-alpha-inference {v['aleph-alpha-inference']}, torch {v['torch']}, "
          f"{meta['dtype']}, {k} of {n_expert} experts per token; torch port in {dtype}; {gguf.name}")

    errs = []

    def report(ok: bool, what: str) -> None:
        if args.tiny:
            print(("PASS " if ok else "FAIL ") + what)
            if not ok:
                errs.append(what)
        else:
            print("INFO " + what)

    runner = Runner(args.llama_cpp) if args.libllama else None
    for name, entry in meta["cases"].items():
        cap = load(args.capture / f"{name}.npz")
        tokens = cap["tokens"]
        if list(map(bool, cap["is_swa"])) != list(W.is_swa):
            raise SystemExit(f"FAIL {name}: the capture's sliding-window pattern differs from {gguf.name}")
        print(f"INFO {name}: {len(tokens)} tokens, vLLM decode logprobs vs its prefill logits max abs diff "
              f"{entry['logprobs_max_abs_diff']:.2e}")
        port = {}
        port_logits = forward(W, tokens, dtype, port).numpy()
        compare(f"{name}, torch port vs vLLM", port, port_logits, cap, entry, W, all_experts, report)
        if runner:
            lib = runner.lib
            ctx = dict(flash_attn_type=lib.LLAMA_FLASH_ATTN_TYPE_DISABLED, type_k=lib.GGML_TYPE_F32,
                       type_v=lib.GGML_TYPE_F32) if args.tiny else {}
            capture = capture_names(W.n_layer)
            overrides = {f"{ARCH}.expert_used_count": k} if k != converted else None
            _, cpu = runner.devices()[0]
            got_logits = runner.logits(gguf, cpu, tokens, len(tokens), overrides, capture=capture,
                                       n_ctx=max(N_CTX, len(tokens)), n_threads=args.threads,
                                       n_threads_batch=args.threads, **ctx)
            got = flat(capture, len(tokens))
            assert EMBD in got
            compare(f"{name}, libllama vs vLLM", got, got_logits, cap, entry, W, all_experts, report)

    if args.tiny:
        print(("FAIL" if errs else "PASS") + f" vLLM capture, {k} of {n_expert} experts: "
              + (f"{len(errs)} check(s) failed" if errs else "torch port and libllama within the tiny bounds"))
        sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
