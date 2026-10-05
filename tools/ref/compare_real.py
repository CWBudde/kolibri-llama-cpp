#!/usr/bin/env python3
"""Compare libllama with the torch reference forward (kolibri_ref.py) on whole models.

--tiny validates the reference itself. It runs on the cmd/kolibri-tiny
checkpoint converted to F32, against libllama on the CPU with an F32 KV cache
and without flash attention, where check_moe.py and check_attn.py show every
step to match the reference semantics:

- with all experts active (no top-k discontinuity), the logits and every
  layer's attn_post_norm, ffn_moe_out, ffn_shexp and l_out must be within
  NMSE 1e-6 of the reference in float64;
- with the converted top-k routing, the same holds where both pick the same
  experts, which they must for at least 98% of the (token, layer) pairs.

--gguf runs a GGUF of the real checkpoint on the CPU, with libllama's default
KV cache and flash-attention settings (those of llama-completion and
llama-perplexity), against the reference in float32:

- a German prompt: per-layer NMSE and expert agreement, both top-5 next
  tokens, the reference's greedy continuation, and whether libllama's greedy
  token agrees along that continuation;
- wikitext-2 test chunk 1 (512 tokens, taken from a llama-perplexity
  --kl-divergence-base file): per-layer NMSE and expert agreement, PPL over the
  second half (as llama-perplexity), and KLD and same-top-token of libllama
  against the reference;
- the sensitivity baseline: the reference in bfloat16 (vLLM's precision)
  against the reference in float32, on the same chunk.

These lines are INFO, not PASS/FAIL: the real model has no tolerance yet
(PLAN Phase 6), and this reference is a port of the vLLM code, not vLLM.

    compare_real.py --llama-cpp third_party/llama.cpp --tiny
    compare_real.py --llama-cpp third_party/llama.cpp --gguf ~/models/Kolibri-1-BF16.gguf \\
        --kld-base ~/models/eval/kld-bf16-c512-n20.bin
"""

import argparse
import random
import struct
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "gguf"))
sys.path.insert(0, str(ROOT / "tools" / "ref"))

from check_model import Runner, nmse  # noqa: E402
from kolibri_ref import Weights, forward, greedy  # noqa: E402
from tiny import convert, generate_tiny  # noqa: E402

ARCH = "kolibri"
NODES = ("attn_post_norm", "ffn_moe_out", "ffn_shexp", "l_out")
MAX_NMSE_TINY = 1e-6
MIN_SAME_EXPERTS_TINY = 0.98
GERMAN = "Die Hauptstadt von Deutschland ist"
N_GREEDY = 16


def capture_names(n_layer: int) -> dict:
    return {f"{node}-{il}": None for il in range(n_layer) for node in NODES + ("ffn_moe_topk",)}


def flat(capture: dict, n_tokens: int) -> dict:
    """One array per node, [n_tokens, width] (each run is one computation)."""
    return {name: parts[-1].reshape(n_tokens, -1) for name, parts in capture.items() if parts}


def same_experts(a, b):
    """Per token, whether both select the same set of experts."""
    import numpy as np

    return (np.sort(a, axis=1) == np.sort(b, axis=1)).all(axis=1)


def layer_table(got: dict, ref: dict, n_layer: int, mask_rows: bool = False) -> list[tuple]:
    """Per layer: NMSE of each node and the share of tokens with the same experts. With mask_rows,
    the NMSE only counts tokens whose experts agree in that layer and in every earlier one."""
    import numpy as np

    rows, ok = [], None
    for il in range(n_layer):
        same = same_experts(got[f"ffn_moe_topk-{il}"], ref[f"ffn_moe_topk-{il}"])
        ok = same if ok is None else ok & same
        keep = ok if mask_rows else np.ones_like(ok)
        vals = [nmse(ref[f"{node}-{il}"][keep], got[f"{node}-{il}"][keep]) if keep.any() else float("nan")
                for node in NODES]
        rows.append((il, *vals, float(same.mean())))
    return rows


def print_table(rows: list[tuple], what: str) -> None:
    print(f"INFO {what}: layer, NMSE " + ", ".join(NODES) + ", same experts")
    for il, *vals, same in rows:
        print(f"INFO   {il:2d}  " + "  ".join(f"{v:.2e}" for v in vals) + f"  {same:6.1%}")


def log_softmax64(x):
    import numpy as np

    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def chunk_stats(base, test, tokens: list[int], first: int) -> dict:
    """llama-perplexity's scoring of one chunk: positions first .. n-2 predict the next token.
    KLD is KL(base || test)."""
    import numpy as np

    lb, lt = log_softmax64(base[first:-1]), log_softmax64(test[first:-1])
    nxt = np.asarray(tokens[first + 1:])
    idx = np.arange(len(nxt))
    return {
        "ppl_base": float(np.exp(-lb[idx, nxt].mean())),
        "ppl_test": float(np.exp(-lt[idx, nxt].mean())),
        "kld": float((np.exp(lb) * (lb - lt)).sum(axis=1).mean()),
        "same_top": float((lb.argmax(1) == lt.argmax(1)).mean()),
    }


def run_tiny(runner: Runner, llama_cpp: Path, seed: int) -> list[str]:
    import json

    import numpy as np

    errs = []

    def report(ok: bool, what: str) -> None:
        print(("PASS " if ok else "FAIL ") + what)
        if not ok:
            errs.append(what)

    lib = runner.lib
    ctx = dict(flash_attn_type=lib.LLAMA_FLASH_ATTN_TYPE_DISABLED, type_k=lib.GGML_TYPE_F32, type_v=lib.GGML_TYPE_F32)
    with tempfile.TemporaryDirectory() as tmp:
        model = generate_tiny(Path(tmp), seed)
        n_expert = json.loads((model / "config.json").read_text())["num_experts"]
        path = convert(model, llama_cpp, "f32")
        W = Weights(path, llama_cpp)
        rng = random.Random(seed)
        tokens = [rng.randrange(128000) for _ in range(100)]
        _, cpu = runner.devices()[0]
        converted = W.n_expert_used
        for label, n_used in ((f"all {n_expert} experts", n_expert), (f"top-{converted} routing", None)):
            capture = capture_names(W.n_layer)
            overrides = {f"{ARCH}.expert_used_count": n_used} if n_used else None
            got_logits = runner.logits(path, cpu, tokens, len(tokens), overrides, capture=capture, **ctx)
            got = flat(capture, len(tokens))
            W.n_expert_used = n_used or converted
            ref = {}
            ref_logits = forward(W, tokens, "float64", ref).numpy()
            rows = layer_table(got, ref, W.n_layer, mask_rows=n_used is None)
            worst = max(v for row in rows for v in row[1:-1])
            same = float(np.mean([row[-1] for row in rows]))
            if n_used:
                v = nmse(ref_logits, got_logits)
                report(v <= MAX_NMSE_TINY and worst <= MAX_NMSE_TINY,
                       f"tiny, {label}: logits NMSE {v:.2e}, worst layer node NMSE {worst:.2e} (bound {MAX_NMSE_TINY:g})")
            else:
                report(same >= MIN_SAME_EXPERTS_TINY and worst <= MAX_NMSE_TINY,
                       f"tiny, {label}: same experts for {same:.1%} of (token, layer) pairs, worst layer node NMSE "
                       f"{worst:.2e} where they agree (bound {MAX_NMSE_TINY:g})")
        W.n_expert_used = converted
    return errs


def run_real(runner: Runner, llama_cpp: Path, gguf: Path, kld_base: Path) -> None:
    import numpy as np
    from tokenizers import Tokenizer

    sys.path.insert(0, str(ROOT / "tools" / "tokenizer"))
    from common import tokenizer_dir

    tok = Tokenizer.from_file(str(tokenizer_dir() / "tokenizer.json"))
    W = Weights(gguf, llama_cpp)
    _, cpu = runner.devices()[0]

    def both(tokens: list[int], what: str):
        t0 = time.time()
        capture = capture_names(W.n_layer)
        got_logits = runner.logits(gguf, cpu, tokens, len(tokens), capture=capture, n_ctx=1024)
        t1 = time.time()
        ref = {}
        ref_logits = forward(W, tokens, "float32", ref).numpy()
        print(f"INFO {what}: {len(tokens)} tokens, libllama {t1 - t0:.0f} s, reference {time.time() - t1:.0f} s")
        return got_logits, flat(capture, len(tokens)), ref_logits, ref

    def maxima(ref: dict, got: dict) -> None:
        kv = np.stack([ref[f"kv_absmax-{il}"] for il in range(W.n_layer)])
        lout = [float(np.abs(got[f"l_out-{il}"]).max()) for il in range(W.n_layer)]
        print(f"INFO activation maxima: |k| {kv[:, 0].max():.1f}, |v| {kv[:, 1].max():.1f} (reference), "
              f"|l_out| {max(lout):.1f} at layer {int(np.argmax(lout))} (libllama), F16 limit 65504; "
              f"non-finite in libllama: {sum(int((~np.isfinite(a)).sum()) for a in got.values())}")

    def margins(ref: dict) -> None:
        gaps, spread = [], []
        for il in range(W.n_layer):
            s = np.sort(ref[f"router_sel-{il}"], axis=1)[:, ::-1]
            gaps.append(s[:, W.n_expert_used - 1] - s[:, W.n_expert_used])
            spread.append(s.std(axis=1))
        g = np.concatenate(gaps)
        print(f"INFO router margin (6th minus 7th of logits + bias, reference): median {np.median(g):.3f}, "
              f"below 0.01 for {(g < 0.01).mean():.1%}, below 0.1 for {(g < 0.1).mean():.1%} of (token, layer) "
              f"pairs; median score spread {np.median(np.concatenate(spread)):.2f}")

    # German prompt
    ids = tok.encode(GERMAN, add_special_tokens=False).ids
    got_logits, got, ref_logits, ref = both(ids, f"prompt {GERMAN!r} {ids}")
    print_table(layer_table(got, ref, W.n_layer), "prompt")
    maxima(ref, got)
    for name, lg in (("libllama", got_logits), ("reference", ref_logits)):
        top = np.argsort(lg[-1])[::-1][:5]
        print(f"INFO {name} next-token top 5: " + ", ".join(f"{tok.decode([int(t)])!r}" for t in top))
    W.cache = {}
    cont = greedy(W, ids, N_GREEDY)
    print(f"INFO reference greedy, float32: {GERMAN + tok.decode(cont)!r}")
    cont16 = greedy(W, ids, N_GREEDY, "bfloat16")
    print(f"INFO reference greedy, bfloat16: {GERMAN + tok.decode(cont16)!r}")
    W.cache = None
    path = ids + cont
    forced = runner.logits(gguf, cpu, path, len(path), n_ctx=1024)
    agree = int((forced[len(ids) - 1:-1].argmax(1) == np.asarray(cont)).sum())
    print(f"INFO libllama greedy token along the reference continuation: equal at {agree}/{N_GREEDY}")

    # wikitext-2 chunk 1
    raw = kld_base.read_bytes()
    assert raw[:8] == b"_logits_", f"{kld_base}: not a llama-perplexity --kl-divergence-base file"
    n_ctx, _, n_chunk = struct.unpack("<iii", raw[8:20])
    tokens = list(np.frombuffer(raw, dtype=np.int32, count=n_ctx, offset=20).tolist())
    got_logits, got, ref_logits, ref = both(tokens, f"wikitext-2 chunk 1 of {n_chunk} from {kld_base.name}")
    rows = layer_table(got, ref, W.n_layer)
    print_table(rows, "chunk 1")
    maxima(ref, got)
    margins(ref)
    first = n_ctx // 2
    s = chunk_stats(ref_logits, got_logits, tokens, first)
    print(f"INFO chunk 1, libllama vs reference float32: PPL {s['ppl_test']:.4f} vs {s['ppl_base']:.4f}, "
          f"KLD {s['kld']:.6f}, same top token {s['same_top']:.1%}")
    t0 = time.time()
    ref16 = forward(W, tokens, "bfloat16").numpy()
    s = chunk_stats(ref_logits, ref16, tokens, first)
    print(f"INFO chunk 1, reference bfloat16 vs float32 ({time.time() - t0:.0f} s): PPL {s['ppl_test']:.4f} vs "
          f"{s['ppl_base']:.4f}, KLD {s['kld']:.6f}, same top token {s['same_top']:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tiny", action="store_true", help="validate the reference on the tiny fixture")
    mode.add_argument("--gguf", type=Path, help="GGUF of the real checkpoint (BF16)")
    ap.add_argument("--kld-base", type=Path, help="llama-perplexity --kl-divergence-base file (with --gguf)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    runner = Runner(args.llama_cpp)
    if args.tiny:
        errs = run_tiny(runner, args.llama_cpp, args.seed)
        if errs:
            print(f"FAIL {len(errs)} problem(s)")
            sys.exit(1)
        return
    if args.kld_base is None:
        ap.error("--gguf needs --kld-base")
    run_real(runner, args.llama_cpp, args.gguf, args.kld_base)


if __name__ == "__main__":
    main()
