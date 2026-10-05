#!/usr/bin/env python3
"""Compare libllama with the torch reference forward (kolibri_ref.py) on whole models.

--tiny validates the reference itself. It runs on the cmd/kolibri-tiny
checkpoint converted to F32, against libllama on the CPU with an F32 KV cache
and without flash attention, where check_moe.py and check_attn.py show every
step to match the reference semantics:

- with all experts active (no top-k discontinuity), the logits, the token
  embeddings and every layer's Q and K after the QK norm, attention output
  before and after its sandwich norm, routed and shared expert output and
  l_out must be within NMSE 1e-6 of the reference in float64;
- with the converted top-k routing, the same holds where both pick the same
  experts, which they must for at least 98% of the (token, layer) pairs;
- in both runs the node lines (node_lines) must hold the same bound for the
  embeddings, the first sliding and the first full layer and the worst layer
  of the routed and shared expert output;
- in both runs the router probe (router_probe.py) must find every layer's
  router logits within NMSE 1e-6 where all earlier layers pick the same
  experts, and the same experts for at least 98% of the pairs.

--gguf runs a GGUF of the real checkpoint on the CPU, with libllama's default
KV cache and flash-attention settings (those of llama-completion and
llama-perplexity), against the reference in float32:

- a German prompt: per-layer NMSE and expert agreement, the node lines,
  logits NMSE, KLD and same top token, the router probe, both top-5 next
  tokens, the reference's greedy continuation, and whether libllama's greedy
  token agrees along that continuation;
- wikitext-2 test chunk 1 (512 tokens, taken from a llama-perplexity
  --kl-divergence-base file): per-layer NMSE and expert agreement, the node
  lines, the router probe per layer and in summary, PPL over the second half
  (as llama-perplexity), and logits NMSE, KLD and same top token of libllama
  against the reference;
- the sensitivity baseline: the reference in bfloat16 (vLLM's precision)
  against the reference in float32, on the same chunk, with the node lines and
  the router probe;
- with --metal, the same chunk on the GPU with the routed experts on the CPU
  (as llama-cli -ngl 99 --cpu-moe), against the reference and against
  libllama on the CPU.

--probe-out DIR writes each router probe as router-<name>.npz.

These lines are INFO, not PASS/FAIL: the real model has no tolerance yet
(PLAN Phase 6), and this reference is a port of the vLLM code, not vLLM.

    compare_real.py --llama-cpp third_party/llama.cpp --tiny
    compare_real.py --llama-cpp third_party/llama.cpp --gguf ~/models/Kolibri-1-BF16.gguf \\
        --kld-base ~/models/eval/kld-bf16-c512-n20.bin [--metal] [--probe-out DIR]
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
from router_probe import probe, same_experts, save, summary, table  # noqa: E402
from tiny import convert, generate_tiny  # noqa: E402

ARCH = "kolibri"
NODES = ("Qcur_normed", "Kcur_normed", "attn_out", "attn_post_norm", "ffn_moe_out", "ffn_shexp", "l_out")
EMBD = "embd"  # the token embeddings (ggml_get_rows of token_embd)
ROUTER = ("ffn_moe_topk", "ffn_moe_logits")
MAX_NMSE_TINY = 1e-6
MIN_SAME_EXPERTS_TINY = 0.98
GERMAN = "Die Hauptstadt von Deutschland ist"
N_GREEDY = 16


def capture_names(n_layer: int) -> dict:
    return {EMBD: None, **{f"{node}-{il}": None for il in range(n_layer) for node in NODES + ROUTER}}


def flat(capture: dict, n_tokens: int) -> dict:
    """One array per node, [n_tokens, width] (each run is one computation)."""
    return {name: parts[-1].reshape(n_tokens, -1) for name, parts in capture.items() if parts}


def router(got: dict, ref: dict, W: Weights) -> dict:
    """The router probe of got against ref, both dumps with ffn_moe_logits-il and ffn_moe_topk-il."""
    import numpy as np

    def stack(d, node):
        return np.stack([d[f"{node}-{il}"] for il in range(W.n_layer)])

    bias = np.stack([W.get(f"blk.{il}.exp_probs_b.bias").float().numpy() for il in range(W.n_layer)])
    return probe(stack(got, "ffn_moe_logits"), stack(got, "ffn_moe_topk"), stack(ref, "ffn_moe_logits"),
                 stack(ref, "ffn_moe_topk"), bias)


def masked_router_nmse(p: dict) -> float:
    """The worst layer's router-logit NMSE over the tokens whose experts agree in every earlier layer."""
    import numpy as np

    ok, worst = np.ones(p["set_same"].shape[1], dtype=bool), 0.0
    for il in range(len(p["logit_nmse"])):
        if ok.any():
            worst = max(worst, nmse(p["ref_logits"][il][ok], p["got_logits"][il][ok]))
        ok &= p["set_same"][il]
    return worst


def emit(lines: list[str]) -> None:
    for line in lines:
        print("INFO " + line)


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


def node_lines(got: dict, ref: dict, W: Weights) -> list[tuple[float, str]]:
    """The PLAN Phase 6 nodes, each as (worst NMSE, text): the token embeddings; in the first sliding
    and the first full layer, Q and K after the QK norm, the attention output before (attn_out) and
    after (attn_post_norm) its sandwich norm, and the routed (ffn_moe_out) and shared (ffn_shexp) expert
    output; and the worst layer of the routed and shared expert output.

    A node counts only the tokens whose experts agree in every layer it depends on: the earlier
    layers for Q, K, the attention output and the shared expert, which a layer computes before its
    router, and this layer too for the routed expert output. NaN if there are none; the worst layer
    skips such layers."""
    import numpy as np

    def worst(*vals) -> float:
        return float(np.max(vals))  # NaN if any is NaN

    def err(node: str, il: int, mask) -> float:
        return nmse(ref[f"{node}-{il}"][mask], got[f"{node}-{il}"][mask]) if mask.any() else float("nan")

    v = nmse(ref[EMBD], got[EMBD])
    lines = [(v, f"token embeddings (embd): NMSE {v:.2e}")]
    first = {"sliding": next(il for il in range(W.n_layer) if W.is_swa[il]),
             "full": next(il for il in range(W.n_layer) if not W.is_swa[il])}
    n = len(got[EMBD])
    before = np.ones(n, dtype=bool)  # the same experts in every earlier layer
    top = {"ffn_moe_out": (float("nan"), -1), "ffn_shexp": (float("nan"), -1)}
    for il in range(W.n_layer):
        through = before & same_experts(got[f"ffn_moe_topk-{il}"], ref[f"ffn_moe_topk-{il}"])
        e = {node: err(node, il, before) for node in ("Qcur_normed", "Kcur_normed", "attn_out", "attn_post_norm",
                                                      "ffn_shexp")}
        e["ffn_moe_out"] = err("ffn_moe_out", il, through)
        for node, (val, _) in top.items():
            if not e[node] <= val and not np.isnan(e[node]):  # the larger, or the first layer
                top[node] = (e[node], il)
        for kind, fil in first.items():
            if il != fil:
                continue
            nb, nt = int(before.sum()), int(through.sum())
            lines.append((worst(e["Qcur_normed"], e["Kcur_normed"], e["attn_out"], e["attn_post_norm"]),
                          f"layer {il} ({kind} attention), {nb} of {n} tokens on the same experts before it: NMSE "
                          f"Q after QK norm {e['Qcur_normed']:.2e}, K after QK norm {e['Kcur_normed']:.2e}, "
                          f"attention output attn_out {e['attn_out']:.2e}, attn_post_norm {e['attn_post_norm']:.2e}"))
            lines.append((worst(e["ffn_moe_out"], e["ffn_shexp"]),
                          f"layer {il} ({kind} attention): NMSE routed expert output ffn_moe_out "
                          f"{e['ffn_moe_out']:.2e} ({nt} tokens on the same experts up to this layer), shared "
                          f"expert output ffn_shexp {e['ffn_shexp']:.2e} ({nb} tokens, before it)"))
        before = through
    (moe, moe_il), (sh, sh_il) = top["ffn_moe_out"], top["ffn_shexp"]
    lines.append((worst(moe, sh), f"worst layer with tokens on the same experts: NMSE routed expert output "
                                  f"ffn_moe_out {moe:.2e} (layer {moe_il}), shared expert output ffn_shexp "
                                  f"{sh:.2e} (layer {sh_il})"))
    return lines


def print_table(rows: list[tuple], what: str) -> None:
    print(f"INFO {what}: layer, NMSE " + ", ".join(NODES) + ", same experts")
    for il, *vals, same in rows:
        print(f"INFO   {il:2d}  " + "  ".join(f"{v:.2e}" for v in vals) + f"  {same:6.1%}")


def log_softmax64(x):
    import numpy as np

    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def logit_stats(base, test) -> dict:
    """Per position NMSE of the logits, KL(base || test) of the next-token distributions and
    whether the top token is the same, over all rows."""
    import numpy as np

    lb, lt = log_softmax64(base), log_softmax64(test)
    return {
        "nmse": nmse(base, test),
        "kld": float((np.exp(lb) * (lb - lt)).sum(axis=1).mean()),
        "same_top": float((lb.argmax(1) == lt.argmax(1)).mean()),
    }


def stats_text(s: dict) -> str:
    return f"logits NMSE {s['nmse']:.2e}, KLD {s['kld']:.6f}, same top token {s['same_top']:.1%}"


def chunk_stats(base, test, tokens: list[int], first: int) -> dict:
    """llama-perplexity's scoring of one chunk: positions first .. n-2 predict the next token.
    Adds logit_stats over the same positions."""
    import numpy as np

    lb, lt = log_softmax64(base[first:-1]), log_softmax64(test[first:-1])
    nxt = np.asarray(tokens[first + 1:])
    idx = np.arange(len(nxt))
    return {
        "ppl_base": float(np.exp(-lb[idx, nxt].mean())),
        "ppl_test": float(np.exp(-lt[idx, nxt].mean())),
        **logit_stats(base[first:-1], test[first:-1]),
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
            for v, text in node_lines(got, ref, W):
                report(v <= MAX_NMSE_TINY, f"tiny, {label}, {text} (bound {MAX_NMSE_TINY:g})")
            p = router(got, ref, W)
            worst = masked_router_nmse(p)
            report(worst <= MAX_NMSE_TINY and p["set_same"].mean() >= MIN_SAME_EXPERTS_TINY,
                   f"tiny, {label}, router probe: worst layer router-logit NMSE {worst:.2e} where every earlier "
                   f"layer picks the same experts (bound {MAX_NMSE_TINY:g}), Top-1 same {p['top1_same'].mean():.1%}, "
                   f"Top-{W.n_expert_used} set same {p['set_same'].mean():.1%}")
        W.n_expert_used = converted
    return errs


def run_real(runner: Runner, llama_cpp: Path, gguf: Path, kld_base: Path, metal: bool, probe_out: Path | None,
             threads: int) -> None:
    import numpy as np
    from tokenizers import Tokenizer

    sys.path.insert(0, str(ROOT / "tools" / "tokenizer"))
    from common import tokenizer_dir

    tok = Tokenizer.from_file(str(tokenizer_dir() / "tokenizer.json"))
    W = Weights(gguf, llama_cpp)
    devs = runner.devices()
    _, cpu = devs[0]
    ctx = dict(n_ctx=1024, n_threads=threads, n_threads_batch=threads)
    if metal and len(devs) < 2:
        raise SystemExit("--metal: no GPU device")
    probes = {}

    def both(tokens: list[int], what: str):
        t0 = time.time()
        capture = capture_names(W.n_layer)
        got_logits = runner.logits(gguf, cpu, tokens, len(tokens), capture=capture, **ctx)
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

    def margins(p: dict) -> None:
        g = p["margin"].ravel()
        spread = (p["ref_logits"] + p["bias"][:, None, :]).std(axis=-1)
        print(f"INFO router margin (6th minus 7th of logits + bias, reference): median {np.median(g):.3f}, "
              f"below 0.01 for {(g < 0.01).mean():.1%}, below 0.1 for {(g < 0.1).mean():.1%} of (token, layer) "
              f"pairs; median score spread {np.median(spread):.2f}")

    def report_router(p: dict, name: str, what: str, per_layer: bool = True) -> None:
        if per_layer:
            emit(table(p, what))
        emit(summary(p, what))
        probes[name] = p

    # German prompt
    ids = tok.encode(GERMAN, add_special_tokens=False).ids
    got_logits, got, ref_logits, ref = both(ids, f"prompt {GERMAN!r} {ids}")
    print_table(layer_table(got, ref, W.n_layer), "prompt")
    emit(f"prompt, libllama CPU vs reference float32, {text}" for _, text in node_lines(got, ref, W))
    maxima(ref, got)
    print(f"INFO prompt, libllama vs reference float32 at all {len(ids)} positions: "
          + stats_text(logit_stats(ref_logits, got_logits)))
    report_router(router(got, ref, W), "prompt-cpu", "prompt, libllama CPU vs reference float32", per_layer=False)
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
    forced = runner.logits(gguf, cpu, path, len(path), **ctx)
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
    emit(f"chunk 1, libllama CPU vs reference float32, {text}" for _, text in node_lines(got, ref, W))
    maxima(ref, got)
    p = router(got, ref, W)
    margins(p)
    report_router(p, "cpu", "chunk 1, libllama CPU vs reference float32")
    first = n_ctx // 2

    def chunk_line(base, test, what: str) -> None:
        s = chunk_stats(base, test, tokens, first)
        print(f"INFO chunk 1, {what}: PPL {s['ppl_test']:.4f} vs {s['ppl_base']:.4f}, " + stats_text(s))

    chunk_line(ref_logits, got_logits, "libllama vs reference float32")
    t0 = time.time()
    ref16d = {}
    ref16 = forward(W, tokens, "bfloat16", ref16d).numpy()
    chunk_line(ref_logits, ref16, f"reference bfloat16 vs float32 ({time.time() - t0:.0f} s)")
    emit(f"chunk 1, reference bfloat16 vs float32, {text}" for _, text in node_lines(ref16d, ref, W))
    report_router(router(ref16d, ref, W), "bf16", "chunk 1, reference bfloat16 vs float32")
    del ref16d

    if metal:
        name, gpu = devs[1]
        t0 = time.time()
        capture = capture_names(W.n_layer)
        mtl_logits = runner.logits(gguf, gpu, tokens, len(tokens), capture=capture, cpu_moe=True, **ctx)
        mtl = flat(capture, len(tokens))
        print(f"INFO chunk 1 on {name}, routed experts on the CPU: libllama {time.time() - t0:.0f} s")
        chunk_line(ref_logits, mtl_logits, f"libllama {name} vs reference float32")
        report_router(router(mtl, ref, W), "metal", f"chunk 1, libllama {name} vs reference float32")
        chunk_line(got_logits, mtl_logits, f"libllama {name} vs libllama CPU")
        report_router(router(mtl, got, W), "metal-vs-cpu", f"chunk 1, libllama {name} vs libllama CPU")

    if probe_out:
        probe_out.mkdir(parents=True, exist_ok=True)
        for name, p in probes.items():
            save(probe_out / f"router-{name}.npz", p)
        print(f"INFO router probes written to {probe_out}: " + ", ".join(f"router-{n}.npz" for n in probes))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tiny", action="store_true", help="validate the reference on the tiny fixture")
    mode.add_argument("--gguf", type=Path, help="GGUF of the real checkpoint (BF16)")
    ap.add_argument("--kld-base", type=Path, help="llama-perplexity --kl-divergence-base file (with --gguf)")
    ap.add_argument("--metal", action="store_true", help="also run the chunk on the GPU, experts on the CPU (with --gguf)")
    ap.add_argument("--probe-out", type=Path, help="directory for the router probes as .npz (with --gguf)")
    ap.add_argument("--threads", type=int, default=10, help="libllama CPU threads (with --gguf)")
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
    run_real(runner, args.llama_cpp, args.gguf, args.kld_base, args.metal, args.probe_out, args.threads)


if __name__ == "__main__":
    main()
