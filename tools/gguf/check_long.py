#!/usr/bin/env python3
"""Check the Kolibri hybrid attention in libllama at long contexts.

check_attn.py covers 1100 tokens. This decodes 8192, 16384 and 65536 tokens,
and 262144 (the real max_position_embeddings) on a GPU with flash attention,
on the cmd/kolibri-tiny -attn fixture (48 query heads, 4 KV heads, head_dim
128, sliding_window 513, layers SSSSFF), converted to F32, with the GGUF
context length set to the real 262144. The tokens are decoded as one sequence
in ubatches that read the KV cache of the earlier ones, with a SWA cache of
window + ubatch cells.

A float64 reference over every position costs O(n^2), so the comparison runs
on checked query rows: the last 32 positions, 32 random ones from the second
half, and the two on either side of the ubatch boundary at n/2. cb_eval keeps
only what those rows need (Runner rows): q at the checked rows, k and v at
every position on the full layers and within the window before a checked row
on the sliding layers.

- KV cache: llama_kv_cache_iswa with n cells for the full layers, the sliding
  layers in the SWA cache;
- kqv_out at the checked rows against the attention recomputed in float64
  from libllama's own q, k and v (all earlier keys on the full layers, the
  513-token window on the sliding layers), against the bounds of
  check_attn.py: the RoPE angles do not enter it;
- Qcur/Kcur_rope at the checked rows against RoPE in float64 of libllama's own
  Qcur/Kcur_normed. At large positions float32 angles lose precision: ggml
  builds them by repeated multiplication (ggml_rope_cache_init), vLLM's
  float32 cos/sin cache by one multiplication per angle. libllama may be off
  by at most MAX_ROPE_VS_VLLM times the error of vLLM's float32 cache (the
  check_attn.py port, applied in float64 so only the cache differs) at the
  same positions.

    check_long.py --llama-cpp third_party/llama.cpp
"""

import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

from check_attn import MAX_NMSE_STRICT, kv_caches, rope, rope_cos_sin_cache
from check_model import MAX_NMSE, Runner, joined, nmse
from tiny import convert, generate_tiny

CONTEXT_LENGTH = 262144  # the real max_position_embeddings
LENGTHS = (8192, 16384, 65536)
# How much further off than vLLM's own float32 cos/sin cache libllama's RoPE may be, both against
# float64; fixed before the first run (the ratio is about 30 at 1100 tokens, see docs/phase5-attention.md).
MAX_ROPE_VS_VLLM = 100


def checked_rows(n: int, rng: random.Random) -> list[int]:
    """The last 32 positions, 32 random ones from [n/2, n - 32), and the two around n/2."""
    return sorted(set(range(n - 32, n)) | set(rng.sample(range(n // 2, n - 32), 32)) | {n // 2 - 1, n // 2})


def check(runner: Runner, gguf: Path, dev, model: Path, tokens: list[int], rows: list[int], strict: bool,
          run: dict):
    """Decodes tokens on dev as run describes and compares the attention at rows with the reference.
    Returns (ok, message) pairs."""
    import torch

    cfg = json.loads((model / "config.json").read_text())
    n_layer, window = cfg["num_hidden_layers"], cfg["sliding_window"]
    n_head, n_head_kv, d = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    full = [t == "full_attention" for t in cfg["layer_types"]]
    n = len(tokens)
    near = sorted({j for r in rows for j in range(max(0, r - window + 1), r + 1)})  # keys of the sliding rows

    want = {}  # captured name -> positions kept (None: all)
    for il in range(n_layer):
        want |= {f"Qcur_normed-{il}": rows, f"kqv_out-{il}": rows}
        if full[il]:
            want |= {f"Kcur_normed-{il}": None, f"Vcur-{il}": None}
        else:
            want |= {f"Qcur_rope-{il}": rows, f"Kcur_normed-{il}": rows, f"Kcur_rope-{il}": near,
                     f"Vcur-{il}": near}
    axis = {s: -2 if s.startswith("kqv_out") else -3 for s in want}
    cap = {s: None for s in want}
    t0 = time.monotonic()
    runner.logits(gguf, dev, tokens, run["n_ubatch"], {"kolibri.context_length": CONTEXT_LENGTH},
                  capture=cap, n_ctx=n, chunks=[run["n_ubatch"]] * (n // run["n_ubatch"]), outputs=rows[-1:],
                  rows={s: (axis[s], pos) for s, pos in want.items() if pos is not None}, swa_full=False,
                  **run["ctx"])
    seconds = time.monotonic() - t0
    missing = [k for k, v in cap.items() if not v]
    if missing:
        return [(False, f"nodes not captured: {missing[:4]}")]
    got = {}
    for k, parts in cap.items():
        v = torch.from_numpy(joined(parts, axis[k])).double()
        got[k] = v.reshape(v.shape[-3], -1, d) if axis[k] == -3 else v.reshape(v.shape[-2], -1)

    # the SWA cache holds the window plus one ubatch, padded to 256 cells (src/llama-kv-cache-iswa.cpp:73),
    # so it evicts: a cache of n cells would hide a wrong window behind cells that are never reused
    n_swa = -(-min(n, window + run["n_ubatch"]) // 256) * 256
    layers, cells = kv_caches(runner.log), kv_caches(runner.log, "cells")
    res = [(layers == {"non-SWA": sum(full), "SWA": n_layer - sum(full)} and cells == {"non-SWA": n, "SWA": n_swa},
            f"KV cache layers {layers}, cells {cells} (llama_kv_cache_iswa: {n} cells for the {sum(full)} full "
            f"layers, {n_swa} = window + ubatch, padded, for the {n_layer - sum(full)} sliding layers; "
            f"decoded in {seconds:.0f} s)")]

    pos = torch.tensor(rows)
    cache64 = rope_cos_sin_cache(cfg["rope_theta"], d, n, torch.float64)
    cache32 = rope_cos_sin_cache(cfg["rope_theta"], d, n, torch.float32)
    near_at = {p: i for i, p in enumerate(near)}
    group = n_head // n_head_kv
    err = {"full": 0.0, "sliding": 0.0}
    rope_err = {"Qcur_rope": [0.0, 0.0], "Kcur_rope": [0.0, 0.0]}  # libllama and vLLM float32, vs float64
    for il in range(n_layer):
        q = got[f"Qcur_normed-{il}" if full[il] else f"Qcur_rope-{il}"]
        ref = torch.empty(len(rows), n_head, d, dtype=torch.float64)
        for i, r in enumerate(rows):
            if full[il]:
                k, v = got[f"Kcur_normed-{il}"][: r + 1], got[f"Vcur-{il}"][: r + 1]
            else:
                lo, hi = near_at[max(0, r - window + 1)], near_at[r] + 1
                k, v = got[f"Kcur_rope-{il}"][lo:hi], got[f"Vcur-{il}"][lo:hi]
            for g in range(n_head_kv):  # query head h reads KV head h // group
                hs = slice(g * group, (g + 1) * group)
                ref[i, hs] = (q[i, hs] @ k[:, g].T * d ** -0.5).softmax(-1) @ v[:, g]
        kind = "full" if full[il] else "sliding"
        err[kind] = max(err[kind], nmse(ref.reshape(len(rows), -1), got[f"kqv_out-{il}"]))
        if not full[il]:
            k_roped = got[f"Kcur_rope-{il}"][[near_at[r] for r in rows]]
            for s, normed, roped in (("Qcur_rope", got[f"Qcur_normed-{il}"], got[f"Qcur_rope-{il}"]),
                                     ("Kcur_rope", got[f"Kcur_normed-{il}"], k_roped)):
                exact = rope(pos, normed, cache64)
                rope_err[s][0] = max(rope_err[s][0], nmse(exact, roped))
                rope_err[s][1] = max(rope_err[s][1], nmse(exact, rope(pos, normed, cache32)))

    bound = MAX_NMSE_STRICT if strict else MAX_NMSE
    at = f"{len(rows)} positions in [{rows[0]}, {rows[-1]}]"
    res.append((err["full"] <= bound, f"kqv_out vs reference on the full layers (all earlier keys) at {at}: "
                f"max NMSE {err['full']:.1e} (<= {bound:g})"))
    res.append((err["sliding"] <= bound, f"kqv_out vs reference on the sliding layers (window {window}) at {at}: "
                f"max NMSE {err['sliding']:.1e} (<= {bound:g})"))
    for s, (ours, vllm) in rope_err.items():
        res.append((ours <= MAX_ROPE_VS_VLLM * vllm, f"{s} vs float64 RoPE at {at}: NMSE {ours:.1e}, vLLM's "
                    f"float32 cos/sin cache {vllm:.1e} (<= {MAX_ROPE_VS_VLLM}x)"))
    return res


def configs(lib, is_cpu: bool):
    """(n_tokens, run, label, strict). Without flash attention, the KQ matrix (n_kv x n_ubatch x
    48 heads, float32) keeps the ubatch at 256; the CPU runs that with an F32 KV cache against the
    strict bounds, up to 16384 tokens (65536 take 620 s, see docs/phase5-attention.md). 262144
    tokens run only on a GPU with flash attention."""
    off, on = lib.LLAMA_FLASH_ATTN_TYPE_DISABLED, lib.LLAMA_FLASH_ATTN_TYPE_ENABLED
    kv_off = lib.GGML_TYPE_F32 if is_cpu else lib.GGML_TYPE_F16
    cells = [(n, off, kv_off, 256) for n in LENGTHS if not (is_cpu and n > 16384)]
    cells += [(n, on, lib.GGML_TYPE_F16, 512) for n in LENGTHS]
    if not is_cpu:
        cells.append((CONTEXT_LENGTH, on, lib.GGML_TYPE_F16, 512))
    threads = os.cpu_count()  # the libllama default of 4 threads doubles the CPU time
    for n, fa, kv, n_ubatch in cells:
        f32 = kv == lib.GGML_TYPE_F32
        label = f"{n} tokens, ubatch {n_ubatch}, flash attn {'on' if fa == on else 'off'}, KV {'f32' if f32 else 'f16'}"
        yield n, {"n_ubatch": n_ubatch, "ctx": {"flash_attn_type": fa, "type_k": kv, "type_v": kv,
                                                "n_threads": threads, "n_threads_batch": threads}}, label, f32


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True, help="llama.cpp checkout with a built libllama")
    ap.add_argument("--seed", type=int, default=1, help="cmd/kolibri-tiny random seed")
    args = ap.parse_args()

    runner = Runner(args.llama_cpp)
    lib = runner.lib
    rng = random.Random(args.seed)
    tokens = rng.choices(range(127900), k=CONTEXT_LENGTH)  # base vocab, no special tokens
    rows = {n: checked_rows(n, rng) for n in LENGTHS + (CONTEXT_LENGTH,)}
    errs = 0
    with tempfile.TemporaryDirectory() as tmp:
        model = generate_tiny(Path(tmp), args.seed, "attn")
        gguf = convert(model, args.llama_cpp, "f32")
        for name, dev in runner.devices():
            is_cpu = lib.ggml_backend_dev_type(dev) == lib.GGML_BACKEND_DEVICE_TYPE_CPU
            for n, run, label, strict in configs(lib, is_cpu):
                try:
                    results = check(runner, gguf, dev, model, tokens[:n], rows[n], strict, run)
                except RuntimeError as e:
                    results = [(False, f"load and decode: {e}")]
                for ok, what in results:
                    print(f"{'PASS' if ok else 'FAIL'} {name} attn [{label}] {what}", flush=True)
                    errs += not ok
    if errs:
        print(f"FAIL {errs} problem(s)")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
