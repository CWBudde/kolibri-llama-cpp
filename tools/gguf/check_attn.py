#!/usr/bin/env python3
"""Check the Kolibri hybrid attention in libllama against the reference.

The reference is Kolibri1Attention of Aleph-Alpha/aleph-alpha-inference at the
pinned commit (aleph_alpha_inference/kolibri1.py:81-121):

- sliding-window layers pass config.sliding_window to vLLM's Attention and use
  get_rope(head_dim, rope_parameters): NeoX style, all head dimensions rotated;
- full-attention layers pass sliding_window=None and have no rotary embedding;
- q and k are RMS-normalized per head before RoPE, the scale is head_dim**-0.5.

vLLM v0.29.0 hands the window to FlashAttention as (sliding_window - 1, 0)
(vllm/v1/attention/backends/flash_attn.py:862): query i sees key j for
0 <= i - j <= sliding_window - 1, so with 513 the 512 preceding tokens and
itself. The RoPE below is a port of vLLM's RotaryEmbedding forward_static and
ApplyRotaryEmb.forward_static.

The fixture is cmd/kolibri-tiny -attn: 48 query heads, 4 KV heads, head_dim
128 and sliding_window 513, as the real model, on 6 layers (4 sliding, 2 full).
It is converted to F32. 1100 tokens (more than two windows) are decoded in
several ways on every device: as one batch; in chunks of 64 that read the KV
cache of the earlier chunks, with a SWA cache of 768 cells (so cells are
evicted) and with a full-size one; and one token at a time across positions
511 to 513. cb_eval captures each layer's attention nodes, and every step is
recomputed in float64 from libllama's own input to that step:

- attn_norm from the layer input (the embedding row, or the previous l_out);
- Qcur/Kcur_normed (per-head RMSNorm) and Vcur from attn_norm;
- Qcur/Kcur_rope from Qcur/Kcur_normed, on the sliding layers; the full layers
  must have no RoPE node;
- kqv_out, the attention, from libllama's own q, k and v, with the window mask
  on the sliding layers and the causal mask on the full layers;
- attn_out (Wo), attn_post_norm and ffn_inp (the residual add).

The window check: from the first position where the masks differ, libllama's
deviation from the reference with window 513 is projected onto the step from
that reference to the one with window 512 (and 514). The coefficient is about
0 for window 513 and about 1 for the other window, whatever the precision of
the backend, so an off-by-one in either direction fails it. (Against the
1e-4 NMSE bound of the F16 runs, an off-by-one has a margin of only 3x.)
The full layers must be much closer to the unrotated reference than to a
rotated one.

    check_attn.py --llama-cpp third_party/llama.cpp
"""

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

from check_model import MAX_NMSE, Runner, joined, nmse
from tiny import convert, generate_tiny

N_TOKENS = 1100
BOUNDARIES = (511, 512, 513, 514)
# How far libllama may move from the window-513 reference toward the window-512/514 one (0: not at all,
# 1: all the way), as a least-squares coefficient over all positions where the masks differ.
MAX_TOWARD = 0.1
# How much further off a rotated reference must be on the full layers than the unrotated one.
MIN_SEPARATION = 100
# The CPU with an F32 KV cache and without flash attention (llama-graph.cpp casts the KV to F16 for
# flash attention) vs a float64 recomputation; set from the first run, see docs/phase5-attention.md.
MAX_NMSE_STRICT = 1e-10
# The RoPE nodes there: ggml_rope_cache_init builds the angles by repeated float32 multiplication
# (observed 6.8e-10 at positions up to 1100; vLLM's own float32 cos/sin cache is off by 2.3e-11).
MAX_NMSE_STRICT_ROPE = 1e-8


def rope_cos_sin_cache(base: float, rotary_dim: int, max_position: int, dtype):
    """vLLM RotaryEmbeddingBase._compute_inv_freq and _compute_cos_sin_cache
    (vllm/model_executor/layers/rotary_embedding/base.py at v0.29.0), in dtype
    instead of float32."""
    import torch

    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=dtype) / rotary_dim))
    t = torch.arange(max_position, dtype=dtype)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.cat((cos, sin), dim=-1)


def apply_rotary_emb(x, cos, sin, is_neox_style: bool = True):
    """vLLM ApplyRotaryEmb.forward_static (rotary_embedding/common.py at v0.29.0)."""
    import torch

    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def rope(positions, x, cos_sin_cache):
    """vLLM RotaryEmbedding.forward_static for one of query or key, with rotary_dim = head_size."""
    cos, sin = cos_sin_cache.index_select(0, positions.flatten()).chunk(2, dim=-1)
    return apply_rotary_emb(x, cos, sin, True)


def attention(q, k, v, window: int | None):
    """softmax(q k^T / sqrt(d)) v per head, q [n, n_head, d], k and v [n, n_head_kv, d].
    Query i sees key j for 0 <= i - j <= window - 1 (FlashAttention window (window - 1, 0)),
    or every j <= i without a window. Returns [n, n_head * d]."""
    import torch

    n, n_head, d = q.shape
    group = n_head // k.shape[1]
    i = torch.arange(n)[:, None]
    j = torch.arange(n)[None, :]
    allowed = (i - j >= 0) & (i - j <= window - 1) if window else i - j >= 0
    out = torch.empty(n, n_head, d, dtype=q.dtype)
    for h in range(k.shape[1]):  # one KV head and its query heads at a time
        qs = q[:, h * group:(h + 1) * group].transpose(0, 1)  # [group, n, d]
        s = (qs @ k[:, h].T * d ** -0.5).masked_fill(~allowed, float("-inf"))
        out[:, h * group:(h + 1) * group] = (s.softmax(-1) @ v[:, h]).transpose(0, 1)
    return out.reshape(n, n_head * d)


def per_pos(ref, got):
    """NMSE per position (row)."""
    return ((ref - got) ** 2).sum(-1) / (ref ** 2).sum(-1)


def check(runner: Runner, gguf: Path, dev, model: Path, tokens: list[int], strict: bool,
          run: dict, overrides: dict | None = None):
    """Decodes tokens on dev as run describes and compares every layer's attention with the
    reference, against the strict bounds or MAX_NMSE. Returns (ok, message) pairs."""
    import torch
    from safetensors.torch import load_file

    cfg = json.loads((model / "config.json").read_text())
    n_layer, eps, window = cfg["num_hidden_layers"], cfg["rms_norm_eps"], cfg["sliding_window"]
    n_head, n_head_kv, d = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    full = [t == "full_attention" for t in cfg["layer_types"]]
    w = {k: v.double() for k, v in load_file(model / "model.safetensors").items()}
    cache = rope_cos_sin_cache(cfg["rope_theta"], d, N_TOKENS, torch.float64)

    heads = ["Qcur_normed", "Kcur_normed", "Vcur", "Qcur_rope", "Kcur_rope"]
    acts = ["attn_norm", "kqv_out", "attn_out", "attn_post_norm", "ffn_inp", "l_out"]
    cap = {f"{s}-{il}": None for il in range(n_layer) for s in heads + acts
           if not (full[il] and s.endswith("_rope"))}
    nodes: dict[str, list[str]] = {}
    runner.logits(gguf, dev, tokens, run["n_ubatch"], overrides, nodes=nodes, capture=cap, n_ctx=2048,
                  chunks=run["chunks"], **run["ctx"])
    missing = [k for k, v in cap.items() if not v]
    if missing:
        return [(False, f"nodes not captured: {missing[:4]}")]
    got = {}
    for k, parts in cap.items():
        is_head = k.rsplit("-", 1)[0] in heads
        got[k] = torch.from_numpy(joined(parts, -3 if is_head else -2)).double().reshape(
            (N_TOKENS, -1, d) if is_head else (N_TOKENS, -1))

    rope_nodes = sorted({int(n[len("Qcur_rope-"):].split()[0]) for n in nodes if n.startswith("Qcur_rope-")})
    want_rope = [il for il in range(n_layer) if not full[il]]
    res = [(rope_nodes == want_rope, f"RoPE nodes in layers {rope_nodes} (sliding layers {want_rope})")]

    def rms_norm(x, weight):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

    pos = torch.arange(N_TOKENS)
    err = {s: 0.0 for s in ["attn_norm", "Qcur_normed", "Kcur_normed", "Vcur", "Qcur_rope", "Kcur_rope",
                            "kqv_out", "attn_out", "attn_post_norm", "ffn_inp"]}
    at = {b: 0.0 for b in BOUNDARIES}  # kqv_out NMSE at single positions, sliding layers
    toward = {512: [], 514: []}  # per sliding layer: the coefficient of the step from window 513 to w
    rotated, unrotated = [], []  # per full layer: NMSE against a rotated and the unrotated reference
    for il in range(n_layer):
        p = f"model.layers.{il}."
        g = {s: got[f"{s}-{il}"] for s in heads + acts if f"{s}-{il}" in got}
        x = w["model.embed_tokens.weight"][tokens] if il == 0 else got[f"l_out-{il - 1}"]
        h = g["attn_norm"]
        err["attn_norm"] = max(err["attn_norm"], nmse(rms_norm(x, w[p + "input_layernorm.weight"]), h))
        q = (h @ w[p + "self_attn.q_proj.weight"].T).reshape(N_TOKENS, n_head, d)
        k = (h @ w[p + "self_attn.k_proj.weight"].T).reshape(N_TOKENS, n_head_kv, d)
        v = (h @ w[p + "self_attn.v_proj.weight"].T).reshape(N_TOKENS, n_head_kv, d)
        for s, ref in (("Qcur_normed", rms_norm(q, w[p + "self_attn.q_norm.weight"])),
                       ("Kcur_normed", rms_norm(k, w[p + "self_attn.k_norm.weight"])), ("Vcur", v)):
            err[s] = max(err[s], nmse(ref, g[s]))

        if full[il]:
            qa, ka = g["Qcur_normed"], g["Kcur_normed"]
            ref = attention(qa, ka, g["Vcur"], None)
            rot = attention(rope(pos, qa, cache), rope(pos, ka, cache), g["Vcur"], None)
            rotated.append(nmse(rot, g["kqv_out"]))
            unrotated.append(nmse(ref, g["kqv_out"]))
        else:
            for s, src in (("Qcur_rope", "Qcur_normed"), ("Kcur_rope", "Kcur_normed")):
                err[s] = max(err[s], nmse(rope(pos, g[src], cache), g[s]))
            qa, ka = g["Qcur_rope"], g["Kcur_rope"]
            ref = attention(qa, ka, g["Vcur"], window)
            e = per_pos(ref, g["kqv_out"])
            for b in BOUNDARIES:
                at[b] = max(at[b], float(e[b]))
            for wo in toward:
                first = min(wo, window)  # the first position where the two masks differ
                step = (attention(qa, ka, g["Vcur"], wo) - ref)[first:]
                toward[wo].append(float(((g["kqv_out"] - ref)[first:] * step).sum() / (step ** 2).sum()))
        err["kqv_out"] = max(err["kqv_out"], nmse(ref, g["kqv_out"]))
        a = ref @ w[p + "self_attn.o_proj.weight"].T
        err["attn_out"] = max(err["attn_out"], nmse(a, g["attn_out"]))
        post = rms_norm(g["attn_out"], w[p + "post_attn_norm.weight"])
        err["attn_post_norm"] = max(err["attn_post_norm"], nmse(post, g["attn_post_norm"]))
        err["ffn_inp"] = max(err["ffn_inp"], nmse(g["attn_post_norm"] + x, g["ffn_inp"]))

    max_nmse = MAX_NMSE_STRICT if strict else MAX_NMSE
    bound = {s: MAX_NMSE_STRICT_ROPE if strict and s.endswith("_rope") else max_nmse for s in err}
    res += [(v <= bound[s], f"{s} vs reference: max NMSE over layers {v:.1e} (<= {bound[s]:g})")
            for s, v in err.items()]
    res.append((max(at.values()) <= max_nmse, "kqv_out at positions " + ", ".join(
        f"{b}: {v:.1e}" for b, v in at.items()) + f" (sliding layers, <= {max_nmse:g})"))
    for wo, c in toward.items():
        worst = max(c, key=abs)
        res.append((abs(worst) <= MAX_TOWARD, f"window {window}, not {wo}: from position {min(wo, window)} on, "
                    f"libllama moves {worst:+.1e} of the way to the window-{wo} reference (|c| <= {MAX_TOWARD})"))
    res.append((min(rotated) >= MIN_SEPARATION * max(max(unrotated), 1e-300),
                f"no RoPE on the full layers: a rotated reference is off by NMSE {min(rotated):.1e}, "
                f"the unrotated one by {max(unrotated):.1e} (>= {MIN_SEPARATION}x)"))
    return res


RUNS = [
    # decode calls, ubatch size, extra llama_context_params
    {"name": "batch", "chunks": [N_TOKENS], "n_ubatch": N_TOKENS, "ctx": {}},
    {"name": "chunks of 64, SWA cache 768", "chunks": [64] * 17 + [12], "n_ubatch": 64,
     "ctx": {"swa_full": False}},
    {"name": "chunks of 64, full SWA cache", "chunks": [64] * 17 + [12], "n_ubatch": 64,
     "ctx": {"swa_full": True}},
    {"name": "single tokens 500-529", "chunks": [500] + [1] * 30 + [570], "n_ubatch": 570,
     "ctx": {"swa_full": False}},
]


def configs(lib, is_cpu: bool):
    """(run, label, strict) for every run, flash attention off and on, and KV cache type. The CPU
    also runs with an F32 KV cache, against the strict bounds; with flash attention, llama-graph.cpp
    casts an F32 KV cache to F16, so that combination equals the F16 one."""
    for run in RUNS:
        for fa in (lib.LLAMA_FLASH_ATTN_TYPE_DISABLED, lib.LLAMA_FLASH_ATTN_TYPE_ENABLED):
            on = fa == lib.LLAMA_FLASH_ATTN_TYPE_ENABLED
            for kv in (lib.GGML_TYPE_F32, lib.GGML_TYPE_F16) if is_cpu and not on else (lib.GGML_TYPE_F16,):
                f32 = kv == lib.GGML_TYPE_F32
                label = f"{run['name']}, flash attn {'on' if on else 'off'}, KV {'f32' if f32 else 'f16'}"
                yield dict(run, ctx=dict(run["ctx"], flash_attn_type=fa, type_k=kv, type_v=kv)), label, f32


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True, help="llama.cpp checkout with a built libllama")
    ap.add_argument("--seed", type=int, default=1, help="cmd/kolibri-tiny random seed")
    args = ap.parse_args()

    runner = Runner(args.llama_cpp)
    lib = runner.lib
    tokens = random.Random(args.seed).choices(range(127900), k=N_TOKENS)  # base vocab, no special tokens
    errs = 0
    with tempfile.TemporaryDirectory() as tmp:
        model = generate_tiny(Path(tmp), args.seed, "attn")
        gguf = convert(model, args.llama_cpp, "f32")
        for name, dev in runner.devices():
            is_cpu = lib.ggml_backend_dev_type(dev) == lib.GGML_BACKEND_DEVICE_TYPE_CPU
            for run, label, strict in configs(lib, is_cpu):
                try:
                    results = check(runner, gguf, dev, model, tokens, strict, run)
                except RuntimeError as e:
                    results = [(False, f"load and decode: {e}")]
                for ok, what in results:
                    print(f"{'PASS' if ok else 'FAIL'} {name} [{label}] {what}", flush=True)
                    errs += not ok
    if errs:
        print(f"FAIL {errs} problem(s)")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
