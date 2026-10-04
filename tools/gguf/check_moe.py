#!/usr/bin/env python3
"""Check the Kolibri router and MoE block in libllama against the reference.

The reference is the routing function of Aleph-Alpha/aleph-alpha-inference
at the pinned commit, ported verbatim below, and the block order of its
Kolibri1DecoderLayer: post_ffn_norm(shared_experts(h) + experts(h)), then
the residual add.

The fixture is cmd/kolibri-tiny -router, the shape and magnitudes of the
reference's test_routing_semantics: 384 experts, top 6, router logits with
standard deviation about 3 and a correction bias with standard deviation 5.
It is converted to F32, so the weights in libllama equal the safetensors
values. 100 tokens are decoded on every device; cb_eval captures each
layer's MoE nodes, and every step is recomputed in float64 from libllama's
own input to that step:

- router logits: ffn_moe_logits = ffn_norm @ W_gate^T;
- Top-6 selection: ffn_moe_topk has expert_used_count = 6 columns and holds
  exactly the experts of top6(logits + bias), with the logits libllama
  computed (a recomputed logit can flip a near-tie, the selection on the same
  logits cannot);
- non-vacuity, as in test_routing_semantics: top6(sigmoid(logits) + bias)
  selects other experts for some tokens in every layer, so the selection
  check tells the two bias placements apart;
- weights: ffn_moe_weights = sigmoid(logits[selected]), and the experts are
  multiplied with exactly that node (no renormalization, no scale);
- routed output (ffn_moe_out), shared expert (ffn_shexp), their sum
  (ffn_out), the post-FFN norm (ffn_post_norm) and the residual add (l_out),
  plus the pre-FFN norm (ffn_norm from ffn_inp).

    check_moe.py --llama-cpp third_party/llama.cpp
"""

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

from check_model import MAX_NMSE, Runner, joined, nmse
from tiny import convert, generate_tiny

N_TOKENS = 100
# The CPU's F32 graph vs a float64 recomputation (observed: at most 3.3e-14).
MAX_NMSE_CPU = 1e-10
# Other devices get the backend tolerance of test-llama-archs (MAX_NMSE, 1e-4): Metal's
# kernel_mul_mm_f32_f32 multiplies F32 matrices in half-precision tiles (observed: up to 1.3e-6).
MAX_WEIGHT_ERR = 1e-6


def sigmoid_logit_add_routing(hidden_states, gating_output, topk, renormalize, e_score_correction_bias):
    """Verbatim from aleph_alpha_inference/kolibri1.py at 049a6a7bd2405b27d6d280d256bd3d585191c7ae
    (without its torch.compile decorator)."""
    import torch

    logits = gating_output.float()
    topk_ids = torch.topk(logits + e_score_correction_bias, k=topk, dim=-1)[1]
    topk_weights = torch.sigmoid(logits.gather(1, topk_ids))
    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    return topk_weights, topk_ids.to(torch.int32)


def check(runner: Runner, gguf: Path, dev, model: Path, tokens: list[int], max_nmse: float,
          overrides: dict | None = None):
    """Decodes tokens on dev and compares every layer's MoE block with the reference.
    Returns (ok, message) pairs."""
    import torch
    from safetensors.torch import load_file

    cfg = json.loads((model / "config.json").read_text())
    n_layer, top_k, eps = cfg["num_hidden_layers"], cfg["num_experts_per_tok"], cfg["rms_norm_eps"]
    w = {k: v.double() for k, v in load_file(model / "model.safetensors").items()}

    steps = ["ffn_inp", "ffn_norm", "ffn_moe_logits", "ffn_moe_topk", "ffn_moe_weights",
             "ffn_moe_out", "ffn_shexp", "ffn_out", "ffn_post_norm", "l_out"]
    cap = {f"{s}-{il}": None for il in range(n_layer) for s in steps}
    nodes: dict[str, list[str]] = {}
    runner.logits(gguf, dev, tokens, N_TOKENS, overrides, nodes=nodes, capture=cap)
    missing = [k for k, v in cap.items() if not v]
    if missing:
        return [(False, f"nodes not captured: {missing[:4]}")]
    # [n_tokens, n] tensors
    got = {k: torch.from_numpy(joined(v).reshape(N_TOKENS, -1)) for k, v in cap.items()}
    shapes = {tuple(got[f"ffn_moe_topk-{il}"].shape) for il in range(n_layer)}
    top_ok = (shapes == {(N_TOKENS, top_k)}, f"Top-{top_k}: ffn_moe_topk is {sorted(shapes)} (n_tokens, n_expert_used)")
    if not top_ok[0]:
        return [top_ok]

    def rms_norm(x, weight):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

    def mlp(x, gate, up, down):
        return (torch.nn.functional.silu(x @ gate.T) * (x @ up.T)) @ down.T

    err = {s: 0.0 for s in steps if s not in ("ffn_inp", "ffn_moe_topk", "ffn_moe_weights")}
    sel_equal, differ, applied, w_err = 0, [], [], 0.0
    for il in range(n_layer):
        p = f"model.layers.{il}."
        g = {s: got[f"{s}-{il}"] for s in steps}
        inp, h = g["ffn_inp"].double(), g["ffn_norm"].double()
        err["ffn_norm"] = max(err["ffn_norm"], nmse(rms_norm(inp, w[p + "post_attention_layernorm.weight"]), h))

        logits = g["ffn_moe_logits"]  # float32, as libllama computed them
        err["ffn_moe_logits"] = max(err["ffn_moe_logits"], nmse(h @ w[p + "mlp.gate.weight"].T, logits.double()))

        bias = w[p + "moe.router.expert_bias"].float()
        ref_w, ref_ids = sigmoid_logit_add_routing(h, logits, top_k, cfg["norm_topk_prob"], bias)
        ref_ids = ref_ids.long()
        ids = g["ffn_moe_topk"].long()
        sel_equal += int((ids.sort(-1)[0] == ref_ids.sort(-1)[0]).all(-1).sum())
        score_add = torch.topk(torch.sigmoid(logits) + bias, k=top_k, dim=-1)[1]
        differ.append(int((score_add.sort(-1)[0] != ref_ids.sort(-1)[0]).any(-1).sum()))

        # weights per expert id, so the order within the top k does not matter
        mine = torch.zeros(N_TOKENS, bias.numel()).scatter(1, ids, g["ffn_moe_weights"])
        ref = torch.zeros(N_TOKENS, bias.numel()).scatter(1, ref_ids, ref_w)
        w_err = max(w_err, float((mine - ref).abs().max()))
        applied.append((nodes.get(f"ffn_moe_weighted-{il}") or [None, None])[1])

        # routed experts from the reference selection, in float64
        routed = torch.zeros_like(h)
        for k in range(top_k):
            e = ref_ids[:, k].tolist()
            gate = torch.stack([w[f"{p}mlp.experts.{x}.gate_proj.weight"] for x in e])
            up = torch.stack([w[f"{p}mlp.experts.{x}.up_proj.weight"] for x in e])
            down = torch.stack([w[f"{p}mlp.experts.{x}.down_proj.weight"] for x in e])
            act = torch.nn.functional.silu(torch.einsum("td,tfd->tf", h, gate)) * torch.einsum("td,tfd->tf", h, up)
            routed += ref_w[:, k:k + 1].double() * torch.einsum("tf,tdf->td", act, down)
        shared = mlp(h, w[p + "mlp.shared_experts.gate_proj.weight"], w[p + "mlp.shared_experts.up_proj.weight"],
                     w[p + "mlp.shared_experts.down_proj.weight"])
        # Kolibri1SparseMoeBlock returns shared_output + fused_output; the decoder layer applies post_ffn_norm
        out = shared + routed
        post = rms_norm(out, w[p + "post_ffn_norm.weight"])
        for s, ref_v in (("ffn_moe_out", routed), ("ffn_shexp", shared), ("ffn_out", out),
                         ("ffn_post_norm", post), ("l_out", post + inp)):
            err[s] = max(err[s], nmse(ref_v, g[s].double()))

    n_sel = n_layer * N_TOKENS
    want_applied = [f"ffn_moe_weights-{il}" for il in range(n_layer)]
    res = [
        top_ok,
        (sel_equal == n_sel, f"selection = top{top_k}(logits + bias) at {sel_equal}/{n_sel} token-layers"),
        (min(differ) > 0, f"non-vacuous: top{top_k}(sigmoid(logits) + bias) selects differently at "
                          f"{sum(differ)}/{n_sel} token-layers, at least {min(differ)} per layer"),
        (w_err <= MAX_WEIGHT_ERR, f"weights = sigmoid(logits[selected]): max abs error {w_err:.1e}"),
        (applied == want_applied, "experts weighted with ffn_moe_weights itself (no norm, no scale)"
                                  + ("" if applied == want_applied else f": got {applied}")),
    ]
    res += [(v <= max_nmse, f"{s} vs reference: max NMSE over {n_layer} layers {v:.1e} (<= {max_nmse:g})")
            for s, v in err.items()]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True, help="llama.cpp checkout with a built libllama")
    ap.add_argument("--seed", type=int, default=1, help="cmd/kolibri-tiny random seed")
    args = ap.parse_args()

    runner = Runner(args.llama_cpp)
    tokens = random.Random(args.seed).choices(range(127900), k=N_TOKENS)  # base vocab, no special tokens
    errs = 0
    with tempfile.TemporaryDirectory() as tmp:
        model = generate_tiny(Path(tmp), args.seed, "router")
        gguf = convert(model, args.llama_cpp, "f32")
        for name, dev in runner.devices():
            is_cpu = runner.lib.ggml_backend_dev_type(dev) == runner.lib.GGML_BACKEND_DEVICE_TYPE_CPU
            try:
                results = check(runner, gguf, dev, model, tokens, MAX_NMSE_CPU if is_cpu else MAX_NMSE)
            except RuntimeError as e:
                results = [(False, f"load and decode: {e}")]
            for ok, what in results:
                print(f"{'PASS' if ok else 'FAIL'} {name} {what}")
                errs += not ok
    if errs:
        print(f"FAIL {errs} problem(s)")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
