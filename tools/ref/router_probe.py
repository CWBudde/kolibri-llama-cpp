"""Router probe: per (token, layer) agreement of two Kolibri routers.

Both sides select experts by the top k of logits + bias and weight them with
sigmoid(logits), so the probe compares router logits, their sigmoid
probabilities, the Top-1 expert (largest logits + bias), the Top-k sets and
their overlap. The margin is the reference's k-th minus (k+1)-th score of
logits + bias: a small margin is a near-tie, where a tiny logit difference
can change the selected set.

compare_real.py runs it for libllama on the CPU and on Metal and for the
reference in bfloat16, each against the reference in float32 or float64.
"""

from pathlib import Path

MARGIN_BUCKETS = (0.001, 0.01, 0.1)


def same_experts(a, b):
    """Per token, whether both select the same set of experts."""
    import numpy as np

    return (np.sort(a, axis=1) == np.sort(b, axis=1)).all(axis=1)


def overlap(a, b):
    """Per token, how many experts both select."""
    return (a[:, :, None] == b[:, None, :]).sum(axis=(1, 2))


def probe(got_logits, got_topk, ref_logits, ref_topk, bias) -> dict:
    """Agreement of got with ref. Logits are [n_layer, n_tokens, n_expert], topk are [n_layer, n_tokens, k]
    and bias is [n_layer, n_expert]. Returns the inputs with per (layer, token) arrays top1_same,
    set_same, overlap and margin, and per layer logit_nmse and prob_nmse."""
    import numpy as np

    from check_model import nmse

    got_logits, ref_logits = np.asarray(got_logits, np.float32), np.asarray(ref_logits, np.float32)
    got_topk, ref_topk = np.asarray(got_topk, np.int32), np.asarray(ref_topk, np.int32)
    bias = np.asarray(bias, np.float32)
    k = ref_topk.shape[-1]
    got_sel, ref_sel = got_logits + bias[:, None, :], ref_logits + bias[:, None, :]
    s = -np.sort(-ref_sel, axis=-1)
    n_layer = ref_logits.shape[0]

    def sig(x):
        return 1 / (1 + np.exp(-x.astype(np.float64)))

    return {
        "got_logits": got_logits, "ref_logits": ref_logits, "got_topk": got_topk, "ref_topk": ref_topk,
        "bias": bias,
        "top1_same": got_sel.argmax(-1) == ref_sel.argmax(-1),
        "set_same": np.stack([same_experts(got_topk[il], ref_topk[il]) for il in range(n_layer)]),
        "overlap": np.stack([overlap(got_topk[il], ref_topk[il]) for il in range(n_layer)]),
        "margin": s[..., k - 1] - s[..., k] if k < s.shape[-1] else np.full(s.shape[:-1], np.inf, np.float32),
        "logit_nmse": np.array([nmse(ref_logits[il], got_logits[il]) for il in range(n_layer)]),
        "prob_nmse": np.array([nmse(sig(ref_logits[il]), sig(got_logits[il])) for il in range(n_layer)]),
    }


def summary(p: dict, what: str) -> list[str]:
    """INFO lines: Top-1 and Top-k agreement over all (token, layer) pairs, the overlap histogram, and
    how often the selected set changes per reference-margin bucket."""
    import numpy as np

    k = p["ref_topk"].shape[-1]
    ov, margin, changed = p["overlap"].ravel(), p["margin"].ravel(), ~p["set_same"].ravel()
    hist = ", ".join(f"{n}: {(ov == n).mean():.2%}" for n in range(k, -1, -1) if (ov == n).any())
    lines = [
        f"router {what}: Top-1 same {p['top1_same'].mean():.2%}, Top-{k} set same {p['set_same'].mean():.2%}, "
        f"mean overlap {ov.mean():.3f}/{k} over {ov.size} (token, layer) pairs; overlap {hist}",
    ]
    edges = (-np.inf,) + MARGIN_BUCKETS + (np.inf,)
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (margin >= lo) & (margin < hi)
        label = f"< {hi:g}" if lo == -np.inf else (f">= {lo:g}" if hi == np.inf else f"{lo:g} to {hi:g}")
        parts.append(f"{label}: {changed[m].mean() if m.any() else float('nan'):.2%} of {int(m.sum())}")
    lines.append(f"router {what}: Top-{k} set changed by reference margin (k-th minus (k+1)-th of logits + bias): "
                 + "; ".join(parts))
    lines.append(f"router {what}: logit NMSE median {np.median(p['logit_nmse']):.2e}, "
                 f"max {p['logit_nmse'].max():.2e}; sigmoid NMSE max {p['prob_nmse'].max():.2e}")
    return lines


def table(p: dict, what: str) -> list[str]:
    """INFO lines: per layer logit and sigmoid NMSE, Top-1 same, Top-k set same and mean overlap."""
    k = p["ref_topk"].shape[-1]
    lines = [f"router {what}: layer, logit NMSE, sigmoid NMSE, Top-1 same, Top-{k} set same, mean overlap"]
    for il in range(len(p["logit_nmse"])):
        lines.append(f"  {il:2d}  {p['logit_nmse'][il]:.2e}  {p['prob_nmse'][il]:.2e}  "
                     f"{p['top1_same'][il].mean():6.1%}  {p['set_same'][il].mean():6.1%}  {p['overlap'][il].mean():.2f}")
    return lines


def save(path: Path, p: dict) -> None:
    """All arrays of p in one compressed .npz."""
    import numpy as np

    np.savez_compressed(path, **p)
