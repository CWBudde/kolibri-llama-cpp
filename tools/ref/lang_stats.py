#!/usr/bin/env python3
"""Per-case sensitivity numbers from the stored e2e reference artifacts, to
compare languages without running a model.

For every case of testdata/e2e/corpus.json it reads <case>.npz from
~/models/eval/e2e (e2e.py --write) and prints:

- router margins: per (token, layer) the reference's k-th minus (k+1)-th score
  of logits + bias in float32; the share below 0.01 and 0.1 and the median
  (a small margin is a near-tie that a small error can flip);
- prompt cases: per generated position, KL(float32 || bfloat16) of the
  reference's two runs and the float32 top probability, so a high mean KLD
  can be traced to the positions it comes from;
- chunk cases: PPL over the second half (as llama-perplexity) and the same loss
  per byte of text (bits per byte), which does not depend on how many bytes a
  token covers. German tokens are longer than English ones, so per-token PPL
  alone is not comparable across the two.

    lang_stats.py [--artifacts ~/models/eval/e2e]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "ref"))
sys.path.insert(0, str(ROOT / "tools" / "tokenizer"))

from common import tokenizer_dir  # noqa: E402
from compare_real import log_softmax64  # noqa: E402

CORPUS = ROOT / "testdata" / "e2e" / "corpus.json"
ARTIFACTS = Path.home() / "models" / "eval" / "e2e"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_dir() / "tokenizer.json"))
    cases = json.loads(CORPUS.read_text())["cases"]
    for name, case in cases.items():
        with np.load(args.artifacts / f"{name}.npz") as z:
            tokens = z["tokens"].tolist()
            s = np.sort(z["router_logits"] + z["router_bias"][:, None, :], axis=-1)[..., ::-1]
            k = z["router_topk"].shape[-1]
            margin = s[..., k - 1] - s[..., k]
            print(f"{name}: router margin < 0.01 for {np.mean(margin < 0.01):.1%}, < 0.1 for "
                  f"{np.mean(margin < 0.1):.1%} of {margin.size} (token, layer) pairs, median {np.median(margin):.3f}")
            n0 = len(case["tokens"])
            if case["kind"] == "prompt":
                a = log_softmax64(z["ref_float32"][n0 - 1:])
                b = log_softmax64(z["ref_bfloat16"][n0 - 1:])
                kld = (np.exp(a) * (a - b)).sum(axis=1)
                top = np.exp(a).max(axis=1)
                print(f"{name}: generated {tokens[n0:n0 + 4]}…; per position from the last prompt token, "
                      f"KLD float32 || bfloat16 " + " ".join(f"{v:.3f}" for v in kld))
                print(f"{name}: float32 top probability " + " ".join(f"{v:.2f}" for v in top))
            else:
                first = len(tokens) // 2
                lp = log_softmax64(z["ref_float32"][first:-1])
                nxt = tokens[first + 1:]
                nll = -lp[np.arange(len(nxt)), nxt]
                n_bytes = len(tok.decode(nxt).encode())
                print(f"{name}: PPL {np.exp(nll.mean()):.4f} over {len(nxt)} tokens, {n_bytes} bytes "
                      f"({n_bytes / len(nxt):.2f} per token), {nll.sum() / np.log(2) / n_bytes:.3f} bits per byte")


if __name__ == "__main__":
    main()
