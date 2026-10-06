#!/usr/bin/env python3
"""The regression suite: rerun after a llama.cpp or upstream change.

Runs three parts, each gated, and fails if any of them fails:

- tokenizer: compare.py's golden cases and its fixed corpus (exact token IDs
  and detokenized bytes against the pinned reference tokenizer), on a
  vocab-only GGUF that vocab_gguf.py writes first;
- tiny reference: compare_real.py --tiny, the token embeddings, Q and K after
  the QK norm, attention output, router logits and selected experts, routed and
  shared expert output, layer output and logits of the F32 tiny model within
  NMSE 1e-6 of the torch reference (kolibri_ref.py);
- real weights: e2e.py on the BF16 GGUF, every libllama node and the logits
  of each corpus case bit-identical to the run recorded in
  testdata/e2e/manifest.json, with the metrics against the stored reference
  artifacts as INFO.

A change that e2e.py reports is accepted with e2e.py --record after review.

    regress.py --llama-cpp third_party/llama.cpp [--gguf ~/models/Kolibri-1-BF16.gguf]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GGUF = Path.home() / "models" / "Kolibri-1-BF16.gguf"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True)
    ap.add_argument("--gguf", type=Path, default=GGUF, help=f"BF16 GGUF of the real checkpoint (default: {GGUF})")
    ap.add_argument("--cases", help="e2e.py: comma-separated case names (default: all)")
    ap.add_argument("--manifest", type=Path, help="e2e.py: the manifest (default: testdata/e2e/manifest.json)")
    ap.add_argument("--threads", type=int, help="e2e.py: libllama CPU threads (default: e2e.py's)")
    args = ap.parse_args()

    py, llama_cpp = sys.executable, str(args.llama_cpp)
    vocab = str(args.llama_cpp / "models" / "ggml-vocab-kolibri.gguf")
    e2e = [py, str(ROOT / "tools" / "ref" / "e2e.py"), "--llama-cpp", llama_cpp, "--gguf", str(args.gguf)]
    for flag, val in (("--cases", args.cases), ("--manifest", args.manifest), ("--threads", args.threads)):
        if val is not None:
            e2e += [flag, str(val)]
    parts = [
        ("tokenizer", [[py, str(ROOT / "tools" / "tokenizer" / "vocab_gguf.py"), "--llama-cpp", llama_cpp,
                        "--out", vocab],
                       [py, str(ROOT / "tools" / "tokenizer" / "compare.py"), "--llama-cpp", llama_cpp,
                        "--vocab", vocab, "--only", "golden", "--only", "corpus"]]),
        ("tiny reference", [[py, str(ROOT / "tools" / "ref" / "compare_real.py"), "--llama-cpp", llama_cpp,
                             "--tiny"]]),
        ("real weights", [e2e]),
    ]

    failed = []
    for name, commands in parts:
        print(f"=== {name}", flush=True)
        t0 = time.time()
        if name == "real weights" and not args.gguf.exists():
            print(f"FAIL {args.gguf} is missing; pass the BF16 GGUF with --gguf")
            ok = False
        else:
            ok = all(subprocess.run(c).returncode == 0 for c in commands)
        print(f"=== {name}: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f} s)", flush=True)
        if not ok:
            failed.append(name)
    if failed:
        print(f"FAIL regression suite: {', '.join(failed)}")
        sys.exit(1)
    print("PASS regression suite: " + ", ".join(name for name, _ in parts))


if __name__ == "__main__":
    main()
