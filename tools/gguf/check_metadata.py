#!/usr/bin/env python3
"""Check the GGUF metadata that llama.cpp's Kolibri converter writes.

Runs convert_hf_to_gguf.py --vocab-only on the pinned BF16 config and
tokenizer files. The vocab-only path still calls set_gguf_parameters(), so
the output holds every hyperparameter but no tensor payload. The values are
then compared with inventory/bf16/config.json, independently of the converter
code. Each line names the PLAN.md Phase 3 metadata item it covers.

    check_metadata.py --llama-cpp third_party/llama.cpp

Without --model-dir it uses the pinned BF16 revision from the Phase 1
inventory, checked against the recorded sha256.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "inventory" / "bf16" / "config.json"
ARCH = "kolibri"


def expected(cfg: dict) -> list[tuple[str, str, object]]:
    """(PLAN.md item, GGUF key, expected value)."""
    return [
        ("architecture", "general.architecture", ARCH),
        ("50 blocks", f"{ARCH}.block_count", cfg["num_hidden_layers"]),
        ("embedding size 2,560", f"{ARCH}.embedding_length", cfg["hidden_size"]),
        ("48/4 Q/KV heads", f"{ARCH}.attention.head_count", cfg["num_attention_heads"]),
        ("48/4 Q/KV heads", f"{ARCH}.attention.head_count_kv", cfg["num_key_value_heads"]),
        ("head dimension / RoPE dimension", f"{ARCH}.attention.key_length", cfg["head_dim"]),
        ("head dimension / RoPE dimension", f"{ARCH}.attention.value_length", cfg["head_dim"]),
        # Sliding layers rotate the full head (Phase 1: get_rope default).
        ("head dimension / RoPE dimension", f"{ARCH}.rope.dimension_count", cfg["head_dim"]),
        ("RMSNorm epsilon", f"{ARCH}.attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"]),
    ]


def convert(llama_cpp: Path, model_dir: Path, out: Path) -> None:
    cmd = [sys.executable, str(llama_cpp / "convert_hf_to_gguf.py"), str(model_dir),
           "--vocab-only", "--outfile", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FAIL convert_hf_to_gguf.py --vocab-only exited {r.returncode}:\n{r.stderr.strip()[-1500:]}")


def same(got, want) -> bool:
    if isinstance(want, float):
        # GGUF stores the epsilon as float32.
        return isinstance(got, float) and abs(got - want) <= 1e-6 * abs(want)
    return got == want


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True, help="llama.cpp checkout with the Kolibri converter")
    ap.add_argument("--model-dir", type=Path, help="directory with config.json, tokenizer.json, tokenizer_config.json")
    args = ap.parse_args()

    if args.model_dir is None:
        sys.path.insert(0, str(ROOT / "tools" / "tokenizer"))
        from common import tokenizer_dir
        args.model_dir = tokenizer_dir()

    sys.path.insert(0, str(args.llama_cpp / "gguf-py"))
    import gguf

    cfg = json.loads(CONFIG.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "kolibri-vocab.gguf"
        convert(args.llama_cpp, args.model_dir, out)
        reader = gguf.GGUFReader(out)
        fields = {name: f.contents() for name, f in reader.fields.items()}

    errs = []
    checked = set()
    for item, key, want in expected(cfg):
        checked.add(key)
        got = fields.get(key)
        ok = same(got, want)
        print(f"{'PASS' if ok else 'FAIL'} {item}: {key} = {got!r}" + ("" if ok else f", want {want!r}"))
        if not ok:
            errs.append(key)

    # Other hyperparameters, for information only: later metadata items own them.
    for key in sorted(k for k in fields if k.startswith(f"{ARCH}.") and k not in checked):
        print(f"info {key} = {fields[key]!r}")

    if errs:
        print(f"FAIL {len(errs)} metadata key(s) differ")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
