#!/usr/bin/env python3
"""Check the GGUF metadata that llama.cpp's Kolibri converter writes.

Runs convert_hf_to_gguf.py --vocab-only on the pinned BF16 config and
tokenizer files. The vocab-only path still calls set_gguf_parameters(), so
the output holds every hyperparameter but no tensor payload. The values are
then compared with inventory/bf16/config.json and the Phase 1 summary
(inventory/bf16/summary.json), independently of the converter code. Each line
names the PLAN.md Phase 3 metadata item it covers.

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
SUMMARY = ROOT / "inventory" / "bf16" / "summary.json"
ARCH = "kolibri"


def swa_layers(cfg: dict, summary: dict) -> list[bool]:
    """Per-layer SWA flags from the Phase 1 attention summary. Checks that the
    summary is consistent with its own period (full layer last)."""
    att = summary["attention"]
    n = cfg["num_hidden_layers"]
    is_swa = [il not in att["full_attention_layers"] for il in range(n)]
    period = att["swa_period"]
    if is_swa != [(il + 1) % period != 0 for il in range(n)]:
        raise SystemExit(f"FAIL inventory: full_attention_layers do not follow swa_period {period}")
    return is_swa


def expected(cfg: dict, summary: dict) -> list[tuple[str, str, object]]:
    """(PLAN.md item, GGUF key, expected value)."""
    shexp = next(c for c in summary["classes"] if c["class"] == "ffn_gate_shexp")
    n_shexp, rest = divmod(shexp["count"], cfg["num_hidden_layers"])
    if rest:
        raise SystemExit(f"FAIL inventory: {shexp['count']} ffn_gate_shexp tensors for {cfg['num_hidden_layers']} layers")
    is_swa = swa_layers(cfg, summary)
    period = summary["attention"]["swa_period"]
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
        ("384 experts / 6 active", f"{ARCH}.expert_count", cfg["num_experts"]),
        ("384 experts / 6 active", f"{ARCH}.expert_used_count", cfg["num_experts_per_tok"]),
        ("expert FFN size 512", f"{ARCH}.expert_feed_forward_length", cfg["moe_intermediate_size"]),
        # One shared gate_proj per layer, with 512 output rows.
        ("shared expert", f"{ARCH}.expert_shared_count", n_shexp),
        ("shared expert", f"{ARCH}.expert_shared_feed_forward_length", shexp["hf_shape"][0]),
        # 513 = 512 preceding tokens + the current one: llama.cpp masks when
        # p1 - p0 >= n_swa (LLAMA_SWA_TYPE_STANDARD), vLLM uses window (512, 0).
        ("SWA size 512", f"{ARCH}.attention.sliding_window", summary["attention"]["sliding_window"]),
        (f"repeating SWA/full pattern (period {period}, full last)",
         f"{ARCH}.attention.sliding_window_pattern", is_swa),
        ("RoPE base 10,000 and SWA-only RoPE", f"{ARCH}.rope.freq_base", cfg["rope_theta"]),
        # Full-attention layers use no positional encoding (Phase 1).
        ("RoPE base 10,000 and SWA-only RoPE", f"{ARCH}.attention.rope_pattern", is_swa),
    ]


def convert(llama_cpp: Path, model_dir: Path, out: Path) -> None:
    cmd = [sys.executable, str(llama_cpp / "convert_hf_to_gguf.py"), str(model_dir),
           "--vocab-only", "--outfile", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FAIL convert_hf_to_gguf.py --vocab-only exited {r.returncode}:\n{r.stderr.strip()[-1500:]}")


def show(v) -> str:
    """Per-layer bool arrays as a compact 0/1 string."""
    if isinstance(v, list) and v and all(isinstance(x, bool) for x in v):
        return "".join("1" if x else "0" for x in v)
    return repr(v)


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
    summary = json.loads(SUMMARY.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "kolibri-vocab.gguf"
        convert(args.llama_cpp, args.model_dir, out)
        reader = gguf.GGUFReader(out)
        fields = {name: f.contents() for name, f in reader.fields.items()}

    errs = []
    checked = set()
    for item, key, want in expected(cfg, summary):
        checked.add(key)
        got = fields.get(key)
        ok = same(got, want)
        print(f"{'PASS' if ok else 'FAIL'} {item}: {key} = {show(got)}" + ("" if ok else f", want {show(want)}"))
        if not ok:
            errs.append(key)

    # Other hyperparameters, for information only: later metadata items own them.
    for key in sorted(k for k in fields if k.startswith(f"{ARCH}.") and k not in checked):
        print(f"info {key} = {show(fields[key])}")

    if errs:
        print(f"FAIL {len(errs)} metadata key(s) differ")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
