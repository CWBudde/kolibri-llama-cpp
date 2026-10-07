#!/usr/bin/env python3
"""The bytes of one routed expert per layer of a GGUF (PLAN Phase 8).

An expert of layer l is its rows of blk.l.ffn_gate_exps, ffn_up_exps and
ffn_down_exps, each stacked over the experts on the outermost axis, so it
takes the tensors' exact n_bytes divided by expert_count. cmd/kolibri-cache
reads them from testdata/locality/expert-bytes.json to estimate the expert
bytes a cache loads per token.

The default run compares the GGUF with the recorded entry and fails (exit 1)
on any difference; --record writes it.

    expert_bytes.py --gguf ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf [--record]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERT_BYTES = ROOT / "testdata" / "locality" / "expert-bytes.json"
KINDS = ("gate", "up", "down")


def measure(llama_cpp: Path, gguf: Path) -> dict:
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    from gguf import GGUFReader

    r = GGUFReader(gguf)
    arch = r.fields["general.architecture"].contents()
    n_layer = int(r.fields[f"{arch}.block_count"].contents())
    n_expert = int(r.fields[f"{arch}.expert_count"].contents())
    tensors = {t.name: t for t in r.tensors}
    layers = []
    for il in range(n_layer):
        entry, total = {}, 0
        for kind in KINDS:
            t = tensors[f"blk.{il}.ffn_{kind}_exps.weight"]
            if int(t.n_bytes) % n_expert:
                raise SystemExit(f"FAIL {t.name}: {int(t.n_bytes)} bytes do not split into {n_expert} experts")
            entry[kind] = t.tensor_type.name
            total += int(t.n_bytes) // n_expert
        layers.append({**entry, "bytes_per_expert": total})
    return {"gguf": gguf.name, "gguf_bytes": gguf.stat().st_size, "n_expert": n_expert, "layers": layers}


def dumps(recorded: dict) -> str:
    """indent 1, one layer per line"""
    out = ["{"]
    for i, (stem, m) in enumerate(recorded.items()):
        out.append(f" {json.dumps(stem)}: {{")
        for key in ("gguf", "gguf_bytes", "n_expert"):
            out.append(f"  {json.dumps(key)}: {json.dumps(m[key])},")
        out.append('  "layers": [')
        out += [f"   {json.dumps(layer)}{',' if j < len(m['layers']) - 1 else ''}" for j, layer in enumerate(m["layers"])]
        out.append("  ]")
        out.append(" }" + ("," if i < len(recorded) - 1 else ""))
    out.append("}")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, default=ROOT / "third_party" / "llama.cpp")
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--record", action="store_true", help="make these sizes the recorded ones")
    args = ap.parse_args()

    recorded = json.loads(EXPERT_BYTES.read_text()) if EXPERT_BYTES.exists() else {}
    got = measure(args.llama_cpp, args.gguf)
    per = sorted({layer["bytes_per_expert"] for layer in got["layers"]})
    sizes = ", ".join(f"{b:,}" for b in per)
    total = sum(layer["bytes_per_expert"] for layer in got["layers"]) * got["n_expert"]
    what = (f"{len(got['layers'])} layers, {sizes} bytes per expert, {total / 1e9:.2f} GB of experts "
            f"({args.gguf.name})")
    if args.record:
        recorded[args.gguf.stem] = got
        EXPERT_BYTES.write_text(dumps(recorded))
        print(f"recorded {what} in {EXPERT_BYTES.relative_to(ROOT)}")
    elif recorded.get(args.gguf.stem) != got:
        old = recorded.get(args.gguf.stem)
        diff = None if old is None else {k: (old.get(k), got[k]) for k in got if k != "layers" and old.get(k) != got[k]}
        if old is not None and old.get("layers") != got["layers"]:
            diff["layers"] = [i for i, (a, b) in enumerate(zip(old["layers"], got["layers"])) if a != b] or "count"
        print(f"FAIL {args.gguf.stem}: differs from {EXPERT_BYTES.relative_to(ROOT)}: {diff or 'not recorded'}")
        sys.exit(1)
    else:
        print(f"PASS {what}, as recorded")


if __name__ == "__main__":
    main()
