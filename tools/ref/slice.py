#!/usr/bin/env python3
"""Write a slice of the real Kolibri-1 checkpoint: chosen layers as a smaller
HF checkpoint, so vLLM itself can run real weights on the CPU.

The tensors come from the BF16 GGUF, which check_real.py showed bit-exact to
the safetensors. The inventory (inventory/bf16/summary.json) maps every GGUF
tensor to its HF names; a stacked expert tensor holds the experts in index
order on its outermost axis, and the converter's F32 tensors (1D norms, the
router and its bias) hold BF16 values, written back as BF16 after checking
that their low 16 bits are zero. The chosen layers are renumbered 0..n-1; the
config is the real one with num_hidden_layers and layer_types cut to them.

A slice computes a smaller, well-defined Kolibri model. vLLM, the torch port
and libllama must agree on it as on any checkpoint, with the real attention
shape (48/4 heads, window 513), the real 384-expert top-6 routing and the real
weights' magnitudes. It does not compute the full model's output.

    slice.py --gguf ~/models/Kolibri-1-BF16.gguf --model-dir ~/models/Kolibri-1-BF16 \\
        --layers 0,4 --out /tmp/kolibri-slice-0-4
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "inventory" / "bf16" / "summary.json"
FILES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json")


def bf16_bits(t, gguf) -> np.ndarray:
    """The tensor's BF16 bit patterns, one row per stacked entry."""
    data = np.ascontiguousarray(t.data)
    if t.tensor_type == gguf.GGMLQuantizationType.BF16:
        return data.view(np.uint16)
    if t.tensor_type == gguf.GGMLQuantizationType.F32:
        u32 = data.view(np.uint32)
        if (u32 & 0xFFFF).any():
            raise SystemExit(f"FAIL {t.name}: F32 values that are not BF16")
        return (u32 >> 16).astype(np.uint16)
    raise SystemExit(f"FAIL {t.name}: type {t.tensor_type.name}")


def write_safetensors(path: Path, tensors: dict) -> None:
    """tensors: name -> (shape, uint16 BF16 bits); one file, data in name order."""
    header, offset = {}, 0
    for name in sorted(tensors):
        shape, bits = tensors[name]
        assert bits.size == int(np.prod(shape)), name
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + bits.nbytes]}
        offset += bits.nbytes
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with path.open("wb") as f:
        f.write(len(raw).to_bytes(8, "little"))
        f.write(raw)
        for name in sorted(tensors):
            f.write(np.ascontiguousarray(tensors[name][1]).tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, default=ROOT / "third_party" / "llama.cpp")
    ap.add_argument("--gguf", type=Path, required=True, help="the BF16 GGUF of the real checkpoint")
    ap.add_argument("--model-dir", type=Path, required=True, help="its config and tokenizer files")
    ap.add_argument("--layers", required=True, help="comma-separated layers, in order")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(args.llama_cpp / "gguf-py"))
    import gguf

    layers = [int(x) for x in args.layers.split(",")]
    config = json.loads((args.model_dir / "config.json").read_text())
    if not all(0 <= il < config["num_hidden_layers"] for il in layers):
        raise SystemExit(f"FAIL --layers {layers}: the checkpoint has {config['num_hidden_layers']} layers")
    reader = gguf.GGUFReader(args.gguf)
    got = {t.name: t for t in reader.tensors}
    classes = json.loads(SUMMARY.read_text())["classes"]

    tensors = {}
    for c in classes:
        targets = [(c["gguf"].replace("{L}", str(il)), c["hf"].replace("{L}", str(i)))
                   for i, il in enumerate(layers)] if "{L}" in c["gguf"] else [(c["gguf"], c["hf"])]
        for gname, hf in targets:
            t = got[gname]
            bits = bf16_bits(t, gguf)
            if "{E}" in hf:
                rows = bits.reshape(c["gguf_shape"][-1], -1)
                for e, row in enumerate(rows):
                    tensors[hf.replace("{E}", str(e))] = (c["hf_shape"], row)
            else:
                tensors[hf] = (c["hf_shape"], bits.reshape(-1))

    args.out.mkdir(parents=True, exist_ok=True)
    write_safetensors(args.out / "model.safetensors", tensors)
    config["num_hidden_layers"] = len(layers)
    config["layer_types"] = [config["layer_types"][il] for il in layers]
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in FILES:
        if (args.model_dir / name).exists():
            shutil.copy(args.model_dir / name, args.out / name)
    (args.out / "slice.json").write_text(json.dumps({"gguf": str(args.gguf), "layers": layers}, indent=1) + "\n")
    size = (args.out / "model.safetensors").stat().st_size
    print(f"INFO slice of layers {layers} ({', '.join(config['layer_types'])}): {len(tensors)} tensors, "
          f"{size / 1e9:.2f} GB, {args.out}")


if __name__ == "__main__":
    main()
