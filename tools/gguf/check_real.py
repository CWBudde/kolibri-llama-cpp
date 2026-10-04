#!/usr/bin/env python3
"""Check a GGUF converted from the real Kolibri-1 BF16 checkpoint.

check_tensors.py covers the converter on the tiny fixture. This check runs the
same comparisons on the full 78B checkpoint, against the checkpoint inventory
(inventory/bf16/summary.json), not against the converter code:

- shards: every safetensors shard has the size and sha256 the inventory pinned
  (skipped with --no-hash);
- tensor set: the inventory's 21 classes expanded over 50 layers, nothing else;
- shape and dtype: the inventory's ggml shape, BF16 matrices, F32 for 1D
  tensors and the router (llama.cpp keeps ffn_gate_inp in F32);
- data: bit-exact against the safetensors bytes for every tensor, experts in
  index order (skipped with --no-data);
- metadata: architecture, block count, expert count and the SWA pattern.

    check_real.py --llama-cpp third_party/llama.cpp \\
        --model-dir ~/models/Kolibri-1-BF16 --gguf ~/models/Kolibri-1-BF16.gguf

Both files are memory-mapped, so the check needs little RAM but reads about
312 GB from disk with the data comparison on.
"""

import argparse
import hashlib
import json
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "inventory" / "bf16" / "summary.json"
CONFIG = ROOT / "inventory" / "bf16" / "config.json"
ARCH = "kolibri"
F32_MATRICES = {"ffn_gate_inp"}


class Shards:
    """Raw BF16 views of the safetensors tensors, without torch."""

    def __init__(self, model_dir: Path):
        self.where = {}
        self.maps = {}
        for path in sorted(model_dir.glob("*.safetensors")):
            with path.open("rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            self.maps[path.name] = np.memmap(path, dtype=np.uint8, mode="r")
            for name, info in header.items():
                if name != "__metadata__":
                    self.where[name] = (path.name, 8 + n, info)

    def u16(self, name: str) -> np.ndarray:
        shard, base, info = self.where[name]
        if info["dtype"] != "BF16":
            raise SystemExit(f"FAIL {name}: source dtype {info['dtype']}, want BF16")
        lo, hi = info["data_offsets"]
        return self.maps[shard][base + lo:base + hi].view(np.uint16)


def bf16_to_f32_bits(u16: np.ndarray) -> np.ndarray:
    """Exact: a BF16 value is the upper half of its F32 bit pattern."""
    return u16.astype(np.uint32) << 16


def expected_tensors(summary: dict, n_layer: int) -> dict:
    """GGUF name -> (class, ggml shape, HF source names in stacking order)."""
    want = {}
    for c in summary["classes"]:
        layers = range(n_layer) if "{L}" in c["gguf"] else [None]
        for il in layers:
            gname = c["gguf"].replace("{L}", str(il))
            hf = c["hf"].replace("{L}", str(il))
            if "{E}" in hf:
                # Experts stack on the outermost ggml axis.
                sources = [hf.replace("{E}", str(e)) for e in range(c["gguf_shape"][-1])]
            else:
                sources = [hf]
            want[gname] = (c["class"], c["gguf_shape"], sources)
    return want


def check_shards(model_dir: Path, summary: dict) -> list[str]:
    errs = []
    t0 = time.time()
    for s in summary["shards"]:
        path = model_dir / s["name"]
        if not path.exists():
            errs.append(s["name"])
            print(f"FAIL shard {s['name']}: missing")
            continue
        h = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(64 << 20):
                h.update(chunk)
        size = path.stat().st_size
        if size != s["size"] or h.hexdigest() != s["sha256"]:
            errs.append(s["name"])
            print(f"FAIL shard {s['name']}: size {size}, sha256 {h.hexdigest()}")
    print(f"{'FAIL' if errs else 'PASS'} shards: {len(summary['shards']) - len(errs)}/{len(summary['shards'])} "
          f"match the pinned size and sha256 ({time.time() - t0:.0f} s)", flush=True)
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True, help="llama.cpp checkout (for gguf-py)")
    ap.add_argument("--model-dir", type=Path, required=True, help="BF16 safetensors directory at the pinned revision")
    ap.add_argument("--gguf", type=Path, help="GGUF from convert_hf_to_gguf.py --outtype bf16")
    ap.add_argument("--no-hash", action="store_true", help="skip the shard sha256 check")
    ap.add_argument("--no-data", action="store_true", help="skip the bit-exact data comparison")
    args = ap.parse_args()

    sys.path.insert(0, str(args.llama_cpp / "gguf-py"))
    import gguf

    summary = json.loads(SUMMARY.read_text())
    cfg = json.loads(CONFIG.read_text())
    errs = [] if args.no_hash else check_shards(args.model_dir, summary)
    if args.gguf is None:
        sys.exit(1 if errs else 0)

    reader = gguf.GGUFReader(args.gguf)
    fields = {name: f.contents() for name, f in reader.fields.items()}
    got = {t.name: t for t in reader.tensors}
    want = expected_tensors(summary, cfg["num_hidden_layers"])

    missing = sorted(want.keys() - got.keys())
    extra = sorted(got.keys() - want.keys())
    ok = not missing and not extra
    print(f"{'PASS' if ok else 'FAIL'} tensor set: {len(got)} tensors, inventory {len(want)}"
          + ("" if ok else f"; missing {missing[:10]}; extra {extra[:10]}"))
    if not ok:
        errs.append("tensor set")

    shards = None if args.no_data else Shards(args.model_dir)
    by_class = defaultdict(list)
    for name in want:
        if name in got:
            by_class[want[name][0]].append(name)
    t0 = time.time()
    for cls, names in by_class.items():
        bad = []
        for name in names:
            _, ne, sources = want[name]
            t = got[name]
            f32 = len(ne) == 1 or cls in F32_MATRICES
            want_type = gguf.GGMLQuantizationType.F32 if f32 else gguf.GGMLQuantizationType.BF16
            problems = []
            if [int(d) for d in t.shape] != ne:
                problems.append(f"ne {[int(d) for d in t.shape]}, want {ne}")
            if t.tensor_type != want_type:
                problems.append(f"{t.tensor_type.name}, want {want_type.name}")
            elif shards is not None:
                # Compare bit patterns: F32 as uint32, BF16 as uint16.
                data = np.ascontiguousarray(t.data).view(np.uint32 if f32 else np.uint16).reshape(len(sources), -1)
                for e, src in enumerate(sources):
                    ref = shards.u16(src)
                    if f32:
                        ref = bf16_to_f32_bits(ref)
                    if not np.array_equal(data[e], ref):
                        problems.append(f"data differs from {src}")
                        break
            if problems:
                bad.append(name)
                print(f"FAIL {name}: " + "; ".join(problems))
        ne = want[names[0]][1]
        n_src = len(want[names[0]][2])
        stacked = f", {n_src} experts stacked" if n_src > 1 else ""
        data = "" if shards is None else ", data bit-exact"
        print(f"{'FAIL' if bad else 'PASS'} {cls} ({len(names) - len(bad)}/{len(names)}): "
              f"ne {ne}, {got[names[0]].tensor_type.name}{stacked}{data}", flush=True)
        errs += bad
    if shards is not None:
        print(f"info data comparison took {time.time() - t0:.0f} s")

    is_swa = [lt == "sliding_attention" for lt in cfg["layer_types"]]
    for key, value in [
        ("general.architecture", ARCH),
        (f"{ARCH}.block_count", cfg["num_hidden_layers"]),
        (f"{ARCH}.expert_count", cfg["num_experts"]),
        (f"{ARCH}.expert_used_count", cfg["num_experts_per_tok"]),
        (f"{ARCH}.attention.sliding_window_pattern", is_swa),
    ]:
        ok = fields.get(key) == value
        print(f"{'PASS' if ok else 'FAIL'} metadata: {key}" + ("" if ok else f" = {fields.get(key)!r}, want {value!r}"))
        if not ok:
            errs.append(key)

    n_param = sum(int(np.prod(t.shape)) for t in got.values())
    ok = n_param == summary["parameters"]
    print(f"{'PASS' if ok else 'FAIL'} parameters: {n_param:,} in the GGUF, inventory {summary['parameters']:,}")
    if not ok:
        errs.append("parameters")

    if errs:
        print(f"FAIL {len(errs)} problem(s)")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
