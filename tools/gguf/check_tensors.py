#!/usr/bin/env python3
"""Check the tensors that llama.cpp's Kolibri converter writes.

Generates the tiny random-weight checkpoint of cmd/kolibri-tiny (the shape of
the reference repo's tests/checkpoints.py, BF16, with the real tokenizer),
converts it with convert_hf_to_gguf.py --outtype bf16, and compares the GGUF
with the generator's manifest.json, which comes from internal/kolibri (the
Phase 1 mapping), not from the converter:

- tensor set: every expected tensor, nothing else;
- shape: the ggml shape, per-expert tensors stacked on the outermost axis;
- dtype: BF16 matrices; F32 for 1D tensors and the router (llama.cpp keeps
  ffn_gate_inp in F32);
- data: bit-exact against the BF16 source, experts in index order;
- metadata: block count, expert count and the per-layer SWA pattern of the
  fixture, through a full (not --vocab-only) conversion.

    check_tensors.py --llama-cpp third_party/llama.cpp

Prints one line per tensor kind and one FAIL line per wrong tensor.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCH = "kolibri"
# 2D tensors that stay F32 (convert_hf_to_gguf.py prepare_tensors).
F32_MATRICES = {"ffn_gate_inp"}


def run(cmd: list[str], what: str) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FAIL {what} exited {r.returncode}:\n{r.stderr.strip()[-1500:]}")


def kind(name: str) -> str:
    """blk.3.ffn_up_exps.weight -> ffn_up_exps."""
    return re.sub(r"^blk\.\d+\.", "", name).rsplit(".", 1)[0]


def expected_bytes(sources, tensors, f32: bool) -> bytes:
    import torch
    data = torch.stack([tensors[s] for s in sources]) if len(sources) > 1 else tensors[sources[0]]
    if f32:
        return data.float().contiguous().numpy().tobytes()
    return data.contiguous().view(torch.uint16).numpy().tobytes()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True, help="llama.cpp checkout with the Kolibri converter")
    ap.add_argument("--seed", type=int, default=1, help="cmd/kolibri-tiny random seed")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "tools" / "tokenizer"))
    from common import tokenizer_dir
    sys.path.insert(0, str(args.llama_cpp / "gguf-py"))
    import gguf
    from safetensors.torch import load_file

    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "kolibri-tiny"
        out = Path(tmp) / "kolibri-tiny-bf16.gguf"
        run(["go", "run", "./cmd/kolibri-tiny", "-out", str(model), "-tokenizer-dir", str(tokenizer_dir()),
             "-seed", str(args.seed)], "cmd/kolibri-tiny")
        run([sys.executable, str(args.llama_cpp / "convert_hf_to_gguf.py"), str(model),
             "--outtype", "bf16", "--outfile", str(out)], "convert_hf_to_gguf.py --outtype bf16")

        cfg = json.loads((model / "config.json").read_text())
        manifest = {m["name"]: m for m in json.loads((model / "manifest.json").read_text())}
        source = load_file(model / "model.safetensors")
        reader = gguf.GGUFReader(out)
        fields = {name: f.contents() for name, f in reader.fields.items()}
        got = {t.name: t for t in reader.tensors}

        errs = []
        missing = sorted(manifest.keys() - got.keys())
        extra = sorted(got.keys() - manifest.keys())
        ok = not missing and not extra
        print(f"{'PASS' if ok else 'FAIL'} tensor set: {len(got)} tensors, manifest {len(manifest)}"
              + ("" if ok else f"; missing {missing}; extra {extra}"))
        if not ok:
            errs.append("tensor set")

        kinds = defaultdict(list)
        for name in manifest:
            if name in got:
                kinds[kind(name)].append(name)
        for k, names in kinds.items():
            bad = []
            for name in names:
                m, t = manifest[name], got[name]
                f32 = len(m["ne"]) == 1 or k in F32_MATRICES
                want_type = gguf.GGMLQuantizationType.F32 if f32 else gguf.GGMLQuantizationType.BF16
                problems = []
                if [int(d) for d in t.shape] != m["ne"]:
                    problems.append(f"ne {[int(d) for d in t.shape]}, want {m['ne']}")
                if t.tensor_type != want_type:
                    problems.append(f"{t.tensor_type.name}, want {want_type.name}")
                elif t.data.tobytes() != expected_bytes(m["sources"], source, f32):
                    problems.append("data differs from " + (m["sources"][0] if len(m["sources"]) == 1
                                                            else f"stack({m['sources'][0]}, ...)"))
                if problems:
                    bad.append(name)
                    print(f"FAIL {name}: " + "; ".join(problems))
            m = manifest[names[0]]
            stacked = f", {len(m['sources'])} experts stacked" if len(m["sources"]) > 1 else ""
            t = got[names[0]]
            print(f"{'FAIL' if bad else 'PASS'} {k} ({len(names) - len(bad)}/{len(names)}): "
                  f"ne {m['ne']}, {t.tensor_type.name}{stacked}, data bit-exact")
            errs += bad

        is_swa = [lt == "sliding_attention" for lt in cfg["layer_types"]]
        for key, want in [
            ("general.architecture", ARCH),
            (f"{ARCH}.block_count", cfg["num_hidden_layers"]),
            (f"{ARCH}.expert_count", cfg["num_experts"]),
            (f"{ARCH}.attention.sliding_window_pattern", is_swa),
        ]:
            ok = fields.get(key) == want
            print(f"{'PASS' if ok else 'FAIL'} metadata: {key} = {fields.get(key)!r}" + ("" if ok else f", want {want!r}"))
            if not ok:
                errs.append(key)

    if errs:
        print(f"FAIL {len(errs)} problem(s)")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
