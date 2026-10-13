#!/usr/bin/env python3
"""The full Kolibri-1 model through vLLM itself on the CPU, as consecutive
slices of the real weights (PLAN Phase 6, without the GPU run of Phase 0).

The 156 GB BF16 checkpoint does not fit vLLM's CPU backend, but a slice of two
layers does (slice.py). A slice that gets the previous slice's output computes
what the full model computes in those layers: vLLM's decoder layer hands on
the hidden state and the residual, and the next layer uses only their sum (its
fused add and RMSNorm), which vllm_capture.py records as l_out. Fed to the next
slice as prompt_embeds (vllm_capture.py --embeds), that sum is the residual of
the slice's first layer and the input of its norm, as in the full model. In
float32 nothing is rounded on the way. Every slice runs the same tokens: the
e2e corpus, each prompt with the torch port's float32 greedy continuation
(vllm_capture.py --sequence).

run: for each slice of --per-slice layers, write it from the BF16 GGUF,
capture it with vllm_capture.py in vLLM's own environment (--vllm-python),
delete it, and stitch the captures into one capture of the whole model, in
vllm_capture.py's layout with the global layer numbers. The logits come from
the last slice, the token embeddings from the first. compare_vllm.py then
reads it as a capture of the full checkpoint. A part whose capture exists is
reused, so an interrupted run resumes.

stitch: the same from existing parts. Each part must have been fed exactly the
l_out of the part before it (sha256 in its capture.json), or it fails.

diff: compares two captures of the same layers, array by array, for the
chaining check: one slice of layers 0-2 against the chain 0-1, 2.

vLLM's greedy continuation is not decoded here; a prompt case's line says
whether vLLM's argmax along the sequence is its next token everywhere, in
which case vLLM's greedy continuation is the sequence's, and the stitched
capture records it as vLLM's greedy tokens.

    vllm_chain.py run --gguf ~/models/Kolibri-1-BF16.gguf --model-dir ~/models/Kolibri-1-BF16 \\
        --vllm-python vllm-venv/bin/python --work /tmp/chain --out /tmp/chain/capture
    vllm_chain.py diff /tmp/slice3-capture /tmp/chain3/capture --max-nmse 0
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "testdata" / "e2e" / "corpus.json"
MANIFEST = ROOT / "testdata" / "e2e" / "manifest.json"
REF = ROOT / "tools" / "ref"
PER_LAYER = ("router_logits", "router_topk", "router_bias", "is_swa")


def array_sha256(a) -> str:
    """As tools/ref/e2e.py and vllm_capture.py: dtype and shape, then the raw bytes."""
    import numpy as np

    a = np.ascontiguousarray(a)
    return hashlib.sha256(f"{a.dtype.str}{a.shape}".encode() + a.tobytes()).hexdigest()


def nmse(ref, got) -> float:
    """check_model.nmse, in float64."""
    import numpy as np

    ref, got = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    return float(((ref - got) ** 2).sum() / (ref ** 2).sum())


def parts_of(n_layer: int, per: int) -> list[list[int]]:
    return [list(range(a, min(a + per, n_layer))) for a in range(0, n_layer, per)]


def run(args) -> None:
    config = json.loads((args.model_dir / "config.json").read_text())
    n_layer = args.n_layers or config["num_hidden_layers"]
    attn = {int(x) for x in args.attn_layers.split(",")} if args.attn_layers else set()
    args.work.mkdir(parents=True, exist_ok=True)
    parts, prev = [], None
    for layers in parts_of(n_layer, args.per_slice):
        part = args.work / f"part-{layers[0]}-{layers[-1]}"
        done = part / "chain.json"
        if done.exists() and json.loads(done.read_text())["layers"] == layers:
            print(f"INFO layers {layers}: reusing {part}", flush=True)
        else:
            t0 = time.time()
            shutil.rmtree(part, ignore_errors=True)
            sdir = args.work / f"slice-{layers[0]}-{layers[-1]}"
            subprocess.run([sys.executable, str(REF / "slice.py"), "--llama-cpp", str(args.llama_cpp),
                            "--gguf", str(args.gguf), "--model-dir", str(args.model_dir),
                            "--layers", ",".join(map(str, layers)), "--out", str(sdir)], check=True)
            local = [str(i) for i, il in enumerate(layers) if il in attn]
            cmd = [str(args.vllm_python), str(REF / "vllm_capture.py"), "--model", str(sdir),
                   "--corpus", str(args.corpus), "--sequence", str(args.manifest), "--dtype", args.dtype,
                   "--kv-cache-gib", str(args.kv_cache_gib),
                   "--attn-layers", ",".join(local) or "none", "--out", str(part)]
            cmd += [x for name in args.case or [] for x in ("--case", name)]
            if prev:
                cmd += ["--embeds", str(prev)]
            if layers[-1] != n_layer - 1:
                cmd.append("--no-logits")
            subprocess.run(cmd, check=True)
            shutil.rmtree(sdir)
            done.write_text(json.dumps({"layers": layers, "seconds": round(time.time() - t0)}) + "\n")
            print(f"INFO layers {layers}: captured in {round(time.time() - t0)} s", flush=True)
        parts.append(part)
        prev = part
    stitch(parts, args.out, args.model_dir)


def stitch(parts: list[Path], out: Path, model_dir: Path | None) -> None:
    import numpy as np

    metas = [json.loads((p / "capture.json").read_text()) for p in parts]
    chains = [json.loads((p / "chain.json").read_text())["layers"] for p in parts]
    flat = [il for layers in chains for il in layers]
    if flat != list(range(len(flat))):
        raise SystemExit(f"FAIL stitch: the parts hold layers {chains}, not 0..{len(flat) - 1} in order")
    for key in ("dtype", "experts_used", "corpus_sha256", "sequence_sha256"):
        if len({json.dumps(m[key]) for m in metas}) != 1:
            raise SystemExit(f"FAIL stitch: the parts differ in {key}")
    if metas[0]["embeds"] or not all(m["embeds"] for m in metas[1:]):
        raise SystemExit("FAIL stitch: the first part must start from token IDs, every other from embeddings")
    tok = None
    if model_dir and (model_dir / "tokenizer.json").exists():
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    out.mkdir(parents=True, exist_ok=True)
    meta = {key: metas[0][key] for key in ("corpus", "corpus_sha256", "random", "dtype", "experts_used",
                                            "pipeline_parallel", "sequence", "sequence_sha256", "versions")}
    meta["attn_layers"] = [layers[il] for m, layers in zip(metas, chains)
                           for il in (range(len(layers)) if m["attn_layers"] is None else m["attn_layers"])]
    meta["chain"] = [{"layers": layers, "capture": str(p), "config_sha256": m["config_sha256"]}
                     for p, m, layers in zip(parts, metas, chains)]
    meta["cases"] = {}
    for name in metas[0]["cases"]:
        arrays, seconds = {}, 0.0
        for k, (p, m, layers) in enumerate(zip(parts, metas, chains)):
            entry = m["cases"][name]
            seconds += entry["seconds"]
            with np.load(p / f"{name}.npz") as z:
                a = {key: z[key] for key in z.files}
            for key, v in a.items():
                if array_sha256(v) != entry["arrays"][key]:
                    raise SystemExit(f"FAIL stitch {name}: {p.name}/{key} differs from its capture.json")
            if k == 0:
                arrays["tokens"] = a["tokens"]
                if "node.embd" in a:
                    arrays["node.embd"] = a["node.embd"]
            else:
                # the part must have been fed exactly the previous part's last l_out
                last = f"l_out-{len(chains[k - 1]) - 1}"
                want = {"node": last, "sha256": metas[k - 1]["cases"][name]["arrays"][f"node.{last}"]}
                if entry.get("embeds") != want:
                    raise SystemExit(f"FAIL stitch {name}: {p.name} was fed {entry.get('embeds')}, not the "
                                     f"{last} of {parts[k - 1].name}")
                if (a["tokens"] != arrays["tokens"]).any():
                    raise SystemExit(f"FAIL stitch {name}: {p.name} ran other tokens")
            for key in PER_LAYER:
                arrays.setdefault(key, []).append(a[key])
            for key, v in a.items():
                if key.startswith("node.") and key != "node.embd":
                    node, il = key.removeprefix("node.").rsplit("-", 1)
                    arrays[f"node.{node}-{layers[int(il)]}"] = v
            if "logits" in a:
                if k != len(parts) - 1:
                    raise SystemExit(f"FAIL stitch {name}: {p.name} has logits, only the last part should")
                arrays["logits"] = a["logits"]
        if "logits" not in arrays:
            raise SystemExit(f"FAIL stitch {name}: the last part has no logits")
        for key in PER_LAYER:
            arrays[key] = np.concatenate(arrays[key])
        np.savez(out / f"{name}.npz", **arrays)

        first = metas[0]["cases"][name]
        n0, seq = first["n_prompt"], first.get("sequence_greedy", [])
        entry = {"kind": first["kind"], "n_tokens": first["n_tokens"], "n_prompt": n0, "greedy": [],
                 "sequence_greedy": seq, "greedy_matches_prefill_argmax": None, "logprobs_max_abs_diff": None,
                 "seconds": round(seconds, 1)}
        if seq:
            top = arrays["logits"][n0 - 1:-1].argmax(axis=1).tolist()
            agree = sum(a == b for a, b in zip(top, seq))
            if top == seq:
                entry["greedy"] = seq  # vLLM's greedy decoding takes exactly these tokens
                more = ", so vLLM's greedy continuation is the sequence's" + (f": {tok.decode(seq)!r}" if tok else "")
            else:
                more = f", first differing at position {next(i for i, (a, b) in enumerate(zip(top, seq)) if a != b)}"
            print(f"INFO {name}: vLLM's argmax along the sequence is its next token at {agree} of {len(seq)} "
                  f"positions{more}", flush=True)
        entry["arrays"] = {key: array_sha256(v) for key, v in arrays.items()}
        meta["cases"][name] = entry
        print(f"INFO {name}: {len(arrays['tokens'])} tokens, {len(arrays)} arrays over {len(flat)} layers from "
              f"{len(parts)} parts, {entry['seconds']} s", flush=True)
    (out / "capture.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"INFO wrote the stitched capture of layers 0..{len(flat) - 1} to {out}")


def diff(args) -> None:
    import numpy as np

    ma = json.loads((args.a / "capture.json").read_text())
    mb = json.loads((args.b / "capture.json").read_text())
    errs = []
    for name in ma["cases"]:
        if name not in mb["cases"]:
            continue
        with np.load(args.a / f"{name}.npz") as za, np.load(args.b / f"{name}.npz") as zb:
            a = {key: za[key] for key in za.files}
            b = {key: zb[key] for key in zb.files}
        if (a["tokens"] != b["tokens"]).any():
            errs.append(name)
            print(f"FAIL {name}: the captures ran other tokens")
            continue
        common, only = sorted(set(a) & set(b)), sorted(set(a) ^ set(b))
        same, worst, bad = 0, (0.0, ""), []
        for key in common:
            if a[key].shape != b[key].shape:
                bad.append(f"{key} shape {a[key].shape} vs {b[key].shape}")
            elif np.array_equal(a[key], b[key]):
                same += 1
            elif a[key].dtype.kind in "biu":  # token IDs, expert IDs, the layer pattern
                bad.append(f"{key} differs at {int((a[key] != b[key]).sum())} entries")
            else:
                v = nmse(a[key], b[key])
                worst = max(worst, (v, key))
                if not v <= args.max_nmse:
                    bad.append(f"{key} NMSE {v:.2e}")
        if bad:
            errs.append(name)
        print(("FAIL " if bad else "PASS ") + f"{name}: {len(common)} arrays in both, {same} bit-identical, worst "
              f"NMSE {worst[0]:.2e}{f' ({worst[1]})' if worst[1] else ''}"
              + (f"; {len(bad)} over the bound: {'; '.join(bad[:4])}" if bad else "")
              + (f"; only in one: {', '.join(only)}" if only else ""))
    print(("FAIL" if errs else "PASS") + f" diff {args.a} vs {args.b}: "
          + (f"{len(errs)} case(s) differ" if errs else f"every common array within NMSE {args.max_nmse:g}"))
    sys.exit(1 if errs else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="slice, capture and stitch")
    r.add_argument("--llama-cpp", type=Path, default=ROOT / "third_party" / "llama.cpp")
    r.add_argument("--gguf", type=Path, required=True, help="the BF16 GGUF of the real checkpoint")
    r.add_argument("--model-dir", type=Path, required=True, help="its config and tokenizer files")
    r.add_argument("--vllm-python", type=Path, required=True, help="the python of vLLM's environment")
    r.add_argument("--corpus", type=Path, default=CORPUS)
    r.add_argument("--manifest", type=Path, default=MANIFEST)
    r.add_argument("--case", action="append", help="only these cases")
    r.add_argument("--n-layers", type=int, help="only the first N layers (default: all)")
    r.add_argument("--per-slice", type=int, default=2)
    r.add_argument("--attn-layers", default="0,4", help="layers whose attention nodes are kept, as e2e.py's")
    r.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    r.add_argument("--kv-cache-gib", type=float, default=1.0, help="vllm_capture.py --kv-cache-gib")
    r.add_argument("--work", type=Path, required=True, help="the slices (deleted after use) and the parts")
    r.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("stitch", help="stitch existing parts, in layer order")
    s.add_argument("--model-dir", type=Path)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("parts", type=Path, nargs="+")
    d = sub.add_parser("diff", help="compare two captures array by array")
    d.add_argument("a", type=Path)
    d.add_argument("b", type=Path)
    d.add_argument("--max-nmse", type=float, default=0.0)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)
    elif args.cmd == "stitch":
        stitch(args.parts, args.out, args.model_dir)
    else:
        diff(args)


if __name__ == "__main__":
    main()
