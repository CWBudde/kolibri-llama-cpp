#!/usr/bin/env python3
"""The experts each locality workload selects, per layer and token (PLAN Phase 8).

Prefills a workload's token IDs (tools/locality/workloads.py writes them to
~/models/eval/locality/<name>.tokens.npy, every answer as generated) through
libllama and captures every layer's selected experts (the ffn_moe_topk node).
The traces are teacher-forced, so one prefill gives the experts the
generations used. It writes ~/models/eval/locality/<name>.<gguf stem>.experts.npy,
int16 [n_layer, n_tokens, n_expert_used] (cmd/kolibri-locality reads it), and
fails (exit 1) when:

- the token IDs differ from testdata/locality/manifest.json;
- a token's selection in some layer is not n_expert_used distinct experts in
  [0, n_expert);
- the sha256 of the selections (their int16 little-endian bytes) differs from
  testdata/locality/experts.json, which --record writes.

A capture replaces the stored selections only once it passes (or, with
--record, once every requested workload is valid); a rejected one is kept as
<name>.<gguf stem>.experts.rejected.npy. The manifest records per workload the
run that captured it (GGUF, llama.cpp commit, device, batch, threads).

--verify re-hashes the stored selection files without running the model.

    experts.py --gguf ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf --device metal [--record]
    experts.py --gguf ~/models/Kolibri-1-Q8_0.gguf --device cpu --no-repack --threads 10 [--record]
    experts.py --gguf ~/models/Kolibri-1-Q8_0.gguf --verify
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "gguf"))

from workloads import MANIFEST, OUT, WORKLOADS, ids_sha256  # noqa: E402

EXPERTS = ROOT / "testdata" / "locality" / "experts.json"
N_UBATCH = 512
N_CTX = 32768


def selections_path(name: str, gguf: Path) -> Path:
    return OUT / f"{name}.{gguf.stem}.experts.npy"


def selections_sha256(sel) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(sel, dtype="<i2").tobytes()).hexdigest()


def shape(llama_cpp: Path, gguf: Path) -> tuple[int, int, int]:
    """n_layer, n_expert and n_expert_used from the GGUF's metadata."""
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    from gguf import GGUFReader

    fields = GGUFReader(gguf).fields
    arch = fields["general.architecture"].contents()
    keys = ("block_count", "expert_count", "expert_used_count")
    return tuple(int(fields[f"{arch}.{key}"].contents()) for key in keys)


def capture(runner, gguf: Path, dev, tokens: list[int], n_layer: int, args):
    """The selected experts, int16 [n_layer, n_tokens, n_expert_used], from one prefill in N_UBATCH chunks."""
    import numpy as np

    from check_model import joined

    names = [f"ffn_moe_topk-{il}" for il in range(n_layer)]
    cap = dict.fromkeys(names)
    chunks = [N_UBATCH] * (len(tokens) // N_UBATCH) + ([len(tokens) % N_UBATCH] if len(tokens) % N_UBATCH else [])
    ctx = {"n_threads": args.threads, "n_threads_batch": args.threads} if args.threads else {}
    # every position is an output, or the last layer would compute only the output rows
    runner.logits(gguf, dev, tokens, N_UBATCH, capture=cap, chunks=chunks, n_ctx=N_CTX, return_logits=False,
                  no_repack=args.no_repack, **ctx)
    # each part is [1, 1, n_tokens, n_expert_used]
    return np.stack([joined(cap[name]).reshape(len(tokens), -1) for name in names]).astype(np.int16)


def invalid(sel, n_expert: int) -> str | None:
    """Why a selection array is not one set of distinct experts per layer and token, or None."""
    import numpy as np

    if sel.min() < 0 or sel.max() >= n_expert:
        return f"expert IDs outside [0, {n_expert}): {sel.min()} to {sel.max()}"
    if (dup := (np.diff(np.sort(sel, axis=-1), axis=-1) == 0).any(axis=-1)).any():
        il, t = map(int, np.argwhere(dup)[0])
        return f"layer {il} token {t} selects an expert twice: {sel[il, t].tolist()}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, default=ROOT / "third_party" / "llama.cpp")
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "metal"), default="metal")
    ap.add_argument("--no-repack", action="store_true", help="keep the CPU weights mmapped (Q8_0 on 48 GB)")
    ap.add_argument("--threads", type=int, help="CPU threads (libllama's default otherwise)")
    ap.add_argument("--workloads", type=lambda s: s.split(","), default=list(WORKLOADS))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--record", action="store_true", help="make these selections the recorded ones")
    mode.add_argument("--verify", action="store_true", help="re-hash the stored selections, run nothing")
    args = ap.parse_args()
    import numpy as np

    tokens_manifest = json.loads(MANIFEST.read_text())["workloads"]
    manifest = json.loads(EXPERTS.read_text()) if EXPERTS.exists() else {}
    recorded = manifest.get(args.gguf.stem, {}).get("workloads", {})
    n_layer, n_expert, n_used = shape(args.llama_cpp, args.gguf)
    errs = []

    def fail(what: str) -> None:
        print("FAIL " + what, flush=True)
        errs.append(what)

    runner = dev = None
    if not args.verify:
        from check_model import Runner

        runner = Runner(args.llama_cpp)
        kind = {"cpu": runner.lib.GGML_BACKEND_DEVICE_TYPE_CPU, "metal": runner.lib.GGML_BACKEND_DEVICE_TYPE_GPU}
        kind = kind[args.device]
        dev_name, dev = next((n, d) for n, d in runner.devices() if runner.lib.ggml_backend_dev_type(d) == kind)
    if not args.verify:
        commit = subprocess.run(["git", "-C", str(args.llama_cpp), "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        provenance = {"gguf": args.gguf.name, "gguf_bytes": args.gguf.stat().st_size, "llama.cpp": commit,
                      "device": dev_name, "n_ubatch": N_UBATCH, "n_ctx": N_CTX, "threads": args.threads,
                      "no_repack": args.no_repack}
    accepted = {}  # name: (selections, manifest entry), written only once accepted
    for name in args.workloads:
        path = selections_path(name, args.gguf)
        if args.verify:
            if not path.exists():
                fail(f"{name}: no {path.name} (run without --verify)")
                continue
            sel = np.load(path)
        else:
            tokens = np.load(OUT / f"{name}.tokens.npy").tolist()
            if ids_sha256(tokens) != tokens_manifest[name]["tokens_sha256"]:
                fail(f"{name}: the token IDs differ from {MANIFEST.relative_to(ROOT)} (run workloads.py)")
                continue
            t0 = time.monotonic()
            sel = capture(runner, args.gguf, dev, tokens, n_layer, args)
            seconds = time.monotonic() - t0
            print(f"{name}: {len(tokens)} tokens in {seconds:.0f} s ({len(tokens) / seconds:.1f} tokens/s) on "
                  f"{dev_name}", flush=True)
        n_tokens = tokens_manifest[name]["n_tokens"]
        entry = {"n_tokens": int(sel.shape[1]), "experts_sha256": selections_sha256(sel)}
        if sel.shape != (n_layer, n_tokens, n_used):
            fail(f"{name}: selections of shape {sel.shape}, not {(n_layer, n_tokens, n_used)}")
        elif why := invalid(sel, n_expert):
            fail(f"{name}: {why}")
        elif args.record:
            accepted[name] = sel, {**entry, **provenance}
            continue
        elif {k: recorded.get(name, {}).get(k) for k in entry} != entry:
            fail(f"{name}: {entry} differs from the recorded "
                 f"{ {k: recorded[name][k] for k in entry} if name in recorded else None}")
        else:
            print(f"PASS {name}: {n_layer} layers x {entry['n_tokens']} tokens x {n_used} distinct experts, "
                  f"as recorded ({args.gguf.name})", flush=True)
            if not args.verify:
                accepted[name] = sel, None
            continue
        # a rejected capture never replaces the stored selections; it is kept beside them for inspection
        if not args.verify:
            rejected = path.with_suffix(".rejected.npy")
            np.save(rejected, sel)
            print(f"{name}: kept the rejected capture as {rejected.name}; {path.name} is unchanged", flush=True)
    if args.record and errs:
        print("recorded nothing: every workload must pass to be recorded", flush=True)
    elif accepted:
        for name, (sel, entry) in accepted.items():
            np.save(selections_path(name, args.gguf), sel)
            if entry is not None:
                manifest.setdefault(args.gguf.stem, {}).setdefault("workloads", {})[name] = entry
        if args.record:
            EXPERTS.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
            print(f"recorded {', '.join(accepted)} in {EXPERTS.relative_to(ROOT)}")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
