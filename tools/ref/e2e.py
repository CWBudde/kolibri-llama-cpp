#!/usr/bin/env python3
"""A small end-to-end corpus with stored reference artifacts for the real checkpoint.

compare_real.py --gguf recomputes the torch reference (kolibri_ref.py) on every
run. This tool stores it once, so a new llama.cpp build can be checked against
it in minutes. The corpus (testdata/e2e/corpus.json) holds six fixed cases with
their token IDs from the pinned tokenizer:

- de-raw, en-raw, code: raw prompts, each followed by the reference's 16 greedy
  tokens;
- de-chat: a German user turn through the released chat template with
  enable_thinking false (the prefilled empty think block), also followed by 16
  greedy tokens;
- wiki-c1: wikitext-2 test chunk 1 (512 tokens) from a llama-perplexity
  --kl-divergence-base file;
- de-long: the first 1024 tokens of the German text in the imatrix calibration
  text, the only case longer than the 513-token sliding window.

--write runs the reference on the BF16 GGUF in float32 and in bfloat16 (vLLM's
precision) and writes one .npz per case to the artifact directory (default
~/models/eval/e2e, outside the repo): the logits of both, the float32 run's
router logits and selected experts per layer, and the router bias. For the
prompts, the sequence is the prompt plus the float32 greedy continuation. It
records in testdata/e2e/manifest.json the sha256 of every array, the greedy
continuations, the reference's bfloat16-vs-float32 metrics, and the libllama
run it then does as --check does.

--check verifies the corpus against its sources (when they are present) and
every artifact against the manifest (FAIL, exit 1, on any difference). It then
runs libllama on the CPU on each case, with libllama's default KV cache and
flash-attention settings, and prints INFO lines against the reference in
float32 and in bfloat16: logits NMSE, KLD and same top token (for wiki-c1 and
de-long over the second half, with PPL, as llama-perplexity), router Top-1 and
Top-6 set agreement, and for the prompts libllama's greedy token along the
reference continuation. Last, per case, whether libllama's logits are
bit-identical to the recorded run, or how its metrics moved. --record makes
the current run the recorded one.

The metrics are INFO, not PASS/FAIL: the real model has no tolerance yet (PLAN
Phase 6), and the reference is a port of the vLLM code, not vLLM.

    e2e.py --llama-cpp third_party/llama.cpp --gguf ~/models/Kolibri-1-BF16.gguf --write
    e2e.py --llama-cpp third_party/llama.cpp --gguf ~/models/Kolibri-1-BF16.gguf [--record]
"""

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "gguf"))
sys.path.insert(0, str(ROOT / "tools" / "ref"))
sys.path.insert(0, str(ROOT / "tools" / "tokenizer"))
sys.path.insert(0, str(ROOT / "tools" / "chat"))
sys.path.insert(0, str(ROOT / "tools" / "quant"))

from check_model import Runner  # noqa: E402
from common import tokenizer_dir  # noqa: E402
from compare_real import GERMAN, N_GREEDY, chunk_stats, flat, logit_stats, stats_text  # noqa: E402
from router_probe import probe  # noqa: E402

CORPUS = ROOT / "testdata" / "e2e" / "corpus.json"
MANIFEST = ROOT / "testdata" / "e2e" / "manifest.json"
EVAL = Path.home() / "models" / "eval"
ARTIFACTS = EVAL / "e2e"
KLD_BASE = EVAL / "kld-bf16-c512-n20.bin"
CALIBRATION = EVAL / "kolibri-calibration.txt"
PROMPTS = {
    "de-raw": GERMAN,
    "en-raw": "The capital of Germany is",
    "code": 'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
}
CHAT = [{"role": "user", "content": "Erkläre in zwei Sätzen, was ein Mixture-of-Experts-Modell ist."}]
CHAT_VARIABLES = {"enable_thinking": False}
N_LONG = 1024
N_CTX = 1024


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def array_sha256(a) -> str:
    """sha256 of an array's dtype, shape and bytes (an .npz file itself carries timestamps)."""
    import numpy as np

    a = np.ascontiguousarray(a)
    return hashlib.sha256(f"{a.dtype.str}{a.shape}".encode() + a.tobytes()).hexdigest()


def build_corpus(tok, kld_base: Path, calibration: Path) -> dict:
    """The corpus from its sources. A case whose source file is missing is None."""
    import numpy as np

    from calibration import blocks
    from check_chat import render_reference

    def encode(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False).ids

    cases = {name: {"kind": "prompt", "source": "text", "text": text, "tokens": encode(text), "n_greedy": N_GREEDY}
             for name, text in PROMPTS.items()}
    template = json.loads((tokenizer_dir() / "tokenizer_config.json").read_text())["chat_template"]
    text = render_reference(template, CHAT, None, CHAT_VARIABLES)
    cases["de-chat"] = {"kind": "prompt", "source": f"chat template, {CHAT_VARIABLES}", "messages": CHAT,
                        "text": text, "tokens": encode(text), "n_greedy": N_GREEDY}
    cases["wiki-c1"] = cases["de-long"] = None
    sources = {}
    if kld_base.exists():
        raw = kld_base.read_bytes()
        assert raw[:8] == b"_logits_", f"{kld_base}: not a llama-perplexity --kl-divergence-base file"
        n_ctx = struct.unpack("<i", raw[8:12])[0]
        tokens = np.frombuffer(raw, dtype=np.int32, count=n_ctx, offset=20).tolist()
        sources[kld_base.name] = sha256(kld_base)
        cases["wiki-c1"] = {"kind": "chunk", "source": f"{kld_base.name}, chunk 1", "text": tok.decode(tokens),
                            "tokens": tokens}
    if calibration.exists():
        # the calibration text interleaves English, German and code blocks; blocks 1 and 4 are the first two
        # German ones, consecutive in the German source
        b = blocks(calibration.read_text())
        tokens = encode(b[1] + b[4])[:N_LONG]
        sources[calibration.name] = sha256(calibration)
        cases["de-long"] = {"kind": "chunk", "source": f"{calibration.name}, German blocks 1 and 4, first {N_LONG} "
                            "tokens", "text": tok.decode(tokens), "tokens": tokens}
    return {"sources": sources, "cases": cases}


def sequence(case: dict, entry: dict | None) -> list[int]:
    """The evaluated tokens: a chunk, or a prompt plus the reference's float32 greedy continuation."""
    return case["tokens"] + (entry["greedy_float32"] if case["kind"] == "prompt" else [])


def stats(case: dict, base, test, tokens: list[int]) -> dict:
    """chunk_stats over the second half for a chunk, logit_stats over every position for a prompt."""
    if case["kind"] == "chunk":
        return chunk_stats(base, test, tokens, len(tokens) // 2)
    return logit_stats(base, test)


def text_of(s: dict) -> str:
    ppl = f"PPL {s['ppl_test']:.4f} vs {s['ppl_base']:.4f}, " if "ppl_base" in s else ""
    return ppl + stats_text(s)


def write_reference(gguf: Path, llama_cpp: Path, case: dict) -> tuple[dict, dict]:
    """The reference arrays of a case and its manifest entry (without the libllama run)."""
    import numpy as np

    from kolibri_ref import Weights, forward, greedy

    W = Weights(gguf, llama_cpp)
    W.cache = {}
    entry, t0 = {}, time.time()
    if case["kind"] == "prompt":
        entry["greedy_float32"] = greedy(W, case["tokens"], case["n_greedy"])
        entry["greedy_bfloat16"] = greedy(W, case["tokens"], case["n_greedy"], "bfloat16")
    tokens = sequence(case, entry)
    dump = {}
    arrays = {"tokens": np.asarray(tokens, np.int32), "ref_float32": forward(W, tokens, "float32", dump).numpy()}
    arrays["ref_bfloat16"] = forward(W, tokens, "bfloat16").numpy()
    arrays["router_logits"] = np.stack([dump[f"ffn_moe_logits-{il}"] for il in range(W.n_layer)])
    arrays["router_topk"] = np.stack([dump[f"ffn_moe_topk-{il}"] for il in range(W.n_layer)]).astype(np.int32)
    arrays["router_bias"] = np.stack([W.get(f"blk.{il}.exp_probs_b.bias").float().numpy() for il in range(W.n_layer)])
    s = stats(case, arrays["ref_float32"], arrays["ref_bfloat16"], tokens)
    entry["reference_bfloat16_vs_float32"] = s
    entry["reference_seconds"] = round(time.time() - t0)
    entry["arrays"] = {name: array_sha256(a) for name, a in sorted(arrays.items())}
    return arrays, entry


def run_libllama(runner: Runner, gguf: Path, threads: int, case: dict, arrays) -> tuple[dict, list[str]]:
    """libllama on the CPU against the stored reference: the record of the run and its INFO lines."""
    import numpy as np

    tokens = arrays["tokens"].tolist()
    n_layer = arrays["router_topk"].shape[0]
    capture = {f"{node}-{il}": None for il in range(n_layer) for node in ("ffn_moe_logits", "ffn_moe_topk")}
    _, cpu = runner.devices()[0]
    t0 = time.time()
    got = runner.logits(gguf, cpu, tokens, len(tokens), capture=capture, n_ctx=N_CTX, n_threads=threads,
                        n_threads_batch=threads)
    seconds = time.time() - t0
    nodes = flat(capture, len(tokens))
    p = probe(np.stack([nodes[f"ffn_moe_logits-{il}"] for il in range(n_layer)]),
              np.stack([nodes[f"ffn_moe_topk-{il}"] for il in range(n_layer)]),
              arrays["router_logits"], arrays["router_topk"], arrays["router_bias"])
    rec = {
        "gguf": gguf.name,
        "logits_sha256": array_sha256(got.astype(np.float32)),
        "vs_float32": stats(case, arrays["ref_float32"], got, tokens),
        "vs_bfloat16": stats(case, arrays["ref_bfloat16"], got, tokens),
        "router_top1_same": float(p["top1_same"].mean()),
        "router_set_same": float(p["set_same"].mean()),
        "non_finite": int((~np.isfinite(got)).sum()),
    }
    k = arrays["router_topk"].shape[-1]
    router = f"router Top-1 same {rec['router_top1_same']:.2%}, Top-{k} set same {rec['router_set_same']:.2%}"
    if case["kind"] == "prompt":
        n = len(case["tokens"])
        cont = np.asarray(tokens[n:])
        rec["greedy_equal"] = int((got[n - 1:-1].argmax(1) == cont).sum())
        router += f"; libllama greedy token along the reference continuation: equal at {rec['greedy_equal']}/{len(cont)}"
    lines = [f"libllama vs reference float32: {text_of(rec['vs_float32'])}; {router} ({seconds:.0f} s)",
             f"libllama vs reference bfloat16: {text_of(rec['vs_bfloat16'])}"]
    if rec["non_finite"]:
        lines.append(f"libllama: {rec['non_finite']} non-finite logits")
    return rec, lines


def moved(old: dict, new: dict) -> str:
    """The metrics of new next to old."""
    parts = []
    for side in ("vs_float32", "vs_bfloat16"):
        for key, fmt in (("kld", ".6f"), ("same_top", ".1%"), ("nmse", ".2e"), ("ppl_test", ".4f")):
            if key in new[side]:
                parts.append(f"{side} {key} {new[side][key]:{fmt}} (recorded {old[side][key]:{fmt}})")
    for key in ("router_top1_same", "router_set_same"):
        parts.append(f"{key} {new[key]:.2%} (recorded {old[key]:.2%})")
    if "greedy_equal" in new:
        parts.append(f"greedy_equal {new['greedy_equal']} (recorded {old['greedy_equal']})")
    return ", ".join(parts)


def load_arrays(path: Path):
    import numpy as np

    with np.load(path) as z:
        return {name: z[name] for name in z.files}


def main() -> None:
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True)
    ap.add_argument("--gguf", type=Path, required=True, help="the GGUF libllama runs (and --write's reference reads)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="compute and store the reference artifacts (slow)")
    mode.add_argument("--record", action="store_true", help="make this libllama run the recorded one")
    ap.add_argument("--cases", help="comma-separated case names (default: all)")
    ap.add_argument("--artifacts", type=Path, default=ARTIFACTS, help=f"artifact directory (default: {ARTIFACTS})")
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--kld-base", type=Path, default=KLD_BASE)
    ap.add_argument("--calibration", type=Path, default=CALIBRATION)
    ap.add_argument("--threads", type=int, default=10, help="libllama CPU threads")
    args = ap.parse_args()

    errs = []

    def fail(what: str) -> None:
        print("FAIL " + what)
        errs.append(what)

    tok = Tokenizer.from_file(str(tokenizer_dir() / "tokenizer.json"))
    built = build_corpus(tok, args.kld_base, args.calibration)
    if args.write:
        missing = [name for name, c in built["cases"].items() if c is None]
        if missing:
            raise SystemExit(f"--write: sources missing for {', '.join(missing)}")
        args.corpus.parent.mkdir(parents=True, exist_ok=True)
        args.corpus.write_text(json.dumps(built, indent=1, ensure_ascii=False) + "\n")
    corpus = json.loads(args.corpus.read_text())
    names = args.cases.split(",") if args.cases else list(corpus["cases"])
    unknown = set(names) - set(corpus["cases"])
    if unknown:
        raise SystemExit(f"unknown cases: {', '.join(sorted(unknown))}")

    # the corpus against its sources
    for name, src in corpus["sources"].items():
        if name in built["sources"] and built["sources"][name] != src:
            fail(f"corpus source {name}: sha256 {built['sources'][name]} (corpus {src})")
    for name in names:
        case, now = corpus["cases"][name], built["cases"][name]
        if now is None:
            print(f"SKIP corpus/{name}: its source is not here ({case['source']})")
        elif now["tokens"] != case["tokens"]:
            i = next((i for i, (a, b) in enumerate(zip(now["tokens"], case["tokens"])) if a != b),
                     min(len(now["tokens"]), len(case["tokens"])))
            fail(f"corpus/{name}: token IDs differ from its source at position {i} "
                 f"({len(now['tokens'])} vs {len(case['tokens'])} tokens)")
        else:
            print(f"PASS corpus/{name}: {len(case['tokens'])} token IDs as from its source")

    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else {"cases": {}}
    runner = Runner(args.llama_cpp)
    if args.write:
        commit = subprocess.run(["git", "-C", str(args.llama_cpp), "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        manifest["reference"] = {"gguf": args.gguf.name, "gguf_bytes": args.gguf.stat().st_size,
                                 "code": "tools/ref/kolibri_ref.py", "dtypes": ["float32", "bfloat16"]}
        manifest["libllama"] = {"llama_cpp": commit, "device": "CPU", "threads": args.threads, "n_ctx": N_CTX,
                                "context": "libllama defaults (KV cache type, flash attention)"}
        args.artifacts.mkdir(parents=True, exist_ok=True)

    for name in names:
        case = corpus["cases"][name]
        path = args.artifacts / f"{name}.npz"
        if args.write:
            import numpy as np

            arrays, entry = write_reference(args.gguf, args.llama_cpp, case)
            np.savez(path, **arrays)
            manifest["cases"][name] = entry
            print(f"INFO {name}: reference written to {path} ({entry['reference_seconds']} s), bfloat16 vs "
                  f"float32: {text_of(entry['reference_bfloat16_vs_float32'])}")
        entry = manifest["cases"].get(name)
        if entry is None:
            fail(f"{name}: not in {args.manifest.name}; record it with --write")
            continue
        if not path.exists():
            fail(f"{name}: {path} is missing; recreate it with --write")
            continue
        try:
            arrays = load_arrays(path)
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            fail(f"{name}: {path.name} is unreadable ({e}); recreate it with --write")
            continue
        bad = [k for k in sorted(set(arrays) | set(entry["arrays"]))
               if k not in arrays or array_sha256(arrays[k]) != entry["arrays"].get(k)]
        if bad:
            fail(f"{name}: {', '.join(bad)} in {path.name} differ from the manifest's sha256")
            continue
        if arrays["tokens"].tolist() != sequence(case, entry):
            fail(f"{name}: the artifact's tokens differ from the corpus")
            continue
        n = len(arrays["tokens"])
        print(f"PASS {name}: {n} tokens, artifacts match the manifest; reference bfloat16 vs float32: "
              f"{text_of(entry['reference_bfloat16_vs_float32'])}")

        rec, lines = run_libllama(runner, args.gguf, args.threads, case, arrays)
        for line in lines:
            print(f"INFO {name}, {line}")
        old = entry.get("libllama")
        if args.write or args.record:
            entry["libllama"] = rec
            print(f"INFO {name}: recorded this libllama run")
        elif old is None:
            print(f"INFO {name}: no recorded libllama run; record one with --record")
        elif old["logits_sha256"] == rec["logits_sha256"]:
            print(f"INFO {name}: unchanged, logits bit-identical to the recorded libllama run ({old['gguf']})")
        else:
            gguf = f"; recorded with {old['gguf']}, this run {rec['gguf']}" if old["gguf"] != rec["gguf"] else ""
            print(f"INFO {name}: CHANGED since the recorded libllama run{gguf}: {moved(old, rec)}")

    if args.write or args.record:
        if args.record:
            commit = subprocess.run(["git", "-C", str(args.llama_cpp), "rev-parse", "HEAD"], capture_output=True,
                                    text=True).stdout.strip()
            manifest["libllama"].update(llama_cpp=commit, threads=args.threads)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"INFO manifest written to {args.manifest}")
    if errs:
        print(f"FAIL {len(errs)} problem(s)")
        sys.exit(1)


if __name__ == "__main__":
    main()
