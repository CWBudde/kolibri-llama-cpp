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
router logits and selected experts per layer, and the router bias. For
wiki-c1 and de-long it also stores a compact set of the float32 run's nodes
(REF_NODES): the attention and expert outputs of the first sliding and the
first full layer and every layer's output l_out, and for wiki-c1 the token
embeddings and every layer's expert outputs too. For the prompts, the sequence
is the prompt plus the float32 greedy continuation. It records in
testdata/e2e/manifest.json the sha256 of every array, the greedy
continuations, the reference's bfloat16-vs-float32 metrics, and the libllama
run it then does as the default mode does.

The default mode verifies the corpus against its sources (when they are
present) and every artifact against the manifest (FAIL, exit 1, on any
difference). It then runs libllama on the CPU on each case, with libllama's
default KV cache and flash-attention settings, captures every node (the token
embeddings and, per layer, Q and K after the QK norm, the attention output
before and after its sandwich norm, the router logits and selected experts,
the routed and shared expert output and l_out) and prints INFO lines against
the reference in float32 and in bfloat16: logits NMSE, KLD and same top token
(for wiki-c1 and de-long over the second half, with PPL, as
llama-perplexity), router Top-1 and Top-6 set agreement, for the prompts
libllama's greedy token along the reference continuation, and where nodes are
stored, their NMSE (compare_real.node_lines) and l_out's.

Last, per case, the gate: with the recorded GGUF and thread count, every
captured node and the logits must be bit-identical to the recorded libllama
run, or the case fails and names the first changed node in graph order. With
another GGUF or thread count, the comparison is INFO. --record makes the
current run the recorded one, after a change has been reviewed. --override
KEY=VALUE changes a GGUF metadata value for libllama, as a negative control.

The metrics against the reference are INFO, not PASS/FAIL: the real model has
no tolerance yet (PLAN Phase 6), and the reference is a port of the vLLM code,
not vLLM.

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

from check_model import Runner, nmse  # noqa: E402
from common import tokenizer_dir  # noqa: E402
from compare_real import (EMBD, GERMAN, N_GREEDY, capture_names, chunk_stats, flat, logit_stats,  # noqa: E402
                          node_lines, stats_text)
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
# the order of a layer's nodes in the graph, after the token embeddings (embd)
ORDER = ("Qcur_normed", "Kcur_normed", "attn_out", "attn_post_norm", "ffn_moe_logits", "ffn_moe_topk",
         "ffn_moe_out", "ffn_shexp", "l_out")
ATTN = ("Qcur_normed", "Kcur_normed", "attn_out", "attn_post_norm")
EXPERTS = ("ffn_moe_out", "ffn_shexp")
# the float32 reference nodes --write stores, per case: whether the token embeddings, the nodes in the
# first sliding and the first full layer, and the nodes in every layer
REF_NODES = {
    "wiki-c1": (True, ATTN + EXPERTS, EXPERTS + ("l_out",)),
    "de-long": (False, ATTN + EXPERTS, ("l_out",)),  # past the window
}


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


def ref_node_names(name: str, is_swa: list[bool]) -> list[str]:
    """The reference nodes stored for a case, in graph order (none for a case not in REF_NODES)."""
    if name not in REF_NODES:
        return []
    embd, first, every = REF_NODES[name]
    firsts = (is_swa.index(True), is_swa.index(False))
    names = [EMBD] if embd else []
    for il in range(len(is_swa)):
        names += [f"{node}-{il}" for node in ORDER if node in every or (il in firsts and node in first)]
    return names


def graph_key(name: str) -> tuple[int, int]:
    """Sort key of a captured node in graph order."""
    if name == EMBD:
        return -1, 0
    node, il = name.rsplit("-", 1)
    return int(il), ORDER.index(node)


def first_changed(old: dict, new: dict) -> str | None:
    """The first node in graph order whose hash differs between two libllama records (or the logits),
    None if they are bit-identical. A record without node hashes compares only the logits."""
    if "nodes_sha256" in old:
        a, b = old["nodes_sha256"], new["nodes_sha256"]
        for name in sorted(set(a) | set(b), key=graph_key):
            if a.get(name) != b.get(name):
                return name if name in a and name in b else f"{name} (only in one run)"
    return None if old["logits_sha256"] == new["logits_sha256"] else "the logits"


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


def write_reference(gguf: Path, llama_cpp: Path, name: str, case: dict) -> tuple[dict, dict]:
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
    is_swa = [bool(x) for x in W.is_swa]
    nodes = ref_node_names(name, is_swa)
    if nodes:
        arrays["is_swa"] = np.asarray(is_swa)
        arrays.update({f"node.{node}": dump[node] for node in nodes})
    del dump
    s = stats(case, arrays["ref_float32"], arrays["ref_bfloat16"], tokens)
    entry["reference_bfloat16_vs_float32"] = s
    entry["reference_seconds"] = round(time.time() - t0)
    entry["arrays"] = {name: array_sha256(a) for name, a in sorted(arrays.items())}
    return arrays, entry


def run_libllama(runner: Runner, gguf: Path, threads: int, case: dict, arrays,
                 overrides: dict | None = None) -> tuple[dict, list[str]]:
    """libllama on the CPU against the stored reference: the record of the run and its INFO lines.
    It captures every node of compare_real.capture_names and records the sha256 of each."""
    import types

    import numpy as np

    tokens = arrays["tokens"].tolist()
    n_layer = arrays["router_topk"].shape[0]
    capture = capture_names(n_layer)
    _, cpu = runner.devices()[0]
    t0 = time.time()
    got = runner.logits(gguf, cpu, tokens, len(tokens), overrides=overrides, capture=capture, n_ctx=N_CTX,
                        n_threads=threads, n_threads_batch=threads)
    seconds = time.time() - t0
    nodes = flat(capture, len(tokens))
    del capture
    p = probe(np.stack([nodes[f"ffn_moe_logits-{il}"] for il in range(n_layer)]),
              np.stack([nodes[f"ffn_moe_topk-{il}"] for il in range(n_layer)]),
              arrays["router_logits"], arrays["router_topk"], arrays["router_bias"])
    rec = {
        "gguf": gguf.name,
        "gguf_bytes": gguf.stat().st_size,
        "threads": threads,
        "logits_sha256": array_sha256(got.astype(np.float32)),
        "nodes_sha256": {name: array_sha256(nodes[name]) for name in sorted(nodes, key=graph_key)},
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
    if "is_swa" in arrays:  # the stored reference nodes
        ref = {name[len("node."):]: a for name, a in arrays.items() if name.startswith("node.")}
        ref.update({f"ffn_moe_topk-{il}": arrays["router_topk"][il] for il in range(n_layer)})
        is_swa = arrays["is_swa"].tolist()
        lines += [f"libllama vs reference float32, {text}"
                  for _, text in node_lines(nodes, ref, types.SimpleNamespace(n_layer=n_layer, is_swa=is_swa))]
        lout = [nmse(ref[f"l_out-{il}"], nodes[f"l_out-{il}"]) for il in range(n_layer)]
        full, top = is_swa.index(False), int(np.argmax(lout))
        lines.append(f"libllama vs reference float32, layer output l_out over all tokens: NMSE {lout[0]:.2e} "
                     f"(layer 0), {lout[full]:.2e} (layer {full}), {lout[top]:.2e} (layer {top}, the worst), "
                     f"{lout[-1]:.2e} (layer {n_layer - 1})")
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
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                    help="a GGUF metadata override for libllama (int, float or true/false), for negative controls")
    args = ap.parse_args()
    overrides = {}
    for o in args.override:
        key, _, val = o.partition("=")
        overrides[key] = (val == "true" if val in ("true", "false") else
                          int(val) if val.lstrip("-").isdigit() else float(val))
    if overrides and (args.write or args.record):
        raise SystemExit("--override changes the model: it cannot be written or recorded")

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

            arrays, entry = write_reference(args.gguf, args.llama_cpp, name, case)
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

        rec, lines = run_libllama(runner, args.gguf, args.threads, case, arrays, overrides)
        del arrays
        for line in lines:
            print(f"INFO {name}, {line}")
        old = entry.get("libllama")
        if args.write or args.record:
            entry["libllama"] = rec
            print(f"INFO {name}: recorded this libllama run ({len(rec['nodes_sha256'])} nodes and the logits)")
            continue
        if old is None:
            print(f"INFO {name}: no recorded libllama run; record one with --record")
            continue
        changed = first_changed(old, rec)
        # the gate: the same GGUF with the same thread count must give the recorded bits
        other = [f"recorded with {old[k]}, this run {rec[k]}" for k in ("gguf", "gguf_bytes", "threads")
                 if old.get(k, manifest.get("libllama", {}).get(k, rec[k])) != rec[k]]
        nodes = f"{len(old['nodes_sha256'])} nodes and the logits" if "nodes_sha256" in old else "the logits"
        if other:
            state = "unchanged" if changed is None else f"CHANGED (first changed node {changed}): {moved(old, rec)}"
            print(f"INFO {name}, not gated ({'; '.join(other)}): {state}")
        elif changed is None:
            print(f"PASS {name}: bit-identical to the recorded libllama run ({nodes})")
        else:
            fail(f"{name}: changed since the recorded libllama run, first changed node {changed}; {moved(old, rec)}; "
                 "accept a reviewed change with --record")

    if args.record and errs:
        # a partly recorded baseline would report later regressions as unchanged
        print(f"INFO {args.manifest.name} not written: --record records nothing after a FAIL")
    elif args.write or args.record:
        # --write keeps writing: the manifest must describe the artifacts it already stored
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
