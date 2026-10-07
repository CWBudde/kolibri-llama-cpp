#!/usr/bin/env python3
"""Reproduce the upstream llama.cpp issues this port ran into, against any
llama.cpp tree with a shared-library build in build/bin (typically a current
ggml-org master):

  utf8              invalid UTF-8 through llama_tokenize, with upstream's own
                    BPE vocab models/ggml-vocab-qwen2.gguf: U+110000 aborts the
                    process, an overlong "/" is tokenized as "/", and a UTF-16
                    surrogate passes through as its raw bytes. Each case runs
                    in a forked child, so the abort is reported, not fatal;
  tokenizer-random  tests/test-tokenizer-random.py itself: LibLlamaModel passes
                    a llama_model * to llama_tokenize, which takes a
                    llama_vocab *; under transformers 5, TokenizerGroundtruth's
                    vocab and added-token lists collapse into one string each
                    (shown with the pinned Kolibri tokenizer; the cause is
                    batch_decode, not the model);
  cpu-moe           -ngl 99 --cpu-moe with a GGUF larger than the Metal working
                    set: Metal maps the file from its first to its last
                    offloaded tensor, routed experts included, and the warmup
                    decode either dies with SIGBUS or fails after a Metal
                    out-of-memory error. The tree must load the GGUF (for
                    Kolibri: patches 0001-0008); --expect ok checks a build with
                    the fix (patch 0009) instead.

Each check prints REPRODUCED or NOT REPRODUCED (with --expect ok: RUNS or
FAILS). The exit code is 0 only if every check met its expectation, so an
upstream fix turns the run red.

    repro.py utf8 --llama-cpp DIR
    repro.py tokenizer-random --llama-cpp DIR
    repro.py cpu-moe --llama-cpp DIR --gguf FILE [--expect sigbus|ok]
"""

import argparse
import multiprocessing
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tokenizer"))
from common import libllama, load_upstream, tokenizer_dir  # noqa: E402
from compare import Llama  # noqa: E402

VOCAB = Path("models") / "ggml-vocab-qwen2.gguf"


class Checks:
    def __init__(self):
        self.ok = True

    def report(self, met: bool, issue: str, observed: str, yes="REPRODUCED", no="NOT REPRODUCED"):
        self.ok &= met
        print(f"{yes if met else no} {issue}: {observed}")


def _encode_child(llama: Llama, raw: bytes, send) -> None:
    ids = llama.encode_bytes(raw)
    send.send((ids, llama.decode_bytes(ids)))


def encode_forked(llama: Llama, raw: bytes):
    """(ids, detokenized bytes), or the negative signal number if the child died."""
    ctx = multiprocessing.get_context("fork")
    recv, send = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_encode_child, args=(llama, raw, send))
    child.start()
    child.join()
    return recv.recv() if child.exitcode == 0 else child.exitcode


def run_utf8(tree: Path, c: Checks) -> None:
    llama = Llama(load_upstream(tree), tree, tree / VOCAB)

    got = encode_forked(llama, b"\xf4\x90\x80\x80")
    c.report(got == -signal.SIGABRT, "utf8 abort",
             "F4 90 80 80 (U+110000) " + (f"killed the process with signal {-got}" if isinstance(got, int)
                                          else f"returned {got[0]}"))

    slash = encode_forked(llama, b"/")
    got = encode_forked(llama, b"\xc0\xaf")
    if isinstance(got, int):
        c.report(False, "utf8 overlong", f"C0 AF killed the process with signal {-got}")
    else:
        c.report(got[0] == slash[0], "utf8 overlong", f"C0 AF gives {got[0]}, '/' gives {slash[0]}")

    got = encode_forked(llama, b"\xed\xa0\x80")
    if isinstance(got, int):
        c.report(False, "utf8 surrogate", f"ED A0 80 killed the process with signal {-got}")
    else:
        c.report(got[1] == b"\xed\xa0\x80", "utf8 surrogate",
                 f"ED A0 80 gives {got[0]}, detokenized {got[1]!r}")


def run_tokenizer_random(tree: Path, c: Checks) -> None:
    import transformers

    up = load_upstream(tree)
    model = up.LibLlamaModel(libllama(tree, up), str(tree / VOCAB),
                             mparams=dict(vocab_only=True), cparams=dict(n_ctx=4096))
    try:
        ids = model.tokenize("Hello")
        c.report(False, "test-tokenizer-random LibLlamaModel", f"tokenize('Hello') returned {ids}")
    except TypeError as e:
        c.report(True, "test-tokenizer-random LibLlamaModel", f"tokenize('Hello') raised TypeError: {e}")

    truth = up.TokenizerGroundtruth(str(tokenizer_dir()))
    n_vocab, n_added = len(truth.model.get_vocab()), len(truth.model.added_tokens_encoder)
    c.report(len(truth.vocab) == 1 and n_vocab > 1, "test-tokenizer-random TokenizerGroundtruth",
             f"transformers {transformers.__version__}: {len(truth.vocab)} vocab entry for {n_vocab} tokens, "
             f"{len(truth.added_tokens)} added-token entry for {n_added} added tokens")


def run_cpu_moe(tree: Path, gguf: Path, expect: str, c: Checks) -> None:
    cmd = [str(tree / "build" / "bin" / "llama-completion"), "-m", str(gguf), "-ngl", "99", "--cpu-moe",
           "--no-repack", "-c", "512", "-n", "1", "-p", "Hallo", "-no-cnv", "-v"]
    print("$ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       errors="replace", env={**os.environ, "GGML_NO_BACKTRACE": "1"})
    mapped = re.findall(r"MTL0_Mapped model buffer size\s*=\s*([\d.]+) MiB", p.stdout)
    oom = "kIOGPUCommandBufferCallbackErrorOutOfMemory" in p.stdout
    size = f"GGUF {gguf.stat().st_size / 2**20:,.0f} MiB, Metal maps {sum(map(float, mapped)):,.0f} MiB of it"
    how = f"killed by signal {-p.returncode}" if p.returncode < 0 else f"exit {p.returncode}"
    how += ", Metal out of memory" if oom else ""
    if expect == "ok":
        c.report(p.returncode == 0 and not oom, "cpu-moe", f"{size}: {how}", yes="RUNS", no="FAILS")
    else:
        # The same cause ends in either of two ways: the CPU expert matmul dies
        # with SIGBUS, or decode fails after the Metal command buffer's OOM.
        c.report(p.returncode == -signal.SIGBUS or (p.returncode != 0 and oom), "cpu-moe", f"{size}: {how}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=["utf8", "tokenizer-random", "cpu-moe"])
    ap.add_argument("--llama-cpp", type=Path, required=True, help="llama.cpp tree with a shared build in build/bin")
    ap.add_argument("--gguf", type=Path, help="cpu-moe: a GGUF larger than the Metal working set")
    ap.add_argument("--expect", choices=["sigbus", "ok"], default="sigbus", help="cpu-moe: the expected outcome")
    args = ap.parse_args()
    os.environ.setdefault("GGML_NO_BACKTRACE", "1")  # the expected abort would print a gdb trace

    head = subprocess.run(["git", "-C", str(args.llama_cpp), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(args.llama_cpp), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    print(f"llama.cpp {head.stdout.strip() or '?'}{' (modified)' if dirty else ''}")

    c = Checks()
    if args.check == "utf8":
        run_utf8(args.llama_cpp, c)
    elif args.check == "tokenizer-random":
        run_tokenizer_random(args.llama_cpp, c)
    else:
        if not args.gguf:
            ap.error("cpu-moe needs --gguf")
        run_cpu_moe(args.llama_cpp, args.gguf, args.expect, c)
    sys.exit(0 if c.ok else 1)


if __name__ == "__main__":
    main()
