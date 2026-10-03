#!/usr/bin/env python3
"""Compare llama.cpp's tokenizer with the Kolibri-1 reference tokenizer.

Loads libllama through the cffi wrapper in llama.cpp's
tests/test-tokenizer-random.py and checks, with vocab_only=true:

  golden  every case in testdata/tokenizer/golden.jsonl: identical token IDs,
          and the detokenized bytes equal the input text;
  fuzz    llama.cpp's own differential generators (random characters,
          Unicode, vocab words, added tokens, ...) against the live
          reference tokenizer;
  utf8    invalid UTF-8 input (report only). The reference cannot receive
          it at all: vLLM tokenizes Python str. Each case runs in a forked
          child so that a llama.cpp abort is reported instead of ending the run;
          the output is compared with the reference on the U+FFFD-replaced
          text (Python's errors="replace").

    compare.py --llama-cpp third_party/llama.cpp --vocab ggml-vocab-kolibri.gguf [--iterations N]
"""

import argparse
import importlib.util
import multiprocessing
import os
import random
import sys
import time
from pathlib import Path

from common import load_golden, tokenizer_dir


def load_upstream(llama_cpp: Path):
    spec = importlib.util.spec_from_file_location("tokrand", llama_cpp / "tests" / "test-tokenizer-random.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Llama:
    """Minimal vocab-only libllama binding. The LibLlamaModel class in
    test-tokenizer-random.py predates llama_vocab and no longer works, so only
    its cffi loader is reused."""

    def __init__(self, up, llama_cpp: Path, vocab: Path):
        lib = up.LibLlama(
            path_llama_h=str(llama_cpp / "include" / "llama.h"),
            path_includes=[str(llama_cpp / "ggml" / "include"), str(llama_cpp / "include")],
            path_libllama=str(llama_cpp / "build" / "bin" / "libllama.so"),
        )
        self.lib, self.ffi = lib.lib, lib.ffi
        self.model = self.lib.llama_model_load_from_file(str(vocab).encode(), lib.model_default_params(vocab_only=True))
        if not self.model:
            raise SystemExit(f"failed to load {vocab}")
        self.vocab = self.lib.llama_model_get_vocab(self.model)
        self.ids = self.ffi.new("llama_token[]", 1 << 16)
        self.buf = self.ffi.new("char[]", 1 << 20)

    def encode(self, text: str) -> list[int]:
        return self.encode_bytes(text.encode())

    def encode_bytes(self, raw: bytes) -> list[int]:
        # add_special=True adds nothing for Kolibri (no BOS/EOS); parse_special=True
        # matches the reference, which always matches added tokens
        # (split_special_tokens=false).
        n = self.lib.llama_tokenize(self.vocab, raw, len(raw), self.ids, len(self.ids), True, True)
        if n < 0:  # -n is the required size
            self.ids = self.ffi.new("llama_token[]", -n)
            n = self.lib.llama_tokenize(self.vocab, raw, len(raw), self.ids, len(self.ids), True, True)
        return list(self.ids[0:n])

    def decode_bytes(self, ids: list[int]) -> bytes:
        if len(ids) > len(self.ids):
            self.ids = self.ffi.new("llama_token[]", len(ids))
        for i, t in enumerate(ids):
            self.ids[i] = t
        # remove_special=False, unparse_special=True: render special tokens as text
        n = self.lib.llama_detokenize(self.vocab, self.ids, len(ids), self.buf, len(self.buf), False, True)
        if n < 0:  # -n is the required size
            self.buf = self.ffi.new("char[]", -n)
            n = self.lib.llama_detokenize(self.vocab, self.ids, len(ids), self.buf, len(self.buf), False, True)
        return bytes(self.ffi.buffer(self.buf, n))


def first_diff(a, b) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


class Report:
    def __init__(self, title: str):
        self.title, self.n, self.fail, self.t0 = title, 0, 0, time.perf_counter()

    def check_text(self, llama: "Llama", label: str, text: str, want: list[int], decoded: str):
        """One check per string: llama.cpp must produce the reference IDs, and
        detokenizing the reference IDs must give the reference decode."""
        self.n += 1
        got = llama.encode(text)
        out = llama.decode_bytes(want)
        errs = []
        if got != want:
            i = first_diff(want, got)
            errs.append(f"ids differ at {i}: want {want[i:i+6]} got {got[i:i+6]}")
        if out != decoded.encode():
            errs.append(f"detokenized {out[:60]!r}, want {decoded[:60]!r}")
        if errs:
            self.fail += 1
            if self.fail <= 10:
                print(f"  FAIL {label}: {'; '.join(errs)}")

    def done(self) -> bool:
        print(f"{self.title}: {self.n - self.fail}/{self.n} ok ({time.perf_counter() - self.t0:.1f}s)")
        return self.fail == 0


def run_golden(llama: Llama) -> bool:
    r = Report("golden")
    for case in load_golden():
        r.check_text(llama, case["name"], case["text"], case["ids"], case["decoded"])
    return r.done()


def run_fuzz(up, llama: Llama, ref, auto_dir: Path, iterations: int) -> bool:
    random.seed(0)
    truth = up.TokenizerGroundtruth(str(auto_dir))
    # Under transformers 5, batch_decode() of a flat ID list returns one
    # string, so TokenizerGroundtruth's word lists collapse into a single
    # entry. Rebuild them from the reference tokenizer.
    truth.vocab = sorted({ref.decode([i]) for i in range(ref.get_vocab_size())})
    truth.added_tokens = [t.content for t in ref.get_added_tokens_decoder().values()]
    gens = {
        "custom_text": up.generator_custom_text(),
        "custom_text_edge_cases": up.generator_custom_text_edge_cases(),
        "ascii_lr_strip": up.generator_ascii_lr_strip(),
        "apostrophe": up.generator_apostrophe(),
        "unicodes": up.generator_unicodes(),
        "vocab_words": up.generator_vocab_words(truth),
        "added_lr_strip": up.generator_added_lr_strip(truth),
        "random_added_tokens": up.generator_random_added_tokens(truth, iterations),
        "random_chars": up.generator_random_chars(iterations),
        "random_unicodes": up.generator_random_unicodes(iterations),
        "random_vocab_chars": up.generator_random_vocab_chars(truth, iterations),
        "random_vocab_words": up.generator_random_vocab_words(truth, iterations // 2),
    }
    ok = True
    for name, gen in gens.items():
        r = Report(f"fuzz/{name}")
        for text in gen:
            r.check_text(llama, repr(text[:40]), text, ref.encode(text, add_special_tokens=False).ids, text)
        ok = r.done() and ok
    return ok


def _encode_child(llama: Llama, raw: bytes, send) -> None:
    send.send(llama.encode_bytes(raw))


def run_utf8(llama: Llama, ref) -> None:
    cases = [
        b"\xff", b"\xfe\xff", b"\xc3", b"abc\xe2\x82", b"\xe2\x82abc", b"\x80\x80\x80",
        b"Stra\xdfe", b"M\xe4dchen \xfcber", b"ok \xf0\x9f\x9a", b"\xf0\x9f\x9a\x80\xf0",
        b"\xc0\xaf",  # overlong "/"
        b"\xed\xa0\x80",  # UTF-16 surrogate U+D800
        b"\xf4\x90\x80\x80",  # U+110000, beyond Unicode
        b"\xf8\x88\x80\x80\x80",  # obsolete 5-byte form
    ]
    ctx = multiprocessing.get_context("fork")
    same = 0
    for raw in cases:
        recv, send = ctx.Pipe(duplex=False)
        child = ctx.Process(target=_encode_child, args=(llama, raw, send))
        child.start()
        child.join()
        want = ref.encode(raw.decode("utf-8", errors="replace"), add_special_tokens=False).ids
        if child.exitcode != 0:
            print(f"  {raw!r}: llama.cpp CRASHED (exit code {child.exitcode}); reference(replaced) {want}")
            continue
        got = recv.recv()
        same += got == want
        if got != want:
            print(f"  {raw!r}: llama.cpp {got} reference(replaced) {want}")
    print(f"utf8 (report only): {same}/{len(cases)} identical to the reference on U+FFFD-replaced text")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True, help="vocab-only Kolibri GGUF")
    ap.add_argument("--iterations", type=int, default=1000, help="iterations per random fuzz generator")
    ap.add_argument("--only", choices=["golden", "fuzz", "utf8"], action="append")
    args = ap.parse_args()

    import tokenizers

    os.environ.setdefault("GGML_NO_BACKTRACE", "1")  # an expected abort in the utf8 report would print a gdb trace
    d = tokenizer_dir()
    ref = tokenizers.Tokenizer.from_file(str(d / "tokenizer.json"))
    up = load_upstream(args.llama_cpp)
    llama = Llama(up, args.llama_cpp, args.vocab)

    only = set(args.only or ["golden", "fuzz", "utf8"])
    ok = True
    if "golden" in only:
        ok = run_golden(llama) and ok
    if "utf8" in only:
        run_utf8(llama, ref)
    if "fuzz" in only:
        ok = run_fuzz(up, llama, ref, d, args.iterations) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
