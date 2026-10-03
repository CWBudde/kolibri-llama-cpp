#!/usr/bin/env python3
"""Check the kolibri architecture registration in a patched llama.cpp.

gguf-py:  MODEL_ARCH.KOLIBRI is named "kolibri", and MODEL_TENSORS[KOLIBRI]
          lists exactly the tensor kinds of the Phase 1 HF -> GGUF mapping
          (inventory/bf16/summary.json, "classes").
libllama: a GGUF whose general.architecture is "kolibri" is a known
          architecture. Until Phase 4 adds llama_model_kolibri, loading it
          must fail with "unsupported model architecture", not with
          "unknown model architecture".

    check_arch.py --llama-cpp third_party/llama.cpp
"""

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "inventory" / "bf16" / "summary.json"
ARCH = "kolibri"

sys.path.insert(0, str(ROOT / "tools" / "tokenizer"))
from common import libllama  # noqa: E402


def check_tensors(gguf) -> list[str]:
    errs = []
    if not hasattr(gguf.MODEL_ARCH, "KOLIBRI"):
        return ["gguf.MODEL_ARCH has no KOLIBRI"]
    arch = gguf.MODEL_ARCH.KOLIBRI
    if gguf.MODEL_ARCH_NAMES.get(arch) != ARCH:
        errs.append(f"MODEL_ARCH_NAMES[KOLIBRI] = {gguf.MODEL_ARCH_NAMES.get(arch)!r}, want {ARCH!r}")

    # "blk.{L}.attn_q.weight" -> "blk.{bid}.attn_q". Compared by name, because
    # some MODEL_TENSOR kinds are aliases (FFN_PRE_NORM == FFN_NORM).
    classes = json.loads(SUMMARY.read_text())["classes"]
    want = {re.sub(r"\.(weight|bias)$", "", c["gguf"]).replace("{L}", "{bid}") for c in classes if c["role"] == "weight"}
    have = {gguf.TENSOR_NAMES[t] for t in gguf.MODEL_TENSORS.get(arch, [])}
    if missing := sorted(want - have):
        errs.append(f"MODEL_TENSORS[KOLIBRI] lacks {missing}")
    if extra := sorted(have - want):
        errs.append(f"MODEL_TENSORS[KOLIBRI] has tensors the checkpoint does not: {extra}")
    print(f"gguf-py: {len(have)} tensor names registered, {len(want)} in the Phase 1 mapping")
    return errs


def check_libllama(gguf, llama_cpp: Path) -> list[str]:
    # In-process with a log callback: llama-tokenize's asynchronous logger can
    # drop its last lines when the output is a pipe.
    lib = libllama(llama_cpp)
    log: list[str] = []

    @lib.ffi.callback("void(enum ggml_log_level, const char *, void *)")
    def on_log(level, text, user_data):
        log.append(lib.ffi.string(text).decode(errors="replace"))

    lib.lib.llama_log_set(on_log, lib.ffi.NULL)
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "probe.gguf"
        w = gguf.GGUFWriter(probe, ARCH)  # writes general.architecture
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        model = lib.lib.llama_model_load_from_file(str(probe).encode(), lib.model_default_params(vocab_only=True))
    lib.lib.llama_log_set(lib.ffi.NULL, lib.ffi.NULL)
    text = "".join(log)
    if model:
        lib.lib.llama_model_free(model)
        return [f"libllama loaded an empty '{ARCH}' GGUF; update this check for Phase 4"]
    if f"unsupported model architecture: '{ARCH}'" in text:
        print(f"libllama: '{ARCH}' is a known architecture without a model class (expected before Phase 4)")
        return []
    if f"unknown model architecture: '{ARCH}'" in text:
        return [f"libllama does not know the architecture '{ARCH}'"]
    return [f"unexpected libllama log: {text.strip()[-300:]}"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(args.llama_cpp / "gguf-py"))
    import gguf

    errs = check_tensors(gguf) + check_libllama(gguf, args.llama_cpp)
    for e in errs:
        print(f"FAIL {e}")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
