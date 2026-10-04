"""Shared helpers for the Kolibri-1 tokenizer tools."""

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "inventory" / "bf16" / "summary.json"
GOLDEN = ROOT / "testdata" / "tokenizer" / "golden.jsonl"
TOKENIZER_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")


def pinned_source() -> tuple[str, str, dict[str, str]]:
    """Returns repo, revision and the sha256 of every file recorded in the inventory."""
    s = json.loads(SUMMARY.read_text())
    return s["repo"], s["revision"], {f["name"]: f["sha256"] for f in s["files"]}


def tokenizer_dir() -> Path:
    """Downloads (or reuses from the HF cache) the tokenizer files of the
    pinned BF16 revision and verifies them against the checkpoint inventory."""
    from huggingface_hub import hf_hub_download

    repo, revision, hashes = pinned_source()
    paths = [Path(hf_hub_download(repo, name, revision=revision)) for name in TOKENIZER_FILES]
    for name, path in zip(TOKENIZER_FILES, paths):
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != hashes[name]:
            raise SystemExit(f"{path}: sha256 {got} does not match the inventory ({hashes[name]})")
    return paths[0].parent


def load_upstream(llama_cpp: Path):
    """llama.cpp's tests/test-tokenizer-random.py, for its cffi loader and
    its differential text generators."""
    spec = importlib.util.spec_from_file_location("tokrand", llama_cpp / "tests" / "test-tokenizer-random.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def libllama(llama_cpp: Path, up=None):
    """cffi bindings for the shared libllama in llama_cpp/build/bin, through
    the LibLlama loader of test-tokenizer-random.py."""
    up = up or load_upstream(llama_cpp)
    cls = up.LibLlama
    name = "libllama.so"
    if sys.platform == "darwin":
        cls = _darwin_libllama(up, llama_cpp)
        name = "libllama.dylib"
    return cls(
        path_llama_h=str(llama_cpp / "include" / "llama.h"),
        path_includes=[str(llama_cpp / "ggml" / "include"), str(llama_cpp / "include")],
        path_libllama=str(llama_cpp / "build" / "bin" / name),
    )


def _darwin_libllama(up, llama_cpp: Path):
    """The upstream loader feeds the whole preprocessed llama.h to cffi. The
    macOS SDK headers it pulls in (nullability qualifiers, __asm aliases,
    inline functions in stdio.h) do not parse, so keep only the declarations
    from llama.cpp's own headers; cffi knows FILE, size_t and int32_t itself."""
    import cffi

    root = llama_cpp.resolve()

    class LibLlama(up.LibLlama):
        def _load_libllama_cffi(self, path_llama_h, path_includes, path_libllama):
            cmd = ["cc", "-E", "-D__attribute__(x)="] + ["-I" + p for p in path_includes] + [path_llama_h]
            out = subprocess.run(cmd, stdout=subprocess.PIPE, check=True, text=True).stdout
            keep, own = [], False
            for line in out.splitlines():
                if m := re.match(r'# \d+ "([^"]+)"', line):
                    own = Path(m.group(1)).resolve().is_relative_to(root)
                elif own:
                    keep.append(line)
            source = "\n".join(keep)
            ffi = cffi.FFI()
            for t in ("int", "void *", "size_t", "int32_t"):  # pycparser cannot evaluate sizeof
                source = re.sub(rf"sizeof ?\({re.escape(t)}\)", str(ffi.sizeof(t)), source)
            ffi.cdef(source, override=True)
            return ffi, ffi.dlopen(path_libllama)

    return LibLlama


def load_golden() -> list[dict]:
    with GOLDEN.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]
