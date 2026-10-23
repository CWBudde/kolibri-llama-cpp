"""Shared steps of the checks that run on the cmd/kolibri-tiny checkpoint."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str], what: str) -> None:
    # from the repo root, so go run ./cmd/... works from any directory
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"FAIL {what} exited {r.returncode}:\n{r.stderr.strip()[-1500:]}")


def generate_tiny(tmp: Path, seed: int, preset: str = "") -> Path:
    """Writes the tiny checkpoint, with the real tokenizer, into tmp.
    preset "router" selects the 384-expert top-6 variant (kolibri-tiny -router), "attn" the real
    attention heads and sliding window (kolibri-tiny -attn), "pattern" the real 50-layer SWA/full
    pattern (kolibri-tiny -pattern)."""
    sys.path.insert(0, str(ROOT / "tools" / "tokenizer"))
    from common import tokenizer_dir

    model = tmp / ("kolibri-tiny" + (f"-{preset}" if preset else ""))
    run(["go", "run", "./cmd/kolibri-tiny", "-out", str(model), "-tokenizer-dir", str(tokenizer_dir()),
         "-seed", str(seed)] + ([f"-{preset}"] if preset else []), "cmd/kolibri-tiny")
    return model


def convert(model: Path, llama_cpp: Path, outtype: str, rekey: bool = False) -> Path:
    """Converts a checkpoint with convert_hf_to_gguf.py --outtype outtype. With rekey, the GGUF
    is then rewritten by cmd/kolibri-rekey into upstream's kolibri1 layout (only the header
    changes), so upstream libllama loads what this port's converter wrote."""
    out = model.parent / f"{model.name}-{outtype}.gguf"
    run([sys.executable, str(llama_cpp / "convert_hf_to_gguf.py"), str(model),
         "--outtype", outtype, "--outfile", str(out)], f"convert_hf_to_gguf.py --outtype {outtype}")
    if not rekey:
        return out
    keyed = out.with_name(out.stem + "-kolibri1.gguf")
    run(["go", "run", "./cmd/kolibri-rekey", "-copy", "-in", str(out), "-out", str(keyed)], "cmd/kolibri-rekey")
    return keyed


def arch(path: Path, llama_cpp: Path) -> str:
    """general.architecture of a GGUF: kolibri for this port's converter, kolibri1 for upstream's."""
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    import gguf

    return gguf.GGUFReader(path).fields["general.architecture"].contents()
