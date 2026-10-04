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


def generate_tiny(tmp: Path, seed: int) -> Path:
    """Writes the tiny checkpoint, with the real tokenizer, into tmp."""
    sys.path.insert(0, str(ROOT / "tools" / "tokenizer"))
    from common import tokenizer_dir

    model = tmp / "kolibri-tiny"
    run(["go", "run", "./cmd/kolibri-tiny", "-out", str(model), "-tokenizer-dir", str(tokenizer_dir()),
         "-seed", str(seed)], "cmd/kolibri-tiny")
    return model


def convert(model: Path, llama_cpp: Path, outtype: str) -> Path:
    """Converts a checkpoint with convert_hf_to_gguf.py --outtype outtype."""
    out = model.parent / f"{model.name}-{outtype}.gguf"
    run([sys.executable, str(llama_cpp / "convert_hf_to_gguf.py"), str(model),
         "--outtype", outtype, "--outfile", str(out)], f"convert_hf_to_gguf.py --outtype {outtype}")
    return out
