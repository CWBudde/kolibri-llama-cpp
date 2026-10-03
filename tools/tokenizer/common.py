"""Shared helpers for the Kolibri-1 tokenizer tools."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "inventory" / "bf16" / "summary.json"
GOLDEN = ROOT / "testdata" / "tokenizer" / "golden.jsonl"
TOKENIZER_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")


def pinned_source() -> tuple[str, str, dict[str, str]]:
    """Returns repo, revision and the sha256 of every file recorded in Phase 1."""
    s = json.loads(SUMMARY.read_text())
    return s["repo"], s["revision"], {f["name"]: f["sha256"] for f in s["files"]}


def tokenizer_dir() -> Path:
    """Downloads (or reuses from the HF cache) the tokenizer files of the
    pinned BF16 revision and verifies them against the Phase 1 inventory."""
    from huggingface_hub import hf_hub_download

    repo, revision, hashes = pinned_source()
    paths = [Path(hf_hub_download(repo, name, revision=revision)) for name in TOKENIZER_FILES]
    for name, path in zip(TOKENIZER_FILES, paths):
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != hashes[name]:
            raise SystemExit(f"{path}: sha256 {got} does not match the inventory ({hashes[name]})")
    return paths[0].parent


def load_golden() -> list[dict]:
    with GOLDEN.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]
