#!/usr/bin/env python3
"""Build the imatrix calibration text for Kolibri-1.

Kolibri routes each token to 6 of 384 experts, and the routing depends on
the domain. On English wikitext alone, 8 chunks of 512 tokens leave 12 to 28%
of the experts in the late layers without a single token. llama-quantize gives
an expert without imatrix data uniform weights, so it gains nothing from the
imatrix.

The calibration text therefore follows the model card's pre-training mix
(about 62.5% English, 23.9% German, 13.6% code), with German and code
weighted up to about 45% English, 35% German and 20% code by token count:

- English: the start of wikitext-2 wiki.train.raw (the KLD evaluation uses
  wiki.test.raw, a disjoint split);
- German: articles of German Wikipedia (wikimedia/wikipedia, 20231101.de)
  from fixed offsets, through the Hugging Face datasets server;
- code: C++, Python and Go source files at fixed paths of this repo and of
  third_party/llama.cpp. third_party/ is not a submodule and may come from a
  moving branch, so every code slice is checked against the sha256 it had in
  the documented corpus (llama.cpp e1a553f5f).

The sections are interleaved in blocks, so a run cut short with --chunks still
sees all three domains.

    calibration.py --wikitext ~/models/eval/wikitext-2-raw/wiki.train.raw \\
        -o ~/models/eval/kolibri-calibration.txt
"""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROWS_URL = ("https://datasets-server.huggingface.co/rows?dataset=wikimedia/wikipedia"
            "&config=20231101.de&split=train&offset={offset}&length={length}")
DE_OFFSETS = [0, 100000, 400000, 800000, 1200000, 1600000, 2000000, 2400000, 2800000]
DE_ROWS_PER_OFFSET = 25
DE_MAX_ARTICLE = 800
LLAMA_CPP_REV = "e1a553f5fa1f82edfb702e93012c4daa70634e92"
# Path, characters taken from the start, sha256 of those characters (UTF-8).
CODE_FILES = [
    ("third_party/llama.cpp/src/llama-graph.cpp", 35000,
     "e8ed5db69b6c78daf1c127c31524e591cc8f6c61199524c3f06f09ffa914250e"),
    ("third_party/llama.cpp/gguf-py/gguf/gguf_writer.py", 20000,
     "52e99c707a35894fdb0e13b53d7a47e484a7b2b0cd5553d7412956864f6c2612"),
    ("third_party/llama.cpp/tools/server/server-context.cpp", 20000,
     "54ca626f53b2ccf8f86636cabb9ce28bb28ac8e227d32a65da3b833742b34e35"),
    ("internal/safetensors/safetensors.go", 7000,
     "08d8aaf05996b708023f5e25841e1241e898e12ce134181bb4d9d664b9637aad"),
    ("cmd/kolibri-tiny/main.go", 10000,
     "e332048198abbcd6acab264c679a5529a072c257c450277edafed2e16574e6f4"),
]
EN_BYTES = 260000
BLOCK = 4000


def german(cache: Path) -> str:
    """German Wikipedia articles, cached as the raw JSON rows."""
    parts = []
    for offset in DE_OFFSETS:
        path = cache / f"dewiki-{offset}.json"
        if not path.exists():
            url = ROWS_URL.format(offset=offset, length=DE_ROWS_PER_OFFSET)
            with urllib.request.urlopen(url, timeout=60) as r:
                path.write_bytes(r.read())
            time.sleep(1)
        for row in json.loads(path.read_text())["rows"]:
            text = row["row"]["text"][:DE_MAX_ARTICLE]
            parts.append(f"{row['row']['title']}\n\n{text}")
    return "\n\n".join(parts)


def code() -> str:
    parts = []
    changed = []
    for rel, limit, want in CODE_FILES:
        text = (ROOT / rel).read_text()[:limit]
        if hashlib.sha256(text.encode()).hexdigest() != want:
            changed.append(rel)
        parts.append(f"// {rel}\n{text}")
    if changed:
        raise SystemExit("FAIL code inputs differ from the documented corpus: " + ", ".join(changed)
                         + f"; check out llama.cpp {LLAMA_CPP_REV[:9]} and this repo's matching commit")
    return "\n\n".join(parts)


def blocks(text: str) -> list[str]:
    """Split at line ends near every BLOCK characters."""
    out = []
    while text:
        cut = text.find("\n", BLOCK)
        cut = len(text) if cut < 0 else cut + 1
        out.append(text[:cut])
        text = text[cut:]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wikitext", type=Path, required=True, help="wikitext-2 wiki.train.raw")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--cache", type=Path, help="directory for the downloaded rows (default: next to --output)")
    args = ap.parse_args()

    cache = args.cache or args.output.parent / "calibration-cache"
    cache.mkdir(parents=True, exist_ok=True)
    sections = {
        "en": args.wikitext.read_text()[:EN_BYTES],
        "de": german(cache),
        "code": code(),
    }
    queues = {k: blocks(v) for k, v in sections.items()}
    out = []
    while any(queues.values()):
        for q in queues.values():
            if q:
                out.append(q.pop(0))
    text = "".join(out)
    args.output.write_text(text)
    for k, v in sections.items():
        print(f"{k}: {len(v.encode()):,} bytes")
    print(f"total: {len(text.encode()):,} bytes, sha256 {hashlib.sha256(text.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
