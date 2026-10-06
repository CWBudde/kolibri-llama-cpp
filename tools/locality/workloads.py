#!/usr/bin/env python3
"""Fixed chat traces for measuring expert locality (PLAN Phase 8).

Four workloads in testdata/locality/<name>.json, each a complete conversation
through the released chat template:

- coding: reading, fixing and refactoring this repo's Go and Python code, plus
  two tasks outside Kolibri's expected strengths (Object Pascal, VHDL);
- research: web research in German and English with repeated web_search and
  fetch_page rounds;
- hr: German questions against a synthetic HR API whose responses follow
  Personio's v1 JSON shape;
- medtech: German QM work on long excerpts of the EU Medical Device
  Regulation (EU) 2017/745.

A trace fixes the system prompt, the tools, the user turns, the assistant's
tool calls and the tool results. Every final assistant answer was generated
once (greedy, thinking off; the GGUF is recorded in the manifest) with the
<tool_call> token banned, since the tool rounds are scripted, and is frozen
in the trace, so the whole conversation is teacher-forced: in a causal
model a token's experts depend only on the tokens before it, so prefilling a
trace selects the experts that generating it did, on any backend, in one pass.

The template runs with enable_thinking false and preserve_thinking true
(llama-server's default), so every assistant turn carries the empty think
block and each answer's prompt is a prefix of the full conversation.

Third-party text is not committed. A trace refers to it by source and range
and pins the sha256 of the resolved text:

- {"git": commit, "path": ...}: a file of this repo at a fixed commit;
- {"wikitext": "wiki.test.raw", "article": title}: one wikitext-2 article
  (~/models/eval/wikitext-2-raw);
- {"dewiki": offset, "row": i}: one German Wikipedia article from the rows
  tools/quant/calibration.py caches (~/models/eval/calibration-cache);
- {"mdr": [heading, ...]}: articles or annexes of the regulation's German
  XHTML from the EU Publications Office, which --fetch downloads.

Any of them may add "chars": n to keep only the first n characters.

The default mode checks each workload and fails (exit 1) on any difference:
every source against its sha256; every answer slot (each assistant message
without tool calls) generated; the rendered conversation tokenized by the
pinned reference tokenizer and by libllama on the vocab GGUF, which must
agree; each answer's prompt and text a prefix of the whole, the prompt's IDs
those the answer was generated from, and the answer's generated token IDs
(stored with it) decoding to its text; the token IDs, answers as generated,
decoding to the conversation; the length within the workload's range; and
the sha256 of the rendered text and of the token IDs against
testdata/locality/manifest.json. It writes the token IDs to
~/models/eval/locality/<name>.tokens.npy for the locality measurements and
prints the token count by role.

The token IDs hold each answer as generated, not as the tokenizer would split
its text afresh: a model can generate a split the tokenizer would not choose
(an INFO line names the answer and the first differing token). Each answer
was generated from the IDs of the conversation before it, earlier answers as
generated, and records that prompt's length and sha256; the check fails
unless the conversation's IDs start with exactly that prompt, so one prefill
reproduces every generation's context.

    workloads.py --fetch
    workloads.py --answer --gguf ~/models/Kolibri-1-Q8_0.gguf
    workloads.py [--record]
"""

import argparse
import bisect
import copy
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "tokenizer"))
sys.path.insert(0, str(ROOT / "tools" / "chat"))
sys.path.insert(0, str(ROOT / "tools" / "quant"))

from calibration import DE_ROWS_PER_OFFSET, ROWS_URL  # noqa: E402
from common import tokenizer_dir  # noqa: E402

TRACES = ROOT / "testdata" / "locality"
MANIFEST = TRACES / "manifest.json"
EVAL = Path.home() / "models" / "eval"
OUT = EVAL / "locality"
WIKITEXT = EVAL / "wikitext-2-raw"
DEWIKI = EVAL / "calibration-cache"
MDR_URL = "http://publications.europa.eu/resource/celex/32017R0745"
MDR_FILE = "mdr-32017R0745-de.xhtml"
MDR_SHA256 = "35ebc66854b8e425b0dd6773f677de87be4df6c81976fa46c576a9f24352bab6"
WORKLOADS = ("coding", "research", "hr", "medtech")
VARIABLES = {"enable_thinking": False, "preserve_thinking": True}
# (minimum, maximum) tokens of each workload; the maximum is the 32k context
LENGTH = {"coding": (8192, 32768), "research": (8192, 32768), "hr": (8192, 32768), "medtech": (24576, 32768)}
N_ANSWER = 384
# <tool_call>: the tool rounds are scripted, so an answer may not open another one
TOOL_CALL = 127909
EOS = (127901, 127906)  # <|endoftext|>, <|im_end|>


class SourceChanged(Exception):
    pass


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


class Paragraphs(HTMLParser):
    """The text of an Official Journal XHTML: one line per paragraph or table row, with the paragraph's class."""

    def __init__(self):
        super().__init__()
        self.lines: list[tuple[str | None, str]] = []
        self.cur: list[str] = []
        self.cells: list[str] = []
        self.cls: str | None = None
        self.depth = 0  # table nesting

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.flush()
            self.depth += 1
        elif tag == "p" and not self.depth:
            self.flush()
            self.cls = dict(attrs).get("class")

    def handle_endtag(self, tag):
        if tag == "table":
            self.depth -= 1
        elif tag == "p" and not self.depth:
            self.flush()
        elif tag == "td" and self.depth == 1:
            self.cells.append(self.take())
        elif tag == "tr" and self.depth == 1:
            if row := " ".join(c for c in self.cells if c):
                self.lines.append(("oj-table", row))
            self.cells = []

    def handle_data(self, data):
        self.cur.append(data)

    def take(self) -> str:
        text = re.sub(r"\s+", " ", "".join(self.cur)).strip()
        self.cur = []
        return text

    def flush(self):
        if text := self.take():
            self.lines.append((self.cls, text))


def mdr_sections(path: Path) -> dict[str, str]:
    """The regulation's articles ("Artikel 10") and annexes ("ANHANG I") by heading, as plain text."""
    p = Paragraphs()
    p.feed(path.read_text(encoding="utf-8"))
    p.flush()
    sections: dict[str, list[str]] = {}
    name = None
    for cls, text in p.lines:
        if cls in ("oj-ti-art", "oj-doc-ti") and re.fullmatch(r"Artikel \d+|ANHANG [IVX]+", text):
            name = text
            sections[name] = []
        if name:
            sections[name].append(text)
    return {k: "\n".join(v) for k, v in sections.items()}


def wikitext_article(path: Path, title: str) -> str:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    start = lines.index(f" = {title} = \n")
    end = next((i for i in range(start + 1, len(lines)) if re.fullmatch(r" = [^=].* = \n", lines[i])), len(lines))
    return "".join(lines[start:end]).strip("\n") + "\n"


def dewiki_article(offset: int, row: int) -> str:
    path = DEWIKI / f"dewiki-{offset}.json"
    if not path.exists():
        raise SourceChanged(f"{path} is missing: run --fetch")
    r = json.loads(path.read_text())["rows"][row]["row"]
    return f"{r['title']}\n\n{r['text']}"


def resolve(ref: dict) -> str:
    """The text a source reference names, checked against its sha256."""
    text = source_text(ref)
    if (got := sha256_text(text)) != ref.get("sha256"):
        what = json.dumps({k: v for k, v in ref.items() if k != "sha256"}, ensure_ascii=False)
        raise SourceChanged(f"{what}: sha256 {got}, pinned {ref.get('sha256')}")
    return text


def source_text(ref: dict) -> str:
    """The text a source reference names, unchecked."""
    if "git" in ref:
        text = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref['git']}:{ref['path']}"], check=True,
                              capture_output=True, text=True).stdout
    elif "wikitext" in ref:
        text = wikitext_article(WIKITEXT / ref["wikitext"], ref["article"])
    elif "dewiki" in ref:
        text = dewiki_article(ref["dewiki"], ref["row"])
    elif "mdr" in ref:
        if not (OUT / MDR_FILE).exists():
            raise SourceChanged(f"{OUT / MDR_FILE} is missing: run --fetch")
        sections = mdr_sections(OUT / MDR_FILE)
        text = "\n\n".join(sections[name] for name in ref["mdr"])
    else:
        raise SystemExit(f"unknown source {ref}")
    return text[:ref["chars"]] if "chars" in ref else text


def content(c) -> str:
    """A message's content: a string, a source reference, {"json": value}, or a list of those, concatenated."""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(content(part) for part in c)
    if "json" in c:
        return json.dumps(c["json"], ensure_ascii=False)
    return resolve(c)


def refs(trace: dict):
    """The trace's source references."""
    def walk(c):
        if isinstance(c, list):
            for part in c:
                yield from walk(part)
        elif isinstance(c, dict) and "json" not in c:
            yield c

    for m in trace["messages"]:
        if m.get("content") is not None:
            yield from walk(m["content"])


def messages(trace: dict, upto: int | None = None) -> list[dict]:
    """The trace's first upto messages (all by default) as vLLM hands them to the template: tool call arguments
    as objects (check_chat.reference_messages)."""
    out, n_call = [], 0
    for m in trace["messages"][:upto]:
        msg = {"role": m["role"], "content": content(m["content"]) if m.get("content") is not None else ""}
        if "tool_calls" in m:
            msg["tool_calls"] = []
            for tc in m["tool_calls"]:
                msg["tool_calls"].append({"type": "function", "id": f"call_{n_call}", "function": {
                    "name": tc["name"], "arguments": copy.deepcopy(tc["arguments"])}})
                n_call += 1
        out.append(msg)
    return out


def render(template: str, trace: dict, upto: int | None = None, prompt: bool = False) -> str:
    """The first upto messages as the reference renders them; with prompt, as the prompt for the next answer."""
    from transformers.utils.chat_template_utils import render_jinja_template

    if upto == 0:
        return ""
    rendered, _ = render_jinja_template(conversations=[messages(trace, upto)], tools=trace.get("tools"),
                                        chat_template=template, add_generation_prompt=prompt, **VARIABLES)
    return rendered[0]


def slots(trace: dict) -> list[int]:
    """The indices of the generated answers: the assistant messages without tool calls, whether generated yet or
    not."""
    return [i for i, m in enumerate(trace["messages"]) if m["role"] == "assistant" and "tool_calls" not in m]


def first_difference(a: list[int], b: list[int]) -> int | None:
    """The first index at which a and b differ, the shorter length if one is a prefix of the other, or None."""
    if a == b:
        return None
    return next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


class Boundary(Exception):
    """An answer that does not start and end on token boundaries of the fresh tokenization."""


def tokens(template: str, trace: dict, tok, upto: int | None = None, prompt: bool = False):
    """The conversation's token IDs as generated (with upto and prompt, those of that answer's prompt): the text
    tokenized afresh, then each recorded answer's span replaced by the IDs the model generated, which can be a
    split the tokenizer would not choose. Returns the text, the IDs, each ID's start offset in the text, and per
    recorded answer its index, fresh IDs and generated IDs."""
    text = render(template, trace, upto, prompt)
    enc = tok.encode(text, add_special_tokens=False)
    fresh, starts = enc.ids, [s for s, _ in enc.offsets]
    ids, id_starts, spans, prev = [], [], [], 0
    for i in slots(trace):
        m = trace["messages"][i]
        if upto is not None and i >= upto:
            break
        if "answer" not in m:
            continue
        begin = len(render(template, trace, i, prompt=True))
        end = begin + len(m["content"])
        s, e = bisect.bisect_left(starts, begin), bisect.bisect_left(starts, end)
        if starts[s:s + 1] != [begin] or starts[e:e + 1] not in ([end], []):
            raise Boundary(i)
        gen = [t for t in m["answer"]["tokens"] if t not in EOS]
        ids += fresh[prev:s] + gen
        id_starts += starts[prev:s] + [begin] * len(gen)
        spans.append((i, fresh[s:e], gen))
        prev = e
    return text, ids + fresh[prev:], id_starts + starts[prev:], spans


def load(name: str) -> dict:
    return json.loads((TRACES / f"{name}.json").read_text())


def save(name: str, trace: dict) -> None:
    (TRACES / f"{name}.json").write_text(json.dumps(trace, ensure_ascii=False, indent=1) + "\n")


def fetch() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / MDR_FILE
    if not path.exists():
        req = urllib.request.Request(MDR_URL, headers={"Accept": "application/xhtml+xml", "Accept-Language": "deu"})
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != MDR_SHA256:
        raise SystemExit(f"FAIL {path}: sha256 {got}, pinned {MDR_SHA256}")
    print(f"{path}: sha256 {got}")
    DEWIKI.mkdir(parents=True, exist_ok=True)
    for offset in sorted({ref["dewiki"] for name in WORKLOADS for ref in refs(load(name)) if "dewiki" in ref}):
        p = DEWIKI / f"dewiki-{offset}.json"
        if not p.exists():
            with urllib.request.urlopen(ROWS_URL.format(offset=offset, length=DE_ROWS_PER_OFFSET), timeout=60) as r:
                p.write_bytes(r.read())
            time.sleep(1)
        print(f"{p}: present")


def complete(port: int, ids: list[int]) -> dict:
    body = {"prompt": ids, "n_predict": N_ANSWER, "temperature": 0, "cache_prompt": True, "return_tokens": True,
            "logit_bias": [[TOOL_CALL, False]]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=None) as r:
        return json.load(r)


def answer(args, template: str, tok) -> None:
    """Generate every empty answer slot, saving the trace after each one, so a rerun resumes."""
    from check_chat import Server

    flags = ["-dev", "none", "--no-repack", "-t", str(args.threads), "-c", "32768", "-np", "1"]
    server = Server(args.llama_cpp, args.gguf, flags, args.log)
    try:
        for name in args.workloads:
            trace = load(name)
            for i in slots(trace):
                m = trace["messages"][i]
                if "answer" in m:
                    continue
                _, ids, _, _ = tokens(template, trace, tok, i, prompt=True)
                t0 = time.monotonic()
                r = complete(server.port, ids)
                gen = r["tokens"]
                m["content"] = r["content"]
                m["answer"] = {"n_prompt": len(ids), "prompt_sha256": ids_sha256(ids), "stop": r.get("stop_type"),
                               "tokens": gen,
                               "prompt_per_second": round(r["timings"]["prompt_per_second"], 2),
                               "predicted_per_second": round(r["timings"]["predicted_per_second"], 2)}
                save(name, trace)
                print(f"{name} message {i}: {len(gen)} tokens ({m['answer']['stop']}) after a {len(ids)}-token "
                      f"prompt in {time.monotonic() - t0:.0f} s", flush=True)
    finally:
        server.close()
    commit = subprocess.run(["git", "-C", str(args.llama_cpp), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest["answers"] = {"gguf": args.gguf.name, "llama.cpp": commit, "llama-server flags": flags,
                           "request": {"n_predict": N_ANSWER, "temperature": 0, "logit_bias": [[TOOL_CALL, False]]}}
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")


def by_role(template: str, trace: dict, starts: list[int]) -> str:
    """Token counts by the message a token starts in, given each token's start offset: system (with the tools),
    user, tool, assistant (scripted tool calls) and answer (generated)."""
    msgs, answers = trace["messages"], set(slots(trace))
    ends = [len(render(template, trace, k + 1)) for k in range(len(msgs))]
    counts: dict[str, int] = {}
    k = 0
    for start in starts:
        while k < len(msgs) - 1 and start >= ends[k]:
            k += 1
        role = "answer" if k in answers else msgs[k]["role"]
        counts[role] = counts.get(role, 0) + 1
    return ", ".join(f"{r} {n}" for r, n in counts.items())


def check(args, template: str, tok) -> int:
    import numpy as np

    from compare import Llama, load_upstream

    llama = Llama(load_upstream(args.llama_cpp), args.llama_cpp, args.llama_cpp / "models" / "ggml-vocab-kolibri.gguf")
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    recorded = manifest.setdefault("workloads", {})
    errs = []

    def fail(what: str) -> None:
        print("FAIL " + what)
        errs.append(what)

    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.workloads:
        trace = load(name)
        n_errs, same = len(errs), 0
        try:
            text, ids, starts, spans = tokens(template, trace, tok)
        except SourceChanged as e:
            fail(f"{name}: source changed: {e}")
            continue
        except Boundary as e:
            fail(f"{name} message {e}: the answer does not start and end on token boundaries")
            continue
        fresh = tok.encode(text, add_special_tokens=False).ids
        if (k := first_difference(llama.encode(text), fresh)) is not None:
            fail(f"{name}: libllama's tokens differ from the reference's from token {k}")
        if tok.decode(ids, skip_special_tokens=False) != text:
            fail(f"{name}: the token IDs with the generated answers do not decode to the conversation")
        for i in slots(trace):
            m = trace["messages"][i]
            if "answer" not in m or m.get("content") is None:
                fail(f"{name} message {i}: no recorded generation (run --answer)")
                continue
            prompt, p_ids, _, _ = tokens(template, trace, tok, i, prompt=True)
            if not text.startswith(prompt + m["content"]):
                fail(f"{name} message {i}: prompt and answer are not a prefix of the conversation")
                continue
            a = m["answer"]
            # Each answer must sit in the conversation on exactly the prompt it was generated from, every earlier
            # answer included as generated, so one prefill of the IDs reproduces every generation's context.
            if len(p_ids) != a["n_prompt"] or ids_sha256(p_ids) != a.get("prompt_sha256"):
                fail(f"{name} message {i}: the prompt is not the one the answer was generated from")
            gen = [t for t in a["tokens"] if t not in EOS]
            if tok.decode(gen, skip_special_tokens=False) != m["content"]:
                fail(f"{name} message {i}: the generated tokens do not decode to the answer")
            elif ids[len(p_ids):len(p_ids) + len(gen)] != gen:
                fail(f"{name} message {i}: the conversation does not hold the generated tokens after the prompt")
        for i, f, gen in spans:
            if (k := first_difference(f, gen)) is None:
                same += 1
            else:
                print(f"INFO {name} message {i}: replayed as generated; the tokenizer splits the answer differently "
                      f"from token {k} ({len(f)} tokens afresh, {len(gen)} generated)")
        lo, hi = LENGTH[name]
        if not lo <= len(ids) <= hi:
            fail(f"{name}: {len(ids)} tokens, outside [{lo}, {hi}]")
        np.save(OUT / f"{name}.tokens.npy", np.array(ids, dtype=np.int32))
        print(f"{name}: {len(ids)} tokens: {by_role(template, trace, starts)}")
        entry = {"n_tokens": len(ids), "text_sha256": sha256_text(text), "tokens_sha256": ids_sha256(ids)}
        if args.record:
            recorded[name] = entry
        elif recorded.get(name) != entry:
            fail(f"{name}: {entry} differs from the manifest's {recorded.get(name)}")
        elif len(errs) == n_errs:
            print(f"PASS {name}: text and token IDs as recorded, libllama's tokens identical, "
                  f"{len(slots(trace))} answers on the prompts they were generated from, {same} of them split as "
                  f"the tokenizer would")
    if args.record and not errs:
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
        print(f"recorded {MANIFEST.relative_to(ROOT)}")
    return len(errs)


def main() -> None:
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, default=ROOT / "third_party" / "llama.cpp")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--fetch", action="store_true", help="download the regulation and missing Wikipedia rows")
    mode.add_argument("--answer", action="store_true", help="generate the empty answer slots with --gguf")
    mode.add_argument("--record", action="store_true", help="make this check's hashes the recorded ones")
    ap.add_argument("--gguf", type=Path, help="--answer: the GGUF that generates (CPU-only)")
    ap.add_argument("--threads", type=int, default=10, help="--answer: llama-server CPU threads")
    ap.add_argument("--log", type=Path, default=Path("llama-server-locality.log"), help="--answer: server log")
    ap.add_argument("--workloads", type=lambda s: s.split(","), default=list(WORKLOADS))
    args = ap.parse_args()

    if args.fetch:
        fetch()
        return
    template = json.loads((tokenizer_dir() / "tokenizer_config.json").read_text())["chat_template"]
    tok = Tokenizer.from_file(str(tokenizer_dir() / "tokenizer.json"))
    if args.answer:
        if not args.gguf:
            raise SystemExit("--answer needs --gguf")
        answer(args, template, tok)
        return
    sys.exit(1 if check(args, template, tok) else 0)


if __name__ == "__main__":
    main()
