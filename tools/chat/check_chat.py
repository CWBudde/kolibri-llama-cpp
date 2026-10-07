#!/usr/bin/env python3
"""Check the Kolibri chat template, reasoning modes, tool calls and stop tokens in llama.cpp.

The reference is the released template rendered by Hugging Face transformers
(render_jinja_template, the renderer of aleph-alpha-inference's
tests/test_reasoning.py) with the arguments vLLM passes. llama.cpp renders the
template with its own Jinja engine, behind llama-server's request handling.

- template: tokenizer.chat_template in the vocab GGUF, in the GGUF that
  convert_hf_to_gguf.py writes for the cmd/kolibri-tiny checkpoint, in the
  fork's models/templates file and in llama-server's /props is the released
  template from tokenizer_config.json, byte for byte;
- render: llama-server's /apply-template gives the same prompt as the
  reference for every message set (the reference's single turn and tool loop,
  a system message, tools, parallel tool calls, reasoning in the history,
  several turns) and every way to choose the reasoning mode: the reference's
  10 chat_template_kwargs sets (reasoning_effort wins over enable_thinking),
  the OpenAI reasoning_effort field for each level, alone and next to an
  enable_thinking kwarg (vLLM lets the field win), and the --reasoning off
  and --reasoning-effort defaults, each with llama-server's default
  --reasoning-preserve (preserve_thinking=true) and with
  --no-reasoning-preserve. The prompt ends in the prefilled empty think block
  exactly when the reference's thinking_enabled() is false;
- continue: continuing a final assistant message (content, reasoning and
  content, reasoning only, after a tool loop) through llama-server's default
  --prefill-assistant and every continue_final_message mode, in every reasoning
  mode. A content continuation gives the reference's continue_final_message
  prompt, with the think block closed whatever the mode. A reasoning
  continuation ("reasoning_content", or a message with reasoning only) has no
  reference: llama.cpp leaves the think block open after the reasoning, the
  reference closes it. That prompt is checked exactly;
- stop tokens: in both GGUFs, libllama ends generation on exactly the two
  eos_token_id of generation_config.json, <|im_end|> and <|endoftext|>; the
  six tag tokens (<think>, <tool_call>, ...) are text, so they reach the chat
  parser, and the control tokens render as nothing.

How llama.cpp parses the reasoning and the tool calls out of the generated
text is checked in the fork's tests/test-chat.cpp (ctest -R test-chat).

    check_chat.py --llama-cpp third_party/llama.cpp
"""

import argparse
import copy
import hashlib
import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "gguf"))
sys.path.insert(0, str(ROOT / "tools" / "tokenizer"))
from common import libllama, tokenizer_dir  # noqa: E402
from tiny import convert, generate_tiny  # noqa: E402

# sha256 of the chat_template in tokenizer_config.json of the pinned revision
TEMPLATE_SHA256 = "9ba35d4bd6baa26b66aa75d03a922dfee98b16bb1fa37481b195d247267b0f97"
TEMPLATE_FILE = "models/templates/Aleph-Alpha-Kolibri-1.jinja"
EOS_IDS = [127901, 127906]  # generation_config.json eos_token_id
EOS, PAD = 127906, 127901  # tokenizer_config.json eos_token, pad_token
CONTROL = {127901: "<|endoftext|>", 127904: "<|im_start|>", 127906: "<|im_end|>"}
TAGS = {127907: "<think>", 127908: "</think>", 127909: "<tool_call>", 127910: "</tool_call>",
        127911: "<tool_response>", 127912: "</tool_response>"}

# aleph-alpha-inference@049a6a7 tests/test_reasoning.py
THINKING_OFF_KWARGS = [
    {"enable_thinking": False},
    {"reasoning_effort": None, "enable_thinking": False},
    {"reasoning_effort": "none"},
    {"reasoning_effort": "none", "enable_thinking": True},
]
THINKING_ON_KWARGS = [
    {},
    {"reasoning_effort": None},
    {"enable_thinking": True},
    {"enable_thinking": None},
    {"reasoning_effort": "high"},
    {"reasoning_effort": "low", "enable_thinking": False},
]
PREFILLED_BLOCK = "<think>\n\n</think>\n\n"
EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]


def thinking_enabled(kwargs: dict) -> bool:
    """aleph_alpha_inference/reasoning.py: reasoning_effort wins, then enable_thinking."""
    effort = kwargs.get("reasoning_effort")
    if effort is not None:
        return effort != "none"
    return kwargs.get("enable_thinking") is not False


def vllm_template_kwargs(request: dict) -> dict:
    """The template arguments vLLM 0.29 derives from a chat request
    (ChatCompletionRequest.build_chat_params, renderers/params.py merge_kwargs): the
    reasoning_effort field overrides chat_template_kwargs and, unless the request sets
    enable_thinking, sets enable_thinking = effort != "none"."""
    user = request.get("chat_template_kwargs") or {}
    effort = request.get("reasoning_effort")
    extra = {"reasoning_effort": effort}
    if effort is not None and "enable_thinking" not in user:
        extra["enable_thinking"] = effort != "none"
    return user | {k: v for k, v in extra.items() if v not in (None, "auto")}


def call(name: str, args: dict, i: int) -> dict:
    """An OpenAI tool call; the arguments are a JSON string, as the API sends them."""
    return {"type": "function", "id": f"call_{i}", "function": {"name": name, "arguments": json.dumps(args)}}


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Aktuelles Wetter für eine Stadt",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]}}}
USER = [{"role": "user", "content": "hi"}]
# (name, messages, tools); the first two are the reference's USER_MESSAGES and TOOL_LOOP_MESSAGES
MESSAGES = [
    ("single", USER, None),
    ("tool_loop", USER + [{"role": "assistant", "content": "", "tool_calls": [call("lookup", {}, 0)]},
                          {"role": "tool", "tool_call_id": "call_0", "content": "result"}], None),
    ("system", [{"role": "system", "content": "Du bist ein hilfreicher Assistent."}] + USER, None),
    ("tools", [{"role": "user", "content": "Wie ist das Wetter in Köln?"}], [WEATHER]),
    ("parallel_calls", [
        {"role": "user", "content": "Wetter in Köln und Heidelberg?"},
        {"role": "assistant", "content": "Ich sehe nach.", "tool_calls": [
            call("get_weather", {"city": "Köln"}, 0), call("get_weather", {"city": "Heidelberg", "days": 2}, 1)]},
        {"role": "tool", "tool_call_id": "call_0", "content": "{\"temperature\": 18}"},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temperature\": 21}"}], [WEATHER]),
    ("history_reasoning", USER + [
        {"role": "assistant", "content": "Hallo!", "reasoning_content": "Der Nutzer grüßt."},
        {"role": "user", "content": "Wie geht's?"}], None),
    ("multi_turn", [
        {"role": "system", "content": "Antworte kurz."},
        {"role": "user", "content": "Eins?"}, {"role": "assistant", "content": "Zwei."},
        {"role": "user", "content": "Drei?"}, {"role": "assistant", "content": "Vier."},
        {"role": "user", "content": "Fünf?"}], None),
]


def reference_messages(messages: list[dict]) -> list[dict]:
    """vLLM's chat_utils._postprocess_messages: tool call arguments go to the template as objects."""
    out = copy.deepcopy(messages)
    for m in out:
        for tc in m.get("tool_calls") or []:
            tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
    return out


def render_reference(template: str, messages: list[dict], tools, variables: dict,
                     continue_final: bool = False, add_generation_prompt: bool = True) -> str:
    """continue_final: vLLM's continue_final_message, which also turns the generation prompt off."""
    from transformers.utils.chat_template_utils import render_jinja_template

    rendered, _ = render_jinja_template(conversations=[reference_messages(messages)], tools=tools,
                                        chat_template=template,
                                        add_generation_prompt=add_generation_prompt and not continue_final,
                                        continue_final_message=continue_final, **variables)
    return rendered[0]


# (name, messages): continuing the final assistant message
CONTINUE_MESSAGES = [
    ("content", USER + [{"role": "assistant", "content": "Hallo, ich"}]),
    ("reasoning_content", USER + [
        {"role": "assistant", "content": "Hallo, ich", "reasoning_content": "Der Nutzer grüßt."}]),
    ("reasoning_only", USER + [{"role": "assistant", "content": "", "reasoning_content": "Der Nutzer"}]),
    ("after_tool_loop", MESSAGES[1][1] + [{"role": "assistant", "content": "Das Ergebnis"}]),
]
# how a request continues it: llama-server's default --prefill-assistant (a final assistant
# message), vLLM's continue_final_message, and llama.cpp's two explicit modes
CONTINUATIONS = [{}] + [{"continue_final_message": c, "add_generation_prompt": False}
                        for c in (True, "content", "reasoning_content")]


def continued_in_reasoning(message: dict, how: dict) -> bool:
    """common_chat_templates_apply: "reasoning_content", or AUTO on a message with reasoning but no content."""
    c = how.get("continue_final_message", True)
    return c == "reasoning_content" or (c is True and bool(message.get("reasoning_content"))
                                        and not message.get("content"))


def gguf_template(llama_cpp: Path, path: Path) -> str:
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    from gguf import GGUFReader

    return GGUFReader(path).fields["tokenizer.chat_template"].contents()


class Server:
    """llama-server on a free local port, for /apply-template and /props."""

    def __init__(self, llama_cpp: Path, gguf: Path, flags: list[str], log: Path):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.proc = subprocess.Popen(
            [str(llama_cpp / "build" / "bin" / "llama-server"), "-m", str(gguf), "--host", "127.0.0.1",
             "--port", str(self.port), "-c", "1024", "--no-webui", "--jinja"] + flags,
            stdout=log.open("w"), stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 120
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            if self.proc.poll() is not None or time.monotonic() > deadline:
                self.close()
                raise RuntimeError(f"llama-server {flags} did not start:\n{log.read_text()[-1500:]}")
            time.sleep(0.2)

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=30) as r:
            return json.load(r)

    def post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def first_diff(a: str, b: str) -> str:
    i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"at char {i}: llama.cpp {a[max(0, i - 40):i + 40]!r}, reference {b[max(0, i - 40):i + 40]!r}"


def check_render(server: Server, template: str, label: str, preserve: bool, defaults: dict,
                 requests: list[dict]) -> list[tuple[bool, str]]:
    """requests: the request fields that choose the reasoning mode; defaults: the template
    arguments the server's flags stand for. One result per message set."""
    res = []
    for name, messages, tools in MESSAGES:
        equal = prefill = 0
        bad = []
        for extra in requests:
            official = defaults | vllm_template_kwargs(extra)
            body = {"messages": messages, **extra}
            if tools:
                body["tools"] = tools
            got = server.post("/apply-template", body)["prompt"]
            # the template reads preserve_thinking only when it is defined and true
            want = render_reference(template, messages, tools,
                                    official | ({"preserve_thinking": True} if preserve else {}))
            equal += got == want
            prefill += got.endswith("<|im_start|>assistant\n" + PREFILLED_BLOCK) == (not thinking_enabled(official))
            if got != want and len(bad) < 2:
                bad.append(f"{json.dumps(extra)}: {first_diff(got, want)}")
        n = len(requests)
        res.append((equal == n and prefill == n,
                    f"render [{label}] {name}: {equal}/{n} prompts equal the reference, prefilled think block "
                    f"iff thinking off {prefill}/{n}" + (f"; {' | '.join(bad)}" if bad else "")))
    return res


def check_continue(server: Server, template: str, label: str, preserve: bool, defaults: dict,
                   requests: list[dict]) -> list[tuple[bool, str]]:
    """Continuing the final assistant message, for every way to choose the reasoning mode.
    A content continuation must equal the reference's continue_final_message render. A
    reasoning continuation has no reference: vLLM always renders the message with its think
    block closed, while llama.cpp leaves it open after the reasoning. That prompt is checked
    exactly, and the reference must still close the block."""
    res = []
    for name, messages in CONTINUE_MESSAGES:
        final = messages[-1]
        equal = known = 0
        bad = []
        for extra in requests:
            official = (defaults | vllm_template_kwargs(extra)
                        | ({"preserve_thinking": True} if preserve else {}))
            want = render_reference(template, messages, None, official, continue_final=True)
            for how in CONTINUATIONS:
                got = server.post("/apply-template", {"messages": messages, **extra, **how})["prompt"]
                if continued_in_reasoning(final, how):
                    open_block = (render_reference(template, messages[:-1], None, official, add_generation_prompt=False)
                                  + "<|im_start|>assistant\n<think>\n" + (final.get("reasoning_content") or ""))
                    ok = got == open_block and want.endswith("\n</think>\n\n" + final["content"])
                    known += ok
                    diff = first_diff(got, open_block) if got != open_block else f"reference {want[-60:]!r}"
                else:
                    ok = got == want
                    equal += ok
                    diff = first_diff(got, want)
                if not ok and len(bad) < 2:
                    bad.append(f"{json.dumps(extra | how)}: {diff}")
        n = len(requests) * len(CONTINUATIONS)
        res.append((equal + known == n,
                    f"continue [{label}] {name}: {equal}/{n} prompts equal the reference"
                    + (f", reasoning continued in an open think block {known}/{n} (the reference closes it)"
                       if known else "") + (f"; {' | '.join(bad)}" if bad else "")))
    return res


def check_stop_tokens(llama_cpp: Path, gguf: Path, what: str) -> list[tuple[bool, str]]:
    ll = libllama(llama_cpp)
    lib, ffi = ll.lib, ll.ffi
    quiet = ffi.callback("void(enum ggml_log_level, const char *, void *)", lambda *_: None)
    lib.llama_log_set(quiet, ffi.NULL)
    model = lib.llama_model_load_from_file(str(gguf).encode(), ll.model_default_params(vocab_only=True))
    if not model:
        return [(False, f"stop tokens [{what}]: failed to load {gguf}")]
    vocab = lib.llama_model_get_vocab(model)
    buf = ffi.new("char[]", 256)

    def piece(t: int, special: bool) -> str:
        n = lib.llama_token_to_piece(vocab, t, buf, len(buf), 0, special)
        return bytes(ffi.buffer(buf, n)).decode(errors="replace")

    n_vocab = lib.llama_vocab_n_tokens(vocab)
    eog = [t for t in range(n_vocab) if lib.llama_vocab_is_eog(vocab, t)]
    eos, eot, pad = lib.llama_vocab_eos(vocab), lib.llama_vocab_eot(vocab), lib.llama_vocab_pad(vocab)
    tags = {t: piece(t, False) for t in TAGS}
    tag_eog = [t for t in TAGS if lib.llama_vocab_is_eog(vocab, t) or lib.llama_vocab_is_control(vocab, t)]
    control = {t: (piece(t, False), piece(t, True)) for t in CONTROL}
    lib.llama_model_free(model)
    return [
        (eog == EOS_IDS, f"stop tokens [{what}]: libllama EOG tokens over all {n_vocab} ids {eog} "
                         f"(generation_config.json eos_token_id {EOS_IDS})"),
        (eos == EOS and pad == PAD and eot in EOS_IDS,
         f"stop tokens [{what}]: eos {eos}, pad {pad} (tokenizer_config.json: {EOS}, {PAD}), eot {eot} "
         f"(one of {EOS_IDS}; the GGUF has no eot key, libllama picks it by token text)"),
        (tags == TAGS and not tag_eog, f"stop tokens [{what}]: tag tokens render as text {list(tags.values())}, "
                                       f"none is EOG or control"),
        (all(a == "" and b == CONTROL[t] for t, (a, b) in control.items()),
         f"stop tokens [{what}]: control tokens render as '' without special, as their text with special "
         f"({', '.join(f'{t}: {b!r}' for t, (_, b) in control.items())})"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True,
                    help="llama.cpp checkout with a built libllama and llama-server")
    ap.add_argument("--vocab", type=Path, help="vocab GGUF (default: models/ggml-vocab-kolibri.gguf in --llama-cpp)")
    ap.add_argument("--seed", type=int, default=1, help="cmd/kolibri-tiny random seed")
    args = ap.parse_args()
    llama_cpp = args.llama_cpp.resolve()
    vocab = (args.vocab or llama_cpp / "models" / "ggml-vocab-kolibri.gguf").resolve()

    results: list[tuple[bool, str]] = []
    template = json.loads((tokenizer_dir() / "tokenizer_config.json").read_text())["chat_template"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gguf = convert(generate_tiny(tmp, args.seed), llama_cpp, "f32")
        fork_file = llama_cpp / TEMPLATE_FILE
        sources = {
            "tokenizer_config.json": template,
            "vocab GGUF": gguf_template(llama_cpp, vocab),
            "converted GGUF": gguf_template(llama_cpp, gguf),
            TEMPLATE_FILE: fork_file.read_text() if fork_file.exists() else "",
        }
        sha = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sources.items()}

        # the reference's kwarg sets, the OpenAI field per level, and the field next to an enable_thinking kwarg
        requests = ([{"chat_template_kwargs": kw} for kw in THINKING_OFF_KWARGS + THINKING_ON_KWARGS]
                    + [{"reasoning_effort": e} for e in EFFORTS]
                    + [{"reasoning_effort": e, "chat_template_kwargs": {"enable_thinking": t}}
                       for e in EFFORTS for t in (True, False)])
        servers = [  # (label, flags, preserve_thinking, template arguments the flags stand for, requests)
            ("default", [], True, {}, requests),
            ("--no-reasoning-preserve", ["--no-reasoning-preserve"], False, {}, requests),
            ("--reasoning off", ["--reasoning", "off"], True, {"enable_thinking": False}, [{}]),
            ("--reasoning-effort low", ["--reasoning-effort", "low"], True, {"reasoning_effort": "low"}, [{}]),
        ]
        for label, flags, preserve, defaults, requests in servers:
            server = Server(llama_cpp, gguf, flags, tmp / "server.log")
            try:
                if label == "default":
                    props = server.get("/props")["chat_template"]
                    sha["llama-server /props"] = hashlib.sha256(props.encode()).hexdigest()
                    results.append((all(s == TEMPLATE_SHA256 for s in sha.values()),
                                    "template: " + ", ".join(f"{k} {v[:12]}" for k, v in sha.items())
                                    + f" (released {TEMPLATE_SHA256[:12]}, {len(template)} chars)"))
                results += check_render(server, template, label, preserve, defaults, requests)
                results += check_continue(server, template, label, preserve, defaults, requests)
            finally:
                server.close()

        results += check_stop_tokens(llama_cpp, vocab, "vocab GGUF")
        results += check_stop_tokens(llama_cpp, gguf, "converted GGUF")

    errs = 0
    for ok, what in results:
        print(f"{'PASS' if ok else 'FAIL'} chat {what}", flush=True)
        errs += not ok
    if errs:
        print(f"FAIL {errs} problem(s)")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
