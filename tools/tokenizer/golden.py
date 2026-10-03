#!/usr/bin/env python3
"""Generate the Kolibri-1 tokenizer golden file from the reference tokenizer.

The reference is the HF `tokenizers` library loading the released
tokenizer.json, which is what vLLM (through transformers) runs. Every case is
also encoded through transformers.AutoTokenizer; the two must agree.

    golden.py            # rewrite testdata/tokenizer/golden.jsonl
    golden.py --check    # verify the committed file against the installed reference
    golden.py --upstream third_party/llama.cpp
                         # write models/ggml-vocab-kolibri.gguf.{inp,out} for
                         # llama.cpp's test-tokenizer-0, exactly as
                         # convert_hf_to_gguf_update.py would

Each line holds {"name", "text", "ids", "decoded"}. "decoded" is the reference
decode(ids, skip_special_tokens=False); it equals "text" for every case
unless the case says otherwise.
"""

import argparse
import ast
import json
import sys
from pathlib import Path

from common import GOLDEN, tokenizer_dir

GERMAN = {
    "bundessozialgerichtes": "Bundessozialgerichtes",
    "protokolldaten": "Protokolldaten",
    "compounds-lower": "bundessozialgerichtes protokolldaten",
    "compounds-upper": "BUNDESSOZIALGERICHTES PROTOKOLLDATEN",
    "compound-long": "Donaudampfschifffahrtsgesellschaftskapitän",
    "compound-law": "Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz",
    "compound-hyphen": "Kfz-Haftpflichtversicherung und E-Mail-Adresse",
    "sentence": "Das Bundessozialgericht hat in seinem Urteil vom 3. Oktober 2026 entschieden, dass die Protokolldaten gelöscht werden müssen.",
    "sentence-leading-space": " Die Datenschutz-Grundverordnung (DSGVO) gilt seit dem 25.05.2018.",
}

UMLAUTS = {
    "umlauts": "Ärger über Öl und Übermut, schön, müde, Bär",
    "umlauts-isolated": "ä ö ü Ä Ö Ü ß ẞ",
    "eszett": "Straße STRASSE Straẞe Maß Mass Fußball",
    "nfd": "Ma\u0308dchen und Bu\u0308cher",  # decomposed umlauts; there is no normalizer
    "nfc-vs-nfd": "Mädchen Ma\u0308dchen",
    "french-spanish": "Déjà vu, garçon, naïve, coöperation, mañana, pingüino",
}

ENGLISH = {
    "plain": "The quick brown fox jumps over the lazy dog.",
    "contractions": "I've been told he's there, you're sure? We'll see, I'd say it doesn't matter.",
    "contractions-case": "I'VE YOU'RE He'S 'Tis 'twas rock'n'roll O'Neill",
    "apostrophes": "'s 't 're 've 'm 'll 'd ''' ' '",
    "numbers": "In 2026, 3.14159 and 1,000,000 and 1e-6 and 0xDEADBEEF and 12345678901234567890.",
    "punctuation": "Hello, world! What's up?! ... -- (parenthesis) [brackets] {braces} \"quotes\" ‘curly’ “double”",
    "url-email": "See https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16 or mail info@example.com.",
}

WHITESPACE = {
    "empty": "",
    "space": " ",
    "spaces-2": "  ",
    "spaces-3": "   ",
    "tab": "\t",
    "tabs": "\t\t",
    "newline": "\n",
    "newlines-3": "\n\n\n",
    "crlf": "line one\r\nline two\r\n",
    "cr": "a\rb",
    "mixed": " \n \t\n  \n",
    "trailing": "word   ",
    "leading": "   word",
    "inner": "a  b   c    d",
    "paragraphs": "Erster Absatz.\n\nZweiter Absatz.\n\n\nDritter.",
    "nbsp": "10\u00a0km und 5\u202fkg",
    "unicode-spaces": "a\u2003b\u3000c\u2028d\u2029e\u0085f",
    "long-run": " " * 600 + "x",
    "indent-newlines": "\n    indented\n\tTabbed\n",
}

UNICODE = {
    "emoji": "🚀 😀 👍🏽 ❤\ufe0f",
    "emoji-zwj": "👩\u200d💻 👨\u200d👩\u200d👧\u200d👦 🏳\ufe0f\u200d🌈",
    "flags": "🇩🇪🇺🇸🇪🇺",
    "cjk": "我想在苹果公司工作。東京は日本の首都です。서울은 한국의 수도입니다.",
    "arabic-hebrew": "مرحبا بالعالم שלום עולם",
    "indic": "नमस\u094dत\u0947 द\u0941निया বাংলা தமிழ\u0bcd",
    "thai-khmer": "สว\u0e31สด\u0e35ชาวโลก កាន\u17cbតែព\u17b7សេសអាច",
    "cyrillic-greek": "Привет, мир! Γειά σου Κόσμε",
    "combining": "e\u0301 a\u0300\u0301\u0302 Z\u0335\u0321a\u0337l\u0336g\u0334o",
    "math": "∀x ∈ ℝ: x² ≥ 0, ∑ᵢ aᵢ = ∫₀¹ f(x) dx ≈ π",
    "symbols": "© ® ™ € £ ¥ § ¶ † ‡ • … ‰ ← → ↑ ↓",
    "invisible": "a\u200bb\u200cc\u200dd\u2060e\ufefff",
    "bidi": "abc\u200fdef\u202ehij\u202c",
    "replacement-char": "� broken �",
    "private-use": "\U000f0000",
    "astral": "\U0001d400\U0001f9a9\U00020000\U0010fffd",
    "noncharacters": "﷐￾￿\U0010ffff",
    "control-chars": "a\x00b\x01c\x07d\x1be\x7ff",
    "superscript-digits": "x² x³ ½ ¾ ① ⅷ ٣ ३ ０１２",
}

CODE = {
    "python": 'def fib(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n',
    "go": "func main() {\n\tfor i := 0; i < 10; i++ {\n\t\tfmt.Printf(\"%d\\n\", i)\n\t}\n}\n",
    "cpp": "template <typename T>\nstd::vector<T> f(const std::vector<T> & v) {\n    return {v.begin(), v.end()}; // copy\n}\n",
    "json": '{"name": "Kolibri", "params": [1, 2.5, -3e-4], "nested": {"ok": true, "x": null}}',
    "html": '<div class="box"><p>Hallo &amp; tsch&uuml;ss</p><br/></div>',
    "shell": "for f in *.txt; do\n  grep -n 'foo' \"$f\" | wc -l\ndone 2>/dev/null",
    "sql": "SELECT id, name FROM users WHERE created_at >= '2026-01-01' ORDER BY id DESC LIMIT 10;",
    "regex": r"^(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+)@\w+\.\w{2,}$",
    "operators": "a+=b; c->d; e::f; g<<=1; h!==i; j===k; l??m; n?.o; p=>q; ...rest",
    "deep-indent": "\n".join(" " * (4 * i) + f"level{i}" for i in range(8)),
}

SPECIAL = {
    "im-start-end": "<|im_start|>user\nHallo, wie geht es dir?<|im_end|>\n<|im_start|>assistant\n",
    "adjacent": "<|im_end|><|im_start|><|im_end|>",
    "think": "<think>\nDer Nutzer fragt nach dem Wetter.\n</think>\n\nEs ist sonnig.",
    "think-inline": "abc<think>def</think>ghi",
    "tool-call": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Heidelberg"}}\n</tool_call>',
    "tool-response": "<tool_response>\n{\"temperature\": 21}\n</tool_response>",
    "role-tokens": "<|text|><|endoftext|><|pad|><|chat|><|/role|>",
    "pii": "Mail <|pii-email-address|>, IBAN <|pii-iban|>, IP <|pii-ip-address|>, Tel <|pii-phone-number|> <|pii_4|><|pii_9|>",
    # The IDs in tokenizer.json's added_tokens skip 127923 and 127924, but the
    # tokenizers library numbers added tokens contiguously, so the runtime ID
    # of <|reserved-token-2|> is 127923, not the 127925 listed in the file.
    "reserved-shifted": "<|reserved-token-2|><|reserved-token-3|><|reserved-token-76|>",
    "reserved-missing": "<|reserved-token-0|><|reserved-token-1|><|reserved-token-77|>",
    "spaces-around": " <|im_end|> \n<|im_start|> x <think> y </think> ",
    "partial": "<|im_end <|im_start| <think </think <|im-end|> <THINK>",
    "nested-brackets": "<<|im_end|>> <<think>>",
}


def chat_cases(tok) -> dict[str, str]:
    """Renders the released chat template, which exercises special tokens,
    reasoning and tool calls the way a server would send them."""
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Aktuelles Wetter für eine Stadt",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }]
    messages = [
        {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
        {"role": "user", "content": "Wie ist das Wetter in Heidelberg?"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Ich sollte das Wetter-Tool aufrufen.",
            "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Heidelberg"}}}],
        },
        {"role": "tool", "content": '{"temperature": 21, "condition": "sonnig"}'},
        {"role": "assistant", "content": "In Heidelberg ist es sonnig bei 21 °C."},
        {"role": "user", "content": "Danke! Und morgen?"},
    ]
    return {
        "template-simple": tok.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True),
        "template-tools": tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True),
    }


def all_cases(tok) -> list[tuple[str, str]]:
    groups = {
        "german": GERMAN, "umlauts": UMLAUTS, "english": ENGLISH, "whitespace": WHITESPACE,
        "unicode": UNICODE, "code": CODE, "special": SPECIAL, "chat": chat_cases(tok),
    }
    return [(f"{g}/{k}", v) for g, cases in groups.items() for k, v in cases.items()]


def upstream_tests(llama_cpp: Path) -> list[str]:
    """Extracts the `tests` list from convert_hf_to_gguf_update.py without
    running that script (it downloads every registered tokenizer)."""
    tree = ast.parse((llama_cpp / "convert_hf_to_gguf_update.py").read_text(encoding="utf-8"))
    consts: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "CHK_TXT":
                consts[name] = ast.literal_eval(node.value)
            elif name == "tests":
                return [consts[e.id] if isinstance(e, ast.Name) else ast.literal_eval(e) for e in node.value.elts]
    raise SystemExit("tests list not found in convert_hf_to_gguf_update.py")


def write_upstream(llama_cpp: Path, auto) -> None:
    base = llama_cpp / "models" / "ggml-vocab-kolibri.gguf"
    tests = upstream_tests(llama_cpp)
    with open(f"{base}.inp", "w", encoding="utf-8") as f:
        f.writelines(f"{text}\n__ggml_vocab_test__\n" for text in tests)
    with open(f"{base}.out", "w") as f:
        f.writelines("".join(f" {r}" for r in auto.encode(text, add_special_tokens=False)) + "\n" for text in tests)
    print(f"wrote {len(tests)} upstream tests to {base}.inp/.out")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare instead of rewriting")
    ap.add_argument("--upstream", type=Path, metavar="LLAMA_CPP", help="write test-tokenizer-0 files into this llama.cpp checkout")
    args = ap.parse_args()

    import tokenizers
    import transformers
    from transformers import AutoTokenizer

    d = tokenizer_dir()
    ref = tokenizers.Tokenizer.from_file(str(d / "tokenizer.json"))
    # fix_mistral_regex=False only silences a false-positive warning: the
    # heuristic fires for any local config.json without transformers_version.
    auto = AutoTokenizer.from_pretrained(d, fix_mistral_regex=False)

    if args.upstream:
        write_upstream(args.upstream, auto)
        return

    records = []
    for name, text in all_cases(auto):
        ids = ref.encode(text, add_special_tokens=False).ids
        auto_ids = auto.encode(text)  # what vLLM does for a prompt; Kolibri adds no BOS/EOS
        if ids != auto_ids:
            sys.exit(f"{name}: tokenizers {ids} != AutoTokenizer {auto_ids}")
        decoded = ref.decode(ids, skip_special_tokens=False)
        records.append({"name": name, "text": text, "ids": ids, "decoded": decoded})

    if args.check:
        from common import load_golden
        want = load_golden()
        bad = [f"{w['name']}: golden {w['ids']} != now {r['ids']}" for w, r in zip(want, records) if w != r]
        if len(want) != len(records) or bad:
            sys.exit("\n".join(bad) or f"{len(want)} golden cases, {len(records)} generated")
        print(f"{len(records)} golden cases match tokenizers {tokenizers.__version__}, transformers {transformers.__version__}")
        return

    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    with GOLDEN.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lossy = [r["name"] for r in records if r["decoded"] != r["text"]]
    print(f"wrote {len(records)} cases to {GOLDEN} (tokenizers {tokenizers.__version__}, transformers {transformers.__version__})")
    print(f"cases whose reference decode differs from the input: {lossy or 'none'}")


if __name__ == "__main__":
    main()
