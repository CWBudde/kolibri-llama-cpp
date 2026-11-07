#!/usr/bin/env python3
"""Capture the residual stream at the last prompt token, through libllama's cb_eval, for issue #63.

A layer N of the issue (1 to n_layer) is the output of the N-th block, node l_out-(N-1) of
src/models/kolibri1.cpp, which numbers its blocks from 0. It is the block's residual sum, after the
post-FFN norm and before the next block's input norm; layer n_layer is that node of the last block,
before output_norm (result_norm). The first decode checks this against the graph and fails if it
does not hold. docs/layer-probe.md has the details.

cb_eval only reads: it makes the scheduler stop the graph at a read node, and ggml_backend_tensor_get
copies the tensor out. All prompts are first decoded by a model and context without any callback, and
the logits of the capturing decodes must equal those, exactly. With --repeat the first prompts are
decoded again at the end, and their activations and logits must equal the first time too.

The output directory holds layer-<N>.npy, float32 [n_rows, n_embd] in the dataset's row order, and
capture.json with what kolibri-probe report checks, and the model's, libllama's and the template's
hashes. Nothing is written through the model's path.

    capture.py --llama-cpp third_party/llama.cpp --model model.gguf --out DIR \
        [--dataset testdata/probe/smoke.jsonl] [--layers 10,25,40,50] [--expert-cache N] [--device MTL0]
"""

import argparse
import fcntl
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "gguf"))
sys.path.insert(0, str(ROOT / "tools" / "tokenizer"))
from check_model import Runner  # noqa: E402

SCHEMA = "kolibri-probe-capture-v1"
DEFAULT_LAYERS = "10,25,40,50"
DEFAULT_DATASET = ROOT / "testdata" / "probe" / "smoke.jsonl"
# The chat template's reasoning mode. "none" prefills the empty think block, so the last prompt
# token is where the answer starts: the point of the execute decision. A definition: see the doc.
DEFAULT_REASONING = "none"
PILOT_REPEATS = 4
# build_norm of the post-FFN norm, as the graph holds it, from the l_out add down
POST_FFN_OPS = ["MUL", "RMS_NORM", "ADD"]
POST_FFN_WEIGHT = "post_ffw_norm.weight"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if platform.system() == "Darwin":
            fcntl.fcntl(f.fileno(), fcntl.F_NOCACHE, 1)  # hashing 35 GB must not fill the page cache
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def node_for_layer(layer: int) -> str:
    """The graph node of the issue's layer N: one-based in the issue, zero-based in the graph."""
    return f"l_out-{layer - 1}"


def parse_layers(text: str, n_layer: int) -> list[int]:
    layers = [int(x) for x in text.split(",")]
    if sorted(set(layers)) != layers:
        raise SystemExit(f"--layers {text}: want increasing layers")
    for n in layers:
        if not 1 <= n <= n_layer:
            raise SystemExit(f"--layers: layer {n} outside 1 to {n_layer} (one-based, as in the issue)")
    return layers


def render(template: str, prompt: str, reasoning: str | None) -> str:
    """The reference's renderer (see tools/chat/check_chat.py), with the arguments vLLM passes."""
    from transformers.utils.chat_template_utils import render_jinja_template

    kwargs = {} if reasoning is None else {"reasoning_effort": reasoning}
    rendered, _ = render_jinja_template(conversations=[[{"role": "user", "content": prompt}]],
                                        chat_template=template, add_generation_prompt=True, **kwargs)
    return rendered[0]


class Capture:
    """One model and one context, with or without the callback (the expert cache serves one context
    per model, so the two passes load the model apart). Prompts are decoded on a cleared memory."""

    def __init__(self, llama_cpp: Path, model: Path, device: str | None, n_ctx: int, n_ubatch: int, threads: int,
                 expert_cache: int, stats_path: Path | None = None, mutate: str | None = None, callback: bool = True):
        self.r = Runner(llama_cpp)
        self.ffi, self.lib = self.r.ffi, self.r.lib
        self.mutate = mutate  # control M77 only: a name of a node to overwrite
        devs = self.r.devices()
        names = [n for n, _ in devs]
        self.device = device or next((n for n in names if n != "CPU"), "CPU")
        if self.device not in names:
            raise SystemExit(f"--device {self.device}: this build has {names}")
        dev = dict(devs)[self.device]
        self.dev_list = self.ffi.new("ggml_backend_dev_t[]", [dev, self.ffi.NULL])
        no_repack = expert_cache > 0
        mparams = self.r.ll.model_default_params(devices=self.dev_list, n_gpu_layers=-1, use_extra_bufts=not no_repack)
        self.keepalive = []
        if expert_cache:
            ec = self.ffi.new("struct llama_expert_cache_params *", self.lib.llama_expert_cache_default_params())
            ec.n_slots = expert_cache
            if stats_path is not None:
                path = self.ffi.new("char[]", str(stats_path).encode())
                self.keepalive.append(path)
                ec.stats_path = path
            self.keepalive.append(ec)
            mparams.expert_cache = ec
        self.model = self.lib.llama_model_load_from_file(str(model).encode(), mparams)
        if not self.model:
            raise SystemExit("load failed: " + self.r.errors())
        self.vocab = self.lib.llama_model_get_vocab(self.model)
        self.n_vocab = self.lib.llama_vocab_n_tokens(self.vocab)
        self.n_layer = self.lib.llama_model_n_layer(self.model)
        self.n_embd = self.lib.llama_model_n_embd(self.model)
        self.n_ctx, self.n_ubatch, self.threads = n_ctx, n_ubatch, threads
        self.want: set[str] = set()
        self.parts: dict[str, list] = {}
        self.nodes: dict[str, list[list[str]]] = {}
        self.post_ffn: dict[str, dict] = {}
        self.record_nodes = False

        @self.ffi.callback("bool(struct ggml_tensor *, bool, void *)")
        def on_node(t, ask, user_data):
            name = self.ffi.string(t.name).decode()
            if ask:
                if self.record_nodes:
                    srcs = [self.ffi.string(s.name).decode() for s in t.src if s != self.ffi.NULL]
                    self.nodes.setdefault(name, []).append(srcs)
                    if name in self.want:
                        self.post_ffn[name] = self._post_ffn(t)
                return name in self.want
            if name in self.want:
                self._on_read(t, name)
            return True

        self._on_node = on_node
        self.ctx = self._context(on_node if callback else None)
        self.n_ubatch_effective = self.lib.llama_n_ubatch(self.ctx)
        self.batch = self.lib.llama_batch_init(n_ctx, 0, 1)

    def _context(self, cb):
        cparams = self.r.ll.context_default_params(n_ctx=self.n_ctx, n_batch=self.n_ctx, n_ubatch=self.n_ubatch,
                                                   n_threads=self.threads, n_threads_batch=self.threads,
                                                   op_offload=False, no_perf=True)
        if cb is not None:
            cparams.cb_eval = cb
        ctx = self.lib.llama_init_from_model(self.model, cparams)
        if not ctx:
            raise SystemExit("context failed: " + self.r.errors())
        return ctx

    def _on_read(self, t, name: str) -> None:
        """Keeps one graph computation's rows: [n_rows, n_embd], from [1, 1, n_rows, n_embd] in numpy order."""
        import numpy as np

        if t.type != self.lib.GGML_TYPE_F32:
            raise RuntimeError(f"{name}: ggml type {t.type}, not F32")
        if self.mutate == name:  # control M77: a capture that writes into the graph
            data = np.full(int(self.lib.ggml_nbytes(t)) // 4, 1.5, np.float32)
            self.lib.ggml_backend_tensor_set(t, self.ffi.from_buffer(data), 0, data.nbytes)
        part = self.r._read(t, np)
        if part.shape[0] != 1 or part.shape[1] != 1 or part.shape[3] != self.n_embd:
            raise RuntimeError(f"{name}: shape {part.shape}, want [1, 1, n_rows, {self.n_embd}]")
        self.parts[name].append(part[0, 0])

    def _post_ffn(self, l_out) -> dict:
        """The residual sum's first source, walked by pointer, since those nodes have no names of their
        own: the weight's multiply, the RMS norm under it and the sum of the experts under that."""
        names = lambda t: [self.ffi.string(s.name).decode() for s in t.src if s != self.ffi.NULL]
        ops, weight, norm, experts = [], None, None, None
        t = l_out.src[0]
        for depth in range(3):
            if t == self.ffi.NULL:
                break
            ops.append(self.ffi.string(self.lib.ggml_op_name(t.op)).decode())
            srcs = names(t)
            if depth == 0:
                weight = srcs[1] if len(srcs) > 1 else None
            elif depth == 1:
                norm = self.ffi.string(t.name).decode()
            else:
                experts = srcs
            t = t.src[0]
        return {"ops": ops, "weight": weight, "norm": norm, "experts": experts}

    def tokenize(self, text: str) -> list[int]:
        raw = text.encode()
        n_max = len(raw) + 8
        buf = self.ffi.new("llama_token[]", n_max)
        n = self.lib.llama_tokenize(self.vocab, raw, len(raw), buf, n_max, True, True)
        if n < 0:
            raise RuntimeError(f"tokenize: {n}")
        return [buf[i] for i in range(n)]

    def piece(self, token: int) -> str:
        buf = self.ffi.new("char[]", 64)
        n = self.lib.llama_token_to_piece(self.vocab, token, buf, 64, 0, True)
        return bytes(self.ffi.buffer(buf, max(n, 0))).decode(errors="replace")

    def template(self) -> str:
        t = self.lib.llama_model_chat_template(self.model, self.ffi.NULL)
        if t == self.ffi.NULL:
            raise SystemExit("the model has no tokenizer.chat_template")
        return self.ffi.string(t).decode()

    def meta(self, key: str) -> str | None:
        buf = self.ffi.new("char[]", 256)
        n = self.lib.llama_model_meta_val_str(self.model, key.encode(), buf, 256)
        return self.ffi.string(buf).decode() if n >= 0 else None

    def decode(self, tokens: list[int], all_outputs: bool = False):
        """Decodes on a cleared memory with an output at the last position (or all). The logits of
        the last position, as a copy."""
        import numpy as np

        if len(tokens) > self.n_ctx:
            raise RuntimeError(f"{len(tokens)} tokens, --n-ctx {self.n_ctx}")
        ctx = self.ctx
        self.lib.llama_memory_clear(self.lib.llama_get_memory(ctx), True)
        b = self.batch
        for i, tok in enumerate(tokens):
            b.token[i] = tok
            b.pos[i] = i
            b.n_seq_id[i] = 1
            b.seq_id[i][0] = 0
            b.logits[i] = all_outputs or i == len(tokens) - 1
        b.n_tokens = len(tokens)
        if (rc := self.lib.llama_decode(ctx, b)) != 0:
            raise RuntimeError(f"llama_decode returned {rc}: " + self.r.errors())
        return np.frombuffer(self.ffi.buffer(self.lib.llama_get_logits_ith(ctx, -1), 4 * self.n_vocab),
                             dtype=np.float32).copy()

    def capture_rows(self, tokens: list[int], layers: list[int], all_outputs: bool = True):
        """Logits and {layer: [n_rows, n_embd] of the prompt's last graph computation}. Checks that
        each node saw every token over its ubatches: the last block, whose rows the graph cuts to the
        outputs, saw the outputs only."""
        import numpy as np

        names = {n: node_for_layer(n) for n in layers}
        self.want = set(names.values())
        self.parts = {name: [] for name in self.want}
        logits = self.decode(tokens, all_outputs)
        out = {}
        for n, name in names.items():
            parts = self.parts[name]
            if not parts:
                raise RuntimeError(f"layer {n}: node {name} was not read; the graph has no such node")
            total = sum(p.shape[0] for p in parts)
            want = len(tokens) if n != self.n_layer or all_outputs else 1
            if total != want or parts[-1].shape[0] == 0:
                raise RuntimeError(f"layer {n}: node {name} has {total} rows over {len(parts)} ubatches, want {want}, "
                                   f"the last token's in the last ubatch")
            out[n] = parts[-1]
        self.want = set()
        return logits, out

    def capture(self, tokens: list[int], layers: list[int]):
        """Logits and {layer: activation at the last prompt token}, with an output there only."""
        logits, rows = self.capture_rows(tokens, layers, all_outputs=False)
        return logits, {n: r[-1] for n, r in rows.items()}

    def check_graph(self, tokens: list[int], layers: list[int]) -> dict:
        """Reads this model's graph during one decode and checks it with check_wiring."""
        self.record_nodes = True
        self.nodes, self.post_ffn = {}, {}
        try:
            self.capture(tokens, layers)
        finally:
            self.record_nodes = False
        return check_wiring(self.nodes, self.post_ffn, layers, self.n_layer)

    def close(self) -> None:
        self.lib.llama_free(self.ctx)
        self.lib.llama_batch_free(self.batch)
        self.lib.llama_model_free(self.model)


def check_wiring(nodes: dict, post_ffn: dict, layers: list[int], n_layer: int) -> dict:
    """What the numbering and the normalization rest on: a layer's l_out is the residual add of
    ffn_inp and post_ffw_norm(routed + shared experts), feeds the next block's first norm, and the
    last one feeds output_norm. Any other wiring refuses the capture."""
    out = {}
    for n in layers:
        il, name = n - 1, node_for_layer(n)
        srcs = nodes.get(name)
        if not srcs or len(srcs[-1]) != 2 or srcs[-1][1] != f"ffn_inp-{il}":
            raise SystemExit(f"layer {n}: {name} <- {srcs}, want the post-FFN norm and ffn_inp-{il}")
        norm = post_ffn.get(name)
        want = {"ops": POST_FFN_OPS, "weight": f"blk.{il}.{POST_FFN_WEIGHT}", "norm": f"norm-{il}",
                "experts": [f"ffn_moe_out-{il}", f"ffn_shexp_out-{il}"]}
        if norm != want:
            raise SystemExit(f"layer {n}: {name}'s first source is {norm}, want the post-FFN norm {want}")
        following = f"norm-{il + 1}" if il + 1 < n_layer else "norm"
        if [name] not in nodes.get(following, []):
            raise SystemExit(f"layer {n}: {following} <- {nodes.get(following)}, does not read {name}")
        out[name] = {"sources": srcs[-1], "post_ffn": norm, "read_by": following}
    return out


def git_head(path: Path) -> str | None:
    r = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def released_template_sha() -> tuple[str | None, str | None]:
    """The released template's sha256 and the HF revision it is pinned at, if the tokenizer files are cached."""
    from common import pinned_source, tokenizer_dir

    try:
        path = tokenizer_dir() / "tokenizer_config.json"
    except Exception:  # offline without the cache: the capture still records the model's template
        return None, None
    text = json.loads(path.read_text())["chat_template"]
    return hashlib.sha256(text.encode()).hexdigest(), pinned_source()[1]


def run(args, mutate_node: str | None = None) -> int:
    """mutate_node is control M77 only: tools/probe/check_capture.py passes it, the command line cannot."""
    import numpy as np

    if (args.out / "capture.json").exists():
        raise SystemExit(f"{args.out} already holds a capture, and results may exist nowhere else: choose a new directory")
    dataset_bytes = args.dataset.read_bytes()
    rows = [json.loads(line) for line in dataset_bytes.decode().splitlines()]
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model_sha = sha256_file(args.model)
    stats_path = args.out / "expert-cache-stats.json" if args.expert_cache else None
    session = lambda stats, callback: Capture(args.llama_cpp, args.model, args.device, args.n_ctx, args.ubatch, args.threads,
                                              args.expert_cache, stats, mutate_node if callback else None, callback)
    reasoning = None if args.reasoning == "default" else args.reasoning

    # the plain pass: a model and context that never had a callback
    plain_logits, t_plain = {}, 0.0
    if not args.skip_logits_check:
        c = session(None, False)
        parse_layers(args.layers, c.n_layer)
        template = c.template()
        for row in rows:
            tokens = c.tokenize(render(template, row["prompt"], reasoning))
            t0 = time.perf_counter()
            plain_logits[row["id"]] = c.decode(tokens)
            t_plain += time.perf_counter() - t0
        c.close()

    c = session(stats_path, True)
    layers = parse_layers(args.layers, c.n_layer)
    template = c.template()
    released, hf_revision = released_template_sha()
    template_sha = hashlib.sha256(template.encode()).hexdigest()

    records, acts = [], {n: [] for n in layers}
    logits_first = []
    graph = None
    differs = []
    t_capture = 0.0
    for row in rows:
        text = render(template, row["prompt"], reasoning)
        tokens = c.tokenize(text)
        if graph is None:
            graph = c.check_graph(tokens, layers)
        t0 = time.perf_counter()
        logits, act = c.capture(tokens, layers)
        t_capture += time.perf_counter() - t0
        if row["id"] in plain_logits and not np.array_equal(plain_logits[row["id"]], logits):
            differs.append(row["id"])
        for n in layers:
            acts[n].append(act[n])
        if len(logits_first) < PILOT_REPEATS:
            logits_first.append((tokens, logits))
        records.append({"id": row["id"], "n_tokens": len(tokens), "last_token": tokens[-1],
                        "last_piece": c.piece(tokens[-1]), "rendered_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "tokens": tokens})
        print(f"{row['id']}: {len(tokens)} tokens, last token {tokens[-1]} {c.piece(tokens[-1])!r}", flush=True)

    repeatable = None
    if args.repeat:
        repeatable = True
        for i, (tokens, logits) in enumerate(logits_first):
            again, act = c.capture(tokens, layers)
            same = np.array_equal(again, logits) and all(np.array_equal(act[n], acts[n][i]) for n in layers)
            print(f"repeat {records[i]['id']}: {'equal' if same else 'DIFFERENT'}")
            repeatable = repeatable and same

    layer_records = []
    for n in layers:
        matrix = np.stack(acts[n]).astype("<f4")
        if not np.isfinite(matrix).all():
            print(f"FAIL layer {n}: a non-finite activation")
            return 1
        path = args.out / f"layer-{n}.npy"
        np.save(path, matrix)
        layer_records.append({"one_based": n, "zero_based": n - 1, "node": node_for_layer(n), "file": path.name,
                              "sha256": sha256_file(path), "dtype": "float32", "shape": list(matrix.shape),
                              "normalization": "pre output_norm; post-FFN norm inside the residual sum"})
    last_tokens = sorted({r["last_token"] for r in records})
    lib_dir = args.llama_cpp / "build" / "bin"
    lib = next(p for p in (lib_dir / "libllama.dylib", lib_dir / "libllama.so") if p.exists())
    manifest = {
        "schema": SCHEMA,
        "dataset": str(args.dataset), "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "model": str(args.model), "model_realpath": str(args.model.resolve()), "model_sha256": model_sha,
        "model_general_name": c.meta("general.name"), "model_file_type": c.meta("general.file_type"),
        "model_architecture": c.meta("general.architecture"),
        "n_layer": c.n_layer, "n_embd": c.n_embd,
        # libllama does what a server would; there is no server process in this path
        "server": str(lib.resolve()), "server_sha256": sha256_file(lib.resolve()),
        "backend_sha256": {p.name: sha256_file(p) for p in sorted(lib_dir.glob("libggml*"))
                           if not p.is_symlink() and (p.suffix in (".dylib", ".so") or ".so." in p.name)},
        "llama_cpp_head": git_head(args.llama_cpp),
        "template_source": "tokenizer.chat_template of the model",
        "template_sha256": template_sha,
        "template_revision": hf_revision,
        "template_is_released": template_sha == released if released else None,
        "reasoning": args.reasoning,
        "add_generation_prompt": True,
        "last_tokens": last_tokens,
        "device": c.device, "threads": args.threads, "n_ctx": args.n_ctx,
        "n_ubatch_requested": args.ubatch, "n_ubatch_effective": c.n_ubatch_effective,
        "expert_cache_slots": args.expert_cache,
        "row_read": "the last row of the node in the prompt's last ubatch",
        "graph_check": graph,
        "logits_check": "differ" if differs else ("skipped" if args.skip_logits_check else "equal"),
        "logits_differ": differs,
        "repeatable": repeatable,
        "wall_seconds": time.time() - started,
        "plain_decode_seconds": t_plain, "capture_decode_seconds": t_capture,
        "rows": records,
        "layers": layer_records,
    }
    (args.out / "capture.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    c.close()
    if differs:
        print(f"FAIL the capture changed the logits of {differs}")
        return 1
    if repeatable is False:
        print("FAIL a repeated decode differs")
        return 1
    checked = manifest["logits_check"] == "equal" and repeatable is True
    print(f"{'PASS' if checked else 'UNCHECKED, kolibri-probe report refuses it:'} captured {len(rows)} prompts at layers {layers}; "
          f"logits {manifest['logits_check']} against a context without a callback; repeat {repeatable}; {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True, help="llama.cpp checkout with a built libllama")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="new directory for the activations and capture.json")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--layers", default=DEFAULT_LAYERS, help="one-based layers, as in the issue; scale them on a smaller model")
    ap.add_argument("--device", help="CPU or MTL0 (default: the GPU if there is one)")
    ap.add_argument("--n-ctx", type=int, default=512, help="context; a prompt must fit")
    ap.add_argument("--ubatch", type=int, default=512, help="physical batch; the expert cache caps it at slots/k")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--expert-cache", type=int, default=0, help="expert cache slots per layer, 0 for resident experts")
    ap.add_argument("--reasoning", default=DEFAULT_REASONING, help="reasoning_effort for the template, or 'default' to leave it unset")
    ap.add_argument("--repeat", action=argparse.BooleanOptionalAction, default=True, help="decode the first prompts again and compare; without it, kolibri-probe report refuses the capture")
    ap.add_argument("--skip-logits-check", action="store_true", help="no decode without a callback; capture.json says skipped, and kolibri-probe report refuses it")
    args = ap.parse_args()
    args.model = args.model.expanduser()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
