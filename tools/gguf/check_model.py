#!/usr/bin/env python3
"""Load and run the tiny Kolibri GGUF in libllama.

Converts the cmd/kolibri-tiny checkpoint to BF16 and to F32 GGUFs, loads
them through the libllama C API on every backend device, and decodes the
same 100 tokens each time. 100 > sliding_window (65), so the sliding-window
mask cuts in, and the fixture has both layer types (SSSSFF).

- load: llama_model_kolibri accepts both GGUFs (libllama fails on a missing
  tensor, a wrong shape, or a tensor the model class does not use), and the
  logits of all 100 positions are finite;
- no routing (F32, expert_used_count overridden to all 8 experts): decoding
  in ubatches of 16, with the KV cache carrying the context, matches one
  ubatch of 100, and every device matches the CPU (NMSE <= 1e-4, the
  test-llama-archs threshold);
- top-2 routing (F32, as converted): the same comparisons pick the same
  greedy token at every position, and at most 2 of the 100 positions exceed
  NMSE 1e-4;
- graph wiring (CPU, top-2): the graph nodes, recorded through cb_eval, have
  the inputs of the Kolibri block in every layer. The selection bias is
  added to the raw router logits, the expert weights come from the unbiased
  probabilities, ffn_out sums the routed and the shared experts, both
  sandwich norms feed the residual adds, and RoPE nodes exist exactly on the
  sliding-window layers. This is the graph's structure; its numerics against
  the reference are not checked here.

Why the routing is taken out for the strict comparison: top-k selection is
discontinuous. Where two experts score almost the same for a token, a
rounding difference between two code paths picks the other expert, and
that one position moves the NMSE over 1e-4 (observed: 1 of 100 positions).
With all experts active the same comparisons stay below 3e-6 per position.
F32 instead of BF16 for the same reason: the CPU rounds the activations to
BF16 for BF16 matmuls, the other paths do not.

The weights are random, so this checks that the graph runs consistently,
not that it matches the reference implementation.

    check_model.py --llama-cpp third_party/llama.cpp
"""

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

from tiny import ROOT, convert, generate_tiny

ARCH = "kolibri"
N_TOKENS = 100
MAX_NMSE = 1e-4  # tests/test-llama-archs.cpp
# top-k near-ties allowed under real routing, see the module docstring
MAX_FLIPS = 2


def nmse(a, b) -> float:
    return float(((a - b) ** 2).sum() / (a ** 2).sum())


def joined(parts: list, token_axis: int = -2):
    """One captured node over all graph computations: the parts concatenated along the token
    axis (-2 for an activation [.., n_tokens, n_embd], -3 for heads [.., n_tokens, n_head, d])."""
    import numpy as np

    return np.concatenate(parts, axis=token_axis)


class Runner:
    def __init__(self, llama_cpp: Path):
        sys.path.insert(0, str(ROOT / "tools" / "tokenizer"))
        from common import libllama

        self.ll = libllama(llama_cpp)
        self.ffi, self.lib = self.ll.ffi, self.ll.lib
        self.log: list[str] = []

        @self.ffi.callback("void(enum ggml_log_level, const char *, void *)")
        def on_log(level, text, user_data):
            self.log.append(self.ffi.string(text).decode(errors="replace"))

        self._on_log = on_log  # keep the callback alive
        self.lib.llama_log_set(on_log, self.ffi.NULL)

    def devices(self) -> list[tuple[str, object]]:
        devs = []
        for i in range(self.lib.ggml_backend_dev_count()):
            dev = self.lib.ggml_backend_dev_get(i)
            kind = self.lib.ggml_backend_dev_type(dev)
            if kind in (self.lib.GGML_BACKEND_DEVICE_TYPE_CPU, self.lib.GGML_BACKEND_DEVICE_TYPE_GPU):
                devs.append((self.ffi.string(self.lib.ggml_backend_dev_name(dev)).decode(), dev))
        # the CPU first: it is the reference the other devices are compared with
        return sorted(devs, key=lambda d: self.lib.ggml_backend_dev_type(d[1]) != self.lib.GGML_BACKEND_DEVICE_TYPE_CPU)

    def errors(self) -> str:
        """The libllama error lines logged since the last load."""
        lines = "".join(self.log).splitlines()
        return " | ".join(ln.strip() for ln in lines if "error" in ln.lower()) or (lines[-1] if lines else "")

    def _read(self, t, np):
        """A graph tensor's data. A view is read from its root tensor with its own strides."""
        dtype = {self.lib.GGML_TYPE_F32: np.float32, self.lib.GGML_TYPE_I32: np.int32}.get(t.type)
        if dtype is None:
            raise RuntimeError(f"{self.ffi.string(t.name).decode()}: ggml type {t.type}, not F32 or I32")
        root, offs = (t.view_src, t.view_offs) if t.view_src != self.ffi.NULL else (t, 0)
        buf = bytearray(self.lib.ggml_nbytes(root))
        self.lib.ggml_backend_tensor_get(root, self.ffi.from_buffer(buf), 0, len(buf))
        return np.ndarray(shape=[t.ne[i] for i in reversed(range(4))], dtype=dtype, buffer=buf, offset=offs,
                          strides=[t.nb[i] for i in reversed(range(4))]).copy()

    def logits(self, path: Path, dev, tokens: list[int], n_ubatch: int, overrides: dict | None = None,
               nodes: dict[str, list[str]] | None = None, capture: dict | None = None,
               chunks: list[int] | None = None, n_ctx: int = 256, outputs: list[int] | None = None,
               rows: dict[str, tuple[int, list[int]]] | None = None, cpu_moe: bool = False, **ctx_params):
        """Logits [len(tokens), n_vocab] with the model and its computation on dev only.
        overrides maps GGUF keys to int, float or bool values that replace the file's.
        If nodes is given, it receives every graph node's name with the names of its inputs.
        If capture is given, it receives the data of every node whose name it already holds as a
        key: a list with one numpy array per graph computation (one per ubatch), in ggml order
        reversed ([1, 1, n_tokens, n_embd] for an activation); see joined. If a computation has
        several nodes of that name (Vcur is the matmul and its reshape), the last one counts.
        With capture, every chunk must fit into one ubatch, so each llama_decode is one computation.
        rows maps a captured name to (token axis as in joined, positions): of that node only the
        given positions are kept, so a long context does not keep every activation.
        chunks are the sizes of consecutive llama_decode calls (default: one call), so the later
        calls read the KV cache the earlier ones wrote. outputs are the positions that get logits
        (default: all), and the result has one row per output position, in order. cpu_moe keeps the
        routed experts (ffn_*_exps) in CPU memory and computes them there, as llama-cli --cpu-moe,
        while the rest stays on dev. ctx_params set further llama_context_params fields (swa_full,
        flash_attn_type, type_k, ...)."""
        import numpy as np

        self.log.clear()
        devs = self.ffi.new("ggml_backend_dev_t[]", [dev, self.ffi.NULL])
        mparams = self.ll.model_default_params(devices=devs, n_gpu_layers=-1)
        if overrides:
            kv = self.ffi.new("struct llama_model_kv_override[]", len(overrides) + 1)  # zeroed: the last ends the list
            for o, (key, val) in zip(kv, overrides.items()):
                o.key = key.encode()
                if isinstance(val, bool):
                    o.tag, o.val_bool = self.lib.LLAMA_KV_OVERRIDE_TYPE_BOOL, val
                elif isinstance(val, int):
                    o.tag, o.val_i64 = self.lib.LLAMA_KV_OVERRIDE_TYPE_INT, val
                else:
                    o.tag, o.val_f64 = self.lib.LLAMA_KV_OVERRIDE_TYPE_FLOAT, val
            mparams.kv_overrides = kv
        if cpu_moe:
            cpu = next(d for _, d in self.devices()
                       if self.lib.ggml_backend_dev_type(d) == self.lib.GGML_BACKEND_DEVICE_TYPE_CPU)
            pattern = self.ffi.new("char[]", rb"\.ffn_(up|down|gate)_exps")
            buft = self.ffi.new("struct llama_model_tensor_buft_override[]", 2)  # zeroed: the last ends the list
            buft[0].pattern, buft[0].buft = pattern, self.lib.ggml_backend_dev_buffer_type(cpu)
            mparams.tensor_buft_overrides = buft
        model = self.lib.llama_model_load_from_file(str(path).encode(), mparams)
        if not model:
            raise RuntimeError("load failed: " + self.errors())
        chunks = chunks or [len(tokens)]
        assert sum(chunks) == len(tokens) and (capture is None or max(chunks) <= n_ubatch)
        cparams = self.ll.context_default_params(n_ctx=n_ctx, n_batch=max(chunks), n_ubatch=n_ubatch,
                                                 op_offload=False, **ctx_params)
        want = None if outputs is None else set(outputs)
        span = [0, 0]  # the positions of the current llama_decode call
        keep = {name: np.asarray(sorted(pos)) for name, (_, pos) in (rows or {}).items()}
        if nodes is not None or capture is not None:
            computation, seen = [0], {}  # the llama_decode call, and per name the call of its last capture

            @self.ffi.callback("bool(struct ggml_tensor *, bool, void *)")
            def on_node(t, ask, user_data):
                name = self.ffi.string(t.name).decode()
                if ask:
                    if nodes is not None:
                        srcs = [t.src[i] for i in range(len(t.src))]
                        nodes[name] = [self.ffi.string(x.name).decode() for x in srcs if x != self.ffi.NULL]
                    return capture is not None and name in capture
                parts = capture[name] or []
                if seen.get(name) == computation[0]:
                    parts.pop()  # a later node of the same name in the same computation
                part = self._read(t, np)
                ax = rows[name][0] if name in keep else None
                if ax is not None and part.shape[ax] == span[1] - span[0]:  # not an earlier node of that name
                    pos = keep[name]
                    part = np.take(part, pos[(pos >= span[0]) & (pos < span[1])] - span[0], axis=ax)
                capture[name], seen[name] = parts + [part], computation[0]
                return True

            cparams.cb_eval = on_node
        ctx = self.lib.llama_init_from_model(model, cparams)
        if not ctx:
            self.lib.llama_model_free(model)
            raise RuntimeError("context failed: " + self.errors())
        batch = self.lib.llama_batch_init(max(chunks), 0, 1)
        try:
            n_vocab = self.lib.llama_vocab_n_tokens(self.lib.llama_model_get_vocab(model))
            out, start = [], 0
            for n in chunks:
                for i in range(n):
                    batch.token[i] = tokens[start + i]
                    batch.pos[i] = start + i
                    batch.n_seq_id[i] = 1
                    batch.seq_id[i][0] = 0
                    batch.logits[i] = want is None or start + i in want
                batch.n_tokens = n
                span[:] = start, start + n
                if capture is not None:
                    computation[0] += 1
                if (rc := self.lib.llama_decode(ctx, batch)) != 0:
                    raise RuntimeError(f"llama_decode returned {rc} at position {start}: " + self.errors())
                out += [np.frombuffer(self.ffi.buffer(self.lib.llama_get_logits_ith(ctx, i), 4 * n_vocab),
                                      dtype=np.float32).copy() for i in range(n) if batch.logits[i]]
                start += n
            return np.stack(out)
        finally:
            self.lib.llama_batch_free(batch)
            self.lib.llama_free(ctx)
            self.lib.llama_model_free(model)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True, help="llama.cpp checkout with a built libllama")
    ap.add_argument("--seed", type=int, default=1, help="cmd/kolibri-tiny random seed")
    args = ap.parse_args()
    import numpy as np

    runner = Runner(args.llama_cpp)
    tokens = random.Random(args.seed).choices(range(127900), k=N_TOKENS)  # base vocab, no special tokens
    errs = []

    def report(ok: bool, what: str) -> None:
        print(f"{'PASS' if ok else 'FAIL'} {what}")
        if not ok:
            errs.append(what)

    with tempfile.TemporaryDirectory() as tmp:
        model = generate_tiny(Path(tmp), args.seed)
        n_expert = json.loads((model / "config.json").read_text())["num_experts"]
        devs = runner.devices()
        cpu_name, cpu = devs[0]
        ggufs = {t: convert(model, args.llama_cpp, t) for t in ("bf16", "f32")}

        for outtype, path in ggufs.items():
            for name, dev in devs:
                try:
                    got = runner.logits(path, dev, tokens, n_ubatch=N_TOKENS)
                except RuntimeError as e:
                    report(False, f"{outtype} load and decode on {name}: {e}")
                    continue
                report(got.shape == (N_TOKENS, 128000) and bool(np.isfinite(got).all()),
                       f"{outtype} load and decode on {name}: logits {got.shape}, all finite")
        if errs:
            print(f"FAIL {len(errs)} problem(s)")
            sys.exit(1)

        for label, n_used in ((f"no routing (top-{n_expert} of {n_expert})", n_expert), ("top-2 routing", None)):
            used = {f"{ARCH}.expert_used_count": n_used} if n_used else None
            ref = runner.logits(ggufs["f32"], cpu, tokens, N_TOKENS, used)
            others = [(f"{cpu_name} ubatch 16 vs 100", runner.logits(ggufs["f32"], cpu, tokens, 16, used))]
            others += [(f"{name} vs {cpu_name}", runner.logits(ggufs["f32"], dev, tokens, N_TOKENS, used))
                       for name, dev in devs[1:]]
            for what, got in others:
                per_pos = ((ref - got) ** 2).sum(1) / (ref ** 2).sum(1)
                flips = int((per_pos > MAX_NMSE).sum())
                same = int((ref.argmax(1) == got.argmax(1)).sum())
                if n_used is not None:
                    v = nmse(ref, got)
                    report(v <= MAX_NMSE, f"f32 {label}, {what}: NMSE {v:.2e}")
                else:
                    report(flips <= MAX_FLIPS and same == N_TOKENS,
                           f"f32 {label}, {what}: {N_TOKENS - flips}/{N_TOKENS} positions with NMSE <= {MAX_NMSE:g}, "
                           f"greedy token equal at {same}/{N_TOKENS}")

        nodes: dict[str, list[str]] = {}
        runner.logits(ggufs["f32"], cpu, tokens, N_TOKENS, nodes=nodes)
        layer_types = json.loads((model / "config.json").read_text())["layer_types"]

        def origin(name: str) -> str:
            # unnamed intermediates (the last layer's get_rows of the output rows) lead back to their source
            while name.startswith("node_") and nodes.get(name):
                name = nodes[name][0]
            return name

        bad = []
        for il, layer_type in enumerate(layer_types):
            blk = f"blk.{il}"
            want = {
                # selection on the raw logits plus bias; weights from the unbiased probs
                f"ffn_moe_probs_biased-{il}": [f"ffn_moe_logits-{il}", f"{blk}.exp_probs_b.bias"],
                f"ffn_moe_weights-{il}": [f"ffn_moe_probs-{il} (reshaped)", f"ffn_moe_topk-{il}"],
                f"ffn_moe_down-{il}": [f"{blk}.ffn_down_exps.weight", f"ffn_moe_swiglu-{il}", f"ffn_moe_topk-{il}"],
                # routed + shared, then the post-FFN norm into the residual add
                f"ffn_shexp-{il}": [f"{blk}.ffn_down_shexp.weight", f"ffn_swiglu-{il}"],
                f"ffn_out-{il}": [f"ffn_moe_out-{il}", f"ffn_shexp-{il}"],
                f"ffn_post_norm-{il}": [f"norm-{il}", f"{blk}.post_ffw_norm.weight"],
                f"l_out-{il}": [f"ffn_post_norm-{il}", f"ffn_inp-{il}"],
                # the post-attention norm into the first residual add
                f"attn_post_norm-{il}": [f"norm-{il}", f"{blk}.post_attention_norm.weight"],
            }
            for node, srcs in want.items():
                if nodes.get(node) != srcs:
                    bad.append(f"{node} <- {nodes.get(node)}, want {srcs}")
            ffn_inp = [origin(n) for n in nodes.get(f"ffn_inp-{il}", [])]
            if not ffn_inp or ffn_inp[0] != f"attn_post_norm-{il}":
                bad.append(f"ffn_inp-{il} <- {ffn_inp}, want attn_post_norm-{il} first")
            has_rope = f"Qcur_rope-{il}" in nodes and f"Kcur_rope-{il}" in nodes
            if has_rope != (layer_type == "sliding_attention"):
                bad.append(f"layer {il} ({layer_type}): RoPE nodes {'present' if has_rope else 'missing'}")
        for b in bad:
            print(f"FAIL graph wiring: {b}")
        n_rope = sum(lt == "sliding_attention" for lt in layer_types)
        report(not bad, f"graph wiring: {len(layer_types)} layers x 9 node checks; RoPE on the {n_rope} sliding layers only "
                        f"({len(nodes)} nodes)")

    if errs:
        print(f"FAIL {len(errs)} problem(s)")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
