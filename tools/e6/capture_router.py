#!/usr/bin/env python3
"""Stage E6: the experts that layer l's hidden state selects in the routers of layers l+1 and l+2.

Prefills each Stage A workload's token IDs (~/models/eval/locality/<wl>.tokens.npy) exactly as
tools/locality/experts.py does: Runner.logits of tools/gguf/check_model.py (libllama through cffi),
llama_decode calls of 512 tokens, n_ubatch 512, n_ctx 32768, every position an output, the same
device, repack and thread settings. Through cb_eval it reads, per layer l and ubatch:

- l_out-l, the residual stream after block l. In src/models/kolibri1.cpp it is the add of ffn_inp-l
  and post_ffw_norm(routed + shared experts), and it is inpL of block l+1: the input of that block's
  attn_norm and the residual of its ffn_inp. It is not ffn_inp-(l+1), which adds block l+1's
  attention output, and so not the router input of block l+1;
- ffn_moe_topk-l, the six experts block l selects, written in experts.py's layout, int16
  [n_layer, n_tokens, n_expert_used], and compared with testdata/locality/experts.json.

For h in 1, 2 with l+h < n_layer the predictor applies block l+h's router to l_out-l, the way the
graph applies block l+h's router to ffn_inp-(l+h):

    x      = rms_norm(l_out-l, eps) * blk.(l+h).ffn_norm.weight
    scores = blk.(l+h).ffn_gate_inp.weight . x + blk.(l+h).exp_probs_b.bias
    pred   = the top-k experts by score, best first (k = 48, at most n_expert)

eps is <arch>.attention.layer_norm_rms_epsilon; rms_norm is x / sqrt(mean(x^2) + eps) (the mean in
float64, the rest float32). The scores are the selection's (biased logits, before any sigmoid).
Every Kolibri block is MoE; a block without ffn_gate_inp would be dense, and its rows are -1, as are
the rows of layers without a target block (l+h >= n_layer). Hidden states are never stored.

Outputs in --out (a new or empty directory), per workload and <model> = the GGUF's stem:

- <wl>.<model>.pred-h1.npy, <wl>.<model>.pred-h2.npy: int16 [n_layer, n_tokens, k], row l is layer
  l's hidden state through block l+h's router;
- <wl>.<model>.experts.npy: the selections captured in the same pass;
- manifest.json: settings, GGUF identity (size, mtime, sha256 of its first 64 MiB), libllama,
  per-file sha256, timings, and the comparison with the recorded selections.

A workload whose selections differ from the recorded ones (MISMATCH) gets its files with a
.rejected.npy suffix instead, and the tool exits 1. With --max-tokens the selections are compared
with the prefix of the stored ~/models/eval/locality/<wl>.<model>.experts.npy (read, never written).

--self-check also reads ffn_inp-l, ffn_norm-l (the graph's router input) and ffn_moe_logits-l in the
first ubatch, and requires that this predictor with h = 0 (rms_norm(ffn_inp-l) * ffn_norm[l] through
block l's router) selects ffn_moe_topk-l, up to near-ties. Those reads split the first graph
computation at more nodes than experts.py did; the selection comparison still covers the result.

    capture_router.py --llama-cpp third_party/llama.cpp \\
        --gguf ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf --device metal \\
        [--out ~/models/eval/e6/capture] [--workloads coding,hr] [--max-tokens 2048] [--self-check]
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
for sub in ("gguf", "locality", "probe"):
    sys.path.insert(0, str(ROOT / "tools" / sub))

import experts as ex  # noqa: E402  (N_UBATCH, N_CTX, selections_sha256, invalid, EXPERTS, OUT)
from workloads import MANIFEST, WORKLOADS, ids_sha256  # noqa: E402

VERSION = "e6-capture-router-1"
TOP_K = 48
HORIZONS = (1, 2)
DEFAULT_OUT = Path.home() / "models" / "eval" / "e6" / "capture"
HEAD_BYTES = 64 << 20
# the least near-tie band of the self-check, relative to the score (see Pass._self_check_layer)
TIE_RTOL = 1e-4


def sha256_file(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    left = limit
    with open(path, "rb") as f:
        if platform.system() == "Darwin":
            fcntl.fcntl(f.fileno(), fcntl.F_NOCACHE, 1)
        while left is None or left > 0:
            chunk = f.read(1 << 24 if left is None else min(1 << 24, left))
            if not chunk:
                break
            h.update(chunk)
            if left is not None:
                left -= len(chunk)
    return h.hexdigest()


def rms_norm(x, eps: float):
    """x / sqrt(mean(x^2) + eps) per row, float32; the mean summed in float64 as ggml's CPU does."""
    import numpy as np

    ms = np.mean(np.square(x, dtype=np.float64), axis=1)
    return x * (1.0 / np.sqrt(ms + eps)).astype(np.float32)[:, None]


def top_k(scores, k: int):
    """The k best columns per row, best first; ties go to the lower expert ID. int16 [n, k]."""
    import numpy as np

    # A stable sort of every column: a partition to k first would cut a tie at the k-th place anywhere.
    return np.argsort(-scores, axis=1, kind="stable")[:, :k].astype(np.int16)


class Routers:
    """Every block's router, from the GGUF: ffn_norm.weight, ffn_gate_inp.weight, exp_probs_b.bias,
    as float32. A block without ffn_gate_inp is dense and has none (None)."""

    def __init__(self, gguf: Path, llama_cpp: Path):
        import numpy as np

        sys.path.insert(0, str(llama_cpp / "gguf-py"))
        from gguf import GGUFReader

        r = GGUFReader(gguf)
        f = r.fields
        self.arch = f["general.architecture"].contents()
        key = lambda k: f[f"{self.arch}.{k}"].contents()
        self.n_layer, self.n_expert, self.n_used = int(key("block_count")), int(key("expert_count")), int(key("expert_used_count"))
        self.n_embd = int(key("embedding_length"))
        self.eps = float(key("attention.layer_norm_rms_epsilon"))
        tensors = {t.name: t for t in r.tensors}

        def f32(name: str, shape: tuple):
            t = tensors[name]
            if t.tensor_type.name != "F32":
                raise SystemExit(f"{name}: {t.tensor_type.name}, want F32 (llama-quantize keeps routers and norms in F32)")
            a = np.array(t.data, dtype=np.float32).reshape(shape)
            return np.ascontiguousarray(a)

        self.norm, self.gate, self.bias, self.types = [], [], [], {}
        for il in range(self.n_layer):
            if f"blk.{il}.ffn_gate_inp.weight" not in tensors:
                self.norm.append(None), self.gate.append(None), self.bias.append(None)
                continue
            self.norm.append(f32(f"blk.{il}.ffn_norm.weight", (self.n_embd,)))
            self.gate.append(f32(f"blk.{il}.ffn_gate_inp.weight", (self.n_expert, self.n_embd)))  # ne [n_embd, n_expert]
            self.bias.append(f32(f"blk.{il}.exp_probs_b.bias", (self.n_expert,)))
        self.dense = [il for il in range(self.n_layer) if self.gate[il] is None]

    def scores(self, normed, target: int, bias: bool = True, norm_layer: int | None = None):
        """Block target's selection scores [n, n_expert] for rms-normed rows. bias and norm_layer exist
        for the negative controls only."""
        w = self.norm[target if norm_layer is None else norm_layer]
        s = (normed * w) @ self.gate[target].T
        return s + self.bias[target] if bias else s


class Stream(dict):
    """The capture mapping Runner.logits fills, consumed as it fills. Runner.logits stores each graph
    computation's part as capture[name] = (capture[name] or []) + [part]; here capture[name] is always
    empty, so the part is handed on and nothing accumulates. A name read twice in one computation makes
    Runner.logits pop from that empty list and the decode fails. extra names are captured only until
    the first part of `until` arrives, which ends the first graph computation."""

    def __init__(self, names, on_part, extra=(), until: str | None = None):
        super().__init__(dict.fromkeys(names))
        self.on_part, self.extra, self.until = on_part, set(extra), until
        self.error: BaseException | None = None

    def __contains__(self, name):
        return dict.__contains__(self, name) or name in self.extra

    def __getitem__(self, name):
        return None

    def __setitem__(self, name, parts):
        try:
            if len(parts) != 1:
                raise RuntimeError(f"{name}: {len(parts)} parts in one computation")
            self.on_part(name, parts[0])
            if name == self.until:
                self.extra = set()
        except BaseException as e:  # cffi swallows a callback's exception: keep it for after the decode
            self.error = self.error or e
            raise


class Pass:
    """One workload's prefill: predictions and selections, filled per ubatch from the callback."""

    def __init__(self, routers: Routers, n_tokens: int, k: int, total_ubatches: int, done_ubatches: int,
                 self_check: bool, control: str | None = None):
        import numpy as np

        self.r, self.k, self.n_tokens = routers, k, n_tokens
        L = routers.n_layer
        self.pred = {h: np.full((L, n_tokens, k), -1, np.int16) for h in HORIZONS}
        self.sel = np.full((L, n_tokens, routers.n_used), -1, np.int16)
        self.seen: dict[str, int] = {}
        self.total_ubatches, self.done_ubatches = total_ubatches, done_ubatches
        self.self_check, self.control = self_check, control
        self.check_parts: dict[str, object] = {}
        self.check: dict[int, dict] = {}
        self.t0 = None
        self.t_score = 0.0
        self.ubatches = 0
        self.projected = None

    def names(self):
        L = self.r.n_layer
        sources = [f"l_out-{l}" for l in range(L) if any(l + h < L and self.r.gate[l + h] is not None for h in HORIZONS)]
        return sources + [f"ffn_moe_topk-{il}" for il in range(L)]

    def extra(self):
        if not self.self_check:
            return ()
        return [f"{n}-{il}" for il in range(self.r.n_layer) if self.r.gate[il] is not None
                for n in ("ffn_inp", "ffn_norm", "ffn_moe_logits")]

    def on_part(self, name: str, part) -> None:
        import numpy as np

        kind, il = name.rsplit("-", 1)
        il = int(il)
        rows = part.reshape(-1, part.shape[-1])  # [1, 1, n, width] in numpy order
        if kind in ("ffn_inp", "ffn_norm", "ffn_moe_logits"):
            self.check_parts[name] = rows.astype(np.float32, copy=True)
            return
        off = self.seen.get(name, 0)
        n = rows.shape[0]
        if off + n > self.n_tokens:
            raise RuntimeError(f"{name}: rows {off}..{off + n} beyond {self.n_tokens} tokens")
        self.seen[name] = off + n
        if kind == "l_out":
            if rows.dtype != np.float32 or rows.shape[1] != self.r.n_embd:
                raise RuntimeError(f"{name}: {rows.dtype} {rows.shape}, want float32 [n, {self.r.n_embd}]")
            t = time.perf_counter()
            normed = rms_norm(rows, self.r.eps)
            for h in HORIZONS:
                target = il + h
                if target < self.r.n_layer and self.r.gate[target] is not None:
                    self.pred[h][il, off:off + n] = top_k(self.r.scores(normed, target), self.k)
            self.t_score += time.perf_counter() - t
        elif kind == "ffn_moe_topk":
            if rows.shape[1] != self.r.n_used:
                raise RuntimeError(f"{name}: {rows.shape[1]} experts per token, want {self.r.n_used}")
            self.sel[il, off:off + n] = rows
            if self.self_check and il not in self.check and f"ffn_inp-{il}" in self.check_parts:
                self._self_check_layer(il, rows)
            if il == self.r.n_layer - 1:
                self._ubatch_done()
        else:
            raise RuntimeError(f"unexpected node {name}")

    def _self_check_layer(self, il: int, topk) -> None:
        """h = 0: block il's own router on block il's ffn_inp, against the graph's selection."""
        import numpy as np

        inp = self.check_parts.pop(f"ffn_inp-{il}")
        graph_x = self.check_parts.pop(f"ffn_norm-{il}")
        graph_logits = self.check_parts.pop(f"ffn_moe_logits-{il}")
        n_used = self.r.n_used
        bias = self.control != "no-bias"
        norm_layer = (il + 1) % self.r.n_layer if self.control == "wrong-norm" else None
        mine = self.r.scores(rms_norm(inp, self.r.eps), il, bias=bias, norm_layer=norm_layer)
        via_graph_x = graph_x @ self.r.gate[il].T + self.r.bias[il]  # the graph's router input, already weighted
        graph_scores = graph_logits + self.r.bias[il]
        want = np.sort(topk.astype(np.int64), axis=1)
        # The graph's rounding per token: how far its logits lie from the float32 ones of the correct
        # x (Metal's mul_mm rounds F32 operands through half). A differing selection counts as a
        # near-tie when the graph's margin between its last selected and first unselected expert is
        # within twice that. Measured with the correct norm and no bias, so a control cannot widen it.
        x_mine = rms_norm(inp, self.r.eps) * self.r.norm[il]
        dev = np.abs(x_mine @ self.r.gate[il].T - graph_logits).max(axis=1)
        gs = np.sort(graph_scores, axis=1)[:, ::-1]
        margin = gs[:, n_used - 1] - gs[:, n_used]
        ties = margin <= np.maximum(2 * dev, TIE_RTOL * np.maximum(np.abs(gs[:, n_used - 1]), 1.0))
        out = {}
        for label, s in (("predictor", mine), ("graph_router_input", via_graph_x)):
            got = top_k(s, n_used).astype(np.int64)
            same = (np.sort(got, axis=1) == want).all(axis=1)
            ordered = (got == topk).all(axis=1)
            out[label] = {"tokens": int(len(same)), "same_set": int(same.sum()), "same_order": int(ordered.sum()),
                          "differ_near_tie": int((~same & ties).sum()), "differ_not_tie": int((~same & ~ties).sum()),
                          "max_margin_of_differing": float(margin[~same].max()) if (~same).any() else None}
        out["max_abs_x_vs_ffn_norm"] = float(np.abs(x_mine - graph_x).max())
        out["max_abs_logits_vs_ffn_moe_logits"] = float(dev.max())
        out["max_abs_logits"] = float(np.abs(graph_logits).max())
        self.check[il] = out

    def _ubatch_done(self) -> None:
        self.ubatches += 1
        if self.ubatches == 1 and self.done_ubatches == 0:
            dt = time.perf_counter() - self.t0
            self.projected = dt * self.total_ubatches
            print(f"first ubatch: {dt:.2f} s (scoring {self.t_score:.2f} s, includes warm-up); projected "
                  f"{self.projected / 60:.1f} min for {self.total_ubatches} ubatches", flush=True)


def check_self(check: dict, layers: list[int]) -> tuple[bool, str]:
    if set(check) != set(layers):
        return False, f"self-check covered layers {sorted(check)}, want {layers}"
    tot = sum(c["predictor"]["tokens"] for c in check.values())
    same = sum(c["predictor"]["same_set"] for c in check.values())
    gx = sum(c["graph_router_input"]["same_set"] for c in check.values())
    bad = sum(c["predictor"]["differ_not_tie"] + c["graph_router_input"]["differ_not_tie"] for c in check.values())
    tie = sum(c["predictor"]["differ_near_tie"] for c in check.values())
    dx = max(c["max_abs_x_vs_ffn_norm"] for c in check.values())
    dl = max(c["max_abs_logits_vs_ffn_moe_logits"] for c in check.values())
    text = (f"self-check (h = 0, first ubatch, {len(layers)} layers): predictor selects ffn_moe_topk in {same}/{tot} "
            f"(token, layer) pairs ({100 * same / tot:.3f}%), the graph's ffn_norm through the GGUF router in {gx}/{tot}; "
            f"{tie} differ at a near-tie, {bad} beyond one; max |x - ffn_norm| {dx:.2e}, max |logits - ffn_moe_logits| {dl:.2e}")
    return bad == 0, text


def model_identity(gguf: Path) -> dict:
    st = gguf.stat()
    return {"path": str(gguf), "realpath": str(gguf.resolve()), "bytes": st.st_size, "mtime": st.st_mtime,
            "head_bytes": min(HEAD_BYTES, st.st_size), "head_sha256": sha256_file(gguf, HEAD_BYTES)}


def library_identity(llama_cpp: Path) -> dict:
    lib_dir = llama_cpp / "build" / "bin"
    lib = next(p for p in (lib_dir / "libllama.dylib", lib_dir / "libllama.so") if p.exists())
    head = subprocess.run(["git", "-C", str(llama_cpp), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(llama_cpp), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip() != ""
    return {"llama_cpp": str(llama_cpp), "head": head, "dirty": dirty, "libllama": str(lib.resolve()),
            "libllama_sha256": sha256_file(lib.resolve())}


def refuse_out(out: Path) -> None:
    """A new or empty directory, never under the locality traces: results may exist nowhere else."""
    locality = ex.OUT.resolve()
    real = out.resolve()
    if real == locality or locality in real.parents:
        raise SystemExit(f"--out {out}: under {ex.OUT}, which holds the Stage A traces; choose another directory")
    if out.is_symlink() or (out.exists() and not out.is_dir()):
        raise SystemExit(f"--out {out} is not a directory: choose a new or empty one")
    if out.is_dir() and any(out.iterdir()):
        raise SystemExit(f"--out {out} is not empty, and results may exist nowhere else: choose a new or empty directory")


def reference(name: str, gguf: Path, tokens_ok: bool, full: bool, sel, recorded: dict) -> tuple[str, str]:
    """MATCH, MISMATCH or UNCHECKED, with why."""
    import numpy as np

    if not tokens_ok:
        return "UNCHECKED", "token IDs are not the workload's recorded ones"
    if full:
        rec = recorded.get(name)
        if rec is None:
            return "UNCHECKED", f"no record for {gguf.stem} in {ex.EXPERTS.relative_to(ROOT)}"
        got = ex.selections_sha256(sel)
        if got == rec["experts_sha256"] and sel.shape[1] == rec["n_tokens"]:
            return "MATCH", f"sha256 {got[:16]}.. as recorded in {ex.EXPERTS.relative_to(ROOT)}"
        return "MISMATCH", f"sha256 {got[:16]}.., recorded {rec['experts_sha256'][:16]}.."
    stored = ex.selections_path(name, gguf)
    if not stored.exists():
        return "UNCHECKED", f"prefix: no {stored}"
    ref = np.load(stored, mmap_mode="r")[:, :sel.shape[1]]
    if np.array_equal(ref, sel):
        return "MATCH", f"the first {sel.shape[1]} tokens of {stored.name}"
    diff = np.argwhere((ref != sel).any(axis=-1))
    return "MISMATCH", f"prefix differs from {stored.name} in {len(diff)} (layer, token) pairs, first {diff[0].tolist()}"


def main() -> int:
    import numpy as np

    from check_model import Runner

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, default=ROOT / "third_party" / "llama.cpp",
                    help="llama.cpp checkout with a built libllama (default: third_party/llama.cpp)")
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "metal"), default="metal")
    ap.add_argument("--no-repack", action="store_true", help="keep the CPU weights mmapped, as experts.py --no-repack")
    ap.add_argument("--threads", type=int, help="CPU threads (libllama's default otherwise, as the recorded IQ3 run)")
    ap.add_argument("--workloads", type=lambda s: s.split(","), default=list(WORKLOADS))
    ap.add_argument("--tokens-dir", type=Path, default=ex.OUT, help="where <wl>.tokens.npy are read (default: the traces)")
    ap.add_argument("--allow-other-tokens", action="store_true",
                    help="fixtures only: token IDs that differ from testdata/locality/manifest.json are UNCHECKED, not a failure")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="new or empty directory")
    ap.add_argument("--max-tokens", type=int, help="the first N tokens of each workload (smoke test)")
    ap.add_argument("--top-k", type=int, default=TOP_K, help="predicted experts per (layer, horizon, token), at most n_expert")
    ap.add_argument("--self-check", action="store_true", help="h = 0 against ffn_moe_topk on the first ubatch")
    ap.add_argument("--control", choices=("no-bias", "wrong-norm"), help=argparse.SUPPRESS)  # negative controls of the check script
    args = ap.parse_args()
    args.gguf = args.gguf.expanduser()
    args.out = args.out.expanduser()

    refuse_out(args.out)
    started = time.time()
    routers = Routers(args.gguf, args.llama_cpp)
    k = min(args.top_k, routers.n_expert)
    print(f"{args.gguf.name}: {routers.arch}, {routers.n_layer} layers, n_embd {routers.n_embd}, {routers.n_expert} experts, "
          f"top {routers.n_used}, eps {routers.eps:g}; dense blocks {routers.dense or 'none'}; top-{k} per horizon", flush=True)

    tokens_manifest = json.loads(MANIFEST.read_text())["workloads"]
    recorded = json.loads(ex.EXPERTS.read_text()).get(args.gguf.stem, {}).get("workloads", {}) if ex.EXPERTS.exists() else {}
    inputs = {}
    for name in args.workloads:
        ids = np.load(args.tokens_dir.expanduser() / f"{name}.tokens.npy").tolist()
        ok = name in tokens_manifest and ids_sha256(ids) == tokens_manifest[name]["tokens_sha256"]
        if not ok and not args.allow_other_tokens:
            raise SystemExit(f"{name}: the token IDs differ from {MANIFEST.relative_to(ROOT)} (run workloads.py)")
        full = args.max_tokens is None or args.max_tokens >= len(ids)
        inputs[name] = (ids if full else ids[:args.max_tokens], ok, full)
    total_ubatches = sum(-(-len(t) // ex.N_UBATCH) for t, _, _ in inputs.values())

    runner = Runner(args.llama_cpp)
    kind = {"cpu": runner.lib.GGML_BACKEND_DEVICE_TYPE_CPU, "metal": runner.lib.GGML_BACKEND_DEVICE_TYPE_GPU}[args.device]
    dev_name, dev = next((n, d) for n, d in runner.devices() if runner.lib.ggml_backend_dev_type(d) == kind)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "tool": VERSION, "gguf": model_identity(args.gguf), "library": library_identity(args.llama_cpp),
        "settings": {"device": dev_name, "n_ubatch": ex.N_UBATCH, "n_batch": ex.N_UBATCH, "n_ctx": ex.N_CTX,
                     "threads": args.threads, "no_repack": args.no_repack, "outputs": "every position",
                     "top_k": k, "horizons": list(HORIZONS), "max_tokens": args.max_tokens,
                     "self_check": args.self_check, "control": args.control},
        "model": {"arch": routers.arch, "n_layer": routers.n_layer, "n_embd": routers.n_embd, "n_expert": routers.n_expert,
                  "n_expert_used": routers.n_used, "rms_eps": routers.eps, "dense_blocks": routers.dense},
        "predictor": "pred-hH[l] = top-k of blk.(l+H).ffn_gate_inp . (rms_norm(l_out-l) * blk.(l+H).ffn_norm.weight) "
                     "+ blk.(l+H).exp_probs_b.bias, best first, ties to the lower ID; -1 where l+H >= n_layer or "
                     "block l+H is dense",
        "workloads": {},
    }
    failed = False
    done_ubatches = 0
    for name, (tokens, tokens_ok, full) in inputs.items():
        p = Pass(routers, len(tokens), k, total_ubatches, done_ubatches, args.self_check, args.control)
        stream = Stream(p.names(), p.on_part, p.extra(), until=f"ffn_moe_topk-{routers.n_layer - 1}")
        n = ex.N_UBATCH
        chunks = [n] * (len(tokens) // n) + ([len(tokens) % n] if len(tokens) % n else [])
        ctx = {"n_threads": args.threads, "n_threads_batch": args.threads} if args.threads else {}
        t0 = time.monotonic()
        p.t0 = time.perf_counter()
        runner.logits(args.gguf, dev, tokens, n, capture=stream, chunks=chunks, n_ctx=ex.N_CTX, return_logits=False,
                      no_repack=args.no_repack, **ctx)
        seconds = time.monotonic() - t0
        if stream.error is not None:
            raise RuntimeError(f"{name}: the callback failed") from stream.error
        missing = [nm for nm in p.names() if p.seen.get(nm) != len(tokens)]
        if missing:
            raise RuntimeError(f"{name}: nodes without every token's row: {missing[:5]}")
        done_ubatches += len(chunks)
        why_invalid = ex.invalid(p.sel, routers.n_expert)
        status, why = ("MISMATCH", why_invalid) if why_invalid else reference(name, args.gguf, tokens_ok, full, p.sel, recorded)
        print(f"{status} {name}: {len(tokens)} tokens in {seconds:.0f} s ({len(tokens) / seconds:.1f} tokens/s, scoring "
              f"{p.t_score:.0f} s) on {dev_name}; {why}", flush=True)
        entry = {"n_tokens": len(tokens), "tokens_sha256": ids_sha256(tokens), "tokens_recorded": tokens_ok,
                 "full": full, "experts": status, "experts_detail": why, "seconds": seconds, "scoring_seconds": p.t_score,
                 "projected_seconds_after_first_ubatch": p.projected, "files": {}}
        if args.self_check:
            moe = [il for il in range(routers.n_layer) if routers.gate[il] is not None]
            ok, text = check_self(p.check, moe)
            print(f"{'PASS' if ok else 'FAIL'} {name}: {text}", flush=True)
            entry["self_check"] = {"pass": ok, "summary": text, "layers": {str(il): c for il, c in sorted(p.check.items())}}
            failed = failed or not ok
        suffix = ".npy" if status != "MISMATCH" else ".rejected.npy"
        for label, arr in [(f"pred-h{h}", p.pred[h]) for h in HORIZONS] + [("experts", p.sel)]:
            path = args.out / f"{name}.{args.gguf.stem}.{label}{suffix}"
            np.save(path, np.ascontiguousarray(arr, dtype="<i2"))
            entry["files"][path.name] = {"shape": list(arr.shape), "dtype": "int16", "sha256": sha256_file(path)}
        entry["experts_sha256"] = ex.selections_sha256(p.sel)
        manifest["workloads"][name] = entry
        manifest["wall_seconds"] = time.time() - started
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        failed = failed or status == "MISMATCH"
    print(f"{'FAIL' if failed else 'DONE'} {len(inputs)} workloads in {(time.time() - started) / 60:.1f} min; {args.out}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
