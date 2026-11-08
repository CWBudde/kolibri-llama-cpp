#!/usr/bin/env python3
"""Check tools/e6/capture_router.py on the tiny 384-expert top-6 checkpoint (kolibri-tiny -router),
without any real weight. Token IDs are random (two "workloads" of 1100 and 700 tokens: several
512-token decodes and a partial one), written to a temporary directory with the outputs.

- the command line, on each device, with --self-check: block l's own router on rms_norm(ffn_inp-l)
  selects ffn_moe_topk-l (h = 0), up to near-ties within the graph's measured rounding;
- shapes and fill: pred-h1 and pred-h2 int16 [n_layer, n_tokens, 48], distinct IDs in [0, 384)
  where block l+h exists, -1 in the rows without one (l = n_layer-1 for h = 1, the last two for h = 2);
- the selections equal, bit for bit, what tools/locality/experts.py's capture() returns for the same
  tokens on the same device, with and without --self-check (its extra reads split the first graph
  computation);
- the predictions equal a float64 reference built from l_out-l captured whole by Runner.logits, a
  different code path: per token the top-48 in the same order, except where two adjacent scores tie
  within 1e-4 of the scale. The reference reads ffn_moe_topk too: on Metal, splitting the graph at
  those nodes changes the later blocks' activations (an INFO line prints by how much), so the tool,
  experts.py and this reference all read the same set;
- refuse_out refuses ~/models/eval/locality and a non-empty directory (called directly; nothing runs).

Controls: --control no-bias (block l's selection without exp_probs_b) and --control wrong-norm (block
l+1's ffn_norm.weight in block l's router) must make the self-check FAIL and the tool exit 1.

    check_capture_router.py --llama-cpp third_party/llama.cpp [--devices CPU,MTL0]
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "e6"))
sys.path.insert(0, str(ROOT / "tools" / "gguf"))
import capture_router as cr  # noqa: E402
from tiny import convert, generate_tiny  # noqa: E402

TOOL = ROOT / "tools" / "e6" / "capture_router.py"
LENGTHS = {"coding": 1100, "hr": 700}
THREADS = 4


def main() -> int:
    import numpy as np

    import experts as ex
    from check_model import Runner

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--devices", help="comma-separated, e.g. CPU (default: every device of the build)")
    args = ap.parse_args()
    failures: list[str] = []

    def report(ok: bool, what: str) -> None:
        print(f"{'PASS' if ok else 'FAIL'} {what}", flush=True)
        if not ok:
            failures.append(what)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        gguf = convert(generate_tiny(tmpdir, args.seed, "router"), args.llama_cpp, "f32")
        routers = cr.Routers(gguf, args.llama_cpp)
        L, k = routers.n_layer, min(cr.TOP_K, routers.n_expert)
        tokens_dir = tmpdir / "tokens"
        tokens_dir.mkdir()
        rng = np.random.default_rng(args.seed)
        tokens = {}
        for name, n in LENGTHS.items():
            tokens[name] = rng.integers(1000, 120000, n).astype(np.int64)
            np.save(tokens_dir / f"{name}.tokens.npy", tokens[name])

        def tool(device: str, out: Path, *extra: str) -> subprocess.CompletedProcess:
            cmd = [sys.executable, str(TOOL), "--llama-cpp", str(args.llama_cpp), "--gguf", str(gguf), "--device", device,
                   "--threads", str(THREADS), "--workloads", ",".join(LENGTHS), "--tokens-dir", str(tokens_dir),
                   "--allow-other-tokens", "--out", str(out), *extra]
            return subprocess.run(cmd, capture_output=True, text=True)

        def lines(r) -> list[str]:
            return [ln for ln in r.stdout.splitlines() if ln.startswith(("PASS", "FAIL", "UNCHECKED", "MATCH", "MISMATCH", "DONE"))]

        runner = Runner(args.llama_cpp)
        devs = runner.devices()
        names = args.devices.split(",") if args.devices else [n for n, _ in devs]
        for dev_name in names:
            dev = dict(devs)[dev_name]
            device = "cpu" if runner.lib.ggml_backend_dev_type(dev) == runner.lib.GGML_BACKEND_DEVICE_TYPE_CPU else "metal"
            what = f"{dev_name}"
            outs = {}
            for mode, extra in (("self-check", ("--self-check",)), ("plain", ())):
                out = tmpdir / f"out-{dev_name}-{mode}"
                r = tool(device, out, *extra)
                for ln in lines(r):
                    print("  " + ln)
                checks = [ln for ln in lines(r) if ln.startswith(("PASS", "FAIL")) and "self-check" in ln]
                report(r.returncode == 0 and (mode == "plain" or (len(checks) == len(LENGTHS) and all(c.startswith("PASS") for c in checks))),
                       f"{what} {mode}: the tool exits 0" + (", self-check PASS on every workload" if mode != "plain" else "")
                       + ("" if r.returncode == 0 else f"; stderr: {r.stderr.strip()[-800:]}"))
                outs[mode] = out

            ctx_args = argparse.Namespace(threads=THREADS, no_repack=False)
            for name, ids in tokens.items():
                stem = f"{name}.{gguf.stem}"
                pred = {h: np.load(outs["plain"] / f"{stem}.pred-h{h}.npy") for h in cr.HORIZONS}
                want_shape = (L, len(ids), k)
                ok = all(p.shape == want_shape and p.dtype == np.int16 for p in pred.values())
                for h, p in pred.items():
                    ok = ok and (p[L - h:] == -1).all()
                    body = p[:L - h]
                    ok = ok and body.min() >= 0 and body.max() < routers.n_expert
                    ok = ok and not (np.diff(np.sort(body, axis=-1), axis=-1) == 0).any()
                report(ok, f"{what} {name}: pred-h1/h2 int16 {list(want_shape)}, distinct IDs where block l+h exists, "
                           f"-1 in the last 1 and 2 layers")

                # experts.py's own capture for the same tokens, as the reference layout and values
                ref = ex.capture(runner, gguf, dev, ids.tolist(), L, ctx_args)
                for mode, out in outs.items():
                    sel = np.load(out / f"{stem}.experts.npy")
                    report(sel.shape == ref.shape and sel.dtype == ref.dtype and np.array_equal(sel, ref)
                           and ex.selections_sha256(sel) == ex.selections_sha256(ref),
                           f"{what} {name} {mode}: experts.npy {list(sel.shape)} equals experts.py capture() bit for bit")

                # A float64 reference from l_out captured whole through Runner.logits' own dict. It reads
                # ffn_moe_topk too, as the tool and experts.py do: on Metal, splitting the graph there
                # changes the later blocks' activations (see the informational line below).
                cap = {f"l_out-{l}": None for l in range(L - 1)} | {f"ffn_moe_topk-{l}": None for l in range(L)}
                n = ex.N_UBATCH
                chunks = [n] * (len(ids) // n) + ([len(ids) % n] if len(ids) % n else [])
                runner.logits(gguf, dev, ids.tolist(), n, capture=cap, chunks=chunks, n_ctx=ex.N_CTX, return_logits=False,
                              n_threads=THREADS, n_threads_batch=THREADS)
                exact, total, bad = 0, 0, 0
                for l in range(L - 1):
                    hid = np.concatenate(cap[f"l_out-{l}"], axis=-2).reshape(len(ids), -1).astype(np.float64)
                    x = hid / np.sqrt((hid ** 2).mean(axis=1, keepdims=True) + routers.eps)
                    for h in cr.HORIZONS:
                        t = l + h
                        if t >= L:
                            continue
                        s = (x * routers.norm[t]) @ routers.gate[t].T.astype(np.float64) + routers.bias[t]
                        order = np.argsort(-s, axis=1, kind="stable")[:, :k]
                        got = pred[h][l].astype(np.int64)
                        same = (order == got).all(axis=1)
                        exact += int(same.sum())
                        total += len(same)
                        for i in np.flatnonzero(~same):  # allowed only where adjacent scores tie
                            j = int(np.argmax(order[i] != got[i]))
                            ss = np.sort(s[i])[::-1]
                            gap = ss[j] - ss[j + 1] if j + 1 < len(ss) else 0
                            bad += gap > 1e-4 * max(1.0, abs(ss[j]))
                report(bad == 0 and exact > 0.99 * total,
                       f"{what} {name}: predictions equal the float64 reference from whole l_out captures in {exact}/{total} "
                       f"(layer, horizon, token) top-{k} lists, {total - exact - bad} near-ties, {bad} otherwise")

            # informational: l_out of the last source layer with and without the ffn_moe_topk reads
            ids = tokens["hr"].tolist()
            n = ex.N_UBATCH
            chunks = [n] * (len(ids) // n) + ([len(ids) % n] if len(ids) % n else [])
            last = f"l_out-{L - 2}"
            got = {}
            for label, extra in (("without", {}), ("with", {f"ffn_moe_topk-{l}": None for l in range(L)})):
                cap = {last: None} | extra
                runner.logits(gguf, dev, ids, n, capture=cap, chunks=chunks, n_ctx=ex.N_CTX, return_logits=False,
                              n_threads=THREADS, n_threads_batch=THREADS)
                got[label] = np.concatenate(cap[last], axis=-2)
            d = float(np.abs(got["with"] - got["without"]).max() / np.abs(got["without"]).max())
            print(f"INFO {what}: {last} with the ffn_moe_topk reads vs without: max relative difference {d:.2e}", flush=True)

            for control in ("no-bias", "wrong-norm"):
                r = tool(device, tmpdir / f"out-{dev_name}-{control}", "--self-check", "--control", control)
                checks = [ln for ln in lines(r) if "self-check" in ln]
                for ln in checks:
                    print("  " + ln)
                report(r.returncode == 1 and checks and all(c.startswith("FAIL") for c in checks),
                       f"{what} control {control}: the self-check FAILs and the tool exits {r.returncode} (want 1)")

        locality = ex.OUT / "e6-must-be-refused"
        for path, why in ((locality, "under the locality traces"), (tmpdir / "tokens", "not empty")):
            try:
                cr.refuse_out(path)
                refused = False
            except SystemExit:
                refused = True
            report(refused and not locality.exists(), f"refuse_out refuses {path} ({why})")

    print(f"{'FAIL' if failures else 'PASS'} check_capture_router: {len(failures)} failures", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
