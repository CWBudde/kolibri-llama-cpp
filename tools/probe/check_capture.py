#!/usr/bin/env python3
"""Check the activation capture of tools/probe/capture.py on tiny checkpoints, without any real weight.

Layer indices scale with --layers: the default 6-layer tiny model reads 2,3,5,6 where the real model
reads 10,25,40,50, and the 50-layer -pattern model reads the real ones.

- non-mutating: on each device, for all 44 prompts, the logits of a model and context that never saw
  a callback equal those of the capturing context with np.array_equal: exact, and NaN is not equal;
  also when every layer is read, so that the graph is split at every block;
- numbering: the graph reads as documented on the model (capture.check_graph), the l_out of the last
  block feeds output_norm; layer N is node l_out-(N-1);
- the row: the last prompt token's row. Decoding with an output at every position, the last row of
  each node equals what the capture kept, and that of the last block, which the graph cuts to the
  outputs, too;
- repeatability: a second decode of a prompt gives equal activations and logits;
- ubatches and the expert cache: with the cache's capped ubatch a prompt spans several computations;
  the capture keeps the last, and equals a resident run at the same ubatch;
- the command line, end to end, on the dataset, through kolibri-probe report.

Controls: M77, a capture that writes into a node, changes the logits, and the check and kolibri-probe
report catch it. M79, zero-based numbering for the issue's layers, is refused by the graph check, and
by the missing node at the last layer. M87, a graph whose residual add lacks the post-FFN norm, is
refused by the check, and by the report from capture.json. M88, a capture that was not checked, gets
no report, also if it wrote into a node. M75, M76 and M78 are Go tests in internal/probe. The capture
prints its own FAIL line for M77 and an UNCHECKED line for M88, which is the controls working; the
check's result is its last line.

    check_capture.py --llama-cpp third_party/llama.cpp
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "probe"))
sys.path.insert(0, str(ROOT / "tools" / "gguf"))
import capture as cap  # noqa: E402
from check_model import MAX_NMSE  # noqa: E402
from tiny import convert, generate_tiny  # noqa: E402

DATASET = ROOT / "testdata" / "probe" / "smoke.jsonl"
SCALED = {"": [2, 3, 5, 6], "pattern": [10, 25, 40, 50]}
THREADS = 4
SLOTS = 6  # top-2 of 8 experts: the cache caps the ubatch at 3


def cap_nmse(a, b) -> float:
    d = a.astype("f8") - b.astype("f8")
    return float((d * d).sum() / (a.astype("f8") ** 2).sum())


def main() -> int:
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True)
    ap.add_argument("--convert-llama-cpp", type=Path, help="checkout whose converter converts (default --llama-cpp)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--devices", help="comma-separated, e.g. CPU (default: every device of the build)")
    args = ap.parse_args()
    failures: list[str] = []

    def report(ok: bool, what: str) -> None:
        print(f"{'PASS' if ok else 'FAIL'} {what}", flush=True)
        if not ok:
            failures.append(what)

    rows = [json.loads(line) for line in DATASET.read_text().splitlines()]
    conv = args.convert_llama_cpp or args.llama_cpp

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        tiny = convert(generate_tiny(tmpdir, args.seed), conv, "f32")
        bf16 = convert(tmpdir / "kolibri-tiny", conv, "bf16")
        pattern = convert(generate_tiny(tmpdir, args.seed, "pattern"), conv, "f32")
        layers = SCALED[""]

        def session(path: Path, device: str, ubatch: int = 512, slots: int = 0, callback: bool = True):
            return cap.Capture(args.llama_cpp, path, device, 512, ubatch, THREADS, slots, callback=callback)

        probe = session(tiny, "CPU")
        template = probe.template()
        prompts = [probe.tokenize(cap.render(template, row["prompt"], cap.DEFAULT_REASONING)) for row in rows]
        report(len({p[-1] for p in prompts}) == 1 and all(len(p) <= 512 for p in prompts),
               f"{len(prompts)} prompts render through the model's template, end in one token "
               f"({probe.piece(prompts[0][-1])!r}) and fit the context")
        probe.close()
        devices = args.devices.split(",") if args.devices else [n for n, _ in cap.Runner(args.llama_cpp).devices()]

        for gguf, name in ((tiny, "f32"), (bf16, "bf16")):
            for device in devices:
                what = f"{name} on {device}"
                # plain logits from a model that never had a callback
                plain = session(gguf, device, callback=False)
                base = [plain.decode(p) for p in prompts]
                plain.close()
                c = session(gguf, device)
                graph = c.check_graph(prompts[0], layers)
                report(set(graph) == {"l_out-1", "l_out-2", "l_out-4", "l_out-5"} and graph["l_out-5"]["read_by"] == "norm"
                       and graph["l_out-5"]["post_ffn"]["norm"] == "norm-5",
                       f"{what}: the graph reads as documented: l_out-(N-1) is the residual add with the post-FFN norm; the last one feeds output_norm")
                got = [c.capture(p, layers) for p in prompts]
                report(all(np.array_equal(b, g[0]) for b, g in zip(base, got)),
                       f"{what}: capture at {layers}, {len(prompts)} prompts: logits exactly those without a callback")
                every = list(range(1, c.n_layer + 1))
                report(all(np.array_equal(b, c.capture(p, every)[0]) for b, p in zip(base, prompts)),
                       f"{what}: capture of all {c.n_layer} layers: logits exactly those without a callback")
                report(all(a.dtype == np.float32 and a.shape == (c.n_embd,) and np.isfinite(a).all()
                           for g in got for a in g[1].values()), f"{what}: activations are finite float32 [{c.n_embd}]")
                # the row: every position has an output in this decode, so every node has every row
                ok, nearest, worst = True, True, 0.0
                for p, g in zip(prompts[:6], got):
                    _, act_all = c.capture_rows(p, layers)
                    ok = ok and all(np.array_equal(act_all[n][-1], g[1][n]) for n in layers if n != c.n_layer)
                    # The last block runs over one row here and all rows there. The CPU gives equal bits;
                    # Metal's kernels may sum in another order, as another ubatch does (docs/layer-probe.md).
                    last = act_all[c.n_layer]
                    errs = [cap_nmse(g[1][c.n_layer], row) for row in last]
                    worst = max(worst, errs[-1])
                    nearest = nearest and int(np.argmin(errs)) == len(errs) - 1
                report(ok, f"{what}: the kept row is exactly the all-outputs decode's last row, below the last block")
                report(nearest and worst <= MAX_NMSE, f"{what}: at the last block it is nearest the last row, NMSE {worst:.1e}")
                report(not np.array_equal(got[0][1][layers[0]], got[1][1][layers[0]]),
                       f"{what}: two prompts give different activations")
                again = [c.capture(p, layers) for p in prompts[:4]]
                report(all(np.array_equal(a[0], g[0]) and all(np.array_equal(a[1][n], g[1][n]) for n in layers)
                           for a, g in zip(again, got)), f"{what}: a second decode gives equal logits and activations")
                c.close()

        # the expert cache: the ubatch capped at 3 splits a prompt into many computations
        for device in devices:
            what = f"expert cache {SLOTS} slots on {device}"
            try:
                plain = session(tiny, device, slots=SLOTS, callback=False)
            except SystemExit as e:
                report(device == "CPU", f"{what}: refused at load: {e}")
                continue
            base = [plain.decode(p) for p in prompts]
            plain.close()
            c = session(tiny, device, slots=SLOTS)
            report(c.n_ubatch_effective == 3, f"{what}: the context caps the ubatch at {c.n_ubatch_effective}")
            res = session(tiny, device, ubatch=c.n_ubatch_effective)
            got = [c.capture(p, layers) for p in prompts]
            ref = [res.capture(p, layers) for p in prompts]
            report(all(np.array_equal(a, g[0]) for a, g in zip(base, got)),
                   f"{what}: logits exactly those of the cached model without a callback")
            report(all(np.array_equal(r[0], g[0]) and all(np.array_equal(r[1][n], g[1][n]) for n in layers)
                       for r, g in zip(ref, got)),
                   f"{what}: logits and activations exactly a resident run's at the same ubatch")
            res.close()
            c.close()

        # the real layer indices, on the real 50-layer pattern
        c = session(pattern, "CPU")
        graph = c.check_graph(prompts[0], SCALED["pattern"])
        report(sorted(graph) == sorted(["l_out-9", "l_out-24", "l_out-39", "l_out-49"]) and c.n_layer == 50,
               "the real indices 10,25,40,50 on the 50-layer pattern are nodes l_out-9, -24, -39, -49")
        report(graph["l_out-49"]["read_by"] == "norm", "layer 50 feeds output_norm: it is read before the model's final norm")
        plain = session(pattern, "CPU", callback=False)
        base = plain.decode(prompts[0])
        plain.close()
        report(np.array_equal(base, c.capture(prompts[0], SCALED["pattern"])[0]), "50 layers: logits exactly those without a callback")
        c.close()

        # control M79: the issue's numbers taken as zero-based
        right = cap.node_for_layer
        cap.node_for_layer = lambda n: f"l_out-{n}"
        c = session(tiny, "CPU")
        for label, ls in (("the scaled layers", [2, 3]), ("the last layer", [6])):
            try:
                c.check_graph(prompts[0], ls)
                report(False, f"M79: zero-based numbering of {label} passed")
            except (SystemExit, RuntimeError) as e:
                report(True, f"M79: zero-based numbering of {label} is refused: {e}")
        c.close()
        cap.node_for_layer = right

        # control M87: the wiring as a rebase could leave it, on what the real graph recorded
        c = session(tiny, "CPU")
        c.check_graph(prompts[0], layers)
        real_nodes, real_norm = c.nodes, c.post_ffn
        c.close()
        wrong = {"no post-FFN norm": {"ops": ["ADD", "MUL", "RMS_NORM"], "weight": "ffn_shexp_out-1", "norm": None, "experts": ["ffn_moe_out-1", "node_9"]},
                 "another block's norm": real_norm["l_out-1"] | {"weight": "blk.4.post_ffw_norm.weight"},
                 "only the routed experts": real_norm["l_out-1"] | {"experts": ["ffn_moe_out-1", "node_9"]}}
        for label, norm in wrong.items():
            try:
                cap.check_wiring(real_nodes, real_norm | {"l_out-1": norm}, layers, 6)
                report(False, f"M87: {label} passed the graph check")
            except SystemExit as e:
                report("post-FFN norm" in str(e), f"M87: {label} is refused by the graph check")
        try:
            cap.parse_layers("0,6", 6)
            report(False, "M79: layer 0 passed")
        except SystemExit:
            try:
                cap.parse_layers("7", 6)
                report(False, "M79: layer 7 of 6 passed")
            except SystemExit:
                report(True, "M79: layers outside 1 to n_layer are refused")

        # the command line, end to end
        def namespace(out: Path, **kw) -> argparse.Namespace:
            base = dict(llama_cpp=args.llama_cpp, model=tiny, out=out, dataset=DATASET, layers=",".join(map(str, layers)),
                        device="CPU", n_ctx=512, ubatch=512, threads=THREADS, expert_cache=0,
                        reasoning=cap.DEFAULT_REASONING, repeat=True, skip_logits_check=False)
            return argparse.Namespace(**(base | kw))

        def probe_report(d: Path) -> subprocess.CompletedProcess:
            return subprocess.run(["go", "run", "./cmd/kolibri-probe", "report", "-capture", str(d)],
                                  capture_output=True, text=True, cwd=ROOT)

        out = tmpdir / "capture"
        report(cap.run(namespace(out)) == 0, "the command line captures the dataset on tiny")
        manifest = json.loads((out / "capture.json").read_text())
        arr = np.load(out / "layer-6.npy")
        report(arr.dtype == np.dtype("<f4") and arr.shape == (44, manifest["n_embd"]), f"layer-6.npy is {arr.dtype} {arr.shape}")
        report(all(manifest[k] for k in ("model_sha256", "server_sha256", "template_sha256", "template_revision", "dataset_sha256"))
               and manifest["logits_check"] == "equal" and manifest["repeatable"] is True and manifest["template_is_released"] is True,
               "capture.json records the model, server and template hashes, the logits check and the repeat")
        p = probe_report(out)
        report(p.returncode == 0 and "It selects no layer" in p.stdout, "kolibri-probe report reads the capture"
               if p.returncode == 0 else "kolibri-probe report: " + p.stderr[-400:])
        try:
            cap.run(namespace(out))
            report(False, "a second capture overwrote the first")
        except SystemExit:
            report(True, "a capture refuses a directory that holds one: results may exist nowhere else")
        partial = tmpdir / "partial"
        partial.mkdir()
        kept = {name: f"results of {name}".encode() for name in ("layer-2.npy", "expert-cache-stats.json")}
        for name, data in kept.items():
            (partial / name).write_bytes(data)
        try:
            cap.run(namespace(partial))
            report(False, "a capture wrote into a directory that holds results without a capture.json")
        except SystemExit:
            report({p.name: p.read_bytes() for p in partial.iterdir()} == kept,
                   "a capture refuses a directory with anything in it and leaves it as it was")

        # the real command's shape: the cache on the GPU where there is one, the capped ubatch, the report
        device = next((d for d in devices if d != "CPU"), "CPU")
        cached = tmpdir / "capture-cached"
        cached.mkdir()  # an empty directory is as good as a new one
        report(cap.run(namespace(cached, expert_cache=SLOTS, device=device)) == 0 and (cached / "expert-cache-stats.json").exists(),
               f"the command line captures the dataset with --expert-cache {SLOTS} on {device}")
        m = json.loads((cached / "capture.json").read_text())
        report(m["n_ubatch_effective"] == 3 and m["expert_cache_slots"] == SLOTS and m["logits_check"] == "equal" and m["repeatable"] is True,
               "capture.json records the capped ubatch, the slots, the logits check and the repeat")
        report(probe_report(cached).returncode == 0, "kolibri-probe report reads that capture")

        # control M87, the report's side: it reads what the graph check found
        no_norm = tmpdir / "no-norm"
        shutil.copytree(out, no_norm)
        (no_norm / "report.md").unlink()
        (no_norm / "report.json").unlink()
        m = json.loads((no_norm / "capture.json").read_text())
        m["graph_check"]["l_out-5"]["post_ffn"]["ops"] = ["ADD", "MUL", "RMS_NORM"]
        (no_norm / "capture.json").write_text(json.dumps(m))
        p = probe_report(no_norm)
        report(p.returncode != 0 and "post-FFN norm" in p.stderr, "M87: kolibri-probe report refuses a capture whose graph lacks the post-FFN norm")
        shutil.rmtree(no_norm)

        # control M88: a capture that was not checked gets no report, also if it wrote into a node
        unchecked = tmpdir / "unchecked"
        report(cap.run(namespace(unchecked, skip_logits_check=True), mutate_node="l_out-2") == 0,
               "M88: --skip-logits-check lets a capture that writes into a node through the capture")
        p = probe_report(unchecked)
        report(p.returncode != 0 and "logits_check skipped" in p.stderr, "M88: kolibri-probe report refuses that capture")
        norepeat = tmpdir / "norepeat"
        report(cap.run(namespace(norepeat, repeat=False)) == 0, "M88: --no-repeat captures")
        p = probe_report(norepeat)
        report(p.returncode != 0 and "no repeated decode ran" in p.stderr, "M88: kolibri-probe report refuses a capture without a repeat")

        # control M77: a capture that writes into the graph
        mutated = tmpdir / "mutated"
        rc = cap.run(namespace(mutated), mutate_node="l_out-2")
        m = json.loads((mutated / "capture.json").read_text())
        report(rc == 1 and m["logits_check"] == "differ" and len(m["logits_differ"]) == 44,
               f"M77: a capture that overwrites l_out-2 changes the logits of {len(m['logits_differ'])} of 44 prompts, and the exact check fails")
        p = probe_report(mutated)
        report(p.returncode != 0 and "changed the logits" in p.stderr, "M77: kolibri-probe report refuses that capture")
        shutil.rmtree(mutated)

    if failures:
        print(f"FAIL {len(failures)} problem(s)")
        return 1
    print("PASS: 0 failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
