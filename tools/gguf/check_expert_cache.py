#!/usr/bin/env python3
"""Check libllama's expert cache (patch 0011) on the tiny Kolibri checkpoint.

The fixture is cmd/kolibri-tiny -router (384 experts, top 6), converted to F32 and to Q8_0. On
every device, 40 prompt tokens are prefilled and 60 more decoded one by one, once with the experts
resident and once through the expert cache with 6, 16, 64 and 384 slots per layer, with and without
prefetch, and with buffered reads (no F_NOCACHE). The cache caps the ubatch at slots / 6, so the resident run uses that ubatch too, and on
the CPU its weights are not repacked (the slots are not either).

- logits: bit-identical to the resident run, at every position. The slots hold the same bytes as
  the stacked tensors and go through the same kernels, so nothing may differ;
- counters (--expert-cache-stats): per phase, hits + hits_loading + misses = needed, and the
  demand bytes are misses times an expert's bytes; every step of every layer is a remap (a ubatch
  of one token counts as decode, so with a ubatch of 1 the prompt does too); loads into a slot
  never outnumber the slots plus what freed one (evictions, cancellations, failed loads); and the
  predictor runs in decode exactly when asked for, never in prefill; the read mode is recorded; the
  decode timing counts every decode ubatch, and its remap and wait fit within its wall time. The
  resident run writes the decode timing alone (stats without the cache). Control M42: the graph
  splits per decoded token are the known counts, on the GPU 2 resident and 2 + 2 per layer with
  the cache (each remap is a CPU split), on the CPU 1;
- the ubatch cap is logged, and fewer slots than the experts of one token fail to load;
- slots per layer (n_slots_layer) give bit-identical logits too, and a model with the cache serves
  one context: a second one fails;
- placement: with only half the layers on the GPU, every layer's slots go to its own device, and
  with the CPU's repacking buffer types enabled the slots still hold the file's bytes; both again
  bit-identical to the resident run without repacking;
- the command line: llama-completion with --expert-cache-layers, --expert-prefetch,
  --expert-cache-buffered and --expert-cache-stats generates with the cache as asked (the stats
  record the slots, the prefetch and the read mode), and greedily the same text as without it;
  --expert-cache-stats without the cache writes the decode timing alone; with a quote, a backslash and a tab in the
  model's file name the stats are still valid JSON and name the file.

With --real GGUF it checks a real checkpoint on the GPU instead: the first 160 tokens of a
locality workload (128 prefilled, 32 decoded), resident against the cache with 128 and 64 slots per
layer, with and without prefetch, again bit-identical, with the same counter checks. Its reads may
come from the page cache, since the resident run maps the whole file first.

    check_expert_cache.py --llama-cpp third_party/llama.cpp
    check_expert_cache.py --llama-cpp third_party/llama.cpp --real ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from check_model import Runner
from tiny import convert, generate_tiny

N_PROMPT, N_DECODE = 40, 60
SLOTS = [6, 16, 64, 384]
VARIANTS = [(0, False), (12, False), (0, True)]  # (prefetch, buffered reads)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True, help="llama.cpp checkout with a built libllama")
    ap.add_argument("--seed", type=int, default=1, help="cmd/kolibri-tiny random seed")
    ap.add_argument("--real", type=Path, help="a real Kolibri GGUF, checked on the GPU instead of the tiny checkpoint")
    ap.add_argument("--tokens", type=Path, default=Path.home() / "models/eval/locality/coding.tokens.npy",
                    help="the token IDs of a locality workload, for --real")
    args = ap.parse_args()
    import numpy as np

    r = Runner(args.llama_cpp)
    failed = 0

    def report(ok: bool, msg: str) -> None:
        nonlocal failed
        print(("PASS " if ok else "FAIL ") + msg, flush=True)
        failed += not ok

    def check(gguf: Path, dev_name: str, dev, tokens: list[int], n_prompt: int, slots_list: list[int],
              n_layer: int, k: int, stats: Path, label: str) -> None:
        chunks = [n_prompt] + [1] * (len(tokens) - n_prompt)
        for slots in slots_list:
            ub = min(n_prompt, max(1, slots // k))  # n_batch is the largest chunk, the prompt
            # the prompt's ubatches; one of a single token counts as decode
            sizes = [ub] * (n_prompt // ub) + ([n_prompt % ub] if n_prompt % ub else [])
            n_pre = sum(n > 1 for n in sizes)
            n_dec = len(tokens) - n_prompt + sum(n == 1 for n in sizes)

            # resident, with the decode timing only (stats without the cache)
            stats.unlink(missing_ok=True)
            ref = r.logits(gguf, dev, tokens, ub, chunks=chunks, no_repack=True, expert_cache={"stats_path": str(stats)})
            s = json.loads(stats.read_text())
            d = s["decode"]
            # M42: the scheduler's splits per decoded token are known: on the GPU 2 resident (docs/real-checkpoint.md)
            # and 2 more per layer with the cache (each layer's remap is a CPU split); on the CPU 1 either way
            on_gpu = r.lib.ggml_backend_dev_type(dev) == r.lib.GGML_BACKEND_DEVICE_TYPE_GPU
            splits_resident, splits_cache = (2, 2 + 2 * n_layer) if on_gpu else (1, 1)
            report(s["expert_cache"] is None and d["tokens"] == n_dec and d["wall_seconds"] > 0
                   and d["splits"] == splits_resident * d["tokens"] and "max_phys_footprint" in s["process"],
                   f"{label} {dev_name}, resident, ubatch {ub}: decode timing without the cache ({d['tokens']} tokens, "
                   f"{d['splits'] / max(d['tokens'], 1):.2f} splits and {(d['metal_commits'] or 0) / max(d['tokens'], 1):.2f} "
                   f"Metal commits per token, {1e3 * d['wall_seconds'] / max(d['tokens'], 1):.3f} ms per token)")
            for pf, buffered in VARIANTS:
                what = f"{label} {dev_name}, {slots} slots, ubatch {ub}, prefetch {pf}" + (", buffered" if buffered else "")
                stats.unlink(missing_ok=True)
                got = r.logits(gguf, dev, tokens, n_prompt, chunks=chunks, no_repack=True,
                               expert_cache={"n_slots": slots, "prefetch_k": pf, "stats_path": str(stats), "buffered": buffered})
                log = "".join(r.log)
                same = np.array_equal(got, ref)
                diff = "" if same else f", max |diff| {float(np.abs(got - ref).max()):.3g}"
                report(same, f"{what}: logits of {len(tokens)} positions bit-identical to the resident run{diff}")
                if ub < n_prompt:
                    report(f"n_ubatch {n_prompt} capped to {ub}" in log, f"{what}: the context caps n_ubatch at {ub}")

                s = json.loads(stats.read_text())
                eb = s["expert_cache"]["expert_bytes"]
                pre, dec = s["prefill"], s["decode"]
                ok = all(c["hits"] + c["hits_loading"] + c["misses"] == c["needed"] and c["demand_bytes"] == c["misses"] * eb
                         for c in (pre, dec))
                ok = ok and pre["steps"] == n_pre * n_layer and dec["steps"] == n_dec * n_layer
                # the read mode as asked; the decode timing: every decode ubatch, its remaps and waits within its wall time
                ok = ok and s["expert_cache"]["buffered"] == (buffered or sys.platform != "darwin")
                ok = ok and dec["tokens"] == n_dec and min(pre["remap_seconds"], dec["remap_seconds"]) >= 0 \
                    and dec["remap_seconds"] + dec["wait_seconds"] <= dec["wall_seconds"] and dec["splits"] == splits_cache * dec["tokens"]
                tot = {key: pre[key] + dec[key] for key in pre}
                ok = ok and tot["misses"] + tot["prefetch_issued"] <= \
                    slots * n_layer + tot["evictions"] + tot["prefetch_cancelled"] + tot["load_failures"]
                ran = dec["prefetch_issued"] + dec["prefetch_skipped"]
                ok = ok and (ran > 0) == (pf > 0) and pre["prefetch_issued"] + pre["prefetch_skipped"] == 0
                ok = ok and tot["load_failures"] == 0
                ok = ok and 0 <= tot["io_prefetch_seconds"] <= tot["io_seconds"] and (tot["prefetch_bytes"] > 0) == (tot["io_prefetch_seconds"] > 0)
                report(ok, f"{what}: counters consistent (prefill {pre['steps']} remaps, {pre['misses']} misses; "
                           f"decode {dec['steps']} remaps, {dec['misses']} misses, {dec['hits']} hits, "
                           f"{dec['hits_loading']} awaited, prefetch {dec['prefetch_used']}/{dec['prefetch_issued']} used, "
                           f"{dec['prefetch_skipped']} skipped, {dec['prefetch_cancelled']} cancelled; "
                           f"wait {dec['wait_seconds']:.3f} s, I/O {dec['io_seconds']:.3f} s ({dec['io_prefetch_seconds']:.3f} s prefetch) in decode; "
                           f"per decoded token {1e3 * dec['wall_seconds'] / max(dec['tokens'], 1):.3f} ms, "
                           f"{1e3 * dec['remap_seconds'] / max(dec['tokens'], 1):.3f} ms remap, {dec['splits'] / max(dec['tokens'], 1):.2f} splits, "
                           f"{(dec['metal_commits'] or 0) / max(dec['tokens'], 1):.2f} Metal commits; "
                           f"{s['process']['disk_bytes_read']} bytes read from disk)")

    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "stats.json"
        if args.real:
            tokens = [int(t) for t in np.load(args.tokens)[:160]]
            gpu = [(n, d) for n, d in r.devices() if r.lib.ggml_backend_dev_type(d) == r.lib.GGML_BACKEND_DEVICE_TYPE_GPU]
            check(args.real, *gpu[0], tokens, 128, [128, 64], 50, 6, stats, args.real.stem)
        else:
            model = generate_tiny(Path(tmp), args.seed, "router")
            cfg = json.loads((model / "config.json").read_text())
            n_layer, k = cfg["num_hidden_layers"], cfg["num_experts_per_tok"]
            rng = random.Random(args.seed)
            tokens = [rng.randrange(1000, 100000) for _ in range(N_PROMPT + N_DECODE)]
            for outtype in ["f32", "q8_0"]:
                gguf = convert(model, args.llama_cpp, outtype)
                for dev_name, dev in r.devices():
                    check(gguf, dev_name, dev, tokens, N_PROMPT, SLOTS, n_layer, k, stats, outtype)

                # fewer slots than one token's experts cannot work
                try:
                    r.logits(gguf, r.devices()[0][1], tokens[:4], 1, expert_cache={"n_slots": k - 1})
                    report(False, f"{outtype}: {k - 1} slots per layer loaded")
                except RuntimeError:
                    report(f"at least {k} needed" in "".join(r.log), f"{outtype}: {k - 1} slots per layer fail to load")

                # slots per layer: the fewest decide the ubatch
                per_layer = [12, 64, 18, 384, 30, 100][:n_layer]
                ub = min(per_layer) // k
                for dev_name, dev in r.devices():
                    ref = r.logits(gguf, dev, tokens, ub, chunks=[N_PROMPT] + [1] * N_DECODE, no_repack=True)
                    stats.unlink(missing_ok=True)
                    got = r.logits(gguf, dev, tokens, N_PROMPT, chunks=[N_PROMPT] + [1] * N_DECODE, no_repack=True,
                                   expert_cache={"n_slots_layer": per_layer, "prefetch_k": 8, "stats_path": str(stats)})
                    s = json.loads(stats.read_text())["expert_cache"]
                    report(np.array_equal(got, ref) and (s["slots_min"], s["slots_max"]) == (min(per_layer), max(per_layer)),
                           f"{outtype} {dev_name}, slots per layer {per_layer}, ubatch {ub}: logits bit-identical to the resident run")

                # half the layers on the GPU; and the CPU with repacking on, which the slots never use
                gpu = [(n, d) for n, d in r.devices() if r.lib.ggml_backend_dev_type(d) == r.lib.GGML_BACKEND_DEVICE_TYPE_GPU]
                cpu = [(n, d) for n, d in r.devices() if r.lib.ggml_backend_dev_type(d) == r.lib.GGML_BACKEND_DEVICE_TYPE_CPU]
                cases = [(f"{gpu[0][0]} with {n_layer // 2} of {n_layer} layers", gpu[0][1], n_layer // 2, True)] if gpu else []
                cases.append((f"{cpu[0][0]} with repacking", cpu[0][1], -1, False))
                for what, dev, ngl, no_repack in cases:
                    ref = r.logits(gguf, dev, tokens, 2, chunks=[N_PROMPT] + [1] * N_DECODE, no_repack=True, n_gpu_layers=ngl)
                    stats.unlink(missing_ok=True)
                    got = r.logits(gguf, dev, tokens, N_PROMPT, chunks=[N_PROMPT] + [1] * N_DECODE, no_repack=no_repack,
                                   n_gpu_layers=ngl, expert_cache={"n_slots": 16, "stats_path": str(stats)})
                    log = "".join(r.log)
                    placed = all(f"MiB for {name}" in log for name in ({gpu[0][0], cpu[0][0]} if ngl > 0 else {cpu[0][0]}))
                    report(np.array_equal(got, ref) and placed,
                           f"{outtype} {what}, 16 slots: logits bit-identical to the resident run without repacking"
                           + ("" if placed else f" (slots not on every layer's device: {[ln for ln in log.splitlines() if 'slots per layer' in ln]})"))

            # one context per model
            gguf = model.parent / f"{model.name}-q8_0.gguf"
            ec = r.ffi.new("struct llama_expert_cache_params *", r.lib.llama_expert_cache_default_params())
            ec.n_slots = 16
            mparams = r.ll.model_default_params(expert_cache=ec)
            m = r.lib.llama_model_load_from_file(str(gguf).encode(), mparams)
            cparams = r.ll.context_default_params(n_ctx=256, n_batch=32, n_ubatch=32)
            c1 = r.lib.llama_init_from_model(m, cparams)
            r.log.clear()
            c2 = r.lib.llama_init_from_model(m, cparams)
            report(c1 != r.ffi.NULL and c2 == r.ffi.NULL and "one context per model" in "".join(r.log),
                   "a model with the expert cache serves one context: the second fails")
            r.lib.llama_free(c1)
            c3 = r.lib.llama_init_from_model(m, cparams)
            report(c3 != r.ffi.NULL, "after the first context is freed, a new one gets the cache")
            r.lib.llama_free(c3)
            r.lib.llama_model_free(m)

            # the command line
            bin_dir = args.llama_cpp / "build" / "bin"
            layers = Path(tmp) / "slots.txt"
            layers.write_text("12, 64 18\n384,30 100\n")
            prompt = "Die Energiewende in Deutschland hat das Ziel"
            base = [str(bin_dir / "llama-completion"), "-m", str(gguf), "-p", prompt, "-n", "24", "--temp", "0",
                    "--ignore-eos", "-no-cnv", "--no-display-prompt", "-c", "256", "--no-warmup"]
            plain = subprocess.run(base + ["-ub", "2"], capture_output=True, text=True)
            stats.unlink(missing_ok=True)
            cached = subprocess.run(base + ["--expert-cache-layers", str(layers), "--expert-prefetch", "8,4,0.25",
                                            "--expert-cache-threads", "2", "--expert-cache-buffered", "--expert-cache-stats", str(stats)],
                                    capture_output=True, text=True)
            ok = plain.returncode == 0 and cached.returncode == 0 and stats.exists()
            if ok:
                s = json.loads(stats.read_text())
                c = s["expert_cache"]
                ok = (c["slots_min"], c["slots_max"], c["prefetch_k"], c["prefetch_half_life"], c["prefetch_mix"], c["io_threads"], c["buffered"]) == \
                     (12, 384, 8, 4, 0.25, 2, True) and s["decode"]["steps"] > 0 and s["decode"]["tokens"] > 0
            # without the cache, --expert-cache-stats writes the decode timing alone
            stats.unlink(missing_ok=True)
            resident = subprocess.run(base + ["-ub", "2", "--expert-cache-stats", str(stats)], capture_output=True, text=True)
            try:
                rs = json.loads(stats.read_text())
                rok = rs["expert_cache"] is None and rs["decode"]["tokens"] >= 23 and rs["decode"]["splits"] > 0
            except (OSError, ValueError, KeyError) as e:
                rs, rok = str(e), False
            report(resident.returncode == 0 and rok and resident.stdout == plain.stdout,
                   f"llama-completion --expert-cache-stats without the cache: decode timing only, same text ({rs if not rok else rs['decode']})")
            odd = Path(tmp) / 'we"ird\\na\tme.gguf'
            shutil.copyfile(gguf, odd)
            stats.unlink(missing_ok=True)
            run = subprocess.run([str(bin_dir / "llama-completion"), "-m", str(odd), "-p", prompt, "-n", "4", "-no-cnv",
                                  "-c", "256", "--no-warmup", "--expert-cache", "16", "--expert-cache-stats", str(stats)],
                                 capture_output=True, text=True)
            try:
                named = json.loads(stats.read_text())["expert_cache"]["file"]
            except (OSError, ValueError) as e:
                named = f"unreadable: {e}"
            report(run.returncode == 0 and Path(named).name == odd.name,
                   f"stats of a model named {odd.name!r} are valid JSON and name it ({named!r})")
            report(ok and plain.stdout == cached.stdout and plain.stdout.strip() != "",
                   f"llama-completion --expert-cache-layers/--expert-prefetch/--expert-cache-buffered/--expert-cache-stats: as asked, same text as without the cache "
                   f"({plain.stdout.strip()[:60]!r})" + ("" if ok else f" rc {plain.returncode}/{cached.returncode}: {cached.stderr.strip()[-300:]}"))

    print(f"{'FAIL' if failed else 'PASS'}: {failed} failures")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
