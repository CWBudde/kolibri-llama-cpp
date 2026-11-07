#!/usr/bin/env python3
"""Check libllama's expert cache (patch 0011) on the tiny Kolibri checkpoint.

The fixture is cmd/kolibri-tiny -router (384 experts, top 6), converted to F32 and to Q8_0. On
every device, 40 prompt tokens are prefilled and 60 more decoded one by one, once with the experts
resident and once through the expert cache with 6, 16, 64 and 384 slots per layer, with and without
prefetch, and with buffered reads (no F_NOCACHE). The cache caps the ubatch at slots / 6, so the resident run uses that ubatch too, and on
the CPU its weights are not repacked (the slots are not either).

- logits: bit-identical to the resident run, at every position. The slots hold the same bytes as
  the stacked tensors and go through the same kernels, so nothing may differ. On the GPU the
  resident run is split where the cache's CPU remap splits the graph, after each layer's
  ffn_moe_topk: Metal fuses ops across those points (RMS_NORM + MUL + ADD), and the fused kernels
  round differently. The difference to an unsplit resident run is reported as info;
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

Waves (--expert-cache-waves): only on fork branch feat/kolibri1-waves, not in the patch series
(patch 0013 is retired, docs/expert-streaming.md, "Stage E4"). Against a checkout without them the
wave checks are skipped with a note, and --waves-only is an error. The prompt runs in one ubatch,
uncapped, with 16, 18, 24, 64 and 384 slots on every device, with and without prefetch and buffered
reads; and a prompt of 4 tokens at 18 slots, whose union does not fit the built waves at their usual
size.

- logits: bit-identical to the resident run at the same ubatch (the whole prompt), split as above;
- with at least 3 * 6 slots nothing is capped; below that the ubatch is capped as before and a
  warning says why; with 384 slots every expert fits, so there are no waves;
- counters: one plan per layer and prefill ubatch, waves only in prefill and only where the
  prompt's experts can exceed the slots, the graph's passes as built (as few as waves of up to
  slots - 6 experts take), the decode counts as without waves.

Side-file (--expert-cache-sidefile): patch 0014. Against a checkout without it the side-file checks are
skipped with a note. cmd/kolibri-sidefile writes and verifies the side-file of each fixture; then, next
to every run of the GGUF path above, the same run reads its misses from the side-file (and so with
prefetch and buffered reads):

- logits: bit-identical to the same resident run, and to the GGUF path's run;
- counters: every phase's loads are what the misses and prefetches read, at exactly 1 pread per expert
  on the side-file and 3 on the GGUF path; the stats name the side-file (null on the GGUF path); without
  prefetch, hits and misses are the GGUF path's;
- the ranges: test-expert-cache --sidefile reads every expert on both paths, with the ranges taken from
  the GGUF's own header by ggml's gguf API as the model's loader takes them, and compares the bytes;
- refusals at load, each by name of what is wrong: a flipped table byte, a truncated file, no file (no
  fallback), another GGUF's rows, another GGUF's header, a side-file without the cache; a flipped data
  byte loads, since the load does not hash 31 GB, and kolibri-sidefile verify reports it.

With --real GGUF it checks a real checkpoint on the GPU instead: the first 160 tokens of a
locality workload (128 prefilled, 32 decoded), resident against the cache with 128 and 64 slots per
layer, with and without prefetch, again bit-identical, with the same counter checks; with waves the
128 prompt tokens run in one ubatch (4 and 7 waves per layer built), 6 tokens at 18 slots grow
their waves, and 512 tokens at 128 slots check the benchmark's own ubatch. Its reads may come from
the page cache, since the resident run maps the whole file first. With side-file support the side-file
is <GGUF>.experts (or --sidefile) and must exist; the --real run checks the pair's bytes on 8 experts
of each layer and not the whole file: kolibri-sidefile verify does that.

    check_expert_cache.py --llama-cpp third_party/llama.cpp
    check_expert_cache.py --llama-cpp third_party/llama.cpp --real ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf
    check_expert_cache.py --llama-cpp third_party/llama.cpp --waves-only   # the wave checks alone, on a waves build
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
from tiny import ROOT, convert, generate_tiny

N_PROMPT, N_DECODE = 40, 60
SLOTS = [6, 16, 64, 384]
VARIANTS = [(0, False), (12, False), (0, True)]  # (prefetch, buffered reads)
WAVE_SLOTS = [16, 18, 24, 64, 384]
N_GROWN = 4  # 4 tokens' 24 pairs at 18 slots: 2 waves built, and a union above 2 * 6 grows them
N_GROWN_REAL = 6  # likewise on a real checkpoint: 36 pairs at 18 slots, 3 waves built
N_BENCH_UBATCH = 512  # testdata/bench/e4.json's arms prefill in ubatches of the server's default


def has_waves(r: Runner, llama_cpp: Path) -> bool:
    """Whether the checkout has --expert-cache-waves. cffi parses the struct from the checkout's
    llama.h, so the header decides what the harness writes. A libllama built from another tree has
    the same size and offsets, with waves in what is padding there, so it would ignore waves=true
    and silently stay capped; the two must agree: the library has the stats key waves_built exactly
    when the header has the field."""
    header = any(name == "waves" for name, _ in r.ffi.typeof("struct llama_expert_cache_params").fields)
    path = llama_cpp / "build" / "bin" / ("libllama.dylib" if sys.platform == "darwin" else "libllama.so")
    library = b"waves_built" in path.read_bytes()
    if header != library:
        raise SystemExit(f"{llama_cpp}: include/llama.h {'has' if header else 'has no'} expert cache field waves, "
                         f"but {path.name} was built {'without' if header else 'with'} it: rebuild the checkout")
    return header


def has_sidefile(r: Runner, llama_cpp: Path) -> bool:
    """Whether the checkout has --expert-cache-sidefile, by the same rule as has_waves: the header and
    the built library must agree."""
    header = any(name == "sidefile_path" for name, _ in r.ffi.typeof("struct llama_expert_cache_params").fields)
    path = llama_cpp / "build" / "bin" / ("libllama.dylib" if sys.platform == "darwin" else "libllama.so")
    library = b'"sidefile"' in path.read_bytes()
    if header != library:
        raise SystemExit(f"{llama_cpp}: include/llama.h {'has' if header else 'has no'} expert cache field sidefile_path, "
                         f"but {path.name} was built {'without' if header else 'with'} it: rebuild the checkout")
    return header


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True, help="llama.cpp checkout with a built libllama")
    ap.add_argument("--seed", type=int, default=1, help="cmd/kolibri-tiny random seed")
    ap.add_argument("--real", type=Path, help="a real Kolibri GGUF, checked on the GPU instead of the tiny checkpoint")
    ap.add_argument("--tokens", type=Path, default=Path.home() / "models/eval/locality/coding.tokens.npy",
                    help="the token IDs of a locality workload, for --real")
    ap.add_argument("--sidefile", type=Path, help="the side-file of --real (default: <GGUF>.experts)")
    ap.add_argument("--waves-only", action="store_true", help="only the wave checks")
    args = ap.parse_args()
    import numpy as np

    r = Runner(args.llama_cpp)
    waves_on = has_waves(r, args.llama_cpp)
    if args.waves_only and not waves_on:
        ap.error(f"--waves-only: {args.llama_cpp} has no expert cache waves; they need a build of fork branch feat/kolibri1-waves")
    if not waves_on:
        print(f"note: {args.llama_cpp} has no expert cache waves (fork branch feat/kolibri1-waves): skipping check_waves", flush=True)
    sidefile_on = has_sidefile(r, args.llama_cpp)
    if not sidefile_on:
        print(f"note: {args.llama_cpp} has no expert cache side-file (patch 0014): skipping the side-file checks", flush=True)
    failed = 0

    def report(ok: bool, msg: str) -> None:
        nonlocal failed
        print(("PASS " if ok else "FAIL ") + msg, flush=True)
        failed += not ok

    def on_gpu(dev) -> bool:
        return r.lib.ggml_backend_dev_type(dev) == r.lib.GGML_BACKEND_DEVICE_TYPE_GPU

    def resident(gguf: Path, dev, tokens: list[int], ub: int, **kw):
        """The resident run the cache must match: on the GPU split after each layer's ffn_moe_topk,
        where the cache's CPU remap splits the graph."""
        return r.logits(gguf, dev, tokens, ub, no_repack=True, split_after=("ffn_moe_topk",) if on_gpu(dev) else (), **kw)

    def reads_ok(s: dict, side: Path | None) -> bool:
        """Every phase's loads are what its misses and prefetches read, at one pread per expert from
        the side-file and three from the GGUF; the stats name the file they used."""
        if not sidefile_on:
            return True
        eb, per = s["expert_cache"]["expert_bytes"], 1 if side else 3
        ok = s["expert_cache"]["sidefile"] == (str(side) if side else None)
        for c in (s["prefill"], s["decode"]):
            ok = ok and c["loads"] * eb == c["demand_bytes"] + c["prefetch_bytes"] and c["preads"] == per * c["loads"]
        return ok

    def check(gguf: Path, dev_name: str, dev, tokens: list[int], n_prompt: int, slots_list: list[int],
              n_layer: int, k: int, stats: Path, label: str, side: Path | None = None) -> None:
        chunks = [n_prompt] + [1] * (len(tokens) - n_prompt)
        for slots in slots_list:
            ub = min(n_prompt, max(1, slots // k))  # n_batch is the largest chunk, the prompt
            # the prompt's ubatches; one of a single token counts as decode
            sizes = [ub] * (n_prompt // ub) + ([n_prompt % ub] if n_prompt % ub else [])
            n_pre = sum(n > 1 for n in sizes)
            n_dec = len(tokens) - n_prompt + sum(n == 1 for n in sizes)

            # resident, with the decode timing only (stats without the cache)
            stats.unlink(missing_ok=True)
            plain = r.logits(gguf, dev, tokens, ub, chunks=chunks, no_repack=True, expert_cache={"stats_path": str(stats)})
            ref = resident(gguf, dev, tokens, ub, chunks=chunks) if on_gpu(dev) else plain
            s = json.loads(stats.read_text())
            d = s["decode"]
            # M42: the scheduler's splits per decoded token are known: on the GPU 2 resident (docs/real-checkpoint.md)
            # and 2 more per layer with the cache (each layer's remap is a CPU split); on the CPU 1 either way
            splits_resident, splits_cache = (2, 2 + 2 * n_layer) if on_gpu(dev) else (1, 1)
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
                if on_gpu(dev):
                    print(f"info {what}: max |diff| to the unsplit resident run {float(np.abs(got - plain).max()):.3g}")
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
                ok = ok and reads_ok(s, None)
                report(ok, f"{what}: counters consistent (prefill {pre['steps']} remaps, {pre['misses']} misses; "
                           f"decode {dec['steps']} remaps, {dec['misses']} misses, {dec['hits']} hits, "
                           f"{dec['hits_loading']} awaited, prefetch {dec['prefetch_used']}/{dec['prefetch_issued']} used, "
                           f"{dec['prefetch_skipped']} skipped, {dec['prefetch_cancelled']} cancelled; "
                           f"wait {dec['wait_seconds']:.3f} s, I/O {dec['io_seconds']:.3f} s ({dec['io_prefetch_seconds']:.3f} s prefetch) in decode; "
                           f"per decoded token {1e3 * dec['wall_seconds'] / max(dec['tokens'], 1):.3f} ms, "
                           f"{1e3 * dec['remap_seconds'] / max(dec['tokens'], 1):.3f} ms remap, {dec['splits'] / max(dec['tokens'], 1):.2f} splits, "
                           f"{(dec['metal_commits'] or 0) / max(dec['tokens'], 1):.2f} Metal commits; "
                           f"{s['process']['disk_bytes_read']} bytes read from disk)")

                if side is None:
                    continue
                gguf_pre, gguf_dec = pre, dec
                stats.unlink(missing_ok=True)
                on_side = r.logits(gguf, dev, tokens, n_prompt, chunks=chunks, no_repack=True,
                                   expert_cache={"n_slots": slots, "prefetch_k": pf, "stats_path": str(stats), "buffered": buffered,
                                                 "sidefile_path": str(side)})
                same = np.array_equal(on_side, ref)
                diff = "" if same else f", max |diff| {float(np.abs(on_side - ref).max()):.3g}"
                report(same, f"{what}, side-file: logits of {len(tokens)} positions bit-identical to the resident run{diff}")
                report(np.array_equal(on_side, got), f"{what}, side-file: logits bit-identical to the GGUF path")
                s = json.loads(stats.read_text())
                pre, dec = s["prefill"], s["decode"]
                ok = reads_ok(s, side) and pre["load_failures"] + dec["load_failures"] == 0
                ok = ok and s["expert_cache"]["buffered"] == (buffered or sys.platform != "darwin")
                # prefetches race with the remaps, so only a run without them has the same hits and misses
                if pf == 0:
                    ok = ok and all(a[key] == b[key] for a, b in ((pre, gguf_pre), (dec, gguf_dec)) for key in ("needed", "hits", "misses", "demand_bytes", "evictions"))
                report(ok, f"{what}, side-file: {pre['preads']} and {dec['preads']} preads for {pre['loads']} and {dec['loads']} experts in prefill and decode "
                           f"({gguf_pre['preads']} and {gguf_dec['preads']} on the GGUF path); prefetch {dec['prefetch_used']}/{dec['prefetch_issued']} used")

    def kolibri_sidefile(*cmd: str) -> subprocess.CompletedProcess:
        return subprocess.run(["go", "run", "./cmd/kolibri-sidefile", *cmd], capture_output=True, text=True, cwd=ROOT)

    def pair(gguf: Path, side: Path, label: str, per_layer: int = 0) -> None:
        """Both read paths on the GGUF's own ranges, in test-expert-cache."""
        run = subprocess.run([str(args.llama_cpp / "build" / "bin" / "test-expert-cache"), "--sidefile", str(gguf), str(side), str(per_layer)],
                             capture_output=True, text=True)
        report(run.returncode == 0 and " 0 differences" in run.stdout, f"{label}: {run.stdout.strip()}{run.stderr.strip()[-300:]}")

    def write_sidefile(gguf: Path, label: str) -> Path:
        """kolibri-sidefile write and verify on a fixture, then both read paths on its own ranges."""
        side = gguf.with_name(gguf.name + ".experts")
        run = kolibri_sidefile("write", "-gguf", str(gguf), "-force")
        if run.returncode != 0:
            raise SystemExit(f"kolibri-sidefile write {gguf}: {run.stderr.strip()[-600:]}")
        run = kolibri_sidefile("verify", "-gguf", str(gguf))
        report(run.returncode == 0, f"{label}: kolibri-sidefile verify: {(run.stdout + run.stderr).strip().splitlines()[0][:120]}")
        pair(gguf, side, label)
        return side

    def refused(gguf: Path, side: Path, tokens: list[int], needle: str, what: str) -> None:
        try:
            r.logits(gguf, r.devices()[0][1], tokens[:4], 1, expert_cache={"n_slots": 16, "sidefile_path": str(side)})
            report(False, f"{what}: loaded")
        except RuntimeError:
            log = "".join(r.log)
            line = next((ln.strip() for ln in log.splitlines() if "side-file" in ln), log[-200:])
            report(needle in log, f"{what}: refused at load: {line[:240]}")

    def check_refusals(gguf: Path, other: Path | None, side: Path, tokens: list[int], outtype: str) -> None:
        """What the load refuses, and the one fault it leaves to kolibri-sidefile verify."""
        data = side.read_bytes()
        bad = side.with_name(side.name + ".bad")
        def corrupt(edit) -> Path:
            bad.write_bytes(edit(bytearray(data)))
            return bad
        def flip(b: bytearray, at: int) -> bytearray:
            b[at] ^= 1
            return b
        refused(gguf, corrupt(lambda b: flip(b, 200)), tokens, "table checksum", f"{outtype} side-file with a flipped table byte")
        refused(gguf, corrupt(lambda b: b[:len(data) - 16384]), tokens, "file size", f"{outtype} truncated side-file")
        refused(gguf, corrupt(lambda b: b[:100]), tokens, "too short", f"{outtype} side-file cut inside its header")
        refused(gguf, side.with_name("no-such-file"), tokens, "no fallback", f"{outtype} missing side-file")
        if other is not None:
            refused(other, side, tokens, "row bytes", f"{outtype} side-file on a GGUF of another type")
        # the same tensors, one metadata byte apart
        header = bytearray(gguf.read_bytes())
        at = header.index(b"general.name") + len(b"general.name") + 4 + 8
        header[at] ^= 1
        changed = gguf.with_name("other-header-" + gguf.name)
        changed.write_bytes(header)
        refused(changed, side, tokens, "another GGUF", f"{outtype} side-file on a GGUF with another header")
        changed.unlink()
        # a data byte: the load cannot know, the tool does
        flipped = corrupt(lambda b: flip(b, len(data) - 16384 + 3))
        shutil.copyfile(str(side) + ".manifest.json", str(flipped) + ".manifest.json")
        run = kolibri_sidefile("verify", "-gguf", str(gguf), "-o", str(flipped))
        loaded = True
        try:
            r.logits(gguf, r.devices()[0][1], tokens[:4], 1, expert_cache={"n_slots": 16, "sidefile_path": str(flipped)})
        except RuntimeError:
            loaded = False
        report(loaded and run.returncode != 0 and "extent differs" in run.stderr,
               f"{outtype} side-file with a flipped data byte: the load does not hash the data ({'loaded' if loaded else 'refused'}), "
               f"kolibri-sidefile verify reports it: {run.stderr.strip()[:160]}")
        Path(str(flipped) + ".manifest.json").unlink()
        bad.unlink()

    def check_waves(gguf: Path, dev_name: str, dev, tokens: list[int], n_prompt: int, slots_list: list[int],
                    n_layer: int, n_expert: int, k: int, stats: Path, label: str, grown: bool = False) -> None:
        if not waves_on:
            return
        chunks = [n_prompt] + [1] * (len(tokens) - n_prompt)
        n_dec = len(tokens) - n_prompt
        n_ctx = max(256, len(tokens))
        for slots in slots_list:
            waves = slots >= 3 * k
            ub = n_prompt if waves else min(n_prompt, max(1, slots // k))
            n_pre = -(-n_prompt // ub)
            touch = min(n_expert, ub * k)
            built = -(-touch // (slots - k)) if waves and touch > slots else 1  # waves per layer the graph holds
            ref = resident(gguf, dev, tokens, ub, chunks=chunks, n_ctx=n_ctx)
            for pf, buffered in VARIANTS:
                what = f"{label} {dev_name}, waves, {slots} slots, ubatch {ub}, prefetch {pf}" + (", buffered" if buffered else "")
                stats.unlink(missing_ok=True)
                got = r.logits(gguf, dev, tokens, n_prompt, chunks=chunks, no_repack=True, n_ctx=n_ctx,
                               expert_cache={"n_slots": slots, "prefetch_k": pf, "stats_path": str(stats), "buffered": buffered, "waves": True})
                log = "".join(r.log)
                same = np.array_equal(got, ref)
                diff = "" if same else f", max |diff| {float(np.abs(got - ref).max()):.3g}"
                report(same, f"{what}: logits of {len(tokens)} positions bit-identical to the resident run at ubatch {ub}{diff}")
                capped = f"n_ubatch {n_prompt} capped to {ub}" in log
                report(capped == (ub < n_prompt) and (waves or "the ubatch stays capped" in log),
                       f"{what}: " + ("not capped" if waves else f"capped at {ub}, with the warning"))

                s = json.loads(stats.read_text())
                eb = s["expert_cache"]["expert_bytes"]
                pre, dec = s["prefill"], s["decode"]
                ok = s["expert_cache"]["waves"] == waves
                ok = ok and all(c["hits"] + c["hits_loading"] + c["misses"] == c["needed"] and c["demand_bytes"] == c["misses"] * eb
                                for c in (pre, dec))
                ok = ok and pre["steps"] == n_pre * n_layer and dec["steps"] == n_dec * n_layer and dec["waves"] == 0
                ok = ok and ((pre["waves"] > 0) == (built > 1)) and pre["waves"] <= n_pre * n_layer * built
                ok = ok and pre["waves_built"] == (n_pre * n_layer * built if built > 1 else 0) and dec["waves_built"] == 0
                ok = ok and (pre["wave_parked"] > 0) == (built > 1) and pre["load_failures"] + dec["load_failures"] == 0
                ok = ok and pre["prefetch_issued"] + pre["prefetch_skipped"] == 0
                # some layer's union did not fit its built waves at (slots - k) // 2 experts each
                ok = ok and (not grown or pre["needed"] > n_layer * built * ((slots - k) // 2))
                report(ok, f"{what}: counters consistent ({built} waves per layer built, {pre['waves']} of {pre['waves_built']} passes used over {n_layer} layers, "
                           f"{pre['wave_parked']} pairs parked; prefill {pre['needed']} needed, {pre['misses']} misses, "
                           f"{pre['hits']} hits, {pre['wait_seconds']:.3f} s wait; decode {dec['steps']} remaps, {dec['misses']} misses)")

    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "stats.json"
        if args.real:
            workload = [int(t) for t in np.load(args.tokens)]
            tokens = workload[:160]
            gpu = [(n, d) for n, d in r.devices() if r.lib.ggml_backend_dev_type(d) == r.lib.GGML_BACKEND_DEVICE_TYPE_GPU]
            side = None
            if sidefile_on and not args.waves_only:
                side = args.sidefile or args.real.with_name(args.real.name + ".experts")
                if not side.exists():
                    raise SystemExit(f"{side} does not exist: go run ./cmd/kolibri-sidefile write -gguf {args.real}")
                pair(args.real, side, f"{args.real.stem} side-file", 8)
            if not args.waves_only:
                check(args.real, *gpu[0], tokens, 128, [128, 64], 50, 6, stats, args.real.stem, side)
            check_waves(args.real, *gpu[0], tokens, 128, [128, 64], 50, 384, 6, stats, args.real.stem)
            check_waves(args.real, *gpu[0], tokens[:40], N_GROWN_REAL, [18], 50, 384, 6, stats, f"{args.real.stem} grown waves", grown=True)
            # the benchmark's ubatch: on a real workload about a third of the layers' unions outgrow 4 waves of 61
            check_waves(args.real, *gpu[0], workload[:N_BENCH_UBATCH + 32], N_BENCH_UBATCH, [128], 50, 384, 6, stats, f"{args.real.stem} benchmark ubatch")
        elif args.waves_only:
            model = generate_tiny(Path(tmp), args.seed, "router")
            cfg = json.loads((model / "config.json").read_text())
            rng = random.Random(args.seed)
            tokens = [rng.randrange(1000, 100000) for _ in range(N_PROMPT + N_DECODE)]
            for outtype in ["f32", "q8_0"]:
                gguf = convert(model, args.llama_cpp, outtype)
                for dev_name, dev in r.devices():
                    check_waves(gguf, dev_name, dev, tokens, N_PROMPT, WAVE_SLOTS, cfg["num_hidden_layers"],
                                cfg["num_experts"], cfg["num_experts_per_tok"], stats, outtype)
                    check_waves(gguf, dev_name, dev, tokens, N_GROWN, [18], cfg["num_hidden_layers"],
                                cfg["num_experts"], cfg["num_experts_per_tok"], stats, f"{outtype} grown waves", grown=True)
        else:
            model = generate_tiny(Path(tmp), args.seed, "router")
            cfg = json.loads((model / "config.json").read_text())
            n_layer, k = cfg["num_hidden_layers"], cfg["num_experts_per_tok"]
            rng = random.Random(args.seed)
            tokens = [rng.randrange(1000, 100000) for _ in range(N_PROMPT + N_DECODE)]
            sides = {}
            for outtype in ["f32", "q8_0"]:
                gguf = convert(model, args.llama_cpp, outtype)
                side = sides[outtype] = write_sidefile(gguf, f"{outtype} side-file") if sidefile_on else None
                for dev_name, dev in r.devices():
                    check(gguf, dev_name, dev, tokens, N_PROMPT, SLOTS, n_layer, k, stats, outtype, side)
                    check_waves(gguf, dev_name, dev, tokens, N_PROMPT, WAVE_SLOTS, n_layer, cfg["num_experts"], k, stats, outtype)
                    check_waves(gguf, dev_name, dev, tokens, N_GROWN, [18], n_layer, cfg["num_experts"], k, stats, f"{outtype} grown waves", grown=True)

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
                    ref = resident(gguf, dev, tokens, ub, chunks=[N_PROMPT] + [1] * N_DECODE)
                    stats.unlink(missing_ok=True)
                    got = r.logits(gguf, dev, tokens, N_PROMPT, chunks=[N_PROMPT] + [1] * N_DECODE, no_repack=True,
                                   expert_cache={"n_slots_layer": per_layer, "prefetch_k": 8, "stats_path": str(stats)})
                    s = json.loads(stats.read_text())["expert_cache"]
                    report(np.array_equal(got, ref) and (s["slots_min"], s["slots_max"]) == (min(per_layer), max(per_layer)),
                           f"{outtype} {dev_name}, slots per layer {per_layer}, ubatch {ub}: logits bit-identical to the resident run")
                    if side is not None:
                        stats.unlink(missing_ok=True)
                        got = r.logits(gguf, dev, tokens, N_PROMPT, chunks=[N_PROMPT] + [1] * N_DECODE, no_repack=True,
                                       expert_cache={"n_slots_layer": per_layer, "prefetch_k": 8, "stats_path": str(stats), "sidefile_path": str(side)})
                        report(np.array_equal(got, ref) and reads_ok(json.loads(stats.read_text()), side),
                               f"{outtype} {dev_name}, slots per layer {per_layer}, ubatch {ub}, side-file: logits bit-identical to the resident run, one pread per expert")

                # half the layers on the GPU; and the CPU with repacking on, which the slots never use
                gpu = [(n, d) for n, d in r.devices() if r.lib.ggml_backend_dev_type(d) == r.lib.GGML_BACKEND_DEVICE_TYPE_GPU]
                cpu = [(n, d) for n, d in r.devices() if r.lib.ggml_backend_dev_type(d) == r.lib.GGML_BACKEND_DEVICE_TYPE_CPU]
                cases = [(f"{gpu[0][0]} with {n_layer // 2} of {n_layer} layers", gpu[0][1], n_layer // 2, True)] if gpu else []
                cases.append((f"{cpu[0][0]} with repacking", cpu[0][1], -1, False))
                for what, dev, ngl, no_repack in cases:
                    ref = resident(gguf, dev, tokens, 2, chunks=[N_PROMPT] + [1] * N_DECODE, n_gpu_layers=ngl)
                    stats.unlink(missing_ok=True)
                    got = r.logits(gguf, dev, tokens, N_PROMPT, chunks=[N_PROMPT] + [1] * N_DECODE, no_repack=no_repack,
                                   n_gpu_layers=ngl, expert_cache={"n_slots": 16, "stats_path": str(stats)})
                    log = "".join(r.log)
                    placed = all(f"MiB for {name}" in log for name in ({gpu[0][0], cpu[0][0]} if ngl > 0 else {cpu[0][0]}))
                    report(np.array_equal(got, ref) and placed,
                           f"{outtype} {what}, 16 slots: logits bit-identical to the resident run without repacking"
                           + ("" if placed else f" (slots not on every layer's device: {[ln for ln in log.splitlines() if 'slots per layer' in ln]})"))
                    if side is not None:
                        stats.unlink(missing_ok=True)
                        got = r.logits(gguf, dev, tokens, N_PROMPT, chunks=[N_PROMPT] + [1] * N_DECODE, no_repack=no_repack,
                                       n_gpu_layers=ngl, expert_cache={"n_slots": 16, "stats_path": str(stats), "sidefile_path": str(side)})
                        report(np.array_equal(got, ref) and reads_ok(json.loads(stats.read_text()), side),
                               f"{outtype} {what}, 16 slots, side-file: logits bit-identical to the resident run without repacking, one pread per expert")

                if side is not None:
                    other = model.parent / f"{model.name}-f32.gguf" if outtype != "f32" else None
                    check_refusals(gguf, other, side, tokens, outtype)

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
            if sidefile_on:
                sf = sides["q8_0"]
                stats.unlink(missing_ok=True)
                on_side = subprocess.run(base + ["--expert-cache", "16", "--expert-cache-sidefile", str(sf), "--expert-prefetch", "8",
                                                 "--expert-cache-threads", "2", "--expert-cache-stats", str(stats)], capture_output=True, text=True)
                s = json.loads(stats.read_text()) if on_side.returncode == 0 and stats.exists() else None
                report(s is not None and on_side.stdout == plain.stdout and reads_ok(s, sf) and s["expert_cache"]["io_threads"] == 2,
                       f"llama-completion --expert-cache-sidefile: same text as without the cache, one pread per expert ({s['decode']['preads'] if s else on_side.stderr[-300:]} in decode)")
                gone = subprocess.run(base + ["--expert-cache", "16", "--expert-cache-sidefile", str(sf) + ".gone"], capture_output=True, text=True)
                report(gone.returncode != 0 and gone.stdout.strip() == "" and "no fallback" in gone.stderr,
                       f"llama-completion with a missing side-file exits {gone.returncode} without generating, no fallback to the GGUF")
                alone = subprocess.run(base + ["--expert-cache-sidefile", str(sf)], capture_output=True, text=True)
                report(alone.returncode != 0 and alone.stdout.strip() == "" and "without the cache" in alone.stderr,
                       f"llama-completion with a side-file and no --expert-cache exits {alone.returncode} without generating")
            report(ok and plain.stdout == cached.stdout and plain.stdout.strip() != "",
                   f"llama-completion --expert-cache-layers/--expert-prefetch/--expert-cache-buffered/--expert-cache-stats: as asked, same text as without the cache "
                   f"({plain.stdout.strip()[:60]!r})" + ("" if ok else f" rc {plain.returncode}/{cached.returncode}: {cached.stderr.strip()[-300:]}"))

    print(f"{'FAIL' if failed else 'PASS'}: {failed} failures")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
