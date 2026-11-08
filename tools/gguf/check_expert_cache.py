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

GPU remap (--expert-cache-gpu-remap): patch 0016, on top of the side-file (fork PR #16). Without it
these checks are skipped with a note, and --gpu-remap-only is an error. The
router's ids become slots on the GPU, with no CPU split: the logits must be bit-identical to the
UNSPLIT resident run on Metal (Stage E2's requirement; a same-split reference would not do), and
the counters equal the CPU remap's, since a gate defers but never changes what the remaps do. With
fusion on the CPU remap can route differently, because its splits round differently; so the checks
run once more under GGML_METAL_FUSION_DISABLE=1, where it routes alike and the counters must match
exactly. Split control (M61): 2 scheduler splits per decoded token, as resident; and a pass that
repeats earlier routing stops nowhere and costs what a resident token does (2 command buffers, 3
commits). Metal commits: one per command buffer, plus 1 for each output copy. Requesting it where
it cannot work fails to load: on the CPU, with some layers off Metal, with prefetch, and without a
cache, where it would otherwise load fully resident. What it cannot run stops with its own diagnosis,
in a child process, since both abort: a cb_eval that stops between a layer's matmuls, and
llama_set_warmup, whose router ids are wider than the gate's slot ids. A cb_eval that stops outside
the matmuls still runs, and a decode the abort callback ended does not stop the next one.

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

LoRA (patch 0011): an adapter on a stacked expert tensor is refused at load with the cache, for gate, up and
down, since the matmuls read the slots and the delta would be left out. The same adapter loads without the
cache and changes the text, and an adapter on the router's input weight, which the cache does not hold, gives
the resident run's text with the cache.

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
    check_expert_cache.py --llama-cpp third_party/llama.cpp-e2 --gpu-remap-only   # the GPU remap checks alone
    check_expert_cache.py --llama-cpp third_party/llama.cpp-e2e3 --stack-only     # the side-file and the GPU remap together
    check_expert_cache.py --llama-cpp third_party/llama.cpp-skip --skip-only      # the lossy skip alone
    check_expert_cache.py --llama-cpp third_party/llama.cpp-skip --skip-only --real ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf

The stack (--expert-cache-sidefile with --expert-cache-gpu-remap): patches 0014 and 0016 together, the
build that fork branch feat/kolibri1-e2-e3 benchmarked. On a build with both
options the logits must be bit-identical to the UNSPLIT resident run and to the GPU remap on the GGUF
path, with its counters, one pread per expert and 2 scheduler splits per decoded token. --real checks
the same on the real weight.

Skip (--expert-cache-skip): patch 0018, lossy and opt-in (Stage E5). Against a checkout without it the
skip checks are skipped with a note, and --skip-only is an error. Per token the remap ranks the ids by
router weight; an id at rank 4 or below whose weight is under skip_weight and whose expert is a true
miss (neither in a slot nor loading), and that no other id of the ubatch needs, is not read: it reads a
ready slot (its token's rank-1 expert's, or rank 2's with the test hook skip_map_rank 2) and its weight
is masked to 0 in the graph. The router fixture has 6 experts per token, so ranks 4 to 6 exist. On every
device, F32 and Q8_0, at 6 slots (ubatch 1), 16 (ubatch 2) and 384 (a slot for every expert, so only
first touches miss, and those skip too):

- skip_weight 0, and 1e-30, which runs the mask path but lies below every sigmoid weight: logits
  bit-identical to the lossless run with the same slots (a mask of 1.0 is exact), nothing skipped, the
  same counters;
- skip_weight 1, every eligible miss skipped: decode skips, the logits are finite and differ from the
  lossless run (max |diff| as info), and the counters add up: hits + hits_loading + misses = needed; a
  decode remap's 6 ids are distinct, so needed + skipped = 6 per remap and skipped = skipped_ids; skipped
  ids never exceed ranks 4 to 6 of the remaps' tokens; with 384 slots nothing is evicted;
- metamorphic (control M108, the mask left out): skip_map_rank 2 gives logits and counters
  bit-identical to rank 1, since a weight of 0 cancels whichever expert the slot holds;
- the rank rule at its edge, with expert_used_count overridden: at 3 experts per token nothing skips
  (bit-identical to lossless), at 4 the rank-4 id does, at most once per token and layer;
- refusals at load, each by name: with the GPU remap (its gate reads no weights), without a cache, a
  negative weight, map rank 3; and llama-completion: --expert-cache-skip 0 gives the lossless text, 1
  writes skip_weight and the skipped counts to the stats, and what the library refuses exits non-zero.

--real with skip support runs the weight 0, 1e-30, 1 and map-rank-2 runs on the real weight on the GPU,
128 prompt tokens and 32 decoded at 64 slots (alone with --skip-only).
"""

import argparse
import json
import os
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
# the skip: the fewest legal slots (one token's 6, ubatch 1, all decode), a few (ubatch 2, so prefill
# remaps of 2 tokens), and a slot for every expert (nothing evicted, only first touches miss)
SKIP_SLOTS = [6, 16, 384]
SKIP_SLOTS_REAL = 64  # --real: 160 tokens at ubatch 10, five runs

# What the GPU remap must refuse aborts the process, so check_gpu_remap_refusals runs it here, in a
# child: python -c GPU_REMAP_CHILD <llama.cpp> <gguf> <what> <slots>
GPU_REMAP_CHILD = """
import sys
from pathlib import Path
from check_model import Runner

llama_cpp, gguf, what, n_slots = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
r = Runner(llama_cpp)
dev = next(d for _, d in r.devices() if r.lib.ggml_backend_dev_type(d) == r.lib.GGML_BACKEND_DEVICE_TYPE_GPU)
tokens = list(range(1000, 1040))
if what.startswith("split_after="):
    r.logits(gguf, dev, tokens, 10, no_repack=True, split_after=(what.split("=")[1],), expert_cache={"n_slots": n_slots, "gpu_remap": True})
    print("decoded")
    sys.exit(0)
devs = r.ffi.new("ggml_backend_dev_t[]", [dev, r.ffi.NULL])
ec = r.ffi.new("struct llama_expert_cache_params *", r.lib.llama_expert_cache_default_params())
ec.n_slots, ec.gpu_remap = n_slots, True
model = r.lib.llama_model_load_from_file(gguf.encode(), r.ll.model_default_params(devices=devs, n_gpu_layers=-1, expert_cache=ec, use_extra_bufts=False))
ctx = r.lib.llama_init_from_model(model, r.ll.context_default_params(n_ctx=256, n_batch=len(tokens), n_ubatch=10, op_offload=False))
batch = r.lib.llama_batch_init(len(tokens), 0, 1)
for i, t in enumerate(tokens):
    batch.token[i], batch.pos[i], batch.n_seq_id[i], batch.seq_id[i][0], batch.logits[i] = t, i, 1, 0, True
batch.n_tokens = len(tokens)
if what == "warmup":
    r.lib.llama_set_warmup(ctx, True)
    print("decoded", r.lib.llama_decode(ctx, batch))
elif what == "abort":
    # end one decode at each of the callback's calls in turn, until a decode gets through. Only the
    # CPU backend hears it through llama_set_abort_callback, so these end before their ubatch's gates
    calls, stop_at, rc = [0], [0], 2

    @r.ffi.callback("bool(void *)")
    def stop(user_data):
        calls[0] += 1
        return calls[0] == stop_at[0]

    r.lib.llama_set_abort_callback(ctx, stop, r.ffi.NULL)
    while rc == 2 and stop_at[0] < 20000:
        calls[0] = 0
        stop_at[0] += 1
        r.lib.llama_memory_clear(r.lib.llama_get_memory(ctx), True)  # the ubatches before the end stay
        rc = r.lib.llama_decode(ctx, batch)
    print("aborted", stop_at[0] - 1, "then decoded", rc)
r.lib.llama_batch_free(batch)
r.lib.llama_free(ctx)
r.lib.llama_model_free(model)
"""


def write_lora(llama_cpp: Path, path: Path, base: Path, targets: list[str], rank: int = 2, seed: int = 1) -> None:
    """A rank-limited LoRA adapter of random weights for these tensors of the GGUF base."""
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    import gguf
    import numpy as np

    dims = {t.name: [int(d) for d in t.shape] for t in gguf.GGUFReader(base).tensors}
    w = gguf.GGUFWriter(path, gguf.GGUFReader(base).fields["general.architecture"].contents())
    w.add_type("adapter")
    w.add_string("adapter.type", "lora")
    rng = np.random.default_rng(seed)
    for name in targets:
        d = dims[name]  # ggml order; numpy's is the reverse
        a = rng.standard_normal(tuple(([d[0], rank] + d[2:])[::-1]), dtype=np.float32) * 0.5
        b = rng.standard_normal(tuple(([rank, d[1]] + d[2:])[::-1]), dtype=np.float32) * 0.5
        w.add_tensor(f"{name}.lora_a", a)
        w.add_tensor(f"{name}.lora_b", b)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


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


def has_gpu_remap(r: Runner, llama_cpp: Path) -> bool:
    """Whether the checkout has --expert-cache-gpu-remap, by the same rule as has_waves: header and
    library must agree, or gpu_remap=true would land in padding and silently compare the CPU remap
    with itself."""
    header = any(name == "gpu_remap" for name, _ in r.ffi.typeof("struct llama_expert_cache_params").fields)
    path = llama_cpp / "build" / "bin" / ("libllama.dylib" if sys.platform == "darwin" else "libllama.so")
    library = b"gpu_stops" in path.read_bytes()
    if header != library:
        raise SystemExit(f"{llama_cpp}: include/llama.h {'has' if header else 'has no'} expert cache field gpu_remap, "
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


WAIT_MODES = {"sleep": 0, "spin": 1, "hybrid": 2, "qos": 3}  # enum llama_expert_cache_wait


def wait_mode(value: str) -> str:
    """--wait-mode: sleep, spin, hybrid:US or qos, as llama-completion's --expert-cache-wait takes it."""
    name, _, us = value.partition(":")
    if name not in WAIT_MODES or (name == "hybrid") != bool(us) or (us and not (us.isdigit() and int(us) > 0)):
        raise argparse.ArgumentTypeError(f"unknown wait mode {value!r}: want sleep, spin, hybrid:US or qos")
    return value


def wait_fields(value: str) -> dict:
    """The llama_expert_cache_params fields of a --wait-mode."""
    name, _, us = value.partition(":")
    return {"wait_mode": WAIT_MODES[name], "wait_spin_us": int(us or 0)}


def has_wait(r: Runner, llama_cpp: Path) -> bool:
    """Whether the checkout has --expert-cache-wait, by the same rule as has_waves: the header and
    the built library must agree."""
    header = any(name == "wait_mode" for name, _ in r.ffi.typeof("struct llama_expert_cache_params").fields)
    path = llama_cpp / "build" / "bin" / ("libllama.dylib" if sys.platform == "darwin" else "libllama.so")
    library = b'"wait": ' in path.read_bytes()
    if header != library:
        raise SystemExit(f"{llama_cpp}: include/llama.h {'has' if header else 'has no'} expert cache field wait_mode, "
                         f"but {path.name} was built {'without' if header else 'with'} it: rebuild the checkout")
    return header


def has_skip(r: Runner, llama_cpp: Path) -> bool:
    """Whether the checkout has --expert-cache-skip, by the same rule as has_waves: the header and
    the built library must agree. The library's key is the stats' "skipped", with its quotes, which
    the older "prefetch_skipped" does not contain."""
    header = any(name == "skip_weight" for name, _ in r.ffi.typeof("struct llama_expert_cache_params").fields)
    path = llama_cpp / "build" / "bin" / ("libllama.dylib" if sys.platform == "darwin" else "libllama.so")
    library = b'"skipped"' in path.read_bytes()
    if header != library:
        raise SystemExit(f"{llama_cpp}: include/llama.h {'has' if header else 'has no'} expert cache field skip_weight, "
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
    ap.add_argument("--gpu-remap-only", action="store_true", help="only the GPU remap checks")
    ap.add_argument("--stack-only", action="store_true", help="only the checks of the side-file and the GPU remap together")
    ap.add_argument("--skip-only", action="store_true", help="only the checks of the lossy skip (--expert-cache-skip)")
    ap.add_argument("--wait-mode", type=wait_mode, metavar="MODE",
                    help="how the cache runs wait for their loads: sleep, spin, hybrid:US or qos (--expert-cache-wait; "
                         "the resident reference runs never get it; default: the library's, sleep)")
    args = ap.parse_args()
    import numpy as np

    r = Runner(args.llama_cpp)
    waves_on = has_waves(r, args.llama_cpp)
    if args.waves_only and not waves_on:
        ap.error(f"--waves-only: {args.llama_cpp} has no expert cache waves; they need a build of fork branch feat/kolibri1-waves")
    if not waves_on:
        print(f"note: {args.llama_cpp} has no expert cache waves (fork branch feat/kolibri1-waves): skipping check_waves", flush=True)
    gpu_remap_on = has_gpu_remap(r, args.llama_cpp)
    if args.gpu_remap_only and not gpu_remap_on:
        ap.error(f"--gpu-remap-only: {args.llama_cpp} has no GPU remap; it needs a build of fork branch feat/kolibri1-gpu-remap")
    if not gpu_remap_on:
        print(f"note: {args.llama_cpp} has no GPU remap (fork branch feat/kolibri1-gpu-remap): skipping check_gpu_remap", flush=True)
    sidefile_on = has_sidefile(r, args.llama_cpp)
    if not sidefile_on:
        print(f"note: {args.llama_cpp} has no expert cache side-file (patch 0014): skipping the side-file checks", flush=True)
    if args.stack_only and not (gpu_remap_on and sidefile_on):
        ap.error(f"--stack-only: {args.llama_cpp} lacks the GPU remap or the side-file; it needs a build of fork branch feat/kolibri1-e2-e3")
    skip_on = has_skip(r, args.llama_cpp)
    if args.skip_only and not skip_on:
        ap.error(f"--skip-only: {args.llama_cpp} has no --expert-cache-skip; it needs a build with patch 0018")
    if not skip_on:
        print(f"note: {args.llama_cpp} has no expert cache skip (patch 0018): skipping check_skip", flush=True)
    if args.wait_mode is not None:
        if not has_wait(r, args.llama_cpp):
            ap.error(f"--wait-mode: {args.llama_cpp} has no --expert-cache-wait; it needs a build of fork branch feat/kolibri1-load-spin")
        # every run with slots waits as asked; the resident runs, with no slots, stay as they are
        logits = r.logits

        def logits_with_wait(*a, expert_cache: dict | None = None, **kw):
            if expert_cache is not None and ("n_slots" in expert_cache or "n_slots_layer" in expert_cache):
                expert_cache = {**expert_cache, **wait_fields(args.wait_mode)}
            return logits(*a, expert_cache=expert_cache, **kw)

        r.logits = logits_with_wait
        print(f"note: the cache runs wait with --expert-cache-wait {args.wait_mode}", flush=True)
    fusion_disabled = bool(os.environ.get("GGML_METAL_FUSION_DISABLE"))
    failed = 0

    def cli(cmd: list[str]) -> list[str]:
        """A llama-completion command line, with --expert-cache-wait when it runs the cache."""
        if args.wait_mode is not None and ("--expert-cache" in cmd or "--expert-cache-layers" in cmd):
            return cmd + ["--expert-cache-wait", args.wait_mode]
        return cmd

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
                # the wait mode as asked, the library's default without --wait-mode
                ok = ok and s["expert_cache"].get("wait", "sleep") == (args.wait_mode or "sleep")
                report(ok, f"{what}: counters consistent (wait {s['expert_cache'].get('wait', 'sleep')}; prefill {pre['steps']} remaps, {pre['misses']} misses; "
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

    def check_gpu_remap(gguf: Path, dev_name: str, dev, tokens: list[int], n_prompt: int, slots_list: list[int],
                        n_layer: int, k: int, stats: Path, label: str) -> None:
        """Stage E2: bit identity to the UNSPLIT resident run, with the CPU remap's counters."""
        if not gpu_remap_on:
            return
        chunks = [n_prompt] + [1] * (len(tokens) - n_prompt)
        same_fields = ("steps", "needed", "hits", "hits_loading", "misses", "evictions", "demand_bytes", "load_failures")
        if not on_gpu(dev):
            # it cannot work here, and a benchmark must not silently compare the CPU remap with itself
            try:
                r.logits(gguf, dev, tokens[:n_prompt], k, no_repack=True, expert_cache={"n_slots": 64, "gpu_remap": True})
                report(False, f"{label} {dev_name}: GPU remap loaded on the CPU")
            except RuntimeError:
                report("needs every expert layer on Metal" in "".join(r.log), f"{label} {dev_name}: GPU remap fails to load, and says why")
            return
        for slots in slots_list:
            ub = min(n_prompt, max(1, slots // k))
            sizes = [ub] * (n_prompt // ub) + ([n_prompt % ub] if n_prompt % ub else [])
            n_dec = len(tokens) - n_prompt + sum(n == 1 for n in sizes)
            ref = r.logits(gguf, dev, tokens, ub, chunks=chunks, no_repack=True)
            stats.unlink(missing_ok=True)
            cpu_arm = r.logits(gguf, dev, tokens, n_prompt, chunks=chunks, no_repack=True, expert_cache={"n_slots": slots, "stats_path": str(stats)})
            c = json.loads(stats.read_text())
            for buffered in (False, True):
                what = f"{label} {dev_name}, GPU remap, {slots} slots, ubatch {ub}" + (", buffered" if buffered else "")
                stats.unlink(missing_ok=True)
                got = r.logits(gguf, dev, tokens, n_prompt, chunks=chunks, no_repack=True,
                               expert_cache={"n_slots": slots, "stats_path": str(stats), "buffered": buffered, "gpu_remap": True})
                same = np.array_equal(got, ref)
                diff = "" if same else f", max |diff| {float(np.abs(got - ref).max()):.3g}"
                report(same, f"{what}: logits of {len(tokens)} positions bit-identical to the UNSPLIT resident run{diff}")
                print(f"info {what}: max |diff| to the CPU remap {float(np.abs(got - cpu_arm).max()):.3g}")
                s = json.loads(stats.read_text())
                pre, dec = s["prefill"], s["decode"]
                diffs = [f"{ph} {key} {x.get(key)} != {y.get(key)}" for ph, x, y in (("prefill", pre, c["prefill"]), ("decode", dec, c["decode"]))
                         for key in same_fields if x.get(key) != y.get(key)]
                # The CPU remap's splits round differently, and that can route an expert differently
                # (E0.1), so its counters can only be held to these where its logits equal the
                # unsplit run's; GGML_METAL_FUSION_DISABLE=1 removes the difference, and main runs
                # these checks once more with it, which makes the comparison exact.
                routes_alike = np.array_equal(cpu_arm, ref)
                if fusion_disabled:
                    report(routes_alike, f"{what}: without fusion the CPU remap equals the unsplit resident run")
                strict = not diffs if routes_alike else all(pre.get(key) == c["prefill"].get(key) for key in ("steps", "needed")) \
                    and all(dec.get(key) == c["decode"].get(key) for key in ("steps", "needed", "load_failures"))
                report(strict and s["expert_cache"]["gpu_remap"] is True and c["expert_cache"]["gpu_remap"] is False,
                       f"{what}: the stats record the mode, and the remaps' counters equal the CPU remap's "
                       f"{'exactly' if routes_alike else 'as far as its different rounding lets them'} "
                       f"(decode {dec['hits']} hits, {dec['misses']} misses, {dec['evictions']} evictions)" + (f": {diffs}" if diffs else ""))
                # M61: no CPU split: 2 splits per decoded token, as resident
                report(dec["tokens"] == n_dec and dec["splits"] == 2 * n_dec,
                       f"{what}: {dec['splits'] / max(dec['tokens'], 1):.2f} scheduler splits per decoded token, want 2 (CPU remap: "
                       f"{c['decode']['splits'] / max(c['decode']['tokens'], 1):.2f})")
                # one commit per command buffer, 1 per output copy; a layer that hits adds none
                runs, stops = dec["gpu_runs"], dec["gpu_stops"]
                grown = 2 + len(bin(n_layer)) - 2
                ok = dec["metal_commits"] == runs + n_dec and stops <= dec["misses"] and (dec["misses"] == 0 or stops > 0) \
                    and 2 * n_dec <= runs <= 2 * n_dec + stops * grown
                report(ok, f"{what}: {runs / max(n_dec, 1):.2f} command buffers and {stops / max(n_dec, 1):.2f} stops per decoded token, "
                           f"{dec['metal_commits'] / max(n_dec, 1):.2f} commits (CPU remap {c['decode']['metal_commits'] / max(n_dec, 1):.2f}), "
                           f"{1e3 * dec['wall_seconds'] / max(n_dec, 1):.3f} ms per token (not a result)")

            # as a server decodes: the last layer of a prefill ubatch sees only the output rows
            outs = list(range(n_prompt - 1, len(tokens)))
            want = r.logits(gguf, dev, tokens, ub, chunks=chunks, no_repack=True, outputs=outs)
            got = r.logits(gguf, dev, tokens, n_prompt, chunks=chunks, no_repack=True, outputs=outs, expert_cache={"n_slots": slots, "gpu_remap": True})
            report(np.array_equal(got, want), f"{label} {dev_name}, GPU remap, {slots} slots, ubatch {ub}, only {len(outs)} output positions: logits bit-identical to the UNSPLIT resident run")

        # what cannot work fails to load, instead of running the CPU remap
        try:
            r.logits(gguf, dev, tokens[:n_prompt], k, no_repack=True, expert_cache={"n_slots": 64, "gpu_remap": True, "prefetch_k": 8})
            report(False, f"{label} {dev_name}: GPU remap with prefetch loaded")
        except RuntimeError:
            report("does not work with --expert-prefetch" in "".join(r.log), f"{label} {dev_name}: GPU remap with prefetch fails to load, and says why")
        try:
            r.logits(gguf, dev, tokens[:n_prompt], k, no_repack=True, n_gpu_layers=n_layer // 2, expert_cache={"n_slots": 64, "gpu_remap": True})
            report(False, f"{label} {dev_name}: GPU remap with {n_layer // 2} of {n_layer} layers on Metal loaded")
        except RuntimeError:
            report("needs every expert layer on Metal" in "".join(r.log), f"{label} {dev_name}: GPU remap with some layers off Metal fails to load, and says why")

    def check_stack(gguf: Path, dev_name: str, dev, side: Path | None, tokens: list[int], n_prompt: int, slots_list: list[int],
                    n_layer: int, k: int, stats: Path, label: str) -> None:
        """The side-file and the GPU remap together: where a build has both, a miss reads the side-file
        and a hit takes no CPU, and neither changes a logit or what the other one counts."""
        if not (gpu_remap_on and side):
            return
        if not on_gpu(dev):
            try:
                r.logits(gguf, dev, tokens[:n_prompt], k, no_repack=True, expert_cache={"n_slots": 64, "gpu_remap": True, "sidefile_path": str(side)})
                report(False, f"{label} {dev_name}: the stack loaded on the CPU")
            except RuntimeError:
                report("needs every expert layer on Metal" in "".join(r.log), f"{label} {dev_name}: the stack fails to load, and says why")
            return
        chunks = [n_prompt] + [1] * (len(tokens) - n_prompt)
        same_fields = ("steps", "needed", "hits", "hits_loading", "misses", "evictions", "demand_bytes", "load_failures", "gpu_runs", "gpu_stops")
        for slots in slots_list:
            ub = min(n_prompt, max(1, slots // k))
            sizes = [ub] * (n_prompt // ub) + ([n_prompt % ub] if n_prompt % ub else [])
            n_dec = len(tokens) - n_prompt + sum(n == 1 for n in sizes)
            ref = r.logits(gguf, dev, tokens, ub, chunks=chunks, no_repack=True)
            stats.unlink(missing_ok=True)
            on_gguf = r.logits(gguf, dev, tokens, n_prompt, chunks=chunks, no_repack=True, expert_cache={"n_slots": slots, "stats_path": str(stats), "gpu_remap": True})
            g = json.loads(stats.read_text())
            report(reads_ok(g, None) and g["expert_cache"]["gpu_remap"] is True, f"{label} {dev_name}, stack, {slots} slots: the GPU remap on the GGUF path takes 3 preads per expert")
            for buffered in (False, True):
                what = f"{label} {dev_name}, stack, {slots} slots, ubatch {ub}" + (", buffered" if buffered else "")
                stats.unlink(missing_ok=True)
                got = r.logits(gguf, dev, tokens, n_prompt, chunks=chunks, no_repack=True,
                               expert_cache={"n_slots": slots, "stats_path": str(stats), "buffered": buffered, "gpu_remap": True, "sidefile_path": str(side)})
                same = np.array_equal(got, ref)
                diff = "" if same else f", max |diff| {float(np.abs(got - ref).max()):.3g}"
                report(same, f"{what}: logits of {len(tokens)} positions bit-identical to the UNSPLIT resident run{diff}")
                report(np.array_equal(got, on_gguf), f"{what}: logits bit-identical to the GPU remap on the GGUF path")
                s = json.loads(stats.read_text())
                pre, dec = s["prefill"], s["decode"]
                report(reads_ok(s, side) and s["expert_cache"]["gpu_remap"] is True,
                       f"{what}: the stats name the side-file and the GPU remap; {pre['preads']} and {dec['preads']} preads for {pre['loads']} and {dec['loads']} experts in prefill and decode")
                diffs = [f"{ph} {key} {x.get(key)} != {y.get(key)}" for ph, x, y in (("prefill", pre, g["prefill"]), ("decode", dec, g["decode"]))
                         for key in same_fields if x.get(key) != y.get(key)]
                report(not diffs, f"{what}: the counters equal the GPU remap's on the GGUF path (decode {dec['hits']} hits, {dec['misses']} misses, "
                                  f"{dec.get('gpu_stops')} stops)" + (f": {diffs}" if diffs else ""))
                report(dec["tokens"] == n_dec and dec["splits"] == 2 * n_dec,
                       f"{what}: {dec['splits'] / max(dec['tokens'], 1):.2f} scheduler splits per decoded token, want 2")
            # as a server decodes: the last layer of a prefill ubatch sees only the output rows
            outs = list(range(n_prompt - 1, len(tokens)))
            want = r.logits(gguf, dev, tokens, ub, chunks=chunks, no_repack=True, outputs=outs)
            got = r.logits(gguf, dev, tokens, n_prompt, chunks=chunks, no_repack=True, outputs=outs,
                           expert_cache={"n_slots": slots, "gpu_remap": True, "sidefile_path": str(side)})
            report(np.array_equal(got, want), f"{label} {dev_name}, stack, {slots} slots, ubatch {ub}, only {len(outs)} output positions: logits bit-identical to the UNSPLIT resident run")
        # what cannot work with the GPU remap still fails to load with a side-file
        try:
            r.logits(gguf, dev, tokens[:n_prompt], k, no_repack=True, expert_cache={"n_slots": 64, "gpu_remap": True, "prefetch_k": 8, "sidefile_path": str(side)})
            report(False, f"{label} {dev_name}: the stack with prefetch loaded")
        except RuntimeError:
            report("does not work with --expert-prefetch" in "".join(r.log), f"{label} {dev_name}: the stack with prefetch fails to load, and says why")

    def check_gpu_remap_refusals(gguf: Path, dev_name: str, dev, stats: Path, label: str) -> None:
        """What the mode cannot load or run says so. Tiny checkpoints only: the refused run without a
        cache would load the model fully resident if the refusal failed."""
        if not gpu_remap_on:
            return
        what = f"{label} {dev_name}: GPU remap without a cache"
        try:
            r.logits(gguf, dev, list(range(1000, 1010)), 10, no_repack=True, expert_cache={"gpu_remap": True})
            report(False, f"{what} loaded")
        except RuntimeError:
            report("needs --expert-cache or --expert-cache-layers" in "".join(r.log), f"{what} fails to load, and says why")
        if not on_gpu(dev):
            return

        # the command line reaches that refusal, also when only the stats ask for params
        tool = [str(args.llama_cpp / "build" / "bin" / "llama-completion"), "-m", str(gguf), "-p", "Hello", "-n", "4", "-no-cnv", "-c", "256", "--no-warmup", "--expert-cache-gpu-remap"]
        for extra, how in (([], "alone"), (["--expert-cache-stats", str(stats)], "with --expert-cache-stats alone")):
            run = subprocess.run(tool + extra, capture_output=True, text=True)
            report(run.returncode != 0 and "needs --expert-cache or --expert-cache-layers" in run.stderr and run.stdout.strip() == "",
                   f"{label}: llama-completion --expert-cache-gpu-remap {how} exits {run.returncode} and says why")

        def child(child_what: str, n_slots: int) -> subprocess.CompletedProcess:
            env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent)}
            return subprocess.run([sys.executable, "-c", GPU_REMAP_CHILD, str(args.llama_cpp), str(gguf), child_what, str(n_slots)],
                                  capture_output=True, text=True, env=env)

        for node in ("ffn_moe_up", "ffn_moe_gate"):
            run = child(f"split_after={node}", 64)
            report(run.returncode != 0 and "reached its Metal gate again" in run.stderr and "went through the Metal gate" not in run.stderr,
                   f"{label} {dev_name}: a cb_eval stopping after {node}, inside the layer, exits {run.returncode} and says why")
        # the cut M69 makes, and one after the layer's last matmul: no layer is cut in two
        for node in ("ffn_moe_topk", "ffn_moe_down"):
            run = child(f"split_after={node}", 64)
            report(run.returncode == 0 and run.stdout.strip().endswith("decoded"), f"{label} {dev_name}: a cb_eval stopping after {node}, outside the matmuls, runs, rc {run.returncode}")
        # with 384 slots the widened ids fit the cache, and only the width check can stop them
        for n_slots in (64, 384):
            run = child("warmup", n_slots)
            report(run.returncode != 0 and "wider than the expert gate's slot ids" in run.stderr,
                   f"{label} {dev_name}, {n_slots} slots: llama_set_warmup's wider ids exit {run.returncode} and say why")
        run = child("abort", 64)
        tail = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else ""
        report(run.returncode == 0 and tail.startswith("aborted ") and tail.endswith(" then decoded 0") and tail != "aborted 0 then decoded 0",
               f"{label} {dev_name}: decodes the abort callback ended, one at each of its calls, leave the next one free to run: {tail!r}, rc {run.returncode}")

    def check_gpu_remap_hits(gguf: Path, dev_name: str, dev, tokens: list[int], k: int, stats: Path, label: str, n_slots: int = 384) -> None:
        """What a layer that hits costs (control M61): the same one-token decodes again after clearing
        the KV cache route alike, so with every expert in a slot the last pass stops nowhere, and a
        token takes the 2 scheduler splits, 2 command buffers and 3 commits of a resident token."""
        if not gpu_remap_on or not on_gpu(dev):
            return

        def passes(n: int) -> dict:
            stats.unlink(missing_ok=True)
            devs = r.ffi.new("ggml_backend_dev_t[]", [dev, r.ffi.NULL])
            ec = r.ffi.new("struct llama_expert_cache_params *", r.lib.llama_expert_cache_default_params())
            ec.n_slots, ec.gpu_remap = n_slots, True
            for key, val in (wait_fields(args.wait_mode) if args.wait_mode is not None else {}).items():
                setattr(ec, key, val)
            path = r.ffi.new("char[]", str(stats).encode())
            ec.stats_path = path
            model = r.lib.llama_model_load_from_file(str(gguf).encode(), r.ll.model_default_params(devices=devs, n_gpu_layers=-1, expert_cache=ec, use_extra_bufts=False))
            ctx = r.lib.llama_init_from_model(model, r.ll.context_default_params(n_ctx=256, n_batch=1, n_ubatch=1, op_offload=False))
            batch = r.lib.llama_batch_init(1, 0, 1)
            for _ in range(n):
                r.lib.llama_memory_clear(r.lib.llama_get_memory(ctx), True)
                for i, t in enumerate(tokens):
                    batch.token[0], batch.pos[0], batch.n_seq_id[0], batch.seq_id[0][0], batch.logits[0] = t, i, 1, 0, True
                    batch.n_tokens = 1
                    if r.lib.llama_decode(ctx, batch) != 0:
                        raise RuntimeError("llama_decode failed: " + r.errors())
            r.lib.llama_batch_free(batch)
            r.lib.llama_free(ctx)
            r.lib.llama_model_free(model)
            return json.loads(stats.read_text())["decode"]

        a, b = passes(3), passes(4)
        n = len(tokens)
        d = {key: b[key] - a[key] for key in ("tokens", "splits", "metal_commits", "gpu_runs", "gpu_stops", "hits", "misses", "needed")}
        n_layer = a["needed"] // (3 * n * k)
        report(d == {"tokens": n, "splits": 2 * n, "metal_commits": 3 * n, "gpu_runs": 2 * n, "gpu_stops": 0, "hits": n * n_layer * k, "misses": 0, "needed": n * n_layer * k},
               f"{label} {dev_name}, GPU remap, all hits: a further pass of {n} tokens costs {d}, want 2 splits, 2 command buffers and 3 commits per token and no stop")

    def check_gpu_remap_exact() -> None:
        """The counters exactly: without fusion the CPU remap routes as the unsplit run does."""
        if not gpu_remap_on or fusion_disabled:
            return
        sub = subprocess.run([sys.executable, __file__, "--llama-cpp", str(args.llama_cpp), "--seed", str(args.seed), "--gpu-remap-only"]
                             + (["--wait-mode", args.wait_mode] if args.wait_mode is not None else []),
                             capture_output=True, text=True, env={**os.environ, "GGML_METAL_FUSION_DISABLE": "1"})
        n_pass = sub.stdout.count("\nPASS ")
        report(sub.returncode == 0 and n_pass > 0, f"GPU remap with GGML_METAL_FUSION_DISABLE=1: {n_pass} checks passed, rc {sub.returncode}"
               + ("" if sub.returncode == 0 else ": " + " | ".join(ln for ln in sub.stdout.splitlines() if ln.startswith("FAIL"))[:800]))

    def check_gpu_remap_cli(base: list[str], gguf: Path, stats: Path, want: str) -> None:
        """The option on the command line: it runs and says so in the stats, or the tool fails."""
        stats.unlink(missing_ok=True)
        run = subprocess.run(cli(base + ["--expert-cache", "64", "--expert-cache-gpu-remap", "--expert-cache-stats", str(stats)]),
                             capture_output=True, text=True)
        try:
            mode = json.loads(stats.read_text())["expert_cache"]["gpu_remap"]
        except (OSError, ValueError, KeyError) as e:
            mode = str(e)
        report(run.returncode == 0 and mode is True and run.stdout == want,
               f"llama-completion --expert-cache-gpu-remap: the stats say gpu_remap {mode}, same text as resident -ub 2 ({run.stdout.strip()[:40]!r}), rc {run.returncode}")
        # as a server starts: with its warmup decode
        warm = subprocess.run(cli([a for a in base if a != "--no-warmup"] + ["--expert-cache", "64", "--expert-cache-gpu-remap"]), capture_output=True, text=True)
        report(warm.returncode == 0 and warm.stdout == want, f"llama-completion --expert-cache-gpu-remap with the warmup decode: same text, rc {warm.returncode}")
        for extra, msg, what in ((["-ngl", "0"], "needs every expert layer on Metal", "-ngl 0"),
                                 (["--expert-prefetch", "8"], "does not work with --expert-prefetch", "--expert-prefetch 8")):
            run = subprocess.run(cli(base + ["--expert-cache", "64", "--expert-cache-gpu-remap"] + extra), capture_output=True, text=True)
            report(run.returncode != 0 and msg in run.stderr, f"llama-completion --expert-cache-gpu-remap with {what} exits {run.returncode} and says why")

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

    def check_skip(gguf: Path, dev_name: str, dev, tokens: list[int], n_prompt: int, slots_list: list[int],
                   n_layer: int, n_expert: int, k: int, stats: Path, label: str, ranks: bool = True) -> None:
        """The lossy skip (--expert-cache-skip). The rule ranks a token's ids by router weight and
        never skips ranks 1 to 3, so it needs k >= 4 to do anything: the router fixture has k = 6, and
        the checks refuse a fixture below 4 instead of passing vacuously. ranks adds the runs with
        expert_used_count overridden to 3 and 4 (tiny only)."""
        if not skip_on:
            return
        if k < 4:
            report(False, f"{label} {dev_name}: the skip needs at least 4 experts per token, the fixture has {k}")
            return
        chunks = [n_prompt] + [1] * (len(tokens) - n_prompt)
        same_fields = ("steps", "needed", "hits", "hits_loading", "misses", "skipped", "skipped_ids", "evictions", "demand_bytes", "load_failures")

        def run(slots: int, ub: int, kk: int = k, **ec):
            stats.unlink(missing_ok=True)
            got = r.logits(gguf, dev, tokens, n_prompt, chunks=chunks, no_repack=True,
                           overrides={"kolibri1.expert_used_count": kk} if kk != k else None,
                           expert_cache={"n_slots": slots, "stats_path": str(stats), **ec})
            return got, json.loads(stats.read_text())

        def phases(s: dict) -> str:
            return "; ".join(f"{ph} {s[ph]['needed']} needed, {s[ph]['misses']} misses, {s[ph].get('skipped', 0)} skipped "
                             f"({s[ph].get('skipped_ids', 0)} ids)" for ph in ("prefill", "decode"))

        def counted(s: dict, ub: int, n_pre: int, n_dec: int, slots: int, kk: int) -> list[str]:
            """What the counters must say with the skip on or off; the reasons they do not."""
            pre, dec = s["prefill"], s["decode"]
            bad = []
            for ph, c in (("prefill", pre), ("decode", dec)):
                sk, ids = c.get("skipped", 0), c.get("skipped_ids", 0)
                if c["hits"] + c["hits_loading"] + c["misses"] != c["needed"]:
                    bad.append(f"{ph}: hits + hits_loading + misses != needed")
                # distinct experts skipped never outnumber the ids that chose them, and only ranks 4..k skip
                if not sk <= ids <= c["steps"] * (ub if ph == "prefill" else 1) * (kk - 3):
                    bad.append(f"{ph}: {sk} skipped, {ids} ids, more than ranks 4 to {kk} of {c['steps']} remaps allow")
                if c["load_failures"]:
                    bad.append(f"{ph}: {c['load_failures']} load failures")
            if pre["steps"] != n_pre * n_layer or dec["steps"] != n_dec * n_layer:
                bad.append(f"remaps {pre['steps']} and {dec['steps']}, want {n_pre * n_layer} and {n_dec * n_layer}")
            # a decode remap has one token, whose kk ids are distinct experts: each is needed or skipped, once
            if dec["needed"] + dec.get("skipped", 0) != dec["steps"] * kk or dec.get("skipped", 0) != dec.get("skipped_ids", 0):
                bad.append(f"decode: needed + skipped = {dec['needed'] + dec.get('skipped', 0)}, want {dec['steps']} remaps * {kk}")
            if pre["needed"] + pre.get("skipped", 0) > pre["steps"] * ub * kk:
                bad.append("prefill: needed + skipped exceed the ids")
            # no prefetch: every load is a miss, and a slot is only reused by an eviction
            if pre["misses"] + dec["misses"] > slots * n_layer + pre["evictions"] + dec["evictions"]:
                bad.append("more misses than slots and evictions")
            if s["expert_cache"].get("wait", "sleep") != (args.wait_mode or "sleep"):
                bad.append(f"wait {s['expert_cache'].get('wait')}")
            return bad

        for slots in slots_list:
            ub = min(n_prompt, max(1, slots // k))
            sizes = [ub] * (n_prompt // ub) + ([n_prompt % ub] if n_prompt % ub else [])
            n_pre = sum(n > 1 for n in sizes)
            n_dec = len(tokens) - n_prompt + sum(n == 1 for n in sizes)
            what = f"{label} {dev_name}, skip, {slots} slots, ubatch {ub}"

            lossless, ls = run(slots, ub)
            # (b) off: the graph is the lossless one, so the logits and counters are too
            off, s0 = run(slots, ub, skip_weight=0.0)
            bad = [f"{ph} {key}" for ph in ("prefill", "decode") for key in same_fields if s0[ph].get(key, 0) != ls[ph].get(key, 0)]
            bad += counted(s0, ub, n_pre, n_dec, slots, k)
            report(np.array_equal(off, lossless) and not bad and s0["expert_cache"].get("skip_weight") == 0
                   and s0["prefill"].get("skipped", 0) + s0["decode"].get("skipped", 0) == 0,
                   f"{what}, weight 0: logits of {len(tokens)} positions bit-identical to the lossless run, nothing skipped, same counters"
                   + (f": {bad}" if bad else ""))

            # (c) on, but below every router weight: the remap reads the weights and the mask op runs,
            # yet nothing is skipped, so the mask is all 1.0 and weights * 1.0 must be exact. No
            # sigmoid weight of this router comes near 1e-30
            tiny_w, s1 = run(slots, ub, skip_weight=1e-30)
            bad = [f"{ph} {key}" for ph in ("prefill", "decode") for key in same_fields if s1[ph].get(key, 0) != ls[ph].get(key, 0)]
            bad += counted(s1, ub, n_pre, n_dec, slots, k)
            same = np.array_equal(tiny_w, lossless)
            report(same and not bad and s1["expert_cache"].get("skip_weight") == 1e-30,
                   f"{what}, weight 1e-30: the mask path runs and skips nothing: logits bit-identical to the lossless run, same counters"
                   + ("" if same else f", max |diff| {float(np.abs(tiny_w - lossless).max()):.3g}") + (f": {bad}" if bad else ""))

            # (d) every true miss at ranks 4..k is eligible: the router's weights are sigmoids, below 1
            lossy, s2 = run(slots, ub, skip_weight=1.0)
            pre, dec = s2["prefill"], s2["decode"]
            bad = counted(s2, ub, n_pre, n_dec, slots, k)
            # with a slot for every expert nothing is evicted, so each is read at most once per layer;
            # what still skips is an expert's first touch, which is a true miss even here
            if slots >= n_expert and (pre["evictions"] + dec["evictions"] > 0 or pre["misses"] + dec["misses"] > n_expert * n_layer):
                bad.append(f"{pre['evictions'] + dec['evictions']} evictions and {pre['misses'] + dec['misses']} misses with a slot for every expert")
            finite = bool(np.isfinite(lossy).all())
            changed = not np.array_equal(lossy, lossless)
            report(dec.get("skipped", 0) > 0 and finite and changed and not bad and s2["expert_cache"].get("skip_weight") == 1,
                   f"{what}, weight 1: decode skips {dec.get('skipped', 0)} experts ({100 * dec.get('skipped', 0) / max(dec['steps'] * k, 1):.1f}% of its ids), "
                   f"logits finite and changed, counters consistent ({phases(s2)}; lossless: {phases(ls)})" + (f": {bad}" if bad else ""))
            print(f"info {what}, weight 1: max |diff| to the lossless run {float(np.abs(lossy - lossless).max()):.3g}, "
                  f"argmax agrees at {int((lossy.argmax(-1) == lossless.argmax(-1)).sum())} of {len(tokens)} positions", flush=True)

            # (e) metamorphic: a skipped id's weight is masked to 0, so which ready slot it reads
            # cannot matter. Without the mask (control M108) rank 1's and rank 2's experts differ
            other, s3 = run(slots, ub, skip_weight=1.0, skip_map_rank=2)
            diffs = [f"{ph} {key} {s3[ph].get(key)} != {s2[ph].get(key)}" for ph in ("prefill", "decode") for key in same_fields
                     if s3[ph].get(key) != s2[ph].get(key)]
            same = np.array_equal(other, lossy)
            report(same and not diffs,
                   f"{what}, weight 1: a skipped id on rank 2's slot instead of rank 1's gives bit-identical logits and counters"
                   + ("" if same else f", max |diff| {float(np.abs(other - lossy).max()):.3g}") + (f": {diffs}" if diffs else ""))

        if ranks:
            # the rank rule at its edge, on the fewest slots: with 3 experts per token no id has rank 4,
            # so nothing may skip; with 4 the rank-4 id may, at most one per token and layer
            slots = slots_list[0]
            for kk in (3, 4):
                ub = min(n_prompt, max(1, slots // kk))
                sizes = [ub] * (n_prompt // ub) + ([n_prompt % ub] if n_prompt % ub else [])
                n_pre = sum(n > 1 for n in sizes)
                n_dec = len(tokens) - n_prompt + sum(n == 1 for n in sizes)
                what = f"{label} {dev_name}, skip, {kk} experts per token, {slots} slots, ubatch {ub}"
                lossless, ls = run(slots, ub, kk)
                lossy, s = run(slots, ub, kk, skip_weight=1.0)
                bad = counted(s, ub, n_pre, n_dec, slots, kk)
                n_sk = s["prefill"].get("skipped_ids", 0) + s["decode"].get("skipped_ids", 0)
                if kk == 3:
                    report(np.array_equal(lossy, lossless) and n_sk == 0 and not bad,
                           f"{what}, weight 1: no id has rank 4, so nothing skips: logits bit-identical to the lossless run ({phases(s)})" + (f": {bad}" if bad else ""))
                else:
                    report(s["decode"].get("skipped", 0) > 0 and not np.array_equal(lossy, lossless) and not bad,
                           f"{what}, weight 1: the rank-4 id skips, at most once per token and layer ({phases(s)})" + (f": {bad}" if bad else ""))

        # what cannot work fails to load: the GPU remap's gate has no weights to read nor a mask to
        # write, and without a cache the model would load fully resident (tiny only)
        if gpu_remap_on and on_gpu(dev):
            try:
                r.logits(gguf, dev, tokens[:n_prompt], k, no_repack=True, expert_cache={"n_slots": 64, "gpu_remap": True, "skip_weight": 0.5})
                report(False, f"{label} {dev_name}: GPU remap with the skip loaded")
            except RuntimeError:
                report("does not work with --expert-cache-skip" in "".join(r.log), f"{label} {dev_name}: GPU remap with the skip fails to load, and says why")
        if not ranks:
            return
        for ec, needle, how in (({"skip_weight": 0.5}, "needs --expert-cache or --expert-cache-layers", "without a cache"),
                                ({"n_slots": 16, "skip_weight": -0.5}, "want a finite weight >= 0", "with weight -0.5"),
                                ({"n_slots": 16, "skip_weight": 0.5, "skip_map_rank": 3}, "want 1 or 2", "with map rank 3")):
            try:
                r.logits(gguf, dev, tokens[:8], 8, no_repack=True, expert_cache=ec)
                report(False, f"{label} {dev_name}: the skip {how} loaded")
            except RuntimeError:
                report(needle in "".join(r.log), f"{label} {dev_name}: the skip {how} fails to load, and says why")

    def check_skip_cli(base: list[str], stats: Path, want: str) -> None:
        """--expert-cache-skip on the command line: 0 generates the lossless text; 1 skips and says so
        in the stats; what the library refuses, the tool exits on."""
        if not skip_on:
            return
        off = subprocess.run(cli(base + ["--expert-cache", "16", "--expert-cache-skip", "0"]), capture_output=True, text=True, errors="replace")
        report(off.returncode == 0 and off.stdout == want, f"llama-completion --expert-cache-skip 0: same text as without the cache, rc {off.returncode}")
        stats.unlink(missing_ok=True)
        run = subprocess.run(cli(base + ["--expert-cache", "16", "--expert-cache-skip", "1", "--expert-cache-stats", str(stats)]),
                             capture_output=True, text=True, errors="replace")  # lossy: the tokens may not be UTF-8
        try:
            s = json.loads(stats.read_text())
            got = (s["expert_cache"]["skip_weight"], s["prefill"]["skipped"] + s["decode"]["skipped"])
        except (OSError, ValueError, KeyError) as e:
            got = (str(e), 0)
        report(run.returncode == 0 and run.stdout.strip() != "" and got[0] == 1 and got[1] > 0,
               f"llama-completion --expert-cache-skip 1: the stats say skip_weight {got[0]}, {got[1]} experts skipped ({run.stdout.strip()[:40]!r}), rc {run.returncode}")
        cases = [([], "needs --expert-cache or --expert-cache-layers", "without a cache"),
                 (["--expert-cache", "16", "--expert-cache-skip", "-1"], "want a finite weight >= 0", "-1")]
        if gpu_remap_on:
            cases.append((["--expert-cache", "64", "--expert-cache-gpu-remap"], "does not work with --expert-cache-skip", "and --expert-cache-gpu-remap"))
        for extra, msg, what in cases:
            args_ = extra if "--expert-cache-skip" in extra else extra + ["--expert-cache-skip", "0.5"]
            run = subprocess.run(cli(base + args_), capture_output=True, text=True, errors="replace")
            report(run.returncode != 0 and run.stdout.strip() == "" and msg in run.stderr,
                   f"llama-completion --expert-cache-skip {what} exits {run.returncode} without generating and says why")

    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "stats.json"
        if args.real:
            workload = [int(t) for t in np.load(args.tokens)]
            tokens = workload[:160]
            gpu = [(n, d) for n, d in r.devices() if r.lib.ggml_backend_dev_type(d) == r.lib.GGML_BACKEND_DEVICE_TYPE_GPU]
            if args.gpu_remap_only:
                check_gpu_remap(args.real, *gpu[0], tokens, 128, [128, 64], 50, 6, stats, args.real.stem)
                # 16 tokens' 96 experts per layer at most fit 128 slots, so a repeated pass stops nowhere
                check_gpu_remap_hits(args.real, *gpu[0], tokens[:16], 6, stats, args.real.stem, n_slots=128)
                print(f"{'FAIL' if failed else 'PASS'}: {failed} failures")
                sys.exit(1 if failed else 0)
            if args.stack_only:
                side = args.sidefile or args.real.with_name(args.real.name + ".experts")
                if not side.exists():
                    raise SystemExit(f"{side} does not exist: go run ./cmd/kolibri-sidefile write -gguf {args.real}")
                pair(args.real, side, f"{args.real.stem} side-file", 8)
                check_stack(args.real, *gpu[0], side, tokens, 128, [128, 64], 50, 6, stats, args.real.stem)
                print(f"{'FAIL' if failed else 'PASS'}: {failed} failures")
                sys.exit(1 if failed else 0)
            if args.skip_only:
                check_skip(args.real, *gpu[0], tokens, 128, [SKIP_SLOTS_REAL], 50, 384, 6, stats, args.real.stem, ranks=False)
                print(f"{'FAIL' if failed else 'PASS'}: {failed} failures")
                sys.exit(1 if failed else 0)
            side = None
            if sidefile_on and not args.waves_only:
                side = args.sidefile or args.real.with_name(args.real.name + ".experts")
                if not side.exists():
                    raise SystemExit(f"{side} does not exist: go run ./cmd/kolibri-sidefile write -gguf {args.real}")
                pair(args.real, side, f"{args.real.stem} side-file", 8)
            if not args.waves_only:
                check(args.real, *gpu[0], tokens, 128, [128, 64], 50, 6, stats, args.real.stem, side)
                check_gpu_remap(args.real, *gpu[0], tokens, 128, [128, 64], 50, 6, stats, args.real.stem)
                check_gpu_remap_hits(args.real, *gpu[0], tokens[:16], 6, stats, args.real.stem, n_slots=128)
                check_stack(args.real, *gpu[0], side, tokens, 128, [128, 64], 50, 6, stats, args.real.stem)
                check_skip(args.real, *gpu[0], tokens, 128, [SKIP_SLOTS_REAL], 50, 384, 6, stats, args.real.stem, ranks=False)
            check_waves(args.real, *gpu[0], tokens, 128, [128, 64], 50, 384, 6, stats, args.real.stem)
            check_waves(args.real, *gpu[0], tokens[:40], N_GROWN_REAL, [18], 50, 384, 6, stats, f"{args.real.stem} grown waves", grown=True)
            # the benchmark's ubatch: on a real workload about a third of the layers' unions outgrow 4 waves of 61
            check_waves(args.real, *gpu[0], workload[:N_BENCH_UBATCH + 32], N_BENCH_UBATCH, [128], 50, 384, 6, stats, f"{args.real.stem} benchmark ubatch")
        elif args.gpu_remap_only:
            model = generate_tiny(Path(tmp), args.seed, "router")
            cfg = json.loads((model / "config.json").read_text())
            rng = random.Random(args.seed)
            tokens = [rng.randrange(1000, 100000) for _ in range(N_PROMPT + N_DECODE)]
            for outtype in ["f32", "q8_0"]:
                gguf = convert(model, args.llama_cpp, outtype)
                for dev_name, dev in r.devices():
                    check_gpu_remap(gguf, dev_name, dev, tokens, N_PROMPT, SLOTS, cfg["num_hidden_layers"],
                                    cfg["num_experts_per_tok"], stats, outtype)
                    check_gpu_remap_hits(gguf, dev_name, dev, tokens[:N_PROMPT], cfg["num_experts_per_tok"], stats, outtype)
                    check_gpu_remap_refusals(gguf, dev_name, dev, stats, outtype)
            check_gpu_remap_exact()
        elif args.stack_only:
            model = generate_tiny(Path(tmp), args.seed, "router")
            cfg = json.loads((model / "config.json").read_text())
            rng = random.Random(args.seed)
            tokens = [rng.randrange(1000, 100000) for _ in range(N_PROMPT + N_DECODE)]
            for outtype in ["f32", "q8_0"]:
                gguf = convert(model, args.llama_cpp, outtype)
                side = write_sidefile(gguf, f"{outtype} side-file")
                for dev_name, dev in r.devices():
                    check_stack(gguf, dev_name, dev, side, tokens, N_PROMPT, SLOTS, cfg["num_hidden_layers"],
                                cfg["num_experts_per_tok"], stats, outtype)
        elif args.skip_only:
            model = generate_tiny(Path(tmp), args.seed, "router")
            cfg = json.loads((model / "config.json").read_text())
            rng = random.Random(args.seed)
            tokens = [rng.randrange(1000, 100000) for _ in range(N_PROMPT + N_DECODE)]
            for outtype in ["f32", "q8_0"]:
                gguf = convert(model, args.llama_cpp, outtype)
                for dev_name, dev in r.devices():
                    check_skip(gguf, dev_name, dev, tokens, N_PROMPT, SKIP_SLOTS, cfg["num_hidden_layers"], cfg["num_experts"],
                               cfg["num_experts_per_tok"], stats, outtype)
            base = [str(args.llama_cpp / "build" / "bin" / "llama-completion"), "-m", str(gguf), "-p", "Die Energiewende in Deutschland hat das Ziel",
                    "-n", "24", "--temp", "0", "--ignore-eos", "-no-cnv", "--no-display-prompt", "-c", "256", "--no-warmup"]
            check_skip_cli(base, stats, subprocess.run(base + ["-ub", "2"], capture_output=True, text=True).stdout)
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
                    check_gpu_remap(gguf, dev_name, dev, tokens, N_PROMPT, SLOTS, n_layer, k, stats, outtype)
                    check_gpu_remap_hits(gguf, dev_name, dev, tokens[:N_PROMPT], k, stats, outtype)
                    check_gpu_remap_refusals(gguf, dev_name, dev, stats, outtype)
                    check_stack(gguf, dev_name, dev, side, tokens, N_PROMPT, SLOTS, n_layer, k, stats, outtype)
                    check_skip(gguf, dev_name, dev, tokens, N_PROMPT, SKIP_SLOTS, n_layer, cfg["num_experts"], k, stats, outtype)

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
            for key, val in (wait_fields(args.wait_mode) if args.wait_mode is not None else {}).items():
                setattr(ec, key, val)
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
            cached = subprocess.run(cli(base + ["--expert-cache-layers", str(layers), "--expert-prefetch", "8,4,0.25",
                                                "--expert-cache-threads", "2", "--expert-cache-buffered", "--expert-cache-stats", str(stats)]),
                                    capture_output=True, text=True)
            ok = plain.returncode == 0 and cached.returncode == 0 and stats.exists()
            if ok:
                s = json.loads(stats.read_text())
                c = s["expert_cache"]
                ok = (c["slots_min"], c["slots_max"], c["prefetch_k"], c["prefetch_half_life"], c["prefetch_mix"], c["io_threads"], c["buffered"]) == \
                     (12, 384, 8, 4, 0.25, 2, True) and s["decode"]["steps"] > 0 and s["decode"]["tokens"] > 0 \
                     and c.get("wait", "sleep") == (args.wait_mode or "sleep")
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
            # the MoE cache would take over the same matmuls, and the graph has one slot map for them
            both = subprocess.run(cli(base + ["--expert-cache", "16", "--moe-cache-mib", "8"]), capture_output=True, text=True, errors="replace")
            report(both.returncode != 0 and both.stdout.strip() == "" and "cannot serve one model" in both.stderr,
                   f"llama-completion with --expert-cache and --moe-cache-mib exits {both.returncode} without generating, refused by name, no abort ({both.stderr[-200:] if both.returncode < 0 else 'no signal'})")
            # a LoRA on the experts the cache holds is refused; without the cache, or on what it does not hold, it applies
            for kind in ("gate", "up", "down"):
                lora = Path(tmp) / f"lora-{kind}.gguf"
                write_lora(args.llama_cpp, lora, gguf, [f"blk.0.ffn_{kind}_exps.weight"])
                refused_run = subprocess.run(cli(base + ["--expert-cache", "16", "--lora", str(lora)]), capture_output=True, text=True, errors="replace")
                report(refused_run.returncode != 0 and refused_run.stdout.strip() == "" and "reads from its slots" in refused_run.stderr,
                       f"LoRA on ffn_{kind}_exps with the cache: exits {refused_run.returncode} without generating, refused by name")
                applied = subprocess.run(base + ["-ub", "2", "--lora", str(lora)], capture_output=True, text=True, errors="replace")
                report(applied.returncode == 0 and applied.stdout.strip() != "" and applied.stdout != plain.stdout,
                       f"LoRA on ffn_{kind}_exps without the cache: loads and changes the text ({applied.stdout.strip()[:40]!r})")
            lora = Path(tmp) / "lora-router.gguf"
            write_lora(args.llama_cpp, lora, gguf, ["blk.0.ffn_gate_inp.weight"])
            on_resident = subprocess.run(base + ["-ub", "2", "--lora", str(lora)], capture_output=True, text=True, errors="replace")
            on_cache = subprocess.run(cli(base + ["--expert-cache", "16", "--lora", str(lora)]), capture_output=True, text=True, errors="replace")
            report(on_resident.returncode == 0 and on_cache.returncode == 0 and on_resident.stdout != plain.stdout and on_cache.stdout == on_resident.stdout,
                   f"LoRA on the router's input with the cache: same text as without it, changed by the adapter ({on_cache.stdout.strip()[:40]!r}{'' if on_cache.returncode == 0 else ' ' + on_cache.stderr[-300:]})")
            odd = Path(tmp) / 'we"ird\\na\tme.gguf'
            shutil.copyfile(gguf, odd)
            stats.unlink(missing_ok=True)
            run = subprocess.run(cli([str(bin_dir / "llama-completion"), "-m", str(odd), "-p", prompt, "-n", "4", "-no-cnv",
                                      "-c", "256", "--no-warmup", "--expert-cache", "16", "--expert-cache-stats", str(stats)]),
                                 capture_output=True, text=True, errors="replace")  # no --temp 0: the tokens may not be UTF-8
            try:
                named = json.loads(stats.read_text())["expert_cache"]["file"]
            except (OSError, ValueError) as e:
                named = f"unreadable: {e}"
            report(run.returncode == 0 and Path(named).name == odd.name,
                   f"stats of a model named {odd.name!r} are valid JSON and name it ({named!r})")
            if sidefile_on:
                sf = sides["q8_0"]
                stats.unlink(missing_ok=True)
                on_side = subprocess.run(cli(base + ["--expert-cache", "16", "--expert-cache-sidefile", str(sf), "--expert-prefetch", "8",
                                                     "--expert-cache-threads", "2", "--expert-cache-stats", str(stats)]), capture_output=True, text=True)
                s = json.loads(stats.read_text()) if on_side.returncode == 0 and stats.exists() else None
                report(s is not None and on_side.stdout == plain.stdout and reads_ok(s, sf) and s["expert_cache"]["io_threads"] == 2,
                       f"llama-completion --expert-cache-sidefile: same text as without the cache, one pread per expert ({s['decode']['preads'] if s else on_side.stderr[-300:]} in decode)")
                gone = subprocess.run(cli(base + ["--expert-cache", "16", "--expert-cache-sidefile", str(sf) + ".gone"]), capture_output=True, text=True)
                report(gone.returncode != 0 and gone.stdout.strip() == "" and "no fallback" in gone.stderr,
                       f"llama-completion with a missing side-file exits {gone.returncode} without generating, no fallback to the GGUF")
                # refused before any tensor is created: on the real model the load would wire the weights first
                alone = subprocess.run(base + ["--expert-cache-sidefile", str(sf), "-v"], capture_output=True, text=True, errors="replace")
                loading = subprocess.run(base + ["-n", "1", "-v"], capture_output=True, text=True, errors="replace")
                report(alone.returncode != 0 and alone.stdout.strip() == "" and "without the cache" in alone.stderr
                       and "load_tensors:" not in alone.stderr and "load_tensors:" in loading.stderr,
                       f"llama-completion with a side-file and no --expert-cache exits {alone.returncode} without generating or loading a tensor")
            report(ok and plain.stdout == cached.stdout and plain.stdout.strip() != "",
                   f"llama-completion --expert-cache-layers/--expert-prefetch/--expert-cache-buffered/--expert-cache-stats: as asked, same text as without the cache "
                   f"({plain.stdout.strip()[:60]!r})" + ("" if ok else f" rc {plain.returncode}/{cached.returncode}: {cached.stderr.strip()[-300:]}"))
            if gpu_remap_on:
                check_gpu_remap_cli(base, gguf, stats, plain.stdout)
            check_skip_cli(base, stats, plain.stdout)
            check_gpu_remap_exact()

    print(f"{'FAIL' if failed else 'PASS'}: {failed} failures")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
