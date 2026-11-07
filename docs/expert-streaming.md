# Expert streaming: the runtime expert cache

Phase 10, Stage D, item 1: a bounded, byte-accounted cache of routed experts inside libllama (patch 0011, fork branch `feat/kolibri-expert-cache`). It is experimental and Kolibri only. Stage D was started by the user's decision (2026-10-14) although no predictor passed the offline gates of Stages A and B ([expert-policy.md](expert-policy.md)). This document covers the prototype and its correctness checks. The benchmark (item 2) and the gates (item 3) are in "Benchmark" below; the speeds before it are INFO, measured on a busy machine. Phase 11's one read per expert, whose deciding gates passed, is in "Stage E3: one read per expert". Phase 11's prefill experiment, whose code stays out of the patch series, is in "Stage E4: waves for prefill". Stage E2's GPU remap was benchmarked, its deciding gate failed at 16 GiB and passed at 0 GiB, and its code also stays out of the patch series; it is in "Stage E2: GPU remap".

## How it works

- **The file keeps the experts.** The stacked expert tensors `ffn_{gate,up,down}_exps` of every layer become lazy tensors: mapped on the CPU, excluded from the loader's `WILLNEED` advice, and never read through the mapping. Nothing places them on the GPU.
- **Slots.** Each layer gets slot tensors of the same type and row size, with `n_slots` experts instead of 384, on the device of the layer's other weights.
  - The slot memory is always host memory, since the reads write into it. For a CPU layer it is the CPU's plain buffer type, never a repacking one. For a GPU layer it is an anonymous mapping that the device uses in place (`buffer_from_host_ptr`; Metal does, as for mapped weights).
  - A device that cannot (CUDA, for example) is rejected at load time.
  - Slots never written cost no memory.
  - The buffers count as weights, so the scheduler runs each layer's expert matmuls where its slots are.
- **The remap op.** In `build_moe_ffn`, after the router's selection and its weights, a CPU custom op maps the router's ids to slots, `ffn_moe_slots-<layer>`. The three `mul_mat_id` read the slot tensors with these slot ids. The expert weights keep the router's own ids, and the router chooses every expert; a prediction only decides what is read ahead.
- **Loads.** A missing expert is read with `pread` into its slot: three ranges per expert (gate, up, down), from a second descriptor of the GGUF with `F_NOCACHE` (macOS), so the bytes do not stay in the page cache. On unified memory there is no host-to-device copy: the read from the SSD into the slot *is* the transfer. Reads of pages that are already in the page cache are served from it, `F_NOCACHE` notwithstanding.
  With `--expert-cache-sidefile` a miss is one `pread` of one extent instead (patch 0014, "Stage E3" below).
- **Bookkeeping** (`src/llama-expert-slots.*`, independent of ggml):
  - LRU per layer;
  - a demand queue ahead of a prefetch queue, served by I/O threads;
  - a remap pins the experts its ubatch needs, waits for a prefetch of a needed expert instead of reading it twice, and drops the queued prefetches of its layer that it does not need;
  - a loading slot is never a victim;
  - a layer is in flight from its remap to the next one: no prefetch replaces its slots meanwhile. The scheduler's sync before every CPU split guarantees that the previous layer's matmuls are done when the next remap runs;
  - a failed prefetch falls back to a demand load; a failed demand load aborts with the file and offset.
- **Prefetch** (optional): a C++ port of the decayed-count Markov predictor of `internal/locality`, same-token horizon. After layer *l* of a decode step it ranks the experts of layer *l*+2 of the same token and queues the best *k* that are not resident.
- **The ubatch cap.** A ubatch must not need more distinct experts than a layer has slots. The context caps `n_ubatch` at the fewest slots of a layer divided by 6 (logged as a warning). One path, all on the GPU, serves prefill and decode; prefill gets slower with small caches.

## Using it

Options of every `common` tool (llama-cli, llama-completion, llama-server):

| Option | Meaning |
|---|---|
| `--expert-cache N` | at most N experts per layer in device memory, the others read from the file on demand |
| `--expert-cache-layers FNAME` | slots per layer instead: one integer per layer, separated by commas or white space |
| `--expert-cache-threads N` | I/O threads (default 4) |
| `--expert-prefetch K[,HALF_LIFE[,MIX]]` | prefetch K experts per layer, ranked by the predictor (default off; half-life 1 token, mix 0.75, Stage B's same-token choice) |
| `--expert-cache-stats FNAME` | write the counters as JSON when the model is freed; without `--expert-cache` the decode timing alone (patch 0012) |
| `--expert-cache-buffered` | read misses through the page cache instead of with `F_NOCACHE` (macOS; default off, patch 0012) |
| `--expert-cache-sidefile FNAME` | read misses from this side-file, one `pread` per expert; the load fails if its table does not match the model and the GGUF, with no fallback (patch 0014) |

```sh
B=third_party/llama.cpp/build/bin
$B/llama-completion -m ~/models/Kolibri-1-Q8_0.gguf -ngl 99 -t 10 -c 4096 --expert-cache 128 \
    --expert-cache-stats q8.json -p "..." -n 128 --temp 0 -no-cnv
```

In the C API it is `llama_model_params.expert_cache`, a `llama_expert_cache_params` (`llama_expert_cache_default_params()`).

## Counters

`--expert-cache-stats` writes one JSON object:

- `expert_cache`: the configuration, the slot buffer's bytes, one expert's bytes and the file.
- `prefill` and `decode`, the same counters per phase:
  - `steps`, one per layer and ubatch; a ubatch of one token counts as decode;
  - `needed`, the distinct experts the remaps needed, split into `hits`, `hits_loading` (a prefetch of it was still queued or running) and `misses`;
  - `evictions`, `demand_bytes` and `prefetch_bytes`;
  - `prefetch_issued`, `_used`, `_wasted` (evicted unused), `_cancelled` (dropped from the queue) and `_skipped` (resident or loading already, or its layer in flight);
  - `load_failures`;
  - `wait_seconds`, the time the remaps blocked on loads, which is the cost on the critical path;
  - `io_seconds`, the loads' own time, summed over the I/O threads;
  - `io_prefetch_seconds`, the part of it that prefetches took. It overlaps compute, unless a remap waited for the prefetch (`hits_loading`). Demand loads overlap nothing, except each other on the I/O threads.
- Patch 0012 adds (Phase 11, Stage E1):
  - `expert_cache.buffered`, whether misses are read through the page cache (always true on Linux);
  - `remap_seconds` per phase, the remap op's time minus its wait on loads: the bookkeeping's own CPU cost;
  - in `decode`, over every ubatch of one token: `tokens`; `wall_seconds`, from the graph's build or reuse to the outputs' copy, synchronized; `splits`, the scheduler's graph splits; and `metal_commits`, the Metal command buffers committed meanwhile (`null` without the Metal backend). `wall_seconds − wait_seconds − remap_seconds` is the rest: compute, syncs and graph handling.
  - Without the cache, `--expert-cache-stats` writes `{"expert_cache": null, "decode": {tokens, wall_seconds, splits, metal_commits}, "process": {...}}`, the process counters measured since the model was loaded.
  - Known counts per decoded token on Metal (control M42, in the tiny check): 2 splits resident, 2 + 2 per layer with the cache (each layer's remap is a CPU split); IQ3: 2 and 102. Commits: 2 per Metal split plus 1 for the output copy, so 3 and 103.
- Patch 0014 adds (Phase 11, Stage E3):
  - `expert_cache.sidefile`, the side-file as given, or `null` on the GGUF path;
  - `loads` and `preads` per phase: the experts read into a slot, and the read calls they took. A run on the side-file has one per expert and a run on the GGUF path three, so `preads / loads` proves which path ran.
- `process`, measured since the cache was set up: `disk_bytes_read` and `pageins` from `proc_pid_rusage`, `major_faults` from `getrusage`, and the lifetime peak `max_phys_footprint`. A counter the platform does not measure is `null`: on Linux all but `major_faults`, on iOS, tvOS and visionOS the same.
- The file name is a JSON string, escaped.

## Checks

**Unit tests.** `build/bin/test-expert-cache testdata/locality/markov-fixture.json` (also in ctest, there without the fixture):

- LRU order;
- pins, also when the only other slot is loading;
- the capacity limit;
- an awaited prefetch;
- cancellation of queued prefetches;
- the layer in flight;
- failure fallback;
- a 400-step stress run with four I/O threads. Every id maps to a slot that holds it, the slots of the layer in flight stay intact until the next remap, and no two loads write one slot.

Output: "markov fixture: 978 steps, 2528 predictions over 4 parameter sets, as the Go predictor", "all expert cache tests passed"; 30 runs in a row all pass.

The fixture comes from `internal/locality`'s `TestMarkovFixture`: a random conversation replayed through the Go predictor for four parameter sets (half-lives 1, 4, 16, 64; mixes 0.75, 0.25, 0, 1). The Go test fails if the predictor and the file disagree.

**Tiny checkpoint** (`tools/gguf/check_expert_cache.py`):
- **Setup:** `cmd/kolibri-tiny -router` (384 experts, top 6) as F32 and Q8_0, on the CPU and Metal. 40 prompt tokens are prefilled and 60 decoded, resident against the cache with 6, 16, 64 and 384 slots, with and without prefetch, at the same ubatch.
- **Also checked:**
  - slots per layer (12, 64, 18, 384, 30, 100), bit-identical as well;
  - fewer than 6 slots fail to load;
  - a second context on the same model fails, and after the first is freed a new one gets the cache;
  - `llama-completion` with `--expert-cache-layers`, `--expert-prefetch 8,4,0.25`, `--expert-cache-threads 2` and `--expert-cache-stats` runs as asked (the stats record it) and generates the same greedy text as without the cache;
  - placement: with 3 of 6 layers on Metal, each layer's slots are on its own device; with the CPU's repacking enabled, the slots still hold the file's bytes. Both are bit-identical to the resident run without repacking;
  - a model file named with a quote, a backslash and a tab still gives valid stats JSON.
- **Result:** "PASS: 0 failures", 102 checks; 163 with patch 0012 (buffered reads, decode timing, M42). The logits of all 100 positions are bit-identical in every configuration, and the counters are consistent.

**Since the rebase onto upstream master (2026-10-23, Stage E0.1):** on the GPU the resident reference is split after each layer's `ffn_moe_topk`, where the cache's CPU remap splits the graph. Master's Metal backend fuses ops (`RMS_NORM + MUL + ADD`) across those points in an unsplit graph and rounds differently, so the unsplit resident run differs by up to 0.134 on the tiny Q8_0 fixture and 3.23 on IQ3; the check reports that as info. With the same split the logits are bit-identical; the CPU still compares against the plain resident run. Tiny: 164 checks, rc 0; IQ3: "PASS: 0 failures". Details and control M45 in [upstream.md](upstream.md#the-rebase-stage-e01).

**IQ3, real weights** (`check_expert_cache.py --real ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf`):
- **Setup:** the first 160 tokens of the coding workload, 128 prefilled and 32 decoded, on Metal.
- **Result:** with 128 slots (ubatch 21) and with 64 (ubatch 10), with and without prefetch, the logits are bit-identical to resident Metal at the same ubatch. "PASS: 0 failures".
- **Decode hits:** 8634 of 9600 needed at 128 slots, 7714 at 64.
- **Prefetch:** at 128 slots it issues nothing, since every predicted expert is resident already. At 64 slots 8 of 23 prefetches are used.

**Q8_0, larger than memory** (83 GB file, 48 GB Mac, `llama-completion` as above, 22 prompt tokens, 128 generated with `--ignore-eos`):

| Run | Decode tokens/s | Decode hit rate | Read per decoded token | Peak footprint |
|---|--:|--:|--:|--:|
| Metal, `--expert-cache 128` | 11.25 | 94.2% | 72.6 MB | 19.2 GB |
| Metal, `--expert-cache 128 --expert-prefetch 12` | 7.67 | 94.3% | 71.0 MB + 1.83 GB prefetched in all | 20.5 GB |
| CPU only (`-dev none --no-repack`) | 0.31 | page cache | | 0.34 GB (28.5 GB RSS, mapped) |

All three generate the same 679 characters.
- **Disk reads:** the process read 19.12 GB from disk against 18.92 GB of demand loads (`F_NOCACHE` goes to the SSD), with 67 pageins.
- **Footprint:** the slot buffer is 26.7 GB, but slots never written cost no memory: the footprint stays at 19.2 GB.
- **Prefetch:** it wastes most of its reads (111 of 576 used, 138 cancelled) and is slower.
- **The CPU-only run** thrashed the page cache: 0.31 tokens/s against about 10 earlier on a quiet machine ([real-checkpoint.md](real-checkpoint.md)).
- **Machine state:** load average 12 to 38 from other processes, 14 GB of swap in use. These are not benchmark numbers.

**Negative controls.**
- **M32**, an off-by-one slot id in the remap: all 32 logit comparisons of the tiny check fail (max |diff| 0.7 to 1.1).
- **M33**, victims chosen without the pins: `test-expert-cache` fails, both the pin test while a slot loads and the stress run's counters.
- **M34**, one predicted expert changed in the fixture: "params 0, step 19 (token 70, layer 1): predicted 2 0 1 2 6 7, Go 2 1 1 2 6 7", exit 1.

## Benchmark

Stage D, items 2 and 3: resident Metal, `--cpu-moe`, the demand-loaded cache and the cache with prefetch, on the same checkpoint and workloads, cold and warm, at two memory ceilings. `cmd/kolibri-bench` runs the matrix `testdata/bench/expert-cache.json` and writes the report; results and logs are in `~/models/eval/expert-bench/`.

### Method

- **Machine:** M5 Pro, 48 GiB, macOS 27.0.1, shared with other sessions. Fork `2da6ec661` (patch 0011).
- **Server, every arm:** `llama-server -ngl 99 -t 10 -c 4096 -np 1 --fit off`, one slot, no prompt cache between requests.
- **Requests** (`kolibri-stream -tokens -summary`): the first 2048 tokens of a locality workload as the prompt, 128 tokens generated at temperature 0 with `ignore_eos`. Per run one cold request (coding) right after the server is up, then four warm ones (coding, research, hr, medtech).
- **Cold:** before every run `kolibri-evict` drops the GGUF from the page cache (`msync(MS_INVALIDATE)`, no root). On the IQ3 file it took 576976 resident pages to 0; reading 1 GiB afterwards cost 66531 page-ins, against 19 warm.
- **Memory ceiling:** 0 or 16 GiB wired by `kolibri-memhog` (`mlock`, which compression and swap cannot undo) before the server starts, so the arms see 48 or 32 GiB.
- **Quiet machine:** a run starts only when the 1-minute load is below 3, swap grew by less than 100 MiB in the last minute, and the Mac is on AC power (the AC condition was added for the last four runs; `pmset -g log` shows every earlier reported run on AC as well). The other processes' CPU during the run is recorded (`ps`, without the driver, server and hog); it is in every `run.json`.
- **Order:** repetitions outermost, so drift spreads over the arms. 3 repetitions per arm and ceiling; the `-ub 21` resident control once.
- **Arms:**
  - IQ3 (`Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf`, 35.6 GB) at 0 and 16 GiB: resident; `--cpu-moe`; `--expert-cache 128` (which caps the ubatch at 21); `--expert-cache 128 --expert-prefetch 12`; and resident with `-ub 21`, the equivalence control, at 0 GiB;
  - Q8_0 (83.1 GB, beyond the memory) at 0 GiB: `--expert-cache 128`, the same with `--expert-prefetch 12`, and `--cpu-moe --no-repack`. There is no resident arm.
- **Reported per arm and ceiling:** the median over the repetitions of each run's median over its four warm requests, with the range over the repetitions in brackets. Decode tokens/s and the p50/p95 per-token latency come from the stream's timestamps (a chunk of k tokens counts as k intervals); the prompt rate is the server's `prompt_per_second`.

**The gates, fixed before the runs** (`gates` in the matrix, `evalGate`):
- **Correctness:** `check_expert_cache.py` (tiny) and `--real` IQ3 give "PASS: 0 failures" on the final build; every cache and prefetch run generates the same token IDs as the resident `-ub 21` control in all five requests; no expert load failure.
- **Complete:** an exact equivalence passes only if every request of every configured run of its arms was compared; an arm runs at a ceiling only if all its repetitions completed, does not if none did, and anything in between withholds the gate's verdict.
- **Performance, at 16 GiB:** the demand cache wins if every repetition's warm decode rate is above the best repetition of the best other arm that runs at that ceiling, or every repetition's p95 below that arm's best. The prefetch arm is kept only if it beats the demand cache by the same rule. Prefill is reported, not gated; no hit rate counts. The same comparisons at 0 GiB and for Q8_0 are INFO.

### What happened on the way

- **Resident IQ3 at 16 GiB does not fit, and macOS does not fail it.** With `-ngl 99` every weight is in Metal buffers, which are wired like the hog: 33.1 GiB of weights and the 16 GiB hog are 49.1 GiB, beyond the 48 GiB of memory. On 2026-10-15 that run hung the Mac until a watchdog panic rebooted it (no check-ins from watchdogd in 93 s). The driver now checks the wired sum against `hw.memsize` and records such a run as "does not fit" without starting it. That is the result for this arm at 16 GiB.
- **Q8_0 `--cpu-moe` repacks its experts.** By default the CPU backend repacks Q8_0 into its own buffers instead of leaving the experts mapped. That server was killed (SIGKILL) 3.5 minutes into loading, most likely by memory pressure. The arm runs with `--no-repack`, as the CPU-only Q8_0 run above did. IQ3_XXS and IQ4_XS have no repacked types; IQ3 `--cpu-moe` keeps a footprint of about 1 GiB.
- **Sleep.** From 2026-10-16 07:10 the Mac was on battery and slept on and off; four runs were timed across hours of sleep (prompt rates of 0.2 to 0.4 tokens/s), and again when repeated on battery. They are set aside, not reported, and were repeated on AC power: the driver now waits for AC and holds `caffeinate -i -s`. All reported runs ran on AC.
- **The quiet gate holds at the start of a run only.** Other sessions' tests ran alongside: the other processes' CPU averaged 0.1 to 5.6 cores over a run, with peaks up to 13. The two noisiest runs are the first repetition of IQ3 `--cpu-moe` (5.6 and 3.8 cores on average); they show as the low ends of its ranges.

### Results

| Config | Ceiling | Runs | Load s | Cold TTFT s | Cold decode t/s | Warm prompt t/s | Warm decode t/s | Warm p50 ms | Warm p95 ms | Page-ins MiB/token | Peak RSS GiB | Peak footprint GiB | Decode hits | Disk GB read |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| iq3-resident | 0 GiB | 3 ok | 12.1 (12.1–25.5) | 1.84 (1.83–2.09) | 60.01 (52.92–60.10) | 1110.4 (1076.2–1181.0) | 60.13 (54.37–60.46) | 16.5 (16.5–17.8) | 17.1 (16.9–20.4) | 0.089 (0.051–0.244) | 33.2 (0.5–33.3) | 0.9 (0.9–0.9) | – | – |
| iq3-resident | 16 GiB | 0 ok, r1 does not fit, r2 does not fit, r3 does not fit | – | – | – | – | – | – | – | – | – | – | – | – |
| iq3-cpu-moe | 0 GiB | 3 ok | 9.3 (9.3–14.1) | 13.04 (12.46–96.20) | 17.05 (8.87–17.47) | 273.2 (22.9–273.5) | 17.42 (6.56–17.59) | 57.8 (57.7–88.9) | 60.9 (60.0–354.1) | 0.240 (0.225–16.173) | 32.8 (20.1–32.8) | 1.1 (1.0–1.1) | – | – |
| iq3-cpu-moe | 16 GiB | 3 ok | 11.3 (11.1–12.9) | 153.62 (125.51–180.25) | 10.37 (3.72–11.20) | 29.2 (13.9–33.2) | 11.86 (5.27–12.61) | 77.9 (72.3–177.2) | 135.9 (123.2–299.0) | 21.847 (17.265–79.155) | 17.5 (8.5–18.5) | 1.0 (1.0–1.1) | – | – |
| iq3-cache128 | 0 GiB | 3 ok | 1.5 (1.5–1.5) | 33.01 (32.78–33.31) | 16.79 (15.48–17.89) | 82.8 (80.9–83.7) | 18.51 (18.25–18.77) | 53.1 (51.5–53.3) | 78.9 (77.8–80.7) | 0.136 (0.134–0.237) | 13.6 (13.6–13.6) | 11.0 (11.0–11.0) | 90.9 (90.9–90.9)% | 448.40 (448.40–448.40) |
| iq3-cache128 | 16 GiB | 3 ok | 1.5 (1.5–2.3) | 33.24 (32.90–34.86) | 17.72 (15.31–17.83) | 80.6 (80.4–83.2) | 17.13 (16.92–19.05) | 55.4 (49.8–58.0) | 81.9 (81.6–85.3) | 0.170 (0.149–0.794) | 13.5 (13.2–13.6) | 11.0 (11.0–11.0) | 90.9 (90.9–90.9)% | 448.40 (448.40–448.43) |
| iq3-cache128-prefetch12 | 0 GiB | 3 ok | 1.5 (1.5–1.8) | 32.98 (32.95–33.58) | 16.89 (16.47–18.39) | 82.8 (74.5–83.8) | 18.43 (18.22–18.52) | 53.2 (51.4–53.8) | 78.9 (75.5–83.6) | 0.129 (0.129–0.509) | 13.6 (13.6–13.6) | 11.0 (11.0–11.0) | 90.9 (90.9–90.9)% | 455.18 (455.17–455.19) |
| iq3-cache128-prefetch12 | 16 GiB | 3 ok | 1.5 (1.5–1.5) | 33.55 (32.55–33.71) | 17.22 (15.63–17.53) | 81.6 (80.0–82.9) | 18.29 (17.77–18.91) | 51.8 (51.2–52.6) | 77.4 (77.0–80.7) | 0.152 (0.122–0.568) | 13.4 (13.2–13.6) | 11.0 (11.0–11.0) | 90.9 (90.9–90.9)% | 455.18 (455.13–455.20) |
| iq3-resident-ub21 | 0 GiB | 1 ok | 19.7 | 9.75 | 59.19 | 210.2 | 60.19 | 16.5 | 17.4 | 0.064 | 6.3 | 0.8 | – | – |
| q8-cache128 | 0 GiB | 3 ok | 1.8 (1.8–1.8) | 47.11 (47.05–48.01) | 12.25 (12.07–12.84) | 53.9 (53.7–54.9) | 13.00 (12.59–13.09) | 74.4 (72.9–76.7) | 111.9 (110.3–117.3) | 14.832 (13.266–26.261) | 27.3 (27.3–27.3) | 26.0 (26.0–26.0) | 90.3 (90.3–90.3)% | 939.69 (920.23–969.96) |
| q8-cache128-prefetch12 | 0 GiB | 3 ok | 1.8 (1.8–1.8) | 47.35 (46.71–47.46) | 11.81 (11.16–12.07) | 53.7 (53.0–54.2) | 12.73 (12.02–12.81) | 74.2 (74.1–79.5) | 116.1 (115.2–118.1) | 18.081 (15.845–21.328) | 27.3 (27.3–27.3) | 26.0 (26.0–26.0) | 90.4 (90.4–90.4)% | 951.99 (933.81–973.46) |
| q8-cpu-moe | 0 GiB | 3 ok | 17.3 (16.9–17.6) | 223.48 (223.36–228.62) | 6.40 (6.14–6.52) | 11.0 (10.7–11.0) | 6.45 (6.03–6.85) | 129.7 (129.0–142.1) | 322.8 (300.6–334.1) | 100.953 (95.824–115.066) | 33.4 (32.1–34.5) | 1.3 (1.3–1.3) | – | – |

Cold TTFT is the cold request's time to the first token, which includes its 2048-token prefill from an evicted page cache. Peak footprint is `phys_footprint`, which leaves out mapped file pages: the resident arm's 33 GiB of weights show in its RSS only.

**Reads.** Per run, the IQ3 cache read 448 GB from the disk for 5 × 2048 prompt and 5 × 128 generated tokens. Prefill needs most of it: 406 GB of demand loads against 29.6 GB in decode (46 MB per generated token, 90.9% decode hits). The remaps waited 51.6 s on loads in prefill and 9.2 s in decode, 14 ms of the about 58 ms per generated token. Q8_0 read 920 to 973 GB per run, against about 1074 GB of demand loads: reads of pages that are in the page cache are served from it, `F_NOCACHE` notwithstanding.

**Prefetch.** At 12 per layer it issued 4252 prefetches per IQ3 run, 1541 used and 2027 evicted unused, and read 6.8 GB in addition. The decode misses hardly change (17315 against 17424 of 190800 needed), nor does the wait on loads (9.5 s against 9.2 s).

### Equivalence

- PASS equivalence iq3-cache128, iq3-cache128-prefetch12 against iq3-resident-ub21: 60 of 60 requests generate the same token IDs
- PASS equivalence q8-cache128-prefetch12 against q8-cache128: 15 of 15 requests generate the same token IDs
- PASS expert load failures: 0 in the 18 of 18 cache runs that completed and wrote their stats
- INFO equivalence against resident IQ3 at the default ubatch: 0 of 65 requests are the same. The `-ub 21` control and the demand cache (every repetition, both ceilings) first differ at token 15, 15, 12, 14 and 22 (cold coding, then warm coding, research, hr, medtech); `--cpu-moe` at token 57, 57, 13, 27 and 15.
- INFO equivalence of Q8_0 `--cpu-moe --no-repack` against the Q8_0 cache: 0 of 15 the same, first differences at token 74, 74, 16, 2 and 0.

The `-ub 21` control differs from resident IQ3 at the default ubatch (512) from token 12 to 22 of each request: the ubatch changes the order of the sums. The cache arms differ from it at the same positions, since they generate exactly what the control generates. `--cpu-moe` runs the experts on the CPU with other kernels and differs from token 13 to 57.

### Gates

- GATE iq3-cache128 at 16 GiB against the best of iq3-resident, iq3-cpu-moe: passed: decode 17.13 (16.92–19.05) against iq3-cpu-moe's 11.86 (5.27–12.61) t/s (win true), p95 81.9 (81.6–85.3) against 135.9 (123.2–299.0) ms (win true)
- GATE iq3-cache128-prefetch12 at 16 GiB against the best of iq3-cache128: passed: decode 18.29 (17.77–18.91) against iq3-cache128's 17.13 (16.92–19.05) t/s (win false), p95 77.4 (77.0–80.7) against 81.9 (81.6–85.3) ms (win true)
- INFO iq3-cache128 at 0 GiB against the best of iq3-resident, iq3-cpu-moe: not passed: decode 18.51 (18.25–18.77) against iq3-resident's 60.13 (54.37–60.46) t/s (win false), p95 78.9 (77.8–80.7) against 17.1 (16.9–20.4) ms (win false)
- INFO iq3-cache128-prefetch12 at 0 GiB against the best of iq3-cache128: not passed: decode 18.43 (18.22–18.52) against iq3-cache128's 18.51 (18.25–18.77) t/s (win false), p95 78.9 (75.5–83.6) against 78.9 (77.8–80.7) ms (win false)
- INFO q8-cache128 at 0 GiB against the best of q8-cpu-moe: passed: decode 13.00 (12.59–13.09) against q8-cpu-moe's 6.45 (6.03–6.85) t/s (win true), p95 111.9 (110.3–117.3) against 322.8 (300.6–334.1) ms (win true)
- INFO q8-cache128-prefetch12 at 0 GiB against the best of q8-cache128: not passed: decode 12.73 (12.02–12.81) against q8-cache128's 13.00 (12.59–13.09) t/s (win false), p95 116.1 (115.2–118.1) against 111.9 (110.3–117.3) ms (win false)

**Verdict** (2026-10-20):
- **Correctness gate: passed.** The tiny and `--real` IQ3 checks give "PASS: 0 failures" on fork `2da6ec661`; all 60 cache and prefetch requests of IQ3 generate the token IDs of the `-ub 21` resident control, Q8_0 with prefetch those without; no expert load failed. The router chooses every expert by construction (the remap passes its ids through, and M32 fails all checks when a slot id is off by one).
- **Performance gate, demand cache: passed.** At the 16 GiB ceiling resident IQ3 does not fit, and the cache beats `--cpu-moe`, the best arm that runs there, on both decode (every repetition at 16.9 to 19.1 tokens/s, against at most 12.6) and p95 (81.6 to 85.3 ms, against at least 123.2). Without a ceiling (INFO) resident Metal is 3.2 times faster in decode (60.1 against 18.5 tokens/s) and 13 times in prefill, and `--cpu-moe` prefills 3.3 times faster than the cache (273 against 83 tokens/s) with a lower p95 (60.9 against 78.9 ms): the cache pays off only once the experts no longer fit. Q8_0, which never fits (INFO): 13.0 against 6.5 tokens/s and 112 against 323 ms p95 for `--cpu-moe --no-repack`.
- **Performance gate, prefetch: passed by the rule, on tail latency alone.** At 16 GiB every repetition's p95 (77.0 to 80.7 ms) is below the demand cache's best (81.6 ms); decode overlaps (17.8 to 18.9 against 16.9 to 19.1 tokens/s). The margin is about 5%. It does not hold without a ceiling (p95 75.5 to 83.6 against 77.8 to 80.7 ms, INFO) or for Q8_0 (116 against 112 ms, INFO). Its third repetition at 16 GiB ran on 2026-10-20, four days after the demand cache's, since the earlier attempts slept. Prefetch reads 6.8 GB more per run without fewer misses.
- Whether the patch and the prefetch option are kept is the user's decision; this benchmark only reports.

## Stage E3: one read per expert

Phase 11, Stage E3. A side-file next to the GGUF holds each expert's gate, up and down rows in one aligned extent, and a miss reads it with one `pread`. E3 is not built on waves.

### Verdict

**All four deciding gates passed (2026-11-07), so patch 0014 stays in the series.** The gates were committed before the first timed run, and they were not moved after the results.

- **Decode at 0 GiB:** 20.30 (18.90–20.79) against 17.30 (16.94–17.42) t/s, and warm p95 70.6 (67.6–72.0) against 86.1 (80.0–88.9) ms. Both win.
- **Decode at 16 GiB:** 19.75 (18.99–21.14) against 17.08 (16.55–18.38) t/s, a win. **Warm p95 71.4 (66.9–79.3) against 79.7 (77.6–84.2) ms did not win by the rule.** The gate passes on tokens/s alone, because the decode rule takes either metric.
- **Prefill:** 106.5 against 88.4 t/s at 0 GiB, and 106.8 against 87.0 at 16 GiB.
- **Cold TTFT, not deciding:** 26.68 against 33.16 s, and 26.75 against 33.13 s.
- **The same output:** 30 of 30 side-file requests generate the token IDs of the GGUF path.
- **One read per expert:** 1,539,444 preads for 1,539,444 experts on the side-file, 3 per expert on the GGUF path. 0 load failures, and one build in all 12 runs.
- **Phase 11's targets are still far.** They are decode of at least 30 t/s, warm prefill above 273 t/s and cold TTFT under 10 s. E3's matrix has no #25294 arm, so no gate compares the two. E4's #25294 runs, a few hours earlier in another matrix, gave 351.3 and 226.7 t/s of prefill and 7.44 and 10.95 s of TTFT at 0 and 16 GiB.

**Merged in the order fork PR #15 into `feat/kolibri1` (`c0b2de9be`), then PR #72.** The README's option A (the patches by glob) and option B (the fork branch) give the same tree, `db0c6d55`.

**The runs did not use the series' current code.** They ran on `dd2040eaf` (tree `53739515`, build digest `e5dcda2e6e56`). The series now gives `c0b2de9be` (tree `db0c6d55`), which adds review fixes after the runs: at load, a refused LoRA on the routed experts, a bound on the GGUF header length that the table claims, a draft model without the cache or the side-file, a refused MoE cache and a check that the reopened file is the loader's; and a synchronize before each ubatch ([upstream.md](upstream.md)). The diff does not change the path that reads a miss. They were not timed: see "Limits".

### Results

`testdata/bench/e3.json`, `~/models/eval/expert-bench-e3/report.md`. All 12 runs have status ok. "GGUF" is `iq3-cache128`, "Side-file" is `iq3-cache128-sidefile`, and every other flag is the same. A cell is the median (min–max) over 3 repetitions, and a warm value is first the median of a run's 4 warm requests, as in Stage D. The 12 `run.json` files were recomputed independently of the report, and all 88 table cells and all GATE, INFO and PASS lines match.

| Metric | GGUF 0 GiB | GGUF 16 GiB | Side-file 0 GiB | Side-file 16 GiB |
|---|--:|--:|--:|--:|
| Load failures (decode+prefill) | 0 | 0 | 0 | 0 |
| Warm prompt t/s | 88.4 (85.0–88.8) | 87.0 (82.6–87.9) | 106.5 (105.4–107.1) | 106.8 (96.7–107.4) |
| Cold TTFT s | 33.16 (33.08–33.24) | 33.13 (33.07–33.34) | 26.68 (26.61–28.31) | 26.75 (26.62–28.36) |
| Cold decode t/s | 16.37 (15.98–17.52) | 16.42 (15.69–17.35) | 18.71 (18.13–25.17) | 17.16 (16.45–20.36) |
| Warm decode t/s | 17.30 (16.94–17.42) | 17.08 (16.55–18.38) | 20.30 (18.90–20.79) | 19.75 (18.99–21.14) |
| Warm p50 ms | 55.3 (54.5–56.5) | 56.2 (53.6–56.2) | 48.0 (47.8–51.2) | 50.3 (45.5–50.4) |
| Warm p95 ms | 86.1 (80.0–88.9) | 79.7 (77.6–84.2) | 70.6 (67.6–72.0) | 71.4 (66.9–79.3) |
| Decode ms/token | 57.84 (57.07–59.59) | 58.07 (54.69–60.99) | 49.48 (47.16–53.38) | 52.17 (51.16–54.16) |
| Wait ms/token | 14.48 (14.47–14.75) | 14.63 (14.50–15.64) | 8.85 (8.79–8.85) | 8.86 (8.84–10.51) |
| Remap ms/token | 0.683 (0.664–0.719) | 0.653 (0.640–0.675) | 0.631 (0.631–0.647) | 0.658 (0.591–0.698) |
| Other ms/token | 42.65 (41.93–44.16) | 42.76 (39.54–44.70) | 39.98 (37.75–43.90) | 42.66 (41.63–43.06) |
| Splits/token | 102.0 | 102.0 | 102.0 | 102.0 |
| Commits/token | 103.0 | 103.0 | 103.0 | 103.0 |
| Disk GB read | 448.68 (448.68–448.68) | 448.68 (448.68–448.69) | 437.19 (437.19–437.19) | 437.19 (437.19–437.23) |
| Others CPU mean % | 72.1 (61.3–86.8) | 75.5 (74.5–86.5) | 73.6 (66.1–82.1) | 65.8 (65.2–145.6) |
| Others CPU max % | 143 (117–794) | 167 (132–355) | 160 (139–222) | 140 (136–777) |
| Quiet wait s | 90.1 (60.1–90.1) | 90.1 (60.1–90.1) | 120.1 (60.1–150.1) | 90.1 (90.1–90.1) |
| Peak RSS GiB | 13.6 (13.6–13.6) | 13.6 (13.5–13.6) | 13.6 (13.6–13.6) | 13.6 (13.4–13.6) |
| Peak footprint GiB | 11.0 | 11.0 | 11.0 | 11.0 |
| Max footprint GiB | 10.9 | 10.9 | 10.9 | 10.9 |
| Page cache peak GiB | 2.7 | 2.7 | 2.7 | 2.7 |
| Page cache end GiB | 2.7 (2.7–2.7) | 2.7 (2.6–2.7) | 2.7 (2.7–2.7) | 2.7 (2.4–2.7) |
| Page-ins MiB/token | 0.064 (0.062–0.069) | 0.059 (0.059–0.265) | 0.048 (0.042–0.059) | 0.058 (0.056–0.262) |
| Decode hits | 91.0% | 91.0% | 91.0% | 91.0% |

**Gates.** Every side-file repetition must beat the GGUF path's fastest repetition, and the decode gate passes on either metric:

| Gate | Slowest side-file rep | GGUF's fastest rep | Result |
|---|--:|--:|---|
| Decode 0 GiB, t/s | 18.899 | 17.424 | win |
| Decode 0 GiB, p95 ms | 72.018 | 79.997 | win |
| Decode 16 GiB, t/s | 18.990 | 18.377 | win; the gate passes on this alone |
| Decode 16 GiB, p95 ms | 79.319 | 77.579 | no win |
| Prefill 0 GiB, t/s | 105.429 | 88.770 | win |
| Prefill 16 GiB, t/s | 96.727 | 87.886 | win |
| TTFT 0 GiB, s (info) | 28.307 | 33.076 | win |
| TTFT 16 GiB, s (info) | 28.355 | 33.072 | win |

**The 16 GiB decode pass is narrow.** Its margin is 0.61 t/s (3.3%), and the GGUF path's fastest repetition there is r1, which has that arm's lowest "other" (39.54 ms). The p95 no-win comes from one repetition, `sidefile/16g/r3` at 79.32 ms. r1 and r2 (71.39 and 66.93 ms) are below the baseline's 77.58. That repetition is also the most disturbed run (others' CPU mean 145.6, max 777). The prefill gate at 16 GiB still passes with its 96.7 t/s. No run was excluded.

### What the numbers show

These are observations. Each says what it does not establish.

**In decode, the wait fell and nothing else clearly moved.**

| Gain, ms/token | Total | Wait | Remap | Other |
|---|--:|--:|--:|--:|
| 0 GiB | 8.36 | 5.63 (67%) | 0.05 (1%) | 2.66 (32%) |
| 16 GiB | 5.90 | 5.78 (98%) | -0.01 | 0.10 (2%) |

- The columns are differences of medians, so they need not add up exactly.
- The wait's change is stable in every same-repetition pair: 5.69, 5.63 and 5.90 at 0 GiB, and 5.65, 5.80 and 5.13 at 16 GiB.
- The change in "other" is not separated from its spread. At 0 GiB the ranges overlap (41.93–44.16 against 37.75–43.90), and 2.66 is less than the side-file's own spread of 6.16. At 16 GiB the side-file's range lies inside the baseline's. Same-repetition changes are +4.18, +2.66 and +0.26, and -3.11, +1.13 and +1.64. So the data establish the wait's gain and do not establish an effect on "other".
- The plan expected up to the 14 ms wait. 8.85 ms of it remain.
- **Prefill's wait fell too, and it is about 55% of the prefill gain.** `run.json` keeps only part of the counters; each run's `stats.json` has `prefill.wait_seconds`. Median per run: 50.66 (50.57–51.07) to 37.05 (36.84–37.27) s at 0 GiB, and 50.77 (50.67–51.92) to 36.85 (36.80–38.99) s at 16 GiB, so 13.61 and 13.92 s less. Prefill time, the requests' `prompt_ms` summed per run, went from 135.01 to 110.66 s and from 135.01 to 109.86 s, so 24.35 and 25.15 s less. The wait is about 56% and 55% of that. The remaining 44 to 45% is not attributed.

**The disk bytes fit whole 16 KiB pages on the GGUF path, as arithmetic only.**

- The work is the same: 256,574 loads in every run (17,213 decode, 239,361 prefill), 190,500 needed, 173,287 hits, 1,699,840 bytes per load by the three rows' sizes.
- Side-file: 256,574 × 1,703,936 (the padded extent) is 437,185,675,264 bytes. The counter exceeds that by 0 to 43.5 MB, and by exactly 0 in `sidefile/0g/r2`.
- GGUF path: 256,574 × 1,699,840 is 436,134,748,160 bytes. The counter exceeds that by 12.547 to 12.551 GB, which is 48,902 to 48,918 bytes per load.
- From the manifest's offsets, widening each of an expert's three rows to 16 KiB page bounds adds a mean of 49,152 bytes per expert over all 19,200 experts (36,864–69,632). Widening to 4 KiB adds 12,288, far too little. So the numbers fit unaligned GGUF preads that transfer whole 16 KiB pages. The side-file's padding costs 4,096 bytes per expert against about 48,900.
- **This is a fit, not a measured mechanism.** There is no per-pread trace, and `run.json` has no per-expert load counts, so 48.9 KB can only be compared with a whole-file mean. The counter is `ri_diskio_bytesread` minus a baseline taken in the cache code, and when that baseline is taken relative to the load was not checked.

**The side-file's decode rate spreads more per request, and the cause is not shown.**

- The 25.17 t/s in the cold decode row is `sidefile/0g/r1`, whose other repetitions give 18.71 and 18.13. Next widest is `sidefile/16g`: 20.36, 17.16 and 16.45.
- The data show no position effect. Cold is always a run's first request. r1 is the fastest cold repetition in 3 of 4 cells, by only 1.15 and 0.93 t/s on the baseline, and `sidefile/16g`'s fastest is r3.
- The same token IDs in the same run differ widely: run 3's warm coding is 16.79 against its cold 25.17, while `sidefile/16g/r1` goes 17.16 cold to 21.98 warm. The 5 requests with a p50 under 42 ms are all on the side-file arm, at different positions. The GGUF arm's lowest p50 is 46.3 ms.
- Same-workload decode SD is 2.81, 3.11, 3.17 and 1.16 on the side-file (coding, research, hr, medtech) against 0.84, 0.75, 1.02 and 1.33 on the GGUF path. Wait and "other" are recorded per run, not per request, so the requests cannot be split. Page-ins correlate with decode rate at -0.42 on requests under 1,000 page-ins, which establishes nothing.

**The baseline did not move beyond its run-to-run range, with one exception and small differences.**

| `iq3-cache128`, 0 / 16 GiB | E3 | E4 | E0 |
|---|--:|--:|--:|
| Warm prompt t/s | 88.4 / 87.0 | 86.0 / 85.9 | 81.5 / 79.0 |
| Cold TTFT s | 33.16 / 33.13 | 32.53 / 32.84 | 35.05 / 35.50 |
| Warm decode t/s | 17.30 / 17.08 | 17.39 / 18.07 | 17.71 / 17.96 |
| Decode ms/token | 57.84 / 58.07 | 55.77 / 56.16 | 56.50 / 56.22 |
| Wait ms/token | 14.48 / 14.63 | 14.65 / 14.35 | 15.05 / 16.09 |
| Other ms/token | 42.65 / 42.76 | 41.07 / 41.28 | 40.90 / 39.57 |

E3's fastest warm prompt at 0 GiB (88.77) is above E4's (86.7) by 2.4%. The exception: E3's cold TTFT at 0 GiB (33.08–33.24 s) is below E0's whole range (34.38–35.71 s, 2 repetitions), by 1.1 s or more. Disk is about 448.7 GB in all three, and the token IDs of this arm in E4 and E0 equal E3's in all 5 requests of every repetition. E4's undisturbed repetitions ran with others' CPU mean 36–52, against 61–87 here. E4's and E0's `run.json` have no `server_sha256`, server path, loads or preads, so builds and load counts cannot be compared across stages from these files.

**E3's and E4's prefill gains cannot be added.** 106 and 300 t/s (waves) are the same kind of number on the same matrix args and workloads, with similar baselines, but not the same work. The side-file arm has its baseline's loads, hits and token IDs. E4's two waves arms have decode hits of 92.3%, 25.0 GB of decode demand, and token IDs equal to E3's in 0 of 5 requests in every repetition. `iq3-cache128-waves` reads 299.96 GB of disk, and `iq3-cache128-waves-buffered` a median of 33.89 GB at 0 GiB and 286.91 GB at 16 GiB. No run combines the two, and waves' code is not in E3's build. The ratios to each stage's own baseline are 1.2× (E3) and about 3.5× (E4, 301.4 against 86.0). For reference, E0's buffered arm gave 110.0 t/s at 0 GiB but 53.8 at 16 GiB, and the side-file gives 106.5 and 106.8.

**For E6's condition, the facts only.** On the side-file at 0 GiB, decode takes 49.48 ms/token: wait 8.85 (17.9%), remap 0.63 (1.3%), other 39.98 (80.8%). At 16 GiB: 52.17, wait 8.86 (17.0%), remap 0.66 (1.3%), other 42.66 (81.8%). On the GGUF path the wait is 25.0% and 25.2%. So the wait is smaller than "other" by a factor of about 4.5 to 4.8, but "other" is a remainder, not one measured cost. This build still has 102 splits per token, so it does not contain E2's change, and there are no timed E2 results here. These are E3's facts alone, not "E2 and E3". E6's decision is left open.

### Limits

- **The review fixes were not timed.** The 12 runs ran on `dd2040eaf`. `1f6ce476c`, `634e34bb4` and `4a4661220` came after them. `ba4bd5bbc` after those changes only what iOS, tvOS and visionOS compile, and `19001f2e0` only clears the side-file option when params are derived for a draft model. In `git diff dd2040eaf bf7bf16b6`, `llama_expert_read_rows` and the remap's reads do not change, but no run measures that. The synchronize before each ubatch is a no-op for a decode, which is already synchronized before its next call. A prefill's wait moves ahead of its first split, and no run measures that either.
- **The fork PR was squashed on merge.** `feat/kolibri1` has one commit for the patch, `c0b2de9be`, whose diff is patch 0014. The timed `dd2040eaf` is not an ancestor of it. It stays reachable from fork branch `feat/kolibri1-sidefile`, which has the PR's eight commits and the same tree.
- **The alignment rationale for `F_NOCACHE` is unmeasured.** What XNU's cluster layer wants comes from its source as remembered. No arm isolates alignment: the benchmark measures one aligned read against three unaligned ones as a whole.
- **Reads take the padded extent.** That is 4,096 bytes per expert more than the rows hold, on every miss.
- **The load's identity check is no security boundary.** The reviewer's finding on `gguf_head`, that a table with a valid checksum could claim the whole model and force an allocation of tens of GB, was fixed in `1f6ce476c` after the runs (control M81, [upstream.md](upstream.md)). The table's checksum has no key, and `gguf_head` has no lower bound, so a crafted table can still pass the identity check. That stays as it is, because the side-file is a local file the user wrote.
- **The decode rule passes on either metric, and the 16 GiB pass rests on tokens/s, not p95.** Whether to set a stricter bar is the owner's call.
- **One machine, 3 repetitions, a busy background.** All 12 runs started with a load average of 1.65 to 2.91 and about 4.1 GiB of swap. All r3 repetitions are the slowest block for decode ms/token and have the highest others' CPU mean, in all 4 cells. Time order and background load cannot be separated. The arms are interleaved, so both are affected.

### The side-file

- **Where:** `<GGUF>.experts`, with its manifest as `<GGUF>.experts.manifest.json`. For IQ3 that is `~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf.experts`: 32,715,735,040 bytes (30.5 GiB, 78.8 MB of it table and padding, 0.24%), sha256 `83c6c88e60e55fd517ba002e1c1cab3c1b47bea4570403e594868edef9692bfe`.
- **Rebuild:** `go run ./cmd/kolibri-sidefile write -gguf ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf -force` (48.0 s), then `verify` with the same `-gguf` (37.6 s). `write` refuses to replace a file without `-force`, and it renames only a complete file into place. It is derived data: nothing is lost if it goes, and `kolibri-models` has no master for it.
- **Layout:** a table (the format is at the top of `cmd/kolibri-sidefile/format.go` and in `src/llama-expert-sidefile.h`), then one extent per layer and expert: the rows as the GGUF has them, then zeros up to the alignment. A read takes the padded extent, so it is aligned at both ends and never meets the end of the file. Row sizes come from each tensor, because IQ3's `down` differs: 696,320 bytes against 501,760.
- **Alignment, 16 KiB:** the VM page on Apple silicon and a multiple of the SSD's 4 KiB block. No page holds two experts, so buffered reads do not drag in a neighbour and `mincore` counts per expert. As far as XNU's cluster layer goes, a direct `F_NOCACHE` read needs both the file offset and the destination aligned. That comes from its source as remembered and was not measured here.
- **One read, then copies:** a slot's three rows sit in three tensors, and they are no multiples of a block (IQ3's gate rows are 122.5 blocks of 4 KiB). So the read goes to a page-aligned staging buffer per I/O thread, and three copies take the rows out. That costs about 1.7 MB of memory per thread, and the copy's time is part of `io_seconds`.

### What is checked when

**The load checks the table. It does not check the data.** It refuses, saying what failed: bad magic, the version, the table's sha256, the layer and expert counts, each tensor's row size, type and GGUF offset, extents out of range, misaligned or overlapping, the file's size, and the GGUF's identity (its size and the sha256 of its header). A missing file fails the load: there is no fallback to the GGUF. **A flipped byte in an extent still loads, and a run that uses that expert reads the wrong bytes.** Hashing about 31 GB at every load would cost tens of seconds, so only `kolibri-sidefile verify` checks the data: it rehashes every extent and the whole file, compares every extent byte for byte with the GGUF, and accepts no field of the manifest on trust. The loader does not read the JSON manifest; the checksummed table in the side-file is what it checks.

**Concurrency needed no change.** A remap queues all of a layer's misses, and the I/O threads (default 4, as E1's curve advises) take them from one queue. `test-expert-cache` holds a load until all four threads are inside one remap's loads. Prefetch uses the same load path, so it works on both paths.

### Lossless evidence

Bit identity with `np.array_equal` against a resident run at the same ubatch and, on the GPU, the same split (`check_expert_cache.py`), never loosened. The timed runs add token IDs: 30 of 30 side-file requests equal the reference, and all 60 requests of both arms are equal, compared on the IDs themselves. The check counts below are from before the review fixes. After them the tiny check gives "PASS: 0 failures", `test-expert-cache` passes, and the real IQ3 pair has 0 differences on 450 experts of 50 layers (`test-expert-cache --sidefile`, argument 8; commit `da2f581` calls that 8 experts per layer). That pair was not rerun for the review: its output line is copied from the session into `~/models/eval/expert-bench-e3.runner/review-controls.log`, which also holds the reruns below. The 39 real checks are not recorded as repeated on `1f6ce476c`.

| Check | Result |
|---|---|
| Tiny, F32 and Q8_0, CPU and Metal: 6, 16, 64 and 384 slots, with and without prefetch and buffered reads, slots per layer, half the layers on Metal, repacking on | "PASS: 0 failures", 335 checks. The 48 side-file runs of that grid are bit-identical both to the resident run and to the GGUF path |
| Real IQ3 on Metal (`--real`, no hog): 128 and 64 slots, the same variants | "PASS: 0 failures", 39 checks. At 128 slots 7,173 and 966 preads for 7,173 and 966 experts in prefill and decode, against 21,519 and 2,898 on the GGUF path |
| Build without the option (`third_party/llama.cpp`, the waves build) | "PASS: 0 failures", 379 checks, the side-file checks skipped with a note |
| `test-expert-cache`, also in ctest | "all expert cache tests passed": the three tensors' bytes on both paths, 27 refusals by name, a read past the end, 3 and 1 preads |
| `test-expert-cache --sidefile`, the ranges from the GGUF's own header by ggml's gguf API | tiny: all 2,304 experts; IQ3: 450 experts, the first and last of each layer among them. No difference; `verify` covers all of IQ3 |
| `kolibri-sidefile verify` on IQ3 | every extent byte-identical to the GGUF |

### Negative controls

- **M50, wrong extent offsets:** the writer's table sends each expert to its neighbour's extent, with a valid checksum, so the load accepts it. All 56 side-file lines against the resident run fail (max |diff| up to 1.46), "FAIL: 139 failures", and the same-misses and pair checks fail with them.
- **M51, a corrupt manifest**, on an APFS clone of the real side-file: a flipped table byte is refused at load ("table checksum mismatch", exit 1, no text generated), and `verify` reports it. One edited extent hash in the JSON manifest: `verify` reports "layer 7 expert 11: sha256 ..., the manifest says ...", exit 1.
- **M52, the load without its checksum test:** `test-expert-cache` fails 3 checks, and the tiny check fails 2 ("FAIL: 2 failures").
- **M53, `kolibri-bench` without the side-file's eviction:** `TestEvictSidefile` fails.
- **M54, eviction works on the real file:** `shasum` left 1,820,300 of 1,996,810 pages resident (28,442 MiB), and `kolibri-evict` took them to 0. `shasum`'s read is the hazard the next section guards against.
- **M55, `kolibri-bench` without the GGUF's residency limit:** `TestEvictSidefile` fails on a resident GGUF in both arms.
- **M56, `readsCheck` accepting runs with no loads:** `TestReadsCheck` fails, with "0 preads for 0 experts" printed as PASS.
- **M57, one build's runs not compared:** with `buildCheck` accepting two digests, `TestBuildCheck` fails. With `serverBuild` hashing only the binary, `TestServerBuild` fails on a rebuilt library.
- **M58, `kolibri-bench` not reading the server's `--help`:** `TestCheckServer` fails.
- **M59, the refusal of a side-file without the cache moved back behind the load:** the tiny check fails 1 check ("FAIL: 1 failures"), the one that wants no `load_tensors:` line in the refused run's verbose log. Peak RSS of the refused run on the tiny router model, by `/usr/bin/time -l`, on `1f6ce476c` with this mutation and without it: 521 to 522 MB against 128 to 130 MB on F32, and 248 to 250 MB against 128 to 130 MB on Q8_0 (3 runs per file, 6 without). Peak memory footprint is 128 to 132 MB in all of them, so the RSS difference does not show in the footprint. Log: `~/models/eval/expert-bench-e3.runner/review-controls.log`.

### Benchmark validity

The page cache serves resident pages even with `F_NOCACHE`, and reading the file for a hash leaves it resident (M54), so an arm could read from memory and look fast. So:

- **Every run of every arm evicts the GGUF and every side-file of the matrix**, and `run.json` records it (`evict`, `evict_sidefiles`, and `sidefile` for the arm's own file). The driver stops, with no result, if a file is missing, or if the GGUF or a side-file stays more than 1% resident. `kolibri-evict` exits 0 whatever stays, and an idle Metal server left on the file keeps pages resident with no `--mlock`: a real `-ngl 99` llama-server on the tiny router GGUF kept 244 MiB of it, and the driver stopped. All 30 E4 runs show 0 pages after eviction, so the limit would not have stopped any of them. The driver does not look for stray servers that hold other files.
- **The hidden tier counts for the arm as for any other.** `peak_page_cache_mib` and `end_page_cache_mib` sum the GGUF and the side-file, with the side-file's part beside them. All arms sample the same files, so the sampling costs them the same.
- **A missing side-file is loud:** the driver stops before its first run, and a later loss fails the server's load, which fails the report. Nothing falls back to the GGUF.
- **The report proves the path:** with a side-file arm in the matrix it fails unless each run's stats name the file its flags gave and show 1 pread per expert (3 on the GGUF path) in both phases, with experts loaded in both. A build without patch 0014 writes no such counters, and 0 preads for 0 loads does not pass: E4's `iq3-cache128/0g` runs, reported against this matrix, fail.
- **The report proves one build:** every run records `server` and `server_sha256`, and the report fails if the completed runs of one build differ, or, with a side-file arm, if a run has no digest. The digest covers the binary and the libraries beside it, because the expert cache lives in `libllama` and a rebuild can leave the binary unchanged. So a resume after a rebuild or with another `-server` fails the report.
- **The driver refuses a server without the flag:** with a side-file arm in the matrix it stops before the first run if the server's `--help` does not list `--expert-cache-sidefile`, as the default `-server` does not. This holds with `-only` too, so no baseline runs on the wrong build.
- **`fits` and the quiet rule do not special-case the arm.** Its `--expert-cache 128` counts as the GGUF arm's does (a test holds them equal).
- **A smoke run came before the timing.** The exact flags with 8 generated tokens gave 5,117 and 219 preads for 5,117 and 219 experts on the side-file, 15,351 and 657 on the GGUF path, the same token IDs, and at most the table's 10 pages of the side-file in the page cache. Its speed was not read.

**What the 12 runs show about these guards:**

- **Status and sleep:** all 12 are ok, from 02:35:57 to 03:27:17 on 2026-11-07, under a `caffeinate` assertion, on AC power. `pmset -g log` has no Sleep, Wake or DarkWake inside any run's window, and the last true event is a Wake about 28 hours before the first run. Wall time from `run.json`'s `started` to the driver log's "ok" exceeds `load_seconds` plus each request's `prompt_ms` and `predicted_ms` by about 3.4 to 9.0 s per run, so no gap suggests a clock jump. The log has whole seconds, which can add up to 1 s to that.
- **Eviction:** in all 12 runs the GGUF and the side-file show 0 pages resident after eviction. Before it, the GGUF had 576 to 178,332 of 2,170,182 pages and the side-file 0 or 10 of 1,996,810. Every location reads "(home volume)".
- **The hidden tier:** `peak_sidefile_cache_mib` is 0 everywhere. `end_sidefile_cache_mib` is 0.156 MiB (10 pages) on 5 of 6 side-file runs and 0 on `sidefile/16g/r3`.
- **The build:** all 12 runs carry `server_sha256` `e5dcda2e6e56...` and one server path, and the build on disk still gives that digest after the runs.
- **The side-file:** after the runs its streamed sha256 and its size equal the manifest's. It was hashed with `F_NOCACHE`, and residency was not checked afterwards. Every benchmark run evicts it anyway.
- **Disturbance stays in the results.** Peak others' CPU is 117 to 222 in 9 runs. The other three: `sidefile/16g/r3` (145.6 mean, 777 max; warm hr decode 13.46 t/s, p95 158.1 ms, 138,796 page-ins; 43.5 MB extra read), `iq3-cache128/0g/r3` (86.8, 794; warm research 3,369 page-ins) and `iq3-cache128/16g/r3` (86.5, 355; warm hr 18,502 and medtech 3,705 page-ins). `iq3-cache128/16g/r2` also has 3,005 page-ins in warm coding, with normal others' CPU.
- **Activity during the runs is in the log, its effect is not.** `pmset` shows a user-activity assertion and a 3 s `cloudd` network task at 03:12:47–50 (run 9), HID input near 02:59 and 03:12, and three other `caffeinate` clients that ended at 03:19:18, 03:20:26 and 03:24:21 (runs 11 and 12). What those clients ran is not in the log, so no link to a specific slowdown is established.

### Reproduce

The matrix `testdata/bench/e3.json` holds 12 runs: `iq3-cache128` and `iq3-cache128-sidefile`, with every other flag the same, at 0 and 16 GiB, 3 repetitions each. Decode and prefill of the side-file against the GGUF path decide at both ceilings, by `kolibri-bench`'s existing rule. The decode gate is the Stage D rule, so a warm p95 win alone passes it, even with a lower median decode rate, and the report's "(win true)" and "(win false)" show which one it was. TTFT is info only. Token IDs must be exactly equal.

The result above came from `third_party/llama.cpp-e3` at `dd2040eaf`. A build of today's `feat/kolibri1-sidefile` has another digest, so its runs are a new result directory, not a resume: the report fails on two builds.

```sh
go run ./cmd/kolibri-sidefile verify -gguf ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf
go run ./cmd/kolibri-bench -matrix testdata/bench/e3.json -out ~/models/eval/expert-bench-e3 \
  -server third_party/llama.cpp-e3/build/bin/llama-server
.venv/bin/python tools/gguf/check_expert_cache.py --llama-cpp third_party/llama.cpp-e3
.venv/bin/python tools/gguf/check_expert_cache.py --llama-cpp third_party/llama.cpp-e3 --real ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf
```

## Stage E4: waves for prefill

Phase 11, Stage E4. `--expert-cache-waves` lifts the cache's ubatch cap in prefill. **The option exists only on fork branch `feat/kolibri1-waves` (commit `90311e5a2`, fork PR #14, open and not merged). It is not in the patch series and not on `feat/kolibri1`.**

### Verdict

**The deciding gates failed at 0 GiB (2026-11-06), so patch 0013 left the series.**

- **At 0 GiB** `iq3-cache128-waves` lost to #25294 (`iq3-moestream128`) on warm prefill (301.4 against 351.3 t/s) and on cold TTFT (8.91 against 7.44 s).
- **At 16 GiB** it passed both (301.8 against 226.7 t/s; 8.93 against 10.95 s).
- **Decode-not-slower** passed at both ceilings.
- **The stop rule:** a deciding gate failed, so the failure is documented here, the number 0013 is retired, and the code stays on the fork branch. The gates were committed before the first timed run, and they were not moved after the results.
- **Buffered waves decide nothing.** `iq3-cache128-waves-buffered` at 0 GiB was in the matrix before the first run, non-deciding: 476.1 t/s and 6.61 s, ahead of #25294 on both. The same arm at 16 GiB was added after the first 27 results (commit `f47e8a0`, three runs), because they had shown a split: 217.2 against 226.7 t/s and 12.31 against 10.95 s, behind #25294 on both.
- **So no single waves mode wins at both ceilings in these runs.** Buffered wins at 0 GiB, `F_NOCACHE` at 16 GiB. Whether to pre-register a new gate, or a choice per ceiling, is open and is the user's decision. This document does not recommend adopting the code.
- **Not built:** layer streaming, E4's other variant, so the two were never compared. Prefill admission was not implemented either. Both stay open in [PLAN.md](../PLAN.md).

### Results

`testdata/bench/e4.json`, `~/models/eval/expert-bench-e4/report.md`. All 30 runs have status ok.

| Config | Ceiling | Status | Warm prompt t/s | Cold TTFT s | Warm decode t/s | Disk GB read | Others CPU mean % | Others CPU max % | Quiet wait s | Peak RSS GiB | Peak footprint GiB | Page cache peak GiB | Load failures (decode+prefill) |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| iq3-cache128 | 0 GiB | 3 ok | 86.0 (51.5–86.7) | 32.53 (32.42–51.16) | 17.39 (15.16–18.38) | 448.68 (448.68–448.69) | 51.6 (35.7–383.6) | 227 (87–1223) | 60.1 (60.1–210.1) | 13.6 (13.5–13.6) | 11.0 (11.0–11.0) | 2.7 (2.7–2.7) | 0+0 |
| iq3-cache128 | 16 GiB | 3 ok | 85.9 (73.0–86.1) | 32.84 (32.08–35.71) | 18.07 (16.90–18.93) | 448.68 (448.68–448.71) | 48.4 (41.8–155.2) | 159 (148–675) | 90.1 (60.1–210.1) | 13.6 (13.6–13.6) | 11.0 (11.0–11.0) | 2.7 (2.7–2.7) | 0+0 |
| iq3-cache128-waves | 0 GiB | 3 ok | 301.4 (246.6–302.0) | 8.91 (8.90–12.51) | 18.58 (15.50–19.64) | 299.96 (299.96–299.96) | 45.0 (33.1–417.0) | 138 (133–967) | 90.1 (90.1–180.1) | 13.6 (13.6–13.6) | 11.0 (11.0–11.0) | 2.7 (2.7–2.7) | 0+0 |
| iq3-cache128-waves | 16 GiB | 3 ok | 301.8 (301.0–301.9) | 8.93 (8.86–9.01) | 18.86 (17.47–20.79) | 299.96 (299.96–299.97) | 44.9 (42.8–52.9) | 130 (100–147) | 90.1 (60.1–300.3) | 13.6 (13.6–13.6) | 11.0 (11.0–11.0) | 2.7 (2.7–2.7) | 0+0 |
| iq3-cache128-waves-buffered | 0 GiB | 3 ok | 476.1 (400.4–483.8) | 6.61 (6.17–6.80) | 24.10 (22.46–24.28) | 33.89 (33.42–72.88) | 69.0 (45.2–96.5) | 213 (129–226) | 60.1 (60.1–60.1) | 13.4 (13.3–13.4) | 11.0 (11.0–11.0) | 27.2 (24.6–27.4) | 0+0 |
| iq3-cache128-waves-buffered | 16 GiB | 3 ok | 217.2 (203.6–222.9) | 12.31 (11.87–12.79) | 18.30 (18.26–18.71) | 286.91 (285.94–287.78) | 44.7 (35.9–96.3) | 96 (75–296) | 60.1 (60.1–720.4) | 13.3 (13.1–13.3) | 11.0 (11.0–11.0) | 15.9 (14.8–16.1) | 0+0 |
| iq3-cpu-moe | 0 GiB | 3 ok | 281.2 (166.3–311.9) | 12.16 (8.63–12.21) | 15.34 (15.11–15.68) | 0.39 (0.00–15.56) | 69.9 (26.0–140.4) | 628 (77–909) | 60.1 (60.0–240.2) | 33.5 (32.9–34.1) | 1.1 (1.0–1.1) | 33.0 (32.8–33.1) | 0+0 |
| iq3-cpu-moe | 16 GiB | 3 ok | 26.6 (15.3–55.5) | 129.87 (59.14–141.94) | 8.68 (5.89–11.86) | 240.29 (88.36–328.59) | 55.8 (39.0–212.8) | 210 (205–1109) | 60.1 (60.1–90.1) | 19.2 (17.6–22.4) | 1.1 (1.1–1.1) | 20.4 (18.7–23.3) | 0+0 |
| iq3-moestream128 | 0 GiB | 3 ok | 351.3 (321.8–351.4) | 7.44 (6.89–7.72) | 23.98 (22.13–24.23) | n/a | 42.0 (32.1–50.1) | 115 (111–122) | 60.1 (60.1–90.1) | 14.0 (14.0–14.0) | 14.0 (14.0–14.0) | 24.7 (21.0–24.9) | n/a |
| iq3-moestream128 | 16 GiB | 3 ok | 226.7 (136.1–232.9) | 10.95 (10.57–11.19) | 18.52 (8.14–19.39) | n/a | 38.2 (37.7–526.7) | 118 (84–1010) | 60.1 (60.1–60.1) | 14.0 (13.6–14.0) | 14.0 (14.0–14.0) | 13.4 (6.9–14.8) | n/a |

**Aggregation.** Each repetition's value is the median over its 4 warm requests; cold TTFT is the single cold request. A cell is the median (min–max) over the 3 repetition values, as in Stage D. Pooling per request gives other numbers (waves at 16 GiB: 301.1, 240.2–306.1), so compare only with this rule.

`iq3-moestream128` writes no `stats.json`, so its disk read and load failures are unknown.

Peak footprint here is the sampled column, which moves in 1 GiB steps at this size. `max_phys_footprint` from `stats.json` is 10.90 GiB for the capped cache and 10.98 GiB for waves, about 90 MiB more.

**Gates, as the report prints them:**

- GATE (prefill) iq3-cache128-waves at 0 GiB against the best of iq3-cache128, iq3-moestream128: not passed: warm prompt 301.4 (246.6–302.0) against iq3-moestream128's 351.3 (321.8–351.4) t/s
- GATE (prefill) iq3-cache128-waves at 16 GiB against the best of iq3-cache128, iq3-moestream128: passed: warm prompt 301.8 (301.0–301.9) against iq3-moestream128's 226.7 (136.1–232.9) t/s
- GATE (ttft) iq3-cache128-waves at 0 GiB against the best of iq3-cache128, iq3-moestream128: not passed: cold TTFT 8.91 (8.90–12.51) against iq3-moestream128's 7.44 (6.89–7.72) s
- GATE (ttft) iq3-cache128-waves at 16 GiB against the best of iq3-cache128, iq3-moestream128: passed: cold TTFT 8.93 (8.86–9.01) against iq3-moestream128's 10.95 (10.57–11.19) s
- GATE (decode-not-slower) iq3-cache128-waves at 0 GiB against the best of iq3-cache128: passed: decode median 18.58 against iq3-cache128's slowest of 15.16 (median 17.39, 15.16–18.38) t/s
- GATE (decode-not-slower) iq3-cache128-waves at 16 GiB against the best of iq3-cache128: passed: decode median 18.86 against iq3-cache128's slowest of 16.90 (median 18.07, 16.90–18.93) t/s
- INFO (prefill) iq3-cache128-waves at 0 GiB against the best of iq3-cpu-moe: not passed: warm prompt 301.4 (246.6–302.0) against iq3-cpu-moe's 281.2 (166.3–311.9) t/s
- INFO (prefill) iq3-cache128-waves at 16 GiB against the best of iq3-cpu-moe: passed: warm prompt 301.8 (301.0–301.9) against iq3-cpu-moe's 26.6 (15.3–55.5) t/s
- INFO (prefill) iq3-cache128-waves-buffered at 0 GiB against the best of iq3-cache128, iq3-moestream128: passed: warm prompt 476.1 (400.4–483.8) against iq3-moestream128's 351.3 (321.8–351.4) t/s
- INFO (ttft) iq3-cache128-waves-buffered at 0 GiB against the best of iq3-cache128, iq3-moestream128: passed: cold TTFT 6.61 (6.17–6.80) against iq3-moestream128's 7.44 (6.89–7.72) s
- INFO (prefill) iq3-cache128-waves-buffered at 16 GiB against the best of iq3-cache128, iq3-moestream128: not passed: warm prompt 217.2 (203.6–222.9) against iq3-moestream128's 226.7 (136.1–232.9) t/s
- INFO (ttft) iq3-cache128-waves-buffered at 16 GiB against the best of iq3-cache128, iq3-moestream128: not passed: cold TTFT 12.31 (11.87–12.79) against iq3-moestream128's 10.95 (10.57–11.19) s
- PASS expert load failures: 0 in the 18 of 18 cache runs that completed and wrote their stats

The report's equivalence line is INFO (`exact: false`): 0 of 120 requests of the capped-cache, waves and #25294 arms generate `--cpu-moe`'s token IDs. See "Lossless".

The rules (`evalGate`, fixed before the runs): prefill passes if every repetition's warm prompt rate is above the best baseline's fastest repetition; TTFT if every repetition's cold TTFT is below the best baseline's lowest; decode-not-slower if the candidate's median warm decode is at least the baseline's slowest repetition. The prefill and TTFT baselines are Stage D's cache and #25294, rerun in the same session.

An independent recomputation from `run.json`, using the gate rule as `report.go`'s comment states it, reproduced all 220 table cells of `report.md`, all 12 GATE and INFO lines, the equivalence line and the PASS line.

The two decode lines were relabelled after the runs, because they printed the median as "slowest". This is display only. `kolibri-bench -report` regenerated `report.md` from the same `run.json`, and only those two lines changed; the first report is kept as `~/models/eval/expert-bench-e4.runner/report.out`.

### How waves work

- **A ubatch is not capped.** With `--expert-cache-waves` (`llama_expert_cache_params.waves`), a ubatch whose experts can exceed a layer's slots runs its expert matmuls once per wave, and the per-wave outputs are summed. A decoded token touches 6 experts and takes the existing remap path.
- **Built passes.** The graph is fixed at build and holds `ceil(n_touch / (slots - k))` passes, with `n_touch = min(n_expert, n_tokens * k)` and `k` the experts per token. At ubatch 512 that is 4 passes at 128 slots and 7 at 64. Every built pass runs its GEMMs, also when a plan leaves it empty.
- **Wave size.** Waves stay at the cap of `(slots - k) / 2` experts, 61 at 128 slots, so a wave, the next wave's preloads and up to `k` parking slots fit at once. They grow only if a union would otherwise need more waves than were built. The union is sorted least used first, so the most used experts are what the slots hold afterwards.
- **Parking.** Pairs of other waves are parked on resident slots, distinct within a token's `k`, and masked to 0, so the sum adds every pair's own result once.
- **Loading.** Each stage loads its wave, queues the next wave's loads behind it, and waits for its own wave. It pins the next wave's resident experts only as far as room allows, so the current wave always has a victim.
- **Too few slots.** Waves need at least `3k` slots per layer, 18 on Kolibri. Below that the cache keeps the cap and warns: "waves need 18 slots per layer, the fewest are N: the ubatch stays capped".
- **Counters.** `expert_cache.waves` says whether waves are on. Per phase, `waves` counts passes staged with experts of their own, `waves_built` the passes the graph ran, and `wave_parked` the parked pairs. Decode reports 0 for all three.
- **Reads per run.** `F_NOCACHE` waves read 299.96 GB, against 448.68 GB for the capped cache. Prefill demand is 266.58 GB per run (156,827 misses of 1,699,840 B), against Stage D's 406 GB.

### Lossless

Bit identity with `np.array_equal` against a resident run at the same ubatch and, on the GPU, the same split (`check_expert_cache.py`):

- **Tiny, `--waves-only`:** F32 and Q8_0, CPU and Metal; 16, 18, 24, 64 and 384 slots, without prefetch, with prefetch, and buffered; plus grown waves (4 tokens at 18 slots). 216 checks, "PASS: 0 failures".
- **Real IQ3, `--real --waves-only`, on Metal:** 36 checks, "PASS: 0 failures" (37 PASS lines with that summary line):
  - 128 and 64 slots at ubatch 128;
  - grown waves, 18 slots at ubatch 6;
  - 128 slots at ubatch 512, the benchmark's own, in 3 variants (no prefetch, prefetch 12, buffered): logits of 544 positions, 11,566 experts needed, 195 of 200 passes used, 460,800 pairs parked.
- **Full tiny check on the waves build:** 379 checks, 0 failures, on 5 of 6 runs. In the sixth a pre-existing flake stopped the script after 377 checks with no failure: its odd-filename `llama-completion` call has no fixed seed and can crash with `UnicodeDecodeError` on random output. The logs of 4 clean runs are kept in `~/models/eval/expert-bench-e4.runner/checks/`. The other clean run's and the stopped run's logs were overwritten; `flake.txt` there holds what the session's tool output preserved of them, the traceback included.

The 30 benchmark runs do not test lossless. E4 has no resident arm, and its equivalence entry is `exact: false`. What they do show: waves, buffered waves and #25294 generate identical tokens to each other, 128 of 128 on all 5 request kinds, across all repetitions and ceilings (90 requests).

The ubatch 512 case is on real routing, where a replay put 6 to 35% of the layers' unions above 244 experts, so waves should grow there. The check has no counter for grown waves, so it does not confirm that any grew in those 512 tokens.

### Negative controls

Both mutate only `src/llama-expert-slots.cpp` and were reverted (`cmp` identical to the saved copy, then a forced rebuild and a rerun of the checks).

| Control | `test-expert-cache` | Bit identity (tiny `--waves-only`) |
|---|---|---|
| **M48:** `wave_mask` always writes 1, so parked pairs are kept | fails: "FAIL mask[i] == (in ? 1.0f : 0.0f)", 17910 checks failed | fails: "FAIL: 60 failures", max \|diff\| up to 1.96 |
| **M49, plain:** the current wave's pins are cleared before the preload, so the next wave's preload may evict its experts | fails: 2 checks failed (`hits + hits_loading + misses == needed`, no load failure) | **passes**, "PASS: 0 failures": the wait loop reloads an evicted expert before the wave's slot ids are written |
| **M49, strong:** after the wave's ids are written, the next wave's load overwrites a slot of this wave | fails: 12530 checks failed | fails: "FAIL: 45 failures", max \|diff\| up to 1.11 |

M49's plain form fails only the unit test. Its strong form was added because the plain form did not reach bit identity. Logs: `m48.log`, `m49.log` and `m49b.log` in `~/models/eval/expert-bench-e4.runner/checks/`.

### Validity

- **Sleep and power.** No run overlaps a true system sleep. `pmset -g log` has its last Sleep and Wake on Nov 5 (22:11:47, 22:17:13) and none on Nov 6, and `kern.sleeptime` and `kern.waketime` agree. It has no "Using Batt" line on Nov 6. The display was off from 21:52:20 to 23:29:49, which is not sleep.
- **A disturbed block, no verdict depends on it.** Six consecutive runs from 22:03:40 to 22:48:01 have others' CPU at or above 100% on average (mean/max): moestream128/16g/r1 527/1010, cpu-moe/0g/r2 140/909, cpu-moe/16g/r2 213/1109, cache128/0g/r2 384/1223, cache128/16g/r2 155/675, waves/0g/r2 417/967. Their warm prompt is 0.58 to 0.85 of the arm's median. The process responsible is not identified. Recomputing all 12 gates without those 6 runs flips none: waves at 0 GiB still fails prefill (301.4, 302.0 against 351.4) and TTFT (8.90, 8.91 against 6.89).
- **Swap.** At run start it went 1792 MiB (through 22:03), 2665, 9046 (22:32), about 7000 (through 23:27), then 4718 to 4414 (23:31 onward). Inside runs `samples.txt` reaches 21491 MiB in cpu-moe/16g/r2 and 24343 MiB in cache128/0g/r2.
- **Memory pressure** reached level 4 in single samples: cpu-moe/16g/r2 at 22:23:20, and cache128/0g/r2 at 22:32:17 and 22:33:48. Level 2 ran 22:21:29 to 22:22:13, 22:22:43 to 22:23:58 and 22:26:51 to 22:28:39 in cpu-moe/16g/r2, 22:32:19 to 22:33:52 in cache128/0g/r2, and 22:04:54 to 22:06:33 in moestream128/16g/r1. In cache128/0g/r2 there is no hog and the server's RSS is at most 13.5 GiB, yet swap reached 23.8 GiB, so something else held the memory; what is not recorded.
- **`--cpu-moe` at 16 GiB varies widely** also below 100% others' CPU: r1 26.6 t/s and 129.87 s (others 56%), r3 55.5 t/s and 59.14 s (others 39%), with page cache peaks of 18.7 and 23.3 GiB. Only INFO lines use this arm.
- **Page-cache arms have a lower r1 at 0 GiB** with no others' CPU signal: moestream128 321.8 against about 351, buffered waves 400.4 against 476 to 484, with a lower page cache peak in r1 for both. Buffered r1 also read 72.88 GB against about 33.6. Repetition order and swap level are confounded here.
- **Disk GB read does not measure `--cpu-moe`'s load at 0 GiB.** r3 reports 0.00 GB and 27985 process page-ins, although `kolibri-evict` emptied the file and page cache was 33.1 GiB by the first sample.
- **The three buffered 16 GiB runs** ran consecutively in a later session (23:30 onward), at about 4.5 GiB of swap, with r3 after a 720 s quiet wait. They are not interleaved with the other arms. All other result rows in `report.md` match the report saved after the first 27 runs exactly.

### Observations

These are observations on open questions. None of them changes a gate.

**The size of the 0 GiB loss moves with the baseline's state.**
- The loss holds repetition by repetition: #25294's slowest (321.8) beats waves' fastest (302.0) by 6.6%, and its fastest (351.4) beats waves' r1 (301.4) by 16.6%.
- E0's and E4's #25294 repetitions span 274 to 351 t/s, while waves' two clean repetitions stay at 301 to 302. Across those 5 repetitions the warm prompt rises with the page cache peak (274, 292, 322, 351, 351 t/s at 17.2, 18.6, 21.0, 24.9, 24.7 GiB), and others' CPU does not track it. That is n = 5 over two sessions and does not establish a cause. Disk state is not recorded in either benchmark.
- "The same binary" cannot be verified: E0's server is gone, and `server.log` has no build line. The reference build was rebuilt on Nov 6 from `8a8ddf827`, and macOS changed between the two (27.2 was installed on 2026-10-29). Other arms moved too: `--cpu-moe` 218.7 to 281.2 (+29%), #25294 +24%, the capped cache 81.5 to 86.0 (+5.5%).
- Against E0's #25294 (median 283.2 t/s) the gate would still not pass as measured: the rule needs every repetition above the baseline's fastest (292.1), and waves' r2 is 246.6. TTFT would fail either way (8.90 against E0's lowest, 8.18). The gate stays against the baseline of its own session.

**`F_NOCACHE` waves read at about 7 GB/s, and what limits them is not established.**
- At 16 GiB: 266.58 GB of prefill demand over 37.6 to 37.8 s of summed prompt time, 7.05 to 7.09 GB/s. Only this per-run average exists, because per-request bytes are not recorded.
- [ssd-curve.md](ssd-curve.md) gives 9.45 GB/s (three reads) or 12.9 GB/s (one read) at the arm's QD 4 and 1,699,840 B, and 13 to 14 GB/s at saturation for larger reads or QD 16.
- The I/O threads are busy about 70% of prefill time (`io_seconds` 106.0 to 106.8 over 4 threads), which is 10.0 GB/s while busy and 0.677 ms per expert, close to the curve's three-read QD 4 p50 of 0.68 ms. The graph waits 16.1 to 16.3 s, 43% of prefill wall time.
- So the data shows neither saturation nor a full queue.
- Clean `F_NOCACHE` prefill reads did not drift from 21:55 to 23:20. The run data records no free-disk figure, so no correlation with free disk can be computed. The pre-flight's `df -h` at 21:27:37 showed 477 GiB free and the end status (`df -g`) 439 GiB, and the 20 MiB of results do not account for that; the cause is not established. Both outputs, with the status snapshots in between, are saved in `~/models/eval/expert-bench-e4.runner/checks/preflight.txt`.

**The low ends of two ranges.**
- `--cpu-moe` at 0 GiB, 166.3 t/s, is r2. It is not a first repetition, and all three repetitions reach about 33 GiB of page cache by the first sample. The change happens inside r2: page cache falls from 31.0 to 23.8 GiB during warm coding, and process page-ins are 14.3 GiB against about 0.45 GiB in r1 and r3. It follows moestream128/16g/r1 after 240.2 s of quiet wait, and its own others' CPU is 140/909.
- Waves at 0 GiB, 246.6 t/s, is r2 at 22:46, inside the disturbed block (others' CPU 417/967, 180 s of quiet wait before it, 300 s after). Page cache is flat at 2.7 GiB in all three repetitions. The next run, waves/16g/r2, is clean at 301.8. Contention fits, but the data does not say whether it was CPU, disk or memory.

**Others' CPU correlates with the results, swap level does not.** Spearman over the 30 runs, each normalised to its arm's median: others' CPU against prefill −0.70, decode −0.49, TTFT +0.57; swap at start against prefill +0.18, decode −0.19, TTFT +0.05. `F_NOCACHE` arms are unaffected by swap level once settled (waves/16g: 301.0 at 1792 MiB, 301.9 at 6990 MiB). Among the 24 runs below 100% others' CPU, swap against prefill is +0.48, but there the page-cache arms are higher in r2 and r3, confounded with repetition order.

**Tokens against `--cpu-moe`.**
- E4's `--cpu-moe`, capped-cache and #25294 tokens equal E0's exactly. As in E0, #25294 first differs from `--cpu-moe` at tokens 15, 13, 14 and 15, and the capped cache at 72, 12, 27 and 15. Waves agree with `--cpu-moe` at 126, 127, 32 and 15 of 128 positions for coding, research, hr and medtech.
- Ubatch does not explain waves or #25294 against `--cpu-moe`, since neither sets `-ub`.
- Whether the split contributes stays open. `--cpu-moe` and waves have the same decode split count (102 splits, 103 commits per token), but #25294 records no splits, prefill splits are not recorded, and equal counts do not show equal split points.
- The other known difference is E0's, GPU against CPU expert matmuls. The tokens fit it (waves also equal E0's #29887 arm and Stage D's resident run at the default ubatch), but no run isolates it.
- Ubatch does fit the capped cache against waves: the same decode split counts, ubatch 21 against uncapped, first differences 15, 15, 12, 14 and 22, exactly those of Stage D's resident against resident `-ub 21`. The capped cache against `--cpu-moe` differs in both ubatch and device, so those two cannot be separated.

### Deviations and known limits

- **Empty passes still run full GEMMs,** because a static graph cannot skip them. A replay of recorded routing estimated 3 to 13% of passes empty at ubatch 512 and 29 to 40% at ubatch 128. The stats of all 12 waves runs count 3606 of 3920 prefill passes used. The time these passes cost was not measured.
- **Wave growth deviates from the older design,** which said `n_waves = ceil(touch / cap)`, 7 passes at 128 slots. #25294 uses that rule (`src/llama-graph.cpp:2396`). The fork's rule builds 4 and grows waves when a union needs it, and then only part of the next wave can load ahead.
- **Repetitions were raised from 2 to 3 before the runs.**
- **A fifth arm was added before the runs.** The design listed four arms, `iq3-moestream128` among them. `iq3-cache128-waves-buffered` at 0 GiB was added in review, non-deciding, and is in the first commit of `e4.json`.
- **16 GiB of buffered waves was added after results,** as stated in the verdict.
- **Layer streaming was not compared.** E4's plan named two variants, and only waves were built.
- **Admission was not implemented,** so nothing gates whether prefill flushes the decode working set.

### Reproduce

`testdata/bench/e4.json` stays as the record. Its waves arms need a build of `feat/kolibri1-waves`: on a series build `llama-server` exits with "error: invalid argument: --expert-cache-waves", `kolibri-bench` records "load failed", and the gates give no verdict. `check_expert_cache.py` skips its wave checks on a series build with a note, and `--waves-only` is an error there. It exits with an error if the checkout's `llama.h` and its built `libllama` disagree about the field.

The #25294 server is built as in [upstream.md](upstream.md#reproduce).

```sh
git -C third_party/llama.cpp fetch origin feat/kolibri1-waves
git -C third_party/llama.cpp switch feat/kolibri1-waves   # 90311e5a2
cmake --build third_party/llama.cpp/build --target llama llama-server llama-completion test-expert-cache -j
go run ./cmd/kolibri-bench -matrix testdata/bench/e4.json -out ~/models/eval/expert-bench-e4 \
  -servers pr25294=third_party/llama.cpp-pr25294/build/bin/llama-server
.venv/bin/python tools/gguf/check_expert_cache.py --llama-cpp third_party/llama.cpp --waves-only
```

## Stage E2: GPU remap

Phase 11, Stage E2. `--expert-cache-gpu-remap` removes the scheduler split and CPU sync at every expert layer that hits. **The option exists only on fork branch `feat/kolibri1-gpu-remap` (commit `0bb0f5444`, fork PR [#16](https://github.com/CWBudde/llama.cpp/pull/16), draft, open and not merged). It is not in the patch series and not on `feat/kolibri1`.**

### Verdict

**The deciding gate failed at 16 GiB and passed at 0 GiB (2026-11-07), so patch 0015 is removed and its number is retired.**

- **The rule.** Stage D's, in `evalGate`: the candidate wins a ceiling if its slowest repetition beats the best baseline's fastest repetition on decode, or if its highest p95 is below the baseline's lowest. The baseline is whichever of `iq3-cache128` and `iq3-cache128-prefetch12` has the better median decode at that ceiling. The gates in `testdata/bench/e2.json` were last changed in `4dca7b2` (03:41), before the first timed run (04:23), and they were not moved after the results.
- **At 16 GiB: not passed on either metric.** Against `iq3-cache128-prefetch12`, decode is 18.94 against 19.00 t/s, 0.054 t/s (0.29%) short. p95 is 79.33 against 78.58 ms, 0.75 ms over.
- **At 0 GiB: passed on both.** Against `iq3-cache128`, decode is 19.02 against 18.91 t/s, 0.110 t/s (0.58%) ahead. p95 is 79.67 against 80.75 ms, 1.08 ms under.
- **The stop rule** ([PLAN.md](../PLAN.md), Phase 11): a stage whose gate fails is documented, and its code stays out of the patch set. Both ceilings are deciding, so one passing does not save the other, and the margin does not enter the rule. Patch 0015 is removed, the number 0015 is retired, and the code stays on the fork branch.
- **Lossless: holds.** 30 of 30 benchmark requests generate the token IDs of `iq3-resident-ub21`, and `check_expert_cache.py --real --gpu-remap-only` says "PASS: 0 failures", with logits bit-identical to the unsplit resident run.
- **Splits: as designed.** 2 per token against 102 for the two CPU remap arms, in every run.
- **The earlier estimate was not met.** It was unmeasured: about 44 ms/token and 23 t/s. The run measured 52.10 ms/token at both ceilings and a warm decode of 19.61 and 19.96 t/s (medians). See "Estimate and measurement".
- **Not run: a stack benchmark of E2 and E3.** It was conditional on E2 passing.
- This document does not recommend adopting the code.

### Results

`testdata/bench/e2.json`, `~/models/eval/expert-bench-e2/report.md`, finished at 05:42 on 2026-11-07. All 19 runs have status ok, with 0 restarts and 0 expert load failures in the 18 cache runs.

Per run, from `run.json` and `stats.json`. Decode and p95 are the median over the 4 warm requests. The bold rows set the 16 GiB result.

| Run | Decode t/s | p95 ms | others_cpu mean/max % | Splits/token | Commits/token | Wall ms/token | Wait | Remap | Other |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cache128/0g/r1 | 17.50 | 84.3 | 60/126 | 102 | 103 | 58.28 | 14.69 | 0.703 | 42.89 |
| cache128/0g/r2 | 17.78 | 80.7 | 46/123 | 102 | 103 | 56.82 | 14.39 | 0.674 | 41.76 |
| cache128/0g/r3 | 18.91 | 84.6 | 36/85 | 102 | 103 | 52.98 | 14.10 | 0.600 | 38.27 |
| cache128/16g/r1 | 18.46 | 80.6 | 46/131 | 102 | 103 | 54.31 | 14.56 | 0.676 | 39.08 |
| cache128/16g/r2 | 17.32 | 84.7 | 32/91 | 102 | 103 | 56.66 | 14.35 | 0.669 | 41.63 |
| cache128/16g/r3 | 19.49 | 76.4 | 40/87 | 102 | 103 | 52.59 | 14.16 | 0.646 | 37.78 |
| gpuremap/0g/r1 | 19.02 | 78.5 | 66/127 | 2 | 33.97 | 52.84 | 15.35 | 0.197 | 37.30 |
| gpuremap/0g/r2 | 20.29 | 78.6 | 28/75 | 2 | 33.97 | 52.10 | 15.11 | 0.192 | 36.79 |
| gpuremap/0g/r3 | 19.61 | 79.7 | 40/92 | 2 | 33.97 | 51.51 | 15.15 | 0.190 | 36.17 |
| **gpuremap/16g/r1** | **18.94** | **79.3** | **50/123** | 2 | 33.97 | 52.82 | 15.41 | 0.197 | 37.21 |
| gpuremap/16g/r2 | 19.96 | 78.2 | 44/164 | 2 | 33.97 | 52.10 | 15.28 | 0.192 | 36.63 |
| gpuremap/16g/r3 | 21.69 | 71.4 | 34/85 | 2 | 33.97 | 49.17 | 15.18 | 0.187 | 33.81 |
| prefetch12/0g/r1 | 16.26 | 85.7 | 52/172 | 102 | 103 | 60.81 | 14.54 | 0.990 | 45.28 |
| prefetch12/0g/r2 | 17.33 | 80.4 | 46/130 | 102 | 103 | 58.13 | 14.29 | 1.045 | 42.79 |
| prefetch12/0g/r3 | 17.42 | 83.5 | 40/123 | 102 | 103 | 56.39 | 14.33 | 1.057 | 41.00 |
| **prefetch12/16g/r1** | 18.47 | **78.6** | **41/94** | 102 | 103 | 56.70 | 14.56 | 0.999 | 41.14 |
| prefetch12/16g/r2 | 18.54 | 82.6 | 46/132 | 102 | 103 | 56.10 | 14.25 | 1.011 | 40.84 |
| **prefetch12/16g/r3** | **19.00** | 78.8 | **42/255** | 102 | 103 | 53.90 | 14.37 | 0.999 | 38.53 |
| resident-ub21/0g/r1 | 60.41 | 17.0 | 40/128 | 2 | 3 | 16.54 | 0.00 | 0.000 | 16.54 |

Wall, wait, remap and other are ms per decoded token. Other is the remainder, so on a cache arm it holds compute as well as overhead.

Gates, by the rule above. The 16 GiB row for `iq3-cache128` is a counterfactual; that baseline was not selected.

| Ceiling | Best baseline (median decode) | Decode: candidate min against baseline max | p95: candidate max against baseline min | Verdict |
|---|---|--:|--:|---|
| 0 GiB | `iq3-cache128` (17.7759; prefetch12 17.3259) | 19.0199 against 18.9098, win (+0.1102) | 79.670 against 80.746, win (1.076 ms under) | passed on both |
| 16 GiB | `iq3-cache128-prefetch12` (18.5366; cache128 18.4633) | 18.9427 against 18.9972, no (−0.0544) | 79.330 against 78.584, no (0.746 ms over) | **not passed on either** |
| 16 GiB, against `iq3-cache128` instead | not selected | 18.9427 against 19.4866, no | 79.330 against 76.416, no | also not passed |

As the report prints them:

- GATE iq3-cache128-gpuremap at 0 GiB against the best of iq3-cache128, iq3-cache128-prefetch12: passed: decode 19.61 (19.02–20.29) against iq3-cache128's 17.78 (17.50–18.91) t/s (win true), p95 78.6 (78.5–79.7) against 84.3 (80.7–84.6) ms (win true)
- GATE iq3-cache128-gpuremap at 16 GiB against the best of iq3-cache128, iq3-cache128-prefetch12: not passed: decode 19.96 (18.94–21.69) against iq3-cache128-prefetch12's 18.54 (18.47–19.00) t/s (win false), p95 78.2 (71.4–79.3) against 78.8 (78.6–82.6) ms (win false)
- PASS equivalence iq3-cache128-gpuremap against iq3-resident-ub21: 30 of 30 requests generate the same token IDs

The other columns, median (min–max) over 3 repetitions from the report. Decode hits are 91.0% on both CPU remap arms and 90.0% on the GPU arm, in all their runs. Peak footprint is 11.0 GiB on every cache arm.

| Config | Ceiling | Cold TTFT s | Warm prompt t/s | Disk GB read |
|---|--:|--:|--:|--:|
| iq3-cache128 | 0 GiB | 33.03 (32.30–33.49) | 85.8 (83.9–86.2) | 448.68 (448.68–448.68) |
| iq3-cache128 | 16 GiB | 33.77 (32.74–33.78) | 84.8 (83.6–86.3) | 448.68 (448.68–448.68) |
| iq3-cache128-gpuremap | 0 GiB | 31.95 (31.83–32.00) | 86.6 (86.3–88.4) | 451.73 (451.73–451.73) |
| iq3-cache128-gpuremap | 16 GiB | 31.92 (31.43–32.39) | 85.7 (85.6–86.7) | 451.73 (451.72–451.73) |
| iq3-cache128-prefetch12 | 0 GiB | 32.67 (32.54–33.64) | 84.8 (84.2–85.8) | 454.82 (454.81–454.83) |
| iq3-cache128-prefetch12 | 16 GiB | 33.69 (33.31–34.27) | 84.0 (83.8–85.4) | 454.83 (454.81–454.83) |
| iq3-resident-ub21 | 0 GiB | 10.23 | 200.3 | 0.06 |

Prefill and TTFT are not gates of this stage. Phase 11's decode target of 30 t/s at 128 slots is not met.

**Token IDs.** `iq3-cache128-gpuremap` matches `iq3-resident-ub21` in 30 of 30 requests, by the report's rule and again by exact phase and workload. The two CPU remap arms match resident in 6 of 30, as `iq3-cache128-gpuremap` matches `iq3-cache128` in 6 of 30 (INFO): their splits round differently from the unsplit graph (E0.1). IDs do not vary across runs within any arm.

**Splits and commits,** exact integers from `stats.json`: the GPU arm has 1270 splits per 635 tokens in all 6 runs, and the CPU remap arms 64770 in all 12. Commits are 21573 of 635 (33.97) on the GPU arm, 65405 (103) on the CPU remap arms and 1905 (3) on resident. Resident also has 2 splits per token, as expected.

### Estimate and measurement

The estimate below, "Estimate, not a result", was written before any real run and was never measured. On the GPU arm (`stats.json`, per decoded token):

| | Estimate | Measured |
|---|--:|--:|
| Misses | about 30 | 30.14 |
| Stops | at most about 23.5 | 20.64 |
| Command buffers | about 25 | 32.97 |
| Commits | about 26 | 33.97 |
| Saved against the CPU remap | about 13.5 ms of 57.8 | 4.72 ms at 0 GiB (56.82 to 52.10), 2.21 ms at 16 GiB (54.31 to 52.10) |
| Decode | about 44 ms/token | 52.10 ms/token at both ceilings (49.17–52.84) |
| Warm decode | about 23 t/s | 19.61 and 19.96 t/s |

- **Stops are below the estimate's bound and command buffers above the estimate.** Tiny's pattern of stops + 2 does not hold at this size. Why was not measured.
- **Commits alone do not carry E1's overhead.** "Other" falls by 4.97 and 2.45 ms for 69.03 fewer commits, about 0.072 and 0.035 ms per commit, against the 0.176 ms per commit that E1's 17.6 ms over 100 commits implies. These are differences of the report's medians.
- **What remains of the gap to resident:** 35.56 ms/token (52.10 against 16.54). The wait is about 15.2 ms, "other" above resident's 16.54 is 20.1 to 20.25 ms, and the remap 0.19 ms. Stream time per token (the `tokens.log` span) is 0.09 to 0.16 ms above `stats.json`'s wall in every run.
- **The GPU arm has a lower decode hit rate:** 90.0% against 91.0%, 30.14 against 27.11 misses per token. Its wait rises about 0.8 to 0.9 ms/token and its remap falls from 0.67 to 0.19 ms/token. The extra wait offsets part of the fall in "other". The arms differ in 24 of 30 requests' tokens, but the records do not say why the hit rate differs.

**E6's condition, stated and not decided.** PLAN says E6 runs "only if E2 and E3 leave the load wait as the largest remaining cost". On the GPU arm the wait is 15.15 and 15.28 ms of 52.10 (29.1% and 29.3%), and "other" is 36.79 and 36.63 ms (70.6% and 70.3%). Taking resident's 16.54 ms out of "other" leaves 20.25 and 20.09 ms, still above the wait. E3's write-up ("Stage E3") gives, on the side-file, 8.85 ms of 49.48 at 0 GiB and 8.86 of 52.17 at 16 GiB. "Other" is a remainder, not one measured cost, and no run has both E2 and E3. E6 is not started.

### Validity

- **Sleep and power.** 0 of the 257 true Sleep, Wake or DarkWake events in `pmset -g log` fall inside the span 04:23:23 to 05:42:08 or any run's window. The last before it is the Wake at 2026-11-05 22:17:13. The driver's `caffeinate` lived exactly that span. Power is not recorded per run: AC rests on the driver's gate, the final status ("AC Power") and the absence of any event in the span other than assertions and the display's notification.
- **Location and eviction.** Every `evict` record says home volume, and all 19 left 0 of 2170182 pages resident. Before eviction the page cache held 8.22% in 17 runs, 72.18% in run 1 and 99.79% in run 8, the run after the resident arm. On 16 GiB runs the hog wired 16.00 GiB.
- **Build.** All 19 runs carry one `server_sha256`, `40cec522…ad86356`, which recomputes from the files on disk now. The newest of those files is from 03:56:36, before the first run. Fork worktree HEAD is `0bb0f5444`, clean and equal to the pushed branch.
- **The driver is not from this branch.** The runner's `kolibri-bench` reports `vcs.revision` `0b19754` (E3's branch, not modified), and `kolibri-stream`, `kolibri-evict` and `kolibri-memhog` report `b451cd6` (also E3's). This branch's own driver cannot print the report's "PASS build" line. The verdicts should not depend on it: `evalGate`, `gateValues`, `perRep` and `median` are byte-identical between this branch and `0b19754`, and an independent recomputation gives the same report. The `kolibri-bench` `main.go` diff of 303 changed lines was only skimmed, by grep for arguments, eviction and quiet-gate lines. Its side-file code is keyed to flags this matrix does not use, and its new eviction guard only stops a run.
- **Repetition 1 is not like 2 and 3.** The display was on from 2026-11-06 23:30:21 to 04:38:44, so through runs 1 to 3 and part of run 4. Run 4 is `gpuremap/16g/r1`, the repetition that decides the 16 GiB result. Its warm coding and research requests finished before 04:38:44, and warm hr was running then. Its median mixes hr (18.93) and coding (18.96). Others' CPU is also highest in repetition 1: the mean over runs is 50.7% against 40.2% and 38.8%. No record shows that either factor slowed decode. This is a confound, not a cause, and the rule stands.
- **The margins are small, and the two verdicts are not symmetric.** `evalGate` passes a ceiling on one win, decode or p95. At 16 GiB neither won, so either metric alone would flip the failure: decode by 0.29% (0.054 of 19.00 t/s) or p95 by 0.95% (0.746 of 78.584 ms). At 0 GiB both won, so flipping the pass would take losing both: decode by 0.58% (0.110 of 18.91 t/s) and p95 by 1.33% (1.076 of 80.746 ms). The spread within one arm is far larger: `gpuremap/16g` runs from 18.94 to 21.69 t/s. Three repetitions cannot resolve margins this small. The rule was fixed before the runs, and this changes neither verdict.
- **The fastest baseline repetition at 16 GiB,** `prefetch12/16g/r3`, has the highest others' CPU maximum of the benchmark (255%, mean 42%). Only the mean and the maximum are recorded, so when that peak fell relative to decode is unknown. `others_cpu` is a 5 s `ps` sample over the server's lifetime, prefill included, and says nothing about the decode phase on its own. The `coreaudiod` release under "Other activity" falls in this run's cold prefill, and no record connects the two.
- **Other activity.** `pmset -g log` has these events in the span, apart from the driver's own `caffeinate`, the periodic summaries and the display's end above. 5 `cloudd` assertions of 3 to 4 s (04:26:03, 04:36:14 twice, 04:42:18 twice). 3 `apsd` push tasks timing out at each of 04:45:28 (run 6's cold prefill), 05:00:38 (run 10's wait) and 05:30:09 (run 17's wait). `AddressBookSourceSync` releasing an assertion of 30 s at 05:07:01, in run 11 (`gpuremap/16g/r2`), which spans warm research's decode and warm hr's prefill. Foreign `caffeinate`s that died at 04:23:33 (run 1's wait) and 05:21:49 (run 15's wait). `coreaudiod` releasing 5 audio contexts (10 assertions, held 12:51 to 14:42 in hours and minutes) at 05:39:33, in the cold prefill of run 19, `prefetch12/16g/r3`, before that request's first token at 05:39:46. The gate's values come from warm requests. No record shows that any of these changed decode, and none is taken as a cause. Swap stayed flat within every run (4008 or 4016 MiB), and memory pressure stayed at level 1. Quiet waits were 60 to 120 s.
- **Recomputation.** A separate program regenerated `report.md` from `run.json` and `stats.json`, byte for byte. The 95 requests were also recomputed from `tokens.log`: the largest deviation is 0.0142 t/s on decode and 0.935 ms on p95, because its stamps are whole milliseconds, and both verdicts come out the same. TTFT, load time, prompt t/s and page-ins come only from `run.json` and `summary.json`.

### Observations

These change no gate.

- **The rule compares extremes, so single repetitions decide it.** At 16 GiB only `gpuremap/16g/r1` decides on the candidate's side, on both metrics. On decode it loses to `prefetch12/16g/r3` alone (18.94 against 19.00). On p95 it loses to `prefetch12/16g/r1` and `r3` (79.33 against 78.58 and 78.79). `r2` and `r3` of the candidate lie beyond every baseline repetition on both metrics. Choosing `iq3-cache128` as the baseline also fails, so the verdict does not depend on which baseline was selected.
- **At 0 GiB** decode is decided by `gpuremap/0g/r1` (19.02, others' CPU 66/127) against `cache128/0g/r3` (18.91, 36/85), and p95 by `gpuremap/0g/r3` (79.67) against `cache128/0g/r2` (80.75). It would also pass against `iq3-cache128-prefetch12`.

### Open decisions

These are the owner's. None of them is planned, and merging this write-up decides none of them.

- Whether to repeat E2 with a new matrix, pre-registered before its first timed run. This result stands either way.
- Whether to benchmark E2 merged with E3. The fork branches' conflicts (Known limits) are not resolved, and no such build exists.
- Whether E6 starts, against its condition above.
- Whether fork PR #16 stays open. It is left open and draft, as a place for the code.

### The problem

E1 measured it: with all 384 experts cached, decode takes 35.6 ms/token against 16.8 resident, with 102 splits per token against 2; 17.6 ms of the gap comes with those splits ([expert-diagnostics.md](expert-diagnostics.md)).

### The decision

PLAN names two candidates and says a measured prototype decides. Measuring needs a quiet machine, so this choice rests on split and commit counts on tiny checkpoints and on reasoning.

- **Fewer, larger splits is out, by count.** The scheduler already puts a layer's attention and the next layer's router in one Metal split: 2 + 2 per layer is what one CPU op per layer costs at least, so the candidate has nothing left to fuse. It also keeps the split the requirement forbids: the logits would still differ from the unsplit resident run (0.134 on tiny Q8_0, 3.23 on IQ3).
- **GPU remap, but not with an `MTLSharedEvent` wait per layer.** A shader cannot signal or skip an encoded wait, so a wait at every layer is a CPU round trip at every layer, hit or miss. That is a lighter form of what E2 is meant to remove.
- **Chosen: a gate per layer that stops the command buffer on a miss.** A hit takes no CPU at all.

### How it works

- **The graph is the resident run's, node for node.** No CPU op, no extra node: the three `MUL_MAT_ID` get two extra sources, the layer's slot ids and its table. They reorder and fuse as in the unsplit run, which is why the logits are bit-identical to it.
- **Fusion stays intact.** The fusion table has no pattern with a `MUL_MAT_ID`, so a command buffer may end before a gate without cutting a fused group. The plain path's own encoder boundary (`n_nodes_0`) stays too, because a pattern across it does not fuse there.
- **The table.** Per layer, one `int32` per expert: its slot, or -1. It lives in the same shared host memory as the slots.
- **The gate.** Before a layer's first matmul, `kernel_expert_gate` maps the router's ids through the table into the slot ids, which the three matmuls read, and logs the router's ids for the CPU.
- **A stop.** On an id without a slot, the gate sets a flag and zeroes the threadgroup counts of every later dispatch of the command buffer: after a command buffer's first gate, all its dispatches are indirect. So nothing after the gate runs, including that layer's matmuls.
- **The miss path is the old one.** The backend waits for the buffer and calls the same `llama_expert_slots::remap` the CPU op uses: pins, LRU, loads, waits, the abort on a failed read. It then copies the layer's table and resumes at that gate.
- **Hits' bookkeeping is deferred, not dropped.** When a buffer has completed (or stopped), `remap` runs for each layer that passed, in layer order. LRU is per layer, so the order within a layer is what it was.
- **Command buffers.** Two gates per buffer after a stop, doubling while no gate stops, so a stop wastes little encoding and a run of hits costs few commits.

### What replaces the sync guarantee

Before, the scheduler's sync before each CPU split guaranteed that the previous layer's matmuls were done when the next remap ran, and a layer was in flight from its remap to the next one, so no prefetch replaced its slots. Now:

1. **The table and the slots' bytes change only while the GPU is idle:** in the remap of a stopped layer, after `waitUntilCompleted` of the only command buffer running for the graph.
2. **A stopped gate has run nothing after it.** That layer's matmuls did not start, and no later node has overwritten a tensor the resumed run reads. (M62 shows what happens otherwise: NaN.)
3. **The layers before it are complete:** their matmuls precede the gate in a completed buffer.
4. **There is no asynchronous writer.** The cache refuses `--expert-prefetch` in this mode, so all loads are demand loads, and the remap returns only when its layer's loads are done. The table holds only `READY` slots (`copy_table`), so a gate cannot pass an unread expert.
5. **Resuming must make progress:** a gate that stops again after its loads aborts (`stopped again after its loads`) rather than loop.

Pin, cancel and fall back: `remap` is unchanged, so `test-expert-cache`'s checks still hold it to them. Cancelling and the prefetch fallback have nothing to do in this mode, since it has no prefetch.

### No silent fallback

The model fails to load, and says why, when the option is asked for without a cache (no `--expert-cache` or `--expert-cache-layers`, where it would otherwise load fully resident, the hazardous memory case on this machine), with `--expert-prefetch`, with an expert layer not on Metal (the CPU, `-ngl 0`, part of the layers), or without a Metal backend. The process aborts, and says why, on a `cb_eval` that stops between a layer's matmuls and on the wider ids of `llama_set_warmup` (Known limits). After each ubatch, every layer must have been through its gate, or the process aborts. `expert_cache.gpu_remap` in the stats records the mode, and `prefill` and `decode` gain `gpu_runs` and `gpu_stops`. Off by default; the default path builds the same graph as before and its stats differ by that one key.

### Counts

On the tiny router checkpoint (6 layers, top 6), Metal, per decoded token, 40 prompt and 60 decoded tokens:

| Slots | Mode | Scheduler splits | Metal commits | Stops |
|--:|---|--:|--:|--:|
| any | resident | 2 | 3 | |
| any | CPU remap | 14 | 15 | |
| 6 | GPU remap | 2 | 8.99 | 5.99 |
| 16 | GPU remap | 2 | 7.43 | 3.90 |
| 64 | GPU remap | 2 | 4.73 | 1.15 |
| 384 | GPU remap | 2 | 4.73 | 1.15 |
| all hits (M61) | GPU remap | 2 | 3 | 0 |

The figures are F32; Q8_0's commits are within 0.07 and its stops within 0.12. The last row repeats a pass after clearing the KV cache, so the routing repeats and every expert has a slot: exactly 2 splits, 2 command buffers and 3 commits per token, as resident. **The expected decode split count is 2 per token for every model, whatever its layers,** and M61 is the control for it, as M42 is for the CPU remap's 2 + 2 per layer. At 64 and 384 slots tiny's stops are first uses: its 100 tokens do not fill the slots, so nothing is evicted, and both rows are the same.

### Estimate, not a result

Kept as written before the run; "Estimate and measurement" above has what the run measured. For IQ3 at 128 slots, Stage D's hit rate (8634 of 9600) means about 30 misses per token, so at most about 23.5 of 50 layers stop (independent misses: 1 - 0.9^6 = 47%). Tiny's pattern (6 stops, 8 command buffers) gives about 25 command buffers, or about 26 commits against 103. If E1's 17.6 ms scale with commits alone, that saves about 13.5 ms of 57.8: about 44 ms/token, or 23 tokens/s against 17.6. If all of it goes with 384 slots, about 18 ms/token against 35.6.

Evidence this estimate lacked: what one indirect dispatch costs on the GPU, since most dispatches of a token become indirect; how much of E1's 0.18 ms per split is the scheduler and how much is commit and wait; the encoding a stop wastes (one layer per stop); and whether the 7.5 ms that E1 could attribute only to misses stays. The benchmark did not separate these, so they stay unmeasured. An event wait per layer stays untested: no re-encoding, but a CPU round trip at every layer.

### Checks

| Check | Result |
|---|---|
| `test-expert-cache` | "all expert cache tests passed"; new: the table holds exactly the `READY` slots, and the gate's deferred bookkeeping gives the same counters as a remap per layer over 400 tokens, with no load while the "GPU" runs |
| fork `ctest` | 63 of 64, before the refusals below; not rerun since. `test-tokenizers-ggml-vocabs` fails: the `.gguf` files of the vocabulary repo it clones are git-lfs pointers here. It fails on the E3 build as well (reason not read there) |
| `check_expert_cache.py`, tiny, all | "PASS: 0 failures", 270 checks |
| `check_expert_cache.py --gpu-remap-only`, tiny | "PASS: 0 failures", 103 checks, plus 118 under `GGML_METAL_FUSION_DISABLE=1` |
| `check_expert_cache.py --real --gpu-remap-only`, IQ3, Metal | "PASS: 0 failures", 21 checks (`real-attempt2.log`): 128 slots at ubatch 21 and 64 slots at ubatch 10, `F_NOCACHE` and buffered, logits of 160 positions bit-identical to the UNSPLIT resident run, and also with only 33 output positions; 2 splits per token; the load refusals with prefetch and with some layers off Metal; all hits cost 2 splits, 2 command buffers and 3 commits per token. A first attempt (`real-attempt1.log`) ends after 8 PASS lines with no failure and no summary, and the log does not say why |
| `go test ./cmd/... ./internal/...` | 21 packages ok, with `/bin` first in `PATH` for `kolibri-rekey` |

- **Bit identity is against the UNSPLIT resident run,** `np.array_equal`, F32 and Q8_0, 6, 16, 64 and 384 slots, `F_NOCACHE` and buffered, as PLAN requires. It passed on the first build. On the CPU the option fails to load, by design, so the CPU has no second arm to compare.
- **A server's shape of decode found a bug.** The last layer keeps only the output rows, so a prefill ubatch with no output has no matmuls there and no gate; the first version aborted on it. The check now also runs with only the last positions as outputs, and `llama-completion` with its warmup decode. `llama-server` on tiny, three requests with `cache_prompt`, gave the same text as resident `-ub 10`, with 2 splits per decoded token. That run is by hand, not in the check, and no log was kept.
- **Counters.** With fusion on, the CPU remap can route an expert differently, because its splits round differently: on tiny Q8_0 one hit in 1,762 at 6 slots. So the check holds it to the GPU remap's counters exactly only where its logits equal the unsplit run's, and runs again under `GGML_METAL_FUSION_DISABLE=1`, where they do: hits, misses, evictions and bytes are then equal in every configuration.

### Negative controls

Each was applied, recorded and reverted; then a forced rebuild and a rerun of the checks. Logs: `~/models/eval/expert-bench-e2.runner/checks/`.

| Control | `test-expert-cache` | Tiny `--gpu-remap-only` |
|---|---|---|
| **M60:** `copy_table` swaps the slot of every odd expert: a wrong slot id in the GPU table | fails: 4667 checks failed | fails: "FAIL: 37 failures", max \|diff\| 1.11 |
| **M61:** a CPU op per layer back in the GPU arm's graph, as the CPU remap has | passes | fails: "FAIL: 75 failures": 14 scheduler splits per token instead of 2, and bit identity, max \|diff\| 0.134 |
| **M62:** the gate stops but leaves the later dispatches on: the invariant's second point | passes | fails: logits NaN on the first configurations, then the guard aborts ("gate 3 stopped again after its loads") |
| **M63:** `copy_table` never writes -1, so an evicted expert stays in the table | fails: 7891 checks failed | fails: "FAIL: 17 failures", max \|diff\| 1.11 |
| **M64:** the run resumes one node after the stopped gate | passes | fails: "FAIL: 37 failures", max \|diff\| 2.23 |
| **M65:** the remap does not wait for loads, the table lists loading slots, and a load takes 2 ms longer: the invariant's fourth point | fails: 5680 checks failed | fails: "FAIL: 34 failures", max \|diff\| 1.26 |
| **M67:** the plain path's `n_nodes_0` encoder boundary is dropped | passes | fails: "FAIL: 27 failures", max \|diff\| 0.134 |
| **M69:** cb_eval splits the GPU arm's computation after each `ffn_moe_topk` | passes | fails: "FAIL: 33 failures": bit identity, max \|diff\| 0.134, and the command buffer bound. The split count stays 2, because the scheduler does not count these |

Two more controls change what a run costs or counts, not what it computes, so bit identity holds under them by design. They do not meet the rule that a control fails bit identity; they guard the counts:

| Control | `test-expert-cache` | Tiny `--gpu-remap-only` |
|---|---|---|
| **M66:** command buffers never grow past two gates | passes | fails: "FAIL: 3 failures", the all-hits cost (and its run under fusion off); bit identity holds |
| **M68:** a layer that hits skips its bookkeeping | passes | fails: "FAIL: 17 failures", `steps` and the other counters; bit identity holds |

Five more guard the refusals. M69 was rerun after them and still fails as above, "FAIL: 33 failures": a `cb_eval` that stops outside the matmuls runs and changes the logits, and it is not refused.

| Control | `test-expert-cache` | Tiny `--gpu-remap-only` |
|---|---|---|
| **M82:** the model does not refuse the option without a cache | passes | fails: "FAIL: 9 failures": it loads on the CPU and on Metal, and `llama-completion` exits 0 |
| **M83:** `common` builds no params for the option alone | passes | fails: "FAIL: 3 failures": `llama-completion --expert-cache-gpu-remap` exits 0. With `--expert-cache-stats` the model still refuses |
| **M84:** a layer's second gate is not refused | passes | fails: "FAIL: 5 failures": a `cb_eval` after `ffn_moe_up` or `ffn_moe_gate` exits -6 without saying why. The old message was "11 of 6 expert layers went through the Metal gate" |
| **M85:** the slot ids' width is not checked | passes | fails: "FAIL: 5 failures": at 384 slots `llama_set_warmup` exits 0, with the gate writing past the tensor; at 64 it exits -6 without the width message |
| **M86:** the gate state is not reset after a graph that did not complete | passes | **passes:** "PASS: 0 failures" |

M86 shows a gap, not a guard: **the reset after an aborted graph has no check.** The check ends a decode at each call of the abort callback, and each next decode still works without the reset. `llama_set_abort_callback` reaches only backends that register `ggml_backend_set_abort_callback`, and Metal does not, so the callback runs in the CPU backend. The 4 calls and M86's result fit a CPU split before the gates; the split order was not read directly. The reset stays because a graph that ends past a gate would otherwise refuse the next one.

Commit `4397b54`'s subject calls these five M70 to M74, and their logs keep those names (`m70.log` to `m74.log` and `m74-rerun.log`, in `~/models/eval/expert-bench-e2.runner/checks/`). They moved to M82 to M86 because PR #77 uses M72 to M74e, and PR #72's `docs/upstream.md` gives M70 and M71 as the old numbers of its M80 and M81.

M67 is more than a control. Dropping that one boundary moves tiny Q8_0's logits by the same maximum, 0.134, that the CPU remap's splits do, so a resident run's own numbers depend on where its encoder boundary falls. The GPU remap computes the boundary by the same formula, on the same node count. Why the two maxima agree was not investigated.

### Reproduce

`testdata/bench/e2.json` stays as the record, and its GPU arm needs a build of fork branch `feat/kolibri1-gpu-remap` (`0bb0f5444`). The real-weight phase, on a quiet machine, nothing else holding the weights. The server is `third_party/llama.cpp-e2/build/bin/llama-server`, which serves both arms:

```sh
go run ./cmd/kolibri-models check testdata/bench/e2.json
.venv/bin/python tools/gguf/check_expert_cache.py --llama-cpp third_party/llama.cpp-e2 \
  --real ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf --gpu-remap-only
# from a checkout of feat/phase11-e3-sidefile, with -matrix naming this branch's file
go run ./cmd/kolibri-bench -matrix /path/to/this/branch/testdata/bench/e2.json -out ~/models/eval/expert-bench-e2 \
  -server third_party/llama.cpp-e2/build/bin/llama-server
```

The benchmark used the driver of E3's branch (`feat/phase11-e3-sidefile`, `0b19754`), not this branch's: it has the eviction limit and the server digest. Its validation accepts this matrix, as a test against that branch's package showed; the matrix has no side-file arm, so the side-file guards do nothing here. `--real` covers the load refusals with prefetch and with some layers off Metal. The other refusal checks run on tiny only: without a cache, on the CPU, on the command line, and the `cb_eval` and warmup aborts. So does the abort-callback check.

The check must say "PASS: 0 failures". `e2.json` is 19 runs: 3 arms × 2 ceilings × 3 repetitions, and one resident reference.

- **Gates:** decode by the Stage D rule at 0 and 16 GiB, against the best of `iq3-cache128` and `iq3-cache128-prefetch12`, since that rule names the best other arm at that ceiling. Resident at 0 GiB is the token reference, not a baseline: Stage D's deciding gate was at 16 GiB, where resident does not fit, and E2's bar is beating Stage D's cache.
- **Token IDs:** `iq3-cache128-gpuremap` must generate exactly those of `iq3-resident-ub21`, the same ubatches, or the driver exits 1. The comparison with `iq3-cache128` stays INFO: the CPU remap's splits round differently from the unsplit graph (E0.1).
- **Splits:** the report's decode table must show 2 per token for `iq3-cache128-gpuremap` and 102 for the two CPU remap arms; otherwise the arms did not compare what they claim to.

A server without the option exits with "error: invalid argument: --expert-cache-gpu-remap", and the gates then give no verdict.

### Known limits

- No prefetch in this mode. Same-token prefetch would need its own policy, since a hit layer no longer reaches the CPU in time to predict the layers after it. Stage D's prefetch did not earn its reads either.
- Metal only.
- A `cb_eval` may stop outside a layer's matmuls, as M69 does after `ffn_moe_topk`, and that changes the logits like any split. It may not stop between them: each graph compute finds its own gates, so the layer would get two, and the process aborts ("reached its Metal gate again"). So capturing `ffn_moe_up` or `ffn_moe_gate` needs the CPU remap or a resident run.
- `llama_set_warmup` makes the router's ids `n_expert` wide, and the slot ids tensor is `n_expert_used` wide, so the gate asserts. `common`'s warmup decode does not call it.
- `kolibri-bench`'s equivalence with the CPU remap arm is not exact, since its rounding differs from the unsplit graph (E0.1).
- Merging with E3: 0014 is on `feat/kolibri1` (fork PR #15), and the fork branch here has no side-file. `git merge-tree` of the fork branches reports conflicts in `common/arg.cpp`, `common/common.cpp`, `common/common.h`, `include/llama.h`, `src/llama-expert-cache.cpp`, `src/llama-expert-cache.h`, `src/llama-model.cpp` and `tests/test-expert-cache.cpp`. They include `llama_expert_cache_params` and `stats_json`. They were not resolved or read one by one, so this does not say whether any logic overlaps. The main repo holds both stages: `check_expert_cache.py` skips each stage's checks on a build without its option.
- The removed patch 0015 was the plain diff of the fork branch against `feat/kolibri1` at `634e34bb4`, so `git diff 634e34bb4 0bb0f5444` in the fork gives it again. Applied to that tree it gave the branch's tree, `cabe6384f`, checked again before the removal. The branch merged `feat/kolibri1` for that: it had gained the refusal of a LoRA on the cache's experts, which the diff would otherwise have taken out.

## Limits

- One context per model; a second context with the cache fails to be created. The cache is taken at the end of the context's construction, so a construction that fails earlier does not hold it.
- The expert layers must be on the CPU or on a device that can use host memory in place (Metal); CUDA and the like are rejected.
- Kolibri only. A LoRA on the routed experts is refused at load, since the matmuls read the slots and its delta would be left out; a LoRA on anything else, the router's input weight among them, works as without the cache. `--moe-cache-mib` with the cache is refused when the context is created. A draft model runs without the cache. The experts must be in one file; the load refuses them otherwise.
- A ubatch of one token counts as decode, also in a prompt. In a prefill ubatch the last layer sees only the output rows, so the predictor's counts of that layer cover fewer tokens than the trace replays of Stage B.
- LRU only. Layer budgets can be set with `--expert-cache-layers`, but nothing writes the Stage A allocation to such a file yet.
- `-fit` does not know about the slot buffer.
- libllama's INFO lines, including the cache's setup line, appear only with `-v`; the counters go to `--expert-cache-stats`.
- Platforms:
  - macOS opens the file again through `F_GETPATH`, refuses it if device and inode are not the loader's, and reads it with `F_NOCACHE`;
  - Linux reopens `/proc/self/fd/N`, without bypassing the page cache, and measures only the major faults;
  - other systems compile the cache but reject it at load time.
