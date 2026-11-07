# Expert streaming: the runtime expert cache

Phase 10, Stage D, item 1: a bounded, byte-accounted cache of routed experts inside libllama (patch 0011, fork branch `feat/kolibri-expert-cache`). It is experimental and Kolibri only. Stage D was started by the user's decision (2026-10-14) although no predictor passed the offline gates of Stages A and B ([expert-policy.md](expert-policy.md)). This document covers the prototype and its correctness checks. The benchmark (item 2) and the gates (item 3) are in "Benchmark" below; the speeds before it are INFO, measured on a busy machine. Phase 11's one read per expert, whose benchmark has not run yet, is in "Stage E3: one read per expert". Phase 11's prefill experiment, whose code stays out of the patch series, is in "Stage E4: waves for prefill".

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
- `process`, measured since the cache was set up: `disk_bytes_read` and `pageins` from `proc_pid_rusage`, `major_faults` from `getrusage`, and the lifetime peak `max_phys_footprint`. A counter the platform does not measure is `null`: on Linux all but `major_faults`.
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

Phase 11, Stage E3. A side-file next to the GGUF holds each expert's gate, up and down rows in one aligned extent, and a miss reads it with one `pread`. **The code and the checks exist. The benchmark has not run, so nothing here says whether it is faster.** Patch 0014 is on fork branch `feat/kolibri1-sidefile` (fork PR #15, open), not yet on `feat/kolibri1`. By the plan's stop rule it stays in the series only if the gates pass. E3 is not built on waves.

### The side-file

- **Where:** `<GGUF>.experts`, with its manifest as `<GGUF>.experts.manifest.json`. For IQ3 that is `~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf.experts`: 32,715,735,040 bytes (30.5 GiB, 78.8 MB of it table and padding, 0.24%), sha256 `83c6c88e60e55fd517ba002e1c1cab3c1b47bea4570403e594868edef9692bfe`.
- **Rebuild:** `go run ./cmd/kolibri-sidefile write -gguf ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf -force` (48.0 s), then `verify` with the same `-gguf` (37.6 s). `write` refuses to replace a file without `-force`, and it renames only a complete file into place. It is derived data: nothing is lost if it goes, and `kolibri-models` has no master for it.
- **Layout:** a table (the format is at the top of `cmd/kolibri-sidefile/format.go` and in `src/llama-expert-sidefile.h`), then one extent per layer and expert: the rows as the GGUF has them, then zeros up to the alignment. A read takes the padded extent, so it is aligned at both ends and never meets the end of the file. Row sizes come from each tensor, because IQ3's `down` differs: 696,320 bytes against 501,760.
- **Alignment, 16 KiB:** the VM page on Apple silicon and a multiple of the SSD's 4 KiB block. No page holds two experts, so buffered reads do not drag in a neighbour and `mincore` counts per expert. As far as XNU's cluster layer goes, a direct `F_NOCACHE` read needs both the file offset and the destination aligned. That comes from its source as remembered and was not measured here.
- **One read, then copies:** a slot's three rows sit in three tensors, and they are no multiples of a block (IQ3's gate rows are 122.5 blocks of 4 KiB). So the read goes to a page-aligned staging buffer per I/O thread, and three copies take the rows out. That costs about 1.7 MB of memory per thread, and the copy's time is part of `io_seconds`.

### What is checked when

**The load checks the table. It does not check the data.** It refuses, saying what failed: bad magic, the version, the table's sha256, the layer and expert counts, each tensor's row size, type and GGUF offset, extents out of range, misaligned or overlapping, the file's size, and the GGUF's identity (its size and the sha256 of its header). A missing file fails the load: there is no fallback to the GGUF. **A flipped byte in an extent still loads, and a run that uses that expert reads the wrong bytes.** Hashing about 31 GB at every load would cost tens of seconds, so only `kolibri-sidefile verify` checks the data: it rehashes every extent and the whole file, compares every extent byte for byte with the GGUF, and accepts no field of the manifest on trust. The loader does not read the JSON manifest; the checksummed table in the side-file is what it checks.

**Concurrency needed no change.** A remap queues all of a layer's misses, and the I/O threads (default 4, as E1's curve advises) take them from one queue. `test-expert-cache` holds a load until all four threads are inside one remap's loads. Prefetch uses the same load path, so it works on both paths.

### Correctness so far

Bit identity with `np.array_equal` against a resident run at the same ubatch and, on the GPU, the same split (`check_expert_cache.py`), never loosened:

| Check | Result |
|---|---|
| Tiny, F32 and Q8_0, CPU and Metal: 6, 16, 64 and 384 slots, with and without prefetch and buffered reads, slots per layer, half the layers on Metal, repacking on | "PASS: 0 failures", 336 checks. The 48 side-file runs of that grid are bit-identical both to the resident run and to the GGUF path |
| Real IQ3 on Metal (`--real`, no hog): 128 and 64 slots, the same variants | "PASS: 0 failures", 40 checks. At 128 slots 7,173 and 966 preads for 7,173 and 966 experts in prefill and decode, against 21,519 and 2,898 on the GGUF path |
| Build without the option (`third_party/llama.cpp`, the waves build) | "PASS: 0 failures", 380 checks, the side-file checks skipped with a note |
| `test-expert-cache`, also in ctest | "all expert cache tests passed": the three tensors' bytes on both paths, 27 refusals by name, a read past the end, 3 and 1 preads |
| `test-expert-cache --sidefile`, the ranges from the GGUF's own header by ggml's gguf API | tiny: all 2,304 experts; IQ3: 450 experts, the first and last of each layer among them. No difference; `verify` covers all of IQ3 |
| `kolibri-sidefile verify` on IQ3 | every extent byte-identical to the GGUF |

### Negative controls

- **M50, wrong extent offsets:** the writer's table sends each expert to its neighbour's extent, with a valid checksum, so the load accepts it. All 56 side-file lines against the resident run fail (max |diff| up to 1.46), "FAIL: 139 failures", and the same-misses and pair checks fail with them.
- **M51, a corrupt manifest**, on an APFS clone of the real side-file: a flipped table byte is refused at load ("table checksum mismatch", exit 1, no text generated), and `verify` reports it. One edited extent hash in the JSON manifest: `verify` reports "layer 7 expert 11: sha256 ..., the manifest says ...", exit 1.
- **M52, the load without its checksum test:** `test-expert-cache` fails 3 checks, and the tiny check fails 2 ("FAIL: 2 failures").
- **M53, `kolibri-bench` without the side-file's eviction:** `TestEvictSidefile` fails.
- **M54, eviction works on the real file:** `shasum` left 1,820,300 of 1,996,810 pages resident (28,442 MiB), and `kolibri-evict` took them to 0. `shasum`'s read is the hazard the next section guards against.

### Benchmark validity

The page cache serves resident pages even with `F_NOCACHE`, and reading the file for a hash leaves it resident (M54), so an arm could read from memory and look fast. So:

- **Every run of every arm evicts the GGUF and every side-file of the matrix**, and `run.json` records it (`evict`, `evict_sidefiles`, and `sidefile` for the arm's own file). The driver stops, with no result, if a side-file is missing or stays more than 1% resident.
- **The hidden tier counts for the arm as for any other.** `peak_page_cache_mib` and `end_page_cache_mib` sum the GGUF and the side-file, with the side-file's part beside them. All arms sample the same files, so the sampling costs them the same.
- **A missing side-file is loud:** the driver stops before its first run, and a later loss fails the server's load, which fails the report. Nothing falls back to the GGUF.
- **The report proves the path:** with a side-file arm in the matrix it fails unless each run's stats name the file its flags gave and show 1 pread per expert (3 on the GGUF path) in both phases.
- **`fits` and the quiet rule do not special-case the arm.** Its `--expert-cache 128` counts as the GGUF arm's does (a test holds them equal).
- **No timing yet.** A smoke run of the exact flags with 8 generated tokens gave 5,117 and 219 preads for 5,117 and 219 experts on the side-file, 15,351 and 657 on the GGUF path, the same token IDs, and at most the table's 10 pages of the side-file in the page cache. Its speed was not read.

### Reproduce

The matrix `testdata/bench/e3.json` holds 12 runs: `iq3-cache128` and `iq3-cache128-sidefile`, with every other flag the same, at 0 and 16 GiB, 3 repetitions each. Decode and prefill of the side-file against the GGUF path decide at both ceilings, by `kolibri-bench`'s existing rule. TTFT is info only. Token IDs must be exactly equal.

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

## Limits

- One context per model; a second context with the cache fails to be created. The cache is taken at the end of the context's construction, so a construction that fails earlier does not hold it.
- The expert layers must be on the CPU or on a device that can use host memory in place (Metal); CUDA and the like are rejected.
- Kolibri only, no LoRA on the routed experts. The experts must be in one file.
- A ubatch of one token counts as decode, also in a prompt. In a prefill ubatch the last layer sees only the output rows, so the predictor's counts of that layer cover fewer tokens than the trace replays of Stage B.
- LRU only. Layer budgets can be set with `--expert-cache-layers`, but nothing writes the Stage A allocation to such a file yet.
- `-fit` does not know about the slot buffer.
- libllama's INFO lines, including the cache's setup line, appear only with `-v`; the counters go to `--expert-cache-stats`.
- Platforms:
  - macOS opens the file again through `F_GETPATH` and reads it with `F_NOCACHE`;
  - Linux reopens `/proc/self/fd/N`, without bypassing the page cache, and measures only the major faults;
  - other systems compile the cache but reject it at load time.
