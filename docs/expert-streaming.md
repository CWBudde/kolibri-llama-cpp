# Expert streaming: the runtime expert cache

Phase 10, Stage D, item 1: a bounded, byte-accounted cache of routed experts inside libllama (patch 0011, fork branch `feat/kolibri-expert-cache`). It is experimental and Kolibri only. Stage D was started by the user's decision (2026-10-14) although no predictor passed the offline gates of Stages A and B ([expert-policy.md](expert-policy.md)). This document covers the prototype and its correctness checks. The benchmark (item 2) and the gates (item 3) are in "Benchmark" below; the speeds before it are INFO, measured on a busy machine.

## How it works

- **The file keeps the experts.** The stacked expert tensors `ffn_{gate,up,down}_exps` of every layer become lazy tensors: mapped on the CPU, excluded from the loader's `WILLNEED` advice, and never read through the mapping. Nothing places them on the GPU.
- **Slots.** Each layer gets slot tensors of the same type and row size, with `n_slots` experts instead of 384, on the device of the layer's other weights.
  - The slot memory is always host memory, since the reads write into it. For a CPU layer it is the CPU's plain buffer type, never a repacking one. For a GPU layer it is an anonymous mapping that the device uses in place (`buffer_from_host_ptr`; Metal does, as for mapped weights).
  - A device that cannot (CUDA, for example) is rejected at load time.
  - Slots never written cost no memory.
  - The buffers count as weights, so the scheduler runs each layer's expert matmuls where its slots are.
- **The remap op.** In `build_moe_ffn`, after the router's selection and its weights, a CPU custom op maps the router's ids to slots, `ffn_moe_slots-<layer>`. The three `mul_mat_id` read the slot tensors with these slot ids. The expert weights keep the router's own ids, and the router chooses every expert; a prediction only decides what is read ahead.
- **Loads.** A missing expert is read with `pread` into its slot: three ranges per expert (gate, up, down), from a second descriptor of the GGUF with `F_NOCACHE` (macOS), so the bytes do not stay in the page cache. On unified memory there is no host-to-device copy: the read from the SSD into the slot *is* the transfer. Reads of pages that are already in the page cache are served from it, `F_NOCACHE` notwithstanding.
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
