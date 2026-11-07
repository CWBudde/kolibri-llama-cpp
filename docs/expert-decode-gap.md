# Where the decode gap goes

A diagnosis of the 20 ms per token that Stage E2's benchmark left unexplained on the expert cache at 128 slots. It proposes no fix and passes no gate; the code it ran is private and is on no branch. It covers the IQ3 weight on Metal at ubatch 21, on this M5 Pro.

Numbers are ms per decoded token: the median of three runs, with the range in brackets, unless a table says otherwise. Anything labeled *Inference* was not observed directly.

## Results

**The 19.5 ms is split along one ordering of the arms, and that split is not unique.** On the GPU remap at 128 slots, a token took 49.31 ms against 16.46 resident on the replay pattern. Load wait is 13.33 of the difference, which leaves 19.52. The rows below are a telescoping chain of arm differences, so they add up to the total of 19.47 by construction, on any data: 26.32 minus 15.90 in GPU busy and 9.61 minus 0.56 in the rest, whatever the arms in between. The 0.05 left over is only the median of wall against the sum of medians (49.31 against 26.32 + 13.33 + 9.61 = 49.26), and the -0.27 is a remainder by definition. The runs of `gpu128` differ by 1.36 ms in wall, and its later passes by more: their means run from 41.74 to 53.44 with the same misses.

| Cause | GPU busy | Rest | Together | How it was separated |
|---|--:|--:|--:|---|
| **The thread waiting for loads sleeps** | +8.33 | +1.12 | **9.45** | same misses, the waiter spins instead (`gpu128` against `gpu128-loadspin`) |
| **Extra command buffers and stops** | +1.68 | +5.70 | **7.38** | GPU: stops with no load and no idle (`n32-fs18` against `gpu128-n32`). Rest: `gpu128-loadspin` against `gpu128-n32` |
| **The gated path at all hits** | +0.68 | +2.23 | **2.91** | `gpu128-n32` on tokens with no miss against `resident` on the same tokens |
| Fresh weights, GPU caches, TLB | -0.27 | | not separated | what is left of GPU busy: `gpu128-loadspin` against `n32-fs18` |

"GPU busy" is the sum of `GPUEndTime - GPUStartTime`. "Rest" is everything else except load wait: encoding, commit, the lag from commit to GPU start, and the lag from GPU end to the waiter's return. The rows are differences of medians from different arms. Their sum is no evidence of additivity, and the effects interact: a sleep adds 8.33 to GPU busy at real stops (`gpu128` against `gpu128-loadspin`) and 18.46 at forced stops (`n32-fs18-sleep` against `n32-fs18`). So each row's size depends on the order of the differences.

Compute and the slots are a small part of the cost above the resident path. **With no miss, the GPU remap takes 19.38 ms, 2.9 above resident. With misses and a spinning waiter it takes 38.90. With misses and the current sleeping waiter it takes 49.31.**

### The arms

| Arm | Wall | GPU busy | Load wait | Rest | Misses | Stops | Commits |
|---|--:|--:|--:|--:|--:|--:|--:|
| `resident` | 16.46 (16.42–16.49) | 15.91 (15.91–15.93) | 0 | 0.55 (0.51–0.56) | 0 | 0 | 3 |
| `gpu128` | 49.31 (48.10–49.46) | 26.32 (25.43–26.42) | 13.33 (13.16–13.38) | 9.61 (9.51–9.72) | 26.2 | 17.5 | 30.7 |
| `gpu128-loadspin` | 38.90 (38.75–39.45) | 17.99 (17.91–18.04) | 12.46 (12.34–12.84) | 8.49 (8.45–8.57) | 26.2 | 17.5 | 30.7 |
| `cpu128`, 2 runs | 52.79 (51.64–53.94) | 28.13 (27.03–29.22) | 12.63 (12.62–12.64) | 12.03 (11.99–12.08) | 25.0 | | 103 |
| `gpu384`, 1 run | 19.33 | 16.66 | 0 | 2.67 | 0 | 0 | 3 |
| `cpu384`, 1 run | 40.17 | 28.36 | 0.05 | 11.76 | 0 | | 103 |
| `gpu384-c2`, 1 run | 23.78 | 16.60 | 0 | 7.18 | 0 | 0 | 27 |

These are the later replay passes: the coding workload's prompt once, then the same 128 tokens again and again, 640 tokens per run.

The next table holds the same 32 tokens, except `cpu128-n32`, whose 32 tokens are the CPU remap's own, on later passes with no miss, 192 tokens per run (`resident` 160):

| Arm | Wall | GPU busy | Rest | Injected idle | Encoding | Start lag | Wake lag | Stops | Commits |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `resident` | 16.46 (16.44–16.54) | 15.90 (15.89–15.92) | 0.56 | | 1.43 | 0 | 0.10 | 0 | 3 |
| `gpu128-n32` | 19.38 (19.30–19.46) | 16.58 (16.57–16.63) | 2.79 | | 1.60 | 0.47 | 0.38 | 0 | 3 |
| `gpu384`, first 32 tokens, 1 run | 19.23 | 16.63 | 2.59 | | 1.59 | 0.43 | 0.24 | 0 | 3 |
| `cpu128-n32`, 2 runs | 43.13 (41.65–44.62) | 30.38 (29.43–31.33) | 12.76 | | 2.04 | | 3.37 | | 103 |
| `n32-fs18` | 29.50 (29.32–30.19) | 18.26 (18.21–18.88) | 11.24 | | 3.34 | 3.56 | 3.78 | 18 | 39 |
| `n32-fs18-sleep` | 73.98 (70.72–76.54) | 36.72 (34.30–39.00) | 37.26 | 18.60 | 5.77 | 7.96 | 3.90 | 18 | 39 |
| `n32-c2-spin` | 47.63 (47.26–47.81) | 17.90 (17.36–17.93) | 29.90 | 22.50 | 1.26 | 2.55 | 2.95 | 0 | 27 |
| `n32-c2-sleep` | 107.34 (102.13–108.00) | 62.09 (58.94–62.55) | 45.25 | 22.81 | 4.05 | 14.08 | 3.12 | 0 | 27 |

### What each experiment rules in or out

**A sleeping waiter: ruled in, and the largest single cost of the 19.5 ms above load wait. Load wait itself is larger, at 13.33.**
- **Real misses.** With the same 26.2 misses and 17.5 stops per token, making the waiter spin takes GPU busy from 26.32 to 17.99 and wall from 49.31 to 38.90. Encoding drops too, from 3.24 to 2.47, for the same number of nodes.
- **No I/O at all.** The same 0.9 ms of idle GPU, 26 command buffers per token, no miss: as a sleep GPU busy is 62.09, as a spin it is 17.90. The GPU runs the same dispatches. Encoding on the CPU is 4.05 against 1.26. So it is not the idle GPU that does this, nor anything about the data, and it is not only the GPU that slows down.
- *Inference:* when the CPU sleeps while the GPU is idle, the chip drops to a lower performance state, and the next command buffer runs in it. No frequency was observed: `powermetrics` needs root, and `xctrace` was not tried, to avoid a prompt on an unattended machine.
- *Inference:* a sleep with no I/O costs more (GPU busy 36.72 at 18 stops) than real waits (26.32 at 17.5 stops), so the I/O threads' activity may hold part of the performance state up. But `usleep(680)` gives 1.03 ms per stop in effect, against a real wait of 0.76 ms. The arm also has 40% more idle (18.60 against 13.33) and more wasted nodes (3943 against 2530), either of which could explain the difference without any effect from the I/O threads. No experiment isolated that.

**Command buffers and stops: ruled in, a smaller cost. Two levels only, one stop count and one chunk size, so linearity is untested.**
- **Per command buffer.** 24 more of them with no stop and no idle add 4.51 to the rest and nothing to GPU busy (`gpu384-c2` against `gpu384`, one run each): 0.19 ms per buffer, mostly in the lags from commit to GPU start and from GPU end to the waiter's return.
- **Per stop.** Forced stops with no load add 1.45 of encoding over `gpu384-c2`, and 1.68 of GPU busy over `gpu128-n32`, at 18 stops. Of that 1.68, about 0.72 is at the token median (17.29 against 16.57). The other 0.96 or so is in slow windows of `n32-fs18` (p95 26.6 to 28.6, pass means 17.24 to 21.71). `n32-c2-spin`, which has no stops, has such windows too (p95 22.5 to 33.2); `gpu128-n32` and `gpu384-c2` have none (p95 at most 16.8). *Inference:* the zeroed dispatches and the re-encoding of what a stop threw away are in the median's part, which gives (1.45 + 0.72) / 18, about 0.12 ms per stop.
- **The control does not match the real stops.** `n32-fs18` has 39 commits, 7130 encoded nodes and 3943 wasted, against 30.65, 5717 and 2530 at real misses: 219 wasted nodes per stop against 144.

**The slot indirection itself: a small cost, and the size of the slot buffer does not matter.**
- At all hits GPU busy is 16.58 at 128 slots and 16.63 at 384 (one run), against 15.90 resident. So the gate, the indirect dispatches and a slot buffer of 10.1 or 30.4 GiB cost about 0.7 together, on the GPU.
- The rest rises from 0.56 to 2.79. *Inference:* the gated path encodes before it commits, while the plain path's GPU starts on the first command buffer during encoding; encoding is 1.60 of that rest. It was not tested separately.

**Fresh weights, GPU cache and TLB effects: not separated.** With 44.5 MB of new weights per token and a spinning waiter, the token median of GPU busy matches that of forced stops with no new data, at 17.3 (17.28 and 17.29; means 17.99 and 18.26). But the control does more stop work than the real stops, as above. They are also different token sets (128 and 32), whose resident GPU busy is the same (15.91 and 15.90). A cost under about 1 ms is not excluded.

**First touch and page-ins: ruled out as a decode cost.**
- The slots fill during prefill: of 20961 decode misses in the first `gpu128` run, 116 went to an empty slot, the others evicted.
- In a separate test on the cache's read pattern, a first write of a slot takes 628 µs per expert and a rewrite 597 µs (`standalone/touch.txt`, one thread, 800 experts). So the 116 first touches cost about 3.6 ms more than rewrites would have (116 × 31 µs) over 767 tokens, under 0.01 ms per token. The reads themselves, about 73 ms at 628 µs each, happen for a rewrite too.
- The `gpu128` process had 42 page-ins in the whole first run.

**Bytes copied into slots: nothing is copied.** A miss reads 1.70 MB from the SSD straight into the slot. At 26.2 misses that is 44.5 MB per token. That read takes 597 µs on one thread, and a `memcpy` of the same bytes 52.6 µs. The wait grows by about 0.345 ms per miss (fits per run, R² 0.82 to 0.87), and it is already counted as load wait, not in the 20 ms.

**The CPU remap is a different case.**
- **GPU busy changes over time at 103 commits.** It is about 16.7 on some stretches of tokens, the gated path's fast value, and up to 57 on others. `cpu128`: 256 and 302 of 640 tokens are under 20 ms, p5 16.6, p75 36.2 to 39.5, p95 49 to 50, pass means 18.86 to 39.39.
- **The means hide the distributions.** Mean GPU busy is about 28 with and without misses (28.13 and 28.36), but `cpu384`, at all hits and 1 run, has p5 24.0, p95 35.5 and none of 640 tokens under 20. `cpu128-n32` has 3 and 31 of 224 at all hits. `cpu128` minus `cpu384` is 12.62 and the load wait 12.63, but that compares 2 runs with a range of 2.30 against 1 run, so the 0.01 is coincidence.
- **About 40 at all hits is an average, not a floor.** `cpu128-n32` r2, pass 1, at all hits and 103 commits, has GPU busy of 16.4 to 17.4 on 26 of 32 tokens and a median wall of 28.06. The no-miss table leaves that pass out, as it does for every arm. The stable part is GPU busy of 16.5 to 16.7 plus a rest of 10.7 to 11.4. The fast tokens have the same 103 commits, so the buffer count cannot explain the slow stretches.
- *Inference:* E2's "other" of 39 to 42 is the same average, not a structural cost.
- *Inference, untested on this path:* the slow stretches may be the same performance-state effect as the sleeping waiter's.
- It generates other tokens than the GPU remap and resident, as E2 recorded, and two different token hashes over its own passes. All GPU remap arms of 128 tokens and `resident` have one and the same hash.

**A full run of the benchmark's own five requests** ties this to E2:

| | Wall | GPU busy | Load wait | Rest | Encoding | Start lag | Wake lag | Misses | Stops | Commits |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `full/gpu128`, all five requests | 53.37 (51.35–53.55) | 28.05 (26.39–28.29) | 15.23 (15.18–15.25) | 10.03 (9.78–10.07) | 3.04 | 3.61 | 2.80 | 30.1 | 20.6 | 34.0 |
| `full/gpu128`, four warm requests | 52.03 (51.79–52.10) | 27.23 (27.09–27.38) | 14.91 (14.85–14.93) | 9.87 (9.77–9.89) | 3.03 | 3.52 | 2.76 | 29.4 | 20.2 | 33.5 |

- **Same tokens and counts as E2, wall 1.27 higher (2.4%).** E2 recorded 52.10 (51.51–52.84) decode, 15.15 wait and 34 commits for this arm at 0 GiB, without timers. All five requests' text hashes match E2 on all three runs, and so do misses, stops and commits (19138, 13109 and 21573). The ranges overlap, but the median and a second run lie above E2's maximum. The 1.27 is unexplained, and timer overhead on the gated path is not separately bounded.
- **On E2's own numbers,** 53.37 minus 15.23 wait minus 16.54 resident leaves 21.60: 12.14 in GPU busy over the replay's resident 15.91, and 9.48 in the rest over its 0.55.
- **Same split as the replay.** More misses and stops (30.1 and 20.6, against 26.2 and 17.5) give somewhat larger terms of the same kind. Applying the table's split to this workload is *Inference*, because the spin and the forced stops were run on the replay only.

### Candidate next steps

For the owner to decide; none is proposed as a fix. Sizes are measured on the replay pattern, except where a line says otherwise.

| Candidate | Measured size | What is not known |
|---|---|---|
| Keep a CPU core awake during load waits. The experiment spins the waiting thread | 49.31 to 38.90 wall, -10.41; 20.3 to 25.7 tokens/s in decode terms. Prefill barely moves: 73.1 to 74.4 prompt tokens/s | Energy and heat of a core at 100% during waits; sustained runs; a hog; whether something cheaper than a spin holds the state (a shorter sleep, a QoS class, a timer); the cold request. `--expert-cache-threads` was left at 4 |
| Fewer command buffers per token at misses | 0.19 per buffer, 27.7 extra buffers: at most about 5.2 (*Inference:* slope times count) | How few are possible: a miss still needs an idle GPU to write the slot |
| Less work thrown away at a stop | about 0.12 per stop, 17.5 stops: about 2.1 (*Inference:* median-based slope times count, from a control that throws away more per stop) | A stop still costs its round trip, counted in the line above |
| Overlap encoding and GPU on the gated path | at most 2.23, the rest at all hits | Whether a gate allows it |
| Fewer or shorter waits: side-file, recorded in E3 | wait 14.48 to 8.85 on the CPU remap at 0 GiB | Not run here. The side-file and the GPU remap are on separate branches, so the combination has never been measured. Shorter waits may change what a sleep costs, in either direction |
| The CPU remap's slow stretches of GPU busy | mean 28.36 against 16.66 at all hits; its fast stretches reach 16.5 to 16.7 | Cause of the slow stretches unknown. It matters only if the CPU remap stays; merging the GPU remap is an open decision |

**E6's condition, for the owner to decide.** PLAN runs E6 "only if E2 and E3 leave the load wait as the largest remaining cost". E2's write-up (PR #76) could put "other" above the wait only because "other" was an unsplit remainder. With this split, on this replay and without the side-file, the wait of 13.33 is the largest single named cost, though the split is one ordering of the arms.

### What was not run, and deviations

- **Fewer repetitions than three, for lack of time** (hard stop at 08:15): `cpu128` and `cpu128-n32` have 2 runs; `gpu384`, `cpu384`, `gpu384-c2` and `gpu384-c2-idle` have 1. Statements resting on them are marked "1 run" or "2 runs".
- **Not run:**
  - the five requests on `resident`, `cpu128` and `gpu128-loadspin`. So the spin's saving is measured on the replay only, and the five-request `gpu128` is compared with E2's resident control;
  - a no-idle `n32-c2`. `gpu384-c2` stands in for it, on the same 32 tokens, 1 run;
  - any hog or 16 GiB ceiling, any Q8_0, the side-file build and the plain series build. The clone is the E2 branch, which contains the series build's `634e34bb4` as its merge-base;
  - Instruments and `xctrace`; `powermetrics` needs root. No clock or power figure was observed;
  - `build3` with no switch, so the spin and forced-stop comparisons cross two binaries. The diff adds only environment-gated branches, so a large effect is not expected, but none was measured. The arms also ran apart in time: `gpu128` 07:19 to 07:42, the spin 07:43 to 07:54. `gpu128-nice0`, at 08:07 on `build`, limits the time effect and not the build effect.
- **The spin is an experiment, not a design:** a 10 µs spin with the lock released between checks. It also moved load wait slightly, 13.33 to 12.46, which the -10.41 includes and the table at the top does not.
- **Run order:** repetition 1 of most arms came first and the others later, not repetitions outermost as in `kolibri-bench`.
- **Quiet rule:** up to 07:42 a run started when the 1-minute load was below 3. After that, to fit the time, it started after a 6 s settle: the load was the decay of the previous run. Nothing else was started by this experiment; the 624.6% below is unexplained. Each `run.txt` records the load and, from then on, the summed `ps` CPU of all processes at the run's start: 1 to 34%, except 624.6% at the start of `gpu128-loadspin/r1`, cause unknown. That run is the highest of the three (wall 39.45, wait 12.84). Its first later pass is disturbed: wall 43.38 and wait 15.26, against 38.03 to 38.67 and 12.18 to 12.27 in its other four. Neither r2 nor r3 has such a pass, and its pass 0 matches theirs (43.45 against 43.28 and 43.24). The pass comes about 40 s after the reading, and neither cause is known. The table keeps it. Without it r1 is 38.46 and 12.24, the lowest of the three, and the medians move: wall 38.90 to 38.75, wait 12.46 to 12.34, the spin's saving 10.41 to 10.56. Swap read 3992 to 4000 MiB at every start and end (`run.txt`). The Mac was on AC.
- **Priority:** the runner's servers ran at nice 5, because zsh lowers background jobs. One `gpu128` run at nice 0 (`replay/gpu128-nice0`, 1 run) took 50.75 with 27.12 GPU busy, so the priority does not explain the gap. How E2's driver was started, and so its priority, is not on record.
- **`tools/run.sh.v1`** is the runner before the settle change. `failed-starts/` holds one start that failed on a bug in that runner.

## Method

Everything is under `~/models/eval/decode-gap-20261107/`.

**Build.** A private clone, `third_party/llama.cpp-gap` (gitignored, no remote), of `llama.cpp-e2` at `0bb0f5444`, the GPU remap branch, with timers added. It has both remap paths: without `--expert-cache-gpu-remap` it runs the CPU remap. It is not a patch and is on no pushed branch. CMake options as in `llama.cpp-e2`, except that tests and the UI were not built. The diffs are in `tools/`:

- `private-timers.diff` (`build`): the timers. Used by `resident`, `gpu128`, `cpu128`, `gpu128-n32`, `cpu128-n32`, `gpu384`, `cpu384`, `gpu384-c2`, `gpu384-c2-idle`, `gpu128-nice0` and `full/gpu128`.
- `private-timers-v3.diff` (`build3`): adds four environment switches (`KOLIBRI_LOAD_SPIN`, `KOLIBRI_IDLE_SPIN`, `KOLIBRI_GATE_FORCE_STOP`, `KOLIBRI_STOP_IDLE_US`), used by `gpu128-loadspin` and the `n32-*` arms. Each run's `run.txt` holds the binary's SHA-256 and its switches. `private-timers-v2.diff` was built as `build2`, which no reported run used.

**Timers.** One CSV row per one-token ubatch, `gap.csv` beside `stats.json`, in the window of the existing `decode.wall_seconds`:

| Column | What it measures |
|---|---|
| `wall` | the existing decode timer |
| `gpu_busy` | sum of `GPUEndTime - GPUStartTime` over the graph's command buffers |
| `load_wait` | the existing `wait_seconds` |
| `enc` | encoding, command buffer creation to just before `commit` |
| `start_lag` | gated path: `GPUStartTime` minus the return of `commit` |
| `wake_lag` | the return of `waitUntilCompleted` minus `GPUEndTime` |
| `remap_op` | the existing remap time, without the wait |
| `sched_*` | the scheduler's per-split sync, input copy and CPU or Metal compute |
| `idle` | the injected idle |
| `misses`, `stops`, `commits`, `nodes_enc`, `nodes_wasted` | counts per token |

`rest` is `wall - gpu_busy - load_wait`. On the gated path nothing overlaps the GPU, so `rest` should be `enc + commit + start_lag + wake_lag + remap_op` plus what no timer covers. On the plain path encoding overlaps the GPU, so `enc` is not additive there, and `sched_copy` contains the wait for the GPU.

**Instrument control.** Resident with timers: 16.46 ms per token on the replay, against 16.54 without timers in E2. That path has 3 commits, so it does not bound the timers' cost at 34. The GPU remap with timers on E2's five requests: see "A full run" in the results.

**Runs.** `tools/run.sh`: confirm that `lsof` shows no holder of the weight and that no `llama-server` runs; `kolibri-evict` from E2's `bin/`; start the server with E2's arguments (`-ngl 99 -t 10 -c 4096 -np 1 --fit off`, plus the arm's); run the requests; stop it with SIGINT and wait for it to exit. Every `evict.txt` says "home volume". The ubatch is 21 everywhere: the cache caps 128 slots at 21, and the resident and 384-slot arms pass `-ub 21`. Weight: `Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf`. Metal.

**Two request patterns:**

- `full/`: E2's five requests through `kolibri-stream` (cold coding, warm coding, research, hr, medtech), 2048 prompt tokens and 128 generated, with E2's stream arguments.
- `replay/`: the coding prompt, then the same request again with `cache_prompt` (`tools/replay.go`): 6 passes of 128 tokens, or 8 passes of 32 for `n32`. Passes after the first have no prefill, so slots and timings are stationary and a run takes about a minute. Tables use the later passes.

**All hits at 128 slots.** Replay of 128 tokens does not reach all hits: 128 slots hold too little of the pass, so LRU keeps missing 26 experts per token, in every pass. Replay of 32 tokens fits. From the second replay on, tokens with no miss are selected, and the comparison arms use their rows for the same 32 tokens.

**Arms:**

| Arm | Server arguments and switches |
|---|---|
| `resident` | `-ub 21` |
| `gpu128` | `--expert-cache 128 --expert-cache-gpu-remap` |
| `gpu128-loadspin` | the same, `KOLIBRI_LOAD_SPIN=1`: the thread waiting for loads spins instead of sleeping on its condition variable |
| `cpu128` | `--expert-cache 128` |
| `gpu128-n32`, `cpu128-n32` | as `gpu128`, `cpu128`, 32 tokens |
| `gpu384`, `cpu384` | `--expert-cache 384 -ub 21`, with and without the GPU remap, about 30.4 GiB of slots, no hog |
| `gpu384-c2` | `gpu384`, `KOLIBRI_GATE_CHUNK_MAX=2`: a command buffer per two gates, no stops |
| `gpu384-c2-idle`, `n32-c2-sleep` | that with `KOLIBRI_GATE_IDLE_US=600`: `usleep(600)` between command buffers, 0.9 ms in effect; no I/O |
| `n32-c2-spin` | that with a 900 µs spin instead of the sleep, `KOLIBRI_IDLE_SPIN=1`; the GPU is idle as long, the CPU is not |
| `n32-fs18` | `KOLIBRI_GATE_FORCE_STOP=18`: before each token 18 layers' gate tables are emptied, so their gates stop though every expert has a slot. Stops, zeroed dispatches and re-encoding, with no load and no idle |
| `n32-fs18-sleep` | that with `KOLIBRI_STOP_IDLE_US=680`: a sleep at each stop: 1.03 ms per stop in effect, against a real wait of 0.76 ms |
