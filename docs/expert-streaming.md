# Expert streaming: the runtime expert cache

Phase 10, Stage D, item 1: a bounded, byte-accounted cache of routed experts inside libllama (patch 0011, fork branch `feat/kolibri-expert-cache`). It is experimental and Kolibri only. Stage D was started by the user's decision (2026-10-14) although no predictor passed the offline gates of Stages A and B ([expert-policy.md](expert-policy.md)). This document covers the prototype and its correctness checks. The benchmark (item 2) and the gates (item 3) are still open, so every speed below is INFO, measured on a busy machine.

## How it works

- **The file keeps the experts.** The stacked expert tensors `ffn_{gate,up,down}_exps` of every layer become lazy tensors: mapped on the CPU, excluded from the loader's `WILLNEED` advice, and never read through the mapping. Nothing places them on the GPU.
- **Slots.** Each layer gets slot tensors of the same type and row size, with `n_slots` experts instead of 384. They live in one buffer of the layers' device (on Apple Silicon a shared Metal buffer). The buffer counts as weights, so the scheduler runs the expert matmuls where the slots are.
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
| `--expert-cache-stats FNAME` | write the counters as JSON when the model is freed |

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
- `process`, measured since the cache was set up: `disk_bytes_read` and `pageins` from `proc_pid_rusage`, `major_faults` from `getrusage`, and the lifetime peak `max_phys_footprint`.

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
  - `llama-completion` with `--expert-cache-layers`, `--expert-prefetch 8,4,0.25`, `--expert-cache-threads 2` and `--expert-cache-stats` runs as asked (the stats record it) and generates the same greedy text as without the cache.
- **Result:** "PASS: 0 failures", 97 checks. The logits of all 100 positions are bit-identical in every configuration, and the counters are consistent.

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

## Limits

- One context per model; a second context with the cache fails to be created.
- Kolibri only, no LoRA on the routed experts. The experts must be in one file.
- A ubatch of one token counts as decode, also in a prompt. In a prefill ubatch the last layer sees only the output rows, so the predictor's counts of that layer cover fewer tokens than the trace replays of Stage B.
- LRU only. Layer budgets can be set with `--expert-cache-layers`, but nothing writes the Stage A allocation to such a file yet.
- `-fit` does not know about the slot buffer.
- libllama's INFO lines, including the cache's setup line, appear only with `-v`; the counters go to `--expert-cache-stats`.
- Platforms:
  - macOS opens the file again through `F_GETPATH` and reads it with `F_NOCACHE`;
  - Linux reopens `/proc/self/fd/N`, without bypassing the page cache;
  - other systems are unsupported.
