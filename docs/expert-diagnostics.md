# Expert cache diagnostics (Phase 11, Stage E1)

Stage E1 measures where the Stage D expert cache loses time before anything new
is built ([PLAN.md](../PLAN.md), Phase 11). It has four parts: an overhead
control, a page-cache A/B, the SSD's random-read curve, and two upstream
eviction policies replayed offline. All runs are on the M5 Pro (48 GiB,
APPLE SSD AP1024Z), on AC power, with no other heavy process.

## Verdict

- **E2 is warranted.** With all 384 experts cached, so that nothing is
  loaded after warmup, decode still takes 35.6 ms/token against 16.8 ms for
  resident weights at the same ubatch size. The load wait is 0.7 ms and the
  remap op 0.5 ms; the remaining 17.6 ms come with the 100 extra graph
  splits per token (102 against 2). E2's threshold was 10 ms/token.
- **Buffered reads stay opt-in.** They win without a ceiling (IQ3 decode
  21.1 against 17.6 tokens/s) and lose at the 16 GiB ceiling (15.2 against
  16.9), where the page cache competes with the hog. The page cache then
  holds up to 20.8 GiB that `phys_footprint` does not show.
- **The cache's I/O path, not the SSD, limits misses.** One IQ3 expert in
  three reads costs 0.61 ms on the SSD (5,560 experts/s at four in flight);
  about 27 decode misses per token should cost about 5 ms. The cache waits
  15.3 ms. One contiguous read per expert is 2.2× faster at one in flight
  and 1.37× at four, which supports E3.
- **Shared SLRU and tuned hotness pass the Stage A gate** on both models.
  Their decode gains are small (under 1 point of hit rate at 128 experts per
  layer); prefill gains are large (about 21% to 34–36% at 128 per layer).
  They are candidates for E4's prefill admission, not for the decode default.
- **Open: Q8_0 decodes at 5.5 tokens/s against Stage D's 13.0.** See
  [Open question](#open-question-q8_0-against-stage-d).

## Overhead control and decode split

Patch 0012 (`patches/llama.cpp/0012-kolibri-expert-cache-diag.patch`, measured at fork
`746be1eae`) adds decode wall time, remap time, graph splits and Metal
command-buffer commits to the stats JSON, also for resident runs. "Other" is
wall time minus load wait minus remap.

`go run ./cmd/kolibri-bench -matrix testdata/bench/e1.json -out ~/models/eval/expert-bench-e1`,
2 repetitions, ranges in brackets, IQ3 = `Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf`:

| Config | Ceiling | Decode ms/token | Wait | Remap | Other | Splits/token | Commits/token | Warm decode t/s |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| iq3-resident-ub21 | 0 GiB | 16.82 (16.69–16.96) | 0.00 | 0.000 | 16.82 | 2 | 3 | 59.87 |
| iq3-cache384-ub21 | 0 GiB | 35.64 (35.09–36.20) | 0.71 | 0.530 | 34.41 | 102 | 103 | 28.73 |
| iq3-cache128 | 0 GiB | 57.81 (56.40–59.22) | 15.27 | 0.629 | 41.91 | 102 | 103 | 17.61 |
| iq3-cache128 | 16 GiB | 59.90 (59.72–60.09) | 15.67 | 0.638 | 43.60 | 102 | 103 | 16.94 |

- Every cached decode token runs 2 + 2 × 50 splits: each MoE layer's CPU remap
  op splits the graph, and the scheduler syncs before and after it. Each
  Metal split costs one commit, plus one for the output copy.
- 17.6 ms of 18.8 ms overhead at 384 slots is neither waiting nor remapping,
  about 0.18 ms per extra split. That is the cost E2 removes.
- At 128 slots "other" is 7.5 ms higher than at 384. That part is not yet
  attributed; candidates are the miss path's I/O submission and the remap's
  wait handshake.
- Stage D's "about 23 ms unexplained" is therefore about 17.6 ms of
  split overhead plus about 7.5 ms that appears only with misses.
- **Control M42:** resident decode runs 2 splits per token, the count in
  [real-checkpoint.md](real-checkpoint.md). The tiny check
  (`tools/gguf/check_expert_cache.py`) asserts 2 resident and 2 + 2 per layer
  with the cache on Metal; adding one split per ubatch made 64 checks fail.

## Page-cache A/B

`--expert-cache-buffered` reads misses through the page cache instead of
with `F_NOCACHE`. "Page cache" is the GGUF's resident pages (`mincore`),
sampled during the run.

| Config | Ceiling | Warm decode t/s | Warm p95 ms | Wait ms/token | Disk GB read | Footprint GiB | Page cache peak GiB | Gate |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| iq3-cache128 | 0 GiB | 17.61 | 84.9 | 15.27 | 448 | 10.9 | 2.7 | |
| iq3-cache128-buffered | 0 GiB | 21.13 | 65.1 | 6.07 | 107 | 10.9 | 20.8 | passed |
| iq3-cache128 | 16 GiB | 16.94 | 83.5 | 15.67 | 448 | 10.9 | 2.7 | |
| iq3-cache128-buffered | 16 GiB | 15.22 | 94.9 | 24.31 | 414 | 10.9 | 8.3 | not passed |
| q8-cache128 | 0 GiB | 5.50 | 819.2 | 127.86 | 963 | 25.6 | 10.1 | |
| q8-cache128-buffered | 0 GiB | 8.10 | 550.8 | 87.07 | 1080 | 25.6 | 9.4 | passed |

- Without a ceiling, buffered reads let the page cache act as a second tier
  of about 20 GiB, and disk reads drop from 448 to 107 GB per run. The
  footprint does not show it.
- At 16 GiB the page cache shrinks to about 8 GiB, the hog evicts it, and
  the wait rises above the `F_NOCACHE` run's. Buffered is worse there.
- Q8_0 at 0 GiB passes by the rule, with wide ranges (6.87–9.34 tokens/s).
- All buffered runs generate the token IDs of their `F_NOCACHE` arm, and all
  IQ3 cache runs those of resident `-ub 21` (50 of 50 requests). No expert
  load failed.

## SSD curve

`cmd/kolibri-ssd` ([ssd-curve.md](ssd-curve.md)) reads the GGUF's expert
tensors at random after evicting the file. At four reads in flight,
`F_NOCACHE`:

| Read | IQ3 (1,699,840 B) | Q8_0 (4,177,920 B) |
|---|--:|--:|
| one expert, three reads, QD 1 | 0.61 ms, 1,633/s | 0.92 ms, 1,058/s |
| one expert, one read, QD 1 | 0.28 ms, 3,223/s | 0.48 ms, 2,036/s |
| one expert, three reads, QD 4 | 5,558/s, 9.4 GB/s | 2,926/s, 12.2 GB/s |
| one expert, one read, QD 4 | 7,595/s, 12.9 GB/s | 3,294/s, 13.8 GB/s |

- The SSD saturates at about 13–14 GB/s from four reads in flight.
- Three reads per expert are latency-bound at low queue depth; one read is
  2.2× faster at QD 1 for IQ3. That is E3's case.
- `F_NOCACHE` and buffered reads have the same cold speed within a few
  percent.
- **Control M41:** warm buffered reads of up to 256 KiB are 9–35× faster than
  evicted ones, so the per-cell eviction works.

## Eviction policies offline

`go run ./cmd/kolibri-policy -models Kolibri-1-IQ3_XXS-IQ4_XS-down-imx,Kolibri-1-Q8_0 -md docs/expert-policy.md`
replays SLRU (upstream #29887) and decaying hotness (#25294) with the Stage A
splits and gate ([expert-policy.md](expert-policy.md), "Stage E1").

| Experts per layer | Phase | Layer LRU | Shared LRU | SLRU per layer | SLRU shared | Hotness | Hotness tuned |
|---|---|--:|--:|--:|--:|--:|--:|
| 16 | decode | 51.5% | 50.9% | 53.7% | 53.7% | 54.0% | 55.4% |
| 128 | decode | 91.4% | 92.5% | 92.2% | 93.1% | 92.1% | 92.2% |
| 128 | prefill | 20.7% | 7.8% | 34.4% | 30.3% | 36.1% | 35.8% |

- All new policies pass the gate on both models. The shared SLRU at 128 is
  the first causal policy to beat the shared LRU there for IQ3: 35 against
  38 MB/token.
- **A step is one event now.** The replay counts step evictions: an
  eviction of an expert the running step already used, which the runtime
  never does. LFU, ARC and the per-layer LRU (prefill only) had them; each
  has a step-pinned variant, and the published numbers are unchanged.
- Upstream's SLRU half split collapses when the entry segment holds less
  than one token's experts across all layers: shared SLRU at 8 per layer
  hits 11.5% in decode against 34% for LRU.
- **Control M40:** SLRU with a protected fraction of 0 replays exactly like
  LRU wherever a step fits the cache (`TestPolicySLRUZeroIsLRU`).

## Open question: Q8_0 against Stage D

Q8_0 with 128 slots decodes at 5.50 tokens/s here and at 13.00 in Stage D,
with 127.9 ms/token of load wait. The hit rate (90.3%) and the bytes read
(963 against 940 GB) match Stage D. A first pass of this matrix with Docker
Desktop running gave 5.12; the IQ3 arms then matched these within a few
percent. The SSD reads Q8_0 experts at 2,926/s, so 29 misses per token
should cost about 10 ms.

Not yet ruled out: patch 0012 (it adds a sync after each decode ubatch when
stats are on), the disk being 98% full (22 GiB free), and the state of the
Q8_0 file. One Stage D build run of `q8-cache128` (fork `2da6ec661`, about
14 min) separates the patch from the machine. Until that is done, the Q8_0
numbers here compare only with each other.

## Reproduce

```
go run ./cmd/kolibri-policy -models Kolibri-1-IQ3_XXS-IQ4_XS-down-imx,Kolibri-1-Q8_0 -md docs/expert-policy.md
go build -o ~/models/eval/ssd/kolibri-ssd ./cmd/kolibri-ssd
~/models/eval/ssd/kolibri-ssd -json ~/models/eval/ssd/sweep.jsonl
~/models/eval/ssd/kolibri-ssd -expert -json ~/models/eval/ssd/expert.jsonl
~/models/eval/ssd/kolibri-ssd -no-evict -modes buffered -chunks 64KiB,256KiB -json ~/models/eval/ssd/m41-warm.jsonl
go run ./cmd/kolibri-bench -matrix testdata/bench/e1.json -out ~/models/eval/expert-bench-e1
go run ./cmd/kolibri-bench -matrix testdata/bench/e1.json -out ~/models/eval/expert-bench-e1 -report
```

The results are in `~/models/eval/expert-bench-e1/` (fork `746be1eae`). The
first pass, with Docker Desktop running, is kept in
`~/models/eval/expert-bench-e1-docker/` and is not used here.
