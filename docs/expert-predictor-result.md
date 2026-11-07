# Expert-ID prediction does not beat a plain expert cache on Kolibri-1

No predictor built from recorded expert IDs passed the gate. Last-N reuse, a decayed-count Markov predictor and a trained linear predictor were replayed causally on held-out decode, over two horizons and three declared links, for IQ3 and Q8_0. None loaded fewer bytes or ran faster than the best simple cache policy at any of 10 budgets. Without its CPU cost, no Markov or linear variant is faster either; the gate's best case is equal, at 384 experts per layer. At runtime, prefetch passed one of three comparisons, at 16 GiB on p95 alone, by about 5%, with no fewer misses.

The gap to the offline optimum is still large: on held-out decode at 128 experts per layer the best causal policy loads about 2.1 to 2.6 times the bytes of Belady's MIN. These predictors do not attack that gap. They prefetch into an LRU and leave its victim choice alone, and no predictor was tried as an eviction rule.

This writes up Phases 10 and 11 with Zhang's reporting checklist ([arXiv 2608.07911](https://arxiv.org/pdf/2608.07911)). Every number comes from a file in this repo, named beside it. Nothing was rerun for this document. A value marked "derived" is arithmetic on recorded values. Where the checklist asks for a number the repo does not hold, this document says it is not measured. Those places are collected in [Not measured](#not-measured).

The checklist and Zhang's own numbers come from [PLAN.md](../PLAN.md) and the research notes (`research_notes/AirLLM style offloading for Kolibri/moe_offloading.md`). The paper was not reread for this document. The notes describe it as single-author and unreviewed.

## Setup

| Item | Value | Source |
|---|---|---|
| Model | 50 layers, 384 experts per layer, sigmoid Top-6 routing | [offloading-survey.md](offloading-survey.md); [expert-policy.md](expert-policy.md), "Stage B: the decayed-count Markov predictor" |
| Quantizations | `Kolibri-1-IQ3_XXS-IQ4_XS-down-imx` and `Kolibri-1-Q8_0`, traces never pooled | [expert-policy.md](expert-policy.md), "LFU half-life, tuned on training prefixes" |
| Traces | Four workloads (coding, research, hr, medtech), teacher-forced, one int16 selection per layer and token | `tools/locality/experts.py`, `testdata/locality/manifest.json` |
| Replay order | As an online server: prompts in ubatches of 512, answers token by token | [expert-policy.md](expert-policy.md), head |
| Held-out split | Chronological, first floor(n/2) answers train, 4,402 decoded test tokens per model | [expert-policy.md](expert-policy.md), head and "Stage A gate"; `testdata/locality/splits.json` |
| Budgets | 8 to 384 experts per layer, equal bytes for every policy (128 per layer: 10.88 GB IQ3, 26.74 GB Q8_0) | [expert-policy.md](expert-policy.md), workload tables |
| Time model | An assumption: serial links at 5, 25 and 100 GB/s (ssd, pcie, ram), 15.7 ms per decoded token of resident compute | [expert-policy.md](expert-policy.md), head |
| Runtime machine | M5 Pro, 48 GiB unified memory | [expert-streaming.md](expert-streaming.md), "Method" |

PLAN and the survey state that no public study covers 384 experts with Top-6 sigmoid routing on unified memory. That was the state of the survey's search of 2026-10-22. In the notes, Zhang's models have 40, 64 and 128 experts.

## The gate

A predictor is retained only if, on all test suffixes of a model together, it beats the best Stage A policy at that budget: fewer bytes loaded, or less estimated time on a link, its CPU cost included ([expert-policy.md](expert-policy.md), "Stage B gate"). Ties and differences within a part in 10⁹ do not count. Prefetched bytes count as loaded.

**Result: 0 of 24 gate rows retained** (Markov and linear × 2 horizons × 3 links × 2 models). Last-N is not retained for either model either ("Stage A gate"). Stage E1's policies are not in the reference, so they cannot have caused this.

Decode ms/token on all test suffixes together at 128 experts per layer, with the predictor's cost and, in brackets, without it. All three columns are estimates from the time model above:

| Model | Predictor | Horizon | ssd | pcie | ram |
|---|---|---|--:|--:|--:|
| IQ3 | best Stage A policy | | 23.3 | 17.2 | 16.1 |
| IQ3 | Markov | next token | 23.5 (23.3) | 17.4 (17.2) | 16.3 (16.1) |
| IQ3 | Markov | same token | 23.6 (23.4) | 17.5 (17.2) | 16.3 (16.1) |
| IQ3 | linear | next token | 24.9 (24.6) | 17.8 (17.5) | 16.5 (16.1) |
| IQ3 | linear | same token | 25.3 (25.0) | 18.2 (17.8) | 16.6 (16.2) |
| Q8_0 | best Stage A policy | | 34.4 | 19.4 | 16.6 |
| Q8_0 | Markov | next token | 34.7 (34.5) | 19.7 (19.5) | 16.9 (16.6) |
| Q8_0 | Markov | same token | 35.0 (34.8) | 19.7 (19.5) | 16.9 (16.7) |
| Q8_0 | linear | next token | 38.8 (38.5) | 20.5 (20.2) | 17.2 (16.8) |
| Q8_0 | linear | same token | 40.7 (40.4) | 20.6 (20.3) | 17.3 (17.0) |

Source: [expert-policy.md](expert-policy.md), "Stage B: the decayed-count Markov predictor" and "Stage B: a trained linear predictor". A printed tie such as 23.3 against 23.3 is a rounded one; the gate compares unrounded times, and it finds the predictors without cost at best equal, at 384 experts per layer ([PLAN.md](../PLAN.md), Stage B, gate item).

Last-N at the same budget, against the shared LRU, on the same three links: IQ3 28.5, 18.2, 16.3 against 23.3, 17.2, 16.1; Q8_0 47.1, 22.0, 17.3 against 34.5, 19.5, 16.6 ("Last-N reuse, the first prediction baseline").

**Margins** ([PLAN.md](../PLAN.md), Stage B):
- With its cost the Markov predictor is at least 0.96% (IQ3) and 0.77% (Q8_0) slower than the best Stage A policy on its own link, at every size and for both horizons.
- With its cost the linear predictor is at least 2.2% slower everywhere.

**Small budgets are where prediction does harm.** At 8 experts per layer on ssd, IQ3 takes 79.0 ms/token with the best Stage A policy, 115.0 and 176.1 with next-token and same-token Markov, and 182.5 and 177.2 with linear.

**Cost.** Markov: 4.4 µs per decoded layer step, 0.22 ms per token (1.4% of 15.7 ms), 59 MB of tables. Linear: 6.8 µs, 0.34 ms per token (2.2%), 59 MB of weights. The replays add this to the critical path ("Cost" paragraphs in both Stage B sections).

### What the predictors can see

Recall of the test suffixes' decoded experts per layer and token, IQ3; Q8_0 is within 0.1 points. Whole Top-6 sets in brackets.

| Predictor | k 6 | k 12 | k 24 | k 48 |
|---|--:|--:|--:|--:|
| Markov, next token | 35.6% (0.2%) | 50.3% (3.3%) | 63.5% (11.9%) | 75.2% (26.0%) |
| Markov, same token | 37.9% (0.2%) | 52.1% (3.5%) | 65.8% (13.8%) | 78.3% (34.0%) |
| linear, next token | 29.6% (0.2%) | 42.3% (2.7%) | 55.5% (9.1%) | 68.2% (19.7%) |
| linear, same token | 36.1% (0.5%) | 49.6% (4.9%) | 62.2% (14.2%) | 73.1% (28.3%) |

Baseline: last-N with n 1 covers 34.4% (0.4%) with 6.3 candidates, and with n 8 64.5% (12.5%) with 30.4 candidates. Source: [expert-policy.md](expert-policy.md), the three predictor sections.

**Recall does not turn into savings.** At 128 experts per layer on ssd, IQ3, next-token Markov uses 33 prefetches over the whole test set, because its top 6 are already in the cache ([PLAN.md](../PLAN.md), Stage B, horizons item). Same-token Markov uses 2,386, of which 94.8% arrive in time. Linear uses 29,829 same-token prefetches (82.9% in time, 5,102 late) and wastes 10 MB/token; PLAN reads that as crowding out cached experts. Recall among the experts that would otherwise miss is not measured, so for the same-token horizon this explanation is an inference.

**The lead is short.** Next token has 15.38 ms, enough for 45 IQ3 experts on ssd. Same token has 0.31 ms, enough for none on ssd, 4 on pcie and 18 on ram. Q8_0: 18, 92 and 368 experts next token; 0, 1 and 7 same token. Source: "Lead time" in the Markov section.

## Zhang's checklist

| Item | State | Gap |
|---|---|---|
| Per-event replay | Done for every cache policy since Stage E1, with step evictions counted | The oracle is not step-pinned; predictor replays do not report step evictions; no gate was rerun with step-pinned references |
| Decode-only hit rate | Reported apart from prefill for every cache policy | No hit rate for the predictors; no lifetime rate |
| Per-step union against capacity | Exact for decode, since a token needs six experts per layer | Prefill's union is not measured as a distribution |
| Belady gap | Per workload, whole trace at every budget, held out at 128 | No pooled held-out oracle; no step-pinned oracle |

### Per-event replay

A step's experts are one event: a decoded token's six in a layer, or a ubatch's union in a layer. A replay that takes them one access at a time can evict an expert the same step has already used, which no server does.

**Recorded step evictions, whole online replays of all workloads, at 128 experts per layer and over all sizes** ([expert-policy.md](expert-policy.md), "Stage E1"):

| Policy | IQ3 decode at 128 | IQ3 prefill at 128 | IQ3 decode, all sizes | IQ3 prefill, all sizes |
|---|--:|--:|--:|--:|
| shared LRU | 0 | 0 | 0 | 0 |
| layer LRU | 0 | 370,595 | 0 | 5,452,832 |
| LFU | 10,378 | 103,845 | 2,793,754 | 2,443,589 |
| ARC | 2,222 | 71,381 | 13,412 | 2,661,674 |
| all step-pinned variants, SLRU, hotness | 0 | 0 | 0 | 0 |
| oracle, per layer † | 3,128 | 222,253 | 354,489 | 1,611,702 |
| oracle, shared † | 1,436 | 144,513 | 232,868 | 915,403 |

Q8_0's table in the same section shows the same pattern.

**What per-event replay changes here:**
- **Prefill of the layer LRU moves most.** Held out at 128, IQ3, its hit rate goes from 20.7% to 32.1% when step-pinned (29 to 25 MB/token). Q8_0: 20.8% to 32.0% (70 to 60 MB/token). So the unpinned replay overstates this policy's prefill misses, the effect the notes report from Zhang. His 27.4% for LRU comes from other models and cache sizes and is not comparable.
- **Decode barely moves.** At 128 the layer LRU goes from 91.4% to 91.6%. It has no decode step evictions, so that difference comes from the state its prefill leaves behind. In the pooled held-out decode tables, LFU and ARC change by at most 1 MB/token at any budget from 8 to 256 (Q8_0 at 8 experts per layer: 818 to 819 and 809 to 810; the rest print equal).
- **The negative result does not depend on it.** The shared LRU has 0 step evictions at every size, and the predictors wrap it. IQ3 on ssd, next-token Markov with its cost is slower than the shared LRU at all 10 budgets, for example 23.5 against 23.3 at 128 and 115.0 against 82.8 at 8. Without its cost it equals the shared LRU, as printed to 0.1 ms, at every budget from 32 up ("Stage B: the decayed-count Markov predictor" and "Last-N reuse").

**Gaps:**
- **The oracle may evict within a step,** as the table shows. It is built from access order (`internal/locality/policies.go`). Any event-atomic schedule is also a valid access-level schedule, so its bytes remain a lower bound. A step-pinned oracle may load more, so the recorded gap may overstate the gap to an event-atomic optimum. By how much is not measured. The notes report that Zhang's per-access replay inflates Belady by 4.3%, which points the other way. That was not reconciled, since the paper was not reread.
- **The Stage B gate predates step-pinned policies.** Its reference still holds the unpinned LFU and ARC. It was not rerun with a pinned reference.
- **Stage B's tables have no step eviction column.**

### Decode-only hit rate

All workloads' test suffixes together, decode, MB/token (hit rate). Prefill is reported apart, since a prefill hit serves a teacher-forced ubatch and is no decode speedup.

| Experts per layer | IQ3 shared LRU | IQ3 SLRU, shared | IQ3 layer LRU | Q8_0 shared LRU | Q8_0 SLRU, shared |
|--:|--:|--:|--:|--:|--:|
| 8 | 335 (34.2%) | 451 (11.5%) | 336 (34.2%) | 824 (34.2%) | 1109 (11.5%) |
| 16 | 250 (50.9%) | 236 (53.7%) | 247 (51.5%) | 615 (50.9%) | 581 (53.6%) |
| 32 | 170 (66.7%) | 161 (68.3%) | 171 (66.5%) | 418 (66.7%) | 397 (68.3%) |
| 64 | 97 (80.9%) | 93 (81.7%) | 102 (79.9%) | 240 (80.8%) | 230 (81.7%) |
| 128 | 38 (92.5%) | 35 (93.1%) | 44 (91.4%) | 94 (92.5%) | 86 (93.1%) |
| 192 | 15 (97.0%) | 13 (97.4%) | 19 (96.2%) | 37 (97.0%) | 33 (97.4%) |
| 256 | 4 (99.2%) | 4 (99.3%) | 6 (98.8%) | 10 (99.2%) | 9 (99.3%) |

Source: [expert-policy.md](expert-policy.md), "Stage E1", "Held out". No cache loads 510 MB per decoded IQ3 token and 1,253 for Q8_0 (workload sections).

**Prefill at the same budget tells a different story.** At 128 per layer the IQ3 shared LRU hits 7.8% in prefill against 92.5% in decode, and step-pinned LFU 38.9%. So a single rate over both phases would say little about decode.

**The runtime agrees with the replay's range.** The fork's layer LRU cache at 128 slots measured 90.9% decode hits on IQ3 and 90.3% on Q8_0 ([expert-streaming.md](expert-streaming.md), "Results"). These runs use other requests (2,048 prompt and 128 generated tokens), so this is not a validation of the replay.

**Not recorded:** a predictor's decode hit rate, and its MB/token (the gate tables print only the verdict). A lifetime hit rate over both phases is not recorded for any policy.

### Per-step union against capacity

The notes report that Zhang takes this ratio, not batch size or cache fraction, as the regime variable: the experts a step needs, over what a layer can hold.

**Decode is exact.** A decoded token needs six distinct experts per layer (`tools/locality/experts.py` rejects any other selection), so the union is 6. Derived ratio, beside the recorded pooled decode hit rate of the IQ3 shared LRU:

| Experts per layer | Union / capacity (derived) | IQ3 shared LRU hit rate |
|--:|--:|--:|
| 8 | 0.75 | 34.2% |
| 16 | 0.38 | 50.9% |
| 32 | 0.19 | 66.7% |
| 48 | 0.13 | 74.8% |
| 64 | 0.09 | 80.9% |
| 96 | 0.06 | 88.0% |
| 128 | 0.05 | 92.5% |
| 192 | 0.03 | 97.0% |
| 256 | 0.02 | 99.2% |
| 384 | 0.02 | 100.0% |

So a decoded step's union never exceeds a layer's capacity at any budget tested. The notes place the regime where smarter policies win over a thrashing LRU at a union above capacity, and single-stream decode does not reach it here.

**A shared cache has its own limit.** A token needs 300 experts across 50 layers (derived). Upstream's SLRU protects at most half the budget, and its entry segment then holds fewer than one token's experts at 8 per layer. There the shared SLRU falls to 11.5% against the shared LRU's 34.2% ([expert-diagnostics.md](expert-diagnostics.md), "Eviction policies offline").

**Prefill is not measured as a distribution.** What the repo holds:
- **An upper bound:** `n_touch = min(n_expert, n_tokens * k)` ([expert-streaming.md](expert-streaming.md), "How waves work"). Derived: 384 at ubatch 512, a ratio of up to 3.0 at 128 slots. At the cache's cap of 21 tokens it is 126, a ratio of 0.98, which is why the cap exists ("The ubatch cap", same document).
- **One range:** a replay put 6 to 35% of the layers' unions at ubatch 512 above 244 experts ("Lossless", same document). No file records that replay's parameters or per-layer values.
- **Effects:** at 128 per layer the layer LRU has 370,595 prefill step evictions and none in decode, and held-out prefill set recall is at most 2.9% for every policy in the table, the oracle included ([expert-policy.md](expert-policy.md), workload sections).

The mean, median and p95 union per ubatch and layer are not measured. Prefill unions are also not this result's subject: the predictors predict decoded steps only.

### Belady gap

Decode on each workload's held-out suffix at 128 experts per layer. Best causal is the lowest MB/token among the causal rows of that table (ties on printed MB broken by hit rate). The ratio is derived from the printed, rounded MB/token.

| Model | Workload | Best causal | MB/token | Hit | Whole Top-6 sets | Oracle, shared †: MB/token | Hit | Whole Top-6 sets | Ratio | ssd ms/token, best causal against oracle |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| IQ3 | coding | SLRU, shared | 47 | 90.7% | 60.0% | 21 | 95.9% | 79.3% | 2.2 | 25.1 / 19.9 |
| IQ3 | research | SLRU, shared | 32 | 93.7% | 70.1% | 15 | 97.1% | 85.0% | 2.1 | 22.1 / 18.6 |
| IQ3 | hr | SLRU, shared | 29 | 94.4% | 73.7% | 12 | 97.7% | 88.2% | 2.4 | 21.4 / 18.0 |
| IQ3 | medtech | LFU | 31 | 93.9% | 70.9% | 12 | 97.6% | 87.1% | 2.6 | 21.9 / 18.1 |
| Q8_0 | coding | SLRU, shared | 117 | 90.7% | 59.9% | 52 | 95.9% | 79.3% | 2.3 | 39.0 / 26.0 |
| Q8_0 | research | SLRU, shared | 79 | 93.7% | 69.9% | 36 | 97.1% | 85.0% | 2.2 | 31.5 / 22.9 |
| Q8_0 | hr | SLRU, shared | 70 | 94.4% | 73.8% | 29 | 97.7% | 88.2% | 2.4 | 29.7 / 21.5 |
| Q8_0 | medtech | LFU | 77 | 93.9% | 70.8% | 30 | 97.6% | 87.1% | 2.6 | 31.0 / 21.7 |

Source: [expert-policy.md](expert-policy.md), the "Held-out test suffix at 128 experts per layer" table of each workload.

**The earlier "2.4–2.5×" needs its scope.** [PLAN.md](../PLAN.md), Stage A, and the survey give that ratio for whole traces at 128 on IQ3, before Stage E1's policies existed. With them, whole-trace best causal against the shared oracle at 128 is 49/22, 32/13, 31/13 and 30/12 MB/token.

**The ratio mostly grows with the budget, while the bytes at stake fall.** Whole traces, decode, IQ3, best causal row against the shared oracle, MB/token (hit rate), and the derived ratio range over the four workloads for both models. The table stops at 192 because the rounded MB/token above that are too small for a ratio:

| Experts per layer | coding | research | hr | medtech | IQ3 ratio | Q8_0 ratio |
|--:|--:|--:|--:|--:|--:|--:|
| 8 | 337 (33.9%) / 224 (56.1%) | 297 (41.7%) / 191 (62.5%) | 314 (38.5%) / 203 (60.2%) | 308 (39.5%) / 199 (60.9%) | 1.50–1.55 | 1.50–1.56 |
| 16 | 261 (48.8%) / 162 (68.2%) | 214 (58.1%) / 130 (74.5%) | 228 (55.2%) / 139 (72.7%) | 221 (56.7%) / 139 (72.7%) | 1.59–1.65 | 1.59–1.64 |
| 32 | 189 (63.0%) / 105 (79.4%) | 145 (71.6%) / 79 (84.5%) | 153 (70.0%) / 84 (83.6%) | 151 (70.3%) / 87 (82.9%) | 1.74–1.84 | 1.73–1.84 |
| 64 | 116 (77.2%) / 56 (89.0%) | 83 (83.6%) / 40 (92.2%) | 82 (84.0%) / 40 (92.2%) | 89 (82.5%) / 42 (91.8%) | 2.05–2.12 | 2.06–2.12 |
| 128 | 49 (90.4%) / 22 (95.8%) | 32 (93.8%) / 13 (97.5%) | 31 (93.9%) / 13 (97.5%) | 30 (94.0%) / 12 (97.7%) | 2.23–2.50 | 2.28–2.59 |
| 192 | 20 (96.0%) / 8 (98.5%) | 11 (97.8%) / 4 (99.1%) | 13 (97.5%) / 6 (98.8%) | 10 (98.0%) / 4 (99.3%) | 2.17–2.75 | 2.13–2.80 |

Source: "Decode, MB/token (hit rate)" of each workload section. The oracle's bound is only on bytes: no cache of the budget, prefetching or not, loads fewer ([expert-policy.md](expert-policy.md), head).

**Prefetch cannot close this gap in bytes.** It loads no fewer bytes than the misses it replaces, and can only hide their time ("Stage B: the decayed-count Markov predictor"). In the notes, 84.3–96.6% of Zhang's gap comes from knowing which resident expert is used furthest ahead, and a causal next-use predictor used as the eviction rule recovers −11.4% of it. He scopes that to residency and next-use prediction, not next-expert prefetch. Stage B is next-expert prefetch, so the two results are complementary rather than one confirming the other. **This repo has no causal next-use eviction predictor, so it has no result on victim prediction.**

**Not recorded:** an oracle pooled over all test suffixes, so the pooled table above has no oracle column. Held-out oracle values exist only at 128 experts per layer. The notes do not define the denominator of Zhang's "44.2–45.9% gap", so it is not compared with these ratios.

## The runtime prefetch

Stage D ported the Markov predictor's same-token horizon to C++ as `--expert-prefetch`, because the user chose on 2026-10-14 to build the prototype although no predictor had passed. Its benchmark is `testdata/bench/expert-cache.json`, with 3 repetitions per arm and ceiling ([expert-streaming.md](expert-streaming.md), "Gates").

| Comparison, `--expert-prefetch 12` against the demand cache | Decode t/s | p95 ms | Verdict |
|---|--:|--:|---|
| IQ3, 16 GiB | 18.29 (17.77–18.91) against 17.13 (16.92–19.05) | 77.4 (77.0–80.7) against 81.9 (81.6–85.3) | passed by the rule, on p95 alone |
| IQ3, 0 GiB (INFO) | 18.43 (18.22–18.52) against 18.51 (18.25–18.77) | 78.9 (75.5–83.6) against 78.9 (77.8–80.7) | not passed |
| Q8_0, 0 GiB (INFO) | 12.73 (12.02–12.81) against 13.00 (12.59–13.09) | 116.1 (115.2–118.1) against 111.9 (110.3–117.3) | not passed |

At 12 per layer it issues 4,252 prefetches per IQ3 run. 1,541 are used, 2,027 are evicted unused, and it reads 6.8 GB more. Decode misses hardly move (17,315 against 17,424 of 190,800 needed), nor does the wait on loads (9.5 against 9.2 s). A loaded-machine Q8_0 smoke run gave 7.67 against 11.25 tokens/s (INFO, not a benchmark).

So the runtime's one pass, about 5% at the p95 under one ceiling, comes with no fewer misses. It is not evidence that prediction beats the cache. The offline gate does not contradict it either, since the offline replay models neither ceilings nor tail latency. The cause of the p95 change is not established. The third repetition at 16 GiB ran four days after the demand cache's. Stage D left keeping the option to the user, and PLAN records no decision.

**Not recorded:** any prefetch arm on the rebased fork (the matrices of E0, E1 and E4 have none), and Q8_0 prefetch at 16 GiB, where 128 slots do not fit ([expert-diagnostics.md](expert-diagnostics.md), "Page-cache A/B").

## What this result does not cover

- **Hidden states.** Stage B used expert IDs only, since no router logits or hidden states are recorded. Stage E6, a hidden-state predictor, remains open. [PLAN.md](../PLAN.md), Stage E6, cites 90–96% one layer ahead on Mixtral and DeepSeek-MoE, 67% on Qwen2-MoE, and nothing for Kolibri's routing.
- **Other link speeds.** Only the three declared links ran. [ssd-curve.md](ssd-curve.md) measures 13–14 GB/s at saturation on this Mac, between ssd and pcie. Predictors fail on both sides of it, but that speed was not replayed.
- **Real compute time.** The replay computes 0.314 ms per layer, from resident speed. The cache's real decode takes 35.6 ms/token even with all 384 experts cached ([expert-diagnostics.md](expert-diagnostics.md)), so real lead times are longer. Stage E2 may change this. No replay uses those times.
- **Concurrent misses.** The time model serves one load at a time. The runtime has four I/O threads.
- **Other requests and models.** One deterministic trace set of four workloads, single stream. The notes' batch and multi-request results are not addressed.
- **Lossy miss handling.** This is lossless only; Stage E5 is separate.

## Not measured

Each of these is absent from the repo, and no estimate is given in its place.

1. Prefill's union per ubatch and layer as a distribution (mean, median, p95).
2. A step-pinned or otherwise event-atomic oracle, and so the gap to it.
3. A rerun of the Stage B gate with step-pinned references.
4. Step evictions of the predictor replays.
5. A predictor's decode hit rate and MB/token (only recall, ms/token, used and unused prefetches, and the verdict are in the tables).
6. An oracle pooled over all test suffixes, and held-out oracle values at budgets other than 128.
7. A lifetime hit rate over prefill and decode together.
8. Recall restricted to experts that would otherwise have missed. This would directly measure the "already resident" explanation, which now rests on the used-prefetch counts.
9. A causal next-use predictor as the eviction rule.
10. Template diversity, which the notes list beside the four items PLAN names. The workloads have no within-class expert Jaccard or check for identical generated prefixes.
11. Replays at the measured 13–14 GB/s, or with the cache's real step time.

Items 1 to 8, 10 and 11 can be computed from the selection files under `~/models/eval/locality/`, which `cmd/kolibri-policy` reads, without loading a weight. They need new replay code or a new report. Neither was part of this document, and item 9 needs a new predictor. `kolibri-policy` also writes `policy-<model>.json` outside the repo, with unused prefetch bytes and per-layer recall; it was not read here.

## Reproduce

```sh
go run ./cmd/kolibri-policy -models Kolibri-1-IQ3_XXS-IQ4_XS-down-imx,Kolibri-1-Q8_0 -md docs/expert-policy.md
```

The last duration PLAN records is 22 minutes, with 80 PASS lines, from before Stage E1 ([PLAN.md](../PLAN.md), Stage B). Selection files are verified with `tools/locality/experts.py --verify`. Recording them needs the weights; replaying them does not. Tests are in `internal/locality` and `cmd/kolibri-policy`, for example `TestReplayCausal`, `TestTuneNoLeak`, `TestMarkovNoLeak`, `TestLinearNoLeak`, `TestStageBGate` and `TestPolicyOracleOptimal`.
