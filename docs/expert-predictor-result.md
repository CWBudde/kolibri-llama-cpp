# Expert-ID prediction does not beat a plain expert cache on Kolibri-1

No predictor built from recorded expert IDs passed the gate. Last-N reuse, a decayed-count Markov predictor and a trained linear predictor were replayed causally on held-out decode, over two horizons and three declared links, for IQ3 and Q8_0. None loaded fewer bytes or ran faster than the best simple cache policy at any of 10 budgets. Without its CPU cost, no Markov or linear variant is faster either; the gate's best case is equal, at 384 experts per layer. At runtime, prefetch passed one of three comparisons, at 16 GiB on p95 alone, by about 5%, with hardly fewer misses.

The gap to the offline optimum is still large: on held-out decode at 128 experts per layer the best causal policy loads at least 2.71 to 3.09 times the bytes of a bound on that decode, which is Belady's MIN with the training prefix and every prefill free. Holding a step's experts together, as a server must, hardly changes that: 2.70 to 3.07 times. Against the schedule MIN picks for the whole trace, which pays for those too, it is 2.20 to 2.54 times. These predictors do not attack that gap. They prefetch into an LRU and leave its victim choice alone, and no predictor was tried as an eviction rule.

This writes up Phases 10 and 11 with Zhang's reporting checklist ([arXiv 2608.07911](https://arxiv.org/pdf/2608.07911)). Every number comes from a file in this repo, named beside it. Stage E7 filled the checklist's gaps with one new replay of the same selection files, with the same rules, so the earlier tables keep their values; its tables are in [expert-policy.md](expert-policy.md), "Stage E7". A value marked "derived" is arithmetic on recorded values. Where this document says a number is not measured, the repo does not hold it. Those places are collected in [Not measured](#not-measured).

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

| Item | State | What is left |
|---|---|---|
| Per-event replay | Done for every cache policy, the predictors and a step-pinned oracle, with step evictions counted, and the Stage B gate rerun with pinned references | Nothing on this list |
| Decode-only hit rate | Reported apart from prefill for every cache policy and predictor, with a lifetime rate beside it | Nothing on this list |
| Per-step union against capacity | Exact for decode, since a token needs six experts per layer, and measured as a distribution for prefill | Nothing on this list |
| Belady gap | Per workload and pooled, held out, at every budget, against a bound on held-out decode and its step-pinned twin | The pooled oracle is a sum of per-workload oracles, not one over joined traces. The bound pays for no prefill, so at large budgets it says little about a schedule that does |

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

**The step-pinned oracle barely moves the bound.** The oracle above may evict within a step, and it may use an expert once without keeping it, so it is not a schedule a server could run. A step-pinned oracle holds a step's experts together. Every expert of a model has one size (`testdata/locality/expert-bytes.json`), and the largest union of any ubatch, 333 experts on IQ3 and 332 on Q8_0, fits the shared budget at every size, so the pinned shared oracle is an atomic schedule throughout. On small random traces `TestPinnedOracleOptimal` finds it equal to the event-atomic optimum by exhaustive search, which is a test and not a proof. Its bytes above the plain oracle's, for 8 to 256 experts per layer, derived from unrounded bytes:

| Pinned against plain oracle, shared | Decode | Prefill |
|---|--:|--:|
| Whole traces, IQ3 | 0.47% to 1.53% | 0.04% to 2.31% |
| Whole traces, Q8_0 | 0.33% to 1.58% | 0.02% to 2.31% |
| Held-out test suffixes, IQ3, pinned against the held-out bound | 0.00% to 3.30% | |
| Held-out test suffixes, Q8_0, pinned against the held-out bound | 0.00% to 3.05% | |

At 384 they are equal. The held-out rows are per workload, for 8 to 256 experts per layer, and compare the two oracles built for held-out decode alone ([Belady gap](#belady-gap)); pooled they are 0.00% to 2.59% and 0.00% to 2.33%. So the gap to the bound does not overstate the gap to an event-atomic optimum by more than these margins. An earlier version of this table compared slices of the whole-trace oracles, got -0.10% to 0.83%, and put the negative values down to those oracles being built for the whole trace. That was the sign the slices are no bound, and it is what the section below corrects.

**A per-layer budget is different.** The pinned per-layer oracle loads 7.02% to 29.36% more decode bytes than the plain one on whole traces, for 8 to 256 per layer. That is not an atomic schedule below the unions in the next section: what does not fit is used once. The cause of the margin was not isolated. Candidates are the lost use-once at small budgets and a prefill that overflows the layer.

**The notes report that Zhang's per-access replay inflates Belady by 4.3%,** which points the other way from these margins. That was not reconciled, since the paper was not reread.

**The negative result does not depend on references that evict within a step.** Stage B's gate was rerun against only the shared LRU, the step-pinned layer LRU, LFU and ARC, and tuned last-N, each with 0 step evictions in every test suffix, a check the replay enforces (control M94). LFU with a tuned half-life and the allocated layer budgets were left out, having no pinned twin. So this reference is weaker: it needs up to 8.89% (IQ3) and 6.95% (Q8_0) more bytes and up to 3.59% more time on ssd than the recorded one, and a predictor passes it more easily. **Result: 0 of 24 rows retained again** ([expert-policy.md](expert-policy.md), "Stage E7: Stage B gate, step-pinned references"). The closest, in both models, is next-token Markov with k tuned for ssd. Excluding the tie at 384, it loads 0.07% (IQ3) and 0.24% (Q8_0) more bytes at 128 experts per layer. It is at best 0.95% (IQ3, 64 per layer) and 0.76% (Q8_0, 128 per layer) slower on ssd. A predictor wraps the shared LRU, which stays in the reference, so it has to beat its own base in any case.

**The predictors themselves have no step evictions:** 0 in all 1,280 replays, 640 per model, at every k and budget.

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

**A predictor's hit rate rises without saving bytes.** All test suffixes together, decode, with k tuned for ssd on the own prefix, MB/token counting everything loaded, prefetched or missed, used or not ([expert-policy.md](expert-policy.md), "The predictors' hit rates, bytes and step evictions"):

| Experts per layer | Shared LRU | Markov, next token | Markov, same token | Linear, next token | Linear, same token |
|--:|--:|--:|--:|--:|--:|
| IQ3, 8 | 335 (34.2%) | 496 (27.4%) | 801 (38.0%) | 832 (0.0%) | 806 (36.3%) |
| IQ3, 128 | 38 (92.5%) | 38 (92.5%) | 39 (92.6%) | 50 (92.6%) | 56 (93.2%) |
| Q8_0, 8 | 824 (34.2%) | 1259 (25.6%) | 1969 (38.0%) | 2046 (0.0%) | 1984 (36.2%) |
| Q8_0, 128 | 94 (92.5%) | 94 (92.5%) | 96 (92.6%) | 124 (92.6%) | 138 (93.2%) |

At 128, IQ3, same-token linear has the highest hit rate, 0.7 points above the shared LRU, for 47% more bytes (derived from 56.2 and 38.1 MB/token). At 8 per layer next-token linear's prefetches push out what the step needs: a hit rate of 0.0%.

**A lifetime hit rate ranks the policies by prefill.** Hits over accesses in both phases, whole online replays, all workloads together, IQ3 at 128 per layer; Q8_0 is within 0.1 points ([expert-policy.md](expert-policy.md), "Lifetime hit rate"):

| Policy | Lifetime | Decode, held out (table above) |
|---|--:|--:|
| hotness, per layer | 74.2% | |
| SLRU, per layer | 73.8% | |
| layer LRU, step-pinned | 72.5% | |
| LFU | 70.3% | |
| SLRU, shared | 69.2% | 93.1% |
| layer LRU | 68.9% | 91.4% |
| ARC | 65.6% | |
| shared LRU | 64.9% | 92.5% |
| oracle, shared † | 86.9% | |
| oracle, shared, step-pinned † | 86.7% | |

Prefill is 31% of the accesses here (derived from 1,087,122 of 3,480,222, shared LRU), since a step counts each expert of its union once. So the shared LRU comes last on this rate although it beats the layer LRU in decode.

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

**Prefill's union is measured.** A step is one ubatch at one layer, and every ubatch counts, including a prompt's short last one ([expert-policy.md](expert-policy.md), "Prefill's union per ubatch and layer"). Ranges are over the four workloads; Q8_0's differ by at most 0.4 points, 1 expert in the quantiles and 2 in the maximum:

| IQ3 | Mean | Median | p95 | Max | Above 128 | Above 244 |
|---|--:|--:|--:|--:|--:|--:|
| Ubatch 512 | 164.8–207.8 | 175–223 | 239–277 | 299–333 | 77.5–87.5% | 3.2–22.1% |
| Ubatch 21 | 46.1–54.5 | 46–55 | 66–74 | 90–97 | 0.0% | 0.0% |

**So prefill at ubatch 512 is in the regime decode never reaches.** At 128 per layer the median ratio of union to capacity is 1.37 to 1.74 (derived), and most steps cannot fit. The maximum of 333 stays below the upper bound of 384. At the cache's cap of 21 tokens the largest union seen is 97, a ratio of 0.76, against a bound of 126.

**The earlier "6 to 35% above 244" is not reproduced.** [expert-streaming.md](expert-streaming.md), "Lossless", took it from a replay with no recorded parameters. Here the share of steps above 244 at ubatch 512 is 2.9 to 22.1% per workload, and the share among a layer's full ubatches runs from 0% to 100% in three workloads and to 58.8% in the fourth. Its conclusion holds, that some unions pass 244 there.

**Effects:** at 128 per layer the layer LRU has 370,595 prefill step evictions and none in decode, and held-out prefill set recall is at most 2.9% for every policy in the table, the oracle included ([expert-policy.md](expert-policy.md), workload sections). Prefill unions are not this result's subject: the predictors predict decoded steps only.

### Belady gap

**A slice of the oracle is not a bound.** MIN minimizes the bytes of the whole replay. What one schedule loads in a phase or a test suffix bounds nothing, since a schedule can load more elsewhere to load less there. With room for one expert and A, B, A | B, whole-trace MIN misses B in the suffix, while a schedule with the same 3 misses overall misses nothing there (`TestHeldOutOracleWholeTraceSliceIsNoBound`). The review of this result raised the suffix; the phase is the same fault. So the tables in this section that compare with "Oracle, shared" compare with one hindsight schedule that pays for everything, and the bound on held-out decode is in the last table.

Decode on each workload's held-out suffix at 128 experts per layer, against that schedule. Best causal is the lowest MB/token among the causal rows of that table (ties on printed MB broken by hit rate). The ratio is derived from the printed, rounded MB/token.

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

Source: "Decode, MB/token (hit rate)" of each workload section. The oracle's bound is on the bytes of the whole replay, both phases: no cache of the budget, prefetching or not, loads fewer ([expert-policy.md](expert-policy.md), head). Its decode column is one schedule's share of those.

**Prefetch cannot close this gap in bytes over a whole replay.** It loads no fewer bytes than the misses it replaces, and can only hide their time ("Stage B: the decayed-count Markov predictor"). In the notes, 84.3–96.6% of Zhang's gap comes from knowing which resident expert is used furthest ahead, and a causal next-use predictor used as the eviction rule recovers −11.4% of it. He scopes that to residency and next-use prediction, not next-expert prefetch. Stage B is next-expert prefetch, so the two results are complementary rather than one confirming the other. **This repo has no causal next-use eviction predictor, so it has no result on victim prediction.**

**Held out, at every budget, against the bound.** All test suffixes together, decode, MB/token (hit rate) ([expert-policy.md](expert-policy.md), "The Belady gap on held-out decode, at every budget", which also has each workload and the per-layer oracles).

- **Bound:** MIN with one objective, the decoded misses after the cut. The training prefix and every prefill cost nothing, so they only fill the cache, and an expert whose next use is free can be dropped for nothing. With one expert size, which both models have, no demand cache of the budget misses less in that decode, by `TestHeldOutOracleOptimal` against exhaustive search on small random traces: a test, not a proof. The replay also checks it against every cache policy and the whole-trace oracle, at every size and workload. A cache that prefetches in a prefill is not bound by it, since that costs decode nothing.
- **Pinned:** the bound's objective, holding a step's experts together. Where every step fits it is an event-atomic schedule, so it bounds that optimum from above. It never loads more than the pinned whole-trace oracle's slice at any size (replay's PASS lines).
- **Slice:** the whole-trace shared oracle's bytes in this decode, as in the tables above. It pays for its prefix and prefills, so Best / slice is the gap to a schedule that pays for everything, though not necessarily the one with the least decode.

Each oracle is built per workload, so this sums per-workload oracles and is not one oracle over joined traces. Best causal is one policy over all workloads, picked among unpinned and pinned ones alike, so LFU at 192 can evict within a step. The ratios are from unrounded bytes. At 384 all ratios are 1.00:

| Experts per layer | IQ3 best causal | Bound, shared † | Pinned † | Slice † | Best / bound | Best / pinned | Best / slice | Q8_0 best / bound | Best / pinned | Best / slice |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 8 | hotness, per layer: 317 (37.9%) | 206 (59.6%) | 207 (59.4%) | 206 (59.6%) | 1.54 | 1.53 | 1.54 | 1.54 | 1.53 | 1.54 |
| 16 | hotness, per layer: 234 (54.0%) | 143 (71.9%) | 144 (71.8%) | 143 (71.9%) | 1.63 | 1.63 | 1.63 | 1.63 | 1.63 | 1.63 |
| 32 | SLRU, shared: 161 (68.3%) | 89 (82.6%) | 89 (82.5%) | 89 (82.6%) | 1.82 | 1.81 | 1.81 | 1.82 | 1.81 | 1.81 |
| 48 | SLRU, shared: 121 (76.2%) | 61 (88.1%) | 61 (88.1%) | 61 (88.0%) | 2.01 | 2.00 | 1.99 | 2.00 | 2.00 | 1.98 |
| 64 | SLRU, shared: 93 (81.7%) | 43 (91.5%) | 43 (91.5%) | 44 (91.4%) | 2.16 | 2.15 | 2.12 | 2.16 | 2.15 | 2.12 |
| 96 | SLRU, shared: 57 (88.8%) | 23 (95.5%) | 23 (95.5%) | 25 (95.2%) | 2.50 | 2.49 | 2.34 | 2.50 | 2.49 | 2.34 |
| 128 | SLRU, shared: 35 (93.1%) | 12 (97.6%) | 12 (97.6%) | 15 (97.1%) | 2.85 | 2.83 | 2.35 | 2.85 | 2.83 | 2.35 |
| 192 | LFU: 13 (97.4%) | 1 (99.7%) | 1 (99.7%) | 5 (99.0%) | 10.00 | 9.75 | 2.59 | 9.93 | 9.71 | 2.64 |
| 256 | SLRU, shared: 4 (99.3%) | 0 (100.0%) | 0 (100.0%) | 1 (99.8%) | 20.48 | 20.48 | 3.06 | 20.32 | 20.32 | 3.02 |

**What the correction moves.** Up to 32 per layer the slice is within 0.3% of the bound, pooled. Then the two part: 7% at 96, 21% at 128, 276 to 286% at 192 and about 570% at 256. The likely reason, which was not isolated: a cache this large holds nearly all of a decode's experts, so what remains is mostly what free prefills and a free state at the cut can supply. The ratios at 192 and 256 also rest on few bytes: 13 and 4 MB/token against 510 uncached, IQ3. **They do not say that an attainable schedule is 10 to 20 times better.** No cheapest schedule that pays for everything was computed, so the true gap lies between Best / slice and Best / bound. Since the bound is never above the slice, the earlier ratios understated the gap to the optimum on this decode, wherever they were read as that, and did not overstate it.

**From unrounded bytes, per workload at 128:** best causal against the bound 2.74 (coding), 2.94 (research), 3.09 (hr) and 2.71 (medtech) on IQ3, and 2.75, 2.95, 3.09 and 2.71 on Q8_0. Against the pinned one 2.72, 2.91, 3.07, 2.70 and 2.72, 2.92, 3.07, 2.70. Against the slice 2.26, 2.20, 2.45, 2.54 and 2.25, 2.21, 2.43, 2.54. The per-workload table above derives its 2.1 to 2.6 from rounded MB/token.

The notes do not define the denominator of Zhang's "44.2–45.9% gap", so it is not compared with these ratios.

## The runtime prefetch

Stage D ported the Markov predictor's same-token horizon to C++ as `--expert-prefetch`, because the user chose on 2026-10-14 to build the prototype although no predictor had passed. Its benchmark is `testdata/bench/expert-cache.json`, with 3 repetitions per arm and ceiling ([expert-streaming.md](expert-streaming.md), "Method", "Results" and "Gates").

| Comparison, `--expert-prefetch 12` against the demand cache | Decode t/s | p95 ms | Verdict |
|---|--:|--:|---|
| IQ3, 16 GiB | 18.29 (17.77–18.91) against 17.13 (16.92–19.05) | 77.4 (77.0–80.7) against 81.9 (81.6–85.3) | passed by the rule, on p95 alone |
| IQ3, 0 GiB (INFO) | 18.43 (18.22–18.52) against 18.51 (18.25–18.77) | 78.9 (75.5–83.6) against 78.9 (77.8–80.7) | not passed |
| Q8_0, 0 GiB (INFO) | 12.73 (12.02–12.81) against 13.00 (12.59–13.09) | 116.1 (115.2–118.1) against 111.9 (110.3–117.3) | not passed |

At 12 per layer it issues 4,252 prefetches per IQ3 run. 1,541 are used, 2,027 are evicted unused, and it reads 6.8 GB more. Decode misses hardly move (17,315 against 17,424 of 190,800 needed), nor does the wait on loads (9.5 against 9.2 s). A loaded-machine Q8_0 smoke run gave 7.67 against 11.25 tokens/s ("Checks", same document; INFO, not a benchmark).

So the runtime's one pass, about 5% at the p95 under one ceiling, comes with hardly fewer misses. It is not evidence that prediction beats the cache. The offline gate does not contradict it either, since the offline replay models neither ceilings nor tail latency. The cause of the p95 change is not established. The third repetition at 16 GiB ran four days after the demand cache's. Stage D left keeping the option to the user, and PLAN records no decision.

**Not recorded:** any prefetch arm on the rebased fork (the matrices of E0, E1 and E4 have none), and Q8_0 prefetch at 16 GiB, where 128 slots do not fit ([expert-diagnostics.md](expert-diagnostics.md), "Page-cache A/B").

## What this result does not cover

- **Hidden states.** Stage B used expert IDs only, since no router logits or hidden states are recorded. Stage E6, a hidden-state predictor, remains open. [PLAN.md](../PLAN.md), Stage E6, cites 90–96% one layer ahead on Mixtral and DeepSeek-MoE, 67% on Qwen2-MoE, and nothing for Kolibri's routing.
- **Other link speeds.** Only the three declared links ran. [ssd-curve.md](ssd-curve.md) measures 13–14 GB/s at saturation on this Mac, between ssd and pcie. Predictors fail on both sides of it, but that speed was not replayed.
- **Real compute time.** The replay computes 0.314 ms per layer, from resident speed. The cache's real decode takes 35.6 ms/token even with all 384 experts cached ([expert-diagnostics.md](expert-diagnostics.md)), so real lead times are longer. Stage E2 may change this. No replay uses those times.
- **Concurrent misses.** The time model serves one load at a time. The runtime has four I/O threads.
- **Other requests and models.** One deterministic trace set of four workloads, single stream. The notes' batch and multi-request results are not addressed.
- **Lossy miss handling.** This is lossless only; Stage E5 is separate.

## Not measured

Each of these is absent from the repo, and no estimate is given in its place. None is on the checklist's four items.

1. Recall restricted to experts that would otherwise have missed. This would directly measure the "already resident" explanation, which now rests on the used-prefetch counts.
2. A causal next-use predictor as the eviction rule.
3. Template diversity, which the notes list beside the four items PLAN names. The workloads have no within-class expert Jaccard or check for identical generated prefixes.
4. Replays at the measured 13–14 GB/s, or with the cache's real step time.
5. One oracle over the joined traces, as a pooled bound.
6. The cheapest decode of a schedule that also pays for its prefix and prefills, which would close the bracket between Best / slice and Best / bound.

Items 1, 3, 4, 5 and 6 can be computed from the selection files under `~/models/eval/locality/`, which `cmd/kolibri-policy` reads, without loading a weight. They need new replay code or a new report. Item 2 needs a new predictor.

## Reproduce

```sh
go run ./cmd/kolibri-policy -models Kolibri-1-IQ3_XXS-IQ4_XS-down-imx,Kolibri-1-Q8_0 -out ~/models/eval/locality-e7-heldout -md docs/expert-policy.md
```

`-out` keeps a rerun from overwriting `policy-<model>.json` under `~/models/eval/locality/`. The Stage E7 run took about 12 minutes on the M5 Pro, with 96 PASS lines and `CHECK` lines reporting 0 of 12 retained per model on both gates. The first Stage E7 run, with 88 PASS lines and no held-out oracles, changed `docs/expert-policy.md` by additions only, and rerunning it reproduced the report. The run with them changes only "The Belady gap on held-out decode, at every budget" and adds to two lines of the head, what a row of the oracle means. Its results are under `~/models/eval/locality-e7-heldout/`, beside the earlier `locality-e7/`. Selection files are verified with `tools/locality/experts.py --verify`. Recording them needs the weights; replaying them does not. Tests are in `internal/locality` and `cmd/kolibri-policy`, for example `TestReplayCausal`, `TestTuneNoLeak`, `TestMarkovNoLeak`, `TestLinearNoLeak`, `TestStageBGate`, `TestPolicyOracleOptimal`, `TestPinnedOracleOptimal`, `TestHeldOutOracleOptimal`, `TestStepUnionsMatchReplay` and `TestCheckPinnedRefs`.

**Controls.** Each makes its tests fail, and each was reverted:
- M92: the pinned oracles ignore the pins (`TestPinnedOracleOptimal`: "trial 0, 1 experts: 8 step evictions").
- M93: a union counts every selection instead of distinct experts (`TestStepUnions`: "prefill unions [4 4], want [3 2]").
- M94: a pinned reference swapped back to the unpinned LFU (`TestCheckPinnedRefs`: "reference \"LFU\" has 7 step evictions").
- M95: the pinned oracle keeps its use-once of an expert with no later use (`TestPinnedOracleOptimal`: "trial 0, 3 experts: the pinned oracle misses 9, the event-atomic optimum 12").
- M99: every access costs the held-out oracle something, which is whole-trace MIN (`TestHeldOutOracleWholeTraceSliceIsNoBound`: "the held-out oracle misses 1 in the suffix, want 0").
- M100: a test suffix's prefill costs it something (`TestHeldOutOracleOptimal`: "trial 3, 2 experts: the held-out oracle misses 2, the optimum 1").
- M101: an expert whose next use is free ranks by that use (`TestHeldOutOracleWholeTraceSliceIsNoBound`: "the held-out oracle misses 1 in the suffix, want 0").
- M102: the pinned held-out oracle does not pin a step when it starts (`TestPinnedHeldOutHoldsTheStep`: "evicted 1 of the running step to load 2").
- M103: the replay's check leaves the cache policies out (`TestCheckHeldOut`: "a cache below the shared bound passes").
