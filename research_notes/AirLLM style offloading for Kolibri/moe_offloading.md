# MoE expert offloading, caching and prefetching during inference (survey as of 2026-10-22)

Scope: MoE-specific offloading systems (expert caches, prefetch/prediction, mixed-precision or lossy miss handling, CPU-vs-accelerator placement) and open-source implementations. Dates are first arXiv submission (or PR/issue open date) unless noted. Numbers are as reported by the authors. Many 2026 community numbers (GitHub PRs and issues) are self-reported and unreviewed.

Reference point for relevance (from the assignment, not researched): Kolibri-1, 50 layers, 384 routed experts per layer, top-6 sigmoid routing, one shared expert, expert FFN width 512. The prototype is a per-layer LRU slot cache in Metal-visible host memory with pread (F_NOCACHE) misses. At 128/384 slots it gets about 91% decode hit rate and 17–18 tok/s, against 60 tok/s fully resident. Markov prefetch improved p95 by about 5%, and offline no predictor beat LRU-type policies.

## Q1. Survey and comparison of the main systems (mechanism, cache policy, prediction, hardware, speedups, models)

### Takeaway
Almost every system since late 2023 combines three things: (1) an expert cache in fast memory, usually LRU, LFU or a weighted hybrid; (2) a prefetcher, most often the next layer's gate applied to the current hidden state; (3) a miss path, which is either fetch, CPU compute, a low-precision copy, substitution or skip. Most were evaluated on Mixtral-8x7B (8 experts) or 60–64-expert models over PCIe 4.0 with consumer GPUs. Only a handful of 2025–2026 systems (MoE-CORE, kTransformers, SeqMoE, community llama.cpp/MLX work) touch 128–384-expert models. Headline speedups are against weak baselines (on-demand loading, HF `device_map`, DeepSpeed) and are not comparable across papers.

### Cited Findings

**Comparison table** (abbreviations: LoD = load on demand; NLG = next-layer gate applied to the current hidden state)

| System (date) | Mechanism | Cache policy | Prediction / prefetch | HW | Models | Reported result |
|---|---|---|---|---|---|---|
| Mixtral-offloading, Eliseev & Mazur (2023-12-28) | GPU expert cache, mixed quant (HQQ: attention 4-bit, experts 2–3-bit) | Per-layer LRU, k=2 (12 GB GPU) or k=4 (16 GB) | Speculative loading via NLG, tested 1/2/10 layers ahead | RTX 3060/3080M, T4, A100 | Mixtral-8x7B | 2.28 tok/s on RTX 3060 (2-bit experts), 2.09 tok/s on T4 Colab — [arXiv 2312.17238](https://arxiv.org/html/2312.17238) |
| Pre-gated MoE (2023-08-23) | Architecture change: block N's gate selects block N+1's experts (needs fine-tuning) | n/a (on-demand + pre-gate) | Exact by construction (it *is* the routing) | A100, PCIe4 | Switch-Base 8/64/128, Switch-Large 128 | ~1.5× throughput vs MoE-OnDemand, ~81% of GPU-only throughput, peak GPU mem ~23% of GPU-only — [arXiv 2308.12066](https://arxiv.org/html/2308.12066) |
| EdgeMoE (2023-08-28) | Experts on storage, per-expert bitwidth (INT2/4/8) | Score = activation frequency / layer distance | Lookup table conditioned on the previous **two** layers' activations; preload 1–3 experts | Jetson TX2, RPi 4B, Snapdragon 8 Gen3 | Switch Transformers, GPTSAN | 2.64–3.03× vs IO-EXP on TX2; INT4 for all experts costs 1.30–1.44% accuracy — [arXiv 2308.14352](https://arxiv.org/html/2308.14352) |
| SwapMoE (2023-08-29) | Small dynamic set of "virtual experts" in memory; routing remapped onto the resident set (lossy) | Importance-based resident set | n/a | edge | Switch Transformer | 14.2 → 4.7 GiB, 50% latency reduction, Rouge-2 −0.041 — [arXiv 2308.15030](https://arxiv.org/abs/2308.15030) |
| MoE-Infinity (2024-01-25) | Request-level Expert Activation Matrix (EAM) tracing | Priority from predicted activation × layer decay (not LRU) | Match the current request's EAM to historical EAMs (cosine), prefetch | RTX A5000, PCIe4 | DeepSeek-V2-Lite, Switch-128, NLLB-128, Arctic-128, Mixtral | 3.1–16.7× per-token latency vs vLLM/Ollama/DeepSpeed/BrainStorm (the conclusion says 2.7–13.7×); built around batch size 1 — [arXiv 2401.14361](https://arxiv.org/html/2401.14361v3) |
| Fiddler (2024-02-10) | CPU computes non-resident experts instead of transferring weights | Static placement of popular experts | n/a | Quadro RTX 6000 / RTX 6000 Ada | Mixtral-8x7B fp16, Phi-3.5-MoE | 1.26× single-batch vs llama.cpp, 11.57× beam search — [arXiv 2402.07033](https://arxiv.org/abs/2402.07033) |
| AdapMoE (2024-08-19) | Sensitivity-based adaptive gating (top-2 → top-1), prefetch, DP cache allocation across layers | DP/knapsack per-layer cache sizes; early layers get more | NLG (≈90% accuracy for most layers); small trained gate for layer 1 | RTX 4090, A6000 | Mixtral-8x7B/8x22B | 1.35× avg vs (their improved) Mixtral-offloading; 25% fewer activated experts — [arXiv 2408.10284](https://arxiv.org/html/2408.10284) |
| ExpertFlow (2024-10-23; v2 rewritten) | Transformer routing-path predictor for all layers in one pass, token scheduling, predictive cache | Predictive cache + runtime correction | Learned predictor | single GPU | various | up to 93.72% less GPU mem, up to 10× throughput — [arXiv 2410.17954](https://arxiv.org/abs/2410.17954) |
| ProMoE (2024-10-29) | Proactive caching on consumer GPU | Per-layer LRU (prefetches update LRU order) | Learned 2-layer MLP per layer (~2M params, runs on CPU, ~200 µs) | RTX 4090, PCIe4 | DeepSeek-MoE-16B, DeepSeek-V2-Lite, Qwen1.5/2-MoE, Mixtral | 2.20× prefill and 2.07× decode on average vs existing offloading — [arXiv 2410.22134](https://arxiv.org/html/2410.22134) |
| HOBBIT (2024-11-03) | Mixed precision: on a miss, load a low-bit copy for less important experts, skip the least important (built on llama.cpp) | Multidimensional: LRU+LFU+LHU+FLD weighted | Stacked gating, predicting 1–3 layers ahead | RTX 4090, Jetson AGX Orin (32 GB unified) | Mixtral-8x7B, Phi-MoE | up to 9.93× decode vs MoE-Infinity; 1.31–1.42× vs llama.cpp on 4090+CPU — [arXiv 2411.01433](https://arxiv.org/html/2411.01433) |
| MoE-Lightning (2024-11-18) | Throughput-oriented CPU-GPU-I/O pipeline (CGOPipe) + hierarchical roofline model | Paged weights | n/a (batch) | T4 | Mixtral 8x7B/8x22B, DBRX | up to 10.3× throughput vs offloading systems — [arXiv 2411.11217](https://arxiv.org/abs/2411.11217) |
| DAOP (2024-12-16) | Per-sequence CPU/GPU expert allocation; pre-computes predicted experts on CPU; "graceful degradation" | Data-aware | Predictive pre-calculation | on-device | Mixtral/Phi | up to 8.20× vs caching+prefetch, 1.35× vs offloading — [arXiv 2501.10375](https://arxiv.org/abs/2501.10375) |
| FineMoE / fMoE (2025-02-07, EuroSys'26) | Fine-grained "expert maps" from iteration-level trajectories + prompt semantic embeddings | LFU-like guided by maps | Trajectory similarity search + semantic hints | 6× RTX 3090, A100 | Mixtral-8x7B, Qwen1.5-MoE, Phi-3.5-MoE | −47% latency, +39% hit rate vs SOTA — [arXiv 2502.05370](https://arxiv.org/abs/2502.05370), [EuroSys PDF](https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf) |
| Klotski (2025-02-09) | Expert-aware multi-batch pipeline (CPU+disk) | — | Correlation-aware expert prefetcher | — | — | up to 85.12× throughput — [arXiv 2502.06888](https://arxiv.org/abs/2502.06888) |
| Fate (2025-02-17) | Cross-layer gate (adjacent-layer gate inputs) + quantization | "Shallow-favoring" caching, 99% hit rate claimed | Cross-layer gate | edge | — | up to 4.1× decode vs LoD — [arXiv 2502.12224](https://arxiv.org/abs/2502.12224) |
| MoE-Gen (2025-03-12) | Module-based batching for offline throughput on a single GPU | — | — | single GPU | DeepSeek, Mixtral | 8–31× throughput vs FlexGen/MoE-Lightning/DeepSpeed — [arXiv 2503.09716](https://arxiv.org/abs/2503.09716) |
| HybriMoE (2025-04-08) | Dynamic intra-layer CPU/GPU scheduling on top of kTransformers | Score-based caching | "Impact-driven" inter-layer prefetch | — | 3 MoE LLMs | 1.33× prefill, 1.70× decode vs kTransformers — [arXiv 2504.05897](https://arxiv.org/abs/2504.05897) |
| FloE (2025-05-09) | Compresses expert internals (lossy) + sparse prediction | — | low-cost sparse predictor | RTX 3090 | Mixtral-8x7B | 9.3× per-expert compression, 4.4–7.6% avg degradation — [arXiv 2505.05950](https://arxiv.org/abs/2505.05950) |
| kTransformers (SOSP'25, Oct 2025) | Experts *computed on CPU* (AMX/AVX-512 kernels), attention and shared on GPU, CUDA graphs, "Expert Deferral" | Static (no expert movement) | n/a | 2× Xeon 8452Y + A100/RTX 4080, PCIe4 | DeepSeek-V3 (256 exp, top-8), DeepSeek-V2.5 (160, top-6), Qwen2-57B | 4.62–19.74× prefill, 1.25–4.09× decode vs Fiddler/llama.cpp — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) |
| SP-MoE (2025-10-11) | Speculative-decoding-aware prefetch (draft model predicts target experts) | — | Draft/target correspondence | — | — | 1.07–3.5× TPOT — [arXiv 2510.10302](https://arxiv.org/abs/2510.10302) |
| BuddyMoE (2025-11-13) | Substitute a co-activated resident "buddy" on a prefetch miss (in llama.cpp CUDA) | — | — | A100 PCIe | DeepSeek-V2-Lite (64, top-6) | up to ~10% throughput at a 5.4% accuracy drop — [arXiv 2511.10054](https://arxiv.org/html/2511.10054) |
| OD-MoE (2025-12-03) | Cacheless, distributed over 10 edge nodes, just-in-time loading | none | "Emulative" predictor, multiple layers ahead, 99.94% | 10-node testbed | — | ~75% of fully cached decode speed at 1/3 GPU memory — [arXiv 2512.03927](https://arxiv.org/abs/2512.03927) |
| SpecMD (2026-02-03) | Benchmark framework for cache policies + new "Least-Stale" eviction | Least-Stale (stale vs current-pass queues) | NLG top-k / score-based | A100 with capped cache | OLMoE (64), Mixtral, Qwen1.5-MoE, Phi-3.5-MoE | Collision misses up to 85× fewer than LRU (other sections say 2.6–8.6× or 3–9×); >88% hit at 5% capacity on OLMoE; −34.7% TTFT — [arXiv 2602.03921](https://arxiv.org/html/2602.03921) |
| DALI (2026-02-03) | Greedy 0-1 CPU/GPU assignment, residual-based prefetching, workload-aware replacement (kTransformers-based) | Workload-aware | Residual-based | local PC | various | "significant" speedups (no number in abstract) — [arXiv 2602.03495](https://arxiv.org/abs/2602.03495) |
| Speculating Experts (2026-03-09) | Execute speculated experts directly from internal representations | — | From current representations | — | multiple | up to 14% TPOT reduction vs on-demand CPU loading — [arXiv 2603.19289](https://arxiv.org/abs/2603.19289) |
| WiSP (2026-06-20) | Working-set paging, byte-identical, KV and experts share VRAM; MV-WSA allocator | Allocation by marginal latency per byte | **Found prefetching useless in single-stream decode** | RTX 3090 24 GB | — | up to 2.0× decode vs static offload; 1.19× from live re-allocation — [arXiv 2606.21868](https://arxiv.org/abs/2606.21868) |
| SpecPrefetch (2026-06-24) | Shared light adapter predicts next-layer candidates for transfer only (router unchanged) | window-aware scheduler | Learned adapter | Snapdragon 8 Elite | Qwen3-VL-30B-A3B, DeepSeek-VL2-Tiny | best recall in 9/10 settings; up to +20% decode throughput — [arXiv 2607.24787](https://arxiv.org/abs/2607.24787) |
| SeqMoE (2026-09-11) | Seq2seq multi-step, multi-layer prediction; JSD prefetch scheduling; probabilistic Belady eviction; CUDA-graph-compatible runtime | Probabilistic Belady | Sequence model | — | — | 96.97% avg hit and 80.22% of full-load perf at 45% residency — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978) |
| Edge0 / "Other Half of the Memory Wall" (2026-09-16) | SSD streaming with a trained "prerouter" whose prediction *replaces* the routing; recovery LoRA | — | Trained per-layer head, one token ahead | single 24 GB machine | 35B MoE int4 | 20 tok/s at 3 GiB peak active memory, "within a few points" of the fp16 teacher — [arXiv 2609.18063](https://arxiv.org/abs/2609.18063) |
| MoE-CORE (2026-10-01) | Layer-wise slot reallocation, domain hot-expert init, RHAC replacement, NLG prefetch of 1 expert, optional substitution (AES) | RHAC: recency + frequency + moving average + router score + age | NLG (next layer's router on current hidden state) | Ascend 950PR NPU | DeepSeek-V4-Flash W4A8 (256 exp), GLM-5.2 W4A8C8 | 1.39–1.75× TPOT vs LoD on DeepSeek — [arXiv 2610.01950](https://arxiv.org/html/2610.01950) |

Additional individual findings:
- MoE-Infinity reports that for models with about 100 experts, fewer than 5% of experts are repeatedly activated within a request (about 25% for Mixtral). Expert reuse is skewed within one request but close to uniform across many requests, which is why tracing is per request — [arXiv 2401.14361](https://arxiv.org/html/2401.14361v3)
- MoE-CORE (DeepSeek-V4-Flash, 256 experts per layer) reports mean exact hit rate rising from 47.47% at 12 experts per layer to 96.43% at 120 experts per layer. Removing its RHAC policy raises TPOT from 36.0 to 46.8 ms; removing domain profiles or layer-wise allocation costs only 1.4% and 1.1% — [arXiv 2610.01950](https://arxiv.org/html/2610.01950)
- MoE-CORE only benchmarks against LoD and vLLM Prefetch. MoE-Infinity, FineMoE, Fate, AdapMoE and BuddyMoE are discussed but not run — [arXiv 2610.01950](https://arxiv.org/html/2610.01950)
- The SOSP'25 kTransformers paper's DeepSeek-V3 Fiddler-style baseline ran at only 70.02 tok/s prefill and 4.68 tok/s decode with GPU utilization below 30%, i.e. CPU-bound. Expert Deferral keeps the accuracy drop within 0.5% and adds up to 1.45× throughput — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- An energy study found SSD offloading of MoE weights costs 3.8×–12.5× more per-token energy for Mixtral than the baseline (datacenter-scale study) — [arXiv 2508.06978](https://arxiv.org/pdf/2508.06978) (figure from search summary, not re-read)

### Inferences
- The Kolibri prototype's core mechanism has been built many times, in research and in llama.cpp/MLX forks: a per-layer LRU slot cache with demand misses served by async pread from the model file. Mixtral-offloading, ProMoE and the llama.cpp PR #25294 all use LRU or LRU-like caches (see Q6). It is not novel as a mechanism.
- The Kolibri regime is underrepresented in the literature: 384 very small experts (width 512), top-6 sigmoid routing, decode at batch 1, unified memory, SSD rather than PCIe as the slow tier. The closest published regimes are MoE-CORE (256 experts, NPU), kTransformers (256 experts, CPU compute) and community Apple Silicon work on Qwen3.x 256-expert models.
- A "decayed-count Markov two layers ahead" prefetcher resembles EdgeMoE's lookup table conditioned on the previous two layers' activations, and ProMoE's "token-based" predictor. Neither is new.

### Gaps
- I could not read the figures with actual hit-rate-by-capacity curves for Mixtral-offloading (Fig. 2) or FineMoE's LRU-vs-LFU ablation (Fig. 14b). Only qualitative statements are available in text.
- I did not verify ExpertFlow v2, Klotski, DAOP or MoE-Lightning beyond their abstracts. The hardware and models for some rows are not specified here for that reason.
- I found no published system evaluated on a 384-expert model (e.g. Kimi-K2-class) with per-layer hit-rate data.

## Q2. Which predictors work, with reported accuracy and lead time

### Takeaway
The cheapest predictor, the next layer's gate applied to the current hidden state, works well when adjacent-layer hidden states are similar. Reported accuracy is about 90–96% for one layer ahead on Mixtral and DeepSeek-MoE-16B, but it is model-dependent: 66.9% on Qwen2-MoE. Accuracy falls as lead time grows. Learned predictors (ProMoE MLP, MoE-Beyond, SeqMoE, SpecPrefetch) gain accuracy at training cost. Token-history predictors (Markov or "token-based") are clearly weaker, about 58% in ProMoE. No paper reports results for a 384-expert, top-6 sigmoid router.

### Cited Findings
- **NLG (next-layer gate on the current hidden state):**
  - Mixtral-offloading introduced it, tested 1, 2 and 10 layers ahead; the numeric recall is only in a figure — [arXiv 2312.17238](https://arxiv.org/html/2312.17238)
  - HOBBIT reports top-1 next-layer accuracy of 96% on average and about 90% for 2–3 layers ahead on Mixtral-8x7B. It recommends 1–3 layers ahead. Prefetching cut prefill latency by about 10% but gave "only modest decoding gains" — [arXiv 2411.01433](https://arxiv.org/html/2411.01433)
  - AdapMoE reports about 90% accuracy for most layers, based on high cosine similarity between adjacent layers' MoE inputs — [arXiv 2408.10284](https://arxiv.org/html/2408.10284)
- **ProMoE's head-to-head comparison** (DeepSeek-MoE-16B with 64 experts top-6, and Qwen2-MoE with 64 experts top-8):
  - Learned 2-layer MLP: 84.7% average.
  - Skip-based (NLG): high on DS-1 but drops in head and tail layers; 66.9% on QW-2, because Qwen2's gate is sensitive to small input changes.
  - Token-based (iteration-wise): 58.3%.
  - Adjacent-layer input cosine similarity is about 90%. Extending the stride by one more layer costs about 5% accuracy.
  - [arXiv 2410.22134](https://arxiv.org/html/2410.22134)
- **Fate (cross-layer gate):** adjacent-layer gate inputs give "high prediction accuracy without additional GPU overhead" — [arXiv 2502.12224](https://arxiv.org/abs/2502.12224). A vLLM RFC cites Fate at 83% cosine similarity and 97–99% prediction accuracy — [vLLM #38256](https://github.com/vllm-project/vllm/issues/38256)
- **MoE-Beyond** (learned transformer predictor on 66M traces, DeepSeek-V2-Lite): 97.5% accuracy and 86.6% F1. In simulation, GPU cache hit rate rises from 17% to 72% when 10% of experts fit — [arXiv 2508.17137](https://arxiv.org/abs/2508.17137)
- **OD-MoE** "emulative" predictor: 99.94% activation prediction accuracy, multiple layers ahead — [arXiv 2512.03927](https://arxiv.org/abs/2512.03927)
- **SeqMoE** (seq2seq multi-step forecasts): 84.56–96.97% average hit at 15–45% cache, against 60.02–88.50% for LRU — [Pith summary](https://pith.science/paper/2609.12978); [arXiv 2609.12978](https://arxiv.org/abs/2609.12978). A reviewer notes the predictor was tested only on traces from the same five datasets, with no zero-shot test on new model families — [Pith](https://pith.science/paper/2609.12978)
- **SpecPrefetch:** a shared adapter predicts next-layer candidates for transfer only. It gets the best average recall in 9 of 10 model-benchmark settings on Qwen3-VL-30B-A3B (128 experts) and DeepSeek-VL2-Tiny — [arXiv 2607.24787](https://arxiv.org/abs/2607.24787)
- **Prediction-as-routing:**
  - Pre-gated MoE is exact by construction but requires fine-tuning — [arXiv 2308.12066](https://arxiv.org/html/2308.12066)
  - Edge0 trains a per-layer "prerouter" one token ahead, uses its output as the actual routing, and adds a recovery LoRA to recover quality — [arXiv 2609.18063](https://arxiv.org/abs/2609.18063)
  - Speculating Experts executes the speculated experts directly: up to 14% TPOT reduction, with accuracy "generally" preserved — [arXiv 2603.19289](https://arxiv.org/abs/2603.19289)
- **Lead time:**
  - FineMoE: hit rates drop as prefetch distance grows, and models with larger selection spaces (Qwen1.5-MoE, 60 experts × 24 layers) have higher routing entropy and are less predictable than Mixtral — [arXiv 2502.05370](https://arxiv.org/html/2502.05370v2)
  - MoE-CORE: hit rate is 89.05% without prefetch and 92.43%, 93.81% and 94.05% at depths 1, 2 and 3; depth 1 is the default — [arXiv 2610.01950](https://arxiv.org/html/2610.01950)
- **Cross-layer selection overlap on a fine-grained model (MLX, Apple Silicon):** on Qwen3.6-35B-A3B (256 experts, top-8), per-token adjacent-layer Jaccard of selected experts was 0.017, against a uniform baseline of 0.016. The author concluded there is "no signal to prefetch layer L+1 from layer L" from selections alone — [mlx-lm #1438 (mabaeyens, 2026-07-19)](https://github.com/ml-explore/mlx-lm/issues/1438)

### Inferences
- Kolibri's Markov predictor uses token history and expert-ID co-occurrence. That is the class the literature finds weakest (58.3% in ProMoE), and the mlx-lm measurement suggests that on fine-grained, load-balanced routers expert-ID cross-layer correlation is close to chance. Kolibri's offline null result for Markov and linear predictors is consistent with this.
- The one predictor class Kolibri apparently has not tried is the hidden-state-based NLG: apply layer L+1's (and L+2's) sigmoid router to layer L's residual stream. Its accuracy depends on residual-stream similarity, not expert-ID correlation, so a near-zero Jaccard of selections does not rule it out. It is cheap: one extra 384×d GEMV per layer. Worth measuring recall@6 and recall@12 offline before building anything.
- Even good predictors add only a few points of hit rate at moderate residency (MoE-CORE: +3.4 pp at depth 1 from an 89% base). Kolibri is already at about 91%, so the ceiling for prefetch is small.

### Gaps
- No paper reports NLG accuracy for sigmoid (non-softmax) top-k routers with a shared expert, such as DeepSeek-V3-style or Kolibri routing. It is not known whether sigmoid scores are more or less stable under the hidden-state shift between layers.
- Lead-time accuracy curves beyond 2–3 layers are sparse. Only HOBBIT (about 90% at 2–3 layers on Mixtral) and ProMoE (−5% per extra stride) give numbers.

## Q3. Does prediction add little over plain LRU/LFU at small batch on fine-grained MoE? Agreement and disagreement with Kolibri's offline result

### Takeaway
The 2026 literature that measures carefully at batch 1 on fine-grained models largely agrees with Kolibri's result:
- **WiSP:** prefetching gives no benefit in single-stream decode.
- **Reproducibility study (Zhang):** lightweight causal predictors recover none of the offline-optimal gap, and LRU or LFRU picks the optimal victim far more often than a learned next-use predictor.
- **Community MLX work on Qwen3.6 (256 experts):** plain LRU is close to the best shippable policy, frozen hot sets hurt, and cross-layer prefetch has no signal.
- **Practitioner reports:** prefetch is net-negative at low concurrency, and "strong temporal locality, no static skew" favours LRU.

Papers claiming large wins over LRU (FineMoE, SpecMD, SeqMoE, MoE-Beyond) mostly operate in different regimes: tiny caches of 1–15%, multi-request serving, prefill-heavy or TTFT metrics, coarse experts, or trace simulators with replay semantics that penalise recency policies.

### Cited Findings
**Agreeing (prediction or smart policies add little at batch 1):**
- **WiSP** (RTX 3090 24 GB): "predicting and prefetching future experts did not help in single-stream decode." The bottleneck is PCIe bandwidth, so speculative transfers compete with demand transfers. The authors reframe the problem as allocation (experts vs KV) — [arXiv 2606.21868](https://arxiv.org/abs/2606.21868)
- **Reproducible Evaluation of MoE Expert Caching** (Yu Zhang, 2026-08-04, v4 2026-08-25, single author):
  - Setup: trace-driven, event-atomic simulator over Granite-3.1-3B-A800M (40 experts), OLMoE (64) and Qwen3-30B-A3B 4-bit (128). The Qwen3 traces were collected with MLX on an M4 Pro Mac mini.
  - After the corrections, a stable 44.2–45.9% gap to the offline optimum remains. 84.3–96.6% of that gap comes from knowing which resident expert is used furthest in the future.
  - A causal next-use predictor used as the eviction rule recovers −11.4% of the gap. It picks an optimal victim 3.4% of the time, against 2.4% for a random resident and 20.6–22.1% for LRU and LFRU.
  - The authors scope this: it concerns residency and next-use prediction, not next-expert prefetch, and lossless service only.
  - [arXiv 2608.07911](https://arxiv.org/pdf/2608.07911)
- **Same paper, simulator pitfalls:** "sequential" per-access replay of a fused event inflates LRU by 27.4%, LFRU by 28.5% and Least-Stale by 27.4%, but LFU by only 3.4% and Belady by 4.3%. That inverts the policy ranking and makes static or frequency policies look better than recency. Separately, single-template probe sets create byte-identical generated prefixes that masquerade as semantic locality: within-class expert Jaccard falls from 0.229 to 0.092 once such pairs are excluded, against a 0.078 cross-category baseline — [arXiv 2608.07911](https://arxiv.org/pdf/2608.07911)
- **mlx-lm #1438** (Qwen3.6-35B-A3B 4-bit, 256 experts top-8, per-layer LRU at 30% capacity, about 4,400 decode tokens per topic):
  - LRU decode hit rate 0.831 (code), 0.871 (prose), 0.795 (agentic), against Belady ceilings of 0.917, 0.940 and 0.902.
  - Pinning a profiled hot set is −6 to −30 points cross-topic, and even net negative within a session (code −6.6, agentic −9.5, prose +1.0).
  - Online LFU gave +0.27 points.
  - The author's later summary lists "reactive-only fetch (three independent prefetch nulls)" and "plain LRU over any frozen resident set" as findings.
  - [mlx-lm #1438](https://github.com/ml-explore/mlx-lm/issues/1438)
- **Same thread, measured sweep** (Qwen3.6-35B-A3B 8-bit, M-series Mac): decode hit rates of 0.836 / 0.881 / 0.896 / 0.910 at resident fractions 0.3 / 0.4 / 0.45 / 0.5, giving 8.88 / 9.79 / 10.17 / 10.58 tok/s. Marginal cost per miss is about 240–280 µs, and it rises with residency as the page cache shrinks — [mlx-lm #1438](https://github.com/ml-explore/mlx-lm/issues/1438)
- **llama.cpp PR #27861** (GPU-resident LRU expert cache): the author found "no exploitable static skew" but "strong temporal locality," hence LRU over static pinning — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861)
- **vLLM RFC #38256 community reports** (self-reported):
  - On Apple Silicon, an LFU variant was 1.4–2.3× worse than LRU, and a bigger cache barely changed the hit rate.
  - In a compute-offload setup, prefetch was net-negative at low concurrency.
  - [vLLM #38256](https://github.com/vllm-project/vllm/issues/38256)
- **HOBBIT:** prefetching gave only modest decoding gains; its multidimensional cache reduces miss penalty by just 4.69–8.68% vs LRU and 2.13–4.19% vs LFU — [arXiv 2411.01433](https://arxiv.org/html/2411.01433)
- **MoE-CORE:** depth-1 prefetch adds +3.4 pp hit rate on top of 89.05% — [arXiv 2610.01950](https://arxiv.org/html/2610.01950)
- **"Cacheable by Design?"** (Qwen3-30B traces): adjacent-token expert reuse is 2.0× chance; 95% of traffic uses 52.5% of experts; an LRU cache holding 13.4% of experts serves 66% of requests. "Domain-primed prefetching does not help." — [arXiv 2608.18261](https://arxiv.org/abs/2608.18261)

**Disagreeing (prediction or smarter policy clearly beats LRU), with regime:**
- **FineMoE:** "LRU performs poorly," LFU beats LRU, and FineMoE beats both; +39% hit rate vs SOTA. Evaluated on Mixtral (8 experts), Qwen1.5-MoE (60) and Phi-3.5-MoE (16) with serving workloads and batch 1–8. Gains are smaller on an A100 than on 3090s — [arXiv 2502.05370](https://arxiv.org/html/2502.05370v2)
- **SpecMD:** "MoE expert access is not consistent with temporal locality assumptions (e.g. LRU, LFU)." Least-Stale cuts collision misses up to 85× vs LRU. Tested at 1–25% cache capacity, metric mostly TTFT, i.e. prefill-heavy — [arXiv 2602.03921](https://arxiv.org/html/2602.03921)
- **SeqMoE:** LRU 60.02–88.50% vs SeqMoE 84.56–96.97% at 15–45% residency — [Pith](https://pith.science/paper/2609.12978)
- **MoE-Beyond:** 17% → 72% hit at 10% capacity, simulated at batch 1 — [arXiv 2508.17137](https://arxiv.org/abs/2508.17137)
- **ProMoE:** 2.07× decode vs reactive LRU offloading on PCIe consumer GPUs — [arXiv 2410.22134](https://arxiv.org/html/2410.22134)
- **Practitioner counter-report:** a llama.cpp fork author says "simple LRU performs very poorly" and uses frequency-based admission and eviction instead — [PR #29887 comment (pashko-k)](https://github.com/ggml-org/llama.cpp/pull/29887)

### Inferences
- Kolibri's result matches the "careful batch-1 decode" cluster: WiSP, Zhang, mlx-lm and HOBBIT's modest decode gains. The reasons line up:
  - At about 33% residency with strong token-to-token reuse, LRU already captures most of what is capturable, about 91% hit against roughly 7–11 points of Belady headroom in the comparable mlx-lm case.
  - Misses are bandwidth- or latency-bound at the slow tier, so speculative reads compete with demand reads.
  - Fine-grained load-balanced routers decorrelate expert IDs across layers.
- The disagreeing papers win in regimes where LRU genuinely thrashes:
  - Caches of 1–15% of experts, where the per-step union exceeds the per-layer capacity. Zhang argues the right regime variable is the per-step expert union relative to per-layer capacity, not batch size or normalized cache fraction.
  - Prefill and multi-request serving, where scans pollute LRU.
  - Coarse experts (Mixtral with k=2 slots).
  - Simulators that flatten fused events, which inflate recency-policy misses by about 27%.
- Kolibri should report its offline results with Zhang's checklist: event-atomic replay, decode-only vs lifetime hit rate, per-step union/capacity ratio and template diversity. This makes the negative result defensible.
- The real remaining headroom is Belady-style victim choice, which is hard to approximate causally per Zhang, and lossy miss handling (Q4). More predictor engineering is not where the gain is.

### Gaps
- No study I found uses 384 experts per layer with top-6; the closest are 256-expert models. Whether the per-step union/capacity ratio at Kolibri's 128 slots behaves like the 256-expert Qwen3.6 case is unverified.
- Zhang's paper is single-author with an unusual affiliation (China National Chemical Equipment Co. Ltd.) and is unreviewed. The mlx-lm and vLLM numbers are GitHub comments. They agree with each other but are not peer-reviewed.

## Q4. Mixed-precision or dynamic quantization of cold experts, expert skipping or substitution on miss, and their quality costs

### Takeaway
Lossy miss handling is where the published gains at batch 1 are largest. Options, roughly in order of quality cost:
- **Low-bit fallback copy** (HOBBIT): ≤1% task-accuracy loss on Mixtral and Phi-MoE.
- **Substituting a resident expert for low-score misses** (MoE-CORE AES): raises availability from 92.4% to 99.97%. Quality is not quantified in the parts read.
- **Buddy substitution** (BuddyMoE): about 5% accuracy drop for 7–10% throughput.
- **Adaptive top-k reduction** (AdapMoE, HOBBIT skip): about 25% fewer experts, claimed lossless on Mixtral.
- **Cache-aware rerouting or trained locality:** about 80% fewer misses at up to 3.4% perplexity.

Many-small-expert models tolerate dropping better than few-large-expert models (SpecMD), which is favourable for Kolibri's top-6 of 384.

### Cited Findings
- **HOBBIT:**
  - Policy: on a miss, an unimportance score from normalized gating weights decides between loading high precision (score ≤ T1), loading a low-precision copy (T1–T2) or skipping (> T2). The top-1 expert is always high precision.
  - For Mixtral, T1=0.6 and T2=0.9 gives about 67% high, 30% low and 3% skipped. Bit pairs are fp16+int4 on RTX 4090 and int8+int2 on Jetson Orin.
  - Gating weight correlates 0.99 with expert output magnitude.
  - Accuracy drop is "no more than 1%" (GSM8K 0.52 → 0.51 on Mixtral). Skipping 10% of experts raises perplexity by more than 1%.
  - [arXiv 2411.01433](https://arxiv.org/html/2411.01433)
- **EdgeMoE:** per-expert bitwidth adaptation (INT2/4/8/FP32) ranked by INT2 sensitivity. INT4 for all experts costs 1.30–1.44% accuracy; INT2 on SAMsum costs 5.82% — [arXiv 2308.14352](https://arxiv.org/html/2308.14352)
- **Mixtral-offloading:** experts at 2–3-bit HQQ and attention at 4-bit, as a static choice — [arXiv 2312.17238](https://arxiv.org/html/2312.17238)
- **AdapMoE:** drops from top-2 to top-1 when a Fisher-based sensitivity test passes. 25% fewer activated experts, gating alone 1.25× speedup; the authors claim no accuracy loss on MMLU and ARC-C (exact numbers not in text) — [arXiv 2408.10284](https://arxiv.org/html/2408.10284)
- **MoE-CORE AES:** replaces eligible low-score misses with resident unselected experts (α=0.1). Effective availability goes from 92.43% to 99.972%; the best configuration with AES and MTP is 21.5 ms TPOT at batch 2 — [arXiv 2610.01950](https://arxiv.org/html/2610.01950)
- **BuddyMoE** (DeepSeek-V2-Lite, 64 experts top-6, llama.cpp CUDA): substitutes co-activated buddies chosen offline from conditional co-activation frequency.
  - Cache rate 0.75: 34.23 → 36.75 tok/s, ARC accuracy 0.735 → 0.695.
  - Cache rate 0.375: about +10.3% throughput.
  - Random replacement collapses accuracy to 0.16–0.55.
  - [arXiv 2511.10054](https://arxiv.org/html/2511.10054)
- **SpecMD:** wide MoEs with many small experts (OLMoE) tolerate expert dropping and routing bias better, because more active experts dilute the impact of dropping one. Qwen at 4-bit suffers larger losses from dropping — [arXiv 2602.03921](https://arxiv.org/html/2602.03921)
- **"Cacheable by Design?"** (pre-registered):
  - Training routers for locality reduces misses by up to 60%, and static pinning reaches 99% hit. But every configuration failed the ≤1% perplexity-increase gate on 137M and 340M multi-domain models.
  - Training-free cache-aware rerouting plus trained locality gives about 80% miss reduction at up to 3.4% perplexity cost.
  - [arXiv 2608.18261](https://arxiv.org/abs/2608.18261)
- **FloE** (compressing expert internals, lossy): 9.3× per-expert compression, 4.4–7.6% average degradation on Mixtral — [arXiv 2505.05950](https://arxiv.org/abs/2505.05950)
- **SwapMoE:** a virtual-expert remap loses 0.041 Rouge-2 on Switch Transformer summarization for a 3× memory reduction — [arXiv 2308.15030](https://arxiv.org/abs/2308.15030)
- **kTransformers Expert Deferral** (defers some routed experts' output to the next layer): ≤0.5% average accuracy drop — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

### Inferences
- **HOBBIT-style fallback** is the most promising lossy extension for Kolibri. Keep a resident low-bit copy of all experts, e.g. IQ1/IQ2 of 384×50 small experts, and on a miss for a low-weight expert (ranks 4–6 of 6 under sigmoid weights) use the low-bit copy instead of waiting on pread. Kolibri's main experts are already IQ3, so the remaining bit gap is small and the quality cost must be measured with the existing KLD-vs-base-logits harness. The literature's ≤1% numbers are for fp16/int8 → int4/int2 on Mixtral.
- **Skipping or substituting rank-5 and rank-6 misses** is cheap to try because sigmoid top-6 weights are often flat in the tail. SpecMD's observation that wide MoEs tolerate drops supports this, but BuddyMoE's DeepSeek-V2-Lite numbers (−4 points ARC) show it is not free.
- **Cache-aware rerouting**, i.e. biasing the router toward resident experts, is the most aggressive lever, and its perplexity cost (up to 3.4%) is likely too high for a quality-sensitive deployment.

### Gaps
- No quality numbers exist for low-bit fallback when the primary copy is already ~3-bit, which is Kolibri's case.
- MoE-CORE's AES quality impact was not extracted (only availability). SpecMD's drop/substitution quality numbers were not extracted in detail.

## Q5. CPU computes the cold expert vs load to the accelerator: when does each win, especially on unified memory?

### Takeaway
On discrete GPUs, Fiddler and kTransformers show that computing a non-resident expert on the CPU beats moving its weights over PCIe when few tokens hit that expert (decode, batch 1). Weight transfer wins for long prefills. GPU caches of hot experts beat pure CPU-MoE once the hit rate is high, for example upstream llama.cpp's new MoE cache: 1.57–2.20× over `--fit`/`-cmoe`. On unified memory there is no transfer, so the trade-off becomes CPU vs GPU kernel efficiency on the same DRAM, plus SSD-read cost when the model does not fit. I found no rigorous published measurement of `--cpu-moe` vs Metal expert compute on Apple Silicon.

### Cited Findings
- **Fiddler** rule per expert:
  - If resident, run on the GPU.
  - Else run on the GPU (transferring weights) only if `cpu_lat(s) > gpu_lat(s) + trans_lat`, where s is the number of tokens routed to that expert.
  - Else run on the CPU.
  - GPU latency is about constant in s (memory-bound), CPU latency is about linear in s. Mixtral experts are over 300 MB in fp16, and transfer takes 2–5× longer than GPU compute. The CPU path is impractical for long prefills.
  - [arXiv 2402.07033](https://arxiv.org/html/2402.07033)
- **kTransformers:** never moves experts; routed experts run on CPU with AMX (prefill) or AVX-512 (decode) kernels, because "PCIe bandwidth limits (e.g. 32 GB/s with PCIe 4.0)" make on-demand transfer a bottleneck. The CPU MoE kernel reaches 21.3 TFLOPS on DeepSeek-V3. Decode is 2.42–4.09× faster than Fiddler and 1.25–1.76× faster than llama.cpp (BF16), and 1.77–1.93× faster than llama.cpp quantized — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- **HybriMoE and DALI** (both on kTransformers) dynamically split experts between CPU and GPU per layer. HybriMoE reports 1.70× decode over kTransformers — [arXiv 2504.05897](https://arxiv.org/abs/2504.05897); [arXiv 2602.03495](https://arxiv.org/abs/2602.03495)
- **HOBBIT on unified-memory Jetson AGX Orin** (32 GB): 13.0× (Mixtral) and 18.9× (Phi-MoE) decode over llama.cpp. On RTX 4090+CPU it is roughly at parity with Fiddler on Mixtral (0.99×) — [arXiv 2411.01433](https://arxiv.org/html/2411.01433)
- **Upstream llama.cpp MoE cache** (PR #29887, merged 2026-10-07) on Qwen3.8-Flash-Next Q4_0 with PCIe4:
  - RTX 4090: 25.0 → 39.4 tok/s (72% hit) with `--fit --moe-cache-mib 6544`, and 40.7 tok/s (77% hit) with `-cmoe --moe-cache-mib 11000`.
  - RTX 5090: 30.8 → 67.8 tok/s (89% hit).
  - [PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887)
- **Same PR, multi-GPU follow-up** (#30112, 2× RTX 4090): decode 34.78 → 64.39 tok/s (95.8% hit), but prompt throughput drops to 0.73× — [PR #30112](https://github.com/ggml-org/llama.cpp/pull/30112)
- **Low-hit warning:** a commenter notes that at about 5% of expert size the hit rate can be so low that uploading plus GPU compute is slower than CPU compute of the whole layer. They propose restricting the cache to some layers — [PR #29887 comment (transwarp829)](https://github.com/ggml-org/llama.cpp/pull/29887)
- **PR #27861 negative results:** with about 10% of the model in VRAM on an RTX 2080 Ti, decode fell from 15.5 to 6.6 tok/s. Upload threads contending with compute threads halved speed for another user — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861)
- **PR #25294** (SSD streaming, GB10 with 128 GB unified memory, GLM-5.2 254 GB): about 2.4× faster decode than mmap with `--cpu-moe` (2.08 vs 0.87 tok/s) and 5.3× faster prefill. The author explicitly targets unified-memory machines — [PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294)
- **Apple Silicon CPU-only expert streaming** (llama.cpp Discussion #27149, M1 16 GB, Qwen3-30B-A3B): CPU compute ceiling about 4.4 tok/s. With Q4 experts (~2.2 MB each), double-buffered pread hides all I/O even at 0% hit, so the CPU path is compute-bound. A Metal PoC with LRU expert paging reportedly reached 13 tok/s on an M1 Pro 16 GB (Discussion #23324, as cited) — [llama.cpp Discussion #27149](https://github.com/ggml-org/llama.cpp/discussions/27149)
- **MLX offload on a 32 GB Mac** (Qwen3.6-35B-A3B 4-bit, which fits): offload at fraction 0.3 costs about 5× decode (57.1 → 10.8 tok/s) and 8–12× prefill (957 → 76 tok/s warm), because a diverse prefill touches nearly all 256 experts — [mlx-lm #1438](https://github.com/ml-explore/mlx-lm/issues/1438)
- **Apple "LLM in a flash"** (dense models, 2023-12): windowing plus row-column bundling for flash reads; runs models up to 2× DRAM with 4–5× (CPU) and 20–25× (GPU) speedups over naive loading — [arXiv 2312.11514](https://arxiv.org/abs/2312.11514)

### Inferences
- **On Kolibri's M5 Pro**, "load to GPU" is a page-cache or SSD read into Metal-visible memory with no PCIe copy. "CPU compute" (`--cpu-moe`) reads the same DRAM but runs on CPU cores. On Apple SoCs the GPU generally sustains more of the shared bandwidth than the CPU clusters, though no CPU-vs-GPU split was found in sources. The Fiddler crossover therefore does not apply directly:
  - When the expert is resident, the GPU wins (60 tok/s fully resident vs ~12 tok/s `--cpu-moe` under pressure in Kolibri's own data).
  - When it is not resident, both paths pay the same SSD read, so CPU compute only helps if it lets the read and compute overlap better, or avoids synchronising the Metal command buffer on a miss.
- **A Fiddler/kTransformers-style hybrid could still matter on a Mac**: compute a missed expert on the CPU directly from the pread buffer while the GPU continues other experts, avoiding a GPU stall on the miss. The kTransformers paper attributes much of its gain to removing CPU-GPU synchronisation (CUDA graphs), the same kind of cost a Metal miss path pays.
- **Upstream llama.cpp's merged cache** caches `is_host` weight buffers into a GPU/IGPU device buffer. On Apple unified memory with mmap'd weights, its benefit would come from residency (keeping hot experts out of the evictable page cache), not from bandwidth. This is an inference from the code, not a measured result.

### Gaps
- No published, controlled measurement of llama.cpp `--cpu-moe` / `-ot exps=CPU` vs Metal expert compute on Apple Silicon (M3/M4/M5) was found.
- No paper models the Fiddler crossover for unified memory with an SSD tier.

## Q6. Implementations in llama.cpp, ik_llama.cpp, MLX, vLLM, SGLang

### Takeaway
As of October 2026, upstream llama.cpp has a merged MoE expert cache: PR #29887 (2026-10-07) plus multi-GPU support in #30112 (2026-10-08). It is a segmented LRU (probation/protected) cache of host-resident experts in a GPU buffer, decode and small batches only (≤32 tokens), no prefetch. SSD streaming (#25294) is still an open PR. MLX has an open feature request (#1438) and community implementations. vLLM has an open RFC with an LFRU cache PR. SGLang relies on kTransformers-style CPU experts. I found no expert cache in ik_llama.cpp.

### Cited Findings
- **llama.cpp PR #29887** "llama : add a GPU cache for MoE experts kept in host memory" (am17an):
  - Status: opened 2026-10-03, merged 2026-10-07. A port of the tetherto qvac-fabric MoE cache. Enable with `--moe-cache-mib N`; "10% of the total expert size is good."
  - Scope: MUL_MAT_ID on host experts runs on the GPU with only misses uploaded; ≤32-token batches.
  - Code: `src/llama-moe-cache.cpp` implements an SLRU (probation and protected lists, protected capped at half the slots, hits promote). Layers of the same device and expert layout share one SLRU and its banks, so slots are shared across a group of layers, not partitioned per layer. Misses are sorted so uploads coalesce into consecutive slots. Both GPU and IGPU device types are accepted.
  - [PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887); source read from `ggml-org/llama.cpp` master `src/llama-moe-cache.cpp` via the GitHub API.
- **llama.cpp PR #30112** (merged 2026-10-08): multi-GPU MoE cache, 1.74–1.85× decode on 2× RTX 4090 — [PR #30112](https://github.com/ggml-org/llama.cpp/pull/30112)
- **llama.cpp PR #25294** "stream MoE routed experts from disk" (opened 2026-07-04, still open):
  - Per-layer device-side cache of `n_slots` expert slabs, a CPU id-remap op after router top-k, async pread pool with optional O_DIRECT.
  - Eviction by "decaying route hotness" with LRU tie-break. Wave-partitioned prefill loads each expert once per ubatch.
  - Flags: `--moe-stream`, `--moe-stream-cache`, `--moe-stream-io-threads`, `--moe-stream-direct`.
  - GB10 GLM-5.2: 73–79% hit, 1.83–2.20 tok/s decode.
  - Reviewers asked for pluggable eviction and an RFC first; one argued against merging until competing approaches are evaluated.
  - [PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294)
- **llama.cpp PR #27861** (draft, opened 2026-08-28): per-layer GPU LRU of host experts, decode-only, throttled asynchronous uploads, no prefetch. +31% (18.4 → 24.2 tok/s) on Qwen3.8-Flash-Next; many community ports and edge cases reported — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861)
- **llama.cpp PR #26563** (opened 2026-08-04, closed for redesign): CUDA-only hot-expert heatmap tier with hysteresis and dwell on swaps; cold experts on CPU. 33 → 56 tok/s on Qwen3.6-35B-A3B with 8 GB VRAM — [PR #26563](https://github.com/ggml-org/llama.cpp/pull/26563); secondary report says Qwen3.5-122B-A10B was slower with caching — [prismix.dev](https://prismix.dev/news/7ffdd0b8ae0b)
- **Other llama.cpp threads:**
  - Issue #20757: two-tier GPU+RAM cache with pluggable eviction — [issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757)
  - Issue #27562: JIT expert streaming — [issue #27562](https://github.com/ggml-org/llama.cpp/issues/27562)
  - Discussion #27149: SSD streaming for CPU-only machines, with expert-contiguous re-layout cutting page faults 36× (192 vs 6,912 pages per expert) and calibrated pinning + LRU at 56% hit — [Discussion #27149](https://github.com/ggml-org/llama.cpp/discussions/27149)
- **Related llama.cpp PRs** (list by Green-Sky): #17044, #21609, #21614, #23170, #24524, #25932, #26414, #26824 — [PR #29887 comments](https://github.com/ggml-org/llama.cpp/pull/29887)
- **ik_llama.cpp:** GitHub search for "expert cache" or "offload experts" found no expert-cache PR. Related items: an open "ATSInfer Tensor Placement Solver" PR (#2259, 2026-08-05) and an open "NUMA Mirror" PR (#2396) — searched via `gh search prs --repo ikawrakow/ik_llama.cpp` on 2026-10-22.
- **MLX:**
  - mlx-lm issue #1438 (opened 2026-06-27, open) requests SSD expert streaming. The thread documents an implementation (mlx-lm PR #1588 referenced) with lazy load, per-layer LRU and byte-range reads from safetensors, plus measurements:
    - LRU decode hit about 0.80–0.91.
    - Prefetch nulls.
    - Coalescing up/gate/down plus scales into one read: +14–18% decode on Qwen 8-bit (~1 MB slices), only +9.7% on gpt-oss (~4 MB slices, bandwidth-bound).
    - Measurement traps: F_NOCACHE does not bypass already-resident pages, and lifetime vs decode-only hit rates differ.
    - [mlx-lm #1438](https://github.com/ml-explore/mlx-lm/issues/1438)
  - Other Apple Silicon projects: Jakevin/moe-offload (SSD + per-layer LRU for MLX and llama.cpp) — [GitHub](https://github.com/Jakevin/moe-offload); streamlx (per-layer LRU pool, bitwise identical) — [GitHub](https://github.com/srcterm/streamlx/blob/main/README.md); ssd-moe/deepseek-v4-flash-mlx (~4.5–5 tok/s on a 48 GB Mac, SSD-bandwidth-bound) — [GitHub](https://github.com/ssd-moe/deepseek-v4-flash-mlx); SwiftLM — [GitHub](https://github.com/SharpAI/SwiftLM). All numbers are self-reported.
- **vLLM:**
  - RFC #38256 (opened 2026-03-26, open): `ExpertWeightProvider` with an LFRU (frequency/age) GPU cache. PR 1 #37190 is open (~980 LOC, needs `--enforce-eager`). Async H2D plus temporal prediction come in PR 2. No maintainer replies visible.
  - Earlier monolithic RFC #33869 / PR #31938 closed as too large. Static selective CPU offload #34535 merged.
  - [vLLM #38256](https://github.com/vllm-project/vllm/issues/38256)
- **SGLang:** no expert-cache PR found. SGLang integrates kTransformers CPU experts, e.g. open PR #43500 "Propagate SwiGLU clamp to KT CPU experts" (2026-10-10). A UVM-based MoE expert offloading PR #20126 was closed (2026-03) — searched via `gh search prs --repo sgl-project/sglang` on 2026-10-22. The LMSYS blog covers the kTransformers integration — [LMSYS blog](https://www.lmsys.org/blog/2025-10-22-KTransformers)
- **antirez/ds4 PR #647** (SSD-streaming LRU on DeepSeek V4 Flash): 1.03 → 2.96 tok/s at a 100 GB cache budget and about 81% hit; long sessions converge to 96–98% — [GitHub](https://github.com/antirez/ds4/pull/647) (from search summary, not opened)

### Inferences
- Kolibri's prototype now overlaps directly with upstream llama.cpp. #29887 is merged (slot cache, misses uploaded, decode-only) and #25294 (pread from GGUF, O_DIRECT, async I/O threads, decaying hotness + LRU) is pending. Differences worth noting:
  - Kolibri: per-layer partition, Metal unified memory with no upload, F_NOCACHE.
  - Upstream: SLRU shared across layer groups, CUDA/IGPU.
- Upstreaming Kolibri's work would likely mean aligning with #29887's `llama_moe_cache` abstraction (SLRU, banks, `--moe-cache-mib`) and #25294's disk tier, rather than a third independent implementation. Reviewers on #25294 explicitly asked for pluggable eviction and convergence.

### Gaps
- I did not verify whether #29887's cache is active on the Metal backend (it accepts IGPU device types, but host-buffer semantics on Apple unified memory were not checked by running it).
- I could not confirm mlx-lm PR #1588's merge status.
- No ik_llama.cpp discussion threads on expert caching were found. Its search index may have missed issues or discussions.

## Q7. What is novel in the Kolibri prototype vs already done, and which ideas are worth adopting next

### Takeaway
Already done elsewhere:
- Per-layer LRU slot cache with async pread misses.
- O_DIRECT/F_NOCACHE reads.
- Markov / token-history prefetch.
- The negative result that predictors barely beat LRU at batch 1.

Less covered, and so potentially novel:
- Measurements at 384 experts with top-6 sigmoid routing and tiny (512-wide) experts.
- Unified-memory Metal with no upload step.
- A rigorous offline policy study in that regime.

The best-supported next steps are not better predictors. They are: SLRU or scan-resistant admission and cross-layer slot sharing; hidden-state NLG prefetch as one cheap experiment; I/O layout coalescing, since small experts make reads latency-bound; and lossy miss handling (low-bit fallback or skipping low-weight misses) measured with KLD.

### Cited Findings
- SLRU with probation and protected segments, shared across layer groups, is what upstream llama.cpp merged — [PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887)
- Layer-adaptive cache sizing:
  - AdapMoE's DP gives early layers more slots — [arXiv 2408.10284](https://arxiv.org/html/2408.10284)
  - MoE-CORE moves slots from insensitive to low-hit layers, though this was worth only 1.1% TPOT in its ablation — [arXiv 2610.01950](https://arxiv.org/html/2610.01950)
  - WiSP allocates memory by marginal value per byte between experts and KV — [arXiv 2606.21868](https://arxiv.org/abs/2606.21868)
- I/O coalescing pays when reads are latency-bound (about 1 MB slices: +14–18% decode) and fades when bandwidth-bound (about 4 MB slices: +9.7%) — [mlx-lm #1438](https://github.com/ml-explore/mlx-lm/issues/1438). Expert-contiguous layout cuts page faults 36× — [Discussion #27149](https://github.com/ggml-org/llama.cpp/discussions/27149)
- Prefill is the biggest unexamined lever:
  - Zhang: layer-major vs chunk-major prefill ordering differs "by roughly two orders of magnitude in total expert traffic for long contexts" under a uniform-routing approximation — [arXiv 2608.07911](https://arxiv.org/pdf/2608.07911)
  - Wave-partitioned prefill in #25294 gave about 2–2.5× prefill (2.28 → 5.65 tok/s) — [PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294)
  - MLX offload prefill is 8–12× slower than resident — [mlx-lm #1438](https://github.com/ml-explore/mlx-lm/issues/1438)
- Low-bit fallback on miss: ≤1% accuracy cost — [HOBBIT](https://arxiv.org/html/2411.01433). Low-score-miss substitution: 92.4% → 99.97% availability — [MoE-CORE](https://arxiv.org/html/2610.01950)
- Speculative decoding or MTP interacts with the cache: SP-MoE (1.07–3.5× TPOT) — [arXiv 2510.10302](https://arxiv.org/abs/2510.10302). Community #27861 runs with MTP2 report +12–25% to 30–32 tok/s — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861). The vLLM RFC warns speculative decoding multiplies the expert working set — [vLLM #38256](https://github.com/vllm-project/vllm/issues/38256)

### Inferences
Ranked suggestions for Kolibri (my judgement from the above):
1. **Write up the negative predictor result** with Zhang's reporting checklist: event-atomic replay, decode-only hit rate, per-step union/capacity, Belady gap, template diversity. At 384 experts it is a genuine data point the literature lacks.
2. **Try SLRU and a global (cross-layer) slot pool** in the offline simulator, then online. This is cheap, matches upstream, and handles prefill pollution and uneven per-layer demand.
3. **Run one hidden-state NLG experiment** (layer L+1's and L+2's sigmoid router on layer L's residual) to close the "which predictor" question properly. Expect a small ceiling (+3 pp in MoE-CORE).
4. **Coalesce each expert's gate/up/down (+ scales) into one contiguous extent** in a repacked side-file. Kolibri's 512-wide IQ3 experts are small, so reads are IOPS- or latency-bound, the regime where coalescing paid +14–18% in MLX.
5. **Lossy miss path**: for misses at sigmoid rank ≥4 with low weight, either skip or use a resident IQ1/IQ2 copy. Gate it on KLD against the base logits. This attacks the p95 stall directly, which prefetch did not.
6. **Prefill**: expert-major or wave-partitioned prefill so each expert is read once per ubatch.
7. **Optionally**, a Fiddler-style CPU fallback that computes a missed expert from the pread buffer on CPU while the GPU proceeds. This is only worth it if Metal miss synchronisation (not SSD bandwidth) dominates p95.

### Gaps
- None of these have been validated at 384 experts top-6. The expected gains are extrapolated from 8–256-expert models and different hardware.
- I found no published p95/p99 tail-latency analysis of expert-cache misses. Nearly all papers report mean TPOT and throughput.
