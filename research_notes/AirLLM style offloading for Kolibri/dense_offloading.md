# Weight-offloading and flash/SSD inference systems for (mostly dense) LLMs: AirLLM relatives and alternatives

Scope: non-MoE-expert-specific offloading and flash-streaming systems, compared on mechanism, bytes moved per token, tok/s, batch assumptions and maturity, plus which ideas transfer to llama.cpp/Metal on a 48 GB Apple Silicon Mac running Kolibri-1 (MoE). Research date: 2026-10-22. Older results are dated inline.

## 1. Apple "LLM in a Flash" (Alizadeh et al., arXiv 2312.11514, Dec 2023, rev. 2024)

### Takeaway
The paper is the closest published precedent to the Kolibri expert cache. It ran on Apple Silicon (M1 Max/M2 Ultra) at batch 1 and keeps about half the model in DRAM. On OPT-6.7B it cuts flash traffic from 13.4 GB to about 0.2 GB per token by combining four things: a low-rank sparsity predictor, a sliding "window" cache of recently active neurons, row-column bundling (32 KiB+ reads), and a preallocated, swap-delete DRAM layout. End-to-end speedup over naive loading is about 4-5x on CPU and 20-25x on GPU. Apple released no code. PowerInfer-2's authors had to reimplement it.

### Cited Findings
- Abstract claims: models up to 2x available DRAM, and 4-5x (CPU) and 20-25x (GPU) inference speedup over naive loading. Windowing "reuses previously activated neurons"; row-column bundling "increases the size of data chunks read from flash" — [arXiv abs](https://arxiv.org/abs/2312.11514)
- Hardware: M1 Max with 1 TB flash. A 1 GiB linear read of an uncached file reaches >6 GiB/s. Random reads need "32KiB or larger random reads across multiple threads" for good throughput. Reads use 32 threads, with the OS cache disabled during benchmarks. Also tested on M2 Ultra (2 TB SSD) and an RTX 4090 Linux box. The setup assumes about half of memory is available for the model — [arXiv HTML v3](https://arxiv.org/html/2312.11514v3)
- I/O breakdown for OPT-6.7B FP16 on M1 Max (Table 2). Naive: 13.4 GB flash→DRAM per token, 2196 ms I/O. Hybrid (half in DRAM): 6.7 GB, 1090 ms. Adding the predictor: 0.9 GB, 738 ms. Adding windowing: 0.2 GB, 164 ms. Adding bundling: 0.2 GB at 2.25 GB/s effective, 87 ms — [arXiv HTML v3](https://arxiv.org/html/2312.11514v3)
- End-to-end latency per token, naive → all techniques (Table 3). OPT-6.7B: CPU 3182→669 ms, Metal M1 2389→565 ms, Metal M2 2270→305 ms, GPU 2218→84 ms. Falcon-7B CPU: 3095→706 ms. Persimmon-8B CPU: 3806→1041 ms. Llama-2-7B CPU: 3095→994 ms. So the best Apple Silicon result is about 3.3 tok/s, on Metal M2 with OPT-6.7B — [arXiv HTML v3](https://arxiv.org/html/2312.11514v3)
- Sparsity levels: OPT-6.7B about 97% FFN sparsity, about 2.4% of FFN neurons used per token. Relufied Falcon-7B about 95%. Llama-2 about 90%, using FATReLU rather than stock SiLU. The method relies on ReLU-type activations — [arXiv HTML v3](https://arxiv.org/html/2312.11514v3)
- Predictors are low-rank: rank 128-1152 per layer, trained on 10k C4 samples in about 4 h on an A100. Over 2.4% overhead in non-embedding weights and FLOPs. For OPT, about 5% false negatives and 7% false positives. They flag about 3x the neurons that actually activate — [arXiv HTML v3](https://arxiv.org/html/2312.11514v3)
- Windowing: neurons active in the last k tokens (k=2-5 depending on model) stay in DRAM, and only newly needed rows are loaded. The paper is internally inconsistent about k for OPT (4 vs. 5) — [arXiv HTML v3](https://arxiv.org/html/2312.11514v3)
- Row-column bundling: the i-th up-projection column and the i-th down-projection row activate together, so they are stored contiguously. This doubles the chunk size to 2·d_model·bytes (32 KiB for fp32 OPT). Bundling by co-activation ("closest friend") was a negative result, because hot neurons got reloaded repeatedly — [arXiv HTML v3](https://arxiv.org/html/2312.11514v3)
- DRAM management: each layer gets one preallocated matrix of Req_i × 2·d_model. Deleting a neuron swaps in the last row, costing O(c·d_model). New neurons are appended at the end with no reallocation. Memory management cost is comparable to flash time; for Falcon it was 161 ms flash vs. 90 ms memory management per token — [arXiv HTML v3](https://arxiv.org/html/2312.11514v3)
- Batch size is 1 throughout. MoE is mentioned only in related work, as something that "can leverage our method" (not evaluated) — [arXiv HTML v3](https://arxiv.org/html/2312.11514v3)
- Apple did not open-source it. PowerInfer-2's authors reimplemented "LLMFlash" in llama.cpp as a baseline — [PowerInfer-2 HTML](https://arxiv.org/html/2406.06282v3)
- Open community derivative: mlx-flash (github, 138 stars, MIT) cites the paper as inspiration. It lazily mmaps safetensors, paces `os.pread` with a token bucket, and prefetches layer N+1 while layer N computes. It reports no tok/s numbers, only a load-time comparison; MoE lookahead routing is on its roadmap (v0.6) — [mlx-flash](https://github.com/matt-k-wong/mlx-flash)

### Inferences
- The Kolibri LRU expert cache already applies "hybrid + windowing" at expert granularity, with the router as a perfect, zero-cost "predictor" for the current layer. What it lacks from LLM-in-a-Flash is intra-expert sparsity and the bundle-aware storage layout.
- Arithmetic from the user's numbers: Q8_0 83 GB / (50 layers × 384 experts) gives at most about 4.3 MB per expert. Top-6 × 50 = 300 expert reads per token. A 91% hit rate leaves about 27 misses, or about 110-120 MB per token. At 13 tok/s that is about 1.5 GB/s, well under the >6 GiB/s sequential figure measured on an M1 Max SSD. That suggests per-token latency is bound by synchronous read latency or memory management, not by flash bandwidth. LLM-in-a-Flash's 32-thread parallel reads and overlap are the relevant lessons.
- A per-neuron bundle (gate row + up row + down column) within a 512-wide Kolibri expert is about 4.3 MB / 512 ≈ 8 KB at Q8_0. That is the same scale as PowerInfer-2's 8 KB bundles, and below the 32 KiB that LLM-in-a-Flash found necessary for good random-read throughput on Mac SSDs. So neuron-level loading inside experts would need multithreaded, queue-deep reads to pay off.

### Gaps
- No published LLM-in-a-Flash numbers for M3/M4/M5 SSDs. No measured M5 Pro SSD random-read curve was found.
- No official Apple code. No measured reproduction on llama.cpp/Metal was found.

## 2. PowerInfer (2023), PowerInfer-2 (2024), TurboSparse (2024)

### Takeaway
The PowerInfer line exploits hot/cold neuron skew in ReLU-sparse (or dReLU-retrofitted) models. Hot neurons are pinned in fast memory (GPU or NPU), and cold neurons are computed on CPU or streamed from flash in small bundles through a cache. On a OnePlus 12 phone, PowerInfer-2 ran TurboSparse-Mixtral-47B at 11.68 tok/s with 19 GB and 2.13 tok/s with 7 GB. It beats a reimplemented LLM-in-a-Flash by 2.9-3.8x on average. The catch is that it needs models with real activation sparsity, which means retraining to ReLU/dReLU; stock SiLU models do not benefit much.

### Cited Findings
- PowerInfer (Dec 2023): hot neurons ("consistently activated across inputs") are preloaded on the GPU, and cold ones are computed on the CPU. Up to 11.69x faster than llama.cpp on one RTX 4090. For OPT-30B it reaches 82% of A100 speed — [arXiv 2312.12456](https://arxiv.org/abs/2312.12456)
- PowerInfer-2 (Jun 2024): matrix operations are split into "neuron clusters". Dense clusters run on the NPU and sparse ones on the CPU. It uses a segmented neuron cache and a cluster-level I/O-compute pipeline. Up to 27.8x faster than prior frameworks. Reported as the first 47B LLM served on a phone (11.68 tok/s) — [arXiv 2406.06282](https://arxiv.org/abs/2406.06282)
- Hardware: OnePlus 12 (SD 8 Gen 3, 24 GB, UFS 4.0) and Ace 2 (16 GB, UFS 3.1). UFS 4.0 sequential reads run from 450 MB/s at 4 KB to 4 GB/s at 512 KB. 4 KB random reads give about 1 GB/s within a 128 MB range and under 850 MB/s across 512 MB — [PowerInfer-2 HTML](https://arxiv.org/html/2406.06282v3)
- Results on OnePlus 12 with 50% of FFN offloaded: on average 3.84x faster than LLMFlash (max 4.63x) and 24.6x faster than llama.cpp mmap (max 27.8x). TurboSparse-Mixtral-47B reaches 11.68 tok/s with 19 GB (21.2x over llama.cpp) and 2.13 tok/s with 7 GB (400 MB neuron cache = 1.8% of FFN). Table 6: Mistral(SiLU)-7B 5.3 tok/s vs. LLMFlash 2.18. Bamboo-7B 11.1 vs. 2.4 — [PowerInfer-2 HTML](https://arxiv.org/html/2406.06282v3)
- Ablation on Bamboo-7B, 50% offloaded: 0.4 tok/s → 1.1 (bundling) → 4.18 (neuron cache) → 9.60 (pipeline) → 11.07 (XPU). I/O is 13.7% of time vs. 76.7% for LLMFlash. Neuron cache hit rate is 95% for Bamboo. For TurboSparse-Mixtral the mean miss rate is 3.5% and P99 is 18.9% — [PowerInfer-2 HTML](https://arxiv.org/html/2406.06282v3)
- I/O layout: an FP16 Mistral neuron is 8 KB, and the gate-up-down bundle is read as one 24 KB random read. For INT4, a bundle is 7.5 KB padded to 8 KB. Loading is two-phase: the gate is read first, and up/down only if the gate output is non-zero. About 80% co-activation across matrices for the same neuron. Hot neurons and attention use sequential reads; cold neurons use small random reads — [PowerInfer-2 HTML](https://arxiv.org/html/2406.06282v3)
- Batch dependence: under 1% of neurons are active at batch 1, but the hot share rises to about 75% at batch 32 (Bamboo-7B, layer 10). The planner covers batch 1-4 only. This is a single-user latency design — [PowerInfer-2 HTML](https://arxiv.org/html/2406.06282v3)
- TurboSparse (Jun 2024): dReLU retrofit. Mistral-7B averages about 90% inactive neurons. In Mixtral, routing alone gives 75% sparsity, and neuron sparsity inside experts (about 85% per expert) raises the total to about 97%. Active parameters per step: about 2.5B (Mistral) and 4.3B (Mixtral). Predictors are trained per expert. 2-5x decoding speedup, and about 11 tok/s on phones. The 85%/97% figures come from different sections and are not fully reconciled in the snippets — [TurboSparse arXiv 2406.05955](https://arxiv.org/html/2406.05955v1)

### Inferences
- TurboSparse is the only work found that applies neuron-level sparsity *inside* MoE experts, and it required a dReLU retrofit plus continued pretraining. For Kolibri, whose experts are only 512 wide and stock SiLU/SwiGLU (presumably), the extra gain from intra-expert sparsity is uncertain. The 128/384 LRU cache already behaves like PowerInfer-2's "segmented neuron cache" at expert granularity.
- Transferable ideas without retraining: the segmented cache (the existing LRU), an overlapped I/O-compute pipeline across layers (the 4.18 → 9.60 tok/s step in the ablation was the pipeline), and a two-phase gate-first read (only with real sparsity).

### Gaps
- Searches surfaced no 2025-2026 PowerInfer follow-up papers. Repo activity in 2025-26 was not checked.
- Bytes per token for PowerInfer-2 is not stated in the paper.

## 3. AirLLM and throughput-oriented offloading: FlexGen, DeepSpeed ZeRO-Inference, HF Accelerate, vLLM/SGLang

### Takeaway
These systems stream *all* (or most) weights per forward pass, layer by layer, and only become efficient by amortizing each layer load over a big batch. FlexGen managed OPT-175B at about 1 tok/s, but only with an effective batch of 144-256. ZeRO-Inference reached Llama-3-70B at 6-26 tok/s, but at batch 96. At batch 1 (Accelerate/ZeRO with disk) OPT-175B ran at 0.01 tok/s. AirLLM is the same idea at batch 1 and is therefore latency-bound by full-model reads per token. None of this fits single-user MoE decode, where only about 1-2% of weights are needed per token.

### Cited Findings
- AirLLM: splits the model into per-layer files and keeps "only one layer on the GPU at a time", with optional prefetch overlap. Optional 4/8-bit block-wise compression is claimed to give "up to 3x" speed. macOS is supported via MLX (Llama/Mistral classes only). The README lists VRAM footprints (e.g., Llama-3.x 70B ~4 GB), not tok/s. Latest release v3.0 (2026/06), training support added 2026/09 — [AirLLM GitHub](https://github.com/lyogavin/airllm)
- FlexGen (Mar 2023): an LP-based policy search over GPU/CPU/disk placement, with 4-bit weight and KV compression. OPT-175B reaches 1 tok/s at effective batch 144 on one 16 GB GPU — [arXiv 2303.06865](https://arxiv.org/abs/2303.06865)
- FlexGen benchmark: T4 16 GB, 208 GB DRAM, 1.5 TB SSD, prompt 512 and output 32, batch tuned to maximize throughput. OPT-175B: Accelerate 0.01 tok/s (bs 2, disk), ZeRO-Inference 0.01 (bs 1, disk), FlexGen 0.69 (bs 256, disk), FlexGen with compression 1.12 (bs 144, CPU). OPT-30B: Accelerate 0.62, ZeRO 0.60, FlexGen 7.32 (bs 144). The README warns it is "significantly slower… especially for small-batch cases". Its "block schedule" reuses each loaded weight across a block of batches instead of row-by-row. The repo was archived 2024-12-01 — [FlexGen GitHub](https://github.com/FMInference/FlexGen)
- DeepSpeed ZeRO-Inference with DeepNVMe: Llama-3-70B on an A100-80GB with 4 Gen4 NVMe drives gives 7 tok/s with GDS and 6 without, at prompt 512, gen 32, **batch 96**. With 4 Gen5 drives, 17 tok/s; with 8 Gen5 drives, 26 tok/s. Raw reads up to 9.6 GB/s — [PyTorch blog: DeepNVMe](https://pytorch.org/blog/deepnvme-affordable-i-o-scaling-for-deep-learning-applications/); original ZeRO-Inference announcement (2022) — [deepspeed.ai](https://www.deepspeed.ai/2022/09/09/zero-inference.html)
- HF Accelerate CPU offload: Llama-2-70B on an L40 ran at 5.5-11 s/token vs. 0.56 s/token with Sequoia speculative decoding — [Sequoia arXiv 2402.12374](https://arxiv.org/pdf/2402.12374). Disk tier in PIPO: Llama-3.1-8B with FlexGen gave 1.9 tok/s with disk weights vs. 5.5 tok/s with CPU weights; PIPO's pipelined disk path gave 4.9 tok/s — [PIPO arXiv 2504.03664](https://arxiv.org/pdf/2504.03664)
- vLLM `--cpu-offload-gb`: a "virtual" GPU memory extension. Offloaded weights are copied CPU→GPU "on the fly in each model forward pass" and need a fast interconnect. No official benchmark numbers — [vLLM docs](https://docs.vllm.ai/en/latest/examples/basic/offline_inference/). SGLang has the same flag. A vendor blog says its default path does not overlap transfers, so the copy stalls the layer (vendor claim, unverified) — [Doubleword blog](https://blog.doubleword.ai/flashoffload)
- The Mixtral-offloading paper (Eliseev & Mazur, Dec 2023) states the general rule: offloading is "significantly slower" for chat-style generation because the system "spends most of the time waiting for next layer's parameters to be loaded" — [arXiv 2312.17238](https://arxiv.org/pdf/2312.17238)

### Inferences
- Bytes per token at batch 1 for a dense layer-streaming design (AirLLM) equal the whole off-RAM model. For an 83 GB Q8_0 Kolibri at about 6 GiB/s, that is ≥13 s/token. Expert-granular caching (about 0.1 GB/token) is two orders of magnitude better. AirLLM-style whole-layer streaming is the wrong granularity for Kolibri.
- The useful borrowable idea is FlexGen's block schedule, *if* Kolibri prefill or multi-request batching is a goal. During prefill of N tokens, a non-resident expert can be loaded once and applied to every token routed to it. This mirrors FlexGen's reuse of a loaded layer across a batch block.
- On Apple Silicon unified memory, "CPU offload" (vLLM/Accelerate) collapses to nothing: there is no PCIe hop. Only the SSD tier matters.

### Gaps
- No 2025-2026 head-to-head of vLLM vs. SGLang CPU offload was found.
- The FlexGen paper's "zig-zag" naming is not on the README. The paper PDF was not fetched.

## 4. llama.cpp's own mechanisms for models bigger than RAM

### Takeaway
llama.cpp relies on mmap plus the OS page cache. A model larger than RAM "works", but reads go through page faults with kernel readahead and eviction you cannot control. Reported results for big MoEs from SSD are about 1-5 tok/s. Direct I/O (`-dio`, O_DIRECT, Linux async loader) was added in early 2026 and then made opt-in. It is a *load-time* path that disables mmap, not a runtime streaming cache. Recent manpages mark `--mmap`/`--mlock`/direct-io as deprecated in favor of `--load-mode`. There is no upstream expert or weight streaming cache, which is the gap the Kolibri fork fills.

### Cited Findings
- 2023 design: the GGUF/ggml layout change let weights be mmapped directly, so "the kernel can make its file cache pages directly accessible" and those pages are less likely to be evicted — [commit 78ca9838](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/commit/78ca9838ee36660a776e97e3391b6fb5dcaacf7f)
- Current CLI: `-dio/--direct-io` means "use DirectIO if available", default disabled. `--mmap` is on by default. Disabling mmap gives slower load but "may reduce pageouts if mlock is not used". Several manpages say `--mmap`, `--mlock` and direct-io are deprecated in favor of `--load-mode` — [Debian llama-cli manpage](https://dyn.manpages.debian.org/testing/llama.cpp-tools/llama-cli.1.en.html); [llama-server manpage](https://manpages.debian.org/unstable/llama.cpp-tools/llama-server.1.en.html)
- Direct I/O history: #18012 "Async DirectIO model loading on Linux" (O_DIRECT, with fallback) — [commit](https://app.semanticdiff.com/gh/ngxson/llama.cpp/commit/4d4f4cacd1c18975e5a97f20369fc30a225dc51f). #19109 made it off by default, merged around 2026-01-27 — [PR #19109](https://app.semanticdiff.com/gh/ggerganov/llama.cpp/pull/19109/overview). #18778 added `direct_io` to llama-bench — [PR #18778](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/pull/18778/overview). #18841 (Jan 2026) set llama-bench defaults to mmap=false and direct_io=false — [PR #18841](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/pull/18841/overview). A user fixed RPC large-model load hangs with `-dio` — [issue #19745](https://github.com/ggml-org/llama.cpp/issues/19745)
- Community results (dated, secondary). DeepSeek-R1 671B via mmap on a 96 GB RAM rig with GPU disabled: about 2.13 tok/s, with "the rest of your system RAM acts as disk cache for the active weights" (r/LocalLLaMA, Jan 2025, via aggregator). Swap-file approach: about 0.3 tok/s and unstable. A Level1Techs thread got 8.44 tok/s on CPU with a 211 GiB Q2_K and found a fast-IOPS RAID "does not provide significantly better performance" than a single Gen5 x4 NVMe — [Level1Techs thread](https://forum.level1techs.com/t/deepseek-r1-671b-backed-by-fast-read-iops/225008); [r/LocalLLaMA mirror](https://axebps-redlib.hf.space/r/LocalLLaMA/s/CibplDLFNx)
- GPT-OSS-120B on a 24 GB M4 Pro MacBook Pro via CPU and mmap: about 4.5 tok/s peak. "DeepSeek V4 Flash" on the same machine: about 2 tok/s (blog, unverified) — [gopenai blog](https://blog.gopenai.com/running-oversized-gguf-llms-on-low-ram-hardware-using-system-memory-mapping-da11e2db6106)
- Older community advice (2024): mmap can run anything that fits on disk, but slowly. Hammering an SSD with churn is discouraged — [llama.cpp issue #5207](https://github.com/ggerganov/llama.cpp/issues/5207)

### Inferences
- The user's own A/B (Q8_0: about 13 tok/s with pread + F_NOCACHE + LRU vs. about 6.5 tok/s with `--cpu-moe` + mmap) matches the literature. Application-managed caches with known granularity beat kernel page-cache heuristics: readahead pulls in neighboring experts, and eviction is unaware of the router. This is the same lesson as LLM-in-a-Flash disabling the OS cache and managing DRAM itself.
- `-ot`/`--override-tensor` and `--cpu-moe` only decide *placement* (CPU vs. GPU buffers). On unified-memory Macs they do not create a streaming tier; with mmap, CPU-placed tensors still page-fault from SSD.
- `--mlock` and wired memory are counter-productive when model > RAM. Pinning more than about 48 GB is impossible. Per user memory notes, wired-memory pressure can panic the kernel on this Mac.

### Gaps
- `--load-mode` values and semantics were not fetched. Whether llama.cpp's DirectIO is implemented on macOS (F_NOCACHE) or Linux-only (O_DIRECT) was not confirmed; the PR title says "on Linux".
- No reproducible Apple Silicon benchmark of llama.cpp with a model larger than RAM from internal SSD was found in primary sources.

## 5. MLX / mlx-lm for models bigger than memory

### Takeaway
mlx-lm has no SSD weight-streaming mode. Its guidance for large models is to raise `iogpu.wired_limit_mb` and wire the model, which only helps if the model fits. An open MLX feature request (Dec 2025) asks for out-of-core streaming. Third-party projects (mlx-flash, SwiftLM `--stream-experts`) try to fill the gap but publish few or unverifiable numbers.

### Cited Findings
- mlx-lm README: "Models which are large relative to the total RAM available on the machine can be slow." mlx-lm wires model and cache memory (requires macOS 15+). Users can raise the limit with `sudo sysctl iogpu.wired_limit_mb=N`, where N should exceed model size but stay below RAM. No lazy-loading or streaming option is documented — [mlx-lm GitHub](https://github.com/ml-explore/mlx-lm)
- MLX issue #2878 (Dec 2025): a feature request for on-disk (out-of-core) weight streaming via mmap and block-wise loading, apparently unimplemented. A separate request asks for Flash-MoE-style expert streaming — [MLX issue #2878 mirror](https://upd.dev/ml-explore/mlx/issues/2878)
- SwiftLM `--stream-experts` on an M1 Max 64 GB logged an overcommit ratio of 1.3 and an estimated 3.3 tok/s (hands-on blog) — [lilting.ch](https://lilting.ch/en/articles/swiftlm-m1max-hands-on)
- mlx-flash: lazy mmap, pread prefetch of layer N+1, and KV older than 128 tokens quantized to disk. No tok/s published — [mlx-flash](https://github.com/matt-k-wong/mlx-flash)
- oMLX's SSD cache is a *KV* cache and does not enable models bigger than RAM — [runaihome review](https://runaihome.com/blog/omlx-apple-silicon-llm-inference-review-2026/)

### Inferences
- The MLX ecosystem is behind the Kolibri fork here, so there is no mature MLX design to copy. The only MLX-side lesson is the general one that wiring is incompatible with oversubscription.

### Gaps
- The SwiftLM expert-streaming design is MoE-specific (covered by the other researcher) and was not investigated further.

## 6. Activation-sparsity methods (Deja Vu, CATS, TEAL, ReLU-fication) and applicability to MoE expert FFNs

### Takeaway
Training-free methods (TEAL, CATS) reach 40-50% activation sparsity with 1.15-1.8x decode speedups for dense 7-70B models. ReLU-fication or dReLU (LLM-in-a-Flash's relufied models, TurboSparse) reaches about 90-97%, but needs continued training. Only TurboSparse demonstrates sparsity inside MoE experts. For a model with tiny (512-wide) experts and top-6 routing, the extra I/O savings from intra-expert sparsity are bounded and unproven.

### Cited Findings
- Deja Vu (ICML 2023): contextual (input-dependent) sparsity over attention heads and MLP neurons, predicted on the fly by a low-cost predictor. OPT-175B latency is over 2x lower than FasterTransformer and over 6x lower than HF — [arXiv 2310.17157](https://arxiv.org/abs/2310.17157)
- CATS (COLM 2024): a thresholded activation in Gated-MLP blocks, with per-block cutoffs set from 500 samples. Within 1-2% of the base model at 50% sparsity without fine-tuning. Custom GPU kernel gives about 15% faster token generation (Llama-2-7B, Mistral-7B). Not tested on Mixtral experts — [CATS arXiv 2404.08763](https://arxiv.org/pdf/2404.08763); [Stanford blog](https://scalingintelligence.stanford.edu/blogs/cats)
- TEAL (Aug 2024): training-free magnitude sparsity on all hidden states, 40-50% model-wide. Up to 1.53x decode speedup at 40% and 1.8x at 50%, on Llama-2/3 and Mistral 7B-70B. The abstract does not mention MoE — [arXiv 2408.14690](https://arxiv.org/abs/2408.14690)
- ReLU-fication: LLM-in-a-Flash uses relufied Falcon (about 95%) and FATReLU Llama-2 (about 90%) — [LLM in a Flash HTML](https://arxiv.org/html/2312.11514v3). TurboSparse dReLU in Mixtral experts gives about 85% per expert and about 97% overall with routing, using per-expert predictors — [TurboSparse](https://arxiv.org/html/2406.05955v1)

### Inferences
- For *bandwidth* (bytes from SSD) rather than compute, TEAL/CATS-style sparsity helps only if the storage layout allows fetching a neuron subset of an expert (row-column bundling). At about 8 KB per neuron bundle in a Q8_0 Kolibri expert, skipping 40-50% of neurons would halve miss traffic. But it requires many small random reads, and quantization blocks (Q8_0's 32-element blocks along the reduction dimension) complicate column-subsetting the down projection.
- TEAL-style input sparsification (sparsifying the hidden state x before the up/gate GEMV) would skip *columns* of gate/up, i.e. a d_model dimension. That does not reduce which rows must be loaded unless weights are stored transposed. Output-side (intermediate) sparsity, as in CATS or ReLU, maps better to row/column bundling.
- Expected value for Kolibri is lower than for dense models. With a 91% expert hit rate, misses are already about 9% of expert reads. Even 50% intra-expert sparsity would cut SSD bytes per token from about 0.11 GB to about 0.06 GB (estimated) and would not touch the resident 91%. Compute savings on Metal from sparse GEMV at batch 1 are also uncertain.

### Gaps
- No paper found applying TEAL or CATS training-free to fine-grained MoEs (many small experts). No quality data for sparsifying SiLU experts of width 512.
- Deja Vu details (batch size, predictor architecture) were not extracted beyond the abstract.

## 7. Cross-cutting comparison and transfer to Kolibri on a 48 GB M5 Pro

### Takeaway
Two families exist. (a) Throughput offloaders (FlexGen, ZeRO-Inference, Accelerate, vLLM/SGLang `--cpu-offload-gb`, AirLLM) stream whole layers and need big batches. (b) Latency-oriented sparse streamers (LLM in a Flash, PowerInfer-2) fetch only the needed rows through an app-managed cache with I/O-compute pipelining. Kolibri's expert cache belongs to (b) at expert granularity. The most transferable remaining ideas are pipelining/prefetch, parallel deep-queue reads, a preallocated slot layout, batch-block reuse during prefill, and possibly neuron-level bundling. Layer-streaming à la AirLLM is not.

### Cited Findings (summary table; every number cited above)
| System | Granularity | Batch assumption | Best reported decode | Bytes/token from slow tier | Maturity |
|---|---|---|---|---|---|
| LLM in a Flash (2023-24) | FFN neuron (row+col bundle) | 1 | OPT-6.7B: 305 ms/token on Metal M2, 84 ms on GPU ([src](https://arxiv.org/html/2312.11514v3)) | about 0.2 GB of 13.4 GB ([src](https://arxiv.org/html/2312.11514v3)) | Paper only, no Apple code |
| PowerInfer-2 (2024) | Neuron cluster + cache | 1-4 | TurboSparse-Mixtral-47B 11.68 tok/s on a phone with 19 GB ([src](https://arxiv.org/html/2406.06282v3)) | Not stated. 3.5% mean cache miss ([src](https://arxiv.org/html/2406.06282v3)) | Research code; needs sparse models |
| PowerInfer (2023) | Hot (GPU) / cold (CPU) neuron | 1 | Up to 11.69x llama.cpp on RTX 4090 ([src](https://arxiv.org/abs/2312.12456)) | n/a (CPU DRAM, not flash) | Research |
| FlexGen (2023) | Layer, block schedule | 144-256 | OPT-175B 0.69-1.12 tok/s throughput ([src](https://github.com/FMInference/FlexGen)) | Whole model per block | Archived Dec 2024 |
| ZeRO-Inference + DeepNVMe | Layer | 96 | Llama-3-70B 7-26 tok/s throughput ([src](https://pytorch.org/blog/deepnvme-affordable-i-o-scaling-for-deep-learning-applications/)) | Whole model per step | Maintained |
| HF Accelerate offload | Layer | 1-2 | OPT-175B 0.01 tok/s from disk ([src](https://github.com/FMInference/FlexGen)) | Whole model | Maintained, slow |
| AirLLM | Layer | 1 | No tok/s published ([src](https://github.com/lyogavin/airllm)) | Whole model | Active (v3.0, 2026) |
| llama.cpp mmap | Page (kernel) | 1 | R1-671B about 2 tok/s, 96 GB box ([src](https://forum.level1techs.com/t/deepseek-r1-671b-backed-by-fast-read-iops/225008)) | Kernel-driven | Mainline |

### Inferences (ranked by expected value for the Kolibri fork)
1. **Overlap I/O with compute (PowerInfer-2's pipeline: 4.18 → 9.60 tok/s in its ablation).** In MoE, layer L's router output is known only at layer L. Speculatively prefetching layer L+1's likely experts (e.g., using layer L's hidden state through L+1's router, or the previous token's set) while layer L computes is the analog. MoE-specific prefetch designs belong to the other researcher's scope.
2. **Parallel, deep-queue reads of ≥32 KiB (LLM-in-a-Flash: 32 threads, chunk-size-dependent throughput).** At about 4 MB per expert, reads are already large. Issuing all about 27 misses per token concurrently, rather than serially per layer, should raise effective bandwidth. The current about 1.5 GB/s effective vs. >6 GiB/s possible suggests headroom (estimate).
3. **Preallocated slot arrays with swap-delete (LLM-in-a-Flash DRAM management)** avoid allocation churn. Memory management cost rivalled flash time in that paper (90 ms vs. 161 ms for Falcon).
4. **Block reuse during prefill and batching (FlexGen block schedule):** load each missing expert once per ubatch and apply it to all tokens routed to it.
5. **Intra-expert neuron sparsity (TurboSparse, CATS, TEAL):** lowest priority. Gains are bounded by the about 9% miss traffic, it needs quality validation, and 8 KB bundles are below Apple SSD's efficient random-read size.
6. **Not transferable:** whole-layer streaming (AirLLM, Accelerate, ZeRO, FlexGen at batch 1), CPU→GPU offload flags (no PCIe tier on unified memory), and `--mlock`/wired limits when model > RAM.

### Gaps
- No measured random-read throughput curve for the M5 Pro internal SSD. Its bandwidth is the key parameter for points 2 and 5 and should be measured locally (e.g., a small Go pread benchmark with F_NOCACHE at 8 KB / 32 KB / 4 MB chunks and varying thread counts).
- No published number for llama.cpp + Metal with a model larger than RAM on M-series hardware from a primary source.
