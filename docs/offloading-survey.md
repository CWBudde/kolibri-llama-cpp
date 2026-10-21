# Kolibri needs expert caches, not AirLLM streaming

AirLLM's whole-layer streaming is the wrong tool for interactive Kolibri-1 decode. It re-reads the entire model from disk for every token. The author's own flagship measurement is **292 s/token**, and users report 3 s to 2 min per token. For an 83 GB Q8_0 Kolibri at realistic Mac SSD speeds, that puts it **roughly 200 to 400 times slower** than the expert cache this repo already has. Since 2023 the field has moved to finer granularities. These are neuron bundles (LLM in a Flash, PowerInfer-2), routed-expert caches with SSD or PCIe misses (Mixtral-offloading, flash-moe, llama.cpp PR #25294, this repo), CPU compute of cold experts (Fiddler, kTransformers) and lossy miss handling (HOBBIT, BuddyMoE). For batch-1 decode the expert cache is the right family. The careful 2026 evidence agrees with this repo's finding that predictors barely beat LRU at that batch size. I found no public SSD-streamed or AirLLM run of Kolibri-1. This repo's 17–18 tok/s IQ3 and 13 tok/s Q8_0 figures look like the only published ones. The more useful finding comes from the repo's own Stage D data. A cached decode token takes about 58 ms: about 16.5 ms is the compute a resident model needs and **only 14 ms is waiting on SSD reads**. The remaining ~27 ms is overhead that no cache policy or predictor can touch. So the next wins are mainly engineering rather than research. In order: remove the per-layer synchronisation cost, rebuild prefill (where AirLLM-style whole-layer streaming becomes the right idea), align with upstream llama.cpp, and add a KLD-gated lossy miss path.

## AirLLM pages whole layers and pays the full model per token

AirLLM splits a Hugging Face checkpoint into one safetensors file per layer and builds the model on the `meta` device. Forward hooks stream each layer to the GPU just before it runs and evict it right after, with a depth-1 prefetch on a single worker thread ([airllm_base.py](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/airllm_base.py#L58-L71)). Nothing is reused across tokens, so **bytes read per token equal the whole model**. The 2023 launch post said so directly: disk reading is "the slowest bottleneck" and the method is "not very suitable for interactive scenarios like chatbots" ([HF blog](https://huggingface.co/blog/lyogavin/airllm)). It descends from a Kaggle multiple-choice competition, where a single forward pass per question amortizes the I/O. Autoregressive decode removes that amortization.

The headline figures ("70B on 4 GB", "Kimi K3 2.8T on under 4 GB") are peak-VRAM numbers, not speeds ([README](https://github.com/lyogavin/airllm#tiny-gpu-huge-models)). The only author-measured v3/v4 throughput is **Kimi K3 at 292 s/token** ([PR #316](https://github.com/lyogavin/airllm/pull/316)). User reports fill in the rest:

- Qwen2.5-32B from the page cache runs at **13.3 s/token** ([issue #330](https://github.com/lyogavin/airllm/issues/330)).
- Qwen2.5-3B on NVMe runs at 28.6 s/token ([issue #364](https://github.com/lyogavin/airllm/issues/364)).
- A 1.5B model runs at 0.1 tok/s streamed against **32.2 tok/s fully loaded** ([PR #324](https://github.com/lyogavin/airllm/pull/324)).

The "up to 3x" compression claim does not hold on fast storage. Compression is storage-only NF4/8-bit that is dequantized on every load, and it disables prefetch. Users measured it **3.8–8× slower** ([issue #330](https://github.com/lyogavin/airllm/issues/330)).

Three properties rule AirLLM out for Kolibri specifically:

- **No Apple Silicon support for Kolibri's architecture.** The macOS path is a separate MLX port that accepts only `LlamaForCausalLM`/`MistralForCausalLM`. It rebuilds and reloads every layer for every token ([auto_model.py](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/auto_model.py)).
- **Kolibri would stream all 384 experts.** True per-expert streaming exists only for Kimi K3, where experts are an `nn.ModuleList`. Packed 3D expert tensors fall back to whole-layer streaming ([airllm_qwen4_exp.py](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/airllm_qwen4_exp.py)), which means all 384 experts per layer instead of 6.
- **Its per-expert mode has no cache and no batching of reads.** K3's 292 s/token for ~25–30 GB of expert reads per token implies about 0.1 GB/s effective. That points to thousands of tiny synchronous reads, not SSD bandwidth, as the bottleneck.

The project is active again (v4.0.0, 2026-09), but CI covers only releases, and Hacker News readers describe it as "vibe coded" ([HN](https://news.ycombinator.com/item?id=49155378)).

Throughput offloaders such as FlexGen and DeepSpeed ZeRO-Inference stream layers too. They escape the per-token cost only by amortizing each weight pass over large batches. FlexGen reaches 0.69–1.12 tok/s on OPT-175B at effective batch 144–256 ([FlexGen](https://github.com/FMInference/FlexGen)). ZeRO-Inference reaches 7–26 tok/s on Llama-3-70B, but **at batch 96** ([PyTorch blog](https://pytorch.org/blog/deepnvme-affordable-i-o-scaling-for-deep-learning-applications/)). That batch-amortization idea is wrong for Kolibri decode, but it becomes relevant again for prefill (see below).

## Granularity decides everything: a comparison of the approaches

The approaches fall into three groups. The first moves whole layers and needs a batch to amortize the cost. The second moves only what the current token needs (neurons or routed experts) through an application-managed cache. The third avoids moving weights at all, either by computing cold experts where they live or by degrading gracefully on a miss.

Kolibri-1 has 50 layers, 384 routed experts of FFN width 512, top-6 routing and hidden size 2560 ([HF card](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16)). That gives 3 × 2560 × 512 ≈ **3.93 M parameters per expert**: about 4.2 MB at Q8_0 and about 1.7 MB in the repo's IQ3 mix. The second figure matches the repo's measured 510 MB per decoded token with no cache, i.e. 300 expert reads ([expert-policy.md](expert-policy.md)). The table uses these sizes for the "Kolibri bytes" column.

| Approach | Representative systems | Mechanism | Bytes from slow tier per Kolibri token (estimate unless noted) | Measured speed (as reported) | Batch assumption | Maturity |
|---|---|---|---|---|---|---|
| Whole-layer streaming | AirLLM, HF Accelerate disk offload | Load each layer, run it, evict it; no reuse | Whole model: ~83 GB (Q8_0), ≥13–17 s/token at 5–6 GB/s | K3 292 s/token ([PR #316](https://github.com/lyogavin/airllm/pull/316)); OPT-175B 0.01 tok/s via Accelerate ([FlexGen](https://github.com/FMInference/FlexGen)) | 1 | Active, but no Kolibri or MoE support on Mac |
| Batched layer streaming | FlexGen, ZeRO-Inference + DeepNVMe | Same, amortized over a block of sequences | Whole model per batch step | Llama-3-70B 7–26 tok/s aggregate at bs 96 ([PyTorch](https://pytorch.org/blog/deepnvme-affordable-i-o-scaling-for-deep-learning-applications/)) | 96–256 | FlexGen archived 2024; ZeRO maintained |
| Kernel paging | llama.cpp mmap, `--cpu-moe` | Page faults, kernel readahead and eviction | Router-blind; repo Q8_0 measured 101 MiB page-ins/token | Repo Q8_0 6.45 tok/s ([expert-streaming.md](expert-streaming.md)); R1-671B ~2.1 tok/s ([Level1Techs](https://forum.level1techs.com/t/deepseek-r1-671b-backed-by-fast-read-iops/225008)) | 1 | Mainline |
| Neuron-level flash streaming / activation sparsity | LLM in a Flash, PowerInfer-2, TurboSparse, TEAL/CATS | Predict active FFN rows, window cache, row-column bundles | OPT-6.7B: 13.4 → 0.2 GB ([arXiv 2312.11514](https://arxiv.org/html/2312.11514v3)); for Kolibri only the ~9% miss traffic would shrink | 305 ms/token on M2 Metal (OPT-6.7B); 11.68 tok/s Mixtral-47B on a phone ([PowerInfer-2](https://arxiv.org/html/2406.06282v3)) | 1–4 | Paper or research code; needs ReLU-sparse models |
| Per-expert streaming without cache | AirLLM K3 mode | Load the routed experts on demand, evict immediately | ~1.3 GB (Q8_0) + all dense weights | 292 s/token (K3) | 1 | One model only |
| Expert cache, SSD misses | **This repo**, llama.cpp #25294, flash-moe, Metal PoC #23324, mlx-lm #1438 | Resident slot cache; misses `pread` from the file | Repo IQ3 **46 MB measured** at 90.9% hits | Repo 17–18 tok/s IQ3, 13 tok/s Q8_0; flash-moe 4.4 tok/s (397B, M3 Max 48 GB) ([flash-moe](https://github.com/danveloper/flash-moe)) | 1 | Forks and open PRs |
| Expert cache, RAM→VRAM misses | llama.cpp #29887 (merged) | GPU cache of host-RAM experts; misses uploaded over PCIe | 0 from SSD (assumes experts fit in RAM) | 1.57–2.20× decode on RTX 4090/5090 ([PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887)) | ≤32 tokens | Merged 2026-10-07 |
| CPU compute of cold experts | Fiddler, kTransformers, HybriMoE | Compute where the weights live; no transfer | 0 over PCIe; on a Mac the same SSD read either way | kTransformers 1.25–4.09× decode vs Fiddler/llama.cpp ([SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)) | small | Production (kTransformers, SGLang) |
| Prediction / prefetch | Next-layer gate, ProMoE, SeqMoE, Markov | Speculatively load next experts | Adds bytes (repo: +6.8 GB per run) | +3.4 pp hit at depth 1 ([MoE-CORE](https://arxiv.org/html/2610.01950)); nulls in single-stream decode ([WiSP](https://arxiv.org/abs/2606.21868)) | 1 | Mostly papers |
| Lossy miss handling | HOBBIT, BuddyMoE, MoE-CORE AES, AdapMoE | On a miss, use a low-bit copy, substitute or skip | Cuts miss bytes and stalls | HOBBIT ≤1% accuracy loss ([arXiv 2411.01433](https://arxiv.org/html/2411.01433)); BuddyMoE +10% at −4 pts ARC ([arXiv 2511.10054](https://arxiv.org/html/2511.10054)) | 1 | Papers; HOBBIT built on llama.cpp |

The table shows a ladder. Whole-layer streaming at batch 1 moves the whole model per token. An uncached per-expert stream moves about 1.3 GB per token. The repo's cache moves tens of MB. The remaining gap to the fully resident model is no longer about bytes.

Two families look attractive but transfer poorly:

- **Activation sparsity inside experts.** The gain is bounded by the ~9% of expert reads that miss. A per-neuron bundle in a Kolibri expert is about 8 KB at Q8_0, below the 32 KiB that LLM in a Flash needed for good random-read throughput on Mac SSDs ([arXiv 2312.11514](https://arxiv.org/html/2312.11514v3)). TurboSparse is the only method demonstrated inside MoE experts, and it required a dReLU retrofit with continued pretraining ([arXiv 2406.05955](https://arxiv.org/html/2406.05955v1)).
- **CPU compute of cold experts.** It wins on discrete GPUs because it avoids PCIe. On unified memory both paths pay the same SSD read. The repo's own data shows `--cpu-moe` at 11.86 tok/s against the cache's 17.13 at a 16 GiB ceiling ([expert-streaming.md](expert-streaming.md)).

## Nobody has published SSD-streamed Kolibri speeds, and upstream now overlaps

As of 2026-10-22, targeted searches found **no AirLLM, accelerate disk-offload or kTransformers run of Kolibri-1**. The closest efforts are four:

- **colibri `kolibri1` engine.** A correctness-focused port for the colibri SSD-streaming engine. Its draft PR is CPU-only, failed one of 80 logit-tolerance cells, and publishes no tok/s ([colibri PR #2049](https://github.com/JustVugg/colibri/pull/2049)). It found that **40.2% of routed experts in the FP8 release are split across two safetensors shards** ([colibri #1944](https://github.com/JustVugg/colibri/issues/1944)), a hazard for anyone streaming from the original checkpoint rather than a GGUF.
- **vLLM CPU offload.** A vLLM + plugin baseline with 58 GB of UVA CPU offload on an RTX 3090 Ti, used as a numerical reference and not timed.
- **"moe-cache" llama.cpp fork.** A report of ~90 tok/s at 4-bit on an RTX 3090 using an undocumented fork ([HF discussion #3](https://huggingface.co/Aleph-Alpha/Kolibri-1/discussions/3)).
- **Fully resident Mac runs.** Every concrete Mac speed found comes from fully resident 2–6-bit builds. A snippet-only claim of "64 tok/s on a 48 GB Mac through Metal" for a separate llama.cpp port ([kingy.ai](https://kingy.ai/news/kolibri-1-local-hardware-ram-vram-mac/)) probably refers to this repo, whose resident IQ3 measures 60 tok/s.

Aleph Alpha's cards target 2× A100-class GPUs and give no offload guidance ([HF Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1)).

The notes reported three upstream developments. A GitHub check on 2026-10-22 confirmed each:

- **Kolibri support merged.** [PR #30673](https://github.com/ggml-org/llama.cpp/pull/30673) by riteme was merged on 2026-10-17. It uses NeoX RoPE on the sliding layers (window 513), NoPE on the full layers, bias-on-logits selection with unbiased sigmoid weights, and the qwen2 pre-tokenizer. The feature request [#29922](https://github.com/ggml-org/llama.cpp/issues/29922) is still open.
- **`--moe-cache-mib` expert cache merged.** [PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887) by am17an was merged on 2026-10-07, with multi-GPU follow-up [#30112](https://github.com/ggml-org/llama.cpp/pull/30112). It is a port of the qvac-fabric cache. `MUL_MAT_ID` on host-RAM experts runs on the GPU, an SLRU cache shared across groups of layers holds the hot experts, and only misses are uploaded over PCIe. It covers batches of ≤32 tokens. It was benchmarked only on RTX 4090/5090 (1.57–2.20× decode at 72–89% hit). **It does not stream from SSD; it assumes the experts fit in host RAM.** On Apple unified memory there is no host-to-VRAM copy to save, so its value on a Mac is doubtful and untested. It solves a different problem from the repo's cache, which exists because the model exceeds the memory ceiling.
- **`--moe-stream` still open.** [PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294) (opened 2026-07-04) is the real analogue of this repo: a per-layer slot cache, an id-remap op, an async `pread` pool with optional O_DIRECT, decaying-hotness eviction, and wave-partitioned prefill. On GB10 it reports 2.4× decode over mmap + `--cpu-moe`, close to the repo's 2.0× on Q8_0. Two caveats remain unverified here: its auto-sizing reportedly **aborts when `n_expert_used >= 6`** (Kolibri is top-6), and it has had no review from ggerganov or CISC.

Some smaller claims also remain unverified: the b11709 release tag, bartowski's b11716 build, and the CyberNativeAI finding that only 7 of 40 Hub quants loaded on mainline without `--override-kv` fixes ([HF discussion #3](https://huggingface.co/Aleph-Alpha/Kolibri-1/discussions/3)).

## Where the evidence disagrees

**Page cache vs. a custom cache.** flash-moe found the strongest Mac result without any custom cache. On a 48 GB M3 Max the OS page cache gave about a 71% hit rate. Removing its Metal LRU gained 38%. F_NOCACHE gained only 3%, and only on 2-bit. mmap'd expert files were 5× slower than `pread` ([flash-moe](https://github.com/danveloper/flash-moe)). That seems to contradict this repo's design and its 2× win over mmap. The two results are reconcilable once you look at what each compares:

- flash-moe's loser was **mmap**, not buffered `pread`, and its winner was **parallel buffered `pread`**. The repo's loser is likewise mmap via `--cpu-moe`.
- flash-moe's "17.5 GB/s measured" is about 2.5× every independent M3/M4 Max SSD benchmark (~6–7.3 GB/s; [MacDailyNews/The Verge](https://macdailynews.com/2026/03/10/apples-new-m5-max-macbook-pro-delivers-a-huge-ssd-read-write-speed-boost/)). It almost certainly counts page-cache hits.
- The repo found that F_NOCACHE does not bypass pages already resident: the Q8_0 runs read 920–973 GB from disk against about 1074 GB of demand loads ([expert-streaming.md](expert-streaming.md)). mlx-lm #1438 reports the same trap ([mlx-lm #1438](https://github.com/ml-explore/mlx-lm/issues/1438)).

The real disagreement is narrower than it looks. With F_NOCACHE the repo gives up a free second-tier cache in order to cap its footprint at 11 GiB. That is the right choice under a memory ceiling, but it is untested when RAM is spare.

**Smarter policies vs. LRU.** These results conflict:

- FineMoE, SpecMD, SeqMoE and MoE-Beyond report large wins over LRU. They operate at 1–15% cache capacity, with coarse experts, prefill-heavy metrics, or simulators that inflate recency-policy misses by about 27% ([Zhang, arXiv 2608.07911](https://arxiv.org/pdf/2608.07911)).
- WiSP finds prefetching useless in single-stream decode ([arXiv 2606.21868](https://arxiv.org/abs/2606.21868)).
- Zhang finds causal next-use predictors recover none of the gap to Belady.
- The mlx-lm community on 256-expert Qwen3.6 reports cross-layer expert-ID Jaccard at chance (0.017 against 0.016) and three prefetch nulls ([mlx-lm #1438](https://github.com/ml-explore/mlx-lm/issues/1438)).
- The repo's offline study agrees: the best causal policy still loads 2.4–2.5× Belady's bytes, and no predictor passed the gate ([expert-policy.md](expert-policy.md)).

The outlier is flash-moe's M5 Max fork, where temporal prediction raised decode from 12.5 to 16.4 tok/s ([gorroai/flash-moe](https://github.com/gorroai/flash-moe)), at K=4 of 512 experts with heavy page-cache reliance. SLRU is contested as well:

- Issue #20757 reports +8–15 pp over LRU ([llama.cpp #20757](https://github.com/ggml-org/llama.cpp/issues/20757)).
- The Metal slot-pool PoC saw no improvement ([Discussion #23324](https://github.com/ggml-org/llama.cpp/discussions/23324)).
- One vLLM thread found LFU 1.4–2.3× worse than LRU on Apple Silicon ([vLLM #38256](https://github.com/vllm-project/vllm/issues/38256)), while FineMoE says LFU beats LRU.

The repo's own replay settles part of this. LFU and LRU tie in decode (89–94% hits), but **LFU nearly doubles prefill hit rates (27–41% against 17–21% for layer LRU)** ([expert-policy.md](expert-policy.md)). So scan resistance matters for prefill, not decode.

**Conflicting PoC tables.** The same Metal slot-pool PoC reports 22.1 tok/s at 32 slots in the llama.cpp discussion but 10.6 in its ik_llama.cpp cross-post ([Discussion #23324](https://github.com/ggml-org/llama.cpp/discussions/23324); [ik_llama.cpp #1833](https://github.com/ikawrakow/ik_llama.cpp/discussions/1833)). Both report the same **19–21% overhead from the interception mechanism alone**, with 128 slots and no eviction. That overhead number is the one that matters most for this repo.

## The repo's 58 ms token is mostly overhead, not I/O

The Stage D benchmark explains where a cached decode token's time goes ([expert-streaming.md](expert-streaming.md)):

- **Total:** IQ3 at 128 slots decodes at about 58 ms per token.
- **Resident compute:** the resident model needs 16.5 ms. The `-ub 21` resident control still runs at 60.19 tok/s, so the ubatch cap does not explain the gap.
- **Load waits:** the remaps waited on loads for **14 ms per generated token**, reading 46 MB per token at 90.9% hits. That is an effective ~3.3 GB/s during stalls, under the ~6.3 GB/s measured on a base M5 SSD ([Tom's Hardware](https://www.tomshardware.com/laptops/macbooks/m5-macbook-pros-ssd-is-2-5x-faster-on-average-than-last-gen-m4-exceeding-apples-own-claims-m5-achieves-6-000-mb-s-across-both-read-and-write-speeds)). I found no trustworthy M5 Pro measurement.
- **Unexplained:** **about 27 ms per token**. The likeliest cause is the cache's structure. A CPU remap op in every MoE layer forces a scheduler sync before each CPU split: 50 GPU→CPU round trips per token, plus remap and bookkeeping time. That matches the PoC's 19–21% interception overhead. kTransformers attributes much of its gain to removing CPU–GPU synchronisation with CUDA graphs ([SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)).

Two consequences follow:

- **Prediction has little to win.** Even a perfect prefetcher removes at most the 14 ms. The Markov prefetch in fact left misses unchanged (17,315 against 17,424) and read 6.8 GB more per run. Its ~5% p95 gain is the size of effect the literature predicts: MoE-CORE adds 3.4 pp of hit rate from an 89% base ([arXiv 2610.01950](https://arxiv.org/html/2610.01950)).
- **Removing the overhead would roughly double decode.** If the sync overhead went away, decode would land near 16.5 + 14 ≈ 30 ms, or **~33 tok/s**. This is an estimate that the 384-slot control below would confirm or refute.

Prefill is worse:

- **The gap.** The cache prefills at 83 tok/s against 1,110 resident, and a cold 2048-token prompt takes 33 s to its first token.
- **The cause.** The ubatch is capped at 128/6 = 21 tokens, so each small ubatch re-touches most of a layer's experts. The run moved **406 GB of demand loads for 10,240 prompt tokens (~40 MB per prompt token)**.
- **The fix is AirLLM's idea.** With ubatches of 1,000+ tokens, nearly all 384 experts in a layer are needed anyway. Streaming each layer's full expert block once per large ubatch is FlexGen's block schedule applied to experts. That block is about 0.6 GB at IQ3 and 1.6 GB at Q8_0, so double-buffered it costs 1.2–3.2 GB. Per 2048-token ubatch it moves ~29 GB of IQ3 experts, against ~81 GB today: **about 2.8× fewer bytes** (estimate). #25294's wave-partitioned prefill already reported 2–2.5× prefill gains ([PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294)). Zhang estimates that layer-major vs chunk-major prefill ordering changes expert traffic by about two orders of magnitude at long context ([arXiv 2608.07911](https://arxiv.org/pdf/2608.07911)).

## Ranked next steps for the repo

| Rank | Step | Why (evidence) | Expected gain / cost |
|---|---|---|---|
| 1 | **Measure and cut the per-layer sync overhead.** First a 100%-hit control (IQ3 with 384 slots, 0 GiB ceiling) to isolate interception cost from load waits. Then move the id→slot remap onto the GPU, or fuse it so no CPU split per layer is needed: keep slot tables device-visible and let the CPU only service miss queues. | ~27 ms of 58 ms is neither compute nor I/O; the Metal PoC's 19–21% interception overhead; kTransformers' sync-removal gains | Up to ~2× decode (estimate, to ~33 tok/s); medium–high effort in ggml/Metal |
| 2 | **Rebuild prefill.** Lift the 21-token ubatch cap with a prefill-only path that streams each layer's whole expert block once per large ubatch (double-buffered), or adopt #25294's wave partitioning. Use LFU or another scan-resistant admission for prefill. | 406 GB of loads per 10k prompt tokens; 83 vs 1,110 tok/s; #25294's 2–2.5×; repo's LFU prefill hits 27–41% vs 17–21% | Est. 2–3× fewer prefill bytes and much lower TTFT; medium effort |
| 3 | **Rebase onto upstream #30673 and compare the port.** Diff routing (bias-on-logits, unbiased sigmoid, no renorm), RoPE/NoPE split, window 513 and pre-tokenizer against the fork, using the existing bit-identity and KLD harnesses. Then shape the cache as a backend of the `llama_moe_cache` abstraction or #25294's flags rather than a third implementation. Test whether `--moe-cache-mib` does anything on Metal, and check #25294's top-6 auto-size abort. | Upstream merged Kolibri and an SLRU expert cache; #25294 reviewers asked for convergence and pluggable eviction | Removes patch-carrying cost; upstreaming path; low–medium effort |
| 4 | **KLD-gated lossy miss path.** For misses at sigmoid rank ≥4 with low weight, skip them, or read a low-bit (IQ1/IQ2) copy from a side-file, which is roughly half the bytes of an IQ3 expert. Keeping a low-bit copy of *all* experts resident would cost about 15–19 GB (whole-model IQ1_S is 17.6 GB; [bartowski](https://huggingface.co/bartowski/Kolibri-1-GGUF)), so that only fits without a memory ceiling. Try skipping first. | HOBBIT ≤1% accuracy loss; MoE-CORE substitution 92.4% → 99.97% availability; SpecMD says wide MoEs tolerate drops; BuddyMoE shows it is not free | Attacks the 14 ms wait and the p95 tail directly; quality must be measured, since no data exists for a ~3-bit primary |
| 5 | **Page-cache A/B.** A buffered-`pread` arm (no F_NOCACHE) when no memory ceiling is set, reporting page-cache residency separately from `phys_footprint`. | flash-moe's "trust the OS" (+38% without a custom LRU, F_NOCACHE +3%); F_NOCACHE already fails to bypass resident pages | Cheap; possibly higher Q8_0 hit rates on a 48 GB Mac with spare RAM |
| 6 | **Coalesce gate/up/down (and scales) into one extent** via a repacked side-file, and issue each layer's misses concurrently. Run a Go `pread` sweep (F_NOCACHE, 1–4 MB chunks, queue depth 1–16) to get the M5 Pro's real curve. | Three `pread`s per expert today; coalescing gave +14–32% in mlx-lm at 1–4 MB slices; LLM in a Flash needed deep queues | Bounded by the 14 ms decode wait; larger effect in prefill, where load waits are roughly 40% of prefill time (51.6 s of waits against ~123 s of prefill per run, estimated from the warm prompt rate) |
| 7 | **One hidden-state predictor experiment.** Record residual streams and test layer L+1's and L+2's sigmoid router on layer L's hidden state (recall@6/12). | Stage B used expert IDs only; the next-layer gate is 90–96% accurate on Mixtral but 66.9% on Qwen2-MoE ([ProMoE](https://arxiv.org/html/2410.22134)); untested on sigmoid top-6 | Closes the predictor question; expected ceiling small (+3 pp) |
| 8 | **Publish the negative predictor result** with Zhang's checklist (event-atomic replay, decode-only hit rate, per-step union/capacity, Belady gap). | No published study covers 384 experts × top-6, sigmoid routing and unified memory | Low effort; a genuine contribution |

Several things should be explicitly ruled out:

- AirLLM-style layer streaming for decode.
- CPU→GPU offload flags, which have nothing to offload on unified memory.
- Intra-expert activation sparsity, which needs retraining, has 8 KB bundles that are too small, and is bounded by 9% miss traffic.
- Cache-aware rerouting, which costs up to 3.4% perplexity ([arXiv 2608.18261](https://arxiv.org/abs/2608.18261)).
- Raising `iogpu.wired_limit_mb`. Wired-memory overcommit panics macOS instead of OOM-killing the process ([mlx-lm #883](https://github.com/ml-explore/mlx-lm/issues/883)), which the repo has already experienced with a watchdog reboot.

## Conclusion

The AirLLM question has a clear answer: whole-layer streaming at batch 1 is a memory-minimizing pager, not an inference strategy, and Kolibri's 4.4%-active sparsity makes it doubly wasteful. The more interesting result is what the repo's own numbers say about the frontier. The expert cache has already shrunk SSD traffic to the point where the SSD is no longer the main cost. Better predictors and policies therefore have little left to recover, which explains the Stage B nulls more simply than "predictors are hard". The remaining factor of three to resident speed is mostly synchronisation structure in decode and ubatch structure in prefill. Both are problems the field has solved elsewhere: CUDA graphs in kTransformers, and block schedules in FlexGen and #25294.

The strategic implication is to converge rather than diverge. Upstream now carries Kolibri and a merged expert cache, and has an open SSD-streaming PR that matches this repo's design closely. The repo has the strongest public Kolibri evidence base: bit-identity gates, a KLD harness, Belady-bounded policy replays, and the only SSD-streamed Kolibri speeds found. Contributing that evidence and the Metal path to #25294 is likely to matter more than another round of predictor research. Lossy miss handling is the one research direction still worth opening, and only behind the KLD gate.
