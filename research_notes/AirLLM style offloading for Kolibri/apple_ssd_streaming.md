# SSD-streamed MoE inference on Apple Silicon (2024 to Oct 2026)

Research date: 2026-10-22. All numbers are as reported by the cited source (mostly self-reported by project authors, not independently reproduced). Where a page was only reached via a secondary write-up, that is noted.

## 1. Known runs of over-RAM MoE models on Macs: tool, quant, machine, tok/s

### Takeaway
The strongest Mac result is flash-moe (custom C/Metal engine, parallel `pread` of only the K active experts): Qwen3.5-397B-A17B at 4-bit runs at 4.4 tok/s on a 48 GB M3 Max and at about 20-21 tok/s on a 128 GB M5 Max after optimization. Stock llama.cpp mmap streaming of a model larger than RAM is consistently the slow baseline, at roughly 1 to 6 tok/s. Expert-cache prototypes in llama.cpp and MLX land at about 2 to 5x over mmap, which matches our ~2x (13 vs 6.5 tok/s on Q8_0). I found no measured Mac runs of DeepSeek-V3/R1, Kimi K2, GLM-4.x or Llama-4 Maverick that stream from SSD with a credible tok/s figure.

### Cited Findings

**flash-moe (danveloper), original M3 Max run (March 2026)**
- Model: Qwen3.5-397B-A17B, 60 layers (45 GatedDeltaNet, 15 full attention), 512 experts/layer, K=4 active (the model default is 10) plus 1 shared expert, hidden 4096. 4-bit experts take 209 GB on disk at about 6.75 MB per expert. 2-bit takes 120 GB. Non-expert weights are 5.5 GB, mmapped — [danveloper/flash-moe](https://github.com/danveloper/flash-moe)
- Machine: MacBook Pro M3 Max, 48 GB, 1 TB SSD, macOS 26.2. The README reports "17.5 GB/s sequential read (measured)" — [danveloper/flash-moe](https://github.com/danveloper/flash-moe)
- Throughput: 4-bit with the FMA kernel is 4.36 tok/s (the current best, with full tool calling). The 4-bit baseline is 3.90. 2-bit "trust OS" is 5.74, but it breaks tool calling (malformed JSON). The 2-bit warm-cache peak for a single token is 7.05 — [danveloper/flash-moe](https://github.com/danveloper/flash-moe)
- Per-layer time is 4.28 ms on 4-bit. Of that, the parallel `pread` of the K=4 experts (~27 MB/layer) takes 2.41 ms, and attention plus delta-net on the GPU takes 1.22 ms. The SSD is the bottleneck — [danveloper/flash-moe](https://github.com/danveloper/flash-moe)
- Simon Willison relayed ~5.7 tok/s using 5.5 GB of active memory, streaming at ~17 GB/s (Mar 2026). That is the 2-bit figure — [Simon Willison on X](https://x.com/simonw/status/2034365182751973541)
- Quality: K was cut from 10 to 4 to fit the SSD budget. The author says the biggest quality drop came at K=3 — [buildmvpfast summary](https://www.buildmvpfast.com/blog/flash-moe-weight-streaming-benchmarks-quality-tradeoffs-2026) (secondary)

**flash-moe fork / autoresearch on M5 Max 128 GB (March to April 2026)**
- Exp1 baseline (4-bit, `cache-io-split 4`) ran at 12.5 tok/s. Temporal expert prediction raised it to 16.4. Q3 GGUF experts plus temporal prediction reached 18.7. Command-buffer pre-encoding and Q/K/V fusion reached 20.34. After an erroneous `--k 10` was removed, the April 2026 figure is 21.48 tok/s — [gorroai/flash-moe](https://github.com/gorroai/flash-moe)
- The fork also lists, per format: MLX 4-bit at 9.5 tok/s, Unsloth Q3 GGUF experts (5.44 MB/expert, 1.30 GB of expert I/O per token) at 12.9 tok/s, and MLX 2-bit (3.75 MB/expert) at 14.5 tok/s — [gorroai/flash-moe](https://github.com/gorroai/flash-moe)
- A secondary tracker reports "11.5 tok/s sustained on an M4 Mac mini with 16 GB". I could not open the source page (HTTP 503), so this is unverified — [tomevault tracker (search snippet)](https://tomevault.io/tome/tayoun/flash-moe)

**llama.cpp-based expert caches on Mac**
- kisasexypantera94 posted a Metal PoC (May 19, 2026): a per-layer slot pool in Metal shared memory, a CPU sidecar doing LRU and `pread`, and an `MTLSharedEvent` handshake. It uses Qwen3-30B-A3B Q6_K on an M3 Pro 36 GB. The llama.cpp discussion lists vanilla at 38.1 tok/s, 128 slots at 30, 64 slots at 27.5 (11.1 GiB saved), 32 slots at 22.1 and 8 slots at 12. An M1 Pro 16 GB reached about 13 tok/s — [llama.cpp Discussion #23324](https://github.com/ggml-org/llama.cpp/discussions/23324)
  - The ik_llama.cpp cross-post of the same PoC gives different numbers: vanilla 39.9, 128 slots 32.4, 64 slots 26.5, 32 slots 10.6, 8 slots 4.7, and M1 Pro 16 GB at ~9.5 tok/s. The two tables conflict below 64 slots, and the reason is unclear (probably different revisions) — [ik_llama.cpp Discussion #1833](https://github.com/ikawrakow/ik_llama.cpp/discussions/1833)
  - Both versions report about a 19-21% overhead from the interception mechanism alone (128 slots, no eviction, vs vanilla) — same sources
- Jakevin/moe-offload ran on a Mac mini M4 Pro 64 GB (2026). Nemotron 3.5 Bolt 74B-A9B (240 experts, top-9) at IQ2_XXS on llama.cpp Metal reaches 11.07 / 11.05 / 11.28 tok/s with 4 / 8 / 12 GB caches (hit rates 86.1 / 93.8 / 97.0%). The resident baseline is 13.9 tok/s at 20.8 GiB. On a 16 GB M4 the 4 GB cache gave 4.39 tok/s, and the 8 GB cache thrashed — [Jakevin/moe-offload](https://github.com/Jakevin/moe-offload)
- Same repo, MLX path, "Qwen3.8-Flash-Next" 125B MoE (512 experts, top-10) at Q2 with a 20% cache (~7.4 GB): 8.51 tok/s with Python `pread` (84.3% hit). Native pthread `pread` gave 9.17 tok/s. Router prefetch raised it to 9.73 tok/s (94.7% hit) — [Jakevin/moe-offload](https://github.com/Jakevin/moe-offload)
- doramirdor/mbolt ran Qwen3-Next-80B-A3B IQ3_XXS (33 GB, larger than available RAM) on an M5 Pro with CPU experts. Stock llama.cpp gave 5.80 tok/s, stock layout plus an explicit-read prefetcher 7.60, and a co-activation-interleaved file plus prefetcher 9.00 (1.55x). Under a 32 GB squeeze it was 9.80 vs 6.00. The mmap fault path is "layout-blind" (parity) — [doramirdor/mbolt](https://github.com/doramirdor/mbolt)
- jerryjokesalot used a CPU-only prototype on a 16 GB M1 MacBook. qwen3:30b-a3b under Ollama ran at 1.7 tok/s. Expert-aware reads (31 MB/layer) gave 4.7 tok/s, against 0.9 tok/s for full-layer reads — [llama.cpp Discussion #27149](https://github.com/ggml-org/llama.cpp/discussions/27149)
- A non-Mac comparison for the main llama.cpp PR #25294 used a GB10 (128 GB unified, PCIe 4.0 SSD) with GLM-5.2-UD-Q2_K_XL (~254 GB). Decode was ~1.83 tok/s at 64 slots (73% hit) and ~2.20 tok/s at 90 slots (79% hit). Against mmap plus `--cache-cpu-moe`, decode was 2.08 vs 0.87 tok/s (2.4x) and prefill 5.58 vs 1.06 tok/s (5.3x) — [llama.cpp PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294)

**MLX-based offload**
- In mlx-lm #1438 (June to July 2026), mabaeyens ran Qwen3.6-35B-A3B 4-bit on a 32 GB Mac. Offload off gave 57.1 tok/s; offload on with 0.3 resident gave 10.8 tok/s, and peak memory fell from 19.3 to 7.3 GB. The 8-bit model (over DRAM) gave 6.6 tok/s at 0.3 resident — [mlx-lm issue #1438](https://github.com/ml-explore/mlx-lm/issues/1438)
- In the same thread, lBroth ran Qwen3-235B-A22B-4bit on a 48 GB M5 Pro. Read coalescing gave +32.3% decode and -26.3% TTFT, cutting preads per token from 2394 to 798. The absolute tok/s did not appear in the part of the thread I read — [mlx-lm issue #1438](https://github.com/ml-explore/mlx-lm/issues/1438)
- gpt-oss-120b-MXFP4 (~4 MB slices) showed a 1.29x I/O gain from coalescing and +9.7% decode. No absolute tok/s was given — [doramirdor/mbolt cross-check table](https://github.com/doramirdor/mbolt)

**Very large models**
- Deltafin fork (argonautlabsai/deltafin, from gavamedia/deltafin) runs Kimi K3 2.8T on an M5 Max 128 GB with up to four NVMe SSDs, using layer streaming with all 16 experts. Decode is 1.00 tok/s with a draft model and 0.92 without. TTFT is 6.3 min for a 512-token prompt. Measured 2026-09-08, single cold runs. Upstream reportedly managed 0.29 tok/s on an M1 Max — [Context Studios blog](https://www.contextstudios.ai/de/blog/deltafin-2-8t-kimi-k3-von-vier-ssds-auf-dem-macbook-layer-streaming-als)
- DeepSeek-R1: I found no measured Mac SSD-streaming figure. The only number is a non-Mac Linux result of ~2.13 tok/s on a 96 GB RAM gaming PC, using llama.cpp mmap from NVMe with an Unsloth dynamic quant. With swap instead of mmap it fell to 0.3 tok/s and was unstable (Jan-Feb 2025) — [Level1Techs forum](https://forum.level1techs.com/t/deepseek-r1-671b-backed-by-fast-read-iops/225008). An "~20 tok/s on 128 GB M4 Max" claim in the same thread came from LLM-generated analysis, not a measurement — same source. The 131 GB UD-IQ1_S failed to load on a 128 GB M4 Mac in one anecdote — [lilys.ai note (secondary, Italian)](https://lilys.ai/it/notes/bitcoin-bear-market-20251214/deepseek-apple-silicon-benchmarks)

### Inferences
- The hard ceiling on a Mac is roughly "bytes missed per token / effective SSD GB/s". flash-moe's M5 Max 20 tok/s with 1.3-1.7 GB of expert I/O per token is only possible because a large share of reads hit the page cache or are hidden by prediction (see Q5).
- Our Kolibri numbers fit the community pattern. Q8_0 via the cache is 2x mmap/`--cpu-moe`, the same as PR #25294's 2.4x decode gain. The IQ3 cache at 17-18 tok/s vs 60 tok/s resident (~0.3x) is in line with Qwen3.6-35B-A3B MLX (10.8 vs 57.1, ~0.19x at 0.3 resident) and the 32-slot Metal PoC (22.1 vs 38.1, ~0.58x).
- With 384 experts at 128 slots (33%), Kolibri's resident fraction is similar to the 0.3 resident setting in mlx-lm #1438 and to the 64/128 knee in the Metal PoC.

### Gaps
- I found no measured Mac SSD-streaming runs for DeepSeek-V3/R1, Kimi K2, GLM-4.x, Llama-4 Maverick or Qwen3-480B, nor any Awni Hannun, ggerganov or exo (Alex Cheema) posts with SSD-streaming numbers. X/Twitter content was not searchable beyond one Simon Willison post.
- The M4 Mac mini 16 GB flash-moe figure (11.5 tok/s) is unverified.
- The model names "Qwen3.8-Flash-Next" and "Nemotron 3.5 Bolt" come only from the Jakevin repo; I did not independently confirm them.

## 2. Apple SSD read bandwidth in practice

### Takeaway
Measured sequential read on the internal SSD is roughly 2-3 GB/s on base M4 (512 GB), about 6-7 GB/s on M1 Max through M4 Max and the base M5, and 12.7-13.6 GB/s on the M5 Max (4 TB), against Apple's claim of 14.5 GB/s. I found no clean M5 Pro measurement. Effective random-read throughput for parallel MB-sized `pread`s is close to sequential when there are enough requests in flight. The best Mac data point is flash-moe: 4 parallel reads of 6.75 MB (~27 MB) took 2.41 ms, which is about 11 GB/s effective, possibly helped by the page cache.

### Cited Findings
- On M1 Max (1 TB), a 1 GiB linear read of an uncached file exceeded 6 GiB/s. Random-read throughput rises with chunk size and thread count, so sparse loading needs "32 KiB or larger random reads across multiple threads". The experiments used 32 threads, F_NOCACHE via `fcntl()`, and `purge` before runs. Dense load ran at 6.10 GB/s, sparse reads without bundling at 1.25 GB/s, and sparse reads with bundling at 2.25 GB/s — [Apple, "LLM in a flash" (arXiv 2312.11514v3)](https://arxiv.org/html/2312.11514v3)
- Base M5 MacBook Pro 14" (Oct 2025): Blackmagic on 512 GB measured 6,323 MB/s read, against 2,031 MB/s for the M4 — [Tom's Hardware](https://www.tomshardware.com/laptops/macbooks/m5-macbook-pros-ssd-is-2-5x-faster-on-average-than-last-gen-m4-exceeding-apples-own-claims-m5-achieves-6-000-mb-s-across-both-read-and-write-speeds). A 1 TB unit measured 6,520 MB/s read — [PetaPixel M5 review](https://petapixel.com/2025/10/21/apple-macbook-pro-with-m5-review-the-best-laptop-for-photographers/)
- Base M-chips route 4 PCIe lanes to the SSD and Pro/Max/Ultra route 8 — [TechSpot (Max Tech summary)](https://www.techspot.com/community/topics/early-tests-show-the-m5-macbook-pros-ssd-is-about-2-5x-faster-than-the-m4.295221/)
- M5 Max MacBook Pro (March 2026): Apple claims up to 14.5 GB/s. The Verge measured 13.6 GB/s sustained read on a 4 TB unit, 86% faster than its 4 TB M4 Max (which implies ~7.3 GB/s for the M4 Max) — [MacDailyNews quoting The Verge](https://macdailynews.com/2026/03/10/apples-new-m5-max-macbook-pro-delivers-a-huge-ssd-read-write-speed-boost/). PetaPixel's Blackmagic run on a 4 TB M5 Max gave 12,695 MB/s read — [PetaPixel M5 Max review](https://petapixel.com/2026/03/13/macbook-pro-with-m5-max-review-this-is-better-than-your-desktop/)
- M5 Pro: one review claims "12 GB/s" but reports ~1,200 MB/s measured, which is internally inconsistent — [Irish Examiner](https://www.irishexaminer.com/business/technology/arid-41848644.html)
- flash-moe on M3 Max: 4 experts of ~6.75 MB (~27 MB per layer) were read with parallel `pread` through GCD dispatch groups in 2.41 ms per layer. The README states 17.5 GB/s sequential read (measured) — [danveloper/flash-moe](https://github.com/danveloper/flash-moe)
- On the M5 Max fork, 4-bit expert I/O took 0.74 ms for 27.0 MB per layer and Q3 took 0.44 ms for 21.8 MB per layer — [gorroai/flash-moe](https://github.com/gorroai/flash-moe)
- Sequential layer reads measured 2.97 GB/s on a 16 GB M1 MacBook. A double-buffered pipeline reached 6.06 GB/s. Re-laying out experts contiguously cut pages per expert from 6,912 to 192 — [llama.cpp Discussion #27149](https://github.com/ggml-org/llama.cpp/discussions/27149)
- On M5 Pro, an explicit-read prefetcher raised the average read size from 450 KB to 1.5 MB (28.6k to 8.5k reads for ~13 GB) and gave +31-52% decode over mmap faults — [doramirdor/mbolt](https://github.com/doramirdor/mbolt)
- mbolt's MLX slice-size model assumes BW ≈ 6 GB/s and t_op ≈ 100 µs per read operation (model parameters, not measurements) — [doramirdor/mbolt](https://github.com/doramirdor/mbolt)
- 4K random I/O: an M3 MacBook Air measured 37 MB/s on random 4K writes at QD1 (vs 185 MB/s on a Snapdragon X Elite). The poster suspects the 16K page size. This is a write test, not reads — [TechBoards thread](https://techboards.net/threads/what-is-going-on-with-random-writes-on-apple-silicon.4706)

### Inferences
- flash-moe's "17.5 GB/s measured" on an M3 Max is about 2.5x every independent Blackmagic or Verge number for M3/M4 Max (~6-7.3 GB/s). It almost certainly includes page-cache hits (the README reports a ~71% OS page-cache hit rate), so it should not be read as raw NAND bandwidth. On the M5 Max fork, 27 MB in 0.74 ms is about 36 GB/s, which is also only possible with cache hits.
- On an M5 Pro, assume roughly 6-14 GB/s for cold parallel reads of 1-6 MB chunks at queue depth ≥4, pending our own fio or `pread` sweep. The M5 Max 4 TB number is the upper bound, and capacity matters (more NAND dies means more parallelism).
- Small reads are dominated by per-operation latency (~100 µs assumed). Coalescing gate/up/down of one expert into one contiguous read is worth +14-32% (mlx-lm #1438) when reads are scattered.

### Gaps
- I found no published fio or `pread` sweep of random 1-6 MB reads vs queue depth on M3/M4/M5 internal SSDs, and no 4K random-read IOPS for M-series internal SSDs.
- I found no M5 Pro (48 GB, 1 TB / 2 TB) SSD measurement I would trust.
- No source directly compares mmap page-in throughput with `pread` on macOS beyond flash-moe's "mmap expert files: -5x" and mbolt's "mmap fault path is layout-blind".

## 3. macOS-specific issues: page cache, wired memory, Metal buffers, panics, SSD wear

### Takeaway
Wired memory, which Metal needs for GPU-resident buffers, cannot be compressed or swapped. When it plus the KV cache grows near physical RAM, macOS has repeatedly panicked in the IOGPU driver instead of killing the process. Raising `iogpu.wired_limit_mb` does not prevent this and may make it worse. The OS page cache plus parallel `pread` is a surprisingly strong expert cache on macOS. Flash-moe found every custom cache it tried slower than "trust the OS", and F_NOCACHE gave only +3% (2-bit only). Reads do not count against TBW, but read disturb can cause background rewrites on dense 3D NAND. That is a minor and not Mac-specific concern.

### Cited Findings
**Wired memory and panics**
- mlx-lm issue #883 (Feb 2026, Mac Studio M3 Ultra 96 GB, macOS 26.3) reports a kernel panic `"completeMemory() prepare count underflow" @IOGPUMemory.cpp:550` as wired memory reached ~80 GB with ~0.01 GB free. The server wires about 75% of RAM through `mx.set_wired_limit` and the KV cache was unbounded. Maintainer angeloskath said it was "probably related to the wired limit being too high". The issue was closed via PR #906, but panics recurred on macOS 26.4 ("nine panics in six days", filed as ml-explore/mlx#3346). A second panic string was `"Memory object unexpectedly not found in fPendingMemorySet"` at `IOGPUGroupMemory.cpp:219` — [mlx-lm issue #883](https://github.com/ml-explore/mlx-lm/issues/883)
- llama.cpp issue #19825 (Feb 2026, M4 Pro 24 GB, macOS 26.2) reports system-wide kernel panics, page-fault storms and audio distortion when MoE models exceed RAM. The reporter calls it a "swap death spiral" as routing switches experts. `sudo sysctl iogpu.wired_limit_mb=22000` did not help. ik_llama.cpp `--cpu-moe` ran Qwen 30B IQ4_KSS at 35+ tok/s. The issue was closed with no maintainer comment — [llama.cpp issue #19825](https://github.com/ggml-org/llama.cpp/issues/19825)
- One blogger reports that raising `iogpu.wired_limit_mb` made crashes worse, because wired memory cannot be reclaimed (single-source reasoning). The default Metal working-set limit is about 75% of RAM — [Hannecke, "How my local coding agent crashed my Mac"](https://medium.com/@michael.hannecke/how-my-local-coding-agent-crashed-my-mac-and-what-i-learned-about-mlx-memory-management-e0cbad01553c); [Hannecke, "Tuning llama.cpp on Apple Silicon"](https://medium.com/@michael.hannecke/tuning-llama-cpp-on-apple-silicon-843f37a6c3dc)
- The Metal expert-pool PoC argues that Metal must wire buffers before execution, so OS paging through mmap is not enough on Apple. It requires `--no-mmap` and `--no-warmup` (to avoid OOM) and uses `MADV_FREE` to release CPU pages. Apple's CPU_REPACK layout is incompatible with reading raw GGUF bytes, so REPACK buffer types must be skipped. A ubatch may not request more unique experts than there are slots (`ub * n_expert_used <= n_slots`). All pools are allocated upfront, which causes a warm-up memory spike — [ik_llama.cpp Discussion #1833](https://github.com/ikawrakow/ik_llama.cpp/discussions/1833); [llama.cpp Discussion #23324](https://github.com/ggml-org/llama.cpp/discussions/23324)

**Page cache, F_NOCACHE, mmap, prefetch hints**
- flash-moe uses no custom cache. The OS page cache (~35 GB on a 48 GB machine) gave a ~71% hit rate. Dropping its Metal LRU cache gave +38%, F_NOCACHE gave +3% (2-bit only), and F_RDADVISE prefetch netted 0% because SSD DMA slowed GPU work by 73%. `dispatch_io` was -70% and mmap of expert files was -5x. LZ4-compressed experts were -13%, and expert file clustering gave 0%. The author notes that SSD DMA and GPU compute share the memory controller — [danveloper/flash-moe](https://github.com/danveloper/flash-moe)
- PR #25294 reads through O_DIRECT, with a probe and a fallback to buffered reads. A tester (darksideX1, Linux) found `--moe-stream-direct` slower than buffered reads with a warm cache — [llama.cpp PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294)
- Apple's "LLM in a flash" used F_NOCACHE plus `purge` as a conservative lower bound — [arXiv 2312.11514v3](https://arxiv.org/html/2312.11514v3)

**`newBufferWithBytesNoCopy` / host-pointer Metal buffers**
- No source in this sweep discusses `newBufferWithBytesNoCopy` for expert streaming. The Metal PoC uses a pool tensor in Metal shared memory instead — [ik_llama.cpp Discussion #1833](https://github.com/ikawrakow/ik_llama.cpp/discussions/1833)

**SSD wear from reads**
- TBW and host-write counters track host writes only. Read disturb, however, causes background rewrites: a KAIST study found that 3D TLC/QLC SSDs "can be worn-out by low read-only workloads within their warranty period" because of read-disturb-induced rewrites — [KAIST DSpace](https://dspace.kaist.ac.kr/handle/10203/286383?mode=full)

### Inferences
- Our watchdog panic fits the pattern documented in mlx-lm #883 and llama.cpp #19825: wired plus Metal allocations near physical RAM make the IOGPU driver fail without a clean OOM kill. The practical guidance is to keep total wired memory (pool plus KV plus compute buffers) well below RAM, leaving ~8-16 GB of headroom, and not to depend on raising `iogpu.wired_limit_mb`.
- On macOS, F_NOCACHE `pread` into a wired pool (our design) and buffered `pread` with the page cache as a second-tier cache (flash-moe) are both valid. Ours trades a smaller footprint (11 GiB) for losing the free page-cache tier. A hybrid is worth an A/B test: drop F_NOCACHE so misses can hit the page cache when spare RAM exists. flash-moe saw only +3% from F_NOCACHE, and only on 2-bit.
- Wear from reads is negligible for a laptop doing inference. The KAIST effect concerns sustained heavy read loads on high-density NAND, and nothing Apple-specific was found.

### Gaps
- I found no Apple documentation on how wired-memory overcommit interacts with the GPU watchdog, and no source on unified-memory compression (the WKdm compressor) for streamed experts.
- I found no measurements of `newBufferWithBytesNoCopy` vs shared-storage buffers for streaming.
- I found no Apple-specific SSD read-disturb data.

## 4. Expert-cache / SSD-offload patches for llama.cpp and MLX, and their status

### Takeaway
As of 2026-10-22 nothing is merged upstream in llama.cpp or mlx-lm for SSD expert streaming. The main llama.cpp PR (#25294, `--moe-stream`) has been open since July 2026 with no maintainer review. Several discussions and forks report Mac results, and mlx-lm has a small opt-in primitive PR (#1588), also unmerged as far as I saw.

### Cited Findings
- **llama.cpp PR #25294** "llama : stream MoE routed experts from disk" (freedomljc), opened 2026-07-04, still open as of a 2026-10-19 comment. It keeps a per-layer device cache of `n_slots` expert slabs and uses a CPU id-remap op after top-k. Misses are demand-loaded by an async `pread` thread pool. Eviction uses decaying hotness with an LRU tiebreak. It supports O_DIRECT, wave-partitioned prefill, and auto-disables mmap. Flags are `--moe-stream`, `--moe-stream-cache <NG|Ns>`, `--moe-stream-io-threads` and `--moe-stream-direct`. It was validated on OLMoE (CPU and Metal) and GLM-5.2 (CUDA), and is "primarily for unified memory PCs such as Mac and DGX Spark". Review issues found so far:
  - Auto-sizing aborts when `n_expert_used >= 6` (suggested floor: 3×n_expert_used).
  - The byte-identical-output claim was retracted (2026-09-06).
  - A bot flagged it as too large, and it needs a rebase.
  - No review from ggerganov or CISC.

  — [llama.cpp PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294)
- **llama.cpp issue #20757**: a two-tier GPU+RAM expert cache with pluggable eviction (SLRU with admission filter, which gave +8-15 pp hit rate over LRU). A Python PoC of gpt-oss-120b on an 8 GB GPU reached 12-14 tok/s at steady state with ~98-100% hit, against 0.5-1 tok/s without the cache. Opened 2026-03-19 and closed by the author on 2026-04-08. Related PRs are #27861 (`--moe-expert-cache N`, GPU-resident LRU), #26563, #26824 and RFC #28248 (RTX 4090: 11.55 to 21.21 tok/s with a 64-slot pool) — [llama.cpp issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757)
- **llama.cpp Discussion #23324** (Metal slot pool PoC, 2026-05-19): open with no maintainer response. Code is in the author's fork PR (kisasexypantera94/llama.cpp #2). The SLRU variant from #20757 showed no improvement over LRU here — [llama.cpp Discussion #23324](https://github.com/ggml-org/llama.cpp/discussions/23324). It was cross-posted to ik_llama.cpp with no reply — [ik_llama.cpp Discussion #1833](https://github.com/ikawrakow/ik_llama.cpp/discussions/1833)
- **llama.cpp Discussion #27149** (expert-aware SSD streaming for CPU-only 8-16 GB machines, 2026-08-15): proposes `ggml_backend_buffer_ssd` with double-buffered I/O. Text-calibrated pinning gave a 42% hit rate, and calibrated pinning plus LRU gave 56%. No maintainer engagement; prototype repo "tinygiant" — [llama.cpp Discussion #27149](https://github.com/ggml-org/llama.cpp/discussions/27149)
- **mlx-lm issue #1438** (2026-06-27, open): a request for MoE SSD streaming. The community findings summary (v1.5.1) recommends reactive fetch with plain per-layer LRU. It advises skipping predictive prefetch, expert pinning, and coalescing for blocks of 19 MB or more, and sizing the resident fraction to the table/RAM ratio. **mlx-lm PR #1588** (mabaeyens) adds an opt-in `enable_offload` to `SwitchLinear` / `QuantizedSwitchLinear`. No maintainer response appeared in the part of the thread I read. Out-of-tree projects include mu-hashni/mlx-moe, SharpAI/SwiftLM (Swift MLX with SSD expert streaming), mabaeyens/mira-core, huckiyang/mlx-moe-offload and PhilipJohnBasile/iliria (a C/Metal GLM-5.2 NVMe streamer) — [mlx-lm issue #1438](https://github.com/ml-explore/mlx-lm/issues/1438)
- **Other**: Jakevin/moe-offload (MLX plus llama.cpp, SSD plus per-layer LRU, pinned llama.cpp commit 620d222) — [GitHub](https://github.com/Jakevin/moe-offload). doramirdor/mbolt (GGUF co-activation re-layout plus prefetcher) — [GitHub](https://github.com/doramirdor/mbolt). Ollama PR #12333 only exposes CPU offload of MoE weights and does not stream from SSD — [ollama PR #12333](https://github.com/ollama/ollama/pull/12333)

### Inferences
- Our design (128/384 slots per layer, F_NOCACHE `pread`, 4 I/O threads) is close to PR #25294 and the #23324 Metal PoC. The PR #25294 bug with `n_expert_used >= 6` is directly relevant to Kolibri's Top-6.
- Community consensus by Oct 2026 is that plain per-layer LRU is hard to beat. SLRU, pinning and predictive prefetch each show mixed or null results, with flash-moe temporal prediction on the M5 Max (+31-55%) as the notable exception. Coalescing each expert into one contiguous read is the reliable win.

### Gaps
- I could not confirm the merge status of mlx-lm PR #1588 or llama.cpp PR #27861.
- I found no public statement from ggerganov or Awni Hannun on the direction of SSD streaming.

## 5. Theoretical throughput: bytes per token vs SSD GB/s, and reported hit rates

### Takeaway
Decode time per token is roughly (1 − hit_rate) × expert_bytes_per_token / SSD_GBps + compute time, and decode tok/s is at most the reciprocal of that. Reported hit rates for per-layer LRU at about 20-35% resident are ~70-95%, which is why measured speeds land 2-5x above a zero-hit SSD ceiling.

### Cited Findings
- flash-moe states its own ceiling: 2-bit needs 943 MB/token, and 943 MB ÷ 17.5 GB/s ≈ 18.6 tok/s. The 4-bit build exceeded that at 20.34 and 21.48 tok/s. With perfect prediction ("oracle") the ceiling is 29.3 tok/s. Expert I/O per token is 1.62 GB for 4-bit and 1.30 GB for Q3, and on the M5 Max it takes 44.7 ms (47% of decode) and 26.7 ms (35%) respectively — [gorroai/flash-moe](https://github.com/gorroai/flash-moe)
- Hit rates reported:
  - flash-moe OS page cache: ~71% on a 48 GB M3 Max — [danveloper/flash-moe](https://github.com/danveloper/flash-moe)
  - flash-moe temporal predictor: ~24-25% hit rate at K=4; cross-token routing correlation 27%; cross-layer prediction near zero; MLP router predictor 31% accuracy — [gorroai/flash-moe](https://github.com/gorroai/flash-moe)
  - Nemotron Bolt (top-9 of 240) at 4/8/12 GB caches: 86.1/93.8/97.0%. Qwen 125B MoE at a 20% cache: 84.3%, or 94.7% with router prefetch — [Jakevin/moe-offload](https://github.com/Jakevin/moe-offload)
  - GLM-5.2 (256 experts) at 64/90 slots: 73/79% — [llama.cpp PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294)
  - gpt-oss-120b steady state: 98-100% (8 GB GPU cache, after warm-up); cold start 48-56% — [llama.cpp issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757)
  - Calibrated pinning: 42%; pinning plus LRU: 56% — [llama.cpp Discussion #27149](https://github.com/ggml-org/llama.cpp/discussions/27149)
- mbolt gives a speedup model for faster cold reads: 1/((1−s)+s/g), where s is the cold-read share of token time (≈0.96 at 0.2 resident) and g is the per-cold-expert I/O gain. Plain LRU at near-1 table/RAM ratios gives +14-18% decode — [doramirdor/mbolt](https://github.com/doramirdor/mbolt)
- Co-activation: 23% of expert pairs co-activate at more than 2× independence (Qwen3-Next-80B) — [doramirdor/mbolt](https://github.com/doramirdor/mbolt)

### Inferences
- **Worked estimate for Kolibri** (50 layers × Top-6 = 300 routed-expert reads per token). The expert sizes are approximations: dividing total file size by 50×384 = 19,200 experts overestimates slightly because non-expert weights are included.
  - Q8_0 (83 GB): at most ~4.3 MB per expert, so ≤ ~1.3 GB/token if every expert misses.
  - IQ3 (33 GiB): ~1.8 MB per expert, so ~0.55 GB/token if every expert misses.
  - At 0% hit and ~6-12 GB/s cold reads, the I/O-only ceiling is ~5-9 tok/s for Q8_0 and ~11-22 tok/s for IQ3.
  - At 80% hit it is ~23-46 and ~55-110 tok/s respectively, before compute.
  - Our measured 13 tok/s (Q8_0) and 17-18 tok/s (IQ3) suggest compute and synchronization overhead now dominates, not raw SSD bandwidth. This matches the ~19-21% interception overhead in the Metal PoC and the 41% Python/sync share in Jakevin's v1 profile. Measuring our hit rate and per-miss latency would confirm this.
- Kolibri's expert slices (~1.8-4.3 MB) are in the 1-4 MB band where coalescing each expert's gate/up/down into one read still pays (+14-32%). Above ~19 MB, coalescing stops mattering.
- Flash-moe's "SSD DMA slows GPU by 73%" finding suggests that overlapping prefetch with Metal compute on the same memory controller may give less than expected on M-series chips.

### Gaps
- The hit-rate figures come from different models with different expert counts and Top-K values. None is a 384-expert / Top-6 model like Kolibri, so transfer is approximate.
- There are no published per-miss latency distributions (p50/p99) for MB-sized `pread` on Apple SSDs.
