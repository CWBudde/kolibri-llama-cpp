# Kolibri-1 inference outside Aleph Alpha's vLLM plugin: public state as of 22 October 2026

Scope note: about 16 search/fetch calls. GitHub searches were run with `gh`. Hugging Face pages were fetched live. Some speed figures come only from search-engine summaries of blog posts and are marked "(unverified snippet)". Relative dates on HF and HN pages ("18 days ago") were read on 2026-10-22.

## Official facts relevant to offloading

### Takeaway
Kolibri-1 was released on 2026-10-03. It is a 78.1B-total / 3.46B-active MoE: 50 layers, 384 routed experts, top-6 sigmoid routing, 1 shared expert, expert FFN width 512, and 4:1 sliding-window/full GQA. It ships as FP8 (default repo, ~78 GB) and BF16 (~156 GB) safetensors under Apache 2.0. Aleph Alpha targets datacenter GPUs only, and neither model card gives any CPU-offload or consumer-hardware guidance.

### Cited Findings
- Release date 2026-10-03. Developer: Aleph Alpha Research GmbH. German/English, with an explicit reasoning mode (effort `none`/`low`/`medium`/`high`) and tool calling — [HF Kolibri-1-BF16](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16)
- Total 78,103,074,560 params, active 3,457,573,120 per token — [HF Kolibri-1-BF16](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16)
- Architecture:
  - MoE, 50 layers, 384 routed experts per MoE layer, top-6, token-choice sigmoid routing, 1 shared expert per MoE layer, expert hidden size 512.
  - Attention is 4:1 sliding-window GQA to full causal GQA. The window is 512 preceding tokens plus the current one.
  - 48 query heads, 4 KV heads, head dim 128, hidden 2560, vocab 128,000. Per-head QK RMSNorm, SwiGLU.
  - RoPE (base 10,000) in the sliding layers only.
  - Source: [HF Kolibri-1-BF16](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16)
- Context: 262,144 native, 1,048,576 by extrapolation, ≤262,144 recommended — [HF Kolibri-1-BF16](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16)
- License: Apache 2.0, covering only the weights and config files. Code, architecture, parameter settings and training methods are excluded — [HF Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1); [heise, 2026-10-06](https://www.heise.de/en/news/AI-from-Germany-Aleph-Alpha-introduces-open-weight-model-Kolibri-11477528.html)
- Default repo `Aleph-Alpha/Kolibri-1`:
  - Weights: FP8 `float8_e4m3fn` in 128×128 blocks. Embeddings, LM head, norms and router stay BF16. About 78 GB.
  - Minimum hardware: 2× A100 80 GB, 2× H100 SXM5, or 1× H200/B200/B300.
  - Source: [HF Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1)
- BF16 repo:
  - About 156 GB.
  - Minimum: 4× A100 80 GB, 4× H100 SXM5, 2× H200, 1× B200 or 1× B300.
  - Recommended: 4× H100, 2× H200, 2× B200 or 1× B300.
  - Source: [HF Kolibri-1-BF16](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16)
- Official inference stacks on the cards:
  - vLLM with the `aleph-alpha-inference>=1` plugin (pip package, or container `ghcr.io/aleph-alpha/aleph-alpha-inference`), using `--reasoning-parser kolibri1`, `--tool-call-parser kolibri1` and `--enable-auto-tool-choice`.
  - Transformers ≥5.20 "natively".
  - SGLang and Docker Model Runner, listed under Local Apps.
  - llama.cpp, Ollama, LM Studio and MLX are not mentioned on the FP8 card.
  - Source: [HF Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1); [HF Kolibri-1-BF16](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16)
- Memory-saving guidance amounts to `--kv-cache-dtype fp8` for vLLM and `device_map="auto"` in the Transformers examples. There is no explicit CPU-offload or disk-offload guidance — [HF Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1); [MindStudio local guide, 2026-10-04](https://www.mindstudio.ai/blog/aleph-alpha-kolibri-1-local)
- Recommended sampling: temperature 1.0, top_p 0.97, top_k 128 — [HF Kolibri-1-BF16](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16)
- Training: from scratch on 768 B200 GPUs in Germany and Finland — [DataCamp](https://www.datacamp.com/blog/aleph-alpha-kolibri-1)
- About 23% of pre-training tokens are German. Synthetic data came partly from GLM-5.2/5.3 and Qwen3.8-27B. Benchmarks are self-reported and unconfirmed — [heise](https://www.heise.de/en/news/AI-from-Germany-Aleph-Alpha-introduces-open-weight-model-Kolibri-11477528.html)
- Company context: a merger with Cohere was agreed in September 2026 and still needs regulatory approval — [heise](https://www.heise.de/en/news/AI-from-Germany-Aleph-Alpha-introduces-open-weight-model-Kolibri-11477528.html)
- Checkpoint detail that matters for expert streaming: 7,720 of 19,200 routed experts (40.2%) have their gate/up/down tensors split across two safetensors shards. The FP8 release has 116,303 tensors — [colibri issue #1944 (bokiko)](https://github.com/JustVugg/colibri/issues/1944)
- Downloads and likes, as of 2026-10-22:
  - FP8 repo: 21,280 downloads last month, 962 likes, 24 quantizations, 4 finetunes, 1 adapter, 2 Spaces — [HF Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1)
  - BF16 repo: 3,037 downloads, 63 likes, 35 quantizations, 2 finetunes — [HF Kolibri-1-BF16](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16)

### Inferences
- The FP8 card's minimum (2× 80 GB GPUs) means every consumer deployment (Macs, 24 GB GPUs, 64–96 GB RAM boxes) depends on community quantizations or offloading.
- The sparsity makes the model a natural candidate for offloading or expert caching. Only about 4.4% of parameters are active per token, the experts are small (width 512), and only 10 of 50 layers keep a full-context KV cache. Aleph Alpha does not discuss this.

### Gaps
- Transformers support conflicts between sources. The cards (fetched 2026-10-22) say Transformers ≥5.20 supports the model natively. A DGX Spark user on 2026-10-03 said Transformers did not support it yet ([NVIDIA forum](https://forums.developer.nvidia.com/t/aleph-alpha-kolibri-1-interesting-new-model-for-german-and-english-texts/384940)). Support likely landed after launch, but no release date was found.
- The tool-call parser also conflicts. MindStudio describes Hermes-style parsing, while the BF16 card says `--tool-call-parser kolibri1`.
- The official launch blog (aleph-alpha.com/en/blog/kolibri-has-landed-a-sovereign-open-weight-model/, cited by modelfit.io) was not fetched.
- No LinkedIn or X posts by Aleph Alpha staff were found or checked.

## Hugging Face derivatives (GGUF, MLX, AWQ/GPTQ, EXL, NVFP4) and discussions

### Takeaway
Within three weeks, about 35 quantized derivatives appeared. They include many GGUFs (bartowski, InsidiousFiddler, Eliasfpv28, huihui-ai abliterated), a family of MLX builds, NVFP4/FP8/INT8 variants, REAP-pruned W4A16 models and one EXL3. Most early GGUFs were made with pre-merge llama.cpp patches, and mainline rejects many of them.

### Cited Findings
- Quantization list for the BF16 base (first 30 of 35 shown; downloads as listed) — [HF quantized filter](https://huggingface.co/models?other=base_model:quantized:Aleph-Alpha/Kolibri-1-BF16):
  - **GGUF**:
    - bartowski/Kolibri-1-GGUF: 2.82k downloads, updated about 4 days ago
    - huihui-ai/Huihui-Kolibri-1-BF16-abliterated-GGUF: 7.28k
    - InsidiousFiddler/Kolibri-1-GGUF: 3.06k
    - Eliasfpv28/Kolibri-1-Q3_K_S-GGUF: 1.63k
  - **MLX**:
    - eins78: 3-, 4-, 5-, 6- and 8-bit, plus mixed 4/8
    - Sawfwair: MLX-Mixed-2bit and MLX-8bit, with about 750 and 950 downloads
    - baa-ai/Kolibri-1-RAM-50GB-MLX: 621
  - **NVFP4**:
    - primitive-ai/Kolibri-1-NVFP4
    - primitive-ai mixed NVFP4-FP8
    - 0xBakeer TandemLLM NVFP4
    - Blackfrost-AI DERISKED NVFP4/FP8
  - **W4A16 REAP-pruned** (expert pruning, which shrinks total parameters):
    - MediSync-AI/Kolibri-1-REAP-40B-A3B-W4A16
    - MediSync-AI/Kolibri-1-REAP-31B-A3B-W4A16
  - **EXL3**: Code0000/Kolibri-1-EXL3-K6
  - **Other**:
    - mbehr90 fp8-channel and int8-w8a8
    - a "GLQ-trellis 3bpw" build
    - mobbitxt "oQ4e"
- bartowski/Kolibri-1-GGUF:
  - Built with llama.cpp b11716. Architecture string `kolibri1`.
  - imatrix from a custom chat-template calibration set (535×512 tokens, 137 conversations).
  - Sizes: Q8_0 83.14 GB, Q4_K_M 49.76 GB (recommended), IQ3_XXS 33.65 GB, Q2_K 29.92 GB, IQ1_S 17.61 GB.
  - Custom per-tensor layouts: Q4_K_M KLD 0.0696 against 0.1122 for the standard layout, at a 4.1% larger file.
  - No speed numbers.
  - Source: [bartowski/Kolibri-1-GGUF](https://huggingface.co/bartowski/Kolibri-1-GGUF)
- baa-ai/Kolibri-1-RAM-50GB-MLX:
  - Mixed precision averaging 5.12 bpw. Routed experts are 3–5 bit in layers 0–30 and 6–8 bit in layers 31–49. Attention, router and shared expert are 8-bit.
  - 49.9 GB on disk. Needs the unreleased mlx-lm PR #1945.
  - Targets Macs with 96 GB+, or 64 GB "with care" after raising the wired limit via sysctl.
  - MMLU-100 73% against 74% for the 8-bit reference.
  - No offloading, no tok/s.
  - Source: [baa-ai/Kolibri-1-RAM-50GB-MLX](https://huggingface.co/baa-ai/Kolibri-1-RAM-50GB-MLX)
- Further builds named by modelfit.io:
  - MLX: velaia 2-bit (25.5 GB), 3-bit (~35 GB) and 4-bit (44.3 GB); here-be-dragons-ai/Kolibri-1-MLX-3bit (33 GiB)
  - GGUF: Hob-forge/Kolibri-1-GGUF Q4_K_M (47.5 GB, CPU-tested on 64 GB RAM)
  - Source: [modelfit.io](https://modelfit.io/blog/kolibri-1-aleph-alpha-mac-memory-requirements/)
- HF discussion "llama.cpp support?" on the FP8 repo — [HF discussion #3](https://huggingface.co/Aleph-Alpha/Kolibri-1/discussions/3):
  - **ulymp** reported about 90 tok/s with a 4-bit GGUF on an RTX 3090, using a llama.cpp fork branch (GenerelSchwerz "moe-cache") plus a patch.
  - **CyberNativeAI** tested 40 quants from 11 Hub repos against mainline b11712. Only 7 loaded as-is. Others failed on a missing sliding-window key or an unknown pre-tokenizer.
    - Workaround: `--override-kv tokenizer.ggml.pre=str:qwen2` and `--override-kv kolibri1.expert_gating_func=int:2`.
    - Verified only on a Q2_K, on CPU: PPL 46.5190 mainline against 46.5191 patched.
  - No Aleph Alpha staff replied.
- Unanswered questions in the same thread: coding quality, and prefill speed — [HF discussion #3](https://huggingface.co/Aleph-Alpha/Kolibri-1/discussions/3)

### Inferences
- The 3090 report is 4-bit (~45–50 GB) on a 24 GB card, so most expert weights must have lived in system RAM. A fork named "moe-cache" suggests expert caching or offload. This is the closest public analogue to the expert-cache work in this repo. The thread does not describe the mechanism, and I could not find the fork itself (see Gaps).
- The REAP-pruned 31B/40B builds are a different route to consumer hardware: they remove experts instead of streaming them.

### Gaps
- No search returned the GenerelSchwerz "moe-cache" fork, so its mechanism and the 3090 offload configuration are unverified.
- Page 2 of the quantization list (5 more models) was not read. No GPTQ or AWQ repos with those exact labels were seen. "W4A16" builds may be GPTQ/AWQ-style, but this is unconfirmed.

## Offloading attempts (AirLLM, accelerate, vLLM --cpu-offload-gb, kTransformers, SSD streaming) and reported speeds

### Takeaway
I found no public AirLLM, accelerate disk-offload or kTransformers runs of Kolibri-1. The only documented offload use is a vLLM + plugin baseline with "58 GB UVA CPU offload" on an RTX 3090 Ti, run as a numerical reference rather than a speed test. The only SSD/expert-streaming effort found is an in-progress `kolibri1` engine for the colibri disk-streaming MoE engine; its draft PR is CPU-only and not qualified.

### Cited Findings
- **AirLLM, accelerate disk offload, kTransformers:** targeted searches returned nothing on Kolibri-1. Results were generic kTransformers/colibri material — [noze.it on KTransformers](https://www.noze.it/en/insights/ktransformers-hybrid-cpu-gpu-inference/) (generic, no Kolibri)
- **vLLM CPU offload:** bokiko's official-plugin baseline was a "locked O1 stack with 58 GB UVA CPU offload". The underlying stack was vLLM 0.29.0 + plugin, Marlin W8A16, on an RTX 3090 Ti.
  - Across 13 inputs of 2k–16k tokens, its median per-document mean KL against an FP64 reference was 0.022–0.23.
  - Run-to-run variation was large: symmetric pairwise KL about 0.02–0.18.
  - "Offload numerical equivalence unestablished." No tok/s given.
  - Source: [colibri issue #1944](https://github.com/JustVugg/colibri/issues/1944)
- **colibri (JustVugg) `kolibri1` engine by bokiko:**
  - Background: colibri is a C engine that keeps dense weights in RAM and streams experts from SSD with an LRU cache ([JustVugg/colibri](https://github.com/JustVugg/colibri)).
  - The `kolibri1` engine reads the FP8 release directly, loads all 116,303 tensors in about 0.3 s, and has a CPU forward pass with an expert LRU and FP64 residual stream. It plugs into colibri's CUDA expert tier.
  - Test box: Ryzen 9 9955HX, 92 GB RAM, RTX 3090 Ti over OcuLink, Samsung 980 PRO NVMe.
  - The CUDA expert tier allocates about 6.02 MiB per Kolibri expert against the planner's assumed ~3.75 MiB, so VRAM expert capacity was over-predicted by about 1.6×.
  - Routing near-ties at 16k context flip some top-1 tokens (e.g. 37 of 16,384 positions on a 16k EN doc).
  - Sources: [issue #1944](https://github.com/JustVugg/colibri/issues/1944); request [issue #1869](https://github.com/JustVugg/colibri/issues/1869)
  - Draft PR #2049, opened 2026-10-13 and still open: "[Draft] kolibri1: experimental CPU engine … (qualification incomplete)". It is CPU engine plus gateway only, with the CUDA expert tier deferred. Fresh G3 validation failed: 1 of 80 cells exceeded the logit tolerance (0.01406 against 0.01). The truncated body showed no tok/s — [colibri PR #2049](https://github.com/JustVugg/colibri/pull/2049)
- **Consumer-RAM failure case:** a 16 GB M4 MacBook Pro running the MLX 2-bit build needed 24,278 MB against a 12,124 MB recommended maximum. Swap passed 22 GB and no first token came after 4 minutes — [modelfit.io](https://modelfit.io/blog/kolibri-1-aleph-alpha-mac-memory-requirements/)
- **Name collision:** "Colibri"/"Colibrì" (JustVugg) is an unrelated project name but a related technique. Coverage of it (e.g. GLM-5.2 744B on 25 GB RAM at about 0.1–1 tok/s) is not about Kolibri-1 — [daniliants.com](https://daniliants.com/insights/colibri-streams-a-370-gb-moe-model-off-ssd-on-25-gb-of-ram/); [MindStudio Colibri](https://www.mindstudio.ai/blog/run-744b-ai-model-consumer-laptop-colibri)

### Inferences
- As of 2026-10-22, no public report I found gives a tok/s figure for Kolibri-1 streamed from SSD or disk-offloaded. Two efforts sit closest: the colibri `kolibri1` engine (correctness-focused, no speed published) and the 3090 "moe-cache" llama.cpp fork (~90 tok/s at 4-bit, mechanism undocumented). This repo's SSD expert cache (~17 tok/s) therefore appears to be the only public SSD-streaming speed figure. That conclusion rests on the absence of search results and is not proven.

### Gaps
- PR #2049 may contain CPU decode speeds beyond the 6,000 characters read.
- The "moe-cache" fork code was not found.
- No `--n-cpu-moe` or `-ot` expert-offload reports for Kolibri-1 on llama.cpp were found, though they are likely to exist informally.

## Upstream llama.cpp / MLX support and independent ports, with reported speeds

### Takeaway
Upstream llama.cpp merged Kolibri-1 (arch `kolibri1`) on 2026-10-17 through ggml-org PR #30673, shipping in b11709+. mlx-lm support is proposed in PR #1945, which was open and unreleased as of the latest pages read. Independent GPU and CPU ports exist (TensorFold, colibri, mistral.rs reference work), and DGX Spark reports cluster around 40–50 tok/s single-stream.

### Cited Findings
- **llama.cpp request:** issue #29922 "Feature Request: Aleph-Alpha/Kolibri-1", opened 2026-10-03 19:38 UTC — [ggml-org/llama.cpp#29922](https://github.com/ggml-org/llama.cpp/issues/29922)
- **llama.cpp PR #30673** "model : add Kolibri-1 (Aleph-Alpha) support", by riteme — [ggml-org/llama.cpp#30673](https://github.com/ggml-org/llama.cpp/pull/30673):
  - Opened 2026-10-16, merged 2026-10-17 by ngxson. Reviewed by CISC and ggerganov, with follow-up #30865.
  - Implementation details:
    - Sliding layers (window 513) use NeoX RoPE and full layers use NoPE.
    - Routing selects on logits + bias, then weights by sigmoid of the unbiased logits, with no renorm.
    - Pre-tokenizer is qwen2.
    - The converter reuses AfmoeModel, and FP8 goes through the `weight_scale_inv` path.
  - Q4_K_M is about 45 GiB. On 4× A800: pp2048 2558 tok/s, tg512 106 tok/s. Backends were not enumerated.
- **Downstream reaction:** the LocalAI fork dropped its carried patch, java-llama.cpp carried a "patch 0016", and the mistral.rs (lu-zero) work cites llama.cpp as an independent implementation — [PR #30673 references](https://github.com/ggml-org/llama.cpp/pull/30673); [java-llama.cpp PR #478](https://github.com/bernardladenthin/java-llama.cpp/pull/478)
- **Pre-merge community llama.cpp ports** included this repo's own work (CWBudde/llama.cpp PRs #1, #7, #9 and CWBudde/kolibri-llama-cpp), which is a self-hit to exclude from "independent" — [CWBudde/llama.cpp#7](https://github.com/CWBudde/llama.cpp/pull/7); [CWBudde/kolibri-llama-cpp](https://github.com/CWBudde/kolibri-llama-cpp). Before the merge, modelfit.io said GGUFs needed a patch applied at upstream commit `836d571` — [modelfit.io](https://modelfit.io/blog/kolibri-1-aleph-alpha-mac-memory-requirements/)
- **MLX:** mlx-lm 0.31.3 lacks `kolibri1`. velaia builds ship a `run.py` that registers `kolibri1.py`. Upstream PR ml-explore/mlx-lm#1945 was still open on 2026-10-04 — [modelfit.io](https://modelfit.io/blog/kolibri-1-aleph-alpha-mac-memory-requirements/). baa-ai still calls it unreleased — [baa-ai card](https://huggingface.co/baa-ai/Kolibri-1-RAM-50GB-MLX). Related: lablup/mlxcel issue #2357 (2026-10-16), and apple/coreai-models issue #330 — [mlxcel#2357](https://github.com/lablup/mlxcel/issues/2357); [coreai-models#330](https://github.com/apple/coreai-models/issues/330)
- **Other engine requests:**
  - TensorRT-LLM #19844 (2026-10-04) — [link](https://github.com/NVIDIA/TensorRT-LLM/issues/19844)
  - SGLang #43256 (2026-10-09), even though the card lists SGLang — [link](https://github.com/sgl-project/sglang/issues/43256)
- **Ollama / LM Studio:**
  - Pre-merge, neither could load `kolibri1`, and the author's Ollama binary contained no `kolibri1` string — [modelfit.io](https://modelfit.io/blog/kolibri-1-aleph-alpha-mac-memory-requirements/)
  - bartowski's card now offers LM Studio and `ollama run hf.co/bartowski/Kolibri-1-GGUF:Q4_K_M` commands but makes no explicit compatibility claim — [bartowski](https://huggingface.co/bartowski/Kolibri-1-GGUF)
  - At launch, there was no GGUF or Ollama build — [MarkTechPost, 2026-10-04](https://www.marktechpost.com/2026/10/04/aleph-alpha-releases-kolibri-a-78-1b-open-weight-english-german-moe-model-with-only-3-46b-active-parameters/)
- **DGX Spark (GB10, 128 GB unified), vLLM + plugin:**
  - tg128 about 48.3 tok/s single-stream and about 101 tok/s aggregate at 4 streams; pp2048 about 4,333 tok/s; tg at 16k depth about 46.6 tok/s (cosinus, 2026-10-03)
  - aisyndicate: just under 49 tok/s on the FP8 checkpoint
  - Source: [NVIDIA forum](https://forums.developer.nvidia.com/t/aleph-alpha-kolibri-1-interesting-new-model-for-german-and-english-texts/384940)
- **DGX Spark, TensorFold PR #328** (jschmied; custom CUDA FP8 path): single-stream decode about 40.3–40.6 tok/s, 4-stream about 100–112 tok/s. Sliding layers need about 215 MB per stream, and 4 streams at 262k context fit in about 20.8 GiB — [NVIDIA forum](https://forums.developer.nvidia.com/t/aleph-alpha-kolibri-1-interesting-new-model-for-german-and-english-texts/384940). A 2× DGX Spark TensorFold recipe claims about 2× decode — [ajensenwaud/Kolibri-1-2x-DGX-Spark-TensorFold](https://github.com/ajensenwaud/Kolibri-1-2x-DGX-Spark-TensorFold)
- **Other DGX Spark recipe repos** — [mrexodia](https://github.com/mrexodia/Kolibri-1-vLLM-DGX-Spark), [pulseandthread](https://github.com/pulseandthread/kolibri-1-dgx-spark), [GroveMinting](https://github.com/GroveMinting/kolibri-1-solo-dgx-spark-eugr), [15ky3 (stock vLLM 0.30)](https://github.com/15ky3/Kolibri-1-DGX-Spark), [mARTin-B78](https://github.com/mARTin-B78/aleph-alpha-inference-kolibri-1-dgx-spark)
- **H200 evaluation** (throughput, prompt injection, NIS2) — [sbeierle/kolibri1-praxistest](https://github.com/sbeierle/kolibri1-praxistest)
- **RTX Pro 6000 (96 GB):** about 170 tok/s at FP8 (HN user martianvoid) — [HN](https://news.ycombinator.com/item?id=49942706)
- **Mac speeds:**
  - Publisher-reported 52–56 tok/s on an M1 Max for the velaia MLX builds. The modelfit author measured none themselves — [modelfit.io](https://modelfit.io/blog/kolibri-1-aleph-alpha-mac-memory-requirements/)
  - Further figures from search snippets only (unverified snippets) — [developersdigest](https://www.developersdigest.tech/blog/run-kolibri-locally-gguf-mlx-2026); [kingy.ai](https://kingy.ai/news/kolibri-1-local-hardware-ram-vram-mac/):
    - about 50 tok/s on an M4 Pro 64 GB
    - about 70 tok/s short context and 40 tok/s at 96k on an M5 Pro 48 GB with a 3/6-bit MLX build
    - 91–116 tok/s on an M5 Max 128 GB at 6-bit
    - "a separate llama.cpp port reports 64 tok/s on a 48 GB Mac through Metal", which may well refer to this repo

### Inferences
- Upstream support makes `-ot`/`--n-cpu-moe` expert offload available to anyone from b11709 on. Many early Hub GGUFs, however, need re-conversion or `--override-kv` fixes.
- Every concrete Mac speed I found comes from fully resident quants (2–6 bit MLX/GGUF in 36–128 GB). None comes from streaming.

### Gaps
- The developersdigest and kingy.ai articles were not fetched, so their Mac numbers and the "64 tok/s llama.cpp port" attribution are unverified.
- The current status of mlx-lm #1945 (merged or not) could not be retrieved. `gh` returned an error for that PR.
- Whether current LM Studio and Ollama releases bundle a llama.cpp ≥ b11709 was not checked.

## Community reception (Reddit, HN, German press, staff posts) and unrelated "Kolibri" hits

### Takeaway
HN gave the release a large thread (713 points, 336 comments). Commenters praised its openness but were skeptical of its quality relative to Qwen 3.8 27B, and Aleph Alpha trainers took part. heise covered it neutrally. I found no r/LocalLLaMA thread and no golem or t3n article.

### Cited Findings
- **HN "Kolibri: A Sovereign Open-Weight Model": 713 points, 336 comments** — [HN](https://news.ycombinator.com/item?id=49942706)
  - Self-identified Aleph Alpha trainers posted: ivo-42, befelix and peterBlue75.
  - Hardware estimates:
    - ekidd: comfortable in 96 GB RAM with a decent quant
    - sspiff: a mixed 8/4-bit quant fits in 64 GB
    - hypfer: usable community quants exist on HF
  - Themes:
    - praise for the tech report's transparency
    - "Qwen 3.8 27B is stronger" skepticism
    - debate over "sovereignty" given the Cohere merger
    - an abstention failure on a made-up song
    - copyright debate
  - Only the first 100k of 170k characters were read.
- **A second HN thread**, "Aleph Alpha Kolibri: How the sovereign German LLM works" (about 162 comments, links tej.as), came up only in a search snippet (unverified snippet) — [tej.as](https://tej.as/blog/aleph-alpha-kolibri)
- **heise (2026-10-06):** specs and the Apache 2.0 caveat; performance data self-reported and pending independent confirmation; nothing on local, quantized or offloaded use — [heise](https://www.heise.de/en/news/AI-from-Germany-Aleph-Alpha-introduces-open-weight-model-Kolibri-11477528.html)
- **trendingtopics.eu headline:** "Aleph Alpha's Kolibri Is No Match for the Open-Weight Leaders" — [trendingtopics](https://www.trendingtopics.eu/aleph-alpha-kolibri-open-weight/)
- **NVIDIA DGX Spark forum:** cosinus judged that Qwen3.8-27B outperforms it but that Kolibri-1 still suits EU-compliance-driven German use. Users hoped for NVFP4 quants for the RTX PRO 6000 — [NVIDIA forum](https://forums.developer.nvidia.com/t/aleph-alpha-kolibri-1-interesting-new-model-for-german-and-english-texts/384940)
- **Secondary or explainer coverage** with no original measurements, mostly relaying the model card — [MindStudio](https://www.mindstudio.ai/blog/aleph-alpha-kolibri-1-sovereign-moe), [DataCamp](https://www.datacamp.com/blog/aleph-alpha-kolibri-1), [MarkTechPost](https://www.marktechpost.com/2026/10/04/aleph-alpha-releases-kolibri-a-78-1b-open-weight-english-german-moe-model-with-only-3-46b-active-parameters/), [testingcatalog](https://www.testingcatalog.com/aleph-alpha-releases-open-weight-kolibri-with-1m-context/), [innfactory](https://innfactory.ai/en/ai-models/aleph-alpha-kolibri/), [Medium](https://medium.com/data-science-in-your-pocket/first-german-llm-kolibri-1-1d4368a1a649)
- **Unrelated "Kolibri" hits:**
  - KolibriOS ([Wikipedia](https://en.wikipedia.org/wiki/KolibriOS); anu-prakash-dev/KolibriOS-1 mirror)
  - Learning Equality's Kolibri platform (learningequality/* repos and snorkel-marlin PR copies; koenswings "Demo Kolibri" issues)
  - public-ui/kolibri (a German UI component library)
  - kolibril13 (manim user)
  - kolibri.finance (Hover-Labs indexer)
  - Kolibree ([Wikipedia](https://en.wikipedia.org/wiki/Kolibree))
  - the JustVugg "Colibri" MoE engine, which is name-adjacent but now relevant because of its kolibri1 port
  - huzjie/kolibriforge, a "Kolibri-direction" research repo that is not a port
  - Aleph Alpha's older Luminous models did not surface in the Kolibri-1 results

### Inferences
- Community energy has gone into DGX Spark recipes, MLX/GGUF quants and llama.cpp support. Interest in offloading is implicit (24 GB GPUs, 48–64 GB Macs) but almost undocumented.

### Gaps
- No r/LocalLLaMA thread was found. Search returned only 2023 posts, and Reddit was not queried directly.
- No golem.de or t3n articles were found.
- LinkedIn and X posts by Aleph Alpha staff were not searched.
