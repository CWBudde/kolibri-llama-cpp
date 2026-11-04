# AirLLM (lyogavin/airllm): mechanism, measured performance, limits (state as of 2026-10-22)

Scope note: code citations point to commit `22d72c4` (HEAD of `main`, 2026-10-21), i.e. AirLLM v4.0.0 plus the merges of 2026-10-19. "Claim" means a README or release statement; "measured" means a reported run with a number attached. All dates are given explicitly because the project changed substantially between the 2023 v2.x line and the 2026 v3/v4 line.

Permalink base used below: `https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/` (abbreviated as BASE).

## 1. Mechanism: what AirLLM actually does

### Takeaway
AirLLM is a Python/PyTorch wrapper that splits a Hugging Face checkpoint into one safetensors file per module (embed, each decoder layer, norm, lm_head) and runs the stock `transformers` model on the `meta` device. Forward hooks load a layer's weights from disk to the GPU right before it runs and drop them right after. Nothing is cached between tokens, so each decoded token re-reads the whole model from disk (or, since July 2026 and for one model family only, the routed experts). The KV cache stays on the GPU through normal `transformers` generation. I/O overlap is a single-thread, depth-1 prefetch of the next layer. The optional "compression" is bitsandbytes NF4/8-bit storage that is dequantized back to fp16 on every load.

### Cited Findings
**Origin and design intent (2023)**
- The technique comes from SimJeg's Kaggle notebook "platypus2-70b-with-wikipedia-rag" for the Kaggle *LLM Science Exam* competition. The README credits it: "A lot of the code are based on SimJeg's great work in the Kaggle exam competition" — [README](https://github.com/lyogavin/airllm#acknowledgement)
- The author's launch post (Hugging Face blog, 2023-11-30) describes the method as follows. A 70B model has 80 layers. "We can load whichever layer is needed from disk when executing that layer". Per-layer memory is "1/80 of the full model, around 1.6GB". The weights are pre-sharded by layer as safetensors (memory-mapped loading). The model is built on the accelerate `meta` device. Tested on a 16 GB T4, it used "less than 4GB GPU memory" — [HF blog](https://huggingface.co/blog/lyogavin/airllm)
- The same post says: "Disk reading speed is actually the slowest bottleneck in the whole inference process", and it is "Not very suitable for interactive scenarios like chatbots. More suited for some offline data analytics like RAG, PDF analysis etc." The post gives no tokens/s figure — [HF blog](https://huggingface.co/blog/lyogavin/airllm)
- KV cache sizing in the 2023 post uses the formula `2 * input_length * num_layers * num_heads * vector_dim * 4`. Its example result ("= 30MB") is about half of what the formula actually gives (~65 MB). The post's arithmetic is sloppy — [HF blog](https://huggingface.co/blog/lyogavin/airllm)

**Current (v3/v4, 2026) runtime architecture**
- The class docstring says the checkpoint "is split into per-layer shards on disk. The real transformers model is instantiated on the `meta` device … AirLLM only attaches forward hooks to each big module (embeddings, every decoder layer, the final norm and the lm_head) to stream that module's weights disk -> GPU right before it runs and free them right after, prefetching the next module on a worker thread." — [airllm_base.py L58-71](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/airllm_base.py#L58-L71)
- The v3.0.0 release (2026-06-30) "Reworked layer streaming to build on the standard Transformers model path". It also made the runtime dtype follow the model's native dtype (bf16), because forced fp16 had produced "garbled output on very deep models" — [v3.0.0 release](https://github.com/lyogavin/airllm/releases/tag/v3.0.0)
- Pre-hook: it takes the prefetched state dict if one is ready, otherwise it loads synchronously. It moves the tensors to the device, then submits the *next* layer to a `ThreadPoolExecutor(max_workers=1)`. Prefetch depth is therefore exactly one layer — [airllm_base.py L186-193, L889-904](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/airllm_base.py#L889-L904)
- Post-hook: `module.to('meta')` (or per-tensor eviction to meta), then `clean_memory()`. That function runs `gc.collect()`, `malloc_trim(0)` and `torch.cuda.empty_cache()` after **every layer** — [airllm_base.py L906-919](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/airllm_base.py#L906-L919); [utils.py L78-85](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/utils.py#L78-L85)
- Host-side loading uses `safetensors.torch.load_file` on the per-layer file, which reads the whole layer into host memory. The data is pinned (`pin_memory()`, a copy) only when CUDA is present and the layer is ≤ 2 GiB. The code comment explains why: "a frontier MoE checkpoint has ~17GB layers, and with prefetching two are in flight at once, which would lock up ~34GB of RAM" — [airllm_base.py L73-75, L430-458](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/airllm_base.py#L430-L458); [persist/safetensor_model_persister.py](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/persist/safetensor_model_persister.py)
- What stays resident: the rotary/other buffers (moved to the device once), the input embedding if `tie_word_embeddings` is set, and "resident" side modules such as vision towers. Everything else is evicted after use — [airllm_base.py L368-373, L673-687, L757-765](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/airllm_base.py#L749-L775)
- Models with huge lookup tables (Qwen3.8-Flash-Next's ~51B-parameter n-gram/PLE embedding, ~102 GB bf16) keep that table **file-mmapped on the host**. It is gathered on the CPU and never streamed to the GPU — [airllm_base.py L689-747](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/airllm_base.py#L689-L747); [v3.3.0 release](https://github.com/lyogavin/airllm/releases/tag/v3.3.0)
- KV cache: generation is delegated to `self.model.generate(...)` with `use_cache=True` (README example). The device property is patched so that `transformers` places "inputs/cache tensors on the right device", meaning the GPU. There is no AirLLM-specific KV offload. `max_seq_len` is stored but not used anywhere in the base class — [airllm_base.py L375-403, L923-924](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/airllm_base.py#L375-L403); [README quickstart](https://github.com/lyogavin/airllm#quickstart)
- The Kimi K3 PR itself notes: "The 3.72 GB figure was measured on a short prompt … KV cache growth at long context was not tested." — [PR #316](https://github.com/lyogavin/airllm/pull/316)
- Disk preparation: the first run splits the checkpoint into per-layer files. A naive split needs about 2× the checkpoint size on disk ("a naive split would want 3.12TB for a 1.56TB model"). For K3 the split files were hardlinked instead of copied. `delete_original=True` reclaims the original. Flash-Next needs ~360 GB of disk — [v3.1.0 release](https://github.com/lyogavin/airllm/releases/tag/v3.1.0); [v3.3.0 release](https://github.com/lyogavin/airllm/releases/tag/v3.3.0); [utils.py link_or_copy_file](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/utils.py#L517-L535)

**"Compression" (block-wise 4/8-bit)**
- Compression is applied at split time. `compression='4bit'` uses `bnb.functional.quantize_nf4(..., blocksize=64)`; `'8bit'` uses `quantize_blockwise(..., blocksize=2048)` — [utils.py L486-504](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/utils.py#L486-L504)
- On every layer load, `uncompress_layer_state_dict` dequantizes each tensor back to full precision on CUDA (`dequantize_nf4` / `dequantize_blockwise`). Compute therefore runs in fp16/bf16. Only the on-disk bytes shrink — [utils.py L104-132](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/utils.py#L104-L132)
- The README rationale: "the bottleneck is mainly at the disk loading, we only need to make the model loading size smaller. So, we get to only quantize the weights' part". It claims "up to 3x" speedup and "almost ignorable accuracy loss", citing the block-wise quantization paper arXiv:2212.09720 — [README "Model Compression"](https://github.com/lyogavin/airllm#model-compression---3x-inference-speed-up)
- Compression force-disables prefetching ("prefetching is not supported together with compression for now"). It also disables per-expert streaming, falling back to whole-layer streaming — [airllm_base.py L188-190, L800-808](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/airllm_base.py#L186-L193)
- The only accuracy test in the repo is RMSE on two random (32,128) tensors. There is no model-level perplexity or task evaluation — [issue #295](https://github.com/lyogavin/airllm/issues/295)
- Pre-quantized checkpoints (FP8 block-FP8, MXFP4 via compressed-tensors) are loaded "verbatim" without a dtype cast. MXFP4 crosses PCIe packed and is expanded on the GPU ("moving 4x less data") — [airllm_base.py L490-544, L634-649](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/airllm_base.py#L490-L544); [v3.1.0 release](https://github.com/lyogavin/airllm/releases/tag/v3.1.0)

### Inferences
- AirLLM's core is a "layer pager" with zero weight reuse across tokens. The resident set is about one or two layers plus the KV cache. That is the opposite design point from the Kolibri prototype, which keeps all dense weights resident and caches 128 of 384 experts per layer.
- The Kaggle origin explains the design. The LLM Science Exam was a multiple-choice task in which each question needs one forward (prefill/scoring) pass and no long autoregressive decode. Streaming each layer once per *batch* of questions amortizes the I/O well there. Autoregressive decode reverses that economics. (This reading of the origin is my inference from the competition name and README credit; I could not load the Kaggle discussion page.)
- Per-layer `gc.collect()` + `malloc_trim` + `empty_cache`, a fresh `safe_open` per expert, and Python-level hooks add fixed overhead per layer and per expert. This probably matters a lot when layers or experts are small (see the 3B-model and K3 numbers in section 2).
- "Compression" is storage-only quantization: it trades GPU dequant work for disk bytes. It is not low-bit compute like llama.cpp's IQ3/Q8_0 kernels, so it cannot reduce compute or VRAM bandwidth.

### Gaps
- No author-published breakdown of per-layer time (disk vs H2D vs compute) for v3/v4. A `profiling_mode` exists, but no results are published.
- Whether the HF blog's "flash attention / O(log n) memory" claims ever corresponded to code. In v3 attention is simply `sdpa`, falling back to `eager`, through transformers.

## 2. Measured throughput vs. marketing claims; why decode is slow

### Takeaway
Every 2026 headline number ("70B on 4 GB", "405B on 8 GB", "DeepSeek-V3 671B on ~12 GB", "Kimi K3 2.8T on <4 GB") is a **peak-VRAM** figure. The only throughput the author has published for v3/v4 is Kimi K3 at **292 s/token**. User reports range from ~3 s/token (1.1B model) to minutes per token (32-70B). Several users report wall times of 20-40 minutes per sentence. Decode is slow by construction: weight bytes read per token ≈ the whole model (dense or whole-layer MoE), and nothing is cached across tokens. AirLLM only makes sense for offline, prefill-heavy or batch scoring, which is what the author recommended in 2023.

### Cited Findings
**Marketing claims (VRAM, not speed)**
- README (2026-10): "letting 70B large language models run on a single 4GB GPU card — without quantization, distillation, or pruning … Kimi K3 (2.8T) … on under 4GB, Qwen3.8-Flash-Next (125B) on 6GB, and DeepSeek-V3 (671B) on ~12GB". The "Tiny GPU, huge models" table lists Llama 3.1 405B at ~8 GB and Qwen3-235B at ~3 GB — [README](https://github.com/lyogavin/airllm#tiny-gpu-huge-models)
- The 405B example notebook loads `unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit`, a pre-quantized 4-bit model. The notebook has no executed outputs, memory capture or timing — [issue #295 (2026-06-21)](https://github.com/lyogavin/airllm/issues/295)
- The 2026 PRs report only VRAM. Qwen3.8-27B: "Measured on an RTX 3090: 3.33GB peak VRAM, greedy decode produced `Paris`" — [PR #343](https://github.com/lyogavin/airllm/pull/343). Qwen3.8-Flash-Next: "Measured on an RTX 4090: 5.95GB peak VRAM" — [PR #353](https://github.com/lyogavin/airllm/pull/353). Neither reports a time.

**Author-measured throughput**
- Kimi K3 (2.8T, MXFP4, 1.56 TB checkpoint) on one RTX 6000 Ada 48 GB: peak VRAM 3.72 GB; "Init + split/link 900 s"; "Generation: 4 tokens in 1168 s (292 s/token, disk-bound)" (2026-07-29) — [PR #316](https://github.com/lyogavin/airllm/pull/316); [v3.1.0 release](https://github.com/lyogavin/airllm/releases/tag/v3.1.0)
- Author's own 2023 notebooks contain tqdm progress output; hardware is not stated. On macOS (MLX), Yi-34B shows 60 layers in 53-55 s per pass (≈54 s/token), and Platypus2-7B shows 32 layers in 7-8 s (≈8 s/token) — [run_on_macos.ipynb](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/examples/run_on_macos.ipynb). Mixtral-8x7B on CUDA shows 35 modules in 4:29-4:30 per pass (≈270 s/token) — [test_mixtral.ipynb](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/tests/test_notebooks/test_mixtral.ipynb). Platypus2-7B with 4-bit compression on CUDA shows 35 modules in 10-11 s per pass after the first — [test_compression.ipynb](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/tests/test_notebooks/test_compression.ipynb)
- The v2.5 notes (2023-12-18) say prefetching gave a "10% speed improvement". v2.0 (2023-12-01) claims "Support compressions: 3x run time speed up!" — [README Updates](https://github.com/lyogavin/airllm#updates)

**User-measured throughput (GitHub issues)**
- Qwen2.5-32B-Instruct on an NVIDIA GB10 (split fits in a 76 GB page cache), 10 tokens: no compression **13.3 s/token**, 1.60 GB peak, 975 MB/layer. 8-bit: **50.5 s/token** (3.8× slower), 3.16 GB. 4-bit: **106.7 s/token** (8.0× slower), 2.04 GB (2026-08-04). "Speed tracks dequantization work, not bytes read" — [issue #330](https://github.com/lyogavin/airllm/issues/330)
- A contributor confirms the mechanism from code: compression disables prefetch and dequantizes every layer on every token. The contributor did not reproduce the numbers — [issue #330 comment](https://github.com/lyogavin/airllm/issues/330)
- Qwen2.5-3B on an RTX 5060 Ti 16 GB, Windows, NVMe, airllm 4.0.0: 4 tokens in 114.5 s = **28.6 s/token**, ~6% GPU utilization. The reporter computes an effective ~210 MB/s disk→GPU throughput (2026-09-10) — [issue #364](https://github.com/lyogavin/airllm/issues/364)
- TinyLlama-1.1B on M1 Pro 16 GB via the MLX backend: 32 tokens in 96.9 s = **0.33 tok/s (~3 s/token)**. The reporter's diagnosis: "every generated token re-streams all `n_layers` from disk — `use_cache=True` only caches attention state, not weights" (2026-09-27) — [issue #364 comment](https://github.com/lyogavin/airllm/issues/364)
- A 1.5B model on an RTX 4060 Laptop 8 GB: streaming **0.1 tok/s**. The same model fully loaded (proposed `AIRLLM_FULL_LOAD`) runs at **32.2 tok/s**, i.e. 322× faster. The maintainer replied (2026-10-19): "streaming is the wrong default *only* when the whole checkpoint fits" — [PR #324](https://github.com/lyogavin/airllm/pull/324)
- In the same thread, a commenter reports that greedy partial layer residency (keep as many layers on the GPU as fit, e.g. a 14B model on a 7B-capacity GPU) gave "nearly no measurable performance gain" over pure streaming — [PR #324](https://github.com/lyogavin/airllm/pull/324)
- gpt-oss-20b-bf16 on an RTX 3090: "around 30 minutes" for one ~120-token answer (≈15 s/token, 2026-10-11) — [issue #374](https://github.com/lyogavin/airllm/issues/374)
- Qwen2.5-7B on an RTX 3090: "about 40 minutes to generate one sentence" (128 in / 128 out, 2024-09-26) — [issue #186](https://github.com/lyogavin/airllm/issues/186)
- Qwen2.5-Coder-32B with 4-bit compression on an RTX 3050 6 GB: "just above 2 minutes per token" from an HDD and **13 s/token** from a RAM disk (2024-12-17) — [issue #186 comment](https://github.com/lyogavin/airllm/issues/186)
- Llama 3.1 405B (bnb-4bit) on one A100: "inference time for one sentence takes 20+ minutes" (2024-08-06) — [issue #166](https://github.com/lyogavin/airllm/issues/166)
- Dolphin-Mixtral-8x7B with 512 new tokens: still running after 160 minutes (2023-12-21) — [issue #84](https://github.com/lyogavin/airllm/issues/84)
- Kimi-K2.5 adapted by a user: "25分钟" (25 minutes) to get a result — [issue #330 comment](https://github.com/lyogavin/airllm/issues/330)

**Community reception**
- HN thread "AirLLM 70B inference with single 4GB GPU" (2026-08-03, 232 points). The top reply on speed cites the 292 s/token K3 figure. Another commenter computes 0.003 tok/s. Several readers had misread the README's unlabeled chart as tok/s rather than s/tok — [HN 49154228](https://news.ycombinator.com/item?id=49154228); [comment 49155253](https://news.ycombinator.com/item?id=49155253)
- One HN commenter asks the obvious comparison question: "What's the benefit of these projects over something like taking an unsloth quant and running llama.cpp with appropriate flags (-cmoe/-mmap)?" — [HN 49162451](https://news.ycombinator.com/item?id=49162451)
- In the 2023 HN thread about the 8 GB MacBook port, a commenter points to text-generation-webui users reporting it is "very slow" — [HN 38794505](https://news.ycombinator.com/item?id=38794505)
- Secondary blogs (2026) echo "20 seconds per token" for 70B on NVMe, and "batch work, not chat". These are not primary measurements — [runaihome.com](https://runaihome.com/blog/airllm-70b-4gb-vram-inference-guide-2026/); [dev.to](https://dev.to/umesh_malik/run-70b-llm-on-4gb-gpu-airllms-real-tradeoff-19oh)

### Inferences
- Effective throughput derived from the reports above:
  - #330: 62 GB split / 13.3 s ≈ 4.7 GB/s, from the page cache. This is about the best case and is memory-copy/H2D-bound, not disk-bound.
  - #364: ≈210 MB/s on NVMe/Windows, an overhead-bound case.
  - K3: if a token needs "~1GB" expanded expert weights per layer (PR text), that is ≈0.25-0.3 GB packed MXFP4 per layer. Over 92 layers that gives ≈25-30 GB per token, and 292 s implies ≈0.1 GB/s. This is far below NVMe sequential bandwidth. It suggests that 82,432 tiny synchronous per-expert reads (a fresh `safe_open` per expert, no prefetch, decompression per expert, per-layer GC) dominate, not raw SSD bandwidth.
- TTFT (prefill) costs one streaming pass over the model, the same I/O as a single decode token. Each additional token costs another full pass. So, for a P-token prompt and N output tokens, total I/O ≈ (1+N) × model bytes, and prefill compute is the only part that scales with P. AirLLM is therefore efficient only for prefill-heavy or scoring workloads (classification, multiple-choice log-likelihood, embedding, ranking) or large batches, where one weight pass is amortized over many sequences. It is unusable for interactive chat. That matches the author's own 2023 framing ("offline data analytics").
- The "3x speedup" compression claim is at best a 2023 HDD/slow-disk artifact. On fast storage, users measure it 3.8-8× *slower*.

### Gaps
- No primary, dated tok/s measurement for the flagship 70B-on-4GB claim on a stated NVMe disk exists in the repo, issues, or author posts.
- No measured throughput for DeepSeek-V3, Qwen3-235B, or Flash-Next inference (only VRAM).
- No published TTFT measurements separated from per-token decode, and no batch-size scaling data.
- r/LocalLLaMA threads could not be retrieved through the search tool. No Reddit numbers are included.

## 3. Supported backends: CUDA, Apple Silicon / MLX, CPU, others

### Takeaway
CUDA is the primary and best-tested path. On macOS, `AutoModel` always routes to a separate, hand-written MLX port of the Llama decoder block. That port has no prefetch, no compression in the merged code and no MoE, and it supports only `LlamaForCausalLM`/`MistralForCausalLM`. It rebuilds the module and reloads every layer from disk for every token. Measured MLX speeds are ~3 s/token (1.1B) to ~54 s/token (34B). A CPU path exists (2024), and device autodetect (CUDA → MPS → CPU) was added in 2026. ROCm and DirectML are unofficial and reported as effectively CPU-only and slow.

### Cited Findings
- macOS routing: "On macOS every model goes through AirLLMLlamaMlx, an MLX port of the Llama decoder block, so only checkpoints with that layout can run there." `MLX_ARCHITECTURES = {"LlamaForCausalLM", "MistralForCausalLM"}`. Since a PR merged on 2026-10-19, other architectures are rejected before download — [auto_model.py](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/auto_model.py); [PR #384](https://github.com/lyogavin/airllm/pull/384); [README MacOS](https://github.com/lyogavin/airllm#macos)
- Before that fix, macOS silently used the Llama/MLX backend for every architecture — [issue #371](https://github.com/lyogavin/airllm/issues/371)
- The MLX generate loop runs the same way for the prompt and for every new token. It constructs a fresh `TransformerBlock` for each layer, `update()`s it with weights loaded from disk through the MLX persister (`np.savez` files), runs `mx.eval`, then `del` + `gc.collect()`. The embedding, norm and lm_head are also reloaded per token. The KV cache is a Python list of per-layer (k, v) MLX arrays in unified memory — [airllm_llama_mlx.py L289-461](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/airllm_llama_mlx.py#L289-L461); [persist/mlx_model_persister.py](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/persist/mlx_model_persister.py)
- MLX support dates from v2.8.2 (2023-12-25): "Support MacOS running 70B large language models"; HN post "AirLLM enables 8GB MacBook run 70B LLM" (2023-12-28) — [README Updates](https://github.com/lyogavin/airllm#updates); [HN 38790385](https://news.ycombinator.com/item?id=38790385)
- An open PR (2026-05-08, not merged as of 2026-10-22) proposes module reuse, `mx.fast.rms_norm`/`scaled_dot_product_attention`, functional `mx.compile`, and native `mx.quantize` 4-bit. Its measurements: TinyLlama 1.1B 2.85 → 0.37 s/tok, Platypus2-7B 8.74 → 2.16 s/tok, with 4-bit shards 3.55× smaller — [PR #286](https://github.com/lyogavin/airllm/pull/286)
- A Metal crash report (2026-09-16): 67/67 layers at 34.04 s/it (38 min for one pass), then `kIOGPUCommandBufferCallbackErrorTimeout` abort on Apple Silicon — [issue #367](https://github.com/lyogavin/airllm/issues/367)
- MLX measured speeds: Yi-34B ≈54 s/token and Platypus2-7B ≈8 s/token (author notebook, 2023) — [run_on_macos.ipynb](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/examples/run_on_macos.ipynb). TinyLlama-1.1B on M1 Pro: 0.33 tok/s (2026-09) — [issue #364 comment](https://github.com/lyogavin/airllm/issues/364)
- CPU inference was added in v2.10.1 (2024-08-18), contributed by @NavodPeiris (fork `NavodPeiris/airllm-cpu-inference-support`) — [README Updates](https://github.com/lyogavin/airllm#updates); [fork](https://github.com/NavodPeiris/airllm-cpu-inference-support)
- `default_device()` (merged 2026-10-19) picks "CUDA, else Apple MPS, else CPU". It applies only to the PyTorch path; on macOS `AutoModel` still dispatches to MLX — [utils.py L88-101](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/utils.py#L88-L101); [PR #366](https://github.com/lyogavin/airllm/pull/366)
- Compression requires bitsandbytes, and dequant is hard-coded to `.cuda()` — [utils.py L104-132](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/utils.py#L104-L132)
- Windows ROCm (RX 9070 XT) and DirectML: "GPU utilization stays near 0%, and TinyLlama (1.1B) is extremely slow" — [issue #301](https://github.com/lyogavin/airllm/issues/301). A ROCm/Strix Halo enablement PR is open — [PR #373](https://github.com/lyogavin/airllm/pull/373)
- Kimi K3 needs flash-attn, so it needs CUDA 12 torch and `transformers` 4.56.x — [v3.1.0 release](https://github.com/lyogavin/airllm/releases/tag/v3.1.0)

### Inferences
- On Apple Silicon, AirLLM's model is irrelevant to the Kolibri use case on three counts. It cannot load Kolibri's architecture (MoE, hybrid SWA, non-Llama layout), it has no MoE logic, and its per-token full-layer reload ignores the main advantage of unified memory: the page cache and resident weights are the same RAM the GPU reads.
- On unified-memory Macs, "VRAM" savings are meaningless. The limiting resource is total RAM vs. SSD bandwidth, and AirLLM makes no attempt to use spare RAM as a cache.

### Gaps
- No published MLX measurement for any model ≥ 70B with stated hardware and SSD.
- No measurements of the PyTorch-MPS path (`device="mps"` via `AirLLMBaseModel` directly).

## 4. MoE support: whole layers vs. routed experts

### Takeaway
AirLLM runs Mixtral, DeepSeek V2/V3/R1, Qwen3-MoE and others through the generic `transformers` path, but it streams **whole decoder layers including all experts**. Since v3.1.0 (2026-07-29), true per-expert, on-demand loading exists only where the model exposes experts as an `nn.ModuleList` and the subclass sets `expert_prefix`. Today that is only Kimi K3. Per-expert loading is synchronous, has no prefetch or cache, and evicts each expert immediately after use. Packed/fused 3D expert tensors (e.g. Qwen3.8-Flash-Next) cannot use it.

### Cited Findings
- README model list: "Qwen (… including MoE, Flash-Next, FP8 …) · DeepSeek (V2 / V3 / R1) · Mistral & Mixtral … · Kimi K3". Its VRAM table includes "Qwen3-30B / Mixtral (MoE) ~1–3 GB", "Qwen3-235B (MoE) ~3 GB", "DeepSeek-V3 ~12 GB" — [README](https://github.com/lyogavin/airllm#supported-models)
- Mixtral support was added in v2.7 (2023-12-20) — [README Updates](https://github.com/lyogavin/airllm#updates)
- Per-expert streaming design: "A sparse MoE layer holds hundreds of experts but routes each token to a handful of them … Because the model calls each selected expert as its own module (and skips unselected ones), a forward hook per expert loads exactly the experts that actually run. This needs `expert_prefix` in layer_names_dict and safetensors shards" — [airllm_base.py L779-864](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/airllm_base.py#L779-L864)
- Expert pre-hook: `load_layer_subset(...)` opens the layer's safetensors file and reads only that expert's tensors, then moves them to the device. The post-hook moves them back to `meta`. There is no reuse across tokens or between the experts of consecutive tokens, and no prefetch — [airllm_base.py L866-876](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/airllm_base.py#L866-L876); [utils.py L431-441](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/utils.py#L431-L441)
- Only `AirLLMKimiK3` sets `'expert_prefix': 'block_sparse_moe.experts'`. No other subclass does, and the generic class used for DeepSeek/Qwen3-MoE/Mixtral does not — [airllm_kimi_k3.py](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/airllm_kimi_k3.py); [auto_model.py ARCH_OVERRIDES](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/auto_model.py)
- K3 numbers: "K3 holds 896 experts per layer and routes each token to 16 of them — expanded, a layer's experts are ~55GB, but a token only needs ~1GB." Across the model that is "82,432 experts across 92 layers". A K3 layer is "~17GB packed and ~56GB expanded, so streaming whole layers cannot work" — [v3.1.0 release](https://github.com/lyogavin/airllm/releases/tag/v3.1.0); [PR #316](https://github.com/lyogavin/airllm/pull/316)
- compressed-tensors' decompress-everything hook is removed because it "undoes per-expert streaming (a K3 layer becomes ~56GB)" — [airllm_base.py L356-362](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/airllm_base.py#L347-L362)
- Flash-Next (48-layer MoE, 512 experts, 10 routed + 1 shared): "Packed MoE experts are 3D tensors on a single module, not a `ModuleList`, so Kimi-style per-expert module hooks do not apply. A whole MoE layer is a few GB of bf16; that is what streams today." — [airllm_qwen4_exp.py docstring](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/airllm_qwen4_exp.py)
- The maintainer acknowledges that "MoE detection via `expert_prefix` misses Flash-Next (packed 3D expert tensors, no `ModuleList`)" — [PR #324 maintainer comment, 2026-10-19](https://github.com/lyogavin/airllm/pull/324)
- Per-expert streaming is disabled when `compression` is set — [PR merged 2026-10-19 #344](https://github.com/lyogavin/airllm/pull/344); [airllm_base.py L800-808](https://github.com/lyogavin/airllm/blob/22d72c46a67c9bf91da0615831fbb619e622c158/air_llm/airllm/airllm_base.py#L800-L808)
- Gemma-4-26B-A4B fails to load because of nested-prefix parsing (open issues) — [issue #345](https://github.com/lyogavin/airllm/issues/345); [issue #346](https://github.com/lyogavin/airllm/issues/346)

### Inferences
- The "DeepSeek-V3 on ~12 GB" figure is consistent with one whole FP8 MoE layer (all 256 routed experts) resident at a time. So decode bytes per token for DeepSeek-V3 ≈ the whole ~0.7 TB checkpoint. "Qwen3-235B on ~3 GB" is consistent with the FP8 variant's per-layer size. (Inferred from the layer-size arithmetic; the author does not state it.)
- For a model shaped like Kolibri (384 experts/layer, top-6, experts likely packed into 3D tensors in a modern `transformers` or GGUF layout), AirLLM would fall back to whole-layer streaming. It would read all 384 experts per layer per token, i.e. ~64× more expert bytes than needed for top-6.
- Even K3-style per-expert streaming lacks every feature that makes the Kolibri prototype work: a cache of hot experts, keeping attention and the shared expert resident, async or batched expert reads, and router-ahead prefetch. Its single data point (0.003 tok/s) shows that per-expert granularity without caching or batching of I/O is overhead-dominated.

### Gaps
- No measurement of Mixtral, DeepSeek or Qwen-MoE tok/s on v3/v4.
- No data on expert-hit locality, since AirLLM does not cache experts.

## 5. Project status (Oct 2026), forks, and similar "layered inference" tools

### Takeaway
After a ~21-month dormancy (v2.11.0 in Sep 2024 → v3.0.0 in Jun 2026), the project was revived. The 2026 work is single-author, Cursor-assisted, and focused on headline models (FP8, Kimi K3, Qwen3.8), plus streamed LoRA training (v4.0.0, 2026-09-05). As of 2026-10-22 the repo has ~40.1k stars, ~4.1k forks and 148 open issues+PRs, and the maintainer merged a batch of 11 community PRs on 2026-10-19. There is no test CI; the workflows cover only release and the star chart. Derived projects are small wrappers or forks. None has published real-model throughput better than AirLLM.

### Cited Findings
- Repo stats (GitHub API, 2026-10-22): 40,146 stars, 4,090 forks, 148 open issues+PRs, Apache-2.0, not archived, last push 2026-10-21 (daily automated star-history commits) — [GitHub repo](https://github.com/lyogavin/airllm)
- PyPI release history: 0.9.x (Nov 2023) → 2.0.0 (2023-12-02) → 2.8.x (Dec 2023) → 2.9-2.11.0 (Jul-Sep 2024) → 3.0.0 (2026-06-30), 3.0.1, 3.1.0 (2026-07-29, Kimi K3), 3.2.0 (2026-08-18, Qwen3.8-27B), 3.3.0 (2026-08-28, Flash-Next), 4.0.0 (2026-09-05, streamed LoRA) — [PyPI](https://pypi.org/project/airllm/#history); [GitHub releases](https://github.com/lyogavin/airllm/releases)
- v4.0.0 training: "Streamed LoRA: frozen weights stay on disk, adapters stay on GPU. Qwen3.8-Flash-Next trains under 6GB (RTX 3060 Ti)". It reports VRAM and loss curves only, with no step time — [v4.0.0 release](https://github.com/lyogavin/airllm/releases/tag/v4.0.0); [PR #359](https://github.com/lyogavin/airllm/pull/359)
- The 2026 author PRs are marked "Made with [Cursor]" — [PR #316](https://github.com/lyogavin/airllm/pull/316); [PR #353](https://github.com/lyogavin/airllm/pull/353)
- On 2026-10-19 the maintainer merged #370, #366, #344, #363, #375, #384, #329, #382, #385, #356 (fixes for MLX input types, rope_theta, macOS arch check, device autodetect, the expert-streaming/compression guard, and docs) — [commit log](https://github.com/lyogavin/airllm/commits/main)
- Open criticism and requests: unverifiable headline claims ([#295](https://github.com/lyogavin/airllm/issues/295)), false compression claim ([#330](https://github.com/lyogavin/airllm/issues/330)), a docs PR on compression trade-offs ([#334](https://github.com/lyogavin/airllm/pull/334)), proposals for double-buffered prefetch ([#298](https://github.com/lyogavin/airllm/issues/298)), Disk→RAM→pinned→GPU staging ([#269](https://github.com/lyogavin/airllm/issues/269)), full-load bypass ([#324](https://github.com/lyogavin/airllm/pull/324)), and a CI/benchmark harness ([#312](https://github.com/lyogavin/airllm/issues/312))
- HN perception (2026-08): "Seeing a lot of these 'run 1TB models with 1GB RAM' projects recently. Most seem vibe coded and probably won't be maintained." — [HN 49155378](https://news.ycombinator.com/item?id=49155378)
- The README carries affiliate "AI Agents Recommendation" links — [README](https://github.com/lyogavin/airllm)

**Forks / derived projects**
- `0xSojalSec/airllm` (~3.1k stars, a fork created 2025-09) is essentially a README mirror of upstream with the same claims. Its README shows no technical divergence — [repo](https://github.com/0xSojalSec/airllm)
- RabbitLLM (MIT, created 2026-02): an AirLLM fork with the same split → stream → prefetch design and optional bnb 4/8-bit. "only Qwen2 and Qwen3 are currently tested"; "Apple … is not supported" — [ManuelSLemos/RabbitLLM](https://github.com/ManuelSLemos/RabbitLLM)
- AirLLM++ (2026-02/04): groups layers into "super-blocks" (e.g. 4 layers), uses large sequential reads, prefetch depth ≥2 and an LRU block cache. Its only results are on **dummy data with mock compute** (2.44× at depth 2) — [howardleegeek/airllm-plus](https://github.com/howardleegeek/airllm-plus)
- Wrappers: OpenAI-compatible servers and chat UIs — [BretMcDanel/airllm-server](https://github.com/BretMcDanel/airllm-server); [mkamranr/airllm-openai-server](https://github.com/mkamranr/airllm-openai-server); [GhislainAdon/airllm-chat](https://github.com/GhislainAdon/airllm-chat). Also a runner repo for Kimi K3 — [anorakadrian/airllm-runners](https://github.com/anorakadrian/airllm-runners)
- CPU fork: `NavodPeiris/airllm-cpu-inference-support`, upstreamed in v2.10.1 — [repo](https://github.com/NavodPeiris/airllm-cpu-inference-support)

**Related offloading systems (different design point)**
- FlexGen (2023) is a throughput-oriented offloading scheduler (GPU/CPU/disk) with large batches. Its paper reports on a single T4: OPT-175B 0.69 tok/s vs. DeepSpeed ZeRO-Inference 0.01; OPT-30B 7.32 vs. 0.60; OPT-6.7B 25.26 vs. 9.28. Throughput = generated tokens / (prefill + decode time) with large effective batch — [FlexGen paper](https://arxiv.org/pdf/2303.06865)
- DeepSpeed ZeRO-Inference streams weights from CPU or NVMe layer by layer. Its NVMe/GDS blog reports LLaMA3-70B on one A100-80GB with 4 NVMe SSDs at ~7 tok/s with GDS (6 without), **at batch size 96** — [ZeRO-Inference blog](https://www.deepspeed.ai/2022/09/09/zero-inference.html); [DeepSpeed GDS blog](https://source-hub.etos-pro.ru/microsoft/DeepSpeed/src/branch/master/blogs/deepspeed-gds)
- An independent paper notes that FlexGen "is designed for high throughput, as opposed to low latency" — [TURNIP](https://arxiv.org/pdf/2405.16283)

### Inferences
- What separates AirLLM from FlexGen and ZeRO-Inference is that the latter amortize each weight pass over a large batch, which makes their aggregate tok/s respectable. AirLLM's API and examples are batch-1, so it gets the cost of weight streaming without the amortization.
- Relative to llama.cpp-style approaches (mmap, `--cpu-moe`, the Kolibri expert LRU cache), AirLLM is a different category: a minimal-VRAM pager, not a bandwidth-optimized MoE runtime.

### Gaps
- Not researched here for lack of tool budget: "LLM in a flash" (Apple), PowerInfer, KTransformers, llama.cpp `--cpu-moe`/mmap SSD-streaming numbers, and 2026 "run 1TB models with 1GB RAM" projects other than AirLLM forks (HN mentions several but names none I could verify).
- The Kaggle discussion page (SimJeg) did not render, so the original T4 timings for the competition notebook are not captured.

## 6. Fundamental cost model: bytes read per token vs. SSD bandwidth

### Takeaway
For AirLLM in its default (whole-layer) mode, decode time per token ≥ (bytes of all streamed weights) / (effective storage→device bandwidth) + per-layer overhead. Bytes per token ≈ the full model (minus a tied embedding). With per-expert mode, bytes ≈ all non-expert weights + L × k × expert_bytes, still with zero reuse. Prefill costs one such pass regardless of prompt length. Measured effective bandwidths range from ~0.1 GB/s (K3, per-expert) to ~4.7 GB/s (page-cache-resident 32B on GB10).

### Cited Findings
- The author's own framing is "Disk reading speed is actually the slowest bottleneck in the whole inference process" — [HF blog](https://huggingface.co/blog/lyogavin/airllm)
- Measured effective rates: ≈210 MB/s on NVMe/Windows for a 3B model ([#364](https://github.com/lyogavin/airllm/issues/364)); 62 GB split read per token in 13.3 s from the page cache on GB10 ([#330](https://github.com/lyogavin/airllm/issues/330)); K3 at 292 s/token "disk-bound" ([PR #316](https://github.com/lyogavin/airllm/pull/316))
- A user diagnosis of the MLX path: "throughput is roughly O(tokens × model size in bytes) regardless of backend" — [issue #364 comment](https://github.com/lyogavin/airllm/issues/364)
- Storage location dominates: 4-bit Qwen2.5-Coder-32B ran at ~2 min/token from an HDD vs. 13 s/token from a RAM disk — [issue #186](https://github.com/lyogavin/airllm/issues/186)
- K3 per-token expert need is "~1GB" expanded per layer vs. "~55GB" for all experts — [v3.1.0 release](https://github.com/lyogavin/airllm/releases/tag/v3.1.0)

### Inferences
These are arithmetic estimates under stated assumptions, not sourced measurements.
- **Whole-layer AirLLM on Kolibri.** Bytes per token = the checkpoint size: ~83 GB (Q8_0-equivalent) or ~156 GB (BF16).
  - At an assumed sustained 5 GB/s SSD read, that is ≈17 s/token (Q8) or ≈31 s/token (BF16), i.e. 0.03-0.06 tok/s.
  - At 10 GB/s it is ≈8-16 s/token.
  - The Kolibri prototype measures ~13 tok/s (≈0.077 s/token) on the same 83 GB Q8_0 file, which is roughly **200-400× faster** than the theoretical best case for AirLLM-style streaming.
- **Per-expert AirLLM on Kolibri (K3-style, no cache).** Kolibri has 50 layers × 384 experts = 19,200 experts. If the routed experts dominate an 83 GB Q8_0 file, an upper bound is ≈4.3 MB per expert.
  - Top-6 per layer then needs ≈50 × 6 × 4.3 MB ≈ 1.3 GB of expert reads per token, plus re-reading all attention and shared-expert weights every token, because AirLLM evicts them.
  - The prototype's 128/384 resident LRU cache with ~91% hit rate cuts expert reads to ≈0.12 GB/token and keeps the dense weights resident. That is roughly an order of magnitude fewer bytes per token than uncached per-expert streaming, and two to three orders fewer than whole-layer streaming.
- **I/O granularity matters as much as volume.** The K3 data point (≈0.1 GB/s effective) shows that thousands of small synchronous per-expert reads per token, each with Python and allocator overhead, underuse the SSD badly. Batched or parallel `pread`s (as in the prototype's F_NOCACHE path) are the relevant alternative.
- **Where AirLLM-style streaming is defensible.** Only when (a) the model cannot fit in RAM plus SSD cache at all, and (b) the workload is prefill/scoring-dominated or heavily batched. For Kolibri on a 48 GB Mac, a fully resident IQ3 quant (~60 tok/s) and the expert cache (~13-18 tok/s) both dominate it for interactive use.

### Gaps
- No sourced figure for M5 Pro internal SSD sustained random-read bandwidth was gathered here; the 5-10 GB/s values above are assumptions.
- Kolibri's per-expert byte size was not sourced. The 4.3 MB estimate is an upper bound derived from file size and expert count, not from the architecture's hidden size.
