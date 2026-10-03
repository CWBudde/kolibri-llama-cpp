# Phase 1: Checkpoint and tensor inventory

Status: **done**. Every tensor needed for a forward pass is identified and
mapped to a GGUF name, and both checkpoints match `config.json` exactly.

## Pinned sources

| Source | Revision |
|---|---|
| `Aleph-Alpha/Kolibri-1-BF16` (HF) | `7a8f290e7858825c3cf5e4c447ba68345de9f1d3` |
| `Aleph-Alpha/Kolibri-1` (HF, FP8) | `e52eb4627d11516b0c01de49210ab5a4e4061444` |
| `Aleph-Alpha/aleph-alpha-inference` | `049a6a7bd2405b27d6d280d256bd3d585191c7ae` (release 1.0.0) |
| vLLM, for the inherited Qwen3-MoE and FusedMoE code | `v0.29.0` |
| `ggml-org/llama.cpp`, for GGUF names and existing semantics | `1537a0a8b2f8711d840878b0a0677ab2213c882c` |

`tokenizer.json` and `tokenizer_config.json` are byte-identical in both HF
repos. The `tokenizer.json` sha256 is `5d4798f2…4a4c13`. The full hashes of all
metadata files are in `inventory/*/summary.json`.

## Reproducing the inventory

```sh
go run ./cmd/kolibri-inventory -repo Aleph-Alpha/Kolibri-1-BF16 \
    -revision 7a8f290e7858825c3cf5e4c447ba68345de9f1d3 -out inventory/bf16
go run ./cmd/kolibri-inventory -repo Aleph-Alpha/Kolibri-1 \
    -revision e52eb4627d11516b0c01de49210ab5a4e4061444 -out inventory/fp8
go test ./...
```

The tool reads only the 8-byte length prefix and the JSON header of each
shard, using HTTP range requests (64 requests, a few seconds). It checks:

- each header against the index;
- every tensor's shape against `config.json`;
- that no tensor is missing or unexpected;
- that tensors tile each shard without gaps or overlaps;
- that the header-implied file sizes match the repo listing;
- that the index `total_size` matches the headers.

Both runs report **0 errors**, and repeated runs produce byte-identical output.
`TestCommittedInventory` re-checks the committed `tensors.jsonl.gz` files
against the current classification code.

`cmd/kolibri-peek` reads the payload of individual small tensors the same way.
With `-compare-repo`, it diffs a tensor against the other checkpoint, with FP8
dequantization applied.

## Totals

| | BF16 | FP8 |
|---|---|---|
| Shards | 32 | 32 |
| Tensors | 58,353 | 116,303 (58,353 + 57,950 scales) |
| Data bytes | 156,206,149,120 | 78,827,029,120 |
| Parameters | 78,103,074,560 | 78,103,074,560 |

Both parameter counts match the model card. The card's 3,457,573,120 active
parameters per token are reproduced as follows:

- all non-expert parameters except the embedding table: 2,277,922,560;
- plus 6 of 384 routed experts × 3 projections × 50 layers: 1,179,648,000;
- plus one embedding row: 2,560.

Each layer's tensors sit in at most 2 consecutive shards: 20 layers fit in one
shard and 30 span two. Layer order follows shard order. A converter that buffers
one layer's 1,152 expert tensors is enough. That buffer is about 3.0 GB at BF16.

## HF → GGUF mapping

Shapes in the HF column are in PyTorch order `[out, in]`. Shapes in the GGUF
column are in ggml `ne` order (innermost first). `L` is 0..49 and `E` is
0..383.

| Group | HF tensor | HF shape | GGUF tensor | GGUF ne | BF16 ckpt | FP8 ckpt |
|---|---|---|---|---|---|---|
| embedding | `model.embed_tokens.weight` | [128000, 2560] | `token_embd.weight` | [2560, 128000] | BF16 | BF16 |
| block norm | `model.layers.L.input_layernorm.weight` | [2560] | `blk.L.attn_norm.weight` | [2560] | BF16 | BF16 |
| attention | `model.layers.L.self_attn.q_proj.weight` | [6144, 2560] | `blk.L.attn_q.weight` | [2560, 6144] | BF16 | F8_E4M3 + scale [48, 20] |
| attention | `model.layers.L.self_attn.k_proj.weight` | [512, 2560] | `blk.L.attn_k.weight` | [2560, 512] | BF16 | F8_E4M3 + scale [4, 20] |
| attention | `model.layers.L.self_attn.v_proj.weight` | [512, 2560] | `blk.L.attn_v.weight` | [2560, 512] | BF16 | F8_E4M3 + scale [4, 20] |
| QK norm | `model.layers.L.self_attn.q_norm.weight` | [128] | `blk.L.attn_q_norm.weight` | [128] | BF16 | BF16 |
| QK norm | `model.layers.L.self_attn.k_norm.weight` | [128] | `blk.L.attn_k_norm.weight` | [128] | BF16 | BF16 |
| attention | `model.layers.L.self_attn.o_proj.weight` | [2560, 6144] | `blk.L.attn_output.weight` | [6144, 2560] | BF16 | F8_E4M3 + scale [20, 48] |
| block norm | `model.layers.L.post_attn_norm.weight` | [2560] | `blk.L.post_attention_norm.weight` | [2560] | BF16 | BF16 |
| block norm | `model.layers.L.post_attention_layernorm.weight` | [2560] | `blk.L.ffn_norm.weight` | [2560] | BF16 | BF16 |
| router | `model.layers.L.mlp.gate.weight` | [384, 2560] | `blk.L.ffn_gate_inp.weight` | [2560, 384] | BF16 | BF16 (not converted) |
| router | `model.layers.L.moe.router.expert_bias` | [384] | `blk.L.exp_probs_b.bias` | [384] | BF16 | BF16 |
| routed expert | `model.layers.L.mlp.experts.E.gate_proj.weight` | [512, 2560] ×384 | `blk.L.ffn_gate_exps.weight` | [2560, 512, 384] | BF16 | F8_E4M3 + scale [4, 20] |
| routed expert | `model.layers.L.mlp.experts.E.up_proj.weight` | [512, 2560] ×384 | `blk.L.ffn_up_exps.weight` | [2560, 512, 384] | BF16 | F8_E4M3 + scale [4, 20] |
| routed expert | `model.layers.L.mlp.experts.E.down_proj.weight` | [2560, 512] ×384 | `blk.L.ffn_down_exps.weight` | [512, 2560, 384] | BF16 | F8_E4M3 + scale [20, 4] |
| shared expert | `model.layers.L.mlp.shared_experts.gate_proj.weight` | [512, 2560] | `blk.L.ffn_gate_shexp.weight` | [2560, 512] | BF16 | F8_E4M3 + scale [4, 20] |
| shared expert | `model.layers.L.mlp.shared_experts.up_proj.weight` | [512, 2560] | `blk.L.ffn_up_shexp.weight` | [2560, 512] | BF16 | F8_E4M3 + scale [4, 20] |
| shared expert | `model.layers.L.mlp.shared_experts.down_proj.weight` | [2560, 512] | `blk.L.ffn_down_shexp.weight` | [512, 2560] | BF16 | F8_E4M3 + scale [20, 4] |
| block norm | `model.layers.L.post_ffn_norm.weight` | [2560] | `blk.L.post_ffw_norm.weight` | [2560] | BF16 | BF16 |
| block norm | `model.norm.weight` | [2560] | `output_norm.weight` | [2560] | BF16 | BF16 |
| LM head | `lm_head.weight` | [128000, 2560] | `output.weight` | [2560, 128000] | BF16 | BF16 |

This table mirrors `internal/kolibri/tensors.go` (`Specs`), which is the
source of truth. The per-class numbers are in `inventory/*/summary.json`.

## Answers to the Phase 1 questions

### Packed or per-expert?

**Per-expert.** There are 19,200 separate tensors for each of gate, up, and
down: 50 layers × 384 experts. Gate and up are stored separately, not fused.
The converter must stack the 384 experts of each layer along a new outermost
axis, as `Qwen2MoeModel.modify_tensors` already does.

### Block structure and residual order

The structure comes from `Kolibri1DecoderLayer` with vLLM's fused add-RMSNorm.
It is a sandwich-norm block like Gemma2 and OLMo2:

```
x  = x + RMSNorm_post_attn( Attn( RMSNorm_input(x) ) )
x  = x + RMSNorm_post_ffn ( MoE ( RMSNorm_pre_ffn(x) ) )
...
logits = LMHead( RMSNorm_final(x) )                      # head in FP32, see below
MoE(h) = Routed(h) + Shared(h)
```

- All norms are plain `RMSNorm(x) * w` with eps 1e-6.
- The weights are stored as `w`, not `1 + w`. vLLM uses `RMSNorm`, not
  `GemmaRMSNorm`. Sampled values agree:
  - `post_attn_norm` weights in layer 0 have a mean of 0.034;
  - `post_ffn_norm` weights in layer 49 have a mean of 8.5;
  - `model.norm` weights have a mean of 42.6.

  The converter must therefore **not** add 1 to these weights, as the Gemma
  converters do.
- The shared expert is **ungated**: there is no `shared_expert_gate`. It runs on
  the same normalized input as the routed experts.
- vLLM's `MoERunner` returns `shared_output + fused_output` with
  `routed_scaling_factor = 1.0`. `post_ffn_norm` is then applied to that sum.

### Router semantics

From `sigmoid_logit_add_routing` and `test_routing_semantics`:

1. `logits = h @ W_gate^T`, computed in FP32 (`GateLinear(out_dtype=float32)`).
2. Experts are selected as `top6(logits + expert_bias)`. The bias is added to the
   **raw logits**, not to `sigmoid(logits)`.
3. Each selected expert's weight is `sigmoid(logits[selected])`. The bias does
   not enter the weights.
4. There is no renormalization (`norm_topk_prob = false`) and no scale
   (`route_scale = 1.0`).

`expert_bias` is stored as BF16 in both checkpoints. Sampled ranges:

- layer 0: [-7.6, 1.8];
- layer 4: [-2.0, 3.0];
- layer 49: [-1.2, 1.0].

The reference test mentions magnitudes of up to about 20.

Attention details from `Kolibri1Attention`:

- Q, K, and V have no bias.
- `q_norm` and `k_norm` are RMSNorms over `head_dim = 128`, applied per head
  and **before** RoPE.
- Sliding layers use NEOX-style RoPE (`get_rope` default) over the full
  `head_dim`, with theta 10000.
- Full-attention layers use **no positional encoding** (RNoPE).
- The attention scale is 1/√128.
- The full-attention layers are 4, 9, …, 49: a period of 5 with the full layer
  last. This matches llama.cpp `set_swa_pattern(5)`.
- `sliding_window = 513` in `config.json`. vLLM passes this to FlashAttention
  as `window = (512, 0)`, so a token sees the 512 preceding tokens plus itself.
  llama.cpp `LLAMA_SWA_TYPE_STANDARD` masks when `p1 - p0 >= n_swa`, so storing
  **`n_swa = 513` unchanged** gives the same mask.

LM head: `config.json` has `head_dtype: "float32"`. In vLLM 0.29,
`LogitsProcessor` honors it: it casts the BF16 `lm_head` weight and the hidden
state to FP32, or accumulates in FP32 on CUDA. The weight itself is stored as
BF16. In llama.cpp, the output matmul should run at `GGML_PREC_F32`.

### BF16 vs FP8 differences

- FP8 uses `quant_method = fp8`, dynamic activations, and 128×128 weight
  blocks. Scales are stored as F32 `weight_scale_inv` with shape
  `[ceil(out/128), ceil(in/128)]`. Weights are `F8_E4M3` (e4m3fn).
- FP8 is applied to all 7 linear projection types: q, k, v, o, and the
  gate/up/down projections of both the routed and shared experts.
- These stay BF16: the router (`mlp.gate`, listed in `modules_to_not_convert`),
  `expert_bias`, all norms, the embedding table, and `lm_head`.
- Unquantized tensors are **bit-identical** between the two repos. The sampled
  tensors were the router, `expert_bias`, `q_norm`, and `post_ffn_norm`.
- Dequantized FP8 weights differ from BF16 by about **2.6 % relative RMSE**.
  Sampled tensors: k_proj in layer 0, expert 200 down_proj in layer 25, and the
  shared gate_proj in layer 49. This matches the error expected from E4M3's
  3-bit mantissa. The FP8 repo is a quantization of the BF16 repo, not a
  separately trained model.
- llama.cpp's `ModelBase.dequant_model` already handles `weight_scale_inv` with
  `weight_block_size`. A direct FP8-source conversion is therefore possible
  later, but BF16 stays the source of truth (see PLAN.md).
- With FP8 you download 78.9 GB instead of 156.2 GB. But the resulting GGUF is
  2.6 % RMSE away from BF16 before any llama.cpp quantization is applied.

## Pitfalls for Phase 3/4 found during the audit

1. **Norm names collide with gguf-py's generic `TensorNameMap`.**
   - `post_attn_norm` maps to `ATTN_OUT_NORM` (via grok-2). Kolibri needs
     `ATTN_POST_NORM`.
   - `post_attention_layernorm` is listed under both `FFN_NORM` (Qwen, correct
     here) and `ATTN_POST_NORM` (Gemma2/OLMo2, wrong here).

   The Kolibri converter must map all four block norms explicitly and not rely
   on `map_tensor_name`.
2. **The router bias is applied differently.** In `build_moe_ffn`,
   `exp_probs_b` is added to `probs = sigmoid(logits)`, which is DeepSeek-V3
   semantics. Kolibri needs `selection = logits + exp_probs_b` with
   `weights = sigmoid(logits)`. Because sigmoid is monotonic, the two choices
   pick the same experts only when the bias is zero, and Kolibri's bias is not
   zero. Phase 4 therefore needs an arch-specific selection branch, like the
   existing special cases for `LLM_ARCH_LLAMA4` and `LLM_ARCH_GROVEMOE`.
3. **`exp_probs_b` precision.** The source is BF16, but GGUF should store it as
   F32 (vLLM keeps it as an FP32 parameter). Router logits should also be
   computed in F32.
4. **The FP32 LM head** (see above).
5. **The tokenizer** (input to Phase 2):
   - type: byte-level `BPE` with `byte_fallback: true`;
   - 127,900 base vocab entries plus 98 added tokens;
   - 127,644 merges, stored as pairs;
   - no normalizer;
   - pre-tokenizer: `Split` regex followed by `ByteLevel(use_regex=false)`; the
     regex splits single digits (`\p{N}{1}`).

   The file's `added_tokens` skip IDs 127923 and 127924, but the tokenizers
   library ignores those IDs and numbers added tokens contiguously. At runtime
   `<|reserved-token-2|>` … `<|reserved-token-76|>` therefore sit two IDs
   lower than the file says, and **127998 and 127999 are the unassigned IDs**.
   The summary's `runtime_id` and `runtime_unassigned` fields record this; see
   `docs/phase2-tokenizer.md`. (An earlier version of this note named
   127923/127924, the file IDs.)
   `<think>`, `</think>`, `<tool_call>`, and `<tool_response>` are added tokens
   with `special: false`. Other details:

   - EOS is 127906 `<|im_end|>`;
   - the generation config also stops on 127901 `<|endoftext|>`, which is also
     the pad token;
   - there is no BOS token and `add_bos_token` is false.
6. **The chat template** in `tokenizer_config.json` equals the reference repo's
   `tests/kolibri1_chat_template.jinja`, apart from that file's header comment.

## Machine-readable outputs

- `inventory/{bf16,fp8}/summary.json`: pinned revisions, file hashes, shard
  sizes and LFS sha256, per-class dtype/shape/byte totals, the attention layout,
  the tokenizer summary, and the check results.
- `inventory/{bf16,fp8}/tensors.jsonl.gz`: one record per tensor, with name,
  shard, dtype, shape, data offsets, class, role, layer, expert, and GGUF target.
- `inventory/{bf16,fp8}/{config,generation_config,tokenizer_config}.json`:
  verbatim copies at the pinned revisions. They are Apache-2.0.
