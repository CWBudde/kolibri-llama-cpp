"""A whole-model Kolibri-1 forward in torch, independent of libllama.

The model is aleph_alpha_inference/kolibri1.py at
049a6a7bd2405b27d6d280d256bd3d585191c7ae and the vLLM v0.29.0 Qwen3Moe pieces it
inherits:

- Qwen3MoeModel: embedding, the decoder layers, the final norm;
- Kolibri1DecoderLayer: input_layernorm, attention, post_attn_norm, the fused
  residual add with post_attention_layernorm, the MoE block, post_ffn_norm, and
  the residual add in the next layer's (or the final) norm;
- Kolibri1Attention: qkv, per-head q/k RMSNorm, RoPE on the sliding-window
  layers only, attention with scale head_dim**-0.5 (window 513 on the sliding
  layers), o_proj;
- Kolibri1SparseMoeBlock: F32 router logits, sigmoid_logit_add_routing, SwiGLU
  experts, plus the ungated shared expert;
- the LM head in F32 (config head_dtype).

The weights come from a GGUF converted by convert_hf_to_gguf.py. For the real
checkpoint, check_real.py shows that its tensors are bit-exact to the
safetensors. The RoPE, attention and routing functions are the ports in
tools/gguf/check_attn.py and check_moe.py.

dtype "float32" or "float64" computes everything in that precision.
"bfloat16" emulates vLLM's BF16 inference on top of float32: matmul inputs and
outputs, norm outputs, the RoPE tables and the residual stream are rounded to
BF16, while the router logits and the LM head stay F32.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "gguf"))

from check_attn import attention, rope, rope_cos_sin_cache  # noqa: E402
from check_moe import sigmoid_logit_add_routing  # noqa: E402


class Weights:
    """The tensors and hyperparameters of a Kolibri GGUF, read lazily from its memory map."""

    def __init__(self, path: Path, llama_cpp: Path):
        sys.path.insert(0, str(llama_cpp / "gguf-py"))
        import gguf

        self.gguf = gguf
        reader = gguf.GGUFReader(path)
        self.t = {t.name: t for t in reader.tensors}
        f = {name: fld.contents() for name, fld in reader.fields.items()}
        arch = f["general.architecture"]
        self.n_layer = f[f"{arch}.block_count"]
        self.n_head = f[f"{arch}.attention.head_count"]
        self.n_head_kv = f[f"{arch}.attention.head_count_kv"]
        self.head_dim = f[f"{arch}.attention.key_length"]
        self.eps = f[f"{arch}.attention.layer_norm_rms_epsilon"]
        self.rope_base = f[f"{arch}.rope.freq_base"]
        self.n_expert_used = f[f"{arch}.expert_used_count"]
        self.norm_topk = f[f"{arch}.expert_weights_norm"]
        self.window = f[f"{arch}.attention.sliding_window"]
        self.is_swa = f[f"{arch}.attention.sliding_window_pattern"]
        self.has_rope = f[f"{arch}.attention.rope_pattern"]
        self.cache: dict | None = None  # a dict keeps the non-expert tensors across forward calls

    def rows(self, name: str, ids: list[int]):
        """Rows of a matrix (the embedding rows of tokens) as float32."""
        return self._f32(self.t[name], self.t[name].data[ids])

    def get(self, name: str, index: int | None = None):
        """A tensor as float32 in numpy order ([n_out, n_in] for a matrix), or one expert of a stacked one."""
        if index is None and self.cache is not None and name in self.cache:
            return self.cache[name]
        t = self.t[name]
        out = self._f32(t, t.data if index is None else t.data[index])
        if index is None and self.cache is not None:
            self.cache[name] = out
        return out

    def _f32(self, t, data):
        import numpy as np
        import torch

        if t.tensor_type == self.gguf.GGMLQuantizationType.BF16:
            bits = np.ascontiguousarray(data).view(np.uint16).astype(np.uint32) << 16
            return torch.from_numpy(bits.view(np.float32))
        if t.tensor_type == self.gguf.GGMLQuantizationType.F32:
            return torch.from_numpy(np.array(data, dtype=np.float32))
        raise ValueError(f"{t.name}: {t.tensor_type.name}, not BF16 or F32")


class Precision:
    def __init__(self, dtype: str):
        import torch

        self.bf16 = dtype == "bfloat16"
        self.dtype = torch.float64 if dtype == "float64" else torch.float32

    def round(self, x):
        """Rounds to BF16 in the bfloat16 mode."""
        import torch

        return x.to(torch.bfloat16).to(self.dtype) if self.bf16 else x

    def w(self, x):
        return x.to(self.dtype)

    def mm(self, x, w):
        """x @ w^T with BF16 inputs and output in the bfloat16 mode (vLLM: F32 accumulation)."""
        return self.round(self.round(x) @ self.w(w).T)


def rms_norm(p: Precision, x, weight, eps: float):
    """vLLM RMSNorm.forward_static without the residual: normalize in F32, cast, then scale."""
    import torch

    v = x.to(torch.float64 if p.dtype == torch.float64 else torch.float32)
    v = v * torch.rsqrt(v.pow(2).mean(dim=-1, keepdim=True) + eps)
    return p.round(p.round(v.to(p.dtype)) * p.w(weight))


def swiglu(p: Precision, x, w_gate, w_up, w_down):
    import torch

    return p.mm(p.round(torch.nn.functional.silu(p.mm(x, w_gate)) * p.mm(x, w_up)), w_down)


def forward(W: Weights, tokens: list[int], dtype: str = "float32", dump: dict | None = None):
    """Logits [len(tokens), n_vocab] as float32. dump, if given, receives per layer the
    libllama-named activations attn_post_norm-il, ffn_moe_out-il, ffn_shexp-il, l_out-il
    ([n_tokens, n_embd]), the selected experts ffn_moe_topk-il ([n_tokens, k]), the selection
    scores router_sel-il (logits + bias, [n_tokens, n_expert]) and kv_absmax-il (max |k|, max |v|)."""
    import torch

    p = Precision(dtype)
    n = len(tokens)
    pos = torch.arange(n)
    d = W.head_dim
    cos_sin = rope_cos_sin_cache(W.rope_base, d, n, torch.float32)
    if p.bf16:
        cos_sin = p.round(cos_sin.to(p.dtype))

    h = p.round(p.w(W.rows("token_embd.weight", tokens)))  # the residual stream
    for il in range(W.n_layer):
        blk = f"blk.{il}."
        x = rms_norm(p, h, W.get(blk + "attn_norm.weight"), W.eps)

        q = p.mm(x, W.get(blk + "attn_q.weight")).view(n, W.n_head, d)
        k = p.mm(x, W.get(blk + "attn_k.weight")).view(n, W.n_head_kv, d)
        v = p.mm(x, W.get(blk + "attn_v.weight")).view(n, W.n_head_kv, d)
        q = rms_norm(p, q, W.get(blk + "attn_q_norm.weight"), W.eps)
        k = rms_norm(p, k, W.get(blk + "attn_k_norm.weight"), W.eps)
        if W.has_rope[il]:
            q = p.round(rope(pos, q, cos_sin.to(p.dtype)))
            k = p.round(rope(pos, k, cos_sin.to(p.dtype)))
        a = p.round(attention(q, k, v, W.window if W.is_swa[il] else None))
        a = p.mm(a, W.get(blk + "attn_output.weight"))
        a = rms_norm(p, a, W.get(blk + "post_attention_norm.weight"), W.eps)
        h = p.round(h + a)

        x = rms_norm(p, h, W.get(blk + "ffn_norm.weight"), W.eps)
        router_in = p.round(x).to(torch.float32)
        logits = router_in @ W.get(blk + "ffn_gate_inp.weight").T  # F32, as GateLinear(out_dtype=float32)
        weights, topk = sigmoid_logit_add_routing(x, logits, W.n_expert_used, W.norm_topk,
                                                  W.get(blk + "exp_probs_b.bias"))
        moe = torch.zeros_like(x)
        for e in torch.unique(topk).tolist():
            rows, slot = (topk == e).nonzero(as_tuple=True)
            y = swiglu(p, x[rows], W.get(blk + "ffn_gate_exps.weight", e), W.get(blk + "ffn_up_exps.weight", e),
                       W.get(blk + "ffn_down_exps.weight", e))
            moe.index_add_(0, rows, y * weights[rows, slot, None].to(p.dtype))
        moe = p.round(moe)
        shexp = swiglu(p, x, W.get(blk + "ffn_gate_shexp.weight"), W.get(blk + "ffn_up_shexp.weight"),
                       W.get(blk + "ffn_down_shexp.weight"))
        m = rms_norm(p, p.round(moe + shexp), W.get(blk + "post_ffw_norm.weight"), W.eps)
        h = p.round(h + m)

        if dump is not None:
            dump[f"attn_post_norm-{il}"] = a.float().numpy()
            dump[f"ffn_moe_out-{il}"] = moe.float().numpy()
            dump[f"ffn_shexp-{il}"] = shexp.float().numpy()
            dump[f"l_out-{il}"] = h.float().numpy()
            dump[f"ffn_moe_topk-{il}"] = topk.numpy()
            dump[f"router_sel-{il}"] = (logits + W.get(blk + "exp_probs_b.bias")).numpy()
            dump[f"kv_absmax-{il}"] = torch.stack([k.abs().max(), v.abs().max()]).float().numpy()

    x = rms_norm(p, h, W.get("output_norm.weight"), W.eps)
    return (x.to(torch.float32) @ W.get("output.weight").T).float()  # head_dtype float32


def greedy(W: Weights, tokens: list[int], n: int, dtype: str = "float32") -> list[int]:
    """n greedy tokens after tokens, each recomputed over the whole sequence (no KV cache)."""
    out = list(tokens)
    for _ in range(n):
        out.append(int(forward(W, out, dtype)[-1].argmax()))
    return out[len(tokens):]
