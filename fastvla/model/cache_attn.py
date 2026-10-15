"""Reading the frozen VLM prefix KV cache: RoPE undo/redo, layer fusion, cross-attention.

The cache holds, for every layer of the DM0.5 action expert, the keys and values
of the prefix (instruction + first-frame images). Keys are stored *already
rotated* with that layer's RoPE, and DM0.5 (Gemma-3) alternates two frequency
bases across layers. Mixing keys across layers therefore requires rotating each
of them back to a canonical frame first and re-rotating the fused key with the
target layer's frequencies:

    K_j = RoPE_j( RMSNorm( Wk_j( sum_i a_ji * RoPE_i^{-1}(K_i) ) ) )
    V_j =                  Wv_j( sum_i a_ji * V_i )
"""
import torch
import torch.nn.functional as F
from torch import nn


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_vlm_rope(x, cos, sin):
    """x: [B, H, L, hd]; cos/sin: [B, L, hd]."""
    cos = cos.unsqueeze(1).to(x.dtype)
    sin = sin.unsqueeze(1).to(x.dtype)
    return x * cos + rotate_half(x) * sin


def undo_vlm_rope(x, cos, sin):
    """Inverse rotation (rotate by -theta)."""
    return apply_vlm_rope(x, cos, -sin)


def rope_at(rope, i):
    """(cos, sin) of layer i. ``rope`` is either one shared pair or a per-layer list."""
    if rope is None:
        return None
    return rope if torch.is_tensor(rope[0]) else rope[i]


def prepare_canonical_kv(keys, values, src_rope, dtype=torch.float32):
    """Undo the RoPE of every cached layer once per batch.

    Returns (K, V), each [n_cache, B, kv_heads, L, hd]. Computed in fp32 because the
    undo/redo rotations followed by a weighted sum accumulate phase error in bf16.
    """
    K = torch.stack([undo_vlm_rope(k.to(dtype), *(r.to(dtype) for r in rope_at(src_rope, i)))
                     for i, k in enumerate(keys)], 0)
    V = torch.stack([v.to(dtype) for v in values], 0)
    return K, V


def build_layer_map(n_dec, n_cache):
    """Which cache layer the j-th decoder layer is initially centred on."""
    if n_dec == 1:
        return [n_cache - 1]
    return [round(j * (n_cache - 1) / (n_dec - 1)) for j in range(n_dec)]


class KVFusion(nn.Module):
    """Mixes all cached layers into one K/V for a decoder layer.

    Mixing weights are softmax(alpha + router(mean_query)), where the router's last
    layer is zero-initialised and alpha is biased towards ``init_layer``, so training
    starts from reading a single, depth-aligned cache layer. Wk / Wv start as identity.
    """

    def __init__(self, n_cache, head_dim, d, eps=1e-6, init_layer=None, init_logit=8.0):
        super().__init__()
        self.n_cache, self.head_dim = n_cache, head_dim
        a0 = torch.zeros(n_cache)
        if init_layer is not None:
            a0[init_layer] = init_logit
        self.alpha = nn.Parameter(a0)
        self.Wk = nn.Linear(head_dim, head_dim, bias=False)
        self.Wv = nn.Linear(head_dim, head_dim, bias=False)
        nn.init.eye_(self.Wk.weight)
        nn.init.eye_(self.Wv.weight)
        self.k_norm_weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps
        self.router = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(),
                                    nn.Linear(d // 2, n_cache))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def mix_weights(self, z):
        """-> [B, n_cache]."""
        return F.softmax(self.alpha.unsqueeze(0) + self.router(z.mean(dim=1)), dim=-1)

    def _rms(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + self.eps)).to(x.dtype) * self.k_norm_weight

    def forward(self, K_canon, V_stack, dst_rope, z):
        """dst_rope: (cos, sin) of the target layer at the prefix positions."""
        dt = self.Wk.weight.dtype
        w = self.mix_weights(z).t().view(self.n_cache, -1, 1, 1, 1).to(dt)
        K = (K_canon.to(dt) * w).sum(0)
        V = (V_stack.to(dt) * w).sum(0)
        K = self._rms(self.Wk(K))
        K = apply_vlm_rope(K, *(r.to(dt) for r in dst_rope))
        return K, self.Wv(V)


class CacheXAttn(nn.Module):
    """Action queries attend to the (fused) VLM prefix cache.

    Queries are projected to the VLM head_dim and rotated with the VLM RoPE at
    positions ``effective_prefix_len + t`` so their phase matches the cached keys.
    GQA is handled by SDPA. With ``n_seg > 0`` a learned, zero-initialised vector is
    added to every key according to its segment (0 = text / chat template,
    1 = image soft token), telling the decoder where the language tokens are.
    """

    def __init__(self, d, head_dim=256, nhead=8, kv_heads=4, gate_init=1.0, n_seg=0):
        super().__init__()
        self.h, self.hd, self.kvh = nhead, head_dim, kv_heads
        self.n = nn.LayerNorm(d)
        self.q = nn.Linear(d, nhead * head_dim, bias=False)
        self.o = nn.Linear(nhead * head_dim, d, bias=False)
        self.gate = nn.Parameter(torch.full((1,), float(gate_init)))
        self.n_seg = n_seg
        self.seg_emb = nn.Parameter(torch.zeros(n_seg, head_dim)) if n_seg else None

    def forward(self, z, k, v, kv_pad, q_rope, kv_seg=None):
        """k/v: [B, kv_heads, L, hd]; kv_pad: [B, L] with True = visible."""
        B, L, _ = z.shape
        q = self.q(self.n(z)).view(B, L, self.h, self.hd).transpose(1, 2)
        q = apply_vlm_rope(q, *(r.to(q.dtype) for r in q_rope))
        k = k.to(q.dtype)
        if self.seg_emb is not None:
            assert kv_seg is not None, "model was built with segments but kv_seg is missing"
            k = k + self.seg_emb.to(q.dtype)[kv_seg].unsqueeze(1)
        mask = kv_pad[:, None, None, :]
        if mask.device.type == "npu":
            # Ascend's fused attention kernel only accepts an attn_mask of [B, N, Sq, Skv] /
            # [B, 1, Sq, Skv] / [1, 1, Sq, Skv] / [Sq, Skv], so the broadcast over the query
            # dim has to be materialised. Mathematically a no-op.
            mask = mask.expand(B, 1, L, mask.shape[-1]).contiguous()
        h = F.scaled_dot_product_attention(q, k, v.to(q.dtype), attn_mask=mask,
                                           enable_gqa=True)
        return z + self.gate * self.o(h.transpose(1, 2).reshape(B, L, -1))
