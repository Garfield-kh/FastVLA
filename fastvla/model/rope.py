"""Rotary embeddings and attention blocks of the action decoder.

Two unrelated RoPEs appear in the model and must not be mixed up:

* ``TimeRoPE`` (this file) lives in the decoder's own space
  (head_dim = d // nhead = 32). Positions are *frame indices* on a single
  signed time axis: history frames are negative, the current frame is 0 and
  the t-th predicted action is t.
* The VLM RoPE (``fastvla.model.cache_attn``) lives in the DM0.5 action-expert
  space (head_dim = 256, a different frequency base per layer). It is only used
  to read the frozen prefix KV cache.
"""
import torch
import torch.nn.functional as F
from torch import nn


def rotate_half(x):
    """HF / Gemma "half" convention: channel i is paired with channel i + d/2."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    """x: [B, H, L, hd]; cos/sin: [1, 1, L, hd]."""
    return x * cos + rotate_half(x) * sin


class TimeRoPE(nn.Module):
    """Lookup table of (cos, sin) for integer, non-negative positions."""

    def __init__(self, head_dim, max_pos=128, base=10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim, self.max_pos = head_dim, max_pos
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        f = torch.outer(torch.arange(max_pos).float(), inv)
        self.register_buffer("cos", torch.cat([f.cos(), f.cos()], -1), persistent=False)
        self.register_buffer("sin", torch.cat([f.sin(), f.sin()], -1), persistent=False)

    def at(self, pos):
        """pos: [L] long -> (cos, sin), each [1, 1, L, head_dim]."""
        assert pos.dtype == torch.long
        assert int(pos.max()) < self.max_pos, f"position {int(pos.max())} >= {self.max_pos}"
        return self.cos[pos][None, None], self.sin[pos][None, None]


class SelfAttnRoPE(nn.Module):
    """Pre-norm residual self-attention over the action queries, with TimeRoPE."""

    def __init__(self, d, nhead=8, rope=None):
        super().__init__()
        assert d % nhead == 0
        self.d, self.h, self.hd = d, nhead, d // nhead
        self.n = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.rope = rope

    def forward(self, z, pos=None):
        """z: [B, L, d]; pos: [L] time index of every query (not its flat index)."""
        B, L, _ = z.shape
        q, k, v = self.qkv(self.n(z)).view(B, L, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        if self.rope is not None:
            assert pos is not None and pos.shape[0] == L
            cos, sin = self.rope.at(pos)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        h = F.scaled_dot_product_attention(q, k, v)
        h = self.o(h.transpose(1, 2).reshape(B, L, self.d))
        return z + torch.nan_to_num(h)


class CrossAttnTimeRoPE(nn.Module):
    """Cross-attention from action queries to visual tokens on a shared time axis.

    Queries sit at position ``base + t`` (t-th future action) and visual tokens at
    ``base + offset`` (offset <= 0 is the frame the token was observed at), so
    q^T k depends only on (action time - observation time). Camera identity is
    *not* part of this axis; it is carried by the per-camera view embedding.
    ``qpos`` is an additional learned per-step position vector added to the queries.
    """

    def __init__(self, d, nhead=8, rope=None):
        super().__init__()
        assert d % nhead == 0 and rope is not None
        self.d, self.h, self.hd = d, nhead, d // nhead
        self.nq = nn.LayerNorm(d)
        self.nkv = nn.LayerNorm(d)
        self.wq = nn.Linear(d, d)
        self.wkv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)
        self.rope = rope

    def forward(self, q, kv, qpos=None, q_pos=None, kv_pos=None):
        B, Lq, _ = q.shape
        Lk = kv.shape[1]
        assert q_pos.shape[0] == Lq and kv_pos.shape[0] == Lk
        qn = self.nq(q)
        if qpos is not None:
            qn = qn + qpos
        qh = self.wq(qn).view(B, Lq, self.h, self.hd).transpose(1, 2)
        k, v = self.wkv(self.nkv(kv)).view(B, Lk, 2, self.h, self.hd).permute(2, 0, 3, 1, 4)
        cq, sq = self.rope.at(q_pos)
        ck, sk = self.rope.at(kv_pos)
        qh, k = apply_rope(qh, cq, sq), apply_rope(k, ck, sk)
        h = F.scaled_dot_product_attention(qh, k, v)
        return q + self.o(h.transpose(1, 2).reshape(B, Lq, self.d))


class QPosTable(nn.Module):
    """Learned per-step position vector re-added to the queries in every layer."""

    def __init__(self, chunk, d, n_arms=1):
        super().__init__()
        self.chunk, self.n_arms = chunk, n_arms
        self.emb = nn.Parameter(torch.zeros(chunk, d))

    def forward(self):
        """-> [chunk * n_arms, d], flattened as idx = t * n_arms + arm."""
        if self.n_arms == 1:
            return self.emb
        return self.emb.repeat_interleave(self.n_arms, dim=0)
