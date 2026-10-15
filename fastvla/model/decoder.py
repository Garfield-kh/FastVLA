"""The action decoder stack.

Each layer runs, in this order: self-attention over the action queries ->
cross-attention to the current visual tokens -> cross-attention to the VLM
prefix cache -> FFN. The per-arm local residual (wrist CLS + proprioception) is
applied by the model before every layer.
"""
from torch import nn

from .cache_attn import CacheXAttn, KVFusion, build_layer_map, rope_at
from .rope import CrossAttnTimeRoPE, SelfAttnRoPE


class FFNBlock(nn.Module):
    def __init__(self, d, mult=4):
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.f = nn.Sequential(nn.Linear(d, d * mult), nn.GELU(), nn.Linear(d * mult, d))

    def forward(self, x):
        return x + self.f(self.n(x))


class DecoderLayer(nn.Module):
    def __init__(self, d, nhead, n_cache, head_dim, kv_heads, tgt_layer,
                 time_rope, obs_rope, n_cache_heads=8, n_seg=0):
        super().__init__()
        self.tgt_layer = tgt_layer
        self.self_attn = SelfAttnRoPE(d, nhead, rope=time_rope)
        self.vis_xattn = CrossAttnTimeRoPE(d, nhead, rope=obs_rope)
        self.cache_xattn = CacheXAttn(d, head_dim, n_cache_heads, kv_heads, n_seg=n_seg)
        self.ffn = FFNBlock(d)
        self.fusion = KVFusion(n_cache, head_dim, d=d, init_layer=tgt_layer)

    def forward(self, z, vis, kv_pad, prefix_rope, query_rope, time_pos, qpos,
                K_canon, V_stack, q_time, vis_time, kv_seg=None):
        z = self.self_attn(z, pos=time_pos)
        z = self.vis_xattn(z, vis, qpos=qpos, q_pos=q_time, kv_pos=vis_time)
        k, v = self.fusion(K_canon, V_stack, rope_at(prefix_rope, self.tgt_layer), z=z)
        z = self.cache_xattn(z, k, v, kv_pad, rope_at(query_rope, self.tgt_layer),
                             kv_seg=kv_seg)
        return self.ffn(z)


class Decoder(nn.Module):
    def __init__(self, d, nhead, n_dec, n_cache, head_dim, kv_heads, time_rope,
                 obs_rope, n_cache_heads=8, n_seg=0):
        super().__init__()
        self.layer_map = build_layer_map(n_dec, n_cache)
        self.layers = nn.ModuleList([
            DecoderLayer(d, nhead, n_cache, head_dim, kv_heads, self.layer_map[j],
                         time_rope, obs_rope, n_cache_heads, n_seg=n_seg)
            for j in range(n_dec)])
