"""FastVLA: a light action decoder that reads a frozen VLM's prefix KV cache.

The VLM (DM0.5) runs once per episode on the instruction and the first frame.
Its per-layer prefix keys/values ("task memory") are kept fixed for the whole
episode. At every control step only this decoder runs: it looks at the current
camera frames through per-camera DINOv2 encoders, reads the task memory through
cross-attention, and emits an action chunk for every arm.

Query layout (used everywhere, never changed): a chunk of T steps for A arms is
flattened as ``idx = t * A + a``.
"""
import torch
from torch import nn

from .cache_attn import prepare_canonical_kv
from .decoder import Decoder
from .rope import QPosTable, TimeRoPE
from .vision import MultiViewDino

DIM_PER_ARM = 7          # 6 joints / end-effector dims + 1 gripper
N_CACHE_SEG = 2          # 0 = text / template, 1 = image soft token


class QueryGrid(nn.Module):
    """q[t, a] = time_emb[t] + arm_emb[a], flattened to [T * A, d]."""

    def __init__(self, chunk, n_arms, d, std=0.02):
        super().__init__()
        self.chunk, self.n_arms, self.d = chunk, n_arms, d
        self.time_emb = nn.Parameter(torch.randn(chunk, d) * std)
        self.arm_emb = nn.Parameter(torch.randn(n_arms, d) * std)

    def forward(self, batch):
        q = self.time_emb[:, None, :] + self.arm_emb[None, :, :]
        return q.reshape(self.chunk * self.n_arms, self.d)[None].expand(batch, -1, -1)

    def time_index(self, device=None):
        """Time step of every flattened query; both arms at step t get t."""
        return torch.arange(self.chunk, device=device).repeat_interleave(self.n_arms)


class LocalCondition(nn.Module):
    """wrist CLS + proprioception + role embedding -> one conditioning vector."""

    def __init__(self, d, wrist_dim, state_dim, n_roles=4):
        super().__init__()
        self.wrist_proj = nn.Linear(wrist_dim, d)
        self.state_proj = nn.Linear(state_dim, d)
        self.role_emb = nn.Embedding(n_roles, d)
        nn.init.normal_(self.role_emb.weight, std=0.02)
        self.mlp = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, wrist_feat, state, role):
        return self.mlp(torch.cat([self.wrist_proj(wrist_feat),
                                   self.state_proj(state),
                                   self.role_emb(role)], -1))


class ArmLocalCondition(nn.Module):
    """One LocalCondition per arm: [B, A, v], [B, A, s], [B, A] -> [B, A, d]."""

    def __init__(self, d, wrist_dim, state_dim_per_arm, n_arms, n_roles=4):
        super().__init__()
        self.n_arms = n_arms
        self.blocks = nn.ModuleList(
            [LocalCondition(d, wrist_dim, state_dim_per_arm, n_roles) for _ in range(n_arms)])

    def forward(self, w_cls, state, role):
        return torch.stack([self.blocks[a](w_cls[:, a], state[:, a], role[:, a])
                            for a in range(self.n_arms)], dim=1)


class ArmLocalResidual(nn.Module):
    """Adds each arm's conditioning vector to that arm's queries (gated residual MLP)."""

    def __init__(self, d, hidden_mult=2, gate_init=1.0):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(2 * d, hidden_mult * d), nn.GELU(),
                                 nn.Linear(hidden_mult * d, d))
        self.gate = nn.Parameter(torch.full((1,), float(gate_init)))

    def forward(self, z, c_local, chunk, n_arms):
        """z: [B, T*A, d]; c_local: [B, A, d]."""
        B, L, d = z.shape
        zc, cc = z.to(self.norm.weight.dtype), c_local.to(self.norm.weight.dtype)
        c = cc[:, None].expand(B, chunk, n_arms, d).reshape(B, L, d)
        h = self.mlp(torch.cat([self.norm(zc), c], -1))
        return z + (self.gate * h).to(z.dtype)


def split_state_by_arm(state, n_arms):
    """[B, D] -> [B, A, D/A]; RoboTwin state is [left 7 | right 7]."""
    B, D = state.shape
    return state.view(B, n_arms, D // n_arms)


def assemble_action(pred, chunk, n_arms, valid_per_arm):
    """[B, T*A, out_dim] -> [B, T, A * valid_per_arm]."""
    B, L, pad = pred.shape
    z = pred.view(B, chunk, n_arms, pad)[..., :valid_per_arm]
    return z.reshape(B, chunk, n_arms * valid_per_arm)


class FastVLA(nn.Module):
    """Action decoder over a frozen VLM prefix cache.

    Args:
        cache_spec: dict(n_layers, kv_heads, head_dim) of the cached VLM layers.
        cam_of_slot / slot_time: camera index and frame offset (<= 0) of every image
            slot. The first ``n_view`` slots are the current frames of all cameras;
            history slots are appended after them.
        state_dim: total proprioception size (split evenly across arms).
        action_dim: width of the output head; the first 7 dims of every arm are used.
    """

    def __init__(self, cache_spec, d=256, nhead=8, n_dec=4, n_cache_heads=8, chunk=50,
                 n_view=3, n_arms=2, state_dim=14, action_dim=32, n_roles=4,
                 backbone="facebook/dinov2-base", cam_of_slot=None, slot_time=None,
                 cache_seg=True, pretrained_backbone=True):
        super().__init__()
        assert state_dim % n_arms == 0
        self.chunk, self.n_view, self.n_arms = chunk, n_view, n_arms
        self.cam_of_slot = tuple(range(n_view) if cam_of_slot is None else cam_of_slot)
        self.n_slot = len(self.cam_of_slot)
        self.slot_time = tuple([0] * self.n_slot if slot_time is None else slot_time)
        assert len(self.slot_time) == self.n_slot
        assert all(o <= 0 for o in self.slot_time), "slot offsets must be <= 0 (no future frames)"
        assert tuple(self.slot_time[:n_view]) == (0,) * n_view
        self.time_base = -min(self.slot_time)

        self.dino = MultiViewDino(n_view, backbone, cam_of_slot=self.cam_of_slot,
                                  pretrained=pretrained_backbone)
        v = self.dino.dim
        # Projection and view embedding are per camera; slot_emb tells frames of the
        # same camera apart in time.
        self.view_proj = nn.ModuleList([nn.Linear(v, d) for _ in range(n_view)])
        self.view_emb = nn.Parameter(torch.zeros(n_view, d))
        self.slot_emb = nn.Parameter(torch.zeros(self.n_slot, d))

        self.arm_local = ArmLocalCondition(d, v, state_dim // n_arms, n_arms, n_roles)
        self.query = QueryGrid(chunk, n_arms, d)
        self.local_res = nn.ModuleList([ArmLocalResidual(d) for _ in range(n_dec)])

        self.time_rope = TimeRoPE(d // nhead, max_pos=max(128, 2 * chunk))
        self.qpos = QPosTable(chunk, d, n_arms=n_arms)
        self.obs_rope = TimeRoPE(d // nhead, max_pos=max(128, 2 * (self.time_base + chunk)))

        self.cache_seg = cache_seg
        self.dec = Decoder(d, nhead, n_dec, cache_spec["n_layers"], cache_spec["head_dim"],
                           cache_spec["kv_heads"], time_rope=self.time_rope,
                           obs_rope=self.obs_rope, n_cache_heads=n_cache_heads,
                           n_seg=(N_CACHE_SEG if cache_seg else 0))
        self.head = nn.Linear(d, action_dim)

    def visual(self, imgs):
        tok, cls = self.dino(imgs)
        vis = torch.cat([self.view_proj[self.cam_of_slot[s]](tok[:, s])
                         + self.view_emb[self.cam_of_slot[s]]
                         + self.slot_emb[s]
                         for s in range(self.n_slot)], 1)
        return vis, cls, tok.shape[2]

    def arm_cls(self, cls):
        """CLS of each arm's wrist camera (current frame). Camera 0 is the global view;
        with one arm the wrist is camera 1, with two arms cameras 1 / 2."""
        return torch.stack([cls[:, 1 + a] for a in range(self.n_arms)], 1)

    def forward(self, keys, values, kv_pad, prefix_rope, query_rope,
                imgs, state, role, kv_seg=None, **_):
        """
        keys/values: tuple over cached layers of [B, kv_heads, L, head_dim].
        kv_pad: [B, L] bool, True = valid prefix token.
        prefix_rope / query_rope: per-layer (cos, sin) at prefix / query positions.
        imgs: [B, n_slot, 3, H, W] in [0, 1]; state: [B, state_dim] (normalised);
        role: [B, n_arms] long. Returns normalised actions [B, chunk, n_arms * 7].
        """
        B = imgs.shape[0]
        vis, cls, per_view_L = self.visual(imgs)
        c_local = self.arm_local(self.arm_cls(cls), split_state_by_arm(state, self.n_arms), role)
        K_canon, V_stack = prepare_canonical_kv(keys, values, prefix_rope)

        dev = imgs.device
        time_pos = self.query.time_index(dev)
        qpos = self.qpos()
        q_time = time_pos + self.time_base
        vis_time = (torch.tensor(self.slot_time, device=dev, dtype=torch.long)
                    + self.time_base).repeat_interleave(per_view_L)

        z = self.query(B)
        for j, layer in enumerate(self.dec.layers):
            z = self.local_res[j](z, c_local, self.chunk, self.n_arms)
            z = layer(z, vis, kv_pad, prefix_rope, query_rope, time_pos=time_pos, qpos=qpos,
                      K_canon=K_canon, V_stack=V_stack, q_time=q_time, vis_time=vis_time,
                      kv_seg=kv_seg)
        return assemble_action(self.head(z), self.chunk, self.n_arms, DIM_PER_ARM)
