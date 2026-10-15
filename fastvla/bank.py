"""Offline first-frame cache bank: reading and writing.

A bank is one or more ``shard_*.pt`` files (or a single ``.pt``), each a dict:

    kv          [n, n_layers, 2, kv_heads, L, head_dim]   bf16, or fp8 (e4m3) + kv_scale
    kv_scale    [n, n_layers, 2, kv_heads]                 (fp8 only) per-head scale
    attn_mask   [n, L] bool     eff_len [n] long     input_ids [n, L] long
    ep_ids      list[str]       instructions list[str]   L int

Rows are always looked up by episode id, never by row index.
"""
import glob
import os

import torch

from .vlm import cache_kwargs

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def fp8_quantize(x):
    """[..., h, L, hd] -> (fp8, scale[..., h]); one scale per head."""
    amax = x.abs().amax(dim=(-2, -1), keepdim=True).float()
    scale = (amax / FP8_MAX).clamp_min(1e-12)
    q = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q, scale.squeeze(-1).squeeze(-1)


def _shard_paths(path):
    if os.path.isdir(path):
        out = sorted(glob.glob(os.path.join(path, "shard_*.pt")))
        assert out, f"no shard_*.pt in {path}"
        return out
    return [path]


class BankReader:
    """Memory-maps a bank and assembles the model's cache kwargs for a batch."""

    def __init__(self, bank_path, rope, n_image_tokens, mmap=True):
        self.shards = [torch.load(p, map_location="cpu", weights_only=False, mmap=mmap)
                       for p in _shard_paths(bank_path)]
        self.fp8 = "kv_scale" in self.shards[0]
        self.row, self.ep_ids, self.instructions = {}, [], {}
        for si, d in enumerate(self.shards):
            assert ("kv_scale" in d) == self.fp8, "mixed fp8 / bf16 shards"
            for r, e in enumerate(d["ep_ids"]):
                assert e not in self.row, f"episode id {e!r} appears in two shards"
                self.row[e] = (si, r)
                self.ep_ids.append(e)
            if "instructions" in d:
                self.instructions.update(zip(d["ep_ids"], d["instructions"]))
        Ls = {int(d["L"]) for d in self.shards}
        assert len(Ls) == 1, f"shards have different L: {sorted(Ls)}"
        self.L = Ls.pop()
        self.rope = rope
        self.n_image_tokens = n_image_tokens

    def __len__(self):
        return len(self.ep_ids)

    def spec(self):
        _, n_l, _, kvh, _, hd = self.shards[0]["kv"].shape
        return dict(n_layers=int(n_l), kv_heads=int(kvh), head_dim=int(hd))

    def check_instructions(self, instr):
        """The instruction encoded in the cache must be the one the dataset reports."""
        common = set(instr) & set(self.instructions)
        bad = [e for e in common if instr[e] != self.instructions[e]]
        assert not bad, (f"{len(bad)} instructions differ between dataset and bank, "
                         f"e.g. {bad[0]}: {instr[bad[0]]!r} vs {self.instructions[bad[0]]!r}")

    def gather_rows(self, episode_ids):
        """CPU-side gather in request order (safe to call from DataLoader workers)."""
        miss = [e for e in episode_ids if e not in self.row]
        assert not miss, f"episodes missing from the bank: {miss[:3]}"
        where = [self.row[e] for e in episode_ids]
        k = len(where)
        kv = scale = am = eff = ids = None
        for si, d in enumerate(self.shards):
            dst = [j for j, (s, _) in enumerate(where) if s == si]
            if not dst:
                continue
            rows = torch.tensor([where[j][1] for j in dst], dtype=torch.long)
            dst = torch.tensor(dst, dtype=torch.long)
            slab = d["kv"][rows]
            if kv is None:
                kv = torch.empty((k, *slab.shape[1:]), dtype=slab.dtype)
                am = torch.empty((k, d["attn_mask"].shape[1]), dtype=d["attn_mask"].dtype)
                eff = torch.empty((k,), dtype=d["eff_len"].dtype)
                ids = torch.empty((k, d["input_ids"].shape[1]), dtype=torch.long)
                if self.fp8:
                    scale = torch.empty((k, *d["kv_scale"].shape[1:]), dtype=d["kv_scale"].dtype)
            kv[dst] = slab
            am[dst], eff[dst], ids[dst] = d["attn_mask"][rows], d["eff_len"][rows], d["input_ids"][rows]
            if self.fp8:
                scale[dst] = d["kv_scale"][rows]
        out = dict(bank_kv=kv, bank_am=am, bank_eff=eff, bank_ids=ids)
        if self.fp8:
            out["bank_scale"] = scale
        return out

    def assemble(self, pre, repeat, chunk, n_arms, device, with_seg=True):
        """GPU side: dequantise, repeat per frame, build RoPE and segment ids."""
        kv = pre["bank_kv"].to(device, non_blocking=True)
        am = pre["bank_am"].to(device, non_blocking=True)
        eff = pre["bank_eff"].to(device, non_blocking=True)
        ids = pre["bank_ids"].to(device, non_blocking=True)
        if self.fp8:
            sc = pre["bank_scale"].to(device, non_blocking=True)
            kv = kv.to(torch.bfloat16) * sc[..., None, None].to(torch.bfloat16)
        if repeat > 1:
            # Frames of a batch are grouped by episode: repeat_interleave, not repeat.
            kv, am, eff, ids = (x.repeat_interleave(repeat, dim=0) for x in (kv, am, eff, ids))
        n_layers = kv.shape[1]
        keys = tuple(kv[:, l, 0] for l in range(n_layers))
        values = tuple(kv[:, l, 1] for l in range(n_layers))
        self.rope.to(device)
        return cache_kwargs(keys, values, am, eff, self.rope, chunk, n_arms,
                            input_ids=ids if with_seg else None,
                            n_image_tokens=self.n_image_tokens)

    def batch(self, episode_ids, repeat, chunk, n_arms, device, with_seg=True):
        return self.assemble(self.gather_rows(episode_ids), repeat, chunk, n_arms, device,
                             with_seg)
