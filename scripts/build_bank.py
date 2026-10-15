#!/usr/bin/env python
"""Precompute the first-frame VLM cache of every training episode ("bank").

    # RoboTwin: 27,500 episodes -> 22 fp8 shards (~1.6 TiB). One GPU per shard.
    for s in $(seq 0 21); do
      CUDA_VISIBLE_DEVICES=$((s % 8)) python scripts/build_bank.py --config configs/robotwin.yaml --shard $s &
    done; wait
    python scripts/build_bank.py --config configs/robotwin.yaml --check

    # LIBERO: 2,000 episodes -> 2 bf16 shards of 1,000 (~150 GiB), minutes on one GPU.
    python scripts/build_bank.py --config configs/libero.yaml --shard 0
    python scripts/build_bank.py --config configs/libero.yaml --shard 1
    python scripts/build_bank.py --config configs/libero.yaml --check

Episodes are prefilled in batches of 8 aligned to the start of the dataset (shards of
1,250 RoboTwin / 1,000 LIBERO episodes), the same batching that produced the banks
behind the released checkpoints. Shard 0 also exports the VLM's
RoPE tables (``rope_tables.pt``) needed for training.
"""
import argparse
import json
import os
import sys
import time

import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from fastvla.bank import fp8_quantize  # noqa: E402
from fastvla.vlm import build_prompt, dump_rope_tables, load_vlm, prefill  # noqa: E402

SHARD_EPS = {"robotwin": 1250, "libero": 1000}
BATCH = 8


def rel(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def episodes(cfg):
    """-> dataset, [(episode_id, instruction)] in dataset order."""
    from fastvla.normalize import Normalizer
    d = rel(cfg["data_dir"])
    norm = Normalizer.load(os.path.join(d, "norm_stats.json"), cfg["norm_mode"])
    if cfg["benchmark"] == "robotwin":
        from fastvla.data.robotwin import RobotwinEpisodes
        ds = RobotwinEpisodes(os.path.join(d, "index.json"), rel(cfg["instructions"]), norm)
        # Bank order (and hence prefill batching) follows the bank the released model was
        # trained on: clean before randomized, then task, then the hdf5 path as a *string*
        # (episode0, episode1, episode10, ...). Rows are looked up by id, so the order only
        # matters for reproducing the bank bit-exactly.
        src = {"clean": 0, "rand": 1}
        order = sorted(range(len(ds)), key=lambda i: (src[ds.eps[i]["episode_id"].split("/")[0]],
                                                      ds.eps[i]["task"], ds.eps[i]["path"]))
        return ds, [(i, ds.episode_id(i), ds.instr[ds.episode_id(i)]) for i in order]
    from fastvla.data.libero import LiberoEpisodes
    ds = LiberoEpisodes(d, norm)
    return ds, [(i, ds.episode_id(i), ds.instruction(i)) for i in range(len(ds))]


def pad_length(dm05, proc, bench, ds, eps):
    """Common prefix length L of all shards: image/template overhead + longest instruction."""
    tok = proc.tokenizer
    i0, _, s0 = eps[0]
    keys, _, _ = prefill(dm05, proc, bench, [s0], [ds.first_frame(i0)])
    overhead = keys[0].shape[2] - len(tok(build_prompt(bench, s0))["input_ids"])
    return overhead + max(len(tok(build_prompt(bench, s))["input_ids"]) for _, _, s in eps)


def build_shard(cfg, shard):
    bench, out = cfg["benchmark"], rel(cfg["bank"])
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"shard_{shard:02d}.pt")
    if os.path.exists(path):
        print(f"[bank] {path} exists, skipping")
        return
    rope_path = os.path.join(out, "rope_tables.pt")
    if shard == 0 and not os.path.exists(rope_path):
        dump_rope_tables(cfg["vlm"], rope_path, cfg["vlm_revision"])
        torch.cuda.empty_cache()

    ds, eps = episodes(cfg)
    per = SHARD_EPS[bench]
    lo, hi = shard * per, min((shard + 1) * per, len(eps))
    assert lo < hi, f"shard {shard} is empty ({len(eps)} episodes)"
    dm05, proc = load_vlm(cfg["vlm"], cfg["vlm_revision"])
    L = pad_length(dm05, proc, bench, ds, eps)
    fp8 = bench == "robotwin"
    n = hi - lo
    kv = scale = None
    am = torch.zeros((n, L), dtype=torch.bool)
    ids = torch.zeros((n, L), dtype=torch.long)
    eff = torch.zeros((n,), dtype=torch.long)
    t0 = time.time()
    for s in range(lo, hi, BATCH):
        sl = range(s, min(s + BATCH, hi))
        keys, values, meta = prefill(dm05, proc, bench, [eps[i][2] for i in sl],
                                     [ds.first_frame(eps[i][0]) for i in sl])
        Lb = keys[0].shape[2]
        assert Lb <= L, f"prefix {Lb} longer than L={L}"
        blob = torch.stack([torch.stack([k, v], 1) for k, v in zip(keys, values)], 1)
        j = slice(s - lo, s - lo + len(sl))
        if fp8:
            q, sc = fp8_quantize(blob)
            if kv is None:
                kv = torch.empty((n, *q.shape[1:-2], L, q.shape[-1]), dtype=q.dtype)
                scale = torch.zeros((n, *sc.shape[1:]), dtype=torch.float32)
            kv[j, ..., :Lb, :] = q.cpu()
            scale[j] = sc.float().cpu()
        else:
            if kv is None:
                kv = torch.zeros((n, *blob.shape[1:-2], L, blob.shape[-1]), dtype=torch.bfloat16)
            kv[j, ..., :Lb, :] = blob.to(torch.bfloat16).cpu()
        am[j, :Lb] = meta["attention_mask"][:, :Lb].bool().cpu()
        ids[j, :meta["input_ids"].shape[1]] = meta["input_ids"].cpu()
        eff[j] = meta["effective_prefix_len"].cpu()
        if (s - lo) // BATCH % 20 == 0:
            print(f"[bank {shard}] {s - lo + len(sl)}/{n}  {time.time() - t0:.0f}s", flush=True)
    d = dict(kv=kv, attn_mask=am, eff_len=eff, input_ids=ids, L=L,
             ep_ids=[eps[i][1] for i in range(lo, hi)],
             instructions=[eps[i][2] for i in range(lo, hi)])
    if fp8:
        d["kv_scale"] = scale
    torch.save(d, path + ".tmp")
    os.replace(path + ".tmp", path)
    print(f"[bank {shard}] {n} episodes, L={L} -> {path} ({time.time() - t0:.0f}s)")


def check(cfg):
    from fastvla.bank import BankReader
    from fastvla.vlm import RopeTables
    out = rel(cfg["bank"])
    _, eps = episodes(cfg)
    rope = RopeTables.load(os.path.join(out, "rope_tables.pt"))
    b = BankReader(out, rope, n_image_tokens=None)
    missing = {e for _, e, _ in eps} - set(b.ep_ids)
    assert not missing, f"{len(missing)} episodes missing, e.g. {sorted(missing)[:3]}"
    b.check_instructions({e: s for _, e, s in eps})
    print(f"[bank] OK: {len(b)} episodes, L={b.L}, {'fp8' if b.fp8 else 'bf16'}, {b.spec()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.check:
        return check(cfg)
    assert a.shard is not None, "give --shard N (or --check)"
    build_shard(cfg, a.shard)


if __name__ == "__main__":
    main()
