#!/usr/bin/env python
"""Train FastVLA on a precomputed first-frame cache bank.

    # RoboTwin (released checkpoint: 2 GPUs)
    torchrun --nproc_per_node=2 scripts/train.py --config configs/robotwin.yaml --out runs/robotwin
    # LIBERO (released checkpoint: 1 GPU)
    python scripts/train.py --config configs/libero.yaml --out runs/libero

Any config key can be overridden, e.g. ``--set steps=1000 batch=16``.
Checkpoints are written in the released format (``runs/<name>/step_XXXXXX/``) and can be
served directly with ``scripts/serve.py``.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from fastvla.aug import MILD, aug_views  # noqa: E402
from fastvla.bank import BankReader  # noqa: E402
from fastvla.data.common import make_loader  # noqa: E402
from fastvla.normalize import Normalizer  # noqa: E402
from fastvla.policy import build_model  # noqa: E402
from fastvla.vlm import VLM_TOKENS_PER_IMAGE, RopeTables  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--resume", default="", help="a step_XXXXXX directory of this run")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--stop-at", type=int, default=0,
                    help="stop after this step without changing the LR schedule (debugging)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return a, cfg


def rel(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def masked_l1(pred, action, mask):
    elem = (pred.float() - action.float()).abs()
    return ((elem * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)).mean()


def main():
    a, cfg = parse()
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp = world > 1
    if ddp:
        import torch.distributed as dist
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    is_main = rank == 0
    dev = f"cuda:{local_rank}" if ddp else "cuda"
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "train.log"), "a") if is_main else None

    def log(msg):
        if is_main:
            s = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
            print(s, flush=True)
            logf.write(s + "\n")
            logf.flush()

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    bench = cfg["benchmark"]
    data_dir, bank_dir = rel(cfg["data_dir"]), rel(cfg["bank"])
    norm_path = os.path.join(data_dir, "norm_stats.json")
    norm = Normalizer.load(norm_path, cfg["norm_mode"])

    # ---------------------------------------------------------------- data
    if bench == "robotwin":
        from fastvla.data.robotwin import (N_ARMS, RobotwinBatcher, RobotwinEpisodes,
                                           img_hw_for_backbone)
        img_hw = img_hw_for_backbone(cfg["backbone"], cfg["img_tokens"])
        ds = RobotwinEpisodes(os.path.join(data_dir, "index.json"), rel(cfg["instructions"]),
                              norm, chunk=cfg["chunk"], img_hw=img_hw,
                              hist_offsets=cfg["hist_offsets"], hist_cams=cfg["hist_cams"])
        n_view, n_arms, state_dim, action_dim, roles = 3, N_ARMS, 14, 14, [1, 2]
        batcher = RobotwinBatcher(ds, cfg["batch"], cfg["k"], seed=cfg["seed"],
                                  bank_path=bank_dir if cfg["bank_in_worker"] else None)
        instr = ds.instr
    else:
        from fastvla.data.libero import LiberoBatcher, LiberoEpisodes
        img_hw = tuple(cfg["img_hw"])
        ds = LiberoEpisodes(data_dir, norm, chunk=cfg["chunk"], img_hw=img_hw,
                            hist_offsets=cfg["hist_offsets"], hist_cams=cfg["hist_cams"])
        n_view, n_arms, state_dim, action_dim, roles = 2, 1, 8, 7, [0]
        batcher = LiberoBatcher(ds, cfg["batch"], cfg["k"], seed=cfg["seed"])
        instr = {ds.episode_id(i): ds.instruction(i) for i in range(len(ds))}
    plan = ds.view_plan()
    log(f"{bench}: {len(ds)} episodes / {ds.meta['n_frames']} frames; image {img_hw}; "
        f"slots {plan}")

    rope = RopeTables.load(os.path.join(bank_dir, "rope_tables.pt"))
    bank = BankReader(bank_dir, rope, n_image_tokens=VLM_TOKENS_PER_IMAGE * n_view)
    missing = {ds.episode_id(i) for i in range(len(ds))} - set(bank.ep_ids)
    assert not missing, f"{len(missing)} training episodes are not in the bank"
    bank.check_instructions(instr)
    log(f"bank: {len(bank)} episodes, L={bank.L}, {'fp8' if bank.fp8 else 'bf16'}, "
        f"spec={bank.spec()}")

    model_cfg = dict(
        benchmark=bench, d=cfg["d"], nhead=cfg["nhead"], n_dec=cfg["n_dec"],
        n_cache_heads=cfg["n_cache_heads"], chunk=cfg["chunk"], n_view=n_view, n_arms=n_arms,
        state_dim=state_dim, action_dim=action_dim, backbone=cfg["backbone"],
        cam_of_slot=[c for c, _ in plan], slot_time=[o for _, o in plan], cache_seg=True,
        cache_spec=bank.spec(), img_hw=list(img_hw), norm_mode=cfg["norm_mode"], roles=roles,
        vlm=cfg["vlm"], vlm_revision=cfg["vlm_revision"], train=cfg)
    model = build_model(model_cfg)          # DINOv2 from the pretrained backbone
    log(f"trainable params {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")

    # ---------------------------------------------------------------- loader
    start = 0
    state = None
    if a.resume:
        state = torch.load(os.path.join(a.resume, "training_state.pt"), map_location="cpu",
                           weights_only=False)
        from safetensors.torch import load_file
        model.load_state_dict(load_file(os.path.join(a.resume, "model.safetensors")))
        start = int(state["step"])
        log(f"resume from step {start}")
    # The data stream is a pure function of (seed, global batch index): resuming is a shift.
    loader = make_loader(batcher, cfg["workers"], cfg["prefetch"], base_seed=cfg["seed"],
                         total_batches=cfg["steps"] - start + 64, offset=start * world,
                         rank=rank, stride=world)
    it = iter(loader)
    first = next(it)                          # fork the workers before CUDA is initialised

    model = model.to(dev)
    raw = model
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    dino = [p for n, p in raw.named_parameters() if p.requires_grad and n.startswith("dino.")]
    rest = [p for n, p in raw.named_parameters() if p.requires_grad and not n.startswith("dino.")]
    groups = [dict(params=rest, lr=cfg["lr"]),
              dict(params=dino, lr=cfg["lr"] * cfg["lr_dino_mult"])]
    opt = torch.optim.AdamW(groups, lr=cfg["lr"], weight_decay=1e-10, betas=(0.9, 0.95))

    def lr_lambda(s):
        if s < cfg["warmup"]:
            return s / max(cfg["warmup"], 1)
        return 0.5 * (1 + math.cos(math.pi * (s - cfg["warmup"]) / max(cfg["steps"] - cfg["warmup"], 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    if state is not None:
        opt.load_state_dict(state["opt"])
        for _ in range(start):
            sched.step()
    eff = cfg["batch"] * world
    log(f"effective batch {eff}; {cfg['steps']} steps = "
        f"{cfg['steps'] * eff / ds.meta['n_frames']:.2f} epochs")

    def save(step):
        d = os.path.join(a.out, f"step_{step:06d}")
        os.makedirs(d, exist_ok=True)
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in raw.state_dict().items()},
                  os.path.join(d, "model.safetensors"))
        json.dump(dict(model_cfg, train=dict(cfg, step=step)),
                  open(os.path.join(d, "config.json"), "w"), indent=1)
        json.dump({k: norm.stats[k] for k in ("state", "action")},
                  open(os.path.join(d, "norm_stats.json"), "w"), indent=1)
        shutil.copy(os.path.join(bank_dir, "rope_tables.pt"), os.path.join(d, "rope_tables.pt"))
        torch.save(dict(opt=opt.state_dict(), step=step), os.path.join(d, "training_state.pt"))
        log(f"saved {d}")

    t0, run = time.time(), 0.0
    for step in range(start, cfg["steps"]):
        b, first = (first, None) if first is not None else (next(it), None)
        if "bank_kv" in b:
            cache = bank.assemble(b, b["repeat"], cfg["chunk"], n_arms, dev)
        else:
            cache = bank.batch(b["episode_ids"], b["repeat"], cfg["chunk"], n_arms, dev)
        act = b["action"].to(dev, non_blocking=True)
        msk = b["action_mask"].to(dev, non_blocking=True)
        im = aug_views(b["imgs"].to(dev, non_blocking=True), model_cfg["cam_of_slot"], MILD)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(imgs=im, state=b["state"].to(dev, non_blocking=True),
                         role=b["role"].to(dev, non_blocking=True), **cache)
        loss = masked_l1(pred, act, msk)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([q for g in groups for q in g["params"]], cfg["clip"])
        opt.step()
        sched.step()
        run += float(loss.detach())
        if (step + 1) % a.log_every == 0:
            sec = (time.time() - t0) / (step + 1 - start)
            log(f"step {step+1}/{cfg['steps']} loss {run / a.log_every:.4f} "
                f"lr {sched.get_last_lr()[0]:.2e} {sec:.2f}s/step "
                f"eta {(cfg['steps'] - step - 1) * sec / 3600:.1f}h")
            run = 0.0
        if is_main and ((step + 1) % cfg["save_every"] == 0 or step + 1 == cfg["steps"]):
            save(step + 1)
        if a.stop_at and step + 1 >= a.stop_at:
            break
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
