"""Closed-loop inference: first-frame VLM prefill once per episode, then the decoder only.

A released model directory contains:
    config.json        architecture + benchmark settings
    model.safetensors  decoder + DINOv2 weights
    norm_stats.json    state / action statistics used in training
    rope_tables.pt     DM0.5 action-expert RoPE tables
"""
import json
import os

import numpy as np
import torch

from .data.common import to_float_image
from .model import FastVLA
from .normalize import Normalizer
from .vlm import VLM_TOKENS_PER_IMAGE, RopeTables, cache_kwargs, load_vlm, prefill


def load_config(model_dir):
    return json.load(open(os.path.join(model_dir, "config.json")))


def build_model(cfg, pretrained_backbone=True):
    return FastVLA(cache_spec=cfg["cache_spec"], d=cfg["d"], nhead=cfg["nhead"],
                   n_dec=cfg["n_dec"], n_cache_heads=cfg["n_cache_heads"], chunk=cfg["chunk"],
                   n_view=cfg["n_view"], n_arms=cfg["n_arms"], state_dim=cfg["state_dim"],
                   backbone=cfg["backbone"], cam_of_slot=cfg["cam_of_slot"],
                   slot_time=cfg["slot_time"], cache_seg=cfg["cache_seg"],
                   pretrained_backbone=pretrained_backbone)


def load_model(model_dir, device="cuda"):
    from safetensors.torch import load_file
    cfg = load_config(model_dir)
    model = build_model(cfg, pretrained_backbone=False)
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    return model.to(device).eval(), cfg


class FastVLAPolicy:
    """
    act(views, state, instruction, hist_views=None) -> actions [chunk, action_dim]

    views:  current RGB uint8 HWC frames, one per camera, in the benchmark's camera order
            (RoboTwin: head, left wrist, right wrist; LIBERO: agentview, wrist in the
            raw simulator orientation, i.e. as stored in the LIBERO datasets).
    state:  raw proprioception (RoboTwin: 14 joint targets; LIBERO: 8-D ee state).
    hist_views: {offset (<0): [frames per camera]} for the history slots of the model;
            missing offsets fall back to the current frame.
    The VLM cache is built on the first call after ``reset()``.
    """

    def __init__(self, model_dir, device="cuda", vlm_repo=None, vlm_revision=None):
        self.device = device
        self.model, self.cfg = load_model(model_dir, device)
        c = self.cfg
        self.bench, self.chunk, self.n_arms = c["benchmark"], c["chunk"], c["n_arms"]
        self.n_view, self.action_dim = c["n_view"], c["action_dim"]
        self.cam_of_slot, self.slot_time = tuple(c["cam_of_slot"]), tuple(c["slot_time"])
        self.img_hw = tuple(c["img_hw"])
        self.norm = Normalizer.load(os.path.join(model_dir, "norm_stats.json"), c["norm_mode"])
        self.rope = RopeTables.load(os.path.join(model_dir, "rope_tables.pt"), device=device)
        self.role = torch.tensor([c["roles"]], dtype=torch.long, device=device)
        self.dm05, self.proc = load_vlm(vlm_repo or c["vlm"], vlm_revision or c["vlm_revision"],
                                        device)
        self.cache = None

    @property
    def hist_offsets(self):
        return sorted({o for o in self.slot_time if o != 0})

    def reset(self):
        self.cache = None

    @torch.no_grad()
    def build_cache(self, instruction, views):
        keys, values, meta = prefill(self.dm05, self.proc, self.bench, [instruction],
                                     [list(views[:self.n_view])], self.device)
        self.cache = cache_kwargs(keys, values, meta["attention_mask"],
                                  meta["effective_prefix_len"], self.rope, self.chunk,
                                  self.n_arms, input_ids=meta["input_ids"],
                                  n_image_tokens=VLM_TOKENS_PER_IMAGE * self.n_view)

    @torch.no_grad()
    def act(self, views, state, instruction, hist_views=None):
        if self.cache is None:
            self.build_cache(instruction, views)
        slots = []
        for c, off in zip(self.cam_of_slot, self.slot_time):
            src = views if off == 0 else (hist_views or {}).get(off, views)
            slots.append(to_float_image(src[c], self.img_hw))
        imgs = torch.stack(slots)[None].to(self.device)
        s = torch.from_numpy(self.norm.norm(np.asarray(state, np.float32), "state"))[None]
        with torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16):
            out = self.model(imgs=imgs, state=s.to(self.device), role=self.role, **self.cache)
        a = out[0, :, :self.action_dim].float().cpu().numpy()
        return self.norm.denorm(a, "action")
