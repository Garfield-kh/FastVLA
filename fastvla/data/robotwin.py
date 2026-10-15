"""RoboTwin 2.0 data: raw hdf5 episodes of the official aloha-agilex dataset.

Training set ("full"): per task 50 demo_clean + 500 demo_randomized episodes,
27,500 in total. Episode ids carry the source: ``clean/<task>/episode<N>`` and
``rand/<task>/episode<N>``.

State and action are the same 14-D array ``joint_action/vector``
([6 joints + gripper] x 2 arms, absolute), as in DM0.5 and the RoboTwin baselines.
The action chunk starts at the current frame and the tail is padded by repeating
the last action. JPEG frames are decoded with ``cv2.imdecode`` and used as-is: the
collector encoded RGB arrays with ``cv2.imencode``, so this round trip returns the
original RGB (decoding with PIL would swap red and blue).
"""
import glob
import json
import os
import re
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from ..normalize import Normalizer, compute_stats
from .common import batch_rng, to_float_image

CAMERAS = ("head_camera", "left_camera", "right_camera")   # Head / Left wrist / Right wrist
ACTION_DIM = 14
N_ARMS = 2
ROLE_LEFT, ROLE_RIGHT = 1, 2
MAX_OPEN_H5 = 128          # LRU cap on open hdf5 handles per worker


def decode_frame(raw):
    import cv2
    arr = cv2.imdecode(np.frombuffer(bytes(raw), np.uint8), cv2.IMREAD_COLOR)
    assert arr is not None, "corrupt JPEG"
    return np.ascontiguousarray(arr, dtype=np.uint8)


def img_hw_for_backbone(backbone, target_tokens=192):
    """3:4 image size whose sides are multiples of the patch size (no cropping).

    DINOv2 (patch 14) -> (168, 224): 12 x 16 patches per camera.
    """
    from transformers import AutoConfig
    patch = int(getattr(AutoConfig.from_pretrained(backbone), "patch_size", 14))
    cands = [(3 * k, 4 * k) for k in range(1, 81) if (3 * k) % patch == 0 and (4 * k) % patch == 0]
    return min(cands, key=lambda hw: abs((hw[0] // patch) * (hw[1] // patch) - target_tokens))


# ------------------------------------------------------------------ index / stats
def scan_episodes(root, prefix):
    """-> [(episode_id, hdf5_path)], sorted by task name and numeric episode index."""
    out = []
    for task in sorted(os.listdir(root)):
        if not os.path.isdir(os.path.join(root, task)):
            continue
        for f in sorted(glob.glob(f"{root}/{task}/*/data/episode*.hdf5")):
            n = int(re.search(r"episode(\d+)\.hdf5$", f).group(1))
            out.append((task, n, f))
    out.sort(key=lambda x: (x[0], x[1]))
    return [(f"{prefix}/{t}/episode{n}", f) for t, n, f in out]


def build_index(clean_root, rand_root, out_path):
    """Episode list + lengths (clean first, then randomized)."""
    import h5py
    rec = []
    for prefix, root in (("clean", clean_root), ("rand", rand_root)):
        for eid, path in scan_episodes(root, prefix):
            with h5py.File(path, "r") as h:
                n = int(h["joint_action/vector"].shape[0])
            rec.append(dict(episode_id=eid, path=path, n=n, task=eid.split("/")[-2]))
    meta = dict(n_episodes=len(rec), n_tasks=len({r["task"] for r in rec}),
                n_frames=sum(r["n"] for r in rec))
    assert meta["n_tasks"] == 50, f"expected 50 tasks, found {meta['n_tasks']}"
    json.dump(dict(meta=meta, episodes=rec), open(out_path, "w"))
    return meta


def compute_norm_stats(index_path, out_path):
    import h5py
    idx = json.load(open(index_path))
    mats = []
    for r in idx["episodes"]:
        with h5py.File(r["path"], "r") as h:
            mats.append(np.asarray(h["joint_action/vector"], np.float32))
    stats = compute_stats(mats, mats)        # state and action are the same array
    json.dump(stats, open(out_path, "w"), indent=1)
    return stats


# ------------------------------------------------------------------ dataset
class RobotwinEpisodes(Dataset):
    """One record per episode; frames are chosen by the batcher."""

    def __init__(self, index_path, instr_path, norm: Normalizer, chunk=50, img_hw=(168, 224),
                 hist_offsets=(0,), hist_cams=(0,)):
        idx = json.load(open(index_path))
        self.eps, self.meta = idx["episodes"], dict(idx["meta"])
        self.instr = json.load(open(instr_path))
        missing = {e["episode_id"] for e in self.eps} - set(self.instr)
        assert not missing, f"{len(missing)} episodes have no instruction in {instr_path}"
        self.norm, self.chunk, self.img_hw = norm, chunk, tuple(img_hw)
        self.hist_offsets = tuple(int(o) for o in hist_offsets)
        assert self.hist_offsets[-1] == 0 and all(o <= 0 for o in self.hist_offsets)
        self.hist_cams = tuple(sorted(set(int(c) for c in hist_cams)))
        self._h, self._pid = OrderedDict(), os.getpid()

    def __len__(self):
        return len(self.eps)

    def episode_id(self, i):
        return self.eps[i]["episode_id"]

    def _file(self, i):
        # hdf5 handles must not be shared across forked workers.
        if os.getpid() != self._pid:
            self._h, self._pid = OrderedDict(), os.getpid()
        p = self.eps[i]["path"]
        if p in self._h:
            self._h.move_to_end(p)
            return self._h[p]
        import h5py
        self._h[p] = h5py.File(p, "r")
        while len(self._h) > MAX_OPEN_H5:
            self._h.popitem(last=False)[1].close()
        return self._h[p]

    def view_plan(self):
        """[(camera, frame offset)] in slot order: current frames first, history after."""
        plan = [(c, 0) for c in range(len(CAMERAS))]
        for c in self.hist_cams:
            plan += [(c, o) for o in self.hist_offsets[:-1]]
        return plan

    def raw_slots(self, i, t, plan):
        """History before the episode start is clamped to frame 0."""
        h, n = self._file(i), int(self.eps[i]["n"])
        cache, out = {}, []
        for c, off in plan:
            f = min(max(t + off, 0), n - 1)
            if (c, f) not in cache:
                cache[(c, f)] = decode_frame(h[f"observation/{CAMERAS[c]}/rgb"][f])
            out.append(cache[(c, f)])
        return out

    def first_frame(self, i):
        return self.raw_slots(i, 0, [(c, 0) for c in range(len(CAMERAS))])

    def action_chunk(self, i, t):
        n_all = self.eps[i]["n"]
        k = min(self.chunk, n_all - t)
        raw = self.norm.norm(np.asarray(self._file(i)["joint_action/vector"][t:t + k],
                                        np.float32), "action")
        a = np.zeros((self.chunk, ACTION_DIM), np.float32)
        a[:k] = raw
        a[k:] = raw[-1]                          # tail: repeat the last action
        return a, np.ones((self.chunk, ACTION_DIM), np.float32)

    def sample(self, i, t):
        views = self.raw_slots(i, t, self.view_plan())
        st = self.norm.norm(np.asarray(self._file(i)["joint_action/vector"][t], np.float32),
                            "state")
        a, m = self.action_chunk(i, t)
        return dict(imgs=torch.stack([to_float_image(v, self.img_hw) for v in views]),
                    state=torch.from_numpy(np.asarray(st, np.float32)),
                    action=torch.from_numpy(a), action_mask=torch.from_numpy(m))


class RobotwinBatcher:
    """k episodes x B/k random frames per batch; optionally gathers the bank rows in
    the DataLoader worker so the ~0.5 GB/batch read overlaps with GPU compute."""

    _banks = {}

    def __init__(self, ds, batch_size, eps_per_batch, seed=0, bank_path=None):
        assert batch_size % eps_per_batch == 0
        self.ds, self.k, self.per_ep, self.seed = ds, eps_per_batch, batch_size // eps_per_batch, seed
        self.bank_path = bank_path

    @classmethod
    def _bank(cls, path):
        if path not in cls._banks:
            from ..bank import BankReader
            cls._banks[path] = BankReader(path, rope=None, n_image_tokens=None)
        return cls._banks[path]

    def build(self, ep_idx, batch_id=0):
        rng = batch_rng(self.seed, batch_id)
        frames, groups = [], []
        for i in ep_idx:
            i = int(i)
            n = self.ds.eps[i]["n"]
            for t in rng.randint(0, max(1, n - 1), size=self.per_ep):
                frames.append(self.ds.sample(i, int(t)))
            groups.append(i)
        out = {k: torch.stack([f[k] for f in frames]) for k in frames[0]}
        B = len(frames)
        out["role"] = torch.tensor([ROLE_LEFT, ROLE_RIGHT], dtype=torch.long)[None].expand(
            B, N_ARMS).contiguous()
        out["repeat"] = self.per_ep
        out["episode_ids"] = [self.ds.episode_id(i) for i in groups]
        if self.bank_path is not None:
            out.update(self._bank(self.bank_path).gather_rows(out["episode_ids"]))
        return out
