"""LIBERO data: the four standard suites (spatial / object / goal / 10), 50 demos per task.

The official hdf5 demos are converted once (``scripts/prepare_data.py libero``) into one
uncompressed ``task%02d.npz`` per task, memory-mapped at training time:

    agentview, wrist  (N, 128, 128, 3) uint8   (raw simulator orientation, i.e. upside down)
    proprio           (N, 8)  ee_pos(3) + ee_ori(3) + gripper(2)
    actions           (N, 7)  absolute, as in the dataset
    ep_start, ep_len  (50,)

Episode ids are ``<suite>/t<task:02d>/ep<index:03d>``.
"""
import json
import os
import struct
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset

from ..normalize import Normalizer, compute_stats
from .common import batch_rng, to_float_image

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
ACTION_DIM = 7
ROLE_LIBERO = 0
PROPRIO_KEYS = ("ee_pos", "ee_ori", "gripper_states")


def npz_memmap(path):
    """Real np.memmap views of an uncompressed .npz (np.load ignores mmap_mode for npz)."""
    out = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            assert info.compress_type == zipfile.ZIP_STORED, f"{path} is compressed"
            with open(path, "rb") as fh:
                fh.seek(info.header_offset)
                hdr = fh.read(30)
                fnlen, extralen = struct.unpack("<HH", hdr[26:30])
                fh.seek(info.header_offset + 30 + fnlen + extralen)
                assert fh.read(6) == b"\x93NUMPY"
                major = fh.read(2)[0]
                hl = struct.unpack("<H" if major == 1 else "<I", fh.read(2 if major == 1 else 4))[0]
                head = eval(fh.read(hl).decode("latin1"))
                start = fh.tell()
            out[info.filename[:-4]] = np.memmap(path, dtype=np.dtype(head["descr"]), mode="r",
                                                offset=start, shape=head["shape"])
    return out


def convert_suite(hdf5_dir, suite, out_dir):
    """Official LIBERO hdf5 (``<hdf5_dir>/<suite>/<task>_demo.hdf5``) -> npz + manifest.

    Task order follows ``libero.benchmark``, which is also the evaluation order.
    """
    import h5py
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[suite]()
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"suite": suite, "proprio_keys": list(PROPRIO_KEYS), "tasks": []}
    for tid in range(ts.n_tasks):
        task = ts.get_task(tid)
        path = os.path.join(hdf5_dir, suite, f"{task.name}_demo.hdf5")
        agent, wrist, proprio, actions, starts = [], [], [], [], []
        with h5py.File(path, "r") as h:
            d = h["data"]
            for k in sorted(d.keys(), key=lambda s: int(s.split("_")[1])):
                g = d[k]
                starts.append(sum(x.shape[0] for x in actions))
                agent.append(g["obs"]["agentview_rgb"][:])
                wrist.append(g["obs"]["eye_in_hand_rgb"][:])
                proprio.append(np.concatenate([np.asarray(g["obs"][p][:], np.float32)
                                               for p in PROPRIO_KEYS], axis=1))
                actions.append(np.asarray(g["actions"][:], np.float32))
        np.savez(os.path.join(out_dir, f"task{tid:02d}.npz"),
                 agentview=np.concatenate(agent), wrist=np.concatenate(wrist),
                 proprio=np.concatenate(proprio), actions=np.concatenate(actions),
                 ep_start=np.asarray(starts, np.int64),
                 ep_len=np.asarray([a.shape[0] for a in actions], np.int64))
        manifest["tasks"].append(dict(task_id=tid, name=task.name, language=task.language,
                                      n_demos=len(starts)))
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)


def compute_norm_stats(root, out_path, suites=SUITES):
    st, ac = [], []
    for s in suites:
        d = os.path.join(root, s)
        for f in sorted(x for x in os.listdir(d) if x.startswith("task") and x.endswith(".npz")):
            z = np.load(os.path.join(d, f))
            st.append(z["proprio"].astype(np.float32))
            ac.append(z["actions"].astype(np.float32))
    stats = compute_stats(st, ac)
    json.dump(stats, open(out_path, "w"), indent=1)
    return stats


class LiberoEpisodes(Dataset):
    def __init__(self, root, norm: Normalizer, chunk=10, img_hw=(224, 224), suites=SUITES,
                 hist_offsets=(0,), hist_cams=(0,)):
        self.norm, self.chunk, self.img_hw = norm, chunk, tuple(img_hw)
        self.eps = []
        for s in suites:
            man = json.load(open(os.path.join(root, s, "manifest.json")))
            for t in man["tasks"]:
                f = os.path.join(root, s, f"task{t['task_id']:02d}.npz")
                z = npz_memmap(f)
                for i, (st, ln) in enumerate(zip(z["ep_start"], z["ep_len"], strict=True)):
                    self.eps.append(dict(suite=s, task_id=t["task_id"], file=f, ep=i,
                                         start=int(st), length=int(ln), language=t["language"]))
        self.meta = dict(n_episodes=len(self.eps), n_frames=sum(e["length"] for e in self.eps),
                         n_tasks=len({(e["suite"], e["task_id"]) for e in self.eps}))
        self.hist_offsets = tuple(int(o) for o in hist_offsets)
        assert self.hist_offsets[-1] == 0 and all(o <= 0 for o in self.hist_offsets)
        self.hist_cams = tuple(hist_cams)
        self._cache, self._pid = {}, os.getpid()

    def __len__(self):
        return len(self.eps)

    def _z(self, path):
        if os.getpid() != self._pid:
            self._cache, self._pid = {}, os.getpid()
        if path not in self._cache:
            self._cache[path] = npz_memmap(path)
        return self._cache[path]

    def episode_id(self, i):
        e = self.eps[i]
        return f"{e['suite']}/t{e['task_id']:02d}/ep{e['ep']:03d}"

    def instruction(self, i):
        return self.eps[i]["language"]

    def view_plan(self):
        plan = [(0, 0), (1, 0)]
        for c in self.hist_cams:
            plan += [(c, o) for o in self.hist_offsets[:-1]]
        return plan

    def raw_slots(self, i, t, plan):
        e = self.eps[i]
        z, s, L = self._z(e["file"]), e["start"], e["length"]
        key = ("agentview", "wrist")
        got, out = {}, []
        for c, o in plan:
            tt = min(max(t + o, 0), L - 1)
            if (c, tt) not in got:
                got[(c, tt)] = np.asarray(z[key[c]][s + tt])
            out.append(got[(c, tt)])
        return out

    def first_frame(self, i):
        return self.raw_slots(i, 0, [(0, 0), (1, 0)])

    def action_chunk(self, i, t):
        """Tail beyond the episode end is masked out (not supervised)."""
        e = self.eps[i]
        z, s, L = self._z(e["file"]), e["start"], e["length"]
        n = min(self.chunk, L - t)
        raw = self.norm.norm(np.asarray(z["actions"][s + t: s + t + n], np.float32), "action")
        a = np.zeros((self.chunk, ACTION_DIM), np.float32)
        m = np.zeros((self.chunk, ACTION_DIM), np.float32)
        a[:n], m[:n] = raw, 1.0
        return a, m

    def sample(self, i, t):
        e = self.eps[i]
        st = self.norm.norm(np.asarray(self._z(e["file"])["proprio"][e["start"] + t],
                                       np.float32), "state")
        a, m = self.action_chunk(i, t)
        imgs = torch.stack([to_float_image(v, self.img_hw)
                            for v in self.raw_slots(i, t, self.view_plan())])
        return dict(imgs=imgs, state=torch.from_numpy(np.asarray(st, np.float32)),
                    action=torch.from_numpy(a), action_mask=torch.from_numpy(m))


class LiberoBatcher:
    def __init__(self, ds, batch_size, eps_per_batch, seed=0):
        assert batch_size % eps_per_batch == 0
        self.ds, self.k, self.per_ep, self.seed = ds, eps_per_batch, batch_size // eps_per_batch, seed

    def build(self, ep_idx, batch_id=0):
        rng = batch_rng(self.seed, batch_id)
        frames, groups = [], []
        for i in ep_idx:
            i = int(i)
            L = self.ds.eps[i]["length"]
            for t in rng.randint(0, max(1, L - 1), size=self.per_ep):
                frames.append(self.ds.sample(i, int(t)))
            groups.append(i)
        out = {k: torch.stack([f[k] for f in frames]) for k in frames[0]}
        out["role"] = torch.full((len(frames), 1), ROLE_LIBERO, dtype=torch.long)
        out["repeat"] = self.per_ep
        out["episode_ids"] = [self.ds.episode_id(i) for i in groups]
        return out
