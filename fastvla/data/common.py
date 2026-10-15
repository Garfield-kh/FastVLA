"""Episode-grouped batching shared by both benchmarks.

Every batch takes B/k frames from each of k episodes, so only k first-frame caches
have to be fetched per step. The whole training run is one deterministic stream of
global batch indices: batch g belongs to epoch g // batches_per_epoch, episodes are
reshuffled per (seed, epoch) and frames are drawn with a per-g seed. Resuming or
running on W GPUs is a shift / stride of that stream.
"""
import numpy as np
from torch.utils.data import Dataset


def batch_rng(seed, batch_id):
    return np.random.RandomState((seed * 1_000_003 + batch_id) % (2**31 - 1))


class BatchDataset(Dataset):
    """Item gi = the gi-th batch this rank consumes."""

    def __init__(self, batcher, base_seed=0, total_batches=None, offset=0, rank=0, stride=1):
        self.b = batcher
        self.base_seed, self.offset = base_seed, int(offset)
        self.rank, self.stride = int(rank), int(stride)
        self.per_epoch = len(batcher.ds) // batcher.k
        assert self.per_epoch > 0
        self.total = int(total_batches or self.per_epoch)
        self._perm = {}

    def __len__(self):
        return self.total

    def _order(self, epoch):
        if epoch not in self._perm:
            if len(self._perm) > 4:
                self._perm.clear()
            rng = np.random.RandomState((self.base_seed * 7919 + epoch) % (2**31 - 1))
            self._perm[epoch] = rng.permutation(len(self.b.ds))
        return self._perm[epoch]

    def __getitem__(self, gi):
        g = self.offset + int(gi) * self.stride + self.rank
        ep, bi = divmod(g, self.per_epoch)
        group = self._order(ep)[bi * self.b.k:(bi + 1) * self.b.k]
        return self.b.build(group, batch_id=g)


def make_loader(batcher, workers=8, prefetch=4, base_seed=0, total_batches=None,
                offset=0, rank=0, stride=1):
    ds = BatchDataset(batcher, base_seed, total_batches, offset, rank, stride)
    if workers <= 0:
        return ds
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=None, shuffle=False, num_workers=workers,
                      prefetch_factor=prefetch, persistent_workers=True, pin_memory=True)


def to_float_image(hwc_u8, img_hw):
    """uint8 HWC -> float CHW in [0, 1], bilinear-resized to img_hw (no cropping)."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(hwc_u8).copy()).permute(2, 0, 1).float() / 255.0
    if tuple(x.shape[-2:]) != tuple(img_hw):
        x = torch.nn.functional.interpolate(x[None], size=tuple(img_hw), mode="bilinear",
                                            align_corners=False)[0]
    return x
