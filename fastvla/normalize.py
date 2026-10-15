"""State / action normalisation.

``quantile``: clip to [q01, q99] and map linearly to [-1, 1] (as DM0.5 does).
``zscore``  : (x - mean) / std, no clipping.
Dimensions that are constant in the data map to 0 in both modes.
"""
import json

import numpy as np


def quantile_norm(arr, q01, q99, clip=True):
    lo, hi = np.asarray(q01, np.float32), np.asarray(q99, np.float32)
    arr = np.asarray(arr, np.float32)
    if clip:
        arr = np.clip(arr, lo, hi)
    out = ((arr - lo) / (hi - lo + 1e-6) * 2.0 - 1.0).astype(np.float32)
    return np.where((lo == 0) & (hi == 0), 0.0, out)


def quantile_denorm(arr, q01, q99):
    lo, hi = np.asarray(q01, np.float32), np.asarray(q99, np.float32)
    return (np.asarray(arr, np.float32) + 1.0) / 2.0 * (hi - lo + 1e-6) + lo


def zscore_norm(arr, mean, std):
    m, sd = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    out = ((np.asarray(arr, np.float32) - m) / (sd + 1e-6)).astype(np.float32)
    return np.where(sd < 1e-6, 0.0, out)


def zscore_denorm(arr, mean, std):
    m, sd = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    return (np.asarray(arr, np.float32) * (sd + 1e-6) + m).astype(np.float32)


class Normalizer:
    """Holds the statistics of one dataset and applies one mode to state and action."""

    def __init__(self, stats, mode="quantile"):
        assert mode in ("quantile", "zscore"), mode
        self.stats, self.mode = stats, mode

    @classmethod
    def load(cls, path, mode):
        return cls(json.load(open(path)), mode)

    def norm(self, x, key):
        s = self.stats[key]
        if self.mode == "zscore":
            return zscore_norm(x, s["mean"], s["std"])
        return quantile_norm(x, s["q01"], s["q99"])

    def denorm(self, x, key):
        s = self.stats[key]
        if self.mode == "zscore":
            return zscore_denorm(x, s["mean"], s["std"])
        return quantile_denorm(x, s["q01"], s["q99"])


def compute_stats(arrays_state, arrays_action):
    """q01 / q99 / mean / std per dimension over all frames."""
    out = {}
    for key, arrs in (("state", arrays_state), ("action", arrays_action)):
        v = np.concatenate(arrs).astype(np.float32)
        out[key] = dict(q01=np.quantile(v, 0.01, axis=0).tolist(),
                        q99=np.quantile(v, 0.99, axis=0).tolist(),
                        mean=v.mean(0).tolist(), std=v.std(0).tolist())
    out["_meta"] = dict(n_frames=int(sum(len(a) for a in arrays_state)))
    return out
