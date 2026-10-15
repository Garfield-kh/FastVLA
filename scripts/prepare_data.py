#!/usr/bin/env python
"""Index the raw datasets and compute normalisation statistics.

    # RoboTwin 2.0 (aloha-agilex, 50 clean + 500 randomized demos per task)
    python scripts/prepare_data.py robotwin --clean-root <.../clean> --rand-root <.../randomized> \
        --out data/robotwin
    # LIBERO (official hdf5 demos; run in an environment that has `libero` and `h5py`)
    python scripts/prepare_data.py libero --hdf5-dir <LIBERO>/datasets --out data/libero

Both roots are laid out as ``<root>/<task>/<setting>/data/episode<N>.hdf5`` with the
instructions in ``<root>/<task>/<setting>/instructions/episode<N>.json``.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="bench", required=True)
    r = sub.add_parser("robotwin")
    r.add_argument("--clean-root", required=True)
    r.add_argument("--rand-root", required=True)
    r.add_argument("--out", default="data/robotwin")
    lb = sub.add_parser("libero")
    lb.add_argument("--hdf5-dir", required=True)
    lb.add_argument("--out", default="data/libero")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.bench == "robotwin":
        from fastvla.data.robotwin import build_index, compute_norm_stats
        meta = build_index(a.clean_root, a.rand_root, os.path.join(a.out, "index.json"))
        print(f"[robotwin] {meta}")
        assert meta["n_episodes"] == 27500, "expected 50 tasks x (50 clean + 500 randomized)"
        compute_norm_stats(os.path.join(a.out, "index.json"), os.path.join(a.out, "norm_stats.json"))
    else:
        from fastvla.data.libero import SUITES, compute_norm_stats, convert_suite
        for s in SUITES:
            convert_suite(a.hdf5_dir, s, os.path.join(a.out, s))
            print(f"[libero] {s} done")
        compute_norm_stats(a.out, os.path.join(a.out, "norm_stats.json"))
    print(f"[{a.bench}] wrote {a.out}")


if __name__ == "__main__":
    main()
