#!/usr/bin/env python
"""Check an installation against reference actions of the released checkpoints.

    python tests/test_reference.py --robotwin checkpoints/fastvla-robotwin --libero checkpoints/fastvla-libero

Runs the full policy (DM0.5 prefill + decoder) on stored observations. On the GPU type and
library versions they were recorded with the difference is exactly 0; other setups give
small bf16 differences, so the check uses a tolerance relative to the action range.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastvla.policy import FastVLAPolicy  # noqa: E402

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_actions.npz")


def check(name, model_dir, ref, steps, hist, device="cuda"):
    pol = FastVLAPolicy(model_dir, device)
    instr = str(ref[f"{name}_instruction"])
    ok = True
    for k in range(steps):
        p = f"{name}_{k}/"
        h = {-17: list(ref[p + "hist_m17"]), -8: list(ref[p + "hist_m8"])} if hist else None
        a = pol.act(list(ref[p + "views"]), ref[p + "state"], instr, hist_views=h)
        want = ref[p + "action"]
        err = float(np.abs(a - want).max())
        scale = float(np.abs(want).max()) + 1e-6
        good = err <= 2e-2 * scale
        ok &= good
        print(f"[{name}] call {k}: max|diff| = {err:.2e} ({'OK' if good else 'FAIL'})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robotwin", default=None)
    ap.add_argument("--libero", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    ref = np.load(REF)
    ok = True
    if a.robotwin:
        ok &= check("robotwin", a.robotwin, ref, 2, hist=True, device=a.device)
    if a.libero:
        ok &= check("libero", a.libero, ref, 2, hist=False, device=a.device)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
