#!/usr/bin/env python
"""LIBERO / LIBERO-Plus evaluation client (runs in the simulator environment).

Start the policy server first (``scripts/serve.py --model <fastvla-libero>``), then:

    # LIBERO: 4 suites x 10 tasks x 50 initial states = 2,000 rollouts
    python scripts/eval_libero.py --benchmark libero --suites libero_spatial --out results/libero
    # LIBERO-Plus (needs the LIBERO-Plus fork installed as `libero`): 10,030 variants x 1 rollout
    python scripts/eval_libero.py --benchmark libero_plus --suites libero_10 --out results/plus \
        --shard 0 --nshard 4

Protocols:
  LIBERO       256x256 render, env.seed(7), initial state i for trial i, 10 no-op steps,
               per-suite step limits 220/280/300/520, full chunk executed before replanning.
  LIBERO-Plus  128x128 render upsampled to 256 (bilinear), env.seed(0), first initial state,
               same no-op / step limits. Variants of every factor except "Language
               Instructions" use the original task instruction.
Images are passed to the policy in the orientation of the LIBERO demonstrations (the raw
simulator frame).
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "fastvla"))
from client import PolicyClient  # noqa: E402  (stdlib + numpy only)

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
NUM_STEPS_WAIT = 10
DUMMY_ACTION = [0.0] * 6 + [-1.0]


def quat2axisangle(quat):
    """robosuite's implementation."""
    q = np.asarray(quat, dtype=np.float64).copy()
    q[3] = min(max(q[3], -1.0), 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def robot_state(obs):
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                           obs["robot0_gripper_qpos"]])


def views_std(obs):
    return [np.ascontiguousarray(obs["agentview_image"]),
            np.ascontiguousarray(obs["robot0_eye_in_hand_image"])]


def views_plus(obs):
    """128 -> 256 bilinear upsampling done on the evaluator-oriented (180-degree rotated)
    frame, then rotated back."""
    from PIL import Image

    def f(img):
        x = np.ascontiguousarray(img[::-1, ::-1])
        x = np.asarray(Image.fromarray(x).resize((256, 256), Image.BILINEAR))
        return np.ascontiguousarray(x[::-1, ::-1])
    return [f(obs["agentview_image"]), f(obs["robot0_eye_in_hand_image"])]


def rollout(env, cli, goal, init_state, max_steps, views_fn):
    """-> (success, policy steps)."""
    cli.reset()
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, done, _ = env.step(DUMMY_ACTION)
    t, done = 0, False
    while t < max_steps and not done:
        chunk = cli.act(views_fn(obs), robot_state(obs), goal)
        for a in chunk:
            if t >= max_steps or done:
                break
            obs, _, done, _ = env.step(np.asarray(a).tolist())
            t += 1
    return bool(done), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["libero", "libero_plus"], default="libero")
    ap.add_argument("--suites", nargs="+", default=list(MAX_STEPS))
    ap.add_argument("--trials", type=int, default=50, help="LIBERO only")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0, help="LIBERO-Plus only")
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import torch
    _load = torch.load               # LIBERO(-Plus) init states need weights_only=False

    def _load_compat(*args, **kw):
        kw.setdefault("weights_only", False)
        return _load(*args, **kw)
    torch.load = _load_compat
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    cli = PolicyClient(a.host, a.port)
    assert cli.info["benchmark"] == "libero", f"server runs a {cli.info['benchmark']} model"
    base_lang = json.load(open(os.path.join(ASSETS, "libero_task_language.json")))
    plus = a.benchmark == "libero_plus"
    manifest = json.load(open(os.path.join(ASSETS, "libero_plus_manifest.json"))) if plus else None

    for suite in a.suites:
        ts = benchmark.get_benchmark_dict()[suite]()
        assert (ts.n_tasks > 10) == plus, f"{suite} has {ts.n_tasks} tasks: wrong LIBERO package"
        ms = MAX_STEPS[suite]
        tag = f"{suite}_s{a.shard}of{a.nshard}" if plus else suite
        rows, t0 = [], time.time()
        if plus:
            np.random.seed(0)
            jobs = [r for r in manifest if r["suite"] == suite][a.shard::a.nshard]
        else:
            np.random.seed(7)
            jobs = [dict(idx=i, task_id=i, factor="") for i in range(ts.n_tasks)]
        for r in jobs:
            task = ts.get_task(r["idx"])
            if plus and r["factor"] != "Language Instructions":
                goal = base_lang[suite][r["task_id"]]
            else:
                goal = task.language
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            res = 128 if plus else 256
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=res, camera_widths=res)
            env.seed(0 if plus else 7)
            inits = ts.get_task_init_states(r["idx"])
            for trial in range(1 if plus else a.trials):
                try:
                    ok, steps = rollout(env, cli, goal, inits[trial], ms,
                                        views_plus if plus else views_std)
                except Exception as e:                    # a broken variant counts as failure
                    print(f"[{suite}] idx {r['idx']} trial {trial}: {type(e).__name__}: {e}")
                    ok, steps = False, 0
                rows.append([r["idx"], r["task_id"], r["factor"], trial, int(ok), steps])
            env.close()
            n_ok = sum(x[4] for x in rows)
            print(f"[{suite}] {len(rows)} rollouts, {n_ok} success "
                  f"({100 * n_ok / len(rows):.1f}%), {time.time() - t0:.0f}s", flush=True)
        with open(os.path.join(a.out, f"trials_{tag}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["variant_idx", "task_id", "factor", "trial", "success", "steps"])
            w.writerows(rows)
        json.dump(dict(suite=suite, n=len(rows), success=sum(x[4] for x in rows)),
                  open(os.path.join(a.out, f"result_{tag}.json"), "w"))


if __name__ == "__main__":
    main()
