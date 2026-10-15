"""RoboTwin 2.0 policy adapter for FastVLA.

Copy this folder to ``<RoboTwin>/policy/FastVLA``. It only talks to the FastVLA policy
server (``scripts/serve.py``) over TCP, so it runs in RoboTwin's own environment.

Per replanning step the model receives the current head / left-wrist / right-wrist
RGB frames, the 14-D joint target vector, and the episode instruction; it returns
50 absolute joint targets (``action_type="qpos"``) that are all executed before
replanning. The model also looks at the head camera 17 and 8 steps before the
current frame; those two frames are captured while the previous chunk executes.
"""
import os
import sys

import numpy as np

_FASTVLA = os.environ.get("FASTVLA_ROOT")
if _FASTVLA:
    sys.path.insert(0, os.path.join(_FASTVLA, "fastvla"))
from client import PolicyClient  # noqa: E402  (fastvla/client.py: stdlib + numpy only)


class FastVLAAgent:
    def __init__(self, host, port):
        self.cli = PolicyClient(host, port)
        assert self.cli.info["benchmark"] == "robotwin", self.cli.info
        self.hist_offsets = list(self.cli.info["hist_offsets"])      # e.g. [-17, -8]
        self.hist = {}

    def reset(self):
        self.cli.reset()
        self.hist = {}

    def get_action(self, obs, instruction):
        a = self.cli.act(obs["views"], obs["state"], instruction, self.hist or None)
        self.hist = {}                          # history frames belong to this call only
        return a


def encode_obs(observation):
    o = observation["observation"]
    return dict(views=[np.asarray(o[c]["rgb"], np.uint8)
                       for c in ("head_camera", "left_camera", "right_camera")],
                state=np.asarray(observation["joint_action"]["vector"], np.float32))


def get_model(usr_args):
    return FastVLAAgent(usr_args.get("host", "127.0.0.1"), int(usr_args.get("port", 8765)))


def eval(TASK_ENV, model, observation):
    actions = model.get_action(encode_obs(observation), TASK_ENV.get_instruction())
    n = len(actions)
    # After executing action i (0-based) the env is (i + 1 - n) steps before the next
    # decision point; grab the frames the model wants as history there.
    grab = {n + o: o for o in model.hist_offsets if 0 < n + o <= n}
    for i, action in enumerate(actions):
        TASK_ENV.take_action(action, action_type="qpos")
        if (i + 1) in grab:
            model.hist[grab[i + 1]] = encode_obs(TASK_ENV.get_obs())["views"]
        elif TASK_ENV.eval_video_path is not None:
            TASK_ENV.get_obs()


def reset_model(model):
    model.reset()
