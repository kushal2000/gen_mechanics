"""Twist the cap off the pen (env_cfg.PenTwistCfg), level 1: the twist pen (pen.twist, twist_pen.py) stands upright
on the table, cap up, its barrel fixed where it stands (pen.twist_fixed_barrel), and the hand above twists the cap.

Set-up only so far: the pen is placed and the goal is the pen's own pose, with success turned off (success_steps
out of reach), so the keypoint success the env counts for the other tasks never fires. Getting the cap off -- and the
reward for it -- comes next.
"""
from __future__ import annotations

import math

import torch


class PenTwist:
    name = "pen_twist"
    carried = False
    success_steps = 10 ** 9         # no keypoint success: the cap coming off is this task's success (not built yet)
    perturbed = False

    def __init__(self, env):
        if not (env.cfg.pen.twist and env.cfg.pen.twist_fixed_barrel):
            raise ValueError("pen_twist needs pen.twist=True and pen.twist_fixed_barrel=True")
        self.env = env
        self.cfg = env.cfg.pen_twist
        c, p = self.cfg, env.cfg.pen
        # Upright: the pen's long axis (local +x, the cap's end) turned to world +z, a -90 deg turn about y.
        self.quat = torch.tensor([math.cos(-math.pi / 4), 0.0, math.sin(-math.pi / 4), 0.0], device=env.device)
        self.pos = torch.tensor([c.centre_xy[0], c.centre_xy[1], env.cfg.table.top_height + p.length / 2 + c.lift],
                                device=env.device)

    def reset(self, ids: torch.Tensor) -> None:
        n = len(ids)
        pos, quat = self.pos.repeat(n, 1), self.quat.repeat(n, 1)
        self.env.place_pen(ids, pos, quat)
        self.env.set_goal(ids, pos, quat)

    def next_goal(self, ids: torch.Tensor) -> None:
        pass
