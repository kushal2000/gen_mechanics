"""Pick the pen up off the table, flat, and hold it there while it is pushed and twisted (env_cfg.PenLiftCfg).

The pen starts flat at the table centre at a random yaw. The goal is the same flat pose ``lift_height`` higher; it
never moves. A goal counts after ``hold_steps`` consecutive steps within tolerance, and the next goal is the same
pose, so goals per episode measure how long the pen was held. From the first hit on, while the pen is up, the env's
perturbations (env_cfg.PerturbCfg) push and twist it.
"""
from __future__ import annotations

import math

import torch

from ..goals import yaw_quat


class PenLift:
    name = "pen_lift"
    perturbed = True

    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg.pen_lift
        self.success_steps = int(self.cfg.hold_steps)

    def reset(self, ids: torch.Tensor) -> None:
        env, n = self.env, len(ids)
        yaw = (torch.rand(n, device=env.device) * 2 - 1) * math.pi
        pos = torch.zeros(n, 3, device=env.device)
        pos[:, 0], pos[:, 1] = self.cfg.centre_xy
        pos[:, 2] = env.pen_rest_height
        env.place_pen(ids, pos, yaw_quat(yaw))
        pos[:, 2] += self.cfg.lift_height
        env.set_goal(ids, pos, yaw_quat(yaw))

    def next_goal(self, ids: torch.Tensor) -> None:
        """Hold on: the goal stays where it is."""
