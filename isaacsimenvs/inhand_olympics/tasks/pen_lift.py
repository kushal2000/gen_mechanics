"""Pick the pen up off the table, flat, and hold it there while it is pushed and twisted (env_cfg.PenLiftCfg).

The pen starts flat at the table centre at a random yaw. The goal is the same flat pose ``lift_height`` higher; it
never moves. The first goal (the pickup) counts after ``first_goal_steps`` steps within tolerance, every later one
after ``hold_steps`` consecutive steps, and the next goal is the same pose, so goals per episode beyond the first
measure how long the pen was held. Once it has been held (the second goal), while it is up, the env's perturbations
(env_cfg.PerturbCfg) push and twist it.
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
        self.first_goal_steps = int(self.cfg.first_goal_steps)
        self.perturb_after = 2                      # goals: the pickup, then one full hold

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
