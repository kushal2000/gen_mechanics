"""Repose the pen in 6D in the air: a random walk of goal poses (env_cfg.PenReposeCfg).

The pen starts flat on the table at a random yaw. The first goal is it picked up flat to ``first_height`` at that yaw
(the same pickup as pen_lift and pen_flip_air); each goal after that is the previous one moved by a random step --
its centre by up to ``step_pos`` in a random direction, kept inside a box above the table, and its orientation turned
by ``step_deg`` (drawn from the range) about a random axis -- so the pen's long axis can come to point anywhere.
Roll about the long axis is invisible to the keypoint success test, so in effect a goal is a centre and an axis
direction (5 DOF). A goal counts the step it is reached (success_steps 1). Once the pen has been picked up, while it
is lifted, the env's pushes and twists act on it (env_cfg.PerturbCfg); touching the table after the pickup ends the
episode (carried).
"""
from __future__ import annotations

import math

import torch
from isaaclab.utils.math import normalize, quat_from_angle_axis, quat_mul

from ..goals import yaw_quat


class PenRepose:
    name = "pen_repose"
    carried = True                  # the pen touching the table after the first goal ends the episode
    success_steps = 1
    perturbed = True
    perturb_after = 1               # goals: pushes start once the pen has been picked up

    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg.pen_repose
        # Every goal clear of the table: the lowest centre in the box minus the farthest the pen reaches below its
        # centre in ANY orientation (half its space diagonal for a box; half its overall length, caps included, for a
        # capsule).
        p = env.cfg.pen
        reach = (math.sqrt((p.length / 2) ** 2 + 2 * (p.diameter / 2) ** 2) if p.shape == "box"
                 else p.length / 2)
        self.min_clearance = p.diameter / 2 + self.cfg.height_range[0] - reach     # above the table top, m
        if self.min_clearance <= env.cfg.termination.table_touch_margin:
            raise ValueError(f"pen_repose.height_range {self.cfg.height_range}: a goal can reach "
                             f"{self.min_clearance * 100:.1f} cm from the table (the pen reaches {reach * 100:.1f} cm "
                             f"below its centre); raise height_range[0]")

    def _box(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The box the goal's centre stays in, env frame."""
        c, env = self.cfg, self.env
        lo = torch.tensor([c.centre_xy[0] - c.half_width_xy, c.centre_xy[1] - c.half_width_xy,
                           env.pen_rest_height + c.height_range[0]], device=env.device)
        hi = torch.tensor([c.centre_xy[0] + c.half_width_xy, c.centre_xy[1] + c.half_width_xy,
                           env.pen_rest_height + c.height_range[1]], device=env.device)
        return lo, hi

    def reset(self, ids: torch.Tensor) -> None:
        """Pen flat on the table at a random yaw; the first goal straight above it, flat, at first_height."""
        env, n = self.env, len(ids)
        yaw = env.start_pen_yaw(n)
        pos = torch.zeros(n, 3, device=env.device)
        pos[:, 0], pos[:, 1] = self.cfg.centre_xy
        pos[:, 2] = env.pen_rest_height
        env.place_pen(ids, pos, yaw_quat(yaw))
        pos[:, 2] += self.cfg.first_height
        env.set_goal(ids, pos, yaw_quat(yaw))

    def next_goal(self, ids: torch.Tensor) -> None:
        """The previous goal, moved: centre by up to step_pos (clipped into the box), turned by step_deg about a random
        axis."""
        env, n, c = self.env, len(ids), self.cfg
        lo, hi = self._box()
        direction = normalize(torch.randn(n, 3, device=env.device))
        dist = torch.rand(n, 1, device=env.device) * c.step_pos
        pos = torch.max(torch.min(env.goal_pos[ids] + direction * dist, hi), lo)
        axis = normalize(torch.randn(n, 3, device=env.device))
        a0, a1 = (math.radians(v) for v in c.step_deg)
        angle = a0 + (a1 - a0) * torch.rand(n, device=env.device)
        quat = normalize(quat_mul(quat_from_angle_axis(angle, axis), env.goal_quat[ids]))
        env.set_goal(ids, pos, quat)
