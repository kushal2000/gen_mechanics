"""Repose the pen in 6D in the air: a random walk of goal poses (env_cfg.PenReposeCfg).

The pen starts flat on the table at a random yaw. The first goal is a random pose: a uniformly random orientation at a
random centre within the bounds below (``random_first_goal``; or, with it off, the pen picked up flat to ``first_height``
at its starting yaw, the pen_lift / pen_flip_air pickup). Each goal after that is the previous one moved by a random
step --
its orientation turned by ``step_deg`` (drawn from the range) about a random axis -- so the pen's long axis can come to
point anywhere -- and its centre moved by up to ``step_pos`` in a random direction, then clipped: sideways into a box,
and in height between a floor set by THAT orientation (the pen's lowest point ``min_clearance`` above the table: a
flat goal can sit low, a vertical one stands higher) and ``max_height``.
Roll about the long axis is invisible to the keypoint success test, so in effect a goal is a centre and an axis
direction (5 DOF). A goal counts the step it is reached (success_steps 1). Once the pen has been picked up, while it
is lifted, the env's pushes and twists act on it (env_cfg.PerturbCfg); touching the table after the pickup ends the
episode (carried).
"""
from __future__ import annotations

import math

import torch
from isaaclab.utils.math import matrix_from_quat, normalize, quat_from_angle_axis, quat_mul

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
        p = env.cfg.pen
        self.half = (p.length / 2, p.diameter / 2)
        if self.cfg.min_clearance <= env.cfg.termination.table_touch_margin:
            raise ValueError(f"pen_repose.min_clearance {self.cfg.min_clearance}: goals would count as touching the "
                             f"table (termination.table_touch_margin {env.cfg.termination.table_touch_margin})")
        # The floor for the worst orientation must stay below the ceiling, or no height fits.
        worst = env.cfg.table.top_height + self._reach_worst() + self.cfg.min_clearance
        if worst > env.pen_rest_height + self.cfg.max_height:
            raise ValueError(f"pen_repose.max_height {self.cfg.max_height}: below the floor of the worst orientation "
                             f"({(worst - env.pen_rest_height) * 100:.1f} cm above resting)")

    def _reach_worst(self) -> float:
        """The farthest the pen can reach below its centre in any orientation (m)."""
        h, r = self.half
        return math.sqrt(h * h + 2 * r * r) if self.env.cfg.pen.shape == "box" else h

    def reach_below(self, quat: torch.Tensor) -> torch.Tensor:
        """(N,) how far the pen reaches below its centre at orientation ``quat`` (wxyz): the support of its shape
        along -z. A box: the half-extents times the |z| of its axes; a capsule: its segment's half-length times the
        |z| of its axis, plus its radius."""
        R = matrix_from_quat(quat)                          # columns: the pen's axes in the env frame
        h, r = self.half
        if self.env.cfg.pen.shape == "box":
            return h * R[:, 2, 0].abs() + r * (R[:, 2, 1].abs() + R[:, 2, 2].abs())
        return (h - r) * R[:, 2, 0].abs() + r

    def _box(self, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, 3) lower and upper bounds for the goal centre at orientation ``quat``, env frame: sideways a box,
        in height from the orientation's floor to the ceiling."""
        c, env, n = self.cfg, self.env, quat.shape[0]
        lo = torch.tensor([c.centre_xy[0] - c.half_width_xy, c.centre_xy[1] - c.half_width_xy, 0.0],
                          device=env.device).repeat(n, 1)
        hi = torch.tensor([c.centre_xy[0] + c.half_width_xy, c.centre_xy[1] + c.half_width_xy,
                           env.pen_rest_height + c.max_height], device=env.device).repeat(n, 1)
        lo[:, 2] = env.cfg.table.top_height + self.reach_below(quat) + c.min_clearance
        return lo, hi

    def reset(self, ids: torch.Tensor) -> None:
        """Pen flat on the table at a random yaw. The first goal: a random pose in the bounds (random_first_goal), or
        straight above the pen, flat, at first_height."""
        env, n = self.env, len(ids)
        yaw = env.start_pen_yaw(n)
        pos = torch.zeros(n, 3, device=env.device)
        pos[:, 0], pos[:, 1] = self.cfg.centre_xy
        pos[:, 2] = env.pen_rest_height
        env.place_pen(ids, pos, yaw_quat(yaw))
        if self.cfg.random_first_goal:
            quat = normalize(torch.randn(n, 4, device=env.device))      # uniform over orientations
            quat = quat * torch.where(quat[:, :1] < 0, -1.0, 1.0)       # w >= 0, a canonical sign
            lo, hi = self._box(quat)
            env.set_goal(ids, lo + (hi - lo) * torch.rand(n, 3, device=env.device), quat)
        else:
            pos[:, 2] += self.cfg.first_height
            env.set_goal(ids, pos, yaw_quat(yaw))

    def next_goal(self, ids: torch.Tensor) -> None:
        """The previous goal, turned by step_deg about a random axis, then its centre moved by up to step_pos and
        clipped into the bounds for the NEW orientation (so its lowest point stays min_clearance above the table)."""
        env, n, c = self.env, len(ids), self.cfg
        axis = normalize(torch.randn(n, 3, device=env.device))
        a0, a1 = (math.radians(v) for v in c.step_deg)
        angle = a0 + (a1 - a0) * torch.rand(n, device=env.device)
        quat = normalize(quat_mul(quat_from_angle_axis(angle, axis), env.goal_quat[ids]))
        lo, hi = self._box(quat)
        direction = normalize(torch.randn(n, 3, device=env.device))
        dist = torch.rand(n, 1, device=env.device) * c.step_pos
        pos = torch.max(torch.min(env.goal_pos[ids] + direction * dist, hi), lo)
        env.set_goal(ids, pos, quat)
