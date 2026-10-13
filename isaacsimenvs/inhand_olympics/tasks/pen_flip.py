"""Flip the pen end over end in the air (env_cfg.PenFlipAirCfg).

The pen starts flat on the table at a random yaw. The first goal is it picked up flat to pen_lift's lift_height at
that yaw; each goal after that turns it ``step_deg`` further about the horizontal axis across the pen (one direction
per episode), its centre staying put: flat, tilted, vertical tip up, flat reversed, vertical tip down, ... The pen's
keypoints lie on its axis, so a flat pen turned end for end is indistinguishable from one yawed 180 degrees; the
tilted and vertical goals in between are what make the policy flip it rather than spin it.
"""
from __future__ import annotations

import math

import torch
from isaaclab.utils.math import quat_from_angle_axis, quat_mul

from ..goals import yaw_quat


class PenFlipAir:
    name = "pen_flip_air"
    carried = True                  # the pen touching the table after the first goal ends the episode
    success_steps = 1
    perturbed = False

    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg.pen_flip_air
        self.step = math.radians(self.cfg.step_deg)
        self.height = env.cfg.pen_lift.lift_height            # the same height as the held pen
        n = env.num_envs
        self.yaw = torch.zeros(n, device=env.device)
        self.pitch = torch.zeros(n, device=env.device)
        self.direction = torch.ones(n, device=env.device)

    def _goal(self, ids: torch.Tensor) -> None:
        env = self.env
        pos = torch.zeros(len(ids), 3, device=env.device)
        pos[:, 0], pos[:, 1] = self.cfg.centre_xy
        pos[:, 2] = env.pen_rest_height + self.height
        yaw = self.yaw[ids]
        across = torch.stack([-torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw)], dim=-1)  # horizontal, across
        quat = quat_mul(quat_from_angle_axis(self.pitch[ids], across), yaw_quat(yaw))
        env.set_goal(ids, pos, quat)

    def reset(self, ids: torch.Tensor) -> None:
        """Pen flat on the table at a random yaw; a flip direction; the first goal straight above it, flat."""
        env, n = self.env, len(ids)
        self.yaw[ids] = (torch.rand(n, device=env.device) * 2 - 1) * math.pi
        self.pitch[ids] = 0.0
        self.direction[ids] = torch.where(torch.rand(n, device=env.device) < 0.5, -1.0, 1.0)
        pos = torch.zeros(n, 3, device=env.device)
        pos[:, 0], pos[:, 1] = self.cfg.centre_xy
        pos[:, 2] = env.pen_rest_height
        env.place_pen(ids, pos, yaw_quat(self.yaw[ids]))
        self._goal(ids)

    def next_goal(self, ids: torch.Tensor) -> None:
        self.pitch[ids] = torch.remainder(self.pitch[ids] + self.direction[ids] * self.step, 2 * math.pi)
        self._goal(ids)
