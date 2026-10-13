"""Pen spinning on the table: goals are the pen lying flat at a fixed spot, each one ``step_deg`` further round
than the last, in one direction per episode (env_cfg.PenSpinCfg)."""
from __future__ import annotations

import math

import torch

from ..goals import yaw_quat


class PenSpin:
    name = "pen_spin"
    success_steps = 1
    perturbed = False

    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg.pen_spin
        self.step = math.radians(self.cfg.step_deg)
        self.direction = torch.ones(env.num_envs, device=env.device)
        self.goal_yaw = torch.zeros(env.num_envs, device=env.device)

    def _flat_at_centre(self, ids: torch.Tensor, yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        env = self.env
        pos = torch.zeros(len(ids), 3, device=env.device)
        pos[:, 0], pos[:, 1] = self.cfg.centre_xy
        pos[:, 2] = env.pen_rest_height
        return pos, yaw_quat(yaw)

    def reset(self, ids: torch.Tensor) -> None:
        """Pen flat at the centre at a random yaw; a spin direction; the first goal one step round."""
        env, n = self.env, len(ids)
        yaw = (torch.rand(n, device=env.device) * 2 - 1) * math.pi
        pos, quat = self._flat_at_centre(ids, yaw)
        env.place_pen(ids, pos, quat)
        self.direction[ids] = torch.where(torch.rand(n, device=env.device) < 0.5, -1.0, 1.0)
        self.goal_yaw[ids] = yaw
        self.next_goal(ids)

    def next_goal(self, ids: torch.Tensor) -> None:
        self.goal_yaw[ids] = torch.remainder(self.goal_yaw[ids] + self.direction[ids] * self.step + math.pi,
                                             2 * math.pi) - math.pi
        pos, quat = self._flat_at_centre(ids, self.goal_yaw[ids])
        self.env.set_goal(ids, pos, quat)
