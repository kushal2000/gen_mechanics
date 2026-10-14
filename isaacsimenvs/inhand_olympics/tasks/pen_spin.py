"""Pen spinning, on the table (PenSpin) and in the air (PenSpinAir): goals are the pen flat at a fixed spot, each
one ``step_deg`` further round (about the vertical) than the last, in one direction per episode (env_cfg.PenSpinCfg,
PenSpinAirCfg)."""
from __future__ import annotations

import math

import torch

from ..goals import yaw_quat


class PenSpin:
    name = "pen_spin"
    success_steps = 1
    perturbed = False
    carried = False

    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg.pen_spin
        self.step = math.radians(self.cfg.step_deg)
        self.direction = torch.ones(env.num_envs, device=env.device)
        self.goal_yaw = torch.zeros(env.num_envs, device=env.device)
        self.goal_height = 0.0

    def _flat_at_centre(self, ids: torch.Tensor, yaw: torch.Tensor,
                        height: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """The pen flat at the centre at ``yaw``, ``height`` above where it rests on the table."""
        env = self.env
        pos = torch.zeros(len(ids), 3, device=env.device)
        pos[:, 0], pos[:, 1] = self.cfg.centre_xy
        pos[:, 2] = env.pen_rest_height + height
        return pos, yaw_quat(yaw)

    def reset(self, ids: torch.Tensor) -> None:
        """Pen flat at the centre at a random yaw; a spin direction; the first goal one step round."""
        env, n = self.env, len(ids)
        yaw = env.start_pen_yaw(n)
        pos, quat = self._flat_at_centre(ids, yaw)
        env.place_pen(ids, pos, quat)
        self.direction[ids] = torch.where(torch.rand(n, device=env.device) < 0.5, -1.0, 1.0)
        self.goal_yaw[ids] = yaw
        self.next_goal(ids)

    def next_goal(self, ids: torch.Tensor) -> None:
        self.goal_yaw[ids] = torch.remainder(self.goal_yaw[ids] + self.direction[ids] * self.step + math.pi,
                                             2 * math.pi) - math.pi
        pos, quat = self._flat_at_centre(ids, self.goal_yaw[ids], self.goal_height)
        self.env.set_goal(ids, pos, quat)


class PenSpinAir(PenSpin):
    """Spin the pen in the air: picked up flat to pen_lift's lift_height at its starting yaw (the first goal), then
    each goal ``step_deg`` further round at that height."""

    name = "pen_spin_air"
    carried = True                  # the pen touching the table after the first goal ends the episode

    def __init__(self, env):
        super().__init__(env)
        self.cfg = env.cfg.pen_spin_air
        self.step = math.radians(self.cfg.step_deg)
        self.goal_height = env.cfg.pen_lift.lift_height        # the same height as the held pen

    def reset(self, ids: torch.Tensor) -> None:
        """Pen flat on the table at a random yaw; a spin direction; the first goal straight above it."""
        env, n = self.env, len(ids)
        yaw = env.start_pen_yaw(n)
        env.place_pen(ids, *self._flat_at_centre(ids, yaw))
        self.direction[ids] = torch.where(torch.rand(n, device=env.device) < 0.5, -1.0, 1.0)
        self.goal_yaw[ids] = yaw
        env.set_goal(ids, *self._flat_at_centre(ids, yaw, self.goal_height))
