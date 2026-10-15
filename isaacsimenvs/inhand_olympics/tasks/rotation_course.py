"""Staying on course for the rotating tasks (pen_spin_air, pen_flip_air): their success and termination, not their
reward (env_cfg.PenSpinAirCfg / PenFlipAirCfg forward_only, max_off_axis_deg, max_backward_deg; all off by default).

The task's axis: the vertical for a spin, the horizontal axis across the pen for a flip, signed by the episode's
direction. From the pickup (the first goal) on, the pen's angular velocity about it is integrated each step into how
far it has turned (``progress``), and its furthest so far kept (``best``).
  forward_only      a goal counts only with progress >= (goals after the pickup) x step - step / 2, so the pen must
                    really have turned that far the right way -- not snapped back, nor come round some other way
  max_off_axis_deg  the episode ends once the pen's long axis is more than this out of the task's plane (tilted out
                    of the horizontal while spinning; swung out of the vertical plane of the flip)
  max_backward_deg  the episode ends once progress has fallen this far below best
"""
from __future__ import annotations

import math

import torch
from isaaclab.utils.math import quat_apply


class RotationCourse:
    """Mixed into a rotating task: needs self.env, self.cfg, self.step (rad) and self.direction (N,) set, and
    ``axis()`` -> (N, 3) unit world axis (unsigned)."""

    def init_course(self) -> None:
        n, dev = self.env.num_envs, self.env.device
        self.progress = torch.zeros(n, device=dev)
        self.best = torch.zeros(n, device=dev)

    def reset_course(self, ids: torch.Tensor) -> None:
        self.progress[ids] = 0.0
        self.best[ids] = 0.0

    def gate(self, near: torch.Tensor) -> torch.Tensor:
        """Every step (env._get_dones, all envs; applied to this task's): integrate the turn, gate success."""
        env = self.env
        lifted = env.successes >= 1
        along = (env.pen.data.root_ang_vel_w * self.axis()).sum(-1) * self.direction
        self.progress += torch.where(lifted, along * env.step_dt, torch.zeros_like(along))
        self.best = torch.maximum(self.best, self.progress)
        if not self.cfg.forward_only:
            return near
        return near & (~lifted | (self.progress >= env.successes * self.step - self.step / 2))

    def failed(self) -> torch.Tensor:
        """(N,) off course: the pen out of the task's plane, or turned back too far (after the pickup)."""
        env, c = self.env, self.cfg
        out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        lifted = env.successes >= 1
        if c.max_off_axis_deg > 0:
            _, quat = env.pen_pose()
            long_axis = quat_apply(quat, torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3))
            off = torch.asin((long_axis * self.axis()).sum(-1).abs().clamp(max=1.0))      # out of the plane
            out |= off > math.radians(c.max_off_axis_deg)
        if c.max_backward_deg > 0:
            out |= self.best - self.progress > math.radians(c.max_backward_deg)
        return out & lifted
