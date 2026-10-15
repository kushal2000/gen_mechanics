"""Twist the cap off the pen (env_cfg.PenTwistCfg), level 1: the twist pen (pen.twist, twist_pen.py) stands upright
on the table, cap up, its barrel fixed where it stands (pen.twist_fixed_barrel), and the hand above twists the cap.

Two objects, each with a 6D goal (the env's two-part goals): the pen's is where it stands; the cap's is turned
``step_deg`` (90) further in the unscrewing direction -- counter-clockwise seen from above, +x about the pen's axis --
than the last. The env's keypoint progress and goal bonus act on the worse of the two, so here on the cap; goals per
episode = quarter turns. Level 2 will set the pen's goal too, to wherever it should be held.

A pose can't tell full turns apart: screwed back 270 deg the cap looks just as it does unscrewed 90 deg. So the task
also follows the cap's total turn (the joint angle, which PhysX wraps to +-pi, accumulated step by step; the policy
doesn't see it) and gates success on it: a goal counts only with the total turn within half a step of the goal's.
"""
from __future__ import annotations

import math

import torch
from isaaclab.utils.math import quat_apply, quat_mul


class PenTwist:
    name = "pen_twist"
    carried = False
    success_steps = 1
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
        self.step = math.radians(c.step_deg)
        self.angle = torch.zeros(env.num_envs, device=env.device)        # the current goal's twist (rad)
        self.turned = torch.zeros(env.num_envs, device=env.device)       # the cap's total turn (rad), unwrapped
        self.prev = torch.zeros(env.num_envs, device=env.device)

    def _goal(self, ids: torch.Tensor) -> None:
        """The pen where it stands; the cap at self.angle (the upright pen's orientation turned about its own axis)."""
        n = len(ids)
        pen_pos, pen_quat = self.pos.repeat(n, 1), self.quat.repeat(n, 1)
        half = self.angle[ids] / 2
        twist = torch.stack([torch.cos(half), torch.sin(half), torch.zeros_like(half), torch.zeros_like(half)], -1)
        cap_quat = quat_mul(pen_quat, twist)
        cap_pos = pen_pos + quat_apply(cap_quat, self.env.cap_centre.expand(n, 3))
        self.env.set_goal(ids, pen_pos, pen_quat)
        self.env.set_cap_goal(ids, cap_pos, cap_quat)

    def reset(self, ids: torch.Tensor) -> None:
        n = len(ids)
        self.env.place_pen(ids, self.pos.repeat(n, 1), self.quat.repeat(n, 1))      # the barrel; the cap at 0
        self.angle[ids] = self.step
        self.turned[ids] = 0.0
        self.prev[ids] = 0.0
        self._goal(ids)

    def gate(self, near: torch.Tensor) -> torch.Tensor:
        """Called every step with the envs' pose-based success (all envs; the env applies it to this task's): keep it
        only where the cap's total turn is within half a step of the goal's."""
        now = self.env.pen.data.joint_pos[:, 0]
        self.turned += torch.remainder(now - self.prev + math.pi, 2 * math.pi) - math.pi
        self.prev = now.clone()
        return near & ((self.turned - self.angle).abs() < self.step / 2)

    def next_goal(self, ids: torch.Tensor) -> None:
        self.angle[ids] += self.step
        self._goal(ids)
