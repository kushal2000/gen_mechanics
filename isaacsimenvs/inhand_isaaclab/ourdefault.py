"""Our env's task definition, rebuilt from manager terms so it runs inside IsaacLab's env.

Used by ``InHandIsaacLabOurDefaultCfg``: the reference env moved all the way to our default
in-hand setup except for the observation and the network. Every term here reproduces a piece
of ``isaacsimenvs/pose_reaching_6d`` / ``inhand_reorient`` at OUR magnitudes, with the source
it mirrors cited, so a difference in outcome is a difference in something else.

Magnitudes: RewardManager multiplies every term by ``weight * step_dt`` (reward_manager.py:150);
our env does not. At our 60 Hz (step_dt 1/60) a weight of ``w_ours / step_dt`` therefore pays
exactly our per-step reward -- see the weights in the cfg.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import random_orientation
from isaaclab_tasks.manager_based.manipulation.inhand.mdp.commands.commands_cfg import (
    InHandReOrientationCommandCfg,
)
from isaaclab_tasks.manager_based.manipulation.inhand.mdp.commands.orientation_command import (
    InHandReOrientationCommand,
)

from .keypoint_reward import KEYPOINT_SCALE, REF_EDGE_M, keypoint_residual_normalised


def keypoint_residual_m(env, goal_quat: torch.Tensor, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Our orientation residual in METRES: max_i |(R_obj - R_goal) o_i| over our 4 keypoints.

    observations.py:245-252 with orientation_only_goal. keypoint_residual_normalised divides by
    the keypoint radius |o|; this undoes that so thresholds and rewards use our units.
    """
    scale = getattr(env.cfg.scene.object.spawn, "scale", None)
    edge = REF_EDGE_M * (float(scale[0]) if scale else 1.0)
    radius = 0.5 * KEYPOINT_SCALE * edge * math.sqrt(3.0)
    return keypoint_residual_normalised(env, goal_quat, object_cfg) * radius


# ---------------------------------------------------------------------------------- command

class KeypointGoalCommand(InHandReOrientationCommand):
    """Their command term with OUR success test, OUR goal distribution and OUR per-goal timer.

    * success: keypoint residual <= keypoint_tol_m (observations.py:269 -- tol =
      success_tolerance * keypoint_scale), instead of quat angle < 0.1 rad. At 20 deg this admits
      20-35 deg depending on the rotation axis, as in our env.
    * goals: Haar-uniform on SO(3) (random_orientation, as reset.py's goal sampler), instead of
      rot_x(u pi) * rot_y(v pi), which never asks for a pure yaw.
    * timer: a success zeroes episode_length_buf for that env (termination.py:12-13, "zero the
      length buf so truncation doesn't fire"), so every GOAL gets the full episode budget.
    """

    cfg: "KeypointGoalCommandCfg"

    def _update_metrics(self):
        self.metrics["orientation_error"] = math_utils.quat_error_magnitude(
            self.object.data.root_quat_w, self.quat_command_w)
        self.metrics["position_error"] = torch.norm(
            self.object.data.root_pos_w - self.pos_command_w, dim=1)
        r = keypoint_residual_m(self._env, self.quat_command_w,
                                SceneEntityCfg(self.cfg.asset_name))
        self.metrics["keypoint_residual_m"] = r
        self._success = r <= self.cfg.keypoint_tol_m
        self.metrics["consecutive_success"] += self._success.float()

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        if n:
            self.quat_command_w[env_ids] = random_orientation(n, device=self.device)

    def _update_command(self):
        if not self.cfg.update_goal_on_success or not hasattr(self, "_success"):
            return
        ids = self._success.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel():
            self._resample(ids)
            if self.cfg.reset_timer_on_success:
                self._env.episode_length_buf[ids] = 0


@configclass
class KeypointGoalCommandCfg(InHandReOrientationCommandCfg):
    class_type: type = KeypointGoalCommand
    keypoint_tol_m: float = 0.020302         # 20 deg: 0.045*sqrt(3)*sin(10 deg) * 1.5
    reset_timer_on_success: bool = True


# ---------------------------------------------------------------------------------- rewards

class keypoint_progress_m(ManagerTermBase):
    """rewards.py:40-51 keypoint_reward, in metres: max(0, best_so_far - residual).

    Record cleared on reset and whenever the goal changes (reset.py:586, _clear_goal_trackers).
    The inf sentinel makes a goal's first step pay 0, as our -1.0 sentinel does
    (observations.py:257). ``lifted`` is not modelled: in our in-hand env it is latched true
    at reset (reset.py:610) and can never clear, so it is identically 1.
    """

    def __init__(self, cfg: RewardTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._best = torch.full((env.num_envs,), float("inf"), device=env.device)
        self._goal = torch.zeros((env.num_envs, 4), device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._best[env_ids] = float("inf")
        self._goal[env_ids] = 0.0

    def __call__(self, env, command_name: str,
                 object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
        goal = env.command_manager.get_term(command_name).command[:, 3:7]
        r = keypoint_residual_m(env, goal, object_cfg)
        fresh = torch.isinf(self._best) | ((goal - self._goal).abs().sum(-1) > 1e-6)
        self._best = torch.where(fresh, r, self._best)
        self._goal = goal.clone()
        delta = (self._best - r).clamp(min=0.0)
        self._best = torch.minimum(self._best, r)
        return delta


def keypoint_success(env, command_name: str, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
                     ) -> torch.Tensor:
    """1 on a step where the keypoint residual is inside tolerance (rewards.py reach_goal_bonus
    with success_steps 1). Same test the command uses, evaluated on the same state."""
    term = env.command_manager.get_term(command_name)
    r = keypoint_residual_m(env, term.command[:, 3:7], object_cfg)
    return (r <= term.cfg.keypoint_tol_m).float()


def hand_joint_vel_l1(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """rewards.py action_penalty: an L1 joint-VELOCITY penalty despite the legacy name."""
    return env.scene[asset_cfg.name].data.joint_vel.abs().sum(dim=-1)


# ---------------------------------------------------------------------------------- termination

def object_away_from_palm(env, threshold: float = 0.3, palm_body: str = "palm_link",
                          object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
                          robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """termination.py:54-64: dropped = the object is more than drop_distance_m from the palm.

    Ours measures to the palm CENTRE, a fixed offset from palm_link's origin (~23 mm for the
    Allegro); negligible against 0.3 m. Theirs measured to the articulation root.
    """
    robot = env.scene[robot_cfg.name]
    i = robot.data.body_names.index(palm_body)
    d = env.scene[object_cfg.name].data.root_pos_w - robot.data.body_pos_w[:, i]
    return torch.norm(d, dim=-1) > threshold


# ---------------------------------------------------------------------------------- events

def reset_object_like_ours(env, env_ids: torch.Tensor, jitter: float = 0.005,
                           asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> None:
    """reset.py:428-515 with our in-hand settings: Haar-uniform start orientation, the default
    spawn position, +-in_hand_position_noise (5 mm) jitter across the palm, zero velocity."""
    obj = env.scene[asset_cfg.name]
    n = len(env_ids)
    state = obj.data.default_root_state[env_ids].clone()
    pos = state[:, 0:3] + env.scene.env_origins[env_ids]
    noise = torch.empty(n, 3, device=env.device).uniform_(-jitter, jitter)
    noise[:, 2] = 0.0
    pos = pos + noise
    quat = random_orientation(n, device=env.device)
    obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)


def reset_hand_like_ours(env, env_ids: torch.Tensor, pos_interval: float = 0.1,
                         vel_interval: float = 0.5,
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """reset.py:344-366 _randomize_robot_dof_state for the hand: lerp reset_dof_pos_random_
    interval_fingers (0.1) of the way from the default pose toward a uniform draw over the full
    joint range, and joint velocity U(-reset_dof_vel_random_interval, +)."""
    robot = env.scene[asset_cfg.name]
    default = robot.data.default_joint_pos[env_ids]
    lo = robot.data.joint_pos_limits[env_ids, :, 0]
    hi = robot.data.joint_pos_limits[env_ids, :, 1]
    sampled = lo + (hi - lo) * torch.rand_like(default)
    q = torch.lerp(default, sampled, pos_interval).clamp(lo, hi)
    qd = torch.empty_like(default).uniform_(-vel_interval, vel_interval)
    robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)
    robot.set_joint_position_target(q, env_ids=env_ids)


__all__ = ["KeypointGoalCommand", "KeypointGoalCommandCfg", "hand_joint_vel_l1",
           "keypoint_progress_m", "keypoint_residual_m", "keypoint_success",
           "object_away_from_palm", "reset_hand_like_ours", "reset_object_like_ours"]
