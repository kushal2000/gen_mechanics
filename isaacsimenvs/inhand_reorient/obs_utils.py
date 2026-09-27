"""Observation assembly and the per-step geometry cache (object pose in the
palm frame, rotation error to goal) every reward/termination hook reads."""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_error_magnitude, quat_mul, quat_inv, subtract_frame_transforms

__all__ = [
    "OBS_FIELD_WIDTHS", "derive_spaces", "compute_intermediate_values",
    "pre_physics_step", "build_observations",
]

# Width (last-dim size) of each observation field, independent of the hand.
OBS_FIELD_WIDTHS = {
    "object_pos_palm": 3,
    "object_quat_palm": 4,
    "goal_quat_palm": 4,
    "object_quat_rel_goal": 4,
    "prev_actions": None,   # = num hand joints
    "joint_pos": None,
    "joint_vel": None,
}


def _field_width(name: str, num_joints: int) -> int:
    w = OBS_FIELD_WIDTHS[name]
    return num_joints if w is None else w


def derive_spaces(cfg, spec) -> None:
    """Set action/observation/state space sizes from the hand spec + obs cfg."""
    j = spec.num_hand_joints
    cfg.action_space = j
    cfg.observation_space = sum(_field_width(f, j) for f in cfg.obs.obs_list)
    cfg.state_space = sum(_field_width(f, j) for f in cfg.obs.state_list)


def pre_physics_step(env, actions: torch.Tensor) -> None:
    """Joint-position targets with moving-average smoothing, PoseReach-style."""
    actions = actions.clamp(-1.0, 1.0)
    env._prev_actions_this_step = env._prev_actions.clone()
    env._prev_actions = actions

    spec = env.hand_spec
    lower = env.robot.data.soft_joint_pos_limits[:, :, 0]
    upper = env.robot.data.soft_joint_pos_limits[:, :, 1]
    span = 0.5 * (upper - lower)
    mid = 0.5 * (upper + lower)
    scaled = mid + actions * span * env.cfg.action.dof_speed_scale
    alpha = env.cfg.action.hand_moving_average
    if not hasattr(env, "_cur_targets"):
        env._cur_targets = env.robot.data.joint_pos.clone()
    env._cur_targets = alpha * scaled + (1.0 - alpha) * env._cur_targets
    env._cur_targets = torch.clamp(env._cur_targets, lower, upper)


def compute_intermediate_values(env) -> None:
    """Object pose in the palm frame and the rotation error to goal. Called
    from ``_get_dones`` before terminations/rewards, so both read fresh
    values (mirrors ``pose_reaching_6d``'s hook order)."""
    palm_pos_w = env.robot.data.body_pos_w[:, env.palm_body_idx]
    palm_quat_w = env.robot.data.body_quat_w[:, env.palm_body_idx]
    obj_pos_w = env.object.data.root_pos_w
    obj_quat_w = env.object.data.root_quat_w

    obj_pos_palm, obj_quat_palm = subtract_frame_transforms(
        palm_pos_w, palm_quat_w, obj_pos_w, obj_quat_w)
    _goal_pos_palm, goal_quat_palm = subtract_frame_transforms(
        palm_pos_w, palm_quat_w, obj_pos_w, env._goal_quat_w)

    env._obj_pos_palm = obj_pos_palm
    env._obj_quat_palm = obj_quat_palm
    env._goal_quat_palm = goal_quat_palm
    env._obj_quat_rel_goal = quat_mul(quat_inv(env._goal_quat_w), obj_quat_w)

    env._prev_rot_error = env._rot_error
    env._rot_error = quat_error_magnitude(obj_quat_w, env._goal_quat_w)

    tol = env._current_success_tolerance
    hit = env._rot_error <= tol
    env._consec_success_steps = torch.where(
        hit, env._consec_success_steps + 1, torch.zeros_like(env._consec_success_steps))
    env._is_success = env._consec_success_steps >= env.cfg.termination.success_steps


def build_observations(env) -> dict[str, torch.Tensor]:
    values = {
        "joint_pos": env.robot.data.joint_pos,
        "joint_vel": env.robot.data.joint_vel,
        "prev_actions": env._prev_actions,
        "object_pos_palm": env._obj_pos_palm,
        "object_quat_palm": env._obj_quat_palm,
        "goal_quat_palm": env._goal_quat_palm,
        "object_quat_rel_goal": env._obj_quat_rel_goal,
    }
    clamp = env.cfg.obs.clamp_abs_observations
    obs = torch.cat([values[f] for f in env.cfg.obs.obs_list], dim=-1).clamp(-clamp, clamp)
    state = torch.cat([values[f] for f in env.cfg.obs.state_list], dim=-1).clamp(-clamp, clamp)
    return {"policy": obs, "critic": state}
