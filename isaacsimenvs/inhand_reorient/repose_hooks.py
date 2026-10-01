"""The ``isaaclab_repose`` profile's env hooks (Kit side).

Thin glue between ``InHandReorientEnv`` and the Kit-free arithmetic in
``repose_profile.py``. Hook order and RNG draw order follow NVIDIA's
``InHandManipulationEnv``:

- ``_pre_physics_step``: store the actions (ghost joints zeroed).
- ``_apply_action`` (every physics substep): limit-scaled targets, moving
  average, clamp.
- ``_get_dones``: refresh the palm-frame geometry; fall (distance to the
  in-hand target >= fall_dist) and time-out (one step before the episode
  limit).
- ``_get_rewards``: NVIDIA's reward; a goal is reached when the rotation
  distance is within ``success_tolerance`` (no hold time); reached goals are
  resampled at once (episode length is not reset).
- ``_reset_idx``: new goal, then the object (spawn point + uniform noise,
  random orientation), then the hand (NVIDIA's joint noise formula).

The legacy profile's buffers (``_successes``, ``_rot_error``,
``_is_success``, ``_termination_reasons``, ``_reward_terms``, ...) are kept
filled with this profile's values, so ``design_scoring``, ``nan_guard`` and
the logging in ``reset_utils.log_step_metrics`` work unchanged.
"""

from __future__ import annotations

import torch

from . import design_scoring
from . import repose_profile as rp
from .obs_utils import _fingertip_valid_mask, _joint_valid_mask, update_palm_frame_geometry
from .repose_profile import sample_uniform
from .reset_utils import _object_spawn_offset, _population_default_joint_pos

__all__ = [
    "allocate_repose_buffers", "pre_physics_step", "apply_action", "refresh_geometry",
    "compute_terminations", "compute_rewards_and_goals", "reset_env_state", "build_observations",
]

# Logged as goal_mode_code; the legacy modes are axis=0, delta=1, full=2.
GOAL_MODE_CODE_ISAACLAB = 3.0


def _limits(env):
    limits = env.robot.data.soft_joint_pos_limits
    return limits[..., 0], limits[..., 1]


def allocate_repose_buffers(env) -> None:
    n, device = env.num_envs, env.device
    j = env.hand_spec.num_hand_joints
    k = len(getattr(env, "fingertip_body_idx", []) or [])
    env._repose_actions = torch.zeros(n, j, device=device)
    env._repose_prev_targets = torch.zeros(n, j, device=device)
    env._repose_cur_targets = torch.zeros(n, j, device=device)
    env._repose_consecutive_successes = torch.zeros(1, device=device)
    env._repose_goal_hit = torch.zeros(n, dtype=torch.bool, device=device)
    env._in_hand_pos_palm = torch.zeros(n, 3, device=device)
    env._fingertip_quat_palm = torch.zeros(n, 4 * k, device=device)
    env._fingertip_vel_palm = torch.zeros(n, 6 * k, device=device)
    env._fingertip_pos_palm = torch.zeros(n, 3 * k, device=device)
    # No tolerance curriculum in this profile: a fixed tolerance, published
    # under the legacy name so logging and design_scoring read it as before.
    env._current_success_tolerance = float(env.cfg.repose.success_tolerance)


def pre_physics_step(env, actions: torch.Tensor) -> None:
    actions = actions.clone()
    joint_mask = _joint_valid_mask(env)
    if joint_mask is not None:
        actions = torch.where(joint_mask, actions, torch.zeros_like(actions))
    env._prev_actions_this_step = env._prev_actions
    env._prev_actions = actions
    env._repose_actions = actions


def apply_action(env) -> None:
    lower, upper = _limits(env)
    targets = rp.joint_targets(
        env._repose_actions, env._repose_prev_targets, lower, upper, env.cfg.repose.act_moving_average)
    env._repose_cur_targets = targets
    env._repose_prev_targets = targets
    env.robot.set_joint_position_target(targets)


def _refresh_goal_palm(env) -> None:
    palm_quat_w = env.robot.data.body_quat_w[:, env.palm_body_idx]
    env._goal_quat_palm = rp.quat_mul(rp.quat_conjugate(palm_quat_w), env._goal_quat_w)


def refresh_geometry(env) -> None:
    """Palm-frame object/goal/fingertip state and the rotation distance
    (NVIDIA's formula). Idempotent; safe on a partial reset."""
    update_palm_frame_geometry(env)
    _refresh_goal_palm(env)
    env._rot_error = rp.rotation_distance(env._obj_quat_palm, env._goal_quat_palm)
    tip_idx = getattr(env, "fingertip_body_idx", None)
    if tip_idx:
        data = env.robot.data
        pos, quat, vel = rp.palm_frame_fingertips(
            data.body_pos_w[:, env.palm_body_idx], data.body_quat_w[:, env.palm_body_idx],
            data.body_pos_w[:, tip_idx], data.body_quat_w[:, tip_idx], data.body_vel_w[:, tip_idx])
        env._fingertip_pos_palm = pos
        env._fingertip_quat_palm = quat
        env._fingertip_vel_palm = vel


def compute_terminations(env) -> tuple[torch.Tensor, torch.Tensor]:
    r = env.cfg.repose
    fell, time_out = rp.fall_and_timeout(
        env._obj_pos_palm, env._in_hand_pos_palm, env.episode_length_buf, env.max_episode_length, r.fall_dist)
    if r.max_consecutive_success > 0:
        # NVIDIA: within tolerance resets the episode clock; reaching the
        # success cap ends the episode as a time-out.
        env.episode_length_buf = torch.where(
            env._rot_error <= r.success_tolerance, torch.zeros_like(env.episode_length_buf),
            env.episode_length_buf)
        max_reached = env._successes >= r.max_consecutive_success
        time_out = (env.episode_length_buf >= env.max_episode_length - 1) | max_reached
    else:
        max_reached = torch.zeros_like(fell)
    env._termination_reasons = {"drop": fell, "max_successes": max_reached, "timeout": time_out}
    return fell, time_out


def compute_rewards_and_goals(env) -> torch.Tensor:
    r = env.cfg.repose
    reward, goal_hit, successes, cons, terms = rp.compute_rewards(
        env.reset_buf, torch.zeros_like(env._repose_goal_hit), env._successes.float(),
        env._repose_consecutive_successes, env._obj_pos_palm, env._obj_quat_palm,
        env._in_hand_pos_palm, env._goal_quat_palm, env._repose_actions,
        dist_reward_scale=r.dist_reward_scale, rot_reward_scale=r.rot_reward_scale, rot_eps=r.rot_eps,
        action_penalty_scale=r.action_penalty_scale, success_tolerance=r.success_tolerance,
        reach_goal_bonus=r.reach_goal_bonus, fall_dist=r.fall_dist, fall_penalty=r.fall_penalty,
        av_factor=r.av_factor,
    )
    env._successes = successes.round().long()
    env._repose_consecutive_successes = cons
    env._repose_goal_hit = goal_hit
    env._is_success = goal_hit
    within = env._rot_error <= r.success_tolerance
    env._consec_success_steps = torch.where(
        within, env._consec_success_steps + 1, torch.zeros_like(env._consec_success_steps))
    env._reward_terms = terms

    goal_ids = goal_hit.nonzero(as_tuple=False).squeeze(-1)
    if goal_ids.numel() > 0:
        _reset_target_pose(env, goal_ids)
    return reward


def _reset_target_pose(env, env_ids: torch.Tensor) -> None:
    """NVIDIA's ``_reset_target_pose``: a new goal from ``randomize_rotation``,
    in world axes."""
    rand = sample_uniform(-1.0, 1.0, (env_ids.numel(), 2), device=env.device)
    env._goal_quat_w[env_ids] = rp.randomize_rotation(rand[:, 0], rand[:, 1])
    _refresh_goal_palm(env)
    env._consec_success_steps[env_ids] = 0
    goal_state = env.goal_viz.data.root_state_w[env_ids].clone()
    goal_state[:, 3:7] = env._goal_quat_w[env_ids]
    goal_state[:, 7:] = 0.0
    env.goal_viz.write_root_state_to_sim(goal_state, env_ids=env_ids)


def reset_env_state(env, env_ids: torch.Tensor) -> None:
    """NVIDIA's ``_reset_idx`` (after DirectRLEnv's own), same draw order:
    goal (n, 2), object position noise (n, 3), object rotation (n, 2), joint
    position noise (n, J), joint velocity noise (n, J)."""
    r = env.cfg.repose
    n, device = env_ids.numel(), env.device
    palm_pos_w = env.robot.data.body_pos_w[env_ids, env.palm_body_idx]
    palm_quat_w = env.robot.data.body_quat_w[env_ids, env.palm_body_idx]
    offset = _object_spawn_offset(env, env_ids)
    spawn_w = palm_pos_w + rp.quat_apply(palm_quat_w, offset)

    # Goal first (NVIDIA calls _reset_target_pose before touching the object).
    rand = sample_uniform(-1.0, 1.0, (n, 2), device=device)
    env._goal_quat_w[env_ids] = rp.randomize_rotation(rand[:, 0], rand[:, 1])

    pos_noise = sample_uniform(-1.0, 1.0, (n, 3), device=device)
    obj_pos_w = spawn_w + r.reset_position_noise * pos_noise
    rot_noise = sample_uniform(-1.0, 1.0, (n, 2), device=device)
    obj_quat_w = rp.randomize_rotation(rot_noise[:, 0], rot_noise[:, 1])
    obj_state = torch.cat([obj_pos_w, obj_quat_w, torch.zeros(n, 6, device=device)], dim=-1)
    env.object.write_root_state_to_sim(obj_state, env_ids=env_ids)

    env._spawn_obj_pos_palm[env_ids] = offset
    in_hand_w = spawn_w + torch.as_tensor(r.in_hand_pos_offset, device=device, dtype=torch.float32)
    env._in_hand_pos_palm[env_ids] = rp.quat_apply_inverse(palm_quat_w, in_hand_w - palm_pos_w)

    default_pos = _population_default_joint_pos(env, env_ids)
    lower, upper = _limits(env)
    j = default_pos.shape[-1]
    dof_pos_noise = sample_uniform(-1.0, 1.0, (n, j), device=device)
    dof_pos = rp.reset_joint_positions(
        default_pos, lower[env_ids], upper[env_ids], dof_pos_noise, r.reset_dof_pos_noise)
    dof_vel_noise = sample_uniform(-1.0, 1.0, (n, j), device=device)
    dof_vel = env.robot.data.default_joint_vel[env_ids] + r.reset_dof_vel_noise * dof_vel_noise
    env._repose_prev_targets[env_ids] = dof_pos
    env._repose_cur_targets[env_ids] = dof_pos
    env.robot.set_joint_position_target(dof_pos, env_ids=env_ids)
    env.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

    goal_state = torch.cat(
        [spawn_w, env._goal_quat_w[env_ids], torch.zeros(n, 6, device=device)], dim=-1)
    env.goal_viz.write_root_state_to_sim(goal_state, env_ids=env_ids)

    env._consec_success_steps[env_ids] = 0
    env._prev_episode_successes[env_ids] = env._successes[env_ids]
    env._successes[env_ids] = 0
    env._prev_rot_error[env_ids] = 0.0

    refresh_geometry(env)
    design_scoring.reset_scoring_state(env, env_ids)


def build_observations(env) -> dict[str, torch.Tensor]:
    lower, upper = _limits(env)
    _refresh_goal_palm(env)
    obs = rp.build_observation(
        joint_pos=env.robot.data.joint_pos, joint_vel=env.robot.data.joint_vel, lower=lower, upper=upper,
        object_pos=env._obj_pos_palm, object_quat=env._obj_quat_palm,
        object_lin_vel=env._obj_lin_vel_palm, object_ang_vel=env._obj_ang_vel_palm,
        in_hand_pos=env._in_hand_pos_palm, goal_quat=env._goal_quat_palm,
        fingertip_pos=env._fingertip_pos_palm, fingertip_quat=env._fingertip_quat_palm,
        fingertip_vel=env._fingertip_vel_palm, actions=env._repose_actions,
        vel_obs_scale=env.cfg.repose.vel_obs_scale,
        joint_mask=_joint_valid_mask(env), fingertip_mask=_fingertip_valid_mask(env),
        fields=tuple(env.cfg.obs.obs_list),
    )
    return {"policy": obs, "critic": obs}
