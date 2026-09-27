"""Reset-time state: object spawn, goal sampling, and the buffers the
curriculum and reward need across steps.

Free functions over a duck-typed ``env`` (no Isaac import at module scope
beyond what every caller already has booted), mirroring
``pose_reaching_6d.reset_utils``.
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply, random_orientation

from isaacsimenvs.pose_reaching_6d.reward_utils.curriculum import initial_success_tolerance

__all__ = ["allocate_state_buffers", "reset_env_state", "reset_goal_trackers"]


def allocate_state_buffers(env) -> None:
    """Run once, after the scene is up (mirrors ``PoseReachEnv.__init__``)."""
    n, device = env.num_envs, env.device
    j = env.hand_spec.num_hand_joints

    env._prev_actions = torch.zeros(n, j, device=device)
    env._goal_quat_w = torch.zeros(n, 4, device=device)
    env._goal_quat_w[:, 0] = 1.0
    env._spawn_obj_pos_palm = torch.zeros(n, 3, device=device)
    env._consec_success_steps = torch.zeros(n, device=device, dtype=torch.long)
    env._successes = torch.zeros(n, device=device, dtype=torch.long)
    env._prev_episode_successes = torch.zeros(n, device=device, dtype=torch.long)
    env._prev_rot_error = torch.zeros(n, device=device)
    env._rot_error = torch.zeros(n, device=device)
    env._is_success = torch.zeros(n, device=device, dtype=torch.bool)
    env._termination_reasons: dict[str, torch.Tensor] = {}

    # Curriculum state rlgames_utils.py ferries through the rl_games
    # checkpoint (get_env_state/set_env_state); names match
    # pose_reaching_6d.reward_utils.curriculum exactly on purpose.
    env._frame_counter = 0
    env._last_curriculum_update = 0
    env._current_success_tolerance = initial_success_tolerance(env)


def _sample_goal(env, env_ids: torch.Tensor) -> torch.Tensor:
    n = env_ids.numel()
    if env.cfg.reset.goal_sampling_type == "delta":
        from isaaclab.utils.math import quat_from_angle_axis, quat_mul
        import math

        axis = torch.nn.functional.normalize(torch.randn(n, 3, device=env.device), dim=-1)
        angle = (torch.rand(n, device=env.device) * 2.0 - 1.0) * math.radians(
            env.cfg.reset.delta_rotation_degrees)
        dq = quat_from_angle_axis(angle, axis)
        return quat_mul(dq, env._goal_quat_w[env_ids])
    return random_orientation(n, device=env.device)


def reset_goal_trackers(env, env_ids: torch.Tensor) -> None:
    """A goal was hit mid-episode: resample it, without touching anything
    else (episode_length_buf reset is the caller's job, as in
    ``pose_reaching_6d.reward_utils.termination``)."""
    env._goal_quat_w[env_ids] = _sample_goal(env, env_ids)
    env._consec_success_steps[env_ids] = 0


def reset_env_state(env, env_ids: torch.Tensor) -> None:
    """Full reset: hand joints, object spawn, goal, and per-episode buffers."""
    n = env_ids.numel()
    spec = env.hand_spec

    # Hand joints to their (0) default pose plus uniform noise, clamped inside
    # each joint's own limits.
    default_pos = env.robot.data.default_joint_pos[env_ids]
    lower = env.robot.data.soft_joint_pos_limits[env_ids, :, 0]
    upper = env.robot.data.soft_joint_pos_limits[env_ids, :, 1]
    noise = (torch.rand_like(default_pos) * 2.0 - 1.0) * env.cfg.reset.joint_reset_noise
    joint_pos = torch.clamp(default_pos + noise, lower, upper)
    joint_vel = torch.zeros_like(joint_pos)
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # Object: spawned just above the palm centre, random orientation, small
    # position jitter -- all in the palm's own (world) frame at reset time.
    palm_pos_w = env.robot.data.body_pos_w[env_ids, env.palm_body_idx]
    palm_quat_w = env.robot.data.body_quat_w[env_ids, env.palm_body_idx]
    offset = torch.as_tensor(env.cfg.reset.object_spawn_offset, device=env.device, dtype=torch.float32)
    jitter = (torch.rand(n, 3, device=env.device) * 2.0 - 1.0) * env.cfg.reset.object_position_noise
    local_pos = offset.expand(n, 3) + jitter
    env._spawn_obj_pos_palm[env_ids] = local_pos
    obj_pos_w = palm_pos_w + quat_apply(palm_quat_w, local_pos)
    obj_quat_w = random_orientation(n, device=env.device)

    obj_state = torch.cat([obj_pos_w, obj_quat_w, torch.zeros(n, 6, device=env.device)], dim=-1)
    env.object.write_root_state_to_sim(obj_state, env_ids=env_ids)

    # Goal.
    env._goal_quat_w[env_ids] = random_orientation(n, device=env.device)
    goal_pos_w = palm_pos_w + quat_apply(palm_quat_w, offset.expand(n, 3))
    goal_state = torch.cat(
        [goal_pos_w, env._goal_quat_w[env_ids], torch.zeros(n, 6, device=env.device)], dim=-1)
    env.goal_viz.write_root_state_to_sim(goal_state, env_ids=env_ids)

    # Per-episode buffers.
    env._prev_actions[env_ids] = 0.0
    env._consec_success_steps[env_ids] = 0
    env._prev_episode_successes[env_ids] = env._successes[env_ids]
    env._successes[env_ids] = 0
    env._prev_rot_error[env_ids] = 0.0  # compute_intermediate_values fills the real value next step
