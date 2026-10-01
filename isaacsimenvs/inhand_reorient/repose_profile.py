"""The ``isaaclab_repose`` task profile: NVIDIA's in-hand cube reorientation
spec, ported to our variable hands.

Source of the spec: Isaac Lab 2.3.2's ``Isaac-Repose-Cube-Allegro-Direct-v0``
(``isaaclab_tasks/direct/inhand_manipulation/inhand_manipulation_env.py`` for
the logic, ``isaaclab_tasks/direct/allegro_hand/allegro_hand_env_cfg.py`` for
the numbers), itself the IsaacGymEnvs AllegroHand benchmark. Every function
below mirrors one of theirs; ``tests/test_repose_profile.py`` runs NVIDIA's
own function bodies, read from the installed package, on the same random
tensors and compares the outputs.

What is ours rather than theirs (the only deliberate departures):

- Frame. NVIDIA observes world (env-origin) quantities for one fixed hand
  pose. Our hands differ in pose, so every pose, velocity and quaternion is
  expressed in the palm (articulation root) frame. For a hand whose palm
  sits at the env origin with identity rotation the two coincide exactly;
  for any fixed hand they differ by a constant rigid transform. Scalars
  (rotation distance, distance to the in-hand target) are frame-invariant,
  so rewards and terminations are unchanged.
- Padding. For a grammar population, ghost joints and absent fingertips are
  zeroed in the observation and their actions are zeroed before control
  (``obs_utils``' existing masks), as in the legacy profile.
- Spawn point. NVIDIA places the cube 6 cm above its Allegro's root; our
  hands use their palm-up calibration's spawn point
  (``hand_calibration.json``). The in-hand target is that point plus
  NVIDIA's offset of -0.04 m along world z.

Kit-free (torch only), so the arithmetic is testable under plain pytest.
The quaternion helpers use the (w, x, y, z) convention and the same formulas
as ``isaaclab.utils.math``.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "PROFILE_ISAACLAB_REPOSE", "PROFILE_LEGACY", "TASK_PROFILES", "REPOSE_OBS_FIELDS",
    "repose_field_width", "repose_obs_dim", "is_repose", "apply_profile_to_cfg",
    "quat_mul", "quat_conjugate", "quat_from_angle_axis", "quat_apply", "quat_apply_inverse",
    "scale", "unscale", "saturate", "sample_uniform", "randomize_rotation", "rotation_distance",
    "compute_rewards", "joint_targets", "reset_joint_positions", "fall_and_timeout",
    "palm_frame_fingertips", "build_observation",
]

PROFILE_ISAACLAB_REPOSE = "isaaclab_repose"
PROFILE_LEGACY = "legacy"
TASK_PROFILES = (PROFILE_ISAACLAB_REPOSE, PROFILE_LEGACY)

# NVIDIA's "full" observation (compute_full_observations), field for field,
# in their order. For Allegro (16 joints, 4 fingertips) this is 124 wide.
REPOSE_OBS_FIELDS = (
    "joint_pos_unscaled",          # unscale(dof_pos, lower, upper)
    "joint_vel_scaled",            # vel_obs_scale * dof_vel
    "object_pos_palm",             # object_pos
    "object_quat_palm",            # object_rot
    "object_lin_vel_palm",         # object_linvel (unscaled, as NVIDIA)
    "object_ang_vel_scaled_palm",  # vel_obs_scale * object_angvel
    "in_hand_pos_palm",            # in_hand_pos (the position target)
    "goal_quat_palm",              # goal_rot
    "object_quat_rel_goal_isaaclab",  # quat_mul(object_rot, quat_conjugate(goal_rot))
    "fingertip_pos_palm",          # fingertip_pos, 3 per tip
    "fingertip_quat_palm",         # fingertip_rot, 4 per tip
    "fingertip_vel_palm",          # fingertip_velocities (linear, angular), 6 per tip
    "actions",                     # this step's actions
)
_FIXED_WIDTHS = {
    "object_pos_palm": 3, "object_quat_palm": 4, "object_lin_vel_palm": 3,
    "object_ang_vel_scaled_palm": 3, "in_hand_pos_palm": 3, "goal_quat_palm": 4,
    "object_quat_rel_goal_isaaclab": 4,
}
_PER_JOINT = ("joint_pos_unscaled", "joint_vel_scaled", "actions")
_PER_TIP = {"fingertip_pos_palm": 3, "fingertip_quat_palm": 4, "fingertip_vel_palm": 6}


def repose_field_width(name: str, num_joints: int, num_fingertips: int) -> int:
    if name in _PER_JOINT:
        return num_joints
    if name in _PER_TIP:
        return _PER_TIP[name] * num_fingertips
    return _FIXED_WIDTHS[name]


def repose_obs_dim(num_joints: int, num_fingertips: int) -> int:
    return sum(repose_field_width(f, num_joints, num_fingertips) for f in REPOSE_OBS_FIELDS)


def is_repose(cfg) -> bool:
    profile = getattr(cfg, "task_profile", PROFILE_LEGACY)
    if profile not in TASK_PROFILES:
        raise ValueError(f"task_profile={profile!r}; expected one of {TASK_PROFILES}")
    return profile == PROFILE_ISAACLAB_REPOSE


def apply_profile_to_cfg(cfg) -> None:
    """Write the profile's env-level numbers onto the fields DirectRLEnv and
    the scene read (decimation, physics dt, episode length, the default
    physics material, the observation lists). Everything else the profile
    reads from ``cfg.repose`` directly. A no-op for the legacy profile, so
    that profile's cfg is untouched. Called once, before DirectRLEnv.__init__.
    """
    if not is_repose(cfg):
        return
    r = cfg.repose
    cfg.decimation = int(r.decimation)
    cfg.episode_length_s = float(r.episode_length_s)
    cfg.sim.dt = float(r.sim_dt)
    cfg.sim.render_interval = int(r.decimation)
    cfg.sim.physics_material.static_friction = float(r.static_friction)
    cfg.sim.physics_material.dynamic_friction = float(r.dynamic_friction)
    cfg.sim.physics_material.restitution = float(r.restitution)
    cfg.sim.physx.bounce_threshold_velocity = float(r.bounce_threshold_velocity)
    cfg.obs.obs_list = tuple(REPOSE_OBS_FIELDS)
    cfg.obs.state_list = tuple(REPOSE_OBS_FIELDS)


# --------------------------------------------------------------------------
# Quaternions, (w, x, y, z): the formulas of isaaclab.utils.math.
# --------------------------------------------------------------------------


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([w, x, y, z], dim=-1).view(shape)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1).view(shape)


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    theta = (angle / 2).unsqueeze(-1)
    xyz = torch.nn.functional.normalize(axis, dim=-1) * theta.sin()
    w = theta.cos()
    return torch.cat([w, xyz], dim=-1)


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec - quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


# --------------------------------------------------------------------------
# NVIDIA's module-level helpers (inhand_manipulation_env.py), unchanged.
# --------------------------------------------------------------------------


def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [lower, upper], linear, centred on the limits' midpoint."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return (2.0 * x - upper - lower) / (upper - lower)


def saturate(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(x, upper), lower)


def sample_uniform(lower: float, upper: float, size, device) -> torch.Tensor:
    """isaaclab.utils.math.sample_uniform for scalar bounds (the same draw)."""
    return torch.rand(*size, device=device) * (upper - lower) + lower


def randomize_rotation(rand0: torch.Tensor, rand1: torch.Tensor) -> torch.Tensor:
    """NVIDIA's goal (and object reset) orientation: a rotation of
    ``rand0 * pi`` about world x composed with ``rand1 * pi`` about world y,
    ``rand0, rand1 ~ U(-1, 1)``. A two-parameter family covering every
    "which face is up" outcome, not the uniform (Haar) distribution on SO(3);
    kept exactly, since it defines the benchmark's goal distribution."""
    n = rand0.shape[0]
    x_unit = torch.tensor([1.0, 0.0, 0.0], device=rand0.device).repeat((n, 1))
    y_unit = torch.tensor([0.0, 1.0, 0.0], device=rand0.device).repeat((n, 1))
    return quat_mul(quat_from_angle_axis(rand0 * math.pi, x_unit), quat_from_angle_axis(rand1 * math.pi, y_unit))


def rotation_distance(object_rot: torch.Tensor, target_rot: torch.Tensor) -> torch.Tensor:
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))


def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    actions: torch.Tensor,
    *,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,
):
    """NVIDIA's ``compute_rewards``, line for line, plus a per-term
    breakdown for logging. Returns ``(reward, goal_resets, successes,
    consecutive_successes, terms)``; the first four are exactly theirs.
    ``max_episode_length`` (an unused argument of theirs) is dropped."""
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    rot_dist = rotation_distance(object_rot, target_rot)

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions**2, dim=-1)

    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    fell = goal_dist >= fall_dist
    reward = torch.where(fell, reward + fall_penalty, reward)

    resets = torch.where(fell, torch.ones_like(reset_buf), reset_buf)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    terms = {
        "rotation_rew": rot_rew,
        "distance_rew": dist_rew,
        "goal_bonus_rew": (goal_resets == 1).float() * reach_goal_bonus,
        "action_penalty": action_penalty * action_penalty_scale,
        "drop_penalty": fell.float() * fall_penalty,
        "total_reward": reward,
    }
    return reward, goal_resets, successes, cons_successes, terms


def joint_targets(
    actions: torch.Tensor, prev_targets: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor,
    act_moving_average: float,
) -> torch.Tensor:
    """NVIDIA's ``_apply_action`` target: scale to the joint limits, blend
    with the previous target, clamp to the limits. Called once per physics
    substep, as theirs is (with act_moving_average = 1.0 the blend is the
    identity, so substep and policy-step application coincide)."""
    targets = scale(actions, lower, upper)
    targets = act_moving_average * targets + (1.0 - act_moving_average) * prev_targets
    return saturate(targets, lower, upper)


def reset_joint_positions(
    default_pos: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, noise: torch.Tensor,
    reset_dof_pos_noise: float,
) -> torch.Tensor:
    """NVIDIA's hand reset (``_reset_idx``), with ``noise ~ U(-1, 1)``:
    ``default + reset_dof_pos_noise * (delta_min + (delta_max - delta_min)
    * 0.5 * noise)``, ``delta_* = limit - default``. Not clamped to the
    limits, as theirs is not; the sampled offset is skewed towards the lower
    limit (its range is ``[1.5 l - 0.5 u, (l + u) / 2] - default``)."""
    delta_max = upper - default_pos
    delta_min = lower - default_pos
    rand_delta = delta_min + (delta_max - delta_min) * 0.5 * noise
    return default_pos + reset_dof_pos_noise * rand_delta


def fall_and_timeout(
    object_pos: torch.Tensor, in_hand_pos: torch.Tensor, episode_length_buf: torch.Tensor,
    max_episode_length: int, fall_dist: float,
):
    """NVIDIA's ``_get_dones`` for ``max_consecutive_success == 0``:
    terminate when the object is ``fall_dist`` or more from the in-hand
    target; time out one step before ``max_episode_length``."""
    goal_dist = torch.norm(object_pos - in_hand_pos, p=2, dim=-1)
    out_of_reach = goal_dist >= fall_dist
    time_out = episode_length_buf >= max_episode_length - 1
    return out_of_reach, time_out


def palm_frame_fingertips(
    palm_pos_w: torch.Tensor, palm_quat_w: torch.Tensor,
    tip_pos_w: torch.Tensor, tip_quat_w: torch.Tensor, tip_vel_w: torch.Tensor,
):
    """Fingertip pose and (linear, angular) velocity in the palm frame.
    ``tip_*_w``: ``(n, k, 3|4|6)``; returns ``(n, 3k)``, ``(n, 4k)``, ``(n, 6k)``."""
    n, k = tip_pos_w.shape[:2]
    q = palm_quat_w.unsqueeze(1).expand(n, k, 4).reshape(n * k, 4)
    pos = quat_apply_inverse(q, (tip_pos_w - palm_pos_w.unsqueeze(1)).reshape(n * k, 3))
    quat = quat_mul(quat_conjugate(q), tip_quat_w.reshape(n * k, 4))
    lin = quat_apply_inverse(q, tip_vel_w[..., 0:3].reshape(n * k, 3))
    ang = quat_apply_inverse(q, tip_vel_w[..., 3:6].reshape(n * k, 3))
    vel = torch.cat([lin.reshape(n, k, 3), ang.reshape(n, k, 3)], dim=-1)
    return pos.reshape(n, 3 * k), quat.reshape(n, 4 * k), vel.reshape(n, 6 * k)


def build_observation(
    *,
    joint_pos: torch.Tensor, joint_vel: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor,
    object_pos: torch.Tensor, object_quat: torch.Tensor, object_lin_vel: torch.Tensor,
    object_ang_vel: torch.Tensor, in_hand_pos: torch.Tensor, goal_quat: torch.Tensor,
    fingertip_pos: torch.Tensor, fingertip_quat: torch.Tensor, fingertip_vel: torch.Tensor,
    actions: torch.Tensor, vel_obs_scale: float,
    joint_mask: torch.Tensor | None = None, fingertip_mask: torch.Tensor | None = None,
    fields=REPOSE_OBS_FIELDS,
) -> torch.Tensor:
    """NVIDIA's ``compute_full_observations`` over palm-frame inputs.

    ``fingertip_*`` are flat ``(n, 3k | 4k | 6k)``. ``joint_mask``
    (``(n, J)`` bool) zeroes ghost joints; ``torch.where`` rather than a
    product, because a ghost joint's limits can make ``unscale`` blow up.
    ``fingertip_mask`` (``(n, k)`` bool) zeroes absent fingertips.
    """
    joint_pos_unscaled = unscale(joint_pos, lower, upper)
    joint_vel_scaled = vel_obs_scale * joint_vel
    if joint_mask is not None:
        joint_pos_unscaled = torch.where(joint_mask, joint_pos_unscaled, torch.zeros_like(joint_pos_unscaled))
        joint_vel_scaled = torch.where(joint_mask, joint_vel_scaled, torch.zeros_like(joint_vel_scaled))
        actions = torch.where(joint_mask, actions, torch.zeros_like(actions))
    if fingertip_mask is not None:
        n, k = fingertip_mask.shape
        m = fingertip_mask.unsqueeze(-1).to(fingertip_pos.dtype)
        fingertip_pos = (fingertip_pos.reshape(n, k, 3) * m).reshape(n, 3 * k)
        fingertip_quat = (fingertip_quat.reshape(n, k, 4) * m).reshape(n, 4 * k)
        fingertip_vel = (fingertip_vel.reshape(n, k, 6) * m).reshape(n, 6 * k)
    values = {
        "joint_pos_unscaled": joint_pos_unscaled,
        "joint_vel_scaled": joint_vel_scaled,
        "object_pos_palm": object_pos,
        "object_quat_palm": object_quat,
        "object_lin_vel_palm": object_lin_vel,
        "object_ang_vel_scaled_palm": vel_obs_scale * object_ang_vel,
        "in_hand_pos_palm": in_hand_pos,
        "goal_quat_palm": goal_quat,
        "object_quat_rel_goal_isaaclab": quat_mul(object_quat, quat_conjugate(goal_quat)),
        "fingertip_pos_palm": fingertip_pos,
        "fingertip_quat_palm": fingertip_quat,
        "fingertip_vel_palm": fingertip_vel,
        "actions": actions,
    }
    return torch.cat([values[f] for f in fields], dim=-1)
