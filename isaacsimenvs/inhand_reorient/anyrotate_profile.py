"""The ``anyrotate`` task profile: multi-axis in-hand object rotation.

Source: M. Yang et al., "AnyRotate: Gravity-Invariant In-Hand Object Rotation
with Sim-to-Real Touch", CoRL 2024 (arXiv 2405.07391v3). No code is released;
every function below cites the paper equation or section it implements, and
the departures are listed in ``ANYROTATE_ADAPTATIONS`` (also in the
evolution README).

Task: rotate the object about a commanded unit axis k (in the palm frame).
The goal is an auxiliary target orientation, the current object orientation
rotated by ``goal_increment`` about k; when it is reached a new one is made
the same way (Sec. 3.1, "Auxiliary Goal").

Kit-free (torch only); quaternions are (w, x, y, z) and reuse the helpers of
``repose_profile`` (the formulas of ``isaaclab.utils.math``).
"""

from __future__ import annotations

import math

import torch

from .repose_profile import quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul

__all__ = [
    "PROFILE_ANYROTATE", "ANYROTATE_OBS_FIELDS", "ANYROTATE_PRIV_FIELDS", "anyrotate_field_width",
    "keypoint_offsets", "object_keypoints", "keypoint_distance", "keypoint_reward",
    "quat_to_rotvec", "rotation_about_axis", "rotation_reward", "next_goal", "goal_reached",
    "contact_rewards", "angular_velocity_penalty", "pose_penalty", "work_penalty", "torque_penalty",
    "axis_deviation", "terminations", "reward_curriculum_lambda", "relative_joint_targets",
    "simulated_tactile", "sample_axes", "combine_rewards", "graded_rotation_fitness",
]

PROFILE_ANYROTATE = "anyrotate"

# Teacher observation o_t (Table 2), per-joint fields J wide, per-fingertip
# fields k wide (k = 4 for Allegro: 95 dims, the paper's "full touch" N=95).
ANYROTATE_OBS_FIELDS = (
    "joint_pos",            # q (16)
    "fingertip_pos_palm",   # f^p (12)
    "fingertip_quat_palm",  # f^o (16)
    "prev_actions",         # a_{t-1} (16)
    "target_joint_pos",     # q-bar (16)
    "tip_contact",          # c (4)
    "tip_contact_pose",     # P (8)
    "tip_contact_force",    # F (4)
    "rotation_axis",        # k (3)
)
# Privileged information x_t (Table 3), appended for the teacher.
ANYROTATE_PRIV_FIELDS = (
    "object_pos_palm", "object_quat_palm", "object_ang_vel_palm", "object_dims", "object_com",
    "object_mass", "gravity_palm", "goal_pos_palm", "goal_quat_palm",
)
_PER_JOINT = {"joint_pos": 1, "prev_actions": 1, "target_joint_pos": 1}
_PER_TIP = {"fingertip_pos_palm": 3, "fingertip_quat_palm": 4, "tip_contact": 1, "tip_contact_pose": 2,
            "tip_contact_force": 1}
_FIXED = {"rotation_axis": 3, "object_pos_palm": 3, "object_quat_palm": 4, "object_ang_vel_palm": 3,
          "object_dims": 2, "object_com": 3, "object_mass": 1, "gravity_palm": 3, "goal_pos_palm": 3,
          "goal_quat_palm": 4}


def anyrotate_field_width(name: str, num_joints: int, num_fingertips: int) -> int:
    if name in _PER_JOINT:
        return _PER_JOINT[name] * num_joints
    if name in _PER_TIP:
        return _PER_TIP[name] * num_fingertips
    return _FIXED[name]


# --------------------------------------------------------------------------
# Keypoints and the rotation objective (App. B.1, Eqs. 3-5)
# --------------------------------------------------------------------------


def keypoint_offsets(distance: float, device=None) -> torch.Tensor:
    """N = 6 keypoints ``distance`` from the object origin along each of its
    principal axes, both signs (App. B.1: 5 cm). ``(6, 3)``."""
    eye = torch.eye(3, device=device)
    return torch.cat([eye, -eye], dim=0) * float(distance)


def object_keypoints(pos: torch.Tensor, quat: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """``(n, 6, 3)`` keypoint positions of a body at ``pos``/``quat``."""
    n, m = pos.shape[0], offsets.shape[0]
    q = quat.unsqueeze(1).expand(n, m, 4).reshape(n * m, 4)
    o = offsets.unsqueeze(0).expand(n, m, 3).reshape(n * m, 3)
    return pos.unsqueeze(1) + quat_apply(q, o).reshape(n, m, 3)


def keypoint_distance(obj_pos, obj_quat, goal_pos, goal_quat, offsets) -> torch.Tensor:
    """kp_dist = (1/N) sum_i ||k_i^o - k_i^g|| (App. B.1)."""
    ko = object_keypoints(obj_pos, obj_quat, offsets)
    kg = object_keypoints(goal_pos, goal_quat, offsets)
    return (ko - kg).norm(dim=-1).mean(dim=-1)


def keypoint_reward(kp_dist: torch.Tensor, a: float, b: float, scale: float = 1.0) -> torch.Tensor:
    """Eq. 3: r_kp = d_kp / (e^{a x} + b + e^{-a x}), x = kp_dist. The
    numerator d_kp is not defined in the paper; ``scale`` (default 1)."""
    x = kp_dist * a
    return scale / (torch.exp(x) + b + torch.exp(-x))


def quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    """Axis-angle vector of a unit quaternion, angle in [0, pi]."""
    q = torch.where(q[..., :1] < 0, -q, q)
    xyz = q[..., 1:]
    sin_half = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, q[..., :1])
    scale = torch.where(sin_half > 1e-8, angle / sin_half.clamp(min=1e-8), torch.full_like(sin_half, 2.0))
    return xyz * scale


def rotation_about_axis(q_prev: torch.Tensor, q_curr: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Delta Theta . k (Eq. 4): the rotation from ``q_prev`` to ``q_curr``
    (both in the frame ``axis`` is expressed in) projected on ``axis``."""
    return (quat_to_rotvec(quat_mul(q_curr, quat_conjugate(q_prev))) * axis).sum(dim=-1)


def rotation_reward(delta: torch.Tensor, c1: float) -> torch.Tensor:
    """Eq. 4: clip(Delta Theta . k, -c1, c1), c1 = 0.025 rad."""
    return delta.clamp(-c1, c1)


def next_goal(obj_quat: torch.Tensor, axis: torch.Tensor, increment: float) -> torch.Tensor:
    """Sec. 3.1: the current object orientation rotated by ``increment`` (rad)
    about ``axis`` (same frame as ``obj_quat``)."""
    angle = torch.full((obj_quat.shape[0],), float(increment), device=obj_quat.device)
    return quat_mul(quat_from_angle_axis(angle, axis), obj_quat)


def goal_reached(kp_dist: torch.Tensor, rot_dist: torch.Tensor, d_tol: float, metric: str) -> torch.Tensor:
    """Eq. 5's condition. The paper thresholds kp_dist at d_tol = 0.15 (teacher;
    Table 5), which cannot be metres: its own drop threshold is kp_dist > 0.1 m
    and a 30 degree goal moves the 5 cm keypoints by 0.017-0.026 m on average.
    ``metric="rotation_rad"`` (default) reads d_tol as the rotation distance
    to the goal in radians (0.15 rad = 8.6 degrees, under the 30 degree
    increment); ``metric="kp_dist_m"`` applies it to kp_dist literally."""
    if metric == "rotation_rad":
        return rot_dist < d_tol
    if metric == "kp_dist_m":
        return kp_dist < d_tol
    raise ValueError(f"goal tolerance metric {metric!r}")


# --------------------------------------------------------------------------
# Contact and stability terms (App. B.1, Eqs. 6-11)
# --------------------------------------------------------------------------


def contact_rewards(tip_contact: torch.Tensor, nontip_contacts: torch.Tensor):
    """Eq. 6: r_gc = 1 if >= 2 fingertips touch the object. Eq. 7: r_bc = 1
    if any non-fingertip body touches it (the paper prints ">= 0", which would
    hold always; ``> 0`` is the stated intent), weighted negatively (a
    penalty)."""
    r_gc = (tip_contact.float().sum(dim=-1) >= 2).float()
    r_bc = (nontip_contacts > 0).float()
    return r_gc, r_bc


def angular_velocity_penalty(ang_vel: torch.Tensor, omega_max: float) -> torch.Tensor:
    """Eq. 8, as described ("penalises the agent if the angular velocity of
    the object exceeds the maximum"): -max(||w|| - w_max, 0). The printed
    -min(||w|| - w_max, 0) would reward speeds below the maximum."""
    return -(ang_vel.norm(dim=-1) - omega_max).clamp(min=0.0)


def _masked(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    return x if mask is None else torch.where(mask, x, torch.zeros_like(x))


def pose_penalty(q: torch.Tensor, q0: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Eq. 9: -||q - q0||, q0 the canonical grasp pose."""
    return -_masked(q - q0, mask).norm(dim=-1)


def work_penalty(torque: torch.Tensor, dq: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Eq. 10, "controller work-done penalty": -|tau^T dq| over the control
    step (dq the joint displacement during the step). The printed tau^T q-bar
    multiplies torque by the absolute target position."""
    return -_masked(torque * dq, mask).sum(dim=-1).abs()


def torque_penalty(torque: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Eq. 11: -||tau||."""
    return -_masked(torque, mask).norm(dim=-1)


# --------------------------------------------------------------------------
# Terminations (Eq. 12) and the adaptive reward curriculum (App. B.3)
# --------------------------------------------------------------------------


def axis_deviation(ang_vel: torch.Tensor, axis: torch.Tensor, min_speed: float):
    """Angle (rad) between the object's current rotation axis (its angular
    velocity direction) and ``axis``, and whether it is defined
    (``||w|| >= min_speed``; the paper gives no gate, but the direction of a
    near-zero angular velocity is noise)."""
    speed = ang_vel.norm(dim=-1)
    cos = (ang_vel * axis).sum(dim=-1) / speed.clamp(min=1e-8)
    return torch.acos(cos.clamp(-1.0, 1.0)), speed >= min_speed


def terminations(kp_dist, d_max, axis_dev, axis_valid, axis_dev_max):
    """Eq. 12: dropped (kp_dist > d_max = 0.1) or rotating about the wrong
    axis (deviation > 45 degrees)."""
    dropped = kp_dist > d_max
    off_axis = axis_valid & (axis_dev > axis_dev_max)
    return dropped, off_axis


def reward_curriculum_lambda(g_eval: float, g_min: float, g_max: float) -> float:
    """Eq. 15: lambda_rew = (g_eval - g_min) / (g_max - g_min), clipped to
    [0, 1] (the curriculum is "active" only inside [g_min, g_max])."""
    return float(min(max((g_eval - g_min) / max(g_max - g_min, 1e-9), 0.0), 1.0))


def combine_rewards(terms: dict, weights: dict, lam_rew: float) -> torch.Tensor:
    """Eq. 2: r = (l_kp r_kp + l_rot r_rot + l_goal r_goal)
    + l_rew (l_gc r_gc - l_bc r_bc) + l_rew (l_w r_w + l_pose r_pose
    + l_work r_work + l_torque r_torque) - l_penalty r_penalty.
    ``terms`` hold the unweighted r_*; r_bc and r_penalty are indicators
    (subtracted), the stability terms are already <= 0."""
    w = weights
    rotation = w["kp"] * terms["kp"] + w["rot"] * terms["rot"] + w["goal"] * terms["goal"]
    contact = w["gc"] * terms["gc"] - w["bc"] * terms["bc"]
    stable = (w["omega"] * terms["omega"] + w["pose"] * terms["pose"] + w["work"] * terms["work"]
              + w["torque"] * terms["torque"])
    return rotation + lam_rew * (contact + stable) - w["penalty"] * terms["penalty"]


# --------------------------------------------------------------------------
# Action, axes and tactile (Sec. 3.1, App. F)
# --------------------------------------------------------------------------


def relative_joint_targets(prev_target, action, prev_action, eta, action_scale, lower, upper):
    """Sec. 3.1, "Action Space": a_t in [-1, 1] scaled to Delta theta in
    [-0.026, 0.026] rad; a~_t = eta a_t + (1 - eta) a_{t-1}; q-bar_t =
    q-bar_{t-1} + a~_t, kept inside the joint limits. Returns
    ``(q-bar_t, a_t scaled)``."""
    a = action.clamp(-1.0, 1.0) * action_scale
    a_prev = prev_action.clamp(-1.0, 1.0) * action_scale
    smoothed = eta * a + (1.0 - eta) * a_prev
    target = torch.max(torch.min(prev_target + smoothed, upper), lower)
    return target, a


def sample_axes(n: int, mode: str, device=None, generator=None) -> torch.Tensor:
    """Commanded rotation axes in the palm frame. "sphere": uniform on S^2
    ("arbitrary rotation axes"; the paper does not give its distribution);
    "principal": one of +-x, +-y, +-z; "z": the palm normal (+z)."""
    if mode == "sphere":
        v = torch.randn(n, 3, device=device, generator=generator)
        return torch.nn.functional.normalize(v, dim=-1)
    if mode == "principal":
        idx = torch.randint(0, 6, (n,), device=device, generator=generator)
        return keypoint_offsets(1.0, device=device)[idx]
    if mode == "z":
        out = torch.zeros(n, 3, device=device)
        out[:, 2] = 1.0
        return out
    raise ValueError(f"rotation axis sampling {mode!r}")


def simulated_tactile(force_tip, contact_dir_tip, force_prev, *, alpha, threshold, beta_f, f_max,
                      beta_p, p_max):
    """App. F. ``force_tip``: (n, k, 3) object contact force on each fingertip
    (tip frame); ``contact_dir_tip``: (n, k, 3) direction from the tip origin
    to the contact point (tip frame; zeros where there is no contact);
    ``force_prev``: last step's smoothed force. Returns ``(c, P, F, F_smooth)``:
    Eq. 17 F = alpha F_t + (1 - alpha) F_{t-1}; Eq. 16 c = ||F|| > 0.25 N;
    Eq. 18 F = beta_F clip(||F||, 0, F_max); Eq. 19 P = beta_P clip(P, -P_max,
    P_max), P the two tilt angles of the contact direction about the tip's
    x and y axes; P and F masked by c."""
    f_smooth = alpha * force_tip + (1.0 - alpha) * force_prev
    mag = f_smooth.norm(dim=-1)
    c = mag > threshold
    d = contact_dir_tip
    pose = torch.stack([torch.atan2(d[..., 1], d[..., 2]), torch.atan2(d[..., 0], d[..., 2])], dim=-1)
    pose = beta_p * pose.clamp(-p_max, p_max) * c.unsqueeze(-1)
    force = beta_f * mag.clamp(0.0, f_max) * c
    return c, pose, force, f_smooth


# --------------------------------------------------------------------------
# Evaluation metrics -> evolution fitness (Sec. 4 "Evaluation")
# --------------------------------------------------------------------------


def graded_rotation_fitness(rotation_rad: torch.Tensor, time_to_terminate_s: torch.Tensor,
                            episode_s: float, time_weight: float = 0.25) -> torch.Tensor:
    """Per-episode fitness from AnyRotate's two metrics: rotations about k
    (Rot, ``rotation_rad / 2 pi``, negative progress counts as 0) plus
    ``time_weight`` x the fraction of the episode before termination (TTT)."""
    rot = rotation_rad.clamp(min=0.0) / (2.0 * math.pi)
    return rot + time_weight * (time_to_terminate_s / max(episode_s, 1e-6)).clamp(0.0, 1.0)
