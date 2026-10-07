"""Observation assembly and the per-step geometry cache (object pose in the
palm frame, rotation error to goal) every reward/termination hook reads."""

from __future__ import annotations

import torch
from isaaclab.utils.math import (
    quat_apply_inverse, quat_error_magnitude, quat_inv, quat_mul, subtract_frame_transforms,
)

__all__ = [
    "OBS_FIELD_WIDTHS", "derive_spaces", "compute_intermediate_values",
    "update_palm_frame_geometry", "pre_physics_step", "build_observations",
]

# Width (last-dim size) of each observation field, independent of the hand.
# "joint_pos"/"joint_vel"/"prev_actions" scale with the hand's joint count;
# "fingertip_pos_palm" scales with its fingertip count -- both resolved in
# ``_field_width`` from the spec rather than listed here.
OBS_FIELD_WIDTHS = {
    "object_pos_palm": 3,
    "object_quat_palm": 4,
    "object_lin_vel_palm": 3,
    "object_ang_vel_palm": 3,
    "goal_quat_palm": 4,
    "object_quat_rel_goal": 4,
}
_PER_JOINT_FIELDS = ("prev_actions", "joint_pos", "joint_vel")


def _field_width(name: str, spec) -> int:
    if name in _PER_JOINT_FIELDS:
        return spec.num_hand_joints
    if name == "fingertip_pos_palm":
        return 3 * spec.num_fingertips
    if name in OBS_FIELD_WIDTHS:
        return OBS_FIELD_WIDTHS[name]
    # The isaaclab_repose and anyrotate profiles' fields.
    from .anyrotate_profile import ANYROTATE_OBS_FIELDS, ANYROTATE_PRIV_FIELDS, anyrotate_field_width
    from .repose_profile import repose_field_width

    if name in ANYROTATE_OBS_FIELDS or name in ANYROTATE_PRIV_FIELDS or name.startswith("hora_"):
        return anyrotate_field_width(name, spec.num_hand_joints, spec.num_fingertips)
    return repose_field_width(name, spec.num_hand_joints, spec.num_fingertips)


def derive_spaces(cfg, spec) -> None:
    """Set action/observation/state space sizes from the hand spec + obs cfg."""
    cfg.action_space = spec.num_hand_joints
    cfg.observation_space = sum(_field_width(f, spec) for f in cfg.obs.obs_list)
    cfg.state_space = sum(_field_width(f, spec) for f in cfg.obs.state_list)


def _joint_valid_mask(env) -> torch.Tensor | None:
    """`(num_envs, 36)` bool, `True` where that env's own design has a joint
    the policy controls in that envelope slot (a real finger joint or a
    leader palm joint; not a ghost, locked or follower carrier), in
    ARTICULATION-VIEW column order -- `None` for the single-hand path. Same computation as
    `reward_utils._joint_valid_mask` (duplicated here, not imported: that
    module pulls in `isaacsimenvs.pose_reaching_6d`, which bootstraps Kit on
    import -- see this module's own lazy-import discipline elsewhere in the
    package, e.g. `drop_detection.py`'s docstring)."""
    tables = getattr(env, "hand_tables", None)
    if tables is None:
        return None
    design_idx = env.scene_record["design_idx"]
    valid = torch.as_tensor(tables.joint_valid, device=env.device, dtype=torch.bool)
    valid = valid[design_idx]
    perm = env.scene_record.get("slot_of_phys_col")
    if perm is not None:
        valid = valid[:, perm]
    return valid


def pre_physics_step(env, actions: torch.Tensor) -> None:
    """Joint-position targets with moving-average smoothing, PoseReach-style.

    Actions in [-1, 1] scale PIECEWISE-LINEARLY around each joint's DEFAULT
    position (``env.robot.data.default_joint_pos``, the calibrated rest pose
    -- see ``hand_calibration.json``), not around the raw midpoint of its
    limits: action=+1 still lands exactly on the upper limit and action=-1 on
    the lower limit, but action=0 now reproduces the rest pose exactly. This
    matters because ``hand_only.build_hand_only_spec`` sets each joint's
    default to ``clamp(0.0, lower, upper)``, which for an asymmetric-limit
    joint (e.g. allegro_right's thumb joint_12, limits [0.263, 1.396], default
    0.263) sits far from the limits' midpoint -- roughly half the joint's span
    away. Under the old (mid-of-limits) mapping, "zero action" pulled that
    joint away from its good rest pose toward the middle of its range on
    every single step from the first reset onward, before any policy
    learning; a from-scratch policy (whose action mean starts near 0) would
    have to learn a per-joint bias just to re-find the rest pose before it
    could learn anything about rotating the object.
    """
    actions = actions.clamp(-1.0, 1.0)
    joint_mask = _joint_valid_mask(env)  # None on the single-hand path
    if joint_mask is not None:
        # Review item 9: a ghost column's action penalty is already zeroed
        # in reward_utils.compute_rewards, but the RAW action was still
        # stored into `env._prev_actions` -- which is itself an observation
        # field (obs_utils.build_observations's "prev_actions") -- giving
        # the policy a free, unpenalized read/write memory channel across
        # steps with no physical grounding (it can never move a ghost joint;
        # its own limits are (0, 1e-8) regardless). Zero it at the source so
        # neither the observation nor the reward can see anything but 0
        # there, closing the channel instead of merely not rewarding it.
        actions = actions * joint_mask
    env._prev_actions_this_step = env._prev_actions.clone()
    env._prev_actions = actions

    lower = env.robot.data.soft_joint_pos_limits[:, :, 0]
    upper = env.robot.data.soft_joint_pos_limits[:, :, 1]
    # Review risk 11: on the population path, read the per-env default pose
    # from this package's OWN cached tensor (`env.scene_record
    # ["default_joint_pos"]`, set once by `scene_utils.
    # _resolve_population_joint_permutation`), not Isaac Lab's own
    # `default_joint_pos` buffer -- which reverts to the scene-wide CFG
    # template (env 0's design) if the articulation re-initializes. `None`
    # on the single-hand path, unchanged.
    cached_default = (
        env.scene_record.get("default_joint_pos") if getattr(env, "hand_tables", None) is not None else None
    )
    default_pos = cached_default if cached_default is not None else env.robot.data.default_joint_pos
    scale = env.cfg.action.dof_speed_scale
    span_up = (upper - default_pos) * scale
    span_dn = (default_pos - lower) * scale
    scaled = default_pos + torch.where(actions >= 0, actions * span_up, actions * span_dn)

    alpha = env.cfg.action.hand_moving_average
    if not hasattr(env, "_cur_targets"):
        env._cur_targets = env.robot.data.joint_pos.clone()
    env._cur_targets = alpha * scaled + (1.0 - alpha) * env._cur_targets
    env._cur_targets = torch.clamp(env._cur_targets, lower, upper)


def _fingertip_valid_mask(env) -> torch.Tensor | None:
    """`(num_envs, 6)` bool, `True` where that env's own design has a finger
    in that finger slot (every finger slot ends in a fingertip body `f{f}_tip`
    at the real fingertip, see `grammar_envelope.tip_offsets`). `None` on the
    single-hand path, same pattern as `reward_utils._joint_valid_mask`."""
    tables = getattr(env, "hand_tables", None)
    if tables is None:
        return None
    design_idx = env.scene_record["design_idx"]
    valid = torch.as_tensor(tables.fingertip_valid, device=env.device, dtype=torch.bool)
    return valid[design_idx]  # (num_envs, 6)


def tie_joints(env, x: torch.Tensor, env_ids=None) -> torch.Tensor:
    """`x` (`(n, num_joints)` per-joint values in articulation-view column
    order, for `env_ids` or every env) with each follower carrier's column
    set to its leader's: a mimic joint holds the two equal, so every write of
    joint positions (resets) must too, or PhysX snaps them together. `x`
    unchanged on the single-hand path."""
    if getattr(env, "hand_tables", None) is None:
        return x
    idx = env.scene_record.get("tie_index")
    if idx is None:
        return x
    if env_ids is not None:
        idx = idx[env_ids]
    return x.gather(1, idx)


def update_palm_frame_geometry(env) -> None:
    """Object/goal pose in the palm frame and the rotation error to goal.
    Idempotent and safe to call from a partial reset (unlike
    ``compute_intermediate_values``, it touches no per-episode counter)."""
    palm_pos_w = env.robot.data.body_pos_w[:, env.palm_body_idx]
    palm_quat_w = env.robot.data.body_quat_w[:, env.palm_body_idx]
    obj_pos_w = env.object.data.root_pos_w
    obj_quat_w = env.object.data.root_quat_w

    obj_pos_palm, obj_quat_palm = subtract_frame_transforms(
        palm_pos_w, palm_quat_w, obj_pos_w, obj_quat_w)
    # Orientation only: the goal has no position of its own (its "position"
    # IS the object's spawn point, tracked separately as
    # ``env._spawn_obj_pos_palm``) -- the previous version of this line
    # passed ``obj_pos_w`` as the goal's translation, computing (and then
    # discarding) a position that was actually just ``obj_pos_palm`` again.
    goal_quat_palm = quat_mul(quat_inv(palm_quat_w), env._goal_quat_w)

    env._obj_pos_palm = obj_pos_palm
    env._obj_quat_palm = obj_quat_palm
    env._obj_lin_vel_palm = quat_apply_inverse(palm_quat_w, env.object.data.root_lin_vel_w)
    env._obj_ang_vel_palm = quat_apply_inverse(palm_quat_w, env.object.data.root_ang_vel_w)
    env._goal_quat_palm = goal_quat_palm
    env._obj_quat_rel_goal = quat_mul(quat_inv(env._goal_quat_w), obj_quat_w)
    env._rot_error = quat_error_magnitude(obj_quat_w, env._goal_quat_w)

    tip_idx = getattr(env, "fingertip_body_idx", None)
    if tip_idx:
        n, k = env.num_envs, len(tip_idx)
        tip_pos_w = env.robot.data.body_pos_w[:, tip_idx, :]  # (n, k, 3)
        rel_w = tip_pos_w - palm_pos_w.unsqueeze(1)
        palm_quat_flat = palm_quat_w.unsqueeze(1).expand(n, k, 4).reshape(n * k, 4)
        tip_pos_palm = quat_apply_inverse(palm_quat_flat, rel_w.reshape(n * k, 3)).reshape(n, k, 3)
        fingertip_mask = _fingertip_valid_mask(env)  # None on the single-hand path
        if fingertip_mask is not None:
            tip_pos_palm = tip_pos_palm * fingertip_mask.unsqueeze(-1)
        env._fingertip_pos_palm = tip_pos_palm.reshape(n, k * 3)


def compute_intermediate_values(env) -> None:
    """Called from ``_get_dones`` once per step, before terminations/rewards
    (mirrors ``pose_reaching_6d``'s hook order): refreshes the palm-frame
    geometry AND advances the per-step success bookkeeping. Reset uses
    ``update_palm_frame_geometry`` alone -- see its docstring."""
    env._prev_rot_error = env._rot_error
    update_palm_frame_geometry(env)

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
        "object_lin_vel_palm": env._obj_lin_vel_palm,
        "object_ang_vel_palm": env._obj_ang_vel_palm,
        "goal_quat_palm": env._goal_quat_palm,
        "object_quat_rel_goal": env._obj_quat_rel_goal,
        "fingertip_pos_palm": env._fingertip_pos_palm,
    }
    clamp = env.cfg.obs.clamp_abs_observations
    obs = torch.cat([values[f] for f in env.cfg.obs.obs_list], dim=-1).clamp(-clamp, clamp)
    state = torch.cat([values[f] for f in env.cfg.obs.state_list], dim=-1).clamp(-clamp, clamp)
    return {"policy": obs, "critic": state}
