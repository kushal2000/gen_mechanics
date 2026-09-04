"""Observation assembly and step-shared geometry caches for PoseReach."""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import convert_quat, quat_apply, quat_from_angle_axis, quat_mul


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------


# The field-size table and keypoint count live in Kit-free ``layout.py`` so the
# network can build exact gather indices without booting the simulator.
from .layout import (  # noqa: E402
    NUM_KEYPOINTS, compute_obs_dim, obs_field_sizes,
)

# Object-frame keypoint corners before scaling.
KEYPOINT_CORNERS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 1),
    (1, 1, -1),
    (-1, -1, 1),
    (-1, -1, -1),
)

# Joint count, fingertip count, and the palm/fingertip geometry offsets used to
# be module constants here (29 / 5 / two fixed 3-vectors). They are now read
# from the RobotSpec, so obs dims follow the selected hand instead of being
# pinned to SHARPA. See hand_sampler/spec.py.


def _stack_obs_dict(obs_dict: dict[str, torch.Tensor], field_list) -> torch.Tensor:
    """Concatenate named tensors in config order."""
    return torch.cat(
        [obs_dict[f].reshape(obs_dict[f].shape[0], -1) for f in field_list],
        dim=-1,
    )


# ----------------------------------------------------------------------------
# Quaternion / keypoint helpers
# ----------------------------------------------------------------------------


def _perturb_quat(q_wxyz: torch.Tensor, max_deg: float) -> torch.Tensor:
    """Apply random-axis rotation noise to wxyz quaternions."""
    n = q_wxyz.shape[0]
    axis = torch.nn.functional.normalize(
        torch.randn(n, 3, device=q_wxyz.device), dim=-1
    )
    angle = torch.empty(n, device=q_wxyz.device).uniform_(
        -max_deg, max_deg
    ) * (math.pi / 180.0)
    dq = quat_from_angle_axis(angle, axis)
    return quat_mul(dq, q_wxyz)


def _apply_local_offset(
    pos_w: torch.Tensor,
    rot_wxyz: torch.Tensor,
    offset,
    batch_shape: tuple[int, ...],
) -> torch.Tensor:
    """Apply a local-frame offset to batched world poses.

    ``offset`` is either one 3-vector broadcast over the whole batch (the palm
    center) or a per-item ``(F, 3)`` stack (fingertip pad centers, which differ
    per finger on hands with asymmetric distal geometry).
    """
    # A (3,) offset broadcasts across the whole batch; an (F, 3) stack
    # broadcasts across the leading env dim, giving each finger its own offset.
    offset_t = torch.as_tensor(offset, device=pos_w.device, dtype=pos_w.dtype)
    offset_t = offset_t.expand(*batch_shape, 3)
    shifted = quat_apply(
        rot_wxyz.reshape(-1, 4), offset_t.reshape(-1, 3)
    ).reshape(*batch_shape, 3)
    return pos_w + shifted


def _keypoints_world(
    center_pos: torch.Tensor,    # (N, 3)
    center_rot: torch.Tensor,    # (N, 4) wxyz
    kp_offsets: torch.Tensor,    # (N, K, 3)
) -> torch.Tensor:
    """Rotate + translate object-frame keypoints."""
    n_envs, k, _ = kp_offsets.shape
    rot_r = center_rot.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4)
    offsets_r = kp_offsets.reshape(-1, 3)
    return center_pos.unsqueeze(1) + quat_apply(rot_r, offsets_r).reshape(n_envs, k, 3)


def _episode_start(env) -> torch.Tensor:
    return (env.episode_length_buf == 0) & (env._successes == 0)


def _sample_delay(
    queue: torch.Tensor,
    values: torch.Tensor,
    env,
    flush: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Push current values into a rolling queue and sample per-env delay."""
    if flush is not None and flush.any():
        queue[flush] = values[flush].unsqueeze(1).expand(-1, queue.shape[1], -1)

    queue = torch.roll(queue, shifts=1, dims=1)
    queue[:, 0, :] = values
    idx = torch.randint(0, queue.shape[1], (env.num_envs,), device=env.device)
    delayed = queue[torch.arange(env.num_envs, device=env.device), idx]
    return queue, delayed


def _normalize_joint_pos(raw, lower, upper) -> torch.Tensor:
    return 2.0 * (raw - lower) / (upper - lower) - 1.0


def _quat_apply_broadcast(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Quaternion rotation with broadcast views, avoiding flattened copies."""
    xyz = quat[..., 1:].expand_as(vec)
    t = torch.cross(xyz, vec, dim=-1) * 2.0
    return vec + quat[..., 0:1] * t + torch.cross(xyz, t, dim=-1)


def _canonical_joint_obs(env) -> tuple[torch.Tensor, ...]:
    """Return current/history proprioception and applied targets in policy order."""
    perm = env._perm_lab_to_canon
    joint_pos_raw = env.robot.data.joint_pos[:, perm]
    joint_pos = _normalize_joint_pos(
        joint_pos_raw, env._joint_lower_canon, env._joint_upper_canon
    )
    prev_joint_pos = _normalize_joint_pos(
        env._prev_joint_pos_canon, env._joint_lower_canon, env._joint_upper_canon
    )
    return (
        joint_pos,
        env.robot.data.joint_vel[:, perm],
        prev_joint_pos,
        env._prev_joint_vel_canon,
        env._prev_targets[:, perm],
    )


def _joint_link_geometry_obs(env, palm_center_pos_w, palm_rot, env_origins):
    """Batched SHARPA link boxes in the normalized palm frame plus joint origins."""
    state = env.robot.data.body_state_w[:, env._joint_link_body_ids, :]
    body_pos = state[:, :, 0:3]
    body_rot = state[:, :, 3:7]
    n, joints, points, _ = env._joint_link_bbox_local.shape

    box_world = body_pos.unsqueeze(2) + _quat_apply_broadcast(
        body_rot.unsqueeze(2), env._joint_link_bbox_local
    )

    palm_inv = palm_rot.clone()
    palm_inv[:, 1:] *= -1.0
    box_palm = _quat_apply_broadcast(
        palm_inv[:, None, None, :],
        box_world - palm_center_pos_w[:, None, None, :],
    )
    box_palm = box_palm / env._hand_scale[:, None, None, :]

    valid = env._joint_geometry_valid
    box_palm = box_palm * valid[:, :, None, None].to(box_palm.dtype)
    joint_origins = body_pos - env_origins.unsqueeze(1)
    return box_palm, joint_origins, valid


def _object_keypoints_rel_joint(
    object_keypoints, joint_origins, palm_rot, hand_scale, valid
) -> torch.Tensor:
    """Object keypoints from every joint, in normalized palm-frame vectors."""
    rel_world = object_keypoints[:, None, :, :] - joint_origins[:, :, None, :]
    palm_inv = palm_rot.clone()
    palm_inv[:, 1:] *= -1.0
    rel_palm = _quat_apply_broadcast(palm_inv[:, None, None, :], rel_world)
    rel_palm = rel_palm / hand_scale[:, None, None, :]
    return rel_palm * valid[:, :, None, None].to(rel_palm.dtype)


# ----------------------------------------------------------------------------
# Step-shared intermediate values (feeds _get_dones + _get_rewards)
# ----------------------------------------------------------------------------


def compute_intermediate_values(env) -> None:
    """Update shared geometric state for rewards and terminations."""
    from ..reward_utils import update_near_goal_steps  # local import to avoid cycle

    rew_cfg = env.cfg.reward
    term_cfg = env.cfg.termination
    env_origins = env.scene.env_origins

    obj_pos = env.object.data.root_pos_w - env_origins
    obj_rot = env.object.data.root_quat_w
    goal_pos = env.goal_viz.data.root_pos_w - env_origins
    goal_rot = env.goal_viz.data.root_quat_w

    ft_state = env.robot.data.body_state_w[:, env._fingertip_body_ids, :]
    ft_pos = ft_state[:, :, 0:3] - env_origins.unsqueeze(1)
    env._curr_fingertip_distances = torch.norm(
        ft_pos - obj_pos.unsqueeze(1), dim=-1
    )  # (N, S)
    # Ghosted slots carry a real link pose and a meaningless distance. Zeroing
    # here, at the single point where the distance is produced, makes every
    # downstream reduction inert for them at once: the reward sums deltas (0),
    # termination takes a max against 1.5 m (0 never trips), the running minimum
    # stays 0, and the observation shows a constant. Masking at each consumer
    # instead would leave whichever one gets forgotten reading ghost geometry.
    # For an unpadded spec the mask is all-true and this is a no-op.
    env._curr_fingertip_distances = torch.where(
        env._fingertip_mask, env._curr_fingertip_distances,
        torch.zeros_like(env._curr_fingertip_distances),
    )

    if rew_cfg.fixed_size_keypoint_reward:
        kp_offsets = env._keypoint_offsets_fixed
    else:
        kp_offsets = env._keypoint_offsets

    obj_kp = _keypoints_world(obj_pos, obj_rot, kp_offsets)
    goal_kp = _keypoints_world(goal_pos, goal_rot, kp_offsets)

    env._keypoints_max_dist = torch.norm(obj_kp - goal_kp, dim=-1).max(dim=-1).values

    # Legacy -1 sentinel: first observed value becomes closest-so-far.
    sentinel = env._closest_keypoint_max_dist < 0.0
    env._closest_keypoint_max_dist = torch.where(
        sentinel, env._keypoints_max_dist, env._closest_keypoint_max_dist
    )
    sentinel_ft = env._closest_fingertip_dist < 0.0
    env._closest_fingertip_dist = torch.where(
        sentinel_ft, env._curr_fingertip_distances, env._closest_fingertip_dist
    )

    if hasattr(env, "_keypoint_success_tolerance_m"):
        tol = env._keypoint_success_tolerance_m()
    else:
        tol = env._current_success_tolerance * rew_cfg.keypoint_scale
    env._near_goal = env._keypoints_max_dist <= tol
    env._near_goal_steps = update_near_goal_steps(
        near_goal=env._near_goal,
        near_goal_steps=env._near_goal_steps,
        force_consecutive=term_cfg.force_consecutive_near_goal_steps,
    )
    env._is_success = env._near_goal_steps >= term_cfg.success_steps


# ----------------------------------------------------------------------------
# Observation builder (Phase D)
# ----------------------------------------------------------------------------


def _apply_object_state_dr(env, obj_pos, obj_rot, obj_linvel, obj_angvel):
    """Apply object-state delay and pose noise."""
    dr = env.cfg.domain_randomization
    state = torch.cat([obj_pos, obj_rot, obj_linvel, obj_angvel], dim=-1)
    env._object_state_queue, delayed = _sample_delay(
        env._object_state_queue, state, env, flush=_episode_start(env)
    )
    noisy_pos = delayed[:, 0:3] + torch.randn_like(delayed[:, 0:3]) * dr.object_state_xyz_noise_std
    noisy_rot = _perturb_quat(delayed[:, 3:7], dr.object_state_rotation_noise_degrees)
    noisy_vel = delayed[:, 7:13]
    return noisy_pos, noisy_rot, noisy_vel


def _apply_obs_delay(env, policy_tensor: torch.Tensor) -> torch.Tensor:
    """Apply per-env policy-observation delay."""
    env._obs_queue, delayed = _sample_delay(
        env._obs_queue, policy_tensor, env, flush=_episode_start(env)
    )
    return delayed


def build_observations(env) -> dict[str, torch.Tensor]:
    """Assemble actor-critic observations with obs-side DR."""
    dr = env.cfg.domain_randomization
    env_origins = env.scene.env_origins

    (
        joint_pos,
        joint_vel,
        prev_joint_pos,
        prev_joint_vel,
        prev_targets_canon,
    ) = _canonical_joint_obs(env)

    palm_state = env.robot.data.body_state_w[:, env._palm_body_id, :]  # (N, 13)
    palm_pos_w = palm_state[:, 0:3]
    palm_rot = palm_state[:, 3:7]  # wxyz (Isaac Lab convention)
    palm_vel = palm_state[:, 7:13]

    palm_center_pos_w = _apply_local_offset(
        palm_pos_w, palm_rot, env._palm_center_offset, (env.num_envs,)
    )
    palm_pos = palm_center_pos_w - env_origins

    # Fingertip pad centres. Restored verbatim from 7a8a98c: dropping this
    # field when the per-joint token fields landed is what the MLP control
    # arm regressed on. The joint_link_bbox block does carry the distal links'
    # geometry, but only implicitly, and a dense first layer evidently cannot
    # recover it -- done_hand_far went from 1.3% of episodes to 54%, while the
    # transformer on the SAME observation stayed at 1.6%.
    ft_state = env.robot.data.body_state_w[:, env._fingertip_body_ids, :]
    ft_pos_w = _apply_local_offset(
        ft_state[:, :, 0:3], ft_state[:, :, 3:7], env._fingertip_offsets,
        (env.num_envs, env._num_fingertips),
    )
    fingertip_pos_rel_palm = (
        (ft_pos_w - env_origins.unsqueeze(1)) - palm_pos.unsqueeze(1)
    )  # (N, S, 3)
    # Ghosted slots would otherwise feed the policy the pose of a finger that
    # is not there. Zero is the "absent" value, matching the distance field.
    fingertip_pos_rel_palm = fingertip_pos_rel_palm * (
        env._fingertip_mask.unsqueeze(-1).to(fingertip_pos_rel_palm.dtype)
    )

    obj_pos = env.object.data.root_pos_w - env_origins
    obj_rot = env.object.data.root_quat_w  # wxyz
    obj_linvel = env.object.data.root_lin_vel_w
    obj_angvel = env.object.data.root_ang_vel_w
    obj_vel = torch.cat([obj_linvel, obj_angvel], dim=-1)

    goal_pos = env.goal_viz.data.root_pos_w - env_origins
    goal_rot = env.goal_viz.data.root_quat_w  # wxyz

    kp_offsets = env._keypoint_offsets * env._object_scale_multiplier.unsqueeze(1)
    obj_kp = _keypoints_world(obj_pos, obj_rot, kp_offsets)
    goal_kp = _keypoints_world(goal_pos, goal_rot, kp_offsets)

    # With object-state DR off the "noisy" object IS the clean one, so every
    # tensor derived from it is the clean tensor. Aliasing rather than
    # recomputing is bit-for-bit identical and skips a second pass over the
    # (N, 22, 4, 3) per-joint keypoint transform -- the widest tensor the
    # observation builds. Identity (`is`) is the test throughout: it is true
    # exactly when the inputs came from the same object, which is the only
    # case where reuse is sound.
    if dr.use_object_state_delay_noise:
        noisy_obj_pos, noisy_obj_rot, noisy_obj_vel = _apply_object_state_dr(
            env, obj_pos, obj_rot, obj_linvel, obj_angvel
        )
        noisy_obj_kp = _keypoints_world(noisy_obj_pos, noisy_obj_rot, kp_offsets)
    else:
        noisy_obj_pos, noisy_obj_rot, noisy_obj_vel = obj_pos, obj_rot, obj_vel
        noisy_obj_kp = obj_kp

    # Optional per-env yaw noise on the observed goal (world +Z about goal_pos).
    goal_yaw_obs_noise = getattr(env, "goal_yaw_obs_noise", None)
    if goal_yaw_obs_noise is not None and torch.any(goal_yaw_obs_noise != 0):
        z_axis = torch.tensor(
            [0.0, 0.0, 1.0], device=goal_rot.device, dtype=goal_rot.dtype
        ).unsqueeze(0).expand(env.num_envs, -1)
        yaw_q = quat_from_angle_axis(goal_yaw_obs_noise, z_axis)
        noisy_goal_rot = quat_mul(yaw_q, goal_rot)
        noisy_goal_kp = _keypoints_world(goal_pos, noisy_goal_rot, kp_offsets)
    else:
        noisy_goal_kp = goal_kp

    object_is_clean = noisy_obj_kp is obj_kp
    keypoints_rel_palm_clean = obj_kp - palm_pos.unsqueeze(1)
    keypoints_rel_palm_noisy = (
        keypoints_rel_palm_clean if object_is_clean
        else noisy_obj_kp - palm_pos.unsqueeze(1)
    )
    keypoints_rel_goal_clean = obj_kp - goal_kp
    keypoints_rel_goal_noisy = (
        keypoints_rel_goal_clean
        if object_is_clean and noisy_goal_kp is goal_kp
        else noisy_obj_kp - noisy_goal_kp
    )

    joint_link_bbox, joint_origins, joint_geometry_valid = (
        _joint_link_geometry_obs(env, palm_center_pos_w, palm_rot, env_origins)
    )
    object_keypoints_rel_joint_clean = _object_keypoints_rel_joint(
        obj_kp, joint_origins, palm_rot, env._hand_scale, joint_geometry_valid
    )
    object_keypoints_rel_joint_noisy = (
        object_keypoints_rel_joint_clean if object_is_clean
        else _object_keypoints_rel_joint(
            noisy_obj_kp, joint_origins, palm_rot, env._hand_scale,
            joint_geometry_valid,
        )
    )

    object_scales_obs = env.scene_record.object_scale * env._object_scale_multiplier

    # Policy obs use legacy Isaac Gym xyzw; internal math stays wxyz.
    palm_rot_xyzw = convert_quat(palm_rot, to="xyzw")
    obj_rot_xyzw = convert_quat(obj_rot, to="xyzw")
    noisy_obj_rot_xyzw = (
        obj_rot_xyzw if noisy_obj_rot is obj_rot
        else convert_quat(noisy_obj_rot, to="xyzw")
    )

    obs_clean: dict[str, torch.Tensor] = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "prev_joint_pos": prev_joint_pos,
        "prev_joint_vel": prev_joint_vel,
        "prev_action_targets": prev_targets_canon,
        "joint_link_bbox": joint_link_bbox,
        "joint_lower": env._joint_lower_hand,
        "joint_upper": env._joint_upper_hand,
        "joint_enabled": env._joint_enabled,
        "object_keypoints_rel_joint": object_keypoints_rel_joint_clean,
        "hand_scale": env._hand_scale,
        "palm_pos": palm_pos,
        "palm_rot": palm_rot_xyzw,
        "fingertip_pos_rel_palm": fingertip_pos_rel_palm,
        "palm_vel": palm_vel,
        "object_rot": obj_rot_xyzw,
        "object_vel": obj_vel,
        "keypoints_rel_palm": keypoints_rel_palm_clean,
        "keypoints_rel_goal": keypoints_rel_goal_clean,
        "object_scales": object_scales_obs,
        "closest_keypoint_max_dist": env._closest_keypoint_max_dist.unsqueeze(-1),
        "closest_fingertip_dist": env._closest_fingertip_dist,
        # Only present on the multi-embodiment env; a single-hand scene has no
        # morphology to describe and never lists the field.
        **({"morphology": env._morphology_per_env}
           if getattr(env, "_morphology_per_env", None) is not None else {}),
        "lifted_object": env._lifted_object.float().unsqueeze(-1),
        "progress": torch.log(env.episode_length_buf.float() / 10.0 + 1.0).unsqueeze(-1),
        "successes": torch.log(env._successes.float() + 1.0).unsqueeze(-1),
        "reward": (env.reward_buf * 0.01).unsqueeze(-1),
    }

    obs_noisy = dict(obs_clean)
    obs_noisy["object_rot"] = noisy_obj_rot_xyzw
    obs_noisy["object_vel"] = noisy_obj_vel
    obs_noisy["keypoints_rel_palm"] = keypoints_rel_palm_noisy
    obs_noisy["keypoints_rel_goal"] = keypoints_rel_goal_noisy
    obs_noisy["object_keypoints_rel_joint"] = object_keypoints_rel_joint_noisy
    if dr.joint_velocity_obs_noise_std > 0:
        obs_noisy["joint_vel"] = (
            joint_vel + torch.randn_like(joint_vel) * dr.joint_velocity_obs_noise_std
        )
        obs_noisy["prev_joint_vel"] = (
            prev_joint_vel
            + torch.randn_like(prev_joint_vel) * dr.joint_velocity_obs_noise_std
        )

    state_tensor = _stack_obs_dict(obs_clean, env.cfg.obs.state_list)
    policy_tensor = _stack_obs_dict(obs_noisy, env.cfg.obs.obs_list)

    if dr.use_obs_delay:
        policy_tensor = _apply_obs_delay(env, policy_tensor)

    clip = env.cfg.obs.clamp_abs_observations
    policy_tensor = policy_tensor.clamp(-clip, clip)
    state_tensor = state_tensor.clamp(-clip, clip)

    return {"policy": policy_tensor, "critic": state_tensor}


def derive_spaces(cfg, spec) -> None:
    """Write the action/observation/state widths onto ``cfg``.

    Mutates: rl_games and DirectRLEnv read these off the configclass, so the env
    must set them before ``super().__init__``. ``action_space`` 0 means derive;
    a stale non-zero would truncate the action vector.
    """
    if cfg.action_space not in (0, spec.num_joints):
        raise ValueError(
            f"cfg.action_space={cfg.action_space} disagrees with robot_spec "
            f"{spec.name!r} ({spec.num_joints} joints). Leave it 0, or build a "
            "fresh cfg -- this one may have been used for another robot.")
    cfg.action_space = spec.num_joints
    cfg.observation_space = compute_obs_dim(cfg.obs.obs_list, spec)
    cfg.state_space = compute_obs_dim(cfg.obs.state_list, spec)


def force_morphology_field(cfg, n_designs: int) -> None:
    """Put ``morphology`` in both obs lists, or strip it for the ablation.

    Forced, not trusted: the YAML overlay is applied after the configclass
    defaults and dropped it once, showing up only as obs 186 wide, not 329.
    """
    include = cfg.include_morphology_obs
    for name in ("obs_list", "state_list"):
        fields = tuple(getattr(cfg.obs, name))
        if include and "morphology" not in fields:
            fields += ("morphology",)
        elif not include:
            fields = tuple(f for f in fields if f != "morphology")
        setattr(cfg.obs, name, fields)
    if not include:
        print(f"[pose_reach] ABLATION: no morphology descriptor; the policy "
              f"cannot tell its {n_designs} designs apart.", flush=True)
