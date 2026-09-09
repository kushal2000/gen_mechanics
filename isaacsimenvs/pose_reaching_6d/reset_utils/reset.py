"""State allocation and reset helpers for PoseReach."""

from __future__ import annotations


import numpy as np
import torch

from isaaclab.utils.math import random_orientation

from ..obs_utils import sample_log_uniform
from .goal_sampling import sample_absolute_goal_pose, sample_delta_goal_pose
from ..obs_utils import KEYPOINT_CORNERS


def allocate_state_buffers(env) -> None:
    """Populate every per-env buffer + index cache used by the hooks.

    Called once from ``__init__`` after ``super().__init__`` (which runs
    ``_setup_scene`` and makes ``env.robot``/``env.object``/``env.goal_viz``
    available).

    This is the single place the robot spec meets the live articulation: joint
    and body ids, the canonical<->Lab permutations, joint limits, and the
    geometry offsets are all resolved here and cached on ``env``.
    """
    dr = env.cfg.domain_randomization
    rew = env.cfg.reward
    spec = env.scene_record.robot_spec

    # --- Joint/body id caches ---
    # Joints are *selected* by exact name from the spec (not by a regex like
    # "left_.*", which assumes a naming convention), then sorted into ascending
    # Lab order.
    #
    # The sort matters. Isaac Lab's parser interleaves the hand joints relative
    # to canonical order (SHARPA lands as 7, 12, 17, 22, 27, 8, ...), so
    # preserving spec order here would reorder the summation in the hand action
    # penalty. That is mathematically identical but not identical in float32:
    # it perturbed the reward by ~3e-8 and broke bitwise parity. Ascending order
    # also means the reward cannot depend on the arbitrary order a spec happens
    # to list its joints in.
    #
    # Nothing downstream needs canonical order from these: the action pipeline,
    # limit tables, and target buffers all index with the same id tensors, so
    # any consistent order is correct. Canonical order is reached separately via
    # _perm_lab_to_canon.
    env._arm_joint_ids = sorted(
        env.robot.find_joints(list(spec.arm_joint_names), preserve_order=True)[0]
    )
    env._hand_joint_ids = sorted(
        env.robot.find_joints(list(spec.hand_joint_names), preserve_order=True)[0]
    )
    env._palm_body_id = env.robot.find_bodies(spec.palm_body_name)[0][0]
    # Fingertips keep spec order: column i of the fingertip observations is
    # finger i. Addressed by SLOT, so a padded design's ghost fingers occupy
    # their own columns and fingertip_valid masks them.
    tips = list(spec.fingertip_body_names)
    env._fingertip_body_ids = env.robot.find_bodies(tips, preserve_order=True)[0]

    env._num_joints = spec.num_joints
    env._num_hand_joints = spec.num_hand_joints
    # Width of every fingertip-shaped buffer and observation field.
    env._num_fingertips = spec.num_fingertips

    if len(env._fingertip_body_ids) != spec.num_fingertips:
        raise RuntimeError(
            f"{spec.name}: found {len(env._fingertip_body_ids)} fingertip bodies, "
            f"spec declares {spec.num_fingertips} ({tips})"
        )

    # (1, S) so it broadcasts over envs; the per-env robot path replaces this
    # with a genuine (N, S) mask, one row per design. A ghosted finger's distal
    # link still exists and still has a pose, and that pose is meaningless --
    # every reduction over the fingertip axis has to exclude it.
    # Everything downstream — action routing, the arm/hand reward penalties, the
    # limit tables — assumes these two id sets tile the joint vector exactly.
    if sorted(env._arm_joint_ids + env._hand_joint_ids) != list(range(spec.num_joints)):
        raise RuntimeError(
            f"{spec.name}: arm ids {env._arm_joint_ids} and hand ids "
            f"{env._hand_joint_ids} do not partition range({spec.num_joints})"
        )

    # Geometry offsets, spec-provided so they follow the hand. Set below, with
    # the rest of the per-design tables: the palm centre depends on palm length,
    # so a population carries one per design and a fixed hand one for all.


    # Convert between Lab parser order and canonical policy order.
    canonical = spec.joint_names_canonical
    lab_names = list(env.robot.data.joint_names)
    if set(lab_names) != set(canonical):
        only_urdf = sorted(set(lab_names) - set(canonical))
        only_spec = sorted(set(canonical) - set(lab_names))
        raise RuntimeError(
            f"{spec.name}: URDF joints do not match the spec "
            f"(in URDF only: {only_urdf}; in spec only: {only_spec})"
        )
    env._perm_canon_to_lab = torch.tensor(
        [canonical.index(n) for n in lab_names],
        device=env.device, dtype=torch.long,
    )
    env._perm_lab_to_canon = torch.tensor(
        [lab_names.index(n) for n in canonical],
        device=env.device, dtype=torch.long,
    )

    # --- hand-token geometry ------------------------------------------------
    # From the spec, so this path is identical for an imported hand and a
    # generated one. The rollout only gathers these body poses and applies one
    # batched transform.
    if not spec.joint_link_bodies:
        raise RuntimeError(f"{spec.name}: spec carries no joint tokens")
    env._joint_link_body_ids = env.robot.find_bodies(
        list(spec.joint_link_bodies), preserve_order=True
    )[0]
    if len(env._joint_link_body_ids) != spec.num_hand_joints:
        raise RuntimeError(
            f"{spec.name}: found {len(env._joint_link_body_ids)} controlled-link "
            f"bodies, expected {spec.num_hand_joints}"
        )
    # One design or many: a population gathers per env, a fixed hand broadcasts
    # the spec's own row. Both end up (N, ...) so nothing downstream branches.
    population = env.scene_record.population
    if population is None:
        per_env = {
            "joint_link_bbox_local": np.asarray(spec.joint_link_boxes, np.float32)[None],
            "joint_geometry_valid": np.asarray(spec.joint_geometry_valid, bool)[None],
            "hand_scale": np.full((1, 1), spec.hand_scale, np.float32),
            "fingertip_valid": np.ones((1, spec.num_fingertips), bool),
            "palm_center_offset": np.asarray(spec.palm_center_offset, np.float32)[None],
        }
        expand = True
    else:
        # hand_sampler is numpy-only and must stay importable without torch,
        # so the device round-trip happens here, not in per_env.
        per_env = population.per_env(
            env.scene_record.robot_design_index.detach().cpu().numpy())
        expand = False

    def _to(name, dtype):
        t = torch.as_tensor(per_env[name], device=env.device, dtype=dtype)
        return t.expand(env.num_envs, *t.shape[1:]) if expand else t

    env._joint_link_bbox_local = _to("joint_link_bbox_local", torch.float32)
    env._joint_geometry_valid = _to("joint_geometry_valid", torch.bool)
    env._hand_scale = _to("hand_scale", torch.float32)
    # A ghost FINGER's template tip body sits at the palm, so its distance is
    # meaningless; zeroing at the single point the distance is produced makes
    # every reduction over the fingertip axis inert for it at once.
    env._fingertip_mask = _to("fingertip_valid", torch.bool)
    env._palm_center_offset = _to("palm_center_offset", torch.float32)  # (N, 3)

    limits = env.robot.data.joint_pos_limits  # (N, num_joints, 2), Lab order

    # Canonical-order limits for normalizing joint_pos observations.
    #
    # PER ENV, not env 0 broadcast. With one design per env the limits genuinely
    # differ -- the sampler draws each hand's abduction range independently --
    # and normalizing every env by env 0's range would hand the policy a
    # joint_pos observation that is wrong everywhere except one env, and wrong
    # in a way nothing downstream can detect. For a single-robot scene all rows
    # are identical, so this is the same arithmetic on the same values.
    env._joint_lower_canon = limits[:, :, 0][:, env._perm_lab_to_canon]  # (N, J)
    env._joint_upper_canon = limits[:, :, 1][:, env._perm_lab_to_canon]
    hand_slice = slice(spec.num_arm_joints, spec.num_joints)
    env._joint_lower_hand = env._joint_lower_canon[:, hand_slice]
    env._joint_upper_hand = env._joint_upper_canon[:, hand_slice]
    env._joint_enabled = (
        env._joint_upper_hand - env._joint_lower_hand > 1e-6
    ).to(torch.float32)

    # Lab-order limits for action target clamping.
    env._arm_lower = limits[:, env._arm_joint_ids, 0]
    env._arm_upper = limits[:, env._arm_joint_ids, 1]
    env._hand_lower = limits[:, env._hand_joint_ids, 0]
    env._hand_upper = limits[:, env._hand_joint_ids, 1]

    # --- Action target buffers  ---
    action_space = env.cfg.action_space
    env._cur_targets = torch.zeros(env.num_envs, action_space, device=env.device)
    env._prev_targets = torch.zeros(env.num_envs, action_space, device=env.device)
    env._prev_joint_pos_canon = env.robot.data.joint_pos[
        :, env._perm_lab_to_canon
    ].clone()
    env._prev_joint_vel_canon = env.robot.data.joint_vel[
        :, env._perm_lab_to_canon
    ].clone()

    # --- Keypoint offsets in object local frame ---
    corners = torch.tensor(
        KEYPOINT_CORNERS, device=env.device, dtype=torch.float32
    )  # (4, 3)

    env._keypoint_offsets = (
        corners.unsqueeze(0)
        * (env.scene_record.object_scale * rew.object_base_size * rew.keypoint_scale * 0.5).unsqueeze(1)
    )  # (N, 4, 3)

    # Fixed-size reward keypoints follow legacy: fixed_size * keypoint_scale / 2.
    env._keypoint_offsets_fixed = (
        corners
        * (0.5 * rew.keypoint_scale * torch.tensor(rew.fixed_size, device=env.device)).unsqueeze(0)
    ).unsqueeze(0).expand(env.num_envs, -1, -1).contiguous()

    # --- Per-env DR priors (re-sampled on reset; seeded once here) ---
    lo, hi = dr.object_scale_noise_multiplier_range
    env._object_scale_multiplier = torch.empty(
        env.num_envs, 3, device=env.device
    ).uniform_(lo, hi)

    # --- Reward / termination trackers  ---
    # whether the object is lifted
    env._lifted_object = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    # the maximum distance between the object and the goal since the last goal reset
    env._closest_keypoint_max_dist = torch.full(
        (env.num_envs,), -1.0, device=env.device
    )
    # the minimum distance between a fingertip and the object since the last goal reset
    env._closest_fingertip_dist = torch.full(
        (env.num_envs, spec.num_fingertips), -1.0, device=env.device
    )
    # the number of succeesses in the current episode
    env._successes = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    # the number of consecutive steps that the object is near the goal
    env._near_goal_steps = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )

    # --- Tolerance curriculum state ---
    # resume_success_tolerance continues a curriculum mid-flight; without it the
    # curriculum restarts at its beginning, which on a resumed run trains an
    # easier task on a different reward scale than the checkpoint was written
    # under (the tolerance also scales the keypoint reward).
    from ..reward_utils.curriculum import initial_success_tolerance

    _term = env.cfg.termination
    _resume_tol = float(getattr(_term, "resume_success_tolerance", 0.0) or 0.0)
    env._current_success_tolerance: float = initial_success_tolerance(env)
    if _resume_tol > 0.0:
        print(f"[curriculum] resuming success tolerance at "
              f"{env._current_success_tolerance:.6f} "
              f"(curriculum start {_term.success_tolerance}, "
              f"floor {_term.target_success_tolerance})", flush=True)
    env._prev_episode_successes = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    env._frame_counter: int = 0
    env._last_curriculum_update: int = 0

    # --- Object lifted-reward reference z (updated on each _reset_object_pose) ---
    init_z = env.cfg.reset.table_reset_z + env.cfg.reset.table_object_z_offset
    env._object_init_z = torch.full(
        (env.num_envs,), init_z, device=env.device
    )

    # --- Per-env table surface z (randomized in _reset_table_pose) ---
    env._table_z_per_env = torch.full(
        (env.num_envs,), env.cfg.reset.table_reset_z, device=env.device
    )

    # --- DR rolling buffers ---
    env._object_state_queue = torch.zeros(
        env.num_envs,
        max(1, dr.object_state_delay_max),
        13,  # pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
        device=env.device,
    )
    env._obs_queue = torch.zeros(
        env.num_envs,
        max(1, dr.obs_delay_max),
        env.cfg.observation_space,
        device=env.device,
    )
    env._action_queue = torch.zeros(
        env.num_envs,
        max(1, dr.action_delay_max),
        action_space,
        device=env.device,
    )

    # --- Wrench DR state (Phase C/D) ---
    env._random_force_prob = sample_log_uniform(
        dr.force_prob_range, env.num_envs
    ).to(env.device)
    env._random_torque_prob = sample_log_uniform(
        dr.torque_prob_range, env.num_envs
    ).to(env.device)
    env._object_forces = torch.zeros(env.num_envs, 1, 3, device=env.device)
    env._object_torques = torch.zeros(env.num_envs, 1, 3, device=env.device)
    env._object_mass = env.object.data.default_mass[:, 0:1].to(env.device)  # (N, 1)

    # --- Step-shared caches populated by compute_intermediate_values (Phase F) ---
    env._keypoints_max_dist = torch.zeros(env.num_envs, device=env.device)
    env._curr_fingertip_distances = torch.zeros(
        env.num_envs, spec.num_fingertips, device=env.device
    )
    env._near_goal = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._is_success = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )

    # set the reward buffer to 0
    env.reward_buf = torch.zeros(env.num_envs, device=env.device)


def _randomize_robot_dof_state(env, env_ids: torch.Tensor) -> None:
    """Reset DOF state and seed previous targets from the reset pose."""
    cfg = env.cfg.reset
    default_pos = env.robot.data.default_joint_pos[env_ids]  # (n, num_dofs)
    lower = env.robot.data.joint_pos_limits[env_ids, :, 0]
    upper = env.robot.data.joint_pos_limits[env_ids, :, 1]

    reset_scale = torch.zeros_like(default_pos)
    reset_scale[:, env._arm_joint_ids] = cfg.reset_dof_pos_random_interval_arm
    reset_scale[:, env._hand_joint_ids] = cfg.reset_dof_pos_random_interval_fingers

    sampled_pos = lower + (upper - lower) * torch.rand_like(default_pos)
    joint_pos = torch.lerp(default_pos, sampled_pos, reset_scale).clamp(lower, upper)
    joint_vel = torch.empty_like(default_pos).uniform_(
        -cfg.reset_dof_vel_random_interval,
        cfg.reset_dof_vel_random_interval,
    )

    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    env._prev_targets[env_ids] = joint_pos
    env._cur_targets[env_ids] = joint_pos
    env._prev_joint_pos_canon[env_ids] = joint_pos[:, env._perm_lab_to_canon]
    env._prev_joint_vel_canon[env_ids] = joint_vel[:, env._perm_lab_to_canon]


def _reset_table_pose(env, env_ids: torch.Tensor) -> None:
    """Randomize the table's pose per env and write the new transform.

    Z position is required (the policy uses `_table_z_per_env` to set the
    object init height). XY position and yaw are gated on
    `table_reset_xy_range_m` / `table_reset_yaw_range_deg` — default ranges
    are zero so this is a no-op for runs that don't opt in.
    """
    cfg = env.cfg.reset
    n = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    dz = torch.empty(n, device=env.device).uniform_(
        -cfg.table_reset_z_range, cfg.table_reset_z_range
    )
    table_z = cfg.table_reset_z + dz
    env._table_z_per_env[env_ids] = table_z

    pos_local = torch.zeros(n, 3, device=env.device)
    pos_local[:, 2] = table_z

    # XY position noise — per-env independent uniform half-widths.
    xy_range = tuple(float(v) for v in cfg.table_reset_xy_range_m)
    if xy_range[0] > 0.0 or xy_range[1] > 0.0:
        rx = torch.empty(n, device=env.device).uniform_(-xy_range[0], xy_range[0])
        ry = torch.empty(n, device=env.device).uniform_(-xy_range[1], xy_range[1])
        pos_local[:, 0] = rx
        pos_local[:, 1] = ry

    # Yaw noise — sample uniform [-r, r] degrees, build a z-axis rotation quat.
    yaw_range_deg = float(cfg.table_reset_yaw_range_deg)
    if yaw_range_deg > 0.0:
        yaw_rad = (
            torch.empty(n, device=env.device).uniform_(-1.0, 1.0)
            * yaw_range_deg * (torch.pi / 180.0)
        )
        half = yaw_rad * 0.5
        w = torch.cos(half)
        z = torch.sin(half)
        quat = torch.stack([w, torch.zeros_like(w), torch.zeros_like(w), z], dim=-1)
    else:
        quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32
        ).unsqueeze(0).expand(n, -1)

    pose = torch.cat([pos_local + env_origins, quat], dim=-1)
    env.table.write_root_pose_to_sim(pose, env_ids=env_ids)

def _reset_object_pose(env, env_ids: torch.Tensor) -> None:
    """Reset object pose and lifted-reward reference height."""
    cfg = env.cfg.reset
    n = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    if cfg.fixed_start_pose is not None:
        fixed = torch.as_tensor(cfg.fixed_start_pose, device=env.device, dtype=torch.float32)
        pos_local = fixed[:3].unsqueeze(0).expand(n, -1)
        quat = fixed[3:].unsqueeze(0).expand(n, -1)
    else:
        noise = torch.empty(n, 3, device=env.device).uniform_(-1.0, 1.0)
        pos_local = torch.stack(
            (
                noise[:, 0] * cfg.reset_position_noise_x,
                noise[:, 1] * cfg.reset_position_noise_y,
                env._table_z_per_env[env_ids]
                + cfg.table_object_z_offset
                + noise[:, 2] * cfg.reset_position_noise_z,
            ),
            dim=-1,
        )
        quat = random_orientation(n, device=env.device)

    pose = torch.cat([pos_local + env_origins, quat], dim=-1)
    env.object.write_root_pose_to_sim(pose, env_ids=env_ids)
    env.object.write_root_velocity_to_sim(
        torch.zeros(n, 6, device=env.device), env_ids=env_ids
    )

    env._object_init_z[env_ids] = pos_local[:, 2]


def _reset_goal_pose(env, env_ids: torch.Tensor, mode: str) -> None:
    """Resample the goal pose and write it to GoalViz."""
    cfg = env.cfg.reset
    n = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    if cfg.fixed_goal_pose is not None:
        fixed = torch.as_tensor(cfg.fixed_goal_pose, device=env.device, dtype=torch.float32)
        new_pos_local = fixed[:3].unsqueeze(0).expand(n, -1)
        new_quat = fixed[3:].unsqueeze(0).expand(n, -1)
        pose = torch.cat([new_pos_local + env_origins, new_quat], dim=-1)
        env.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)
        return

    if mode == "delta":
        prev_pos_local = env.goal_viz.data.root_pos_w[env_ids] - env_origins
        prev_quat = env.goal_viz.data.root_quat_w[env_ids]
        new_pos_local, new_quat = sample_delta_goal_pose(
            prev_pos=prev_pos_local,
            prev_quat_wxyz=prev_quat,
            delta_distance=cfg.delta_goal_distance,
            delta_rotation_degrees=cfg.delta_rotation_degrees,
            mins=cfg.target_volume_mins,
            maxs=cfg.target_volume_maxs,
            scale=cfg.target_volume_region_scale,
        )
    elif mode == "absolute":
        new_pos_local, new_quat = sample_absolute_goal_pose(
            mins=cfg.target_volume_mins,
            maxs=cfg.target_volume_maxs,
            scale=cfg.target_volume_region_scale,
            n_envs=n,
            device=env.device,
        )
    else:
        raise ValueError(f"unknown goal sampling mode: {mode}")

    pose = torch.cat([new_pos_local + env_origins, new_quat], dim=-1)
    env.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)


def _clear_goal_trackers(env, env_ids: torch.Tensor) -> None:
    env._closest_keypoint_max_dist[env_ids] = -1.0
    env._closest_fingertip_dist[env_ids] = -1.0
    env._near_goal_steps[env_ids] = 0


def reset_goal_trackers(env, env_ids: torch.Tensor) -> None:
    """Clear per-goal trackers and sample the next goal."""
    _clear_goal_trackers(env, env_ids)
    _reset_goal_pose(env, env_ids, mode=env.cfg.reset.goal_sampling_type)


def reset_env_state(env, env_ids: torch.Tensor) -> None:
    """Full per-env reset after ``super()._reset_idx``."""
    n = env_ids.numel()

    _randomize_robot_dof_state(env, env_ids)
    _reset_table_pose(env, env_ids)
    _reset_object_pose(env, env_ids)
    _reset_goal_pose(env, env_ids, mode="absolute")  # full reset → always absolute

    env._prev_episode_successes[env_ids] = env._successes[env_ids]

    _clear_goal_trackers(env, env_ids)
    env._lifted_object[env_ids] = False
    env._successes[env_ids] = 0

    env._action_queue[env_ids] = 0.0
    env._obs_queue[env_ids] = 0.0
    env._object_state_queue[env_ids] = 0.0
    env._object_forces[env_ids] = 0.0
    env._object_torques[env_ids] = 0.0

    dr = env.cfg.domain_randomization
    env._random_force_prob[env_ids] = sample_log_uniform(
        dr.force_prob_range, n
    ).to(env.device)
    env._random_torque_prob[env_ids] = sample_log_uniform(
        dr.torque_prob_range, n
    ).to(env.device)
    lo, hi = dr.object_scale_noise_multiplier_range
    env._object_scale_multiplier[env_ids] = torch.empty(
        n, 3, device=env.device
    ).uniform_(lo, hi)


__all__ = [
    "allocate_state_buffers",
    "reset_env_state",
    "reset_goal_trackers",
]
