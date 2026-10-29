"""Observations, in in-hand reorientation's format (pose_reaching_6d/obs_utils/observations.py): named fields
stacked in ``cfg.obs.obs_list`` order for the actor and ``cfg.obs.state_list`` order for the critic, with the same
names, frames and scalings, built with the same helpers. Joints are in canonical order, wrist (as in-hand's arm)
then hand; quaternions are xyzw in the observation; the palm frame is the canonical one at the palm centre.

Differences from in-hand: the object is the pen, its keypoints 4 points along its long axis (tip to tail, so the
keypoint fields keep in-hand's width of 12); ``keypoints_rel_goal`` is the full 6D residual (goals are poses, not
orientations); ``closest_fingertip_dist`` is per fingertip as there; and ``task_onehot`` is new.
"""
from __future__ import annotations

import torch
from isaaclab.utils.math import convert_quat, quat_mul

from isaacsimenvs.common.obs_geometry import (
    apply_local_offset, joint_link_geometry_obs, normalize_joint_pos, object_keypoints_rel_joint, rotate_into,
)
from .goals import keypoints

OBS_LIST = (
    "joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel", "prev_action_targets",
    "joint_link_bbox", "joint_lower", "joint_upper", "joint_enabled",
    "keypoints_rel_ee", "keypoints_rel_goal", "object_vel", "palm_extents", "task_onehot",
)
STATE_LIST = OBS_LIST + (
    "object_keypoints_rel_joint", "palm_keypoints", "ee_pos", "ee_rot", "ee_vel", "object_rot",
    "fingertip_pos_rel_palm", "closest_keypoint_max_dist", "closest_fingertip_dist",
    "progress", "successes", "reward",
)


def field_sizes(n_joints: int, n_hand: int, n_tips: int, n_tasks: int) -> dict[str, int]:
    """Width of every field (Kit-free: the spaces are set before the sim exists)."""
    per_joint = dict.fromkeys(("joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel",
                               "prev_action_targets"), n_joints)
    return {
        **per_joint,
        "joint_link_bbox": 12 * n_hand, "joint_origin": 3 * n_hand, "object_keypoints_rel_joint": 12 * n_hand,
        "joint_lower": n_hand, "joint_upper": n_hand, "joint_enabled": n_hand, "joint_moving": n_hand,
        "palm_keypoints": 12, "palm_extents": 3,
        "ee_pos": 3, "ee_rot": 4, "ee_vel": 6,
        "object_rot": 4, "object_vel": 6,
        # The twist pen's two objects and their goals: position (env frame) + quaternion (xyzw).
        "pen_pose": 7, "cap_pose": 7, "pen_goal_pose": 7, "cap_goal_pose": 7,
        "keypoints_rel_ee": 12, "keypoints_rel_goal": 12, "keypoints_rel_palm": 12,
        "fingertip_pos_rel_palm": 3 * n_tips,
        "closest_keypoint_max_dist": 1, "closest_fingertip_dist": n_tips,
        "progress": 1, "successes": 1, "reward": 1,
        "task_onehot": n_tasks,
    }


def width(fields, sizes: dict[str, int]) -> int:
    unknown = [f for f in fields if f not in sizes]
    if unknown:
        raise ValueError(f"unknown observation fields {unknown}; known: {sorted(sizes)}")
    return sum(sizes[f] for f in fields)


def build_observations(env) -> dict[str, torch.Tensor]:
    origins = env.scene.env_origins
    perm = env.perm_lab_to_canon
    lower, upper = env.joint_lower_canon, env.joint_upper_canon

    palm = env.robot.data.body_state_w[:, env.palm_body]
    palm_pos_w, palm_rot_body, palm_vel = palm[:, 0:3], palm[:, 3:7], palm[:, 7:13]
    centre_w = apply_local_offset(palm_pos_w, palm_rot_body, env.palm_center_offset, (env.num_envs,))
    palm_rot = quat_mul(palm_rot_body, env.palm_frame_quat)          # canonical palm frame
    ee_pos = centre_w - origins
    ones = torch.ones(env.num_envs, 1, device=env.device)

    pen_pos, pen_quat = env.pen_pose()
    barrel_pos, barrel_quat, cap_pos, cap_quat = env.pen_part_poses()
    kp = keypoints(pen_pos, pen_quat, env.pen_offsets)
    kp_goal = keypoints(env.goal_pos, env.goal_quat, env.pen_offsets)
    bbox, joint_origins, valid = joint_link_geometry_obs(env, centre_w, palm_rot, origins, ones)
    tips = env.fingertips()

    fields = {
        "joint_pos": normalize_joint_pos(env.robot.data.joint_pos[:, perm], lower, upper),
        "joint_vel": env.robot.data.joint_vel[:, perm],
        "prev_joint_pos": normalize_joint_pos(env.prev_joint_pos_canon, lower, upper),
        "prev_joint_vel": env.prev_joint_vel_canon,
        "prev_action_targets": env.targets[:, perm],
        "joint_link_bbox": bbox,
        # Masked by joint_enabled, not by the box's validity: a multi-hand scene's ghost slot reads the palm's body.
        "joint_origin": rotate_into(palm_rot, joint_origins - ee_pos.unsqueeze(1)) * env.joint_enabled.unsqueeze(-1),
        "joint_lower": env.joint_lower_hand, "joint_upper": env.joint_upper_hand,
        "joint_enabled": env.joint_enabled,
        "joint_moving": env.joint_moving,
        "object_keypoints_rel_joint": object_keypoints_rel_joint(kp, joint_origins, palm_rot, ones, valid),
        "palm_keypoints": env.palm_keypoints_canon,
        "palm_extents": env.palm_extents,
        "ee_pos": ee_pos, "ee_rot": convert_quat(palm_rot, to="xyzw"), "ee_vel": palm_vel,
        "object_rot": convert_quat(pen_quat, to="xyzw"),
        "object_vel": torch.cat([env.pen.data.root_lin_vel_w, env.pen.data.root_ang_vel_w], dim=-1),
        "pen_pose": torch.cat([barrel_pos, convert_quat(barrel_quat, to="xyzw")], dim=-1),
        "cap_pose": torch.cat([cap_pos, convert_quat(cap_quat, to="xyzw")], dim=-1),
        "pen_goal_pose": torch.cat([env.goal_pos, convert_quat(env.goal_quat, to="xyzw")], dim=-1),
        "cap_goal_pose": (torch.cat([env.cap_goal_pos, convert_quat(env.cap_goal_quat, to="xyzw")], dim=-1)
                          if env.two_part else torch.zeros(env.num_envs, 7, device=env.device)),
        "keypoints_rel_ee": rotate_into(palm_rot, kp - ee_pos.unsqueeze(1)),
        "keypoints_rel_goal": kp - kp_goal,
        "keypoints_rel_palm": kp - ee_pos.unsqueeze(1),
        "fingertip_pos_rel_palm": tips - ee_pos.unsqueeze(1),
        "closest_keypoint_max_dist": torch.nan_to_num(env.closest_goal_dist, nan=0.0).unsqueeze(-1),
        "closest_fingertip_dist": torch.nan_to_num(env.closest_tip_dist, nan=0.0),
        "progress": torch.log(env.episode_length_buf.float() / 10.0 + 1.0).unsqueeze(-1),
        "successes": torch.log(env.successes + 1.0).unsqueeze(-1),
        "reward": (env.reward_buf * 0.01).unsqueeze(-1),
        "task_onehot": env.task_onehot(),
    }
    clamp = env.cfg.obs.clamp_abs_observations

    def stack(names):
        return torch.cat([fields[f].reshape(env.num_envs, -1) for f in names], dim=-1).clamp(-clamp, clamp)

    return {"policy": stack(env.cfg.obs.obs_list), "critic": stack(env.cfg.obs.state_list)}
