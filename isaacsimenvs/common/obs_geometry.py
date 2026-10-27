"""Batched geometry for observations, shared by every hand task (in-hand olympics, pose reaching, in-hand
reorientation): the palm frame, link boxes and object keypoints relative to each joint, world vectors in the palm's
orientation. Moved from pose_reaching_6d/obs_utils/observations.py (27 Oct).

``joint_link_geometry_obs`` reads what the env must provide: ``robot.data.body_state_w``, ``_joint_link_body_ids``,
``_joint_link_bbox_local`` (N, J, 4, 3) and ``_joint_geometry_valid`` (N, J).
"""
from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply


def apply_local_offset(
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


def normalize_joint_pos(raw, lower, upper) -> torch.Tensor:
    # A slot with no range -- a multi-hand scene's ghost slot, limits (0, 0) -- reads 0, not 0/0.
    # (Generated designs' ghosts carry a 1e-8 range for the same reason.)
    rng = upper - lower
    ok = rng > 1e-6
    return torch.where(ok, 2.0 * (raw - lower) / torch.where(ok, rng, 1.0) - 1.0, torch.zeros_like(raw))


def quat_apply_broadcast(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Quaternion rotation with broadcast views, avoiding flattened copies."""
    xyz = quat[..., 1:].expand_as(vec)
    t = torch.cross(xyz, vec, dim=-1) * 2.0
    return vec + quat[..., 0:1] * t + torch.cross(xyz, t, dim=-1)


def joint_link_geometry_obs(env, origin_w, palm_rot, env_origins, scale):
    """Batched link boxes relative to ``origin_w`` in the palm body's orientation,
    divided by ``scale``, plus joint origins."""
    state = env.robot.data.body_state_w[:, env._joint_link_body_ids, :]
    body_pos = state[:, :, 0:3]
    body_rot = state[:, :, 3:7]
    n, joints, points, _ = env._joint_link_bbox_local.shape

    box_world = body_pos.unsqueeze(2) + quat_apply_broadcast(
        body_rot.unsqueeze(2), env._joint_link_bbox_local
    )

    palm_inv = palm_rot.clone()
    palm_inv[:, 1:] *= -1.0
    box_palm = quat_apply_broadcast(
        palm_inv[:, None, None, :],
        box_world - origin_w[:, None, None, :],
    )
    box_palm = box_palm / scale[:, None, None, :]

    valid = env._joint_geometry_valid
    box_palm = box_palm * valid[:, :, None, None].to(box_palm.dtype)
    joint_origins = body_pos - env_origins.unsqueeze(1)
    return box_palm, joint_origins, valid


def object_keypoints_rel_joint(
    object_keypoints, joint_origins, palm_rot, scale, valid
) -> torch.Tensor:
    """Object keypoints from every joint, in the palm body's orientation, / scale."""
    rel_world = object_keypoints[:, None, :, :] - joint_origins[:, :, None, :]
    palm_inv = palm_rot.clone()
    palm_inv[:, 1:] *= -1.0
    rel_palm = quat_apply_broadcast(palm_inv[:, None, None, :], rel_world)
    rel_palm = rel_palm / scale[:, None, None, :]
    return rel_palm * valid[:, :, None, None].to(rel_palm.dtype)


def rotate_into(palm_rot, vectors_w):
    """World vectors ``(N, K, 3)`` expressed in the palm body's orientation."""
    palm_inv = palm_rot.clone()
    palm_inv[:, 1:] *= -1.0
    return quat_apply_broadcast(palm_inv[:, None, :], vectors_w)


__all__ = ["apply_local_offset", "joint_link_geometry_obs", "normalize_joint_pos", "object_keypoints_rel_joint",
           "quat_apply_broadcast", "rotate_into"]
