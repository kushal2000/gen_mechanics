"""Forward kinematics: local product of exponentials over the model tree.

Convention (fixed, do not deviate): ``Joint.origin`` places the joint frame in the
parent body frame; the child body frame equals the joint frame at q=0; the axis
passes through the joint-frame origin. rpy is fixed-axis XYZ: R = Rz(yaw)*Ry(pitch)*Rx(roll).
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from .kinematics import KinematicModel, Pose


def rpy_to_matrix(rpy) -> np.ndarray:
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def pose_to_matrix(pose: Pose) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(pose.rpy)
    T[:3, 3] = [float(v) for v in pose.xyz]
    return T


def rodrigues(axis, angle: float) -> np.ndarray:
    a = np.asarray([float(v) for v in axis], dtype=float)
    n = np.linalg.norm(a)
    if n == 0.0:
        raise ValueError("zero-length rotation axis")
    a = a / n
    x, y, z = a
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    theta = float(angle)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _joint_motion_matrix(joint, qval: float) -> np.ndarray:
    T = np.eye(4)
    if joint.type == "fixed":
        return T
    if joint.type in ("revolute", "continuous"):
        T[:3, :3] = rodrigues(joint.axis, qval)
        return T
    if joint.type == "prismatic":
        a = np.asarray([float(v) for v in joint.axis], dtype=float)
        n = np.linalg.norm(a)
        if n == 0.0:
            raise ValueError(f"joint {joint.name!r} has zero-length axis")
        a = a / n
        T[:3, 3] = a * float(qval)
        return T
    raise ValueError(f"joint {joint.name!r} has unsupported type {joint.type!r}")


def forward_kinematics(model: KinematicModel, q: Mapping[str, float]) -> Dict[str, np.ndarray]:
    """Return 4x4 root-frame transforms for every body and every named frame.

    Missing entries in ``q`` default to 0. Keys of ``q`` that are not joint names
    raise ``ValueError``.
    """
    joint_names = {j.name for j in model.joints}
    unknown = set(q) - joint_names
    if unknown:
        raise ValueError(f"q contains unknown joint names: {sorted(unknown)}")

    children = {}
    for j in model.joints:
        children.setdefault(j.parent, []).append(j)

    transforms: Dict[str, np.ndarray] = {model.root: np.eye(4)}
    stack = [model.root]
    while stack:
        parent_body = stack.pop()
        T_parent = transforms[parent_body]
        for j in children.get(parent_body, []):
            qval = float(q.get(j.name, 0.0))
            origin_T = pose_to_matrix(j.origin)
            motion_T = _joint_motion_matrix(j, qval)
            transforms[j.child] = T_parent @ origin_T @ motion_T
            stack.append(j.child)

    for f in model.frames:
        transforms[f.name] = transforms[f.body] @ pose_to_matrix(f.pose)

    return transforms


def rotation_error(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices, via atan2 (never arccos)."""
    E = np.asarray(Ra)[:3, :3].T @ np.asarray(Rb)[:3, :3]
    skew = E - E.T
    vee = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    sin_term = np.linalg.norm(vee) / 2.0
    cos_term = (np.trace(E) - 1.0) / 2.0
    return float(np.arctan2(sin_term, cos_term))


def position_error(Ta: np.ndarray, Tb: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(Ta)[:3, 3] - np.asarray(Tb)[:3, 3]))
