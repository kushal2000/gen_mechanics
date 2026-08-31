"""Rotation conversions, on scipy.

One convention, stated once: URDF roll-pitch-yaw is ``R = Rz(yaw) @ Ry(pitch)
@ Rx(roll)``, which is scipy's *extrinsic* ``"xyz"`` (lowercase; uppercase would
be intrinsic and is the easy mistake). Quaternions are (w, x, y, z) here and
(x, y, z, w) in scipy, so every conversion reorders explicitly.

These were five hand-rolled trig blocks across four files, one of them
duplicated verbatim. Each was correct -- checked against scipy to 1e-16 -- but a
wrong one would never raise: it would silently misplace a mount or corrupt the
morphology descriptor the policy conditions on.

Lives in hand_sampler because it is the base of the dependency chain, so the
task package can use it too.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rpy_to_mat(rpy) -> np.ndarray:
    """URDF RPY -> 3x3 rotation matrix."""
    return Rotation.from_euler("xyz", np.asarray(rpy, dtype=float)).as_matrix()


def mat_to_rpy(mat) -> tuple[float, float, float]:
    """3x3 rotation matrix -> URDF RPY.

    At gimbal lock scipy resolves the free angle differently from a hand-rolled
    branch, but both decompose to the same rotation, which is all any caller
    round-trips through.
    """
    r, p, y = Rotation.from_matrix(np.asarray(mat, dtype=float)[:3, :3]).as_euler("xyz")
    return float(r), float(p), float(y)


def rpy_to_quat_wxyz(rpy) -> tuple[float, float, float, float]:
    """URDF RPY -> (w, x, y, z)."""
    x, y, z, w = Rotation.from_euler("xyz", np.asarray(rpy, dtype=float)).as_quat()
    return float(w), float(x), float(y), float(z)


def rpy_to_rot6d(rpy) -> list[float]:
    """First two columns of the rotation matrix for an RPY triple.

    The 6D representation rather than Euler angles or a quaternion: it is
    continuous over SO(3), so nearby orientations map to nearby vectors. Euler
    angles wrap and quaternions double-cover, and both put a discontinuity
    somewhere in a space the sampler draws uniformly over (mount roll is
    U(0, 2*pi), so it visits every wrap point).
    """
    m = rpy_to_mat(rpy)
    return [float(v) for v in m[:, 0]] + [float(v) for v in m[:, 1]]


def mat_to_pos_quat(m):
    """4x4 transform -> (translation, (w, x, y, z))."""
    m = np.asarray(m, dtype=float)
    x, y, z, w = Rotation.from_matrix(m[:3, :3]).as_quat()
    return tuple(float(v) for v in m[:3, 3]), (float(w), float(x), float(y), float(z))


__all__ = ["rpy_to_mat", "mat_to_rpy", "rpy_to_quat_wxyz", "rpy_to_rot6d",
           "mat_to_pos_quat"]
