"""Rotation conventions and rigid-body mass properties.

``rotations`` owns rpy/quaternion/rot6d; ``design_space`` owns the axis-angle
construction these do not cover. The inertia half is used for authored objects
as well as links, so it takes primitives rather than a hand.
"""

from __future__ import annotations

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


# --- mass properties ---

import math


def compute_mass_and_inertia(scale, density: float):
    """Capsule-approximation for cylinders; exact for cuboids.

    ``scale`` is (lx, ly, lz) for a cuboid or (height, diameter) for a capsule.
    Returns (m, ixx, iyy, izz) with scale-axis = z (caller flips if needed).
    """
    if len(scale) == 3:
        lx, ly, lz = scale
        v = lx * ly * lz
        m = v * density
        ixx = (1 / 12) * m * (ly * ly + lz * lz)
        iyy = (1 / 12) * m * (lx * lx + lz * lz)
        izz = (1 / 12) * m * (lx * lx + ly * ly)
        return m, ixx, iyy, izz
    if len(scale) == 2:
        h, d = scale[0], scale[1]
        r = d / 2
        # Capsule mass = cylinder + two hemispheres.
        m_c = density * math.pi * r * r * h
        m_h = density * (2 / 3) * math.pi * r ** 3
        m = m_c + 2 * m_h
        # Cylinder inertia about centroid (axis = z).
        i_c_axis = 0.5 * m_c * r * r
        i_c_perp = (1 / 12) * m_c * (3 * r * r + h * h)
        # Hemisphere inertia about its own centroid.
        i_h_axis = (2 / 5) * m_h * r * r
        i_h_perp = (83 / 320) * m_h * r * r
        d_com = (h / 2) + (3 * r / 8)
        izz = i_c_axis + 2 * i_h_axis
        ixx = iyy = i_c_perp + 2 * (i_h_perp + m_h * d_com * d_com)
        return m, ixx, iyy, izz
    raise ValueError(f"Invalid scale: {scale}")


__all__ = ["compute_mass_and_inertia"]


# --- task geometry: the scene a hand is judged in ---

import numpy as np

from hand_sampler import resolve as resolve_repo_path


# Task geometry, from ResetCfg.
GOAL_VOLUME_MINS = (-0.35, -0.2, 0.6)
GOAL_VOLUME_MAXS = (0.35, 0.2, 0.95)
TABLE_Z = 0.38
TABLE_URDF = "assets/urdf/table_narrow.urdf"


def table_extents() -> tuple[float, float, float]:
    """Box dimensions read from the actual table asset.

    This used to be a hardcoded TABLE_SIZE = (1.2, 0.8), which is 2.5x too wide
    in x and 2x in y against the asset's 0.475 x 0.4 x 0.3. An oversized table
    swallows the arm's link_3 and link_4, which looks exactly like the robot
    colliding with the table when nothing is wrong with the robot.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(resolve_repo_path(TABLE_URDF)).getroot()
    for link in root.findall("link"):
        for coll in link.findall("collision"):
            box = coll.find("geometry/box")
            if box is not None:
                return tuple(float(v) for v in box.get("size").split())
    raise RuntimeError(f"no box collision geometry in {TABLE_URDF}")


def _hull_collision_scene(urdf) -> int:
    """Replace each collision mesh with its convex hull, in place.

    Isaac Lab stamps ``approximation="convexHull"`` on every mesh collider
    (verified in the baked USD: 34/34 on SHARPA, 26/26 on Allegro), so PhysX
    never resolves contacts against the triangle meshes the URDF declares -- it
    uses their hulls, with every concavity between phalanges filled in.

    Showing the declared mesh therefore overstates the fidelity of the
    simulation. Hulling the collision scene before ViserUrdf reads it means the
    "collision" view is the geometry that actually decides contacts, and it
    still follows the joint sliders because only the geometry is swapped, not
    the scene graph.

    Generated hands are unaffected -- their capsules and palm box are analytic
    primitives with approximation "None", i.e. simulated exactly as declared.
    """
    import trimesh

    scene = getattr(urdf, "collision_scene", None)
    if scene is None:
        return 0
    n = 0
    for key, geom in list(scene.geometry.items()):
        if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 3:
            try:
                scene.geometry[key] = geom.convex_hull
                n += 1
            except Exception:
                pass
    return n


__all__ = ["GOAL_VOLUME_MINS", "GOAL_VOLUME_MAXS", "TABLE_Z", "TABLE_URDF",
           "table_extents", "_hull_collision_scene"]
