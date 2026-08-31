"""Mass and inertia for the primitive shapes used by hands and objects.

Lives here rather than beside the object generator because both sides need it:
``hand_sampler.urdf`` writes inertials for capsule finger segments, and the
task's object generator writes them for cylinders and cuboids. Keeping it in
``hand_sampler`` is what lets the URDF backend stay free of any dependency on
``isaacsimenvs``.
"""

from __future__ import annotations

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
