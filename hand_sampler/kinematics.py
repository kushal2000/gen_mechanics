"""Geometry of a hand tree: joint axes, mount frames, forward kinematics.

Pure numpy -- nothing here writes a file or imports a simulator, which is what
lets the validator, the operators and the viewers all run offline.
``rotations.py`` owns rpy/quaternion conventions; this owns the axis-angle
construction those do not cover.
"""

from __future__ import annotations

import math

import numpy as np

from hand_sampler.genotype import (
    CAPSULE_RADIUS,
    FINGER_FACES,
    MOUNT_EDGE_MARGIN,
    GRASP_DIR,
    Finger,
    Hand,
    Joint,
    Mount,
    Palm,
)

_EPS = 1e-9


def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation about ``axis`` by ``angle``. Axis need not be normalised."""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0.0, -a[2], a[1]],
                  [a[2], 0.0, -a[0]],
                  [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


# --- joint axes -------------------------------------------------------------

def axis_of(joint: Joint) -> np.ndarray:
    """The hinge axis in the joint's own frame, where the link runs along +x.

        axis(theta, phi) = [cos phi, sin phi sin theta, sin phi cos theta]

    (0, pi/2) is +z (flexion), (pi/2, pi/2) is +y (abduction), phi -> 0 collapses
    onto the link itself. theta needs only [0, pi) and phi only (0, pi/2]: the
    redundant halves would give a hand two spellings and break design identity.
    """
    ct, st = math.cos(joint.theta), math.sin(joint.theta)
    cp, sp = math.cos(joint.phi), math.sin(joint.phi)
    return np.array([cp, sp * st, sp * ct])


# --- palm faces -------------------------------------------------------------

def face_frame(face: str, palm: Palm) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                               np.ndarray, float, float]:
    """``(centre, normal, t_u, t_v, span_u, span_v)`` for a palm face.

    Palm frame: origin at the centre of the WRIST face, so the palm occupies
    z in [0, length] and an arm attaches at the origin. A normalised mount (u, v)
    places at ``centre + (u - 0.5) span_u t_u + (v - 0.5) span_v t_v``.
    """
    t, w, l = palm.thickness, palm.width, palm.length
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])
    table = {
        "+y": (np.array([0.0, w / 2, l / 2]), y, x, z, t, l),
        "-y": (np.array([0.0, -w / 2, l / 2]), -y, x, z, t, l),
        "+z": (np.array([0.0, 0.0, l]), z, x, y, t, w),
    }
    if face not in table:
        raise ValueError(f"{face!r} is not a finger face")
    return table[face]


def face_from_normal(n: np.ndarray) -> str | None:
    """Which finger face has this outward normal, if any.

    On an axis-aligned box a face's tangents are exactly its neighbours' normals,
    which is what lets ``mutate.move_mount`` walk between faces without a
    hand-written cube net. None means no finger mounts that way.
    """
    axis = int(np.argmax(np.abs(n)))
    sign = "+" if n[axis] > 0 else "-"
    face = f"{sign}{'xyz'[axis]}"
    return face if face in FINGER_FACES else None


def mount_uv_bounds(face: str, palm: Palm) -> tuple[float, float, float, float]:
    """``(u_lo, u_hi, v_lo, v_hi)`` -- the normalised box a mount may occupy.

    MOUNT_EDGE_MARGIN converted per face, since a face's two axes have different
    spans. A face narrower than twice the margin centres the mount instead: the
    finger overhangs either way, and centring overhangs symmetrically.
    """
    _, _, _, _, span_u, span_v = face_frame(face, palm)
    m = MOUNT_EDGE_MARGIN
    lo_u, hi_u = ((m / span_u, 1.0 - m / span_u) if span_u > 2 * m else (0.5, 0.5))
    lo_v, hi_v = ((m / span_v, 1.0 - m / span_v) if span_v > 2 * m else (0.5, 0.5))
    return lo_u, hi_u, lo_v, hi_v


def mount_position(mount: Mount, palm: Palm) -> np.ndarray:
    centre, _, t_u, t_v, span_u, span_v = face_frame(mount.face, palm)
    return (centre
            + (mount.u - 0.5) * span_u * t_u
            + (mount.v - 0.5) * span_v * t_v)


def mount_direction(mount: Mount, palm: Palm) -> np.ndarray:
    """Which way the finger points, as a unit vector in the palm frame.

    ``(alpha, beta)`` are polar about the face normal, so beta names nothing at
    alpha = 0 -- which is why ``mutate.perturb_direction`` jitters the direction
    in the tangent plane rather than stepping the two independently.
    """
    _, n, t_u, t_v, _, _ = face_frame(mount.face, palm)
    tangential = math.cos(mount.beta) * t_u + math.sin(mount.beta) * t_v
    d = math.cos(mount.alpha) * n + math.sin(mount.alpha) * tangential
    return d / np.linalg.norm(d)


def _frame_from_axis(axis: np.ndarray) -> np.ndarray:
    """Orthonormal frame with ``axis`` as its first column.

    The remaining rotation about ``axis`` is gauge -- exactly what a mount roll
    would have carried, absorbed into each joint's theta -- so it only has to be
    deterministic. It is still chosen to be meaningful: local +z is perpendicular
    to both the finger and GRASP_DIR, so theta = 0 curls the tip toward the palm.
    That degenerates when a finger points along GRASP_DIR, which a mount tilt can
    reach, so the fallback picks the world axis least aligned with the finger.
    """
    a = axis / np.linalg.norm(axis)
    ref = GRASP_DIR
    if np.linalg.norm(np.cross(a, ref)) < 1e-6:
        ref = np.eye(3)[int(np.argmin(np.abs(a)))]
    fe = np.cross(a, ref)
    fe /= np.linalg.norm(fe)
    aa = np.cross(fe, a)
    return np.column_stack([a, aa, fe])


def mount_frame(mount: Mount, palm: Palm) -> tuple[np.ndarray, np.ndarray]:
    """``(position, R)`` for a finger's base, in the palm frame.

    Columns of R are (finger axis, local +y, local +z). The link runs along the
    first; a joint at theta = 0 rotates about the third.
    """
    return mount_position(mount, palm), _frame_from_axis(mount_direction(mount, palm))


# --- forward kinematics -----------------------------------------------------

def forward_kinematics(finger: Finger, palm: Palm,
                       angles: dict[int, float] | None = None,
                       ) -> tuple[list[np.ndarray], list[tuple]]:
    """``(joint_positions, capsules)`` for one finger, in the palm frame.

    ``angles`` maps segment index to radians; missing entries are 0.
    ``joint_positions`` has one entry per segment plus the fingertip.

    Each capsule is ``(start, end, radius, segment_index)``. The index is carried
    even though it currently equals the capsule's own position, because
    ``build.py`` will skip geometry for ghosted joints -- and a caller zipping
    them positionally would then read the wrong joint SILENTLY.
    """
    angles = angles or {}
    p, R = mount_frame(finger.mount, palm)
    joints: list[np.ndarray] = []
    capsules: list[tuple] = []

    for i, seg in enumerate(finger.segments):
        joints.append(p.copy())
        R = R @ rodrigues(axis_of(seg.joint), angles.get(i, 0.0))
        nxt = p + R[:, 0] * seg.length
        if seg.length > _EPS:
            capsules.append((p.copy(), nxt.copy(), CAPSULE_RADIUS, i))
        p = nxt

    joints.append(p.copy())
    return joints, capsules


def fingertip(finger: Finger, palm: Palm,
              angles: dict[int, float] | None = None) -> np.ndarray:
    return forward_kinematics(finger, palm, angles)[0][-1]


# --- the two measures that carry signal -------------------------------------

def segment_distance(p0: np.ndarray, p1: np.ndarray,
                     q0: np.ndarray, q1: np.ndarray) -> float:
    """Closest distance between two 3-D line segments, closed form.

    Two capsules of radius r intersect exactly when this drops below 2r, so this
    is the whole of a capsule-capsule test -- no mesh, no solver.

    THE CLAMPING IS NOT INDEPENDENT. Solving the unconstrained problem and
    clipping each parameter into [0, 1] separately does not give the closest
    pair: once one is clamped the other must be re-solved against it. Doing that
    wrong overestimates, which in a clearance check reports parts as clear when
    they overlap. Parameters are carried as numerator/denominator pairs so a
    clamp applies before dividing (Ericson, *Real-Time Collision Detection*).
    """
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a, b, c = float(u @ u), float(u @ v), float(v @ v)
    d, e = float(u @ w), float(v @ w)
    det = a * c - b * b
    eps = 1e-12

    if det < eps:                                  # parallel or degenerate
        s_num, s_den, t_num, t_den = 0.0, 1.0, e, (c if c > eps else 1.0)
    else:
        s_den = t_den = det
        s_num, t_num = b * e - c * d, a * e - b * d
        if s_num < 0.0:
            s_num, t_num, t_den = 0.0, e, (c if c > eps else 1.0)
        elif s_num > s_den:
            s_num, t_num, t_den = s_den, e + b, (c if c > eps else 1.0)

    if t_num < 0.0:                                # re-solve s against t = 0
        t_num = 0.0
        if -d < 0.0:
            s_num, s_den = 0.0, 1.0
        elif -d > a:
            s_num, s_den = 1.0, 1.0
        else:
            s_num, s_den = -d, (a if a > eps else 1.0)
    elif t_num > t_den:                            # re-solve s against t = 1
        t_num = t_den
        if (-d + b) < 0.0:
            s_num, s_den = 0.0, 1.0
        elif (-d + b) > a:
            s_num, s_den = 1.0, 1.0
        else:
            s_num, s_den = -d + b, (a if a > eps else 1.0)

    sc = 0.0 if abs(s_num) < eps else s_num / s_den
    tc = 0.0 if abs(t_num) < eps else t_num / t_den
    return float(np.linalg.norm(w + sc * u - tc * v))


def base_capsules(hand: Hand) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each finger's proximal link at the rest pose, as a core segment.

    Computed directly rather than through ``forward_kinematics``: at rest the
    first link is just mount position plus mount direction times length. Running
    the full chain for its first element made the validator 10x slower, and the
    validator runs on every mutation.
    """
    out = []
    for f in hand.fingers:
        p0 = mount_position(f.mount, hand.palm)
        p1 = p0 + mount_direction(f.mount, hand.palm) * f.segments[0].length
        out.append((p0, p1))
    return out


def mount_separations(hand: Hand) -> list[float]:
    """Pairwise distances between finger mounts, in metres.

    First-class because the 24k eval found this one of only two geometry
    parameters that predicted performance -- an inverted U peaking at 4-5 cm
    against a 4 cm object. The other is reach, and the two were independent.
    """
    pos = [mount_position(f.mount, hand.palm) for f in hand.fingers]
    return [float(np.linalg.norm(pos[i] - pos[j]))
            for i in range(len(pos)) for j in range(i + 1, len(pos))]
