"""A minimal hand design space.

Rewritten from gen_mechanics' ``hand_sampler/params.py`` (branch
2026-08-18-analyze_embodiments), stripped to a small, fully-enumerable space.

What changed from the original sampler
--------------------------------------
* palm is FIXED (was: 3 sampled extents)
* finger bases are FIXED at the centres of the 3 non-wrist thin faces
  (was: continuous (u, v) placement on 3 faces, plus a wrist keep-out)
* 2-3 fingers (was: 2-5, with 5 ghosted slots)
* joint locations are MCP/PIP/DIP only -- no CMC (was: CMC/MCP/PIP/DIP)
* AA is available at EVERY location (was: MCP_AA only, and CMC_AA)
* total finger length is CONSTANT, split into 1 cm quanta (was: 4 independent
  continuous lengths, so total reach varied 111-260 mm)
* joint limits are fixed (was: MCP_AA half-range sampled 2-25 deg)
* radii fixed (was: one per-finger scale over a fixed tier table)
* no ghosting: a design is exactly the joints it has

Conventions
-----------
Palm frame: origin at the CENTRE OF THE WRIST FACE, so an arm would attach at
the origin and the palm occupies z in [0, LENGTH].

  +x  palm surface   (the side fingers close toward)   <- the two LARGE faces
  -x  back of hand
  +-y sides                                            <- thin, host fingers
  +z  fingertip end                                    <- thin, hosts a finger
  -z  WRIST                                            <- thin, arm attaches

Per finger, local frame: the link runs along local +x, FE rotates about local
+z, AA about local +y. The two are perpendicular by construction.

Exactly ONE mounting angle is sampled: ``splay``, a rotation about the palm
normal (+x). It rotates the finger WITHIN the palm plane, so it changes which
direction the finger sticks out without ever tilting it out of plane. At zero
flexion every finger therefore lies flat in the palm's mid-plane (x = 0),
parallel to the two large faces.

That works because all three finger faces (+z, +y, -y) have normals lying in the
palm plane already, so a rotation about +x maps the plane to itself.

``splay`` shares its axis with the base MCP AA joint -- both rotate about the
palm normal. Mounting splay is the fixed part of that same motion, AA the
actuated part. Flexion is the only thing that lifts a finger out of the palm
plane, which is what makes it flexion.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import product

import numpy as np

# --- palm (fixed) -----------------------------------------------------------
PALM_THICKNESS = 0.025   # x, thin -- human palm is ~20-30 mm
PALM_WIDTH = 0.060       # y
PALM_LENGTH = 0.060      # z, wrist face at z=0
PALM_EXTENTS = (PALM_THICKNESS, PALM_WIDTH, PALM_LENGTH)

FINGER_FACES: tuple[str, ...] = ("+z", "+y", "-y")
"""The 3 thin faces that host fingers. -z is the wrist; +-x are the large faces."""

GRASP_DIR = np.array([1.0, 0.0, 0.0])
"""Fingers curl toward the palm surface. Flip to (-1,0,0) to close the other way."""

# --- finger ----------------------------------------------------------------
TOTAL_FINGER_LENGTH = 0.100
LINK_QUANTUM = 0.010
MIN_LINK_LENGTH = 0.030
MAX_LINKS = 3
CAPSULE_RADIUS = 0.010

JOINT_LOCATIONS: tuple[str, ...] = ("MCP", "PIP", "DIP")

FE_LIMIT = (math.radians(-10.0), math.radians(120.0))
AA_LIMIT = (math.radians(-20.0), math.radians(20.0))

SPLAY_RANGE = (math.radians(-45.0), math.radians(45.0))
"""Mount rotation about the palm normal, in the palm plane. Same axis as MCP AA."""

N_FINGERS_CHOICES = (2, 3)


@dataclass(frozen=True)
class Finger:
    face: str
    splay: float                        # rad, about the palm normal (+x), in-plane
    link_lengths: tuple[float, ...]     # metres, len 1..3, sums to TOTAL_FINGER_LENGTH
    dofs: tuple[tuple[str, ...], ...]   # per joint location, a subset of ("FE","AA")

    def __post_init__(self) -> None:
        n = len(self.link_lengths)
        if not 1 <= n <= MAX_LINKS:
            raise ValueError(f"{n} links, expected 1..{MAX_LINKS}")
        if len(self.dofs) != n:
            raise ValueError(f"{len(self.dofs)} joint sets for {n} links")
        if abs(sum(self.link_lengths) - TOTAL_FINGER_LENGTH) > 1e-9:
            raise ValueError(f"lengths sum to {sum(self.link_lengths)}, "
                             f"expected {TOTAL_FINGER_LENGTH}")
        if min(self.link_lengths) < MIN_LINK_LENGTH - 1e-9:
            raise ValueError(f"link below {MIN_LINK_LENGTH} m")
        if "FE" not in self.dofs[0]:
            raise ValueError("the base (MCP) joint must have FE")
        for d in self.dofs:
            if not d or not set(d) <= {"FE", "AA"}:
                raise ValueError(f"bad dof set {d}")

    @property
    def n_joints(self) -> int:
        return sum(len(d) for d in self.dofs)


@dataclass(frozen=True)
class Hand:
    fingers: tuple[Finger, ...]

    @property
    def n_joints(self) -> int:
        return sum(f.n_joints for f in self.fingers)


# --- the discrete part, enumerable -----------------------------------------

def link_partitions(n_links: int) -> list[tuple[float, ...]]:
    """Every way to split TOTAL_FINGER_LENGTH into n_links quantised links."""
    total = round(TOTAL_FINGER_LENGTH / LINK_QUANTUM)
    lo = round(MIN_LINK_LENGTH / LINK_QUANTUM)
    out = []
    for combo in product(range(lo, total + 1), repeat=n_links):
        if sum(combo) == total:
            out.append(tuple(c * LINK_QUANTUM for c in combo))
    return out


def dof_patterns(n_links: int) -> list[tuple[tuple[str, ...], ...]]:
    """Every legal per-location DOF assignment. Base must include FE."""
    options = [("FE",), ("AA",), ("FE", "AA")]
    base = [o for o in options if "FE" in o]
    out = []
    for combo in product(base, *([options] * (n_links - 1))):
        out.append(tuple(combo))
    return out


def enumerate_fingers() -> list[tuple[tuple[float, ...], tuple[tuple[str, ...], ...]]]:
    """All (lengths, dofs) pairs -- the finger design space minus splay."""
    return [(lens, dofs)
            for n in range(1, MAX_LINKS + 1)
            for lens in link_partitions(n)
            for dofs in dof_patterns(n)]


# --- sampling ---------------------------------------------------------------

def sample_finger(rng: random.Random, face: str) -> Finger:
    n_links = rng.randint(1, MAX_LINKS)
    return Finger(
        face=face,
        splay=rng.uniform(*SPLAY_RANGE),
        link_lengths=rng.choice(link_partitions(n_links)),
        dofs=rng.choice(dof_patterns(n_links)),
    )


def sample_hand(rng: random.Random, n_fingers: int | None = None) -> Hand:
    if n_fingers is None:
        n_fingers = rng.choice(N_FINGERS_CHOICES)
    faces = rng.sample(FINGER_FACES, n_fingers)
    return Hand(fingers=tuple(sample_finger(rng, f) for f in sorted(faces)))


def sample_population(seed: int, count: int) -> list[Hand]:
    rng = random.Random(seed)
    return [sample_hand(rng) for _ in range(count)]


# --- geometry ---------------------------------------------------------------

def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues."""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def face_origin_normal(face: str) -> tuple[np.ndarray, np.ndarray]:
    """Centre and outward normal of a thin face, in the palm frame."""
    hx, hy, hz = PALM_THICKNESS / 2, PALM_WIDTH / 2, PALM_LENGTH / 2
    table = {
        "+z": (np.array([0.0, 0.0, PALM_LENGTH]), np.array([0.0, 0.0, 1.0])),
        "+y": (np.array([0.0, hy, hz]), np.array([0.0, 1.0, 0.0])),
        "-y": (np.array([0.0, -hy, hz]), np.array([0.0, -1.0, 0.0])),
    }
    if face not in table:
        raise ValueError(f"{face!r} is not a finger face; use {FINGER_FACES}")
    return table[face]


def mount_frame(finger: Finger) -> tuple[np.ndarray, np.ndarray]:
    """Base position and orientation of a finger, in the palm frame.

    Columns of R are (finger axis, AA axis, FE axis).

    The finger axis is the face normal rotated by ``splay`` about the palm normal,
    which keeps it in the palm plane. FE is then the in-plane perpendicular, so
    positive flexion lifts the tip toward the palm surface; AA comes out along the
    palm normal itself, i.e. the same axis ``splay`` turns about.

    No degeneracy is possible here: the face normals are perpendicular to the palm
    normal and a rotation about that normal preserves the property, so the finger
    axis can never line up with GRASP_DIR.
    """
    origin, n = face_origin_normal(finger.face)

    axis = _rot(GRASP_DIR, finger.splay) @ n
    axis /= np.linalg.norm(axis)

    fe = np.cross(axis, GRASP_DIR)
    fe /= np.linalg.norm(fe)
    aa = np.cross(fe, axis)          # == GRASP_DIR, up to numerical noise

    return origin, np.column_stack([axis, aa, fe])


def forward_kinematics(finger: Finger, angles: dict[tuple[int, str], float] | None = None):
    """Joint centres and capsule segments for one finger, in the palm frame.

    ``angles`` maps (location_index, "FE"|"AA") to radians; missing entries are 0.
    Returns (joint_positions, segments) with segments as (start, end, radius).
    """
    angles = angles or {}
    p, R = mount_frame(finger)
    joints, segments = [], []

    for i, length in enumerate(finger.link_lengths):
        joints.append(p.copy())
        # FE about local +z, then AA about local +y -- coincident at one location.
        if "FE" in finger.dofs[i]:
            R = R @ _rot(np.array([0.0, 0.0, 1.0]), angles.get((i, "FE"), 0.0))
        if "AA" in finger.dofs[i]:
            R = R @ _rot(np.array([0.0, 1.0, 0.0]), angles.get((i, "AA"), 0.0))
        nxt = p + R[:, 0] * length
        segments.append((p.copy(), nxt.copy(), CAPSULE_RADIUS))
        p = nxt

    joints.append(p.copy())   # fingertip
    return joints, segments


def flexed(finger: Finger, frac: float) -> dict[tuple[int, str], float]:
    """All FE joints at ``frac`` of their range; AA left at zero."""
    lo, hi = FE_LIMIT
    return {(i, "FE"): lo + frac * (hi - lo)
            for i, d in enumerate(finger.dofs) if "FE" in d}
