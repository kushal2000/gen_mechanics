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

SPLAY_STEP_DEG = 30.0
SPLAY_LIMIT_DEG = 30.0
SPLAY_CHOICES: tuple[float, ...] = tuple(
    math.radians(d) for d in
    [SPLAY_LIMIT_DEG * (-1) + SPLAY_STEP_DEG * i
     for i in range(int(2 * SPLAY_LIMIT_DEG / SPLAY_STEP_DEG) + 1)]
)
"""Mount rotation about the palm normal, in the palm plane. Same axis as MCP AA.

Discretised: +-30 deg in 30 deg steps, 3 values. This was the design space's only
continuous parameter, so quantising it makes the whole space finite and
enumerable -- which is what lets the counts below be exact rather than "times a
continuum"."""

N_FINGERS_CHOICES = (2, 3)

# --- joint coupling ---------------------------------------------------------
#
# Each joint plays exactly one role:
#
#   independent   its own actuator input                     (the v1 behaviour)
#   passive       no input at all; a spring returns it home, external contact
#                 is the only thing that moves it
#   rigid         welded 1:1 to one neighbouring joint, so the pair shares a
#                 single input and no external force can separate them
#
# Rigid pairs may only join ADJACENT joint locations (MCP-PIP or PIP-DIP), but
# may cross DOF type freely -- MCP_FE with PIP_AA is legal. Same-location pairs
# (MCP_FE with MCP_AA) are NOT, since those locations are not adjacent.
#
# "1:1" is in NORMALISED RANGE, not raw angle: a joint at fraction t through its
# own travel drives its partner to fraction t of theirs. That matters because FE
# spans 130 deg and AA only 40, so an identity map would either clip FE or barely
# move AA.
RIGID_ADJACENT_ONLY = True

BASE_JOINT: "Joint" = (0, "FE")
"""MCP FE. Required to exist, and required to be ACTUATED -- it may be rigidly
coupled, but never fully passive. A finger whose base flexion is passive cannot
actively flex at all, which would hollow out the kinematic rule that put it
there."""

MAX_PASSIVE_PER_FINGER = 3
"""Cap on fully passive joints per finger, out of at most 6."""

ONE_PASSIVE_PER_LOCATION = True
"""A joint LOCATION may not have both its FE and AA passive at once.

A knuckle with both DOFs spring-loaded is a floating ball joint -- it contributes
two DOFs that nothing can command and nothing holds in any particular direction.
One passive DOF at a location is a compliant axis; two is a hole in the finger.
Locations carrying only one DOF are unaffected, so an AA-only PIP may be passive.
"""

MAX_ACTUATORS_PER_FINGER = 3
"""Cap on control inputs per finger. A finger may still carry up to 6 joints --
the extras have to earn their place by being rigidly coupled or passive.

    n_actuators = n_joints - n_rigid_pairs - n_passive

so the cap bites only at 4+ joints, and is always satisfiable: the tightest case
is 6 joints with no rigid pairs, which needs exactly 3 passive joints, and 5 are
eligible (all but the base joint)."""

PASSIVE_HOME_IS_MID_RANGE = True
"""A passive joint's spring rests at the MIDDLE of its travel, not at zero.

So passive AA sits at 0 (its range is symmetric) but passive FE sits at +55 deg,
the midpoint of [-10, 120] -- a finger with passive flexion rests pre-curled
rather than splayed flat, which is what a return spring on a real finger does.
"""

PASSIVE_FULL_DEFLECTION_TORQUE = 0.2
"""N.m to drive a passive joint to the end of its travel.

Stiffness is derived per joint as this divided by the joint's range, so FE and
AA springs feel equally stiff *as a fraction of travel* rather than per radian.
0.2 N.m is roughly 2 N at the tip of a 100 mm finger -- firm enough to hold a
pose against gravity, soft enough for contact to move it. A placeholder: nothing
here simulates, so it is metadata for whoever builds the physics.
"""


Joint = tuple[int, str]
"""(joint-location index, "FE" | "AA")."""


@dataclass(frozen=True)
class Coupling:
    """Which joints are passive, and which are welded into rigid pairs.

    Anything not named here is independent. A joint may appear at most once
    across both fields -- a joint cannot be passive *and* coupled, and cannot sit
    in two rigid pairs.
    """

    passive: tuple[Joint, ...] = ()
    rigid: tuple[tuple[Joint, Joint], ...] = ()

    @property
    def coupled_joints(self) -> tuple[Joint, ...]:
        return tuple(j for pair in self.rigid for j in pair)


@dataclass(frozen=True)
class Finger:
    face: str
    splay: float                        # rad, about the palm normal (+x), in-plane
    link_lengths: tuple[float, ...]     # metres, len 1..3, sums to TOTAL_FINGER_LENGTH
    dofs: tuple[tuple[str, ...], ...]   # per joint location, a subset of ("FE","AA")
    coupling: Coupling = Coupling()

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

        joints = set(finger_joints(self))
        named = list(self.coupling.passive) + list(self.coupling.coupled_joints)
        for j in named:
            if j not in joints:
                raise ValueError(f"coupling names {j}, which this finger lacks")
        if len(named) != len(set(named)):
            raise ValueError("a joint may have only one role (passive XOR rigid)")
        if BASE_JOINT in self.coupling.passive:
            raise ValueError("the base joint (MCP FE) may not be fully passive")
        if len(self.coupling.passive) > MAX_PASSIVE_PER_FINGER:
            raise ValueError(f"{len(self.coupling.passive)} passive joints, "
                             f"max {MAX_PASSIVE_PER_FINGER}")
        n_act = len(joints) - len(self.coupling.rigid) - len(self.coupling.passive)
        if n_act > MAX_ACTUATORS_PER_FINGER:
            raise ValueError(f"{n_act} actuators, max {MAX_ACTUATORS_PER_FINGER}; "
                             f"couple or passivate more joints")
        if ONE_PASSIVE_PER_LOCATION and _location_fully_passive(self.coupling.passive):
            raise ValueError("a joint location may not have both DOFs passive")
        legal = set(legal_rigid_pairs(self))
        for pair in self.coupling.rigid:
            if pair not in legal and tuple(reversed(pair)) not in legal:
                raise ValueError(f"rigid pair {pair} is not between adjacent locations")

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
    lens = rng.choice(link_partitions(n_links))
    dofs = rng.choice(dof_patterns(n_links))
    # Coupling first: with an actuator cap, an all-independent Finger would be
    # invalid for 4+ joints, so there is no legal intermediate to build.
    return Finger(
        face=face,
        splay=rng.choice(SPLAY_CHOICES),
        link_lengths=lens,
        dofs=dofs,
        coupling=sample_coupling_for(rng, dofs),
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


# --- coupling ---------------------------------------------------------------

def joints_of(dofs) -> tuple[Joint, ...]:
    """Every joint implied by a DOF pattern, proximal to distal, FE before AA.

    Keyed on ``dofs`` rather than a Finger so coupling can be enumerated and
    sampled BEFORE a Finger exists. With an actuator cap in force, the default
    all-independent Coupling is invalid for any finger with 4+ joints, so there
    is no valid intermediate Finger to hang these off.
    """
    return tuple((li, dof)
                 for li, d in enumerate(dofs)
                 for dof in ("FE", "AA") if dof in d)


def finger_joints(finger: Finger) -> tuple[Joint, ...]:
    return joints_of(finger.dofs)


def _location_fully_passive(passive) -> bool:
    """Does any joint location have BOTH its DOFs in this passive set?"""
    seen: dict[int, int] = {}
    for li, _ in passive:
        seen[li] = seen.get(li, 0) + 1
    return any(c >= 2 for c in seen.values())


def joint_limit(joint: Joint) -> tuple[float, float]:
    return FE_LIMIT if joint[1] == "FE" else AA_LIMIT


def passive_home(joint: Joint) -> float:
    """Rest angle of a passive joint's return spring: the middle of its travel."""
    lo, hi = joint_limit(joint)
    return 0.5 * (lo + hi) if PASSIVE_HOME_IS_MID_RANGE else 0.0


def passive_stiffness(joint: Joint) -> float:
    """N.m/rad for a passive joint's return spring. See the constant's docstring."""
    lo, hi = joint_limit(joint)
    return PASSIVE_FULL_DEFLECTION_TORQUE / (hi - lo)


def rigid_pairs_of(dofs) -> tuple[tuple[Joint, Joint], ...]:
    """Joint pairs that may be rigidly coupled: adjacent locations, any DOFs."""
    out = []
    for li in range(len(dofs) - 1):
        for a in ("FE", "AA"):
            if a not in dofs[li]:
                continue
            for b in ("FE", "AA"):
                if b in dofs[li + 1]:
                    out.append(((li, a), (li + 1, b)))
    return tuple(out)


def legal_rigid_pairs(finger: Finger) -> tuple[tuple[Joint, Joint], ...]:
    return rigid_pairs_of(finger.dofs)


def _matchings(edges: list, used: frozenset = frozenset()):
    """Every set of pairwise-disjoint edges, including the empty one."""
    yield ()
    for i, (a, b) in enumerate(edges):
        if a in used or b in used:
            continue
        for rest in _matchings(edges[i + 1:], used | {a, b}):
            yield ((a, b),) + rest


def couplings_of(dofs) -> list[Coupling]:
    """Every legal role assignment for this finger's joints.

    Exact rather than sampled, because the graphs are tiny (<=6 joints, <=8 legal
    pairs) and having the true count is what makes the design-space size a
    measured number instead of an estimate.
    """
    joints = joints_of(dofs)
    edges = list(rigid_pairs_of(dofs))
    out: list[Coupling] = []
    for m in _matchings(edges):
        matched = {j for pair in m for j in pair}
        free = [j for j in joints if j not in matched and j != BASE_JOINT]
        for mask in range(1 << len(free)):
            passive = tuple(j for k, j in enumerate(free) if (mask >> k) & 1)
            if len(passive) > MAX_PASSIVE_PER_FINGER:
                continue
            if len(joints) - len(m) - len(passive) > MAX_ACTUATORS_PER_FINGER:
                continue
            if ONE_PASSIVE_PER_LOCATION and _location_fully_passive(passive):
                continue
            out.append(Coupling(passive=passive, rigid=tuple(m)))
    return out


def enumerate_couplings(finger: Finger) -> list[Coupling]:
    return couplings_of(finger.dofs)


def sample_coupling_for(rng: random.Random, dofs) -> Coupling:
    """Draw a role assignment, reweighted away from uniform-over-assignments.

    Uniform over the enumerated set puts far too much mass on passivity: passive
    is an independent label on every unmatched joint, so the implied count is
    Binomial(k, 1/2) and concentrates near k/2. Measured that way, 40% of joints
    came out passive and 17% of fingers had no actuators at all.

    Instead the count is drawn UNIFORMLY over how many passive joints there are,
    from 0 to the cap. That encodes no prior on how much passivity a finger should
    have, rather than an accidental one. Rigid pairs are still uniform over
    matchings, which was not the problem.

    Note this deliberately differs from the distribution implied by
    :func:`enumerate_couplings`, which remains the exhaustive LEGAL set.
    """
    joints = joints_of(dofs)
    n_j = len(joints)

    # Only matchings that CAN meet the actuator cap. A matching leaving too many
    # free joints to passivate away is discarded before it is drawn, rather than
    # drawn and rejected -- rejection would quietly reweight toward matchings that
    # happen to be easy to satisfy.
    feasible = []
    for m in _matchings(list(rigid_pairs_of(dofs))):
        matched = {j for pair in m for j in pair}
        elig = [j for j in joints if j not in matched and j != BASE_JOINT]

        # Enumerate the legal passive subsets rather than drawing a size and then
        # sampling: with the one-per-location rule the legal sizes are no longer a
        # contiguous range, and rejection sampling would quietly reweight toward
        # whichever subsets are easy to hit.
        by_size: dict[int, list] = {}
        for mask in range(1 << len(elig)):
            subset = tuple(j for i, j in enumerate(elig) if (mask >> i) & 1)
            if len(subset) > MAX_PASSIVE_PER_FINGER:
                continue
            if n_j - len(m) - len(subset) > MAX_ACTUATORS_PER_FINGER:
                continue
            if ONE_PASSIVE_PER_LOCATION and _location_fully_passive(subset):
                continue
            by_size.setdefault(len(subset), []).append(subset)
        if by_size:
            feasible.append((m, by_size))

    matching, by_size = rng.choice(feasible)
    k = rng.choice(sorted(by_size))                 # uniform over available counts
    passive = rng.choice(by_size[k])
    return Coupling(passive=passive, rigid=tuple(matching))


def sample_coupling(rng: random.Random, finger: Finger) -> Coupling:
    return sample_coupling_for(rng, finger.dofs)


def joint_role(finger: Finger, joint: Joint) -> str:
    if joint in finger.coupling.passive:
        return "passive"
    if joint in finger.coupling.coupled_joints:
        return "rigid"
    return "independent"


def actuators(finger: Finger) -> tuple[tuple, ...]:
    """The finger's control inputs, in canonical order.

    One entry per input: ``("independent", joint)`` or ``("rigid", a, b)``.
    Passive joints contribute none. For a rigid pair the PROXIMAL joint is the
    driver, so the input spans that joint's range.
    """
    out = []
    for j in finger_joints(finger):
        role = joint_role(finger, j)
        if role == "independent":
            out.append(("independent", j))
        elif role == "rigid":
            for a, b in finger.coupling.rigid:
                if j == min(a, b):
                    out.append(("rigid", min(a, b), max(a, b)))
    return tuple(out)


def n_actuators(finger: Finger) -> int:
    return len(actuators(finger))


def expand_inputs(finger: Finger, inputs: dict[Joint, float] | None = None,
                  passive: dict[Joint, float] | None = None) -> dict[Joint, float]:
    """Turn actuator inputs into a per-joint angle dict for forward_kinematics.

    ``inputs`` is keyed by the driver joint of each actuator (see :func:`actuators`).
    A rigid partner is driven to the SAME FRACTION of its own travel, which is what
    "1:1 in range" means. ``passive`` optionally supplies deflections for passive
    joints, standing in for external contact; they rest at 0 otherwise.
    """
    inputs = inputs or {}
    angles: dict[Joint, float] = {
        j: (passive_home(j) if joint_role(finger, j) == "passive" else 0.0)
        for j in finger_joints(finger)
    }
    angles.update({j: v for j, v in (passive or {}).items()})

    for act in actuators(finger):
        if act[0] == "independent":
            j = act[1]
            angles[j] = inputs.get(j, 0.0)
        else:
            _, a, b = act
            va = inputs.get(a, 0.0)
            lo_a, hi_a = joint_limit(a)
            lo_b, hi_b = joint_limit(b)
            t = (va - lo_a) / (hi_a - lo_a)
            angles[a] = va
            angles[b] = lo_b + t * (hi_b - lo_b)
    return angles
