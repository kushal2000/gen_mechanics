"""The hand design space: what a hand is, and where its parts are.

A ``Hand`` is a palm with fingers; a finger is a chain of segments; a segment is
one revolute joint plus the link that follows it. That tree is the genotype, the
kinematic structure, and the graph a per-joint policy message-passes over -- the
same object in three roles. ``DESIGN.md`` has the reasoning.

This module is the vocabulary the rest of the package is written in, and it is
the only one with no internal imports: ``gen_init_pop`` draws from the space,
``mutate_design`` walks it, ``validate_design`` says which points are legal, and
``build`` turns a point into a robot. None of them own the noun, so it lives
here.

Ghosting is how a design reaches the simulator but no longer how it is
represented: ``params.HandParams`` tied joint count to joint identity by
enabling a fixed ladder of slots, and a tree has no ladder.

``__post_init__`` enforces only structural invariants. Design-space BOUNDS are
checked in ``validate_design.py``, so a mutation can build a candidate and then
ask whether it is legal -- a constructor that rejected out-of-range values would
force every operator to pre-validate, which is where two copies of the rules
drift apart.

The geometry half is pure numpy -- nothing here writes a file or imports a
simulator, which is what lets the validator, the operators and the viewers all
run offline. ``rotations.py`` owns rpy/quaternion conventions; this owns the
axis-angle construction those do not cover.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

# --- palm ------------------------------------------------------------------- Mutable,...

PALM_QUANTUM = 0.005
"""Grid the palm dimensions must lie on."""

PALM_STEP = 0.010
"""How far one ``perturb_palm`` moves a dimension -- twice the grid.

Separation's measured optimum band is centimetres wide, which a 5 mm step crawls
across. Seed widths and lengths are multiples of 10 mm, so this keeps a palm on
the coarser grid."""
PALM_THICKNESS_RANGE = (0.015, 0.040)   # x -- NOT MUTATED, see below
PALM_WIDTH_RANGE = (0.040, 0.100)       # y
PALM_LENGTH_RANGE = (0.040, 0.100)      # z, wrist face at z = 0

# Thickness is seeded and never mutated: it is the dimension geometry cares least about, while...
MUTABLE_PALM_DIMS: tuple[str, ...] = ("width", "length")

# --- links ------------------------------------------------------------------

LINK_QUANTUM = 0.005
CAPSULE_RADIUS = 0.010
"""Fixed, on evidence: `radius_scale` scored Spearman -0.005 across a 2x range in
the 24k eval, and every link-volume measure -0.006 to -0.018 across 7-20x."""

MIN_LINK_LENGTH = 0.015
"""The closest two joint axes can sit -- there is exactly one joint per link.

Set BELOW 2 x CAPSULE_RADIUS on purpose. Truly co-located axes need a gimbal and
are excluded (see Segment), so the nearest this space gets to a compact knuckle
is two ordinary revolutes a short spacer apart, and 15 mm is that spacer.

A link shorter than its own diameter is geometrically a SPHERE: the capsule's
cylindrical section vanishes and both joints sit inside one ball of radius
CAPSULE_RADIUS. That is a fair model of a compact knuckle housing, and the
renderers already draw it. ``urdf.py`` drops the collider for such a segment --
build.py should emit the sphere instead of nothing, or short-linked fingers
become transparent to contact."""

MAX_LINK_LENGTH = 0.080
"""Deliberately loose. The measured fingertip-reach optimum is 14.5-16 cm, so
three links at 80 mm puts it inside the space with room either side."""

# --- joints -----------------------------------------------------------------

ANGLE_QUANTUM = math.radians(15.0)
"""Grid for every angle in the genotype: joint theta and offset.

The reason is EXACT INVERSES (DESIGN.md 6), not tidiness -- continuous parameters
cannot give them, so add/remove pairs would leak on every step. It also makes a
design hashable, so fitness can be memoised."""

JOINT_LIMIT = (math.radians(-90.0), math.radians(90.0))
"""Symmetric, for every joint regardless of axis. Anatomical asymmetric ranges
stop meaning anything once the axis is a continuum: there is no principled
interpolation from a flexion range to an abduction one. Range of motion comes
from CONTACT instead -- a finger that bends backwards hits the palm and stops."""

# --- the envelope -----------------------------------------------------------

MIN_FINGERS = 2
"""A one-finger hand cannot oppose anything, so it is excluded rather than left
for selection to discover at the cost of an evaluation. The only floor on
complexity; every other pressure toward simplicity is left to evolution."""

MAX_FINGERS = 7
MAX_JOINTS_PER_FINGER = 6
"""The articulation envelope: a HARD cap, not a rail.

Batched Isaac Lab needs one Articulation view to hold every design, so all of
them must present the same joint count and ``build.py`` ghosts the difference.
Every design pays for the envelope whether it uses it or not. These two numbers
are the only place the simulator reaches back into the genotype."""

MIN_MOUNT_SEPARATION = 0.015
"""Centre-to-centre floor between mounts on DIFFERENT faces. Loose on purpose:
capsules there leave along different normals and diverge."""

MOUNT_EDGE_MARGIN = CAPSULE_RADIUS
"""How far a mount stays from its face boundary, or half the base capsule hangs
off the palm. Tight on the thin axis -- a 25 mm palm carrying a 20 mm finger
leaves 5 mm of play -- which is what a 20 mm finger on a 25 mm palm looks like.

It is also why ``mutate.move_mount`` jumps the band when crossing an edge: a
margin forbidding a mount NEAR an edge forbids one AT it."""

MIN_SAME_FACE_SEPARATION = 2.5 * CAPSULE_RADIUS
"""Floor between mounts on the SAME face, where fingers run parallel and their
base capsules overlap whenever the mounts are closer than the capsules are wide.
Capsules are tangent at 2 x CAPSULE_RADIUS; the extra 0.5 r is clearance."""

FINGER_FACES: tuple[str, ...] = ("+z", "+y", "-y")
"""The three THIN faces. The large faces (`+x`, the palm surface, and `-x`, its
back) are excluded -- a finger growing out of the gripping surface is awkward to
build and to mount an arm behind. `-z` is the wrist.

Opposition comes from `+-y` fingers curling toward `+x` to meet a `+z` finger,
measured closing to 6 mm against a 40 mm object. The three stay connected under
``move_mount``: `+-y` each border `+z`."""

GRASP_DIR = np.array([1.0, 0.0, 0.0])
"""Fingers curl toward the palm surface (+x)."""


# --- the tree ---------------------------------------------------------------

@dataclass(frozen=True)
class Joint:
    """One revolute DOF; ``theta`` and ``phi`` in radians, see kinematics.axis_of.

    theta rotates the hinge within the plane perpendicular to its link (0 flexion,
    pi/2 abduction); phi is the polar angle from the link, so pi/2 is
    perpendicular-to-bone and phi -> 0 is a roll joint.

    phi is PINNED at pi/2 -- no operator moves it and the validator requires it
    (DESIGN.md 11). It stays a field so re-enabling is one line in perturb_axis.

    ``offset`` is the joint's ZERO ANGLE: where the link sits when the actuator is
    at neutral, i.e. the angle the link is assembled at. It is structural, costs
    no motor, and shifts the joint's travel with it. A base joint's offset aims
    the whole finger -- which is what the mount used to carry as (alpha, beta) --
    and an offset further out gives the finger a resting curl, which no mount
    orientation could express.
    """

    theta: float
    phi: float = math.pi / 2
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.offset):
            raise ValueError(f"non-finite joint offset {self.offset}")
        if not math.isfinite(self.theta) or not math.isfinite(self.phi):
            raise ValueError(f"non-finite joint angles ({self.theta}, {self.phi})")


@dataclass(frozen=True)
class Segment:
    """A joint and the link distal to it. A finger is a tuple of these.

    ONE JOINT PER LINK: ``length`` is always at least MIN_LINK_LENGTH, so every
    joint sits at its own point. Zero-length segments used to express a multi-DOF
    knuckle as coincident joints; dropped because coincident axes need a gimbal
    where two axes a MIN_LINK_LENGTH spacer apart are ordinary revolutes
    in series.
    """

    joint: Joint
    length: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.length) or self.length < 0.0:
            raise ValueError(f"bad link length {self.length}")


@dataclass(frozen=True)
class Mount:
    """Where a finger attaches to the palm. Position only -- no orientation.

    ``(u, v)`` are NORMALISED face coordinates, so a palm resize carries every
    mount with it. Mutation still steps in METRES, because faces differ 2-4x in
    span and the spans shrink with the palm.

    A finger leaves along its face normal, and aiming it elsewhere is the base
    joint's ``offset`` (see Joint). The mount used to carry a pointing direction
    (alpha, beta); it was exactly reproducible by (base theta, base offset) and
    strictly less expressive, since it could only aim a whole finger and never
    give one a resting curl.
    """

    face: str
    u: float
    v: float

    def __post_init__(self) -> None:
        if self.face not in FINGER_FACES:
            raise ValueError(f"{self.face!r} is not a finger face; use {FINGER_FACES}")
        for name in ("u", "v"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"non-finite mount {name}")


@dataclass(frozen=True)
class Finger:
    mount: Mount
    segments: tuple[Segment, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("a finger needs at least one segment")

    @property
    def n_joints(self) -> int:
        return len(self.segments)

    @property
    def reach(self) -> float:
        """Fully-extended length from mount to tip."""
        return sum(s.length for s in self.segments)


@dataclass(frozen=True)
class Palm:
    thickness: float   # x
    width: float       # y
    length: float      # z

    @property
    def extents(self) -> tuple[float, float, float]:
        return (self.thickness, self.width, self.length)


@dataclass(frozen=True)
class Hand:
    palm: Palm
    fingers: tuple[Finger, ...]

    def __post_init__(self) -> None:
        if not self.fingers:
            raise ValueError("a hand needs at least one finger")

    @property
    def n_fingers(self) -> int:
        return len(self.fingers)

    @property
    def n_joints(self) -> int:
        return sum(f.n_joints for f in self.fingers)

    @property
    def n_motors(self) -> int:
        """One motor per joint -- couplings are deferred.

        Kept as its own name rather than an alias because it is what the headline
        claim is plotted against, and re-adding underactuation changes this and
        not ``n_joints``."""
        return self.n_joints


# --- complexity -------------------------------------------------------------

def complexity(hand: Hand) -> tuple[int, int]:
    """``(n_motors, n_joints)`` -- readable without touching a simulator.

    The hook the evolution loop needs to age-layer or stratify selection by
    complexity without this package owning that decision."""
    return (hand.n_motors, hand.n_joints)


# --- small helpers ----------------------------------------------------------

def with_finger(hand: Hand, i: int, finger: Finger) -> Hand:
    """Replace finger ``i``. Frozen dataclasses, so every edit rebuilds."""
    fingers = list(hand.fingers)
    fingers[i] = finger
    return replace(hand, fingers=tuple(fingers))


# --- geometry --------------------------------------------------------------- Joint axes,...

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
    """The face normal: a finger leaves perpendicular to the face it sits on.

    Aiming it elsewhere is the base joint's ``offset``, not a mount property.
    """
    return face_frame(mount.face, palm)[1]


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

    ``angles`` maps segment index to radians and is COMMANDED angle, added to
    each joint's ``offset``; missing entries are 0, so the default pose is every
    joint sitting at its own offset.
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
        R = R @ rodrigues(axis_of(seg.joint),
                          seg.joint.offset + angles.get(i, 0.0))
        nxt = p + R[:, 0] * seg.length
        if seg.length > _EPS:
            capsules.append((p.copy(), nxt.copy(), CAPSULE_RADIUS, i))
        p = nxt

    joints.append(p.copy())
    return joints, capsules


def joint_axes(finger: Finger, palm: Palm,
               angles: dict[int, float] | None = None) -> list[np.ndarray]:
    """Each joint's hinge axis as a unit vector in the palm frame.

    ``axis_of`` gives the axis in the joint's own frame; this carries it out to
    the palm frame by the same chain ``forward_kinematics`` walks. A rotation
    leaves its own axis fixed, so it makes no difference whether the joint's own
    offset has been applied yet -- but the joints PROXIMAL to it move the axis,
    which is why this takes the pose.
    """
    angles = angles or {}
    _, R = mount_frame(finger.mount, palm)
    out: list[np.ndarray] = []
    for i, seg in enumerate(finger.segments):
        a = axis_of(seg.joint)
        out.append(R @ a)
        R = R @ rodrigues(a, seg.joint.offset + angles.get(i, 0.0))
    return out


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
        p0, R = mount_frame(f.mount, hand.palm)
        seg = f.segments[0]
        R = R @ rodrigues(axis_of(seg.joint), seg.joint.offset)
        out.append((p0, p0 + R[:, 0] * seg.length))
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
