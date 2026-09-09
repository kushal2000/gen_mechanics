"""The hand design space, as a tree.

A ``Hand`` is a palm with fingers; a finger is a chain of segments; a segment is
one revolute joint plus the link that follows it. That tree is the genotype, the
kinematic structure, and the graph a per-joint policy message-passes over -- the
same object in three roles. ``grammar/DESIGN.md`` has the reasoning.

Ghosting is how a design reaches the simulator (``build.py``) but no longer how
it is represented: ``params.HandParams`` tied joint count to joint identity by
enabling a fixed ladder of slots, and a tree has no ladder.

``__post_init__`` enforces only structural invariants. Design-space BOUNDS live
in ``validate.py``, so a mutation can build a candidate and then ask whether it
is legal -- a constructor that rejected out-of-range values would force every
operator to pre-validate, which is where two copies of the rules drift apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

# --- palm -------------------------------------------------------------------
# Mutable, because mount separation is one of only two geometry parameters the
# 24k eval found to carry signal and the palm is what bounds it.

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

# Thickness is seeded and never mutated: it is the dimension geometry cares
# least about, while width and length move separation and reach directly.
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
