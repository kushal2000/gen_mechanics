"""The hand design space: what a hand is, and where its parts are."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, replace

import numpy as np

from hand_sampler import resolve
from hand_sampler import robot_param_constants as rpc

# --- palm ------------------------------------------------------------------- Mutable,...

PALM_QUANTUM = 0.005
"""Grid the finger origins lie on: BOTH of a mount's coordinates, and the step
``move_mount`` takes.

One grid in both directions, so a site 70 mm out is placed as precisely as one
20 mm out. It was a radius on this grid and a bearing on ANGLE_QUANTUM, which
made the far ring three and a half times coarser than the near one.
"""

PALM_MIN_RADIUS = 0.020
"""The palm's own disc: the smallest it can ever be, around its centre.

A palm is not a stored shape any more. It is the convex hull of THIS DISC and
of where the fingers START, so a hand with its fingers drawn in is just the
disc, and spreading them pulls the outline out to meet them. The disc is in the
hull rather than merely bounding the mounts, which is what gives a palm a body
even when every finger crowds one side of it.

Also the inner bound of the mount annulus: a finger starts on this disc's rim
at the closest, never inside it.
"""

PALM_RIM = 0.005
"""How far the plate carries PAST a finger's base, and the radius its corners
are rounded to.

The outline used to be grown by PALM_MIN_RADIUS, which left 20 mm of plate
outboard of every base -- enough to bury a whole first link, since the shortest
is 20 mm and a capsule only 15 mm wide. The palm stops at the bases now, with
one quantum of rim: enough to round every corner and to leave a mount material
to bolt to, not enough to hide what it carries.
"""

MAX_MOUNT_RADIUS = 0.070
"""How far from the centre a finger may start. With PALM_MIN_RADIUS this is the
annulus a mount lives in.

A reach, so it stays a CIRCLE even though a mount is now spelled on a square
grid: the bound is how far a finger can be from the palm's centre, which has no
corners. 70 mm because the vendor hands need 62 (LEAP), 45 (MIDAS) and 41
(wuji2) about their own base centroids.

Tied to WRIST_STANDOFF and PALM_RIM, which is not obvious: those two and the
50 mm the arm sits behind the palm frame add to exactly this 70 mm, so a base
at the far rim pointing straight back at the wrist lands its rim exactly on
ARM_FACE_Z. The outer ring is tangent to the arm, and raising this without
raising the standoff puts part of it inside -- where check_arm_clearance will
refuse every hand that tries to use it, rather than anything here saying so.
"""

ARM_FACE_Z = -rpc.FLANGE_TO_PALM_Z_M
"""Where the ARM stops, in the palm frame. Nothing the hand owns may go behind.

The palm frame's origin is bolted to the flange, and the arm is entirely behind
it: measuring the iiwa14's own meshes in this frame puts its frontmost vertex at
exactly -FLANGE_TO_PALM_Z_M, with nothing in front. So one plane is the whole
rule, and validate_design.check_arm_clearance enforces it.

This REPLACED a wedge on the mount bearing, WRIST_NOGO, which was a proxy for
the same thing and a bad one. A bearing says where a finger STARTS; what hits
the arm is where it points and how far it reaches, and the bearing knows
neither. With the wedge in force, 68% of the hands a mutation walk reached had
something inside the arm.

A wedge on the FACING, the obvious next proxy, is not much better. Over 1200
drifted hands it catches 57% of the ones in the arm at 45 degrees while refusing
15% of the ones that are fine, and 100% at 135 degrees while refusing 53% --
because whether a finger pointed backwards actually gets there depends on where
it starts and how long it is. The plane is exact instead: it refuses every hand
in the arm and nothing else.
"""

WRIST_STANDOFF = 0.025
"""How far forward of the palm FRAME the fingers are centred, along +z.

The frame's origin is where the arm bolts on, and the arm's own last surface
sits 50 mm behind it -- rpc.FLANGE_TO_PALM_Z_M, and measuring the iiwa14's
meshes in this frame puts its frontmost vertex at exactly -50 mm. So everything
the hand owns has to live forward of there, and centring the fingers on the
origin did not: a hand whose thumb reaches back, which every vendor hand's
does, put that thumb 17 mm inside the arm. Sliding the whole polar centre
forward moves palm and fingers together and leaves the bolt pattern alone.
"""

PALM_CENTRE = np.array([0.0, 0.0, WRIST_STANDOFF])
"""Where the fingers are centred, in the palm frame. Everything polar is about
this point rather than about the frame's origin."""

# --- links ------------------------------------------------------------------

LINK_QUANTUM = 0.005
CAPSULE_RADIUS = 0.015
"""Set by the actuator, not by the search.

Every joint carries one XM335-T323-T (19.0 x 35.0 x 22.0 mm). Running its 35 mm
axis along the link leaves a 19 x 22 mm cross-section, whose smallest enclosing
circle has radius 14.5 mm; rounded up to the 5 mm grid. A link's radius was never
really a free parameter -- a motor lives inside it.
"""

# Defined here rather than with the other palm constants: it is tied to the
# link, so it has to follow CAPSULE_RADIUS.
PALM_THICKNESS = 2.0 * CAPSULE_RADIUS - 0.005
"""Fixed for every hand: not sampled, not mutated, not a design variable.

``u`` runs along the thickness axis on every face, so thickness alone decides how
far a mount sits from the palm's two large faces -- but every finger originates on
the MIDPLANE now, so that is not a design variable and thickness only has to house
each finger's BASE motor, which sits in the palm rather than in any link. The
XM335's smallest dimension is 19 mm, so 25 mm clears it.

A link's capsule is 30 mm across and the palm is deliberately THINNER than that,
by one 5 mm quantum. A palm as thick as the fingers reads as a block with
fingers on it rather than as a hand, and nothing needs the extra 5 mm: the base
motor fits in 19.
"""
PALM_THICKNESS_RANGE = (PALM_THICKNESS, PALM_THICKNESS)   # x -- a single point now

MIN_LINK_LENGTH = 0.020
"""The closest two joint axes can sit, for a link that carries a motor.

The XM335 is 19.0 x 35.0 x 22.0 mm, so 20 mm is its shortest dimension rounded to
the grid: the orientation with that axis along the link is the one that packs two
joints closest together.

NOTE the capsule does NOT enclose the motor at this length. That orientation
leaves a 35 x 22 mm cross-section, which needs a 20.7 mm radius against the
15 mm CAPSULE_RADIUS here. The capsule is a nominal collision proxy, not a motor
housing -- the real link bulges where the motor sits, and packaging is resolved
when a design is actually built. Set deliberately: a radius wide enough to
enclose the motor in every orientation would force 55 mm mount separation, which
leaves room for only three fingers on a maximal palm.
"""

MIN_DISTAL_LINK_LENGTH = 0.015
"""The floor for the LAST link of a finger, which drives no child joint.

It carries no motor, so it only has to be long enough to be a fingertip. Its own
joint's motor sits in the link proximal to it, or in the palm when the finger has
a single joint.
"""

MAX_LINK_LENGTH = 0.080
"""Deliberately loose. Stall torque never binds it: a 3-link, 35 mm finger holds
itself horizontal on 7.5% of the XM335's 1.03 N.m."""

MAX_FINGER_LENGTH = 0.200
"""Fully-extended mount-to-tip. MAX_LINK_LENGTH alone allowed 4 x 80 = 320 mm."""

# --- joints -----------------------------------------------------------------

ANGLE_QUANTUM = math.radians(15.0)
"""Grid for a mount's FACING, and for the one angle only an IMPORTED joint can
carry, its offset.

Chosen, not derived. It used to grid a mount's BEARING too, when where a finger
sat was polar; WHERE is cartesian now and only which way it POINTS is an angle.
That is what made the change worth making -- an angular grid sites a far finger
coarsely and a near one finely, while an angular grid on a DIRECTION is the
same 15 degrees wherever the finger is.

Not what a generated JOINT carries: that is a kind and a lean, which have
alphabets of their own rather than a grid.
"""

LEAN_QUANTUM = math.radians(45.0)
"""How far a link may lean off the one before it.

A link is a motor bolted to its parent, and a bracket either carries it straight
on or tips it over by a fixed angle. 45 degrees is as far as a lean goes: a
right angle between two links is not spelled by leaning twice, it is spelled by
putting an abduction joint next to a flexion joint, which is the same corner and
an ACTUATED one.
"""


def _lean_table() -> tuple[tuple[float, float, float], ...]:
    """Straight on, then tipped 45 degrees each of the four ways."""
    c = math.cos(LEAN_QUANTUM)
    s = math.sin(LEAN_QUANTUM)
    return ((1.0, 0.0, 0.0),
            (c, s, 0.0), (c, -s, 0.0),
            (c, 0.0, s), (c, 0.0, -s))


LEANS: tuple[tuple[float, float, float], ...] = _lean_table()
"""Where a link may point in its parent's frame. Index 0 is straight on, which
is what every link of LEAP does."""

N_LEANS = len(LEANS)


def _lean_neighbours() -> tuple[tuple[int, ...], ...]:
    """Two leans are one step apart when they are within 60 degrees.

    Straight on reaches all four tips; a tip reaches straight on and the two
    tips perpendicular to it, but not the one opposite, which is 90 degrees away
    and two steps. Symmetric by construction, so the move back is also one step.
    """
    out = []
    for a in LEANS:
        near = tuple(j for j, b in enumerate(LEANS)
                     if b != a and float(np.dot(a, b)) > math.cos(math.radians(60.5)))
        out.append(near)
    return tuple(out)


LEAN_NEIGHBOURS: tuple[tuple[int, ...], ...] = _lean_neighbours()

ROLL, FLEXION, ABDUCTION = 0, 1, 2
JOINT_KINDS: tuple[int, ...] = (ROLL, FLEXION, ABDUCTION)
JOINT_KIND_NAMES: tuple[str, ...] = ("roll", "flexion", "abduction")

JOINT_AXES: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),    # ROLL -- along the link, so the link spins on its own axis
    (0.0, 1.0, 0.0),    # ABDUCTION in the mount frame: the palm normal
    (0.0, 0.0, 1.0),    # FLEXION in the mount frame: along the knuckle row
)
"""The hinge axis of each kind, in the joint's own frame, where the link is +x.

Named for what they do at the MOUNT, where a +z face puts the link's local z
along the knuckle row and its local y along the palm normal. A lean tips the
frame, so after one the names drift by up to 45 degrees -- they are exact where
it matters and approximate deeper in a leaning finger.
"""
_KIND_AXIS = {ROLL: JOINT_AXES[0], ABDUCTION: JOINT_AXES[1], FLEXION: JOINT_AXES[2]}


def lean_rot(index: int) -> np.ndarray:
    """The fixed rotation a lean applies, by the SHORTEST arc.

    Shortest arc so the lean carries no spin of its own: spin about the link is
    what a joint's own travel does, and two knobs for it is what this grammar
    just stopped having.
    """
    d = np.asarray(LEANS[index % N_LEANS], dtype=float)
    x = np.array([1.0, 0.0, 0.0])
    v = np.cross(x, d)
    s = float(np.linalg.norm(v))
    c = float(x @ d)
    if s < 1e-12:
        return np.eye(3) if c > 0.0 else rodrigues(np.array([0.0, 0.0, 1.0]), math.pi)
    return rodrigues(v, math.atan2(s, c))


JOINT_LIMIT = (math.radians(-90.0), math.radians(90.0))
"""Symmetric, for every joint regardless of axis.

Travel is centred on the joint's own zero offset, so the usable span is 120 deg
placed anywhere the offset can reach -- asymmetric ranges are expressed by moving
the offset, not by widening this. Well inside what the actuator allows; the limit
is a design choice about reach into the palm, not a hardware bound.
"""

# --- the envelope -----------------------------------------------------------

MIN_FINGERS = 2
"""A one-finger hand cannot oppose anything, so it is excluded rather than left for..."""

MAX_FINGERS = 6
MAX_JOINTS_PER_FINGER = 5
"""The articulation envelope: a HARD cap, not a rail."""

MIN_MOUNT_SEPARATION = 2.0 * CAPSULE_RADIUS + 0.005
"""Centre-to-centre floor between ANY two mounts, whatever faces they sit on.

Two capsules running parallel are tangent at 2r, so a floor below that lets
same-face fingers intersect outright; the extra 5 mm is shell-to-shell clearance.
One derived number replaces the earlier pair of chosen ones (a loose across-face
floor and a separate same-face floor).
"""


GRASP_DIR = np.array([1.0, 0.0, 0.0])
"""Fingers curl toward the palm surface (+x)."""


# --- the tree ---------------------------------------------------------------

@dataclass(frozen=True)
class Joint:
    """One revolute DOF: which of the three kinds of hinge this is.

    A generated joint is nothing but its kind. ``offset`` survives only for hands
    that were MEASURED rather than drawn -- a vendor's assembly angle, which the
    grammar itself no longer has a way to say and never mutates.
    """

    kind: int = FLEXION
    # Import-only: a measured hand's assembly angle. Always 0 in a generated one.
    offset: float = 0.0
    # Set only by an imported hand, whose hinge is measured rather than drawn.
    axis_override: tuple[float, float, float] | None = None
    # Travel, in radians. None means the design space's JOINT_LIMIT.
    limits: tuple[float, float] | None = None
    # (effort N.m, velocity rad/s, stiffness, damping, armature). None means the
    # depth-indexed default, which is what a generated joint gets.
    drive: tuple[float, float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.limits is not None and self.limits[0] > self.limits[1]:
            raise ValueError(f"joint limits out of order: {self.limits}")
        if self.drive is not None and len(self.drive) != 5:
            raise ValueError(f"drive must be 5 numbers, got {self.drive}")
        if self.axis_override is not None and len(self.axis_override) != 3:
            raise ValueError(f"axis_override must be 3 numbers, got {self.axis_override}")
        if not math.isfinite(self.offset):
            raise ValueError(f"non-finite joint offset {self.offset}")
        if self.kind not in JOINT_KINDS:
            raise ValueError(f"joint kind must be one of {JOINT_KINDS} "
                             f"({', '.join(JOINT_KIND_NAMES)}), got {self.kind!r}")


@dataclass(frozen=True)
class Segment:
    """A joint and the link distal to it."""

    joint: Joint
    length: float
    # Which way this link leans off the one before it, as an index into LEANS.
    # 0 is straight on -- collinear with the previous link, or along the mount
    # normal for the first segment of a finger. One knob: the palm uses it for
    # the angle a finger leaves at, every later segment for the angle the next
    # link tips through.
    lean: int = 0
    # (bend, axis) extents of the link box. None means the capsule a generated
    # design gets; an imported hand carries its measured cross-section instead.
    cross_section: tuple[float, float] | None = None
    # Collision/visual meshes, for a measured link. None authors a capsule.
    meshes: tuple[str, ...] | None = None
    # An imported hand's box, verbatim, in ITS child-link frame. A generated
    # hand has none: we author its USD, so its links lie along +x by
    # construction, and the box is derived. A measured hand's frames come from
    # its own asset and cannot be re-derived from length and axis alone.
    token_box: tuple | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.length) or self.length < 0.0:
            raise ValueError(f"bad link length {self.length}")
        if self.cross_section is not None and len(self.cross_section) != 2:
            raise ValueError(f"cross_section must be (bend, axis), got {self.cross_section}")
        if not isinstance(self.lean, int) or not 0 <= self.lean < N_LEANS:
            raise ValueError(f"lean must be an index into LEANS (0..{N_LEANS - 1}), "
                             f"got {self.lean!r}")

    @property
    def box(self) -> tuple[float, float, float]:
        """``(length, bend, axis)`` extents of this link, in metres."""
        if self.cross_section is None:
            return (self.length, 2.0 * CAPSULE_RADIUS, 2.0 * CAPSULE_RADIUS)
        return (self.length, self.cross_section[0], self.cross_section[1])


@dataclass(frozen=True)
class Mount:
    """Where a finger starts, and which way it leaves. Both in the palm plane.

    ``y`` and ``z`` say WHERE the base sits, offset from PALM_CENTRE, each on
    PALM_QUANTUM. ``facing`` says which way the finger leaves, as an angle on
    ANGLE_QUANTUM. Where it sits and where it points are separate on purpose: a
    thumb reaches across the palm, so they are not the same question.

    CARTESIAN, where this used to be polar -- a radius on the 5 mm grid and a
    bearing on the 15 degree one. Polar made the ring and MAX_MOUNT_RADIUS
    natural, but it sited a finger less and less precisely the further out it
    went: one bearing step is 5.2 mm of arc at the inner rim and 18.3 mm at the
    outer, so the far ring was placed three and a half times more coarsely than
    the near one. On one grid in both directions every site is 5 mm from its
    neighbours wherever it is. The ring survives as a BOUND -- see
    ``radius`` below and check_layout -- rather than as the way a mount is
    spelled.

    ``facing`` is absolute, measured from +z like any other bearing here, NOT
    from the radial direction. A finger at facing 0 points away from the wrist
    whatever side of the palm it sits on.

    ``facing`` and the first segment's ``lean`` overlap -- both turn the first
    link -- which is accepted: ``facing`` turns it IN the palm plane and a lean
    tips it OUT of that plane, so between them a finger can leave in any
    direction without either knob doing the other's job alone.
    """

    y: float
    z: float
    facing: float

    def __post_init__(self) -> None:
        for name in ("y", "z", "facing"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"non-finite mount {name}")

    @classmethod
    def polar(cls, radius: float, bearing: float, facing: float,
              *, snap: bool = True) -> "Mount":
        """A mount from the polar pair, snapped onto the grid.

        Where a finger goes is often natural to SAY in polar -- "40 mm out at
        30 degrees" -- even though a mount is spelled in y and z. This is the
        one place that conversion lives, so a seed, a measured hand and a test
        all round the same way. ``snap=False`` keeps the exact point, for a hand
        that is a measurement rather than a design.

        It does not clamp: a radius outside the ring converts fine and
        check_layout refuses it, which is the caller's business to handle.
        """
        y, z = radius * math.sin(bearing), radius * math.cos(bearing)
        if snap:
            y = round(y / PALM_QUANTUM) * PALM_QUANTUM
            z = round(z / PALM_QUANTUM) * PALM_QUANTUM
        return cls(y=y, z=z, facing=facing)

    @property
    def radius(self) -> float:
        """How far out the base sits. Derived now, and still what the annulus
        bounds -- MAX_MOUNT_RADIUS is a reach, so it stays a circle even though
        the grid under it is square."""
        return math.hypot(self.y, self.z)

    @property
    def bearing(self) -> float:
        """Which way round the palm the base sits, from +z. Derived, and no
        longer on any grid: it is read for reporting, not stored or snapped."""
        return bearing_of(self.y, self.z)


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
    """All that is stored of a palm. Its OUTLINE is derived from the mounts.

    See ``palm_outline``: a palm is the convex hull of its own minimum disc and
    of where its fingers start. There is nothing here to mutate, which is why
    there is no longer a perturb_palm.
    """

    thickness: float   # x, the one dimension the shape does not give us


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
        """One motor per joint -- couplings are deferred."""
        return self.n_joints


# --- complexity -------------------------------------------------------------

def complexity(hand: Hand) -> tuple[int, int]:
    """``(n_motors, n_joints)`` -- readable without touching a simulator."""
    return (hand.n_motors, hand.n_joints)


# --- small helpers ----------------------------------------------------------

def with_finger(hand: Hand, i: int, finger: Finger) -> Hand:
    """Replace finger ``i``. Frozen dataclasses, so every edit rebuilds."""
    fingers = list(hand.fingers)
    fingers[i] = finger
    return replace(hand, fingers=tuple(fingers))


def palm_center(hand: "Hand") -> tuple[float, float, float]:
    """Centre of the palm box, in the palm frame.

    The centroid of the derived outline, not of a box: a radial palm has no
    symmetry to assume, since where its fingers start decides its shape. In x it
    is still exactly centred, because the plate is extruded evenly either way.
    """
    ring = palm_outline(hand)
    nxt = np.roll(ring, -1, axis=0)
    cr = ring[:, 0] * nxt[:, 1] - nxt[:, 0] * ring[:, 1]
    area = float(cr.sum()) / 2.0
    if abs(area) < 1e-12:
        c = ring.mean(axis=0)
        return (0.0, float(c[0]), float(c[1]))
    cy = float(((ring[:, 0] + nxt[:, 0]) * cr).sum()) / (6.0 * area)
    cz = float(((ring[:, 1] + nxt[:, 1]) * cr).sum()) / (6.0 * area)
    return (0.0, cy, cz)


def palm_area(hand: "Hand") -> float:
    """Area of the derived outline, for mass and inertia."""
    ring = palm_outline(hand)
    nxt = np.roll(ring, -1, axis=0)
    return abs(float((ring[:, 0] * nxt[:, 1] - nxt[:, 0] * ring[:, 1]).sum())) / 2.0


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
    """The hinge axis in the joint's own frame, where the link runs along +x."""
    if joint.axis_override is not None:
        a = np.asarray(joint.axis_override, dtype=float)
        return a / max(float(np.linalg.norm(a)), 1e-12)
    return np.array(_KIND_AXIS[joint.kind], dtype=float)


# --- where things sit on the palm --------------------------------------------

def bearing_of(y: float, z: float) -> float:
    """The bearing of a point in the palm plane, measured from +z, in [0, 2pi).

    From +z because that is the direction a row of fingers tends to point, which
    puts the arm at pi -- directly behind the hand.
    """
    return math.atan2(y, z) % (2.0 * math.pi)


def mount_position(mount: Mount) -> np.ndarray:
    """Where the finger starts, in the palm frame.

    Offset from PALM_CENTRE, and always in the midplane: x is 0 because a palm
    is a flat plate and a finger starts on it, not above or below it.
    """
    return PALM_CENTRE + np.array([0.0, mount.y, mount.z])


def mount_direction(mount: Mount) -> np.ndarray:
    """Which way the finger leaves, in the palm frame.

    Its own angle, NOT the direction it sits in. A thumb sits off one side and
    reaches back across the palm, which a face normal could never say.
    """
    return np.array([0.0, math.sin(mount.facing), math.cos(mount.facing)])


def palm_outline(hand: "Hand", arc: int = 32) -> np.ndarray:
    """The palm as (y, z) vertices, counterclockwise. Derived, never stored.

    The convex hull of two kinds of circle: the palm's own disc, of
    PALM_MIN_RADIUS about PALM_CENTRE, and a disc of PALM_RIM about every place
    a finger starts. So the plate always has a body, it reaches out to each
    base, and it STOPS there -- one rim past the base, where it used to grow a
    full 20 mm and swallow the first link whole. Hulling circles rather than
    points is what keeps the outline smooth: there is no corner anywhere left
    to fillet, and one finger or six fall out of the same construction.

    ``arc`` is how many segments a full circle is drawn with -- a drawing
    resolution, not a design parameter. The hull of the samples is inscribed in
    the true outline, by 0.1 mm at the default.
    """
    a = np.linspace(0.0, 2.0 * math.pi, arc, endpoint=False)
    circle = np.column_stack([np.sin(a), np.cos(a)])
    pts = [PALM_CENTRE[1:] + PALM_MIN_RADIUS * circle]
    pts += [mount_position(f.mount)[1:] + PALM_RIM * circle
            for f in hand.fingers]
    return _hull_ccw(np.vstack(pts))


def _hull_ccw(pts: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D points, counterclockwise, duplicates dropped.

    Monotone chain. Kept hand-written rather than handed to scipy for the
    winding: ``palm_hull`` extrudes this ring and its faces are wound off the
    order the points come back in. It also used to be needed because a two
    finger hand hulled two points and scipy refuses what it cannot triangulate,
    which ``palm_outline`` no longer asks of it -- the minimum disc puts 32
    points in whatever the hand has.
    """
    uniq = np.unique(np.round(pts, 9), axis=0)
    order = np.lexsort((uniq[:, 1], uniq[:, 0]))
    q = uniq[order]
    if len(q) <= 2:
        return q

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for v in q:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], v) <= 0:
            lower.pop()
        lower.append(v)
    upper = []
    for v in q[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], v) <= 0:
            upper.pop()
        upper.append(v)
    return np.array(lower[:-1] + upper[:-1])


def palm_hull(hand: "Hand") -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """The palm as ``(vertices, faces)``: the outline extruded through x.

    ONE convex solid, which is what a physics engine wants -- no decomposition,
    no union of parts, and the same shape the viewer draws.
    """
    ring = palm_outline(hand)
    half = hand.palm.thickness / 2.0
    n = len(ring)
    verts = np.array([[-half, y, z] for y, z in ring]
                     + [[half, y, z] for y, z in ring], dtype=float)
    faces: list[tuple[int, ...]] = [tuple(range(n - 1, -1, -1)),
                                    tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, j + n, i + n))
    return verts, faces


def palm_extents(hand: "Hand") -> tuple[float, float, float]:
    """``(thickness, width, length)`` of the palm's bounding box."""
    ring = palm_outline(hand)
    lo, hi = ring.min(axis=0), ring.max(axis=0)
    return (hand.palm.thickness, float(hi[0] - lo[0]), float(hi[1] - lo[1]))


def _frame_from_axis(axis: np.ndarray) -> np.ndarray:
    """Orthonormal frame with ``axis`` as its first column."""
    a = axis / np.linalg.norm(axis)
    ref = GRASP_DIR
    if np.linalg.norm(np.cross(a, ref)) < 1e-6:
        ref = np.eye(3)[int(np.argmin(np.abs(a)))]
    fe = np.cross(a, ref)
    fe /= np.linalg.norm(fe)
    aa = np.cross(fe, a)
    return np.column_stack([a, aa, fe])


def mount_frame(mount: Mount) -> tuple[np.ndarray, np.ndarray]:
    """``(position, R)`` for a finger's base, in the palm frame."""
    return mount_position(mount), _frame_from_axis(mount_direction(mount))


# --- forward kinematics -----------------------------------------------------

def forward_kinematics(finger: Finger, palm: Palm,
                       angles: dict[int, float] | None = None,
                       ) -> tuple[list[np.ndarray], list[tuple]]:
    """``(joint_positions, capsules)`` for one finger, in the palm frame."""
    angles = angles or {}
    p, R = mount_frame(finger.mount)
    joints: list[np.ndarray] = []
    capsules: list[tuple] = []

    for i, seg in enumerate(finger.segments):
        joints.append(p.copy())
        R = R @ lean_rot(seg.lean)                      # bolt the link on
        R = R @ rodrigues(axis_of(seg.joint),           # then the joint turns
                          seg.joint.offset + angles.get(i, 0.0))
        nxt = p + R[:, 0] * seg.length
        if seg.length > _EPS:
            capsules.append((p.copy(), nxt.copy(), CAPSULE_RADIUS, i))
        p = nxt

    joints.append(p.copy())
    return joints, capsules


def joint_axes(finger: Finger, palm: Palm,
               angles: dict[int, float] | None = None) -> list[np.ndarray]:
    """Each joint's hinge axis as a unit vector in the palm frame."""
    angles = angles or {}
    _, R = mount_frame(finger.mount)
    out: list[np.ndarray] = []
    for i, seg in enumerate(finger.segments):
        R = R @ lean_rot(seg.lean)
        a = axis_of(seg.joint)
        out.append(R @ a)
        R = R @ rodrigues(a, seg.joint.offset + angles.get(i, 0.0))
    return out


def fingertip(finger: Finger, palm: Palm,
              angles: dict[int, float] | None = None) -> np.ndarray:
    return forward_kinematics(finger, palm, angles)[0][-1]


def rearmost(hand: "Hand", links=None) -> tuple[float, int | None]:
    """``(z of the backmost point the hand owns, which finger owns it)``.

    None for the finger means the palm plate owns it. At the REST pose, the same
    convention check_base_clearance keeps: what a joint sweep then does is
    configuration, which is the gate's business rather than the grammar's.

    Exact, and cheap for both halves. The palm is the hull of circles, so its
    backmost point is simply the backmost point of the backmost circle -- no
    outline to build. A link is the AUTHORED capsule, whose axis is inset by a
    radius at each end: taking the joint-to-joint span instead would call every
    link 15 mm longer than the one the simulator builds, and refuse hands that
    clear.

    ``links`` takes a ``rest_capsules`` a caller already has. The validator runs
    this beside check_base_clearance, which needs the same list, and walking the
    kinematics twice per hand cost the test suite 60% of its runtime.
    """
    worst = PALM_CENTRE[2] - PALM_MIN_RADIUS
    who = None
    for i, finger in enumerate(hand.fingers):
        z = float(mount_position(finger.mount)[2]) - PALM_RIM
        if z < worst:
            worst = z                     # still the PLATE, reaching that base
    for fi, _si, a, b in (rest_capsules(hand) if links is None else links):
        z = float(min(a[2], b[2])) - CAPSULE_RADIUS
        if z < worst:
            worst, who = z, fi
    return worst, who


def curl_authority(finger: Finger, palm: Palm) -> float:
    """How well this finger can swing its tip INTO the palm. 0 to 1.

    The one thing a finger has to do to take part in a grasp: close toward where
    an object sits. Each joint contributes ``(axis x (tip - joint)) . GRASP_DIR``
    -- the Jacobian column projected on the grasp direction, which is the speed
    the tip heads for the palm per radian of that joint. Divided by the finger's
    own reach, so a long finger gets no credit for being long.

    The MAX over joints rather than the norm: the question is whether the finger
    has a joint that closes it, not how many. The norm would rank a 4-joint
    finger above a 3-joint one for no reason a grasp can use.

    Exactly 0 for a finger of roll joints, whose axes lie along their own links,
    and for a finger of abduction joints, whose axes lie along GRASP_DIR itself
    -- abduction spreads a hand, it never closes it. Measured at the rest pose,
    which is one forward-kinematics call.
    """
    if finger.reach <= _EPS:
        return 0.0
    pts, _ = forward_kinematics(finger, palm)
    axes = joint_axes(finger, palm)
    tip = pts[-1]
    return max(abs(float(np.cross(a, tip - p) @ GRASP_DIR))
               for a, p in zip(axes, pts[:-1])) / finger.reach


def curl_score(hand: "Hand") -> float:
    """The hand's SECOND-best finger by curl authority.

    Second rather than best, because a grasp needs two things closing on an
    object -- two fingers, or one finger and the palm it presses against. One
    good finger beside a row of dead ones is not a hand.

    Cheap enough to filter sampled designs with: about 230 microseconds for a
    whole hand. LEAP, the one hand fitted into this grammar, scores 1.00, and a
    hand of nothing but roll or abduction joints scores exactly 0. There is no
    second reference yet -- SHARPA is in the repo but only as a capsule built
    against older constants, so it calibrates nothing.
    """
    ranked = sorted((curl_authority(f, hand.palm) for f in hand.fingers),
                    reverse=True)
    return ranked[1] if len(ranked) > 1 else ranked[0]


# --- the two measures that carry signal -------------------------------------

def segment_distance(p0: np.ndarray, p1: np.ndarray,
                     q0: np.ndarray, q1: np.ndarray) -> float:
    """Closest distance between two 3-D line segments, closed form."""
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
    """Each finger's proximal link at the rest pose, as a core segment."""
    out = []
    for f in hand.fingers:
        p0, R = mount_frame(f.mount)
        seg = f.segments[0]
        R = R @ lean_rot(seg.lean)
        R = R @ rodrigues(axis_of(seg.joint), seg.joint.offset)
        out.append((p0, p0 + R[:, 0] * seg.length))
    return out


def capsule_axis(p0: np.ndarray, p1: np.ndarray, radius: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """The AUTHORED capsule's axis for a link spanning ``p0`` to ``p1``.

    A capsule's tip-to-tip extent is the link length, so its axis is inset by
    one radius at each end -- `viewer.capsule_mesh` and `author_hand` both build
    it that way. Using the full span instead treats every link as ``2 * radius``
    longer than the simulator's, which for a clearance test is not conservative
    in a useful way: it rejects designs that do not touch.

    A link shorter than ``2 * radius`` has no cylindrical part at all and is a
    sphere at its midpoint, which is what the simulator authors too.
    """
    d = np.asarray(p1, float) - np.asarray(p0, float)
    length = float(np.linalg.norm(d))
    if length < 1e-12:
        return np.asarray(p0, float), np.asarray(p0, float)
    half = max(length - 2.0 * radius, 0.0) / 2.0
    mid = (np.asarray(p0, float) + np.asarray(p1, float)) / 2.0
    return mid - d / length * half, mid + d / length * half


def rest_capsules(hand: Hand) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """``(finger, depth, axis start, axis end)`` for EVERY link at the rest pose.

    ``base_capsules`` is the proximal link of each finger and nothing else,
    which is every pair a two-joint hand HAS. Mutation makes longer fingers, and
    a distal link folding back onto another finger is invisible to a check that
    only ever looks at the first segment of each.
    """
    out = []
    for fi, f in enumerate(hand.fingers):
        _, caps = forward_kinematics(f, hand.palm)
        for p0, p1, radius, si in caps:
            a, b = capsule_axis(p0, p1, radius)
            out.append((fi, si, a, b))
    return out


def mount_separations(hand: Hand) -> list[float]:
    """Pairwise distances between finger mounts, in metres."""
    pos = [mount_position(f.mount) for f in hand.fingers]
    return [float(np.linalg.norm(pos[i] - pos[j]))
            for i in range(len(pos)) for j in range(i + 1, len(pos))]


# --- rotation conventions, mass properties, and the scene a hand is judged in

from scipy.spatial.transform import Rotation


def rpy_to_mat(rpy) -> np.ndarray:
    """URDF RPY -> 3x3 rotation matrix."""
    return Rotation.from_euler("xyz", np.asarray(rpy, dtype=float)).as_matrix()


def mat_to_rpy(mat) -> tuple[float, float, float]:
    """3x3 rotation matrix -> URDF RPY."""
    r, p, y = Rotation.from_matrix(np.asarray(mat, dtype=float)[:3, :3]).as_euler("xyz")
    return float(r), float(p), float(y)


def rpy_to_quat_wxyz(rpy) -> tuple[float, float, float, float]:
    """URDF RPY -> (w, x, y, z)."""
    x, y, z, w = Rotation.from_euler("xyz", np.asarray(rpy, dtype=float)).as_quat()
    return float(w), float(x), float(y), float(z)


def rpy_to_rot6d(rpy) -> list[float]:
    """First two columns of the rotation matrix for an RPY triple."""
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



def compute_mass_and_inertia(scale, density: float):
    """Capsule-approximation for cylinders; exact for cuboids."""
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


from hand_sampler import resolve as resolve_repo_path


# Task geometry, from ResetCfg.
GOAL_VOLUME_MINS = (-0.35, -0.2, 0.6)
GOAL_VOLUME_MAXS = (0.35, 0.2, 0.95)
TABLE_Z = 0.38
TABLE_URDF = "assets/urdf/table_narrow.urdf"


def table_extents() -> tuple[float, float, float]:
    """Box dimensions read from the actual table asset."""
    import xml.etree.ElementTree as ET

    root = ET.parse(resolve_repo_path(TABLE_URDF)).getroot()
    for link in root.findall("link"):
        for coll in link.findall("collision"):
            box = coll.find("geometry/box")
            if box is not None:
                return tuple(float(v) for v in box.get("size").split())
    raise RuntimeError(f"no box collision geometry in {TABLE_URDF}")


def _hull_collision_scene(urdf) -> int:
    """Replace each collision mesh with its convex hull, in place."""
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


# --- joint tokens: the one description a policy reads, for any hand ---------

def joint_boxes(hand: "Hand") -> tuple[np.ndarray, np.ndarray, float]:
    """``(boxes, valid, hand_scale)`` -- the per-joint link box of every joint."""
    boxes, valid = [], []
    for finger in hand.fingers:
        for seg in finger.segments:
            if seg.token_box is not None:
                boxes.append(np.asarray(seg.token_box, dtype=np.float32).reshape(4, 3))
                valid.append(True)
                continue
            length, width, height = seg.box
            axis = axis_of(seg.joint)
            distal = np.array([1.0, 0.0, 0.0])          # the link runs along +x
            distal = distal - axis * float(distal @ axis)
            n = float(np.linalg.norm(distal))
            if n < 1e-8:                                 # a roll joint: pick any transverse ray
                distal = np.cross(axis, np.eye(3)[int(np.argmin(np.abs(axis)))])
                n = float(np.linalg.norm(distal))
            distal = distal / max(n, 1e-12)
            bend = np.cross(axis, distal)
            bend = bend / max(float(np.linalg.norm(bend)), 1e-12)
            p0 = -0.5 * width * bend - 0.5 * height * axis
            boxes.append(np.asarray([p0, p0 + length * distal,
                                     p0 + width * bend, p0 + height * axis], dtype=np.float32))
            valid.append(length > 0.0)
    if not boxes:
        return np.zeros((0, 4, 3), np.float32), np.zeros((0,), bool), 0.0
    arr = np.stack(boxes)
    scale = float(max(np.linalg.norm(arr[:, i] - arr[:, 0], axis=-1).max() for i in (1, 2, 3)))
    return arr, np.asarray(valid, dtype=bool), scale


# --- importing a measured hand: its URDF read once, never at runtime ---





def _values(node: ET.Element | None, key: str, default) -> np.ndarray:
    if node is None or node.get(key) is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in node.get(key).split()], dtype=np.float64)


def _origin(node: ET.Element | None) -> np.ndarray:
    """URDF origin as a homogeneous parent-from-child transform."""
    xyz = _values(node, "xyz", (0.0, 0.0, 0.0))
    r, p, y = _values(node, "rpy", (0.0, 0.0, 0.0))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rot = np.asarray([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = xyz
    return out


def _box_corners(extents) -> np.ndarray:
    half = 0.5 * np.asarray(extents, dtype=np.float64)
    return np.asarray([
        (sx * half[0], sy * half[1], sz * half[2])
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ])


def _collision_vertices(collision: ET.Element, base_dir: Path) -> np.ndarray:
    geom = collision.find("geometry")
    if geom is None:
        return np.empty((0, 3), dtype=np.float64)
    box = geom.find("box")
    cylinder = geom.find("cylinder")
    sphere = geom.find("sphere")
    mesh = geom.find("mesh")
    if box is not None:
        vertices = _box_corners(_values(box, "size", (0.0, 0.0, 0.0)))
    elif cylinder is not None:
        radius = float(cylinder.get("radius"))
        length = float(cylinder.get("length"))
        # Its AABB is sufficient; the ordered box is not a collision proxy.
        vertices = _box_corners((2.0 * radius, 2.0 * radius, length))
    elif sphere is not None:
        radius = float(sphere.get("radius"))
        vertices = _box_corners((2.0 * radius,) * 3)
    elif mesh is not None:
        import trimesh

        filename = mesh.get("filename")
        if filename.startswith("package://"):
            raise ValueError(f"package URI is not resolvable offline: {filename}")
        loaded = trimesh.load(str(base_dir / filename), force="mesh", process=False)
        vertices = np.asarray(loaded.vertices, dtype=np.float64)
        if mesh.get("scale") is not None:
            vertices = vertices * _values(mesh, "scale", (1.0, 1.0, 1.0))
    else:
        return np.empty((0, 3), dtype=np.float64)

    transform = _origin(collision.find("origin"))
    return vertices @ transform[:3, :3].T + transform[:3, 3]


def _nearest_geometry(
    start_link: str,
    links: dict[str, ET.Element],
    outgoing: dict[str, list[ET.Element]],
    base_dir: Path,
) -> np.ndarray:
    """Nearest collision-bearing descendant, expressed in ``start_link``."""
    frontier = [(start_link, np.eye(4, dtype=np.float64))]
    seen: set[str] = set()
    while frontier:
        found: list[np.ndarray] = []
        following: list[tuple[str, np.ndarray]] = []
        for link_name, start_from_link in frontier:
            if link_name in seen:
                continue
            seen.add(link_name)
            link = links[link_name]
            pieces = [_collision_vertices(c, base_dir) for c in link.findall("collision")]
            pieces = [p for p in pieces if p.size]
            if pieces:
                vertices = np.concatenate(pieces, axis=0)
                found.append(
                    vertices @ start_from_link[:3, :3].T + start_from_link[:3, 3]
                )
                continue
            for joint in outgoing.get(link_name, ()):
                child = joint.find("child").get("link")
                following.append(
                    (child, start_from_link @ _origin(joint.find("origin")))
                )
        if found:
            return np.concatenate(found, axis=0)
        frontier = following
    return np.empty((0, 3), dtype=np.float64)


def _ordered_box(vertices: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, bool]:
    if not vertices.size:
        return np.zeros((4, 3), dtype=np.float32), False

    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    # The collision centroid normally points down the controlled segment.  If
    # the mesh is centred on the pivot, use its farthest transverse vertex.
    distal = vertices.mean(axis=0)
    distal = distal - axis * float(distal @ axis)
    if np.linalg.norm(distal) < 1e-8:
        transverse = vertices - np.outer(vertices @ axis, axis)
        distal = transverse[np.argmax(np.linalg.norm(transverse, axis=1))]
    if np.linalg.norm(distal) < 1e-8:
        basis = np.eye(3)[np.argmin(np.abs(axis))]
        distal = np.cross(axis, basis)
    distal /= np.linalg.norm(distal)

    bend = np.cross(axis, distal)  # positive angular motion of the distal ray
    bend /= max(float(np.linalg.norm(bend)), 1e-12)

    projected = np.stack(
        [vertices @ distal, vertices @ bend, vertices @ axis], axis=-1
    )
    spans = projected.max(axis=0) - projected.min(axis=0)
    # Preserve the collision dimensions but anchor the proximal face at the
    # joint.  Link meshes occasionally overhang slightly behind their pivot.
    length = max(float(spans[0]), float(projected[:, 0].max()), 1e-6)
    width = max(float(spans[1]), 1e-6)
    height = max(float(spans[2]), 1e-6)

    p0 = -0.5 * width * bend - 0.5 * height * axis
    return np.asarray([
        p0,
        p0 + length * distal,
        p0 + width * bend,
        p0 + height * axis,
    ], dtype=np.float32), True


def joint_link_boxes(
    urdf: str | Path | ET.Element,
    joint_names,
    *,
    base_dir: str | Path | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    """Return child bodies, ordered boxes, validity, and characteristic scale."""
    if isinstance(urdf, ET.Element):
        root = urdf
        mesh_base = Path(base_dir or ".")
    else:
        path = Path(resolve(urdf))
        root = ET.parse(path).getroot()
        mesh_base = path.parent if base_dir is None else Path(base_dir)

    links = {link.get("name"): link for link in root.findall("link")}
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    outgoing: dict[str, list[ET.Element]] = {}
    for joint in root.findall("joint"):
        outgoing.setdefault(joint.find("parent").get("link"), []).append(joint)

    child_links: list[str] = []
    boxes: list[np.ndarray] = []
    valid: list[bool] = []
    for name in joint_names:
        if name not in joints:
            raise KeyError(f"URDF has no joint {name!r}")
        joint = joints[name]
        child = joint.find("child").get("link")
        axis_node = joint.find("axis")
        axis = _values(axis_node, "xyz", (1.0, 0.0, 0.0))
        vertices = _nearest_geometry(child, links, outgoing, mesh_base)
        box, ok = _ordered_box(vertices, axis)
        child_links.append(child)
        boxes.append(box)
        valid.append(ok)

    boxes_np = np.stack(boxes).astype(np.float32)
    valid_np = np.asarray(valid, dtype=np.bool_)
    lengths = np.linalg.norm(boxes_np[:, 1] - boxes_np[:, 0], axis=-1)
    scale = float(lengths[valid_np].max()) if valid_np.any() else 1.0
    return child_links, boxes_np, valid_np, max(scale, 1e-6)


__all__ = ["joint_link_boxes"]


def hand_from_urdf(urdf, joint_names, *, base_dir=None, palm=None) -> "Hand":
    """Import a measured hand -- SHARPA -- as a ``Hand``."""
    if isinstance(urdf, ET.Element):
        root, meshes = urdf, base_dir or "."
    else:
        path = resolve(urdf)
        root, meshes = ET.parse(path).getroot(), base_dir or path.parent
    _bodies, boxes, valid, _scale = joint_link_boxes(root, joint_names, base_dir=meshes)
    joints = {j.get("name"): j for j in root.findall("joint")}
    parent_of = {n: j.find("parent").get("link") for n, j in joints.items()}
    child_of = {n: j.find("child").get("link") for n, j in joints.items()}
    wanted = list(joint_names)
    in_hand = set(wanted)
    joint_making = {child_of[n]: n for n in joints}          # child link -> the joint above it

    # A finger is a chain of controlled joints. Fixed joints sit between them in
    # the URDF, so the predecessor is the nearest CONTROLLED ancestor, not the
    # joint immediately above.
    def predecessor(name):
        link = parent_of[name]
        while link in joint_making:
            up = joint_making[link]
            if up in in_hand:
                return up
            link = parent_of[up]
        return None

    prev = {n: predecessor(n) for n in wanted}
    nxt = {}
    for n, pr in prev.items():
        if pr is not None:
            nxt.setdefault(pr, []).append(n)
    chains = []
    for head in [n for n in wanted if prev[n] is None]:
        chain, cur = [head], head
        while nxt.get(cur):
            cur = nxt[cur][0]        # one controlled child per link in a finger
            chain.append(cur)
        chains.append(chain)

    index = {n: i for i, n in enumerate(wanted)}
    fingers = []
    the_palm = palm or Palm(thickness=PALM_THICKNESS)
    for k, chain in enumerate(chains):
        segs = []
        for n in chain:
            i = index[n]
            box = np.asarray(boxes[i], dtype=float)
            length = float(np.linalg.norm(box[1] - box[0]))
            segs.append(Segment(joint=Joint(kind=FLEXION), length=max(length, 1e-6),
                                token_box=tuple(map(tuple, box))))
        # A measured hand's real mount is whatever its URDF says; this only has
        # to be a legal, distinct place to hang each chain from.
        bearing = (2.0 * math.pi * k / max(len(chains), 1)) % (2.0 * math.pi)
        fingers.append(Finger(mount=Mount.polar(PALM_MIN_RADIUS + MIN_MOUNT_SEPARATION,
                                                bearing, bearing),
                              segments=tuple(segs)))
    return Hand(palm=the_palm, fingers=tuple(fingers))
