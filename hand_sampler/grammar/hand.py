"""The hand grammar's genotype, its constants and its rules
(`project-notes/grammar/GRAMMAR-LOCK-2026-10-07.md`, revised 2026-10-14).

A hand is a flat palm plate with 2-6 fingers, built from one motor (the
Feetech HL-3915, `motor.py`). Everything is stored as integers on one grid:
millimetres in steps of 10, degrees in steps of 30.

Frames
------
Palm frame: origin at the palm centre, where the hand mounts (the mount is a
disc behind the plate's back face), x = the palm normal on the grasp side,
y and z in the plate. The plate runs from x = -12 to +12 mm.

Every direction is two angles, `direction(az, el) = (sin el, cos el sin az,
cos el cos az)`: `az` turns in the y-z plane from +z toward +y, `el` lifts
toward +x.

- A finger's base sits at (y, z) on the plate and the finger leaves along
  e = `direction(facing, tilt)`. Its link frame: x = e, y = the closing
  direction c = (+x of the palm turned by the tilt), z = x cross y (the bend
  axis, in the plate). The base joint sits below the front face, as deep as
  its motor needs (`derive.base_depth_mm`).
- Links are straight at the zero pose, so every link of a finger has that
  frame at q = 0, and a joint's axis is `direction(az, el)` in it: (0, 0) is
  z (bend), (90, 0) is y (spread), el = 90 is x (twist). An axis is a line;
  its sign is derived (`derive.joint_sign`), so az is kept in [0, 180) and
  the twist axis is written (0, 90).
- A palm joint is a straight cut across the palm: the line with direction
  `direction(facing, 0)` at signed distance `offset` from the palm centre
  along its normal n = (palm normal) x (line direction). The part beyond the
  line (on the side n points to) hinges on the rest along the line.

Joint j of a finger sits at the start of link j; link j runs to joint j + 1,
and the last link to the fingertip.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import motor as mt

GRAMMAR_VERSION = "hand_grammar/2.0"

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

GRID_MM = 10                # every stored length and position
GRID_DEG = 30               # every stored angle

# Links: 0 mm (two joints at one point) or 10 mm up to the longest link; the
# fingertip link 10 mm up, never 0. A buildable rule set raises the minimum to
# the motor floors (`motor.link_floor_mm`, `motor.tip_floor_mm`).
# The grammar's own limits (longest link and finger, farthest base) are the
# commercial hands' extremes at the scale they are buildable from Feetech
# servos (`build_conformed`, 2026-10-14: 180 mm, SHARPA x 3.1; 450 mm, SHARPA;
# 329 mm, SVH x 3.05), x 1.1, rounded out to 10 mm; the Feetech rules keep
# 90 mm links and bases 40-90 mm out, and fingers at most 225 mm (Martin,
# 2026-10-16).
LINK_MIN_MM = 10
TIP_MIN_MM = 10
LINK_MAX_MM = 200
FINGER_LENGTH_MAX_MM = 500
BASE_DISTANCE_MAX_MM = 370

# Palm: a flat plate as thick as a flat bend motor in its housing (20 + 2 x 2
# mm), the convex hull of the mount disc and the base motors' housings.
PALM_THICKNESS_MM = 24.0
PLATE_HALF_MM = PALM_THICKNESS_MM / 2.0
# The mount: a disc at the palm centre behind the back face, where the hand
# attaches to the arm (or the world). It is the smallest palm. Assumed size.
MOUNT_RADIUS_MM = 20.0
MOUNT_DEPTH_MM = 40.0
MOUNT_POINTS = 16
PALM_OUTLINE_MAX_VERTICES = 32     # a GPU convex hull: 2 x 32 vertices, 34 faces
# A palm cut is an open seam: a 24 mm plate cannot swing +/-30 degrees through
# a zero-width cut. Both parts stop half the seam short of the line (the gap
# that keeps a wall between the two plate edges at 30 degrees).
SEAM_MM = 2.0 * (PLATE_HALF_MM * math.sin(math.radians(30.0)) + mt.WALL_MM) / (1.0 + math.cos(math.radians(30.0)))
# The cut must leave the whole mount on the fixed part.
MIN_CUT_OFFSET_MM = mt.ceil10(MOUNT_RADIUS_MM + SEAM_MM / 2.0)   # 30

# Finger tilt out of the plate, toward the grasp side: from lying in the
# plate's directions (0) to standing straight up (+90, DClaw's fingers). No
# finger points under the palm (Martin, 2026-10-16; the range was -30 to +90
# for Barrett's thumb at -25). Facing covers the full circle unless a rule set
# asks for outward-facing fingers.
TILT_RANGE_DEG = (0, 90)
# A finger whose first joint is a bend has tilt 0: the tilt turns the finger
# about its bend axis (in the plate, whatever the tilt), so a tilted bend-first
# finger is the same chain as an untilted one with its first bend offset by
# the tilt (Martin, 2026-10-16). `bend_first`.

# Joint ranges, degrees; sliding joints in millimetres (Dex1's jaws, -20 to
# +24.5 mm, the only commercial sliding joints, rounded out).
KINDS = ("flexion", "abduction", "roll")
KIND_RANGE_DEG = {"flexion": (-30, 90), "abduction": (-30, 30), "roll": (-90, 90)}
PALM_RANGE_DEG = (-30, 30)
SLIDING_RANGE_MM = (-20, 25)
COUPLING_RATIO = 1.1        # a coupled joint follows the joint before it, offset 0

# The random draw's kind mix (palm-and-axis-study.md, 263 commercial joints).
KIND_MIX = {"flexion": 0.69, "abduction": 0.27, "roll": 0.04}
KIND_AXIS = {"flexion": (0, 0), "abduction": (90, 0), "roll": (0, 90)}

JOINT_TYPES = ("hinge", "coupled", "sliding")   # no continuous joints
REVOLUTE_TYPES = ("hinge", "coupled")

# Physical properties for every hand (Kushal's uniform values), except the
# effort: the Feetech HL-3915-C001's stall torque at 12 V, 14.2 kg cm = 1.39 N m
# (Martin, 2026-10-14: "use feetech torque numbers"; was 0.5 N m). Its rated
# load is 4.5 kg cm (0.44 N m) and its no-load speed 0.1 s per 60 degrees
# (10.5 rad/s); the speed limit stays Kushal's 5 rad/s.
JOINT_EFFORT_NM = 1.39
JOINT_VELOCITY_RAD_S = 5.0
JOINT_STIFFNESS = 3.0
JOINT_DAMPING = 0.078
JOINT_ARMATURE = 0.00058
LINK_DENSITY_KG_M3 = 1750.0
FRICTION = 0.5


# --------------------------------------------------------------------------
# Genotype
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Joint:
    """One joint and the link after it."""

    type: str = "hinge"                  # "hinge" | "coupled" | "sliding"
    axis: Tuple[int, int] = (0, 0)       # (az, el), degrees, in the link frame
    length: int = 50                     # mm, the link after this joint


@dataclass(frozen=True)
class Finger:
    y: int                               # mm, base position on the plate
    z: int
    facing: int = 0                      # degrees, in the plate, from +z toward +y
    tilt: int = 0                        # degrees, out of the plate toward the grasp side
    joints: Tuple[Joint, ...] = (Joint(length=30),)


@dataclass(frozen=True)
class PalmJoint:
    """A straight cut across the palm; the part beyond it hinges along it."""

    facing: int = 0                      # degrees: the line's direction in the plate
    offset: int = MIN_CUT_OFFSET_MM      # mm: the line's distance from the palm centre along its normal


@dataclass(frozen=True)
class Hand:
    fingers: Tuple[Finger, ...]
    palm_joints: Tuple[PalmJoint, ...] = ()


def finger_length_mm(f: Finger) -> int:
    return sum(j.length for j in f.joints)


def base_distance_mm(f: Finger) -> float:
    return math.hypot(f.y, f.z)


def axis_unit(axis: Tuple[int, int]) -> np.ndarray:
    """`direction(az, el)` of a joint axis in its link frame."""
    a, e = math.radians(axis[0]), math.radians(axis[1])
    return np.array([math.sin(e), math.cos(e) * math.sin(a), math.cos(e) * math.cos(a)])


# --------------------------------------------------------------------------
# The cut
# --------------------------------------------------------------------------


def cut_direction_yz(p: PalmJoint) -> np.ndarray:
    """The cut line's direction in the plate, as (y, z)."""
    f = math.radians(p.facing)
    return np.array([math.sin(f), math.cos(f)])


def cut_normal_yz(p: PalmJoint) -> np.ndarray:
    """The cut's normal in the plate, as (y, z): (palm normal) x (direction).
    The moving part lies on the side it points to."""
    f = math.radians(p.facing)
    return np.array([-math.cos(f), math.sin(f)])


def cut_side_mm(p: PalmJoint, y: float, z: float) -> float:
    """Signed distance of the plate point (y, z) beyond the cut line (> 0 on
    the moving part)."""
    n = cut_normal_yz(p)
    return float(n[0] * y + n[1] * z) - p.offset


def finger_section(hand: Hand, i: int) -> int:
    """Which palm part finger i sits on: 0 (the part beyond the cut) or -1
    (the main palm), from its base point."""
    if not hand.palm_joints:
        return -1
    f = hand.fingers[i]
    return 0 if cut_side_mm(hand.palm_joints[0], f.y, f.z) > 0.0 else -1


# --------------------------------------------------------------------------
# Canonical angles
# --------------------------------------------------------------------------


def canonical_axis(az: int, el: int) -> Tuple[int, int]:
    """The canonical (az, el) of the line through `direction(az, el)`:
    az in [0, 180), el in [-90, 90], the twist axis as (0, 90)."""
    az, el = int(az), int(el)
    if el > 90 or el < -90:      # over the pole: the same line seen from the other side
        el = (180 - el) if el > 90 else (-180 - el)
        az += 180
    az %= 360
    if az >= 180:
        az, el = az - 180, -el
    if abs(el) == 90:
        return (0, 90)
    return (az, el)


def canonical_facing(facing: int) -> int:
    return int(facing) % 360


# --------------------------------------------------------------------------
# Rules (the limits sampling and every operator obey)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Rules:
    """Limits that random hands and every mutation keep by construction. The
    grammar's own limits (`NO_RULES`) are the widest allowed values; a rule
    set may only tighten them. `FEETECH_RULES` (the simulator's and
    evolution's) are the hands buildable from Feetech servos: hinges only,
    bases on a 40 mm grid 40-90 mm from the palm centre, fingers facing
    outward, links at least the motor floors, the mount at the palm's centre.

    `mount_at_centre`: the centroid of the base servos' hull lies within half
    a base step of the mount (`derive.palm_centre_mm`), so no two hands
    differ only in where the mount sits (Martin, 2026-10-16). It belongs to
    the rule set because the step does: the grammar's own limits leave the
    mount free and so contain every rule set's hands."""

    joint_types: Tuple[str, ...] = JOINT_TYPES
    min_fingers: int = 2
    max_fingers: int = 6
    max_joints: int = 5
    max_palm_joints: int = 1
    max_finger_length_mm: int = FINGER_LENGTH_MAX_MM
    base_pitch_mm: int = GRID_MM
    base_distance_mm: Tuple[int, int] = (10, BASE_DISTANCE_MAX_MM)
    outward_facing: bool = False
    buildable_links: bool = False
    max_link_mm: int = LINK_MAX_MM
    mount_at_centre: bool = False

    def clamp(self) -> "Rules":
        """The same rules, every field inside the grammar's own limits."""
        g = NO_RULES
        types = ("hinge",) + tuple(t for t in ("coupled", "sliding") if t in self.joint_types)
        max_f = max(g.min_fingers, min(int(self.max_fingers), g.max_fingers))
        lo = max(int(self.base_distance_mm[0]), g.base_distance_mm[0])
        hi = min(int(self.base_distance_mm[1]), g.base_distance_mm[1])
        pitch = max(GRID_MM, int(self.base_pitch_mm) // GRID_MM * GRID_MM)
        return Rules(
            joint_types=types, min_fingers=g.min_fingers, max_fingers=max_f,
            max_joints=max(1, min(int(self.max_joints), g.max_joints)),
            max_palm_joints=max(0, min(int(self.max_palm_joints), g.max_palm_joints)),
            max_finger_length_mm=max(TIP_MIN_MM, min(int(self.max_finger_length_mm), g.max_finger_length_mm)),
            base_pitch_mm=pitch, base_distance_mm=(lo, max(lo, hi)),
            outward_facing=bool(self.outward_facing), buildable_links=bool(self.buildable_links),
            max_link_mm=max(TIP_MIN_MM, min(int(self.max_link_mm) // GRID_MM * GRID_MM, g.max_link_mm)),
            mount_at_centre=bool(self.mount_at_centre),
        )


NO_RULES = Rules()
FEETECH_RULES = Rules(joint_types=("hinge",), base_pitch_mm=40, base_distance_mm=(40, 90),
                      outward_facing=True, buildable_links=True, max_finger_length_mm=225, max_link_mm=90,
                      mount_at_centre=True)
EVOLUTION_RULES = FEETECH_RULES
COMMERCIAL_RULES = Rules(buildable_links=True, mount_at_centre=True)
"""The grammar's own limits with servo link lengths: what a commercial hand,
scaled up until it is buildable from Feetech servos, is fitted under."""


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def bend_first(f: Finger) -> bool:
    """The finger's first joint turns about its bend axis (a hinge on the
    flexion axis), which is what its tilt turns it about: such a finger has
    tilt 0."""
    return bool(f.joints) and f.joints[0].type in REVOLUTE_TYPES and tuple(f.joints[0].axis) == KIND_AXIS["flexion"]


def tilts_for(kind: str) -> List[int]:
    """The tilts a finger whose first joint is of `kind` may take."""
    if kind == "flexion":
        return [0]
    return list(range(TILT_RANGE_DEG[0], TILT_RANGE_DEG[1] + 1, GRID_DEG))


def faces_outward(f: Finger) -> bool:
    """The finger does not point back across the palm: its direction in the
    plate makes at most 90 degrees with the line from the palm centre to its
    base (tangential allowed)."""
    fa = math.radians(f.facing)
    return math.sin(fa) * f.y + math.cos(fa) * f.z >= -1e-9


@lru_cache(maxsize=None)
def _link_floor(ax_a: Tuple[int, int], ax_b: Tuple[int, int], turned_a: bool, turned_b: bool) -> int:
    return mt.link_floor_mm(axis_unit(ax_a), axis_unit(ax_b), turned_a, turned_b)


@lru_cache(maxsize=None)
def _tip_floor(ax: Tuple[int, int], turned: bool) -> int:
    return mt.tip_floor_mm(axis_unit(ax), turned)


@lru_cache(maxsize=None)
def _is_bend(ax: Tuple[int, int]) -> bool:
    return mt.is_bend(axis_unit(ax))


def turned_servo(f: Finger, j: int) -> bool:
    """Whether joint j's servo is turned, its body on the back of the finger
    instead of along the link: a bend servo whose link before it is too short
    for it to lie along the link (Martin, 2026-10-16). Servos at the base and
    after a 0 mm link lie along the finger."""
    if j == 0:
        return False
    jt, prev = f.joints[j], f.joints[j - 1]
    if jt.type == "sliding" or prev.length == 0 or not _is_bend(tuple(jt.axis)):
        return False
    return prev.length < _link_floor(tuple(prev.axis), tuple(jt.axis), turned_servo(f, j - 1), False)


def zero_link_ok(f: Finger, j: int, buildable: bool) -> bool:
    """Whether link j of finger f may be 0 mm: never the fingertip; under
    buildable rules only between two different axes square to the finger,
    and never next to another 0 mm link."""
    n = len(f.joints)
    if j >= n - 1:
        return False
    if not buildable:
        return True
    if not mt.crossing_ok(axis_unit(f.joints[j].axis), axis_unit(f.joints[j + 1].axis)):
        return False
    return not ((j > 0 and f.joints[j - 1].length == 0) or (j + 1 < n - 1 and f.joints[j + 1].length == 0))


def length_floor_mm(f: Finger, j: int, buildable: bool) -> int:
    """The shortest non-zero length link j may have: before a bend servo,
    the length at which it turns (`turned_servo`)."""
    n = len(f.joints)
    if not buildable:
        return TIP_MIN_MM if j == n - 1 else LINK_MIN_MM
    a = tuple(f.joints[j].axis)
    if j == n - 1:
        return _tip_floor(a, turned_servo(f, j))
    nxt = f.joints[j + 1]
    b = tuple(nxt.axis)
    return _link_floor(a, b, turned_servo(f, j), nxt.type != "sliding" and _is_bend(b))


def allowed_lengths(f: Finger, j: int, buildable: bool, max_mm: int = LINK_MAX_MM) -> List[int]:
    """Every length link j of finger f may take, ascending, given its
    neighbours: the floor to `max_mm` on the grid, and 0 where allowed."""
    out = list(range(length_floor_mm(f, j, buildable), max_mm + 1, GRID_MM))
    if zero_link_ok(f, j, buildable):
        out = [0] + out
    return out


def check(hand: Hand, rules: Optional[Rules] = None) -> List[str]:
    """Every way `hand` breaks the grammar or `rules` (default: the grammar's
    own limits). Empty for a valid hand. Sampling and the operators never
    produce a hand with a violation; this is for tests, imports and the
    viewer's status line. Geometry (motors clear of each other, of the mount
    and of the palm) is the clearance check, `viability.c1_self_overlap`."""
    r = rules or NO_RULES
    out: List[str] = []
    n = len(hand.fingers)
    if not (r.min_fingers <= n <= r.max_fingers):
        out.append(f"{n} fingers, allowed {r.min_fingers}-{r.max_fingers}")
    npj = len(hand.palm_joints)
    if npj > r.max_palm_joints:
        out.append(f"{npj} palm joints, allowed at most {r.max_palm_joints}")
    for i, f in enumerate(hand.fingers):
        tag = f"finger {i}"
        if not all(_is_int(v) for v in (f.y, f.z, f.facing, f.tilt)):
            out.append(f"{tag}: mount values must be integers (mm, degrees)")
            continue
        if f.facing % GRID_DEG or f.tilt % GRID_DEG:
            out.append(f"{tag}: facing/tilt off the {GRID_DEG} degree grid")
        if not (0 <= f.facing < 360):
            out.append(f"{tag}: facing {f.facing} outside [0, 360)")
        if not (TILT_RANGE_DEG[0] <= f.tilt <= TILT_RANGE_DEG[1]):
            out.append(f"{tag}: tilt {f.tilt} outside {TILT_RANGE_DEG[0]} to {TILT_RANGE_DEG[1]}")
        if f.tilt and bend_first(f):
            out.append(f"{tag}: tilt {f.tilt} with a bend first (the bend already turns the finger out of "
                       f"the plate; tilt 0)")
        if f.y % r.base_pitch_mm or f.z % r.base_pitch_mm:
            out.append(f"{tag}: base ({f.y}, {f.z}) off the {r.base_pitch_mm} mm grid")
        d = base_distance_mm(f)
        lo, hi = r.base_distance_mm
        if not (lo <= d <= hi):
            out.append(f"{tag}: base {d:.1f} mm from the palm centre, allowed {lo}-{hi}")
        if r.outward_facing and not faces_outward(f):
            out.append(f"{tag}: faces back across the palm")
        nj = len(f.joints)
        if not (1 <= nj <= r.max_joints):
            out.append(f"{tag}: {nj} joints, allowed 1-{r.max_joints}")
        bad_axis = False
        for j, jt in enumerate(f.joints):
            jtag = f"{tag} joint {j}"
            if jt.type not in JOINT_TYPES:
                out.append(f"{jtag}: unknown type {jt.type!r}")
            elif jt.type not in r.joint_types:
                out.append(f"{jtag}: {jt.type} joints not allowed")
            if jt.type == "coupled" and (j == 0 or f.joints[j - 1].type not in REVOLUTE_TYPES):
                out.append(f"{jtag}: a coupled joint must follow a hinge or coupled joint in its finger")
            az, el = jt.axis
            if not (_is_int(az) and _is_int(el)) or az % GRID_DEG or el % GRID_DEG:
                out.append(f"{jtag}: axis off the {GRID_DEG} degree grid")
                bad_axis = True
            elif canonical_axis(az, el) != (az, el):
                out.append(f"{jtag}: axis {jt.axis} not canonical ({canonical_axis(az, el)})")
        if not bad_axis:
            for j, jt in enumerate(f.joints):
                allowed = allowed_lengths(f, j, r.buildable_links, r.max_link_mm)
                if not _is_int(jt.length) or jt.length not in allowed:
                    lo_mm = length_floor_mm(f, j, r.buildable_links)
                    zero = "0 or " if zero_link_ok(f, j, r.buildable_links) else ""
                    out.append(f"{tag} joint {j}: link length {jt.length} mm not allowed "
                               f"({'fingertip ' if j == nj - 1 else ''}{zero}{lo_mm}-{r.max_link_mm} mm "
                               f"in {GRID_MM} mm steps)")
        if finger_length_mm(f) > r.max_finger_length_mm:
            out.append(f"{tag}: {finger_length_mm(f)} mm long, allowed {r.max_finger_length_mm}")
    for k, p in enumerate(hand.palm_joints):
        if not (_is_int(p.facing) and _is_int(p.offset)):
            out.append(f"palm joint {k}: facing and offset must be integers")
            continue
        if p.facing % GRID_DEG or not (0 <= p.facing < 360):
            out.append(f"palm joint {k}: facing {p.facing} off the {GRID_DEG} degree grid or outside [0, 360)")
        if p.offset % GRID_MM:
            out.append(f"palm joint {k}: offset {p.offset} off the {GRID_MM} mm grid")
        if p.offset < MIN_CUT_OFFSET_MM:
            out.append(f"palm joint {k}: cut {p.offset} mm from the palm centre, through the mount "
                       f"(at least {MIN_CUT_OFFSET_MM})")
        moving = sum(1 for i in range(n) if finger_section(hand, i) == 0)
        if moving == 0:
            out.append(f"palm joint {k}: no finger beyond the cut")
        if moving == n:
            out.append(f"palm joint {k}: no finger left on the main palm")
    if r.mount_at_centre and n and not out:
        # the mount sits at the palm's centre (Martin, 2026-10-16): so no two hands differ only in where it sits
        from .derive import palm_centre_mm      # derive builds on this module
        cy, cz = palm_centre_mm(hand)
        h = r.base_pitch_mm / 2.0
        if not (-h <= cy < h and -h <= cz < h):
            out.append(f"palm centre ({cy:.1f}, {cz:.1f}) mm from the mount; the mount sits at the palm's centre, "
                       f"within half a base step ({h:g} mm)")
    return out


def is_valid(hand: Hand, rules: Optional[Rules] = None) -> bool:
    return not check(hand, rules)


def follows(hand: Hand, rules: Rules) -> bool:
    return not check(hand, rules)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def with_finger(hand: Hand, i: int, finger: Finger) -> Hand:
    fs = list(hand.fingers)
    fs[i] = finger
    return replace(hand, fingers=tuple(fs))


def with_joint(hand: Hand, i: int, j: int, joint: Joint) -> Hand:
    f = hand.fingers[i]
    js = list(f.joints)
    js[j] = joint
    return with_finger(hand, i, replace(f, joints=tuple(js)))


def shifted(hand: Hand, dy: int, dz: int) -> Hand:
    """The same hand with the mount moved by (-dy, -dz): every base moved by
    (dy, dz), the cut with them (its offset to the nearest 10 mm)."""
    fingers = tuple(replace(f, y=f.y + dy, z=f.z + dz) for f in hand.fingers)
    cuts = tuple(replace(p, offset=int(round((p.offset + float(cut_normal_yz(p) @ (dy, dz))) / GRID_MM)) * GRID_MM)
                 for p in hand.palm_joints)
    return replace(hand, fingers=fingers, palm_joints=cuts)


def drop_empty_cut(hand: Hand) -> Hand:
    """Remove the cut if a side has no finger (a palm part without a finger
    is not a palm)."""
    if not hand.palm_joints:
        return hand
    sides = {finger_section(hand, i) for i in range(len(hand.fingers))}
    return hand if sides == {-1, 0} else replace(hand, palm_joints=())


def parameter_count(hand: Hand) -> Dict[str, int]:
    """Numbers per hand: 4 per finger (y, z, facing, tilt), 3 per joint (two
    axis angles and the link length; the type is a choice, not a number), 2
    per palm cut (facing, offset)."""
    n_f = len(hand.fingers)
    n_j = sum(len(f.joints) for f in hand.fingers)
    n_p = len(hand.palm_joints)
    return {"fingers": n_f, "joints": n_j, "palm_joints": n_p,
            "numbers": 4 * n_f + 3 * n_j + 2 * n_p,
            "choices": n_j}   # joint types


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def hand_to_dict(hand: Hand) -> dict:
    return {
        "grammar": GRAMMAR_VERSION,
        "fingers": [
            {"y": f.y, "z": f.z, "facing": f.facing, "tilt": f.tilt,
             "joints": [{"type": j.type, "axis": list(j.axis), "length": j.length} for j in f.joints]}
            for f in hand.fingers
        ],
        "palm_joints": [{"facing": p.facing, "offset": p.offset} for p in hand.palm_joints],
    }


def hand_from_dict(d: dict) -> Hand:
    if d.get("grammar") != GRAMMAR_VERSION:
        raise ValueError(f"hand written by grammar {d.get('grammar')!r}, this checkout reads {GRAMMAR_VERSION!r}")
    fingers = tuple(
        Finger(y=int(f["y"]), z=int(f["z"]), facing=int(f["facing"]), tilt=int(f["tilt"]),
               joints=tuple(Joint(type=str(j["type"]), axis=(int(j["axis"][0]), int(j["axis"][1])),
                                  length=int(j["length"])) for j in f["joints"]))
        for f in d["fingers"]
    )
    palm = tuple(PalmJoint(facing=int(p["facing"]), offset=int(p["offset"])) for p in d.get("palm_joints", []))
    return Hand(fingers=fingers, palm_joints=palm)


def hand_to_json(hand: Hand) -> str:
    return json.dumps(hand_to_dict(hand), sort_keys=True, separators=(",", ":"))


def hand_from_json(text: str) -> Hand:
    return hand_from_dict(json.loads(text))
