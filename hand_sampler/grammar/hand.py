"""The hand grammar's genotype, its constants and its rules
(`project-notes/grammar/GRAMMAR-LOCK-2026-10-07.md`).

A hand is a flat palm plate with 2-6 fingers. Everything is stored on the
fine grid as integers: millimetres for positions and lengths, degrees
(multiples of 5) for angles.

Frames
------
Palm frame: origin at the wrist centre (the anchor, where the hand mounts),
x = the palm normal on the grasp side, y = across the palm, z = from the
wrist toward the fingers. The plate's mid-plane is x = 0.

Every direction is two angles, `direction(az, el) = (sin el, cos el sin az,
cos el cos az)`: `az` turns in the y-z plane from +z toward +y, `el` lifts
toward +x.

- A finger leaves its mount (y, z) along `direction(facing, tilt)` in the
  palm frame. Its link frame: x along the finger, y = the closing direction
  (+x of the palm turned by the tilt), z = x cross y (the flexion axis, in the
  plate).
- Links are straight at the zero pose, so every link of a finger has that
  frame at q = 0, and a joint's axis is `direction(az, el)` in it: (0, 0) is
  z (flexion), (90, 0) is y (abduction), el = 90 is x (roll). An axis is a
  line; its sign is derived (`derive.joint_sign`), so az is kept in
  [0, 180) and the roll axis is written (0, 90).
- A palm joint's hinge passes through (y, z) on the plate along
  `direction(az, el)` in the palm frame.

Joint j of a finger sits at the start of link j; link j runs to joint j + 1,
and the last link to the fingertip.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

GRAMMAR_VERSION = "hand_grammar/1.0"

# --------------------------------------------------------------------------
# Constants (GRAMMAR-LOCK items in brackets)
# --------------------------------------------------------------------------

# Links: one rounded-box cross-section for every link of every hand [6].
LINK_WIDTH_MM = 19.0        # along the link's z (the flexion axis)
LINK_HEIGHT_MM = 18.0       # along the link's y (the closing direction)
LINK_RADIUS_MM = 6.0        # corner radius
# Link lengths [8]: 0 mm (two joints at one point) or 15-90 mm; the fingertip
# link 10-90 mm, never 0.
LINK_MIN_MM = 15
TIP_MIN_MM = 10
LINK_MAX_MM = 90

# Palm [1]: a flat plate, the convex hull of the finger bases plus a heel disc
# at the wrist centre. The anchor and the disc are defined here only.
PALM_THICKNESS_MM = 37.0    # palm-and-axis-study.md: median of 15 hands
PALM_ANCHOR_MM = (0, 0)     # the wrist centre, as (y, z) on the plate
PALM_DISC_RADIUS_MM = 20.0  # the heel disc at the anchor
BASE_RIM_MM = LINK_WIDTH_MM / 2.0  # each finger base and hinge adds a disc of half a link width
PALM_OUTLINE_MAX_VERTICES = 32     # a GPU convex hull: 2 x 32 vertices, 34 faces

# Finger bases: distance from the wrist centre, in the plate (re-measured from
# the wrist on the conformed commercial hands: smallest 14.6 mm (ARMS thumb)
# x 0.9 and largest 141.2 mm (Tesollo's middle finger) x 1.1, rounded out to
# 5 mm; GRAMMAR-LOCK item 21).
BASE_DISTANCE_MM = (10, 160)
# Palm joints: the hinge's distance from the wrist centre, in the plate (the
# five commercial palm joints' hinges, 20.0 mm (ARMS CMC5) x 0.9 and 98.1 mm
# (Shadow LFJ5) x 1.1, rounded out to 5 mm).
PALM_HINGE_DISTANCE_MM = (15, 110)
# Fingers sit on the palm's rim (item 24): a base may lie at most this far
# inside the convex hull of its plate's points (the heel disc, the hinges and
# the bases on that plate), never further in. Calibrated on the commercial
# hands: the smallest whole millimetre every one of them passes (Allegro's
# thumb, 8.0 mm inside the hull of its heel disc and index base), except Dex3,
# whose thumb sits 22.5 mm inside (reported, not accommodated: > 10 mm).
RIM_TOLERANCE_MM = 8
HEEL_POINTS = 32       # the heel disc as a regular polygon for the rim rule
# Finger tilt out of the plate (commercial extremes x 1.1, rounded out to the
# 5 degree grid: Barrett's thumb -25 and DClaw's fingers +90, which is the
# most a tilt can be). Facing covers the full circle.
TILT_RANGE_DEG = (-30, 90)

# Joint ranges [5], degrees; sliding joints in millimetres (Dex1's jaws,
# -20 to +24.5 mm, the only commercial sliding joints, rounded out).
KINDS = ("flexion", "abduction", "roll")
KIND_RANGE_DEG = {"flexion": (-30, 90), "abduction": (-30, 30), "roll": (-90, 90)}
PALM_RANGE_DEG = (-30, 30)
SLIDING_RANGE_MM = (-20, 25)
COUPLING_RATIO = 1.1        # [7b]: a coupled joint follows the joint before it, offset 0

# The random draw's kind mix (palm-and-axis-study.md, 263 commercial joints).
KIND_MIX = {"flexion": 0.69, "abduction": 0.27, "roll": 0.04}
KIND_AXIS = {"flexion": (0, 0), "abduction": (90, 0), "roll": (0, 90)}

JOINT_TYPES = ("hinge", "coupled", "sliding")   # [7]; no continuous joints
REVOLUTE_TYPES = ("hinge", "coupled")

# Steps [17]: one rule for every parameter.
STEPS = {"coarse": (10, 30), "fine": (1, 5)}    # (mm, degrees)
GRID_MM, GRID_DEG = STEPS["fine"]

# Physical properties for every hand (Kushal's uniform values) [14].
JOINT_EFFORT_NM = 0.5
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
    length: int = 40                     # mm, the link after this joint


@dataclass(frozen=True)
class Finger:
    y: int                               # mm, base position on the plate
    z: int
    facing: int = 0                      # degrees, in the plate, from +z toward +y
    tilt: int = 0                        # degrees, out of the plate toward the grasp side
    joints: Tuple[Joint, ...] = (Joint(),)
    palm_joint: int = -1                 # -1: on the main palm; else the palm joint it sits on


@dataclass(frozen=True)
class PalmJoint:
    y: int                               # mm, a point of the hinge line, on the plate
    z: int
    axis: Tuple[int, int] = (0, 0)       # (az, el), degrees, in the palm frame


@dataclass(frozen=True)
class Hand:
    fingers: Tuple[Finger, ...]
    palm_joints: Tuple[PalmJoint, ...] = ()


def finger_length_mm(f: Finger) -> int:
    return sum(j.length for j in f.joints)


def base_distance_mm(f: Finger) -> float:
    return math.hypot(f.y - PALM_ANCHOR_MM[0], f.z - PALM_ANCHOR_MM[1])


# --------------------------------------------------------------------------
# Canonical angles
# --------------------------------------------------------------------------


def canonical_axis(az: int, el: int) -> Tuple[int, int]:
    """The canonical (az, el) of the line through `direction(az, el)`:
    az in [0, 180), el in [-90, 90], the roll axis as (0, 90)."""
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
    grammar's own limits are the widest allowed values; a rule set may only
    tighten them. `EVOLUTION_RULES` (the simulator's, shown as "Evolution
    Rules") differ from the grammar (`NO_RULES`) only in the joint types."""

    joint_types: Tuple[str, ...] = JOINT_TYPES
    min_fingers: int = 2
    max_fingers: int = 6
    max_joints: int = 5
    max_palm_joints: int = 6
    max_finger_length_mm: int = 250
    min_spacing_mm: int = 19
    base_distance_mm: Tuple[int, int] = BASE_DISTANCE_MM

    def clamp(self) -> "Rules":
        """The same rules, every field inside the grammar's own limits."""
        g = NO_RULES
        types = ("hinge",) + tuple(t for t in ("coupled", "sliding") if t in self.joint_types)
        max_f = max(g.min_fingers, min(int(self.max_fingers), g.max_fingers))
        lo = max(int(self.base_distance_mm[0]), g.base_distance_mm[0])
        hi = min(int(self.base_distance_mm[1]), g.base_distance_mm[1])
        return Rules(
            joint_types=types, min_fingers=g.min_fingers, max_fingers=max_f,
            max_joints=max(1, min(int(self.max_joints), g.max_joints)),
            max_palm_joints=max(0, min(int(self.max_palm_joints), g.max_palm_joints)),
            max_finger_length_mm=max(TIP_MIN_MM, min(int(self.max_finger_length_mm), g.max_finger_length_mm)),
            min_spacing_mm=max(int(self.min_spacing_mm), g.min_spacing_mm),
            base_distance_mm=(lo, max(lo, hi)),
        )


NO_RULES = Rules()
EVOLUTION_RULES = Rules(joint_types=("hinge", "coupled"))
SIMULATOR = EVOLUTION_RULES


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def link_length_ok(length: int, is_tip: bool, max_mm: int = LINK_MAX_MM) -> bool:
    if is_tip:
        return TIP_MIN_MM <= length <= max_mm
    return length == 0 or LINK_MIN_MM <= length <= max_mm


def check(hand: Hand, rules: Optional[Rules] = None, rim: bool = True) -> List[str]:
    """Every way `hand` breaks the grammar or `rules` (default: the grammar's
    own limits). Empty for a valid hand. Sampling and the operators never
    produce a hand with a violation; this is for tests, imports and the
    viewer's status line. `rim=False` skips the rim rule (the simulator can
    build a commercial hand that breaks it, e.g. Dex3)."""
    r = rules or NO_RULES
    out: List[str] = []
    n = len(hand.fingers)
    if not (r.min_fingers <= n <= r.max_fingers):
        out.append(f"{n} fingers, allowed {r.min_fingers}-{r.max_fingers}")
    npj = len(hand.palm_joints)
    if npj > r.max_palm_joints:
        out.append(f"{npj} palm joints, allowed at most {r.max_palm_joints}")
    users = [0] * npj
    for i, f in enumerate(hand.fingers):
        tag = f"finger {i}"
        if not all(_is_int(v) for v in (f.y, f.z, f.facing, f.tilt, f.palm_joint)):
            out.append(f"{tag}: mount values must be integers (mm, degrees)")
            continue
        if f.facing % GRID_DEG or f.tilt % GRID_DEG:
            out.append(f"{tag}: facing/tilt off the {GRID_DEG} degree grid")
        if not (0 <= f.facing < 360):
            out.append(f"{tag}: facing {f.facing} outside [0, 360)")
        if not (TILT_RANGE_DEG[0] <= f.tilt <= TILT_RANGE_DEG[1]):
            out.append(f"{tag}: tilt {f.tilt} outside {TILT_RANGE_DEG[0]} to {TILT_RANGE_DEG[1]}")
        d = base_distance_mm(f)
        lo, hi = r.base_distance_mm
        if not (lo <= d <= hi):
            out.append(f"{tag}: base {d:.1f} mm from the wrist centre, allowed {lo}-{hi}")
        if f.palm_joint != -1:
            if not (0 <= f.palm_joint < npj):
                out.append(f"{tag}: palm joint {f.palm_joint} does not exist")
            else:
                users[f.palm_joint] += 1
        nj = len(f.joints)
        if not (1 <= nj <= r.max_joints):
            out.append(f"{tag}: {nj} joints, allowed 1-{r.max_joints}")
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
            elif canonical_axis(az, el) != (az, el):
                out.append(f"{jtag}: axis {jt.axis} not canonical ({canonical_axis(az, el)})")
            if not _is_int(jt.length) or not link_length_ok(jt.length, j == nj - 1):
                out.append(f"{jtag}: link length {jt.length} mm not allowed"
                           f" ({'fingertip 10-90' if j == nj - 1 else '0 or 15-90'} mm)")
        if finger_length_mm(f) > r.max_finger_length_mm:
            out.append(f"{tag}: {finger_length_mm(f)} mm long, allowed {r.max_finger_length_mm}")
    for k, u in enumerate(users):
        if u == 0:
            out.append(f"palm joint {k} carries no finger")
    for k, p in enumerate(hand.palm_joints):
        if not (_is_int(p.y) and _is_int(p.z)):
            out.append(f"palm joint {k}: hinge position must be integers (mm)")
        elif not (PALM_HINGE_DISTANCE_MM[0] <= math.hypot(p.y - PALM_ANCHOR_MM[0], p.z - PALM_ANCHOR_MM[1])
                  <= PALM_HINGE_DISTANCE_MM[1]):
            out.append(f"palm joint {k}: hinge {math.hypot(p.y, p.z):.1f} mm from the wrist centre, allowed "
                       f"{PALM_HINGE_DISTANCE_MM[0]}-{PALM_HINGE_DISTANCE_MM[1]}")
        az, el = p.axis
        if not (_is_int(az) and _is_int(el)) or az % GRID_DEG or el % GRID_DEG or canonical_axis(az, el) != (az, el):
            out.append(f"palm joint {k}: axis {p.axis} off the grid or not canonical")
    if rim and all(0 <= f.palm_joint < npj or f.palm_joint == -1 for f in hand.fingers):
        for i, d in enumerate(base_insets_mm(hand)):
            if d > RIM_TOLERANCE_MM + 1e-9:
                out.append(f"finger {i}: base {d:.1f} mm inside its palm plate, not on the rim "
                           f"(allowed {RIM_TOLERANCE_MM:g} mm)")
    for a in range(n):
        for b in range(a + 1, n):
            fa, fb = hand.fingers[a], hand.fingers[b]
            d = math.hypot(fa.y - fb.y, fa.z - fb.z)
            if d < r.min_spacing_mm:
                out.append(f"fingers {a} and {b}: bases {d:.1f} mm apart, need {r.min_spacing_mm}")
    return out


def _hull2d(pts: np.ndarray) -> np.ndarray:
    """Convex hull, counter-clockwise, of 2D points (monotone chain)."""
    P = sorted(set((round(float(x), 9), round(float(y), 9)) for x, y in pts))
    if len(P) <= 2:
        return np.array(P, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for q in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def plate_rim_points_mm(hand: Hand, k: int) -> np.ndarray:
    """The points whose convex hull is plate k's core for the rim rule: the
    main plate (k = -1) is the heel disc, the hinges and its fingers' bases; a
    palm joint's section is its hinge and its fingers' bases."""
    if k < 0:
        t = np.linspace(0.0, 2.0 * math.pi, HEEL_POINTS, endpoint=False)
        pts = [np.stack([PALM_ANCHOR_MM[0] + PALM_DISC_RADIUS_MM * np.cos(t),
                         PALM_ANCHOR_MM[1] + PALM_DISC_RADIUS_MM * np.sin(t)], axis=1)]
        pts += [np.array([[p.y, p.z]], dtype=float) for p in hand.palm_joints]
    else:
        p = hand.palm_joints[k]
        pts = [np.array([[p.y, p.z]], dtype=float)]
    pts += [np.array([[f.y, f.z]], dtype=float) for f in hand.fingers if f.palm_joint == k]
    return np.concatenate(pts, axis=0)


def base_insets_mm(hand: Hand) -> List[float]:
    """Per finger, how far its base lies inside the convex hull of its plate's
    rim points (0 on the hull's boundary)."""
    hulls: Dict[int, np.ndarray] = {}
    out = []
    for f in hand.fingers:
        k = f.palm_joint if 0 <= f.palm_joint < len(hand.palm_joints) else -1
        if k not in hulls:
            hulls[k] = _hull2d(plate_rim_points_mm(hand, k))
        H = hulls[k]
        if len(H) < 3:
            out.append(0.0)
            continue
        q = np.array([f.y, f.z], dtype=float)
        e = np.roll(H, -1, axis=0) - H
        n = np.stack([-e[:, 1], e[:, 0]], axis=1) / np.linalg.norm(e, axis=1)[:, None]   # inward for CCW
        d = ((q[None] - H) * n).sum(axis=1)
        out.append(max(0.0, float(d.min())))
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


def drop_unused_palm_joints(hand: Hand) -> Hand:
    """Remove every palm joint that carries no finger (a palm joint without a
    finger is not a palm) and renumber the rest in their order."""
    used = sorted({f.palm_joint for f in hand.fingers if f.palm_joint >= 0})
    remap = {old: new for new, old in enumerate(used)}
    fingers = tuple(replace(f, palm_joint=remap.get(f.palm_joint, -1)) for f in hand.fingers)
    return Hand(fingers=fingers, palm_joints=tuple(hand.palm_joints[k] for k in used))


def parameter_count(hand: Hand) -> Dict[str, int]:
    """Numbers per hand: 4 per finger (y, z, facing, tilt), 3 per joint (two
    axis angles and the link length; the type is a choice, not a number), 4
    per palm joint (hinge y, z and two axis angles)."""
    n_f = len(hand.fingers)
    n_j = sum(len(f.joints) for f in hand.fingers)
    n_p = len(hand.palm_joints)
    return {"fingers": n_f, "joints": n_j, "palm_joints": n_p,
            "numbers": 4 * n_f + 3 * n_j + 4 * n_p,
            "choices": n_j + n_f}   # joint types, and which palm part each finger sits on


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def hand_to_dict(hand: Hand) -> dict:
    return {
        "grammar": GRAMMAR_VERSION,
        "fingers": [
            {"y": f.y, "z": f.z, "facing": f.facing, "tilt": f.tilt, "palm_joint": f.palm_joint,
             "joints": [{"type": j.type, "axis": list(j.axis), "length": j.length} for j in f.joints]}
            for f in hand.fingers
        ],
        "palm_joints": [{"y": p.y, "z": p.z, "axis": list(p.axis)} for p in hand.palm_joints],
    }


def hand_from_dict(d: dict) -> Hand:
    if d.get("grammar") != GRAMMAR_VERSION:
        raise ValueError(f"hand written by grammar {d.get('grammar')!r}, this checkout reads {GRAMMAR_VERSION!r}")
    fingers = tuple(
        Finger(y=int(f["y"]), z=int(f["z"]), facing=int(f["facing"]), tilt=int(f["tilt"]),
               palm_joint=int(f.get("palm_joint", -1)),
               joints=tuple(Joint(type=str(j["type"]), axis=(int(j["axis"][0]), int(j["axis"][1])),
                                  length=int(j["length"])) for j in f["joints"]))
        for f in d["fingers"]
    )
    palm = tuple(PalmJoint(y=int(p["y"]), z=int(p["z"]), axis=(int(p["axis"][0]), int(p["axis"][1])))
                 for p in d.get("palm_joints", []))
    return Hand(fingers=fingers, palm_joints=palm)


def hand_to_json(hand: Hand) -> str:
    return json.dumps(hand_to_dict(hand), sort_keys=True, separators=(",", ":"))


def hand_from_json(text: str) -> Hand:
    return hand_from_dict(json.loads(text))
