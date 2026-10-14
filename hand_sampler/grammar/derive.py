"""From a `hand.Hand` to kinematics and geometry: derived joint kinds, signs
and ranges, forward kinematics (batched), the `KinematicModel`, the motor
and link solids, the palm outlines and the cut, the object's start point.
Numpy only. Geometry is in millimetres inside, metres at the interfaces.

Derived per joint:
- the link frame (tilt-defined): x = e = cos(tilt) u + sin(tilt) n along the
  finger (u its direction in the plate, n the palm normal), y = c =
  cos(tilt) n - sin(tilt) u (the closing direction), z = x cross y (the bend
  axis, in the plate). It depends on the finger alone.
- kind: the nearest of flexion (bend, link z), abduction (spread, link y)
  and roll (twist, link x); ties go to flexion, then abduction.
- sign: + moves the fingertip toward +y of the link frame (the closing
  direction, so every bend closes toward the palm's front) if the joint can
  move it that way at all; otherwise + moves it toward +y of the palm, then
  +z, then +x; a pure twist turns right-handed about the link. The palm
  joint's + lifts the part beyond the cut toward the grasp side.
- range: the kind's global range; a coupled joint takes 1.1 x its source's
  range; the palm joint +/-30 degrees; sliding joints -20 to +25 mm.

Derived geometry (one motor, `motor.py`):
- every finger's base joint sits below the front face, just deep enough
  that the housings of the motors at its base point (motor 0, and motor 1
  when link 0 is 0 mm) end flush with the face (`base_depth_mm`);
- a crossing (a 0 mm link) slides the earlier motor along its own axis
  (`motor.crossing_slide_mm`);
- solids: the main plate, the part beyond the cut, the mount, the hinge
  motor, each finger's base motor (fixed to its plate) and one box per
  link, square to its frame, around its yoke (the bracket around its
  joint's motor) and its far end (the next motor, which sits in this link,
  or the fingertip pad): links are servo-sized;
- the palm outline: the convex hull of the mount disc, every base motor's
  housing and the hinge motor's housing; a cut splits it into two convex
  plates, each half a seam short of the line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import motor as mt
from .hand import (
    COUPLING_RATIO,
    GRID_MM,
    KIND_RANGE_DEG,
    LINK_DENSITY_KG_M3,
    MOUNT_DEPTH_MM,
    MOUNT_POINTS,
    MOUNT_RADIUS_MM,
    PALM_OUTLINE_MAX_VERTICES,
    PALM_RANGE_DEG,
    PALM_THICKNESS_MM,
    PLATE_HALF_MM,
    REVOLUTE_TYPES,
    SEAM_MM,
    SLIDING_RANGE_MM,
    Finger,
    Hand,
    axis_unit,
    cut_direction_yz,
    cut_normal_yz,
    finger_section,
)
from .kinematics import AffineCoupling, Body, Frame, Joint as KJoint, KinematicModel, Pose, validate

DEG = math.pi / 180.0
MM = 1e-3
EPS = 1e-9

START_CURL = 0.35
"""The episode start pose: every flexion joint at this fraction of its upper
limit, every other joint at 0, coupled joints following their source."""

CLEARANCE_MM = mt.WALL_MM
"""Solids that do not touch by construction keep one printed wall apart."""


# --------------------------------------------------------------------------
# Directions and frames
# --------------------------------------------------------------------------


def direction(az_deg: float, el_deg: float) -> np.ndarray:
    a, e = az_deg * DEG, el_deg * DEG
    return np.array([math.sin(e), math.cos(e) * math.sin(a), math.cos(e) * math.cos(a)])


def frame_of(facing_deg: float, tilt_deg: float) -> np.ndarray:
    """3x3, columns = a finger's link frame in the palm frame at q = 0: x
    along the finger (`direction(facing, tilt)`), y the closing direction
    (+x of the palm turned by the tilt), z = x cross y (in the plate)."""
    fa, t = facing_deg * DEG, tilt_deg * DEG
    x = np.array([math.sin(t), math.cos(t) * math.sin(fa), math.cos(t) * math.cos(fa)])
    y = np.array([math.cos(t), -math.sin(t) * math.sin(fa), -math.sin(t) * math.cos(fa)])
    z = np.array([0.0, math.cos(fa), -math.sin(fa)])
    return np.column_stack([x, y, z])


def finger_frame(f: Finger) -> np.ndarray:
    """3x3, columns = the finger's link frame (x along the finger, y the
    closing direction, z the bend axis) in the palm frame, at q = 0."""
    return frame_of(f.facing, f.tilt)


def finger_direction(f: Finger) -> np.ndarray:
    """The finger's direction at q = 0 (palm frame): `direction(facing, tilt)`."""
    return direction(f.facing, f.tilt)


def kind_of(axis: Sequence[float]) -> str:
    a = np.abs(np.asarray(axis, dtype=float))
    # nearest kind by angle; ties: flexion, abduction, roll
    best = max((a[2], 2), (a[1], 1), (a[0], 0), key=lambda t: (round(t[0], 12), t[1]))
    return {2: "flexion", 1: "abduction", 0: "roll"}[best[1]]


def joint_kind(hand: Hand, i: int, j: int) -> str:
    jt = hand.fingers[i].joints[j]
    if jt.type == "sliding":
        return "sliding"
    return kind_of(direction(*jt.axis))


def _sign_from(v_closing: float, v_palm: np.ndarray, fallback: float) -> float:
    if abs(v_closing) > EPS:
        return 1.0 if v_closing > 0 else -1.0
    for c in (1, 2, 0):
        if abs(v_palm[c]) > EPS:
            return 1.0 if v_palm[c] > 0 else -1.0
    return 1.0 if fallback >= 0 else -1.0


def axis_sign(a: np.ndarray, B: np.ndarray, sliding: bool = False) -> float:
    """The derived sign of a joint with unit axis `a` (link frame) on a
    finger whose zero-pose link frame is `B` (palm frame)."""
    v = a if sliding else np.array([0.0, a[2], -a[1]])   # the tip's velocity (a x link x for a hinge)
    return _sign_from(v[1], B @ v, a[0])


def joint_sign(hand: Hand, i: int, j: int) -> float:
    f = hand.fingers[i]
    jt = f.joints[j]
    return axis_sign(direction(*jt.axis), finger_frame(f), jt.type == "sliding")


def joint_axis(hand: Hand, i: int, j: int) -> np.ndarray:
    """The signed unit axis of finger i's joint j, in its link frame."""
    return joint_sign(hand, i, j) * direction(*hand.fingers[i].joints[j].axis)


def joint_limits(hand: Hand, i: int, j: int) -> Tuple[float, float]:
    """Radians (metres for a sliding joint)."""
    jt = hand.fingers[i].joints[j]
    if jt.type == "sliding":
        return (SLIDING_RANGE_MM[0] * MM, SLIDING_RANGE_MM[1] * MM)
    if jt.type == "coupled":
        lo, hi = joint_limits(hand, i, j - 1)
        return (COUPLING_RATIO * lo, COUPLING_RATIO * hi)
    lo, hi = KIND_RANGE_DEG[joint_kind(hand, i, j)]
    return (lo * DEG, hi * DEG)


def palm_limits() -> Tuple[float, float]:
    return (PALM_RANGE_DEG[0] * DEG, PALM_RANGE_DEG[1] * DEG)


# --------------------------------------------------------------------------
# Small geometry helpers
# --------------------------------------------------------------------------


@dataclass
class Box:
    """An oriented box: centre (3,), axes (3x3, columns), half sizes (3,)."""

    centre: np.ndarray
    axes: np.ndarray
    half: np.ndarray

    def corners(self) -> np.ndarray:
        return mt.box_corners(self.centre, self.axes, self.half)

    def moved(self, T: np.ndarray, scale: float = 1.0) -> "Box":
        """The box in the frame `T` (4x4, its translation in the box's units
        times `scale`) is expressed in."""
        return Box(T[:3, :3] @ self.centre + T[:3, 3] / scale, T[:3, :3] @ self.axes, self.half)

    def grown(self, d: float) -> "Box":
        return Box(self.centre, self.axes, self.half + d)


def box_corners(centre: np.ndarray, half: np.ndarray) -> np.ndarray:
    """(8, 3) corners of an axis-aligned box."""
    return mt.box_corners(centre, np.eye(3), half)


def _hull2d(pts: np.ndarray) -> np.ndarray:
    """Convex hull (counter-clockwise, no repeated end point) of 2D points."""
    P = sorted(set((round(float(x), 9), round(float(y), 9)) for x, y in pts))
    if len(P) <= 2:
        return np.array(P, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def _simplify(poly: np.ndarray, max_vertices: int) -> np.ndarray:
    """Drop the vertex cutting the least area until at most `max_vertices`
    remain (stays convex, inside the original)."""
    poly = [p for p in poly]
    while len(poly) > max_vertices:
        n = len(poly)
        areas = []
        for k in range(n):
            a, b, c = poly[k - 1], poly[k], poly[(k + 1) % n]
            areas.append(abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])))
        del poly[int(np.argmin(areas))]
    return np.array(poly)


def _ring(centre: Sequence[float], radius: float, n: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.asarray(centre, dtype=float)[None] + np.stack([np.cos(t), np.sin(t)], axis=1) * radius


def clip_halfplane(poly: np.ndarray, n: np.ndarray, c: float) -> np.ndarray:
    """The part of the convex polygon `poly` where `p . n <= c`
    (Sutherland-Hodgman, one plane); counter-clockwise stays so."""
    out = []
    m = len(poly)
    for k in range(m):
        a, b = poly[k], poly[(k + 1) % m]
        da, db = float(a @ n) - c, float(b @ n) - c
        if da <= 0:
            out.append(a)
        if (da < 0 < db) or (db < 0 < da):
            t = da / (da - db)
            out.append(a + t * (b - a))
    if len(out) < 3:
        return np.zeros((0, 2))
    return _hull2d(np.array(out))


def line_polygon_interval(poly: np.ndarray, p0: np.ndarray, d: np.ndarray) -> Optional[Tuple[float, float]]:
    """The interval of t with p0 + t d inside the convex counter-clockwise
    polygon, or None."""
    lo, hi = -math.inf, math.inf
    m = len(poly)
    for k in range(m):
        a, b = poly[k], poly[(k + 1) % m]
        e = b - a
        nrm = np.array([e[1], -e[0]])            # outward for CCW
        num = float((p0 - a) @ nrm)
        den = float(d @ nrm)
        if abs(den) < 1e-12:
            if num > 0:
                return None
            continue
        t = -num / den
        if den > 0:
            hi = min(hi, t)
        else:
            lo = max(lo, t)
    if lo > hi:
        return None
    return lo, hi


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    x, y = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    a = cr.sum() / 2.0
    if abs(a) < 1e-12:
        return poly.mean(axis=0)
    return np.array([((x + x1) * cr).sum(), ((y + y1) * cr).sum()]) / (6.0 * a)


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def point_in_polygon(pts: np.ndarray, poly: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """`(N,)` bool: 2D points inside the convex counter-clockwise `poly`
    grown by `margin` (same units)."""
    pts = np.atleast_2d(pts)
    inside = np.ones(len(pts), dtype=bool)
    n = len(poly)
    for k in range(n):
        a, b = poly[k], poly[(k + 1) % n]
        e = b - a
        nrm = np.array([e[1], -e[0]]) / (np.linalg.norm(e) + 1e-15)   # outward for CCW
        inside &= (pts - a[None]) @ nrm <= margin
    return inside


# --------------------------------------------------------------------------
# GJK distance between convex point sets
# --------------------------------------------------------------------------


def _closest_segment(a: np.ndarray, b: np.ndarray):
    ab = b - a
    t = -float(a @ ab) / max(float(ab @ ab), 1e-30)
    if t <= 0.0:
        return a, [a]
    if t >= 1.0:
        return b, [b]
    return a + t * ab, [a, b]


def _closest_triangle(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Closest point to the origin on triangle abc (Ericson, Real-Time
    Collision Detection 5.1.5) and the vertices of the feature it lies on."""
    ab, ac, ap = b - a, c - a, -a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, [a]
    bp = -b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, [b]
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, [a, b]
    cp = -c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, [c]
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, [a, c]
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), [b, c]
    denom = 1.0 / (va + vb + vc)
    v, w = vb * denom, vc * denom
    return a + ab * v + ac * w, [a, b, c]


def _closest_on_simplex(S: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """The point of the simplex S (1-4 points) closest to the origin, and the
    smallest sub-simplex containing it (a full tetrahedron: the origin is
    inside, distance 0)."""
    k = len(S)
    if k == 1:
        return S[0], [S[0]]
    if k == 2:
        return _closest_segment(S[0], S[1])
    if k == 3:
        return _closest_triangle(S[0], S[1], S[2])
    a, b, c, d = S
    best = None
    for tri, other in (((a, b, c), d), ((a, b, d), c), ((a, c, d), b), ((b, c, d), a)):
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        s_origin = float(-tri[0] @ n)
        s_other = float((other - tri[0]) @ n)
        if s_origin * s_other < 0.0 or abs(s_other) < 1e-18:      # the origin is outside this face
            p, sub = _closest_triangle(*tri)
            dd = float(p @ p)
            if best is None or dd < best[0]:
                best = (dd, p, sub)
    if best is None:
        return np.zeros(3), list(S)
    return best[1], best[2]


def gjk_distance(A: np.ndarray, B: np.ndarray, max_iter: int = 64) -> float:
    """Euclidean distance between the convex hulls of point sets `A` and `B`
    (0 when they intersect)."""
    v = A[0] - B[0]
    S = [v]
    for _ in range(max_iter):
        vv = float(v @ v)
        if vv < 1e-20:
            return 0.0
        w = A[int(np.argmax(A @ -v))] - B[int(np.argmax(B @ v))]
        if vv - float(v @ w) <= 1e-10 * max(vv, 1e-12):
            break
        S.append(w)
        v, S = _closest_on_simplex(S)
        if len(S) == 4 or float(v @ v) < 1e-20:
            return 0.0
    return math.sqrt(float(v @ v))


# --------------------------------------------------------------------------
# A finger's motors, yokes and base (finger-local; cached)
# --------------------------------------------------------------------------


@dataclass
class FingerGeometry:
    B: np.ndarray                 # 3x3: the finger's zero-pose frame in the palm frame
    base_mm: np.ndarray           # (3,): joint 0's point, palm frame
    depth_mm: float               # joint 0 below the palm's front face
    axes: np.ndarray              # (n, 3): canonical unit joint axes, link frame
    slides_mm: np.ndarray         # (n,): each motor's slide along its own axis (crossings)
    motors: List[Box]             # motor j in joint j's frame before it turns (origin at the joint)
    yokes: List[Box]              # yoke j in link j's frame
    fars: List[Box]               # link j's far end in its frame: the next motor, or the fingertip pad
    links: List[Box]              # link j's solid in its frame: the box around its yoke and its far end
    base_joints: Tuple[int, ...]  # the joints at the base point (0, and 1 when link 0 is 0 mm)


def _pad_start_mm(a: np.ndarray) -> float:
    """Where the fingertip pad starts along the last link: past the last
    motor's swinging front and a wall (`motor.tip_floor_mm` without the pad)."""
    t = abs(float(a[0]))
    return mt.WALL_MM + mt.MOTOR_HALF_MM[0] * t + mt.front_sweep_mm(a) * math.sqrt(max(0.0, 1.0 - t * t))


def _aabb_box(points: np.ndarray) -> Box:
    c, h = mt.aabb(points)
    return Box(c, np.eye(3), h)


@lru_cache(maxsize=65536)
def finger_geometry(f: Finger) -> FingerGeometry:
    n = len(f.joints)
    B = finger_frame(f)
    axes = np.array([axis_unit(jt.axis) for jt in f.joints])
    slides = np.zeros(n)
    for j in range(n - 1):
        if f.joints[j].length == 0:
            slides[j] = -mt.crossing_slide_mm(axes[j], axes[j + 1])
    motors, yokes = [], []
    for j in range(n):
        M = mt.motor_axes(axes[j])
        motors.append(Box(mt.motor_centre(axes[j], slides[j]), M, mt.MOTOR_HALF_MM.copy()))
        yokes.append(Box(slides[j] * M[:, 0], M, mt.YOKE_HALF_MM.copy()))
    fars, links = [], []
    for j in range(n):
        yc = yokes[j].corners()
        if j < n - 1:
            nxt = motors[j + 1]
            far = Box(nxt.centre + np.array([float(f.joints[j].length), 0.0, 0.0]), nxt.axes, nxt.half)
        else:     # the pad: the yoke's cross-section, from past the motor's swing to the fingertip
            L = float(f.joints[j].length)
            x0 = min(_pad_start_mm(axes[j]), L - 1.0)
            lo, hi = yc.min(axis=0), yc.max(axis=0)
            far = Box(np.array([(x0 + L) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0]), np.eye(3),
                      np.array([(L - x0) / 2.0, (hi[1] - lo[1]) / 2.0, (hi[2] - lo[2]) / 2.0]))
        fars.append(far)
        links.append(_aabb_box(np.concatenate([yc, far.corners()])))
    base_joints = [0]
    for j in range(n - 1):
        if f.joints[j].length != 0:
            break
        base_joints.append(j + 1)
    # depth: the base motors' housings end flush with the front face
    nrm = B[0, :]                                   # the palm normal in the finger frame
    depth = max(mt.support(motors[j].centre, motors[j].axes, mt.ENCLOSURE_HALF_MM, nrm) for j in base_joints)
    base = np.array([PLATE_HALF_MM - depth, float(f.y), float(f.z)])
    return FingerGeometry(B=B, base_mm=base, depth_mm=float(depth), axes=axes, slides_mm=slides, motors=motors,
                          yokes=yokes, fars=fars, links=links, base_joints=tuple(base_joints))


def base_depth_mm(f: Finger) -> float:
    """How far finger f's base joint sits below the palm's front face: 12 mm
    for a bend first, 16.5 for a spread, 11.5 for a twist (flat fingers)."""
    return finger_geometry(f).depth_mm


def mount_point_m(f: Finger) -> np.ndarray:
    """The finger's base joint (joint 0) in the palm frame, metres."""
    return finger_geometry(f).base_mm * MM


def tip_zero_m(f: Finger) -> np.ndarray:
    """The fingertip at q = 0 (the finger is a straight line), palm frame."""
    return mount_point_m(f) + finger_direction(f) * (sum(j.length for j in f.joints) * MM)


def base_motor_boxes_mm(f: Finger) -> List[Box]:
    """The motors at finger f's base point at q = 0 (palm frame, mm):
    motor 0, and motor 1 when link 0 is 0 mm."""
    g = finger_geometry(f)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = g.B, g.base_mm
    return [g.motors[j].moved(T) for j in g.base_joints]


def link_box_mm(f: Finger, j: int) -> Box:
    """Link j's solid in its own frame (mm): the box, square to the link
    frame, around its yoke and its far end (the next motor, or the pad). The
    simulator's collider, the policy's token box and the mass all use it."""
    return finger_geometry(f).links[j]


def link_points_m(hand: Hand, i: int, j: int) -> np.ndarray:
    """(8, 3): link j's corners in its own frame, metres."""
    return link_box_mm(hand.fingers[i], j).corners() * MM


# --------------------------------------------------------------------------
# The palm: mount, outline, cut, hinge (hand-level; cached)
# --------------------------------------------------------------------------


def mount_outline_mm() -> np.ndarray:
    return _ring((0.0, 0.0), MOUNT_RADIUS_MM, MOUNT_POINTS)


def mount_points_m() -> np.ndarray:
    """The mount's prism corners (palm frame, metres): a disc behind the
    back face."""
    ring = mount_outline_mm()
    top = np.column_stack([np.full(len(ring), -PLATE_HALF_MM), ring])
    bot = np.column_stack([np.full(len(ring), -PLATE_HALF_MM - MOUNT_DEPTH_MM), ring])
    return np.concatenate([top, bot]) * MM


@dataclass
class HandGeometry:
    outline: np.ndarray                    # the uncut outline, (y, z) mm, counter-clockwise
    outlines: Dict[int, np.ndarray]        # -1: the main palm, 0: the part beyond the cut
    hinge_mm: Optional[np.ndarray] = None  # (3,): a point of the hinge axis (on the cut, mid-plane)
    hinge_axis: Optional[np.ndarray] = None   # (3,): unit, + lifts the part beyond the cut
    hinge_box: Optional[Box] = None        # the hinge motor, palm frame, mm
    hinge_clear: bool = True               # a spot on the cut clears every base motor
    base_boxes: Dict[int, List[Box]] = field(default_factory=dict)   # per finger, its base motors (palm, mm)


def _hinge_box_mm(p_yz: np.ndarray, d_yz: np.ndarray, n_yz: np.ndarray) -> Box:
    """The hinge motor at the plate point `p_yz` on the cut: its axis along
    the cut, its body on the main palm's side, in the plate's mid-plane."""
    a = np.array([0.0, d_yz[0], d_yz[1]])
    v = -np.array([0.0, n_yz[0], n_yz[1]])
    axes = np.column_stack([a, v, np.cross(a, v)])
    centre = np.array([0.0, p_yz[0], p_yz[1]]) + mt.MOTOR_CENTRE_BACK_MM * v
    return Box(centre, axes, mt.MOTOR_HALF_MM.copy())


def _footprint(box: Box) -> np.ndarray:
    """(8, 2): a box's housing projected on the plate."""
    return box.grown(mt.WALL_MM).corners()[:, 1:]


@lru_cache(maxsize=8192)
def hand_geometry(hand: Hand) -> HandGeometry:
    base_boxes = {i: base_motor_boxes_mm(f) for i, f in enumerate(hand.fingers)}
    pts = [mount_outline_mm()] + [_footprint(b) for bs in base_boxes.values() for b in bs]
    outline0 = _hull2d(np.concatenate(pts))
    if not hand.palm_joints:
        return HandGeometry(outline=_simplify(outline0, PALM_OUTLINE_MAX_VERTICES),
                            outlines={-1: _simplify(outline0, PALM_OUTLINE_MAX_VERTICES)}, base_boxes=base_boxes)
    p = hand.palm_joints[0]
    d, nrm = cut_direction_yz(p), cut_normal_yz(p)
    foot = p.offset * nrm
    span = line_polygon_interval(outline0, foot, d)
    t0, t1 = span if span is not None else (0.0, 0.0)
    tm = 0.5 * (t0 + t1)
    cands = list(range(int(math.ceil(t0 / GRID_MM)) * GRID_MM, int(math.floor(t1 / GRID_MM)) * GRID_MM + 1, GRID_MM))
    cands.sort(key=lambda t: (abs(t - tm), t))
    if not cands:
        cands = [int(round(tm / GRID_MM)) * GRID_MM]
    motor_pts = [b.corners() for bs in base_boxes.values() for b in bs]
    chosen, clear = None, False
    for t in cands:
        box = _hinge_box_mm(foot + t * d, d, nrm)
        C = box.corners()
        if all(gjk_distance(C, M) >= CLEARANCE_MM - 1e-6 for M in motor_pts):
            chosen, clear = (t, box), True
            break
    if chosen is None:
        t = cands[0]
        chosen = (t, _hinge_box_mm(foot + t * d, d, nrm))
    t, hbox = chosen
    outline = _hull2d(np.concatenate([outline0, _footprint(hbox)]))
    main = clip_halfplane(outline, nrm, p.offset - SEAM_MM / 2.0)
    section = clip_halfplane(outline, -nrm, -(p.offset + SEAM_MM / 2.0))
    if len(section) < 3:     # nothing beyond the seam: a sliver so the plate exists (the check fails such hands)
        q = foot + (SEAM_MM / 2.0 + 1.0) * nrm
        section = _hull2d(np.stack([q - d, q + d, q + d + nrm, q - d + nrm]))
    if len(main) < 3:
        main = _hull2d(mount_outline_mm())
    hinge = np.array([0.0, *(foot + t * d)])
    axis = np.array([0.0, d[0], d[1]])
    return HandGeometry(outline=_simplify(outline, PALM_OUTLINE_MAX_VERTICES),
                        outlines={-1: _simplify(main, PALM_OUTLINE_MAX_VERTICES),
                                  0: _simplify(section, PALM_OUTLINE_MAX_VERTICES)},
                        hinge_mm=hinge, hinge_axis=axis, hinge_box=hbox, hinge_clear=clear, base_boxes=base_boxes)


def palm_outlines_mm(hand: Hand) -> Dict[int, np.ndarray]:
    """Plate outlines as (y, z) polygons in mm, counter-clockwise, at most
    `PALM_OUTLINE_MAX_VERTICES` vertices: key -1 is the main palm, key 0 the
    part beyond the cut (if any)."""
    return hand_geometry(hand).outlines


def outline_mm(hand: Hand) -> np.ndarray:
    """The palm's whole outline before the cut."""
    return hand_geometry(hand).outline


def plate_points_m(outline_mm: np.ndarray, origin_mm: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """The plate's prism corners (2 x outline vertices) in metres, relative to
    the point (0, origin) of the palm frame: the outline at x = +/- half the
    palm thickness."""
    o = np.asarray(outline_mm, dtype=float) - np.asarray(origin_mm, dtype=float)[None]
    h = PLATE_HALF_MM
    top = np.column_stack([np.full(len(o), h), o])
    bot = np.column_stack([np.full(len(o), -h), o])
    return np.concatenate([top, bot], axis=0) * MM


def palm_hinge_m(hand: Hand, k: int = 0) -> np.ndarray:
    """A point of palm joint k's hinge axis (palm frame, metres): the hinge
    motor's spot on the cut, in the mid-plane."""
    return hand_geometry(hand).hinge_mm * MM


def palm_axis(hand: Hand, k: int = 0) -> np.ndarray:
    """The signed unit axis of palm joint k (palm frame): along the cut, +
    lifts the part beyond it toward the grasp side."""
    return hand_geometry(hand).hinge_axis.copy()


# --------------------------------------------------------------------------
# Joint vectors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Dof:
    name: str
    kind: str            # flexion / abduction / roll / sliding / palm
    type: str            # hinge / coupled / sliding / palm
    finger: int          # -1 for a palm joint
    index: int           # joint index in its finger, or the palm joint index
    limits: Tuple[float, float]
    source: int          # for a coupled joint: the dof it follows; else -1


def dofs(hand: Hand) -> List[Dof]:
    """Every joint of the hand in one fixed order: the palm joint first, then
    each finger's joints from the palm out."""
    out: List[Dof] = []
    for k in range(len(hand.palm_joints)):
        out.append(Dof(f"palm{k}_joint", "palm", "palm", -1, k, palm_limits(), -1))
    for i, f in enumerate(hand.fingers):
        for j, jt in enumerate(f.joints):
            src = len(out) - 1 if jt.type == "coupled" else -1
            out.append(Dof(f"f{i}_j{j}", joint_kind(hand, i, j), jt.type, i, j, joint_limits(hand, i, j), src))
    return out


def tie(hand: Hand, q: np.ndarray, ds: Optional[List[Dof]] = None) -> np.ndarray:
    """`q` (`(n,)` or `(N, n)`) with every coupled joint set to 1.1 x its source."""
    ds = ds if ds is not None else dofs(hand)
    q = np.array(q, dtype=float, copy=True)
    for k, d in enumerate(ds):
        if d.source >= 0:
            q[..., k] = COUPLING_RATIO * q[..., d.source]
    return q


def curl_q(hand: Hand, curl: float = START_CURL, ds: Optional[List[Dof]] = None) -> np.ndarray:
    """Every flexion joint at `curl` x its upper limit, every other joint at
    0, coupled joints following their source."""
    ds = ds if ds is not None else dofs(hand)
    q = np.zeros(len(ds))
    for k, d in enumerate(ds):
        if d.kind == "flexion" and d.source < 0:
            q[k] = curl * d.limits[1]
    return tie(hand, q, ds)


def start_q(hand: Hand) -> np.ndarray:
    return curl_q(hand, START_CURL)


def sample_q(hand: Hand, n: int, rng: np.random.Generator, ds: Optional[List[Dof]] = None) -> np.ndarray:
    """`(n, dofs)` joint vectors drawn uniformly over every independent
    joint's range, coupled joints following."""
    ds = ds if ds is not None else dofs(hand)
    lo = np.array([d.limits[0] for d in ds])
    hi = np.array([d.limits[1] for d in ds])
    return tie(hand, rng.uniform(lo, hi, size=(n, len(ds))), ds)


# --------------------------------------------------------------------------
# Forward kinematics (batched)
# --------------------------------------------------------------------------


def _rot_batch(axis: np.ndarray, q: np.ndarray) -> np.ndarray:
    """`(N, 3, 3)` rotations about the unit `axis` by angles `q` (`(N,)`)."""
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    s, c = np.sin(q)[:, None, None], np.cos(q)[:, None, None]
    return np.eye(3)[None] + s * K[None] + (1.0 - c) * (K @ K)[None]


@dataclass
class Pose3:
    palm: np.ndarray           # (N, P, 4, 4): palm sections (palm frame)
    links: List[np.ndarray]    # per finger (N, J, 4, 4): link frames (palm frame), after each joint turns
    tips: np.ndarray           # (N, F, 3): fingertips
    pre: List[np.ndarray] = field(default_factory=list)   # per finger (N, J, 4, 4): joint frames before they turn


def fk(hand: Hand, q: Optional[np.ndarray] = None) -> Pose3:
    """Forward kinematics in the palm frame (metres). `q`: `(dofs,)` or
    `(N, dofs)` in `dofs(hand)` order (coupled entries are recomputed), or
    None for the zero pose."""
    ds = dofs(hand)
    q = np.zeros(len(ds)) if q is None else np.asarray(q, dtype=float)
    single = q.ndim == 1
    q = tie(hand, np.atleast_2d(q), ds)
    N = q.shape[0]
    P = len(hand.palm_joints)
    palm = np.tile(np.eye(4), (N, max(P, 0), 1, 1))
    for k in range(P):
        h = palm_hinge_m(hand, k)
        R = _rot_batch(palm_axis(hand, k), q[:, k])
        palm[:, k, :3, :3] = R
        palm[:, k, :3, 3] = h - R @ h
    links: List[np.ndarray] = []
    pres: List[np.ndarray] = []
    tips = np.zeros((N, len(hand.fingers), 3))
    col = P
    for i, f in enumerate(hand.fingers):
        T = np.tile(np.eye(4), (N, 1, 1))
        T[:, :3, :3] = finger_frame(f)
        T[:, :3, 3] = mount_point_m(f)
        sec = finger_section(hand, i)
        if sec >= 0:
            T = palm[:, sec] @ T
        frames = np.zeros((N, len(f.joints), 4, 4))
        pre = np.zeros((N, len(f.joints), 4, 4))
        prev_len = 0.0
        for j, jt in enumerate(f.joints):
            T = T.copy()
            T[:, :3, 3] += T[:, :3, 0] * prev_len
            pre[:, j] = T
            a = joint_axis(hand, i, j)
            if jt.type == "sliding":
                T[:, :3, 3] += (T[:, :3, :3] @ a) * q[:, col][:, None]
            else:
                T[:, :3, :3] = T[:, :3, :3] @ _rot_batch(a, q[:, col])
            frames[:, j] = T
            prev_len = jt.length * MM
            col += 1
        links.append(frames)
        pres.append(pre)
        tips[:, i] = T[:, :3, 3] + T[:, :3, 0] * prev_len
    if single:
        return Pose3(palm=palm[0], links=[l[0] for l in links], tips=tips[0], pre=[p[0] for p in pres])
    return Pose3(palm=palm, links=links, tips=tips, pre=pres)


# --------------------------------------------------------------------------
# KinematicModel
# --------------------------------------------------------------------------


def _matrix_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    from .fk import matrix_to_rpy

    return tuple(float(v) for v in matrix_to_rpy(R))


def derive(hand: Hand, name: str = "grammar_hand") -> KinematicModel:
    """The hand as a `KinematicModel` (metres, radians): root body `palm`,
    the part beyond the cut `palm0` (hinged to the palm by `palm0_joint`),
    finger links `f{i}_link{j}` (joints `f{i}_j{j}`), fingertip frames
    `f{i}_tip`. Coupled joints are revolute joints with an `AffineCoupling`
    (1.1, 0) on the joint before them."""
    bodies = [Body("palm", palm=True)]
    joints: List[KJoint] = []
    frames: List[Frame] = []
    couplings: List[AffineCoupling] = []
    for k in range(len(hand.palm_joints)):
        bodies.append(Body(f"palm{k}", palm=True))
        joints.append(KJoint(f"palm{k}_joint", "revolute", "palm", f"palm{k}",
                             origin=Pose(xyz=tuple(float(v) for v in palm_hinge_m(hand, k))),
                             axis=tuple(float(v) for v in palm_axis(hand, k)), limits=palm_limits()))
    for i, f in enumerate(hand.fingers):
        sec = finger_section(hand, i)
        parent = "palm" if sec < 0 else f"palm{sec}"
        base = mount_point_m(f) - (palm_hinge_m(hand, sec) if sec >= 0 else 0.0)
        for j, jt in enumerate(f.joints):
            child = f"f{i}_link{j}"
            bodies.append(Body(child))
            if j == 0:
                origin = Pose(xyz=tuple(float(v) for v in base), rpy=_matrix_to_rpy(finger_frame(f)))
            else:
                origin = Pose(xyz=(f.joints[j - 1].length * MM, 0.0, 0.0))
            jtype = "prismatic" if jt.type == "sliding" else "revolute"
            joints.append(KJoint(f"f{i}_j{j}", jtype, parent, child, origin=origin,
                                 axis=tuple(float(v) for v in joint_axis(hand, i, j)),
                                 limits=joint_limits(hand, i, j)))
            if jt.type == "coupled":
                couplings.append(AffineCoupling(f"f{i}_j{j}", f"f{i}_j{j - 1}", COUPLING_RATIO, 0.0))
            parent = child
        frames.append(Frame(f"f{i}_tip", parent, Pose(xyz=(f.joints[-1].length * MM, 0.0, 0.0))))
    model = KinematicModel(name=name, root="palm", bodies=tuple(bodies), joints=tuple(joints),
                           frames=tuple(frames), couplings=tuple(couplings))
    validate(model)
    return model


# --------------------------------------------------------------------------
# Solids at a pose
# --------------------------------------------------------------------------


@dataclass
class Solid:
    key: Tuple                 # ("palm", -1 | 0), ("mount",), ("hinge",), ("motor", i), ("link", i, j)
    points: np.ndarray         # (k, 3) corners, palm frame, metres
    box: Optional[Box] = None  # the same solid as a box (palm frame, mm), for boxes
    far: Optional[Box] = None  # a link's far end (palm frame, mm): the next motor or the pad


def _box_solid(key: Tuple, box_local: Box, T: np.ndarray) -> Solid:
    b = box_local.moved(T, MM)            # T in metres; the box in mm
    return Solid(key, b.corners() * MM, b)


def solids_at(hand: Hand, q: Optional[np.ndarray] = None) -> List[Solid]:
    """Every solid of the hand at joint vector `q` (default the zero pose):
    the plates, the mount, the hinge motor, each finger's base motor (fixed
    to its plate) and each link (with the motor it carries)."""
    pose = fk(hand, q)
    g = hand_geometry(hand)
    out: List[Solid] = [Solid(("palm", -1), plate_points_m(g.outlines[-1])), Solid(("mount",), mount_points_m())]
    if hand.palm_joints:
        T = pose.palm[0]
        pts = plate_points_m(g.outlines[0])
        out.append(Solid(("palm", 0), pts @ T[:3, :3].T + T[:3, 3]))
        out.append(Solid(("hinge",), g.hinge_box.corners() * MM, g.hinge_box))
    for i, f in enumerate(hand.fingers):
        fg = finger_geometry(f)
        out.append(_box_solid(("motor", i), fg.motors[0], pose.pre[i][0]))
        for j in range(len(f.joints)):
            s = _box_solid(("link", i, j), fg.links[j], pose.links[i][j])
            s.far = fg.fars[j].moved(pose.links[i][j], MM)
            out.append(s)
    return out


def solid_owner(hand: Hand, key: Tuple) -> Tuple:
    """The rigid body a solid belongs to: ("palm", -1 | 0) or ("link", i, j)."""
    kind = key[0]
    if kind == "palm":
        return key
    if kind in ("mount", "hinge"):
        return ("palm", -1)
    if kind == "motor":
        return ("palm", finger_section(hand, key[1]))
    if kind == "link":
        return key
    raise ValueError(key)


# --------------------------------------------------------------------------
# Mass properties
# --------------------------------------------------------------------------


def box_mass_props(points_m: np.ndarray, density: float = LINK_DENSITY_KG_M3) -> Tuple[float, np.ndarray, np.ndarray]:
    """(mass, centre of mass, diagonal inertia about it) of the axis-aligned
    box around `points_m` (in their frame), at the uniform density."""
    c, h = mt.aabb(np.asarray(points_m, dtype=float))
    a, b, cc = (2.0 * h).tolist()
    m = density * a * b * cc
    I = m / 12.0 * np.array([b * b + cc * cc, a * a + cc * cc, a * a + b * b])
    return float(m), c, I


def link_mass_props(hand: Hand, i: int, j: int) -> Tuple[float, np.ndarray, np.ndarray]:
    return box_mass_props(link_points_m(hand, i, j))


# --------------------------------------------------------------------------
# The object's start point
# --------------------------------------------------------------------------


OBJECT_STEP_MM = 1.0
"""The object start's search step (mm) up from the plate."""
OBJECT_MAX_RAISE_MM = 400.0


@lru_cache(maxsize=8192)
def _object_start_cached(hand: Hand, half_mm: float, clearance_mm: float) -> Tuple[float, float, float]:
    """The lowest height, from `clearance_mm` above the front face up in
    `OBJECT_STEP_MM` steps, at which the cube clears every solid by
    `clearance_mm` at the start pose (exact distances)."""
    yc, zc = polygon_centroid(outline_mm(hand))
    solids = [s for s in solids_at(hand, start_q(hand)) if s.key[0] != "mount"]
    pts = [s.points / MM for s in solids]
    centres = [p.mean(axis=0) for p in pts]
    reach = [float(np.max(np.linalg.norm(p - c, axis=1))) for p, c in zip(pts, centres)]
    cube_reach = half_mm * math.sqrt(3.0)
    x = PLATE_HALF_MM + half_mm + clearance_mm
    top = x + OBJECT_MAX_RAISE_MM
    while x <= top:
        centre = np.array([x, yc, zc])
        cube = box_corners(centre, np.full(3, half_mm))
        clear = True
        for p, c, r in zip(pts, centres, reach):
            if float(np.linalg.norm(c - centre)) > r + cube_reach + clearance_mm:
                continue
            if gjk_distance(cube, p) < clearance_mm - 1e-6:
                clear = False
                break
        if clear:
            return (x, float(yc), float(zc))
        x += OBJECT_STEP_MM
    return (top, float(yc), float(zc))


def object_start_m(hand: Hand, object_half_size_m: float = 0.03, clearance_m: float = 0.005) -> np.ndarray:
    """Where the object (a cube of half size `object_half_size_m`, square to
    the palm axes) starts, in the palm frame: over the centroid of the uncut
    palm outline, at the lowest height (from `clearance_m` above the front
    face, in 1 mm steps) where it clears every solid of the hand by
    `clearance_m` at the start pose. Set by the task, not by the
    fingertips."""
    x, y, z = _object_start_cached(hand, round(object_half_size_m / MM, 6), round(clearance_m / MM, 6))
    return np.array([x, y, z]) * MM
