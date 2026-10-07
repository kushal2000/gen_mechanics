"""From a `hand.Hand` to kinematics and geometry: derived joint kinds, signs
and ranges, forward kinematics (batched), the `KinematicModel`, the rounded
link boxes and the palm outlines. Numpy only.

Derived per joint (GRAMMAR-LOCK item 3):
- kind: the nearest of flexion (link z), abduction (link y) and roll (link
  x); ties go to flexion, then abduction.
- sign: + moves the fingertip toward +y of the link frame (the closing
  direction) if the joint can move it that way at all; otherwise + moves it
  toward +y of the palm, then +z of the palm, then +x of the palm; a pure
  roll turns right-handed about the link. A palm joint uses the centroid of
  its fingertips the same way, closing toward +x of the palm (the grasp
  side).
- range: the kind's global range; a coupled joint takes 1.1 x its source's
  range so the coupling never pushes it into its own limit; palm joints
  +/-30 degrees; sliding joints -20 to +25 mm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .hand import (
    BASE_RIM_MM,
    COUPLING_RATIO,
    KIND_RANGE_DEG,
    LINK_HEIGHT_MM,
    LINK_RADIUS_MM,
    LINK_WIDTH_MM,
    PALM_ANCHOR_MM,
    PALM_DISC_RADIUS_MM,
    PALM_OUTLINE_MAX_VERTICES,
    PALM_RANGE_DEG,
    PALM_THICKNESS_MM,
    REVOLUTE_TYPES,
    SLIDING_RANGE_MM,
    Finger,
    Hand,
)
from .kinematics import AffineCoupling, Body, Frame, Joint as KJoint, KinematicModel, Pose, validate

DEG = math.pi / 180.0
MM = 1e-3
EPS = 1e-9

START_CURL = 0.35
"""The episode start pose: every flexion joint at this fraction of its upper
limit, every other joint at 0, coupled joints following their source."""


# --------------------------------------------------------------------------
# Directions and frames
# --------------------------------------------------------------------------


def direction(az_deg: float, el_deg: float) -> np.ndarray:
    a, e = az_deg * DEG, el_deg * DEG
    return np.array([math.sin(e), math.cos(e) * math.sin(a), math.cos(e) * math.cos(a)])


def finger_frame(f: Finger) -> np.ndarray:
    """3x3, columns = the finger's link frame (x along the finger, y the
    closing direction, z the flexion axis) in the palm frame, at q = 0."""
    fa, t = f.facing * DEG, f.tilt * DEG
    x = np.array([math.sin(t), math.cos(t) * math.sin(fa), math.cos(t) * math.cos(fa)])
    y = np.array([math.cos(t), -math.sin(t) * math.sin(fa), -math.sin(t) * math.cos(fa)])
    z = np.array([0.0, math.cos(fa), -math.sin(fa)])
    return np.column_stack([x, y, z])


def mount_point_m(f: Finger) -> np.ndarray:
    return np.array([0.0, f.y * MM, f.z * MM])


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


def palm_hinge_m(hand: Hand, k: int) -> np.ndarray:
    p = hand.palm_joints[k]
    return np.array([0.0, p.y * MM, p.z * MM])


def palm_axis(hand: Hand, k: int) -> np.ndarray:
    """The signed unit axis of palm joint k, in the palm frame."""
    p = hand.palm_joints[k]
    a = direction(*p.axis)
    tips = [tip_zero_m(f) for f in hand.fingers if f.palm_joint == k]
    c = np.mean(tips, axis=0) if tips else palm_hinge_m(hand, k) + np.array([0.0, 0.0, 0.05])
    v = np.cross(a, c - palm_hinge_m(hand, k))
    return _sign_from(v[0], np.array([0.0, v[1], v[2]]), a[0] if abs(a[0]) > EPS else a[1] + a[2]) * a


def tip_zero_m(f: Finger) -> np.ndarray:
    """The fingertip at q = 0 (the finger is a straight line), palm frame."""
    return mount_point_m(f) + finger_frame(f)[:, 0] * (sum(j.length for j in f.joints) * MM)


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
    """Every joint of the hand in one fixed order: palm joints first, then
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
    links: List[np.ndarray]    # per finger (N, J, 4, 4): link frames (palm frame)
    tips: np.ndarray           # (N, F, 3): fingertips


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
    tips = np.zeros((N, len(hand.fingers), 3))
    col = P
    for i, f in enumerate(hand.fingers):
        T = np.tile(np.eye(4), (N, 1, 1))
        T[:, :3, :3] = finger_frame(f)
        T[:, :3, 3] = mount_point_m(f)
        if f.palm_joint >= 0:
            T = palm[:, f.palm_joint] @ T
        frames = np.zeros((N, len(f.joints), 4, 4))
        prev_len = 0.0
        for j, jt in enumerate(f.joints):
            T = T.copy()
            T[:, :3, 3] += T[:, :3, 0] * prev_len
            a = joint_axis(hand, i, j)
            if jt.type == "sliding":
                T[:, :3, 3] += (T[:, :3, :3] @ a) * q[:, col][:, None]
            else:
                T[:, :3, :3] = T[:, :3, :3] @ _rot_batch(a, q[:, col])
            frames[:, j] = T
            prev_len = jt.length * MM
            col += 1
        links.append(frames)
        tips[:, i] = T[:, :3, 3] + T[:, :3, 0] * prev_len
    if single:
        return Pose3(palm=palm[0], links=[l[0] for l in links], tips=tips[0])
    return Pose3(palm=palm, links=links, tips=tips)


# --------------------------------------------------------------------------
# KinematicModel
# --------------------------------------------------------------------------


def _matrix_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    from .fk import matrix_to_rpy

    return tuple(float(v) for v in matrix_to_rpy(R))


def derive(hand: Hand, name: str = "grammar_hand") -> KinematicModel:
    """The hand as a `KinematicModel` (metres, radians): root body `palm`,
    palm sections `palm{k}` (hinged to the palm by `palm{k}_joint`), finger
    links `f{i}_link{j}` (joints `f{i}_j{j}`), fingertip frames `f{i}_tip`.
    Coupled joints are revolute joints with an `AffineCoupling` (1.1, 0) on
    the joint before them."""
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
        parent = "palm" if f.palm_joint < 0 else f"palm{f.palm_joint}"
        base = mount_point_m(f) - (palm_hinge_m(hand, f.palm_joint) if f.palm_joint >= 0 else 0.0)
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
# Geometry: rounded-box links, palm plates
# --------------------------------------------------------------------------

CORE_HALF_Y_MM = (LINK_HEIGHT_MM - 2 * LINK_RADIUS_MM) / 2.0   # 3.0
CORE_HALF_Z_MM = (LINK_WIDTH_MM - 2 * LINK_RADIUS_MM) / 2.0    # 3.5
MIN_CORE_MM = 1.0


def link_core_x_mm(length: float, is_tip: bool) -> Tuple[float, float]:
    """The core box's extent along the link (x), mm. The rounded shell (r = 6
    mm) adds 6 mm all round: a link between two joints spans from 6 mm
    before its joint to 6 mm past the next one, so links meet at every joint;
    a fingertip link ends at the fingertip; a 0 mm link is a 1 mm core (a
    19 x 18 x 13 mm puck at its joint)."""
    if is_tip:
        return (0.0, max(length - LINK_RADIUS_MM, MIN_CORE_MM))
    if length < MIN_CORE_MM:
        return (length / 2.0 - MIN_CORE_MM / 2.0, length / 2.0 + MIN_CORE_MM / 2.0)
    return (0.0, float(length))


def link_core_box_m(hand: Hand, i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
    """(centre, half extents) of the core box in link j's frame, metres."""
    f = hand.fingers[i]
    x0, x1 = link_core_x_mm(f.joints[j].length, j == len(f.joints) - 1)
    c = np.array([(x0 + x1) / 2.0, 0.0, 0.0]) * MM
    h = np.array([(x1 - x0) / 2.0, CORE_HALF_Y_MM, CORE_HALF_Z_MM]) * MM
    return c, h


def box_corners(centre: np.ndarray, half: np.ndarray) -> np.ndarray:
    s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    return centre[None] + s * half[None]


def rounded_box_mass_props(centre: np.ndarray, half: np.ndarray, r: float = LINK_RADIUS_MM * MM,
                           density: float = None) -> Tuple[float, np.ndarray, np.ndarray]:
    """(mass, centre of mass, diagonal inertia about it) of a rounded box
    (the core box `half` grown by `r`): the exact volume, the inertia of the
    outer box at that mass."""
    from .hand import LINK_DENSITY_KG_M3

    rho = LINK_DENSITY_KG_M3 if density is None else density
    a, b, c = (2.0 * half).tolist()
    vol = a * b * c + 2.0 * r * (a * b + b * c + c * a) + math.pi * r * r * (a + b + c) + 4.0 / 3.0 * math.pi * r ** 3
    m = rho * vol
    A, B, C = a + 2 * r, b + 2 * r, c + 2 * r
    I = m / 12.0 * np.array([B * B + C * C, A * A + C * C, A * A + B * B])
    return m, np.asarray(centre, dtype=float), I


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


def _discs(centres: Sequence[Tuple[float, float]], radius: float, n: int = 16) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    ring = np.stack([np.cos(t), np.sin(t)], axis=1) * radius
    return np.concatenate([np.asarray(c, dtype=float)[None] + ring for c in centres], axis=0)


def palm_outlines_mm(hand: Hand) -> Dict[int, np.ndarray]:
    """Plate outlines as (y, z) polygons in mm, counter-clockwise, at most
    `PALM_OUTLINE_MAX_VERTICES` vertices: key -1 is the main palm (the heel
    disc at the wrist centre, its fingers' bases and every hinge), key k is
    palm joint k's section (its fingers' bases and its hinge). Each base and
    hinge adds a disc of half a link width, so a section with one finger is
    still a plate."""
    main = [tuple(PALM_ANCHOR_MM)]
    main_pts = [_discs(main, PALM_DISC_RADIUS_MM)]
    bases_main = [(f.y, f.z) for f in hand.fingers if f.palm_joint < 0]
    hinges = [(p.y, p.z) for p in hand.palm_joints]
    if bases_main or hinges:
        main_pts.append(_discs(bases_main + hinges, BASE_RIM_MM))
    out = {-1: _simplify(_hull2d(np.concatenate(main_pts)), PALM_OUTLINE_MAX_VERTICES)}
    for k, p in enumerate(hand.palm_joints):
        pts = [(f.y, f.z) for f in hand.fingers if f.palm_joint == k] + [(p.y, p.z)]
        out[k] = _simplify(_hull2d(_discs(pts, BASE_RIM_MM)), PALM_OUTLINE_MAX_VERTICES)
    return out


def plate_points_m(outline_mm: np.ndarray, origin_mm: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """The plate's prism corners (2 x outline vertices) in metres, relative to
    the point (0, origin) of the palm frame: the outline at x = +/- half the
    palm thickness."""
    o = np.asarray(outline_mm, dtype=float) - np.asarray(origin_mm, dtype=float)[None]
    h = PALM_THICKNESS_MM / 2.0
    top = np.column_stack([np.full(len(o), h), o])
    bot = np.column_stack([np.full(len(o), -h), o])
    return np.concatenate([top, bot], axis=0) * MM


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    x, y = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    a = cr.sum() / 2.0
    if abs(a) < 1e-12:
        return poly.mean(axis=0)
    return np.array([((x + x1) * cr).sum(), ((y + y1) * cr).sum()]) / (6.0 * a)


def object_start_m(hand: Hand, object_half_size_m: float = 0.03, clearance_m: float = 0.005) -> np.ndarray:
    """Where the object starts, in the palm frame: above the centroid of the
    main palm's outline, one object half-size plus `clearance_m` above the
    plate's top face. Set by the task, not by the fingertips."""
    c = polygon_centroid(palm_outlines_mm(hand)[-1]) * MM
    return np.array([PALM_THICKNESS_MM / 2.0 * MM + object_half_size_m + clearance_m, c[0], c[1]])


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
