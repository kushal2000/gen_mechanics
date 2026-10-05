"""Non-UI kinematics for the grammar viewer: joint ranges, digit structure,
palm cells, joint classification and per-pose render primitives.

Everything here is a pure function of a `KinematicModel` (plus a joint
vector), so it is testable without a browser. Rendering (viser) lives in
`scene.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hand_sampler.grammar import geometry as ggeom
from hand_sampler.grammar.coords import admissible_box, independent_joints, q_from_u
from hand_sampler.grammar.derive import segment
from hand_sampler.grammar.fk import forward_kinematics
from hand_sampler.grammar.kinematics import KinematicModel, MOVABLE_TYPES
from hand_sampler.grammar.proxy import tip_frames

# Colours (0-255 RGB). Flexion/abduction blend follows hand_sampler/viewer.py.
FLEXION_RGB = (64, 115, 242)
ABDUCTION_RGB = (250, 153, 26)
TWIST_RGB = (190, 60, 200)
PALM_JOINT_RGB = (40, 170, 160)
PRISMATIC_RGB = (120, 120, 120)
PALM_RGB = (150, 155, 165)
OVERLAP_RGB = (230, 30, 30)        # penetration deeper than the oracle's 3 mm gate
OVERLAP_MINOR_RGB = (250, 160, 170)  # penetrating, but within the gate
DIGIT_PALETTE = (  # no reds or pinks: those mark overlaps
    (66, 135, 245), (240, 140, 40), (60, 175, 75), (150, 100, 210), (40, 180, 190),
    (200, 170, 40), (140, 100, 70), (100, 100, 100), (90, 60, 160), (120, 180, 90),
)

CONTINUOUS_SLIDER_RANGE = (-math.pi, math.pi)
TWIST_COS = math.cos(math.radians(35.0))
"""A joint axis within 35 deg of its own link direction is drawn as a twist
(roll) joint rather than placed on the flexion/abduction scale."""


# --------------------------------------------------------------------------
# Joints
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class JointRange:
    name: str
    type: str
    lo: float
    hi: float

    def clamp(self, v: float) -> float:
        return float(min(max(v, self.lo), self.hi))

    def at_fraction(self, f: float) -> float:
        return float(self.lo + f * (self.hi - self.lo))


def joint_ranges(model: KinematicModel) -> List[JointRange]:
    """One entry per INDEPENDENT joint (the slider set), in model order.
    Ranges are `coords.admissible_box` (each joint's own limits intersected
    with the pre-images of its coupled dependents' limits); continuous
    joints get +/- pi instead of the box's +/- 2 pi."""
    box, _conflicts = admissible_box(model)
    jt = {j.name: j.type for j in model.joints}
    out = []
    for name in independent_joints(model):
        lo, hi = box[name]
        if jt[name] == "continuous":
            lo, hi = CONTINUOUS_SLIDER_RANGE
        if lo > hi:  # empty admissible box (conflicting coupled limits): keep it drawable
            lo = hi = 0.5 * (lo + hi)
        out.append(JointRange(name=name, type=jt[name], lo=float(lo), hi=float(hi)))
    return out


def clamp_u(ranges: Sequence[JointRange], u: Mapping[str, float]) -> Dict[str, float]:
    """A full independent-joint vector: every range's joint, value from `u`
    clamped into range (0 clamped when absent)."""
    return {r.name: r.clamp(float(u.get(r.name, 0.0))) for r in ranges}


def curl_u(ranges: Sequence[JointRange], frac: float) -> Dict[str, float]:
    return {r.name: r.at_fraction(frac) for r in ranges}


def random_u(ranges: Sequence[JointRange], rng: np.random.Generator) -> Dict[str, float]:
    return {r.name: float(rng.uniform(r.lo, r.hi)) for r in ranges}


def expand_q(model: KinematicModel, u: Mapping[str, float]) -> Dict[str, float]:
    """Independent values -> every movable joint (couplings applied)."""
    indep = set(independent_joints(model))
    return q_from_u(model, {k: float(u.get(k, 0.0)) for k in indep})


# --------------------------------------------------------------------------
# Structure: palm bodies, digits, counts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DigitInfo:
    index: int               # 0-based, order of appearance in model.joints
    root_body: str           # first phalanx body
    host: str                # body it mounts on
    bodies: Tuple[str, ...]  # its own chain, excluding branch digits
    branch: bool             # mounts on a phalanx (branch digit) rather than a palm body
    top_index: int           # index of the top-level digit this belongs to


@dataclass(frozen=True)
class Structure:
    palm_bodies: Tuple[str, ...]
    jointed_palm_bodies: Tuple[str, ...]
    digits: Tuple[DigitInfo, ...]
    body_digit: Dict[str, int]   # non-palm body -> DigitInfo.index
    n_movable: int
    n_independent: int
    n_couplings: int
    capsule_radius_m: Optional[float]

    @property
    def top_level_digits(self) -> Tuple[DigitInfo, ...]:
        return tuple(d for d in self.digits if not d.branch)

    @property
    def branch_digits(self) -> Tuple[DigitInfo, ...]:
        return tuple(d for d in self.digits if d.branch)

    def phalanges_per_digit(self) -> List[int]:
        return [len(d.bodies) for d in self.top_level_digits]


def structure(model: KinematicModel) -> Structure:
    """Palm bodies are the palm-flagged ones (the grammar flags root and every
    PalmBody). For a model with no palm flags at all (a raw URDF import), the
    root alone is treated as palm."""
    palm = [b.name for b in model.bodies if b.palm]
    if not palm:
        palm = [model.root]
    palm_set = set(palm)
    children: Dict[str, List] = {}
    for j in model.joints:
        children.setdefault(j.parent, []).append(j)
    jointed_palm = tuple(j.child for j in model.joints if j.child in palm_set and j.type in MOVABLE_TYPES)

    digits: List[DigitInfo] = []
    body_digit: Dict[str, int] = {}
    # A digit starts at a non-palm body whose parent is palm (top level) or
    # whose parent is a non-palm body with more than one child (a branch).
    # The digit's own chain follows the FIRST child at every phalanx (the
    # grammar's own continuation order: continuation joints are emitted
    # before branch digits mount on the same body).
    starts: List[Tuple[str, str, bool]] = []
    for j in model.joints:
        if j.child in palm_set:
            continue
        if j.parent in palm_set:
            starts.append((j.child, j.parent, False))
    top_of: Dict[str, int] = {}
    queue = list(starts)
    while queue:
        root_body, host, is_branch = queue.pop(0)
        idx = len(digits)
        top_index = idx if not is_branch else top_of.get(host, 0)
        chain = [root_body]
        cur = root_body
        while True:
            kids = [j for j in children.get(cur, []) if j.child not in palm_set]
            if not kids:
                break
            # Continuation = first child; any further children start branch digits.
            for extra in kids[1:]:
                queue.append((extra.child, cur, True))
            cur = kids[0].child
            chain.append(cur)
        for b in chain:
            body_digit[b] = idx
            top_of[b] = top_index
        digits.append(DigitInfo(index=idx, root_body=root_body, host=host, bodies=tuple(chain),
                                branch=is_branch, top_index=top_index))

    radius = next((float(b.radius) for b in model.bodies if b.radius is not None), None)
    movable = [j for j in model.joints if j.type in MOVABLE_TYPES]
    return Structure(
        palm_bodies=tuple(palm), jointed_palm_bodies=jointed_palm, digits=tuple(digits),
        body_digit=body_digit, n_movable=len(movable), n_independent=len(independent_joints(model)),
        n_couplings=len(model.couplings), capsule_radius_m=radius,
    )


def body_lengths(model: KinematicModel) -> Dict[str, float]:
    """Segment length of every body: its `<body>_tip` frame's z offset (the
    grammar's convention); bodies without a tip frame get 0."""
    out = {b.name: 0.0 for b in model.bodies}
    for f in model.frames:
        if f.name == f"{f.body}_tip":
            out[f.body] = float(np.linalg.norm(np.asarray(f.pose.xyz, dtype=float)))
    return out


def body_tip_local(model: KinematicModel) -> Dict[str, np.ndarray]:
    """Local tip point (in the body's own frame) of every body with a tip frame."""
    out = {}
    for f in model.frames:
        if f.name == f"{f.body}_tip":
            out[f.body] = np.asarray(f.pose.xyz, dtype=float)
    return out


def digit_colour(st: Structure, body: str) -> Tuple[int, int, int]:
    idx = st.body_digit.get(body)
    if idx is None:
        return PALM_RGB
    d = st.digits[idx]
    base = DIGIT_PALETTE[d.top_index % len(DIGIT_PALETTE)]
    if d.branch:  # lighter shade of the host digit's colour
        return tuple(int(c + 0.45 * (255 - c)) for c in base)
    return base


# --------------------------------------------------------------------------
# Palm cells (geometry.build_geometry steps 1-3, without the adjacency pass)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CellMesh:
    body: str
    vertices: np.ndarray  # (n, 3), body-local frame
    faces: np.ndarray     # (m, 3) int triangles


def _triangulate(faces: Sequence[Sequence[int]]) -> np.ndarray:
    tris = []
    for f in faces:
        f = list(f)
        for k in range(1, len(f) - 1):
            tris.append((f[0], f[k], f[k + 1]))
    return np.asarray(tris, dtype=np.int32).reshape(-1, 3)


def palm_cells(model: KinematicModel) -> Dict[str, CellMesh]:
    """Per palm body, its nearest-spine convex cell in the body's local frame.

    Mirrors `geometry.build_geometry`'s cell construction (dense hull samples
    plus capsule points, nearest-spine assignment, per-body hull) but skips its
    pairwise cell-overlap volume pass, which is ~90% of `build_geometry`'s
    runtime and only feeds `adjacent_cells` (collision filtering), never the
    cell shapes. `tests/test_primitives.py` checks the cells match
    `build_geometry`'s."""
    transforms = forward_kinematics(model, {})
    bodies_by_name = {b.name: b for b in model.bodies}
    palm_names = [b.name for b in model.bodies if b.palm]
    if not palm_names:
        return {}
    default_radius = next((b.radius for b in model.bodies if b.radius is not None), 0.0)

    def radius_of(name: str) -> float:
        r = bodies_by_name[name].radius
        return float(r) if r is not None else float(default_radius)

    palm_set = set(palm_names)
    capsule_pts = {}
    for name in palm_names:
        start, end = segment(model, name)
        capsule_pts[name] = ggeom._capsule_hull_points(start, end, radius_of(name))
    pts = np.concatenate([capsule_pts[n] for n in palm_names], axis=0)
    hull_v, hull_f = ggeom.convex_hull_3d(pts, thicken_radius=default_radius)

    parent_of = {j.child: j.parent for j in model.joints if j.child in palm_set}
    depth_of = {model.root: 0}
    for name in palm_names:
        if name == model.root:
            continue
        d, cur = 0, name
        while cur != model.root:
            d += 1
            cur = parent_of[cur]
        depth_of[name] = d
    spine_segs = {name: segment(model, name) for name in palm_names}
    dense = ggeom._dense_hull_samples(hull_v, hull_f, target_count=2000)
    pool = np.concatenate([dense] + [capsule_pts[n] for n in palm_names], axis=0)
    winners = ggeom._assign_nearest_spine(pool, palm_names, spine_segs, depth_of)
    assigned: Dict[str, List[np.ndarray]] = {n: [] for n in palm_names}
    for p, w in zip(pool, winners):
        assigned[w].append(p)

    out = {}
    for name in palm_names:
        own = np.array(assigned[name]) if assigned[name] else np.zeros((0, 3))
        own_pts = np.concatenate([own, capsule_pts[name]], axis=0)
        v, f = ggeom.convex_hull_3d(own_pts, thicken_radius=radius_of(name))
        T = transforms[name]
        local = (np.asarray(v) - T[:3, 3]) @ T[:3, :3]  # R^T (v - t), row-wise
        out[name] = CellMesh(body=name, vertices=local, faces=_triangulate(f))
    return out


# --------------------------------------------------------------------------
# Palm normal and joint classification
# --------------------------------------------------------------------------


def palm_normal_estimate(model: KinematicModel, curl_frac: float = 0.35) -> np.ndarray:
    """Root-frame unit vector from the mean digit mount point (q=0) to the mean
    fingertip at `curl_frac` of every joint range: the same construction as
    `grammar_envelope.palm_up`'s normal, but on the grammar model itself, so it
    exists for designs the envelope cannot admit."""
    st = structure(model)
    ranges = joint_ranges(model)
    q_mid = expand_q(model, curl_u(ranges, curl_frac))
    T0 = forward_kinematics(model, {})
    Tm = forward_kinematics(model, q_mid)
    tips = tip_frames(model)
    mounts = [T0[d.root_body][:3, 3] for d in st.top_level_digits]
    if not mounts or not tips:
        return np.array([0.0, 0.0, 1.0])
    delta = np.mean([Tm[t][:3, 3] for t in tips], axis=0) - np.mean(mounts, axis=0)
    n = float(np.linalg.norm(delta))
    return delta / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class JointClass:
    name: str
    kind: str        # "flexion" | "abduction" | "twist" | "palm" | "prismatic" | "fixed"
    rgb: Tuple[int, int, int]
    theta_deg: float  # 0 = pure flexion, 90 = pure abduction (nan when not on that scale)


def _blend(t: float) -> Tuple[int, int, int]:
    return tuple(int(round((1 - t) * a + t * b)) for a, b in zip(FLEXION_RGB, ABDUCTION_RGB))


def classify_joints(model: KinematicModel, palm_normal: Optional[np.ndarray] = None) -> Dict[str, JointClass]:
    """Classify each joint at the rest pose (q=0), in the root frame.

    For a digit joint with world axis `a` and child link direction `d` (the
    child body's own +z, along which its segment runs): if `|a.d| >= cos 35 deg`
    it is a twist joint. Otherwise the component of `a` perpendicular to `d` is
    compared with the flexion direction `d x n` and the abduction direction
    (palm normal `n` made perpendicular to `d`): theta = 0 deg is pure flexion
    (blue), 90 deg pure abduction (orange), blended in between like the old
    sampler viewer. Palm joints and prismatic joints get their own colours."""
    n = np.asarray(palm_normal if palm_normal is not None else palm_normal_estimate(model), dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    T0 = forward_kinematics(model, {})
    palm_set = set(structure(model).palm_bodies)
    out = {}
    for j in model.joints:
        if j.type == "fixed":
            out[j.name] = JointClass(j.name, "fixed", PALM_RGB, float("nan"))
            continue
        R = T0[j.child][:3, :3]
        a = R @ np.asarray(j.axis, dtype=float)
        a = a / (np.linalg.norm(a) + 1e-12)
        if j.type == "prismatic":
            out[j.name] = JointClass(j.name, "prismatic", PRISMATIC_RGB, float("nan"))
            continue
        if j.child in palm_set:
            out[j.name] = JointClass(j.name, "palm", PALM_JOINT_RGB, float("nan"))
            continue
        d = R[:, 2]
        if abs(float(a @ d)) >= TWIST_COS:
            out[j.name] = JointClass(j.name, "twist", TWIST_RGB, float("nan"))
            continue
        a_perp = a - float(a @ d) * d
        a_perp /= np.linalg.norm(a_perp) + 1e-12
        flex = np.cross(d, n)
        if np.linalg.norm(flex) < 1e-6:  # link along the palm normal: no defined flexion plane
            out[j.name] = JointClass(j.name, "flexion", _blend(0.0), 0.0)
            continue
        flex /= np.linalg.norm(flex)
        abd = n - float(n @ d) * d
        abd /= np.linalg.norm(abd) + 1e-12
        theta = math.atan2(abs(float(a_perp @ abd)), abs(float(a_perp @ flex)))
        t = theta / (math.pi / 2)
        out[j.name] = JointClass(j.name, "flexion" if t < 0.5 else "abduction", _blend(t), math.degrees(theta))
    return out


# --------------------------------------------------------------------------
# Render primitives at a pose
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BodyPrim:
    name: str
    T: np.ndarray          # 4x4 root-frame pose of the body frame
    length: float          # segment length along local +z
    radius: float
    palm: bool
    rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class JointPrim:
    name: str
    child: str
    type: str
    position: np.ndarray   # root frame
    axis: np.ndarray       # unit, root frame
    kind: str
    rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class TipPrim:
    frame: str
    body: str
    position: np.ndarray


@dataclass
class Primitives:
    bodies: Dict[str, BodyPrim]
    joints: List[JointPrim]
    tips: List[TipPrim]
    cell_poses: Dict[str, np.ndarray] = field(default_factory=dict)

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        pts = []
        for b in self.bodies.values():
            pts.append(b.T[:3, 3])
            pts.append((b.T @ np.array([0.0, 0.0, b.length, 1.0]))[:3])
        if not pts:
            return np.zeros(3), np.zeros(3)
        P = np.asarray(pts)
        return P.min(axis=0), P.max(axis=0)


@dataclass
class ModelView:
    """Pose-independent data for one model, computed once per design."""
    model: KinematicModel
    structure: Structure
    ranges: List[JointRange]
    lengths: Dict[str, float]
    radius: float
    joint_classes: Dict[str, JointClass]
    palm_normal: np.ndarray
    tips: List[str]

    @classmethod
    def build(cls, model: KinematicModel, palm_normal: Optional[np.ndarray] = None) -> "ModelView":
        st = structure(model)
        n = np.asarray(palm_normal, dtype=float) if palm_normal is not None else palm_normal_estimate(model)
        return cls(
            model=model, structure=st, ranges=joint_ranges(model), lengths=body_lengths(model),
            radius=float(st.capsule_radius_m if st.capsule_radius_m is not None else 0.01),
            joint_classes=classify_joints(model, n), palm_normal=n, tips=tip_frames(model),
        )

    def primitives(self, u: Mapping[str, float]) -> Primitives:
        q = expand_q(self.model, clamp_u(self.ranges, u))
        T = forward_kinematics(self.model, q)
        palm_set = set(self.structure.palm_bodies)
        bodies = {}
        for b in self.model.bodies:
            bodies[b.name] = BodyPrim(
                name=b.name, T=T[b.name], length=self.lengths.get(b.name, 0.0),
                radius=float(b.radius) if b.radius is not None else self.radius,
                palm=b.name in palm_set, rgb=digit_colour(self.structure, b.name),
            )
        joints = []
        for j in self.model.joints:
            if j.type == "fixed":
                continue
            Tc = T[j.child]
            a = Tc[:3, :3] @ np.asarray(j.axis, dtype=float)
            a = a / (np.linalg.norm(a) + 1e-12)
            jc = self.joint_classes[j.name]
            joints.append(JointPrim(name=j.name, child=j.child, type=j.type, position=Tc[:3, 3].copy(),
                                    axis=a, kind=jc.kind, rgb=jc.rgb))
        tips = [TipPrim(frame=f, body=f[: -len("_tip")], position=T[f][:3, 3].copy()) for f in self.tips]
        cell_poses = {name: T[name] for name in self.structure.palm_bodies if name in T}
        return Primitives(bodies=bodies, joints=joints, tips=tips, cell_poses=cell_poses)


# --------------------------------------------------------------------------
# Capsule meshes (local frame)
# --------------------------------------------------------------------------

CAPSULE_MODES = ("simulator", "grammar")
"""`simulator`: the PhysX capsule `author_grammar` builds and
`grammar_envelope.rest_overlap_pairs` checks -- total extent [0, L] along
local z (core [r, L-r]; a sphere at L/2 when L < 2r). `grammar`:
`geometry.Capsule` / `to_urdf` collision -- core [0, L], so the total extent is
[-r, L+r]."""


def capsule_core(length: float, radius: float, mode: str = "simulator") -> Tuple[float, float]:
    if mode == "grammar":
        return 0.0, float(length)
    half_cyl = max(float(length) - 2.0 * float(radius), 0.0) / 2.0
    c = float(length) / 2.0
    return c - half_cyl, c + half_cyl


def capsule_mesh_local(length: float, radius: float, mode: str = "simulator",
                       n_ring: int = 14, n_cap: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Vertices/faces of a capsule along local +z with the given core."""
    z0, z1 = capsule_core(length, radius, mode)
    r = float(radius)
    rings = []  # (z, ring radius)
    for k in range(n_cap, 0, -1):  # bottom cap, pole excluded
        phi = (math.pi / 2) * k / n_cap
        rings.append((z0 - r * math.sin(phi), r * math.cos(phi)))
    rings.append((z0, r))
    rings.append((z1, r))
    for k in range(1, n_cap + 1):
        phi = (math.pi / 2) * k / n_cap
        rings.append((z1 + r * math.sin(phi), r * math.cos(phi)))
    ang = np.linspace(0.0, 2 * math.pi, n_ring, endpoint=False)
    verts = [(0.0, 0.0, z0 - r)]
    for z, rr in rings:
        for a in ang:
            verts.append((rr * math.cos(a), rr * math.sin(a), z))
    verts.append((0.0, 0.0, z1 + r))
    V = np.asarray(verts, dtype=np.float32)
    F = []
    n_r = len(rings)
    def vid(ri, k):
        return 1 + ri * n_ring + (k % n_ring)
    for k in range(n_ring):
        F.append((0, vid(0, k + 1), vid(0, k)))
    for ri in range(n_r - 1):
        for k in range(n_ring):
            a, b = vid(ri, k), vid(ri, k + 1)
            c, d = vid(ri + 1, k), vid(ri + 1, k + 1)
            F.append((a, b, d))
            F.append((a, d, c))
    top = len(verts) - 1
    for k in range(n_ring):
        F.append((top, vid(n_r - 1, k), vid(n_r - 1, k + 1)))
    return V, np.asarray(F, dtype=np.int32)


def mat_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)
