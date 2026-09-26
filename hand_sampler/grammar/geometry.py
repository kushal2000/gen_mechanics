"""Derived geometry overlay for the hand-kinematics grammar (iteration 7,
start of M2).

Geometry is never a gene: everything here is computed *from* a
``KinematicModel`` (plus the one per-hand ``Body.radius`` the grammar stamps
in ``derive.py``) -- ``build_geometry`` is a pure function of the model.

Conventions (read before touching anything below):

- Every non-palm ("finger") body gets one ``Capsule`` expressed in that
  body's OWN local frame: since ``derive.py``/``rules.py`` always place a
  body's own ``"<body>_tip"`` frame at ``(0, 0, length)`` *in the body's own
  frame*, and a body's own origin is by definition its local ``(0, 0, 0)``,
  every capsule is trivially ``start=(0,0,0)``, ``end=(0,0,length)`` -- no
  forward-kinematics lookup, no rotation, needed to place it.

- Palm bodies are different: their capsules must be combined into a single
  convex hull *in a shared frame* (root frame, q=0) before being cut into
  per-body cells, because different palm bodies sit at different poses. The
  hull and the cutting planes are therefore built in root frame, but each
  resulting ``Cell`` is transformed back into *its own body's* local frame
  before being stored -- so ``Capsule`` and ``Cell`` share the same
  convention (local frame, origin at the body's own origin) and every mass
  property below is "about the body origin" for free, with no further
  parallel-axis bookkeeping needed at the call site.

- Cutting-plane rule (design note, do not deviate): for a joint whose child
  is a palm body, with joint origin point ``o`` (root frame, q=0) and joint
  axis ``a`` (root frame, q=0, unit), and ``d`` the unit direction from the
  joint's PARENT palm body's own origin to ``o``: the cutting plane passes
  through ``o`` with normal ``n = d - (d.a)a``, normalized. If ``|n| <
  1e-9`` (``d`` parallel to ``a``), an arbitrary unit vector perpendicular to
  ``a`` is used instead and a warning is recorded in
  ``GeometrySpec.warnings``.

Stdlib + numpy only (no scipy, no trimesh): the convex hull is a from-scratch
incremental (quickhull-style) 3-D hull, and polytope cutting is from-scratch
half-space (Sutherland-Hodgman-style) polygon clipping.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .derive import segment
from .fk import forward_kinematics
from .kinematics import Joint, KinematicModel

_RING_N = 16
_ABS_TOL = 1e-12


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Capsule:
    body: str
    start: Tuple[float, float, float]
    end: Tuple[float, float, float]
    radius: float


@dataclass(frozen=True)
class Cell:
    body: str
    # Vertices and faces are in the body's OWN local frame (see module
    # docstring). ``faces``: each is an ordered (outward-CCW, planar,
    # convex) tuple of indices into ``vertices``.
    vertices: Tuple[Tuple[float, float, float], ...]
    faces: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class GeometrySpec:
    radius: float
    capsules: Dict[str, Capsule]
    cells: Dict[str, Cell]
    # The combined palm hull, root frame, q=0 -- kept so validate_geometry
    # (and tests) can check "sum of cell volumes == hull volume" without
    # recomputing it. Empty when the model has no palm bodies at all (never
    # true of grammar output, but kept total for imported models).
    hull_vertices: Tuple[Tuple[float, float, float], ...] = ()
    hull_faces: Tuple[Tuple[int, ...], ...] = ()
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MassProps:
    mass: float
    com: Tuple[float, float, float]
    inertia: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]


# --------------------------------------------------------------------------
# Small vector helpers
# --------------------------------------------------------------------------


def _any_perp(d) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    a = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = a - d * np.dot(a, d)
    n = np.linalg.norm(u)
    if n < 1e-12:
        a = np.array([0.0, 0.0, 1.0])
        u = a - d * np.dot(a, d)
        n = np.linalg.norm(u)
    return u / n


def _capsule_hull_points(start, end, radius: float) -> np.ndarray:
    """Sample points bounding a capsule for hull-building purposes (design
    note): segment endpoints, a ring of 16 points around each endpoint, and
    the two end caps' extreme points."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    axis_vec = end - start
    length = float(np.linalg.norm(axis_vec))
    d = axis_vec / length if length > 1e-15 else np.array([0.0, 0.0, 1.0])
    u = _any_perp(d)
    v = np.cross(d, u)
    pts = [start, end]
    if radius > 0.0:
        for base in (start, end):
            for k in range(_RING_N):
                theta = 2.0 * math.pi * k / _RING_N
                pts.append(base + radius * (math.cos(theta) * u + math.sin(theta) * v))
        pts.append(start - radius * d)
        pts.append(end + radius * d)
    return np.array(pts)


# --------------------------------------------------------------------------
# Convex hull (from-scratch incremental / quickhull-style)
# --------------------------------------------------------------------------


def _thicken(pts: np.ndarray, radius: float) -> np.ndarray:
    """Degenerate-hull fallback (design note): coplanar/collinear point sets
    have no interior, so thicken along the two smallest-variance principal
    axes by ``radius`` (a small positive fallback if ``radius`` is ~0)."""
    r = radius if radius > 1e-9 else 1e-6
    centered = pts - pts.mean(axis=0)
    cov = centered.T @ centered
    _eigvals, eigvecs = np.linalg.eigh(cov)
    d0, d1 = eigvecs[:, 0], eigvecs[:, 1]
    extra = np.concatenate([pts + r * d0, pts - r * d0, pts + r * d1, pts - r * d1], axis=0)
    return np.concatenate([pts, extra], axis=0)


def _initial_tetra_indices(pts: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    n = len(pts)
    if n < 4:
        return None
    scale = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1))) if n else 0.0
    tol = max(1e-9, 1e-6 * scale)
    p0 = int(np.argmin(pts[:, 0]))
    diffs = pts - pts[p0]
    dist = np.linalg.norm(diffs, axis=1)
    p1 = int(np.argmax(dist))
    if dist[p1] < tol:
        return None
    ab = pts[p1] - pts[p0]
    abn = ab / np.linalg.norm(ab)
    proj = diffs @ abn
    perp = diffs - np.outer(proj, abn)
    perp_dist = np.linalg.norm(perp, axis=1)
    p2 = int(np.argmax(perp_dist))
    if perp_dist[p2] < tol:
        return None
    normal = np.cross(pts[p1] - pts[p0], pts[p2] - pts[p0])
    normal = normal / np.linalg.norm(normal)
    plane_dist = np.abs((pts - pts[p0]) @ normal)
    p3 = int(np.argmax(plane_dist))
    if plane_dist[p3] < tol:
        return None
    return p0, p1, p2, p3


def _build_hull_raw(pts: np.ndarray) -> Optional[List[Tuple[int, int, int]]]:
    tetra = _initial_tetra_indices(pts)
    if tetra is None:
        return None
    p0, p1, p2, p3 = tetra
    interior = (pts[p0] + pts[p1] + pts[p2] + pts[p3]) / 4.0

    def make_face(i: int, j: int, k: int) -> Tuple[int, int, int]:
        a, b, c = pts[i], pts[j], pts[k]
        nvec = np.cross(b - a, c - a)
        if np.dot(nvec, interior - a) > 0.0:
            return (i, k, j)
        return (i, j, k)

    def is_visible(face: Tuple[int, int, int], p_idx: int) -> bool:
        i, j, k = face
        a, b, c = pts[i], pts[j], pts[k]
        nvec = np.cross(b - a, c - a)
        return bool(np.dot(nvec, pts[p_idx] - a) > _ABS_TOL)

    faces: List[Tuple[int, int, int]] = [
        make_face(p0, p1, p2), make_face(p0, p1, p3), make_face(p0, p2, p3), make_face(p1, p2, p3),
    ]
    used = {p0, p1, p2, p3}
    for m in range(len(pts)):
        if m in used:
            continue
        visible_idx = [fi for fi, f in enumerate(faces) if is_visible(f, m)]
        if not visible_idx:
            continue
        visible_set = set(visible_idx)
        edge_count: Counter = Counter()
        for fi in visible_idx:
            i, j, k = faces[fi]
            for e in ((i, j), (j, k), (k, i)):
                edge_count[frozenset(e)] += 1
        horizon = [e for e, c in edge_count.items() if c == 1]
        faces = [f for fi, f in enumerate(faces) if fi not in visible_set]
        for e in horizon:
            a, b = tuple(e)
            faces.append(make_face(a, b, m))
        used.add(m)
    return faces


def convex_hull_3d(points, thicken_radius: float = 0.0) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """Convex hull of ``points`` (n x 3). Returns ``(vertices, faces)`` with
    only hull-used points kept (re-indexed), ``faces`` a list of outward-CCW
    triangle index tuples. Degenerate (coplanar/collinear) inputs are
    thickened (see ``_thicken``) and retried."""
    cur = np.asarray(points, dtype=float)
    faces = None
    for _attempt in range(4):
        faces = _build_hull_raw(cur)
        if faces is not None:
            break
        cur = _thicken(cur, thicken_radius)
    if faces is None:
        raise RuntimeError("convex_hull_3d: point set remains degenerate after thickening")
    used = sorted({i for f in faces for i in f})
    remap = {old: new for new, old in enumerate(used)}
    verts = cur[used]
    faces_r = [tuple(remap[i] for i in f) for f in faces]
    return verts, faces_r


def polytope_volume(vertices, faces: Sequence[Sequence[int]]) -> float:
    """Signed volume of a closed, outward-CCW-oriented polytope (divergence
    theorem, apex at the origin -- valid for any origin, inside or out, as
    long as the surface is closed and consistently oriented)."""
    vertices = np.asarray(vertices, dtype=float)
    vol6 = 0.0
    for face in faces:
        v0 = vertices[face[0]]
        for a_i, b_i in zip(face[1:-1], face[2:]):
            vol6 += float(np.dot(v0, np.cross(vertices[a_i], vertices[b_i])))
    return vol6 / 6.0


# --------------------------------------------------------------------------
# Half-space polytope clipping
# --------------------------------------------------------------------------


def _dedupe_points(points: List[np.ndarray], tol: float = 1e-9) -> List[np.ndarray]:
    seen: Dict[Tuple[int, int, int], np.ndarray] = {}
    out: List[np.ndarray] = []
    for p in points:
        key = (round(p[0] / tol), round(p[1] / tol), round(p[2] / tol))
        if key not in seen:
            seen[key] = p
            out.append(p)
    return out


def _rebuild_polytope(face_coord_lists: List[List[np.ndarray]], tol: float = 1e-9):
    vindex: Dict[Tuple[int, int, int], int] = {}
    vertices: List[np.ndarray] = []

    def key(p: np.ndarray) -> Tuple[int, int, int]:
        return (round(p[0] / tol), round(p[1] / tol), round(p[2] / tol))

    faces_idx: List[Tuple[int, ...]] = []
    for poly in face_coord_lists:
        idxs = []
        for p in poly:
            k = key(p)
            if k not in vindex:
                vindex[k] = len(vertices)
                vertices.append(p)
            idxs.append(vindex[k])
        dedup: List[int] = []
        for idx in idxs:
            if not dedup or dedup[-1] != idx:
                dedup.append(idx)
        if len(dedup) >= 2 and dedup[0] == dedup[-1]:
            dedup.pop()
        if len(dedup) >= 3:
            faces_idx.append(tuple(dedup))
    return np.array(vertices) if vertices else np.zeros((0, 3)), faces_idx


def _clip_polytope(vertices: np.ndarray, faces: Sequence[Sequence[int]], point, normal,
                    keep_child_side: bool, tol: float = 1e-9):
    """Clip a convex polytope by the half-space through ``point`` with
    ``normal``: ``keep_child_side=True`` keeps ``normal.(x-point) >= 0``
    (the child side of a joint's own cutting plane); ``False`` keeps
    ``normal.(x-point) <= 0`` (the parent side of a child's cutting plane)."""
    point = np.asarray(point, dtype=float)
    normal = np.asarray(normal, dtype=float)
    sign = 1.0 if keep_child_side else -1.0

    def f(x: np.ndarray) -> float:
        return sign * float(np.dot(normal, x - point))

    new_face_polys: List[List[np.ndarray]] = []
    cut_points: List[np.ndarray] = []
    for face in faces:
        coords = [vertices[i] for i in face]
        vals = [f(c) for c in coords]
        n = len(coords)
        poly: List[np.ndarray] = []
        for i in range(n):
            cur, curv = coords[i], vals[i]
            nxt, nxtv = coords[(i + 1) % n], vals[(i + 1) % n]
            cur_in = curv >= -tol
            nxt_in = nxtv >= -tol
            if cur_in:
                poly.append(cur)
            if cur_in != nxt_in:
                denom = nxtv - curv
                t = (0.0 - curv) / denom if abs(denom) > 1e-15 else 0.0
                t = min(max(t, 0.0), 1.0)
                ipt = cur + t * (nxt - cur)
                poly.append(ipt)
                cut_points.append(ipt)
        if len(poly) >= 3:
            new_face_polys.append(poly)

    if cut_points:
        cap = _dedupe_points(cut_points)
        if len(cap) >= 3:
            centroid = np.mean(cap, axis=0)
            outward = -sign * normal
            outward = outward / np.linalg.norm(outward)
            u = _any_perp(outward)
            v = np.cross(outward, u)
            angles = [math.atan2(float(np.dot(p - centroid, v)), float(np.dot(p - centroid, u))) for p in cap]
            order = sorted(range(len(cap)), key=lambda i: angles[i])
            new_face_polys.append([cap[i] for i in order])

    return _rebuild_polytope(new_face_polys)


# --------------------------------------------------------------------------
# Cutting planes
# --------------------------------------------------------------------------


def _cutting_plane(transforms: Dict[str, np.ndarray], joint: Joint):
    """Return ``(point, normal, degenerate)`` for the cutting plane of a
    joint whose child is a palm body (design note, module docstring)."""
    joint_origin = transforms[joint.child][:3, 3]
    parent_origin = transforms[joint.parent][:3, 3]
    R_child = transforms[joint.child][:3, :3]
    axis_local = np.asarray(joint.axis, dtype=float)
    axis_local = axis_local / np.linalg.norm(axis_local)
    a = R_child @ axis_local
    a = a / np.linalg.norm(a)
    d_vec = joint_origin - parent_origin
    dn = float(np.linalg.norm(d_vec))
    origin_coincident = dn <= 1e-12
    d = d_vec / dn if not origin_coincident else np.array([0.0, 0.0, 1.0])
    n = d - np.dot(d, a) * a
    nn = float(np.linalg.norm(n))
    degenerate = nn < 1e-9 or origin_coincident
    if degenerate:
        n = _any_perp(a)
    else:
        n = n / nn
    return joint_origin, n, degenerate


# --------------------------------------------------------------------------
# build_geometry
# --------------------------------------------------------------------------


def build_geometry(model: KinematicModel) -> GeometrySpec:
    transforms = forward_kinematics(model, {})
    bodies_by_name = {b.name: b for b in model.bodies}
    palm_names = [b.name for b in model.bodies if b.palm]
    non_palm_names = [b.name for b in model.bodies if not b.palm]
    default_radius = next((b.radius for b in model.bodies if b.radius is not None), 0.0)
    tip_len = {f.body: float(f.pose.xyz[2]) for f in model.frames if f.name == f"{f.body}_tip"}

    def _radius_of(name: str) -> float:
        r = bodies_by_name[name].radius
        return float(r) if r is not None else float(default_radius)

    capsules: Dict[str, Capsule] = {}
    for name in non_palm_names:
        L = tip_len.get(name, 0.0)
        capsules[name] = Capsule(body=name, start=(0.0, 0.0, 0.0), end=(0.0, 0.0, L), radius=_radius_of(name))

    warnings: List[str] = []
    cells: Dict[str, Cell] = {}
    hull_v: np.ndarray = np.zeros((0, 3))
    hull_f: List[Tuple[int, ...]] = []

    if palm_names:
        palm_set = set(palm_names)
        point_chunks = []
        for name in palm_names:
            start, end = segment(model, name)
            point_chunks.append(_capsule_hull_points(start, end, _radius_of(name)))
        pts = np.concatenate(point_chunks, axis=0)
        hull_v, hull_f = convex_hull_3d(pts, thicken_radius=default_radius)

        palm_joints = [j for j in model.joints if j.child in palm_set]
        planes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for j in palm_joints:
            point, n, degenerate = _cutting_plane(transforms, j)
            planes[j.child] = (point, n)
            if degenerate:
                warnings.append(f"degenerate_normal:{j.name}")

        children_of: Dict[str, List[str]] = {}
        for j in palm_joints:
            children_of.setdefault(j.parent, []).append(j.child)

        # Hierarchical top-down split, one parent at a time -- but siblings
        # under the SAME parent always mount along that parent's own local
        # Z axis (rules.py's mount convention: xyz is always ``(0, 0,
        # frac*length)``, never off-axis), so they are always colinear and
        # must be ordered by distance from the parent's own origin before
        # cutting: a "childless" nearer sibling's own plane alone never
        # stops at a farther sibling's attach point, so each sibling's own
        # candidate region is also cut by the PARENT side of every sibling
        # that sits farther out (not just its own descendants' cuts, and
        # not a plain order-dependent sequential peel, which double-counted
        # or zeroed out cells in earlier versions of this function whenever
        # a body had more than one direct palm-body child).
        def _split(body: str, inherited_v: np.ndarray, inherited_f: List[Tuple[int, ...]]) -> None:
            kids = children_of.get(body, [])
            own_v, own_f = inherited_v, inherited_f
            if kids:
                ordered = sorted(kids, key=lambda c: float(np.linalg.norm(
                    transforms[c][:3, 3] - transforms[body][:3, 3])))
                for i, c in enumerate(ordered):
                    point, n = planes[c]
                    cand_v, cand_f = _clip_polytope(inherited_v, inherited_f, point, n, keep_child_side=True)
                    for cj in ordered[i + 1:]:
                        pj, nj = planes[cj]
                        cand_v, cand_f = _clip_polytope(cand_v, cand_f, pj, nj, keep_child_side=False)
                    _split(c, cand_v, cand_f)
                    own_v, own_f = _clip_polytope(own_v, own_f, point, n, keep_child_side=False)
            T = transforms[body]
            Rm = T[:3, :3]
            t = T[:3, 3]
            local_verts = tuple(tuple((Rm.T @ (v - t)).tolist()) for v in own_v)
            cells[body] = Cell(body=body, vertices=local_verts, faces=tuple(tuple(f) for f in own_f))

        _split(model.root, hull_v, hull_f)

    return GeometrySpec(
        radius=float(default_radius), capsules=capsules, cells=cells,
        hull_vertices=tuple(tuple(v.tolist()) for v in hull_v), hull_faces=tuple(tuple(f) for f in hull_f),
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------
# validate_geometry
# --------------------------------------------------------------------------


def _face_outward_normal(verts: np.ndarray, face: Tuple[int, ...]):
    a, b, c = verts[face[0]], verts[face[1]], verts[face[2]]
    nvec = np.cross(b - a, c - a)
    nn = float(np.linalg.norm(nvec))
    if nn < 1e-18:
        return None, None
    return nvec / nn, a


def validate_geometry(model: KinematicModel, spec: GeometrySpec) -> List[str]:
    issues: List[str] = []
    palm_names = {b.name for b in model.bodies if b.palm}
    if not palm_names:
        return issues

    if len(spec.cells) != len(palm_names):
        issues.append(f"cell_count_mismatch:{len(spec.cells)}!={len(palm_names)}")

    hull_vol = polytope_volume(np.array(spec.hull_vertices), spec.hull_faces) if spec.hull_vertices else 0.0

    total_vol = 0.0
    for name, cell in spec.cells.items():
        verts = np.array(cell.vertices)
        vol = polytope_volume(verts, cell.faces)
        if vol <= 0.0:
            issues.append(f"nonpositive_cell_volume:{name}:{vol!r}")
        total_vol += vol
        for face in cell.faces:
            nvec, a = _face_outward_normal(verts, face)
            if nvec is None:
                continue
            d = float(np.dot(nvec, a))
            viol = float(np.max(verts @ nvec - d))
            if viol > 1e-9:
                issues.append(f"nonconvex_cell:{name}")
                break

    rel_scale = max(abs(hull_vol), 1e-15)
    if abs(total_vol - hull_vol) > 1e-9 * rel_scale:
        issues.append(f"cell_volume_sum_mismatch:{total_vol!r}!={hull_vol!r}")

    for j in model.joints:
        if j.parent in palm_names and j.child not in palm_names:
            cell = spec.cells.get(j.parent)
            if cell is None:
                issues.append(f"missing_cell_for_mount:{j.name}")
                continue
            verts = np.array(cell.vertices)
            mount_pt = np.array(j.origin.xyz, dtype=float)
            for face in cell.faces:
                nvec, a = _face_outward_normal(verts, face)
                if nvec is None:
                    continue
                d = float(np.dot(nvec, a))
                if float(np.dot(nvec, mount_pt)) - d > 1e-9:
                    issues.append(f"mount_point_outside_cell:{j.name}")
                    break

    return issues


# --------------------------------------------------------------------------
# Mass properties
# --------------------------------------------------------------------------


def capsule_mass_props(capsule: Capsule, density: float) -> MassProps:
    """Analytic capsule = cylinder (length L along the local axis) + two
    hemispheres (radius r), about the capsule's own body origin (design
    note). Derivation: see project notes -- verified to reduce to a sphere
    at L=0 and to include the standard solid-cylinder terms in the L>>r
    limit."""
    start = np.array(capsule.start, dtype=float)
    end = np.array(capsule.end, dtype=float)
    axis_vec = end - start
    L = float(np.linalg.norm(axis_vec))
    r = float(capsule.radius)
    a = axis_vec / L if L > 1e-15 else np.array([0.0, 0.0, 1.0])

    V_cyl = math.pi * r * r * L
    V_sph = (4.0 / 3.0) * math.pi * r ** 3
    m_cyl = density * V_cyl
    m_sph = density * V_sph
    M = m_cyl + m_sph
    com = (start + end) / 2.0

    if M <= 0.0:
        return MassProps(mass=0.0, com=tuple(com.tolist()), inertia=((0.0, 0.0, 0.0),) * 3)

    Izz_local = 0.5 * m_cyl * r * r + (2.0 / 5.0) * m_sph * r * r
    Ixx_local = (
        m_cyl * (3.0 * r * r + L * L) / 12.0
        + (83.0 / 320.0) * m_sph * r * r
        + m_sph * (L / 2.0 + 3.0 * r / 8.0) ** 2
    )
    I_local = np.diag([Ixx_local, Ixx_local, Izz_local])

    u = _any_perp(a)
    v = np.cross(a, u)
    Rm = np.column_stack([u, v, a])
    I_com_world = Rm @ I_local @ Rm.T

    r_vec = com - start  # start is the body's own local origin
    I_origin = I_com_world + M * (float(np.dot(r_vec, r_vec)) * np.eye(3) - np.outer(r_vec, r_vec))

    return MassProps(
        mass=float(M), com=tuple(com.tolist()),
        inertia=tuple(tuple(row) for row in I_origin.tolist()),
    )


def cell_mass_props(cell: Cell, density: float) -> MassProps:
    """Convex-cell mass properties by exact tetrahedral decomposition from
    the origin (== the body's own origin, since ``Cell`` vertices are stored
    in the body's local frame -- design note)."""
    verts = np.array(cell.vertices, dtype=float)
    V = 0.0
    M1 = np.zeros(3)
    S = np.zeros((3, 3))
    for face in cell.faces:
        v0 = verts[face[0]]
        for a_i, b_i in zip(face[1:-1], face[2:]):
            v1, v2, v3 = v0, verts[a_i], verts[b_i]
            J = float(np.dot(v1, np.cross(v2, v3)))
            V += J / 6.0
            M1 += J * (v1 + v2 + v3) / 24.0
            pts3 = (v1, v2, v3)
            outer_sum = sum(np.outer(p, p) for p in pts3)
            cross_sum = sum(
                np.outer(pts3[i], pts3[k]) for i in range(3) for k in range(3) if i != k
            )
            S += J * (outer_sum / 60.0 + cross_sum / 120.0)

    if abs(V) < 1e-18:
        return MassProps(mass=0.0, com=(0.0, 0.0, 0.0), inertia=((0.0, 0.0, 0.0),) * 3)

    com = M1 / V
    mass = density * V
    I = density * (np.trace(S) * np.eye(3) - S)
    return MassProps(mass=float(mass), com=tuple(com.tolist()), inertia=tuple(tuple(row) for row in I.tolist()))


def mass_properties(spec: GeometrySpec, density: float) -> Dict[str, MassProps]:
    out: Dict[str, MassProps] = {}
    for name, cap in spec.capsules.items():
        out[name] = capsule_mass_props(cap, density)
    for name, cell in spec.cells.items():
        out[name] = cell_mass_props(cell, density)
    return out


# --------------------------------------------------------------------------
# URDF mesh export helper (see adapters/urdf.py's ``to_urdf(..., geometry=)``)
# --------------------------------------------------------------------------


def write_geometry_meshes(spec: GeometrySpec, out_dir) -> Dict[str, str]:
    """Write one OBJ file per palm cell into ``out_dir/meshes/`` (matching
    the ``package://.../meshes/<body>_cell.obj`` filenames ``to_urdf``
    references when given the same ``spec``). Returns {body: relative path}.
    """
    out_dir = Path(out_dir)
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    for name, cell in spec.cells.items():
        fname = f"{name}_cell.obj"
        lines = [f"v {v[0]!r} {v[1]!r} {v[2]!r}" for v in cell.vertices]
        for face in cell.faces:
            lines.append("f " + " ".join(str(i + 1) for i in face))
        (mesh_dir / fname).write_text("\n".join(lines) + "\n")
        written[name] = f"meshes/{fname}"
    return written
