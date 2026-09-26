"""Derived geometry overlay for the hand-kinematics grammar (iteration 7c:
nearest-spine palm cells).

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
  convex hull *in a shared frame* (root frame, q=0) before being carved into
  per-body cells, because different palm bodies sit at different poses. The
  hull and the per-body assignment are therefore built in root frame, but
  each resulting ``Cell`` is transformed back into *its own body's* local
  frame before being stored -- so ``Capsule`` and ``Cell`` share the same
  convention (local frame, origin at the body's own origin) and every mass
  property below is "about the body origin" for free, with no further
  parallel-axis bookkeeping needed at the call site.

- Palm-cell construction (iteration 7c, nearest-spine assignment --
  supersedes plane-carving entirely). Three successive cutting-plane rules
  each fixed one failure mode but the last ("n = the component of the
  child's spine direction orthogonal to span(joint axis, parent's spine
  direction)", correct for any ONE parent/child pair in isolation) still
  left 59/200 sampled hands with an unresolvable overlap between two
  non-adjacent branches whenever three-plus branches spatially entangled:
  proved by exhaustive search over candidate separating planes (seed 10)
  that NO single supporting plane can simultaneously conserve volume
  exactly and keep both bodies' own spines inside their cells there -- a
  structural limit of "one convex polytope per body, carved by planes",
  not a fixable formula. Cells are therefore no longer an exact,
  non-overlapping partition of the hull. Instead:
    1. Sample the shared hull densely at rest (q=0): every hull vertex,
       plus a barycentric triangle grid over every hull face (``hull_f``
       faces are always triangles, see ``convex_hull_3d``) fine enough to
       give >= 2000 points total (``_dense_hull_samples``), unioned with
       every palm body's own capsule sample points.
    2. Assign every one of those points to the palm body whose spine
       SEGMENT is nearest (point-to-segment distance; ties -- e.g. a point
       exactly at a mount point, distance 0 to both a body and its own
       parent -- go to the shallower body, i.e. the parent).
    3. cell(B) = the convex hull of the points assigned to B in step 2,
       UNIONED with B's own capsule sample points regardless of what they
       were assigned to. This guarantees B's own spine (and every digit
       mounted on it, which sits ON that spine) is always inside cell(B):
       the capsule's own start/end points are always members of the point
       set being hulled, so the hull contains the whole segment between
       them by convexity, with no further proof obligation per joint.
  Two cells CAN legitimately overlap a little near a branch boundary now
  (not asserted to be zero -- reported: mass properties computed per cell
  will double-count material in an overlap by a small, reported amount).
  What IS asserted: every hull sample point lands inside at least one cell
  (coverage), and every pair of cells that actually overlaps with positive
  volume is recorded in ``GeometrySpec.adjacent_cells`` (a set of
  frozensets of body names, together with every direct parent-child pair
  unconditionally) so a simulator can filter exactly those pairs from
  collision checks.

Stdlib + numpy only (no scipy, no trimesh): the convex hull is a from-scratch
incremental (quickhull-style) 3-D hull, and polytope clipping (used only for
the overlap-volume check now, not for cell construction) is from-scratch
half-space (Sutherland-Hodgman-style) polygon clipping.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from .derive import segment
from .fk import forward_kinematics
from .kinematics import KinematicModel

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
    # The combined palm hull, root frame, q=0 -- kept so tests/callers can
    # recover it (e.g. for the dense-sample coverage check) without
    # recomputing it. Empty when the model has no palm bodies at all (never
    # true of grammar output, but kept total for imported models).
    hull_vertices: Tuple[Tuple[float, float, float], ...] = ()
    hull_faces: Tuple[Tuple[int, ...], ...] = ()
    # Reserved for future informational warnings; nothing currently
    # populates it (the degenerate-normal fallback it used to carry was
    # deleted with the cutting-plane machinery -- see module docstring).
    warnings: Tuple[str, ...] = ()
    # Every pair of cells that actually overlaps with positive volume,
    # PLUS every direct parent-child palm pair unconditionally -- see the
    # nearest-spine design note above.
    adjacent_cells: FrozenSet[FrozenSet[str]] = field(default_factory=frozenset)


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
    """Incremental quickhull-style build (design note: performance -- with
    ``build_geometry`` now feeding thousands of dense-sample points into
    this per palm body, per-point/per-face visibility is tracked as two
    parallel arrays (plane normal, a point on the plane) and checked with
    one vectorized dot product per point against every CURRENT face,
    instead of a nested Python loop calling ``np.cross`` per (point, face)
    pair -- ~30x faster on a ~2000-point input, verified by profiling)."""
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

    def face_plane(face: Tuple[int, int, int]):
        i, j, k = face
        a, b, c = pts[i], pts[j], pts[k]
        return np.cross(b - a, c - a), a

    faces: List[Tuple[int, int, int]] = [
        make_face(p0, p1, p2), make_face(p0, p1, p3), make_face(p0, p2, p3), make_face(p1, p2, p3),
    ]
    used = {p0, p1, p2, p3}
    normals = np.array([face_plane(f)[0] for f in faces])
    refs = np.array([face_plane(f)[1] for f in faces])

    for m in range(len(pts)):
        if m in used:
            continue
        vis = np.einsum("ij,ij->i", pts[m] - refs, normals) > _ABS_TOL
        visible_idx = np.nonzero(vis)[0].tolist()
        if not visible_idx:
            continue
        visible_set = set(visible_idx)
        edge_count: Counter = Counter()
        for fi in visible_idx:
            i, j, k = faces[fi]
            for e in ((i, j), (j, k), (k, i)):
                edge_count[frozenset(e)] += 1
        horizon = [e for e, c in edge_count.items() if c == 1]
        keep_mask = np.ones(len(faces), dtype=bool)
        keep_mask[visible_idx] = False
        faces = [f for fi, f in enumerate(faces) if fi not in visible_set]
        normals = normals[keep_mask]
        refs = refs[keep_mask]
        new_faces = [make_face(a, b, m) for a, b in (tuple(e) for e in horizon)]
        faces.extend(new_faces)
        if new_faces:
            new_planes = [face_plane(f) for f in new_faces]
            new_normals = np.array([p[0] for p in new_planes])
            new_refs = np.array([p[1] for p in new_planes])
            normals = np.concatenate([normals, new_normals], axis=0) if len(normals) else new_normals
            refs = np.concatenate([refs, new_refs], axis=0) if len(refs) else new_refs
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
# Nearest-spine palm-cell assignment (iteration 7c)
# --------------------------------------------------------------------------


def _point_segment_distance_batch(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from every row of ``P`` to segment ``a-b`` (vectorized)."""
    ab = b - a
    L2 = float(np.dot(ab, ab))
    if L2 < 1e-18:
        return np.linalg.norm(P - a, axis=1)
    t = np.clip((P - a) @ ab / L2, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(P - proj, axis=1)


def _dense_hull_samples(hull_v: np.ndarray, hull_f: Sequence[Sequence[int]], target_count: int = 2000) -> np.ndarray:
    """Dense sample of the hull surface (design note, module docstring):
    every hull vertex, plus a barycentric triangle grid over every hull
    face (``hull_f`` faces are always triangles -- see ``convex_hull_3d``),
    subdivided finely enough that the total point count is at least
    ``target_count``."""
    n_faces = max(len(hull_f), 1)
    per_face = max(1, -(-target_count // n_faces))  # ceil division
    m = 1
    while (m + 1) * (m + 2) // 2 < per_face:
        m += 1
    pts: List[np.ndarray] = [hull_v[i] for i in range(len(hull_v))]
    for face in hull_f:
        if len(face) < 3:
            continue
        v0 = hull_v[face[0]]
        for t in range(1, len(face) - 1):
            v1 = hull_v[face[t]]
            v2 = hull_v[face[t + 1]]
            for ai in range(m + 1):
                for bi in range(m + 1 - ai):
                    ci = m - ai - bi
                    pts.append((ai * v0 + bi * v1 + ci * v2) / m)
    return np.array(pts)


def _assign_nearest_spine(
    points: np.ndarray,
    palm_names: Sequence[str],
    spine_segs: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    depth_of: Dict[str, int],
    tol: float = 1e-9,
) -> List[str]:
    """Assign every row of ``points`` to the palm body whose spine SEGMENT
    is nearest (design note, module docstring); ties go to the shallower
    (closer-to-root) body -- in particular a point exactly at a mount
    point (distance 0 to both a body and its own parent) goes to the
    parent. Returns one body name per point."""
    order = sorted(palm_names, key=lambda n: depth_of[n])
    D = np.stack(
        [
            _point_segment_distance_batch(points, np.asarray(spine_segs[n][0], dtype=float),
                                           np.asarray(spine_segs[n][1], dtype=float))
            for n in order
        ],
        axis=1,
    )
    mind = D.min(axis=1, keepdims=True)
    mask = D <= mind + tol
    winner_idx = mask.argmax(axis=1)  # first True (shallowest tied body) per row
    return [order[i] for i in winner_idx]


def _intersect_polytopes(vA: np.ndarray, fA, vB: np.ndarray, fB):
    """Intersection polytope of two convex polytopes (clip A by every one
    of B's face half-spaces) -- used only for the overlap-volume check
    (adjacency, mass-property double-count reporting) now, not for cell
    construction."""
    cur_v, cur_f = vA, list(fA)
    for face in fB:
        if len(cur_v) == 0:
            return cur_v, []
        nvec, a_pt = _face_outward_normal(vB, face)
        if nvec is None:
            continue
        cur_v, cur_f = _clip_polytope(cur_v, cur_f, a_pt, nvec, keep_child_side=False)
    return cur_v, cur_f


def _overlap_volume(vA: np.ndarray, fA, vB: np.ndarray, fB) -> float:
    """Volume of the intersection of two convex polytopes."""
    v, f = _intersect_polytopes(vA, fA, vB, fB)
    if len(v) == 0 or not f:
        return 0.0
    return abs(polytope_volume(v, f))



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
    adjacency: set = set()

    if palm_names:
        palm_set = set(palm_names)
        capsule_points_by_body: Dict[str, np.ndarray] = {}
        point_chunks = []
        for name in palm_names:
            start, end = segment(model, name)
            pts_b = _capsule_hull_points(start, end, _radius_of(name))
            capsule_points_by_body[name] = pts_b
            point_chunks.append(pts_b)
        pts = np.concatenate(point_chunks, axis=0)
        hull_v, hull_f = convex_hull_3d(pts, thicken_radius=default_radius)
        hull_vol = polytope_volume(hull_v, hull_f) if len(hull_v) else 0.0

        parent_of: Dict[str, str] = {j.child: j.parent for j in model.joints if j.child in palm_set}
        depth_of: Dict[str, int] = {model.root: 0}
        for name in palm_names:
            if name == model.root:
                continue
            d = 0
            cur = name
            while cur != model.root:
                d += 1
                cur = parent_of[cur]
            depth_of[name] = d

        spine_segs = {name: segment(model, name) for name in palm_names}

        # Steps 1-3 of the nearest-spine design note above: dense hull
        # sample + every body's own capsule points, assigned to whichever
        # spine SEGMENT is nearest, then each body's cell is the convex
        # hull of (its assigned points) UNIONED with (its own capsule
        # points, unconditionally -- guarantees its own spine/mounts are
        # always inside its own cell).
        dense_samples = _dense_hull_samples(hull_v, hull_f, target_count=2000)
        full_pool = np.concatenate([dense_samples] + [capsule_points_by_body[n] for n in palm_names], axis=0)
        winners = _assign_nearest_spine(full_pool, palm_names, spine_segs, depth_of)

        assigned: Dict[str, List[np.ndarray]] = {n: [] for n in palm_names}
        for p, w in zip(full_pool, winners):
            assigned[w].append(p)

        cells_root: Dict[str, Tuple[np.ndarray, List[Tuple[int, ...]]]] = {}
        for name in palm_names:
            own_assigned = np.array(assigned[name]) if assigned[name] else np.zeros((0, 3))
            own_pts = np.concatenate([own_assigned, capsule_points_by_body[name]], axis=0)
            v, f = convex_hull_3d(own_pts, thicken_radius=_radius_of(name))
            cells_root[name] = (v, f)

        # Adjacency (step 4): every direct parent-child pair unconditionally,
        # plus every pair whose cells actually overlap with positive volume.
        adjacency = {frozenset((j.parent, j.child)) for j in model.joints if j.child in palm_set}
        rel_scale = max(abs(hull_vol), 1e-15)
        for i in range(len(palm_names)):
            for j in range(i + 1, len(palm_names)):
                b1, b2 = palm_names[i], palm_names[j]
                v1, f1 = cells_root[b1]
                v2, f2 = cells_root[b2]
                ov = _overlap_volume(v1, f1, v2, f2)
                if ov > 1e-12 * rel_scale:
                    adjacency.add(frozenset((b1, b2)))

        for body, (own_v, own_f) in cells_root.items():
            T = transforms[body]
            Rm = T[:3, :3]
            t = T[:3, 3]
            local_verts = tuple(tuple((Rm.T @ (v - t)).tolist()) for v in own_v)
            cells[body] = Cell(body=body, vertices=local_verts, faces=tuple(tuple(f) for f in own_f))

    return GeometrySpec(
        radius=float(default_radius), capsules=capsules, cells=cells,
        hull_vertices=tuple(tuple(v.tolist()) for v in hull_v), hull_faces=tuple(tuple(f) for f in hull_f),
        warnings=tuple(warnings), adjacent_cells=frozenset(adjacency),
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
    """Structural sanity checks. Does NOT check "cells partition the hull"
    any more (iteration 7c, nearest-spine assignment): cells can legitimately
    overlap a little near branch boundaries now -- see the module docstring
    and ``test_geometry_over_200_seeds`` for the coverage/adjacency checks
    that replace it."""
    issues: List[str] = []
    palm_names = {b.name for b in model.bodies if b.palm}
    if not palm_names:
        return issues

    if len(spec.cells) != len(palm_names):
        issues.append(f"cell_count_mismatch:{len(spec.cells)}!={len(palm_names)}")

    for name, cell in spec.cells.items():
        verts = np.array(cell.vertices)
        vol = polytope_volume(verts, cell.faces)
        if vol <= 0.0:
            issues.append(f"nonpositive_cell_volume:{name}:{vol!r}")
        for face in cell.faces:
            nvec, a = _face_outward_normal(verts, face)
            if nvec is None:
                continue
            d = float(np.dot(nvec, a))
            viol = float(np.max(verts @ nvec - d))
            if viol > 1e-9:
                issues.append(f"nonconvex_cell:{name}")
                break

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
    """Mass properties per body, computed independently per convex cell
    (unchanged from earlier iterations). Note (iteration 7c, nearest-spine
    assignment): two adjacent cells can legitimately overlap a little near
    a branch boundary now (see module docstring, ``GeometrySpec.
    adjacent_cells``), so summing these across a whole hand slightly
    double-counts the material in any overlap region -- reported via the
    200-seed test's mean pairwise-overlap-fraction statistic, not corrected
    for here (each body's own mass properties are still exactly correct in
    isolation)."""
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
