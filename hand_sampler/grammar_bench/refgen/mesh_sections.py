"""Mesh cross-section extraction and shape fitting for the E14 cross-section
study (representation-check plan, item 4: "would a single rounded-rectangle
cross-section represent real finger links better than a capsule?").

Two independent halves:

  * Mesh I/O (needs ``yourdfpy`` + ``trimesh``; imported lazily so importing
    this module never fails on the system Python that ships with this repo's
    test suite): ``load_link_mesh`` resolves a link's collision mesh (or, if
    it has none loadable, its visual mesh) into a single ``trimesh.Trimesh``
    expressed in the link's own local frame; ``slice_mesh`` cuts it with a
    plane and returns the largest closed loop as an (N, 3) array of points.

  * Pure-numpy 2D shape fitting, centred on the link's central axis (the
    joint-to-joint line), which never needs a mesh at all and is what
    ``test_e14_cross_section.py`` exercises directly with synthetic
    contours: signed distance functions for a circle and an
    axis-aligned-to-principal-axes rounded rectangle, plus fits that read
    off a circle's radius directly (equal-area, 95th-percentile) and a
    small coordinate-descent fit for the rounded rectangle's (w, h, r).

Units: everything mesh-derived is in metres (URDF convention); the fitting
functions are unit-agnostic (they are handed whatever ``xy`` is in) but this
study's caller works in millimetres so error thresholds read naturally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class MeshUnavailable(Exception):
    """A link's collision/visual mesh could not be loaded (missing file,
    unresolvable ``package://`` URI, DAE-only with no loader, primitive-only
    geometry with no visual fallback, or ``trimesh``/``yourdfpy`` missing)."""


# --------------------------------------------------------------------------
# Filename resolution: package://, plain-relative and REPO-relative URIs,
# by walking up from the URDF's own directory and trying successively
# shorter suffixes of the URI's path components. Handles every case seen in
# the manifest's mesh-bearing hands (Allegro's "package://drake/.../meshes/
# x.obj" needs the 2-suffix walk; xhand/Tesollo/ORCA's "package://<pkg>/
# meshes/..." resolve at the full suffix; Wuji's "../meshes/..." and
# Ability/Barrett/LEAP/D'Claw's plain relative paths resolve directly).
# --------------------------------------------------------------------------


def resolve_mesh_filename(uri: str, urdf_dir: Path, max_levels: int = 6) -> Optional[Path]:
    if uri.startswith("package://"):
        rest = uri[len("package://"):]
        parts = rest.split("/", 1)
        rel_parts = parts[1].split("/") if len(parts) > 1 else []
    elif uri.startswith("file://"):
        p = Path(uri[len("file://"):])
        return p if p.is_file() else None
    else:
        rel_parts = uri.split("/")

    bases: List[Path] = []
    d = urdf_dir
    for _ in range(max_levels):
        bases.append(d)
        if d.parent == d:
            break
        d = d.parent

    for base in bases:
        for i in range(len(rel_parts)):
            cand = base.joinpath(*rel_parts[i:])
            if cand.is_file():
                return cand.resolve()
    return None


# --------------------------------------------------------------------------
# Mesh loading (lazy yourdfpy + trimesh)
# --------------------------------------------------------------------------


def _geometry_scale(mesh_geom) -> np.ndarray:
    scale = getattr(mesh_geom, "scale", None)
    if scale is None:
        return np.ones(3)
    if isinstance(scale, (int, float)):
        return np.full(3, float(scale))
    return np.asarray(scale, dtype=float)


def _merge_meshes(pieces):
    import trimesh

    if not pieces:
        return None
    if len(pieces) == 1:
        return pieces[0]
    return trimesh.util.concatenate(pieces)


def load_link_mesh(urdf_path: Path, link, source_root: Optional[Path] = None):
    """``link`` is a ``yourdfpy.urdf.Link`` (from ``URDF.load(..., load_meshes=
    False, load_collision_meshes=False, build_scene_graph=False)``, i.e. only
    parsed, not loaded). Returns ``(trimesh.Trimesh, "collision"|"visual")``
    in the link's own local frame (every mesh piece transformed by its own
    ``<collision>``/``<visual>`` origin, then merged). Raises
    ``MeshUnavailable`` if neither collision nor visual has loadable mesh
    geometry (box/cylinder/sphere-only collision with no visual mesh, a
    filename that does not resolve on disk, or an unsupported format such as
    a lone ``.dae`` when the loader cannot read it)."""
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - system python has no trimesh
        raise MeshUnavailable(f"trimesh not importable: {exc}") from exc

    urdf_dir = urdf_path.parent
    for kind, elements in (("collision", link.collisions), ("visual", link.visuals)):
        pieces = []
        for el in elements:
            geom = getattr(el, "geometry", None)
            mesh_geom = getattr(geom, "mesh", None) if geom is not None else None
            if mesh_geom is None:
                continue
            resolved = resolve_mesh_filename(mesh_geom.filename, urdf_dir)
            if resolved is None:
                continue
            try:
                loaded = trimesh.load(str(resolved), force="mesh", process=False)
            except Exception:
                continue
            if not hasattr(loaded, "vertices") or len(loaded.vertices) == 0:
                continue
            scale = _geometry_scale(mesh_geom)
            loaded = loaded.copy()
            loaded.vertices = loaded.vertices * scale
            origin = el.origin if el.origin is not None else np.eye(4)
            loaded.apply_transform(origin)
            pieces.append(loaded)
        merged = _merge_meshes(pieces)
        if merged is not None:
            return merged, kind
    raise MeshUnavailable(f"link {getattr(link, 'name', '?')!r}: no loadable collision or visual mesh geometry")


def load_mesh_with_fixed_fallback(urdf_path: Path, robot, fixed_children: Dict[str, List[str]],
                                   T0: Dict[str, np.ndarray], body: str):
    """Some hands (ORCA's ``*_jointbody`` links, for a zero-mass joint
    pivot) put no geometry at all on the digit-chain body itself and carry
    it instead on a link reached through one or more FIXED joints (the same
    situation ``hand_sampler/design_space.py``'s ``_nearest_geometry``
    handles for its own box-extraction pattern). BFS through ``body``'s
    fixed-joint descendants (``fixed_children``, built by the caller from
    the ORIGINAL ``KinematicModel``) until one has a loadable collision or
    visual mesh, then re-expresses it in ``body``'s own local frame via
    ``T0`` (the SAME zero-config forward-kinematics dict the caller already
    has, which -- like ``projection.py``'s own use of it -- covers every
    body, fixed-joint children included). Returns ``(mesh, kind,
    mesh_body_name)``; raises ``MeshUnavailable`` if nothing on the whole
    fixed subtree has geometry."""
    frontier = [body]
    seen = set()
    while frontier:
        nxt: List[str] = []
        for b in frontier:
            if b in seen:
                continue
            seen.add(b)
            link = robot.link_map.get(b)
            if link is not None:
                try:
                    mesh, kind = load_link_mesh(urdf_path, link)
                except MeshUnavailable:
                    mesh = None
                if mesh is not None:
                    if b != body:
                        T = np.linalg.inv(T0[body]) @ T0[b]
                        mesh = mesh.copy()
                        mesh.apply_transform(T)
                    return mesh, kind, b
            nxt.extend(fixed_children.get(b, []))
        frontier = nxt
    raise MeshUnavailable(f"no loadable mesh on {body!r} or any of its fixed-joint descendants")


def slice_mesh(mesh, plane_origin: np.ndarray, plane_normal: np.ndarray) -> Optional[np.ndarray]:
    """Slice ``mesh`` (in its own local frame) with the plane through
    ``plane_origin`` normal to ``plane_normal`` (both in that same frame).
    Returns the largest-area closed loop as an (N, 3) point array, or
    ``None`` if the plane does not intersect the mesh."""
    try:
        section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    except Exception:
        return None
    if section is None:
        return None
    loops = section.discrete
    if not loops:
        return None
    u, v = _plane_basis(plane_normal)
    best, best_area = None, -1.0
    for loop in loops:
        pts = np.asarray(loop, dtype=float)
        rel = pts - plane_origin
        xy = np.stack([rel @ u, rel @ v], axis=1)
        area = _polygon_area(xy)
        if area > best_area:
            best_area, best = area, pts
    return best


# --------------------------------------------------------------------------
# Plane geometry helpers (pure numpy)
# --------------------------------------------------------------------------


def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - float(np.dot(ref, n)) * n
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def _polygon_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def points_to_axis_plane(points3d: np.ndarray, axis_point: np.ndarray, axis_dir: np.ndarray) -> np.ndarray:
    """Project 3D loop points onto the plane through ``axis_point`` normal to
    ``axis_dir``, returning 2D coordinates with the origin AT ``axis_point``
    (not the loop's own centroid) -- the capsule/rounded-rect models here are
    both centred on the link's central axis, not on the mesh's local
    centroid."""
    u, v = _plane_basis(axis_dir)
    rel = points3d - np.asarray(axis_point, dtype=float)
    return np.stack([rel @ u, rel @ v], axis=1)


def principal_axis_angle(xy: np.ndarray) -> float:
    """Angle (rad) of the contour's major principal axis, from its own
    second moments about its own centroid (orientation only -- the fit
    itself stays centred at the coordinate origin, i.e. the link axis)."""
    c = xy.mean(axis=0)
    d = xy - c
    cov = d.T @ d
    if np.allclose(cov, 0.0):
        return 0.0
    vals, vecs = np.linalg.eigh(cov)
    major = vecs[:, int(np.argmax(vals))]
    return float(math.atan2(major[1], major[0]))


def rotate_xy(xy: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(-angle), math.sin(-angle)
    R = np.array([[c, -s], [s, c]])
    return xy @ R.T


# --------------------------------------------------------------------------
# Signed distance functions (both centred at the origin)
# --------------------------------------------------------------------------


def sdf_circle(xy: np.ndarray, r: float) -> np.ndarray:
    return np.linalg.norm(xy, axis=-1) - r


def sdf_rounded_rect(xy: np.ndarray, w: float, h: float, r: float) -> np.ndarray:
    """Inigo Quilez's 2D rounded-box SDF: ``w``/``h`` are the FULL extents
    (outer bounding box, corners included), ``r`` is the corner radius
    (``0 <= r <= min(w, h) / 2``); ``r == 0`` is a plain rectangle, ``w == h
    == 2r`` is a circle."""
    b = np.array([w / 2.0, h / 2.0])
    q = np.abs(xy) - b + r
    q_pos = np.maximum(q, 0.0)
    return np.linalg.norm(q_pos, axis=-1) + np.minimum(np.max(q, axis=-1), 0.0) - r


# --------------------------------------------------------------------------
# Fits
# --------------------------------------------------------------------------


@dataclass
class CircleFit:
    radius: float
    mean_err: float
    max_err: float
    area_ratio: float  # fitted circle area / polygon area


def fit_circle_equal_area(xy: np.ndarray) -> CircleFit:
    area = _polygon_area(xy)
    r = math.sqrt(area / math.pi) if area > 0 else 0.0
    err = np.abs(sdf_circle(xy, r))
    return CircleFit(r, float(err.mean()), float(err.max()), 1.0)


def fit_circle_p95(xy: np.ndarray) -> CircleFit:
    """Minimal circle (centred at the origin) containing 95% of the boundary
    vertices: radius = the 95th percentile of vertex distance from the
    origin."""
    d = np.linalg.norm(xy, axis=-1)
    r = float(np.percentile(d, 95.0))
    area = _polygon_area(xy)
    err = np.abs(sdf_circle(xy, r))
    area_ratio = (math.pi * r * r / area) if area > 0 else float("nan")
    return CircleFit(r, float(err.mean()), float(err.max()), area_ratio)


@dataclass
class RoundedRectFit:
    w: float
    h: float
    r: float
    mean_err: float
    max_err: float


def _rr_mean_sq_err(xy: np.ndarray, w: float, h: float, r: float) -> float:
    r = min(r, 0.5 * min(w, h))
    e = sdf_rounded_rect(xy, w, h, r)
    return float(np.mean(e * e))


def fit_rounded_rect(xy: np.ndarray, n_rounds: int = 8) -> RoundedRectFit:
    """Coordinate-descent fit of an origin-centred, axis-aligned rounded
    rectangle (caller rotates ``xy`` into principal-axis coordinates first)
    minimising mean squared boundary error. Deterministic, no external
    optimiser dependency -- adequate for this desk study's grid-search-sized
    problem (hundreds of sections, three free parameters each)."""
    ext = np.abs(xy).max(axis=0)
    w, h = float(2 * ext[0]) or 1e-6, float(2 * ext[1]) or 1e-6
    r = 0.25 * min(w, h)

    def _search_1d(get, set_, lo, hi, steps):
        best_v, best_e = get(), _rr_mean_sq_err(xy, *set_(get()))
        for _ in range(6):
            grid = np.linspace(lo, hi, steps)
            for cand in grid:
                e = _rr_mean_sq_err(xy, *set_(cand))
                if e < best_e:
                    best_e, best_v = e, cand
            span = (hi - lo) / (steps - 1)
            lo, hi = max(0.0, best_v - span), best_v + span
        return best_v

    for _ in range(n_rounds):
        w = _search_1d(lambda: w, lambda v: (v, h, r), 0.5 * w, 1.5 * w, 21)
        h = _search_1d(lambda: h, lambda v: (w, v, r), 0.5 * h, 1.5 * h, 21)
        r = _search_1d(lambda: r, lambda v: (w, h, v), 0.0, 0.5 * min(w, h), 21)

    r = min(r, 0.5 * min(w, h))
    e = np.abs(sdf_rounded_rect(xy, w, h, r))
    return RoundedRectFit(w, h, r, float(e.mean()), float(e.max()))


def best_capsule_radius(sections: Sequence[np.ndarray]) -> Tuple[float, float]:
    """One capsule radius for a group of sections (per-link / per-hand /
    global), chosen to minimise the 80th percentile of per-section MAX
    boundary error -- same objective as the template search, so the two are
    comparable. Returns ``(radius, p80_of_max_err)``."""
    if not sections:
        return 0.0, 0.0
    areas = [_polygon_area(s) for s in sections]
    r0 = math.sqrt(max(areas) / math.pi) if max(areas) > 0 else 1e-6
    lo, hi = 0.2 * r0, 3.0 * r0

    def score(r: float) -> float:
        max_errs = [float(np.abs(sdf_circle(s, r)).max()) for s in sections]
        return float(np.percentile(max_errs, 80.0))

    best_r, best_s = r0, score(r0)
    for _ in range(4):
        grid = np.linspace(lo, hi, 25)
        for cand in grid:
            s = score(float(cand))
            if s < best_s:
                best_s, best_r = s, float(cand)
        span = (hi - lo) / 25
        lo, hi = max(1e-9, best_r - span), best_r + span
    return best_r, best_s


def fraction_within(sections: Sequence[np.ndarray], shape_err_fn, tol: float) -> float:
    if not sections:
        return float("nan")
    ok = 0
    for s in sections:
        if float(np.abs(shape_err_fn(s)).max()) <= tol:
            ok += 1
    return ok / len(sections)


def best_template_size(sections: Sequence[np.ndarray], aspect: float, roundedness: float) -> Tuple[float, float]:
    """For a fixed template shape (``aspect=h/w``, ``roundedness=r/h``), the
    one size ``w`` (per this group -- a link, in the template search) that
    minimises the 80th percentile of per-section max boundary error.
    Returns ``(w, p80_of_max_err)``."""
    if not sections:
        return 0.0, 0.0
    areas = [_polygon_area(s) for s in sections]
    w0 = math.sqrt(max(areas)) if max(areas) > 0 else 1e-6
    lo, hi = 0.2 * w0, 3.0 * w0

    def score(w: float) -> float:
        h = aspect * w
        r = min(roundedness * h, 0.5 * min(w, h))
        max_errs = [float(np.abs(sdf_rounded_rect(s, w, h, r)).max()) for s in sections]
        return float(np.percentile(max_errs, 80.0))

    best_w, best_s = w0, score(w0)
    for _ in range(4):
        grid = np.linspace(lo, hi, 25)
        for cand in grid:
            s = score(float(cand))
            if s < best_s:
                best_s, best_w = s, float(cand)
        span = (hi - lo) / 25
        lo, hi = max(1e-9, best_w - span), best_w + span
    return best_w, best_s


__all__ = [
    "MeshUnavailable", "resolve_mesh_filename", "load_link_mesh", "load_mesh_with_fixed_fallback", "slice_mesh",
    "points_to_axis_plane", "principal_axis_angle", "rotate_xy",
    "sdf_circle", "sdf_rounded_rect",
    "CircleFit", "fit_circle_equal_area", "fit_circle_p95",
    "RoundedRectFit", "fit_rounded_rect",
    "best_capsule_radius", "fraction_within", "best_template_size",
]
