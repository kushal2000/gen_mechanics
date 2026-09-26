"""Cheap geometric proxy metrics for evolvability experiments (E0).

Every function here is pure numpy, deterministic given a seed, and reads
only a ``KinematicModel`` (plus, for ``all_proxies``, a seed used to draw
u-configurations via ``coords.sample_configurations``). These proxies are
DIAGNOSTICS ONLY: they exist so tonight's experiments can characterize the
grammar's evolvability (locality, drift, reachability, redundancy) cheaply.
None of them is, or should ever be treated as, a task/fitness score for an
actual manipulation objective.

Conventions used throughout:

- "digit tip": the ``<body>_tip`` frame of a leaf, non-palm body -- a body
  that is not palm-flagged and is not the parent of any joint (see
  ``tip_frames``).
- "digit" (structural, for ``antipodal_pinch`` only): the body directly
  parented by a palm body -- i.e. the topmost non-palm body of a connected
  chain hanging off the palm. Two tips belong to "different digits" iff
  their digit-root bodies (found by walking parent joints up from the tip's
  own body until a palm-flagged parent is reached) differ. This
  deliberately treats a branch digit (see ``rules.py``) as *the same
  digit* as the phalanx chain it branches from, since it shares the same
  digit-root; this under-counts "different digit" pairs for branching
  hands, which is an accepted, documented limitation of this cheap proxy.
- root-frame "front": the root body's own frame is always the identity
  transform at q=0 (the root has no parent joint), so "the +x side of the
  root frame" and "the +x side of the global/world frame" coincide; this
  module always means the latter when it says "front".
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .coords import q_from_u, sample_configurations
from .coords import independent_joints as _independent_joints
from .fk import forward_kinematics
from .geometry import convex_hull_3d, polytope_volume
from .kinematics import KinematicModel

DEFAULT_RADIUS_M = 0.01


def _radius_of(model: KinematicModel, body_name: str) -> float:
    for b in model.bodies:
        if b.name == body_name:
            return float(b.radius) if b.radius is not None else DEFAULT_RADIUS_M
    raise KeyError(f"no such body {body_name!r}")


def tip_frames(model: KinematicModel) -> List[str]:
    """Names of the ``<body>_tip`` frames of leaf, non-palm bodies (digit
    tips): a body that is not palm-flagged and is not the parent of any
    joint. Returned in ``model.bodies`` order (deterministic)."""
    has_child = {j.parent for j in model.joints}
    frame_names = {f.name for f in model.frames}
    out = []
    for b in model.bodies:
        if b.palm or b.name in has_child:
            continue
        tip = f"{b.name}_tip"
        if tip in frame_names:
            out.append(tip)
    return out


def _tip_positions(model: KinematicModel, configs: Sequence[Mapping], frames: Sequence[str]) -> Dict[str, np.ndarray]:
    positions: Dict[str, List[np.ndarray]] = {f: [] for f in frames}
    for u in configs:
        q = q_from_u(model, u)
        T = forward_kinematics(model, q)
        for f in frames:
            positions[f].append(T[f][:3, 3].copy())
    return {f: np.asarray(v, dtype=float) for f, v in positions.items()}


def _digit_root(model: KinematicModel, body_name: str) -> str:
    """Walk parent joints up from ``body_name`` until a palm-flagged parent
    is reached; return the last non-palm body visited (``body_name`` itself
    if its own parent is already a palm body). See module docstring."""
    palm_names = {b.name for b in model.bodies if b.palm}
    parent_of = {j.child: j.parent for j in model.joints}
    cur = body_name
    while True:
        parent = parent_of.get(cur)
        if parent is None or parent in palm_names:
            return cur
        cur = parent


def _root_length_m(model: KinematicModel) -> float:
    for f in model.frames:
        if f.name == "root_tip":
            return float(np.linalg.norm(np.asarray(f.pose.xyz, dtype=float)))
    return 0.0


def opposition(model: KinematicModel, configs: Sequence[Mapping]) -> float:
    """Fraction of digit-tip PAIRS (unordered, over all ``tip_frames``)
    whose minimum tip-tip Euclidean distance over ``configs`` is less than
    the sum of the two tips' own ``Body.radius`` (each defaulting to
    ``DEFAULT_RADIUS_M`` = 0.01 m when unset) -- i.e. two capsule-shaped
    tips that can be brought into (near-)contact somewhere in the sampled
    configurations. Returns 0.0 if fewer than 2 tips exist."""
    frames = tip_frames(model)
    if len(frames) < 2:
        return 0.0
    pos = _tip_positions(model, configs, frames)
    radii = {f: _radius_of(model, f[: -len("_tip")]) for f in frames}
    n = len(frames)
    hits = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            fi, fj = frames[i], frames[j]
            d = np.linalg.norm(pos[fi] - pos[fj], axis=1)
            min_d = float(d.min())
            total += 1
            if min_d < (radii[fi] + radii[fj]):
                hits += 1
    return hits / total if total else 0.0


def reach_coverage(model: KinematicModel, configs: Sequence[Mapping]) -> float:
    """Volume of the convex hull of every sampled tip position that lies in
    front of the root palm (+x side of the world/root frame, see module
    docstring), divided by (root palm segment length)^3. Returns 0.0 if
    there are fewer than 4 such points (degenerate hull) or the root has
    zero length."""
    frames = tip_frames(model)
    if not frames:
        return 0.0
    pos = _tip_positions(model, configs, frames)
    all_pts = np.concatenate([pos[f] for f in frames], axis=0)
    front = all_pts[all_pts[:, 0] > 0.0]
    if len(front) < 4:
        return 0.0
    root_len = _root_length_m(model)
    if root_len <= 0.0:
        return 0.0
    try:
        verts, faces = convex_hull_3d(front)
    except RuntimeError:
        return 0.0
    vol = abs(polytope_volume(verts, faces))
    return float(vol / (root_len ** 3))


def antipodal_pinch(model: KinematicModel, configs: Sequence[Mapping]) -> float:
    """Diagnostic pinch-grasp proxy.

    A sphere of radius 0.03 m is placed at the centroid of every sampled
    tip position (over every ``tip_frames`` frame and every configuration).
    For each (frame, configuration) tip position within 5 mm of the
    sphere's surface (``abs(|p - center| - 0.03) < 0.005``), record its
    "contact point" and its outward normal, defined as the unit vector from
    the sphere center to that point (a sphere-surface approximation, not a
    true contact normal of the hand geometry).

    For every pair of such contact points belonging to different digits
    (see module docstring's structural digit-root definition), compute
    ``cos = dot(n_a, n_b)`` between their two outward normals. If any pair
    has its two normals within 20 degrees of antiparallel (``cos <=
    cos(160 deg)``), the score is 1.0. Otherwise the score is the best
    (maximum) value of ``(1 - cos) / 2`` over every cross-digit pair --
    0.0 when perfectly parallel, approaching 1.0 as pairs approach
    antiparallel. Returns 0.0 if there are fewer than 2 tips, or no
    cross-digit pair of near-surface contacts exists at all.
    """
    frames = tip_frames(model)
    if len(frames) < 2:
        return 0.0
    pos = _tip_positions(model, configs, frames)
    all_pts = np.concatenate([pos[f] for f in frames], axis=0)
    center = all_pts.mean(axis=0)
    radius = 0.03
    tol = 0.005

    contacts: List[Tuple[str, np.ndarray]] = []  # (digit_root, unit outward normal)
    for f in frames:
        digit_root = _digit_root(model, f[: -len("_tip")])
        d = np.linalg.norm(pos[f] - center, axis=1)
        mask = np.abs(d - radius) < tol
        for p, dist in zip(pos[f][mask], d[mask]):
            if dist < 1e-12:
                continue
            n = (p - center) / dist
            contacts.append((digit_root, n))

    best = 0.0
    found_hit = False
    cos_thresh = math.cos(math.radians(160.0))
    for i in range(len(contacts)):
        root_i, n_i = contacts[i]
        for j in range(i + 1, len(contacts)):
            root_j, n_j = contacts[j]
            if root_i == root_j:
                continue
            cos = float(np.dot(n_i, n_j))
            if cos <= cos_thresh:
                found_hit = True
            score = (1.0 - cos) / 2.0
            if score > best:
                best = score
    if found_hit:
        return 1.0
    return best if best > 0.0 else 0.0


def structural_cost(model: KinematicModel) -> float:
    """``len(independent_joints(model))`` (motor count) plus total link
    length (sum of every ``<body>_tip`` frame's segment length) divided by
    0.1 m."""
    n_motors = len(_independent_joints(model))
    total_len = 0.0
    body_names = {b.name for b in model.bodies}
    for f in model.frames:
        if f.name.endswith("_tip") and f.body in body_names and f"{f.body}_tip" == f.name:
            total_len += float(np.linalg.norm(np.asarray(f.pose.xyz, dtype=float)))
    return n_motors + total_len / 0.1


def all_proxies(model: KinematicModel, seed: int, n_configs: int = 32) -> Dict[str, float]:
    """Compute every proxy above using ``n_configs`` seeded u-configurations
    (plus the deterministic extremal configurations ``sample_configurations``
    always appends). Returns a flat dict; deterministic for a given
    ``(model, seed, n_configs)``."""
    configs = sample_configurations(model, n_configs, seed)
    return {
        "opposition": opposition(model, configs),
        "reach_coverage": reach_coverage(model, configs),
        "antipodal_pinch": antipodal_pinch(model, configs),
        "structural_cost": structural_cost(model),
        "n_tips": float(len(tip_frames(model))),
        "n_configs": float(len(configs)),
        "seed": float(seed),
    }
