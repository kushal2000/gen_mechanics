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
- root-frame "front" (I14 fix): the root body's own frame is always the
  identity transform at q=0 (the root has no parent joint), so any fixed
  axis of the root frame coincides with that axis of the global/world
  frame. ``derive.py`` extends every body's own segment along that body's
  LOCAL +z (a ``"<body>_tip"`` frame always sits at local ``(0, 0,
  length)`` -- see ``derive.derive``'s ``root_tip``/phalanx frames), so the
  palm-facing "front" direction by derive.py's own convention is world
  **+z**, not +x. This module used to say +x (an undocumented, unverified
  assumption); every "front" use below (``reach_coverage``, and
  ``antipodal_pinch``'s sphere placement) now uses +z and says so at its
  own definition.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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


def _total_hand_length_m(model: KinematicModel) -> float:
    """Sum of every body's own segment length (its ``"<body>_tip"`` frame's
    distance from that body's own origin), including the root -- i.e. the
    grammar's ``total_length_m`` structural metric (see
    ``phenodist._total_length_m``, duplicated here rather than imported to
    avoid a ``proxy`` <-> ``phenodist`` import cycle, since ``phenodist``
    already imports from ``proxy``). Unlike the root's own length alone
    (0.02-0.08 m, and never mutated by any operator), this is mutated by
    almost every structural operator (``insert_phalanx``, ``add_digit``,
    ``regrow_subtree``, ...), so normalizing by it (I14 fix 4) does not
    give every hand the same fixed denominator "for free"."""
    body_names = {b.name for b in model.bodies}
    total = 0.0
    for f in model.frames:
        if f.body in body_names and f.name == f"{f.body}_tip":
            total += float(np.linalg.norm(np.asarray(f.pose.xyz, dtype=float)))
    return total


def _digit_mount_point(model: KinematicModel, digit_root_body: str,
                        transforms_q0: Mapping[str, np.ndarray]) -> Optional[np.ndarray]:
    """World-frame position (at q=0) of the joint origin that attaches
    ``digit_root_body`` to its palm parent -- the digit's own "base mount
    point" (I14 fix 3). ``None`` if no such joint exists (``digit_root_body``
    is itself the model's root, which is never a digit root in practice)."""
    joint = next((j for j in model.joints if j.child == digit_root_body), None)
    if joint is None or joint.parent not in transforms_q0:
        return None
    parent_T = transforms_q0[joint.parent]
    origin_local = np.asarray(joint.origin.xyz, dtype=float)
    return (parent_T[:3, :3] @ origin_local) + parent_T[:3, 3]


def opposition(model: KinematicModel, configs: Sequence[Mapping]) -> float:
    """Fraction of CROSS-DIGIT tip PAIRS (see module docstring's
    digit-root definition -- a branch tip counts with its host top-level
    digit) that "oppose": at the pair's own closest configuration (the
    ``configs`` index minimizing tip-tip distance), the tip-tip distance is
    less than the sum of the two tips' own ``Body.radius`` (each defaulting
    to ``DEFAULT_RADIUS_M`` = 0.01 m when unset) AND the two digits' own
    base mount points (see ``_digit_mount_point``, at q=0) are at least
    0.02 m apart (I14 fix 3: without this second condition, two parallel,
    closely-mounted neighbouring fingers that merely brush past each other
    counted as "opposing").  Same-digit pairs are excluded entirely (from
    both the numerator and the denominator). Returns 0.0 if fewer than 2
    tips exist, or no cross-digit pair exists at all."""
    frames = tip_frames(model)
    if len(frames) < 2:
        return 0.0
    pos = _tip_positions(model, configs, frames)
    radii = {f: _radius_of(model, f[: -len("_tip")]) for f in frames}
    digit_roots = {f: _digit_root(model, f[: -len("_tip")]) for f in frames}
    transforms0 = forward_kinematics(model, {})
    mount_pts: Dict[str, Optional[np.ndarray]] = {}
    n = len(frames)
    hits = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            fi, fj = frames[i], frames[j]
            root_i, root_j = digit_roots[fi], digit_roots[fj]
            if root_i == root_j:
                continue
            total += 1
            d = np.linalg.norm(pos[fi] - pos[fj], axis=1)
            min_d = float(d.min())
            if min_d >= (radii[fi] + radii[fj]):
                continue
            if root_i not in mount_pts:
                mount_pts[root_i] = _digit_mount_point(model, root_i, transforms0)
            if root_j not in mount_pts:
                mount_pts[root_j] = _digit_mount_point(model, root_j, transforms0)
            mp_i, mp_j = mount_pts[root_i], mount_pts[root_j]
            if mp_i is None or mp_j is None:
                continue
            if float(np.linalg.norm(mp_i - mp_j)) >= 0.02:
                hits += 1
    return hits / total if total else 0.0


def reach_coverage(model: KinematicModel, configs: Sequence[Mapping]) -> float:
    """Volume of the convex hull of every sampled tip position that lies in
    front of the palm (world/root-frame +z -- see module docstring's I14
    fix on the "front" convention), divided by (total hand length, see
    ``_total_hand_length_m``)^3 (I14 fix 4: previously divided by the root
    segment's own length, which no operator ever mutates, so every hand
    shared essentially the same denominator regardless of how much
    structure it actually grew). Returns 0.0 if there are fewer than 4 such
    points (degenerate hull) or the hand has zero total length."""
    frames = tip_frames(model)
    if not frames:
        return 0.0
    pos = _tip_positions(model, configs, frames)
    all_pts = np.concatenate([pos[f] for f in frames], axis=0)
    front = all_pts[all_pts[:, 2] > 0.0]
    if len(front) < 4:
        return 0.0
    total_len = _total_hand_length_m(model)
    if total_len <= 0.0:
        return 0.0
    try:
        verts, faces = convex_hull_3d(front)
    except RuntimeError:
        return 0.0
    vol = abs(polytope_volume(verts, faces))
    return float(vol / (total_len ** 3))


ANTIPODAL_SPHERE_RADIUS_M = 0.03
ANTIPODAL_SPHERE_FORWARD_OFFSET_M = 0.06
ANTIPODAL_CONTACT_TOL_M = 0.005


def antipodal_pinch(model: KinematicModel, configs: Sequence[Mapping]) -> float:
    """Diagnostic pinch-grasp proxy (I14 fixes).

    A sphere of radius ``ANTIPODAL_SPHERE_RADIUS_M`` (0.03 m) is placed at a
    FIXED position in front of the palm: ``root_origin + 0.06 m along the
    root frame's own "front" axis`` -- world/root-frame +z, per this
    module's I14 docstring fix (not +x: see the module docstring). The root
    body has no parent joint, so its own frame is the identity transform
    (world origin, world axes) regardless of configuration, making the
    sphere center the fixed point ``(0, 0, 0.06)`` in world/root
    coordinates (I14 fix: previously the sphere was centered on the
    centroid of ALL sampled tip positions, a data-dependent point with no
    gradient toward "reach here").

    For EACH configuration independently (I14 fix: previously contacts from
    different configurations were pooled together and paired across
    configurations, e.g. one tip's position at config 3 against another
    tip's position at config 7 -- a pair of contacts that can never
    physically co-occur), every tip within ``ANTIPODAL_CONTACT_TOL_M``
    (5 mm) of the sphere's surface is recorded as a "contact", with its
    outward normal := the unit vector from the sphere center to that
    contact point (a sphere-surface approximation, not a true contact
    normal of the hand geometry -- "acceptable" per this fix's own spec).

    The score is ``max`` over every configuration and every pair of
    same-configuration contacts belonging to DIFFERENT digits (module
    docstring's digit-root definition) of ``1 - angle(n_a, -n_b) / pi``:
    0.0 when the two normals are parallel (the two tips push the same
    way -- not a pinch), 1.0 when they are exactly antiparallel (a and b
    close toward each other from opposite sides -- an ideal pinch), and
    everywhere continuous/monotonic in between, so (I14 fix) this proxy
    has a gradient rather than saturating at a fixed threshold. Returns 0.0
    if there are fewer than 2 tips, or no cross-digit pair of
    same-configuration near-surface contacts exists at all.
    """
    frames = tip_frames(model)
    if len(frames) < 2:
        return 0.0
    pos = _tip_positions(model, configs, frames)
    digit_roots = {f: _digit_root(model, f[: -len("_tip")]) for f in frames}
    center = np.array([0.0, 0.0, ANTIPODAL_SPHERE_FORWARD_OFFSET_M], dtype=float)
    radius = ANTIPODAL_SPHERE_RADIUS_M
    tol = ANTIPODAL_CONTACT_TOL_M

    best = 0.0
    found_any = False
    n_configs = len(configs)
    for k in range(n_configs):
        contacts: List[Tuple[str, np.ndarray]] = []  # (digit_root, unit outward normal)
        for f in frames:
            p = pos[f][k]
            dist = float(np.linalg.norm(p - center))
            if dist < 1e-12 or abs(dist - radius) >= tol:
                continue
            contacts.append((digit_roots[f], (p - center) / dist))
        for i in range(len(contacts)):
            root_i, n_i = contacts[i]
            for j in range(i + 1, len(contacts)):
                root_j, n_j = contacts[j]
                if root_i == root_j:
                    continue
                found_any = True
                cos = float(np.dot(n_i, -n_j))
                cos = max(-1.0, min(1.0, cos))
                angle = math.acos(cos)
                score = 1.0 - angle / math.pi
                if score > best:
                    best = score
    return best if found_any else 0.0


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
