"""Structured projection: real hand (``KinematicModel``) -> ``Derivation``
with continuous values (representation-check plan, item 2).

``project_to_derivation`` is an exact gauge change, not an approximation: it
re-expresses the SAME kinematic tree the input ``model`` already represents
in the grammar's own canonical-frame convention (mount/bend fractions and
offsets, ``direction_rpy``/``mount_rpy``/``bend_rpy``), so ``derive()`` of the
returned ``Derivation`` reproduces the original model's forward kinematics
to numerical precision. It never resamples or approximates a value; every
quantity is *computed* from the input model's own geometry.

Stdlib + numpy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..derive import Derivation, DerivationStep, DERIVATION_SCHEMA
from ..fk import forward_kinematics, rpy_to_matrix
from ..kinematics import Joint, KinematicModel, MOVABLE_TYPES
from ..rules import GRAMMAR_VERSION

_EPS = 1e-6


class ProjectionFailure(Exception):
    """Raised when ``model`` contains a construct the current schema cannot
    project (e.g. a genuine in-digit branch, or a non-revolute palm joint).
    ``report`` carries everything discovered before the failure."""

    def __init__(self, message: str, report: Dict[str, Any]):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class ProjectionResult:
    derivation: Derivation
    report: Dict[str, Any]
    name_map: Dict[str, str]
    root_transform: np.ndarray


# --------------------------------------------------------------------------
# Small linear-algebra helpers
# --------------------------------------------------------------------------


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < _EPS:
        return None
    return v / n


def matrix_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    """Deterministic inverse of ``fk.rpy_to_matrix`` (``R = Rz(yaw) Ry(pitch)
    Rx(roll)``), with a gimbal-lock fallback (``pitch`` at +/-90 degrees:
    ``yaw`` is pinned to 0 and ``roll`` recovered from the remaining
    off-diagonal terms). Any valid ``(roll, pitch, yaw)`` reproducing ``R``
    is acceptable -- FK only ever consumes the recomposed matrix, never the
    triple itself -- so gimbal lock is a non-issue for correctness here."""
    r20 = float(R[2, 0])
    cp = math.hypot(float(R[0, 0]), float(R[1, 0]))
    if cp < 1e-9:
        pitch = math.atan2(-r20, cp)
        yaw = 0.0
        roll = math.atan2(-float(R[1, 2]), float(R[1, 1]))
    else:
        pitch = math.atan2(-r20, cp)
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
    return (roll, pitch, yaw)


def _frame(z: np.ndarray, primary: Optional[np.ndarray], fallback: Optional[np.ndarray]) -> np.ndarray:
    """Build a right-handed orthonormal ``[x, y, z]`` (as columns) with the
    given unit ``z``, choosing ``x`` as the unit component of ``primary``
    perpendicular to ``z`` (so ``primary`` -- e.g. a joint axis -- lands in
    the x-z plane with a non-negative x-component by construction), falling
    back to ``fallback``'s perpendicular component, then to world x/y, if
    ``primary``/``fallback`` are parallel to ``z``."""
    z = z / np.linalg.norm(z)
    for cand in (primary, fallback, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
        if cand is None:
            continue
        v = np.asarray(cand, dtype=float)
        v_perp = v - float(np.dot(v, z)) * z
        x = _unit(v_perp)
        if x is not None:
            y = np.cross(z, x)
            return np.stack([x, y, z], axis=1)
    raise ProjectionFailure("could not construct a canonical frame", {})


def _rel_rpy(R_parent: np.ndarray, R_child: np.ndarray) -> Tuple[float, float, float]:
    return matrix_to_rpy(R_parent.T @ R_child)


# --------------------------------------------------------------------------
# Structural analysis of the original model
# --------------------------------------------------------------------------


@dataclass
class _Merged:
    anchor: Dict[str, str]
    # merged parent body name -> movable Joint objects whose composed parent
    # is that merged body (joint.parent/joint.child keep their ORIGINAL
    # names; joint.child is always itself a merged body, since every
    # merged body other than root is exactly the child of some movable
    # joint -- see the module docstring above).
    movable_children: Dict[str, List[Joint]]
    parent_of: Dict[str, str]


def _merge_fixed(model: KinematicModel) -> _Merged:
    children: Dict[str, List[Joint]] = {}
    for j in model.joints:
        children.setdefault(j.parent, []).append(j)

    anchor: Dict[str, str] = {model.root: model.root}
    movable_children: Dict[str, List[Joint]] = {}
    parent_of: Dict[str, str] = {}
    stack: List[Tuple[str, str]] = [(model.root, model.root)]
    while stack:
        b, anc = stack.pop()
        for j in children.get(b, []):
            if j.type == "fixed":
                anchor[j.child] = anc
                stack.append((j.child, anc))
            else:
                anchor[j.child] = j.child
                movable_children.setdefault(anc, []).append(j)
                parent_of[j.child] = anc
                stack.append((j.child, j.child))
    return _Merged(anchor=anchor, movable_children=movable_children, parent_of=parent_of)


def _deepest_fixed_leaf(model: KinematicModel, body: str) -> str:
    """Descend from ``body`` through fixed joints only, returning the
    farthest (by cumulative offset norm) leaf reached -- the fingertip
    candidate for a digit ending at ``body``. Returns ``body`` itself if it
    has no fixed-only descendants."""
    children: Dict[str, List[Joint]] = {}
    for j in model.joints:
        if j.type == "fixed":
            children.setdefault(j.parent, []).append(j)
    T0 = forward_kinematics(model, {})
    best_body, best_dist = body, 0.0
    stack = [body]
    seen = set()
    while stack:
        b = stack.pop()
        if b in seen:
            continue
        seen.add(b)
        for j in children.get(b, []):
            stack.append(j.child)
            d = float(np.linalg.norm(T0[j.child][:3, 3] - T0[body][:3, 3]))
            if d > best_dist:
                best_dist, best_body = d, j.child
    return best_body


# --------------------------------------------------------------------------
# Main projection
# --------------------------------------------------------------------------


def project_to_derivation(
    model: KinematicModel,
    *,
    palm_joints: Sequence[str] = (),
    tip_frames: Optional[Mapping[str, str]] = None,
) -> ProjectionResult:
    tip_frames = dict(tip_frames or {})
    report: Dict[str, Any] = {
        "projection": f"grammar {GRAMMAR_VERSION}",
        "merged_fixed_joints": [],
        "palm_bodies": [],
        "digits": {},
        "fingertip_undefined": [],
        "coupling_not_in_structure": [],
        "dependent_limits_replaced": [],
        "branch_not_projected": [],
    }

    T0 = forward_kinematics(model, {})
    merged = _merge_fixed(model)

    # merged_fixed_joints: every original joint whose type is "fixed".
    report["merged_fixed_joints"] = [j.name for j in model.joints if j.type == "fixed"]

    # merged tree children (by merged body name).
    tree_children: Dict[str, List[str]] = {}
    for m, joints in merged.movable_children.items():
        tree_children[m] = [j.child for j in joints]

    leaf_count_cache: Dict[str, int] = {}

    def leaf_count(m: str) -> int:
        if m in leaf_count_cache:
            return leaf_count_cache[m]
        kids = tree_children.get(m, [])
        val = 1 if not kids else sum(leaf_count(k) for k in kids)
        leaf_count_cache[m] = val
        return val

    all_merged_bodies = {model.root} | {j.child for js in merged.movable_children.values() for j in js}

    palm_set = {model.root} | {m for m in all_merged_bodies if leaf_count(m) >= 2}

    # Explicit palm_joints: force the joint's child (+ path to root) into
    # the palm set.
    joint_by_name = {j.name: j for j in model.joints}
    for jn in palm_joints:
        j = joint_by_name.get(jn)
        if j is None or j.type not in MOVABLE_TYPES:
            raise ProjectionFailure(f"palm_joints entry {jn!r} is not a movable joint in model", report)
        cur = j.child
        while cur not in palm_set:
            palm_set.add(cur)
            cur = merged.parent_of.get(cur)
            if cur is None:
                break

    # Sanity: every merged body with >1 movable child must be in palm_set
    # under this rule (leaf_count >= 2 there too); anything violating this
    # is a genuine in-digit branch this schema cannot project.
    for m in all_merged_bodies:
        if len(tree_children.get(m, [])) > 1 and m not in palm_set:
            report["branch_not_projected"].append(m)
    if report["branch_not_projected"]:
        raise ProjectionFailure(f"in-digit branch not projected: {report['branch_not_projected']}", report)

    for j in model.joints:
        if j.type in MOVABLE_TYPES:
            child_m = j.child
            if child_m in palm_set and j.type != "revolute":
                raise ProjectionFailure(f"palm joint {j.name!r} is not revolute (type={j.type!r})", report)

    # ---- canonical frames --------------------------------------------------

    R_can: Dict[str, np.ndarray] = {}
    length_of: Dict[str, float] = {}
    axis_world: Dict[str, np.ndarray] = {}  # merged body -> world axis of its own driving joint (non-root)
    fingertip_abs: Dict[str, Optional[np.ndarray]] = {}
    fingertip_defined: Dict[str, bool] = {}
    tip_body_of: Dict[str, str] = {}  # last-phalanx merged body -> ORIGINAL tip body name

    def pos(body_or_point) -> np.ndarray:
        if isinstance(body_or_point, str):
            return T0[body_or_point][:3, 3]
        return body_or_point

    # Root frame.
    root_children = tree_children.get(model.root, [])
    if root_children:
        centroid = np.mean([pos(c) for c in root_children], axis=0)
        z_root = _unit(centroid - pos(model.root))
        if z_root is None:
            z_root = np.array([0.0, 0.0, 1.0])
        fallback = pos(root_children[0]) - pos(model.root)
    else:
        z_root = np.array([0.0, 0.0, 1.0])
        fallback = np.array([1.0, 0.0, 0.0])
    R_can[model.root] = _frame(z_root, None, fallback)
    if root_children:
        proj = [float(np.dot(pos(c) - pos(model.root), z_root)) for c in root_children]
        length_of[model.root] = max(max(proj), _EPS)
    else:
        length_of[model.root] = _EPS

    # Non-root palm bodies (BFS order from root so a parent is always
    # already framed before its children).
    order: List[str] = []
    stack = list(root_children)
    while stack:
        b = stack.pop(0)
        order.append(b)
        stack.extend(tree_children.get(b, []))
    # order currently mixes palm and digit-start bodies; process only palm ones now.
    for b in order:
        if b not in palm_set or b in R_can:
            continue
        parent = merged.parent_of[b]
        j = next(jj for jj in merged.movable_children[parent] if jj.child == b)
        axis_world[b] = T0[b][:3, :3] @ np.asarray(j.axis, dtype=float)
        kids = tree_children.get(b, [])
        if kids:
            centroid = np.mean([pos(c) for c in kids], axis=0)
            z_b = _unit(centroid - pos(b))
        else:
            z_b = None
        if z_b is None:
            z_b = _unit(pos(b) - pos(parent))
        if z_b is None:
            z_b = np.array([0.0, 0.0, 1.0])
        fallback = T0[b][:3, 0]
        R_can[b] = _frame(z_b, axis_world[b], fallback)
        if kids:
            proj = [float(np.dot(pos(c) - pos(b), z_b)) for c in kids]
            length_of[b] = max(max(proj), _EPS)
        else:
            length_of[b] = _EPS

    # ---- digits -------------------------------------------------------------

    digit_chains: List[Tuple[str, List[str], List[Joint]]] = []  # (mount_body, chain_bodies, chain_joints)
    for m in sorted(palm_set):
        for j in merged.movable_children.get(m, []):
            if j.child in palm_set:
                continue
            chain_bodies = [j.child]
            chain_joints = [j]
            cur = j.child
            while True:
                kids = tree_children.get(cur, [])
                if len(kids) == 1:
                    nxt_j = merged.movable_children[cur][0]
                    chain_bodies.append(nxt_j.child)
                    chain_joints.append(nxt_j)
                    cur = nxt_j.child
                elif len(kids) == 0:
                    break
                else:
                    raise ProjectionFailure(f"unexpected branch at {cur!r} inside digit", report)
            digit_chains.append((m, chain_bodies, chain_joints))

    def frame_digit(mount_body: str, chain_bodies: List[str], chain_joints: List[Joint], digit_key: str) -> None:
        n = len(chain_bodies)
        for i, (body, j) in enumerate(zip(chain_bodies, chain_joints)):
            axis_world[body] = T0[body][:3, :3] @ np.asarray(j.axis, dtype=float)
            is_last = i == n - 1
            if is_last:
                if digit_key in tip_frames:
                    tip_body = tip_frames[digit_key]
                    tip_pt = pos(tip_body)
                else:
                    tip_body = _deepest_fixed_leaf(model, body)
                    tip_pt = pos(tip_body)
                tip_body_of[body] = tip_body
                dist = float(np.linalg.norm(tip_pt - pos(body)))
                if dist < _EPS:
                    fingertip_defined[body] = False
                    report["fingertip_undefined"].append(digit_key)
                    target = None
                else:
                    fingertip_defined[body] = True
                    target = tip_pt
                fingertip_abs[body] = tip_pt if fingertip_defined[body] else None
            else:
                target = pos(chain_bodies[i + 1])
                dist = float(np.linalg.norm(target - pos(body)))

            if target is not None and dist >= _EPS:
                z_b = _unit(target - pos(body))
            else:
                # coincident: use the next non-coincident point downstream,
                # or fall back to the incoming direction / world z.
                z_b = None
                for k in range(i + 1, n):
                    cand = pos(chain_bodies[k])
                    z_b = _unit(cand - pos(body))
                    if z_b is not None:
                        break
                if z_b is None and is_last and fingertip_abs.get(body) is not None:
                    z_b = _unit(fingertip_abs[body] - pos(body))
                if z_b is None:
                    prev_pt = pos(mount_body) if i == 0 else pos(chain_bodies[i - 1])
                    z_b = _unit(pos(body) - prev_pt)
                if z_b is None:
                    z_b = np.array([0.0, 0.0, 1.0])

            fallback = T0[body][:3, 0]
            R_can[body] = _frame(z_b, axis_world[body], fallback)
            length_of[body] = 0.0 if dist < _EPS else dist

    for (mount_body, chain_bodies, chain_joints) in digit_chains:
        frame_digit(mount_body, chain_bodies, chain_joints, chain_bodies[-1])

    root_transform = np.eye(4)
    root_transform[:3, :3] = R_can[model.root]

    # ---- emit steps ----------------------------------------------------------

    next_uid = [0]

    def alloc_uid() -> int:
        u = next_uid[0]
        next_uid[0] += 1
        return u

    steps: List[DerivationStep] = []
    name_map: Dict[str, str] = {}

    palm_name_of: Dict[str, str] = {model.root: "root"}
    palm_order = [b for b in order if b in palm_set]
    for idx, b in enumerate(palm_order):
        palm_name_of[b] = f"palm{idx}"

    for idx, b in enumerate(palm_order):
        parent_orig = merged.parent_of[b]
        parent_name = palm_name_of[parent_orig]
        j = next(jj for jj in merged.movable_children[parent_orig] if jj.child == b)
        local_vec = R_can[parent_orig].T @ (pos(b) - pos(parent_orig))
        L_parent = length_of[parent_orig]
        mount_frac = float(local_vec[2] / L_parent)
        mount_offset = (float(local_vec[0]), float(local_vec[1]))
        direction_rpy = _rel_rpy(R_can[parent_orig], R_can[b])
        axis_local = tuple(float(v) for v in (R_can[b].T @ axis_world[b]))
        limits = tuple(float(v) for v in j.limits) if j.limits is not None else (0.0, 0.0)
        uid = alloc_uid()
        steps.append(DerivationStep(path=f"palm/{idx}", production="PalmBody", params={
            "name": palm_name_of[b], "parent": parent_name, "mount_frac": mount_frac,
            "mount_offset": mount_offset, "length": float(length_of[b]),
            "direction_rpy": tuple(float(v) for v in direction_rpy),
            "has_joint": True, "axis": axis_local, "limits": limits, "uid": uid,
        }))
        name_map[j.name] = f"{palm_name_of[b]}_j"
        report["palm_bodies"].append(palm_name_of[b])
        # PalmBody steps have no coupling schema: a mimic on a palm joint (e.g.
        # SVH j5 driven by thumb opposition, ARMS CMC5 following CMC4) is kept
        # as an independent palm joint and reported, never dropped silently.
        if any(c.dependent == j.name for c in model.couplings):
            report["coupling_not_in_structure"].append(j.name)

    digit_id_of: Dict[Tuple[str, ...], str] = {}
    next_digit_id = 1
    for (mount_body, chain_bodies, chain_joints) in digit_chains:
        digit_id = str(next_digit_id)
        next_digit_id += 1
        mount_name = palm_name_of[mount_body]
        b0 = chain_bodies[0]
        local_vec = R_can[mount_body].T @ (pos(b0) - pos(mount_body))
        L_mount = length_of[mount_body]
        mount_frac = float(local_vec[2] / L_mount)
        mount_rpy = _rel_rpy(R_can[mount_body], R_can[b0])
        phalanx_count = len(chain_bodies)
        d_uid = alloc_uid()
        steps.append(DerivationStep(path=f"digit/{digit_id}", production="Digit", params={
            "digit_id": digit_id, "mount": mount_name, "mount_frac": mount_frac,
            "mount_rpy": tuple(float(v) for v in mount_rpy), "phalanx_count": phalanx_count,
            "top_level": True, "depth": 0, "uid": d_uid,
        }))

        source_p_for_joint: Dict[str, int] = {}
        digit_report = {"phalanges": phalanx_count, "mount": mount_name}
        for i, (body, j) in enumerate(zip(chain_bodies, chain_joints)):
            axis_l = tuple(float(v) for v in (R_can[body].T @ axis_world[body]))
            module: Dict[str, Any]
            coupling_note = None
            src = None
            for c in model.couplings:
                if c.dependent == j.name:
                    src = c
                    break
            if src is not None and src.source in source_p_for_joint:
                sp = source_p_for_joint[src.source]
                module = {"kind": "Coupled", "axis": axis_l, "source_p": sp,
                          "multiplier": float(src.multiplier), "offset": float(src.offset)}
                # informational: does the DECLARED limit match derive()'s
                # own image-of-source computation?
                src_joint = next(jj for jj in chain_joints if jj.name == src.source)
                if src_joint.limits is not None and j.limits is not None:
                    slo, shi = src_joint.limits
                    lo_img = src.multiplier * slo + src.offset
                    hi_img = src.multiplier * shi + src.offset
                    if lo_img > hi_img:
                        lo_img, hi_img = hi_img, lo_img
                    if (abs(lo_img - j.limits[0]) > 1e-9 or abs(hi_img - j.limits[1]) > 1e-9):
                        report["dependent_limits_replaced"].append(j.name)
            elif src is not None:
                module = {"kind": "R", "axis": axis_l, "limits": tuple(float(v) for v in j.limits)}
                report["coupling_not_in_structure"].append(j.name)
            elif j.type == "revolute":
                module = {"kind": "R", "axis": axis_l, "limits": tuple(float(v) for v in j.limits)}
            elif j.type == "continuous":
                module = {"kind": "C", "axis": axis_l}
            elif j.type == "prismatic":
                module = {"kind": "P", "axis": axis_l, "limits": tuple(float(v) for v in j.limits)}
            else:
                raise ProjectionFailure(f"joint {j.name!r} has unsupported type {j.type!r}", report)

            if module["kind"] == "R":
                source_p_for_joint[j.name] = i

            if i == 0:
                bend_offset = (float(local_vec[0]), float(local_vec[1]))
                bend_rpy = (0.0, 0.0, 0.0)
            else:
                prev = chain_bodies[i - 1]
                lv = R_can[prev].T @ (pos(body) - pos(prev))
                bend_offset = (float(lv[0]), float(lv[1]))
                bend_rpy = tuple(float(v) for v in _rel_rpy(R_can[prev], R_can[body]))

            p_uid = alloc_uid()
            steps.append(DerivationStep(path=f"digit/{digit_id}/phalanx/{i}", production="Phalanx", params={
                "digit_id": digit_id, "p": i, "module": module, "length": float(length_of[body]),
                "branch_digit_count": 0, "uid": p_uid, "bend_rpy": bend_rpy, "bend_offset": bend_offset,
            }))
            name_map[j.name] = f"d{digit_id}p{i + 1}_j"

        last_body = chain_bodies[-1]
        if fingertip_defined.get(last_body, False):
            name_map[tip_body_of[last_body]] = f"d{digit_id}p{phalanx_count}_tip"
        report["digits"][digit_id] = digit_report

    hand_uid = None  # 'hand' step is not counted in the uid stream (has none)
    palm_body_count = len(palm_order)
    digit_count = len(digit_chains)
    root_length = float(length_of[model.root])
    steps.insert(0, DerivationStep(path="hand", production="Hand", params={
        "digit_count": digit_count, "palm_body_count": palm_body_count, "root_length": root_length,
        "capsule_radius_m": 0.01,
    }))

    derivation = Derivation(seed=-1, grammar_version=GRAMMAR_VERSION, steps=tuple(steps))
    return ProjectionResult(derivation=derivation, report=report, name_map=name_map, root_transform=root_transform)
