"""Phenotype distance between two ``KinematicModel``s (E0 experiment infra).

``phenotype_distance`` is a cheap, deterministic dissimilarity measure used
to study neutral drift / redundancy: how far apart (in tip-position and
structural terms) two derived hands are. It is NOT a fitness or task score.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

import numpy as np

from .coords import independent_joints, q_from_u, sample_configurations
from .fk import forward_kinematics
from .kinematics import KinematicModel
from .proxy import tip_frames

DEFAULT_RADIUS_M = 0.01


def _root_length_m(model: KinematicModel) -> float:
    for f in model.frames:
        if f.name == "root_tip":
            return float(np.linalg.norm(np.asarray(f.pose.xyz, dtype=float)))
    return 0.0


def _total_length_m(model: KinematicModel) -> float:
    body_names = {b.name for b in model.bodies}
    total = 0.0
    for f in model.frames:
        if f.body in body_names and f.name == f"{f.body}_tip":
            total += float(np.linalg.norm(np.asarray(f.pose.xyz, dtype=float)))
    return total


def _digit_count(model: KinematicModel) -> int:
    """Number of top-level digits: joints whose parent is palm-flagged and
    whose child is not (a branch digit's first joint mounts on a
    non-palm phalanx body, so this never double-counts branches)."""
    palm_names = {b.name for b in model.bodies if b.palm}
    non_palm_names = {b.name for b in model.bodies if not b.palm}
    return sum(1 for j in model.joints if j.parent in palm_names and j.child in non_palm_names)


def _palm_body_count(model: KinematicModel) -> int:
    return sum(1 for b in model.bodies if b.palm)


def _tip_positions_at(model: KinematicModel, frames: List[str], u: Mapping[str, float]) -> np.ndarray:
    q = q_from_u(model, u)
    T = forward_kinematics(model, q)
    return np.asarray([T[f][:3, 3] for f in frames], dtype=float)


def _greedy_assignment(dist: np.ndarray) -> List[Tuple[int, int]]:
    """Greedy nearest-first assignment: repeatedly match the globally
    smallest remaining (row, col) distance, remove that row and column,
    until one side is exhausted. Not globally optimal (unlike Hungarian)
    but deterministic, dependency-free, and adequate for this diagnostic."""
    d = np.array(dist, dtype=float, copy=True)
    n_rows, n_cols = d.shape
    rows_left = list(range(n_rows))
    cols_left = list(range(n_cols))
    matches: List[Tuple[int, int]] = []
    for _ in range(min(n_rows, n_cols)):
        sub = d[np.ix_(rows_left, cols_left)]
        flat_idx = int(np.argmin(sub))
        ri, ci = divmod(flat_idx, sub.shape[1])
        i, j = rows_left[ri], cols_left[ci]
        matches.append((i, j))
        rows_left.remove(i)
        cols_left.remove(j)
    return matches


def _per_config_mean_distance(pos_a: np.ndarray, pos_b: np.ndarray, penalty: float) -> float:
    n_a, n_b = len(pos_a), len(pos_b)
    if n_a == 0 and n_b == 0:
        return 0.0
    if n_a == 0 or n_b == 0:
        return penalty
    dist = np.linalg.norm(pos_a[:, None, :] - pos_b[None, :, :], axis=-1)
    matches = _greedy_assignment(dist)
    matched_costs = [float(dist[i, j]) for i, j in matches]
    n_unmatched = abs(n_a - n_b)
    total = sum(matched_costs) + n_unmatched * penalty
    denom = max(n_a, n_b)
    return total / denom


def _tip_positions_from_q(model: KinematicModel, frames: List[str], q: Mapping[str, float]) -> np.ndarray:
    T = forward_kinematics(model, q)
    return np.asarray([T[f][:3, 3] for f in frames], dtype=float)


def phenotype_distance(a: KinematicModel, b: KinematicModel, seed: int, n_configs: int = 32) -> Dict[str, float]:
    """Dissimilarity between two derived hands.

    ``tip_displacement_m`` (I14 fix): ``a`` is treated as the PARENT and ``b``
    as the CHILD (this is how every caller -- ``e1_locality``, ``e2_drift``'s
    aggregate readings -- already uses this function). Only ``a``'s own
    u-configurations are drawn (``sample_configurations(a, n_configs,
    seed)``); ``b``'s configuration is built by ALIGNING joints BY NAME
    rather than sampling ``b`` independently: for each of ``a``'s sampled
    configurations, expand it to a full per-joint ``q`` via ``q_from_u``,
    then for ``b`` reuse that same joint's value for every joint name
    present in both ``a.joints`` and ``b.joints`` (this is well-defined
    because ``derive.py``'s ``vary`` operators never rename an existing
    joint), and set the value of every movable joint present ONLY in ``b``
    to 0.0 (clamped into that joint's own declared limits, if any).

    This fixes the previous (pre-I14) behaviour of drawing ``a`` and ``b``'s
    configurations INDEPENDENTLY (via two separate ``sample_configurations``
    calls): whenever a mutation adds or removes a joint, the sorted-name
    independent-joint set shifts, which shifts every later joint's draw from
    the shared ``np.random.default_rng(seed)`` stream even though that
    joint's own value didn't structurally change -- an accepted-review-
    documented noise floor of ~94 mm (a hand vs a hand with an unrelated new
    joint, under independently-drawn configs, showed tip displacement on that
    order even though every SHARED joint was mechanically unaffected).
    Aligning by name means two structurally identical models (or a parent
    and a child that share every joint) get IDENTICAL per-joint values, so
    their tip displacement is exactly 0 regardless of ``seed``.

    For each of ``a``'s ``n_configs`` (+ extremal) configurations, tip
    positions are matched between the two models by greedy nearest-first
    assignment on Euclidean distance; any tip left unmatched (unequal tip
    counts) is penalized by ``(root_length_a + root_length_b) / 2`` instead
    of a real distance, and the per-config score is the mean over
    ``max(n_tips_a, n_tips_b)`` terms. ``tip_displacement_m`` is the mean of
    that per-config score over configurations (0.0 if either model has zero
    tips or zero configurations are usable).

    ``n_shared_joints``: the number of joint NAMES present in both
    ``a.joints`` and ``b.joints`` (regardless of movable/fixed type) -- how
    much of the alignment above is actually reusing a real shared value
    (vs. falling back to the child-only 0.0 default).

    The remaining keys are exact, seed-independent structural deltas.
    """
    frames_a = tip_frames(a)
    frames_b = tip_frames(b)
    configs_a = sample_configurations(a, n_configs, seed)

    joints_by_name_a = {j.name: j for j in a.joints}
    joints_by_name_b = {j.name: j for j in b.joints}
    shared_names = set(joints_by_name_a) & set(joints_by_name_b)
    n_shared_joints = len(shared_names)

    n_common = len(configs_a)
    penalty = (_root_length_m(a) + _root_length_m(b)) / 2.0

    if n_common == 0:
        tip_displacement_m = 0.0
    else:
        scores = []
        for u in configs_a:
            q_a = q_from_u(a, u)
            q_b: Dict[str, float] = {}
            for name, jb in joints_by_name_b.items():
                if name in q_a:
                    # Shared joint (by name) whose value is known for ``a``
                    # (movable there): reuse it verbatim -- see docstring.
                    q_b[name] = q_a[name]
                elif jb.type in ("revolute", "continuous", "prismatic"):
                    # Movable joint only in ``b`` (or shared by name but not
                    # movable in ``a``): 0.0, clamped into its own limits.
                    val = 0.0
                    if jb.limits is not None:
                        lo, hi = jb.limits
                        val = min(max(val, lo), hi)
                    q_b[name] = val
            pos_a = _tip_positions_from_q(a, frames_a, q_a)
            pos_b = _tip_positions_from_q(b, frames_b, q_b)
            scores.append(_per_config_mean_distance(pos_a, pos_b, penalty))
        tip_displacement_m = float(np.mean(scores))

    return {
        "tip_displacement_m": tip_displacement_m,
        "n_shared_joints": float(n_shared_joints),
        "joint_count_delta": float(abs(len(a.joints) - len(b.joints))),
        "digit_count_delta": float(abs(_digit_count(a) - _digit_count(b))),
        "palm_body_delta": float(abs(_palm_body_count(a) - _palm_body_count(b))),
        "motor_delta": float(abs(len(independent_joints(a)) - len(independent_joints(b)))),
        "total_length_delta_m": float(abs(_total_length_m(a) - _total_length_m(b))),
    }
