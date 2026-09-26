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


def phenotype_distance(a: KinematicModel, b: KinematicModel, seed: int, n_configs: int = 32) -> Dict[str, float]:
    """Dissimilarity between two derived hands.

    ``tip_displacement_m``: both models are sampled with ``sample_configurations(model,
    n_configs, seed)`` (independently -- their independent-joint sets generally
    differ, so there is no shared configuration space; ``seed`` is simply reused for
    both draws). For each of the first ``min(len(configs_a), len(configs_b))``
    index-aligned configurations, tip positions are matched between the two
    models by greedy nearest-first assignment on Euclidean distance; any tip
    left unmatched (unequal tip counts) is penalized by ``(root_length_a +
    root_length_b) / 2`` instead of a real distance, and the per-config score
    is the mean over ``max(n_tips_a, n_tips_b)`` terms. ``tip_displacement_m``
    is the mean of that per-config score over configurations (0.0 if either
    model has zero tips and zero configurations are usable).

    The remaining keys are exact, seed-independent structural deltas.
    """
    frames_a = tip_frames(a)
    frames_b = tip_frames(b)
    configs_a = sample_configurations(a, n_configs, seed)
    configs_b = sample_configurations(b, n_configs, seed)
    n_common = min(len(configs_a), len(configs_b))
    penalty = (_root_length_m(a) + _root_length_m(b)) / 2.0

    if n_common == 0:
        tip_displacement_m = 0.0
    else:
        scores = []
        for k in range(n_common):
            pos_a = _tip_positions_at(a, frames_a, configs_a[k])
            pos_b = _tip_positions_at(b, frames_b, configs_b[k])
            scores.append(_per_config_mean_distance(pos_a, pos_b, penalty))
        tip_displacement_m = float(np.mean(scores))

    return {
        "tip_displacement_m": tip_displacement_m,
        "joint_count_delta": float(abs(len(a.joints) - len(b.joints))),
        "digit_count_delta": float(abs(_digit_count(a) - _digit_count(b))),
        "palm_body_delta": float(abs(_palm_body_count(a) - _palm_body_count(b))),
        "motor_delta": float(abs(len(independent_joints(a)) - len(independent_joints(b)))),
        "total_length_delta_m": float(abs(_total_length_m(a) - _total_length_m(b))),
    }
