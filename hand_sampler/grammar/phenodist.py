"""Phenotype distance between two ``KinematicModel``s (E0 experiment infra).

``phenotype_distance`` is a cheap, deterministic dissimilarity measure used
to study neutral drift / redundancy: how far apart (in tip-position and
structural terms) two derived hands are. It is NOT a fitness or task score.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

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


def phenotype_distance(a: KinematicModel, b: KinematicModel, seed: int, n_configs: int = 32,
                        identity_a: Optional[Mapping[str, int]] = None,
                        identity_b: Optional[Mapping[str, int]] = None) -> Dict[str, float]:
    """Dissimilarity between two derived hands.

    ``tip_displacement_m``: ``a`` is treated as the PARENT and ``b`` as the
    CHILD (this is how every caller -- ``e1_locality``, ``e2_drift``'s
    aggregate readings -- already uses this function). Only ``a``'s own
    u-configurations are drawn (``sample_configurations(a, n_configs,
    seed)``); ``b``'s configuration is built by ALIGNMENT rather than
    sampling ``b`` independently (I14 fix; see below for why).

    ``identity_a``/``identity_b`` (I15 fix 1, optional): ``derive.
    joint_identity(derivation)`` maps for ``a`` and ``b`` respectively
    (``{joint_name: uid}``, ``uid`` the STABLE id of the derivation step
    that created that joint). When BOTH are given, alignment matches a
    joint of ``b`` to a joint of ``a`` by shared ``uid`` -- immune to
    insert/delete_phalanx, remove_digit or remove_palm_body renumbering a
    joint's NAME without it being new structure (``opus-review-final.md``'s
    "77/200 insert_phalanx children had >= 2 same-name joints with changed
    records" defect). When either is omitted (the I14 default), alignment
    falls back to matching by joint NAME, which is exact only because
    ``derive.py``'s ``vary`` operators never rename a joint that is
    otherwise unchanged -- true for most operators, but not for the
    renumbering ones above.

    For EITHER alignment key, the value assigned to each of ``b``'s
    INDEPENDENT joints (``coords.independent_joints(b)``) is: the matched
    joint's value in ``a``'s expanded ``q`` (via ``q_from_u(a, u)``) if a
    match exists, clamped into ``b``'s own declared limits (if any);
    else 0.0, clamped the same way. ``b``'s DEPENDENT (coupled) joints are
    then obtained by ``coords.q_from_u(b, u_b)`` -- i.e. through ``b``'s OWN
    coupling formula applied to ``b``'s OWN independent values -- never by
    copying a matched joint's raw value directly (I15 fix 1: the previous
    scheme copied ANY shared-by-name joint's value verbatim, including a
    coupled dependent one, silently ignoring the child's own coupling
    multiplier/offset -- documented in ``opus-review-final.md`` as
    "``step_coupling`` reads 0 mm by construction" and "``step_limits``
    reads 0 because shared values are not clamped").

    This (I14) fixes the pre-I14 behaviour of drawing ``a`` and ``b``'s
    configurations INDEPENDENTLY (via two separate ``sample_configurations``
    calls): whenever a mutation adds or removes a joint, the sorted-name
    independent-joint set shifts, which shifts every later joint's draw from
    the shared ``np.random.default_rng(seed)`` stream even though that
    joint's own value didn't structurally change -- an accepted-review-
    documented noise floor of ~94 mm (a hand vs a hand with an unrelated new
    joint, under independently-drawn configs, showed tip displacement on that
    order even though every SHARED joint was mechanically unaffected).
    Aligning means two structurally identical models (or a parent and a
    child that share every joint, with matching couplings) get IDENTICAL
    per-joint values, so their tip displacement is exactly 0 regardless of
    ``seed``.

    For each of ``a``'s ``n_configs`` (+ extremal) configurations, tip
    positions are matched between the two models by greedy nearest-first
    assignment on Euclidean distance; any tip left unmatched (unequal tip
    counts) is penalized by ``(root_length_a + root_length_b) / 2`` instead
    of a real distance, and the per-config score is the mean over
    ``max(n_tips_a, n_tips_b)`` terms. ``tip_displacement_m`` is the mean of
    that per-config score over configurations (0.0 if either model has zero
    tips or zero configurations are usable).

    ``n_shared_joints``: the number of joints matched between ``a`` and
    ``b`` by the SAME alignment key used above (uid if both identities were
    given, else name) -- how much of the alignment is actually reusing a
    real shared value (vs. falling back to the child-only 0.0 default).

    The remaining keys are exact, seed-independent structural deltas.
    """
    frames_a = tip_frames(a)
    frames_b = tip_frames(b)
    configs_a = sample_configurations(a, n_configs, seed)

    joint_by_name_b = {j.name: j for j in b.joints}
    indep_b = independent_joints(b)

    use_uid = identity_a is not None and identity_b is not None
    if use_uid:
        uid_to_name_a: Dict[int, str] = {}
        for name, uid in identity_a.items():
            uid_to_name_a[uid] = name  # last write wins; uids are expected unique.

        def _match_name_a(name_b: str) -> Optional[str]:
            uid = identity_b.get(name_b)
            return uid_to_name_a.get(uid) if uid is not None else None
    else:
        names_a = {j.name for j in a.joints}

        def _match_name_a(name_b: str) -> Optional[str]:
            return name_b if name_b in names_a else None

    n_shared_joints = sum(1 for name_b in joint_by_name_b if _match_name_a(name_b) is not None)

    n_common = len(configs_a)
    penalty = (_root_length_m(a) + _root_length_m(b)) / 2.0

    if n_common == 0:
        tip_displacement_m = 0.0
    else:
        scores = []
        for u in configs_a:
            q_a = q_from_u(a, u)
            u_b: Dict[str, float] = {}
            for name_b in indep_b:
                jb = joint_by_name_b[name_b]
                name_a = _match_name_a(name_b)
                val = q_a.get(name_a, 0.0) if name_a is not None else 0.0
                if jb.limits is not None:
                    lo, hi = jb.limits
                    val = min(max(val, lo), hi)
                u_b[name_b] = val
            # ``b``'s dependent (coupled) joints come from ITS OWN coupling
            # formula applied to ``u_b`` above -- never copied directly
            # (I15 fix 1; see docstring).
            q_b = q_from_u(b, u_b)
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
