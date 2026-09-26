"""E7: operator-mixture SCHEDULES -- does weighting structural operators
early and small steps late recover growth (under E5b's hard 5x6 envelope)
without giving up parent-child locality late?

E5b (``e5b_evolve_sim.py``) showed that ``UNION_weighted_nopalm`` (a fixed,
small-step-heavy pool) makes a 5-digit ``target_distance`` target unreachable
within 40 generations in 17-42% of restarts once the ``_INS`` insertion
prior is on, even though that same prior controls neutral bloat (E2). This
experiment asks whether a SCHEDULE -- structural operators front-loaded,
small steps back-loaded -- recovers reach without giving up the late-run
locality (small parent-child tip displacement) that motivated small-step
weighting in the first place.

This module reuses ``e5b_evolve_sim``'s evolution loop wholesale (envelope-
checked mutate-with-retry-and-clone-fallback, noise model, shared paired
starts, per-generation trajectory recording, bootstrap CI helpers) --
imported, not copied -- and only adds: (1) a per-generation MIXTURE getter in
place of a fixed pool, (2) a generation-third improvement-rate breakdown by
operator group (structural vs small), (3) a late-window (last 10 of 40
generations) sample of non-cloned parent-child ``phenotype_distance`` pairs
(joint_identity-aligned, as in ``e1_locality``/``e5b``'s own I15 fixes) to
report as a locality diagnostic.

Pools (all ``nopalm`` -- palm operators are dead under the envelope, which
sets ``allow_palm_joints=False``; see ``e5b_evolve_sim``'s own
``UNION_weighted_nopalm``):

- ``DEFAULT_uniform``: ``e2_drift.MIXTURES["DEFAULT_uniform"]`` (equal weight
  on the 7 default ``OPERATORS``), unchanged, every generation.
- ``UNION_nopalm_weighted``: ``e2_drift.MIXTURES["UNION_weighted"]`` with
  every palm operator dropped and renormalized (byte-identical definition to
  ``e5b_evolve_sim.POOLS["UNION_weighted_nopalm"]``, rebuilt here rather than
  imported so this module's own pool table is self-contained), unchanged,
  every generation.
- ``SCHEDULE_linear``: at generation ``g0`` of ``generations`` (0-indexed,
  the initial population's own one-shot mutation counts as ``g0=0``, offspring
  produced for evolutionary step ``g`` in ``1..generations`` use ``g0=g-1``),
  the structural share is ``s(g0) = 0.8 - 0.6 * g0/(generations-1)`` (0.8 at
  ``g0=0``, 0.2 at the last generation), split between the 3 "coarse"
  operators (``regrow_subtree``/``add_digit``/``remove_digit``) and the 4
  nopalm "minimal structural" operators (``add_minimal_digit``,
  ``remove_digit_minimal``, ``insert_phalanx``, ``delete_phalanx``) in
  ``e2_drift.UNION_weighted``'s own coarse:minimal proportions (10%:25%, a
  2:5 ratio); ``resample_parameter`` is fixed at 10% every generation; the
  remainder (``0.9 - s(g0)``, 0.1 at ``g0=0`` up to 0.7 at the last
  generation) is split equally over the 7 small-step operators (the 6
  ``SMALL_STEP_OPERATORS`` plus ``perturb_parameter``, as in ``e2_drift``'s
  ``_SMALL_GROUP``).
- ``SCHEDULE_step``: generation ``g0`` in the first ``round(generations *
  15/40)`` generations (15 of 40 by default) draws from ``DEFAULT_uniform``;
  every later generation draws from ``UNION_nopalm_weighted`` (a hard switch,
  not a blend).

Distributions: ``G_NOBRANCH_INS``, ``G_NOBRANCH`` (plain), ``G_FULL_INS`` --
identical set to E5b, dropping ``G_FULL`` for the same reason (E2 already
isolated the insertion prior's effect; this halves the factor count).

Fitness: ``target_distance`` (E3's ``anthropomorphic_staggered`` structural
distance -- the only fitness under which "reach" is a meaningful concept) and
``antipodal_pinch`` (E5's geometric proxy, included as in E5b so a schedule's
effect on a LOW-dimensional-optimum objective can be compared against its
effect on ``target_distance``'s high-dimensional one). Cost weight in
``{0, 0.01}`` (E5b's ``0.02`` dropped -- E5b already showed ``0.01`` vs
``0.02`` told the same qualitative story, and this experiment's factor count
is already 4 pools x 3 dists x 2 fitness x 2 cost = 48).

Every mechanic besides the mixture-per-generation is exactly E5b's: same hard
envelope (``fits_envelope``, ``max_digits=5``, ``max_joints_per_digit=6``,
``allow_palm_joints=False``, ``allow_branches=False``), same 8-attempt
envelope-checked mutate-with-clone-fallback, same noise model (``sigma = 0.1
* generation-0 raw-proxy range``, drawn per evaluation, PENALISED value used
for selection, RAW value logged), same random tie-breaks (never by cost),
same (16+16) mu/lambda, 40 generations, 24 restarts, starts shared across
every (dist, pool, fitness, cost) at a given (restart, i) exactly as in
``e5b_evolve_sim._shared_start_seed_int``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..derive import Derivation, derive, joint_identity
from ..phenodist import phenotype_distance
from ..variants import G_FULL_INS, G_NOBRANCH, G_NOBRANCH_INS, G_SERIAL
from . import e3_reach as _e3_reach_mod  # noqa: F401 -- see e5b_evolve_sim's own note: deferred attribute
# access only, to avoid the same runner-import cycle documented there. Never
# `from .e3_reach import ...` at module level in this file.
from .e2_drift import MIXTURES as _E2_MIXTURES
from .e2_drift import _COARSE_GROUP, _MINIMAL_GROUP, _RESAMPLE_GROUP, _SMALL_GROUP
from .e5b_evolve_sim import (
    COST_WEIGHTS,
    ENVELOPE_KWARGS,
    FITNESS_NAMES,
    _PHASE_GEN_SEED,
    _PHASE_INIT_MUTATE,
    _PHASE_NOISE,
    _PHASE_OFFSPRING_MUTATE,
    _PHASE_TIEBREAK,
    _bootstrap_ci,
    _diff_ci_paired,
    _drop_and_renormalize,
    _evaluate_population,
    _mutate_retry_envelope,
    _new_op_stats,
    _record_generation,
    _rng,
    _shared_start_seed_int,
)
from .runner import register, run_experiment
from .starts import build_start

# --------------------------------------------------------------------------
# Pools / schedules.
# --------------------------------------------------------------------------

DIST_VARIANTS: Dict[str, Any] = {
    "G_FULL_INS": G_FULL_INS, "G_NOBRANCH_INS": G_NOBRANCH_INS, "G_NOBRANCH": G_NOBRANCH,
}

_PALM_OPS: Tuple[str, ...] = ("add_palm_body", "remove_palm_body", "toggle_palm_joint")
_MINIMAL_NOPALM_GROUP: Tuple[str, ...] = tuple(op for op in _MINIMAL_GROUP if op not in _PALM_OPS)
assert len(_MINIMAL_NOPALM_GROUP) == len(_MINIMAL_GROUP) - len(_PALM_OPS)

POOLS_STATIC: Dict[str, Dict[str, float]] = {
    "DEFAULT_uniform": dict(_E2_MIXTURES["DEFAULT_uniform"]),
    "UNION_nopalm_weighted": _drop_and_renormalize(_E2_MIXTURES["UNION_weighted"], _PALM_OPS),
}
for _name, _mix in POOLS_STATIC.items():
    assert abs(sum(_mix.values()) - 1.0) < 1e-9, f"pool {_name!r} does not sum to 1"

# Every operator ``SCHEDULE_linear``/``SCHEDULE_step`` can ever draw is the
# union of the two static pools' own keys; ``DEFAULT_uniform``'s 7 keys are a
# strict subset of ``UNION_nopalm_weighted``'s 15 (both draw from
# ``derive.OPERATORS``/``MINIMAL_STRUCTURAL_OPERATORS`` minus palm ops), so
# the union is exactly ``UNION_nopalm_weighted``'s own key set.
_UNIVERSAL_OPS: Tuple[str, ...] = tuple(POOLS_STATIC["UNION_nopalm_weighted"].keys())
assert set(POOLS_STATIC["DEFAULT_uniform"]) <= set(_UNIVERSAL_OPS)

# "structural" vs "small" bucketing used ONLY for the per-third improvement-
# rate breakdown below: structural = topology-changing (coarse + nopalm
# minimal-structural, 3 + 4 = 7 ops); small = same-topology parameter edits
# (the 7-op small-step group, incl. ``perturb_parameter``, + the 1
# ``resample_parameter`` op, 8 ops). Together they partition
# ``_UNIVERSAL_OPS`` exactly (asserted below).
_STRUCTURAL_GROUP: Tuple[str, ...] = tuple(_COARSE_GROUP) + _MINIMAL_NOPALM_GROUP
_SMALL_TOTAL_GROUP: Tuple[str, ...] = tuple(_SMALL_GROUP) + tuple(_RESAMPLE_GROUP)
assert sorted(_STRUCTURAL_GROUP + _SMALL_TOTAL_GROUP) == sorted(_UNIVERSAL_OPS)


def _linear_structural_share(g0: int, generations: int) -> float:
    """``s(g0) = 0.8 - 0.6 * g0/(generations-1)``, clamped to
    ``g0 in [0, generations-1]`` (``generations<=1`` clamps the denominator
    to 1, so ``s(0) = 0.8``)."""
    denom = max(generations - 1, 1)
    frac = min(max(g0, 0), denom) / denom
    return 0.8 - 0.6 * frac


def schedule_linear_mixture(g0: int, generations: int) -> Dict[str, float]:
    """Per-generation mixture for ``SCHEDULE_linear`` -- see module
    docstring. Always sums to 1: ``coarse_w + minimal_w == s(g0)``,
    ``resample_w == 0.10`` fixed, ``small_w == 1 - 0.10 - s(g0)``, so the
    four totals sum to exactly 1 regardless of ``g0``."""
    s = _linear_structural_share(g0, generations)
    resample_w = 0.10
    small_w = 1.0 - resample_w - s
    coarse_w = s * (10.0 / 35.0)
    minimal_w = s * (25.0 / 35.0)
    mix: Dict[str, float] = {}
    mix.update({op: coarse_w / len(_COARSE_GROUP) for op in _COARSE_GROUP})
    mix.update({op: minimal_w / len(_MINIMAL_NOPALM_GROUP) for op in _MINIMAL_NOPALM_GROUP})
    mix.update({op: resample_w for op in _RESAMPLE_GROUP})
    mix.update({op: small_w / len(_SMALL_GROUP) for op in _SMALL_GROUP})
    return mix


def _schedule_step_boundary(generations: int) -> int:
    """Number of leading generations (``g0 < boundary``) that draw from
    ``DEFAULT_uniform`` -- ``round(generations * 15/40)`` (15 of 40 by
    default), clamped into ``[1, generations-1]`` so both the
    ``DEFAULT_uniform`` and ``UNION_nopalm_weighted`` phases are non-empty
    whenever ``generations > 1``."""
    if generations <= 1:
        return 0
    boundary = int(round(generations * 15.0 / 40.0))
    return max(1, min(boundary, generations - 1))


def schedule_step_mixture(g0: int, generations: int) -> Dict[str, float]:
    boundary = _schedule_step_boundary(generations)
    if g0 < boundary:
        return dict(POOLS_STATIC["DEFAULT_uniform"])
    return dict(POOLS_STATIC["UNION_nopalm_weighted"])


_SCHEDULE_FNS: Dict[str, Any] = {"SCHEDULE_linear": schedule_linear_mixture, "SCHEDULE_step": schedule_step_mixture}
POOL_NAMES: Tuple[str, ...] = ("DEFAULT_uniform", "UNION_nopalm_weighted", "SCHEDULE_linear", "SCHEDULE_step")


def mixture_for_generation(pool_name: str, g0: int, generations: int) -> Dict[str, float]:
    if pool_name in POOLS_STATIC:
        return POOLS_STATIC[pool_name]
    return _SCHEDULE_FNS[pool_name](g0, generations)


def pool_operator_universe(pool_name: str) -> Tuple[str, ...]:
    if pool_name == "DEFAULT_uniform":
        return tuple(POOLS_STATIC["DEFAULT_uniform"].keys())
    return _UNIVERSAL_OPS


FITNESS_NAMES = tuple(FITNESS_NAMES)
COST_WEIGHTS = {k: v for k, v in COST_WEIGHTS.items() if k in ("0", "0.01")}
assert set(COST_WEIGHTS) == {"0", "0.01"}

DEFAULT_MU = 16
DEFAULT_LAMBDA = 16
DEFAULT_GENERATIONS = 40
DEFAULT_N_PROXY_CONFIGS = 16
_LATE_LOCALITY_WINDOW = 10
_LATE_LOCALITY_N_CONFIGS = 8
_LATE_LOCALITY_MAX_PAIRS_PER_RESTART = 8  # x <=24 restarts -> <=192 <= 200 pairs/condition (design cap).


def all_conditions() -> List[Tuple[str, str, str, str]]:
    return [
        (dist_name, pool_name, fitness_name, cost_name)
        for dist_name in DIST_VARIANTS
        for pool_name in POOL_NAMES
        for fitness_name in FITNESS_NAMES
        for cost_name in COST_WEIGHTS
    ]


def _gen_third(g0: int, generations: int) -> int:
    return min(2, (g0 * 3) // max(generations, 1))


# --------------------------------------------------------------------------
# One (restart, dist, pool, fitness, cost) run.
# --------------------------------------------------------------------------


def e7_evolve_seed(
    task: Tuple[int, str, str, str, str], mu: int = DEFAULT_MU, lam: int = DEFAULT_LAMBDA,
    generations: int = DEFAULT_GENERATIONS, n_proxy_configs: int = DEFAULT_N_PROXY_CONFIGS,
) -> Dict[str, Any]:
    restart_seed, dist_name, pool_name, fitness_name, cost_name = task
    dist = DIST_VARIANTS[dist_name]
    ops = list(pool_operator_universe(pool_name))
    cost_weight = COST_WEIGHTS[cost_name]
    dist_idx = list(DIST_VARIANTS).index(dist_name)
    pool_idx = POOL_NAMES.index(pool_name)
    fitness_idx = FITNESS_NAMES.index(fitness_name)
    cost_idx = list(COST_WEIGHTS).index(cost_name)
    task_idxs = (int(restart_seed), dist_idx, pool_idx, fitness_idx, cost_idx)

    def gen_seed(g: int) -> int:
        return int(_rng(task_idxs, _PHASE_GEN_SEED, g).integers(0, 2**31 - 1))

    op_stats = _new_op_stats(ops)
    # op_improve_third[op][third] = [n_strict_improve, n_noncloned_offspring].
    op_improve_third: Dict[str, List[List[int]]] = {op: [[0, 0] for _ in range(3)] for op in ops}
    n_clone_events = 0
    n_mutate_calls = 0

    population: List[Derivation] = []
    mixture0 = mixture_for_generation(pool_name, 0, generations)
    for i in range(mu):
        seed_int = _shared_start_seed_int(restart_seed, i)
        start_derivation, _, _, _ = build_start(seed_int, G_SERIAL, 2, 2)
        rng_i = _rng(task_idxs, _PHASE_INIT_MUTATE, i)
        _op, mutated, cloned = _mutate_retry_envelope(start_derivation, rng_i, dist, mixture0, op_stats)
        n_mutate_calls += 1
        if cloned:
            n_clone_events += 1
        population.append(mutated)

    evaluated0 = _evaluate_population(population, 0, gen_seed(0), fitness_name, n_proxy_configs)
    raw0 = [e["raw"] for e in evaluated0]
    proxy_range = (max(raw0) - min(raw0)) if raw0 else 0.0
    sigma = 0.1 * proxy_range

    def noisy_penalized(evaluated_row: Dict[str, Any], g: int, i: int) -> float:
        rng_n = _rng(task_idxs, _PHASE_NOISE, g, i)
        noise = float(rng_n.normal(0.0, sigma)) if sigma > 0.0 else 0.0
        return evaluated_row["raw"] + noise - cost_weight * evaluated_row["cost"]

    penal0 = [noisy_penalized(e, 0, i) for i, e in enumerate(evaluated0)]
    trajectory: List[Dict[str, Any]] = [_record_generation(0, evaluated0, penal0)]

    # (parent, child) derivations of every non-cloned offspring produced in
    # the last ``_LATE_LOCALITY_WINDOW`` generations -- sampled below into a
    # bounded-size locality diagnostic (see module docstring).
    late_pairs: List[Tuple[Derivation, Derivation]] = []

    for g in range(1, generations + 1):
        g0 = g - 1
        mixture_g = mixture_for_generation(pool_name, g0, generations)
        third = _gen_third(g0, generations)
        rng_select = _rng(task_idxs, _PHASE_GEN_SEED, g, 999999)  # distinct stream from gen_seed's own draw
        offspring: List[Derivation] = []
        offspring_ops: List[str] = []
        offspring_cloned: List[bool] = []
        parent_idx_of_offspring: List[int] = []
        for j in range(lam):
            p_idx = int(rng_select.integers(0, len(population)))
            parent = population[p_idx]
            rng_mut = _rng(task_idxs, _PHASE_OFFSPRING_MUTATE, g, j)
            op_used, child, cloned = _mutate_retry_envelope(parent, rng_mut, dist, mixture_g, op_stats)
            n_mutate_calls += 1
            if cloned:
                n_clone_events += 1
            else:
                if g0 >= generations - _LATE_LOCALITY_WINDOW:
                    late_pairs.append((parent, child))
            offspring.append(child)
            offspring_ops.append(op_used)
            offspring_cloned.append(cloned)
            parent_idx_of_offspring.append(p_idx)

        combined = population + offspring
        evaluated = _evaluate_population(combined, g, gen_seed(g), fitness_name, n_proxy_configs)
        penal = [noisy_penalized(e, g, i) for i, e in enumerate(evaluated)]
        rng_tie = _rng(task_idxs, _PHASE_TIEBREAK, g)
        tiebreak = [float(rng_tie.random()) for _ in combined]
        order = sorted(range(len(combined)), key=lambda idx: (-penal[idx], tiebreak[idx]))
        selected = order[: len(population)]

        n_parents = len(population)
        for j, (op, cloned, p_idx) in enumerate(zip(offspring_ops, offspring_cloned, parent_idx_of_offspring)):
            if cloned:
                continue
            combined_idx = n_parents + j
            op_improve_third[op][third][1] += 1
            if evaluated[combined_idx]["raw"] > evaluated[p_idx]["raw"]:
                op_improve_third[op][third][0] += 1

        population = [combined[idx] for idx in selected]
        selected_evaluated = [evaluated[idx] for idx in selected]
        selected_penal = [penal[idx] for idx in selected]
        trajectory.append(_record_generation(g, selected_evaluated, selected_penal))

    final_best_raw = trajectory[-1]["best_raw_proxy"]
    reached_target = fitness_name == "target_distance" and final_best_raw >= -1e-9
    gen_first_reach: Optional[int] = None
    if fitness_name == "target_distance":
        for row in trajectory:
            if row["best_raw_proxy"] >= -1e-9:
                gen_first_reach = row["generation"]
                break

    # Late-locality diagnostic: mean ``tip_displacement_m`` (joint_identity-
    # aligned, as in e1_locality/e5b's I15 fixes) over up to
    # ``_LATE_LOCALITY_MAX_PAIRS_PER_RESTART`` non-cloned offspring sampled
    # (without replacement, seeded) from the last-window pool collected above.
    rng_locality = _rng(task_idxs, _PHASE_TIEBREAK, 10_000_000)
    late_locality_mean_tip_displacement_m: Optional[float] = None
    late_locality_n_pairs = 0
    if late_pairs:
        k = min(_LATE_LOCALITY_MAX_PAIRS_PER_RESTART, len(late_pairs))
        chosen = rng_locality.choice(len(late_pairs), size=k, replace=False)
        tip_displacements = []
        for idx in chosen:
            parent_d, child_d = late_pairs[int(idx)]
            parent_model, child_model = derive(parent_d), derive(child_d)
            id_p, id_c = joint_identity(parent_d), joint_identity(child_d)
            row = phenotype_distance(
                parent_model, child_model, int(idx), n_configs=_LATE_LOCALITY_N_CONFIGS,
                identity_a=id_p, identity_b=id_c,
            )
            tip_displacements.append(row["tip_displacement_m"])
        late_locality_mean_tip_displacement_m = float(np.mean(tip_displacements))
        late_locality_n_pairs = len(tip_displacements)

    return {
        "restart_seed": int(restart_seed),
        "dist": dist_name, "pool": pool_name, "fitness": fitness_name, "cost": cost_name,
        "sigma": float(sigma),
        "trajectory": trajectory,
        "op_stats": op_stats,
        "op_improve_third": {
            op: [{"n_improve": v[0], "n_noncloned": v[1]} for v in thirds]
            for op, thirds in op_improve_third.items()
        },
        "n_clone_events": n_clone_events,
        "n_mutate_calls": n_mutate_calls,
        "final_best_raw_proxy": float(final_best_raw),
        "final_mean_motors": float(trajectory[-1]["mean_motors"]),
        "final_mean_joints": float(trajectory[-1]["mean_joints"]),
        "final_mean_digits": float(trajectory[-1]["mean_digits"]),
        "reached_target": bool(reached_target),
        "gen_first_reach": gen_first_reach,
        "late_locality_mean_tip_displacement_m": late_locality_mean_tip_displacement_m,
        "late_locality_n_pairs": late_locality_n_pairs,
    }


register("e7_schedule", e7_evolve_seed)


# --------------------------------------------------------------------------
# Aggregation and summary.md.
# --------------------------------------------------------------------------


def _condition_key(dist_name: str, pool_name: str, fitness_name: str, cost_name: str) -> str:
    return f"{dist_name}|{pool_name}|{fitness_name}|{cost_name}"


def _censored_gen(row: Dict[str, Any], generations: int) -> float:
    return float(row["gen_first_reach"]) if row["gen_first_reach"] is not None else float(generations)


def _aggregate(per_seed: List[Dict[str, Any]], generations: int) -> Dict[str, Any]:
    ok = [r["result"] for r in per_seed if r["ok"]]
    by_cond: Dict[str, List[Dict[str, Any]]] = {}
    for r in ok:
        key = _condition_key(r["dist"], r["pool"], r["fitness"], r["cost"])
        by_cond.setdefault(key, []).append(r)

    per_condition: Dict[str, Any] = {}
    for key, rows in by_cond.items():
        final_raw = [row["final_best_raw_proxy"] for row in rows]
        final_motors = [row["final_mean_motors"] for row in rows]
        final_joints = [row["final_mean_joints"] for row in rows]
        final_digits = [row["final_mean_digits"] for row in rows]
        reached = [bool(row["reached_target"]) for row in rows]
        gens_censored = [_censored_gen(row, generations) for row in rows]
        late_loc = [
            row["late_locality_mean_tip_displacement_m"] for row in rows
            if row["late_locality_mean_tip_displacement_m"] is not None
        ]
        late_loc_n_pairs_total = sum(row["late_locality_n_pairs"] for row in rows)
        n_clone = sum(row["n_clone_events"] for row in rows)
        n_mutate = sum(row["n_mutate_calls"] for row in rows)

        op_stats_total: Dict[str, Dict[str, int]] = {}
        op_improve_third_total: Dict[str, List[List[int]]] = {}
        for row in rows:
            for op, st in row["op_stats"].items():
                tot = op_stats_total.setdefault(op, {"attempts": 0, "envelope_rejected": 0, "inapplicable": 0, "accepted": 0})  # noqa: E501
                for k in tot:
                    tot[k] += st[k]
            for op, thirds in row["op_improve_third"].items():
                tot3 = op_improve_third_total.setdefault(op, [[0, 0], [0, 0], [0, 0]])
                for ti, d in enumerate(thirds):
                    tot3[ti][0] += d["n_improve"]
                    tot3[ti][1] += d["n_noncloned"]

        group_improve_third: Dict[str, List[List[int]]] = {"structural": [[0, 0], [0, 0], [0, 0]],
                                                             "small": [[0, 0], [0, 0], [0, 0]]}
        for op, thirds in op_improve_third_total.items():
            group = "structural" if op in _STRUCTURAL_GROUP else ("small" if op in _SMALL_TOTAL_GROUP else None)
            if group is None:
                continue
            for ti in range(3):
                group_improve_third[group][ti][0] += thirds[ti][0]
                group_improve_third[group][ti][1] += thirds[ti][1]

        per_condition[key] = {
            "n_restarts": len(rows),
            "final_best_raw_proxy": _bootstrap_ci(final_raw),
            "final_mean_motors": _bootstrap_ci(final_motors),
            "final_mean_joints": _bootstrap_ci(final_joints),
            "final_mean_digits": _bootstrap_ci(final_digits),
            "target_reach_rate": _e3_reach_mod._wilson_rate(reached),
            "gen_first_reach_censored": _bootstrap_ci(gens_censored),
            "late_locality_mean_tip_displacement_m": _bootstrap_ci(late_loc),
            "late_locality_n_pairs_total": late_loc_n_pairs_total,
            "clone_rate": (n_clone / n_mutate) if n_mutate else None,
            "rejection_rate_per_operator": {
                op: {
                    "attempts": st["attempts"],
                    "envelope_rejected_rate": (st["envelope_rejected"] / st["attempts"]) if st["attempts"] else None,
                    "inapplicable_rate": (st["inapplicable"] / st["attempts"]) if st["attempts"] else None,
                    "accepted_rate": (st["accepted"] / st["attempts"]) if st["attempts"] else None,
                }
                for op, st in op_stats_total.items()
            },
            "improvement_rate_per_group_by_third": {
                group: [(d[0] / d[1] if d[1] else None) for d in thirds]
                for group, thirds in group_improve_third.items()
            },
        }

    # Paired differences: each schedule minus each static pool, per
    # dist x fitness x cost -- paired by restart index (every restart's
    # starts are shared across every factor here, exactly as in E5b).
    diffs: Dict[str, Any] = {}
    for sched_name in ("SCHEDULE_linear", "SCHEDULE_step"):
        for base_name in ("DEFAULT_uniform", "UNION_nopalm_weighted"):
            for dist_name in DIST_VARIANTS:
                for fitness_name in FITNESS_NAMES:
                    for cost_name in COST_WEIGHTS:
                        key_s = _condition_key(dist_name, sched_name, fitness_name, cost_name)
                        key_b = _condition_key(dist_name, base_name, fitness_name, cost_name)
                        rows_s = by_cond.get(key_s, [])
                        rows_b = by_cond.get(key_b, [])
                        vals_s_raw = [row["final_best_raw_proxy"] for row in rows_s]
                        vals_b_raw = [row["final_best_raw_proxy"] for row in rows_b]
                        vals_s_gen = [_censored_gen(row, generations) for row in rows_s]
                        vals_b_gen = [_censored_gen(row, generations) for row in rows_b]
                        vals_s_reach = [1.0 if row["reached_target"] else 0.0 for row in rows_s]
                        vals_b_reach = [1.0 if row["reached_target"] else 0.0 for row in rows_b]
                        label = f"{sched_name}_minus_{base_name}__{dist_name}__{fitness_name}__{cost_name}"
                        diffs[label] = {
                            "final_best_raw_proxy": _diff_ci_paired(vals_s_raw, vals_b_raw),
                            "gen_first_reach_censored": _diff_ci_paired(vals_s_gen, vals_b_gen),
                            "target_reach_rate": _diff_ci_paired(vals_s_reach, vals_b_reach),
                        }

    return {"n_conditions": len(by_cond), "per_condition": per_condition, "diff_schedule_minus_pool": diffs}


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4g}"


def _summary_md(params: Dict[str, Any], aggregate: Dict[str, Any], wall_time_s: float,
                 note: Optional[str] = None) -> str:
    lines = ["# Experiment: e7_schedule", "", f"Wall time: {wall_time_s:.3f} s", ""]
    if note:
        lines += [f"**Note:** {note}", ""]
    lines += [
        "## Params", "", "```json", json.dumps(params, indent=2, sort_keys=True, default=str), "```", "",
        f"n_conditions: {aggregate.get('n_conditions')}", "",
        "## Per-condition (mean over restarts, 95% CI)", "",
        "| dist | pool | fitness | cost | n | final best raw proxy | 95% CI | final motors | final joints | "
        "final digits | target reach rate | gen first reach (censored, mean) | late-locality mean tip "
        "displacement (m) | clone rate |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, c in sorted(aggregate.get("per_condition", {}).items()):
        dist_name, pool_name, fitness_name, cost_name = key.split("|")
        b = c["final_best_raw_proxy"]
        rr = c["target_reach_rate"]
        g = c["gen_first_reach_censored"]
        ll = c["late_locality_mean_tip_displacement_m"]
        lines.append(
            f"| {dist_name} | {pool_name} | {fitness_name} | {cost_name} | {c['n_restarts']} | "
            f"{_fmt(b['mean'])} | [{_fmt(b['ci_lo'])}, {_fmt(b['ci_hi'])}] | "
            f"{_fmt(c['final_mean_motors']['mean'])} | {_fmt(c['final_mean_joints']['mean'])} | "
            f"{_fmt(c['final_mean_digits']['mean'])} | {_fmt(rr.get('rate'))} | {_fmt(g['mean'])} | "
            f"{_fmt(ll['mean'])} | {_fmt(c['clone_rate'])} |"
        )
    lines.append("")

    lines += ["## Paired difference: schedule minus static pool, per dist x fitness x cost", "",
              "| schedule minus pool | dist | fitness | cost | metric | mean DIFFERENCE | 95% CI | n |",
              "|---|---|---|---|---|---|---|---|"]
    for key, metrics in sorted(aggregate.get("diff_schedule_minus_pool", {}).items()):
        pair, dist_name, fitness_name, cost_name = key.split("__")
        for metric_name, b in metrics.items():
            lines.append(
                f"| {pair} | {dist_name} | {fitness_name} | {cost_name} | {metric_name} | {_fmt(b['mean'])} | "
                f"[{_fmt(b['ci_lo'])}, {_fmt(b['ci_hi'])}] | {b['n']} |"
            )
    lines.append("")

    lines += ["## Improvement rate per operator group (structural vs small), by generation third, per condition", "",
              "| condition | group | third 0 (early) | third 1 (mid) | third 2 (late) |",
              "|---|---|---|---|---|"]
    for key, c in sorted(aggregate.get("per_condition", {}).items()):
        for group, rates in sorted(c["improvement_rate_per_group_by_third"].items()):
            lines.append(f"| {key} | {group} | {_fmt(rates[0])} | {_fmt(rates[1])} | {_fmt(rates[2])} |")
    lines.append("")

    lines += ["## Reading", "",
              "(Numbers only. Caveats: `gen_first_reach_censored` assigns `generations` (40, or the run's "
              "own `generations` param) to any restart that never reached the target -- a simple censoring "
              "imputation, not a Kaplan-Meier estimator, so its mean is a LOWER BOUND on the true mean "
              "generation-to-reach whenever the reach rate is below 1. `late_locality_mean_tip_displacement_m` "
              "is a mean over at most 8 sampled non-cloned offspring per restart from the last 10 of 40 "
              "generations (<=192 pairs per condition across 24 restarts), joint_identity-aligned exactly as "
              "in e1_locality/e5b's I15 fixes -- a diagnostic sample, not an exhaustive count. "
              "`improvement_rate_per_group_by_third` buckets `regrow_subtree`/`add_digit`/`remove_digit`/"
              "`add_minimal_digit`/`remove_digit_minimal`/`insert_phalanx`/`delete_phalanx` as \"structural\" "
              "and the 6 `step_*` operators + `perturb_parameter` + `resample_parameter` as \"small\"; a third "
              "with 0 non-cloned offspring for a group reports `n/a`.)", ""]
    for key, c in sorted(aggregate.get("per_condition", {}).items()):
        b = c["final_best_raw_proxy"]
        rr = c["target_reach_rate"]
        g = c["gen_first_reach_censored"]
        lines.append(
            f"- {key}: final best raw proxy {_fmt(b['mean'])} [{_fmt(b['ci_lo'])}, {_fmt(b['ci_hi'])}] "
            f"(n={b['n']}); target reach rate {_fmt(rr.get('rate'))}; gen first reach (censored, mean) "
            f"{_fmt(g['mean'])}; clone rate {_fmt(c['clone_rate'])}."
        )
    lines.append("")
    return "\n".join(lines)


def run(out_dir: Optional[str] = None, restarts: Sequence[int] = range(24),
        conditions: Optional[Sequence[Tuple[str, str, str, str]]] = None,
        mu: int = DEFAULT_MU, lam: int = DEFAULT_LAMBDA, generations: int = DEFAULT_GENERATIONS,
        n_proxy_configs: int = DEFAULT_N_PROXY_CONFIGS, processes: int = 24,
        allow_dirty: bool = False, note: Optional[str] = None, time_budget_s: float = 1800.0) -> Dict[str, Any]:
    if out_dir is None:
        out_dir = "project-notes/grammar/experiments/E7_schedule"
    if conditions is None:
        conditions = all_conditions()
    conditions = list(conditions)

    # Timing probe (design requirement): one REAL task, then project the
    # full run's wall time from it; if that projection exceeds
    # ``time_budget_s`` (default 1800s = 30 min), drop the
    # antipodal_pinch+cost=0.01 conditions (kept for target_distance) and
    # record the decision in ``note``/params.
    probe_task = (0, *conditions[0])
    t_probe0 = time.time()
    e7_evolve_seed(probe_task, mu=mu, lam=lam, generations=generations, n_proxy_configs=n_proxy_configs)
    probe_time_s = time.time() - t_probe0

    n_restarts_full = len(list(restarts))
    n_tasks_full = len(conditions) * n_restarts_full
    n_proc_full = max(1, min(processes, n_tasks_full))
    projected_full_s = probe_time_s * n_tasks_full / n_proc_full

    probe_note = (
        f"timing probe: probe_task={probe_task} took {probe_time_s:.3f}s; projected full run "
        f"({n_tasks_full} tasks over {n_proc_full} processes) = {projected_full_s:.1f}s "
        f"(budget {time_budget_s:.0f}s)."
    )
    dropped = False
    if projected_full_s > time_budget_s:
        before = len(conditions)
        conditions = [c for c in conditions if not (c[2] == "antipodal_pinch" and c[3] == "0.01")]
        dropped = True
        probe_note += (
            f" DROPPED antipodal_pinch+cost=0.01 conditions ({before} -> {len(conditions)}) to stay in budget."
        )
    else:
        probe_note += " No conditions dropped."
    full_note = probe_note if note is None else f"{note} {probe_note}"

    tasks = [
        (int(r), dist_name, pool_name, fitness_name, cost_name)
        for r in restarts
        for (dist_name, pool_name, fitness_name, cost_name) in conditions
    ]
    params = {
        "mu": mu, "lam": lam, "generations": generations, "n_proxy_configs": n_proxy_configs,
        "n_restarts": n_restarts_full, "n_conditions": len(conditions),
        "envelope": ENVELOPE_KWARGS, "timing_probe_s": probe_time_s, "projected_full_run_s": projected_full_s,
        "dropped_pinch_cost_0_01": dropped,
    }
    fn_params = {"mu": mu, "lam": lam, "generations": generations, "n_proxy_configs": n_proxy_configs}
    result = run_experiment(
        "e7_schedule", e7_evolve_seed, params=fn_params, seeds=tasks, out_dir=out_dir, processes=processes,
        allow_dirty=allow_dirty,
    )
    aggregate = _aggregate(result["per_seed"], generations)
    result["aggregate"] = aggregate
    result["params"] = params
    out_path = Path(out_dir)
    (out_path / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    (out_path / "summary.md").write_text(_summary_md(params, aggregate, result["wall_time_s"], note=full_note))
    return result


if __name__ == "__main__":
    run()
