"""E5b: (mu+lambda) evolution matched to the SIMULATOR's hard envelope --
fix iteration I15's rerun of E5, per ``opus-review-final.md``'s "Next
experiments" #1.

Why a rerun (see ``project-notes/grammar/opus-review-final.md``'s "E5
soundness" section for the full list of defects in the original E5):
E5's cost-aware selection reported the PENALISED fitness, not the raw
proxy, so its "halves motors at a small proxy cost" claim was unsupported;
"none" cost was not actually cost-free (ties broke by lower cost, a
parsimony pressure that vanishes under noise); starts differed across
distributions (unpaired INS-vs-plain); its two structural proxies
(opposition, reach_log) each have a low-dimensional optimum (2 digits, 1
chain respectively) that tells you little about whether growth is being
slowed rather than simply not rewarded; and it never modeled the actual
target this grammar feeds -- a simulator that can only actuate a BOUNDED
hand (few digits, few joints per digit, no spare palm DOF, no branching).

What changes here:

- ``hard 5x6 envelope`` (``envelope.fits_envelope``, defaults ``max_digits=5,
  max_joints_per_digit=6, allow_palm_joints=False, allow_branches=False``):
  every ``vary`` proposal is derived and checked; a proposal that fails the
  envelope is REJECTED (not repaired), retried with a freshly-drawn operator
  up to 8 times total, and if every attempt fails (envelope-reject OR
  ``VariationImpossible``) the offspring falls back to an unmutated CLONE of
  its parent. Rejection/inapplicable/clone counts are recorded per operator.
- ``UNION_weighted_nopalm``: the E2 ``UNION_weighted`` mixture (which already
  has ``perturb_parameter`` in its small-step share -- I15 fix 3) with every
  palm operator (``add_palm_body``, ``remove_palm_body``,
  ``toggle_palm_joint``) removed and the remaining weights renormalized --
  since ``allow_palm_joints=False`` means every palm-joint proposal from
  those operators is rejected outright, a pool that still spends 25%/3 of
  its weight on them is spending proposals the envelope refuses by
  construction (opus-review-final.md's "envelope contradiction").
- starts are SHARED across every (dist, pool) at a given (restart, i): built
  from a FIXED distribution (``G_SERIAL``, exactly ``starts.build_start``'s
  own default) with a seed depending ONLY on ``(restart, i)`` -- never on
  dist/pool/fitness/cost index -- so every one of the 54 conditions at a
  given restart starts its generation-0 mutation from literally the same 16
  reduced derivations (I15 fix, generalizing E2's "same starts across
  mixtures" to every factor here; the prior E5's ``_start_seed_int`` folded
  ``dist_idx`` into the seed, which is the "starts differ across
  distributions" defect above).
- fitness = ``base_proxy + noise`` where ``noise ~ N(0, sigma)`` is drawn
  PER EVALUATION from a seeded rng (``sigma = 0.1 * (max - min) of the raw
  base_proxy over the INITIAL (generation-0) population``, fixed for the
  rest of that restart); the PENALISED value used for selection is
  ``noisy_value - cost_weight * structural_cost`` (``cost_weight`` in
  ``{0, 0.01, 0.02}``); ties are broken by an independent seeded random draw
  (never by cost -- I15 fix, removing E5's lexicographic-by-cost tie-break,
  which was a hidden parsimony pressure under "no cost"). RAW proxy, cost,
  motors, joints, digits are logged separately every generation, in addition
  to the noisy/penalised selection value.
- two fitnesses: ``antipodal_pinch`` (as in E5) and ``target_distance``
  (NEW) = ``-e3_reach``'s own ``anthropomorphic_staggered`` structural
  distance (0 at 5 digits, each >= 3 phalanges, >= 3 distinct mount
  fractions) -- an objective whose optimum needs real growth (many digits,
  many phalanges), unlike E5's opposition/reach_log, which saturate at a
  small, low-dimensional optimum and so cannot distinguish "insertion prior
  slows growth" from "growth was never rewarded" (opus-review-final.md's
  "Next experiments" #2).
- three ``dist`` variants: ``G_FULL_INS``, ``G_NOBRANCH_INS``,
  ``G_NOBRANCH`` (``G_FULL`` dropped -- E2 already showed the insertion
  prior is what matters for drift control, and this halves the factor
  count).

Aggregates (over restarts, per condition, plus PAIRED difference CIs: INS
vs plain pool-for-pool, UNION vs DEFAULT dist-for-dist -- both paired
because every restart's starts are shared across every factor here):
final best RAW proxy, final mean motors/joints/digits, fraction of restarts
reaching the target (raw proxy >= 0, i.e. E3 distance == 0, meaningful only
for ``target_distance``) and the generation of first reach, rejection rates
per operator, per-operator STRICT improvement rate (a non-cloned
offspring's raw proxy, evaluated under the SAME generation seed as its
parent, strictly exceeds the parent's own raw proxy that generation), and
the overall clone rate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..canonical import phenotype_hash
from ..coords import independent_joints, sample_configurations
from ..derive import Derivation, VariationImpossible, derive, vary
from ..envelope import fits_envelope
from ..kinematics import KinematicModel
from ..phenodist import _digit_count
from ..proxy import antipodal_pinch, structural_cost
from ..variants import G_FULL_INS, G_NOBRANCH, G_NOBRANCH_INS, G_SERIAL
from .e2_drift import MIXTURES, _draw_operator
from . import e3_reach as _e3_reach_mod  # noqa: F401 -- deferred attribute access only (see below);
# NEVER `from .e3_reach import TARGETS` at module level: runner.py imports e3_reach BEFORE this module, so
# when e3_reach itself is the entrypoint (standalone `import e3_reach`), its own `from .runner import ...`
# re-enters runner.py, which reaches this module's import while e3_reach is still mid-init (its own
# `TARGETS`/`_wilson_rate` module attributes not yet bound) -- a `from X import name` at that point raises
# ImportError. Accessing `_e3_reach_mod.TARGETS`/`_e3_reach_mod._wilson_rate` lazily, inside functions below
# (only ever called after every module has finished importing), avoids the cycle entirely.
from .runner import register, run_experiment
from .starts import build_start

# --------------------------------------------------------------------------
# Factors (3 x 3 x 2 x 3 = 54 conditions).
# --------------------------------------------------------------------------

DIST_VARIANTS: Dict[str, Any] = {
    "G_FULL_INS": G_FULL_INS, "G_NOBRANCH_INS": G_NOBRANCH_INS, "G_NOBRANCH": G_NOBRANCH,
}

_PALM_OPS = ("add_palm_body", "remove_palm_body", "toggle_palm_joint")


def _drop_and_renormalize(mixture: Dict[str, float], drop: Sequence[str]) -> Dict[str, float]:
    kept = {op: w for op, w in mixture.items() if op not in drop}
    total = sum(kept.values())
    return {op: w / total for op, w in kept.items()}


POOLS: Dict[str, Dict[str, float]] = {
    "DEFAULT_uniform": dict(MIXTURES["DEFAULT_uniform"]),
    "UNION_weighted": dict(MIXTURES["UNION_weighted"]),
    "UNION_weighted_nopalm": _drop_and_renormalize(MIXTURES["UNION_weighted"], _PALM_OPS),
}
for _name, _mix in POOLS.items():
    assert abs(sum(_mix.values()) - 1.0) < 1e-9, f"pool {_name!r} does not sum to 1"

FITNESS_NAMES: Tuple[str, ...] = ("antipodal_pinch", "target_distance")
COST_WEIGHTS: Dict[str, float] = {"0": 0.0, "0.01": 0.01, "0.02": 0.02}

ENVELOPE_KWARGS: Dict[str, Any] = {
    "max_digits": 5, "max_joints_per_digit": 6, "allow_palm_joints": False, "allow_branches": False,
}

DEFAULT_MU = 16
DEFAULT_LAMBDA = 16
DEFAULT_GENERATIONS = 40
DEFAULT_N_PROXY_CONFIGS = 16
_MUTATE_RETRY_ATTEMPTS = 8


def all_conditions() -> List[Tuple[str, str, str, str]]:
    return [
        (dist_name, pool_name, fitness_name, cost_name)
        for dist_name in DIST_VARIANTS
        for pool_name in POOLS
        for fitness_name in FITNESS_NAMES
        for cost_name in COST_WEIGHTS
    ]


# --------------------------------------------------------------------------
# Deterministic RNG streams (same convention as e5_evolve.py).
# --------------------------------------------------------------------------

_PHASE_INIT_MUTATE = 0
_PHASE_OFFSPRING_MUTATE = 1
_PHASE_GEN_SEED = 2
_PHASE_NOISE = 3
_PHASE_TIEBREAK = 4


def _rng(task_idxs: Sequence[int], phase: int, *rest: int) -> np.random.Generator:
    return np.random.default_rng([*task_idxs, phase, *rest])


def _shared_start_seed_int(restart_seed: int, i: int) -> int:
    """Depends ONLY on ``(restart_seed, i)`` -- never on dist/pool/fitness/
    cost -- see module docstring's "starts are SHARED" note."""
    return restart_seed * 1_000_000 + i


# --------------------------------------------------------------------------
# Envelope-aware mutation with rejection/clone bookkeeping.
# --------------------------------------------------------------------------


def _new_op_stats(ops: Sequence[str]) -> Dict[str, Dict[str, int]]:
    return {op: {"attempts": 0, "envelope_rejected": 0, "inapplicable": 0, "accepted": 0} for op in ops}


def _mutate_retry_envelope(
    derivation: Derivation, rng: np.random.Generator, dist, mixture: Dict[str, float],
    op_stats: Dict[str, Dict[str, int]], max_attempts: int = _MUTATE_RETRY_ATTEMPTS,
) -> Tuple[str, Derivation, bool]:
    """One envelope-checked mutation of ``derivation``. Returns
    ``(operator_label, result, cloned)``. Each of up to ``max_attempts``
    tries draws a (possibly fresh) operator from ``mixture``, applies
    ``vary``, and REJECTS the candidate (counted in ``op_stats``, retried)
    if it fails ``envelope.fits_envelope`` or ``vary`` itself raises
    ``VariationImpossible``. If every attempt fails, ``result`` is an
    unmutated clone of ``derivation`` (``cloned=True``), labelled with the
    last operator attempted."""
    op = _draw_operator(rng, mixture)
    for _ in range(max_attempts):
        st = op_stats[op]
        st["attempts"] += 1
        try:
            candidate = vary(derivation, rng, dist, operator=op)
        except VariationImpossible:
            st["inapplicable"] += 1
            op = _draw_operator(rng, mixture)
            continue
        model = derive(candidate)
        ok, _reasons = fits_envelope(model, **ENVELOPE_KWARGS)
        if not ok:
            st["envelope_rejected"] += 1
            op = _draw_operator(rng, mixture)
            continue
        st["accepted"] += 1
        return op, candidate, False
    return op, derivation, True


# --------------------------------------------------------------------------
# Fitness.
# --------------------------------------------------------------------------


def _raw_fitness(model: KinematicModel, derivation: Derivation, fitness_name: str,
                  gen_seed: int, n_proxy_configs: int) -> float:
    if fitness_name == "antipodal_pinch":
        configs = sample_configurations(model, n_proxy_configs, gen_seed)
        return float(antipodal_pinch(model, configs))
    if fitness_name == "target_distance":
        dist = _e3_reach_mod.TARGETS["anthropomorphic_staggered"](model, derivation)
        return -float(dist)
    raise ValueError(f"unknown fitness {fitness_name!r}")


def _evaluate_population(
    derivations: Sequence[Derivation], generation: int, generation_seed: int,
    fitness_name: str, n_proxy_configs: int,
) -> List[Dict[str, Any]]:
    """Derive+evaluate every derivation under ``generation_seed`` (RAW,
    un-noised, un-penalised), caching by ``(generation-local) phenotype_hash``
    -- a fresh cache per call/generation, same rationale as
    ``e5_evolve._evaluate_population``."""
    cache: Dict[str, Dict[str, Any]] = {}
    out = []
    for d in derivations:
        model = derive(d)
        h = phenotype_hash(model)
        cached = cache.get(h)
        if cached is None:
            raw = _raw_fitness(model, d, fitness_name, generation_seed, n_proxy_configs)
            cached = {
                "raw": raw, "cost": float(structural_cost(model)),
                "motors": len(independent_joints(model)), "joints": len(model.joints),
                "digits": _digit_count(model), "hash": h,
            }
            cache[h] = cached
        out.append({**cached, "derivation": d, "model": model})
    return out


def _record_generation(generation: int, evaluated: Sequence[Dict[str, Any]],
                        penalized: Sequence[float]) -> Dict[str, Any]:
    n = len(evaluated)
    raws = [e["raw"] for e in evaluated]
    hashes = [e["hash"] for e in evaluated]
    return {
        "generation": int(generation),
        "best_raw_proxy": float(max(raws)),
        "mean_raw_proxy": float(np.mean(raws)),
        "best_penalized": float(max(penalized)),
        "mean_penalized": float(np.mean(penalized)),
        "mean_cost": float(np.mean([e["cost"] for e in evaluated])),
        "mean_motors": float(np.mean([e["motors"] for e in evaluated])),
        "mean_joints": float(np.mean([e["joints"] for e in evaluated])),
        "mean_digits": float(np.mean([e["digits"] for e in evaluated])),
        "distinct_hash_fraction": float(len(set(hashes)) / n) if n else 0.0,
    }


# --------------------------------------------------------------------------
# One (restart, dist, pool, fitness, cost) run.
# --------------------------------------------------------------------------


def e5b_evolve_seed(
    task: Tuple[int, str, str, str, str], mu: int = DEFAULT_MU, lam: int = DEFAULT_LAMBDA,
    generations: int = DEFAULT_GENERATIONS, n_proxy_configs: int = DEFAULT_N_PROXY_CONFIGS,
) -> Dict[str, Any]:
    restart_seed, dist_name, pool_name, fitness_name, cost_name = task
    dist = DIST_VARIANTS[dist_name]
    mixture = POOLS[pool_name]
    ops = list(mixture.keys())
    cost_weight = COST_WEIGHTS[cost_name]
    dist_idx = list(DIST_VARIANTS).index(dist_name)
    pool_idx = list(POOLS).index(pool_name)
    fitness_idx = FITNESS_NAMES.index(fitness_name)
    cost_idx = list(COST_WEIGHTS).index(cost_name)
    task_idxs = (int(restart_seed), dist_idx, pool_idx, fitness_idx, cost_idx)

    def gen_seed(g: int) -> int:
        return int(_rng(task_idxs, _PHASE_GEN_SEED, g).integers(0, 2**31 - 1))

    op_stats = _new_op_stats(ops)
    op_improve = {op: [0, 0] for op in ops}  # [n_strict_improve, n_noncloned_offspring]
    n_clone_events = 0
    n_mutate_calls = 0

    # Shared start (I15 fix): depends only on (restart_seed, i), built from
    # G_SERIAL -- see module docstring and ``_shared_start_seed_int``.
    population: List[Derivation] = []
    for i in range(mu):
        seed_int = _shared_start_seed_int(restart_seed, i)
        start_derivation, _, _, _ = build_start(seed_int, G_SERIAL, 2, 2)
        rng_i = _rng(task_idxs, _PHASE_INIT_MUTATE, i)
        _op, mutated, cloned = _mutate_retry_envelope(start_derivation, rng_i, dist, mixture, op_stats)
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

    for g in range(1, generations + 1):
        rng_select = _rng(task_idxs, _PHASE_GEN_SEED, g, 999999)  # distinct stream from gen_seed's own draw
        offspring: List[Derivation] = []
        offspring_ops: List[str] = []
        offspring_cloned: List[bool] = []
        parent_idx_of_offspring: List[int] = []
        for j in range(lam):
            p_idx = int(rng_select.integers(0, len(population)))
            parent = population[p_idx]
            rng_mut = _rng(task_idxs, _PHASE_OFFSPRING_MUTATE, g, j)
            op_used, child, cloned = _mutate_retry_envelope(parent, rng_mut, dist, mixture, op_stats)
            n_mutate_calls += 1
            if cloned:
                n_clone_events += 1
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
        selected_set = set(selected)

        n_parents = len(population)
        for j, (op, cloned, p_idx) in enumerate(zip(offspring_ops, offspring_cloned, parent_idx_of_offspring)):
            if cloned:
                continue
            combined_idx = n_parents + j
            op_improve[op][1] += 1
            if evaluated[combined_idx]["raw"] > evaluated[p_idx]["raw"]:
                op_improve[op][0] += 1

        population = [combined[idx] for idx in selected]
        selected_evaluated = [evaluated[idx] for idx in selected]
        selected_penal = [penal[idx] for idx in selected]
        trajectory.append(_record_generation(g, selected_evaluated, selected_penal))
        del selected_set  # only used for clarity above

    final_best_raw = trajectory[-1]["best_raw_proxy"]
    reached_target = fitness_name == "target_distance" and final_best_raw >= -1e-9
    gen_first_reach: Optional[int] = None
    if fitness_name == "target_distance":
        for row in trajectory:
            if row["best_raw_proxy"] >= -1e-9:
                gen_first_reach = row["generation"]
                break

    return {
        "restart_seed": int(restart_seed),
        "dist": dist_name, "pool": pool_name, "fitness": fitness_name, "cost": cost_name,
        "sigma": float(sigma),
        "trajectory": trajectory,
        "op_stats": op_stats,
        "op_improve": {op: {"n_improve": v[0], "n_noncloned": v[1]} for op, v in op_improve.items()},
        "n_clone_events": n_clone_events,
        "n_mutate_calls": n_mutate_calls,
        "final_best_raw_proxy": float(final_best_raw),
        "final_mean_motors": float(trajectory[-1]["mean_motors"]),
        "final_mean_joints": float(trajectory[-1]["mean_joints"]),
        "final_mean_digits": float(trajectory[-1]["mean_digits"]),
        "reached_target": bool(reached_target),
        "gen_first_reach": gen_first_reach,
    }


register("e5b_evolve_sim", e5b_evolve_seed)


# --------------------------------------------------------------------------
# Aggregation and summary.md.
# --------------------------------------------------------------------------


def _bootstrap_ci(vals: Sequence[float], n_resamples: int = 2000, seed: int = 0) -> Dict[str, Optional[float]]:
    arr = np.asarray(vals, dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0}
    if n == 1:
        return {"mean": float(arr[0]), "ci_lo": float(arr[0]), "ci_hi": float(arr[0]), "n": 1}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    return {"mean": float(arr.mean()), "ci_lo": float(np.percentile(means, 2.5)),
            "ci_hi": float(np.percentile(means, 97.5)), "n": n}


def _diff_ci_paired(vals_a: Sequence[float], vals_b: Sequence[float], seed: int = 0) -> Dict[str, Optional[float]]:
    n = min(len(vals_a), len(vals_b))
    if n == 0:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0}
    diff = np.asarray(vals_a[:n], dtype=float) - np.asarray(vals_b[:n], dtype=float)
    return _bootstrap_ci(list(diff), seed=seed)


def _condition_key(dist_name: str, pool_name: str, fitness_name: str, cost_name: str) -> str:
    return f"{dist_name}|{pool_name}|{fitness_name}|{cost_name}"


def _aggregate(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        gens_reached = [row["gen_first_reach"] for row in rows if row["gen_first_reach"] is not None]
        n_clone = sum(row["n_clone_events"] for row in rows)
        n_mutate = sum(row["n_mutate_calls"] for row in rows)

        op_stats_total: Dict[str, Dict[str, int]] = {}
        op_improve_total: Dict[str, List[int]] = {}
        for row in rows:
            for op, st in row["op_stats"].items():
                tot = op_stats_total.setdefault(op, {"attempts": 0, "envelope_rejected": 0, "inapplicable": 0, "accepted": 0})  # noqa: E501
                for k in tot:
                    tot[k] += st[k]
            for op, imp in row["op_improve"].items():
                tot2 = op_improve_total.setdefault(op, [0, 0])
                tot2[0] += imp["n_improve"]
                tot2[1] += imp["n_noncloned"]

        per_condition[key] = {
            "n_restarts": len(rows),
            "final_best_raw_proxy": _bootstrap_ci(final_raw),
            "final_mean_motors": _bootstrap_ci(final_motors),
            "final_mean_joints": _bootstrap_ci(final_joints),
            "final_mean_digits": _bootstrap_ci(final_digits),
            "target_reach_rate": _e3_reach_mod._wilson_rate(reached),
            "gen_first_reach_median": float(np.median(gens_reached)) if gens_reached else None,
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
            "improvement_rate_per_operator": {
                op: (v[0] / v[1] if v[1] else None) for op, v in op_improve_total.items()
            },
        }

    # Paired differences: INS minus plain pool-for-pool (same dist family,
    # since restarts share starts across EVERY factor here -- see module
    # docstring), and UNION_weighted minus DEFAULT_uniform dist-for-dist.
    diff_ins: Dict[str, Any] = {}
    ins_pairs = [("G_FULL_INS", "G_NOBRANCH_INS")]  # both _INS; also compare G_NOBRANCH_INS vs G_NOBRANCH below
    plain_pairs = [("G_NOBRANCH_INS", "G_NOBRANCH")]
    for pool_name in POOLS:
        for fitness_name in FITNESS_NAMES:
            for cost_name in COST_WEIGHTS:
                for ins_name, plain_name in plain_pairs:
                    key_ins = _condition_key(ins_name, pool_name, fitness_name, cost_name)
                    key_plain = _condition_key(plain_name, pool_name, fitness_name, cost_name)
                    rows_ins = by_cond.get(key_ins, [])
                    rows_plain = by_cond.get(key_plain, [])
                    vals_ins = [row["final_best_raw_proxy"] for row in rows_ins]
                    vals_plain = [row["final_best_raw_proxy"] for row in rows_plain]
                    diff_ins[f"{ins_name}_minus_{plain_name}__{pool_name}__{fitness_name}__{cost_name}"] = (
                        _diff_ci_paired(vals_ins, vals_plain)
                    )
    diff_pool: Dict[str, Any] = {}
    if "UNION_weighted" in POOLS and "DEFAULT_uniform" in POOLS:
        for dist_name in DIST_VARIANTS:
            for fitness_name in FITNESS_NAMES:
                for cost_name in COST_WEIGHTS:
                    key_u = _condition_key(dist_name, "UNION_weighted", fitness_name, cost_name)
                    key_d = _condition_key(dist_name, "DEFAULT_uniform", fitness_name, cost_name)
                    vals_u = [row["final_best_raw_proxy"] for row in by_cond.get(key_u, [])]
                    vals_d = [row["final_best_raw_proxy"] for row in by_cond.get(key_d, [])]
                    diff_pool[f"UNION_weighted_minus_DEFAULT_uniform__{dist_name}__{fitness_name}__{cost_name}"] = (
                        _diff_ci_paired(vals_u, vals_d)
                    )

    return {
        "n_conditions": len(by_cond), "per_condition": per_condition,
        "diff_ins_minus_plain": diff_ins, "diff_UNION_weighted_minus_DEFAULT_uniform": diff_pool,
    }


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4g}"


def _summary_md(params: Dict[str, Any], aggregate: Dict[str, Any], wall_time_s: float,
                 note: Optional[str] = None) -> str:
    lines = [
        "# Experiment: e5b_evolve_sim", "",
        f"Wall time: {wall_time_s:.3f} s", "",
    ]
    if note:
        lines += [f"**Note:** {note}", ""]
    lines += [
        "## Params", "", "```json", json.dumps(params, indent=2, sort_keys=True, default=str), "```", "",
        f"n_conditions: {aggregate.get('n_conditions')}", "",
        "## Per-condition (mean over restarts, 95% CI)", "",
        "| dist | pool | fitness | cost | n | final best raw proxy | 95% CI | "
        "final motors | final joints | final digits | target reach rate | gen first reach (median) | clone rate |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, c in sorted(aggregate.get("per_condition", {}).items()):
        dist_name, pool_name, fitness_name, cost_name = key.split("|")
        b = c["final_best_raw_proxy"]
        rr = c["target_reach_rate"]
        lines.append(
            f"| {dist_name} | {pool_name} | {fitness_name} | {cost_name} | {c['n_restarts']} | "
            f"{_fmt(b['mean'])} | [{_fmt(b['ci_lo'])}, {_fmt(b['ci_hi'])}] | "
            f"{_fmt(c['final_mean_motors']['mean'])} | {_fmt(c['final_mean_joints']['mean'])} | "
            f"{_fmt(c['final_mean_digits']['mean'])} | {_fmt(rr.get('rate'))} | "
            f"{_fmt(c['gen_first_reach_median'])} | {_fmt(c['clone_rate'])} |"
        )
    lines.append("")

    lines += ["## Difference CI: an _INS dist minus its plain counterpart, per pool x fitness x cost", "",
              "| pair | pool | fitness | cost | mean DIFFERENCE (final best raw proxy) | 95% CI | n |",
              "|---|---|---|---|---|---|---|"]
    for key, b in sorted(aggregate.get("diff_ins_minus_plain", {}).items()):
        pair, pool_name, fitness_name, cost_name = key.split("__")
        lines.append(
            f"| {pair} | {pool_name} | {fitness_name} | {cost_name} | {_fmt(b['mean'])} | "
            f"[{_fmt(b['ci_lo'])}, {_fmt(b['ci_hi'])}] | {b['n']} |"
        )
    lines.append("")

    lines += ["## Difference CI: UNION_weighted minus DEFAULT_uniform, per dist x fitness x cost", "",
              "| dist | fitness | cost | mean DIFFERENCE (final best raw proxy) | 95% CI | n |",
              "|---|---|---|---|---|---|"]
    for key, b in sorted(aggregate.get("diff_UNION_weighted_minus_DEFAULT_uniform", {}).items()):
        _prefix, dist_name, fitness_name, cost_name = key.split("__")
        lines.append(
            f"| {dist_name} | {fitness_name} | {cost_name} | {_fmt(b['mean'])} | "
            f"[{_fmt(b['ci_lo'])}, {_fmt(b['ci_hi'])}] | {b['n']} |"
        )
    lines.append("")

    lines += ["## Rejection / improvement rate per operator, per condition", ""]
    for key, c in sorted(aggregate.get("per_condition", {}).items()):
        lines += [f"### {key}", "",
                  "| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |",  # noqa: E501
                  "|---|---|---|---|---|---|"]
        rej = c["rejection_rate_per_operator"]
        imp = c["improvement_rate_per_operator"]
        for op in sorted(rej):
            r = rej[op]
            lines.append(
                f"| {op} | {r['attempts']} | {_fmt(r['envelope_rejected_rate'])} | "
                f"{_fmt(r['inapplicable_rate'])} | {_fmt(r['accepted_rate'])} | {_fmt(imp.get(op))} |"
            )
        lines.append("")

    lines += ["## Reading", "",
              "(Numbers only; caveats: `target_distance` fitness's structural-distance function is cached "
              "by phenotype_hash within a generation, so two structurally-equivalent but differently-derived "
              "individuals share one evaluation -- an approximation, not exact per-derivation scoring. "
              "`antipodal_pinch` fitness uses the same n_proxy_configs-sample geometric proxy as E5, still a "
              "diagnostic, never a real task objective.)", ""]
    for key, c in sorted(aggregate.get("per_condition", {}).items()):
        b = c["final_best_raw_proxy"]
        rr = c["target_reach_rate"]
        lines.append(
            f"- {key}: final best raw proxy {_fmt(b['mean'])} [{_fmt(b['ci_lo'])}, {_fmt(b['ci_hi'])}] "
            f"(n={b['n']}); target reach rate {_fmt(rr.get('rate'))}; clone rate {_fmt(c['clone_rate'])}."
        )
    lines.append("")
    return "\n".join(lines)


def run(out_dir: Optional[str] = None, restarts: Sequence[int] = range(24),
        conditions: Optional[Sequence[Tuple[str, str, str, str]]] = None,
        mu: int = DEFAULT_MU, lam: int = DEFAULT_LAMBDA, generations: int = DEFAULT_GENERATIONS,
        n_proxy_configs: int = DEFAULT_N_PROXY_CONFIGS, processes: int = 24,
        allow_dirty: bool = False, note: Optional[str] = None) -> Dict[str, Any]:
    if out_dir is None:
        out_dir = "project-notes/grammar/experiments/E5b_evolve_sim"
    if conditions is None:
        conditions = all_conditions()
    tasks = [
        (int(r), dist_name, pool_name, fitness_name, cost_name)
        for r in restarts
        for (dist_name, pool_name, fitness_name, cost_name) in conditions
    ]
    params = {
        "mu": mu, "lam": lam, "generations": generations, "n_proxy_configs": n_proxy_configs,
        "n_restarts": len(list(restarts)), "n_conditions": len(list(conditions)),
        "envelope": ENVELOPE_KWARGS,
    }
    fn_params = {"mu": mu, "lam": lam, "generations": generations, "n_proxy_configs": n_proxy_configs}
    t0 = time.time()
    result = run_experiment(
        "e5b_evolve_sim", e5b_evolve_seed, params=fn_params, seeds=tasks, out_dir=out_dir, processes=processes,
        allow_dirty=allow_dirty,
    )
    aggregate = _aggregate(result["per_seed"])
    result["aggregate"] = aggregate
    result["params"] = params
    out_path = Path(out_dir)
    (out_path / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    (out_path / "summary.md").write_text(_summary_md(params, aggregate, result["wall_time_s"], note=note))
    return result


if __name__ == "__main__":
    run()
