"""E5: does (mu+lambda) selection under a cheap geometric proxy exploit the
grammar, and which constructs/operators does it actually use?

Purpose (see project-notes/grammar/experiments/E5_evolve for the write-up):
E2/E3 already showed neutral drift is controlled by the insertion
distribution and that the union operator pool is needed to reach palm
articulation. E5 asks the complementary question with actual selection
pressure: given a cheap geometric DIAGNOSTIC (never a real task objective --
see ``proxy.py``'s own module docstring) as fitness, does truncation
selection push a population toward more structure/motors regardless of the
diagnostic's content, and does the choice of operator pool / grammar variant
/ cost penalty change what gets used.

Design
------
(mu+lambda) = (32+32), 40 generations, truncation selection on fitness (ties
broken by LOWER ``proxy.structural_cost``); elitism is implicit in mu+lambda
(a generation's mu+lambda pool is this generation's mu survivors from the
PREVIOUS generation plus lambda fresh offspring -- the previous best
individual is always a candidate for re-selection). Each offspring is one
mutation of a uniformly chosen SURVIVOR (a member of the current mu
population), operator drawn from the config's own pool distribution, retried
up to 8 times on ``VariationImpossible`` (drawing a fresh operator each
retry); if every retry fails the offspring falls back to an unmutated clone
of its parent (labelled with the last operator attempted, for the
per-operator-survival bookkeeping below). 6 restarts (seeds 0..5).

Initial population: for restart ``r`` and grammar variant ``dist``, the same
32 reduced starts (``starts.build_start``, deterministic from
``(r, dist)``-derived integer seeds, independent of pool/fitness/cost --
sharing the same starts across those factors, in the spirit of E2's "same
starts across mixtures") are each mutated once (same retry rule as above,
using the config's own pool) to seed generation 0.

Factors (48 configurations, each run at 6 restarts):

- ``dist`` in {G_FULL, G_NOBRANCH, G_FULL_INS, G_NOBRANCH_INS} (variants.py).
- ``pool`` in {DEFAULT_uniform, UNION_weighted} -- the E2 mixture
  definitions (``e2_drift.MIXTURES``), reused EXACTLY (same operator/weight
  dicts, not re-derived here) so a pool means the same thing in E2 and E5.
- ``fitness`` in {opposition, antipodal_pinch, reach_log = log1p(reach_coverage)}
  (proxy.py).
- ``cost`` in {none, aware}: ``aware`` subtracts ``0.02 * structural_cost``
  from the raw fitness above; ``none`` uses the raw fitness unchanged.
  ``structural_cost`` is always used for the truncation tie-break,
  regardless of ``cost``.

Fitness/noise model
--------------------
Every individual's fitness comes from ``proxy.all_proxies(model,
seed=generation_seed, n_configs=16)``. ``generation_seed`` is the SAME for
every individual evaluated in one generation (derived deterministically from
``(restart, dist, pool, fitness, cost, generation)``) but DIFFERENT across
generations -- so fitness is a noisy estimate of a "true" geometric score,
and comparisons WITHIN a generation are apples-to-apples while comparisons
ACROSS generations are not. Consequently mu survivors carried into a new
generation's mu+lambda pool are RE-EVALUATED under that generation's own
seed (never reusing a fitness value computed under a previous generation's
seed) -- this is what "elitism is implicit in mu+lambda" means operationally:
nothing here ever compares a cached old-seed fitness value against a new one.
A per-generation cache keyed by ``(generation, phenotype_hash)`` (built
fresh at the top of each generation, so it never crosses a generation
boundary) avoids recomputing ``all_proxies`` for duplicate phenotypes WITHIN
one generation (e.g. an unmutated clone fallback, or two offspring that
happen to land on the same phenotype) without ever serving a stale value
from a different generation's seed.

``fixed_eval_seed=True`` (used only by ``test_e5.py``) freezes
``generation_seed`` at generation 0's value for every generation, which
removes the noise above and makes fitness a deterministic function of
phenotype -- under which (mu+lambda) truncation selection is provably
best-fitness-non-decreasing (the previous best individual's fitness cannot
change, so it is always still a top-mu candidate). This is the property
``test_e5.py`` checks; it is not expected to hold (and is not claimed to
hold) in the real, noisy run.

Per-generation record: best/mean fitness; mean motors/digits/palm bodies/
joints; fraction of the (post-selection) population with a palm joint / a
branch / a coupling / a prismatic joint; distinct-phenotype-hash fraction;
per-operator offspring counts and survival counts for that generation
(offspring produced by that operator this generation that made it into the
next generation's population, over offspring produced by that operator this
generation).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..canonical import phenotype_hash
from ..coords import independent_joints
from ..derive import Derivation, VariationImpossible, derive, vary
from ..kinematics import KinematicModel
from ..phenodist import _digit_count, _palm_body_count
from ..proxy import all_proxies
from ..variants import G_FULL, G_FULL_INS, G_NOBRANCH, G_NOBRANCH_INS
from .e2_drift import MIXTURES, _draw_operator
from .runner import register, run_experiment
from .starts import build_start

# --------------------------------------------------------------------------
# Factors.
# --------------------------------------------------------------------------

DIST_VARIANTS: Dict[str, Any] = {
    "G_FULL": G_FULL, "G_NOBRANCH": G_NOBRANCH, "G_FULL_INS": G_FULL_INS, "G_NOBRANCH_INS": G_NOBRANCH_INS,
}

# Reuse E2's mixture definitions EXACTLY -- see module docstring. Never
# re-derive an operator pool's membership/weights here.
POOLS: Dict[str, Dict[str, float]] = {
    "DEFAULT_uniform": MIXTURES["DEFAULT_uniform"],
    "UNION_weighted": MIXTURES["UNION_weighted"],
}

FITNESS_NAMES: Tuple[str, ...] = ("opposition", "antipodal_pinch", "reach_log")
COST_MODES: Tuple[str, ...] = ("none", "aware")
COST_WEIGHT_DEFAULT = 0.02

DEFAULT_MU = 32
DEFAULT_LAMBDA = 32
DEFAULT_GENERATIONS = 40
DEFAULT_N_PROXY_CONFIGS = 16
_MUTATE_RETRY_ATTEMPTS = 8


def all_conditions() -> List[Tuple[str, str, str, str]]:
    """Every ``(dist_name, pool_name, fitness_name, cost_name)`` combination
    (48 total: 4 dists x 2 pools x 3 fitnesses x 2 cost modes)."""
    return [
        (dist_name, pool_name, fitness_name, cost_name)
        for dist_name in DIST_VARIANTS
        for pool_name in POOLS
        for fitness_name in FITNESS_NAMES
        for cost_name in COST_MODES
    ]


# --------------------------------------------------------------------------
# Fitness.
# --------------------------------------------------------------------------


def _base_fitness(proxies: Dict[str, float], fitness_name: str) -> float:
    if fitness_name == "opposition":
        return proxies["opposition"]
    if fitness_name == "antipodal_pinch":
        return proxies["antipodal_pinch"]
    if fitness_name == "reach_log":
        return math.log1p(proxies["reach_coverage"])
    raise ValueError(f"unknown fitness {fitness_name!r}")


def _fitness_value(proxies: Dict[str, float], fitness_name: str, cost_name: str,
                    cost_weight: float) -> float:
    base = _base_fitness(proxies, fitness_name)
    if cost_name == "none":
        return base
    if cost_name == "aware":
        return base - cost_weight * proxies["structural_cost"]
    raise ValueError(f"unknown cost mode {cost_name!r}")


# --------------------------------------------------------------------------
# Derivation-based construct predicates (mirrors e3_reach's own
# derivation-based extraction: exact, no ambiguity about branches/reordering).
# --------------------------------------------------------------------------


def _has_palm_joint(derivation: Derivation) -> bool:
    return any(s.production == "PalmBody" and s.params["has_joint"] for s in derivation.steps)


def _has_branch(derivation: Derivation) -> bool:
    return any(s.production == "Phalanx" and s.params["branch_digit_count"] > 0 for s in derivation.steps)


def _has_coupling(derivation: Derivation) -> bool:
    return any(
        s.production == "Phalanx" and s.params["module"]["kind"] == "Coupled" for s in derivation.steps
    )


def _has_prismatic(derivation: Derivation) -> bool:
    return any(s.production == "Phalanx" and s.params["module"]["kind"] == "P" for s in derivation.steps)


# --------------------------------------------------------------------------
# Deterministic RNG streams. Every stream is derived from the task's own
# (restart, dist_idx, pool_idx, fitness_idx, cost_idx) coordinates plus a
# small integer "phase" tag and any further integers (generation, index) --
# never from wall-clock/OS entropy -- so a run is byte-for-byte reproducible
# (see test_e5.py's determinism check).
# --------------------------------------------------------------------------

_PHASE_INIT_MUTATE = 0
_PHASE_OFFSPRING_MUTATE = 1
_PHASE_GEN_SEED = 2
_PHASE_SELECT = 3


def _rng(task_idxs: Sequence[int], phase: int, *rest: int) -> np.random.Generator:
    return np.random.default_rng([*task_idxs, phase, *rest])


def _start_seed_int(restart_seed: int, dist_idx: int, i: int) -> int:
    """Deterministic plain-integer seed for ``starts.build_start`` -- shared
    across every pool/fitness/cost config for the same ``(restart, dist)``
    (see module docstring's "same starts" note), so only ``restart`` and
    ``dist_idx`` (not pool/fitness/cost) feed into it."""
    return restart_seed * 1_000_000 + dist_idx * 10_000 + i


def _mutate_retry(derivation: Derivation, rng: np.random.Generator, dist, mixture: Dict[str, float],
                   max_attempts: int = _MUTATE_RETRY_ATTEMPTS) -> Tuple[str, Derivation]:
    """One mutation of ``derivation``, operator drawn from ``mixture`` (see
    ``e2_drift._draw_operator``), retried up to ``max_attempts`` times (each
    retry redraws the operator) on ``VariationImpossible``. Returns
    ``(operator_label, result)``; ``operator_label`` is the last operator
    attempted even on total failure, in which case ``result`` is an
    unmutated clone of ``derivation`` (see module docstring)."""
    op = _draw_operator(rng, mixture)
    for _ in range(max_attempts):
        try:
            return op, vary(derivation, rng, dist, operator=op)
        except VariationImpossible:
            op = _draw_operator(rng, mixture)
    return op, derivation


# --------------------------------------------------------------------------
# Evaluation.
# --------------------------------------------------------------------------


def _evaluate_population(
    derivations: Sequence[Derivation], generation: int, generation_seed: int, n_proxy_configs: int,
) -> List[Tuple[Derivation, KinematicModel, str, Dict[str, float]]]:
    """Derive+evaluate every derivation under ``generation_seed``, caching
    ``all_proxies`` by ``(generation, phenotype_hash)`` -- a fresh cache
    built here every call (i.e. every generation), so a phenotype recurring
    in a LATER generation (necessarily under a different seed) is always
    freshly evaluated; see module docstring's noise-model note."""
    cache: Dict[str, Dict[str, float]] = {}
    out = []
    for d in derivations:
        model = derive(d)
        h = phenotype_hash(model)
        proxies = cache.get(h)
        if proxies is None:
            proxies = all_proxies(model, seed=generation_seed, n_configs=n_proxy_configs)
            cache[h] = proxies
        out.append((d, model, h, proxies))
    return out


def _record_generation(
    generation: int, evaluated: Sequence[Tuple[Derivation, KinematicModel, str, Dict[str, float]]],
    fitness_vals: Sequence[float],
) -> Dict[str, Any]:
    n = len(evaluated)
    motors = [len(independent_joints(m)) for _, m, _, _ in evaluated]
    digits = [_digit_count(m) for _, m, _, _ in evaluated]
    palm_bodies = [_palm_body_count(m) for _, m, _, _ in evaluated]
    joints = [len(m.joints) for _, m, _, _ in evaluated]
    hashes = [h for _, _, h, _ in evaluated]
    return {
        "generation": int(generation),
        "best_fitness": float(max(fitness_vals)),
        "mean_fitness": float(np.mean(fitness_vals)),
        "mean_motors": float(np.mean(motors)),
        "mean_digits": float(np.mean(digits)),
        "mean_palm_bodies": float(np.mean(palm_bodies)),
        "mean_joints": float(np.mean(joints)),
        "frac_palm_joint": float(np.mean([_has_palm_joint(d) for d, _, _, _ in evaluated])),
        "frac_branch": float(np.mean([_has_branch(d) for d, _, _, _ in evaluated])),
        "frac_coupling": float(np.mean([_has_coupling(d) for d, _, _, _ in evaluated])),
        "frac_prismatic": float(np.mean([_has_prismatic(d) for d, _, _, _ in evaluated])),
        "distinct_hash_fraction": float(len(set(hashes)) / n) if n else 0.0,
    }


# --------------------------------------------------------------------------
# One (restart, dist, pool, fitness, cost) run.
# --------------------------------------------------------------------------


def e5_evolve_seed(
    task: Tuple[int, str, str, str, str], mu: int = DEFAULT_MU, lam: int = DEFAULT_LAMBDA,
    generations: int = DEFAULT_GENERATIONS, n_proxy_configs: int = DEFAULT_N_PROXY_CONFIGS,
    cost_weight: float = COST_WEIGHT_DEFAULT, fixed_eval_seed: bool = False,
) -> Dict[str, Any]:
    """One restart of one (dist, pool, fitness, cost) configuration. ``task``
    is ``(restart_seed, dist_name, pool_name, fitness_name, cost_name)`` --
    playing the role of ``run_experiment``'s per-task ``seed`` (see
    ``run()`` below, which builds ``restarts x conditions`` tasks so the
    multiprocess pool parallelizes over every restart AND every
    configuration, not just restarts)."""
    restart_seed, dist_name, pool_name, fitness_name, cost_name = task
    dist = DIST_VARIANTS[dist_name]
    mixture = POOLS[pool_name]
    ops = list(mixture.keys())
    dist_idx = list(DIST_VARIANTS).index(dist_name)
    pool_idx = list(POOLS).index(pool_name)
    fitness_idx = FITNESS_NAMES.index(fitness_name)
    cost_idx = COST_MODES.index(cost_name)
    task_idxs = (restart_seed, dist_idx, pool_idx, fitness_idx, cost_idx)

    def gen_seed(g: int) -> int:
        gg = 0 if fixed_eval_seed else g
        return int(_rng(task_idxs, _PHASE_GEN_SEED, gg).integers(0, 2**31 - 1))

    # Initial population: mu reduced starts (shared across pool/fitness/cost
    # for this (restart, dist) -- see _start_seed_int), each mutated once.
    population: List[Derivation] = []
    for i in range(mu):
        seed_int = _start_seed_int(restart_seed, dist_idx, i)
        start_derivation, _, _, _ = build_start(seed_int, dist)
        rng_i = _rng(task_idxs, _PHASE_INIT_MUTATE, i)
        _, mutated = _mutate_retry(start_derivation, rng_i, dist, mixture)
        population.append(mutated)

    op_offspring_total: Dict[str, int] = {op: 0 for op in ops}
    op_offspring_survived: Dict[str, int] = {op: 0 for op in ops}

    evaluated0 = _evaluate_population(population, 0, gen_seed(0), n_proxy_configs)
    fitness0 = [_fitness_value(p, fitness_name, cost_name, cost_weight) for _, _, _, p in evaluated0]
    trajectory: List[Dict[str, Any]] = [_record_generation(0, evaluated0, fitness0)]

    for g in range(1, generations + 1):
        rng_select = _rng(task_idxs, _PHASE_SELECT, g)
        offspring: List[Derivation] = []
        offspring_ops: List[str] = []
        for j in range(lam):
            parent = population[int(rng_select.integers(0, len(population)))]
            rng_mut = _rng(task_idxs, _PHASE_OFFSPRING_MUTATE, g, j)
            op_used, child = _mutate_retry(parent, rng_mut, dist, mixture)
            offspring.append(child)
            offspring_ops.append(op_used)

        combined = population + offspring
        evaluated = _evaluate_population(combined, g, gen_seed(g), n_proxy_configs)
        fitness_vals = [_fitness_value(p, fitness_name, cost_name, cost_weight) for _, _, _, p in evaluated]
        cost_vals = [p["structural_cost"] for _, _, _, p in evaluated]
        order = sorted(range(len(combined)), key=lambda idx: (-fitness_vals[idx], cost_vals[idx]))
        selected = order[: len(population)]
        selected_set = set(selected)

        n_parents = len(population)
        for j, op in enumerate(offspring_ops):
            combined_idx = n_parents + j
            op_offspring_total[op] += 1
            if combined_idx in selected_set:
                op_offspring_survived[op] += 1

        population = [combined[idx] for idx in selected]
        selected_evaluated = [evaluated[idx] for idx in selected]
        selected_fitness = [fitness_vals[idx] for idx in selected]
        trajectory.append(_record_generation(g, selected_evaluated, selected_fitness))

    final_best = trajectory[-1]["best_fitness"]
    threshold = 0.9 * final_best if final_best > 0 else final_best
    gen90 = trajectory[-1]["generation"]
    for row in trajectory:
        if row["best_fitness"] >= threshold:
            gen90 = row["generation"]
            break

    return {
        "restart_seed": int(restart_seed),
        "dist": dist_name,
        "pool": pool_name,
        "fitness": fitness_name,
        "cost": cost_name,
        "trajectory": trajectory,
        "operator_survival": {
            op: {"offspring": op_offspring_total[op], "survived": op_offspring_survived[op]} for op in ops
        },
        "final_best_fitness": float(final_best),
        "gen90": int(gen90),
        "final_mean_motors": float(trajectory[-1]["mean_motors"]),
        "final_mean_joints": float(trajectory[-1]["mean_joints"]),
        "construct_usage_gen40": {
            "frac_palm_joint": trajectory[-1]["frac_palm_joint"],
            "frac_branch": trajectory[-1]["frac_branch"],
            "frac_coupling": trajectory[-1]["frac_coupling"],
            "frac_prismatic": trajectory[-1]["frac_prismatic"],
        },
        "diversity_gen40": trajectory[-1]["distinct_hash_fraction"],
    }


register("e5_evolve", e5_evolve_seed)


# --------------------------------------------------------------------------
# Aggregation (over restarts, per configuration; construct usage per dist x
# pool averaged over fitness/cost; per-operator survival under
# UNION_weighted pooled over configs) and summary.md.
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
    return {
        "mean": float(arr.mean()),
        "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)),
        "n": n,
    }


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[Optional[float], Optional[float]]:
    if n == 0:
        return None, None
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def _wilson_rate(k: int, n: int) -> Dict[str, Optional[float]]:
    if n == 0:
        return {"rate": None, "ci_lo": None, "ci_hi": None, "n": 0}
    lo, hi = _wilson_ci(k, n)
    return {"rate": k / n, "ci_lo": lo, "ci_hi": hi, "n": n}


def _aggregate(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r["result"] for r in per_seed if r["ok"]]
    agg: Dict[str, Any] = {"n_ok": len(ok), "n_failed": len(per_seed) - len(ok)}

    # Table 1: dist x pool x fitness x cost, over restarts.
    by_condition: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for r in ok:
        by_condition.setdefault((r["dist"], r["pool"], r["fitness"], r["cost"]), []).append(r)
    table1: Dict[str, Any] = {}
    for cond, rows in by_condition.items():
        key = "::".join(cond)
        table1[key] = {
            "final_best_fitness": _bootstrap_ci([r["final_best_fitness"] for r in rows]),
            "final_mean_motors": _bootstrap_ci([r["final_mean_motors"] for r in rows]),
            "final_mean_joints": _bootstrap_ci([r["final_mean_joints"] for r in rows]),
            "gen90": _bootstrap_ci([float(r["gen90"]) for r in rows]),
            "diversity_gen40": _bootstrap_ci([r["diversity_gen40"] for r in rows]),
        }
    agg["table1"] = table1

    # Table 2: construct usage at gen 40, per dist x pool, averaged over
    # fitness/cost (pool every restart x fitness x cost row for that cell).
    by_dist_pool: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in ok:
        by_dist_pool.setdefault((r["dist"], r["pool"]), []).append(r)
    table2: Dict[str, Any] = {}
    for (dist_name, pool_name), rows in by_dist_pool.items():
        key = f"{dist_name}::{pool_name}"
        table2[key] = {
            construct: _bootstrap_ci([r["construct_usage_gen40"][construct] for r in rows])
            for construct in ("frac_palm_joint", "frac_branch", "frac_coupling", "frac_prismatic")
        }
    agg["table2"] = table2

    # Table 3: per-operator survival under UNION_weighted, pooled over every
    # (restart, dist, fitness, cost) that used that pool.
    union_rows = [r for r in ok if r["pool"] == "UNION_weighted"]
    op_totals: Dict[str, List[int]] = {}
    for r in union_rows:
        for op, d in r["operator_survival"].items():
            totals = op_totals.setdefault(op, [0, 0])
            totals[0] += d["offspring"]
            totals[1] += d["survived"]
    table3 = {op: _wilson_rate(surv, off) for op, (off, surv) in op_totals.items()}
    agg["table3_union_weighted_operator_survival"] = table3

    return agg


def _fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.4g}"


def _fmt_ci(d: Dict[str, Optional[float]], value_key: str = "mean") -> str:
    return f"{_fmt(d.get(value_key))} [{_fmt(d.get('ci_lo'))}, {_fmt(d.get('ci_hi'))}] (n={d.get('n')})"


def _summary_md(params: Dict[str, Any], aggregate: Dict[str, Any], wall_time_s: float) -> str:
    import json as _json

    lines = [
        "# Experiment: e5_evolve", "",
        f"Wall time: {wall_time_s:.3f} s", "",
        "## Params", "", "```json", _json.dumps(params, indent=2, sort_keys=True, default=str), "```", "",
        f"n_ok: {aggregate.get('n_ok')}  n_failed: {aggregate.get('n_failed')}", "",
        "## Table 1: final best fitness / final motors, per dist x pool x fitness x cost "
        "(mean, 95% bootstrap CI over restarts)", "",
        "| dist | pool | fitness | cost | final best fitness | final mean motors | "
        "gen reaching 90% of final best | diversity @ gen40 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cond, row in sorted(aggregate["table1"].items()):
        dist_name, pool_name, fitness_name, cost_name = cond.split("::")
        lines.append(
            f"| {dist_name} | {pool_name} | {fitness_name} | {cost_name} | "
            f"{_fmt_ci(row['final_best_fitness'])} | {_fmt_ci(row['final_mean_motors'])} | "
            f"{_fmt_ci(row['gen90'])} | {_fmt_ci(row['diversity_gen40'])} |"
        )
    lines += ["", "## Table 2: construct usage at generation 40, per dist x pool "
              "(averaged over fitness/cost, mean +/- 95% bootstrap CI)", "",
              "| dist | pool | frac palm joint | frac branch | frac coupling | frac prismatic |",
              "|---|---|---|---|---|---|"]
    for key, row in sorted(aggregate["table2"].items()):
        dist_name, pool_name = key.split("::")
        lines.append(
            f"| {dist_name} | {pool_name} | {_fmt_ci(row['frac_palm_joint'])} | "
            f"{_fmt_ci(row['frac_branch'])} | {_fmt_ci(row['frac_coupling'])} | "
            f"{_fmt_ci(row['frac_prismatic'])} |"
        )
    lines += ["", "## Table 3: per-operator survival under UNION_weighted "
              "(pooled over dist x fitness x cost x restart; Wilson 95% CI)", "",
              "| operator | survival rate | 95% CI lo | 95% CI hi | n offspring |",
              "|---|---|---|---|---|"]
    for op, row in sorted(aggregate["table3_union_weighted_operator_survival"].items()):
        lines.append(
            f"| {op} | {_fmt(row['rate'])} | {_fmt(row['ci_lo'])} | {_fmt(row['ci_hi'])} | {row['n']} |"
        )
    lines += ["", "## Reading", ""]
    for cond, row in sorted(aggregate["table1"].items()):
        lines.append(
            f"- {cond}: final best fitness {_fmt_ci(row['final_best_fitness'])}; "
            f"final mean motors {_fmt_ci(row['final_mean_motors'])}; "
            f"gen90 {_fmt_ci(row['gen90'])}; diversity@40 {_fmt_ci(row['diversity_gen40'])}."
        )
    lines += ["", "## Caveats", "",
              "- ``opposition``/``antipodal_pinch``/``reach_coverage``/``structural_cost`` are cheap "
              "geometric DIAGNOSTICS (see proxy.py's own module docstring), never a real manipulation "
              "task objective -- a high score under any of these fitnesses says nothing about "
              "grasp quality, only about what the grammar's operators can cheaply push a proxy toward.",
              "- Fitness is evaluated with the SAME seed for every individual within one generation but "
              "a DIFFERENT seed across generations (`proxy.all_proxies(model, seed=generation_seed, "
              "n_configs=16)`), so within-generation comparisons (selection) are apples-to-apples but "
              "the fitness trajectory across generations is a noisy estimate, not a monotone objective; "
              "mu survivors are re-evaluated under each new generation's seed rather than keeping a "
              "stale fitness value (see the module docstring's noise-model note).",
              "- 6 restarts is a small sample for the bootstrap CIs reported here; treat table 1/2 CIs "
              "as indicative, not as a claim of tight precision.",
              ""]
    return "\n".join(lines)


def run(
    out_dir: Optional[str] = None, restarts: Sequence[int] = range(6),
    conditions: Optional[Sequence[Tuple[str, str, str, str]]] = None,
    mu: int = DEFAULT_MU, lam: int = DEFAULT_LAMBDA, generations: int = DEFAULT_GENERATIONS,
    n_proxy_configs: int = DEFAULT_N_PROXY_CONFIGS, cost_weight: float = COST_WEIGHT_DEFAULT,
    fixed_eval_seed: bool = False, processes: int = 24, allow_dirty: bool = False,
) -> Dict[str, Any]:
    """Run ``e5_evolve_seed`` over ``restarts x conditions`` (default: every
    one of the 48 conditions in ``all_conditions()``) via the multiprocess
    runner -- each ``(restart, dist, pool, fitness, cost)`` combination is
    its own task, so the pool parallelizes across configurations as well as
    restarts (rather than looping configurations inside one per-restart
    task, which would leave most of ``processes`` idle whenever
    ``len(restarts) < processes``) -- then overwrite ``result.json``/
    ``summary.md`` with the tables described in this module's docstring."""
    if out_dir is None:
        out_dir = "project-notes/grammar/experiments/E5_evolve"
    if conditions is None:
        conditions = all_conditions()
    tasks = [
        (int(r), dist_name, pool_name, fitness_name, cost_name)
        for r in restarts
        for (dist_name, pool_name, fitness_name, cost_name) in conditions
    ]
    params = {
        "mu": mu, "lam": lam, "generations": generations, "n_proxy_configs": n_proxy_configs,
        "cost_weight": cost_weight, "fixed_eval_seed": fixed_eval_seed,
        "n_restarts": len(list(restarts)), "n_conditions": len(list(conditions)),
    }
    fn_params = {
        "mu": mu, "lam": lam, "generations": generations, "n_proxy_configs": n_proxy_configs,
        "cost_weight": cost_weight, "fixed_eval_seed": fixed_eval_seed,
    }
    result = run_experiment(
        "e5_evolve", e5_evolve_seed, params=fn_params, seeds=tasks, out_dir=out_dir, processes=processes,
        allow_dirty=allow_dirty,
    )
    aggregate = _aggregate(result["per_seed"])
    result["aggregate"] = aggregate
    result["params"] = params
    out_path = Path(out_dir)
    import json as _json

    (out_path / "result.json").write_text(_json.dumps(result, indent=2, sort_keys=True, default=str))
    (out_path / "summary.md").write_text(_summary_md(params, aggregate, result["wall_time_s"]))
    return result


if __name__ == "__main__":
    run()
