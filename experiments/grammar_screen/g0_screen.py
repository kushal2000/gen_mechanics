#!/usr/bin/env python3
"""G0 CPU grammar screen (plan revision 2026-09-27, step 2).

Martin's question: how does the grammar shape exploration of hand designs,
allowing diversity without making the problem intractable? This script
measures, cheaply on the CPU, how grammar variants trade viability against
diversity and explorability, using ONLY the frozen viability oracle
(`isaacsimenvs.inhand_reorient.scene.grammar_envelope.viability_report`) and
`hand_sampler.grammar` -- no Kit, no Isaac, no GPU.

Run with the project's `.venv_isaacsim` interpreter (importing
`isaacsimenvs` may pull in isaaclab-registered modules):

    .venv_isaacsim/bin/python experiments/grammar_screen/g0_screen.py \\
        --out-dir project-notes/grammar/experiments/g0

`--quick` shrinks every sample count for a smoke test (seconds, not
minutes) -- used by the acceptance test, not for real numbers.

Variants (see `hand_sampler.grammar.variants.G0_SCREEN_VARIANTS`):
  V0  G_SERIAL              -- old-sampler-like baseline.
  V1  DEFAULT, constrained  -- revolute only, no branching, envelope-shaped.
  V2  V1 + mount-spacing rule (I29).
  V3  V2 + curl-axis and opposition priors (I30).

Definitions used throughout this script (see `report.md` for the full
write-up):
  - "viable" design: `admitted` AND `fingertips_reachable >= 2` (the
    Viability metric's own "both combined" fraction) -- the pool every
    other section (diversity, evolvability, explorability) draws from.
  - descriptor cell: (digit_count in 1..5, joint_count bin), 5 x 6 = 30
    cells, joint bins [1-5, 6-10, 11-15, 16-20, 21-25, 26-32].
  - explorability fitness: `fingertips_reachable` (the primary criterion;
    0-5) plus `0.01 * proxy.opposition(model, configs)` as a small,
    same-scale tie-break (opposition in [0, 1], so it never changes the
    ranking between two different fingertip counts) -- computed only for
    structurally admitted designs (`fingertips_reachable is not None`).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hand_sampler.grammar.canonical import phenotype_hash  # noqa: E402
from hand_sampler.grammar.coords import sample_configurations  # noqa: E402
from hand_sampler.grammar.derive import (  # noqa: E402
    Derivation, EVOLUTION_OPERATORS, VariationImpossible, derive, sample_derivation, vary,
)
from hand_sampler.grammar.kinematics import KinematicModel, ModelError  # noqa: E402
from hand_sampler.grammar.proxy import opposition as proxy_opposition  # noqa: E402
from hand_sampler.grammar.phenodist import phenotype_distance  # noqa: E402
from hand_sampler.grammar.variants import G0_SCREEN_VARIANTS  # noqa: E402

from isaacsimenvs.inhand_reorient.scene.grammar_envelope import viability_report  # noqa: E402

DIGIT_BINS = 5  # digit_count in {1, ..., 5}, one bin each (envelope max is 5).
JOINT_BIN_EDGES = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 32)]
N_CELLS = DIGIT_BINS * len(JOINT_BIN_EDGES)


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def git_provenance() -> Dict[str, Any]:
    def _run(args: List[str]) -> Optional[str]:
        try:
            return subprocess.run(args, cwd=str(REPO_ROOT), capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    sha = _run(["git", "rev-parse", "HEAD"])
    porcelain = _run(["git", "status", "--porcelain"])
    return {"sha": sha, "dirty": bool(porcelain), "numpy": np.__version__, "python": sys.version.split()[0]}


# --------------------------------------------------------------------------
# Core per-design record
# --------------------------------------------------------------------------


@dataclass
class Record:
    seed: int
    admitted: bool
    reasons: Tuple[str, ...]
    max_rest_overlap_mm: Optional[float]
    fingertips_reachable: Optional[int]
    spawn_height_mm: Optional[float]
    digit_count: Optional[int]
    joint_count: Optional[int]
    wall_s: float
    # Rerun (opus-review-g0.md statistics section): the digit count the
    # GRAMMAR sampled, read straight off the derivation's own "hand" step --
    # unlike ``digit_count`` above (the oracle's own field), this is defined
    # even when the design fails the STRUCTURAL gate, so the per-digit-count
    # viable-fraction table can bucket every one of the 2000 fixed seeds,
    # not only the admitted ones.
    sampled_digit_count: int = 0

    @property
    def viable(self) -> bool:
        return bool(self.admitted) and self.fingertips_reachable is not None and self.fingertips_reachable >= 2


def evaluate_seed(dist, seed: int) -> Tuple[Record, Derivation, Optional[KinematicModel]]:
    t0 = time.perf_counter()
    d = sample_derivation(seed, dist)
    m = derive(d)
    r = viability_report(m)
    dt = time.perf_counter() - t0
    hand_step = next(s for s in d.steps if s.path == "hand")
    rec = Record(
        seed=seed, admitted=r["admitted"], reasons=tuple(r["reasons"]),
        max_rest_overlap_mm=r["max_rest_overlap_mm"], fingertips_reachable=r["fingertips_reachable"],
        spawn_height_mm=r["spawn_height_mm"], digit_count=r["digit_count"], joint_count=r["joint_count"],
        wall_s=dt, sampled_digit_count=int(hand_step.params["digit_count"]),
    )
    return rec, d, m


def _evaluate_seed_for_pool(args: Tuple[Any, int]) -> Record:
    """Top-level (picklable) wrapper for ``multiprocessing.Pool.map`` --
    returns only the ``Record`` (the ``Derivation``/``KinematicModel`` are
    cheap to recompute from ``(dist, seed)`` and are not needed by the
    parallel viability sweep itself; callers that need them for a later
    section re-derive from the seed, exactly as ``scan_viable_pool`` already
    does)."""
    dist, seed = args
    rec, _d, _m = evaluate_seed(dist, seed)
    return rec


def descriptor_cell(digit_count: Optional[int], joint_count: Optional[int]) -> Optional[Tuple[int, int]]:
    if digit_count is None or joint_count is None:
        return None
    if not (1 <= digit_count <= DIGIT_BINS):
        return None
    for j_idx, (lo, hi) in enumerate(JOINT_BIN_EDGES):
        if lo <= joint_count <= hi:
            return (digit_count - 1, j_idx)
    return None


def shannon_entropy_bits(counts: Dict[Tuple[int, int], int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * np.log2(p)
    return float(h)


# --------------------------------------------------------------------------
# Statistics (rerun, opus-review-g0.md "statistics" section): Wilson score
# CIs on a binomial rate, exact McNemar on paired (shared-seed) booleans.
# --------------------------------------------------------------------------

_Z_95 = 1.959963984540054


def wilson_ci(k: int, n: int, z: float = _Z_95) -> Dict[str, Optional[float]]:
    """Wilson score interval for a binomial rate ``k/n`` -- well-behaved
    (never degenerates to a point interval) even when ``k`` is 0 or ``n``,
    which is the common case for these viable-fraction rates."""
    if n == 0:
        return {"rate": None, "ci_lo": None, "ci_hi": None, "n": 0, "k": 0}
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z * z / (4 * n * n))
    return {
        "rate": phat, "ci_lo": max(0.0, center - half), "ci_hi": min(1.0, center + half), "n": n, "k": k,
    }


def exact_mcnemar(a_bools: Sequence[bool], b_bools: Sequence[bool]) -> Dict[str, Any]:
    """Exact (binomial) McNemar test on two PAIRED boolean sequences (same
    length, same underlying seed at each index -- e.g. two variants' own
    ``viable`` flags over the identical 0..1999 fixed-seed sweep). ``b`` is
    the count of seeds where A is True and B is False (discordant pairs
    favouring A); ``c`` the reverse. Two-sided exact p-value via the
    binomial distribution (``scipy.stats.binomtest``), the same test
    opus-review-g0.md itself reports."""
    from scipy.stats import binomtest

    assert len(a_bools) == len(b_bools)
    b = sum(1 for x, y in zip(a_bools, b_bools) if x and not y)
    c = sum(1 for x, y in zip(a_bools, b_bools) if (not x) and y)
    n_discordant = b + c
    if n_discordant == 0:
        p = 1.0
    else:
        p = float(binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided").pvalue)
    higher = "a" if b > c else ("b" if c > b else "tie")
    return {"n_a_only": b, "n_b_only": c, "n_discordant": n_discordant, "p_value": p, "higher": higher}


# --------------------------------------------------------------------------
# 1. Viability + 2. Descriptor coverage + Complexity/cost (one seed sweep)
# --------------------------------------------------------------------------


def run_viability(dist, n_seeds: int, pool: Optional["mp.pool.Pool"] = None) -> Dict[str, Any]:
    reasons_hist: Counter = Counter()
    if pool is not None:
        records: List[Record] = pool.map(_evaluate_seed_for_pool, [(dist, s) for s in range(n_seeds)],
                                          chunksize=max(1, n_seeds // 100))
    else:
        records = [evaluate_seed(dist, seed)[0] for seed in range(n_seeds)]
    for rec in records:
        for reason in rec.reasons:
            key = reason.split(" (")[0].split(":")[0]
            reasons_hist[key[:60]] += 1

    n = len(records)
    n_admitted = sum(1 for r in records if r.admitted)
    n_two_tips = sum(1 for r in records if r.fingertips_reachable is not None and r.fingertips_reachable >= 2)
    n_both = sum(1 for r in records if r.viable)

    viable = [r for r in records if r.viable]
    cell_counts: Counter = Counter()
    for r in viable:
        cell = descriptor_cell(r.digit_count, r.joint_count)
        if cell is not None:
            cell_counts[cell] += 1

    joint_counts_viable = [r.joint_count for r in viable if r.joint_count is not None]
    total_wall = sum(r.wall_s for r in records)

    # Rerun: per-digit-count (SAMPLED, not oracle-reported -- see Record's
    # own docstring) viable fraction, digits 1..5, with Wilson CIs.
    per_digit_count: Dict[str, Any] = {}
    for dc in range(1, DIGIT_BINS + 1):
        bucket = [r for r in records if r.sampled_digit_count == dc]
        n_bucket = len(bucket)
        n_viable_bucket = sum(1 for r in bucket if r.viable)
        per_digit_count[str(dc)] = wilson_ci(n_viable_bucket, n_bucket)

    return {
        "n_seeds": n,
        "frac_admitted": n_admitted / n if n else 0.0,
        "frac_two_or_more_fingertips": n_two_tips / n if n else 0.0,
        "frac_both": n_both / n if n else 0.0,
        "frac_both_wilson_ci": wilson_ci(n_both, n),
        "per_digit_count_viable": per_digit_count,
        "reasons_histogram": dict(reasons_hist.most_common(20)),
        "descriptor_coverage": {
            "cells_occupied": len(cell_counts),
            "n_cells": N_CELLS,
            "shannon_entropy_bits": shannon_entropy_bits(cell_counts),
            "per_cell_counts": {f"d{d + 1}_j{JOINT_BIN_EDGES[j][0]}-{JOINT_BIN_EDGES[j][1]}": c
                                for (d, j), c in sorted(cell_counts.items())},
        },
        "complexity": {
            "joint_count_histogram": dict(Counter(joint_counts_viable)),
            "joint_count_mean": float(np.mean(joint_counts_viable)) if joint_counts_viable else None,
            "joint_count_median": float(np.median(joint_counts_viable)) if joint_counts_viable else None,
        },
        "cost": {
            "total_wall_s": total_wall,
            "mean_wall_s_per_design": total_wall / n if n else None,
            "mean_wall_s_per_viable_design": (total_wall / n_both) if n_both else None,
        },
        "n_viable_found": n_both,
        "records": records,  # kept for reuse by other sections; stripped before JSON dump.
    }


# --------------------------------------------------------------------------
# Viable-pool scanning (for diversity / evolvability / explorability, which
# need MORE viable designs than a low-admit-rate variant's first n_seeds
# fixed viability sweep may contain).
# --------------------------------------------------------------------------


def scan_viable_pool(dist, target_n: int, seed_records: List[Record], extra_seed_start: int,
                      max_extra_seeds: int) -> Tuple[List[Tuple[Record, Derivation]], int]:
    """Reuses viable seeds already found in ``seed_records`` (the fixed
    viability sweep), then scans additional seeds starting at
    ``extra_seed_start`` (never overlapping the fixed sweep) until
    ``target_n`` viable designs are collected or ``max_extra_seeds`` extra
    seeds have been tried. Returns ``(pool, n_extra_seeds_scanned)``; the
    pool may be SHORTER than ``target_n`` (reported, not padded)."""
    pool: List[Tuple[Record, Derivation]] = []
    for r in seed_records:
        if r.viable:
            d = sample_derivation(r.seed, dist)
            pool.append((r, d))
            if len(pool) >= target_n:
                return pool, 0

    n_extra = 0
    seed = extra_seed_start
    while len(pool) < target_n and n_extra < max_extra_seeds:
        rec, d, _m = evaluate_seed(dist, seed)
        n_extra += 1
        seed += 1
        if rec.viable:
            pool.append((rec, d))
    return pool, n_extra


# --------------------------------------------------------------------------
# 3. Diversity
# --------------------------------------------------------------------------


def run_diversity(dist, pool: List[Tuple[Record, Derivation]], n_pairs: int, n_configs: int,
                   rng: np.random.Generator) -> Dict[str, Any]:
    n = len(pool)
    if n < 2:
        return {"n_designs": n, "n_pairs_sampled": 0, "mean_tip_displacement_m": None}
    models = [derive(d) for _r, d in pool]
    all_pairs = n * (n - 1) // 2
    n_pairs = min(n_pairs, all_pairs)
    seen = set()
    dists = []
    attempts = 0
    while len(dists) < n_pairs and attempts < n_pairs * 20:
        attempts += 1
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        pd = phenotype_distance(models[i], models[j], seed=0, n_configs=n_configs)
        dists.append(pd["tip_displacement_m"])
    return {
        "n_designs": n,
        "n_pairs_sampled": len(dists),
        "n_pairs_total_possible": all_pairs,
        "mean_tip_displacement_m": float(np.mean(dists)) if dists else None,
        "std_tip_displacement_m": float(np.std(dists)) if dists else None,
    }


# --------------------------------------------------------------------------
# 4. Evolvability
# --------------------------------------------------------------------------


def run_evolvability(dist, pool: List[Tuple[Record, Derivation]], n_children: int,
                      rng: np.random.Generator) -> Dict[str, Any]:
    n_attempted = 0
    n_produced = 0
    n_viable_children = 0
    n_cell_defined = 0
    n_cell_changed = 0
    n_identical = 0
    n_neutral_not_identical = 0

    for parent_rec, parent_d in pool:
        parent_model = derive(parent_d)
        parent_hash = phenotype_hash(parent_model)
        parent_cell = descriptor_cell(parent_rec.digit_count, parent_rec.joint_count)
        for _ in range(n_children):
            n_attempted += 1
            try:
                child_d = vary(parent_d, rng, dist, operators=EVOLUTION_OPERATORS)
            except VariationImpossible:
                continue
            try:
                child_model = derive(child_d)
            except ModelError:
                continue
            n_produced += 1
            child_report = viability_report(child_model)
            child_viable = child_report["admitted"] and child_report["fingertips_reachable"] is not None \
                and child_report["fingertips_reachable"] >= 2
            if child_viable:
                n_viable_children += 1
            child_cell = descriptor_cell(child_report["digit_count"], child_report["joint_count"])
            if child_cell is not None and parent_cell is not None:
                n_cell_defined += 1
                if child_cell != parent_cell:
                    n_cell_changed += 1
            child_hash = phenotype_hash(child_model)
            if child_hash == parent_hash:
                n_identical += 1
            elif child_cell is not None and parent_cell is not None and child_cell == parent_cell:
                n_neutral_not_identical += 1

    return {
        "n_parents": len(pool),
        "n_children_attempted": n_attempted,
        "n_children_produced": n_produced,
        "child_viability_rate": (n_viable_children / n_produced) if n_produced else None,
        "frac_children_change_cell": (n_cell_changed / n_cell_defined) if n_cell_defined else None,
        "frac_children_identical": (n_identical / n_produced) if n_produced else None,
        "frac_children_identical_or_neutral": ((n_identical + n_neutral_not_identical) / n_produced)
        if n_produced else None,
    }


# --------------------------------------------------------------------------
# 5. Explorability: CPU MAP-Elites, no RL.
# --------------------------------------------------------------------------


def fitness_of(report: Dict[str, Any], model: Optional[KinematicModel]) -> Optional[float]:
    if report["fingertips_reachable"] is None or model is None:
        return None
    cfgs = sample_configurations(model, 4, 0)
    tie = proxy_opposition(model, cfgs)
    return float(report["fingertips_reachable"]) + 0.01 * float(tie)


def _is_viable_report(r: Dict[str, Any]) -> bool:
    return bool(r["admitted"]) and r["fingertips_reachable"] is not None and r["fingertips_reachable"] >= 2


def _run_one_explorability_repeat(args: Tuple[Any, List[Tuple[Record, Derivation]], int, List[int]]) -> Dict[str, Any]:
    """One CPU MAP-Elites repeat -- a top-level (picklable) function so
    ``run_explorability`` can map it over ``n_repeats`` in a shared
    ``multiprocessing.Pool`` (each repeat is an independent archive, so
    this is embarrassingly parallel).

    Opus review of G0 (item 6) fix: the archive admits a child ONLY when it
    is VIABLE (``admitted`` AND ``fingertips_reachable >= 2``) -- the
    previous version admitted anything that merely passed the STRUCTURAL
    gate (``cell is not None``), which let non-viable children pollute the
    coverage/founder-diversity numbers this section reports."""
    dist, founders, n_evals, seed_key = args
    rng = np.random.default_rng(seed_key)
    archive: Dict[Tuple[int, int], Tuple[float, Derivation, int]] = {}

    def try_insert(d: Derivation, founder_seed: int) -> None:
        try:
            m = derive(d)
        except ModelError:
            return
        r = viability_report(m)
        if not _is_viable_report(r):
            return
        cell = descriptor_cell(r["digit_count"], r["joint_count"])
        if cell is None:
            return
        f = fitness_of(r, m)
        if f is None:
            return
        cur = archive.get(cell)
        if cur is None or f > cur[0]:
            archive[cell] = (f, d, founder_seed)

    for rec, d in founders:
        try_insert(d, rec.seed)

    n_success = 0
    attempts = 0
    while n_success < n_evals and attempts < n_evals * 4:
        attempts += 1
        if not archive:
            break
        cells = list(archive.keys())
        parent_cell = cells[int(rng.integers(0, len(cells)))]
        _pf, parent_d, parent_founder = archive[parent_cell]
        try:
            child_d = vary(parent_d, rng, dist, operators=EVOLUTION_OPERATORS)
        except VariationImpossible:
            continue
        n_success += 1
        try_insert(child_d, parent_founder)

    founders_in_archive = {founder for _f, _d, founder in archive.values()}
    return {
        "cells_occupied": len(archive),
        "coverage_frac": len(archive) / N_CELLS,
        "distinct_founders_in_archive": len(founders_in_archive),
        "n_founders": len(founders),
        "n_evals_succeeded": n_success,
        "n_eval_attempts": attempts,
    }


def run_explorability(dist, founders: List[Tuple[Record, Derivation]], n_evals: int,
                       n_repeats: int, base_seed: int, founders_target: int,
                       pool: Optional["mp.pool.Pool"] = None) -> Dict[str, Any]:
    tasks = [(dist, founders, n_evals, [base_seed, rep]) for rep in range(n_repeats)]
    if pool is not None:
        per_repeat: List[Dict[str, Any]] = pool.map(_run_one_explorability_repeat, tasks)
    else:
        per_repeat = [_run_one_explorability_repeat(t) for t in tasks]

    coverages = [p["coverage_frac"] for p in per_repeat]
    founders_ct = [p["distinct_founders_in_archive"] for p in per_repeat]
    return {
        "n_repeats": n_repeats,
        "n_evals_target": n_evals,
        "founders_target": founders_target,
        "founders_found": len(founders),
        # Opus review item 6: every variant is asked for the SAME
        # ``founders_target`` (fair comparison); this is False, honestly,
        # for any variant whose viable-pool scan could not fill it within
        # budget -- never padded.
        "founders_target_met": len(founders) >= founders_target,
        "per_repeat": per_repeat,
        "coverage_frac_mean": float(np.mean(coverages)) if coverages else None,
        "coverage_frac_std": float(np.std(coverages)) if coverages else None,
        "distinct_founders_mean": float(np.mean(founders_ct)) if founders_ct else None,
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def strip_records(viability_result: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(viability_result)
    out.pop("records", None)
    return out


def run_variant(name: str, dist, args, variant_index: int,
                 pool: Optional["mp.pool.Pool"] = None) -> Dict[str, Any]:
    t_variant0 = time.perf_counter()
    print(f"[{name}] viability sweep ({args.viability_seeds} seeds)...", flush=True)
    viability = run_viability(dist, args.viability_seeds, pool=pool)
    print(f"[{name}]   frac_admitted={viability['frac_admitted']:.3f} "
          f"frac_2tips={viability['frac_two_or_more_fingertips']:.3f} "
          f"frac_both={viability['frac_both']:.3f} n_viable={viability['n_viable_found']}", flush=True)

    diversity_pool, n_extra_div = scan_viable_pool(
        dist, args.diversity_n, viability["records"], args.viability_seeds, args.max_extra_seeds,
    )
    evolve_pool, n_extra_evo = scan_viable_pool(
        dist, args.evolvability_parents, viability["records"], args.viability_seeds + args.max_extra_seeds,
        args.max_extra_seeds,
    )
    # Opus review item 6: every variant asks for the SAME founders target
    # (``args.explore_founders``, e.g. 20) -- never padded if the scan
    # cannot fill it (see ``run_explorability``'s own ``founders_target_met``).
    founder_pool, n_extra_founders = scan_viable_pool(
        dist, args.explore_founders, viability["records"],
        args.viability_seeds + 2 * args.max_extra_seeds, args.max_extra_seeds,
    )

    print(f"[{name}] diversity ({len(diversity_pool)} designs, {n_extra_div} extra seeds scanned)...", flush=True)
    rng_div = np.random.default_rng([args.run_seed, variant_index, 1])
    diversity = run_diversity(dist, diversity_pool, args.diversity_pairs, args.diversity_n_configs, rng_div)

    print(f"[{name}] evolvability ({len(evolve_pool)} parents, {n_extra_evo} extra seeds scanned)...", flush=True)
    rng_evo = np.random.default_rng([args.run_seed, variant_index, 2])
    evolvability = run_evolvability(dist, evolve_pool, args.evolvability_children, rng_evo)

    print(f"[{name}] explorability ({len(founder_pool)}/{args.explore_founders} founders, "
          f"{n_extra_founders} extra seeds scanned, "
          f"{args.explore_evals} evals x {args.explore_repeats} repeats)...", flush=True)
    explorability = run_explorability(dist, founder_pool, args.explore_evals, args.explore_repeats,
                                       base_seed=args.run_seed * 10000 + variant_index * 1000,
                                       founders_target=args.explore_founders, pool=pool)

    wall_variant = time.perf_counter() - t_variant0
    print(f"[{name}] done in {wall_variant:.1f}s", flush=True)

    return {
        "variant": name,
        "viability": strip_records(viability),
        "diversity": {**diversity, "n_extra_seeds_scanned": n_extra_div},
        "evolvability": {**evolvability, "n_extra_seeds_scanned": n_extra_evo},
        "explorability": {**explorability, "n_extra_seeds_scanned": n_extra_founders},
        "wall_s_total": wall_variant,
        # Kept only in-memory for the cross-variant McNemar section (never
        # JSON-dumped -- see ``strip_viable_by_seed``): the per-seed
        # ``viable`` boolean over the fixed 0..``viability_seeds`` sweep,
        # aligned by seed index across every variant.
        "_viable_by_seed": [r.viable for r in viability["records"]],
    }


def pairwise_mcnemar(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Exact McNemar's test, every pair of variants, on their SHARED fixed
    seeds (0..n-1, the viability sweep) -- opus-review-g0.md's own
    "statistics" section. Requires every variant in ``results`` to have
    been run with the SAME ``--viability-seeds`` (asserted)."""
    out: Dict[str, Any] = {}
    lengths = {r["variant"]: len(r["_viable_by_seed"]) for r in results}
    if len(set(lengths.values())) > 1:
        out["error"] = f"variants do not share the same seed count: {lengths}"
        return out
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i], results[j]
            key = f"{a['variant']}_vs_{b['variant']}"
            out[key] = exact_mcnemar(a["_viable_by_seed"], b["_viable_by_seed"])
    return out


def write_outputs(results: List[Dict[str, Any]], provenance: Dict[str, Any], args, out_dir: Path,
                   mcnemar: Optional[Dict[str, Any]] = None, tag: str = "v2") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    payload = {"provenance": provenance, "args": vars(args), "results": clean_results, "mcnemar": mcnemar or {}}
    json_path = out_dir / f"g0_results_{tag}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    csv_path = out_dir / f"g0_summary_{tag}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "variant", "n_seeds", "frac_admitted", "frac_two_or_more_fingertips", "frac_both",
            "frac_both_ci_lo", "frac_both_ci_hi",
            "cells_occupied", "n_cells", "shannon_entropy_bits",
            "mean_tip_displacement_m", "n_pairs_sampled",
            "child_viability_rate", "frac_children_change_cell", "frac_children_identical_or_neutral",
            "founders_found", "founders_target", "coverage_frac_mean", "coverage_frac_std",
            "distinct_founders_mean",
            "joint_count_mean", "mean_wall_s_per_design", "mean_wall_s_per_viable_design",
        ])
        for r in clean_results:
            v, dv, ev, ex = r["viability"], r["diversity"], r["evolvability"], r["explorability"]
            ci = v["frac_both_wilson_ci"]
            w.writerow([
                r["variant"], v["n_seeds"], v["frac_admitted"], v["frac_two_or_more_fingertips"], v["frac_both"],
                ci["ci_lo"], ci["ci_hi"],
                v["descriptor_coverage"]["cells_occupied"], v["descriptor_coverage"]["n_cells"],
                v["descriptor_coverage"]["shannon_entropy_bits"],
                dv["mean_tip_displacement_m"], dv["n_pairs_sampled"],
                ev["child_viability_rate"], ev["frac_children_change_cell"], ev["frac_children_identical_or_neutral"],
                ex["founders_found"], ex["founders_target"], ex["coverage_frac_mean"], ex["coverage_frac_std"],
                ex["distinct_founders_mean"],
                v["complexity"]["joint_count_mean"], v["cost"]["mean_wall_s_per_design"],
                v["cost"]["mean_wall_s_per_viable_design"],
            ])

    digit_csv_path = out_dir / f"g0_per_digit_count_{tag}.csv"
    with digit_csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "digit_count", "n", "n_viable", "rate", "ci_lo", "ci_hi"])
        for r in clean_results:
            for dc, stats in sorted(r["viability"]["per_digit_count_viable"].items(), key=lambda kv: int(kv[0])):
                w.writerow([r["variant"], dc, stats["n"], stats["k"], stats["rate"], stats["ci_lo"], stats["ci_hi"]])

    print(f"wrote {json_path}, {csv_path}, and {digit_csv_path}")


def make_plots(results: List[Dict[str, Any]], out_dir: Path, tag: str = "v2") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    names = [r["variant"] for r in results]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(names))
    w = 0.25
    ax.bar(x - w, [r["viability"]["frac_admitted"] for r in results], w, label="admitted")
    ax.bar(x, [r["viability"]["frac_two_or_more_fingertips"] for r in results], w, label=">=2 fingertips")
    both = [r["viability"]["frac_both"] for r in results]
    ci_lo = [r["viability"]["frac_both_wilson_ci"]["ci_lo"] for r in results]
    ci_hi = [r["viability"]["frac_both_wilson_ci"]["ci_hi"] for r in results]
    yerr = [[b - lo for b, lo in zip(both, ci_lo)], [hi - b for b, hi in zip(both, ci_hi)]]
    ax.bar(x + w, both, w, yerr=yerr, capsize=3, label="both (viable), 95% Wilson CI")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("fraction of sampled seeds")
    ax.set_title("G0 rerun viability by variant")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"g0_viability_{tag}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for r in results:
        dcs = sorted(r["viability"]["per_digit_count_viable"].items(), key=lambda kv: int(kv[0]))
        xs = [int(k) for k, _ in dcs]
        rates = [v["rate"] if v["rate"] is not None else 0.0 for _k, v in dcs]
        ax.plot(xs, rates, marker="o", label=r["variant"])
    ax.set_xlabel("digit count (sampled)")
    ax.set_ylabel("viable fraction")
    ax.set_title("G0 rerun: viable fraction by digit count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"g0_per_digit_count_{tag}.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(results), figsize=(3.2 * len(results), 3.2))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        grid = np.zeros((DIGIT_BINS, len(JOINT_BIN_EDGES)))
        for key, count in r["viability"]["descriptor_coverage"]["per_cell_counts"].items():
            d_part, j_part = key.split("_j")
            d_idx = int(d_part[1:]) - 1
            lo = int(j_part.split("-")[0])
            j_idx = [i for i, (a, _b) in enumerate(JOINT_BIN_EDGES) if a == lo][0]
            grid[d_idx, j_idx] = count
        im = ax.imshow(grid, cmap="viridis", aspect="auto")
        ax.set_title(r["variant"])
        ax.set_xlabel("joint bin")
        ax.set_ylabel("digits")
        ax.set_xticks(range(len(JOINT_BIN_EDGES)))
        ax.set_xticklabels([f"{a}-{b}" for a, b in JOINT_BIN_EDGES], rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(DIGIT_BINS))
        ax.set_yticklabels(range(1, DIGIT_BINS + 1))
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / f"g0_descriptor_coverage_{tag}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    cov_mean = [r["explorability"]["coverage_frac_mean"] or 0.0 for r in results]
    cov_std = [r["explorability"]["coverage_frac_std"] or 0.0 for r in results]
    ax.bar(x, cov_mean, yerr=cov_std, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("final archive coverage (fraction of 30 cells)")
    ax.set_title("G0 rerun explorability: MAP-Elites coverage (mean +/- std over repeats)")
    fig.tight_layout()
    fig.savefig(out_dir / f"g0_explorability_{tag}.png", dpi=150)
    plt.close(fig)

    print(f"wrote plots to {out_dir}")


# Rerun (opus-review-g0.md, "Rerun G0 after [oracle-v2] lands"): V1 (kept,
# the fair baseline), V1s/V2s/V3s (fixed), V3 (kept, for before/after
# comparison against V3s) -- V0/V2 (old, superseded) stay selectable via
# ``--variants`` but are not part of the default rerun roster.
DEFAULT_RERUN_VARIANTS = ["V1", "V1s", "V2s", "V3", "V3s"]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", nargs="+", default=DEFAULT_RERUN_VARIANTS,
                   help=f"subset of {list(G0_SCREEN_VARIANTS.keys())} to run "
                        f"(default: {DEFAULT_RERUN_VARIANTS})")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "project-notes/grammar/experiments/g0")
    p.add_argument("--tag", type=str, default="v2",
                   help="output filename suffix, e.g. g0_results_<tag>.json (keeps the original "
                        "g0_results.json etc. from the superseded run untouched)")
    p.add_argument("--run-seed", type=int, default=0)
    p.add_argument("--viability-seeds", type=int, default=2000)
    p.add_argument("--diversity-n", type=int, default=300)
    p.add_argument("--diversity-pairs", type=int, default=3000,
                   help="random subsample of pairs for the mean pairwise phenodist "
                        "(full C(n,2) is expensive at n=300); an unbiased Monte Carlo estimate")
    p.add_argument("--diversity-n-configs", type=int, default=8)
    p.add_argument("--evolvability-parents", type=int, default=200)
    p.add_argument("--evolvability-children", type=int, default=5)
    # Opus review item 6: one fixed founders target for EVERY variant (fair
    # comparison) -- honestly under-filled, never padded, when a variant's
    # scan cannot reach it (see ``run_explorability``'s ``founders_target_met``).
    p.add_argument("--explore-founders", type=int, default=20)
    p.add_argument("--explore-evals", type=int, default=3000)
    p.add_argument("--explore-repeats", type=int, default=3)
    p.add_argument("--max-extra-seeds", type=int, default=20000,
                   help="cap on extra seeds scanned per pool beyond the fixed viability sweep, "
                        "for a low-admit-rate variant that needs more than "
                        "--viability-seeds seeds to fill a pool")
    p.add_argument("--n-workers", type=int, default=None,
                   help="multiprocessing worker count for the viability sweep and explorability "
                        "repeats (default: cpu_count() - 4, leaving >= 4 cores free; 0 or 1 disables "
                        "multiprocessing)")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--quick", action="store_true", help="tiny sample counts, for a fast smoke test")
    return p


def apply_quick(args: argparse.Namespace) -> None:
    args.viability_seeds = 40
    args.diversity_n = 6
    args.diversity_pairs = 10
    args.evolvability_parents = 3
    args.evolvability_children = 2
    args.explore_founders = 3
    args.explore_evals = 15
    args.explore_repeats = 2
    args.max_extra_seeds = 40


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.quick:
        apply_quick(args)

    provenance = git_provenance()
    print(f"G0 screen: git sha={provenance['sha']} dirty={provenance['dirty']} quick={args.quick}")

    n_workers = args.n_workers
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 4)
    print(f"G0 screen: {os.cpu_count()} CPUs detected, using {n_workers} worker process(es) "
          f"(leaving {(os.cpu_count() or 1) - n_workers} free)")

    pool = mp.Pool(processes=n_workers) if n_workers > 1 else None
    try:
        results = []
        for variant_index, name in enumerate(args.variants):
            if name not in G0_SCREEN_VARIANTS:
                raise SystemExit(f"unknown variant {name!r}; choices: {list(G0_SCREEN_VARIANTS)}")
            dist = G0_SCREEN_VARIANTS[name]
            results.append(run_variant(name, dist, args, variant_index, pool=pool))
            # Write after EVERY variant (not just at the end): a screen that
            # finds some variants far more expensive than others should
            # never lose already-completed variants to a later timeout/kill.
            mcnemar = pairwise_mcnemar(results) if len(results) > 1 else {}
            write_outputs(results, provenance, args, args.out_dir, mcnemar=mcnemar, tag=args.tag)
            if not args.no_plots:
                make_plots(results, args.out_dir, tag=args.tag)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
