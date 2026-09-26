"""E12: the balance test suite (grammar 0.5, iteration B) --
``balanced-grammar-synthesis.md`` section 5, items 1-3, 5-6:

(a) Per-pair neutral drift (grammar 0.5, I18 fix 4 -- TWO regimes, both
    reported): mutation-only walks applying ONLY one ``derive.EVOLUTION_PAIRS``
    pair (equal probability over its 1-2 operators, via ``vary_tracked(...,
    operators=pair)``), 128 seeds x 200 steps, under ``G_FULL_INS`` and
    ``G_NOBRANCH_INS``; and the same for the whole ``EVOLUTION_OPERATORS``
    pool (uniform).
      - ``boundary`` regime: starts are ``starts.build_start`` (reduced to
        near-minimal digit/phalanx counts) -- reduced-size starts have an
        inherently lopsided growth/shrink applicability ratio (shrink moves
        are often inapplicable at the boundary), so drift here is EXPECTED
        POSITIVE; reported descriptively (mean, no CI-based PASS/FAIL).
      - ``stationary`` regime: starts are plain ``generate(seed, dist)``
        (unreduced samples from the grammar's own stationary distribution).
        Reports E[delta joints], E[delta digits], E[delta palm bodies] with
        a 95% (unpaired) bootstrap CI. PASS iff the CI includes 0 for every
        pair/pool/dist/metric -- this regime alone determines (a)'s overall
        PASS/FAIL.
    Both regimes also report, per pair, each of its 1-2 operators'
    applicability rate over the walk (successful ``vary_tracked`` draws /
    total draws of that operator) -- the growth/shrink ratio explains any
    residual drift.
(b) Locality bands (grammar 0.5, I18 fix 5 -- per-operator, not pooled):
    reuses ``e1_locality.e1_locality_seed`` (under ``G_FULL``, its own
    hardcoded dist) over a seed sample, then checks EVERY operator in
    ``EVOLUTION_OPERATORS`` individually: PASS iff every ``SMALL_STEP_OPERATORS``
    member's median ``tip_displacement_m`` <= 0.005 m (5 mm) AND every
    "structural" operator (``EVOLUTION_OPERATORS`` minus
    ``SMALL_STEP_OPERATORS``)'s p90 <= 0.060 m (60 mm); violators are listed
    by name.
(c) Reversibility: ``derive.apply_operator`` applies each pair's growth
    move then its shrink move (``target`` = the uid of the material the
    growth move just added -- self-inverse ``toggle_palm_joint`` targets a
    palm body currently WITHOUT a joint, so the growth direction is
    unambiguous), on 500 parents each, under ``G_FULL_INS`` and
    ``G_NOBRANCH_INS``. Pass: >= 95% of APPLICABLE cases recover the
    parent's own ``phenotype_hash`` exactly.
(d) Prior size histograms: joints/digits/motors histograms (and the
    fraction of mass at the observed min/max value -- "saturated at a
    bound") for ``G_FULL_INS``, ``G_NOBRANCH_INS``, ``G_BEND``. Pass: no
    bound holds > 20% of the mass, for any of the 3 metrics x 3 dists.
(e) Redundancy: 5000 seeds under ``G_FULL``, distinct ``phenotype_hash``
    fraction.

``run()`` drives all five, writes ``result.json`` and a ``summary.md`` with
one PASS/FAIL line per criterion (into
``project-notes/grammar/experiments/E12_balance`` by default).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..canonical import phenotype_hash
from ..coords import independent_joints
from ..derive import (
    EVOLUTION_OPERATORS, EVOLUTION_PAIRS, SMALL_STEP_OPERATORS, VariationImpossible, apply_operator,
    derive, generate, vary, vary_tracked,
)
from ..kinematics import KinematicModel
from ..phenodist import _digit_count, _palm_body_count
from ..variants import G_BEND, G_FULL, G_FULL_INS, G_NOBRANCH_INS
from .e1_locality import e1_locality_seed
from .runner import _git_diff_sha256, _git_info
from .starts import build_start

DIST_VARIANTS: Dict[str, Any] = {"G_FULL_INS": G_FULL_INS, "G_NOBRANCH_INS": G_NOBRANCH_INS}

PAIR_LABELS: Tuple[Tuple[str, Tuple[str, ...]], ...] = tuple(
    (f"{growth}/{shrink}" if growth != shrink else growth, (growth,) if growth == shrink else (growth, shrink))
    for growth, shrink in EVOLUTION_PAIRS
) + (("EVOLUTION_pool", tuple(EVOLUTION_OPERATORS)),)

STRUCTURAL_OPS: Tuple[str, ...] = tuple(op for op in EVOLUTION_OPERATORS if op not in SMALL_STEP_OPERATORS)


# --------------------------------------------------------------------------
# (a) Per-pair neutral drift
# --------------------------------------------------------------------------


def _metrics(model: KinematicModel) -> Dict[str, float]:
    return {
        "joints": float(len(model.joints)),
        "digits": float(_digit_count(model)),
        "palm_bodies": float(_palm_body_count(model)),
    }


_REGIME_RNG_TAG = {"boundary": 0, "stationary": 1}


def e12_drift_seed(seed: int, walk_length: int = 200, regime: str = "boundary") -> Dict[str, Any]:
    """``regime``: ``"boundary"`` (``starts.build_start`` -- reduced,
    near-minimal-size starts) or ``"stationary"`` (plain ``generate(seed,
    dist)``, an unreduced sample from the grammar's own distribution) --
    grammar 0.5, I18 fix 4. Each walk step draws one operator from the
    pair/pool (``vary_tracked``, same algorithm as ``vary(..., operators=
    ops)`` but also reports which operator was drawn and whether it was
    applicable) so ``attempt_counts``/``applic_counts`` can report each
    operator's own applicability rate over the walk."""
    if regime not in _REGIME_RNG_TAG:
        raise ValueError(f"unknown drift regime {regime!r}")
    out: Dict[str, Any] = {}
    for dist_name, dist in DIST_VARIANTS.items():
        if regime == "boundary":
            start_derivation, _, _, _ = build_start(seed, dist)
        else:
            start_derivation, _ = generate(seed, dist)
        start_model = derive(start_derivation)
        start_metrics = _metrics(start_model)
        dist_out: Dict[str, Any] = {"start": start_metrics}
        for p_idx, (label, ops) in enumerate(PAIR_LABELS):
            rng = np.random.default_rng([seed, 7_000_000, p_idx, _REGIME_RNG_TAG[regime]])
            current = start_derivation
            attempt_counts: Dict[str, int] = {op: 0 for op in ops}
            applic_counts: Dict[str, int] = {op: 0 for op in ops}
            for _ in range(walk_length):
                candidate, op = vary_tracked(current, rng, dist, ops)
                attempt_counts[op] += 1
                if candidate is None:
                    continue
                applic_counts[op] += 1
                current = candidate
            final_metrics = _metrics(derive(current))
            dist_out[label] = {
                "delta_joints": final_metrics["joints"] - start_metrics["joints"],
                "delta_digits": final_metrics["digits"] - start_metrics["digits"],
                "delta_palm_bodies": final_metrics["palm_bodies"] - start_metrics["palm_bodies"],
                "attempt_counts": attempt_counts,
                "applic_counts": applic_counts,
                "applicability_rate": {
                    op: (applic_counts[op] / attempt_counts[op]) if attempt_counts[op] else None
                    for op in ops
                },
            }
        out[dist_name] = dist_out
    return out


def _bootstrap_ci(vals: Sequence[float], n_resamples: int = 2000, seed: int = 0) -> Dict[str, Optional[float]]:
    arr = np.asarray(vals, dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()), "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)), "n": n,
    }


def _aggregate_drift(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per (dist, pair) CIs for the 3 structural-delta metrics, plus (I18
    fix 4) each pair's own operators' pooled applicability rate over the
    walk: total successful ``vary_tracked`` draws / total draws of that
    operator, summed across every seed's row (a pooled rate, not a mean of
    per-seed rates, since a single seed's walk can draw an operator 0
    times)."""
    agg: Dict[str, Any] = {}
    for dist_name in DIST_VARIANTS:
        dist_agg: Dict[str, Any] = {}
        for label, ops in PAIR_LABELS:
            pair_agg: Dict[str, Any] = {}
            for metric in ("delta_joints", "delta_digits", "delta_palm_bodies"):
                vals = [r[dist_name][label][metric] for r in rows if dist_name in r and label in r[dist_name]]
                pair_agg[metric] = _bootstrap_ci(vals)
            op_applic: Dict[str, Any] = {}
            for op in ops:
                relevant = [r[dist_name][label] for r in rows if dist_name in r and label in r[dist_name]]
                total_attempt = sum(r["attempt_counts"].get(op, 0) for r in relevant)
                total_applic = sum(r["applic_counts"].get(op, 0) for r in relevant)
                op_applic[op] = {
                    "attempts": total_attempt, "applicable": total_applic,
                    "rate": (total_applic / total_attempt) if total_attempt else None,
                }
            pair_agg["applicability"] = op_applic
            dist_agg[label] = pair_agg
        agg[dist_name] = dist_agg
    return agg


# --------------------------------------------------------------------------
# (b) Locality bands (E1 machinery on EVOLUTION_OPERATORS)
# --------------------------------------------------------------------------


_SMALL_STEP_MEDIAN_LIMIT_M = 0.005
_STRUCTURAL_P90_LIMIT_M = 0.060


def _locality_bands(seeds: Sequence[int], n_configs: int = 16) -> Dict[str, Any]:
    """Grammar 0.5, I18 fix 5: per-operator criteria (replacing the old
    pooled-IQR comparison). PASS iff every ``SMALL_STEP_OPERATORS`` member's
    median <= ``_SMALL_STEP_MEDIAN_LIMIT_M`` AND every ``STRUCTURAL_OPS``
    member's p90 <= ``_STRUCTURAL_P90_LIMIT_M``; violators are named."""
    per_op_vals: Dict[str, List[float]] = {op: [] for op in EVOLUTION_OPERATORS}
    for seed in seeds:
        row = e1_locality_seed(seed, n_configs=n_configs)
        for op in EVOLUTION_OPERATORS:
            r = row.get(op)
            if r is None or r["applicable"] != 1.0 or r["tip_displacement_m"] is None:
                continue
            per_op_vals[op].append(r["tip_displacement_m"])

    def _stats(vals: List[float]) -> Dict[str, Optional[float]]:
        if not vals:
            return {"median": None, "p90": None, "n": 0}
        a = np.asarray(vals, dtype=float)
        return {"median": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)), "n": len(vals)}

    per_op_stats = {op: _stats(vals) for op, vals in per_op_vals.items()}

    small_violators = [
        {"op": op, "median": per_op_stats[op]["median"], "n": per_op_stats[op]["n"]}
        for op in SMALL_STEP_OPERATORS
        if per_op_stats[op]["median"] is not None and per_op_stats[op]["median"] > _SMALL_STEP_MEDIAN_LIMIT_M
    ]
    structural_violators = [
        {"op": op, "p90": per_op_stats[op]["p90"], "n": per_op_stats[op]["n"]}
        for op in STRUCTURAL_OPS
        if per_op_stats[op]["p90"] is not None and per_op_stats[op]["p90"] > _STRUCTURAL_P90_LIMIT_M
    ]

    return {
        "n_seeds": len(seeds), "per_op": per_op_stats,
        "small_violators": small_violators, "structural_violators": structural_violators,
        "pass_small_median_5mm": bool(len(small_violators) == 0),
        "pass_structural_p90_60mm": bool(len(structural_violators) == 0),
    }


# --------------------------------------------------------------------------
# (c) Reversibility
# --------------------------------------------------------------------------


def _new_uids(parent_steps, child_steps, production: str) -> set:
    parent_uids = {s.params.get("uid") for s in parent_steps if s.production == production}
    child_uids = {s.params.get("uid") for s in child_steps if s.production == production}
    return child_uids - parent_uids


_GROWTH_PRODUCTION = {
    "add_minimal_digit": "Digit", "insert_phalanx": "Phalanx", "add_palm_body": "PalmBody",
    "add_branch_digit": "Digit",
}


def _reversibility_one(seed: int, dist, growth: str, shrink: str, pair_idx: int) -> Optional[bool]:
    """``None`` = not applicable to this parent; ``True``/``False`` =
    applicable, phenotype recovered or not. ``pair_idx`` (NOT a hash of
    ``growth``/``shrink`` -- Python's built-in ``hash()`` on a string is
    randomized per-process unless ``PYTHONHASHSEED`` is fixed, which would
    break reproducibility) distinguishes this pair's own RNG stream from
    every other pair's, for the same ``seed``."""
    derivation, parent = generate(seed, dist)
    rng = np.random.default_rng([seed, 9_000_000, pair_idx])
    parent_hash = phenotype_hash(parent)

    if growth == shrink:  # toggle_palm_joint: force the growth direction (OFF -> ON).
        off_uids = [
            s.params["uid"] for s in derivation.steps
            if s.production == "PalmBody" and not s.params["has_joint"]
        ]
        if not off_uids:
            return None
        target = off_uids[int(rng.integers(0, len(off_uids)))]
        child = apply_operator(derivation, rng, dist, growth, target=target)
        if child is None:
            return None
        grandchild = apply_operator(child, rng, dist, shrink, target=target)
        if grandchild is None:
            return False
        return phenotype_hash(derive(grandchild)) == parent_hash

    child = apply_operator(derivation, rng, dist, growth)
    if child is None:
        return None
    production = _GROWTH_PRODUCTION[growth]
    new_uids = _new_uids(derivation.steps, child.steps, production)
    if len(new_uids) != 1:
        return False
    target = next(iter(new_uids))
    grandchild = apply_operator(child, rng, dist, shrink, target=target)
    if grandchild is None:
        return False
    return phenotype_hash(derive(grandchild)) == parent_hash


def _reversibility_rates(n_parents: int = 500, seed0: int = 20_000) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for dist_name, dist in DIST_VARIANTS.items():
        dist_out: Dict[str, Any] = {}
        for pair_idx, (growth, shrink) in enumerate(EVOLUTION_PAIRS):
            label = f"{growth}/{shrink}" if growth != shrink else growth
            n_applicable = 0
            n_ok = 0
            for i in range(n_parents):
                seed = seed0 + i
                result = _reversibility_one(seed, dist, growth, shrink, pair_idx)
                if result is None:
                    continue
                n_applicable += 1
                if result:
                    n_ok += 1
            rate = (n_ok / n_applicable) if n_applicable else None
            dist_out[label] = {
                "n_parents": n_parents, "n_applicable": n_applicable, "n_recovered": n_ok,
                "rate": rate, "pass": bool(rate is not None and rate >= 0.95),
            }
        out[dist_name] = dist_out
    return out


# --------------------------------------------------------------------------
# (d) Prior size histograms
# --------------------------------------------------------------------------


def _histogram(vals: Sequence[float]) -> Dict[str, Any]:
    a = np.asarray(vals, dtype=float)
    n = len(a)
    counts: Dict[str, int] = {}
    for v in a:
        key = str(int(v))
        counts[key] = counts.get(key, 0) + 1
    vmin, vmax = float(a.min()), float(a.max())
    frac_at_min = counts.get(str(int(vmin)), 0) / n
    frac_at_max = counts.get(str(int(vmax)), 0) / n
    return {
        "counts": counts, "n": n, "min": vmin, "max": vmax,
        "fraction_at_min": frac_at_min, "fraction_at_max": frac_at_max,
        "pass_no_bound_over_20pct": bool(frac_at_min <= 0.20 and frac_at_max <= 0.20),
    }


def _prior_histograms(n: int = 2000, seed0: int = 30_000) -> Dict[str, Any]:
    dists = {"G_FULL_INS": G_FULL_INS, "G_NOBRANCH_INS": G_NOBRANCH_INS, "G_BEND": G_BEND}
    out: Dict[str, Any] = {}
    for dist_name, dist in dists.items():
        joints: List[float] = []
        digits: List[float] = []
        motors: List[float] = []
        for i in range(n):
            derivation, model = generate(seed0 + i, dist)
            joints.append(len(model.joints))
            digits.append(_digit_count(model))
            motors.append(len(independent_joints(model)))
        out[dist_name] = {
            "joints": _histogram(joints), "digits": _histogram(digits), "motors": _histogram(motors),
        }
    return out


# --------------------------------------------------------------------------
# (e) Redundancy
# --------------------------------------------------------------------------


def _redundancy(n: int = 5000, seed0: int = 0, dist=G_FULL) -> Dict[str, Any]:
    hashes = []
    for i in range(n):
        _, model = generate(seed0 + i, dist)
        hashes.append(phenotype_hash(model))
    distinct = len(set(hashes))
    return {
        "n": n, "n_distinct": distinct, "distinct_fraction": distinct / n,
        "pass_distinct_fraction_99pct": bool(distinct / n >= 0.99),
    }


# --------------------------------------------------------------------------
# run()
# --------------------------------------------------------------------------


def _fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.4g}"


def _summary_md(result: Dict[str, Any]) -> str:
    lines = ["# Experiment: e12_balance", "", f"Wall time: {result['wall_time_s']:.3f} s", "",
             "## Params", "", "```json", json.dumps(result["params"], indent=2, sort_keys=True), "```", ""]

    def _applicability_table(regime_drift: Dict[str, Any], dist_name: str) -> List[str]:
        out_lines = ["| pair | operator | attempts | applicable | rate |", "|---|---|---|---|---|"]
        for label, ops in PAIR_LABELS:
            if label == "EVOLUTION_pool":
                continue  # per-PAIR applicability only -- see module docstring.
            applic = regime_drift[dist_name][label]["applicability"]
            for op in ops:
                a = applic[op]
                out_lines.append(
                    f"| {label} | {op} | {a['attempts']} | {a['applicable']} | {_fmt(a['rate'])} |"
                )
        out_lines.append("")
        return out_lines

    lines += ["## (a) Per-pair neutral drift", ""]

    lines += ["### Boundary regime (reduced minimum-size starts; drift expected positive)", ""]
    boundary = result["drift"]["boundary"]
    for dist_name in DIST_VARIANTS:
        lines += [f"#### dist={dist_name}", "",
                  "| pair | metric | mean | 95% CI lo | 95% CI hi | n | PASS |", "|---|---|---|---|---|---|---|"]
        for label, _ops in PAIR_LABELS:
            for metric in ("delta_joints", "delta_digits", "delta_palm_bodies"):
                b = boundary[dist_name][label][metric]
                # "PASS" here means "drift is >= 0, as expected at this reduced
                # boundary start" -- informational, does not gate (a) overall.
                ok = b["mean"] is None or b["mean"] >= 0.0
                lines.append(
                    f"| {label} | {metric} | {_fmt(b['mean'])} | {_fmt(b['ci_lo'])} | {_fmt(b['ci_hi'])} | "
                    f"{b['n']} | {'PASS' if ok else 'FAIL'} |"
                )
        lines.append("")
        lines += _applicability_table(boundary, dist_name)
    lines.append(
        "**PASS/FAIL boundary regime: informational only (drift here is expected positive by construction "
        "-- reduced starts have a lopsided growth/shrink applicability ratio); it does not gate (a) overall.**"
    )
    lines.append("")

    lines += ["### Stationary regime (random samples from the distribution; 95% CI must include 0)", ""]
    stationary = result["drift"]["stationary"]
    all_drift_pass = True
    for dist_name in DIST_VARIANTS:
        lines += [f"#### dist={dist_name}", "",
                  "| pair | metric | mean | 95% CI lo | 95% CI hi | n | PASS |", "|---|---|---|---|---|---|---|"]
        for label, _ops in PAIR_LABELS:
            for metric in ("delta_joints", "delta_digits", "delta_palm_bodies"):
                b = stationary[dist_name][label][metric]
                ok = b["ci_lo"] is not None and b["ci_lo"] <= 0.0 <= b["ci_hi"]
                all_drift_pass = all_drift_pass and ok
                lines.append(
                    f"| {label} | {metric} | {_fmt(b['mean'])} | {_fmt(b['ci_lo'])} | {_fmt(b['ci_hi'])} | "
                    f"{b['n']} | {'PASS' if ok else 'FAIL'} |"
                )
        lines.append("")
        lines += _applicability_table(stationary, dist_name)
    lines.append(f"**PASS/FAIL (a) per-pair neutral drift (stationary regime): {'PASS' if all_drift_pass else 'FAIL'}**")
    lines.append("")

    lb = result["locality"]
    lines += [
        "## (b) Locality bands (per-operator, EVOLUTION_OPERATORS pool, G_FULL)", "",
        f"n_seeds: {lb['n_seeds']}", "",
        f"criteria: every small-step operator's median tip displacement <= {_SMALL_STEP_MEDIAN_LIMIT_M} m; "
        f"every structural operator's p90 <= {_STRUCTURAL_P90_LIMIT_M} m.", "",
        "| operator | kind | median (m) | p90 (m) | n | PASS |", "|---|---|---|---|---|---|",
    ]
    for op in EVOLUTION_OPERATORS:
        kind = "small-step" if op in SMALL_STEP_OPERATORS else "structural"
        st = lb["per_op"][op]
        if kind == "small-step":
            ok = st["median"] is None or st["median"] <= _SMALL_STEP_MEDIAN_LIMIT_M
        else:
            ok = st["p90"] is None or st["p90"] <= _STRUCTURAL_P90_LIMIT_M
        lines.append(
            f"| {op} | {kind} | {_fmt(st['median'])} | {_fmt(st['p90'])} | {st['n']} | {'PASS' if ok else 'FAIL'} |"
        )
    lines.append("")
    if lb["small_violators"]:
        lines.append(
            "small-step violators (median > "
            f"{_SMALL_STEP_MEDIAN_LIMIT_M} m): "
            + ", ".join(f"{v['op']} ({_fmt(v['median'])} m, n={v['n']})" for v in lb["small_violators"])
        )
    else:
        lines.append("small-step violators: none")
    if lb["structural_violators"]:
        lines.append(
            "structural violators (p90 > "
            f"{_STRUCTURAL_P90_LIMIT_M} m): "
            + ", ".join(f"{v['op']} ({_fmt(v['p90'])} m, n={v['n']})" for v in lb["structural_violators"])
        )
    else:
        lines.append("structural violators: none")
    lines.append("")
    locality_pass = lb["pass_small_median_5mm"] and lb["pass_structural_p90_60mm"]
    lines.append(f"**PASS/FAIL (b) locality bands: {'PASS' if locality_pass else 'FAIL'}**")
    lines.append("")

    lines += ["## (c) Reversibility (>= 95% of applicable cases)", ""]
    all_rev_pass = True
    for dist_name in DIST_VARIANTS:
        lines += ["| pair | n_parents | n_applicable | n_recovered | rate | PASS |",
                  "|---|---|---|---|---|---|"]
        for label, r in result["reversibility"][dist_name].items():
            all_rev_pass = all_rev_pass and r["pass"]
            lines.append(
                f"| {dist_name}/{label} | {r['n_parents']} | {r['n_applicable']} | {r['n_recovered']} | "
                f"{_fmt(r['rate'])} | {'PASS' if r['pass'] else 'FAIL'} |"
            )
        lines.append("")
    lines.append(f"**PASS/FAIL (c) reversibility: {'PASS' if all_rev_pass else 'FAIL'}**")
    lines.append("")

    lines += ["## (d) Prior size histograms (no bound > 20% of mass)", ""]
    all_hist_pass = True
    for dist_name, dist_hist in result["histograms"].items():
        lines += [f"### dist={dist_name}", "",
                  "| metric | min | max | frac_at_min | frac_at_max | PASS |", "|---|---|---|---|---|---|"]
        for metric, h in dist_hist.items():
            all_hist_pass = all_hist_pass and h["pass_no_bound_over_20pct"]
            lines.append(
                f"| {metric} | {_fmt(h['min'])} | {_fmt(h['max'])} | {_fmt(h['fraction_at_min'])} | "
                f"{_fmt(h['fraction_at_max'])} | {'PASS' if h['pass_no_bound_over_20pct'] else 'FAIL'} |"
            )
        lines.append("")
    lines.append(f"**PASS/FAIL (d) prior size histograms: {'PASS' if all_hist_pass else 'FAIL'}**")
    lines.append("")

    red = result["redundancy"]
    lines += [
        "## (e) Redundancy (G_FULL, distinct-hash fraction >= 99%)", "",
        f"n={red['n']} n_distinct={red['n_distinct']} distinct_fraction={_fmt(red['distinct_fraction'])}", "",
        f"**PASS/FAIL (e) redundancy: {'PASS' if red['pass_distinct_fraction_99pct'] else 'FAIL'}**", "",
    ]

    lines += [
        "## Overall", "",
        f"- (a) drift: {'PASS' if all_drift_pass else 'FAIL'}",
        f"- (b) locality: {'PASS' if locality_pass else 'FAIL'}",
        f"- (c) reversibility: {'PASS' if all_rev_pass else 'FAIL'}",
        f"- (d) histograms: {'PASS' if all_hist_pass else 'FAIL'}",
        f"- (e) redundancy: {'PASS' if red['pass_distinct_fraction_99pct'] else 'FAIL'}",
        "",
    ]
    return "\n".join(lines)


def run(out_dir: Optional[str] = None, drift_seeds: int = 128, drift_walk_length: int = 200,
        locality_seeds: int = 200, reversibility_n_parents: int = 500, histogram_n: int = 2000,
        redundancy_n: int = 5000, allow_dirty: bool = False) -> Dict[str, Any]:
    if out_dir is None:
        out_dir = "project-notes/grammar/experiments/E12_balance"
    repo_dir = Path(__file__).resolve().parents[3]
    git_info = _git_info(repo_dir)
    if git_info.get("dirty") and not allow_dirty:
        raise RuntimeError(
            "e12_balance.run: working tree is dirty; pass allow_dirty=True to record a "
            "git_diff_sha256 provenance hash alongside the (stale) git_sha."
        )
    git_diff_sha256 = _git_diff_sha256(repo_dir) if (git_info.get("dirty") and allow_dirty) else None

    params = {
        "drift_seeds": drift_seeds, "drift_walk_length": drift_walk_length,
        "locality_seeds": locality_seeds, "reversibility_n_parents": reversibility_n_parents,
        "histogram_n": histogram_n, "redundancy_n": redundancy_n,
    }
    t0 = time.time()

    drift_rows_boundary = [
        e12_drift_seed(seed, drift_walk_length, regime="boundary") for seed in range(drift_seeds)
    ]
    drift_rows_stationary = [
        e12_drift_seed(seed, drift_walk_length, regime="stationary") for seed in range(drift_seeds)
    ]
    drift = {
        "boundary": _aggregate_drift(drift_rows_boundary),
        "stationary": _aggregate_drift(drift_rows_stationary),
    }

    locality = _locality_bands(range(locality_seeds))
    reversibility = _reversibility_rates(n_parents=reversibility_n_parents)
    histograms = _prior_histograms(n=histogram_n)
    redundancy = _redundancy(n=redundancy_n)

    wall_time_s = time.time() - t0
    result: Dict[str, Any] = {
        "name": "e12_balance", "params": params, "drift": drift, "locality": locality,
        "reversibility": reversibility, "histograms": histograms, "redundancy": redundancy,
        "git_sha": git_info["sha"], "git_dirty": git_info["dirty"], "git_diff_sha256": git_diff_sha256,
        "wall_time_s": wall_time_s,
    }
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    (out_path / "summary.md").write_text(_summary_md(result))
    return result


if __name__ == "__main__":
    run()
