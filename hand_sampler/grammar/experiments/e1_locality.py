"""E1: locality of ``vary``'s mutation operators.

Hypothesis under test: the seven default operators (``derive.OPERATORS``)
are coarse-grained -- a prior review measured ``regrow_subtree`` changing
about 29% of a hand's joints in one application -- while the small-step
operators (``derive.SMALL_STEP_OPERATORS``) are local (each edits exactly
one field of one existing derivation step).

For every parent seed (``generate(seed, G_FULL)``) and every one of the 12
operators (``OPERATORS + SMALL_STEP_OPERATORS``), this module applies
``vary(derivation, rng, G_FULL, operator=op)`` once, with ``rng`` seeded
deterministically from ``(seed, operator_index)``, and records:

- ``applicable``: 0.0/1.0 -- whether ``vary`` found a valid application
  (``VariationImpossible`` is caught, not raised).
- ``null``: 1.0 if the child's ``phenotype_hash`` equals the parent's
  (an accepted mutation that nonetheless produced an identical phenotype --
  e.g. ``step_axis`` stepping to a value the model treats as equivalent),
  else 0.0; ``None`` when not applicable.
- ``tip_displacement_m``, ``joint_count_delta``, ``motor_delta``,
  ``total_length_delta_m``: from ``phenodist.phenotype_distance``.
- ``fraction_joints_changed``: ``1 - |common joints| / max(n_joints)``,
  where "common" is computed on ``canonical_form`` joints (type + rounded
  origin/axis/limits) as a multiset intersection -- robust to any body/joint
  renaming between parent and child (see ``canonical.py``).

``run()`` drives the multiprocess sweep via ``experiments.runner.run_experiment``
and then replaces its generic per-key aggregate with a per-operator table
(applicability/null rates, median/p90 of continuous metrics), written back
into the same ``result.json``/``summary.md``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..canonical import canonical_form, phenotype_hash
from ..derive import OPERATORS, SMALL_STEP_OPERATORS, VariationImpossible, derive, generate, joint_identity, vary
from ..kinematics import KinematicModel
from ..phenodist import phenotype_distance
from ..proxy import tip_frames
from ..variants import G_FULL
from .runner import register, run_experiment

ALL_OPERATORS: tuple = tuple(OPERATORS) + tuple(SMALL_STEP_OPERATORS)

_CONTINUOUS_KEYS = (
    "tip_displacement_m", "joint_count_delta", "fraction_joints_changed",
    "motor_delta", "total_length_delta_m",
)


def _round_tuple(xs, ndigits: int = 9):
    return tuple(round(float(v), ndigits) for v in xs)


def _joint_key(j) -> tuple:
    return (
        j.type,
        _round_tuple(j.origin.xyz),
        _round_tuple(j.origin.rpy),
        _round_tuple(j.axis),
        _round_tuple(j.limits) if j.limits is not None else None,
    )


def _joint_multiset(canon_model: KinematicModel) -> Counter:
    return Counter(_joint_key(j) for j in canon_model.joints)


def fraction_joints_changed(canon_parent: KinematicModel, canon_child: KinematicModel) -> float:
    """``1 - |common joints (by type/origin/axis/limits)| / max(n_joints)``,
    on already-canonicalized models (so this is robust to renaming)."""
    n_p, n_c = len(canon_parent.joints), len(canon_child.joints)
    if max(n_p, n_c) == 0:
        return 0.0
    common = sum((_joint_multiset(canon_parent) & _joint_multiset(canon_child)).values())
    return 1.0 - common / max(n_p, n_c)


_FOOTNOTE_OPERATOR = "insert_phalanx"  # adds a joint -- the case the I14 alignment fix actually changes.


def _legacy_unaligned_tip_displacement(a: KinematicModel, b: KinematicModel, seed: int, n_configs: int = 32) -> float:
    """I14 footnote helper ONLY (fix 9): reproduces the PRE-I14 (unaligned)
    ``phenotype_distance`` sampling scheme -- draw ``a`` and ``b``'s
    u-configurations INDEPENDENTLY (two separate ``sample_configurations``
    calls, same ``seed``) -- purely so the summary artifact can show, for
    one operator, how much the I14 alignment fix (see ``phenodist.py``)
    changed this number. Never used for any other purpose (the real
    per-operator table below always uses the current, aligned
    ``phenotype_distance``)."""
    from ..coords import sample_configurations
    from ..phenodist import _per_config_mean_distance, _root_length_m, _tip_positions_at

    frames_a = tip_frames(a)
    frames_b = tip_frames(b)
    configs_a = sample_configurations(a, n_configs, seed)
    configs_b = sample_configurations(b, n_configs, seed)
    n_common = min(len(configs_a), len(configs_b))
    if n_common == 0:
        return 0.0
    penalty = (_root_length_m(a) + _root_length_m(b)) / 2.0
    scores = []
    for k in range(n_common):
        pos_a = _tip_positions_at(a, frames_a, configs_a[k])
        pos_b = _tip_positions_at(b, frames_b, configs_b[k])
        scores.append(_per_config_mean_distance(pos_a, pos_b, penalty))
    return float(np.mean(scores))


def _footnote_legacy_vs_aligned_median(seeds: Sequence[int], n_configs: int, sample_size: int = 30) -> Dict[str, Any]:
    """Median (over up to ``sample_size`` of ``seeds``) tip_displacement_m
    for ``_FOOTNOTE_OPERATOR`` under BOTH the current aligned
    ``phenotype_distance`` and the legacy unaligned scheme above -- the
    "footnote" required by fix 9."""
    op_idx = ALL_OPERATORS.index(_FOOTNOTE_OPERATOR)
    aligned_vals: List[float] = []
    legacy_vals: List[float] = []
    for seed in list(seeds)[:sample_size]:
        derivation, parent = generate(seed, G_FULL)
        rng = np.random.default_rng([seed, op_idx])
        try:
            child_derivation = vary(derivation, rng, G_FULL, operator=_FOOTNOTE_OPERATOR)
        except VariationImpossible:
            continue
        child = derive(child_derivation)
        id_p, id_c = joint_identity(derivation), joint_identity(child_derivation)
        aligned_vals.append(
            phenotype_distance(parent, child, seed, n_configs=n_configs,
                                identity_a=id_p, identity_b=id_c)["tip_displacement_m"]
        )
        legacy_vals.append(_legacy_unaligned_tip_displacement(parent, child, seed, n_configs=n_configs))
    return {
        "operator": _FOOTNOTE_OPERATOR,
        "n": len(aligned_vals),
        "aligned_median": float(np.median(aligned_vals)) if aligned_vals else None,
        "legacy_unaligned_median": float(np.median(legacy_vals)) if legacy_vals else None,
    }


def e1_locality_seed(seed: int, n_configs: int = 32) -> Dict[str, Any]:
    """Apply every one of the 12 operators once to ``generate(seed, G_FULL)``
    and return a ``{operator_name: metrics_dict}`` mapping (JSON-serializable,
    deterministic given ``seed``)."""
    derivation, parent = generate(seed, G_FULL)
    canon_parent = canonical_form(parent)
    parent_hash = phenotype_hash(parent)
    identity_parent = joint_identity(derivation)  # I15 fix 8: uid-based alignment (see derive.joint_identity).
    out: Dict[str, Any] = {}
    for i, op in enumerate(ALL_OPERATORS):
        rng = np.random.default_rng([seed, i])
        try:
            child_derivation = vary(derivation, rng, G_FULL, operator=op)
        except VariationImpossible:
            out[op] = {
                "applicable": 0.0, "null": None,
                "tip_displacement_m": None, "joint_count_delta": None,
                "fraction_joints_changed": None, "motor_delta": None, "total_length_delta_m": None,
            }
            continue
        child = derive(child_derivation)
        identity_child = joint_identity(child_derivation)
        dist = phenotype_distance(parent, child, seed, n_configs=n_configs,
                                   identity_a=identity_parent, identity_b=identity_child)
        canon_child = canonical_form(child)
        child_hash = phenotype_hash(child)
        out[op] = {
            "applicable": 1.0,
            "null": 1.0 if child_hash == parent_hash else 0.0,
            "tip_displacement_m": dist["tip_displacement_m"],
            "joint_count_delta": dist["joint_count_delta"],
            "fraction_joints_changed": fraction_joints_changed(canon_parent, canon_child),
            "motor_delta": dist["motor_delta"],
            "total_length_delta_m": dist["total_length_delta_m"],
        }
    return out


register("e1_locality", e1_locality_seed)


def _percentile(vals: List[float], p: float) -> float:
    return float(np.percentile(np.asarray(vals, dtype=float), p))


def _aggregate_by_operator(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r["result"] for r in per_seed if r["ok"]]
    n_seeds = len(ok)
    agg: Dict[str, Any] = {"n_seeds": n_seeds}
    for op in ALL_OPERATORS:
        rows = [r[op] for r in ok if op in r]
        n_applicable = sum(1 for r in rows if r["applicable"] == 1.0)
        applicable_rows = [r for r in rows if r["applicable"] == 1.0]
        null_rows = [r for r in applicable_rows if r["null"] == 1.0]
        op_agg: Dict[str, Any] = {
            "n_seeds": len(rows),
            "applicability_rate": (n_applicable / len(rows)) if rows else 0.0,
            # null_rate is a fraction of APPLICABLE mutations, not of all seeds.
            "null_rate": (len(null_rows) / n_applicable) if n_applicable else 0.0,
        }
        for key in _CONTINUOUS_KEYS:
            vals = [r[key] for r in applicable_rows if r[key] is not None]
            if not vals:
                op_agg[key] = {"median": None, "p90": None, "mean": None, "n": 0}
                continue
            op_agg[key] = {
                "median": _percentile(vals, 50), "p90": _percentile(vals, 90),
                "mean": float(np.mean(vals)), "n": len(vals),
            }
        agg[op] = op_agg
    return agg


def _fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.4g}"


def _summary_md(params: Dict[str, Any], aggregate: Dict[str, Any], wall_time_s: float,
                 footnote: Optional[Dict[str, Any]] = None) -> str:
    lines = [
        "# Experiment: e1_locality", "",
        f"Wall time: {wall_time_s:.3f} s", "",
        "## Params", "", "```json", json.dumps(params, indent=2, sort_keys=True, default=str), "```", "",
        f"n_seeds: {aggregate.get('n_seeds')}", "",
        "## Table", "",
        "| operator | applicability_rate | null_rate | tip_disp median | tip_disp p90 | "
        "|joint_count_delta| mean | frac_joints_changed median | frac_joints_changed p90 | "
        "motor_delta mean | total_length_delta median |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for op in ALL_OPERATORS:
        a = aggregate[op]
        lines.append(
            f"| {op} | {_fmt(a['applicability_rate'])} | {_fmt(a['null_rate'])} | "
            f"{_fmt(a['tip_displacement_m']['median'])} | {_fmt(a['tip_displacement_m']['p90'])} | "
            f"{_fmt(a['joint_count_delta']['mean'])} | "
            f"{_fmt(a['fraction_joints_changed']['median'])} | {_fmt(a['fraction_joints_changed']['p90'])} | "
            f"{_fmt(a['motor_delta']['mean'])} | {_fmt(a['total_length_delta_m']['median'])} |"
        )
    lines += ["", "## Reading", ""]
    for op in ALL_OPERATORS:
        a = aggregate[op]
        lines.append(
            f"- {op}: applicable {_fmt(a['applicability_rate'])}, null {_fmt(a['null_rate'])}, "
            f"fraction_joints_changed median {_fmt(a['fraction_joints_changed']['median'])} "
            f"(p90 {_fmt(a['fraction_joints_changed']['p90'])}), "
            f"tip_displacement_m median {_fmt(a['tip_displacement_m']['median'])} "
            f"(p90 {_fmt(a['tip_displacement_m']['p90'])})."
        )
    if footnote is not None:
        lines += ["", "## Footnote: aligned vs. legacy (pre-I14) tip_displacement_m", ""]
        lines.append(
            f"- {footnote['operator']} (n={footnote['n']} seeds): aligned (I14) median "
            f"{_fmt(footnote['aligned_median'])} m; legacy unaligned (pre-I14, independently-sampled "
            f"parent/child configs -- see ``_legacy_unaligned_tip_displacement``) median "
            f"{_fmt(footnote['legacy_unaligned_median'])} m."
        )
    lines.append("")
    return "\n".join(lines)


def run(out_dir: Optional[str] = None, seeds: Sequence[int] = range(1000),
        n_configs: int = 32, processes: int = 24, allow_dirty: bool = False) -> Dict[str, Any]:
    """Run ``e1_locality_seed`` over ``seeds`` via the multiprocess runner,
    then overwrite ``result.json``/``summary.md`` (still in ``out_dir``,
    default ``project-notes/grammar/experiments/E1_locality``) with a
    per-operator aggregate table in place of the runner's generic per-key
    aggregate (which does not understand this experiment's nested,
    per-operator result shape)."""
    if out_dir is None:
        out_dir = "project-notes/grammar/experiments/E1_locality"
    result = run_experiment(
        "e1_locality", e1_locality_seed, params={"n_configs": n_configs},
        seeds=seeds, out_dir=out_dir, processes=processes, allow_dirty=allow_dirty,
    )
    aggregate = _aggregate_by_operator(result["per_seed"])
    result["aggregate"] = aggregate
    footnote = _footnote_legacy_vs_aligned_median(seeds, n_configs)
    result["footnote_legacy_vs_aligned"] = footnote
    out_path = Path(out_dir)
    (out_path / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    (out_path / "summary.md").write_text(_summary_md(result["params"], aggregate, result["wall_time_s"], footnote))
    return result


if __name__ == "__main__":
    run()
