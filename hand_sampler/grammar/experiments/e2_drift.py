"""E2: neutral drift under different operator mixtures and grammar variants.

Hypothesis under test: neutral mutation (no selection -- every accepted
``vary`` step is kept regardless of its effect) with the default operator
mixture drifts toward more joints/digits, the way the old (pre-grammar)
sampler did; a mixture that balances growth and shrink operators, or one
weighted toward the small-step operators, drifts less; whether/how much it
drifts also depends on which grammar variant (``Distribution``) parameterizes
sampling.

Start: for each seed, ``starts.build_start(seed, dist)`` -- ``generate(seed,
dist)`` reduced (via ``remove_digit``/``delete_phalanx``) until digit_count
<= 2 and every digit's phalanx_count <= 2. I14 fix 8: this reduced start is
now built ONCE PER (seed, dist variant) pair and reused across every mixture
under that dist variant ("same starts across mixtures" -- the precondition
for the paired bootstrap below).

Mixtures (see ``MIXTURES``; I14 fix 8, replacing the previous ``uniform``/
``balanced``/``small_heavy`` trio):

- ``DEFAULT_uniform``: equal weight on the 7 default ``OPERATORS``.
- ``UNION_uniform``: equal weight on ``UNION_OPERATORS`` (the 7 default
  operators, the 6 ``SMALL_STEP_OPERATORS`` -- ``step_length`` dropped, see
  ``derive.py``'s I14 fix 5 -- and the 5 ``MINIMAL_STRUCTURAL_OPERATORS``;
  18 operators total).
- ``UNION_weighted``: 55% of the weight split equally over the 6 small-step
  operators; 25% split equally over the "minimal structural" group (the 5
  ``MINIMAL_STRUCTURAL_OPERATORS`` PLUS ``insert_phalanx``/``delete_phalanx``,
  7 operators); 10% on ``resample_parameter`` alone; 10% split equally over
  the remaining 4 "coarse" default operators (``perturb_parameter``,
  ``regrow_subtree``, ``add_digit``, ``remove_digit``).

Grammar variants (``DIST_VARIANTS``; I14 fix 8): ``G_FULL``, ``G_NOBRANCH``,
``G_FULL_INS``, ``G_NOBRANCH_INS`` (see ``variants.py`` -- the ``_INS``
pair route every growth operator's new material through a small
``phalanx_count_range=(1, 3)``, no-branching insertion distribution).

For each (seed, dist variant, mixture), one continuous walk of ``walk_length``
(default 400, I14 fix 8) steps is run from that (seed, dist)'s shared start
(``vary(current, rng, dist, operator=drawn_op)``; a ``VariationImpossible``
proposal keeps the parent and is counted as a rejection for that operator).
Structural metrics (joints, digits, palm bodies, motors, total_length_m) are
recorded at step 0, every step through ``record_through`` (default 40), then
every ``record_every_after`` (default 10) steps through ``walk_length``;
``phenotype_hash`` is recorded at steps ``record_through`` and ``walk_length``
for cross-seed distinctness.

Aggregation uses a PAIRED bootstrap (I14 fix 8): one shared resample index
array (over the seed axis) is drawn once and reused for every (dist,
mixture, metric) combination of the same seed count, rather than an
independent draw per combination -- since every mixture (for a given dist)
shares the exact same per-seed starts, this correctly propagates that
correlation into any CI comparison across mixtures, rather than treating
each mixture's seeds as if drawn afresh.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..canonical import phenotype_hash
from ..coords import independent_joints
from ..derive import MINIMAL_STRUCTURAL_OPERATORS, OPERATORS, SMALL_STEP_OPERATORS, VariationImpossible, derive, vary
from ..kinematics import KinematicModel
from ..phenodist import _digit_count, _palm_body_count, _total_length_m
from ..variants import G_FULL, G_FULL_INS, G_NOBRANCH, G_NOBRANCH_INS
# I14 fix 11: ``build_start`` moved to ``starts.py`` (see that module's own
# docstring for why) -- re-exported here so every existing caller of
# ``e2_drift.build_start`` (this module's own ``e2_drift_seed``, tests, and
# the unfinished ``e5_evolve.py``) keeps working unchanged. Imported BEFORE
# ``.runner`` (which, at ITS own module level, imports this module and --
# guarded -- ``e5_evolve``, which in turn imports ``build_start`` FROM this
# module) so that if this module is ever the first one touched, its own
# ``build_start`` attribute already exists by the time any of that nested
# re-entry happens, regardless of import order.
from .starts import _digit_count_and_max_phalanx, _hand_step, _top_level_digit_steps, build_start  # noqa: F401
from .runner import register, run_experiment

_RECORD_THROUGH = 40
_RECORD_EVERY_AFTER = 10
_WALK_LENGTH = 400

UNION_OPERATORS: Tuple[str, ...] = tuple(OPERATORS) + tuple(SMALL_STEP_OPERATORS) + tuple(MINIMAL_STRUCTURAL_OPERATORS)
assert len(UNION_OPERATORS) == len(set(UNION_OPERATORS)), "UNION_OPERATORS has duplicates"

# I15 fix 3: ``perturb_parameter`` moved from the 10% "coarse" group into
# the 55% small-step share -- it edits exactly one existing step's length
# by one grid increment (see ``derive._op_perturb_parameter``), which is a
# small-step edit in every respect except that it lives in ``OPERATORS``
# rather than ``SMALL_STEP_OPERATORS``; opus-review-final.md flagged it as
# misplaced in the coarse 10% share. The coarse group now holds only
# ``regrow_subtree``/``add_digit``/``remove_digit``.
_SMALL_GROUP: Tuple[str, ...] = tuple(SMALL_STEP_OPERATORS) + ("perturb_parameter",)
_MINIMAL_GROUP: Tuple[str, ...] = tuple(MINIMAL_STRUCTURAL_OPERATORS) + ("insert_phalanx", "delete_phalanx")
_RESAMPLE_GROUP: Tuple[str, ...] = ("resample_parameter",)
_COARSE_GROUP: Tuple[str, ...] = ("regrow_subtree", "add_digit", "remove_digit")
# ``_SMALL_GROUP`` is SMALL_STEP_OPERATORS (disjoint from OPERATORS/
# MINIMAL_STRUCTURAL_OPERATORS) plus the one relocated ``perturb_parameter``
# (which DOES belong to OPERATORS); the remaining three groups must exactly
# partition OPERATORS + MINIMAL_STRUCTURAL_OPERATORS minus that one operator.
assert sorted(_MINIMAL_GROUP + _RESAMPLE_GROUP + _COARSE_GROUP + ("perturb_parameter",)) == sorted(
    OPERATORS + MINIMAL_STRUCTURAL_OPERATORS
)


def _uniform(ops: Sequence[str]) -> Dict[str, float]:
    w = 1.0 / len(ops)
    return {op: w for op in ops}


def _union_weighted_mixture() -> Dict[str, float]:
    mix: Dict[str, float] = {}
    mix.update({op: 0.55 / len(_SMALL_GROUP) for op in _SMALL_GROUP})
    mix.update({op: 0.25 / len(_MINIMAL_GROUP) for op in _MINIMAL_GROUP})
    mix.update({op: 0.10 / len(_RESAMPLE_GROUP) for op in _RESAMPLE_GROUP})
    mix.update({op: 0.10 / len(_COARSE_GROUP) for op in _COARSE_GROUP})
    return mix


MIXTURES: Dict[str, Dict[str, float]] = {
    "DEFAULT_uniform": _uniform(OPERATORS),
    "UNION_uniform": _uniform(UNION_OPERATORS),
    "UNION_weighted": _union_weighted_mixture(),
}

for _name, _mix in MIXTURES.items():
    assert abs(sum(_mix.values()) - 1.0) < 1e-9, f"mixture {_name!r} does not sum to 1"

DIST_VARIANTS: Dict[str, Any] = {
    "G_FULL": G_FULL, "G_NOBRANCH": G_NOBRANCH, "G_FULL_INS": G_FULL_INS, "G_NOBRANCH_INS": G_NOBRANCH_INS,
}


def _draw_operator(rng: np.random.Generator, mixture: Dict[str, float]) -> str:
    ops = list(mixture.keys())
    probs = np.asarray([mixture[o] for o in ops], dtype=float)
    idx = int(rng.choice(len(ops), p=probs))
    return ops[idx]


def _metrics(model: KinematicModel) -> Dict[str, float]:
    return {
        "joints": float(len(model.joints)),
        "digits": float(_digit_count(model)),
        "palm_bodies": float(_palm_body_count(model)),
        "motors": float(len(independent_joints(model))),
        "total_length_m": float(_total_length_m(model)),
    }


def _record_steps(walk_length: int, record_through: int, record_every_after: int) -> set:
    steps = set(range(0, min(walk_length, record_through) + 1))
    steps.update(range(0, walk_length + 1, record_every_after))
    steps.add(walk_length)
    return steps


def _run_one_walk(start_derivation, start_model, start_metrics, dist, mixture: Dict[str, float],
                   rng: np.random.Generator, walk_length: int, record_through: int,
                   record_steps: Sequence[int]) -> Dict[str, Any]:
    current = start_derivation
    acceptance: Dict[str, List[int]] = {op: [0, 0] for op in mixture}
    trajectory: List[Dict[str, float]] = []
    hash_at: Dict[int, str] = {}

    if 0 in record_steps:
        trajectory.append({"step": 0.0, **start_metrics})
        if 0 == record_through or 0 == walk_length:
            hash_at[0] = phenotype_hash(start_model)

    for step in range(1, walk_length + 1):
        op = _draw_operator(rng, mixture)
        acceptance[op][0] += 1
        try:
            candidate = vary(current, rng, dist, operator=op)
            acceptance[op][1] += 1
        except VariationImpossible:
            candidate = current
        current = candidate
        if step in record_steps:
            model = derive(current)
            m = _metrics(model)
            trajectory.append({"step": float(step), **m})
            if step in (record_through, walk_length):
                hash_at[step] = phenotype_hash(model)

    mixture_result: Dict[str, Any] = {
        "trajectory": trajectory,
        "acceptance": {op: {"attempts": a, "accepted": acc} for op, (a, acc) in acceptance.items()},
        f"hash_at_{record_through}": hash_at.get(record_through),
        f"hash_at_{walk_length}": hash_at.get(walk_length),
    }
    by_step = {row["step"]: row for row in trajectory}
    for check_step in (record_through, walk_length):
        row = by_step.get(float(check_step))
        if row is not None:
            for key in ("joints", "digits", "motors", "total_length_m"):
                mixture_result[f"delta_{key}_at_{check_step}"] = row[key] - start_metrics[key]
                mixture_result[f"final_{key}_at_{check_step}"] = row[key]
    return mixture_result


def e2_drift_seed(seed: int, walk_length: int = _WALK_LENGTH, record_through: int = _RECORD_THROUGH,
                   record_every_after: int = _RECORD_EVERY_AFTER,
                   max_start_digits: int = 2, max_start_phalanges: int = 2) -> Dict[str, Any]:
    """Per dist variant in ``DIST_VARIANTS``: build ONE shared reduced start
    for ``(seed, dist)``, then run one walk per mixture in ``MIXTURES`` from
    that SAME start (I14 fix 8's "same starts across mixtures"). Returns a
    JSON-serializable ``{dist_name: {mixture_name: ...}}`` mapping (plus
    each dist's own start metrics)."""
    record_steps = sorted(_record_steps(walk_length, record_through, record_every_after))
    out: Dict[str, Any] = {}
    for d_idx, (dist_name, dist) in enumerate(DIST_VARIANTS.items()):
        start_derivation, start_digits, start_max_phalanx, start_reduced_fully = build_start(
            seed, dist, max_start_digits, max_start_phalanges,
        )
        start_model = derive(start_derivation)
        start_metrics = _metrics(start_model)
        dist_out: Dict[str, Any] = {
            "start_digit_count": float(start_digits),
            "start_max_phalanx_count": float(start_max_phalanx),
            "start_reduced_fully": 1.0 if start_reduced_fully else 0.0,
            "start": start_metrics,
        }
        for mix_idx, (mixture_name, mixture) in enumerate(MIXTURES.items()):
            rng = np.random.default_rng([seed, d_idx, mix_idx])
            dist_out[mixture_name] = _run_one_walk(
                start_derivation, start_model, start_metrics, dist, mixture, rng,
                walk_length, record_through, record_steps,
            )
        out[dist_name] = dist_out
    return out


register("e2_drift", e2_drift_seed)


def _paired_bootstrap_idx(n_seeds: int, n_resamples: int = 2000, seed: int = 0) -> np.ndarray:
    """One shared (n_resamples, n_seeds) resample-index array, drawn ONCE
    and reused for every (dist, mixture, metric) bootstrap of the same
    ``n_seeds`` -- see module docstring's "paired bootstrap" note."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_seeds, size=(n_resamples, n_seeds))


def _bootstrap_ci_paired(vals: List[float], idx: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    arr = np.asarray(vals, dtype=float)
    if len(arr) == 0 or idx is None:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "n": len(arr)}
    means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)),
        "n": len(arr),
    }


def _diff_ci_paired(vals_a: List[float], vals_b: List[float], idx: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    """95% CI on the per-seed PAIRED difference ``vals_a[i] - vals_b[i]``
    (I15 fix 6) -- ``vals_a``/``vals_b`` must be aligned by seed index (both
    built from iterating the same ``ok`` list in the same order, which every
    caller below does). Uses the SAME shared bootstrap resample index
    array ``idx`` as every marginal CI, so it is directly comparable."""
    n = min(len(vals_a), len(vals_b))
    if n == 0 or idx is None:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0}
    diff = np.asarray(vals_a[:n], dtype=float) - np.asarray(vals_b[:n], dtype=float)
    return _bootstrap_ci_paired(list(diff), idx)


def _aggregate(per_seed: List[Dict[str, Any]], record_through: int, walk_length: int) -> Dict[str, Any]:
    ok = [r["result"] for r in per_seed if r["ok"]]
    n_seeds = len(ok)
    agg: Dict[str, Any] = {"n_seeds": n_seeds}
    idx = _paired_bootstrap_idx(n_seeds) if n_seeds > 0 else None
    for dist_name in DIST_VARIANTS:
        dist_agg: Dict[str, Any] = {}
        start_digits = [r[dist_name]["start_digit_count"] for r in ok if dist_name in r]
        start_phalanx = [r[dist_name]["start_max_phalanx_count"] for r in ok if dist_name in r]
        dist_agg["start_digit_count_mean"] = float(np.mean(start_digits)) if start_digits else None
        dist_agg["start_max_phalanx_count_mean"] = float(np.mean(start_phalanx)) if start_phalanx else None
        for mixture_name in MIXTURES:
            rows = [r[dist_name][mixture_name] for r in ok if dist_name in r and mixture_name in r[dist_name]]
            mix_agg: Dict[str, Any] = {}
            for check_step in (record_through, walk_length):
                label = str(check_step)
                for metric in ("joints", "digits", "motors"):
                    key = f"delta_{metric}_at_{label}"
                    vals = [r[key] for r in rows if key in r]
                    mix_agg[key] = _bootstrap_ci_paired(vals, idx)
                    final_key = f"final_{metric}_at_{label}"
                    final_vals = [r[final_key] for r in rows if final_key in r]
                    mix_agg[final_key] = _bootstrap_ci_paired(final_vals, idx)
                hash_key = f"hash_at_{label}"
                hashes = [r[hash_key] for r in rows if r.get(hash_key) is not None]
                mix_agg[f"distinct_hash_fraction_at_{label}"] = (
                    (len(set(hashes)) / len(hashes)) if hashes else None
                )
            op_totals: Dict[str, List[int]] = {}
            for r in rows:
                for op, d in r["acceptance"].items():
                    totals = op_totals.setdefault(op, [0, 0])
                    totals[0] += d["attempts"]
                    totals[1] += d["accepted"]
            mix_agg["acceptance_rate"] = {
                op: (acc / att if att else None) for op, (att, acc) in op_totals.items()
            }
            dist_agg[mixture_name] = mix_agg

        # I15 fix 6: difference CI, UNION_weighted minus DEFAULT_uniform,
        # paired by the shared per-(seed, dist) start (both mixtures walk
        # from the exact same start -- see module docstring).
        if "UNION_weighted" in MIXTURES and "DEFAULT_uniform" in MIXTURES:
            rows_u = [r[dist_name]["UNION_weighted"] for r in ok if dist_name in r and "UNION_weighted" in r[dist_name]]
            rows_d = [r[dist_name]["DEFAULT_uniform"] for r in ok if dist_name in r and "DEFAULT_uniform" in r[dist_name]]
            diff_agg: Dict[str, Any] = {}
            for check_step in (record_through, walk_length):
                label = str(check_step)
                for metric in ("joints", "digits", "motors"):
                    final_key = f"final_{metric}_at_{label}"
                    vals_u = [r[final_key] for r in rows_u if final_key in r]
                    vals_d = [r[final_key] for r in rows_d if final_key in r]
                    diff_agg[f"diff_final_{metric}_at_{label}"] = _diff_ci_paired(vals_u, vals_d, idx)
            dist_agg["diff_UNION_weighted_minus_DEFAULT_uniform"] = diff_agg
        agg[dist_name] = dist_agg

    # I15 fix 6: difference CI, an "_INS" dist minus its plain counterpart,
    # per mixture ("per pool") -- paired by seed, since ``build_start``'s
    # own reduction depends only on ``seed`` (never on ``dist.insertion``,
    # which only ``vary``'s GROWTH operators read -- see
    # ``derive._growth_dist``), so an ``_INS`` variant and its plain
    # counterpart share byte-identical starts for the same seed already.
    ins_pairs = [
        (ins, plain) for ins, plain in (("G_FULL_INS", "G_FULL"), ("G_NOBRANCH_INS", "G_NOBRANCH"))
        if ins in DIST_VARIANTS and plain in DIST_VARIANTS
    ]
    diff_ins: Dict[str, Any] = {}
    for ins_name, plain_name in ins_pairs:
        for mixture_name in MIXTURES:
            rows_ins = [r[ins_name][mixture_name] for r in ok if ins_name in r and mixture_name in r[ins_name]]
            rows_plain = [r[plain_name][mixture_name] for r in ok
                          if plain_name in r and mixture_name in r[plain_name]]
            pair_agg: Dict[str, Any] = {}
            for check_step in (record_through, walk_length):
                label = str(check_step)
                for metric in ("joints", "digits", "motors"):
                    final_key = f"final_{metric}_at_{label}"
                    vals_ins = [r[final_key] for r in rows_ins if final_key in r]
                    vals_plain = [r[final_key] for r in rows_plain if final_key in r]
                    pair_agg[f"diff_final_{metric}_at_{label}"] = _diff_ci_paired(vals_ins, vals_plain, idx)
            diff_ins[f"{ins_name}_minus_{plain_name}__{mixture_name}"] = pair_agg
    agg["diff_ins_minus_plain"] = diff_ins
    return agg


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4g}"


def _summary_md(params: Dict[str, Any], aggregate: Dict[str, Any], wall_time_s: float) -> str:
    lines = [
        "# Experiment: e2_drift", "",
        f"Wall time: {wall_time_s:.3f} s", "",
        "## Params", "", "```json", json.dumps(params, indent=2, sort_keys=True, default=str), "```", "",
        f"n_seeds: {aggregate.get('n_seeds')}", "",
        "Bootstrap CIs are PAIRED across mixtures (same seeds/starts within a "
        "dist variant -- see module docstring).", "",
    ]
    walk_label = str(params["walk_length"])

    lines += ["## Start sizes per dist variant (mean over seeds)", "",
              "| dist | start digit_count mean | start max_phalanx_count mean |", "|---|---|---|"]
    for dist_name in DIST_VARIANTS:
        a = aggregate[dist_name]
        lines.append(
            f"| {dist_name} | {_fmt(a.get('start_digit_count_mean'))} | "
            f"{_fmt(a.get('start_max_phalanx_count_mean'))} |"
        )
    lines.append("")

    lines += [
        f"## Final joints (ABSOLUTE, not a delta) at step {walk_label}, per mixture x dist (paired 95% CI)", "",
        "| dist | mixture | final joints mean | 95% CI lo | 95% CI hi | n |",
        "|---|---|---|---|---|---|",
    ]
    for dist_name in DIST_VARIANTS:
        for mixture_name in MIXTURES:
            b = aggregate[dist_name][mixture_name][f"final_joints_at_{walk_label}"]
            lines.append(
                f"| {dist_name} | {mixture_name} | {_fmt(b['mean'])} | {_fmt(b['ci_lo'])} | "
                f"{_fmt(b['ci_hi'])} | {b['n']} |"
            )
    lines.append("")

    lines += [
        f"## Difference CI: UNION_weighted minus DEFAULT_uniform, final joints/digits/motors "
        f"at step {walk_label} (paired by shared start; I15 fix 6)", "",
        "| dist | metric | mean DIFFERENCE | 95% CI lo | 95% CI hi | n |",
        "|---|---|---|---|---|---|",
    ]
    for dist_name in DIST_VARIANTS:
        d = aggregate[dist_name].get("diff_UNION_weighted_minus_DEFAULT_uniform", {})
        for metric in ("joints", "digits", "motors"):
            b = d.get(f"diff_final_{metric}_at_{walk_label}", {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0})
            lines.append(
                f"| {dist_name} | {metric} | {_fmt(b['mean'])} | {_fmt(b['ci_lo'])} | {_fmt(b['ci_hi'])} | {b['n']} |"
            )
    lines.append("")

    lines += [
        f"## Difference CI: _INS minus plain, per pool (mixture), final joints/digits/motors "
        f"at step {walk_label} (paired by shared start; I15 fix 6)", "",
        "| pair | mixture | metric | mean DIFFERENCE | 95% CI lo | 95% CI hi | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, pair_agg in aggregate.get("diff_ins_minus_plain", {}).items():
        pair_name, mixture_name = key.rsplit("__", 1)
        for metric in ("joints", "digits", "motors"):
            b = pair_agg.get(f"diff_final_{metric}_at_{walk_label}", {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0})  # noqa: E501
            lines.append(
                f"| {pair_name} | {mixture_name} | {metric} | {_fmt(b['mean'])} | {_fmt(b['ci_lo'])} | "
                f"{_fmt(b['ci_hi'])} | {b['n']} |"
            )
    lines.append("")

    labels = [str(params["record_through"]), str(params["walk_length"])]
    for dist_name in DIST_VARIANTS:
        for mixture_name in MIXTURES:
            a = aggregate[dist_name][mixture_name]
            lines += [f"## dist={dist_name} mixture={mixture_name}", ""]
            for label in labels:
                lines += [
                    f"### Step {label}", "",
                    "| metric | mean delta | 95% CI lo | 95% CI hi | n |",
                    "|---|---|---|---|---|",
                ]
                for metric in ("joints", "digits", "motors"):
                    key = f"delta_{metric}_at_{label}"
                    b = a[key]
                    lines.append(
                        f"| {metric} | {_fmt(b['mean'])} | {_fmt(b['ci_lo'])} | {_fmt(b['ci_hi'])} | {b['n']} |"
                    )
                lines += ["", f"distinct_hash_fraction_at_{label}: {_fmt(a[f'distinct_hash_fraction_at_{label}'])}", ""]  # noqa: E501
            lines += ["### Acceptance rate per operator", "", "| operator | acceptance_rate |", "|---|---|"]
            for op, rate in sorted(a["acceptance_rate"].items()):
                lines.append(f"| {op} | {_fmt(rate)} |")
            lines.append("")
    lines += ["## Reading", "",
              "(I15 fix 6: every number below tagged \"Δ\" is a DELTA from that "
              "(seed, dist)'s shared reduced START -- not an absolute joint/digit/motor "
              "count; start sizes are reported separately above.)", ""]
    for dist_name in DIST_VARIANTS:
        da = aggregate[dist_name]
        lines.append(
            f"- {dist_name} start: digit_count mean {_fmt(da.get('start_digit_count_mean'))}, "
            f"max_phalanx_count mean {_fmt(da.get('start_max_phalanx_count_mean'))}."
        )
        for mixture_name in MIXTURES:
            a = da[mixture_name]
            for label in labels:
                parts = []
                for metric in ("joints", "digits", "motors"):
                    b = a[f"delta_{metric}_at_{label}"]
                    parts.append(f"Δ{metric} {_fmt(b['mean'])} [{_fmt(b['ci_lo'])}, {_fmt(b['ci_hi'])}]")
                lines.append(
                    f"- {dist_name}/{mixture_name} step {label}: " + "; ".join(parts) +
                    f"; distinct_hash_fraction {_fmt(a[f'distinct_hash_fraction_at_{label}'])}."
                )
        diff = da.get("diff_UNION_weighted_minus_DEFAULT_uniform", {})
        for metric in ("joints", "digits", "motors"):
            b = diff.get(f"diff_final_{metric}_at_{walk_label}")
            if b is not None:
                lines.append(
                    f"- {dist_name} DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step {walk_label}: "
                    f"Δ{metric} {_fmt(b['mean'])} [{_fmt(b['ci_lo'])}, {_fmt(b['ci_hi'])}] (n={b['n']})."
                )
    for key, pair_agg in aggregate.get("diff_ins_minus_plain", {}).items():
        pair_name, mixture_name = key.rsplit("__", 1)
        for metric in ("joints", "digits", "motors"):
            b = pair_agg.get(f"diff_final_{metric}_at_{walk_label}")
            if b is not None:
                lines.append(
                    f"- DIFFERENCE {pair_name} (mixture={mixture_name}) at step {walk_label}: "
                    f"Δ{metric} {_fmt(b['mean'])} [{_fmt(b['ci_lo'])}, {_fmt(b['ci_hi'])}] (n={b['n']})."
                )
    lines.append("")
    return "\n".join(lines)


def run(out_dir: Optional[str] = None, seeds: Sequence[int] = range(128),
        walk_length: int = _WALK_LENGTH, record_through: int = _RECORD_THROUGH,
        record_every_after: int = _RECORD_EVERY_AFTER, processes: int = 24,
        allow_dirty: bool = False) -> Dict[str, Any]:
    """Run ``e2_drift_seed`` over ``seeds`` via the multiprocess runner, then
    overwrite ``result.json``/``summary.md`` (in ``out_dir``, default
    ``project-notes/grammar/experiments/E2_drift``) with the per-dist x
    per-mixture paired-bootstrap-CI / acceptance-rate aggregate described in
    this module's docstring, in place of the runner's generic per-key
    aggregate."""
    if out_dir is None:
        out_dir = "project-notes/grammar/experiments/E2_drift"
    params = {
        "walk_length": walk_length, "record_through": record_through,
        "record_every_after": record_every_after,
    }
    result = run_experiment(
        "e2_drift", e2_drift_seed, params=params, seeds=seeds, out_dir=out_dir, processes=processes,
        allow_dirty=allow_dirty,
    )
    aggregate = _aggregate(result["per_seed"], record_through, walk_length)
    result["aggregate"] = aggregate
    out_path = Path(out_dir)
    (out_path / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    (out_path / "summary.md").write_text(_summary_md(result["params"], aggregate, result["wall_time_s"]))
    return result


if __name__ == "__main__":
    run()
