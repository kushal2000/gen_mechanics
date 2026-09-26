"""E2: neutral drift under different operator mixtures.

Hypothesis under test: neutral mutation (no selection -- every accepted
``vary`` step is kept regardless of its effect) with the default operator
mixture drifts toward more joints/digits, the way the old (pre-grammar)
sampler did; a mixture that balances growth and shrink operators, or one
weighted toward the small-step operators, drifts less.

Start: for each seed in 0..255, ``generate(seed, G_SERIAL)`` (no branching,
no extra palm bodies, no palm joints -- see ``variants.G_SERIAL``), then
repeatedly ``vary(..., operator="remove_digit")`` while ``digit_count > 2``
and/or ``operator="delete_phalanx"`` while any digit's ``phalanx_count > 2``
(retrying past ``VariationImpossible``, up to ``_MAX_REDUCE_ATTEMPTS``) until
both hold or attempts run out. Digit/phalanx counts are read directly off
the derivation's own ``Hand``/``Digit`` step params (``digit_count``,
``phalanx_count``), not off the derived model -- exact and cheap.

Mixtures (see ``MIXTURES`` below): ``uniform`` (equal weight on the 7
default operators), ``balanced`` (growth ops {add_digit, insert_phalanx}
and shrink ops {remove_digit, delete_phalanx} carry equal total weight;
the remaining weight is split equally among the other 3 default
operators), ``small_heavy`` (70% of the weight split equally over the 5
small-step operators, 30% split equally over the 7 default operators).

For each (seed, mixture), one continuous walk of 500 steps is run from the
shared start (``vary(current, rng, G_SERIAL, operator=drawn_op)``; a
``VariationImpossible`` proposal keeps the parent and is counted as a
rejection for that operator). This single walk is used for BOTH the
"length 40" and "length 500" readings (step 40's state is a prefix of the
500-step walk) -- cheaper than two independent walks and avoids splitting
the rng stream, at the cost of the length-40 reading not being an
independent walk from the length-500 one; this is recorded in ``params``.
Structural metrics (joints, digits, palm bodies, motors, total_length_m)
are recorded at step 0, every step through 40, then every 10 steps through
500 (a superset of "every step for 40, every 10 for 500"); ``phenotype_hash``
is recorded at steps 40 and 500 for cross-seed distinctness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..canonical import phenotype_hash
from ..coords import independent_joints
from ..derive import OPERATORS, SMALL_STEP_OPERATORS, VariationImpossible, derive, generate, vary
from ..kinematics import KinematicModel
from ..phenodist import _digit_count, _palm_body_count, _total_length_m
from ..variants import G_SERIAL
from .runner import register, run_experiment

_MAX_REDUCE_ATTEMPTS = 500
_RECORD_THROUGH = 40
_RECORD_EVERY_AFTER = 10
_WALK_LENGTH = 500

GROWTH_OPS = ("add_digit", "insert_phalanx")
SHRINK_OPS = ("remove_digit", "delete_phalanx")
OTHER_DEFAULT_OPS = tuple(op for op in OPERATORS if op not in GROWTH_OPS and op not in SHRINK_OPS)


def _uniform(ops: Sequence[str]) -> Dict[str, float]:
    w = 1.0 / len(ops)
    return {op: w for op in ops}


def _balanced_mixture() -> Dict[str, float]:
    # Growth total == shrink total == 1/3 each; remaining 1/3 split equally
    # among the other 3 default operators (resample_parameter,
    # perturb_parameter, regrow_subtree).
    growth_total = shrink_total = 1.0 / 3.0
    remainder = 1.0 - growth_total - shrink_total
    mix = {op: growth_total / len(GROWTH_OPS) for op in GROWTH_OPS}
    mix.update({op: shrink_total / len(SHRINK_OPS) for op in SHRINK_OPS})
    mix.update({op: remainder / len(OTHER_DEFAULT_OPS) for op in OTHER_DEFAULT_OPS})
    return mix


def _small_heavy_mixture() -> Dict[str, float]:
    mix = {op: 0.30 / len(OPERATORS) for op in OPERATORS}
    mix.update({op: 0.70 / len(SMALL_STEP_OPERATORS) for op in SMALL_STEP_OPERATORS})
    return mix


MIXTURES: Dict[str, Dict[str, float]] = {
    "uniform": _uniform(OPERATORS),
    "balanced": _balanced_mixture(),
    "small_heavy": _small_heavy_mixture(),
}

for _name, _mix in MIXTURES.items():
    assert abs(sum(_mix.values()) - 1.0) < 1e-9, f"mixture {_name!r} does not sum to 1"


def _draw_operator(rng: np.random.Generator, mixture: Dict[str, float]) -> str:
    ops = list(mixture.keys())
    probs = np.asarray([mixture[o] for o in ops], dtype=float)
    idx = int(rng.choice(len(ops), p=probs))
    return ops[idx]


def _hand_step(derivation):
    return next(s for s in derivation.steps if s.path == "hand")


def _top_level_digit_steps(derivation) -> List:
    return [s for s in derivation.steps if s.production == "Digit" and s.params.get("top_level")]


def _digit_count_and_max_phalanx(derivation) -> Tuple[int, int]:
    digit_count = _hand_step(derivation).params["digit_count"]
    phalanx_counts = [s.params["phalanx_count"] for s in _top_level_digit_steps(derivation)]
    return digit_count, (max(phalanx_counts) if phalanx_counts else 0)


def build_start(seed: int, dist=G_SERIAL, max_digits: int = 2, max_phalanges: int = 2):
    """``generate(seed, dist)`` reduced (via ``remove_digit``/``delete_phalanx``)
    until ``digit_count <= max_digits`` and every digit's ``phalanx_count <=
    max_phalanges``, or ``_MAX_REDUCE_ATTEMPTS`` retries are exhausted.
    Returns ``(derivation, achieved_digit_count, achieved_max_phalanx,
    reduced_fully: bool)``."""
    rng = np.random.default_rng([int(seed), 1_000_000])  # distinguishes this phase from per-mixture rngs (mix_idx 0..2)
    derivation, _ = generate(seed, dist)
    attempts = 0
    while attempts < _MAX_REDUCE_ATTEMPTS:
        digit_count, max_phalanx = _digit_count_and_max_phalanx(derivation)
        if digit_count <= max_digits and max_phalanx <= max_phalanges:
            return derivation, digit_count, max_phalanx, True
        op = "remove_digit" if digit_count > max_digits else "delete_phalanx"
        attempts += 1
        try:
            derivation = vary(derivation, rng, dist, operator=op)
        except VariationImpossible:
            continue
    digit_count, max_phalanx = _digit_count_and_max_phalanx(derivation)
    return derivation, digit_count, max_phalanx, False


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


def e2_drift_seed(seed: int, walk_length: int = _WALK_LENGTH, record_through: int = _RECORD_THROUGH,
                   record_every_after: int = _RECORD_EVERY_AFTER, dist_name: str = "G_SERIAL",
                   max_start_digits: int = 2, max_start_phalanges: int = 2) -> Dict[str, Any]:
    """Build the shared reduced start for ``seed``, then run one walk per
    mixture in ``MIXTURES``, returning a JSON-serializable dict with the
    achieved start sizes and, per mixture, the recorded trajectory,
    per-operator acceptance counts, and the phenotype hash at step 40 and
    at ``walk_length``."""
    dist = G_SERIAL  # only named Distribution used by this experiment
    start_derivation, start_digits, start_max_phalanx, start_reduced_fully = build_start(
        seed, dist, max_start_digits, max_start_phalanges,
    )
    start_model = derive(start_derivation)
    start_metrics = _metrics(start_model)
    record_steps = sorted(_record_steps(walk_length, record_through, record_every_after))

    out: Dict[str, Any] = {
        "start_digit_count": float(start_digits),
        "start_max_phalanx_count": float(start_max_phalanx),
        "start_reduced_fully": 1.0 if start_reduced_fully else 0.0,
    }
    out["start"] = start_metrics

    for mix_idx, (mixture_name, mixture) in enumerate(MIXTURES.items()):
        rng = np.random.default_rng([seed, mix_idx])
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
        # Delta-from-start convenience fields (flat, numeric -- easy to
        # bootstrap-aggregate across seeds). Key labels use the ACTUAL
        # record_through/walk_length values (not literal "40"/"500"), so a
        # smoke-sized call (e.g. record_through=4, walk_length=8) produces
        # correctly-named keys too.
        by_step = {row["step"]: row for row in trajectory}
        for check_step in (record_through, walk_length):
            row = by_step.get(float(check_step))
            if row is not None:
                for key in ("joints", "digits", "motors", "total_length_m"):
                    mixture_result[f"delta_{key}_at_{check_step}"] = row[key] - start_metrics[key]
        out[mixture_name] = mixture_result

    return out


register("e2_drift", e2_drift_seed)


def _bootstrap_ci(vals: List[float], n_resamples: int = 2000, seed: int = 0) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    if len(arr) == 0:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)),
        "n": len(arr),
    }


def _aggregate(per_seed: List[Dict[str, Any]], record_through: int, walk_length: int) -> Dict[str, Any]:
    ok = [r["result"] for r in per_seed if r["ok"]]
    agg: Dict[str, Any] = {"n_seeds": len(ok)}
    for mixture_name in MIXTURES:
        rows = [r[mixture_name] for r in ok if mixture_name in r]
        mix_agg: Dict[str, Any] = {}
        for check_step in (record_through, walk_length):
            label = str(check_step)
            for metric in ("joints", "digits", "motors"):
                key = f"delta_{metric}_at_{label}"
                vals = [r[key] for r in rows if key in r]
                mix_agg[key] = _bootstrap_ci(vals)
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
        agg[mixture_name] = mix_agg
    return agg


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4g}"


def _summary_md(params: Dict[str, Any], aggregate: Dict[str, Any], wall_time_s: float) -> str:
    lines = [
        "# Experiment: e2_drift", "",
        f"Wall time: {wall_time_s:.3f} s", "",
        "## Params", "", "```json", json.dumps(params, indent=2, sort_keys=True, default=str), "```", "",
        f"n_seeds: {aggregate.get('n_seeds')}", "",
    ]
    labels = [str(params["record_through"]), str(params["walk_length"])]
    for mixture_name in MIXTURES:
        a = aggregate[mixture_name]
        lines += [f"## Mixture: {mixture_name}", ""]
        for label in labels:
            lines += [
                f"### Step {label}", "",
                "| metric | mean delta | 95% CI lo | 95% CI hi | n |",
                "|---|---|---|---|---|",
            ]
            for metric in ("joints", "digits", "motors"):
                key = f"delta_{metric}_at_{label}"
                b = a[key]
                lines.append(f"| {metric} | {_fmt(b['mean'])} | {_fmt(b['ci_lo'])} | {_fmt(b['ci_hi'])} | {b['n']} |")
            lines += ["", f"distinct_hash_fraction_at_{label}: {_fmt(a[f'distinct_hash_fraction_at_{label}'])}", ""]
        lines += ["### Acceptance rate per operator", "", "| operator | acceptance_rate |", "|---|---|"]
        for op, rate in sorted(a["acceptance_rate"].items()):
            lines.append(f"| {op} | {_fmt(rate)} |")
        lines.append("")
    lines += ["## Reading", ""]
    for mixture_name in MIXTURES:
        a = aggregate[mixture_name]
        for label in labels:
            parts = []
            for metric in ("joints", "digits", "motors"):
                b = a[f"delta_{metric}_at_{label}"]
                parts.append(f"{metric} {_fmt(b['mean'])} [{_fmt(b['ci_lo'])}, {_fmt(b['ci_hi'])}]")
            lines.append(
                f"- {mixture_name} step {label}: " + "; ".join(parts) +
                f"; distinct_hash_fraction {_fmt(a[f'distinct_hash_fraction_at_{label}'])}."
            )
    lines.append("")
    return "\n".join(lines)


def run(out_dir: Optional[str] = None, seeds: Sequence[int] = range(256),
        walk_length: int = _WALK_LENGTH, record_through: int = _RECORD_THROUGH,
        record_every_after: int = _RECORD_EVERY_AFTER, processes: int = 24) -> Dict[str, Any]:
    """Run ``e2_drift_seed`` over ``seeds`` via the multiprocess runner, then
    overwrite ``result.json``/``summary.md`` (in ``out_dir``, default
    ``project-notes/grammar/experiments/E2_drift``) with the per-mixture
    bootstrap-CI / acceptance-rate aggregate described in this module's
    docstring, in place of the runner's generic per-key aggregate."""
    if out_dir is None:
        out_dir = "project-notes/grammar/experiments/E2_drift"
    params = {
        "walk_length": walk_length, "record_through": record_through,
        "record_every_after": record_every_after,
    }
    result = run_experiment(
        "e2_drift", e2_drift_seed, params=params, seeds=seeds, out_dir=out_dir, processes=processes,
    )
    aggregate = _aggregate(result["per_seed"], record_through, walk_length)
    result["aggregate"] = aggregate
    out_path = Path(out_dir)
    (out_path / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    (out_path / "summary.md").write_text(_summary_md(result["params"], aggregate, result["wall_time_s"]))
    return result


if __name__ == "__main__":
    run()
