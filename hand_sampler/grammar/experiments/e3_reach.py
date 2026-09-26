"""E3: reachability of four target structural signatures by hill-climbing
``vary`` under two operator pools and four grammar variants.

Targets (see ``TARGETS``) are defined as structural signatures over a
derived model's own ``Derivation`` steps (digit mounts/phalanx counts/module
kinds) plus, where a *direction* matters, the derived model's forward
kinematics at q=0 (a joint's world-frame axis is
``forward_kinematics(model, {})[joint.child][:3, :3] @ joint.axis`` -- see
``rules.py``/``fk.py``'s convention: a joint's stored axis is expressed in
its own child-body frame at q=0, so the child body's own root-frame
rotation carries it to world/root frame). A digit's own "mount direction"
is its first phalanx's local +z axis in the root frame (the direction the
digit's own first segment extends in), i.e. the third column of
``transforms[f"d{digit_id}p1"][:3, :3]``.

Each target's distance is a small sum of per-criterion terms, each 0 when
that one criterion is satisfied and a small positive integer otherwise (an
exact count shortfall/excess, or a violating-pair count) -- e.g.
``anthropomorphic_staggered``'s distance is
``|digit_count - 5| + (#digits with < 3 phalanges) + max(0, 3 - #distinct
mount fractions)``. This is an explicit design choice (the task prescribes
"sum of absolute count shortfalls/excesses ... normalized to 1 when
satisfied", which this satisfies: every term is exactly 0 when its own
criterion holds, and O(1) otherwise) documented at each ``_dist_*``
function below. I14 fix: ``arch_palm`` additionally requires >= 2 digits
mounted on a non-root palm body (previously satisfied by an empty,
palm-jointed palm body with every digit still mounted on the root, which
inflated success under pools that easily add empty palm bodies).

Operator pools under test (I14 fix: replaces the previous DEFAULT /
DEFAULT+SMALL / MINIMAL trio):

- ``DEFAULT``: the 7 default ``derive.OPERATORS``.
- ``UNION``: ``DEFAULT`` plus the 6 ``SMALL_STEP_OPERATORS`` (``step_length``
  dropped, ``step_root_length``/``step_radius`` added -- see ``derive.py``'s
  I14 fix 5) plus the 5 ``MINIMAL_STRUCTURAL_OPERATORS`` (18 operators).

Grammar variants under test (``DIST_VARIANTS``): ``G_FULL``, ``G_NOBRANCH``,
``G_FULL_INS``, ``G_NOBRANCH_INS`` (see ``variants.py``).

Hill-climb (``_hill_climb``): from a shared start (``starts.build_start`` --
the same reduced ``G_SERIAL`` start used by E2), repeatedly draw one
operator uniformly from the given pool, apply ``vary(current, rng, dist,
operator=op)`` (a ``VariationImpossible`` proposal still counts as ONE
proposal -- see the budget note below -- and simply produces no candidate).
Accept the resulting candidate iff its distance to the target does not
increase (plateau moves allowed). Stops on ``distance == 0`` (success) or
once ``budget`` proposals have been made.

I14 fix: the stopping/reporting budget is now PROPOSALS (evaluations), not
accepted moves -- the previous ``max_accepted``-based stop unfairly
penalized low-acceptance-rate pools (e.g. small-step-heavy ones), which need
many more proposals to rack up the same number of ACCEPTED moves as a
coarser pool, even at an identical per-proposal success rate. Each restart
now gets a fixed budget of 1500 proposals; success/failure and (for
successes) the proposal count at success are recorded, and failures are
CENSORED at the budget (their true "proposals to success" is >= budget,
unknown) rather than dropped, when computing the reported median.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..derive import (
    EVOLUTION_OPERATORS,
    MINIMAL_STRUCTURAL_OPERATORS,
    OPERATORS,
    SMALL_STEP_OPERATORS,
    Derivation,
    VariationImpossible,
    derive,
    vary,
)
from ..fk import forward_kinematics
from ..kinematics import KinematicModel
from ..variants import G_FULL, G_FULL_INS, G_NOBRANCH, G_NOBRANCH_INS
from .runner import register, run_experiment
from .starts import build_start

# --------------------------------------------------------------------------
# Operator pools and grammar variants under test.
# --------------------------------------------------------------------------

OPERATOR_SETS: Dict[str, Tuple[str, ...]] = {
    "DEFAULT": tuple(OPERATORS),
    "UNION": tuple(dict.fromkeys(tuple(OPERATORS) + tuple(SMALL_STEP_OPERATORS) + tuple(MINIMAL_STRUCTURAL_OPERATORS))),
    # Grammar 0.5 (I18 fix 7): the exact-inverse evolution pool
    # (``derive.EVOLUTION_OPERATORS`` -- 5 growth/shrink pairs plus every
    # small-step operator), drawn uniformly like ``DEFAULT``/``UNION``
    # rather than through the pair-balanced ``EVOLUTION_pool`` weighting
    # used by ``e2_drift``/``e12_balance``, so E3 can compare reachability
    # under the SAME pool E12 certifies for neutral drift.
    "EVOLUTION_uniform": tuple(EVOLUTION_OPERATORS),
}
for _name, _ops in OPERATOR_SETS.items():
    assert len(_ops) == len(set(_ops)), f"operator set {_name!r} has duplicates"

DIST_VARIANTS: Dict[str, Any] = {
    "G_FULL": G_FULL, "G_NOBRANCH": G_NOBRANCH, "G_FULL_INS": G_FULL_INS, "G_NOBRANCH_INS": G_NOBRANCH_INS,
}

DEFAULT_BUDGET = 1500

# --------------------------------------------------------------------------
# Structural extraction helpers (derivation-based -- exact, no ambiguity
# about branches/reordering the way a bare ``KinematicModel`` walk would
# have; ``derive.py`` always names a digit's own first phalanx body
# ``f"d{digit_id}p1"``, so ``model`` is only needed for its forward
# kinematics, to turn a locally-stored axis/direction into a root-frame one).
# --------------------------------------------------------------------------


def _top_level_digits(derivation: Derivation) -> List[Dict[str, Any]]:
    return [s.params for s in derivation.steps if s.production == "Digit" and s.params.get("top_level")]


def _digit_has_prismatic(derivation: Derivation, digit_id: str) -> bool:
    return any(
        s.production == "Phalanx" and s.params["digit_id"] == digit_id and s.params["module"]["kind"] == "P"
        for s in derivation.steps
    )


def _palm_joint_steps(derivation: Derivation) -> List[Any]:
    return [s for s in derivation.steps if s.production == "PalmBody" and s.params["has_joint"]]


def _joint_world_axis(model: KinematicModel, transforms: Dict[str, np.ndarray], joint_name: str) -> Optional[np.ndarray]:
    joint = next((j for j in model.joints if j.name == joint_name), None)
    if joint is None or joint.child not in transforms:
        return None
    R = transforms[joint.child][:3, :3]
    return R @ np.asarray(joint.axis, dtype=float)


def _digit_direction(model: KinematicModel, transforms: Dict[str, np.ndarray], digit_id: str) -> Optional[np.ndarray]:
    body = f"d{digit_id}p1"
    if body not in transforms:
        return None
    return transforms[body][:3, 2]


# --------------------------------------------------------------------------
# Target distance functions. Each returns 0.0 iff every one of the target's
# listed criteria is met, and a small positive number (sum of independent,
# per-criterion shortfall/violation counts) otherwise -- see module
# docstring.
# --------------------------------------------------------------------------


def _dist_anthropomorphic_staggered(model: KinematicModel, derivation: Derivation) -> float:
    """5 digits, >= 3 phalanges each, >= 3 distinct mount fractions among
    them (no palm-joint criterion -- see the target's own docstring in
    ``TARGETS``)."""
    digits = _top_level_digits(derivation)
    d_count = abs(len(digits) - 5)
    n_short_phalanx = sum(1 for d in digits if d["phalanx_count"] < 3)
    n_distinct_fracs = len({d["mount_frac"] for d in digits})
    d_fracs = max(0, 3 - n_distinct_fracs)
    return float(d_count + n_short_phalanx + d_fracs)


def _dist_radial_3(model: KinematicModel, derivation: Derivation) -> float:
    """Exactly 3 digits, >= 2 phalanges each, every pair of digit mount
    directions (root frame) >= 90 degrees apart (dot product <= 0)."""
    digits = _top_level_digits(derivation)
    d_count = abs(len(digits) - 3)
    n_short_phalanx = sum(1 for d in digits if d["phalanx_count"] < 2)
    transforms = forward_kinematics(model, {})
    directions = [
        d for d in (_digit_direction(model, transforms, dd["digit_id"]) for dd in digits) if d is not None
    ]
    n_bad_pairs = 0
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            if float(np.dot(directions[i], directions[j])) > 1e-9:
                n_bad_pairs += 1
    return float(d_count + n_short_phalanx + n_bad_pairs)


def _dist_prismatic_gripper(model: KinematicModel, derivation: Derivation) -> float:
    """Exactly 2 digits, each containing >= 1 prismatic joint, no palm
    joints anywhere."""
    digits = _top_level_digits(derivation)
    d_count = abs(len(digits) - 2)
    n_missing_prismatic = sum(1 for d in digits if not _digit_has_prismatic(derivation, d["digit_id"]))
    n_palm_joints = len(_palm_joint_steps(derivation))
    return float(d_count + n_missing_prismatic + n_palm_joints)


def _dist_arch_palm(model: KinematicModel, derivation: Derivation) -> float:
    """>= 2 non-root palm bodies with palm joints whose root-frame axes are
    non-parallel (at least one non-parallel pair among them), >= 4 digits,
    AND (I15 fix 7, tightening the I14 fix) >= 2 of those digits mounted
    specifically on a non-root palm body that ITSELF carries a palm joint
    (``has_joint`` True on that ``PalmBody`` step) -- not merely any
    non-root palm body -- since a digit mounted on an un-jointed palm body
    is not on the "arch" this target names at all (previously satisfiable
    by digits mounted on a plain, joint-free palm body while the actual
    jointed palm bodies stayed empty)."""
    digits = _top_level_digits(derivation)
    d_count = max(0, 4 - len(digits))
    palm_joint_steps = _palm_joint_steps(derivation)
    k = len(palm_joint_steps)
    d_k = max(0, 2 - k)
    parallel_shortfall = 0.0
    if k >= 2:
        transforms = forward_kinematics(model, {})
        axes = [
            a for a in (
                _joint_world_axis(model, transforms, f"{s.params['name']}_j") for s in palm_joint_steps
            ) if a is not None
        ]
        any_nonparallel = any(
            float(np.linalg.norm(np.cross(axes[i], axes[j]))) > 1e-6
            for i in range(len(axes)) for j in range(i + 1, len(axes))
        )
        if not any_nonparallel:
            parallel_shortfall = 1.0
    jointed_palm_names = {s.params["name"] for s in palm_joint_steps}
    n_on_jointed_palm = sum(1 for d in digits if str(d["mount"]) in jointed_palm_names)
    d_mount = max(0, 2 - n_on_jointed_palm)
    return float(d_count + d_k + parallel_shortfall + d_mount)


TARGETS: Dict[str, Callable[[KinematicModel, Derivation], float]] = {
    "anthropomorphic_staggered": _dist_anthropomorphic_staggered,
    "radial_3": _dist_radial_3,
    "prismatic_gripper": _dist_prismatic_gripper,
    "arch_palm": _dist_arch_palm,
}


# --------------------------------------------------------------------------
# Hill climb.
# --------------------------------------------------------------------------


def _hill_climb(start_derivation: Derivation, dist_fn: Callable[[KinematicModel, Derivation], float],
                 ops: Sequence[str], dist, rng: np.random.Generator, budget: int = DEFAULT_BUDGET) -> Dict[str, Any]:
    current = start_derivation
    current_model = derive(current)
    current_dist = dist_fn(current_model, current)
    n_accepted = 0
    n_proposed = 0
    success = current_dist == 0.0
    success_proposed = 0 if success else None

    while not success and n_proposed < budget:
        op = ops[int(rng.integers(0, len(ops)))]
        n_proposed += 1  # I14 fix: every drawn operator is ONE proposal/evaluation,
        # whether or not it turns out applicable -- see module docstring.
        try:
            candidate = vary(current, rng, dist, operator=op)
        except VariationImpossible:
            continue
        candidate_model = derive(candidate)
        candidate_dist = dist_fn(candidate_model, candidate)
        if candidate_dist <= current_dist:
            current, current_model, current_dist = candidate, candidate_model, candidate_dist
            n_accepted += 1
            if current_dist == 0.0:
                success = True
                success_proposed = n_proposed

    return {
        "success": bool(success),
        "final_distance": float(current_dist),
        "n_accepted": int(n_accepted),
        "n_proposed": int(n_proposed),
        "success_proposed": success_proposed,
        "censored": bool(not success),
    }


def e3_reach_seed(seed: int, budget: int = DEFAULT_BUDGET) -> Dict[str, Any]:
    """One restart (``seed``): build the shared reduced ``G_SERIAL`` start
    (``starts.build_start``), then hill-climb toward every target under
    every operator pool x grammar variant combination (4 targets x 2 pools
    x 4 dists = 32 climbs total), keyed ``"<target>::<opset>::<dist>"``."""
    start_derivation, _, _, _ = build_start(seed)
    out: Dict[str, Any] = {}
    for t_idx, (target_name, dist_fn) in enumerate(TARGETS.items()):
        for o_idx, (opset_name, ops) in enumerate(OPERATOR_SETS.items()):
            for d_idx, (dist_name, dist) in enumerate(DIST_VARIANTS.items()):
                rng = np.random.default_rng([seed, t_idx, o_idx, d_idx])
                out[f"{target_name}::{opset_name}::{dist_name}"] = _hill_climb(
                    start_derivation, dist_fn, ops, dist, rng, budget,
                )
    return out


register("e3_reach", e3_reach_seed)


_Z_95 = 1.959963984540054


def _wilson_ci(k: int, n: int, z: float = _Z_95) -> Tuple[Optional[float], Optional[float]]:
    """Wilson score interval for a binomial rate ``k/n`` (I14 fix: replaces
    a percentile bootstrap, which collapses to a point/degenerate interval
    for a Bernoulli sample whose rate is near 0 or 1 -- exactly the regime
    most of these success rates fall in)."""
    if n == 0:
        return None, None
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def _wilson_rate(successes: List[bool]) -> Dict[str, Optional[float]]:
    n = len(successes)
    if n == 0:
        return {"rate": None, "ci_lo": None, "ci_hi": None, "n": 0}
    k = sum(1 for s in successes if s)
    lo, hi = _wilson_ci(k, n)
    return {"rate": k / n, "ci_lo": lo, "ci_hi": hi, "n": n}


def _censored_median_proposed(rows: List[Dict[str, Any]], budget: int) -> Dict[str, Optional[float]]:
    """Median proposals-to-success over ALL restarts (I14 fix), with a
    failed restart's (unknown, but >= ``budget``) true value censored AT
    the budget for the purpose of the median computation. If half or more
    of the restarts are censored, the true median is itself >= budget (not
    a real number below it); ``is_lower_bound`` flags that case."""
    if not rows:
        return {"median": None, "is_lower_bound": None, "censored_fraction": None, "n": 0}
    vals = sorted(r["success_proposed"] if r["success"] else budget for r in rows)
    n = len(vals)
    censored_fraction = sum(1 for r in rows if not r["success"]) / n
    return {
        "median": float(np.median(vals)),
        "is_lower_bound": bool(censored_fraction >= 0.5),
        "censored_fraction": censored_fraction,
        "n": n,
    }


def _aggregate(per_seed: List[Dict[str, Any]], budget: int) -> Dict[str, Any]:
    ok = [r["result"] for r in per_seed if r["ok"]]
    agg: Dict[str, Any] = {"n_seeds": len(ok)}
    for target_name in TARGETS:
        for opset_name in OPERATOR_SETS:
            for dist_name in DIST_VARIANTS:
                key = f"{target_name}::{opset_name}::{dist_name}"
                rows = [r[key] for r in ok if key in r]
                successes = [bool(r["success"]) for r in rows]
                agg[key] = {
                    "success_rate": _wilson_rate(successes),
                    "median_proposed_censored": _censored_median_proposed(rows, budget),
                    "median_proposed_all": (
                        float(np.median([r["n_proposed"] for r in rows])) if rows else None
                    ),
                    "n": len(rows),
                }
    return agg


def _fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.4g}"


def _summary_md(params: Dict[str, Any], aggregate: Dict[str, Any], wall_time_s: float) -> str:
    lines = [
        "# Experiment: e3_reach", "",
        f"Wall time: {wall_time_s:.3f} s", "",
        "## Params", "", "```json", json.dumps(params, indent=2, sort_keys=True, default=str), "```", "",
        f"n_seeds (restarts): {aggregate.get('n_seeds')}", "",
        "Budget is in PROPOSALS (evaluations), not accepted moves. "
        "success_rate uses a Wilson score interval. median_proposed is censored "
        "at the budget for restarts that never succeeded (reported "
        "'>= budget' via is_lower_bound/censored_fraction when that applies).", "",
        "## Table: target x operator pool x dist", "",
        "| target | pool | dist | success_rate | 95% CI lo | 95% CI hi | "
        "median proposed (censored) | is_lower_bound | censored_fraction | n |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for target_name in TARGETS:
        for opset_name in OPERATOR_SETS:
            for dist_name in DIST_VARIANTS:
                a = aggregate[f"{target_name}::{opset_name}::{dist_name}"]
                sr = a["success_rate"]
                mc = a["median_proposed_censored"]
                lines.append(
                    f"| {target_name} | {opset_name} | {dist_name} | {_fmt(sr['rate'])} | {_fmt(sr['ci_lo'])} | "
                    f"{_fmt(sr['ci_hi'])} | {_fmt(mc['median'])} | {mc['is_lower_bound']} | "
                    f"{_fmt(mc['censored_fraction'])} | {sr['n']} |"
                )
    lines += ["", "## Reading", ""]
    for target_name in TARGETS:
        for opset_name in OPERATOR_SETS:
            for dist_name in DIST_VARIANTS:
                a = aggregate[f"{target_name}::{opset_name}::{dist_name}"]
                sr = a["success_rate"]
                mc = a["median_proposed_censored"]
                lines.append(
                    f"- {target_name} / {opset_name} / {dist_name}: success_rate {_fmt(sr['rate'])} "
                    f"[{_fmt(sr['ci_lo'])}, {_fmt(sr['ci_hi'])}] (n={sr['n']}); "
                    f"median proposed {'>= ' if mc['is_lower_bound'] else ''}{_fmt(mc['median'])} "
                    f"(censored_fraction {_fmt(mc['censored_fraction'])})."
                )
    lines.append("")
    return "\n".join(lines)


def run(out_dir: Optional[str] = None, seeds: Sequence[int] = range(64),
        budget: int = DEFAULT_BUDGET, processes: int = 24, allow_dirty: bool = False) -> Dict[str, Any]:
    if out_dir is None:
        out_dir = "project-notes/grammar/experiments/E3_reach"
    params = {"budget": budget}
    result = run_experiment(
        "e3_reach", e3_reach_seed, params=params, seeds=seeds, out_dir=out_dir, processes=processes,
        allow_dirty=allow_dirty,
    )
    aggregate = _aggregate(result["per_seed"], budget)
    result["aggregate"] = aggregate
    out_path = Path(out_dir)
    (out_path / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    (out_path / "summary.md").write_text(_summary_md(result["params"], aggregate, result["wall_time_s"]))
    return result


if __name__ == "__main__":
    run()
