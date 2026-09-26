"""E3: reachability of four target structural signatures by hill-climbing
``vary`` under three operator sets.

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
function below.

Hill-climb (``_hill_climb``): from a shared start (``e2_drift.build_start``
-- the same reduced ``G_SERIAL`` start used by E2), repeatedly draw one
operator uniformly from the given operator set, apply ``vary(current, rng,
G_FULL, operator=op)`` (retrying up to ``max_attempts_per_step`` times on
``VariationImpossible``); accept the resulting candidate iff its distance
to the target does not increase (plateau moves allowed). Stops on
``distance == 0`` (success) or once ``max_accepted`` mutations have been
accepted, or (safety net against a pathological all-``VariationImpossible``
stall) once ``safety_cap`` total proposals have been made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..derive import (
    OPERATORS,
    SMALL_STEP_OPERATORS,
    MINIMAL_STRUCTURAL_OPERATORS,
    Derivation,
    VariationImpossible,
    derive,
    vary,
)
from ..fk import forward_kinematics
from ..kinematics import KinematicModel
from ..variants import G_FULL
from .e2_drift import build_start
from .runner import register, run_experiment

# --------------------------------------------------------------------------
# Operator sets under test.
# --------------------------------------------------------------------------

OPERATOR_SETS: Dict[str, Tuple[str, ...]] = {
    "DEFAULT": tuple(OPERATORS),
    "DEFAULT+SMALL": tuple(OPERATORS) + tuple(SMALL_STEP_OPERATORS),
    "MINIMAL": tuple(MINIMAL_STRUCTURAL_OPERATORS) + ("insert_phalanx", "delete_phalanx") + tuple(SMALL_STEP_OPERATORS),
}
for _name, _ops in OPERATOR_SETS.items():
    assert len(_ops) == len(set(_ops)), f"operator set {_name!r} has duplicates"


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
    non-parallel (at least one non-parallel pair among them), >= 4 digits."""
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
    return float(d_count + d_k + parallel_shortfall)


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
                 ops: Sequence[str], rng: np.random.Generator, max_accepted: int = 200,
                 max_attempts_per_step: int = 8, safety_cap: int = 4000) -> Dict[str, Any]:
    current = start_derivation
    current_model = derive(current)
    current_dist = dist_fn(current_model, current)
    n_accepted = 0
    n_proposed = 0
    success = current_dist == 0.0
    success_accepted = 0 if success else None
    success_proposed = 0 if success else None

    while not success and n_accepted < max_accepted and n_proposed < safety_cap:
        candidate = None
        for _ in range(max_attempts_per_step):
            if n_proposed >= safety_cap:
                break
            op = ops[int(rng.integers(0, len(ops)))]
            n_proposed += 1
            try:
                candidate = vary(current, rng, G_FULL, operator=op)
                break
            except VariationImpossible:
                candidate = None
        if candidate is None:
            continue
        candidate_model = derive(candidate)
        candidate_dist = dist_fn(candidate_model, candidate)
        if candidate_dist <= current_dist:
            current, current_model, current_dist = candidate, candidate_model, candidate_dist
            n_accepted += 1
            if current_dist == 0.0:
                success = True
                success_accepted = n_accepted
                success_proposed = n_proposed

    return {
        "success": bool(success),
        "final_distance": float(current_dist),
        "n_accepted": int(n_accepted),
        "n_proposed": int(n_proposed),
        "success_accepted": success_accepted,
        "success_proposed": success_proposed,
    }


def e3_reach_seed(seed: int, max_accepted: int = 200, max_attempts_per_step: int = 8,
                   safety_cap: int = 4000) -> Dict[str, Any]:
    """One restart (``seed``): build the shared reduced ``G_SERIAL`` start
    (``e2_drift.build_start``), then hill-climb toward every target under
    every operator set (12 climbs total), keyed ``"<target>::<opset>"``."""
    start_derivation, _, _, _ = build_start(seed)
    out: Dict[str, Any] = {}
    for t_idx, (target_name, dist_fn) in enumerate(TARGETS.items()):
        for o_idx, (opset_name, ops) in enumerate(OPERATOR_SETS.items()):
            rng = np.random.default_rng([seed, t_idx, o_idx])
            out[f"{target_name}::{opset_name}"] = _hill_climb(
                start_derivation, dist_fn, ops, rng, max_accepted, max_attempts_per_step, safety_cap,
            )
    return out


register("e3_reach", e3_reach_seed)


def _bootstrap_ci_rate(successes: List[bool], n_resamples: int = 2000, seed: int = 0) -> Dict[str, Optional[float]]:
    arr = np.asarray(successes, dtype=float)
    if len(arr) == 0:
        return {"rate": None, "ci_lo": None, "ci_hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    means = arr[idx].mean(axis=1)
    return {
        "rate": float(arr.mean()),
        "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)),
        "n": len(arr),
    }


def _aggregate(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r["result"] for r in per_seed if r["ok"]]
    agg: Dict[str, Any] = {"n_seeds": len(ok)}
    for target_name in TARGETS:
        for opset_name in OPERATOR_SETS:
            key = f"{target_name}::{opset_name}"
            rows = [r[key] for r in ok if key in r]
            successes = [bool(r["success"]) for r in rows]
            success_accepted = [r["success_accepted"] for r in rows if r["success"]]
            success_proposed = [r["success_proposed"] for r in rows if r["success"]]
            proposed_all = [r["n_proposed"] for r in rows]
            agg[key] = {
                "success_rate": _bootstrap_ci_rate(successes),
                "median_accepted_at_success": (
                    float(np.median(success_accepted)) if success_accepted else None
                ),
                "median_proposed_at_success": (
                    float(np.median(success_proposed)) if success_proposed else None
                ),
                "median_proposed_all": float(np.median(proposed_all)) if proposed_all else None,
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
        "## Table: target x operator set", "",
        "| target | operator_set | success_rate | 95% CI lo | 95% CI hi | median accepted @ success | "
        "median proposed @ success | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for target_name in TARGETS:
        for opset_name in OPERATOR_SETS:
            a = aggregate[f"{target_name}::{opset_name}"]
            sr = a["success_rate"]
            lines.append(
                f"| {target_name} | {opset_name} | {_fmt(sr['rate'])} | {_fmt(sr['ci_lo'])} | {_fmt(sr['ci_hi'])} | "
                f"{_fmt(a['median_accepted_at_success'])} | {_fmt(a['median_proposed_at_success'])} | {sr['n']} |"
            )
    lines += ["", "## Reading", ""]
    for target_name in TARGETS:
        for opset_name in OPERATOR_SETS:
            a = aggregate[f"{target_name}::{opset_name}"]
            sr = a["success_rate"]
            lines.append(
                f"- {target_name} / {opset_name}: success_rate {_fmt(sr['rate'])} "
                f"[{_fmt(sr['ci_lo'])}, {_fmt(sr['ci_hi'])}] (n={sr['n']}); "
                f"median accepted @ success {_fmt(a['median_accepted_at_success'])}; "
                f"median proposed @ success {_fmt(a['median_proposed_at_success'])}."
            )
    lines.append("")
    return "\n".join(lines)


def run(out_dir: Optional[str] = None, seeds: Sequence[int] = range(64),
        max_accepted: int = 200, max_attempts_per_step: int = 8, safety_cap: int = 4000,
        processes: int = 24) -> Dict[str, Any]:
    if out_dir is None:
        out_dir = "project-notes/grammar/experiments/E3_reach"
    params = {
        "max_accepted": max_accepted, "max_attempts_per_step": max_attempts_per_step,
        "safety_cap": safety_cap,
    }
    result = run_experiment(
        "e3_reach", e3_reach_seed, params=params, seeds=seeds, out_dir=out_dir, processes=processes,
    )
    aggregate = _aggregate(result["per_seed"])
    result["aggregate"] = aggregate
    out_path = Path(out_dir)
    (out_path / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    (out_path / "summary.md").write_text(_summary_md(result["params"], aggregate, result["wall_time_s"]))
    return result


if __name__ == "__main__":
    run()
