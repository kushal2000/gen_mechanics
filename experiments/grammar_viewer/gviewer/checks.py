"""The simulator's viability checks, one at a time.

`grammar_envelope.viability_report` folds every check into one admitted flag
plus a list of reason strings. The essential viewer lets each check be
switched off, so this module evaluates them separately:

- nine STRUCTURAL checks, one per rejection reason `_admit_structural` (and
  the `fits_envelope` call inside it) can produce. They are recomputed here
  from the model with the envelope's own helpers and constants;
  `tests/test_checks.py` checks that their conjunction equals
  `_admit_structural(model).ok` on sampled designs from every variant.
- four PHYSICAL checks, computed with the envelope's own functions
  (`canonicalize`, `palm_up`, `rest_overlap_pairs`,
  `spawn_height_above_palm_m`) and its thresholds, exactly as
  `viability_report` computes them.

The physical checks need the 32-slot envelope design, which `canonicalize`
can only build when every structural check passes. On a hand that fails a
structural check they are "n/a". A disabled structural check therefore lets
such a hand through without the physical checks; `Evaluation.note` says so.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from hand_sampler.grammar.derive import Derivation, derive, sample_derivation
from hand_sampler.grammar.kinematics import KinematicModel, ModelError

from .envload import load_env_modules

PASS, FAIL, NA, SKIPPED = "pass", "fail", "n/a", "skipped"


def ge():
    return load_env_modules().grammar_envelope


@dataclass(frozen=True)
class Check:
    key: str
    label: str
    kind: str      # "structural" | "physical"
    why: str


def _checks() -> Tuple[Check, ...]:
    g = ge()
    gate_mm = g.MAX_REST_PENETRATION_M * 1000.0
    spawn_mm = load_env_modules().palm_calibration.MIN_SPAWN_HEIGHT_ABOVE_PALM_M * 1000.0
    return (
        Check("revolute_only", "revolute joints only", "structural",
              "Every movable joint is a hinge; the simulator's hand asset has no prismatic or continuous joints."),
        Check("no_couplings", "no coupled joints", "structural",
              "One motor per joint; the simulator has no mimic/coupled joints."),
        Check("no_branches", "no branching digits", "structural",
              "A digit is one serial chain; the simulator lays each digit out as one row of joint slots."),
        Check("max_digits", f"at most {g.MAX_DIGITS} digits", "structural",
              f"The simulator's hand has {g.N_FINGERS} finger slots."),
        Check("joints_per_digit", f"at most {g.MAX_JOINTS_PER_DIGIT} joints per digit", "structural",
              f"Each finger slot holds {g.N_JOINTS_PER_FINGER} joints."),
        Check("max_jointed_palm", f"at most {g.MAX_JOINTED_PALM_BODIES} jointed palm bodies", "structural",
              f"The simulator has {g.MAX_JOINTED_PALM_BODIES} palm-joint slots."),
        Check("palm_not_nested", "no jointed palm body on another", "structural",
              "Each palm-joint slot hangs off the root palm, so jointed palm bodies cannot be stacked."),
        Check("carrier_one_digit", "each jointed palm body carries at most 1 digit", "structural",
              "Each palm-joint slot carries exactly one finger slot."),
        Check("finger_slots", f"root digits + jointed palm bodies <= {g.N_FINGERS}", "structural",
              f"Every jointed palm body reserves a finger slot, and only {g.N_FINGERS} exist."),
        Check("overlap_zero", f"no self-overlap > {gate_mm:.0f} mm at the zero pose", "physical",
              "Capsules that interpenetrate at the start make the physics engine push them apart violently."),
        Check("overlap_reset", f"no self-overlap > {gate_mm:.0f} mm at the reset pose", "physical",
              "The same at the slightly curled pose every episode starts from."),
        Check("spawn_height", f"spawn point >= {spawn_mm:.0f} mm above the palm", "physical",
              "The cube must start above the palm so the hand holds it up against gravity."),
        Check("reach", "at least 2 fingertips reach the cube (within 5 cm)", "physical",
              "Without two fingertips near the cube the hand cannot manipulate it."),
    )


CHECKS: Tuple[Check, ...] = _checks()
CHECK_KEYS: Tuple[str, ...] = tuple(c.key for c in CHECKS)
STRUCTURAL_KEYS: Tuple[str, ...] = tuple(c.key for c in CHECKS if c.kind == "structural")
PHYSICAL_KEYS: Tuple[str, ...] = tuple(c.key for c in CHECKS if c.kind == "physical")
CHECK_BY_KEY: Dict[str, Check] = {c.key: c for c in CHECKS}
MIN_REACHING_TIPS = 2


@dataclass(frozen=True)
class CheckResult:
    status: str        # PASS | FAIL | NA | SKIPPED
    value: str


@dataclass
class Evaluation:
    results: Dict[str, CheckResult]
    structural_ok: bool                      # `_admit_structural(model).ok`, the oracle's own verdict
    unexplained: Tuple[str, ...] = ()        # oracle reasons no listed check covers (never seen; see tests)
    design: object = None                    # grammar_envelope.EnvelopeDesign
    pu: object = None                        # grammar_envelope.PalmUpResult (full sweep unless stopped early)
    pairs_q0: List[Tuple[int, int, float]] = field(default_factory=list)
    pairs_reset: List[Tuple[int, int, float]] = field(default_factory=list)

    def failing(self, enabled: Iterable[str]) -> List[str]:
        en = set(enabled)
        return [k for k in CHECK_KEYS if k in en and k in self.results and self.results[k].status == FAIL]

    def passes(self, enabled: Iterable[str]) -> bool:
        return not self.unexplained and not self.failing(enabled)

    @property
    def note(self) -> str:
        if self.structural_ok:
            return ""
        return "overlap, spawn and reach are n/a: the simulator cannot build this hand"


# --------------------------------------------------------------------------
# Structural checks
# --------------------------------------------------------------------------


def _structural(model: KinematicModel) -> Dict[str, CheckResult]:
    """Mirrors `_admit_structural` (and `fits_envelope` with
    allow_palm_joints=True, allow_branches=False), one result per reason."""
    g = ge()
    palm = g._palm_body_names(model)
    children_of: Dict[str, List] = {}
    for j in model.joints:
        children_of.setdefault(j.parent, []).append(j)
    out: Dict[str, CheckResult] = {}

    bad_types = sorted({j.type for j in model.joints if j.type != "fixed"} - {"revolute"})
    out["revolute_only"] = (CheckResult(FAIL, "found " + ", ".join(bad_types)) if bad_types
                            else CheckResult(PASS, "all revolute"))

    n_c = len(model.couplings)
    out["no_couplings"] = CheckResult(FAIL if n_c else PASS, f"{n_c} coupling(s)")

    branchers = sorted(n for n, js in children_of.items() if n not in palm and len(js) >= 2)
    out["no_branches"] = CheckResult(FAIL if branchers else PASS, f"{len(branchers)} branching link(s)")

    digit_roots = [j for j in model.joints if j.parent in palm and j.child not in palm]
    n_digits = len(digit_roots)
    out["max_digits"] = CheckResult(FAIL if n_digits > g.MAX_DIGITS else PASS, f"{n_digits} digit(s)")

    per_digit = []
    for rj in digit_roots:
        count, stack = 1, [rj.child]
        while stack:
            for j in children_of.get(stack.pop(), []):
                count += 1
                stack.append(j.child)
        per_digit.append(count)
    longest = max(per_digit, default=0)
    out["joints_per_digit"] = CheckResult(FAIL if longest > g.MAX_JOINTS_PER_DIGIT else PASS,
                                          f"longest digit {longest} joint(s)")

    jointed = g._jointed_palm_joints(model)
    out["max_jointed_palm"] = CheckResult(FAIL if len(jointed) > g.MAX_JOINTED_PALM_BODIES else PASS,
                                          f"{len(jointed)} jointed")

    pj_by_child = g._palm_joint_by_child(model)
    jointed_children = {j.child for j in jointed}
    nested = 0
    for j in jointed:
        cur = pj_by_child.get(j.child)
        cur = cur.parent if cur is not None else None
        seen = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            if cur in jointed_children:
                nested += 1
                break
            pj = pj_by_child.get(cur)
            cur = pj.parent if pj is not None else None
    out["palm_not_nested"] = CheckResult(FAIL if nested else PASS, f"{nested} stacked")

    carrier_of = [g._carrier_of_mount(j.parent, jointed_children, pj_by_child, model.root) for j in digit_roots]
    counts = Counter(c for c in carrier_of if c is not None)
    most = max(counts.values(), default=0)
    out["carrier_one_digit"] = CheckResult(FAIL if most > 1 else PASS,
                                           f"most on one: {most}" if jointed else "no jointed palm body")

    n_root = sum(1 for c in carrier_of if c is None)
    total = n_root + len(jointed)
    out["finger_slots"] = CheckResult(FAIL if total > g.N_FINGERS else PASS, f"{n_root} + {len(jointed)} = {total}")
    return out


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def _worst(pairs: Sequence[Tuple[int, int, float]]) -> float:
    return max((p for _, _, p in pairs), default=0.0)


def _overlap_result(pairs: Sequence[Tuple[int, int, float]], gate: float) -> CheckResult:
    bad = [p for p in pairs if p[2] > gate]
    worst = _worst(pairs) * 1000.0
    value = f"worst {worst:.1f} mm" + (f", {len(bad)} pair(s) over" if bad else "")
    return CheckResult(FAIL if bad else PASS, value)


def evaluate(model: KinematicModel, enabled: Optional[Iterable[str]] = None, stop_early: bool = False) -> Evaluation:
    """Every check on `model`. With `stop_early`, stop at the first failing
    ENABLED check (cheapest first) and mark the rest SKIPPED: what Random
    uses per draw. Without it, everything is computed (the panel readout)."""
    g = ge()
    en = set(CHECK_KEYS if enabled is None else enabled)
    results = _structural(model)
    oracle = g._admit_structural(model)
    unexplained: Tuple[str, ...] = ()
    if not oracle.ok and all(r.status == PASS for r in results.values()):
        unexplained = tuple(oracle.reasons)
    ev = Evaluation(results=results, structural_ok=oracle.ok, unexplained=unexplained)

    def skip_rest():
        for k in PHYSICAL_KEYS:
            results.setdefault(k, CheckResult(SKIPPED, "not computed"))
        return ev

    if stop_early and (unexplained or ev.failing(en)):
        return skip_rest()
    if not oracle.ok:
        for k in PHYSICAL_KEYS:
            results[k] = CheckResult(NA, "simulator cannot build this hand")
        return ev

    design = g.canonicalize(model)
    ev.design = design
    # palm_up's reach sweep is the slow part; default_q and the spawn point do
    # not depend on it (admit itself uses n_sweep=0 for those).
    pu = g.palm_up(design, n_sweep=0)
    ev.pu = pu
    spawn_min = load_env_modules().palm_calibration.MIN_SPAWN_HEIGHT_ABOVE_PALM_M
    gate = g.MAX_REST_PENETRATION_M

    if not stop_early or "spawn_height" in en:
        h = g.spawn_height_above_palm_m(design, pu)
        results["spawn_height"] = CheckResult(FAIL if h < spawn_min else PASS, f"{h * 1000:.1f} mm")
        if stop_early and ev.failing(en):
            return skip_rest()
    if not stop_early or "overlap_zero" in en:
        ev.pairs_q0 = list(g.rest_overlap_pairs(design))
        results["overlap_zero"] = _overlap_result(ev.pairs_q0, gate)
        if stop_early and ev.failing(en):
            return skip_rest()
    if not stop_early or "overlap_reset" in en:
        ev.pairs_reset = list(g.rest_overlap_pairs(design, q=pu.default_q))
        results["overlap_reset"] = _overlap_result(ev.pairs_reset, gate)
        if stop_early and ev.failing(en):
            return skip_rest()
    if not stop_early or "reach" in en:
        pu = g.palm_up(design)
        ev.pu = pu
        n_tips = sum(1 for d in design.finger_digit_id if d is not None)
        n = int(pu.reachable_fingertips)
        results["reach"] = CheckResult(FAIL if n < MIN_REACHING_TIPS else PASS, f"{n} of {n_tips} tip(s)")
    return skip_rest()


# --------------------------------------------------------------------------
# Random: draw designs until every enabled check passes
# --------------------------------------------------------------------------


@dataclass
class SearchResult:
    derivation: Optional[Derivation]
    model: Optional[KinematicModel]
    seed: int
    tries: int
    evaluation: Optional[Evaluation] = None
    cancelled: bool = False


def search(dist, enabled: Iterable[str], start_seed: int, max_tries: int = 3000,
           cancel: Optional[Callable[[], bool]] = None,
           progress: Optional[Callable[[int], None]] = None) -> SearchResult:
    """Seeds start_seed, start_seed+1, ... until a design passes every enabled
    check. A seed whose derivation fails to derive counts as a try."""
    en = set(enabled)
    seed = int(start_seed)
    for k in range(1, max_tries + 1):
        if cancel is not None and cancel():
            return SearchResult(None, None, seed, k - 1, cancelled=True)
        if progress is not None and k % 25 == 0:
            progress(k)
        d = sample_derivation(seed, dist)
        try:
            m = derive(d)
        except ModelError:
            seed += 1
            continue
        ev = evaluate(m, en, stop_early=True)
        if ev.passes(en):
            return SearchResult(d, m, seed, k, ev)
        seed += 1
    return SearchResult(None, None, seed - 1, max_tries)
