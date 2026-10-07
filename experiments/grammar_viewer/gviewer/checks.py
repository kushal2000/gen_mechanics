"""Viability checks: the physical properties generation cannot guarantee.

The hand grammar has three layers (see hand_sampler/grammar/limits.py):
what the grammar can express, the generation limits that sampling and
mutation obey by construction (the simulator's envelope shape is the
`SIMULATOR` preset), and these viability checks, which can only be measured
on a generated hand:

- `overlap_zero`: no two capsules interpenetrate by more than 3 mm with every
  joint at zero;
- `overlap_reset`: the same at the reset pose every episode starts from;
- `spawn_height` (PROVISIONAL, to be reworked): the cube's spawn point is at
  least 20 mm above the palm once the hand is turned palm-up;
- `reach` (PROVISIONAL, to be reworked): at least 2 fingertips come within
  5 cm of the spawn point in a random joint sweep.

Each is computed with the simulator's own functions and thresholds
(`canonicalize`, `palm_up`, `rest_overlap_pairs`, `spawn_height_above_palm_m`
from grammar_envelope.py, loaded by file path through envload), exactly as
`viability_report` computes them. They need the simulator's 32-slot model of
the hand, which exists only for hands within the SIMULATOR limits
(`_admit_structural`); on any other hand (sampled under looser limits) they
read n/a and do not block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from hand_sampler.grammar.derive import Derivation, derive, sample_derivation
from hand_sampler.grammar.kinematics import KinematicModel, ModelError

from .envload import load_env_modules

PASS, FAIL, NA, SKIPPED = "pass", "fail", "n/a", "skipped"
MIN_REACHING_TIPS = 2


def ge():
    return load_env_modules().grammar_envelope


@dataclass(frozen=True)
class Check:
    key: str
    short: str          # one short line in the panel
    why: str            # hover text
    provisional: bool = False


def _checks() -> Tuple[Check, ...]:
    gate_mm = ge().MAX_REST_PENETRATION_M * 1000.0
    spawn_mm = load_env_modules().palm_calibration.MIN_SPAWN_HEIGHT_ABOVE_PALM_M * 1000.0
    return (
        Check("overlap_zero", "fingers don't overlap (open)",
              f"With every joint at 0 (hand open), no two links interpenetrate by more than {gate_mm:.0f} mm. "
              "Deeper starting overlaps make the physics engine push links apart violently."),
        Check("overlap_reset", "fingers don't overlap (start pose)",
              f"The same at the start pose of every episode (every joint 35% of the way through its range)."),
        Check("spawn_height", "object starts above palm",
              f"With the hand palm-up, the object's start point is at least {spawn_mm:.0f} mm above the palm, so "
              "the hand holds it against gravity. Provisional, to be reworked.", provisional=True),
        Check("reach", "\u22652 fingertips reach object",
              "In a random sweep of the joints, at least 2 fingertips come within 5 cm of the object's start "
              "point. Provisional, to be reworked.", provisional=True),
    )


CHECKS: Tuple[Check, ...] = _checks()
CHECK_KEYS: Tuple[str, ...] = tuple(c.key for c in CHECKS)
CHECK_BY_KEY: Dict[str, Check] = {c.key: c for c in CHECKS}


@dataclass(frozen=True)
class CheckResult:
    status: str        # PASS | FAIL | NA | SKIPPED
    value: str


@dataclass
class Evaluation:
    results: Dict[str, CheckResult]
    buildable: bool                          # `_admit_structural(model).ok`: the simulator can build it
    design: object = None                    # grammar_envelope.EnvelopeDesign
    pu: object = None                        # grammar_envelope.PalmUpResult (full sweep unless stopped early)
    pairs_q0: List[Tuple[int, int, float]] = field(default_factory=list)
    pairs_reset: List[Tuple[int, int, float]] = field(default_factory=list)

    def failing(self, enabled: Iterable[str]) -> List[str]:
        en = set(enabled)
        return [k for k in CHECK_KEYS if k in en and k in self.results and self.results[k].status == FAIL]

    def passes(self, enabled: Iterable[str]) -> bool:
        return not self.failing(enabled)


def _worst(pairs: Sequence[Tuple[int, int, float]]) -> float:
    return max((p for _, _, p in pairs), default=0.0)


def _overlap_result(pairs: Sequence[Tuple[int, int, float]], gate: float) -> CheckResult:
    bad = [p for p in pairs if p[2] > gate]
    return CheckResult(FAIL if bad else PASS, f"{_worst(pairs) * 1000.0:.1f} mm")


def evaluate(model: KinematicModel, enabled: Optional[Iterable[str]] = None, stop_early: bool = False) -> Evaluation:
    """Every check on `model`. With `stop_early`, stop at the first failing
    ENABLED check (cheapest first) and mark the rest SKIPPED: what Random
    uses per draw. Without it, everything is computed (the panel readout)."""
    g = ge()
    en = set(CHECK_KEYS if enabled is None else enabled)
    results: Dict[str, CheckResult] = {}
    buildable = g._admit_structural(model).ok
    ev = Evaluation(results=results, buildable=buildable)
    if not buildable:
        for k in CHECK_KEYS:
            results[k] = CheckResult(NA, "outside simulator")
        return ev

    def skip_rest():
        for k in CHECK_KEYS:
            results.setdefault(k, CheckResult(SKIPPED, "not computed"))
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
        results["spawn_height"] = CheckResult(FAIL if h < spawn_min else PASS, f"{h * 1000:.0f} mm")
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
        results["reach"] = CheckResult(FAIL if n < MIN_REACHING_TIPS else PASS, f"{n} of {n_tips}")
    return skip_rest()


# --------------------------------------------------------------------------
# Random: draw designs (within the limits by construction) until every
# enabled viability check passes
# --------------------------------------------------------------------------


@dataclass
class SearchResult:
    derivation: Optional[Derivation]
    model: Optional[KinematicModel]
    seed: int
    tries: int
    evaluation: Optional[Evaluation] = None
    cancelled: bool = False


def search(dist, enabled: Iterable[str], start_seed: int, max_tries: int = 3000, limits=None,
           cancel: Optional[Callable[[], bool]] = None,
           progress: Optional[Callable[[int], None]] = None) -> SearchResult:
    """Seeds start_seed, start_seed+1, ... sampled under `limits` until a
    design passes every enabled viability check. Every sample is within the
    limits by construction, so `tries` counts viability rejections only (plus
    the accepted design); a seed that fails to derive is skipped uncounted."""
    en = set(enabled)
    seed = int(start_seed)
    tries = 0
    for _ in range(4 * max_tries):
        if tries >= max_tries:
            break
        if cancel is not None and cancel():
            return SearchResult(None, None, seed, tries, cancelled=True)
        d = sample_derivation(seed, dist, limits=limits)
        try:
            m = derive(d)
        except ModelError:
            seed += 1
            continue
        tries += 1
        if progress is not None and tries % 25 == 0:
            progress(tries)
        ev = evaluate(m, en, stop_early=True)
        if ev.passes(en):
            return SearchResult(d, m, seed, tries, ev)
        seed += 1
    return SearchResult(None, None, seed - 1, tries)
