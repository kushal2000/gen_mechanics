"""E11 acceptance tests: ``coverage.coverage``'s ``relax`` argument (support
widening) and the ``e11_support_widening`` experiment built on it.

1. Relaxation is monotone: for the cumulative L0..L8 levels defined by
   ``e11_support_widening``, widening the relax set never removes a hand (or
   a generated design) from support, and never increases its remaining
   out-of-support/missing-construct item count.
2. Every one of the grammar's own generated designs (default distribution)
   stays in support at every level -- a relaxation only ever loosens a
   check, so this must hold as long as it holds at L0 (which
   ``test_coverage_covers_own_generated_output`` in test_coverage.py already
   proves).
3. A synthetic model with an off-grid link length is out of support at L0
   and in support at L3 (once ``length_grid_1mm`` is in effect), exactly as
   specified for E11.
"""

from __future__ import annotations

import pytest

from hand_sampler.grammar.coverage import coverage
from hand_sampler.grammar.derive import generate
from hand_sampler.grammar.distributions import DEFAULT_DISTRIBUTION
from hand_sampler.grammar.experiments.e11_support_widening import (
    LEVEL_ORDER,
    LEVELS,
    _load_hand_models,
)
from hand_sampler.grammar.kinematics import Body, Frame, Joint, KinematicModel, Pose

N_GENERATED_SEEDS = 40


def _remaining(res) -> int:
    return len(res.missing_constructs) + len(res.out_of_support)


# ---------------------------------------------------------------------------
# 1. Monotonicity across the cumulative L0..L8 levels.
# ---------------------------------------------------------------------------


def _assert_monotone_across_levels(model, dist=DEFAULT_DISTRIBUTION, label: str = ""):
    prev_in_support = False
    prev_remaining = None
    for level in LEVEL_ORDER:
        res = coverage(model, dist, relax=LEVELS[level])
        remaining = _remaining(res)
        if prev_remaining is not None:
            assert remaining <= prev_remaining, (
                f"{label} at {level}: remaining item count increased ({prev_remaining} -> {remaining}) "
                "under strictly more relaxation"
            )
        if prev_in_support:
            assert res.in_support, f"{label} at {level}: in_support went True -> False under more relaxation"
        prev_in_support = res.in_support
        prev_remaining = remaining


def test_relaxation_monotone_over_real_hands():
    hands = [e for e in _load_hand_models() if e["model"] is not None]
    assert len(hands) >= 8, "expected most manifest hands to resolve on this machine"
    for entry in hands:
        _assert_monotone_across_levels(entry["model"], label=entry["id"])


@pytest.mark.parametrize("seed", range(N_GENERATED_SEEDS))
def test_relaxation_monotone_over_generated_designs(seed):
    _derivation, model = generate(seed, DEFAULT_DISTRIBUTION)
    _assert_monotone_across_levels(model, label=f"seed={seed}")


def test_relaxation_monotone_single_flag_additions():
    """A finer-grained monotonicity check than the cumulative levels: adding
    ANY single named relaxation on top of an arbitrary existing relax set
    must never turn an in-support hand out of support, over every available
    real hand and a handful of generated designs."""
    from hand_sampler.grammar.coverage import RELAX_NAMES

    models = [e["model"] for e in _load_hand_models() if e["model"] is not None]
    for seed in range(10):
        _derivation, model = generate(seed, DEFAULT_DISTRIBUTION)
        models.append(model)

    base_sets = [frozenset(), frozenset({"rest_bend"}), frozenset({"rest_bend", "limits_continuous"})]
    for model in models:
        for base in base_sets:
            base_res = coverage(model, DEFAULT_DISTRIBUTION, relax=base)
            if not base_res.in_support:
                continue
            for name in RELAX_NAMES:
                widened = coverage(model, DEFAULT_DISTRIBUTION, relax=base | {name})
                assert widened.in_support, (
                    f"adding relax {name!r} to {sorted(base)} removed an in-support model from support"
                )


# ---------------------------------------------------------------------------
# 2. Generated designs stay in support at every level.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(N_GENERATED_SEEDS))
def test_generated_designs_in_support_every_level(seed):
    _derivation, model = generate(seed, DEFAULT_DISTRIBUTION)
    for level in LEVEL_ORDER:
        res = coverage(model, DEFAULT_DISTRIBUTION, relax=LEVELS[level])
        assert res.in_support, f"seed={seed} level={level}: out_of_support={res.out_of_support}"


# ---------------------------------------------------------------------------
# 3. Synthetic off-grid-length model: out at L0, in at L3.
# ---------------------------------------------------------------------------


def _synthetic_offgrid_length_model() -> KinematicModel:
    """One digit, one phalanx (``d1p1``), mounted directly on the (palm)
    root: a single revolute joint with axis/limits both drawn from
    ``DEFAULT_DISTRIBUTION``'s own choice sets (so the only thing wrong with
    this model is its link length), and a ``d1p1_tip`` frame whose z is
    0.017 m -- inside ``link_length_range_m`` (0.015, 0.080) but off the
    default 5 mm grid (15, 20, 25, ... mm), while landing exactly on a 1 mm
    grid from the same 0.015 m origin (2 mm past it)."""
    lo_deg, hi_deg = DEFAULT_DISTRIBUTION.revolute_limit_choices_deg[0]
    import math
    limits = (lo_deg * math.pi / 180.0, hi_deg * math.pi / 180.0)
    bodies = (Body(name="root", palm=True), Body(name="d1p1"))
    joints = (
        Joint(name="d1p1_j", type="revolute", parent="root", child="d1p1", axis=(1.0, 0.0, 0.0), limits=limits),
    )
    frames = (Frame(name="d1p1_tip", body="d1p1", pose=Pose(xyz=(0.0, 0.0, 0.017))),)
    return KinematicModel(name="offgrid_length_test", root="root", bodies=bodies, joints=joints, frames=frames)


def test_synthetic_offgrid_length_out_at_l0_in_at_l3():
    model = _synthetic_offgrid_length_model()

    res_l0 = coverage(model, DEFAULT_DISTRIBUTION, relax=LEVELS["L0"])
    assert res_l0.topology_expressible is True
    assert res_l0.in_support is False
    assert any(item.startswith("link_length_off_grid") for item in res_l0.out_of_support), res_l0.out_of_support

    for level in ("L1", "L2"):
        res = coverage(model, DEFAULT_DISTRIBUTION, relax=LEVELS[level])
        assert res.in_support is False, f"{level}: expected still out of support (no length relaxation yet)"

    res_l3 = coverage(model, DEFAULT_DISTRIBUTION, relax=LEVELS["L3"])
    assert res_l3.in_support is True, res_l3.out_of_support
