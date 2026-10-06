"""The segment-length step (``step_segment_length`` and its two directions
``lengthen_segment``/``shorten_segment``): one existing segment moves by
exactly one 5 mm grid step, never past the variant's range, the two
directions undo each other exactly, it runs under the generation limits, and
``EVOLUTION_OPERATORS`` gained it while the old pool stays available as
``EVOLUTION_OPERATORS_V1``."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hand_sampler.grammar.derive import (
    EVOLUTION_OPERATORS,
    EVOLUTION_OPERATORS_V1,
    INVERSE_OF,
    LENGTH_STEP_PAIR,
    SMALL_STEP_OPERATORS,
    TARGETABLE_OPERATORS,
    VariationImpossible,
    apply_operator,
    derive,
    sample_derivation,
    vary,
)
from hand_sampler.grammar.limits import SIMULATOR, check
from hand_sampler.grammar.variants import G_FULL, G_V1, G_V3S

GRID = 0.005


def _segments(d):
    return {s.params["uid"]: s for s in d.steps if s.production in ("Phalanx", "PalmBody")}


def _changed(parent, child):
    a, b = _segments(parent), _segments(child)
    assert set(a) == set(b)
    out = []
    for uid in a:
        if a[uid].params != b[uid].params:
            diff = {k for k in a[uid].params if a[uid].params[k] != b[uid].params.get(k)}
            assert diff == {"length"}, diff
            out.append((uid, a[uid], b[uid]))
    others_a = [s for s in parent.steps if s.production not in ("Phalanx", "PalmBody")]
    others_b = [s for s in child.steps if s.production not in ("Phalanx", "PalmBody")]
    assert others_a == others_b
    return out


def test_pool_gained_the_operator_and_v1_is_frozen():
    assert EVOLUTION_OPERATORS == EVOLUTION_OPERATORS_V1 + ("step_segment_length",)
    assert len(EVOLUTION_OPERATORS_V1) == 17 and "step_segment_length" not in EVOLUTION_OPERATORS_V1
    assert "step_segment_length" not in SMALL_STEP_OPERATORS          # older pools unchanged
    assert INVERSE_OF["lengthen_segment"] == "shorten_segment" and LENGTH_STEP_PAIR == ("lengthen_segment",
                                                                                       "shorten_segment")
    assert {"step_segment_length", "lengthen_segment", "shorten_segment"} <= TARGETABLE_OPERATORS


@pytest.mark.parametrize("dist", [G_FULL, G_V1, G_V3S], ids=["G_FULL", "G_V1", "G_V3S"])
def test_exactly_one_segment_moves_by_5mm_within_range(dist):
    rng = np.random.default_rng(0)
    for seed in range(80):
        d = sample_derivation(seed, dist)
        child = vary(d, rng, dist, operator="step_segment_length")
        (uid, before, after), = _changed(d, child)
        delta = after.params["length"] - before.params["length"]
        assert abs(abs(delta) - GRID) < 1e-12
        lo, hi = dist.link_length_range_m if after.production == "Phalanx" else dist.palm_length_range_m
        assert lo - 1e-12 <= after.params["length"] <= hi + 1e-12
        derive(child)


def _all_at(dist, d, which):
    steps = []
    for s in d.steps:
        if s.production in ("Phalanx", "PalmBody"):
            lo, hi = dist.link_length_range_m if s.production == "Phalanx" else dist.palm_length_range_m
            s = replace(s, params={**s.params, "length": hi if which == "hi" else lo})
        steps.append(s)
    return replace(d, steps=tuple(steps))


def test_bounds_respected():
    rng = np.random.default_rng(1)
    for seed in range(20):
        d = sample_derivation(seed, G_FULL)
        top = _all_at(G_FULL, d, "hi")
        assert apply_operator(top, rng, G_FULL, "lengthen_segment") is None
        for _ in range(5):
            (_, b, a), = _changed(top, vary(top, rng, G_FULL, operator="step_segment_length"))
            assert a.params["length"] < b.params["length"]
        bottom = _all_at(G_FULL, d, "lo")
        assert apply_operator(bottom, rng, G_FULL, "shorten_segment") is None
        for _ in range(5):
            (_, b, a), = _changed(bottom, vary(bottom, rng, G_FULL, operator="step_segment_length"))
            assert a.params["length"] > b.params["length"]
    # a variant whose ranges are a single length: nothing can move
    fixed = replace(G_FULL, link_length_range_m=(0.03, 0.03), palm_length_range_m=(0.03, 0.03))
    d = sample_derivation(0, fixed)
    with pytest.raises(VariationImpossible):
        vary(d, rng, fixed, operator="step_segment_length")


@pytest.mark.parametrize("first", LENGTH_STEP_PAIR)
def test_inverse_pair_restores_exactly(first):
    rng = np.random.default_rng(2)
    n = 0
    for seed in range(100):
        d = sample_derivation(seed, G_FULL)
        child = apply_operator(d, rng, G_FULL, first)
        if child is None:
            continue
        (uid, _, _), = _changed(d, child)
        back = apply_operator(child, rng, G_FULL, INVERSE_OF[first], target=uid)
        assert back is not None and back.steps == d.steps
        n += 1
    assert n >= 95


def test_target_restricts_to_one_segment():
    rng = np.random.default_rng(3)
    d = sample_derivation(5, G_FULL)
    for uid in _segments(d):
        child = apply_operator(d, rng, G_FULL, "step_segment_length", target=uid)
        assert child is not None
        assert [u for u, _, _ in _changed(d, child)] == [uid]


def test_runs_under_limits():
    """No limit concerns lengths, so under SIMULATOR the operator stays
    applicable and its children stay within the limits."""
    rng = np.random.default_rng(4)
    for seed in range(40):
        d = sample_derivation(seed, G_V3S, limits=SIMULATOR)
        for _ in range(10):
            d = vary(d, rng, G_V3S, operator="step_segment_length", limits=SIMULATOR)
            assert check(d, SIMULATOR).ok
