"""The viability checks (gviewer/checks.py) against the simulator's own
combined verdict on sampled designs."""

import itertools

import pytest

from gviewer import checks as ck
from gviewer import sources as src
from gviewer.envload import load_env_modules
from hand_sampler.grammar.derive import derive, sample_derivation
from hand_sampler.grammar.kinematics import ModelError
from hand_sampler.grammar.limits import SIMULATOR, GenerationLimits, check

# Sampled under the SIMULATOR limits, every design is buildable, so all four
# physical checks are measured.
CASES = [("G_FULL", 40), ("G_V1", 30), ("G_V3S", 30), ("G_V2S", 15)]


def _designs():
    for variant, n in CASES:
        dist = src.distribution(variant)
        for seed in range(n):
            try:
                yield variant, seed, derive(sample_derivation(seed, dist, limits=SIMULATOR))
            except ModelError:
                continue


DESIGNS = list(_designs())


def test_four_physical_checks():
    assert ck.CHECK_KEYS == ("overlap_zero", "overlap_reset", "spawn_height", "reach")
    assert [c.key for c in ck.CHECKS if c.provisional] == ["spawn_height", "reach"]


@pytest.mark.parametrize("variant,seed,model", DESIGNS, ids=[f"{v}-{s}" for v, s, _ in DESIGNS])
def test_checks_match_the_oracle(variant, seed, model):
    ge = load_env_modules().grammar_envelope
    ev = ck.evaluate(model)
    assert ev.buildable
    rep = ge.viability_report(model)
    # the four together are exactly viability_report's "viable" (structure is guaranteed by the limits)
    assert ev.passes(ck.CHECK_KEYS) == (rep["admitted"] and rep["fingertips_reachable"] >= 2)
    assert ev.results["reach"].value.startswith(f"{rep['fingertips_reachable']} of")
    overlap_fail = any(ev.results[k].status == ck.FAIL for k in ("overlap_zero", "overlap_reset"))
    assert overlap_fail == any("rest-overlap" in r for r in rep["reasons"])
    assert (ev.results["spawn_height"].status == ck.FAIL) == any("spawn height" in r for r in rep["reasons"])


def test_outside_the_simulator_is_na_and_does_not_block():
    ge = load_env_modules().grammar_envelope
    dist = src.distribution("G_FULL")
    n = 0
    for seed in range(60):
        m = derive(sample_derivation(seed, dist))
        if ge._admit_structural(m).ok:
            continue
        ev = ck.evaluate(m)
        assert not ev.buildable and all(r.status == ck.NA for r in ev.results.values())
        assert ev.passes(ck.CHECK_KEYS)
        n += 1
    assert n > 10


def test_early_stop_gives_the_same_verdict():
    subsets = [set(c) for r in range(len(ck.CHECK_KEYS) + 1) for c in itertools.combinations(ck.CHECK_KEYS, r)]
    for _variant, _seed, model in DESIGNS[::3]:
        full = ck.evaluate(model)
        for en in subsets:
            assert ck.evaluate(model, en, stop_early=True).passes(en) == full.passes(en)


def test_search_respects_checks_and_limits():
    dist = src.distribution("G_V3S")
    res = ck.search(dist, ck.CHECK_KEYS, 0, limits=SIMULATOR)
    assert res.derivation is not None and res.evaluation.passes(ck.CHECK_KEYS)
    assert check(res.derivation, SIMULATOR).ok
    assert ck.search(dist, set(), 0, limits=SIMULATOR).tries == 1
    # under no limits every G_FULL draw is accepted at once with the checks off
    res = ck.search(src.distribution("G_FULL"), set(), 3, limits=GenerationLimits())
    assert res.tries == 1 and res.seed == 3
