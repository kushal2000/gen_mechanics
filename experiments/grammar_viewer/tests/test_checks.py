"""The per-check evaluation (gviewer/checks.py) against the simulator's own
combined verdicts on sampled designs."""

import itertools

import pytest

from gviewer import checks as ck
from gviewer import sources as src
from gviewer.envload import load_env_modules
from hand_sampler.grammar.derive import derive, sample_derivation
from hand_sampler.grammar.kinematics import ModelError

# G_FULL and G_NOBRANCH exercise every structural check; the V variants reach
# the physical ones (they pass the structural gate by construction).
CASES = [("G_FULL", 80), ("G_NOBRANCH", 40), ("G_V1", 20), ("G_V3S", 20)]


def _designs():
    for variant, n in CASES:
        dist = src.distribution(variant)
        for seed in range(n):
            try:
                yield variant, seed, derive(sample_derivation(seed, dist))
            except ModelError:
                continue


DESIGNS = list(_designs())


def test_every_check_is_listed_once():
    assert len(set(ck.CHECK_KEYS)) == len(ck.CHECKS) == 13
    assert set(ck.STRUCTURAL_KEYS) | set(ck.PHYSICAL_KEYS) == set(ck.CHECK_KEYS)


@pytest.mark.parametrize("variant,seed,model", DESIGNS, ids=[f"{v}-{s}" for v, s, _ in DESIGNS])
def test_checks_match_the_oracle(variant, seed, model):
    ge = load_env_modules().grammar_envelope
    ev = ck.evaluate(model)
    structural = ge._admit_structural(model)
    # the nine structural checks together are exactly _admit_structural
    assert all(ev.results[k].status == ck.PASS for k in ck.STRUCTURAL_KEYS) == structural.ok
    assert not ev.unexplained
    # all thirteen together are exactly viability_report's "viable"
    rep = ge.viability_report(model)
    viable = rep["admitted"] and (rep["fingertips_reachable"] or 0) >= 2
    assert ev.passes(ck.CHECK_KEYS) == viable
    if structural.ok:
        assert ev.results["reach"].value.startswith(f"{rep['fingertips_reachable']} of")
        overlap_fail = any(ev.results[k].status == ck.FAIL for k in ("overlap_zero", "overlap_reset"))
        assert overlap_fail == any("rest-overlap" in r for r in rep["reasons"])
        assert (ev.results["spawn_height"].status == ck.FAIL) == any("spawn height" in r for r in rep["reasons"])
    else:
        assert all(ev.results[k].status == ck.NA for k in ck.PHYSICAL_KEYS)


def test_early_stop_gives_the_same_verdict():
    """Random's stop-early path (palm_up without its sweep until reach is
    needed) agrees with the full evaluation for every subset of the physical
    checks plus a few structural ones."""
    subsets = [set(c) for r in range(len(ck.PHYSICAL_KEYS) + 1) for c in itertools.combinations(ck.PHYSICAL_KEYS, r)]
    subsets += [set(ck.STRUCTURAL_KEYS) | s for s in subsets[:4]]
    for _variant, _seed, model in DESIGNS[::3]:
        full = ck.evaluate(model)
        for en in subsets:
            assert ck.evaluate(model, en, stop_early=True).passes(en) == full.passes(en)


def test_search_respects_the_enabled_set():
    dist = src.distribution("G_V3S")
    res = ck.search(dist, ck.CHECK_KEYS, 0)
    assert res.derivation is not None and res.evaluation.passes(ck.CHECK_KEYS)
    assert ck.search(dist, set(), 0).tries == 1
