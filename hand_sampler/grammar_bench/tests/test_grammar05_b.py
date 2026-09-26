"""Grammar 0.5, iteration B acceptance tests:

  1. ``add_branch_digit``/``remove_branch_digit``: structural validity
     (a branch digit really is mounted on a phalanx body, ``top_level`` is
     False, its host phalanx's own ``branch_digit_count`` is kept accurate)
     and replay (``derive`` is a pure function of the derivation -- calling
     it twice on the same varied derivation gives byte-identical models).
  2. The ``target`` argument on every operator in ``derive.TARGETABLE_OPERATORS``
     restricts that operator to the named uid.
  3. Reversibility: for every ``derive.EVOLUTION_PAIRS`` pair, on 100 parents
     each (fast version of the 500-parent E12 deliverable run), applying the
     growth move then the matching shrink move (targeted at the material the
     growth move just added) recovers the parent's own ``phenotype_hash`` in
     >= 95% of the cases where the growth move was applicable.
  4. Immigrant strata coverage (``immigrants.py``): under ``G_FULL_INS``,
     over 400 immigrants, every one of the grid's strata has >= 1 sample
     and the max/min occupancy ratio is <= 3.
  5. ``e12_balance.run`` (tiny params) writes ``result.json``/``summary.md``
     with a PASS/FAIL line for every one of its 5 criteria.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from hand_sampler.grammar.canonical import phenotype_hash
from hand_sampler.grammar.derive import (
    EVOLUTION_PAIRS, TARGETABLE_OPERATORS, VariationImpossible, apply_operator, derive, generate,
    sample_derivation, vary,
)
from hand_sampler.grammar.experiments import e12_balance
from hand_sampler.grammar.immigrants import all_strata, stratified_population, stratum_occupancy
from hand_sampler.grammar.variants import G_FULL_INS, G_NOBRANCH_INS

DIST_VARIANTS = {"G_FULL_INS": G_FULL_INS, "G_NOBRANCH_INS": G_NOBRANCH_INS}


# ---------------------------------------------------------------------------
# 1. add_branch_digit / remove_branch_digit: structural validity + replay.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_add_branch_digit_structural_validity_and_replay(seed):
    derivation, _ = generate(seed, G_FULL_INS)
    rng = np.random.default_rng([seed, 1])
    try:
        child = vary(derivation, rng, G_FULL_INS, operator="add_branch_digit")
    except VariationImpossible:
        pytest.skip(f"seed={seed}: add_branch_digit not applicable")

    branch_digits = [
        s for s in child.steps
        if s.production == "Digit" and not s.params.get("top_level", True)
    ]
    assert branch_digits, "expected at least one branch digit after add_branch_digit"
    new_branch = max(branch_digits, key=lambda s: s.params["uid"])
    host_body = new_branch.params["mount"]
    assert host_body.startswith("d") and "p" in host_body[1:]

    host_phalanx = next(
        s for s in child.steps
        if s.production == "Phalanx" and f"d{s.params['digit_id']}p{s.params['p'] + 1}" == host_body
    )
    assert host_phalanx.params["branch_digit_count"] >= 1

    # Replay: derive() is pure.
    model_a = derive(child)
    model_b = derive(child)
    assert phenotype_hash(model_a) == phenotype_hash(model_b)


@pytest.mark.parametrize("seed", range(20))
def test_remove_branch_digit_decrements_host_branch_count(seed):
    derivation, _ = generate(seed, G_FULL_INS)
    rng = np.random.default_rng([seed, 2])
    try:
        grown = vary(derivation, rng, G_FULL_INS, operator="add_branch_digit")
    except VariationImpossible:
        pytest.skip(f"seed={seed}: add_branch_digit not applicable")

    branch_digits = [s for s in grown.steps if s.production == "Digit" and not s.params.get("top_level", True)]
    new_branch = max(branch_digits, key=lambda s: s.params["uid"])
    host_body = new_branch.params["mount"]
    host_before = next(
        s for s in grown.steps
        if s.production == "Phalanx" and f"d{s.params['digit_id']}p{s.params['p'] + 1}" == host_body
    )
    before_count = host_before.params["branch_digit_count"]

    shrunk = apply_operator(grown, rng, G_FULL_INS, "remove_branch_digit", target=new_branch.params["uid"])
    assert shrunk is not None
    host_after = next(
        s for s in shrunk.steps
        if s.production == "Phalanx" and f"d{s.params['digit_id']}p{s.params['p'] + 1}" == host_body
    )
    assert host_after.params["branch_digit_count"] == before_count - 1


# ---------------------------------------------------------------------------
# 2. target argument on TARGETABLE_OPERATORS.
# ---------------------------------------------------------------------------


def test_target_argument_restricts_to_named_uid():
    rng = np.random.default_rng(42)
    hits = 0
    for seed in range(60):
        derivation, _ = generate(seed, G_FULL_INS)
        palm_bodies = [s for s in derivation.steps if s.production == "PalmBody"]
        if len(palm_bodies) < 2:
            continue
        target = palm_bodies[0].params["uid"]
        child = apply_operator(derivation, rng, G_FULL_INS, "toggle_palm_joint", target=target)
        if child is None:
            continue
        hits += 1
        toggled = [
            s for s in child.steps
            if s.production == "PalmBody"
            and (s.params["has_joint"] != next(p for p in palm_bodies if p.params["uid"] == s.params["uid"]).params["has_joint"])  # noqa: E501
        ]
        assert len(toggled) == 1
        assert toggled[0].params["uid"] == target
    assert hits >= 10, f"only {hits} applicable toggle_palm_joint(target=...) draws over 60 seeds"


def test_targetable_operators_all_accept_target_kwarg():
    import inspect
    from hand_sampler.grammar import derive as derive_mod

    for op in TARGETABLE_OPERATORS:
        fn = derive_mod._OPERATOR_FNS[op]  # noqa: SLF001 -- whitebox check of the contract itself.
        sig = inspect.signature(fn)
        assert "target" in sig.parameters, f"{op} does not accept a target= kwarg"
        assert sig.parameters["target"].default is None


# ---------------------------------------------------------------------------
# 3. Reversibility >= 95% on 100 parents per pair (fast version of E12's
#    500-parent deliverable run).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dist_name", list(DIST_VARIANTS))
def test_reversibility_95pct_100_parents(dist_name):
    dist = DIST_VARIANTS[dist_name]
    for pair_idx, (growth, shrink) in enumerate(EVOLUTION_PAIRS):
        n_applicable = 0
        n_ok = 0
        for i in range(100):
            seed = 50_000 + i
            result = e12_balance._reversibility_one(seed, dist, growth, shrink, pair_idx)  # noqa: SLF001
            if result is None:
                continue
            n_applicable += 1
            if result:
                n_ok += 1
        if n_applicable == 0:
            continue  # not applicable at all in this (small) sample -- nothing to assert.
        rate = n_ok / n_applicable
        label = f"{growth}/{shrink}" if growth != shrink else growth
        assert rate >= 0.95, f"{dist_name}/{label}: reversibility rate {rate} ({n_ok}/{n_applicable})"


# ---------------------------------------------------------------------------
# 4. Immigrant strata coverage.
# ---------------------------------------------------------------------------


def test_immigrant_strata_coverage():
    strata = all_strata(G_FULL_INS)
    population = stratified_population(400, seed=0, dist=G_FULL_INS)
    assert len(population) == 400
    occ = stratum_occupancy(population, G_FULL_INS, strata=strata)
    counts = list(occ.values())
    empty = [k for k, c in occ.items() if c == 0]

    assert not empty, f"empty strata: {empty}"
    assert max(counts) / min(counts) <= 3.0, f"occupancy ratio too wide: {sorted(counts)}"


def test_sample_immigrant_lands_in_requested_stratum():
    rng = np.random.default_rng(7)
    strata = all_strata(G_FULL_INS)
    n_ok = 0
    n_tried = 0
    for _ in range(30):
        from hand_sampler.grammar.immigrants import sample_immigrant, stratum_of

        idx = int(rng.integers(0, len(strata)))
        target = strata[idx]
        derivation = sample_immigrant(rng, G_FULL_INS, strata=[target])
        n_tried += 1
        if derivation is None:
            continue
        model = derive(derivation)
        n_ok += 1
        assert stratum_of(model) == target
    assert n_ok >= 1


# ---------------------------------------------------------------------------
# 5. e12_balance.run (tiny params) writes PASS/FAIL lines.
# ---------------------------------------------------------------------------


def test_e12_balance_tiny_run_writes_pass_fail_lines(tmp_path):
    out_dir = tmp_path / "E12_balance"
    result = e12_balance.run(
        out_dir=str(out_dir), drift_seeds=3, drift_walk_length=6, locality_seeds=3,
        reversibility_n_parents=5, histogram_n=20, redundancy_n=20, allow_dirty=True,
    )
    assert (out_dir / "result.json").is_file()
    assert (out_dir / "summary.md").is_file()
    summary = (out_dir / "summary.md").read_text()
    for label in ("(a)", "(b)", "(c)", "(d)", "(e)"):
        assert f"PASS/FAIL {label}" in summary, f"missing PASS/FAIL line for {label}"
        assert "PASS" in summary or "FAIL" in summary
    loaded = json.loads((out_dir / "result.json").read_text())
    assert loaded["name"] == "e12_balance"
    for key in ("drift", "locality", "reversibility", "histograms", "redundancy"):
        assert key in result
