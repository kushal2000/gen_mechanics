"""Grammar 0.5, iteration C (I18) acceptance tests:

  1. ``delete_phalanx`` structural invariant (I18 fix 1): over 500 random
     applications (``G_FULL_INS``), the derived model's count of top-level
     digits (``phenodist._digit_count`` -- movable chains leaving palm
     bodies) never changes, and no branch digit ends up mounted on a palm
     body (branch digits must keep ``top_level=False`` and never mount on a
     palm body).
  2. ``delete_phalanx`` refusal: a hand whose only digit has exactly 1
     phalanx is never a deletion candidate -- ``apply_operator`` returns
     ``None`` and ``vary`` raises ``VariationImpossible`` -- so digit
     removal stays the job of ``remove_digit_minimal``/``remove_branch_digit``.
  3. ``remove_palm_body_empty`` (I18 fix 2): refuses (returns ``None``) on
     any palm body that is not a leaf (hosts a ``PalmBody`` child or a
     ``Digit`` mount); on a leaf body it succeeds and is the exact inverse
     of ``add_palm_body`` (undoing ``add_palm_body`` always recovers the
     parent's own ``phenotype_hash``).
  4. ``e12_balance.run`` (tiny params) prints BOTH drift regimes
     ("Boundary regime", "Stationary regime") with PASS/FAIL lines.
"""

from __future__ import annotations

import numpy as np
import pytest

from hand_sampler.grammar.canonical import phenotype_hash
from hand_sampler.grammar.derive import (
    VariationImpossible, apply_operator, derive, generate, vary,
)
from hand_sampler.grammar.experiments import e12_balance
from hand_sampler.grammar.experiments.starts import build_start
from hand_sampler.grammar.phenodist import _digit_count
from hand_sampler.grammar.variants import G_FULL_INS, G_NOBRANCH_INS

# ---------------------------------------------------------------------------
# 1. delete_phalanx: top-level digit count invariant + no branch on a palm
#    body.
# ---------------------------------------------------------------------------


def test_delete_phalanx_preserves_top_level_digit_count_and_branch_mounts():
    n_checked = 0
    for seed in range(5000):
        derivation, model = generate(seed, G_FULL_INS)
        rng = np.random.default_rng([seed, 0xD317])
        try:
            child = vary(derivation, rng, G_FULL_INS, operator="delete_phalanx")
        except VariationImpossible:
            continue
        n_checked += 1
        child_model = derive(child)
        assert _digit_count(model) == _digit_count(child_model), (
            f"seed={seed}: delete_phalanx changed the top-level digit count "
            f"({_digit_count(model)} -> {_digit_count(child_model)})"
        )
        palm_names = {b.name for b in child_model.bodies if b.palm}
        for s in child.steps:
            if s.production == "Digit" and not s.params.get("top_level", True):
                assert s.params.get("top_level") is False
                assert s.params["mount"] not in palm_names, (
                    f"seed={seed}: branch digit {s.params['digit_id']!r} ended up "
                    f"mounted on palm body {s.params['mount']!r}"
                )
        if n_checked >= 500:
            break
    assert n_checked >= 500, f"only {n_checked} delete_phalanx applications succeeded in 5000 seeds"


# ---------------------------------------------------------------------------
# 2. delete_phalanx: refuses on a hand whose only digit has 1 phalanx.
# ---------------------------------------------------------------------------


def test_delete_phalanx_refuses_single_phalanx_only_hand():
    n_tried = 0
    for seed in range(200):
        derivation, digit_count, max_phalanx, reduced_fully = build_start(
            seed, G_NOBRANCH_INS, max_digits=1, max_phalanges=1
        )
        if not (reduced_fully and digit_count == 1 and max_phalanx == 1):
            continue
        n_tried += 1
        rng = np.random.default_rng([seed, 0x4242])
        result = apply_operator(derivation, rng, G_NOBRANCH_INS, "delete_phalanx")
        assert result is None, (
            f"seed={seed}: delete_phalanx should refuse a hand with only one, "
            f"single-phalanx digit"
        )
        with pytest.raises(VariationImpossible):
            vary(derivation, rng, G_NOBRANCH_INS, operator="delete_phalanx")
        if n_tried >= 10:
            break
    assert n_tried >= 5, "could not build enough single-digit, single-phalanx starts to test refusal"


# ---------------------------------------------------------------------------
# 3. remove_palm_body_empty: leaf-only, exact inverse of add_palm_body.
# ---------------------------------------------------------------------------


def test_remove_palm_body_empty_leaf_only_and_reverses_add_palm_body():
    n_nonleaf_refused = 0
    n_leaf_removed = 0
    n_reversible = 0
    for seed in range(500):
        derivation, model = generate(seed, G_FULL_INS)
        palm_steps = [s for s in derivation.steps if s.production == "PalmBody"]
        parent_names = {s.params["parent"] for s in palm_steps}
        mount_names = {s.params["mount"] for s in derivation.steps if s.production == "Digit"}
        nonleaf = [s for s in palm_steps if s.params["name"] in parent_names or s.params["name"] in mount_names]

        if nonleaf:
            rng = np.random.default_rng([seed, 0x111])
            result = apply_operator(
                derivation, rng, G_FULL_INS, "remove_palm_body_empty", target=nonleaf[0].params["uid"]
            )
            assert result is None, f"seed={seed}: remove_palm_body_empty should refuse a non-leaf palm body"
            n_nonleaf_refused += 1

        rng2 = np.random.default_rng([seed, 0x222])
        child = apply_operator(derivation, rng2, G_FULL_INS, "add_palm_body")
        if child is None:
            continue
        parent_uids = {s.params["uid"] for s in derivation.steps if s.production == "PalmBody"}
        child_uids = {s.params["uid"] for s in child.steps if s.production == "PalmBody"}
        new_uids = child_uids - parent_uids
        assert len(new_uids) == 1
        new_uid = next(iter(new_uids))

        grandchild = apply_operator(child, rng2, G_FULL_INS, "remove_palm_body_empty", target=new_uid)
        assert grandchild is not None, f"seed={seed}: remove_palm_body_empty failed to undo add_palm_body"
        n_leaf_removed += 1
        if phenotype_hash(derive(grandchild)) == phenotype_hash(model):
            n_reversible += 1

    assert n_nonleaf_refused >= 5, f"only {n_nonleaf_refused} non-leaf refusals observed"
    assert n_leaf_removed >= 5, f"only {n_leaf_removed} leaf removals observed"
    assert n_reversible == n_leaf_removed, (
        f"remove_palm_body_empty is not an exact inverse of add_palm_body: "
        f"{n_reversible}/{n_leaf_removed} recovered the parent's phenotype_hash"
    )


# ---------------------------------------------------------------------------
# 4. e12_balance.run (tiny params) prints both drift regimes with PASS/FAIL
#    lines.
# ---------------------------------------------------------------------------


def test_e12_balance_tiny_run_prints_both_drift_regimes(tmp_path):
    out_dir = tmp_path / "E12_balance"
    result = e12_balance.run(
        out_dir=str(out_dir), drift_seeds=3, drift_walk_length=6, locality_seeds=3,
        reversibility_n_parents=5, histogram_n=20, redundancy_n=20, allow_dirty=True,
    )
    assert "boundary" in result["drift"] and "stationary" in result["drift"]

    summary = (out_dir / "summary.md").read_text()
    assert "Boundary regime" in summary
    assert "Stationary regime" in summary

    boundary_idx = summary.index("Boundary regime")
    stationary_idx = summary.index("Stationary regime")
    locality_idx = summary.index("## (b)")
    assert boundary_idx < stationary_idx < locality_idx

    boundary_section = summary[boundary_idx:stationary_idx]
    stationary_section = summary[stationary_idx:locality_idx]
    for section, name in ((boundary_section, "boundary"), (stationary_section, "stationary")):
        assert "PASS" in section or "FAIL" in section, f"{name} regime section has no PASS/FAIL line"
