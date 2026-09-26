"""E3 (reachability) / E4 (redundancy) acceptance tests: the five minimal
structural ``vary`` operators (Part A), and the core E3/E4 per-seed
functions run at tiny sizes into tmp dirs via the experiment runner.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hand_sampler.grammar.adapters.json_io import to_json
from hand_sampler.grammar.derive import (
    MINIMAL_STRUCTURAL_OPERATORS,
    OPERATORS,
    SMALL_STEP_OPERATORS,
    VariationImpossible,
    derivation_from_json,
    derivation_to_json,
    derive,
    generate,
    vary,
)
from hand_sampler.grammar.kinematics import validate
from hand_sampler.grammar.experiments.e3_reach import TARGETS, OPERATOR_SETS, e3_reach_seed
from hand_sampler.grammar.experiments.e4_redundancy import (
    VARIANTS,
    OFFSPRING_OP_GROUPS,
    e4_redundancy_seed,
)
from hand_sampler.grammar.experiments.runner import registered_experiments, run_experiment

N_SEEDS = 50


# ---------------------------------------------------------------------------
# Part A: MINIMAL_STRUCTURAL_OPERATORS
# ---------------------------------------------------------------------------


def test_minimal_structural_operators_disjoint_from_default_pool():
    assert set(MINIMAL_STRUCTURAL_OPERATORS).isdisjoint(OPERATORS)
    assert OPERATORS == (
        "resample_parameter", "perturb_parameter", "regrow_subtree", "insert_phalanx",
        "delete_phalanx", "add_digit", "remove_digit",
    ), "default OPERATORS tuple must be unchanged"


def _apply_and_check(opname: str, seed: int, check):
    """Apply ``opname`` to ``generate(seed)`` once; if applicable, run
    ``check(parent_derivation, parent_model, child_derivation, child_model)``
    and assert replay/round-trip; return True iff applied."""
    d0, m0 = generate(seed)
    rng = np.random.default_rng([2024, seed])
    try:
        d1 = vary(d0, rng, operator=opname)
    except VariationImpossible:
        return False
    m1 = derive(d1)
    validate(m1)
    # Replay: deriving the same Derivation twice is byte-identical.
    m1b = derive(d1)
    assert to_json(m1) == to_json(m1b), f"{opname} seed {seed}: replay not exact"
    # JSON round trip of the Derivation itself.
    d1_rt = derivation_from_json(derivation_to_json(d1))
    assert to_json(derive(d1_rt)) == to_json(m1), f"{opname} seed {seed}: derivation JSON round trip changed the model"
    check(d0, m0, d1, m1)
    return True


def test_add_minimal_digit_adds_one_single_phalanx_digit():
    n_applied = 0
    for seed in range(N_SEEDS):
        def check(d0, m0, d1, m1):
            assert d1.steps[next(i for i, s in enumerate(d1.steps) if s.path == "hand")].params["digit_count"] \
                == d0.steps[next(i for i, s in enumerate(d0.steps) if s.path == "hand")].params["digit_count"] + 1
            new_digit_ids = {s.params["digit_id"] for s in d1.steps if s.production == "Digit" and s.params.get("top_level")} \
                - {s.params["digit_id"] for s in d0.steps if s.production == "Digit" and s.params.get("top_level")}
            assert len(new_digit_ids) == 1
            new_id = next(iter(new_digit_ids))
            phalanges = [s for s in d1.steps if s.production == "Phalanx" and s.params["digit_id"] == new_id]
            assert len(phalanges) == 1
            assert phalanges[0].params["module"]["kind"] == "R"
            assert phalanges[0].params["branch_digit_count"] == 0
        if _apply_and_check("add_minimal_digit", seed, check):
            n_applied += 1
    assert n_applied >= 30, f"add_minimal_digit applied on too few seeds ({n_applied}/{N_SEEDS})"


def test_remove_digit_minimal_removes_a_single_phalanx_digit():
    n_applied = 0
    for seed in range(N_SEEDS):
        def check(d0, m0, d1, m1):
            hand0 = next(s for s in d0.steps if s.path == "hand")
            hand1 = next(s for s in d1.steps if s.path == "hand")
            assert hand1.params["digit_count"] == hand0.params["digit_count"] - 1
            removed_ids = {s.params["digit_id"] for s in d0.steps if s.production == "Digit" and s.params.get("top_level")} \
                - {s.params["digit_id"] for s in d1.steps if s.production == "Digit" and s.params.get("top_level")}
            assert len(removed_ids) == 1
        if _apply_and_check("remove_digit_minimal", seed, check):
            n_applied += 1
    assert n_applied >= 10, f"remove_digit_minimal applied on too few seeds ({n_applied}/{N_SEEDS})"


def test_add_palm_body_adds_one_palm_body():
    n_applied = 0
    for seed in range(N_SEEDS):
        def check(d0, m0, d1, m1):
            hand0 = next(s for s in d0.steps if s.path == "hand")
            hand1 = next(s for s in d1.steps if s.path == "hand")
            assert hand1.params["palm_body_count"] == hand0.params["palm_body_count"] + 1
            n_palm0 = sum(1 for b in m0.bodies if b.palm)
            n_palm1 = sum(1 for b in m1.bodies if b.palm)
            assert n_palm1 == n_palm0 + 1
        if _apply_and_check("add_palm_body", seed, check):
            n_applied += 1
    assert n_applied >= 30, f"add_palm_body applied on too few seeds ({n_applied}/{N_SEEDS})"


def test_remove_palm_body_removes_a_leaf_palm_body():
    n_applied = 0
    for seed in range(N_SEEDS):
        def check(d0, m0, d1, m1):
            hand0 = next(s for s in d0.steps if s.path == "hand")
            hand1 = next(s for s in d1.steps if s.path == "hand")
            assert hand1.params["palm_body_count"] == hand0.params["palm_body_count"] - 1
            n_palm0 = sum(1 for b in m0.bodies if b.palm)
            n_palm1 = sum(1 for b in m1.bodies if b.palm)
            assert n_palm1 == n_palm0 - 1
        if _apply_and_check("remove_palm_body", seed, check):
            n_applied += 1
    assert n_applied >= 5, f"remove_palm_body applied on too few seeds ({n_applied}/{N_SEEDS})"


def test_toggle_palm_joint_flips_has_joint_on_one_palm_body():
    n_applied = 0
    for seed in range(N_SEEDS):
        def check(d0, m0, d1, m1):
            palm0 = {s.params["name"]: s.params["has_joint"] for s in d0.steps if s.production == "PalmBody"}
            palm1 = {s.params["name"]: s.params["has_joint"] for s in d1.steps if s.production == "PalmBody"}
            assert set(palm0) == set(palm1)
            flipped = [name for name in palm0 if palm0[name] != palm1[name]]
            assert len(flipped) == 1
        if _apply_and_check("toggle_palm_joint", seed, check):
            n_applied += 1
    assert n_applied >= 30, f"toggle_palm_joint applied on too few seeds ({n_applied}/{N_SEEDS})"


def test_minimal_operators_reachable_via_operators_kwarg():
    n_ok = 0
    for seed in range(20):
        d0, _ = generate(seed)
        rng = np.random.default_rng([seed, 99])
        try:
            d1 = vary(d0, rng, operators=MINIMAL_STRUCTURAL_OPERATORS)
        except VariationImpossible:
            continue
        op_used = d1.lineage[-1][0]
        assert op_used in MINIMAL_STRUCTURAL_OPERATORS
        n_ok += 1
    assert n_ok >= 10, f"MINIMAL_STRUCTURAL_OPERATORS pool succeeded on too few seeds ({n_ok}/20)"


def test_default_operators_and_default_vary_unaffected_by_minimal_ops():
    # Default vary() (no operator/operators kwarg) must never reach a
    # MINIMAL_STRUCTURAL_OPERATORS operator.
    n_ok = 0
    for seed in range(30):
        d0, _ = generate(seed)
        rng = np.random.default_rng(seed)
        try:
            d1 = vary(d0, rng)
        except VariationImpossible:
            continue
        op_used = d1.lineage[-1][0]
        assert op_used in OPERATORS
        assert op_used not in MINIMAL_STRUCTURAL_OPERATORS
        n_ok += 1
    assert n_ok >= 20


# ---------------------------------------------------------------------------
# E3: core function at tiny sizes, via the runner, into tmp dirs.
# ---------------------------------------------------------------------------


def test_e3_reach_seed_returns_all_target_opset_combinations():
    result = e3_reach_seed(0, max_accepted=5, max_attempts_per_step=8, safety_cap=40)
    assert len(result) == len(TARGETS) * len(OPERATOR_SETS)
    for target_name in TARGETS:
        for opset_name in OPERATOR_SETS:
            key = f"{target_name}::{opset_name}"
            assert key in result
            row = result[key]
            for field in ("success", "final_distance", "n_accepted", "n_proposed"):
                assert field in row
            assert row["n_accepted"] <= 5
            assert row["final_distance"] >= 0.0


def test_e3_reach_seed_deterministic():
    r1 = e3_reach_seed(3, max_accepted=8, safety_cap=60)
    r2 = e3_reach_seed(3, max_accepted=8, safety_cap=60)
    assert r1 == r2


def test_e3_reach_writes_result_json_via_runner():
    with tempfile.TemporaryDirectory() as td:
        out_dir = str(Path(td) / "e3_out")
        fn = registered_experiments()["e3_reach"]
        result = run_experiment(
            "e3_reach", fn, params={"max_accepted": 5, "max_attempts_per_step": 8, "safety_cap": 40},
            seeds=list(range(4)), out_dir=out_dir, processes=2,
        )
        result_path = Path(out_dir) / "result.json"
        assert result_path.exists()
        with open(result_path) as f:
            loaded = json.load(f)
        assert loaded["name"] == "e3_reach"
        assert loaded["aggregate"]["n_ok"] == 4
        for key in ("name", "params", "seeds", "per_seed", "aggregate", "grammar_version", "wall_time_s"):
            assert key in loaded


# ---------------------------------------------------------------------------
# E4: core function at tiny sizes, via the runner, into tmp dirs.
# ---------------------------------------------------------------------------


def test_e4_redundancy_seed_shape():
    result = e4_redundancy_seed(0, n_offspring_parents=3)
    assert set(result) == set(VARIANTS)
    for variant_name in VARIANTS:
        entry = result[variant_name]
        assert isinstance(entry["hash"], str)
        assert entry["canonical_reordered"] in (0.0, 1.0)
        assert "offspring" in entry  # seed 0 < n_offspring_parents=3
        for group_name in OFFSPRING_OP_GROUPS:
            og = entry["offspring"][group_name]
            assert og["n_pairs"] == 45  # C(10, 2)
            assert 0 <= og["n_identical_pairs"] <= 45
            assert 0.0 <= og["null_fraction"] <= 1.0

    result_no_offspring = e4_redundancy_seed(5, n_offspring_parents=3)
    for variant_name in VARIANTS:
        assert "offspring" not in result_no_offspring[variant_name]


def test_e4_redundancy_seed_deterministic():
    r1 = e4_redundancy_seed(1, n_offspring_parents=5)
    r2 = e4_redundancy_seed(1, n_offspring_parents=5)
    assert r1 == r2


def test_e4_redundancy_writes_result_json_via_runner():
    with tempfile.TemporaryDirectory() as td:
        out_dir = str(Path(td) / "e4_out")
        fn = registered_experiments()["e4_redundancy"]
        result = run_experiment(
            "e4_redundancy", fn, params={"n_offspring_parents": 3}, seeds=list(range(6)),
            out_dir=out_dir, processes=2,
        )
        result_path = Path(out_dir) / "result.json"
        assert result_path.exists()
        with open(result_path) as f:
            loaded = json.load(f)
        assert loaded["name"] == "e4_redundancy"
        assert loaded["aggregate"]["n_ok"] == 6
        for key in ("name", "params", "seeds", "per_seed", "aggregate", "grammar_version", "wall_time_s"):
            assert key in loaded


def test_e4_offspring_op_groups_use_default_and_small_step_pools():
    assert OFFSPRING_OP_GROUPS["default"] == OPERATORS
    assert OFFSPRING_OP_GROUPS["small"] == SMALL_STEP_OPERATORS
