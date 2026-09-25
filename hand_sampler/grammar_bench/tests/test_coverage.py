"""Iteration-4 acceptance tests for ``hand_sampler/grammar/coverage.py``
(structural inventory + support audit) and ``grammar_bench/evaluate.py``
(the pilot report over the manifest's 14 hands).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hand_sampler.grammar.adapters.urdf import load_urdf
from hand_sampler.grammar.coverage import coverage, inventory
from hand_sampler.grammar.derive import generate
from hand_sampler.grammar.distributions import DEFAULT_DISTRIBUTION
from hand_sampler.grammar.kinematics import Body, Frame, Joint, KinematicModel, LoopClosure, Pose
from hand_sampler.grammar_bench import evaluate

BENCH_DIR = Path(__file__).resolve().parent.parent
ANALYTIC_DIR = BENCH_DIR / "fixtures" / "analytic"
MANIFEST = json.loads((BENCH_DIR / "manifest.json").read_text())

N_GENERATED_SEEDS = 50


# ---------------------------------------------------------------------------
# 1. inventory() on the two analytic fixtures
# ---------------------------------------------------------------------------


def test_inventory_offaxis_tree():
    model = load_urdf(ANALYTIC_DIR / "offaxis_tree.urdf").model
    inv = inventory(model)

    assert {t for t, n in inv.joint_type_counts.items() if n > 0} == {"revolute", "continuous", "prismatic"}
    assert inv.non_unit_axes == 1  # j3: axis xyz="0 3 4"
    assert inv.default_axis_joints >= 1  # j1 and j8 lack <axis>, defaulting to (1,0,0)
    assert inv.branching_bodies == 1
    assert inv.branching_body_names == ("link3",)
    assert inv.n_couplings == 0
    assert inv.asymmetric_limits > 0
    assert inv.palm_bodies == 0
    assert inv.n_closures == 0


def test_inventory_coupled_finger():
    model = load_urdf(ANALYTIC_DIR / "coupled_finger.urdf").model
    inv = inventory(model)

    assert inv.n_couplings == 2
    assert inv.couplings_with_offset == 1  # joint2: offset=0.1
    assert inv.couplings_negative == 1  # joint3: multiplier=-1.0
    assert inv.limit_conflicts == 1  # joint3's own limits are tighter than the image


# ---------------------------------------------------------------------------
# 2. coverage() must cover the grammar's own output exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(N_GENERATED_SEEDS))
def test_coverage_covers_own_generated_output(seed):
    _derivation, model = generate(seed, DEFAULT_DISTRIBUTION)
    result = coverage(model, DEFAULT_DISTRIBUTION)
    assert result.expressible is True, f"seed={seed} missing={result.missing_constructs}"
    assert result.in_support is True, f"seed={seed} out_of_support={result.out_of_support}"
    assert result.missing_constructs == []
    assert result.out_of_support == []


# ---------------------------------------------------------------------------
# 3. coverage() structured result on offaxis_tree (real hand, not grammar output)
# ---------------------------------------------------------------------------


def test_coverage_offaxis_tree_structured_result():
    model = load_urdf(ANALYTIC_DIR / "offaxis_tree.urdf").model
    result = coverage(model, DEFAULT_DISTRIBUTION)

    assert isinstance(result.missing_constructs, list)
    assert all(isinstance(x, str) for x in result.missing_constructs)
    assert isinstance(result.out_of_support, list)
    assert all(isinstance(x, str) for x in result.out_of_support)

    assert result.expressible == (result.missing_constructs == [])
    assert result.in_support == (result.expressible and result.out_of_support == [])


# ---------------------------------------------------------------------------
# 4. A LoopClosure is never expressible
# ---------------------------------------------------------------------------


def _synthetic_closure_model():
    bodies = (Body(name="root"), Body(name="a"), Body(name="b"))
    joints = (
        Joint(name="j_a", type="revolute", parent="root", child="a", limits=(-1.0, 1.0)),
        Joint(name="j_b", type="revolute", parent="root", child="b", limits=(-1.0, 1.0)),
    )
    frames = (
        Frame(name="frame_a", body="a", pose=Pose()),
        Frame(name="frame_b", body="b", pose=Pose()),
    )
    closures = (LoopClosure(name="loop1", frame_a="frame_a", frame_b="frame_b", kind="revolute"),)
    return KinematicModel(
        name="closure_test", root="root", bodies=bodies, joints=joints, frames=frames, closures=closures
    )


def test_coverage_loop_closure_not_expressible():
    model = _synthetic_closure_model()
    inv = inventory(model)
    assert inv.n_closures == 1

    result = coverage(model, DEFAULT_DISTRIBUTION)
    assert result.expressible is False
    assert "loop_closure" in result.missing_constructs
    assert result.in_support is False


# ---------------------------------------------------------------------------
# 5. evaluate.main -> pilot.json / pilot.md
# ---------------------------------------------------------------------------


def test_evaluate_main_produces_pilot_report(tmp_path):
    out_dir = tmp_path / "results"
    evaluate.main(["--out", str(out_dir)])

    pilot_json = out_dir / "pilot.json"
    pilot_md = out_dir / "pilot.md"
    assert pilot_json.is_file()
    assert pilot_md.is_file()

    data = json.loads(pilot_json.read_text())
    hands = data["hands"]
    manifest_ids = {h["id"] for h in MANIFEST["hands"]}
    assert {h["id"] for h in hands} == manifest_ids
    assert len(hands) == 14

    for h in hands:
        assert "split" in h
        assert h["availability"] in {"available", "unavailable", "excluded"}
        assert h["fidelity"] is None or {"n_poses", "max_pos", "max_rot", "tolerance_met"} <= set(h["fidelity"])
        assert h["coverage"] is None or {"expressible", "in_support", "missing_constructs", "out_of_support"} <= set(
            h["coverage"]
        )

    md_text = pilot_md.read_text()
    assert "## Claims" in md_text or "# Claims" in md_text
    assert "No universality claim is made." in md_text
