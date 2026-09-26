"""Iteration-4 acceptance tests for ``hand_sampler/grammar/coverage.py``
(structural inventory + support audit) and ``grammar_bench/evaluate.py``
(the pilot report over the manifest's hands -- 16 as of the representation-
check plan's item 3, which added ``shadow_right_local``/``arms_skel`` and
turned ``svh_right`` from ``excluded`` into a scored local hand).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hand_sampler.grammar.adapters.urdf import load_urdf
from hand_sampler.grammar.coverage import coverage, inventory
from hand_sampler.grammar.derive import generate
from hand_sampler.grammar.distributions import DEFAULT_DISTRIBUTION
from hand_sampler.grammar.kinematics import AffineCoupling, Body, Frame, Joint, KinematicModel, LoopClosure, Pose
from hand_sampler.grammar_bench import evaluate

BENCH_DIR = Path(__file__).resolve().parent.parent
ANALYTIC_DIR = BENCH_DIR / "fixtures" / "analytic"
REAL_DIR = BENCH_DIR / "fixtures" / "real"
ALLEGRO_URDF = REAL_DIR / "allegro_right" / "allegro_hand_description_right.urdf"
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
    assert result.topology_expressible is True, f"seed={seed} missing={result.missing_constructs}"
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

    assert result.topology_expressible == (result.missing_constructs == [])
    assert result.in_support == (result.topology_expressible and result.out_of_support == [])


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
    assert result.topology_expressible is False
    assert "loop_closure" in result.missing_constructs
    assert result.in_support is False


# ---------------------------------------------------------------------------
# 4c. iteration-6 honest-coverage checks: each new missing_constructs
# category, flagged by name, on a small synthetic model.
# ---------------------------------------------------------------------------


def _synthetic_excess_children_model():
    """A digit's own first phalanx (``d1p1``) has 6 child joints -- more than
    ``1 + DEFAULT_DISTRIBUTION.max_branch_digits`` (3) -- which no Phalanx
    production could ever build (ORCA's post-wrist body has 5)."""
    bodies = [Body(name="root", palm=True), Body(name="d1p1")]
    joints = [Joint(name="d1p1_j", type="revolute", parent="root", child="d1p1", limits=(-1.0, 1.0))]
    for i in range(6):
        bodies.append(Body(name=f"c{i}"))
        joints.append(Joint(name=f"c{i}_j", type="revolute", parent="d1p1", child=f"c{i}", limits=(-1.0, 1.0)))
    return KinematicModel(name="excess_children_test", root="root", bodies=tuple(bodies), joints=tuple(joints))


def test_coverage_flags_excess_children_by_name():
    model = _synthetic_excess_children_model()
    result = coverage(model, DEFAULT_DISTRIBUTION)
    assert result.topology_expressible is False
    assert any(item == "excess_children:d1p1" for item in result.missing_constructs), result.missing_constructs


def _synthetic_continuation_pose_model():
    """A mid-digit continuation joint (``d1p2_j``, the sole child of
    ``d1p1``) has a lateral (x/y) origin offset -- the grammar only ever
    builds a continuation as ``Trans(0, 0, prev_len) * Rot(0, 0, 0)``."""
    bodies = (Body(name="root", palm=True), Body(name="d1p1"), Body(name="d1p2"))
    joints = (
        Joint(name="d1p1_j", type="revolute", parent="root", child="d1p1", limits=(-1.0, 1.0)),
        Joint(name="d1p2_j", type="revolute", parent="d1p1", child="d1p2",
              origin=Pose(xyz=(0.01, 0.0, 0.02), rpy=(0.0, 0.0, 0.0)), limits=(-1.0, 1.0)),
    )
    return KinematicModel(name="continuation_pose_test", root="root", bodies=bodies, joints=joints)


def test_coverage_flags_continuation_pose_by_name():
    """Grammar 0.5 (I16 priority 1) contract change: a rest-bend is now a
    genuine grammar primitive (every Phalanx step carries ``bend_rpy``/
    ``bend_offset`` -- see distributions.py/derive.py), so a mid-digit
    lateral offset is topology-EXPRESSIBLE regardless of its value; whether
    this SPECIFIC value is in the (default, "no bend") distribution's own
    support is a grid-membership question, reported in ``out_of_support``
    (like axis/limit/length grid checks) rather than ``missing_constructs``."""
    model = _synthetic_continuation_pose_model()
    result = coverage(model, DEFAULT_DISTRIBUTION)
    assert result.topology_expressible is True
    assert any(item == "continuation_pose:d1p2_j" for item in result.out_of_support), result.out_of_support
    assert result.in_support is False


def _synthetic_cross_digit_coupling_model():
    """A coupling whose dependent (``d1p1_j``) and source (``d2p1_j``) sit on
    two different top-level digits -- rules.py's ModuleCoupled only ever
    sources an earlier phalanx of the SAME digit."""
    bodies = (Body(name="root", palm=True), Body(name="d1p1"), Body(name="d2p1"))
    joints = (
        Joint(name="d1p1_j", type="revolute", parent="root", child="d1p1", limits=(-1.0, 1.0)),
        Joint(name="d2p1_j", type="revolute", parent="root", child="d2p1", limits=(-1.0, 1.0)),
    )
    couplings = (AffineCoupling(dependent="d1p1_j", source="d2p1_j", multiplier=1.0, offset=0.0),)
    return KinematicModel(name="cross_digit_coupling_test", root="root", bodies=bodies, joints=joints,
                           couplings=couplings)


def test_coverage_flags_cross_digit_coupling_by_name():
    model = _synthetic_cross_digit_coupling_model()
    result = coverage(model, DEFAULT_DISTRIBUTION)
    assert result.topology_expressible is False
    assert any(item == "coupling_scope:d1p1_j" for item in result.missing_constructs), result.missing_constructs


# ---------------------------------------------------------------------------
# 4b. digit_count structural fallback (root_chains) for imported models with
# no palm=True bodies at all -- see coverage.py's module docstring.
# ---------------------------------------------------------------------------


def test_coverage_digit_count_fallback_allegro():
    model = load_urdf(ALLEGRO_URDF).model
    result = coverage(model, DEFAULT_DISTRIBUTION)

    assert not any(item.startswith("digit_count_out_of_range") for item in result.out_of_support)
    assert result.digit_count == 4
    assert result.digit_count_source == "root_chains"
    assert "digit_count_source:root_chains" in result.notes


def test_coverage_digit_count_fallback_offaxis_tree():
    # offaxis_tree.urdf's root ("base") has exactly one direct child joint,
    # j1 (revolute) -- a single movable-joint chain leaving the root, so the
    # structural fallback must report 1 digit, not 0.
    model = load_urdf(ANALYTIC_DIR / "offaxis_tree.urdf").model
    result = coverage(model, DEFAULT_DISTRIBUTION)

    assert not any(item.startswith("digit_count_out_of_range") for item in result.out_of_support)
    assert result.digit_count == 1
    assert result.digit_count_source == "root_chains"
    assert "digit_count_source:root_chains" in result.notes


# ---------------------------------------------------------------------------
# 5. evaluate.main -> pilot-report.json / pilot-report.md
# ---------------------------------------------------------------------------


def test_evaluate_main_default_out_dir_is_project_notes_grammar():
    assert evaluate.DEFAULT_OUT_DIR == evaluate.REPO_ROOT / "project-notes" / "grammar"


def test_evaluate_main_produces_pilot_report(tmp_path):
    out_dir = tmp_path / "results"
    evaluate.main(["--out", str(out_dir)])

    pilot_json = out_dir / "pilot-report.json"
    pilot_md = out_dir / "pilot-report.md"
    assert pilot_json.is_file()
    assert pilot_md.is_file()

    data = json.loads(pilot_json.read_text())
    hands = data["hands"]
    manifest_ids = {h["id"] for h in MANIFEST["hands"]}
    assert {h["id"] for h in hands} == manifest_ids
    assert len(hands) == 16  # representation-check plan item 3: was 14; +shadow_right_local, +arms_skel

    for h in hands:
        assert "split" in h
        assert h["availability"] in {"available", "unavailable", "excluded"}
        assert h["fidelity"] is None or {"n_poses", "max_pos", "max_rot", "tolerance_met"} <= set(h["fidelity"])
        assert h["coverage"] is None or {
            "topology_expressible", "in_support", "missing_constructs", "out_of_support",
            "digit_count", "digit_count_source",
        } <= set(h["coverage"])

    md_text = pilot_md.read_text()
    assert "## Claims" in md_text or "# Claims" in md_text
    assert "No universality claim is made." in md_text
    assert "digit count" in md_text
    assert "digit count source" in md_text
