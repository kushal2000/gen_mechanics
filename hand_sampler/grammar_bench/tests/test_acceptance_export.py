"""Iteration-2 acceptance tests: JSON round trip, URDF export fidelity (with
an independent Pinocchio oracle check on the exported file), LossReport
completeness for loop closures and hand_root-cut couplings, SHARPA import,
and the local-only coupled real hands (Ability, Inspire).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from urdf_parser_py.urdf import URDF

from hand_sampler.grammar import fk
from hand_sampler.grammar.adapters.json_io import from_json, to_json
from hand_sampler.grammar.adapters.urdf import LossReport, load_urdf, to_urdf
from hand_sampler.grammar.coords import independent_joints, movable_joints, q_from_u
from hand_sampler.grammar.kinematics import (
    AffineCoupling,
    Body,
    Frame,
    Joint,
    KinematicModel,
    LoopClosure,
    Pose,
)
from hand_sampler.grammar_bench.tolerances import ORACLE_POS_M, ORACLE_ROT_RAD

BENCH_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BENCH_DIR.parent.parent
REFERENCES_DIR = BENCH_DIR / "references"
LOCAL_REFERENCES_DIR = REFERENCES_DIR / "local"
ANALYTIC_DIR = BENCH_DIR / "fixtures" / "analytic"
ORACLE_PYTHON = Path("/home/singularity/anaconda3/envs/piper/bin/python")
ORACLE_FK_SCRIPT = BENCH_DIR / "refgen" / "oracle_fk.py"


def _manifest():
    return json.loads((BENCH_DIR / "manifest.json").read_text())


def _committed_real_hands():
    """(hand_id, urdf_path, hand_root) for every manifest entry with
    commit_allowed=true and a fixture_path, plus SHARPA via its REPO: source_path."""
    out = []
    for h in _manifest()["hands"]:
        if not h.get("commit_allowed"):
            continue
        if h.get("fixture_path"):
            out.append((h["id"], BENCH_DIR / h["fixture_path"], h.get("hand_root")))
        elif isinstance(h.get("source_path"), str) and h["source_path"].startswith("REPO:"):
            out.append((h["id"], REPO_ROOT / h["source_path"][len("REPO:") :], h.get("hand_root")))
    assert out, "expected at least one committed real hand in the manifest"
    return out


def _analytic_fixtures():
    paths = sorted(ANALYTIC_DIR.glob("*.urdf"))
    assert paths, "expected at least one analytic fixture"
    return paths


def _oracle_available() -> bool:
    return ORACLE_PYTHON.is_file() and ORACLE_FK_SCRIPT.is_file()


# --------------------------------------------------------------------------
# 1. JSON round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("urdf_path", _analytic_fixtures(), ids=lambda p: p.name)
def test_json_round_trip_analytic(urdf_path):
    model = load_urdf(urdf_path).model
    s = to_json(model)
    assert from_json(s) == model
    assert to_json(from_json(s)) == s


@pytest.mark.parametrize("case", _committed_real_hands(), ids=lambda c: c[0])
def test_json_round_trip_real_hands(case):
    hand_id, urdf_path, hand_root = case
    model = load_urdf(urdf_path, hand_root=hand_root).model
    s = to_json(model)
    assert from_json(s) == model
    assert to_json(from_json(s)) == s


def test_json_round_trip_rejects_nan():
    bad_model = KinematicModel(
        name="bad",
        root="a",
        bodies=(Body(name="a"),),
        joints=(),
    )
    # Sanity: a normal model serializes fine; float("nan") must be refused.
    to_json(bad_model)
    nan_model = KinematicModel(
        name="bad",
        root="a",
        bodies=(Body(name="a"), Body(name="b")),
        joints=(Joint(name="j", type="revolute", parent="a", child="b", limits=(0.0, float("nan"))),),
    )
    with pytest.raises(ValueError):
        to_json(nan_model)


# --------------------------------------------------------------------------
# 2. URDF export fidelity
# --------------------------------------------------------------------------


def _check_export_fidelity(model, actuation):
    text, _losses = to_urdf(model, actuation=actuation)
    parsed = URDF.from_xml_string(text.encode("utf-8"))
    assert parsed is not None

    expected_mimics = {
        c.dependent: (c.dependent, c.source, c.multiplier, c.offset) for c in model.couplings
    }
    seen_mimics = {}
    for j in parsed.joints:
        if j.mimic is not None:
            seen_mimics[j.name] = (j.name, j.mimic.joint, j.mimic.multiplier, j.mimic.offset)
    assert seen_mimics == expected_mimics

    reimported = load_urdf(text).model
    assert reimported.bodies == model.bodies
    assert reimported.joints == model.joints
    assert reimported.couplings == model.couplings
    assert reimported.root == model.root
    return text


@pytest.mark.parametrize("urdf_path", _analytic_fixtures(), ids=lambda p: p.name)
def test_urdf_export_fidelity_analytic(urdf_path):
    result = load_urdf(urdf_path)
    _check_export_fidelity(result.model, result.actuation)


@pytest.mark.parametrize("case", _committed_real_hands(), ids=lambda c: c[0])
def test_urdf_export_fidelity_real_hands(case):
    hand_id, urdf_path, hand_root = case
    result = load_urdf(urdf_path, hand_root=hand_root)
    _check_export_fidelity(result.model, result.actuation)


# --------------------------------------------------------------------------
# 2c. Independent oracle check on the *exported* file
# --------------------------------------------------------------------------

# Restricted to hands with an existing frozen reference (generated with the
# oracle interpreter): the three committed dev hands plus SHARPA (generated
# in this iteration). Real hands without a frozen reference are still
# covered by the parse + mimic-tuple + reimport-equality checks above.
ORACLE_EXPORT_CASES = [
    ("allegro_right", BENCH_DIR / "fixtures/real/allegro_right/allegro_hand_description_right.urdf", None),
    ("leap_right", BENCH_DIR / "fixtures/real/leap_right/leap_hand_right.urdf", None),
    ("barrett_bh", BENCH_DIR / "fixtures/real/barrett_bh/bhand_model.urdf", None),
    (
        "sharpa_left_on_iiwa14",
        REPO_ROOT / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf",
        "left_hand_C_MC",
    ),
]


@pytest.mark.parametrize("case", ORACLE_EXPORT_CASES, ids=lambda c: c[0])
def test_urdf_export_matches_oracle(case, tmp_path):
    hand_id, urdf_path, hand_root = case
    if not _oracle_available():
        pytest.skip("oracle-unavailable: piper conda interpreter not found")

    ref_path = REFERENCES_DIR / f"{hand_id}.json"
    configs_path = REFERENCES_DIR / f"{hand_id}.configs.json"
    if not (ref_path.is_file() and configs_path.is_file()):
        pytest.fail(f"missing frozen reference/configs for {hand_id!r}")

    result = load_urdf(urdf_path, hand_root=hand_root)
    text, _losses = to_urdf(result.model, actuation=result.actuation)

    exported_urdf_path = tmp_path / f"{hand_id}_export.urdf"
    exported_urdf_path.write_text(text)
    oracle_out_path = tmp_path / f"{hand_id}_export_oracle.json"

    cmd = [
        str(ORACLE_PYTHON),
        str(ORACLE_FK_SCRIPT),
        "--urdf",
        str(exported_urdf_path),
        "--configs",
        str(configs_path),
        "--out",
        str(oracle_out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"oracle_fk.py failed on exported {hand_id}:\n{proc.stdout}\n{proc.stderr}"

    exported_oracle = json.loads(oracle_out_path.read_text())
    frozen_reference = json.loads(ref_path.read_text())

    max_pos = 0.0
    max_rot = 0.0
    n_compared = 0
    n_bodies = None
    for exp_poses, ref_poses in zip(exported_oracle["poses"], frozen_reference["poses"]):
        # The export round trip never adds or drops a body for a real (non-
        # grammar-derived) hand -- these models carry no "<body>_tip" frames
        # (see test_acceptance_grammar.py's own convention note, which only
        # applies to grammar output). For the three plain dev hands the
        # exported and frozen-reference body sets agree exactly; for SHARPA,
        # the frozen reference here is generated against the full
        # iiwa14+hand robot (see references/sharpa_left_on_iiwa14.json),
        # while the exported model only ever has the hand_root-truncated
        # ("kept") bodies -- so reference bodies == kept bodies means every
        # exported (kept) body must appear in the reference, not the reverse.
        assert set(exp_poses) <= set(ref_poses), (
            f"{hand_id}: exported body missing from reference: {set(exp_poses) - set(ref_poses)}"
        )
        n_bodies = len(exp_poses)
        for body, T_exp_list in exp_poses.items():
            T_exp = np.array(T_exp_list)
            T_ref = np.array(ref_poses[body])
            pos_err = fk.position_error(T_exp, T_ref)
            rot_err = fk.rotation_error(T_exp, T_ref)
            max_pos = max(max_pos, pos_err)
            max_rot = max(max_rot, rot_err)
            n_compared += 1
            assert pos_err <= ORACLE_POS_M, f"{hand_id} body={body} pos_err={pos_err}"
            assert rot_err <= ORACLE_ROT_RAD, f"{hand_id} body={body} rot_err={rot_err}"

    assert n_compared > 0
    assert n_bodies is not None and n_compared == len(exported_oracle["poses"]) * n_bodies
    print(
        f"{hand_id} export-oracle: {n_compared} body-pose comparisons, "
        f"max_pos={max_pos:.3e}, max_rot={max_rot:.3e}"
    )


# --------------------------------------------------------------------------
# 3. LossReport completeness
# --------------------------------------------------------------------------


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
    return KinematicModel(name="closure_test", root="root", bodies=bodies, joints=joints, frames=frames, closures=closures)


def test_export_loss_report_names_every_closure():
    model = _synthetic_closure_model()
    text, losses = to_urdf(model)
    assert losses.closures_dropped == ("loop1",)
    assert "loop1" in text


def test_hand_root_cut_names_every_dropped_coupling():
    synthetic_urdf = """<?xml version="1.0"?>
<robot name="cuttest">
  <link name="root"/>
  <link name="mid"/>
  <link name="tip"/>
  <joint name="jointA" type="revolute">
    <parent link="root"/><child link="mid"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="jointC" type="revolute">
    <parent link="mid"/><child link="tip"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
    <mimic joint="jointA" multiplier="1" offset="0"/>
  </joint>
</robot>
"""
    result = load_urdf(synthetic_urdf, hand_root="mid")
    assert result.losses.couplings_cut == ("jointC",)
    assert result.model.couplings == ()


# --------------------------------------------------------------------------
# 4. SHARPA
# --------------------------------------------------------------------------


def test_sharpa_import_and_fk():
    sharpa = next(h for h in _manifest()["hands"] if h["id"] == "sharpa_left_on_iiwa14")
    urdf_path = REPO_ROOT / sharpa["source_path"][len("REPO:") :]
    result = load_urdf(urdf_path, hand_root=sharpa["hand_root"])
    model = result.model

    assert len(movable_joints(model)) == 22
    assert model.root == "left_hand_C_MC"
    assert not any(b.name.startswith("iiwa14") for b in model.bodies)
    assert result.losses.dropped_above_hand_root

    ref_path = REFERENCES_DIR / "sharpa_left_on_iiwa14.json"
    assert ref_path.is_file(), "missing references/sharpa_left_on_iiwa14.json"
    reference = json.loads(ref_path.read_text())

    # SHARPA's oracle reference was generated against the full iiwa14+hand
    # robot (dropped_above_hand_root bodies included), while our own model
    # only ever has the hand_root-truncated ("kept") bodies -- so the
    # reference is a strict superset, never equal, to our own body set.
    # What must hold instead: reference bodies == kept bodies, restricted to
    # the ones our (truncated) model actually has -- every kept body has a
    # reference measurement.
    kept_bodies = {b.name for b in model.bodies}

    max_pos = 0.0
    max_rot = 0.0
    n_compared = 0
    for u, oracle_poses in zip(reference["configs"], reference["poses"]):
        q = q_from_u(model, u)
        ours = fk.forward_kinematics(model, q)
        assert set(ours) == kept_bodies
        assert kept_bodies <= set(oracle_poses), (
            f"missing reference pose(s) for kept body: {kept_bodies - set(oracle_poses)}"
        )
        for body in kept_bodies:
            T_oracle = np.array(oracle_poses[body])
            pos_err = fk.position_error(ours[body], T_oracle)
            rot_err = fk.rotation_error(ours[body], T_oracle)
            max_pos = max(max_pos, pos_err)
            max_rot = max(max_rot, rot_err)
            n_compared += 1
            assert pos_err <= ORACLE_POS_M
            assert rot_err <= ORACLE_ROT_RAD

    assert n_compared > 0
    assert n_compared == len(reference["configs"]) * len(kept_bodies)
    print(f"sharpa_left_on_iiwa14: {n_compared} body-pose comparisons, max_pos={max_pos:.3e}, max_rot={max_rot:.3e}")


# --------------------------------------------------------------------------
# 5. Local-only coupled hands (Ability, Inspire)
# --------------------------------------------------------------------------

LOCAL_ONLY_CASES = {
    "ability_right": "/home/singularity/karma/karma-data/all_urdfs/full_models_as_downloaded/Ability/ability_hand_right_large_no_fsr.urdf",
    "inspire_right": "/home/singularity/karma/karma-data/all_urdfs/full_models_as_downloaded/Inspire/inspire_hand_right.urdf",
}


@pytest.mark.parametrize("hand_id", sorted(LOCAL_ONLY_CASES))
def test_local_only_coupled_hands_match_oracle(hand_id):
    ref_path = LOCAL_REFERENCES_DIR / f"{hand_id}.json"
    if not ref_path.is_file():
        pytest.skip(f"local-only:{hand_id} reference not present on this machine")

    urdf_path = Path(LOCAL_ONLY_CASES[hand_id])
    result = load_urdf(urdf_path)
    model = result.model
    reference = json.loads(ref_path.read_text())

    max_pos = 0.0
    max_rot = 0.0
    n_compared = 0
    n_bodies = None
    for u, oracle_poses in zip(reference["configs"], reference["poses"]):
        q = q_from_u(model, u)
        ours = fk.forward_kinematics(model, q)
        assert set(ours) == set(oracle_poses), (
            f"{hand_id}: body-name mismatch, ours-only={set(ours) - set(oracle_poses)} "
            f"reference-only={set(oracle_poses) - set(ours)}"
        )
        n_bodies = len(ours)
        for body, T_oracle_list in oracle_poses.items():
            T_oracle = np.array(T_oracle_list)
            pos_err = fk.position_error(ours[body], T_oracle)
            rot_err = fk.rotation_error(ours[body], T_oracle)
            max_pos = max(max_pos, pos_err)
            max_rot = max(max_rot, rot_err)
            n_compared += 1
            assert pos_err <= ORACLE_POS_M, f"{hand_id} body={body} pos_err={pos_err}"
            assert rot_err <= ORACLE_ROT_RAD, f"{hand_id} body={body} rot_err={rot_err}"

    assert n_compared > 0
    assert n_bodies is not None and n_compared == len(reference["configs"]) * n_bodies
    print(f"{hand_id}: {n_compared} body-pose comparisons, max_pos={max_pos:.3e}, max_rot={max_rot:.3e}")
