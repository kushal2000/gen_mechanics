"""Hand-only RobotSpec-shaped object, built from the manifest. Needs numpy
(design_space.joint_link_boxes); run with .venv_isaacsim's python -- see
test_urdf_cutter.py's docstring for why anything under isaacsimenvs needs it
anyway (gymnasium, imported eagerly by isaacsimenvs/__init__.py).

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaacsim/bin/python3 -m pytest \
        isaacsimenvs/inhand_reorient/tests/test_hand_only_spec.py -q
"""

from __future__ import annotations

import pytest

from isaacsimenvs.inhand_reorient.hand_only import build_hand_only_spec

HANDS = ["sharpa", "allegro_right", "dclaw"]


@pytest.mark.parametrize("hand_id", HANDS)
def test_builds_a_valid_spec(hand_id, tmp_path):
    spec, cut = build_hand_only_spec(hand_id, out_dir=tmp_path)
    spec.validate()  # __post_init__ already ran this; a second call must be idempotent
    assert spec.num_arm_joints == 0
    assert spec.arm_joint_names == ()
    assert spec.num_hand_joints == len(spec.hand_joint_names) > 0
    assert spec.palm_body_name == cut.hand_root


def test_sharpa_has_22_joints_and_its_own_transcribed_gains(tmp_path):
    spec, _ = build_hand_only_spec("sharpa", out_dir=tmp_path)
    assert spec.num_hand_joints == 22
    from hand_sampler.robot_param_constants import HAND_STIFFNESS

    for j, k in HAND_STIFFNESS.items():
        assert spec.hand_stiffness[j] == k


def test_commercial_hands_get_the_sharpa_mean_default_gains(tmp_path):
    from isaacsimenvs.inhand_reorient.hand_only import DEFAULT_HAND_STIFFNESS

    spec, _ = build_hand_only_spec("allegro_right", out_dir=tmp_path)
    assert set(spec.hand_stiffness.values()) == {DEFAULT_HAND_STIFFNESS}


def test_dclaw_has_9_joints(tmp_path):
    spec, _ = build_hand_only_spec("dclaw", out_dir=tmp_path)
    assert spec.num_hand_joints == 9


@pytest.mark.parametrize("hand_id", HANDS)
def test_joint_link_boxes_shape_matches_joint_count(hand_id, tmp_path):
    spec, _ = build_hand_only_spec(hand_id, out_dir=tmp_path)
    n = spec.num_hand_joints
    assert len(spec.joint_link_bodies) == n
    assert len(spec.joint_link_boxes) == n
    assert len(spec.joint_geometry_valid) == n
    assert spec.hand_scale > 0.0
    for box in spec.joint_link_boxes:
        assert len(box) == 4
        assert all(len(corner) == 3 for corner in box)


@pytest.mark.parametrize("hand_id", HANDS)
def test_joint_limits_are_lower_le_upper(hand_id, tmp_path):
    spec, _ = build_hand_only_spec(hand_id, out_dir=tmp_path)
    assert len(spec.hand_joint_limits) == spec.num_hand_joints
    for lo, hi in spec.hand_joint_limits:
        assert lo <= hi


@pytest.mark.parametrize("hand_id", HANDS)
def test_cut_urdf_is_written_and_parseable(hand_id, tmp_path):
    spec, cut = build_hand_only_spec(hand_id, out_dir=tmp_path)
    import xml.etree.ElementTree as ET

    reparsed = ET.parse(spec.urdf_path).getroot()
    assert reparsed.tag == "robot"
    assert {l.get("name") for l in reparsed.findall("link")} == set(cut.kept_links)


def test_unavailable_commercial_hand_raises_a_clear_error(tmp_path):
    # shadow_right: split "excluded", fixture_path None, and its
    # source_path ("Shadow") is a directory, not a file -- unavailable under
    # both resolution paths regardless of this machine's source_root.
    # (inspire_right used to be the example here: commit_allowed False, no
    # fixture_path. I33 made resolve_hand_urdf prefer the manifest's
    # source_root/source_path copy when it is present and sha-verified --
    # exactly what grammar_bench.evaluate._resolve_hand and
    # scene/population_file._resolve_hand already do -- so inspire_right now
    # resolves via its source copy; see test_resolve_hand_urdf.py.)
    with pytest.raises(ValueError, match="fixture_path"):
        build_hand_only_spec("shadow_right", out_dir=tmp_path)


@pytest.mark.parametrize("hand_id", HANDS)
def test_default_joint_pos_is_within_each_joints_own_limits(hand_id, tmp_path):
    # Regression: allegro_right's thumb joint_12 has limits [0.263, 1.396],
    # which excludes 0 -- a hardcoded 0.0 default crashes Isaac Lab's
    # Articulation._validate_cfg() ("default positions out of the limits")
    # at scene boot, well past anything a CPU-only test would catch.
    spec, _ = build_hand_only_spec(hand_id, out_dir=tmp_path)
    for j, (lo, hi) in zip(spec.hand_joint_names, spec.hand_joint_limits):
        default = spec.hand_default_joint_pos[j]
        assert lo <= default <= hi, f"{hand_id}/{j}: default {default} not in [{lo}, {hi}]"
