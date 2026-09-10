"""Everything about SHARPA that reaches the simulator, pinned to exact values.

A rollout comparison cannot answer "did the physics change": PhysX on GPU is
not bit-deterministic, the policy samples, and resets are randomised, so a few
percent either way is unattributable. These are the deterministic surfaces
instead -- what the spec says and what gets handed to Isaac Lab -- checked
against literals rather than against another build.

Numbers are simtoolreal's, transcribed and validated by a bitwise rollout
parity test against it (see robot_param_constants). If one of these changes,
the run is a different experiment.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def spec():
    """SHARPA_IIWA14 without its package's Kit imports."""
    adjacency = _load(
        "adj", "isaacsimenvs/pose_reaching_6d/scene_utils/robots/adjacency/sharpa_iiwa14.py")
    src = (ROOT / "isaacsimenvs/pose_reaching_6d/scene_utils/robots/sharpa_iiwa14.py").read_text()
    src = src.replace(
        "from isaacsimenvs.pose_reaching_6d.scene_utils.robots.adjacency.sharpa_iiwa14 "
        "import SHARPA_IIWA14_ADJACENT_LINKS",
        "SHARPA_IIWA14_ADJACENT_LINKS = _ADJ")
    ns = {"__name__": "sharpa", "_ADJ": adjacency.SHARPA_IIWA14_ADJACENT_LINKS}
    exec(compile(src, "sharpa_iiwa14.py", "exec"), ns)
    return ns["SHARPA_IIWA14"]


def test_joint_set_and_order(spec):
    """Order is the action layout, and the left_N_ infixes are load-bearing."""
    assert spec.num_arm_joints == 7
    assert spec.num_hand_joints == 22
    assert spec.num_joints == 29
    assert spec.hand_joint_names[0] == "left_1_thumb_CMC_FE"
    assert spec.hand_joint_names[-1] == "left_pinky_DIP"
    assert spec.joint_names_canonical == (*spec.arm_joint_names, *spec.hand_joint_names)


@pytest.mark.parametrize("joint,stiffness,damping,armature,friction", [
    ("left_1_thumb_CMC_FE", 6.95, 0.28676845, 0.0032, 0.132),
    ("left_thumb_CMC_AA", 13.2, 0.40845109, 0.0032, 0.132),
    ("left_2_index_MCP_FE", 4.76, 0.20859232, 0.00265, 0.07456),
    ("left_index_MCP_AA", 6.62, 0.24595532, 0.00265, 0.07456),
    ("left_index_PIP", 0.9, 0.04243185, 0.0006, 0.01276),
    ("left_index_DIP", 0.9, 0.03504461, 0.00042, 0.00378738),
    ("left_5_pinky_CMC", 1.38, 0.02782345, 0.00012, 0.012),
])
def test_drive_parameters(spec, joint, stiffness, damping, armature, friction):
    assert spec.hand_stiffness[joint] == stiffness
    assert spec.hand_damping[joint] == damping
    assert spec.hand_armature[joint] == armature
    assert spec.hand_friction[joint] == friction


def test_every_hand_joint_has_every_drive_parameter(spec):
    """A missing entry means Isaac Lab silently falls back to the USD value."""
    for table in (spec.hand_stiffness, spec.hand_damping,
                  spec.hand_armature, spec.hand_friction):
        assert set(table) == set(spec.hand_joint_names)


def test_joint_friction_is_present_at_all(spec):
    """simtoolreal set it; the first port dropped it and SHARPA ran frictionless.

    Runs launched before it was restored are a different environment.
    """
    assert spec.hand_friction, "hand_friction is empty: the hand has no joint friction"
    assert sum(spec.hand_friction.values()) == pytest.approx(1.10054952)


def test_geometry_the_observation_depends_on(spec):
    assert spec.palm_body_name == "iiwa14_link_7"
    assert spec.palm_center_offset == (-0.0, -0.02, 0.16)
    assert spec.num_fingertips == 5
    assert spec.base_pos == (0.0, 0.8, 0.0)
    assert spec.base_rot == (1.0, 0.0, 0.0, 0.0)
    assert spec.replace_cylinders_with_capsules is False


def test_self_collision_map_is_not_empty(spec):
    """Empty means every hand link collides with its own neighbours at spawn."""
    assert len(spec.adjacent_links) >= 30


def test_joint_tokens_match_the_urdf(spec):
    """The observation's geometry, carried on the spec instead of parsed at run
    start. It must still equal what the asset says."""
    import numpy as np

    from hand_sampler.design_space import joint_link_boxes
    bodies, boxes, valid, scale = joint_link_boxes(spec.urdf_path, spec.hand_joint_names)
    assert tuple(bodies) == spec.joint_link_bodies
    assert np.abs(np.asarray(spec.joint_link_boxes, np.float32) - boxes).max() == 0.0
    assert spec.hand_scale == pytest.approx(scale)
    assert all(spec.joint_geometry_valid)
