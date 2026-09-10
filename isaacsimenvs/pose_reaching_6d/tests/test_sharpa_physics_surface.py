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


@pytest.mark.parametrize("joint,stiffness,damping,armature", [
    ("left_1_thumb_CMC_FE", 6.95, 0.28676845, 0.0032),
    ("left_thumb_CMC_AA", 13.2, 0.40845109, 0.0032),
    ("left_2_index_MCP_FE", 4.76, 0.20859232, 0.00265),
    ("left_index_MCP_AA", 6.62, 0.24595532, 0.00265),
    ("left_index_PIP", 0.9, 0.04243185, 0.0006),
    ("left_index_DIP", 0.9, 0.03504461, 0.00042),
    ("left_5_pinky_CMC", 1.38, 0.02782345, 0.00012),
])
def test_drive_parameters(spec, joint, stiffness, damping, armature):
    assert spec.hand_stiffness[joint] == stiffness
    assert spec.hand_damping[joint] == damping
    assert spec.hand_armature[joint] == armature


def test_every_hand_joint_has_every_drive_parameter(spec):
    """A missing entry means Isaac Lab silently falls back to the USD value."""
    for table in (spec.hand_stiffness, spec.hand_damping, spec.hand_armature):
        assert set(table) == set(spec.hand_joint_names)


def test_no_spec_carries_joint_friction(spec):
    """Joint friction is zero on every robot, so no spec may smuggle a table in.

    HAND_FRICTION survives in robot_param_constants as the record of what
    simtoolreal measured, but nothing reads it: the values only mean anything in
    isaacgym's units, which it never documents, and applied here they moved the
    pretrained checkpoint by 0.089 goals/env against a 0.24 standard error.
    """
    assert not hasattr(spec, "hand_friction")


def test_the_actuator_cfg_pins_friction_to_zero():
    """0.0, never None -- None takes the USD value, which is not uniformity."""
    src = Path(__file__).resolve().parents[1] / "scene_utils" / "assembly.py"
    body = src.read_text()
    body = body[body.index("def build_robot_articulation_cfg"):]
    body = body[:body.index("\ndef ")]
    assert body.count("friction=0.0") == 2, "arm and hand must both pin friction"
    assert "spec.hand_friction" not in body


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


# --- backwards compatibility with the simtoolreal checkpoint ------------------

CHECKPOINT_OBS_LIST = (
    "joint_pos", "joint_vel", "prev_action_targets", "palm_pos", "palm_rot",
    "object_rot", "fingertip_pos_rel_palm", "keypoints_rel_palm",
    "keypoints_rel_goal", "object_scales",
)
"""Verbatim from pretrained_policy/config.yaml, which ships with the weights."""


def test_the_pretrained_checkpoints_observation_can_still_be_built(spec):
    """Its running_mean_std is (140,); the env must be able to produce 140."""
    layout = _load("layout", "isaacsimenvs/pose_reaching_6d/obs_utils/layout.py")
    assert layout.compute_obs_dim(list(CHECKPOINT_OBS_LIST), spec) == 140


def test_fingertip_pad_offsets_exist(spec):
    """Deleted twice now. The first time, the MLP arm's done_hand_far went from
    1.3% of episodes to 54% while the transformer on the same observation stayed
    at 1.6% -- joint_link_bbox carries the distal geometry only implicitly."""
    assert spec.fingertip_offsets
    assert len(spec.fingertip_offsets) == spec.num_fingertips
    assert spec.fingertip_offsets[0] == (0.02, 0.002, 0.0)


def test_current_state_list_is_unchanged_at_800(spec):
    layout = _load("layout", "isaacsimenvs/pose_reaching_6d/obs_utils/layout.py")
    current = [
        "joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel",
        "prev_action_targets", "joint_link_bbox", "joint_lower", "joint_upper",
        "joint_enabled", "object_keypoints_rel_joint", "hand_scale", "palm_pos",
        "palm_rot", "palm_vel", "object_rot", "object_vel", "keypoints_rel_palm",
        "keypoints_rel_goal", "object_scales", "closest_keypoint_max_dist",
        "closest_fingertip_dist", "lifted_object", "progress", "successes", "reward",
    ]
    assert layout.compute_obs_dim(current, spec) == 800


def test_no_undefined_names_in_the_package():
    """Most of this package only executes under Kit, so a free name that resolves
    to nothing is a NameError nobody sees until a job has booted the simulator.
    (`build_robot_articulation_cfg` once read `env.cfg` with no `env` in scope.)"""
    import pathlib
    import pyflakes.api
    import pyflakes.messages
    import pyflakes.reporter

    class _UndefinedOnly(pyflakes.reporter.Reporter):
        def __init__(self):
            self.hits: list[str] = []

        def unexpectedError(self, filename, msg):
            self.hits.append(f"{filename}: {msg}")

        def syntaxError(self, filename, msg, lineno, offset, text):
            self.hits.append(f"{filename}:{lineno}: {msg}")

        def flake(self, message):
            if isinstance(message, (pyflakes.messages.UndefinedName,
                                    pyflakes.messages.UndefinedLocal,
                                    pyflakes.messages.UndefinedExport)):
                self.hits.append(str(message))

    root = pathlib.Path(__file__).resolve().parents[1]
    reporter = _UndefinedOnly()
    for path in sorted(root.rglob("*.py")):
        pyflakes.api.checkPath(str(path), reporter)
    assert not reporter.hits, "\n".join(reporter.hits)
