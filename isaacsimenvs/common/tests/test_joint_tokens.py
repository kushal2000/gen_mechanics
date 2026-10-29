"""Joint-token geometry: the "ordered" link boxes are unchanged, the "extent" boxes cover each link exactly, and the
joint_origin field is an optional 3-wide token field. Kit-free."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from hand_sampler import design_space as ds
from isaacsimenvs.common.obs_layout import build_token_layout
from isaacsimenvs.common.robots import get_robot_spec

ROOT = Path(__file__).resolve().parents[3]
GRAMMAR_000 = "assets/urdf/27oct_grammar_random_hands/coarse_s0_000/coarse_s0_000.spec.json"   # has a twist joint
ALLEGRO = "assets/urdf/unified_dynamics_commercial_hands/allegro/allegro_left.spec.json"


def _outside_mm(box: np.ndarray, vertices: np.ndarray) -> float:
    p0, e = box[0], box[1:] - box[0]
    n = np.linalg.norm(e, axis=1)
    loc = (vertices - p0) @ (e / n[:, None]).T
    return 1e3 * float((np.maximum(0, -loc) + np.maximum(0, loc - n)).max())


@pytest.mark.parametrize("spec_path", [GRAMMAR_000, ALLEGRO])
def test_ordered_is_what_the_spec_has_always_carried(spec_path):
    s = get_robot_spec(spec_path)
    _, boxes, _, _ = ds.joint_link_boxes(s.urdf_path, s.hand_joint_names)
    assert np.abs(boxes - np.asarray(s.joint_link_boxes)).max() == 0.0


@pytest.mark.parametrize("spec_path", [GRAMMAR_000, ALLEGRO])
def test_extent_boxes_cover_each_link_exactly_and_keep_the_axis_edge(spec_path):
    s = get_robot_spec(spec_path)
    urdf = ROOT / s.urdf_path
    root = ET.parse(urdf).getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    joints = {j.get("name"): j for j in root.findall("joint")}
    _, boxes, valid, _ = ds.joint_link_boxes(s.urdf_path, s.hand_joint_names, mode="extent",
                                             capsules=s.replace_cylinders_with_capsules)
    _, old, _, _ = ds.joint_link_boxes(s.urdf_path, s.hand_joint_names)
    worst_old = 0.0
    for k, name in enumerate(s.hand_joint_names):
        if not valid[k]:
            continue
        child = joints[name].find("child").get("link")
        V = np.concatenate([ds._collision_vertices(c, urdf.parent, s.replace_cylinders_with_capsules)
                            for c in links[child].findall("collision")])
        assert _outside_mm(boxes[k], V) < 1e-3, name                     # covers the link, capsule ends included
        worst_old = max(worst_old, _outside_mm(old[k], V))
        axis = np.array(joints[name].find("axis").get("xyz").split(), float)
        e3 = boxes[k][3] - boxes[k][0]                                   # the axis edge: the joint axis, signed
        assert np.allclose(e3 / np.linalg.norm(e3), axis / np.linalg.norm(axis), atol=1e-6), name
    assert worst_old > 5.0          # the ordered box misses some of these links (what the extent mode fixes)


def test_joint_origin_is_an_optional_three_wide_token_field():
    s = get_robot_spec(ALLEGRO)
    base = ["joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel", "prev_action_targets",
            "joint_link_bbox", "joint_lower", "joint_upper", "joint_enabled", "keypoints_rel_ee"]
    old = build_token_layout(s, base)
    new = build_token_layout(s, base[:6] + ["joint_origin"] + base[6:])
    assert new["token_dim"] == old["token_dim"] + 3
    assert new["obs_dim"] == old["obs_dim"] + 3 * s.num_hand_joints
    a, b = new["token_field_slices"]["joint_origin"]
    assert b - a == 3
    with pytest.raises(ValueError):
        ds.joint_link_boxes(s.urdf_path, s.hand_joint_names, mode="tight")
