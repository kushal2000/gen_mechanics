"""URDF-cutter tests. Pure stdlib: run with system python3, no Isaac, no torch.

    PYTHONDONTWRITEBYTECODE=1 python3 -m pytest isaacsimenvs/inhand_reorient/tests/test_urdf_cutter.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from isaacsimenvs.inhand_reorient.urdf_cutter import (
    cut_urdf_to_hand, leaf_link_names, movable_joint_names,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SHARPA_URDF = REPO_ROOT / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
ALLEGRO_URDF = (REPO_ROOT / "hand_sampler/grammar_bench/fixtures/real/allegro_right"
                 / "allegro_hand_description_right.urdf")
DCLAW_URDF = REPO_ROOT / "hand_sampler/grammar_bench/fixtures/real/dclaw/dclaw_gripper.urdf"

SHARPA_HAND_JOINT_NAMES = (
    "left_1_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE",
    "left_thumb_MCP_AA", "left_thumb_IP",
    "left_2_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
    "left_3_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
    "left_4_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
    "left_5_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA",
    "left_pinky_PIP", "left_pinky_DIP",
)


def test_sharpa_cut_keeps_exactly_the_22_hand_joints():
    cut = cut_urdf_to_hand(SHARPA_URDF, "left_hand_C_MC")
    assert cut.hand_root == "left_hand_C_MC"
    movable = movable_joint_names(cut.root)
    assert set(movable) == set(SHARPA_HAND_JOINT_NAMES)
    assert len(movable) == 22
    # Nothing above the hand (the arm, the mount) survives the cut.
    assert "iiwa14_link_0" in cut.dropped_above
    assert "sharpa_mount" in cut.dropped_above
    assert "left_hand_C_MC" not in cut.dropped_above


def test_sharpa_cut_mesh_paths_are_absolute_and_exist():
    cut = cut_urdf_to_hand(SHARPA_URDF, "left_hand_C_MC")
    n_mesh = 0
    for link in cut.root.findall("link"):
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                geom = el.find("geometry")
                mesh = geom.find("mesh") if geom is not None else None
                if mesh is None:
                    continue
                n_mesh += 1
                filename = mesh.get("filename")
                assert Path(filename).is_absolute(), filename
                assert Path(filename).is_file(), filename
    assert n_mesh > 0
    # SHARPA vendors its own meshes, so nothing should have been dropped.
    assert cut.unresolved_meshes == ()


def test_sharpa_cut_joints_have_no_mimic():
    cut = cut_urdf_to_hand(SHARPA_URDF, "left_hand_C_MC")
    for joint in cut.root.findall("joint"):
        assert joint.find("mimic") is None


def test_sharpa_cut_is_a_tree_rooted_at_hand_root():
    cut = cut_urdf_to_hand(SHARPA_URDF, "left_hand_C_MC")
    children = {j.find("child").get("link") for j in cut.root.findall("joint")}
    parents = {j.find("parent").get("link") for j in cut.root.findall("joint")}
    assert "left_hand_C_MC" not in children  # the root is nobody's child
    assert "left_hand_C_MC" in parents
    link_names = {l.get("name") for l in cut.root.findall("link")}
    assert link_names == set(cut.kept_links)
    assert children | {"left_hand_C_MC"} == link_names


def test_sharpa_cut_writes_a_parseable_urdf(tmp_path):
    cut = cut_urdf_to_hand(SHARPA_URDF, "left_hand_C_MC")
    out = cut.write(tmp_path / "sharpa_hand.urdf")
    import xml.etree.ElementTree as ET
    reparsed = ET.parse(out).getroot()
    assert reparsed.tag == "robot"
    assert len(reparsed.findall("link")) == len(cut.kept_links)


def test_hand_root_none_uses_the_urdf_own_root():
    cut = cut_urdf_to_hand(ALLEGRO_URDF, None)
    assert cut.hand_root == "hand_root"
    assert set(cut.kept_links) == {l.get("name") for l in
                                    __import__("xml.etree.ElementTree", fromlist=["ET"])
                                    .parse(ALLEGRO_URDF).getroot().findall("link")}
    assert cut.dropped_above == ()


def test_allegro_box_collisions_need_no_mesh_resolution():
    # Allegro's collision geometry is boxes; only its *visual* meshes are
    # package:// URIs that this repo does not vendor -- unresolved, so
    # dropped, not left dangling for Isaac's importer to choke on.
    cut = cut_urdf_to_hand(ALLEGRO_URDF, None)
    assert len(cut.unresolved_meshes) > 0
    for uri in cut.unresolved_meshes:
        assert uri.startswith("package://")
    for link in cut.root.findall("link"):
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                geom = el.find("geometry")
                mesh = geom.find("mesh") if geom is not None else None
                if mesh is not None:
                    assert Path(mesh.get("filename")).is_absolute()


def test_dclaw_movable_joints_are_the_9_finger_joints():
    cut = cut_urdf_to_hand(DCLAW_URDF, None)
    movable = movable_joint_names(cut.root)
    assert len(movable) == 9
    assert all(j.startswith("joint_f") for j in movable)


def test_leaf_link_names_fallback_tips():
    cut = cut_urdf_to_hand(DCLAW_URDF, None)
    leaves = leaf_link_names(cut.root)
    assert set(leaves) == {"link_f1_head", "link_f2_head", "link_f3_head"}


def test_missing_hand_root_raises():
    with pytest.raises(ValueError):
        cut_urdf_to_hand(SHARPA_URDF, "not_a_real_link")
