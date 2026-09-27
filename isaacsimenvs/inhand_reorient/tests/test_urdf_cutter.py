"""URDF-cutter tests. Pure stdlib: run with system python3, no Isaac, no torch.

    PYTHONDONTWRITEBYTECODE=1 python3 -m pytest isaacsimenvs/inhand_reorient/tests/test_urdf_cutter.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from isaacsimenvs.inhand_reorient.urdf_cutter import (
    cut_urdf_to_hand, leaf_link_names, merged_body_name, merged_terminal_names,
    movable_joint_names,
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


def test_merged_body_name_walks_the_fixed_joint_chain_to_the_actuated_body():
    # left_pinky_fingertip -fixed-> left_pinky_elastomer -fixed-> left_pinky_DP
    # (left_pinky_DP itself is the child of left_pinky_DIP, a REVOLUTE joint):
    # the URDF importer's merge_fixed_joints folds both fixed-joint links into
    # left_pinky_DP, which is the name that actually survives into the
    # imported articulation (confirmed against its own import-time merge log:
    # "link left_pinky_fingertip ... merged into left_pinky_elastomer" then
    # "left_pinky_elastomer ... merged into left_pinky_DP"). I24: a fingertip
    # lookup by the pre-merge leaf name (what leaf_link_names returns) has to
    # resolve through this to find a body Articulation.data.body_names
    # actually has.
    cut = cut_urdf_to_hand(SHARPA_URDF, "left_hand_C_MC")
    assert merged_body_name(cut.root, "left_pinky_fingertip") == "left_pinky_DP"


def test_merged_body_name_is_identity_past_the_nearest_movable_joint():
    cut = cut_urdf_to_hand(SHARPA_URDF, "left_hand_C_MC")
    # left_pinky_DP is itself the child of a REVOLUTE joint (left_pinky_DIP):
    # nothing merges past it walking further up, so it is already its own
    # surviving name.
    assert merged_body_name(cut.root, "left_pinky_DP") == "left_pinky_DP"
    # The articulation root has no parent joint at all; also identity.
    assert merged_body_name(cut.root, cut.hand_root) == cut.hand_root


# --------------------------------------------------------------------------
# merged_terminal_names (I34, 2026-09-27): a synthetic tree reproducing
# xhand_right's thumb (root -revolute-> A -[fixed-> A_back, revolute]-> B
# -[fixed-> B_back, fixed-> B_tip]) without needing that hand's real (mesh-
# only, machine-local) source URDF. A naive leaf-merge (leaf_link_names +
# merged_body_name, what build_hand_only_spec used before I34) reports BOTH
# A (via A_back) and B (via B_back/B_tip) as "fingertips" -- A is a MID-CHAIN
# link (it also continues to B through a movable joint), not a terminus.
# --------------------------------------------------------------------------

def _dead_end_branch_urdf() -> ET.Element:
    xml = """<robot name="synthetic">
      <link name="root"/><link name="A"/><link name="A_back"/>
      <link name="B"/><link name="B_back"/><link name="B_tip"/>
      <joint name="root_A" type="revolute">
        <parent link="root"/><child link="A"/>
      </joint>
      <joint name="A_back_j" type="fixed">
        <parent link="A"/><child link="A_back"/>
      </joint>
      <joint name="A_B" type="revolute">
        <parent link="A"/><child link="B"/>
      </joint>
      <joint name="B_back_j" type="fixed">
        <parent link="B"/><child link="B_back"/>
      </joint>
      <joint name="B_tip_j" type="fixed">
        <parent link="B"/><child link="B_tip"/>
      </joint>
    </robot>"""
    import xml.etree.ElementTree as ET

    return ET.fromstring(xml)


def test_naive_leaf_merge_wrongly_promotes_the_mid_chain_dead_end():
    # Documents the bug merged_terminal_names fixes: the naive combination
    # every caller used before I34 resolves A's dead end up to A itself --
    # A is a knuckle, not a tip, but nothing in leaf_link_names/
    # merged_body_name alone can tell.
    root = _dead_end_branch_urdf()
    naive = tuple(dict.fromkeys(merged_body_name(root, t) for t in leaf_link_names(root)))
    assert set(naive) == {"A", "B"}


def test_merged_terminal_names_excludes_the_mid_chain_dead_end():
    root = _dead_end_branch_urdf()
    assert set(merged_terminal_names(root)) == {"B"}


def test_merged_terminal_names_is_a_noop_when_every_leaf_is_already_a_terminus():
    # dclaw: no dead-end branching, so merged_terminal_names must match the
    # naive merge exactly (regression guard against the filter being
    # over-eager).
    cut = cut_urdf_to_hand(DCLAW_URDF, None)
    naive = tuple(dict.fromkeys(merged_body_name(cut.root, t) for t in leaf_link_names(cut.root)))
    assert set(merged_terminal_names(cut.root)) == set(naive) == {
        "link_f1_3", "link_f2_3", "link_f3_3"}
