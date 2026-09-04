"""Author one URDF per dropped finger, by removing the finger's whole subtree.

Every finger hangs off ``left_hand_C_MC`` by exactly one root joint, so a
variant is a clean subtree deletion: take the root joint's child link, walk
down, and drop every link and joint reached. Nothing is renamed and nothing
else is touched, so the remaining joints keep the names, order, limits and
geometry the checkpoint was trained with -- which is what lets the same
weights drive the reduced hand.

    python experiments/decentralized_control/eval/make_finger_variants.py
"""

from __future__ import annotations

import pathlib
import shutil
import xml.etree.ElementTree as ET

SRC = pathlib.Path("assets/urdf/kuka_sharpa_description/"
                   "iiwa14_left_sharpa_adjusted_restricted.urdf")
OUT = pathlib.Path("experiments/decentralized_control/eval/assets")
# Root joint per finger: the one whose parent is the palm, not the finger.
ROOT_JOINT = {
    "thumb": "left_1_thumb_CMC_FE",
    "index": "left_2_index_MCP_FE",
    "middle": "left_3_middle_MCP_FE",
    "ring": "left_4_ring_MCP_FE",
    "pinky": "left_5_pinky_CMC",
}


def drop_subtree(tree: ET.ElementTree, root_joint: str):
    """Remove root_joint and everything below it. Returns (links, joints) cut."""
    root = tree.getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    links = {l.get("name"): l for l in root.findall("link")}
    children: dict[str, list[str]] = {}
    for j in root.findall("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(j.get("name"))

    cut_joints = {root_joint}
    frontier = [joints[root_joint].find("child").get("link")]
    cut_links: set[str] = set()
    while frontier:
        link = frontier.pop()
        if link in cut_links:
            continue
        cut_links.add(link)
        for jn in children.get(link, []):
            cut_joints.add(jn)
            frontier.append(joints[jn].find("child").get("link"))

    for name in cut_joints:
        root.remove(joints[name])
    for name in cut_links:
        root.remove(links[name])
    return sorted(cut_links), sorted(cut_joints)


def link_meshes() -> None:
    """Symlink the mesh trees next to the variants.

    The URDF references meshes relatively ("left_sharpa_meshes/...",
    "new_iiwa14_meshes/..."), and both trimesh (via joint_link_boxes) and the
    Isaac URDF importer resolve them against the URDF's own directory. Symlinks
    rather than copies: 15 MB, and one source of truth if a mesh is ever fixed.
    """
    for name in ("left_sharpa_meshes", "new_iiwa14_meshes"):
        src = (SRC.parent / name).resolve()
        dst = OUT / name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src, target_is_directory=True)
        print(f"  {dst} -> {src}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    link_meshes()
    # The intact hand is copied in too, so all six variants are read from one
    # directory and the control is not a special case in the eval script.
    shutil.copy(SRC, OUT / "sharpa_intact.urdf")
    base = ET.parse(SRC)
    n_link = len(base.getroot().findall("link"))
    n_joint = len(base.getroot().findall("joint"))
    print(f"source: {n_link} links, {n_joint} joints -> {OUT}/sharpa_intact.urdf\n")

    for finger, root_joint in ROOT_JOINT.items():
        tree = ET.parse(SRC)
        cut_links, cut_joints = drop_subtree(tree, root_joint)
        revolute = [j for j in cut_joints
                    if tree_type(ET.parse(SRC), j) == "revolute"]
        dst = OUT / f"sharpa_no_{finger}.urdf"
        tree.write(dst, encoding="utf-8", xml_declaration=True)
        left_l = len(tree.getroot().findall("link"))
        left_j = len(tree.getroot().findall("joint"))
        print(f"no_{finger:<7} cut {len(cut_links)} links, {len(cut_joints)} joints "
              f"({len(revolute)} revolute)  ->  {left_l} links, {left_j} joints")
        print(f"           links:  {', '.join(cut_links)}")
        print(f"           joints: {', '.join(cut_joints)}")


def tree_type(tree: ET.ElementTree, joint_name: str) -> str:
    for j in tree.getroot().findall("joint"):
        if j.get("name") == joint_name:
            return j.get("type")
    return ""


if __name__ == "__main__":
    main()
