"""Cut a URDF down to one hand, by XML subtree extraction below ``hand_root``.

Pure stdlib (``xml.etree`` + ``pathlib``): no Isaac, no torch, no trimesh at
import time, so this runs under plain system python3.

Two things a hand-only Isaac scene needs that a full arm+hand URDF does not
give directly:

1. Only the subtree at and below ``hand_root`` (the arm, the mount, and
   anything else the hand sits on is not part of the articulation we author).
   ``hand_root=None`` means the whole URDF is already hand-only (its own
   root link becomes the articulation root) -- several manifest entries
   (allegro_right, dclaw, wuji_right, xhand_right, tesollo_dg5f_right) are
   like this.
2. Mesh filenames resolved to absolute paths that exist on disk. Isaac's URDF
   importer has no ROS package path, so a ``package://`` URI never resolves
   inside it; resolution here uses the same directory-walk
   ``hand_sampler.grammar_bench.refgen.mesh_sections.resolve_mesh_filename``
   already uses for the same manifest hands. A reference that does not
   resolve to a file on disk (mesh assets this repo does not vendor, e.g.
   D'Claw's ``meshes/`` are not committed) is DROPPED rather than left
   dangling -- ``design_space.joint_link_boxes`` already degrades a link with
   no collision geometry to ``valid=False`` rather than crashing, whereas an
   unresolvable filename reaching ``trimesh.load`` raises.

Mimic joints are rewritten to independent, unmimicked joints (the ``<mimic>``
child is dropped): "one motor per joint" applies to a cut commercial hand the
same way it applies to a generated one.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from hand_sampler.grammar_bench.refgen.mesh_sections import resolve_mesh_filename

__all__ = ["CutURDF", "cut_urdf_to_hand", "movable_joint_names", "leaf_link_names"]

_GEOMETRY_TAGS = ("visual", "collision")


@dataclass(frozen=True)
class CutURDF:
    """What the cut produced."""

    root: ET.Element
    """The new ``<robot>`` element -- hand_root's subtree only."""
    hand_root: str
    """Root link name (the articulation's base link)."""
    kept_links: tuple[str, ...]
    kept_joints: tuple[str, ...]
    dropped_above: tuple[str, ...]
    """Links that existed in the source URDF but are not under hand_root."""
    unresolved_meshes: tuple[str, ...]
    """Mesh URIs that could not be resolved to a file on disk and were
    dropped from the cut (their <visual>/<collision> element is removed)."""

    def write(self, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(self.root).write(out_path, encoding="unicode", xml_declaration=True)
        return out_path


def _subtree_links(root: ET.Element, hand_root: str) -> tuple[set[str], set[str]]:
    """``(kept_links, kept_joints)`` reachable from ``hand_root`` going down
    (i.e. following joints where the visited link is the PARENT)."""
    joints_by_parent: dict[str, list[ET.Element]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        joints_by_parent.setdefault(parent, []).append(joint)

    kept_links = {hand_root}
    kept_joints: set[str] = set()
    frontier = [hand_root]
    while frontier:
        nxt = []
        for link_name in frontier:
            for joint in joints_by_parent.get(link_name, ()):
                jname = joint.get("name")
                child = joint.find("child").get("link")
                if jname in kept_joints:
                    continue
                kept_joints.add(jname)
                if child not in kept_links:
                    kept_links.add(child)
                    nxt.append(child)
        frontier = nxt
    return kept_links, kept_joints


def _strip_mimic(joint: ET.Element) -> None:
    for mimic in joint.findall("mimic"):
        joint.remove(mimic)


def _resolve_mesh_elements(link: ET.Element, urdf_dir: Path) -> list[str]:
    """Rewrite every ``<mesh filename=...>`` under ``link`` to an absolute,
    existing path; remove the enclosing visual/collision when it does not
    resolve. Returns the unresolved URIs (for reporting)."""
    unresolved: list[str] = []
    for tag in _GEOMETRY_TAGS:
        for el in list(link.findall(tag)):
            geom = el.find("geometry")
            mesh = geom.find("mesh") if geom is not None else None
            if mesh is None:
                continue
            uri = mesh.get("filename", "")
            resolved = resolve_mesh_filename(uri, urdf_dir)
            if resolved is None:
                unresolved.append(uri)
                link.remove(el)
            else:
                mesh.set("filename", str(resolved))
    return unresolved


def cut_urdf_to_hand(urdf_path: str | Path, hand_root: str | None) -> CutURDF:
    """Extract ``hand_root``'s subtree from ``urdf_path`` (or the whole robot
    when ``hand_root`` is None), with mesh paths resolved to absolute."""
    urdf_path = Path(urdf_path)
    source = ET.parse(urdf_path).getroot()
    urdf_dir = urdf_path.parent

    all_links = {link.get("name") for link in source.findall("link")}
    if hand_root is None:
        # The whole URDF is the hand: its own root is whichever link is never
        # a joint's child (URDF requires exactly one, for a tree).
        children = {j.find("child").get("link") for j in source.findall("joint")}
        roots = [name for name in all_links if name not in children]
        if len(roots) != 1:
            raise ValueError(
                f"{urdf_path}: expected exactly one root link with hand_root=None, "
                f"found {roots!r}")
        hand_root = roots[0]

    if hand_root not in all_links:
        raise ValueError(f"{urdf_path}: hand_root {hand_root!r} is not a <link>")

    kept_links, kept_joints = _subtree_links(source, hand_root)

    new_root = ET.Element("robot", {"name": source.get("name", "hand") + "_cut"})
    unresolved: list[str] = []
    for link in source.findall("link"):
        name = link.get("name")
        if name not in kept_links:
            continue
        link_copy = ET.fromstring(ET.tostring(link))
        unresolved.extend(_resolve_mesh_elements(link_copy, urdf_dir))
        new_root.append(link_copy)
    for joint in source.findall("joint"):
        if joint.get("name") not in kept_joints:
            continue
        joint_copy = ET.fromstring(ET.tostring(joint))
        _strip_mimic(joint_copy)
        new_root.append(joint_copy)

    dropped_above = tuple(sorted(all_links - kept_links))
    return CutURDF(
        root=new_root, hand_root=hand_root,
        kept_links=tuple(sorted(kept_links)), kept_joints=tuple(sorted(kept_joints)),
        dropped_above=dropped_above, unresolved_meshes=tuple(unresolved),
    )


def movable_joint_names(root: ET.Element) -> tuple[str, ...]:
    """Non-fixed joints, in document order -- the actuated hand joints."""
    return tuple(
        j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"
    )


def leaf_link_names(root: ET.Element) -> tuple[str, ...]:
    """Links that are no joint's parent, in link-declaration order -- the
    fingertip fallback when the manifest names no tip frames."""
    parents = {j.find("parent").get("link") for j in root.findall("joint")}
    return tuple(l.get("name") for l in root.findall("link") if l.get("name") not in parents)
