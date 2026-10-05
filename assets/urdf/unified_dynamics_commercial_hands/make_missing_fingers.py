"""Missing-finger variants of a uniform-dynamics hand: the same URDF with one finger's subtree removed.

A finger is found from the URDF, not from joint names: for each fingertip in the spec, walk the tree up from the
tip to the palm body; the joint on that path whose parent is the palm is the finger's root. That joint and every
link and joint below it are deleted from the URDF, and the spec JSON drops the finger's joints, fingertip, pad
offset and self-collision pairs. The palm, the other fingers, every dynamic parameter and the palm frame are
unchanged, so a variant differs from the full hand only by the finger. Written next to the full hand (mesh paths
are relative to that folder) as ``<hand>_left_no_<finger>.{urdf,spec.json}``; registered as
``<hand>_left_uniform_handonly_no_<finger>``.

    python3 assets/urdf/unified_dynamics_commercial_hands/make_missing_fingers.py [hand ...]
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Finger names, in the order of each spec's fingertip_body_names.
FINGER_NAMES = {
    "wuji2": ["thumb", "index", "middle", "ring", "pinky"],
    "shadow": ["index", "middle", "ring", "pinky", "thumb"],      # ff, mf, rf, lf, th
    "tesollo": ["thumb", "index", "middle", "ring", "pinky"],     # dg_1 is the thumb (spec thumb_tip)
    "xhand": ["thumb", "index", "middle", "ring", "pinky"],
    "allegro": ["index", "middle", "ring", "thumb"],              # link_15_0 is the thumb (spec thumb_tip)
}


def finger_root(root: ET.Element, palm: str, tip: str) -> str:
    """The joint on the palm -> tip path whose parent is the palm."""
    parent_joint = {j.find("child").get("link"): j for j in root.findall("joint")}
    link = tip
    while link in parent_joint:
        j = parent_joint[link]
        if j.find("parent").get("link") == palm:
            return j.get("name")
        link = j.find("parent").get("link")
    raise ValueError(f"{tip} does not hang off {palm}")


def subtree(root: ET.Element, first_joint: str) -> tuple[set[str], set[str]]:
    """(links, joints) at and below ``first_joint``."""
    children: dict[str, list[ET.Element]] = {}
    for j in root.findall("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(j)
    joint = next(j for j in root.findall("joint") if j.get("name") == first_joint)
    links, joints, stack = set(), {first_joint}, [joint.find("child").get("link")]
    while stack:
        link = stack.pop()
        links.add(link)
        for j in children.get(link, []):
            joints.add(j.get("name"))
            stack.append(j.find("child").get("link"))
    return links, joints


def make(hand: str) -> list[Path]:
    src_urdf = HERE / hand / f"{hand}_left.urdf"
    spec = json.loads((HERE / hand / f"{hand}_left.spec.json").read_text())
    names = FINGER_NAMES[hand]
    tips = spec["fingertip_body_names"]
    if len(names) != len(tips):
        raise ValueError(f"{hand}: {len(names)} finger names for {len(tips)} fingertips")
    if spec.get("thumb_tip") and names[tips.index(spec["thumb_tip"])] != "thumb":
        raise ValueError(f"{hand}: spec thumb_tip {spec['thumb_tip']} is not named 'thumb'")
    written = []
    for finger, tip in zip(names, tips):
        tree = ET.parse(src_urdf)
        root = tree.getroot()
        links, joints = subtree(root, finger_root(root, spec["palm_body"], tip))
        assert tip in links
        for el in list(root):
            if (el.tag == "link" and el.get("name") in links) or (el.tag == "joint" and el.get("name") in joints):
                root.remove(el)
        out_urdf = HERE / hand / f"{hand}_left_no_{finger}.urdf"
        tree.write(out_urdf, encoding="utf-8", xml_declaration=True)

        s = dict(spec)
        keep = [i for i, t in enumerate(tips) if t not in links]
        s["hand"] = f"{hand}_no_{finger}"
        s["urdf"] = str(out_urdf.relative_to(HERE.parents[2]))
        s["hand_joint_names"] = [j for j in spec["hand_joint_names"] if j not in joints]
        s["joint_limits"] = {k: v for k, v in spec["joint_limits"].items() if k not in joints}
        s["hand_default_joint_pos"] = {k: v for k, v in spec["hand_default_joint_pos"].items() if k not in joints}
        s["fingertip_body_names"] = [tips[i] for i in keep]
        s["fingertip_offsets"] = [spec["fingertip_offsets"][i] for i in keep]
        s["adjacent_links"] = {k: [v for v in vs if v not in links]
                               for k, vs in spec["adjacent_links"].items() if k not in links}
        if s.get("thumb_tip") in links:
            s["thumb_tip"] = None
        s["missing_finger"] = {"finger": finger, "removed_joints": sorted(joints), "removed_links": sorted(links)}
        out_spec = HERE / hand / f"{hand}_left_no_{finger}.spec.json"
        out_spec.write_text(json.dumps(s, indent=1))
        n_rev = sum(1 for j in spec["hand_joint_names"] if j in joints)
        print(f"{hand:8s} no {finger:6s}: {len(s['hand_joint_names'])} joints ({n_rev} removed), "
              f"{len(s['fingertip_body_names'])} tips, removed {len(links)} links")
        written += [out_urdf, out_spec]
    return written


if __name__ == "__main__":
    for h in sys.argv[1:] or ["wuji2"]:
        make(h)
