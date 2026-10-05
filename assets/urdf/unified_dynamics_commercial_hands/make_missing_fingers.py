"""Missing-finger variants of a uniform-dynamics hand: the same URDF with one finger's subtree removed.

For each finger -- the joints sharing a name prefix in the spec's canonical order, 4 per finger on Wuji v2 --
the finger's first joint and every link and joint below it are deleted from the URDF, and the spec JSON drops
the finger's joints, fingertip, pad offset and self-collision pairs. The palm, the other fingers, every
dynamic parameter and the palm frame are unchanged, so a variant differs from the full hand only by the
finger. Written next to the full hand (mesh paths are relative to that folder) as
``<hand>_left_no_<finger>.{urdf,spec.json}``; registered as ``<hand>_left_uniform_handonly_no_<finger>``.

    python3 assets/urdf/unified_dynamics_commercial_hands/make_missing_fingers.py [wuji2]
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINGERS = {"wuji2": {"thumb": "l_thumb_", "index": "l_index_finger_", "middle": "l_middle_finger_",
                     "ring": "l_ring_finger_", "pinky": "l_pinky_"}}


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
    written = []
    for finger, prefix in FINGERS[hand].items():
        tree = ET.parse(src_urdf)
        root = tree.getroot()
        fjoints = [j for j in spec["hand_joint_names"] if j.startswith(prefix)]
        links, joints = subtree(root, fjoints[0])
        for el in list(root):
            if (el.tag == "link" and el.get("name") in links) or (el.tag == "joint" and el.get("name") in joints):
                root.remove(el)
        out_urdf = HERE / hand / f"{hand}_left_no_{finger}.urdf"
        tree.write(out_urdf, encoding="utf-8", xml_declaration=True)

        s = dict(spec)
        keep = [i for i, t in enumerate(spec["fingertip_body_names"]) if t not in links]
        s["hand"] = f"{hand}_no_{finger}"
        s["urdf"] = str(out_urdf.relative_to(HERE.parents[2]))
        s["hand_joint_names"] = [j for j in spec["hand_joint_names"] if j not in joints]
        s["joint_limits"] = {k: v for k, v in spec["joint_limits"].items() if k not in joints}
        s["hand_default_joint_pos"] = {k: v for k, v in spec["hand_default_joint_pos"].items() if k not in joints}
        s["fingertip_body_names"] = [spec["fingertip_body_names"][i] for i in keep]
        s["fingertip_offsets"] = [spec["fingertip_offsets"][i] for i in keep]
        s["adjacent_links"] = {k: [v for v in vs if v not in links]
                               for k, vs in spec["adjacent_links"].items() if k not in links}
        s["missing_finger"] = {"finger": finger, "removed_joints": sorted(joints), "removed_links": sorted(links)}
        out_spec = HERE / hand / f"{hand}_left_no_{finger}.spec.json"
        out_spec.write_text(json.dumps(s, indent=1))
        print(f"{hand} no {finger:6s}: {len(s['hand_joint_names'])} joints, {len(s['fingertip_body_names'])} tips, "
              f"removed {len(links)} links / {len(joints)} joints")
        written += [out_urdf, out_spec]
    return written


if __name__ == "__main__":
    for h in sys.argv[1:] or ["wuji2"]:
        make(h)
