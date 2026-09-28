"""Build the unified hand-only SHARPA URDF from the repo's arm+hand description.

    python assets/urdf/unified_commercial_hands/sharpa/unify_from_vendor.py

SHARPA is not a commercial hand -- it lives in this folder because the POINT of the
folder is one pipeline, and having the in-house reference go through the same steps as
a vendor hand is what makes the two comparable. If SHARPA needed a special case here,
that would be a fact about the pipeline worth knowing.

WHAT THIS IS FOR. Every in-hand run so far has trained on sharpa_capsule.json, whose
provenance says `method: sharpa_capsule` -- SHARPA rebuilt inside the design grammar,
where build.author_hand writes CAPSULES and never looks at a mesh. That is a useful
control but it is not the hand. This produces the real one: the measured URDF's own
links, meshes, limits and mount geometry, with the arm removed.

WHAT IS DONE, AND WHAT IS NOT

  1. The hand subtree is extracted, rooted at ``left_hand_C_MC``. That link is the
     palm, and the chain above it -- iiwa14_link_7 -> link_ee (z 0.045) -> sharpa_mount
     (yaw +0.261799) -> left_hand_C_MC (z 0.05, yaw -1.5708) -- composes to exactly
     ``build.flange_to_palm``: yaw -1.309 rad = -75 deg and z 0.095. Verified below,
     because that identity is what lets the spec use the palm's own frame unchanged
     rather than adapting one.

  2. Mesh paths are rewritten to point back at the vendor folder rather than copied.
     Both trees are in-repo, so a relative path resolves for ``joint_link_boxes``
     (which resolves against the URDF's own directory) and 40 STL files are not
     duplicated. Allegro's meshes ARE copied, because they came from outside the repo.

  3. Nothing is repaired. Unlike Allegro's vendor file, this URDF is well formed, its
     collision geometry is authored, and its limits admit the zero pose. If a future
     check finds otherwise, the repair belongs here where it is visible.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
VENDOR_URDF = REPO / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
PALM_LINK = "left_hand_C_MC"
# Where the vendor meshes live, relative to THIS folder.
MESH_PREFIX = "../../kuka_sharpa_description"


def main() -> None:
    root = ET.parse(VENDOR_URDF).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    links = {l.get("name"): l for l in root.findall("link")}

    # --- the mount chain must compose to flange_to_palm ---------------------------
    # Walk iiwa14_link_7 down to the palm, accumulating yaw and z. If this ever stops
    # matching, the spec's palm frame is wrong and every observation with it.
    yaw = z = 0.0
    link, chain = "iiwa14_link_7", []
    while link != PALM_LINK:
        nxt = [j for j in joints.values() if j.find("parent").get("link") == link
               and j.get("type") == "fixed"]
        if not nxt:
            raise SystemExit(f"no fixed joint out of {link}; cannot reach {PALM_LINK}")
        j = nxt[0]
        o = j.find("origin")
        yaw += float((o.get("rpy") or "0 0 0").split()[2])
        z += float((o.get("xyz") or "0 0 0").split()[2])
        link = j.find("child").get("link")
        chain.append(j.get("name"))
    from hand_sampler import robot_param_constants as rpc
    want_yaw = rpc.FLANGE_TO_PALM_YAW_RAD
    want_z = rpc.LINK7_TO_FLANGE_Z_M + rpc.FLANGE_TO_PALM_Z_M
    print(f"mount chain {' -> '.join(chain)}")
    print(f"  composed yaw {yaw:+.6f} rad ({math.degrees(yaw):+.3f} deg), z {z:.6f} m")
    print(f"  flange_to_palm says yaw {want_yaw:+.6f}, z {want_z:.6f}")
    # 1e-5 rad, not 1e-6: the URDF writes its mount angles as 0.261799 and -1.5708,
    # rounded, so the walked chain lands 4.1e-6 rad (0.23 millidegrees) from
    # -radians(75). That is the asset's own precision, not a disagreement -- a tighter
    # bound here just fails on rounding. z is exact because 0.045 and 0.05 are.
    assert abs(yaw - want_yaw) < 1e-5, \
        f"mount yaw {yaw} disagrees with flange_to_palm {want_yaw} by {abs(yaw-want_yaw):.2e}"
    assert abs(z - want_z) < 1e-9, "mount z disagrees with flange_to_palm"

    # --- the subtree ---------------------------------------------------------------
    kids: dict[str, list[ET.Element]] = {}
    for j in joints.values():
        kids.setdefault(j.find("parent").get("link"), []).append(j)
    keep_links, keep_joints, stack = set(), [], [PALM_LINK]
    while stack:
        l = stack.pop()
        if l in keep_links:
            continue
        keep_links.add(l)
        for j in kids.get(l, []):
            keep_joints.append(j)
            stack.append(j.find("child").get("link"))
    print(f"subtree from {PALM_LINK}: {len(keep_links)} links, {len(keep_joints)} joints "
          f"({sum(j.get('type') == 'revolute' for j in keep_joints)} revolute)")

    out = ET.Element("robot", {"name": "sharpa_left_unified"})
    out.append(ET.Comment(" GENERATED by unify_from_vendor.py. Do not hand-edit; regenerate. "))
    for m in root.findall("material"):
        out.append(m)
    for name in sorted(keep_links):
        out.append(links[name])
    for j in keep_joints:
        out.append(j)
    for mesh in out.iter("mesh"):
        mesh.set("filename", f"{MESH_PREFIX}/{mesh.get('filename')}")

    ET.indent(out, space="  ")
    dest = HERE / "sharpa.urdf"
    dest.write_text('<?xml version="1.0"?>\n' + ET.tostring(out, encoding="unicode") + "\n")

    # --- checks --------------------------------------------------------------------
    r = ET.parse(dest).getroot()
    ls = [l.get("name") for l in r.findall("link")]
    js = r.findall("joint")
    children = {j.find("child").get("link") for j in js}
    orphans = [l for l in ls if l not in children and l != PALM_LINK]
    assert not orphans, f"orphan links: {orphans}"
    assert len(ls) == len(js) + 1, f"not a tree: {len(ls)} links, {len(js)} joints"
    rev = {j.get("name") for j in js if j.get("type") == "revolute"}
    assert set(rpc.HAND_JOINT_NAMES) <= rev, \
        f"missing spec joints: {sorted(set(rpc.HAND_JOINT_NAMES) - rev)}"
    assert set(rpc.FINGERTIP_BODY_NAMES) <= set(ls), "a fingertip body is missing"
    for mesh in r.iter("mesh"):
        f = mesh.get("filename")
        assert not f.startswith("package://"), f"package URI: {f}"
        assert (HERE / f).exists(), f"missing mesh {f}"
    print(f"wrote {dest}")
    print(f"checks pass: {len(ls)} links, {len(js)} joints, {len(rev)} revolute, "
          f"all {len(rpc.HAND_JOINT_NAMES)} spec joints and "
          f"{len(rpc.FINGERTIP_BODY_NAMES)} fingertips present, every mesh resolves")


if __name__ == "__main__":
    main()
