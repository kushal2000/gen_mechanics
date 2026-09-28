"""Build the unified Allegro hand URDF from the vendor description.

Run this, do not hand-edit allegro.urdf -- the repairs below are the provenance of
the asset, and a hand-edit loses them.

    python assets/urdf/unified_commercial_hands/allegro/unify_from_vendor.py

WHERE THE INPUTS COME FROM

  vendor hand   /share/portal/kk837/simtoolreal/isaacgym/assets/urdf/
                kuka_allegro_description/allegro.urdf
                Wonik Allegro, right hand, as shipped with Isaac Gym. Its joints are
                index/middle/ring/thumb_joint_0..3, the same naming IsaacLab's
                allegro_hand_instanceable.usd uses -- NOT the joint_0.0 .. joint_15.0
                of ManiSkill's copy, which is a different description entirely.

  inertials     this repo at e4eb91a^, assets/urdf/
                kuka_allegro_iiwa14_description/iiwa14_allegro.urdf
                A previous integration, deleted in e4eb91a ("Delete Allegro"), which
                had already lifted the good index/middle/ring inertials out of the
                vendor's kuka_allegro.urdf. Recovered rather than redone.

WHAT IS REPAIRED, AND WHY EACH ONE MATTERS

  1. The four *_biotac_tip links are DROPPED. index_biotac_tip has no parent joint at
     all -- the vendor file lost it to a stray "-->" at line 231 with no opening
     marker -- leaving 22 links against 20 joints, which is not a tree. The tips are
     massful frames with no geometry, so nothing is lost: the real contact surface is
     the biotac mesh on *_link_3, which becomes the fingertip body.

  2. The palm collider becomes a BOX. base_link.obj is two disconnected pieces: the
     real 100 x 117 x 41 mm slab, and a stray 88 x 117 x 3 mm plate floating 54.5 mm
     behind it with no vertices in between (verified: 8574 verts at z > -0.05, 474 at
     z < -0.095). PhysX cannot use a triangle mesh on a dynamic body, so it would
     convex-hull that into a 98 mm thick box -- 2.4x the real palm -- swallowing the
     whole wrist region. The mesh stays as the VISUAL; only the collider changes.

  3. The palm inertia is recomputed from the slab. The vendor declares 1e-4 isotropic
     against m = 0.4154 kg, which understates izz by 8.2x.

  4. The arm is dropped and the tree is rooted at palm_link. The hand-only task mounts
     the palm directly, so allegro_mount and the iiwa chain are dead weight -- and
     rooting at palm_link makes palm_body_name a real body rather than something that
     has to survive merge_fixed_joints.

  5. Mesh paths are rewritten relative to this file's directory. package:// URIs are
     fatal to design_space.joint_link_boxes (design_space.py:682), which is what reads
     the meshes to build the policy's per-joint token geometry.

NOT REPAIRED, recorded so it is a known quantity rather than a surprise:

  * Collision geometry on the fingers is unsimplified triangle mesh, identical to the
    visual, up to 17850 verts on a biotac. PhysX convex-hulls each one. Fine for a
    first run; decimate if it shows up in throughput.
  * thumb_joint_0's lower limit is +0.2792 rad, so q = 0 is OUT OF RANGE. Left as the
    vendor has it, because it is the hardware's real range -- the spec's
    hand_default_joint_pos must start the thumb at 0.2792, and a naive zero pose will
    fight the limit.
  * The four thumb links keep the vendor's 0.1 kg placeholder masses and the isotropic
    1e-4 inertia; only index/middle/ring have real values upstream.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VENDOR = Path("/share/portal/kk837/simtoolreal/isaacgym/assets/urdf/kuka_allegro_description")
RECOVERED = Path("/tmp/kk_iiwa14_allegro.urdf")   # git show e4eb91a^:.../iiwa14_allegro.urdf

HAND_JOINTS = tuple(
    f"{finger}_joint_{i}" for finger in ("index", "middle", "ring", "thumb") for i in range(4)
)
ROOT_LINK = "palm_link"
PALM_MASS_KG = 0.4154


def _palm_slab(mesh: Path) -> tuple[np.ndarray, np.ndarray]:
    """(size, centre) of base_link.obj's MAIN body, excluding the stray plate.

    Split on the 54.5 mm void rather than a hard-coded z: if the vendor ever ships a
    repaired mesh the split finds nothing to discard and this still returns the slab.
    """
    v = np.array(
        [[float(x) for x in line.split()[1:4]] for line in mesh.read_text().splitlines()
         if line.startswith("v ")]
    )
    z = np.sort(np.unique(np.round(v[:, 2], 4)))
    gaps = [(z[i], z[i + 1]) for i in range(len(z) - 1) if z[i + 1] - z[i] > 0.01]
    if len(gaps) > 1:
        raise SystemExit(f"{mesh.name}: expected at most one void in z, found {gaps}")
    keep = v if not gaps else v[v[:, 2] > gaps[0][1] - 1e-9]
    lo, hi = keep.min(0), keep.max(0)
    print(f"  palm slab from {len(keep)}/{len(v)} verts: size {np.round(hi - lo, 5).tolist()}"
          f"  centre {np.round((hi + lo) / 2, 5).tolist()}")
    return hi - lo, (hi + lo) / 2


def _mesh_name(filename: str) -> str:
    """Vendor or mirrored path -> this folder's layout."""
    stem = Path(filename).name
    # the recovered file flattened names as allegro_meshes__allegro__base_link.obj
    stem = stem.replace("allegro_meshes__", "").replace("__", "/")
    if "/" not in stem:
        stem = ("biotac/" if stem.startswith("biotac") else "allegro/") + stem
    return f"meshes/{stem}"


def main() -> None:
    if not RECOVERED.exists():
        raise SystemExit(
            f"{RECOVERED} missing. Recover it first:\n"
            "  git show e4eb91a^:assets/urdf/kuka_allegro_iiwa14_description/"
            f"iiwa14_allegro.urdf > {RECOVERED}")

    vendor = ET.parse(VENDOR / "allegro.urdf").getroot()
    recovered = ET.parse(RECOVERED).getroot()
    print(f"vendor: {len(vendor.findall('link'))} links, {len(vendor.findall('joint'))} joints")

    # --- which links survive: the hand subtree from palm_link, tips dropped --------
    joints = {j.get("name"): j for j in vendor.findall("joint") if j.get("name") in HAND_JOINTS}
    missing = [n for n in HAND_JOINTS if n not in joints]
    if missing:
        raise SystemExit(f"vendor URDF lacks joints {missing}")
    keep = {ROOT_LINK} | {j.find("child").get("link") for j in joints.values()}
    print(f"keeping {len(keep)} links and {len(joints)} joints "
          f"(dropped the arm, allegro_mount and 4 *_biotac_tip)")

    # --- the good inertials, by link name, from the recovered integration ----------
    better = {l.get("name"): l.find("inertial") for l in recovered.findall("link")
              if l.find("inertial") is not None}

    out = ET.Element("robot", {"name": "allegro_right_unified"})
    # No "--" inside: XML forbids it in a comment, and ET does not check.
    out.append(ET.Comment(
        " GENERATED by unify_from_vendor.py. Do not hand-edit; regenerate. "
        "Repairs and provenance are documented in that file's docstring. "))
    for m in vendor.findall("material"):
        out.append(m)

    size, centre = _palm_slab(HERE / "meshes/allegro/base_link.obj")

    for link in vendor.findall("link"):
        name = link.get("name")
        if name not in keep:
            continue
        new = ET.SubElement(out, "link", {"name": name})

        # inertial: prefer the recovered file's, which carries real masses and COMs
        src = better.get(name, link.find("inertial"))
        if name == ROOT_LINK:
            # recomputed from the slab; the vendor's 1e-4 isotropic understates izz 8.2x
            ine = ET.SubElement(new, "inertial")
            ET.SubElement(ine, "origin", {"xyz": " ".join(f"{v:.6f}" for v in centre),
                                          "rpy": "0 0 0"})
            ET.SubElement(ine, "mass", {"value": str(PALM_MASS_KG)})
            k = PALM_MASS_KG / 12.0
            ET.SubElement(ine, "inertia", {
                "ixx": f"{k * (size[1] ** 2 + size[2] ** 2):.6e}",
                "iyy": f"{k * (size[0] ** 2 + size[2] ** 2):.6e}",
                "izz": f"{k * (size[0] ** 2 + size[1] ** 2):.6e}",
                "ixy": "0", "ixz": "0", "iyz": "0"})
        elif src is not None:
            new.append(src)

        for vis in link.findall("visual"):
            v = ET.SubElement(new, "visual")
            for child in vis:
                v.append(child)
            for mesh in v.iter("mesh"):
                mesh.set("filename", _mesh_name(mesh.get("filename")))

        if name == ROOT_LINK:
            # the box, in place of a mesh whose hull is 2.4x too thick
            col = ET.SubElement(new, "collision")
            ET.SubElement(col, "origin", {"xyz": " ".join(f"{v:.6f}" for v in centre),
                                          "rpy": "0 0 0"})
            geo = ET.SubElement(col, "geometry")
            ET.SubElement(geo, "box", {"size": " ".join(f"{v:.6f}" for v in size)})
        else:
            for colsrc in link.findall("collision"):
                c = ET.SubElement(new, "collision")
                for child in colsrc:
                    c.append(child)
                for mesh in c.iter("mesh"):
                    mesh.set("filename", _mesh_name(mesh.get("filename")))

    for name in HAND_JOINTS:                       # emit in canonical order
        out.append(joints[name])

    ET.indent(out, space="  ")
    dest = HERE / "allegro.urdf"
    dest.write_text('<?xml version="1.0"?>\n' + ET.tostring(out, encoding="unicode") + "\n")
    print(f"wrote {dest.relative_to(Path.cwd()) if dest.is_relative_to(Path.cwd()) else dest}")

    # --- checks that would have caught every defect above -------------------------
    r = ET.parse(dest).getroot()
    links = [l.get("name") for l in r.findall("link")]
    js = r.findall("joint")
    children = {j.find("child").get("link") for j in js}
    orphans = [l for l in links if l not in children and l != ROOT_LINK]
    assert not orphans, f"orphan links: {orphans}"
    assert len(links) == len(js) + 1, f"not a tree: {len(links)} links, {len(js)} joints"
    assert all(j.get("type") == "revolute" for j in js), "a hand joint is not revolute"
    assert [j.get("name") for j in js] == list(HAND_JOINTS), "joint order drifted"
    for mesh in r.iter("mesh"):
        f = mesh.get("filename")
        assert not f.startswith("package://"), f"package URI survives: {f}"
        assert (HERE / f).exists(), f"missing mesh {f}"
    palm = next(l for l in r.findall("link") if l.get("name") == ROOT_LINK)
    assert palm.find("collision/geometry/box") is not None, "palm collider is not a box"
    print(f"checks pass: {len(links)} links, {len(js)} revolute joints, tree, all meshes resolve")


if __name__ == "__main__":
    main()
