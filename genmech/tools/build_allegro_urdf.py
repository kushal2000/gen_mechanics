"""Build the iiwa14 + Allegro URDF by splicing two existing assets.

The study compares *hands*, so the arm must be identical across robots. The only
Allegro asset available ships on an **iiwa7** (different link lengths and joint
limits), so using it directly would confound every result with an arm change.
This script instead grafts the Allegro hand onto the exact iiwa14 chain the
SHARPA robot uses, taken from the SHARPA URDF verbatim.

Generated rather than hand-edited so every number has provenance and the file
can be rebuilt when a mount transform is retuned.

    .venv_isaacsim/bin/python -m genmech.tools.build_allegro_urdf
    .venv_isaacsim/bin/python -m genmech.tools.build_allegro_urdf --mount_yaw 0.7854

**The mount transform.** In the stock asset the flange->palm chain is

    iiwa7_link_7 --(0,0,0.071)--> iiwa7_link_ee --(identity)--> allegro_mount
                 --(rpy 0,-1.5708,0.785398; xyz 0.008219,-0.02063,0.08086)--> palm_link

The iiwa14 flange sits closer to its wrist: ``iiwa14_joint_ee`` is (0,0,0.045).
Attaching ``allegro_mount`` to ``iiwa14_link_ee`` with z = 0.071 - 0.045 = 0.026
therefore reproduces the shipped flange-to-palm geometry exactly, rather than
inventing one. ``--mount_yaw`` rotates the hand about the flange axis, which is
what sets where the thumb points relative to the arm; sweep it in the
reachability viewer before freezing a value.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from genmech.utils.paths import resolve as resolve_repo_path


SHARPA_URDF = "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
ALLEGRO_SRC = "assets/urdf/kuka_allegro_iiwa14_description/_source/kuka_allegro.urdf"
OUT_URDF = "assets/urdf/kuka_allegro_iiwa14_description/iiwa14_allegro.urdf"

# Mirrored (left-hand) mesh copies live here; generated on demand.
MIRRORED_MESH_DIR = "allegro_meshes_mirrored"

# The iiwa14 chain to keep, verbatim, so the arm is byte-identical to SHARPA's.
ARM_LINKS = tuple(f"iiwa14_link_{i}" for i in range(8)) + ("iiwa14_link_ee",)
ARM_JOINTS = tuple(f"iiwa14_joint_{i}" for i in range(1, 8)) + ("iiwa14_joint_ee",)

FINGERS = ("index", "middle", "ring", "thumb")
HAND_LINKS = ("allegro_mount", "palm_link") + tuple(
    f"{f}_link_{i}" for f in FINGERS for i in range(4)
)
HAND_JOINTS = ("allegro_mount_joint",) + tuple(
    f"{f}_joint_{i}" for f in FINGERS for i in range(4)
)

# Flange offsets from each arm's own URDF.
IIWA7_FLANGE_TO_EE_Z = 0.071
IIWA14_FLANGE_TO_EE_Z = 0.045
MOUNT_Z = IIWA7_FLANGE_TO_EE_Z - IIWA14_FLANGE_TO_EE_Z  # 0.026

# Rotation of the hand about the flange axis. Unlike MOUNT_Z this is not derived
# from anything -- it sets where the thumb points relative to the arm, and the
# right value is a judgement about how the hand presents itself to the table.
# Chosen by eye in genmech/tools/reachability_viewer.py, on the LEFT (mirrored)
# hand. Note the two interact: mirroring the hand inverts the sense of this
# rotation, so a yaw picked on the right-hand asset does not carry over. Always
# re-check it visually after changing handedness.
# It is a default rather than a CLI-only flag so that rebuilding the URDF
# without arguments reproduces the shipped robot instead of silently reverting
# the hand to 0 and invalidating a trained policy.
MOUNT_YAW = math.radians(150.0)

# Allegro meshes were copied out of the isaacgym asset tree; rewrite its
# package-rooted prefix to a path relative to the generated URDF's directory,
# matching how the iiwa14 links reference `new_iiwa14_meshes/...`.
MESH_PREFIX_FROM = "kuka_allegro_description/meshes/"
MESH_PREFIX_TO = "allegro_meshes/"
# The arm meshes live in the SHARPA asset directory, so the generated URDF
# reaches them with a relative hop rather than duplicating 16 MB of STLs.
ARM_MESH_PREFIX_TO = "../kuka_sharpa_description/"


def _rpy_to_mat(r: float, p: float, y: float):
    import numpy as np

    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def _mat_to_rpy(R):
    import numpy as np

    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    p = math.asin(sy)
    if abs(abs(sy) - 1.0) < 1e-9:  # gimbal lock
        r, y = math.atan2(-R[1, 2], R[1, 1]), 0.0
    else:
        r, y = math.atan2(R[2, 1], R[2, 2]), math.atan2(R[1, 0], R[0, 0])
    return r, p, y


def _mirror_hand(root: ET.Element, hand_links, hand_joints, mesh_map: dict) -> None:
    """Reflect the hand subtree about the y=0 plane of its mount frame.

    The stock Allegro asset is a RIGHT hand. Its own comment says a left hand
    only needs the sign of each finger's y offset and splay angle flipped, but
    that is incomplete: the palm and thumb-base meshes are not mirror-symmetric
    (34.8 mm and 30.3 mm maximum residual when reflected about y), so that
    recipe yields left-hand kinematics wearing a right-hand palm. This applies
    the full reflection M = diag(1, -1, 1) to origins, rotations, axes, and
    geometry.

    Under a reflection a rotation R maps to M R M -- still a rotation, since
    det(M R M) = det(R) = 1. A revolute axis is a pseudovector, so it maps to
    -M a rather than M a; that extra sign keeps a positive joint angle meaning
    the same motion (flexion stays flexion), which matters because the limits
    are asymmetric and are carried over unchanged.
    """
    import numpy as np

    M = np.diag([1.0, -1.0, 1.0])

    def mirror_origin(el: ET.Element) -> None:
        xyz = [float(v) for v in el.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in el.get("rpy", "0 0 0").split()]
        el.set("xyz", " ".join(f"{v:.9g}" for v in (M @ np.array(xyz))))
        el.set("rpy", " ".join(f"{v:.9g}" for v in _mat_to_rpy(M @ _rpy_to_mat(*rpy) @ M)))

    for joint in root.findall("joint"):
        if joint.get("name") not in hand_joints:
            continue
        origin = joint.find("origin")
        if origin is not None:
            mirror_origin(origin)
        axis = joint.find("axis")
        if axis is not None:
            a = np.array([float(v) for v in axis.get("xyz", "0 0 1").split()])
            axis.set("xyz", " ".join(f"{v:.9g}" for v in (-(M @ a))))

    for link in root.findall("link"):
        if link.get("name") not in hand_links:
            continue
        for tag in ("visual", "collision", "inertial"):
            for el in link.findall(tag):
                o = el.find("origin")
                if o is not None:
                    mirror_origin(o)
        # Point every mesh at its mirrored copy.
        for mesh in link.iter("mesh"):
            fn = mesh.get("filename", "")
            if fn in mesh_map:
                mesh.set("filename", mesh_map[fn])


def _write_mirrored_meshes(urdf_dir, hand_mesh_names) -> dict:
    """Mirror each hand mesh about y and fix winding; return old -> new paths."""
    import numpy as np
    import trimesh

    out_dir = urdf_dir / MIRRORED_MESH_DIR
    mapping = {}
    for rel in sorted(hand_mesh_names):
        src = urdf_dir / rel
        if not src.exists():
            raise SystemExit(f"mesh not found while mirroring: {src}")
        m = trimesh.load(src, force="mesh")
        V = np.asarray(m.vertices).copy()
        V[:, 1] *= -1.0
        F = np.asarray(m.faces).copy()
        # Reflection flips orientation; reverse winding so normals point out.
        F = F[:, ::-1]
        dst_rel = f"{MIRRORED_MESH_DIR}/{pathlib_name(rel)}"
        dst = urdf_dir / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(vertices=V, faces=F, process=False).export(dst)
        mapping[rel] = dst_rel
    return mapping


def pathlib_name(rel: str) -> str:
    """Flatten a nested mesh path into a unique single filename."""
    return rel.replace("/", "__")


def _indent(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + "  " * level
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + "  "
        for child in elem:
            _indent(child, level + 1)
        if not (child.tail or "").strip():
            child.tail = pad
    if level and not (elem.tail or "").strip():
        elem.tail = pad


def _rewrite_meshes(elem: ET.Element, frm: str, to: str) -> int:
    n = 0
    for mesh in elem.iter("mesh"):
        fn = mesh.get("filename", "")
        if fn.startswith(frm):
            mesh.set("filename", to + fn[len(frm):])
            n += 1
        elif frm == "" and not fn.startswith(to) and not fn.startswith("/"):
            mesh.set("filename", to + fn)
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mount_yaw", type=float, default=MOUNT_YAW,
                        help="Rotation of the hand about the flange axis, radians. "
                             "Sets where the thumb points relative to the arm. "
                             f"Default {MOUNT_YAW:.4f} rad "
                             f"({math.degrees(MOUNT_YAW):.0f} deg), chosen visually.")
    parser.add_argument("--mount_z", type=float, default=MOUNT_Z)
    parser.add_argument("--out", default=OUT_URDF)
    parser.add_argument("--hand", choices=("right", "left"), default="left",
                        help="Handedness. The stock asset is right-handed; 'left' "
                             "mirrors the hand subtree AND its meshes so it matches "
                             "SHARPA, which is a left hand. Default left.")
    args = parser.parse_args()

    sharpa = ET.parse(resolve_repo_path(SHARPA_URDF)).getroot()
    allegro = ET.parse(resolve_repo_path(ALLEGRO_SRC)).getroot()

    out = ET.Element("robot", {"name": f"iiwa14_allegro_{args.hand}"})
    out.append(ET.Comment(
        " GENERATED by genmech/tools/build_allegro_urdf.py; do not hand-edit.\n"
        "     arm:  iiwa14 chain copied verbatim from the SHARPA URDF, so the arm is\n"
        "           identical across hands (docs/methodology.md 1).\n"
        "     hand: Allegro subtree from the stock kuka_allegro.urdf (iiwa7 asset).\n"
        f"     mount: iiwa14_link_ee -> allegro_mount at z={args.mount_z:.4f}\n"
        f"            (0.071 iiwa7 flange-to-ee minus 0.045 iiwa14 flange-to-ee), so the\n"
        f"            shipped flange-to-palm geometry is reproduced exactly.\n"
        f"     mount_yaw = {args.mount_yaw:.6f} rad "
        f"({math.degrees(args.mount_yaw):.1f} deg), chosen visually.\n"
        f"     handedness = {args.hand}\n"
    ))

    # Materials from both sources, first definition wins.
    seen_materials: set[str] = set()
    for src in (sharpa, allegro):
        for mat in src.findall("material"):
            name = mat.get("name")
            if name and name not in seen_materials:
                seen_materials.add(name)
                out.append(mat)

    # --- arm, verbatim ---
    arm_n = 0
    for link in sharpa.findall("link"):
        if link.get("name") in ARM_LINKS:
            _rewrite_meshes(link, "", ARM_MESH_PREFIX_TO)
            out.append(link)
            arm_n += 1
    for joint in sharpa.findall("joint"):
        if joint.get("name") in ARM_JOINTS:
            out.append(joint)

    # --- the graft ---
    mount = ET.SubElement(out, "joint", {"name": "iiwa14_allegro", "type": "fixed"})
    ET.SubElement(mount, "parent", {"link": "iiwa14_link_ee"})
    ET.SubElement(mount, "child", {"link": "allegro_mount"})
    ET.SubElement(mount, "origin", {
        "xyz": f"0 0 {args.mount_z}",
        "rpy": f"0 0 {args.mount_yaw}",
    })

    # --- hand ---
    hand_n = mesh_n = 0
    for link in allegro.findall("link"):
        if link.get("name") in HAND_LINKS:
            mesh_n += _rewrite_meshes(link, MESH_PREFIX_FROM, MESH_PREFIX_TO)
            out.append(link)
            hand_n += 1
    hand_j = 0
    for joint in allegro.findall("joint"):
        if joint.get("name") in HAND_JOINTS:
            out.append(joint)
            hand_j += 1

    if args.hand == "left":
        urdf_dir = resolve_repo_path(args.out).parent
        hand_meshes = {
            mesh.get("filename")
            for link in out.findall("link") if link.get("name") in HAND_LINKS
            for mesh in link.iter("mesh")
        }
        mesh_map = _write_mirrored_meshes(urdf_dir, hand_meshes)
        _mirror_hand(out, set(HAND_LINKS), set(HAND_JOINTS), mesh_map)
        print(f"[build]   mirrored hand subtree + {len(mesh_map)} meshes -> LEFT hand")

    missing_links = set(ARM_LINKS + HAND_LINKS) - {
        l.get("name") for l in out.findall("link")
    }
    missing_joints = set(ARM_JOINTS + HAND_JOINTS + ("iiwa14_allegro",)) - {
        j.get("name") for j in out.findall("joint")
    }
    if missing_links or missing_joints:
        raise SystemExit(
            f"splice incomplete: missing links {sorted(missing_links)}, "
            f"joints {sorted(missing_joints)}"
        )

    _indent(out)
    path = resolve_repo_path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(out).write(path, encoding="utf-8", xml_declaration=True)

    actuated = [j.get("name") for j in out.findall("joint") if j.get("type") == "revolute"]
    print(f"[build] wrote {path}")
    print(f"[build]   arm links {arm_n}, hand links {hand_n}, hand joints {hand_j}, "
          f"{mesh_n} allegro mesh paths rewritten")
    print(f"[build]   {len(actuated)} actuated joints: "
          f"{actuated[:7]} + {actuated[7:]}")
    print(f"[build]   mount: iiwa14_link_ee -> allegro_mount "
          f"xyz=(0,0,{args.mount_z}) rpy=(0,0,{args.mount_yaw})")


if __name__ == "__main__":
    main()
