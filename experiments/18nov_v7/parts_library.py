"""Parts library for the v7 grammar (18 Nov 2026): every finger link of the commercial hands (uniform-dynamics vendor
URDFs), its collision geometry in a canonical link frame -- origin at the link's joint, x along the link (to the next
joint, or to the fingertip pad for a finger's last link), z along the finger's flexion axis (its last joint's axis,
the roll-invariant convention), y = z x x. Every collision primitive (box / cylinder / sphere / mesh) of a link is kept as
its own mesh (vertices, faces; mm), so a grammar link can carry the real shapes -- scaled along x to its own length.

Why: a policy trained directly on the commercial hands (~1 goal / episode) handles them with a finger removed
(0.7-1.06) but scores ~0 on box-and-capsule replicas of them even with exact kinematics: the grammar's link geometry,
not its kinematics, is the domain gap (experiments/16nov_v6, 17-18 Nov).

    .venv_isaacsim/bin/python experiments/18nov_v7/parts_library.py [--out debug_outputs/coevolve/v7/parts.json]

Output JSON: {"<hand>/f<i>/<j>": {"hand", "finger", "link", "kind" (flex / abd / roll of its joint), "length_mm",
"n_links" (in its finger), "meshes": [{"v": [[x, y, z], ...] (mm), "f": [[a, b, c], ...]}]}}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
HANDS = ("wuji2", "tesollo", "xhand", "leap", "dex3", "allegro", "shadow")


def _rigid_bodies(u, link_name: str):
    """`link_name` and every body hanging off it through FIXED joints only (a fingertip pad body, ...), each with its
    transform into `link_name`'s frame."""
    out, stack = [], [(link_name, np.eye(4))]
    while stack:
        name, T = stack.pop()
        out.append((name, T))
        for j in u.robot.joints:
            if j.parent == name and j.type == "fixed":
                stack.append((j.child, T @ (j.origin if j.origin is not None else np.eye(4))))
    return out


def link_meshes(urdf_path: Path, link_name: str):
    """Each collision element of `link_name` and of the bodies fixed to it, as a trimesh in the link's frame (m)."""
    import trimesh
    import yourdfpy
    u = yourdfpy.URDF.load(str(urdf_path), load_meshes=False, build_scene_graph=False, build_collision_scene_graph=False)
    out = []
    for name, T_body in _rigid_bodies(u, link_name):
        out += _body_meshes(urdf_path, u.link_map[name], T_body)
    return out


def _body_meshes(urdf_path: Path, link, T_body):
    import trimesh
    out = []
    for c in link.collisions:
        g = c.geometry
        if g.box is not None:
            m = trimesh.creation.box(extents=g.box.size)
        elif g.cylinder is not None:
            m = trimesh.creation.cylinder(radius=g.cylinder.radius, height=g.cylinder.length, sections=16)
        elif g.sphere is not None:
            m = trimesh.creation.icosphere(radius=g.sphere.radius, subdivisions=1)
        elif g.mesh is not None:
            fn = Path(g.mesh.filename)
            fn = fn if fn.is_absolute() else urdf_path.parent / fn
            m = trimesh.load(str(fn), force="mesh")
            if g.mesh.scale is not None:
                m.apply_scale(np.asarray(g.mesh.scale, float))
            m = m.convex_hull                               # the importer colliders are hulls (collider_type convex_hull)
        else:
            continue
        T = T_body @ (c.origin if c.origin is not None else np.eye(4))
        m.apply_transform(T)
        out.append(m)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="debug_outputs/coevolve/v7/parts.json")
    a = ap.parse_args()
    from hand_sampler.design_space import joint_axes_and_link_vectors
    from isaacsimenvs.common.robots import get_robot_spec
    from isaacsimenvs.common.robots.finger_layout import finger_layout
    from coevolution.evolution.descriptors_v3 import _frames, joint_kinds
    lib = {}
    for h in HANDS:
        s = get_robot_spec(f"{h}_left_uniform_handonly")
        urdf = Path(s.urdf_path) if Path(s.urdf_path).is_absolute() else REPO / s.urdf_path
        pose, child_of = _frames(s)
        ax_local, lv = joint_axes_and_link_vectors(s.urdf_path, s.hand_joint_names, s.fingertip_body_names,
                                                   s.fingertip_offsets)
        for fi, (joints, tip) in enumerate(zip(finger_layout(s).fingers, s.fingertip_body_names)):
            base_link = child_of[s.hand_joint_names[joints[0]]]
            Mt = pose(tip)
            pad = Mt[:3, :3] @ np.asarray(s.fingertip_offsets[fi]) + Mt[:3, 3]
            kinds = joint_kinds(s, pose, child_of, ax_local, joints, pad - pose(base_link)[:3, 3])
            last_link = child_of[s.hand_joint_names[joints[-1]]]
            flex = pose(last_link)[:3, :3] @ ax_local[joints[-1]]          # the finger's flexion axis (palm frame)
            for j, jn in enumerate(joints):
                body = child_of[s.hand_joint_names[jn]]
                P = pose(body)
                o = P[:3, 3]
                nxt = pose(child_of[s.hand_joint_names[joints[j + 1]]])[:3, 3] if j + 1 < len(joints) else pad
                x = nxt - o
                L = float(np.linalg.norm(x))
                x = x / max(L, 1e-9)
                z = flex - (flex @ x) * x
                if np.linalg.norm(z) < 1e-6:                 # a link along the flexion axis (roll-like): any normal
                    z = np.cross(x, [0, 0, 1.0]) if abs(x[2]) < 0.9 else np.cross(x, [0, 1.0, 0])
                z = z / np.linalg.norm(z)
                y = np.cross(z, x)
                R = np.stack([x, y, z])                       # palm frame -> canonical link frame
                meshes = []
                for m in link_meshes(urdf, body):
                    v_palm = (P[:3, :3] @ m.vertices.T).T + o   # link frame -> palm frame
                    v_can = ((v_palm - o) @ R.T) * 1000.0       # -> canonical, mm
                    meshes.append({"v": np.round(v_can, 3).tolist(), "f": m.faces.tolist()})
                lib[f"{h}/f{fi}/{j}"] = {"hand": h, "finger": fi, "link": j, "kind": {"F": "flex", "A": "abd", "R": "roll"}.get(kinds[j], kinds[j]),
                                         "length_mm": round(L * 1000, 2), "n_links": len(joints), "meshes": meshes}
        print(f"{h:8s} {sum(1 for k in lib if k.startswith(h + '/'))} links", flush=True)
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lib))
    print(f"[parts] {len(lib)} links -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
