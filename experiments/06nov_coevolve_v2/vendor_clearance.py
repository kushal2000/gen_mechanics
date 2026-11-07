"""Self-collision check for a vendor-hand URDF copy at q = 0, as the sim sees it: one convex hull per link (the
scene's collider_type "convex_hull") from the link's collision meshes, every pair the spec does not filter
(adjacent_links; the scene's self-collision filters) tested with the grammar's GJK distance. Returns the pairs that
intersect (distance 0) or come within --tol mm.

    python experiments/06nov_coevolve_v2/vendor_clearance.py <spec.json> [...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def link_hulls(urdf: Path) -> dict[str, np.ndarray]:
    import trimesh
    import yourdfpy
    u = yourdfpy.URDF.load(str(urdf), load_meshes=False, build_scene_graph=True, build_collision_scene_graph=False)
    u.update_cfg(np.zeros(len(u.actuated_joint_names)))
    out = {}
    for link in u.robot.links:
        pts = []
        for col in link.collisions:
            g = col.geometry
            if g.mesh is not None:
                fn = Path(g.mesh.filename)
                fn = fn if fn.is_absolute() else urdf.parent / fn
                m = trimesh.load(str(fn), force="mesh")
                if g.mesh.scale is not None:
                    m.apply_scale(np.asarray(g.mesh.scale, float))
            elif g.box is not None:
                m = trimesh.creation.box(extents=g.box.size)
            elif g.cylinder is not None:
                m = trimesh.creation.cylinder(radius=g.cylinder.radius, height=g.cylinder.length, sections=16)
            elif g.sphere is not None:
                m = trimesh.creation.icosphere(radius=g.sphere.radius, subdivisions=1)
            else:
                continue
            T = u.get_transform(link.name) @ (col.origin if col.origin is not None else np.eye(4))
            pts.append((np.c_[m.vertices, np.ones(len(m.vertices))] @ T.T)[:, :3])
        if pts:
            P = np.concatenate(pts)
            try:
                P = trimesh.convex.convex_hull(P).vertices
            except Exception:                           # noqa: BLE001  (degenerate: keep the points)
                pass
            out[link.name] = P
    return out


def clearance(spec_path: str, tol_mm: float = 0.0) -> list[tuple[str, str, float]]:
    from hand_sampler.grammar.viability import gjk_distance
    spec = json.loads((REPO / spec_path).read_text())
    hulls = link_hulls(REPO / spec["urdf"])
    adj = {k: set(v) for k, v in spec.get("adjacent_links", {}).items()}
    # fixed joints merge into one body (the converter), and jointed neighbours never collide (PhysX)
    import xml.etree.ElementTree as ET
    joints = [(j.find("parent").get("link"), j.find("child").get("link"), j.get("type"))
              for j in ET.parse(REPO / spec["urdf"]).getroot().findall("joint")]
    body = {}
    def root(l):
        for p, c, t in joints:
            if c == l and t == "fixed":
                return root(p)
        return l
    for p, c, _ in joints:
        body[p], body[c] = root(p), root(c)
    jointed = {(body[p], body[c]) for p, c, t in joints if t != "fixed"}
    names = sorted(hulls)
    bad = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            A, B = names[a], names[b]
            if B in adj.get(A, ()) or A in adj.get(B, ()):
                continue
            bA, bB = body.get(A, A), body.get(B, B)
            if bA == bB or (bA, bB) in jointed or (bB, bA) in jointed:
                continue
            PA, PB = hulls[A], hulls[B]
            ca, cb = PA.mean(0), PB.mean(0)
            ra = np.linalg.norm(PA - ca, axis=1).max()
            rb = np.linalg.norm(PB - cb, axis=1).max()
            if np.linalg.norm(ca - cb) > ra + rb + tol_mm / 1000:
                continue
            d = gjk_distance(PA, PB) * 1000
            if d <= tol_mm:
                bad.append((A, B, d))
    return bad


if __name__ == "__main__":
    for s in sys.argv[1:]:
        bad = clearance(s)
        print(f"{Path(s).name}: {len(bad)} intersecting pairs" + "".join(f"\n    {a} x {b}" for a, b, _ in bad[:12]))
