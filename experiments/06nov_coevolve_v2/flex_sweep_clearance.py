"""Self-collision while flexing: each finger's joints swept (one finger at a time, the others at 0), at every angle
the sim's collision pairs (convex hulls, minus adjacent_links and jointed / fixed-merged neighbours, as
vendor_clearance.py) tested; prints the first angle each finger's sweep hits something, and what.

    python experiments/06nov_coevolve_v2/flex_sweep_clearance.py <spec.json> [joint ...]
"""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
import vendor_clearance as vc


def main():
    spec_path = sys.argv[1]
    spec = json.loads((REPO / spec_path).read_text())
    joints = sys.argv[2:] or [j for j in spec["hand_joint_names"] if "mcp_flex" in j or "cmc_flex" in j or j.endswith("_j0")]
    import trimesh, yourdfpy
    from hand_sampler.grammar.viability import gjk_distance
    urdf = REPO / spec["urdf"]
    u = yourdfpy.URDF.load(str(urdf), load_meshes=False, build_scene_graph=True, build_collision_scene_graph=False)
    # local convex hulls per link (link frame), once
    local = {}
    for link in u.robot.links:
        pts = []
        for col in link.collisions:
            g = col.geometry
            if g.mesh is not None:
                fn = Path(g.mesh.filename); fn = fn if fn.is_absolute() else urdf.parent / fn
                m = trimesh.load(str(fn), force="mesh")
                if g.mesh.scale is not None: m.apply_scale(np.asarray(g.mesh.scale, float))
            elif g.box is not None: m = trimesh.creation.box(extents=g.box.size)
            elif g.cylinder is not None: m = trimesh.creation.cylinder(radius=g.cylinder.radius, height=g.cylinder.length, sections=16)
            elif g.sphere is not None: m = trimesh.creation.icosphere(radius=g.sphere.radius, subdivisions=1)
            else: continue
            T = col.origin if col.origin is not None else np.eye(4)
            pts.append((np.c_[m.vertices, np.ones(len(m.vertices))] @ T.T)[:, :3])
        if pts:
            P = np.concatenate(pts)
            try: P = trimesh.convex.convex_hull(P).vertices
            except Exception: pass
            local[link.name] = P
    import xml.etree.ElementTree as ET
    js = [(j.find("parent").get("link"), j.find("child").get("link"), j.get("type")) for j in ET.parse(urdf).getroot().findall("joint")]
    def root(l):
        for p, c, t in js:
            if c == l and t == "fixed": return root(p)
        return l
    body = {l: root(l) for l in local}
    jointed = {(root(p), root(c)) for p, c, t in js if t != "fixed"}
    adj = {k: set(v) for k, v in spec.get("adjacent_links", {}).items()}
    names = sorted(local)
    def hits(cfg):
        u.update_cfg(cfg)
        W = {n: (np.c_[local[n], np.ones(len(local[n]))] @ u.get_transform(n).T)[:, :3] for n in names}
        out = []
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                A, B = names[a], names[b]
                if B in adj.get(A, ()) or A in adj.get(B, ()) or body[A] == body[B] or (body[A], body[B]) in jointed or (body[B], body[A]) in jointed:
                    continue
                if np.linalg.norm(W[A].mean(0) - W[B].mean(0)) > np.linalg.norm(W[A] - W[A].mean(0), axis=1).max() + np.linalg.norm(W[B] - W[B].mean(0), axis=1).max():
                    continue
                if gjk_distance(W[A], W[B]) <= 0.0:
                    out.append((A, B))
        return out
    base = {n: 0.0 for n in u.actuated_joint_names}
    print(f"{Path(spec_path).name}: at q=0 {len(hits(base))} hits")
    for j in joints:
        lo, hi = spec["joint_limits"][j]
        first = None
        for a in np.linspace(0, hi, 10)[1:]:
            h = hits({**base, j: a})
            if h:
                first = (np.degrees(a), h); break
        print(f"  {j:26s} 0..{np.degrees(hi):3.0f} deg: " + ("clear" if first is None else f"HITS at {first[0]:.0f} deg: {first[1][:3]}"))


if __name__ == "__main__":
    main()
