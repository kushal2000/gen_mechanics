"""Render recorded rollout poses (wuji_reach_check.py with REACH_POSES=<json>): each hand posed at its simulated joint
angles and palm pose (env frame), with the pen (100 mm segment along its x) and the table top, side view (x, z) and
top view (x, y), one column per recorded step.

    python experiments/06nov_coevolve_v2/render_rollout.py <poses.json> <out.png>
"""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]


def quat_mat(q):                                      # wxyz
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def posed_tris(urdf, q, kind_of=None):
    import trimesh, yourdfpy
    from plot_morph_gallery import KIND_RGB, PALM_RGB
    u = yourdfpy.URDF.load(str(urdf), load_meshes=False, build_scene_graph=True, build_collision_scene_graph=False)
    u.update_cfg({n: q.get(n, 0.0) for n in u.actuated_joint_names})
    tris, cols = [], []
    for link in u.robot.links:
        T_link = u.get_transform(link.name)
        for vis in link.collisions:                  # what PhysX collides with
            g = vis.geometry
            if g.mesh is not None:
                fn = Path(g.mesh.filename); fn = fn if fn.is_absolute() else urdf.parent / fn
                m = trimesh.load(str(fn), force="mesh")
                if g.mesh.scale is not None:
                    m.apply_scale(np.asarray(g.mesh.scale, float))
                m = m.convex_hull                     # the scene's collider_type
            elif g.box is not None:
                m = trimesh.creation.box(extents=g.box.size)
            elif g.cylinder is not None:
                m = trimesh.creation.capsule(radius=g.cylinder.radius, height=g.cylinder.length, count=[12, 6])
            elif g.sphere is not None:
                m = trimesh.creation.icosphere(radius=g.sphere.radius, subdivisions=1)
            else:
                continue
            T = T_link @ (vis.origin if vis.origin is not None else np.eye(4))
            v = (np.c_[m.vertices, np.ones(len(m.vertices))] @ T.T)[:, :3]
            pj = next((j.name for j in u.robot.joints if j.child == link.name and j.type != "fixed"), None)
            rgb = PALM_RGB if pj is None else KIND_RGB.get(kind_of(pj) if kind_of else "flex", KIND_RGB["flex"])
            tris.append(v[m.faces]); cols.append(np.repeat(np.array(rgb)[None], len(m.faces), 0))
    return np.concatenate(tris), np.concatenate(cols)


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from isaacsimenvs.common.robots import get_robot_spec
    d = json.loads(Path(sys.argv[1]).read_text())
    hands = d["hands"]
    steps = sorted({int(t) for h in d["poses"].values() for t in h}, key=int)
    fig, axs = plt.subplots(2 * len(hands), len(steps), figsize=(2.3 * len(steps), 2.4 * 2 * len(hands)), squeeze=False)
    light = np.array([0.3, -0.6, 0.75]); light /= np.linalg.norm(light)
    for hi, name in enumerate(hands):
        urdf = REPO / get_robot_spec(name).urdf_path
        for si, t in enumerate(steps):
            P = d["poses"][str(hi)][str(t)]
            tris, cols = posed_tris(urdf, P["q"])
            R, p = quat_mat(P["palm"][3:7]), np.asarray(P["palm"][:3])
            W = (tris @ R.T + p) * 1000
            top = P["table_top"] * 1000
            pen_c = np.asarray(P["pen"]) * 1000; pen_ax = quat_mat(P["pen_quat"])[:, 0] * 50
            for v, (a, b, dep, lab) in enumerate([(0, 2, 1, "side x-z"), (0, 1, 2, "top x-y")]):
                ax = axs[2 * hi + v, si]; ax.set_xticks([]); ax.set_yticks([])
                xy = W[..., [a, b]]; depth = W[..., dep].mean(1)
                nrm = np.cross(W[:, 1] - W[:, 0], W[:, 2] - W[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
                order = np.argsort(depth if v == 0 else depth)
                shade = (0.35 + 0.65 * np.abs(nrm @ light))[order, None]
                ax.add_collection(PolyCollection(xy[order], facecolors=np.clip(cols[order] * shade, 0, 1), edgecolors="none", alpha=0.85))
                e0, e1 = pen_c - pen_ax, pen_c + pen_ax
                ax.plot([e0[a], e1[a]], [e0[b], e1[b]], color="#e8a317", lw=4, solid_capstyle="butt", zorder=5)
                if v == 0:
                    ax.axhline(top, color="#8a8985", lw=1)
                    ax.set_xlim(pen_c[0] - 220, pen_c[0] + 220); ax.set_ylim(top - 30, top + 410)
                else:
                    ax.set_xlim(pen_c[0] - 220, pen_c[0] + 220); ax.set_ylim(pen_c[1] - 220, pen_c[1] + 220)
                ax.set_aspect("equal")
                if si == 0:
                    ax.set_ylabel(f"{Path(name).name[:22]}\n{lab}", fontsize=7)
                if hi == 0 and v == 0:
                    ax.set_title(f"step {t}", fontsize=9)
    fig.tight_layout()
    fig.savefig(sys.argv[2], dpi=90)
    print("[render]", sys.argv[2])


if __name__ == "__main__":
    main()
