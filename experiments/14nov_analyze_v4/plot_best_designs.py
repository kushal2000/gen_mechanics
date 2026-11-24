"""The best design each v4 method found (14 Nov 2026): over all generations, the training design with the highest
goals / episode label (gen_XXXX/labels.json; designs with fewer than --min-episodes logged episodes skipped), drawn
open (q = 0) and curled (every flexion joint at --curl-deg, abductions at 0), face-on, side-on and from an oblique
viewpoint, links coloured by joint kind (flexion blue, abduction red, palm grey), in the canonical palm frame (mm).

    .venv_isaacsim/bin/python experiments/14nov_analyze_v4/plot_best_designs.py [--curl-deg 45]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments/06nov_coevolve_v2")]
RUNS = [("k5_none", "MAP-Elites"), ("k5_global", "Greedy"), ("hod", "House of Dextra")]


def best_design(run: Path, min_episodes: int) -> dict:
    best = None
    for f in sorted(run.glob("gen_*/labels.json")):
        g = int(f.parent.name.split("_")[1])
        for i, (v, n, *_) in json.loads(f.read_text()).items():
            if n >= min_episodes and (best is None or v > best["label"]):
                best = {"id": i, "label": v, "episodes": n, "gen": g}
    reg = json.loads((run / "designs.json").read_text())
    best.update(design=reg[best["id"]]["design"], spec=reg[best["id"]]["spec"], niche=reg[best["id"]]["niche"],
                born=reg[best["id"]].get("born", 0))
    return best


def posed_tris(urdf: Path, q_of_joint, kind_of_joint):
    """Triangles and colours of the URDF's visuals at joint angles q_of_joint(name) (rad)."""
    import trimesh
    import yourdfpy
    from plot_morph_gallery import KIND_RGB, PALM_RGB
    u = yourdfpy.URDF.load(str(urdf), load_meshes=False, build_scene_graph=True, build_collision_scene_graph=False)
    u.update_cfg(np.array([q_of_joint(j) for j in u.actuated_joint_names]))
    parent_joint = {j.child: j.name for j in u.robot.joints if j.type != "fixed"}
    jparent = {j.child: j.parent for j in u.robot.joints}
    tris, cols = [], []
    for link in u.robot.links:
        l, kind = link.name, None
        while l in jparent:
            if l in parent_joint:
                kind = kind_of_joint(parent_joint[l])
                break
            l = jparent[l]
        rgb = KIND_RGB.get(kind, PALM_RGB)
        T_link = u.get_transform(link.name)
        for vis in link.visuals:
            g = vis.geometry
            if g.box is not None:
                m = trimesh.creation.box(extents=g.box.size)
            elif g.cylinder is not None:
                m = trimesh.creation.capsule(radius=g.cylinder.radius, height=g.cylinder.length, count=[16, 8])
            elif g.sphere is not None:
                m = trimesh.creation.icosphere(radius=g.sphere.radius, subdivisions=1)
            elif g.mesh is not None:
                fn = Path(g.mesh.filename)
                m = trimesh.load(str(fn if fn.is_absolute() else urdf.parent / fn), force="mesh")
                if g.mesh.scale is not None:
                    m.apply_scale(np.asarray(g.mesh.scale, float))
            else:
                continue
            T = T_link @ (vis.origin if vis.origin is not None else np.eye(4))
            v = (np.c_[m.vertices, np.ones(len(m.vertices))] @ T.T)[:, :3]
            tris.append(v[m.faces])
            cols.append(np.repeat(np.array(rgb)[None], len(m.faces), 0))
    return np.concatenate(tris), np.concatenate(cols)


def draw(ax, P, cols, view):
    """P: (T, 3, 3) mm in the canonical palm frame [normal, width, length]. view: face / side / oblique."""
    from matplotlib.collections import PolyCollection
    if view == "face":
        xy, depth = P[..., [1, 2]], P[..., 0].mean(1)
    elif view == "side":
        xy, depth = P[..., [2, 0]], P[..., 1].mean(1)
    else:                                                # oblique: turned 40 deg about the normal, tilted back 55 deg
        X = P[..., [1, 2, 0]]                            # (width, length, normal) as (x, y, z)
        az, el = math.radians(40), math.radians(55)
        Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, math.cos(el), -math.sin(el)], [0, math.sin(el), math.cos(el)]])
        Q = X @ (Rx @ Rz).T
        xy, depth = Q[..., [0, 1]], Q[..., 2].mean(1)
    light = np.array([0.4, -0.5, 0.75]); light /= np.linalg.norm(light)
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    o = np.argsort(depth)
    sh = (0.35 + 0.65 * np.abs(nrm @ light))[o, None]
    ax.add_collection(PolyCollection(xy[o], facecolors=np.clip(cols[o] * sh, 0, 1), edgecolors="none"))
    ax.plot([0], [0], "k+", ms=4, zorder=5)
    ax.set_xlim(-240, 240); ax.set_ylim(-240, 240); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#e4e3df")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="debug_outputs/coevolve/v4")
    ap.add_argument("--out", default="experiments/14nov_analyze_v4/plots/best_designs.png")
    ap.add_argument("--min-episodes", type=int, default=20)
    ap.add_argument("--curl-deg", type=float, default=45.0)
    ap.add_argument("--all-views", action="store_true", help="also oblique, side and curled views (default: home pose, top down)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from plot_morph_gallery import KIND_RGB, PALM_RGB, canon
    from hand_sampler.grammar import v3
    v3.configure("v4")

    views = [("open", "face")] if not a.all_views else [("open", "face"), ("open", "oblique"), ("curled", "face"),
                                                        ("curled", "oblique"), ("curled", "side")]
    single = len(views) == 1                      # home pose, top down: the methods side by side
    if single:
        fig, axs = plt.subplots(1, len(RUNS), figsize=(4.0 * len(RUNS), 6.0), squeeze=False, facecolor="white")
    else:
        fig, axs = plt.subplots(len(RUNS), len(views), figsize=(2.3 * len(views) + 2.2, 2.55 * len(RUNS)),
                                squeeze=False, facecolor="white")
    for r, (k, name) in enumerate(RUNS):
        b = best_design(REPO / a.campaign / k, a.min_episodes)
        hand = v3.hand_from_dict(b["design"])
        kinds = {f"f{i}_j{j}": v3.kind_of(f, jt) for i, f in enumerate(hand.fingers) for j, jt in enumerate(f.joints)}
        pats = ["".join(kd[0].upper() for kd in (v3.kind_of(f, jt) for jt in f.joints)) for f in hand.fingers]
        rs, ctr, R = canon(b["spec"])
        urdf = REPO / rs.urdf_path if not Path(rs.urdf_path).is_absolute() else Path(rs.urdf_path)
        f_, m_, gap, kn, reach = b["niche"].split("_")
        info = (f"{f_[1:]} fingers, {m_[1:]} motors, {gap}, {kn}, {reach}\n"
                f"fingers {' '.join(pats)} (base to tip)\nrolls {' '.join(str(v3.roll_of(f)) for f in hand.fingers)}")
        for c, (pose, view) in enumerate(views):
            q = (lambda jn: 0.0) if pose == "open" else (
                lambda jn: math.radians(a.curl_deg) if kinds.get(jn) == "flex" else 0.0)
            tris, cols = posed_tris(urdf, q, lambda jn: kinds.get(jn, "flex"))
            ax = axs[0, r] if single else axs[r, c]
            draw(ax, (tris - ctr) @ R.T * 1000, cols, view)
            if not single and r == 0:
                ax.set_title(f"{pose}, {view}" + (f" ({a.curl_deg:.0f}\N{DEGREE SIGN} flex)" if pose == "curled"
                                                  and c == 2 else ""), fontsize=8.5, color="#6b6a64")
        if single:
            ax = axs[0, r]
            ax.set_title(f"{name}\n{b['label']:.1f} goals / ep", fontsize=11, fontweight="semibold", color="#1f1f1e")
            ax.set_xlabel(f"{b['episodes']} episodes, gen {b['gen']}\n" + info, fontsize=8.5, color="#1f1f1e")
        else:
            axs[r, 0].set_ylabel(f"{name}\n{b['label']:.1f} goals / ep\n({b['episodes']} ep, gen {b['gen']})\n\n"
                                 + info.replace(", " + gap, "\n" + gap), fontsize=8, rotation=0, ha="right",
                                 va="center", labelpad=8, color="#1f1f1e")
        print(f"{name:16s} {b['id']}: {b['label']:.2f} goals/ep over {b['episodes']} ep (gen {b['gen']}, born gen "
              f"{b['born']}); {b['niche']}; fingers {pats}; rolls {[v3.roll_of(f) for f in hand.fingers]}")
    fig.legend(handles=[Patch(color=KIND_RGB["flex"], label="flexion"), Patch(color=KIND_RGB["abd"], label="abduction"),
                        Patch(color=PALM_RGB, label="palm")], loc="upper right", ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle("v4: the best design each method found" + (" (home pose, top down; mm, + = palm centre)" if single
                 else " (highest training-log goals / episode, any generation)"),
                 x=0.01, ha="left", fontsize=11, fontweight="semibold", color="#1f1f1e")
    fig.tight_layout(rect=(0, 0, 1, 0.94) if single else (0.12, 0, 1, 0.95))
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor="white")
    print("[best designs]", out)


if __name__ == "__main__":
    main()
