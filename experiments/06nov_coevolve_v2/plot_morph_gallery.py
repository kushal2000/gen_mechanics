"""Every grammar-morph design and Wuji variant at q = 0, in the tokens' canonical palm frame (palm centre; normal,
width, length from the spec's palm keypoints): a face view (width across, length up, looking along the palm normal)
and a side view (length across, normal up). Links coloured by the kind of the joint that moves them (flexion blue,
abduction red, roll green, palm grey); grammar designs labelled with the grammar's clearance check (C1).

    .venv_isaacsim/bin/python experiments/06nov_coevolve_v2/plot_morph_gallery.py
Writes debug_outputs/coevolve/studies/grammar_morph/gallery_zero_pose.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments/05nov_noevolution/studies"), str(Path(__file__).resolve().parent)]
OUT = REPO / "debug_outputs/coevolve/studies/grammar_morph"
KIND_RGB = {"flex": (0.16, 0.42, 0.85), "abd": (0.86, 0.20, 0.18), "roll": (0.15, 0.65, 0.30)}
PALM_RGB = (0.62, 0.62, 0.64)
GRAMMAR = ["ref", "spread240", "spread180", "spread120", "spread60", "parallel", "line_p60_fan60", "line_p60_fan30",
           "line_p60_fan0", "line_p50_fan60", "line_p50_fan30", "line_p50_fan0", "line_p40_fan30", "line_p40_fan0",
           "line_p40_fan0_thumb300", "ref_wujichain", "ref_wujilens", "ref_knuckle28", "ref_splitflex28", "ref_knuckleFA0", "ref_knuckleAF0"]
WUJI = [("vendor x1", "assets/urdf/unified_dynamics_commercial_hands/wuji2/wuji2_left.spec.json"),
        ("vendor x2", "assets/urdf/unified_dynamics_commercial_hands/wuji2/wuji2_left_x2.spec.json")] + [
    (t, f"debug_outputs/coevolve/studies/wuji_morph/wuji2_left_{t}.spec.json")
    for t in ("x2e", "x2e_lim", "x2e_half", "x2e_star", "x2e_star_lim", "x1_star", "x2e_lockabd",
              "x2e_star_lockabd", "x2e_star_lim_lockabd", "x1_star_lockabd", "x2e_star_lim_lockabd_rc")]


def mesh_tris(urdf: Path, kind_of_joint):
    import trimesh
    import yourdfpy
    u = yourdfpy.URDF.load(str(urdf), load_meshes=False, build_scene_graph=True, build_collision_scene_graph=False)
    u.update_cfg(np.zeros(len(u.actuated_joint_names)))
    parent_joint = {j.child: j.name for j in u.robot.joints if j.type != "fixed"}
    jparent = {j.child: j.parent for j in u.robot.joints}
    tris, cols = [], []
    for link in u.robot.links:
        # the nearest moving joint above this link decides its colour
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
            if g.mesh is not None:
                fn = Path(g.mesh.filename)
                fn = fn if fn.is_absolute() else (urdf.parent / fn)
                m = trimesh.load(str(fn), force="mesh")
                if g.mesh.scale is not None:
                    m.apply_scale(np.asarray(g.mesh.scale, float))
            elif g.box is not None:
                m = trimesh.creation.box(extents=g.box.size)
            elif g.cylinder is not None:                 # the importer makes capsules of cylinders (the sim's shape)
                m = trimesh.creation.capsule(radius=g.cylinder.radius, height=g.cylinder.length, count=[16, 8])
            elif g.sphere is not None:
                m = trimesh.creation.icosphere(radius=g.sphere.radius, subdivisions=1)
            else:
                continue
            T = T_link @ (vis.origin if vis.origin is not None else np.eye(4))
            v = (np.c_[m.vertices, np.ones(len(m.vertices))] @ T.T)[:, :3]
            tris.append(v[m.faces])
            cols.append(np.repeat(np.array(rgb)[None], len(m.faces), 0))
    return np.concatenate(tris), np.concatenate(cols)


def canon(spec_path: str):
    from isaacsimenvs.common.robots import get_robot_spec
    rs = get_robot_spec(spec_path)
    kp = np.asarray(rs.palm_keypoints)
    e = kp[1:] - kp[:1]
    n, w, l = (e / np.linalg.norm(e, axis=1)[:, None])
    return rs, np.asarray(rs.palm_center_offset), np.stack([n, w, l])


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch
    import grammar_morph as G
    from hand_sampler.grammar import hand as gh, viability as gvb

    lad = G.ladder()
    items = []
    for hid in GRAMMAR:
        spec = next((OUT / "hands" / hid).glob("*.spec.json"), None)
        if spec is None:
            print(f"[gallery] {hid}: not built, skipped")
            continue
        d = lad[hid]
        c1 = gvb.c1_self_overlap(gh.hand_from_dict(d))
        kinds = {}
        for i, f in enumerate(d["fingers"]):
            for j, jt in enumerate(f["joints"]):
                kinds[f"f{i}_j{j}"] = {(0, 0): "flex", (90, 0): "abd", (0, 90): "roll"}.get(tuple(jt["axis"]), "flex")
        items.append((hid, str(spec.relative_to(REPO)), lambda jn, k=kinds: k.get(jn, "flex"),
                      "C1 clear" if c1.ok else f"C1 FAIL ({len(c1.pairs)} pairs)", c1.ok))
    for name, spec in WUJI:
        if (REPO / spec).exists():
            from vendor_clearance import clearance
            bad = clearance(spec)
            items.append((f"Wuji {name}", spec, lambda jn: "abd" if jn.endswith("_abd") else "flex",
                          "clear" if not bad else f"COLLIDES ({len(bad)} pairs)", not bad))

    light = np.array([0.4, -0.5, 0.75])
    light /= np.linalg.norm(light)
    ncol = 6
    nrow = int(np.ceil(len(items) / ncol))
    fig, axs = plt.subplots(2 * nrow, ncol, figsize=(2.6 * ncol, 2 * 2.5 * nrow), facecolor="white", squeeze=False)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for k, (title, spec, kind_of, note, ok) in enumerate(items):
        rs, c, R = canon(spec)
        tris, cols = mesh_tris(REPO / rs.urdf_path, kind_of)
        P = (tris - c) @ R.T * 1000                     # mm, columns (normal, width, length)
        r, q = divmod(k, ncol)
        for v, (ia, ib, idepth, xl, yl) in enumerate([(1, 2, 0, "width", "length"), (2, 0, 1, "length", "normal")]):
            ax = axs[2 * r + v, q]
            xy = P[..., [ia, ib]]
            depth = P[..., idepth].mean(1)
            nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
            order = np.argsort(depth if v == 0 else -depth)
            shade = (0.35 + 0.65 * np.abs(nrm @ light))[order, None]
            ax.add_collection(PolyCollection(xy[order], facecolors=np.clip(cols[order] * shade, 0, 1), edgecolors="none"))
            ax.plot([0], [0], "k+", ms=6)
            ax.set_xlim(-260, 260); ax.set_ylim(-260, 260); ax.set_aspect("equal")
            ax.axhline(0, color="#e4e3df", lw=0.5, zorder=-1); ax.axvline(0, color="#e4e3df", lw=0.5, zorder=-1)
            if v == 0:
                ax.set_title(title, fontsize=9, color="#1f1f1e", fontweight="semibold")
                if note:
                    ax.text(0.5, 0.01, note, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5,
                            color="#1f1f1e" if ok else "#b42318", fontweight="semibold")
            ax.text(0.99, 0.99, f"{xl} → / {yl} ↑", transform=ax.transAxes, ha="right", va="top", fontsize=6,
                    color="#8a8985")
    fig.legend(handles=[Patch(color=KIND_RGB[k], label=lab) for k, lab in
                        (("flex", "flexion"), ("abd", "abduction"), ("roll", "roll"))] + [Patch(color=PALM_RGB,
                                                                                               label="palm / base")],
               loc="upper right", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Grammar morph ladder and Wuji variants at q = 0 (canonical palm frame, mm; + = palm centre). "
                 "Rows: face view, side view", x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.subplots_adjust(left=0.01, right=0.995, top=0.95, bottom=0.01, hspace=0.12, wspace=0.03)
    out = OUT / "gallery_zero_pose.png"
    fig.savefig(out, dpi=110, facecolor="white")
    print(f"[gallery] wrote {out}")


if __name__ == "__main__":
    main()
