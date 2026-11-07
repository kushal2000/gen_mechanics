"""v3 hands at q = 0 in the canonical palm frame: one row per cell, per hand a face view (width across, length up) and
a side view (length across, normal up); links coloured by joint kind (flexion blue, abduction red), palm grey; each
hand titled with its URDF-checked cell.

    .venv_isaacsim/bin/python experiments/07nov_v3/plot_v3_gallery.py <hands dir or manifest.json> <out.png> [--per-cell 4]
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments/06nov_coevolve_v2")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--per-cell", type=int, default=4)
    ap.add_argument("--max-cells", type=int, default=12)
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch
    from plot_morph_gallery import mesh_tris, canon, KIND_RGB, PALM_RGB
    from coevolution.evolution.descriptors_v3 import features, cell
    src = REPO / a.src
    if src.suffix == ".json":
        specs = [(h["niche"], REPO / h["spec"]) for h in json.loads(src.read_text())["hands"]]
    else:
        specs = [(None, p) for p in sorted(src.glob("*/*.spec.json"))]
    by = defaultdict(list)
    for want, p in specs:
        c = cell(features(str(p.relative_to(REPO))))
        by[want or c].append((p, c))
    cells = sorted(by)[: a.max_cells]
    ncol = 2 * a.per_cell
    fig, axs = plt.subplots(len(cells), ncol, figsize=(1.9 * ncol, 2.1 * len(cells)), squeeze=False, facecolor="white")
    light = np.array([0.4, -0.5, 0.75]); light /= np.linalg.norm(light)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#e4e3df")
    for r, c in enumerate(cells):
        axs[r, 0].set_ylabel(c.replace("_", "\n"), fontsize=6.5, rotation=0, ha="right", va="center", labelpad=4)
        for k, (p, got) in enumerate(by[c][: a.per_cell]):
            spec = json.loads(p.read_text())
            kinds = {f"f{i}_j{j}": ("flex" if tuple(jt["axis"]) == (0, 0) else "abd")
                     for i, f in enumerate(spec["design"]["fingers"]) for j, jt in enumerate(f["joints"])}
            rs, ctr, R = canon(str(p.relative_to(REPO)))
            tris, cols = mesh_tris(REPO / rs.urdf_path, lambda jn: kinds.get(jn, "flex"))
            P = (tris - ctr) @ R.T * 1000
            for v, (ia, ib, idp) in enumerate([(1, 2, 0), (2, 0, 1)]):
                ax = axs[r, 2 * k + v]
                xy = P[..., [ia, ib]]; depth = P[..., idp].mean(1)
                nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
                o = np.argsort(depth)
                sh = (0.35 + 0.65 * np.abs(nrm @ light))[o, None]
                ax.add_collection(PolyCollection(xy[o], facecolors=np.clip(cols[o] * sh, 0, 1), edgecolors="none"))
                ax.plot([0], [0], "k+", ms=4)
                ax.set_xlim(-200, 200); ax.set_ylim(-200, 200); ax.set_aspect("equal")
                if v == 0 and got != c:
                    ax.set_title(f"URDF: {got}", fontsize=5, color="#b42318")
    fig.legend(handles=[Patch(color=KIND_RGB["flex"], label="flexion"), Patch(color=KIND_RGB["abd"], label="abduction"),
                        Patch(color=PALM_RGB, label="palm")], loc="upper right", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("v3 hands at q = 0 (mm, + = palm centre): per hand, face view then side view", x=0.01, ha="left",
                 fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0.06, 0, 1, 0.97))
    fig.savefig(REPO / a.out, dpi=110, facecolor="white")
    print("[v3 gallery]", a.out)


if __name__ == "__main__":
    main()
