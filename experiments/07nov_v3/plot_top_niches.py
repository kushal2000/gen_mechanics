"""The best niches of one co-evolution generation (7 Nov 2026): cells ranked by their training hands' mean goals / episode
(training-log labels, labels.json), one row per cell in rank order, its best hands by label (face view, then side view;
links coloured by joint kind as in plot_v3_gallery.py), each titled with its goals / episode.

    .venv_isaacsim/bin/python experiments/07nov_v3/plot_top_niches.py debug_outputs/coevolve/v3/k5_none 3 \
        [--cells 8] [--per-cell 3] [--min-hands 4] [--out <png>]
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments/06nov_coevolve_v2")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("gen", type=int)
    ap.add_argument("--cells", type=int, default=8); ap.add_argument("--per-cell", type=int, default=3)
    ap.add_argument("--min-hands", type=int, default=4); ap.add_argument("--out")
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch
    from plot_morph_gallery import mesh_tris, canon, KIND_RGB, PALM_RGB
    gd = REPO / a.run / f"gen_{a.gen:04d}"
    lab = json.loads((gd / "labels.json").read_text())
    man = json.loads((gd / "manifest.json").read_text())
    by = defaultdict(list)
    for h in man["hands"]:
        if h["id"] in lab:
            by[h["niche"]].append((lab[h["id"]][0], lab[h["id"]][1], h))
    ranked = sorted((c for c in by if len(by[c]) >= a.min_hands), key=lambda c: -np.mean([v for v, *_ in by[c]]))
    cells = ranked[: a.cells]
    ncol = 2 * a.per_cell
    fig, axs = plt.subplots(len(cells), ncol, figsize=(1.9 * ncol, 2.15 * len(cells)), squeeze=False, facecolor="white")
    light = np.array([0.4, -0.5, 0.75]); light /= np.linalg.norm(light)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#e4e3df")
    for r, c in enumerate(cells):
        rows = sorted(by[c], key=lambda t: -t[0])
        m = np.mean([v for v, *_ in rows])
        axs[r, 0].set_ylabel(f"#{r + 1}  {m:.3f}\n(n {len(rows)})\n" + c.replace("_", "\n"), fontsize=6.5, rotation=0,
                             ha="right", va="center", labelpad=4)
        print(f"#{r + 1:2d} {c:36s} mean {m:.3f} goals/ep over {len(rows)} hands; best {rows[0][0]:.3f}")
        for k, (v, n, h) in enumerate(rows[: a.per_cell]):
            p = REPO / h["spec"]
            spec = json.loads(p.read_text())
            kinds = {f"f{i}_j{j}": ("flex" if (int(jt["axis"][0]) - int(f.get("roll", 0))) % 180 == 0 else "abd")
                     for i, f in enumerate(spec["design"]["fingers"]) for j, jt in enumerate(f["joints"])}
            rs, ctr, R = canon(str(p.relative_to(REPO)))
            tris, cols = mesh_tris(REPO / rs.urdf_path, lambda jn: kinds.get(jn, "flex"))
            P = (tris - ctr) @ R.T * 1000
            for w, (ia, ib, idp) in enumerate([(1, 2, 0), (2, 0, 1)]):
                ax = axs[r, 2 * k + w]
                xy = P[..., [ia, ib]]; depth = P[..., idp].mean(1)
                nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
                o = np.argsort(depth)
                sh = (0.35 + 0.65 * np.abs(nrm @ light))[o, None]
                ax.add_collection(PolyCollection(xy[o], facecolors=np.clip(cols[o] * sh, 0, 1), edgecolors="none"))
                ax.plot([0], [0], "k+", ms=4)
                ax.set_xlim(-200, 200); ax.set_ylim(-200, 200); ax.set_aspect("equal")
                if w == 0:
                    ax.set_title(f"{v:.3f} goals/ep ({n} ep)", fontsize=6)
    fig.legend(handles=[Patch(color=KIND_RGB["flex"], label="flexion"), Patch(color=KIND_RGB["abd"], label="abduction"),
                        Patch(color=PALM_RGB, label="palm")], loc="upper right", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"{Path(a.run).name} gen {a.gen}: top niches by mean goals / episode (best hands; face, then side view; mm)",
                 x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0.08, 0, 1, 0.97))
    out = Path(a.out) if a.out else gd / "top_niches.png"
    fig.savefig(REPO / out if not out.is_absolute() else out, dpi=110, facecolor="white")
    print("[top niches]", out)


if __name__ == "__main__":
    main()
