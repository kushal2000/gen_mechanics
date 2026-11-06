"""Each niche's best design in a campaign run's archive (by that generation's own-policy 100-env score), drawn in the
niche grid (rows = fingers, columns = motor bins) as deployed: palm down, q = 0, each finger link coloured by the
kind of joint that moves it (flexion blue, abduction red, roll green; palm and base motors grey).

    .venv_isaacsim/bin/python experiments/05nov_coevolve/plot_best_designs.py --run debug_outputs/coevolve/v1/k5_none
        [--gen G]   (default: the latest generation with a complete own-policy eval)

Writes experiments/05nov_coevolve/plots/best_designs_<run>_g<G>.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/oct_2026_experiments/30oct_evolution_evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
KIND = {(0, 0): "flex", (90, 0): "abd", (0, 90): "roll"}
KIND_RGB = {"flex": (0.16, 0.42, 0.85), "abd": (0.86, 0.20, 0.18), "roll": (0.15, 0.65, 0.30)}
PALM_RGB = (0.62, 0.62, 0.64)
FINGERS = [2, 3, 4, 5, 6]


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch
    import yourdfpy
    from plot_niches import LIGHT, _wrist, project
    import plot_search_check as psc

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="debug_outputs/coevolve/v1/k5_none")
    ap.add_argument("--gen", type=int)
    args = ap.parse_args()
    run = REPO / args.run
    g = args.gen if args.gen is not None else max(
        int(gd.name[4:]) for gd in run.glob("gen_[0-9][0-9][0-9][0-9]")
        if len(list((gd / "eval100").glob("c*_fitness.json"))) >= 16)
    gd = run / f"gen_{g:04d}"
    A, T = psc.archive_of(gd), psc.scores(gd, "eval100")
    reg = json.loads((run / "designs.json").read_text())
    if not reg:
        raise SystemExit("empty registry")
    spec_of = {h["id"]: h["spec"] for h in json.loads((gd / "manifest.json").read_text())["hands"]}
    best = {n: max(v, key=lambda i: T.get(i, -1)) for n, v in A.items()}
    bins = sorted({tuple(int(x) for x in n.split("_m")[1].split("-")) for n in A})

    wrist = _wrist()
    light = LIGHT / np.linalg.norm(LIGHT)

    def faces(spec: Path, Rm, design):
        u = yourdfpy.URDF.load(str(spec.with_name(spec.name.replace(".spec.json", ".urdf"))), load_meshes=True,
                               build_collision_scene_graph=False)
        u.update_cfg(np.zeros(len(u.actuated_joint_names)))
        tris, cols = [], []
        for node in u.scene.graph.nodes_geometry:
            Tm, gname = u.scene.graph[node]
            m = u.scene.geometry[gname]
            if not len(getattr(m, "faces", [])):
                continue
            v = (np.c_[m.vertices, np.ones(len(m.vertices))] @ Tm.T)[:, :3]
            rgb = PALM_RGB
            if gname.startswith("f") and "_link" in gname:
                i, j = int(gname.split("_")[0][1:]), int(gname.split("_")[1][4:])
                rgb = KIND_RGB[KIND.get(tuple(design["fingers"][i]["joints"][j]["axis"]), "flex")]
            tris.append(v[m.faces] @ Rm.T)
            cols.append(np.repeat(np.array(rgb)[None], len(m.faces), 0))
        return np.concatenate(tris), np.concatenate(cols)

    nr, nc = len(FINGERS), len(bins)
    fig, axs = plt.subplots(nr, nc, figsize=(2.05 * nc, 2.35 * nr), facecolor="white", squeeze=False)
    span = 0.0
    drawn = {}
    for n, i in best.items():
        spec = REPO / spec_of[i]
        frame = np.asarray(json.loads(spec.read_text())["palm_frame"], float)
        Rm = wrist.palm_down_rotation(frame)
        tris, cols = faces(spec, Rm, reg[i]["design"])
        xy, depth = project(tris)
        drawn[n] = (i, tris, cols, xy, depth)
        c = xy.reshape(-1, 2)
        span = max(span, float(np.ptp(c[:, 0])), float(np.ptp(c[:, 1])))
    for r, f in enumerate(FINGERS):
        for k, (lo, hi) in enumerate(bins):
            ax = axs[r, k]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            n = f"f{f}_m{lo}-{hi}"
            if r == 0:
                ax.set_title(f"{lo}–{hi} motors", fontsize=9, color="#6b6a66", pad=14)
            if k == 0:
                ax.set_ylabel(f"{f} fingers", fontsize=10, color="#6b6a66")
            if n not in drawn:
                ax.set_facecolor("#f3f2ef")
                continue
            i, tris, cols, xy, depth = drawn[n]
            nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
            order = np.argsort(-depth.mean(1))
            shade = (0.35 + 0.65 * np.abs(nrm @ light))[order, None]
            ax.add_collection(PolyCollection(xy[order], facecolors=np.clip(cols[order] * shade, 0, 1),
                                             edgecolors="none"))
            c = xy.reshape(-1, 2).mean(0)
            ax.set_xlim(c[0] - span / 2 * 1.05, c[0] + span / 2 * 1.05)
            ax.set_ylim(c[1] - span / 2 * 1.05, c[1] + span / 2 * 1.05)
            ax.set_aspect("equal")
            ax.text(0.5, -0.02, f"{T[i]:.1f} goals/ep", transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
                    color="#1f1f1e", fontweight="semibold")
            ax.text(0.5, -0.13, i, transform=ax.transAxes, ha="center", va="top", fontsize=6, color="#6b6a66")
    fig.legend(handles=[Patch(color=KIND_RGB[k], label=lab) for k, lab in
                        (("flex", "flexion"), ("abd", "abduction"), ("roll", "roll"))] + [Patch(color=PALM_RGB,
                                                                                               label="palm / base")],
               loc="upper right", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"{run.name} gen {g} – best design per niche (100 envs, gen-{g} policy)", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold", color="#1f1f1e")
    fig.subplots_adjust(left=0.03, right=0.995, top=0.88, bottom=0.05, hspace=0.45, wspace=0.05)
    out = Path(__file__).resolve().parent / "plots" / f"best_designs_{run.name}_g{g}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"[best_designs] wrote {out}")


if __name__ == "__main__":
    main()
