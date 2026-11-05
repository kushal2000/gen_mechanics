"""The 30 niches as a picture: one hand per niche, posed as the env poses it, laid out fingers x (palm cut, finger length).

    .venv_isaacsim/bin/python experiments/oct_2026_experiments/30oct_evolution_evaluation/plot_niches.py                 # the seed (iteration 0)
    .venv_isaacsim/bin/python experiments/oct_2026_experiments/30oct_evolution_evaluation/plot_niches.py \\
        --manifest debug_outputs/evolution/map_elites/gen_0021/manifest.json \\
        --fitness debug_outputs/evolution/map_elites/gen_0021/fitness.json --name map_elites_it21

Each hand is drawn at its home pose (every joint at 0: the template scene requires a zero home pose) and mounted as in
the in-hand olympics env: palm normal down (-z), fingers along +x (isaacsimenvs/inhand_olympics/wrist.py
palm_down_rotation). The camera looks up at it from below and to the side, as from the table. Every panel has the same
scale. With --fitness, a niche holding several hands shows its best and every panel is labelled with its goals per
episode. Writes experiments/oct_2026_experiments/30oct_evolution_evaluation/plots/niches_<name>.png.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "plots"
LENGTH = ("short", "medium", "long")                  # mean joints per finger < 2.5, 2.5-3.5, > 3.5 (descriptors.py)
VIEW = np.array([0.55, 0.75, 1.0])                    # camera -> scene: from below (looking up +z), behind and to the side
LIGHT = np.array([0.3, 0.5, 1.0])


def _wrist():
    """isaacsimenvs/inhand_olympics/wrist.py by path: the package __init__ pulls in Isaac Lab, the module does not."""
    spec = importlib.util.spec_from_file_location("olympics_wrist", REPO / "isaacsimenvs/inhand_olympics/wrist.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hand_faces(spec_path: Path, R: np.ndarray):
    """Every visual triangle of the hand at q = 0, rotated into the env's palm-down mount, with its RGB."""
    import trimesh
    import yourdfpy
    urdf = spec_path.with_name(spec_path.name.replace(".spec.json", ".urdf"))
    u = yourdfpy.URDF.load(str(urdf), load_meshes=True, build_collision_scene_graph=False)
    u.update_cfg(np.zeros(len(u.actuated_joint_names)))
    tris, cols = [], []
    for m in u.scene.dump(concatenate=False):
        if isinstance(m, trimesh.Trimesh) and len(m.faces):
            c = np.asarray(m.visual.main_color[:3], float) / 255.0 if hasattr(m.visual, "main_color") else np.full(3, .7)
            tris.append(m.vertices[m.faces] @ R.T)
            cols.append(np.repeat(c[None], len(m.faces), 0))
    return np.concatenate(tris), np.concatenate(cols)


def project(tris: np.ndarray):
    """Orthographic projection along VIEW, screen up = the palm's wrist-to-fingertip axis (+x) as seen from below."""
    d = VIEW / np.linalg.norm(VIEW)
    right = np.cross(d, [1.0, 0.0, 0.0]); right /= np.linalg.norm(right)
    up = np.cross(right, d)
    return np.stack([tris @ right, tris @ up], -1), tris @ d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="assets/urdf/29oct_niche_seed/manifest.json")
    ap.add_argument("--fitness", default=None, help="eval_hands.py output: label goals/episode, keep each niche's best")
    ap.add_argument("--name", default="seed")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    from coevolution.evolution.descriptors import niche, summary
    wrist = _wrist()
    hands = json.loads((REPO / args.manifest).read_text())["hands"]
    fit = json.loads((REPO / args.fitness).read_text())["hands"] if args.fitness else {}
    cells: dict[str, dict] = {}
    for h in hands:
        spec = REPO / h["spec"]
        design = json.loads(spec.with_name(spec.name.replace(".spec.json", ".design.json")).read_text())
        n = niche(design)
        g = fit.get(h["spec"], {}).get("goals_per_episode")
        if n not in cells or (g is not None and g > (cells[n]["goals"] or -1)):
            cells[n] = {"id": h.get("id", spec.stem), "spec": spec, "design": design, "goals": g}

    light = LIGHT / np.linalg.norm(LIGHT)
    drawn = {}
    for n, c in cells.items():
        frame = np.asarray(json.loads(c["spec"].read_text())["palm_frame"], float)
        tris, cols = hand_faces(c["spec"], wrist.palm_down_rotation(frame))
        xy, depth = project(tris)
        nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        shade = 0.35 + 0.65 * np.abs(nrm @ light)
        order = np.argsort(-depth.mean(1))            # far (large depth along VIEW) first
        drawn[n] = (xy[order], cols[order] * shade[order, None])
    allxy = np.concatenate([xy.reshape(-1, 2) for xy, _ in drawn.values()])
    lo, hi = allxy.min(0) - 0.01, allxy.max(0) + 0.01

    fig, axes = plt.subplots(5, 6, figsize=(18, 16.5))
    for r, f in enumerate(range(2, 7)):
        for k, (cut, j) in enumerate([(c, j) for c in ("nocut", "cut") for j in range(3)]):
            ax = axes[r, k]
            n = f"f{f}_{cut}_j{j}"
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#cccccc")
            if r == 0:
                ax.set_title(f"{'palm cut' if cut == 'cut' else 'no cut'}, {LENGTH[j]} fingers", fontsize=12, pad=14)
            if k == 0:
                ax.set_ylabel(f"{f} fingers", fontsize=13)
            if n not in drawn:
                ax.text(0.5, 0.5, "empty", ha="center", va="center", color="#999999", transform=ax.transAxes)
                continue
            xy, col = drawn[n]
            ax.add_collection(PolyCollection(xy, facecolors=col, edgecolors="none"))
            c = cells[n]
            s = summary(c["design"])
            label = f"{c['id']}\n{s['finger_joints']} joints"
            if c["goals"] is not None:
                label += f", {c['goals']:g} goals/ep"
            ax.text(0.5, -0.02, label, ha="center", va="top", fontsize=8.5, transform=ax.transAxes)
    fig.suptitle(f"Niches: {args.name} ({len(cells)} of 30 filled). Home pose (all joints 0), palm down as mounted in "
                 f"the env, seen from below, wrist-to-fingertip axis up the page; one scale", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(exist_ok=True)
    path = OUT / f"niches_{args.name}.png"
    fig.savefig(path, dpi=110)
    print(path)


if __name__ == "__main__":
    main()
