"""A picture sheet of the 30 seed hands (assets/urdf/29oct_niche_seed/), every joint at 0, three orthographic views
each (along the palm frame's x, y and z axes), at one scale: the Claude designer's input (README.md here).

    .venv_isaacsim/bin/python experiments/29oct_evolution_from_scratch/claude_pick/seed_sheet.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
SEED = REPO / "assets/urdf/29oct_niche_seed"
OUT = pathlib.Path(__file__).resolve().parent / "seed_hands.png"
VIEWS = {"along x": (0, 1, 2), "along y": (0, 2, 1), "along z": (1, 0, 2)}   # (drop axis, horizontal, vertical)
VIEWS = {"view along x (y right, z up)": (0, 1, 2), "view along y (x right, z up)": (1, 0, 2),
         "view along z (x right, y up)": (2, 0, 1)}


def faces_of(urdf: pathlib.Path):
    import trimesh
    import yourdfpy
    u = yourdfpy.URDF.load(str(urdf), load_meshes=True, build_collision_scene_graph=False)
    u.update_cfg(np.zeros(len(u.actuated_joint_names)))
    tris, cols = [], []
    for m in u.scene.dump(concatenate=False):
        if isinstance(m, trimesh.Trimesh) and len(m.faces):
            c = np.asarray(m.visual.main_color[:3], float) / 255.0 if hasattr(m.visual, "main_color") else np.full(3, .7)
            tris.append(m.vertices[m.faces]); cols.append(np.repeat(c[None], len(m.faces), 0))
    return np.concatenate(tris), np.concatenate(cols)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    hands = json.loads((SEED / "manifest.json").read_text())["hands"]
    data = [(h["id"], *faces_of(REPO / h["spec"].replace(".spec.json", ".urdf"))) for h in hands]
    allv = np.concatenate([t.reshape(-1, 3) for _, t, _ in data])
    lo, hi = allv.min(0) - 0.01, allv.max(0) + 0.01
    fig, axes = plt.subplots(10, 9, figsize=(18, 22))
    for k, (hid, tris, cols) in enumerate(data):
        for v, (name, (d, a, b)) in enumerate(VIEWS.items()):
            ax = axes[k // 3, (k % 3) * 3 + v]
            n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            shade = 0.35 + 0.65 * np.abs(n[:, d])
            order = np.argsort(tris[:, :, d].mean(1))           # far to near along +axis d
            ax.add_collection(PolyCollection(tris[order][:, :, [a, b]], facecolors=cols[order] * shade[order, None],
                                             edgecolors="none"))
            ax.set_xlim(lo[a], hi[a]); ax.set_ylim(lo[b], hi[b]); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{hid}" if v == 1 else name.split(" (")[0], fontsize=8)
    fig.suptitle("29 Oct seed hands, joints at 0; per hand: views along the palm frame's x, y, z (same scale)", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT, dpi=90)
    print(OUT)


if __name__ == "__main__":
    main()
