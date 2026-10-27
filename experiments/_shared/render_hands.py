"""Pictures of the Wuji v2 left hand, every joint at 0 (fingers straight and flat), seen straight on the palm (or
the back), fingers up: the full hand and the thumb + index hand, at the same scale. Drawn from the URDFs' visual
meshes (assets/urdf/unified_dynamics_commercial_hands/wuji2/, the uniform-dynamics hands every run here uses) with an
orthographic projection, depth-sorted faces and Lambert shading -- matplotlib only, no OpenGL.

    .venv_isaacsim/bin/python experiments/_shared/render_hands.py [--side palm|back]
    # -> experiments/12oct_inhand_olympics/plots/hands/wuji_<hand>_flat[_back].png
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
URDFS = REPO / "assets/urdf/unified_dynamics_commercial_hands/wuji2"
OUT = REPO / "experiments/12oct_inhand_olympics/plots/hands"           # where its pictures have always gone
HANDS = {"full": "wuji2_left", "thumb_index": "wuji2_left_only_thumb_index"}


def faces_of(urdf_name: str):
    """Every visual triangle at q = 0, in the URDF's base frame, with its RGB."""
    import trimesh
    import yourdfpy
    u = yourdfpy.URDF.load(str(URDFS / f"{urdf_name}.urdf"), load_meshes=True)
    u.update_cfg(np.zeros(len(u.actuated_joint_names)))
    tris, cols = [], []
    for m in u.scene.dump(concatenate=False):
        if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
            continue
        c = np.asarray(m.visual.main_color[:3], dtype=float) / 255.0 if hasattr(m.visual, "main_color") else \
            np.array([0.75, 0.75, 0.75])
        tris.append(m.vertices[m.faces])
        cols.append(np.repeat(c[None], len(m.faces), axis=0))
    return np.concatenate(tris), np.concatenate(cols)


def render(tris, cols, view: np.ndarray, up: np.ndarray, path: pathlib.Path, lims) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    view = view / np.linalg.norm(view)                 # from the camera into the scene
    right = np.cross(view, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, view)
    # Face normals, flipped toward the camera (mesh winding is not reliable across vendor meshes).
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    n *= -np.sign(n @ view)[:, None]
    light = -view + 0.45 * up + 0.35 * right           # a key light from the camera, a little above and right
    light /= np.linalg.norm(light)
    shade = 0.42 + 0.58 * np.clip(n @ light, 0.0, 1.0)
    rgb = np.clip(cols * shade[:, None], 0, 1)
    depth = (tris @ view).mean(axis=1)                 # larger = farther
    order = np.argsort(-depth)                         # painter's algorithm: far first
    xy = np.stack([tris @ right, tris @ up], axis=-1)[order]
    fig, ax = plt.subplots(figsize=(4.0, 4.0 * (lims[3] - lims[2]) / (lims[1] - lims[0])))
    pc = PolyCollection(xy, facecolors=rgb[order], edgecolors=rgb[order], linewidths=0.15, antialiaseds=True)
    ax.add_collection(pc)
    ax.set_xlim(lims[0], lims[1])
    ax.set_ylim(lims[2], lims[3])
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="palm", choices=("palm", "back"))
    args = ap.parse_args()
    # The palm's normal is the base frame's y axis and the fingers point along -z (q = 0); look along -y (palm
    # side) or +y (back), fingers up.
    view = np.array([0.0, 1.0, 0.0]) if args.side == "palm" else np.array([0.0, -1.0, 0.0])
    up = np.array([0.0, 0.0, -1.0])
    meshes = {h: faces_of(u) for h, u in HANDS.items()}
    # One frame for both, from the full hand's extent, so the two pictures share a scale.
    t = meshes["full"][0].reshape(-1, 3)
    v = view / np.linalg.norm(view)
    r = np.cross(v, up)
    r /= np.linalg.norm(r)
    u_ = np.cross(r, v)
    px, py = t @ r, t @ u_
    pad = 0.008
    lims = (px.min() - pad, px.max() + pad, py.min() - pad, py.max() + pad)
    for h, (tris, cols) in meshes.items():
        suffix = "" if args.side == "palm" else "_back"
        render(tris, cols, view, up, OUT / f"wuji_{h}_flat{suffix}.png", lims)


if __name__ == "__main__":
    main()
