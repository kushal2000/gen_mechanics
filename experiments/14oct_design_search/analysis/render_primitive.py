"""Pictures of a primitive hand (primitive_hands.py) from its URDF: looking straight at the grasp face with the fingers
straight, and a three-quarter view with every joint curled -- matplotlib only (experiments/_shared/render_hands.py renderer).

    .venv_isaacsim/bin/python experiments/14oct_design_search/analysis/render_primitive.py \\
        --hand prim4x4_r10l25_g090_090_090_090 [--curl 45]
    # -> experiments/14oct_design_search/plots/<hand>_{face,curl<deg>}.png
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "experiments/_shared"))
sys.path.insert(0, str(HERE.parent))
import primitive_hands as ph  # noqa: E402
import render_hands as rh  # noqa: E402


def faces(urdf: pathlib.Path, q: float):
    import yourdfpy
    u = yourdfpy.URDF.load(str(urdf))
    u.update_cfg({n: q for n in u.actuated_joint_names})
    tris, cols = [], []
    for m in u.scene.dump(concatenate=False):
        c = np.asarray(m.visual.main_color[:3], float) / 255 if hasattr(m.visual, "main_color") else np.full(3, 0.8)
        tris.append(m.vertices[m.faces])
        cols.append(np.repeat(c[None], len(m.faces), 0))
    return np.concatenate(tris), np.concatenate(cols)


def view_limits(tris, view, up, pad=0.01):
    v = view / np.linalg.norm(view)
    up = up - v * (up @ v)
    r = np.cross(v, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, v)
    p = tris.reshape(-1, 3)
    return (float((p @ r).min()) - pad, float((p @ r).max()) + pad, float((p @ u).min()) - pad,
            float((p @ u).max()) + pad), up


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="prim4x4_r10l25_g090_090_090_090")
    ap.add_argument("--curl", type=float, default=45.0, help="degrees on every joint for the curled view")
    ap.add_argument("--extent", type=float, default=0.0,
                    help="face view: a fixed square +-extent (m) around the palm centre, so hands share a scale "
                         "(0: fit each hand)")
    ap.add_argument("--out", default="", help="the folder for the pictures (default: this experiment's plots)")
    args = ap.parse_args()
    urdf = ph.OUT / f"{args.hand}.urdf"
    out = pathlib.Path(args.out) if args.out else HERE.parent / "plots"
    out.mkdir(parents=True, exist_ok=True)
    # ``view`` points from the camera into the scene, so a camera on the grasp side (+x) looks along -x.
    # The grasp face, straight on, bearing 0 (+z) up.
    t, c = faces(urdf, 0.0)
    lims, up = view_limits(t, np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    if args.extent > 0:
        e = args.extent
        lims = (-e, e, -e, e)
    rh.render(t, c, np.array([-1.0, 0.0, 0.0]), up, out / f"{args.hand}_face.png", lims)
    # Curled, three-quarter: from the grasp side and a little to the side.
    t, c = faces(urdf, math.radians(args.curl))
    view, up0 = np.array([-0.75, -0.45, -0.5]), np.array([0.0, 0.3, 1.0])
    lims, up = view_limits(t, view, up0)
    rh.render(t, c, view, up, out / f"{args.hand}_curl{args.curl:.0f}.png", lims)


if __name__ == "__main__":
    main()
