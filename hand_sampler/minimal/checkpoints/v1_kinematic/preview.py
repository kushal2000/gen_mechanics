"""Render sampled hands to a PNG. No browser, no viser -- a sanity check you can
look at, and a fallback if the interactive viewer is inconvenient.

    python preview.py --seed 0 --count 6 --flex 0 --out preview.png
"""

from __future__ import annotations

import argparse
import math
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sampler as S  # noqa: E402

FACE_COLOR = {"+z": "#e87a54", "+y": "#56a0d6", "-y": "#96c86e"}


def draw_hand(ax, hand: S.Hand, flex_deg: float) -> None:
    # palm as a wireframe box
    hx, hy = S.PALM_THICKNESS / 2, S.PALM_WIDTH / 2
    z0, z1 = 0.0, S.PALM_LENGTH
    for sx in (-hx, hx):
        ax.plot([sx] * 5, [-hy, hy, hy, -hy, -hy], [z0, z0, z1, z1, z0],
                color="#888", lw=0.8)
    for sy in (-hy, hy):
        ax.plot([-hx, hx, hx, -hx, -hx], [sy] * 5, [z0, z0, z1, z1, z0],
                color="#888", lw=0.8)

    for f in hand.fingers:
        angles = {(i, "FE"): math.radians(flex_deg)
                  for i, d in enumerate(f.dofs) if "FE" in d}
        joints, segs = S.forward_kinematics(f, angles)
        c = FACE_COLOR[f.face]
        for p0, p1, _ in segs:
            ax.plot(*zip(p0, p1), color=c, lw=6, solid_capstyle="round", alpha=0.85)
        J = np.array(joints)
        ax.scatter(J[:-1, 0], J[:-1, 1], J[:-1, 2], color="k", s=22, depthshade=False)
        ax.scatter(*J[-1], color="#f0dc78", s=34, edgecolors="k",
                   linewidths=0.5, depthshade=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--flex", type=float, default=0.0, help="all FE joints, degrees")
    ap.add_argument("--out", default="preview.png")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    hands = [S.sample_hand(rng) for _ in range(args.count)]

    cols = min(3, args.count)
    rows = math.ceil(args.count / cols)
    fig = plt.figure(figsize=(4.2 * cols, 4.0 * rows))
    for i, h in enumerate(hands):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        draw_hand(ax, h, args.flex)
        lim = 0.16
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-0.02, 2 * lim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x (palm side)", fontsize=7)
        ax.set_ylabel("y", fontsize=7)
        ax.set_zlabel("z (wrist->tip)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=22, azim=-62)
        desc = " | ".join(
            f"{f.face} {'+'.join(str(int(L * 100)) for L in f.link_lengths)}cm "
            f"[{','.join('/'.join(d) for d in f.dofs)}]" for f in h.fingers
        )
        ax.set_title(f"{len(h.fingers)}F {h.n_joints}J\n{desc}", fontsize=6.5)
    fig.suptitle(f"minimal hand sampler - seed {args.seed}, flex {args.flex:.0f} deg",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
