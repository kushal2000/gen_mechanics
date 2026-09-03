"""Render hands to a PNG. No browser, no server.

    python -m hand_sampler.preview --seeds 6 --out seeds.png
    python -m hand_sampler.preview --lineage --seed 0 --out lineage.png

``viewer.py`` is the interactive tool; this is for a remote shell, a figure in
notes, or eyeballing a population at once. Same kinematics, so it shows what the
viewer would. ``--lineage`` renders a seed and each successive mutation, which is
the view that makes an operator's effect obvious: everything holds still except
the one thing it touched.
"""

from __future__ import annotations

import argparse
import math
import random

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection            # noqa: E402

from hand_sampler import genotype as G                             # noqa: E402
from hand_sampler import kinematics as K                           # noqa: E402
from hand_sampler import mutate as M                               # noqa: E402
from hand_sampler import sample as S                               # noqa: E402
from hand_sampler.viewer import theta_colour                       # noqa: E402


def _palm_faces(palm: G.Palm) -> list[np.ndarray]:
    t, w, l = palm.extents
    x, y = t / 2, w / 2
    c = np.array([[-x, -y, 0], [x, -y, 0], [x, y, 0], [-x, y, 0],
                  [-x, -y, l], [x, -y, l], [x, y, l], [-x, y, l]], float)
    idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
           (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
    return [c[list(f)] for f in idx]


def draw(ax, hand: G.Hand, title: str = "", flex: float = 0.0) -> None:
    ax.add_collection3d(Poly3DCollection(
        _palm_faces(hand.palm), facecolor=(0.47, 0.49, 0.53), alpha=0.30,
        edgecolor=(0.3, 0.3, 0.34), linewidths=0.5))

    for finger in hand.fingers:
        angles = {i: flex for i in range(finger.n_joints)}
        joints, capsules = K.forward_kinematics(finger, hand.palm, angles)
        # colour by the segment the capsule BELONGS to, which is carried on the
        # capsule -- capsule and segment indices diverge at a coincident joint
        for p0, p1, _, si in capsules:
            col = np.array(theta_colour(finger.segments[si].joint.theta)) / 255.0
            ax.plot(*zip(p0, p1), color=col, linewidth=6.0, solid_capstyle="round")

        # a joint is a dot; an off-perpendicular hinge (phi != 90) is a red ring,
        # because that is the assumption the parameter exists to test
        for si, p in enumerate(joints[:-1]):
            ax.scatter(*p, s=30, c="#1e1e23", depthshade=False, zorder=5)
            if abs(finger.segments[si].joint.phi - math.pi / 2) > 1e-9:
                ax.scatter(*p, s=80, facecolors="none", edgecolors="#d02828",
                           linewidths=1.8, depthshade=False, zorder=6)
        ax.scatter(*joints[-1], s=34, c="#f5d93c", depthshade=False, zorder=6)

    span = 0.16
    ax.set_xlim(-span / 2, span / 2)
    ax.set_ylim(-span / 2, span / 2)
    ax.set_zlim(0.0, span)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-58)
    if title:
        ax.set_title(title, fontsize=8, pad=0)


def _grid(items, out: str, cols: int = 3, flex: float = 0.0) -> None:
    rows = (len(items) + cols - 1) // cols
    fig = plt.figure(figsize=(3.3 * cols, 3.2 * rows), dpi=130)
    for i, (hand, title) in enumerate(items):
        draw(fig.add_subplot(rows, cols, i + 1, projection="3d"), hand, title, flex)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}  ({len(items)} hands)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=6, help="how many seeds to draw")
    ap.add_argument("--lineage", action="store_true",
                    help="render a mutation sequence instead of a seed population")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--flex", type=float, default=0.0,
                    help="drive every joint to this angle, in degrees. Fingers rest "
                         "pointing straight out of their face; opposition is what "
                         "FLEXION produces, so use this to see a hand close.")
    ap.add_argument("--out", default="preview.png")
    args = ap.parse_args()

    if not args.lineage:
        pop = S.seed_population(args.seed, args.seeds)
        _grid([(h, f"seed {i}  |  {h.n_fingers}f  {h.n_joints}j")
               for i, h in enumerate(pop)], args.out, flex=math.radians(args.flex))
        return

    rng = random.Random(args.seed)
    hand = S.seed_population(args.seed, 1)[0]
    items = [(hand, f"seed {args.seed}  |  {hand.n_fingers}f  {hand.n_joints}j")]
    ops = list(M.OPERATORS)
    while len(items) <= args.steps:
        op = ops[rng.randrange(len(ops))]
        child = M.mutate(rng, hand, op)
        if child is None:
            continue
        hand = child
        items.append((hand, f"{op}  |  {hand.n_fingers}f  {hand.n_joints}j"))
    _grid(items, args.out, flex=math.radians(args.flex))


if __name__ == "__main__":
    main()
