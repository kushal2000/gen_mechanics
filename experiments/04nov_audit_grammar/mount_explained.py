"""Today's mount rule vs the translation-free proposal, on one example (g0058_c00 losing finger 2), in two rows.

    .venv_isaacsim/bin/python experiments/04nov_audit_grammar/mount_explained.py

Row 1, today: the mount is fixed at the origin and the fingers are placed on a grid around it; after an edit the whole
hand is slid by whole 40 mm steps until the mount is near the middle of the fingers. Row 2, proposed: the fingers are
placed relative to each other, with no mount; after an edit nothing moves; the mount is put in last, at the free point
nearest the middle of the fingers (2 mm search). Writes debug_outputs/grammar_audit/mount_explained.png.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon
    from hand_sampler.grammar import derive as dv
    from hand_sampler.grammar import hand as gh
    from hand_sampler.grammar import operators as gops
    from hand_sampler.grammar import viability as gvb

    R = replace(gh.FEETECH_COARSE_RULES, max_palm_joints=0)
    parent = gh.hand_from_dict(json.loads((REPO / "debug_outputs/evolution/map_elites/hands/g0058_c00/"
                                                   "g0058_c00.design.json").read_text()))
    raw = gops.OPERATOR_BY_NAME["remove_finger"].apply(parent, (2,), R)
    slid = gops.recentre(raw, R)
    c = dv.palm_centre_mm(raw)
    exact = gh.shifted(raw, int(round(-c[0])), int(round(-c[1])))
    offs = sorted(((dy, dz) for dy in range(-60, 61, 2) for dz in range(-60, 61, 2)), key=lambda o: math.hypot(*o))
    dy, dz = next(o for o in offs if gvb.clear(gh.shifted(exact, *o)))
    mount_raw = (c[0] - dy, c[1] - dz)                   # where the mount lands, in raw's own coordinates
    keep = [0, 1, 3]

    def hit_by_mount(h):
        return {k[1] for a, b, _ in gvb.c1_self_overlap(h).pairs for k in (a, b)
                if k[0] in ("motor", "link") and "mount" in (a[0], b[0])}

    def draw(ax, h, labels, mount=None, middle=True, grid_origin=None, bad=(), note=""):
        if grid_origin is not None:                        # the base grid, drawn where it sits
            gy, gz = grid_origin
            pts = [(gy + y, gz + z) for y in range(-160, 161, 40) for z in range(-160, 161, 40)]
            ax.scatter([p[1] for p in pts], [p[0] for p in pts], s=10, c="#d4d4d4", zorder=1)
        foot = []
        for i, f in enumerate(h.fingers):
            col = "#d03030" if i in bad else "#2a6fd6"
            for b in dv.base_motor_boxes_mm(f):
                yz = b.corners()[:, 1:]
                foot.append(yz)
                o = yz.mean(0)
                hull = yz[np.argsort(np.arctan2(yz[:, 1] - o[1], yz[:, 0] - o[0]))]
                ax.add_patch(Polygon(hull[:, ::-1], closed=True, fc=col, ec="black", alpha=0.6, zorder=3))
            L = 0.45 * gh.finger_length_mm(f)
            fa = math.radians(f.facing)
            ax.annotate("", xy=(f.z + L * math.cos(fa), f.y + L * math.sin(fa)), xytext=(f.z, f.y),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2), zorder=3, annotation_clip=False)
            ax.text(f.z, f.y, str(labels[i]), ha="center", va="center", fontsize=10, fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec=col))
        if middle:
            m = dv.palm_centre_mm(h)
            ax.plot(m[1], m[0], "+", ms=16, mew=2.5, color="#1baf7a", zorder=7)
        if mount is not None:
            ax.add_patch(Circle((mount[1], mount[0]), gh.MOUNT_RADIUS_MM, fc="#4a4a55", ec="black", alpha=0.8,
                                zorder=4))
            ax.text(mount[1], mount[0], "mount", color="white", ha="center", va="center", fontsize=7.5, zorder=5)
        ax.set_xlim(-150, 150); ax.set_ylim(-150, 150); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=9.5, va="bottom",
                color="#b02020" if bad else "#0b0b0b")

    fig, axes = plt.subplots(2, 3, figsize=(17, 11.5))
    t = dict(loc="left", fontsize=11.5)
    # row 1: today
    draw(axes[0, 0], parent, [0, 1, 2, 3], mount=(0, 0), grid_origin=(0, 0),
         note="the mount is the origin; finger bases sit on\nthe 40 mm grid around it")
    axes[0, 0].set_title("TODAY  1. the design is written around the mount", **t)
    draw(axes[0, 1], raw, keep, mount=(0, 0), grid_origin=(0, 0),
         note="the middle of the fingers (+) is now 25 mm from\nthe mount; the rule allows at most 20")
    axes[0, 1].set_title("TODAY  2. remove finger 2", **t)
    draw(axes[0, 2], slid, keep, mount=(0, 0), grid_origin=(0, 0), bad=hit_by_mount(slid),
         note="the only allowed fix: slide every finger one\nwhole 40 mm grid step -> finger 1 hits the mount\n"
              "-> the edit is REJECTED")
    axes[0, 2].set_title("TODAY  3. slide the hand 40 mm to re-centre", **t)
    # row 2: proposed (raw's own coordinates throughout; nothing moves)
    draw(axes[1, 0], parent, [0, 1, 2, 3], mount=None, grid_origin=(0, 0),
         note="the design is only the fingers, placed\nrelative to each other on the 40 mm grid; no mount")
    axes[1, 0].set_title("PROPOSED  1. the design is just the fingers", **t)
    draw(axes[1, 1], raw, keep, mount=None, grid_origin=(0, 0),
         note="the other fingers do not move; + is the middle\nof the fingers")
    axes[1, 1].set_title("PROPOSED  2. remove finger 2", **t)
    draw(axes[1, 2], raw, keep, mount=mount_raw, grid_origin=(0, 0),
         note=f"the mount goes in last, at the free point nearest\nthe middle (+), searched in 2 mm steps: "
              f"{math.hypot(dy, dz):.0f} mm away\n-> the edit is KEPT")
    axes[1, 2].set_title("PROPOSED  3. place the mount where it fits", **t)
    fig.suptitle("Where does the wrist attach? Today the mount is the fixed reference and the fingers move around it; "
                 "proposed, the fingers are the design and the mount is placed afterwards.\n(palm plane seen from the "
                 "grasp side; blue boxes = base servos, arrows = finger directions, grey dots = the 40 mm grid)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = REPO / "debug_outputs/grammar_audit/mount_explained.png"
    fig.savefig(out, dpi=110, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
