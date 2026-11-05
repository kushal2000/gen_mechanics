"""The mount-centring rule, drawn: g0058_c00 with finger 2 removed, top-down in the palm plane.

    .venv_isaacsim/bin/python experiments/04nov_audit_grammar/centring_figure.py [--hand g0058_c00] [--remove 2]

Four panels, all in the palm frame (y, z mm; the mount at the origin unless moved):
  A  the parent: bases on the 40 mm grid around the mount, the palm centre inside the +-20 mm window
  B  the finger removed, nothing else: clear of the mount, but the palm centre is outside the window
  C  after recentre: every base moved by one 40 mm step so the centre is back in the window -- now a base motor
     overlaps the mount, so C1 rejects the hand
  D  the cleaner alternative: the same arrangement as B, the mount placed at the palm's exact centre
Drawn: the allowed base spots (40-90 mm from the mount), the mount disc, each base servo's footprint, each finger as an
arrow along its facing (its length), the palm outline, the palm centre (x) and the centring window (dashed square).
Writes debug_outputs/grammar_audit/centring_<hand>_remove<i>.png.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="g0058_c00")
    ap.add_argument("--run", default="map_elites")
    ap.add_argument("--remove", type=int, default=2)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle
    from hand_sampler.grammar import derive as dv
    from hand_sampler.grammar import hand as gh
    from hand_sampler.grammar import operators as gops
    from hand_sampler.grammar import viability as gvb

    R = replace(gh.FEETECH_COARSE_RULES, max_palm_joints=0)
    path = REPO / f"debug_outputs/evolution/{args.run}/hands/{args.hand}/{args.hand}.design.json"
    parent = gh.hand_from_dict(json.loads(path.read_text()))
    raw = gops.OPERATOR_BY_NAME["remove_finger"].apply(parent, (args.remove,), R)
    shifted = gops.recentre(raw, R)
    c = dv.palm_centre_mm(raw)
    exact = gh.shifted(raw, int(round(-c[0])), int(round(-c[1])))      # the mount at the palm's centre (to 1 mm)
    # the mount at the clear point nearest the palm centre: offsets from the exact centre on a 2 mm grid, nearest first
    offs = sorted(((dy, dz) for dy in range(-60, 61, 2) for dz in range(-60, 61, 2)), key=lambda o: math.hypot(*o))
    nearest = next(gh.shifted(exact, dy, dz) for dy, dz in offs if gvb.clear(gh.shifted(exact, dy, dz)))
    keep = [i for i in range(len(parent.fingers)) if i != args.remove]       # original finger numbers

    def overlaps_mount(h):
        bad = set()
        for a, b, _ in gvb.c1_self_overlap(h).pairs:
            for k in (a, b):
                if k[0] in ("motor", "link") and "mount" in (a[0], b[0]):
                    bad.add(k[1])
        return bad

    panels = [
        ("A  parent", parent, None),
        (f"B  finger {args.remove} removed, nothing else", raw, None),
        ("C  after recentre (the rule's one allowed shift)", shifted, None),
        ("D  mount at the exact palm centre", exact, None),
        ("E  mount at the nearest clear point to the centre", nearest, None),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(27, 7.4))
    spots = gops.base_spots(R)
    for ax, (title, h, _) in zip(axes, panels):
        problems = [p for p in gh.check(h, R) if not ("grid" in p and title.startswith("D"))]
        clear = gvb.clear(h)
        hit = overlaps_mount(h) if not clear else set()
        ctr = dv.palm_centre_mm(h)
        # allowed base spots relative to the mount, and the 40/90 mm ring
        ax.scatter([s[1] for s in spots], [s[0] for s in spots], s=14, c="#c9c9c9", zorder=1)
        for r in (40, 90):
            ax.add_patch(Circle((0, 0), r, fill=False, ls=":", ec="#b0b0b0", lw=1))
        # palm outline
        for poly in dv.palm_outlines_mm(h).values():
            ax.add_patch(Polygon(np.asarray(poly)[:, ::-1], closed=True, fc="#e9e9ee", ec="#8a8a96", lw=1, zorder=2))
        # mount disc
        ax.add_patch(Circle((0, 0), gh.MOUNT_RADIUS_MM, fc="#4a4a55", ec="black", alpha=0.85, zorder=4))
        ax.text(0, 0, "mount", color="white", ha="center", va="center", fontsize=8, zorder=5)
        # base servo footprints and finger arrows
        for i, f in enumerate(h.fingers):
            col = "#d03030" if i in hit else "#2a6fd6"
            for b in dv.base_motor_boxes_mm(f):
                yz = b.corners()[:, 1:]
                hull = yz[np.argsort(np.arctan2(yz[:, 1] - yz[:, 1].mean(), yz[:, 0] - yz[:, 0].mean()))]
                ax.add_patch(Polygon(hull[:, ::-1], closed=True, fc=col, ec="black", alpha=0.55, lw=0.8, zorder=3))
            L = gh.finger_length_mm(f) * math.cos(math.radians(f.tilt))
            fa = math.radians(f.facing)
            ax.annotate("", xy=(f.z + L * math.cos(fa), f.y + L * math.sin(fa)), xytext=(f.z, f.y),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2), zorder=3, annotation_clip=False)
            lab = i if title.startswith("A") else keep[i]
            ax.text(f.z, f.y, str(lab), ha="center", va="center", fontsize=11, fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec=col))
        # centring window and palm centre
        ax.add_patch(Rectangle((-20, -20), 40, 40, fill=False, ls="--", ec="#1baf7a", lw=1.6, zorder=5))
        inside = (-20 <= ctr[0] < 20) and (-20 <= ctr[1] < 20)
        ax.plot(ctr[1], ctr[0], "x", ms=12, mew=3, color="#1baf7a" if inside else "#eb6834", zorder=7)
        status = []
        status.append(f"palm centre ({ctr[0]:+.0f}, {ctr[1]:+.0f}) mm: " + ("inside" if inside else "OUTSIDE") +
                      " the window")
        status.append("C1 clear" if clear else f"C1 FAILS: finger {sorted(hit)} hits the mount")
        if title.startswith("E"):
            status.append(f"mount {math.hypot(*ctr):.0f} mm from the palm centre")
        ok = clear and (inside or title[0] in "BE")
        ax.set_title(title + "\n" + "\n".join(status), loc="left", fontsize=10.5,
                     color="#0b0b0b" if ok else "#b02020")
        ax.set_xlim(-260, 260); ax.set_ylim(-260, 260); ax.set_aspect("equal")
        ax.set_xlabel("z (mm)"); ax.set_ylabel("y (mm)")
        ax.axhline(0, color="#eee", lw=0.6, zorder=0); ax.axvline(0, color="#eee", lw=0.6, zorder=0)
    fig.suptitle(f"{args.hand}: the mount-centring rule (palm plane, seen from the grasp side). Grey dots: allowed "
                 f"base spots (40 mm grid, 40-90 mm from the mount). Green dashed square: the centring window "
                 f"(+-20 mm). x: palm centre (centroid of the base servos).", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = REPO / "debug_outputs/grammar_audit" / f"centring_{args.hand}_remove{args.remove}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
