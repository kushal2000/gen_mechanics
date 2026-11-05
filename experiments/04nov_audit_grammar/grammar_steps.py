"""How a random hand is built, one step at a time, drawn from the grammar's own code (no palm cuts).

    .venv_isaacsim/bin/python experiments/04nov_audit_grammar/grammar_steps.py --step base [--seed 3]

Steps (one figure each, under debug_outputs/grammar_audit/steps/):
  base  finger base placement: the 20 base spots, the outward facings at a spot, the first joint's kind (which sets
        the base servo's footprint and the allowed tilts), and a hand's fingers placed one by one with what each draw
        was rejected for (spot taken / servo hits the mount / servo hits an earlier finger's servo)
All views are the palm plane seen from the grasp side (z right, y up, mm), the mount at the origin.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = REPO / "debug_outputs/grammar_audit/steps"
KIND_RGB = {"flexion": (0.16, 0.42, 0.85), "abduction": (0.86, 0.20, 0.18), "roll": (0.15, 0.65, 0.30)}


def _footprint(C):
    """(y, z) corners of a box seen from the grasp side, as a convex polygon (z, y) for plotting."""
    yz = np.asarray(C)[:, 1:] * 1e3
    c = yz.mean(0)
    hull = yz[np.argsort(np.arctan2(yz[:, 1] - c[1], yz[:, 0] - c[0]))]
    return hull[:, ::-1]


def base_figure(seed: int) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle
    from hand_sampler.grammar import hand as gh
    from hand_sampler.grammar import operators as op

    R = replace(gh.FEETECH_COARSE_RULES, max_palm_joints=0)
    spots = op.base_spots(R)
    fig, axes = plt.subplots(1, 4, figsize=(26, 7.2))

    def frame(ax, title, lim=130):
        ax.add_patch(Circle((0, 0), gh.MOUNT_RADIUS_MM, fc="#4a4a55", ec="black", alpha=0.85, zorder=4))
        ax.text(0, 0, "mount", color="white", ha="center", va="center", fontsize=8, zorder=5)
        for r in R.base_distance_mm:
            ax.add_patch(Circle((0, 0), r, fill=False, ls=":", ec="#9a9a9a", lw=1))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel("z (mm)"); ax.set_ylabel("y (mm)")
        ax.set_title(title, loc="left", fontsize=10.5)

    # 1. the spots
    ax = axes[0]
    frame(ax, "1. Base spots: a 40 mm grid, kept 40-90 mm from the mount\n"
              f"{len(spots)} spots; a hand uses 2-6 of them, one finger per spot")
    gy = [(y, z) for y in range(-120, 121, 40) for z in range(-120, 121, 40)]
    ax.scatter([z for y, z in gy], [y for y, z in gy], s=18, c="#dddddd", zorder=1, label="grid point, outside the ring")
    ax.scatter([s[1] for s in spots], [s[0] for s in spots], s=46, c="#0b0b0b", zorder=3, label="allowed base spot")
    for s in spots:
        ax.text(s[1] + 4, s[0] + 5, f"{math.hypot(*s):.0f}", fontsize=6.5, color="#555")
    ax.legend(loc="lower left", fontsize=8, frameon=False)

    # 2. facings at two spots
    ax = axes[1]
    frame(ax, "2. Facing: the finger's direction in the plate, 30 degree steps\n"
              "only outward facings (within 90 degrees of centre -> base): 7 on an axis, 6 off it")
    ax.scatter([s[1] for s in spots], [s[0] for s in spots], s=14, c="#c9c9c9", zorder=1)
    for (y, z) in ((0, 80), (-40, -40)):
        ok = set(op.facings_at(y, z, R))
        for fa in range(0, 360, 30):
            d = np.array([math.cos(math.radians(fa)), math.sin(math.radians(fa))]) * 38    # (z, y)
            good = fa in ok
            ax.annotate("", xy=(z + d[0], y + d[1]), xytext=(z, y), zorder=3, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-|>", lw=1.8 if good else 1.0,
                                        color="#1baf7a" if good else "#d0d0d0"))
        ax.plot(z, y, "o", ms=7, color="#0b0b0b", zorder=4)
        ax.text(z + 6, y - 14, f"({y}, {z}): {len(ok)} facings", fontsize=8)
    ax.plot([], [], color="#1baf7a", lw=2, label="allowed (outward)")
    ax.plot([], [], color="#d0d0d0", lw=1, label="forbidden (points back across the palm)")
    ax.legend(loc="lower left", fontsize=8, frameon=False)

    # 3. first joint kind -> base servo: top view and side view, no mount (one finger at (0, 80) facing +z, re-drawn)
    ax = axes[2]
    ax.set_title("3. First joint kind, drawn with the base: flexion 69%, abduction 27%, roll 4%\n"
                 "it sets how the base servo sits in the plate and which tilts are allowed", loc="left", fontsize=10.5)
    for k, dz in (("flexion", -80), ("abduction", 0), ("roll", 80)):
        C = op._base_box_corners(0, 80, 0, 0, k) * 1e3          # palm frame, mm: (x normal, y, z)
        C[:, 2] += dz - 80                                     # the spot drawn at z = dz
        top = C[:, [2, 1]] + np.array([0, 45])                 # top view (z, y), raised
        side = C[:, [2, 0]] + np.array([0, -55])               # side view (z, x): x = palm normal, grasp side up
        for pts in (top, side):
            c0 = pts.mean(0)
            hull = pts[np.argsort(np.arctan2(pts[:, 1] - c0[1], pts[:, 0] - c0[0]))]
            ax.add_patch(Polygon(hull, closed=True, fc=KIND_RGB[k], ec="black", alpha=0.75, zorder=3))
        ax.annotate("", xy=(dz + 30, 45), xytext=(dz, 45), arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4))
        tilts = gh.tilts_for(k)
        ax.text(dz, 92, f"{k} first\nweight {gh.KIND_MIX[k]:.0%}\ntilt {'0 only' if len(tilts) < 2 else '0/30/60/90'}",
                ha="center", fontsize=8.5, color=KIND_RGB[k])
    ax.add_patch(Rectangle((-125, -55 - gh.PLATE_HALF_MM), 250, 2 * gh.PALM_THICKNESS_MM / 2, fc="#e6e6ea",
                           ec="#8a8a96", zorder=1))
    ax.text(-122, -55 + gh.PLATE_HALF_MM + 3, "palm plate (24 mm), grasp side up", fontsize=7.5, color="#555")
    ax.text(-122, 62, "top view (finger points along the arrow)", fontsize=7.5, color="#555")
    ax.text(-122, -100, "side view: the base servo sits in a pocket through the plate", fontsize=7.5, color="#555")
    ax.set_xlim(-130, 130); ax.set_ylim(-130, 130); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    pl, w = op._base_placements(R)
    cnt = Counter(p[4] for p in pl)
    ax.text(-125, -125, f"{len(pl)} placements (spot x facing x tilt x first kind): "
            f"{cnt['flexion']} flexion, {cnt['abduction']} abduction, {cnt['roll']} roll", fontsize=8)

    # 4. a hand placed finger by finger, with the rejected draws
    ax = axes[3]
    rng = np.random.default_rng(seed)
    n = int(rng.integers(R.min_fingers, R.max_fingers + 1))
    cands, _ = op._base_placements(R)
    cdf = op._base_cdf(R)
    placed, fingers, reasons, marks = [], [], [], []
    for i in range(n):
        tally = Counter()
        for _ in range(200):
            c = cands[int(np.searchsorted(cdf, rng.random(), side="right"))]
            if (c[0], c[1]) in {(f.y, f.z) for f in fingers}:
                tally["spot taken"] += 1
                continue
            if not op._clears_mount(*c):
                tally["servo hits the mount"] += 1
                marks.append((c, "mount"))
                continue
            if not op._base_clear(c, placed):
                tally["servo hits an earlier servo"] += 1
                marks.append((c, "earlier"))
                continue
            break
        yb, zb, fab, tb, kb = c
        fingers.append(gh.Finger(y=yb, z=zb, facing=fab, tilt=tb))
        placed.append(op._base_box_corners(*c))
        reasons.append((i, c, dict(tally)))
    frame(ax, f"4. A {n}-finger hand placed one finger at a time (seed {seed})\n"
              "per finger: draw by weight until a placement fits a free spot\nand its servo clears the mount and earlier servos")
    ax.scatter([s[1] for s in spots], [s[0] for s in spots], s=14, c="#c9c9c9", zorder=1)
    for c, why in marks[:60]:
        ax.plot(c[1], c[0], "x", ms=7, mew=1.5,
                color="#eb6834" if why == "mount" else "#b07cc6", zorder=2)
    lines = []
    for i, c, tally in reasons:
        yb, zb, fab, tb, kb = c
        ax.add_patch(Polygon(_footprint(op._base_box_corners(*c)), closed=True, fc=KIND_RGB[kb], ec="black",
                             alpha=0.7, zorder=3))
        d = np.array([math.cos(math.radians(fab)), math.sin(math.radians(fab))]) * 34
        ax.annotate("", xy=(zb + d[0], yb + d[1]), xytext=(zb, yb), zorder=4, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))
        ax.text(zb, yb, str(i), ha="center", va="center", fontsize=10, fontweight="bold", zorder=5,
                bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="black"))
        rej = ", ".join(f"{v} {k}" for k, v in tally.items()) or "first draw fit"
        lines.append(f"finger {i}: ({yb}, {zb}) facing {fab}, {kb} first, tilt {tb}  [{rej}]")
    ax.plot([], [], "x", color="#eb6834", label="rejected: servo hits the mount")
    ax.plot([], [], "x", color="#b07cc6", label="rejected: servo hits an earlier servo")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.text(0.0, -0.13, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=8)
    fig.suptitle("Step 1 of a random hand: where each finger's base goes (palm plane, seen from the grasp side; "
                 "mount at the origin). After this: the finger's joints and links, then the clearance check.",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "step1_base_placement.png"
    fig.savefig(out, dpi=115, facecolor="white")
    return out


def tilt_figure() -> Path:
    """Step 2: tilt -- side views (along the finger, palm normal up) of three cases."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    from hand_sampler.grammar import derive as dv
    from hand_sampler.grammar import hand as gh
    from hand_sampler.grammar import viability as gvb
    from hand_sampler.grammar.hand import Finger, Hand, Joint

    K = {"flexion": (0, 0), "abduction": (90, 0), "roll": (0, 90)}
    KN = {(0, 0): "flexion", (90, 0): "abduction", (0, 90): "roll"}

    def side(ax, hand, i, q=None, title="", bad=False):
        """Finger i's solids, the mount and the plate seen from the side: s along the finger's facing, h the palm
        normal (grasp side up); solids in collision outlined red."""
        f = hand.fingers[i]
        fa = math.radians(f.facing)
        u = np.array([math.sin(fa), math.cos(fa)])                  # (y, z) direction of the finger in the plate
        hits = set()
        res = gvb.c1_self_overlap(hand)
        for a, b, _ in res.pairs:
            hits.update((a, b))
        span = dv.palm_outlines_mm(hand)[-1] @ u
        ax.add_patch(Rectangle((float(span.min()), -gh.PLATE_HALF_MM), float(span.max() - span.min()),
                               gh.PALM_THICKNESS_MM, fc="#e6e6ea", ec="#8a8a96", zorder=1))
        mp = dv.mount_points_m() * 1e3
        for s0 in (mp,):
            pts = np.c_[s0[:, 1:] @ u, s0[:, 0]]
            c0 = pts.mean(0); hull = pts[np.argsort(np.arctan2(pts[:, 1] - c0[1], pts[:, 0] - c0[0]))]
            ax.add_patch(Polygon(hull, closed=True, fc="#4a4a55", ec="red" if ("mount",) in hits else "black",
                                 lw=2.5 if ("mount",) in hits else 0.8, alpha=0.8, zorder=2))
        for sol in dv.solids_at(hand, q):
            k = sol.key
            if k[0] not in ("motor", "link") or k[1] != i:
                continue
            P = sol.points * 1e3
            if sol.radius > 0:                                     # capsule: its segment, thickened
                a, b = P[0], P[1]
                d = b - a
                n = np.cross(d, np.r_[0.0, *u]) if np.linalg.norm(d) > 0 else np.r_[1.0, 0.0, 0.0]
                ring = []
                for t in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                    for e in (a, b):
                        ring.append(e + sol.radius * 1e3 * (np.cos(t) * np.r_[1.0, 0, 0] + np.sin(t) * np.r_[0, *u]))
                P = np.array(ring)
            pts = np.c_[P[:, 1:] @ u, P[:, 0]]
            c0 = pts.mean(0); hull = pts[np.argsort(np.arctan2(pts[:, 1] - c0[1], pts[:, 0] - c0[0]))]
            if k[0] == "motor":
                col = KIND_RGB[KN[tuple(f.joints[0].axis)]]
            else:
                col = KIND_RGB[KN[tuple(f.joints[k[2]].axis)]]
            ax.add_patch(Polygon(hull, closed=True, fc=col, ec="red" if k in hits else "black",
                                 lw=2.5 if k in hits else 0.6, alpha=0.75, zorder=3))
        ax.set_xlim(-60, 260); ax.set_ylim(-60, 220); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ok = res.ok
        ax.set_title(title + ("\nC1 clear" if ok else "\nC1 FAILS (red outline)"), loc="left", fontsize=9.5,
                     color="#0b0b0b" if ok else "#b02020")

    fig, axes = plt.subplots(4, 4, figsize=(19, 17))

    def chain_after(kind, L0):
        return (Joint("hinge", K[kind], L0), Joint("hinge", K["flexion"], 50), Joint("hinge", K["flexion"], 50),
                Joint("hinge", K["flexion"], 30))
    rows = [("A. abduction first, 70 mm first link: every tilt clears", "abduction", 70),
            ("B. abduction first, 50 mm first link: 30 and 60 deg bury the next servo in the plate", "abduction", 50)]
    for r, (lab, kind, L0) in enumerate(rows):
        for c, t in enumerate((0, 30, 60, 90)):
            h = Hand(fingers=(Finger(y=0, z=80, facing=0, tilt=t, joints=chain_after(kind, L0)),))
            side(axes[r, c], h, 0, title=(lab + "\n" if c == 0 else "") + f"tilt {t} deg")
    # C: a flexion-first finger: tilt is not allowed; bending its first joint gives the same shapes at run time
    for c, a in enumerate((0, 30, 60, 90)):
        h = Hand(fingers=(Finger(y=0, z=80, facing=0, tilt=0, joints=chain_after("flexion", 50)),))
        q = np.zeros(len(dv.dofs(h)))
        q[0] = math.radians(a)
        side(axes[2, c], h, 0, q=q if a else None,
             title=("C. flexion first: tilt is always 0 -- bending the first joint\n(which the policy does) already "
                    "makes these shapes, so a tilt would only duplicate designs\n" if c == 0 else "") +
             f"tilt 0, first joint bent {a} deg")
    # D: g0058_c00's finger 1 (abduction on a 0 mm link first) at the four tilts, in its hand
    G = [Finger(y=-80, z=-40, facing=210, tilt=0, joints=(Joint("hinge", K["flexion"], 50), Joint("hinge", K["flexion"], 70),
                Joint("hinge", K["flexion"], 70), Joint("hinge", K["abduction"], 30))),
         Finger(y=-40, z=40, facing=330, tilt=0, joints=(Joint("hinge", K["abduction"], 0), Joint("hinge", K["flexion"], 50),
                Joint("hinge", K["flexion"], 50), Joint("hinge", K["flexion"], 30), Joint("hinge", K["flexion"], 40))),
         Finger(y=0, z=-80, facing=180, tilt=0, joints=(Joint("hinge", K["flexion"], 90), Joint("hinge", K["flexion"], 30),
                Joint("hinge", K["flexion"], 60), Joint("hinge", K["roll"], 30))),
         Finger(y=40, z=0, facing=90, tilt=0, joints=(Joint("hinge", K["flexion"], 30), Joint("hinge", K["flexion"], 40),
                Joint("hinge", K["flexion"], 60), Joint("hinge", K["flexion"], 50)))]
    for c, t in enumerate((0, 30, 60, 90)):
        fs = list(G); fs[1] = replace(fs[1], tilt=t)
        side(axes[3, c], Hand(fingers=tuple(fs)), 1,
             title=("D. g0058_c00 finger 1 (abduction on a 0 mm link):\nonly 0 and 90 deg clear, so +-30 deg steps "
                    "can never reach 90\n" if c == 0 else "") + f"tilt {t} deg")
    fig.suptitle("Step 2: tilt. Side views along each finger (palm normal up, grasp side above the grey plate; the dark "
                 "block under the plate is the mount). Colours: the kind of the joint that moves each link "
                 "(flexion blue, abduction red, roll green).", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "step2_tilt.png"
    fig.savefig(out, dpi=105, facecolor="white")
    return out


def tilt_c1_figure() -> Path:
    """Step 2b: the clearance check on a tilted finger, zoomed: the plate's real extent, the base joint, and the next
    servo (link 0's far end), for first links of 50 and 70 mm at the four tilts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    from hand_sampler.grammar import derive as dv
    from hand_sampler.grammar import hand as gh
    from hand_sampler.grammar import viability as gvb
    from hand_sampler.grammar.hand import Finger, Hand, Joint

    A, F = (90, 0), (0, 0)

    def hull(pts):
        c0 = pts.mean(0)
        return pts[np.argsort(np.arctan2(pts[:, 1] - c0[1], pts[:, 0] - c0[0]))]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
    for r, L0 in enumerate((50, 70)):
        for c, t in enumerate((0, 30, 60, 90)):
            ax = axes[r, c]
            h = Hand(fingers=(Finger(0, 80, 0, t, (Joint("hinge", A, L0), Joint("hinge", F, 50), Joint("hinge", F, 50),
                                                   Joint("hinge", F, 30))),))
            ok = gvb.c1_self_overlap(h).ok
            span = dv.palm_outlines_mm(h)[-1][:, 1]                       # facing 0: the finger runs along +z
            edge = float(span.max())
            ax.add_patch(Rectangle((float(span.min()), -gh.PLATE_HALF_MM), edge - float(span.min()),
                                   gh.PALM_THICKNESS_MM, fc="#dcdce3", ec="#6d6d7a", lw=1.2, zorder=1))
            ax.add_patch(Rectangle((float(span.min()), -gh.PLATE_HALF_MM - 2), edge - float(span.min()) + 2,
                                   gh.PALM_THICKNESS_MM + 4, fill=False, ls=":", ec="#6d6d7a", zorder=1))
            sol = {x.key: x for x in dv.solids_at(h)}
            for key, fc in ((("motor", 0), KIND_RGB["abduction"]), (("link", 0, 0), KIND_RGB["abduction"]),
                            (("link", 0, 1), KIND_RGB["flexion"])):
                P = sol[key].points * 1e3
                ax.add_patch(Polygon(hull(np.c_[P[:, 2], P[:, 0]]), closed=True, fc=fc, ec="black", alpha=0.35,
                                     lw=0.6, zorder=2))
            fp, _ = dv.shape_points(sol[("link", 0, 0)].far)
            inside = not ok
            ax.add_patch(Polygon(hull(np.c_[fp[:, 2], fp[:, 0]]), closed=True, fill=False, lw=2.6,
                                 ec="#d01010" if inside else "#108a10", zorder=4))
            b = dv.mount_point_m(h.fingers[0]) * 1e3
            ax.plot(b[2], b[0], "ko", ms=6, zorder=5)
            ax.annotate("base joint", (b[2], b[0]), (b[2] - 38, b[0] - 30), fontsize=8,
                        arrowprops=dict(arrowstyle="-", color="#444"))
            ax.axvline(edge, color="#6d6d7a", ls="--", lw=1)
            ax.text(edge + 1.5, -40, "plate\nedge", fontsize=8, color="#555")
            ax.set_xlim(40, 175); ax.set_ylim(-45, 90); ax.set_aspect("equal")
            ax.set_xlabel("along the finger (mm)"); ax.set_ylabel("palm normal (mm)")
            low, near = float(fp[:, 0].min()), float(fp[:, 2].min())
            why = ("above the plate" if low >= gh.PLATE_HALF_MM + 2 else
                   "past the plate edge" if near >= edge + 2 else "inside the plate")
            ax.set_title(f"first link {L0} mm, tilt {t} deg\nnext servo (outlined): {why} -> C1 "
                         f"{'clear' if ok else 'FAILS'}", loc="left", fontsize=10,
                         color="#0b0b0b" if ok else "#b02020")
    fig.suptitle("Why some tilts fail the clearance check (C1). The base joint sits inside the plate (the servo is "
                 "pocketed into it), so link 0 starts inside the palm; that is allowed. But the NEXT servo (the far "
                 "end of link 0,\noutlined) must keep one 2 mm wall (dotted) from the plate: it must end up past "
                 "the plate's edge or above it. A short first link tilted 30 or 60 deg leaves it in neither place.",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "step2b_tilt_c1.png"
    fig.savefig(out, dpi=105, facecolor="white")
    return out


def joints_figure(seed: int) -> Path:
    """Step 3: a finger's joints -- how many, and what each kind does when it turns (top and side views)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    from dataclasses import replace as _replace
    from hand_sampler.grammar import derive as dv
    from hand_sampler.grammar import hand as gh
    from hand_sampler.grammar import operators as op
    from hand_sampler.grammar.hand import Finger, Hand, Joint

    K = {"flexion": (0, 0), "abduction": (90, 0), "roll": (0, 90)}
    KN = {v: k for k, v in K.items()}

    def hull(pts):
        c0 = pts.mean(0)
        return pts[np.argsort(np.arctan2(pts[:, 1] - c0[1], pts[:, 0] - c0[0]))]

    def draw(ax, hand, q, view, alpha=0.8, ghost=False):
        """view 'top': (z, y) the palm plane; 'side': (z, x) along the finger with the palm normal up."""
        f = hand.fingers[0]
        for sol in dv.solids_at(hand, q):
            k = sol.key
            if k[0] not in ("motor", "link"):
                continue
            P = sol.points * 1e3
            if sol.radius > 0:
                a, b = P[0], P[1]
                r = sol.radius * 1e3
                P = np.array([e + r * np.array(d) for e in (a, b) for d in
                              [(np.cos(t), np.sin(t), 0) for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)] +
                              [(np.cos(t), 0, np.sin(t)) for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)] +
                              [(0, np.cos(t), np.sin(t)) for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)]])
            xy = np.c_[P[:, 2], P[:, 1]] if view == "top" else np.c_[P[:, 2], P[:, 0]]
            col = KIND_RGB[KN[tuple(f.joints[0 if k[0] == "motor" else k[2]].axis)]]
            ax.add_patch(Polygon(hull(xy), closed=True, fc=col if not ghost else "none", ec=col if ghost else "black",
                                 lw=1.6 if ghost else 0.5, ls="--" if ghost else "-", alpha=alpha, zorder=3))

    fig = plt.figure(figsize=(21, 13))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.05])
    # rows 1-2: one finger (flexion base, then the demo joint, then two flexions); the demo joint turned 45 deg
    for c, kind in enumerate(("flexion", "abduction", "roll")):
        chain = (Joint("hinge", K["flexion"], 50), Joint("hinge", K[kind], 50), Joint("hinge", K["flexion"], 50),
                 Joint("hinge", K["flexion"], 30))
        h = Hand(fingers=(Finger(0, 0, 0, 0, chain),))
        q = np.zeros(len(dv.dofs(h)))
        q[1] = math.radians(45)
        for r, view in enumerate(("top", "side")):
            ax = fig.add_subplot(gs[r, 2 * c:2 * c + 2])
            if view == "side":
                ax.add_patch(Rectangle((-20, -gh.PLATE_HALF_MM), 40, gh.PALM_THICKNESS_MM, fc="#e6e6ea", ec="#8a8a96"))
            draw(ax, h, None, view, alpha=0.85)
            draw(ax, h, q, view, alpha=0.9, ghost=True)
            ax.set_xlim(-20, 230); ax.set_ylim(-110, 110); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            what = {"flexion": "curls the rest of the finger toward the grasp side (side view)",
                    "abduction": "swings the rest of the finger sideways (top view)",
                    "roll": "twists the rest of the finger about its own length:\nthe next flexion now bends sideways"}[kind]
            ax.set_title((f"joint 1 = {kind} ({gh.KIND_MIX[kind]:.0%} of later joints): {what}\n" if r == 0 else "") +
                         f"{'top view (palm plane)' if view == 'top' else 'side view (palm normal up)'}; "
                         "solid: q = 0, dashed: joint 1 at +45 deg", loc="left", fontsize=9.5)
    # row 3: random fingers with 1-5 joints, from random hands
    rng = np.random.default_rng(seed)
    R = gh.FEETECH_COARSE_NOCUT_RULES
    want, got = list(range(1, 6)), {}
    while len(got) < 5:
        hh = op.random_hand(rng, R)
        for f in hh.fingers:
            n = len(f.joints)
            if n in want and n not in got and f.tilt == 0:
                got[n] = f
    for c, n in enumerate(want):
        ax = fig.add_subplot(gs[2, c if c < 5 else 5])
        f = _replace(got[n], y=0, z=0, facing=0)
        h = Hand(fingers=(f,))
        draw(ax, h, None, "top")
        ax.set_xlim(-20, 240); ax.set_ylim(-60, 60); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{n} joint{'s' if n > 1 else ''}: " + ", ".join(f"{KN[tuple(j.axis)][:4]} {j.length}" for j in f.joints),
                     loc="left", fontsize=8.5)
    ax = fig.add_subplot(gs[2, 5]); ax.axis("off")
    ax.text(0, 0.95, "How a finger's joints are drawn:\n\n1. joint count n: uniform 1-5\n2. joint 1's kind: fixed with the base\n"
            "    (step 1)\n3. joints 2..n: flexion 69%, abduction 27%,\n    roll 4% (the commercial-hand mix)\n"
            "4. every axis exactly on its kind\n5. link lengths: step 4\n\nRow 3: real fingers from random hands\n"
            "(top view, laid out pointing right)", va="top", fontsize=10, family="monospace")
    fig.suptitle("Step 3: the finger's joints. Colours: the kind of the joint that moves each link (flexion blue, "
                 "abduction red, roll green); the base servo has the first joint's colour.", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "step3_joints.png"
    fig.savefig(out, dpi=100, facecolor="white")
    return out


def clearance_figure(seed: int) -> Path:
    """Step 5: the clearance check while a hand is built -- each new finger is checked against the hand so far
    (`gvb.clear(focus_finger=...)`, the mount left out under a derived mount); a failing finger is drawn again, up to
    50 times, then the whole hand. Instrumented copy of `operators._draw_fingers`: draws a few hands, keeps every
    rejected finger with the pair that failed, and counts redraws over many hands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter
    from dataclasses import replace as _replace
    from matplotlib.patches import Polygon
    from hand_sampler.grammar import derive as dv
    from hand_sampler.grammar import hand as gh
    from hand_sampler.grammar import operators as op
    from hand_sampler.grammar import viability as gvb
    from hand_sampler.grammar.hand import Finger, Hand

    R = gh.FEETECH_COARSE_NOCUT_RULES
    KN = {(0, 0): "flexion", (90, 0): "abduction", (0, 90): "roll"}

    def build(rng, n):
        fingers, placed, log = [], [], []
        for i in range(n):
            for attempt in range(op.MAX_FINGER_ATTEMPTS):
                base = op._draw_base(rng, R, placed, {(f.y, f.z) for f in fingers})
                if base is None:
                    return None, log
                y, z, fa, t, kind = base
                stub = Finger(y=y, z=z, facing=fa, tilt=t)
                joints = op._draw_joints(rng, R, kind, stub)
                if joints is None:
                    log.append((i, attempt, None, "no joints fit the length cap"))
                    continue
                f = _replace(stub, joints=joints)
                hand = Hand(fingers=tuple(fingers) + (f,))
                bad = gvb.overlaps(hand, None, len(fingers), mount=False)
                if not bad:
                    fingers.append(f)
                    placed.extend(b.corners() * dv.MM for b in dv.base_motor_boxes_mm(f))
                    break
                log.append((i, attempt, (tuple(fingers), f, bad), "C1"))
            else:
                return None, log
        return Hand(fingers=tuple(fingers)), log

    # statistics over many hands
    rng = np.random.default_rng(seed)
    per_n = {n: Counter() for n in range(2, 7)}
    why = Counter()
    for _ in range(240):
        n = int(rng.integers(2, 7))
        h, log = build(rng, n)
        per_n[n]["hands"] += 1
        per_n[n]["restarts"] += h is None
        per_n[n]["rejected fingers"] += len(log)
        for i, a, info, reason in log:
            if info is None:
                why[reason] += 1
                continue
            for ka, kb, _ in info[2]:
                kinds = sorted({k[0] for k in (ka, kb)})
                why["finger hits " + ("palm" if "palm" in kinds else "another finger" if
                                      {ka[1] if len(ka) > 1 else -1, kb[1] if len(kb) > 1 else -1} - {len(info[0])}
                                      else "itself")] += 1
                break
    # examples to draw: the first rejections of a 5-6 finger hand
    rng = np.random.default_rng(seed + 1)
    examples = []
    while len(examples) < 4:
        h, log = build(rng, int(rng.integers(5, 7)))
        examples += [e for e in log if e[2] is not None][: 4 - len(examples)]
    fig, axes = plt.subplots(1, 5, figsize=(26, 6.2))

    def footprint_draw(ax, hand, highlight=None, bad_keys=()):
        for sol in dv.solids_at(hand):
            k = sol.key
            if k[0] not in ("motor", "link"):
                continue
            P = sol.points * 1e3
            if sol.radius > 0:
                a, b = P[0], P[1]
                r = sol.radius * 1e3
                P = np.array([e + r * np.array([0, np.cos(t), np.sin(t)]) for e in (a, b)
                              for t in np.linspace(0, 2 * np.pi, 12, endpoint=False)])
            yz = P[:, 1:]
            c0 = yz.mean(0)
            hull = yz[np.argsort(np.arctan2(yz[:, 1] - c0[1], yz[:, 0] - c0[0]))]
            f = hand.fingers[k[1]]
            col = KIND_RGB[KN[tuple(f.joints[0 if k[0] == "motor" else k[2]].axis)]]
            new = k[1] == highlight
            ax.add_patch(Polygon(hull[:, ::-1], closed=True, fc=col, alpha=0.75 if new else 0.35,
                                 ec="red" if k in bad_keys else ("black" if new else "#777"),
                                 lw=2.8 if k in bad_keys else 0.6, zorder=4 if new else 3))
        ax.set_xlim(-260, 260); ax.set_ylim(-260, 260); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    for c, (i, attempt, (prev, f, bad), _) in enumerate(examples):
        ax = axes[c]
        hand = Hand(fingers=prev + (f,))
        keys = {k for a, b, _ in bad for k in (a, b)}
        footprint_draw(ax, hand, highlight=len(prev), bad_keys=keys)
        a, b, ov = bad[0]
        ax.set_title(f"finger {i} draw {attempt + 1}: REJECTED\n{a} vs {b}: {ov:.1f} mm into the 2 mm wall\n"
                     f"(faded: fingers already placed; bold: the new one; red: the clashing solids)",
                     loc="left", fontsize=9.5, color="#b02020")
    ax = axes[4]; ax.axis("off")
    lines = ["Over 240 random hands (seed %d):" % seed, "", "fingers | hands | rejected fingers | hand restarts"]
    for n in range(2, 7):
        d = per_n[n]
        lines.append(f"   {n}    |  {d['hands']:3d}  |  {d['rejected fingers'] / max(1, d['hands']):5.1f} per hand   |   "
                     f"{d['restarts']}")
    lines += ["", "what failed (first failing pair):"] + [f"  {k}: {v}" for k, v in why.most_common()]
    lines += ["", "checked: every pair of solids that do not touch", "by construction, 2 mm apart, at the zero pose;",
              "only pairs involving the new finger or the palm.", "", "on failure: redraw the finger (new base, joints,",
              "lengths), up to 50 times; then the whole hand", "with the same finger count, up to 20 times."]
    ax.text(0, 1, "\n".join(lines), va="top", fontsize=10, family="monospace")
    fig.suptitle("Step 5: the clearance check while a hand is built (palm plane, seen from the grasp side). Each new "
                 "finger is checked against the fingers already placed; the mount is placed later (step 6).",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "step5_clearance.png"
    fig.savefig(out, dpi=105, facecolor="white")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="base", choices=["base", "tilt", "tilt_c1", "joints", "clearance"])
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    print({"base": lambda: base_figure(args.seed), "tilt": tilt_figure, "tilt_c1": tilt_c1_figure, "joints": lambda: joints_figure(args.seed), "clearance": lambda: clearance_figure(args.seed)}[args.step]())


if __name__ == "__main__":
    main()
