"""Bar chart of the best and worst embodiments, with each hand drawn on the axis.

    .venv_isaacsim/bin/python -m genmech.eval.embodiments.plot_best_worst \\
        --results results/population_eval/<run>_tol3cm --out figures/best_worst.png

The x axis is the point of this figure. A design name like ``gen_0003_10683``
says nothing about the mechanism, so each bar is labelled with a render of that
design's HAND -- arm links dropped, posed at home, drawn in the palm frame at a
shared scale so the ten are actually comparable to each other.

**Ranking.** Bars are raw mean goals, and the top five are the top five by that.
"Worst" needs a tie-break: 1,107 designs score exactly 0.000, so the bottom five
by score alone would be an arbitrary pick out of a 1,107-way tie. They are broken
by the object-controlled z-score -- how a design did against the ~20 designs
holding the IDENTICAL object -- so the five shown are the ones that most
underperformed their own peer group, not five names pulled out of a hat.

Every bar carries that z as well, because a design's raw score is partly its
object's: measured on this run, the object explains 17.9% of the variance across
designs. Two designs at 7.3 goals are not equally good if one was handed a marker
and the other a hammer.

Rendering is pure matplotlib -- Poly3DCollection over the URDF's collision
geometry -- so this needs no GL context, no Isaac Sim and no GPU.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

# dataviz reference palette, categorical slots in fixed order. Slot 1 (blue) and
# slot 2 (orange); never a generated hue, never cycled.
COLOR_BEST = "#2a78d6"
COLOR_WORST = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
SURFACE = "#fcfcfb"

# Hand pictograms: orthographic top-down, looking along -z in the palm frame.
# Measured on the population, +z IS the palm normal -- the palm box spans z in
# [0, palm_extents[2]] and every finger link sits above it -- so projecting onto
# (x, y) is the view straight down onto the palm, which is the one that shows
# where the fingers are mounted and how far they spread. A perspective 3D view
# spent most of its pixels on the palm box and told you almost nothing.
PALM_FACE = "#c9c6bd"
PALM_EDGE = "#8a8880"
FINGER_FACE = "#8f9aa6"
FINGER_EDGE = "#4a5560"
# Active joints are marked in near-black ink with a light ring, NOT in a third
# hue: the figure already spends blue and orange on best/worst, and a coloured
# joint dot would read as a third category rather than as an annotation.
JOINT_FACE = "#14181d"
JOINT_RING = "#f7f6f2"


def load_rows(results_dir: Path) -> list[dict]:
    with open(results_dir / "per_design.csv") as f:
        return list(csv.DictReader(f))


def object_controlled_z(rows: list[dict]) -> np.ndarray:
    """Each design's score relative to the designs holding the IDENTICAL object.

    env i held object i % pool_size for the whole run, so designs sharing a
    pool_index are the only strictly like-for-like comparison available.
    """
    goals = np.array([float(r["mean_goals"]) for r in rows])
    pool = np.array([int(r["pool_index"]) for r in rows])
    groups = defaultdict(list)
    for p, v in zip(pool, goals):
        groups[p].append(v)
    mean = {p: float(np.mean(v)) for p, v in groups.items()}
    # A degenerate group (every member identical) would divide by zero; such a
    # group carries no ranking information anyway, so its z is 0.
    std = {p: float(np.std(v)) for p, v in groups.items()}
    return np.array([
        0.0 if std[p] <= 1e-9 else (v - mean[p]) / std[p]
        for p, v in zip(pool, goals)
    ])


def pick(rows: list[dict], n: int) -> tuple[list[int], list[int]]:
    goals = np.array([float(r["mean_goals"]) for r in rows])
    z = object_controlled_z(rows)
    best = list(np.argsort(-goals)[:n])
    # Sort by score, then by z: identical scores (the 1107-way tie at zero) are
    # ordered by how badly they did against their own object's peer group.
    worst = list(np.lexsort((z, goals))[:n])
    return best, worst


def draw_hand(ax, hand, half_extent: float) -> None:
    """Draw one top-down hand into an existing axes, at a SHARED scale.

    Drawn straight into a figure axes rather than rasterised and pasted with
    OffsetImage: that path scales by the ratio of image dpi to figure dpi, so a
    100-dpi pictogram in a 200-dpi figure comes out half the requested size --
    which is how the first version of this figure ended up with invisible icons.
    """
    from matplotlib.patches import Polygon

    link_hulls = hand["hulls"]
    for is_palm, hull in link_hulls:
        ax.add_patch(Polygon(
            hull, closed=True,
            facecolor=PALM_FACE if is_palm else FINGER_FACE,
            edgecolor=PALM_EDGE if is_palm else FINGER_EDGE,
            linewidth=1.1 if is_palm else 0.8,
            zorder=2 if is_palm else 3))
    # Joint marks encode which axes actually move at each knuckle:
    #   filled black          flexion only
    #   filled black + white  flexion AND abduction
    #   white                 abduction only (flexion ghosted)
    # Shape and fill rather than a third hue -- the figure already spends blue
    # and orange on best/worst, and this is a property of a mark, not a category.
    for pts, face, edge, width in (
        (hand["fe_only"], JOINT_FACE, JOINT_FACE, 0.0),
        (hand["fe_and_aa"], JOINT_FACE, JOINT_RING, 1.6),
        (hand["aa_only"], JOINT_RING, JOINT_FACE, 1.1),
    ):
        if not len(pts):
            continue
        ax.scatter(pts[:, 0], pts[:, 1], s=34, facecolor=face, edgecolor=edge,
                   linewidths=width, zorder=6)
    ax.set_xlim(-half_extent, half_extent)
    ax.set_ylim(-half_extent, half_extent)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)


def hand_hulls(design: str):
    """Per-link hulls of a design's HAND at HOME POSE, seen palm-down from above.

    Returns ``[(is_palm, (N,2) hull), ...]`` or None.

    Two things this has to get right, and both were wrong in the first version.

    **Home pose.** yourdfpy reports transforms at the URDF's default
    configuration, which is all zeros -- not the pose the robot actually rests
    in. The spec carries the real one as a name-keyed dict, so it is applied with
    ``update_cfg`` before any transform is read.

    **Which way is "palm down".** There is no fixed axis for it. The palm box is
    thinnest in x on design 10683 and in z on 11658, and because the sampler can
    mount fingers on any face of the palm, some designs splay their fingers along
    the palm's own normal. So the view direction is derived per design: PCA over
    the hand's points, view along the least-varying direction (the one the hand
    is flattest in, which IS the palm normal for a hand-shaped hand), with the
    most-varying direction drawn vertical so fingers point up the page. That is
    the palm-down top view for anything hand-like, and still the most informative
    silhouette for the designs that are not.
    """
    import yourdfpy
    from scipy.spatial import ConvexHull

    from genmech.robots.generated.population import load_population
    from genmech.robots.generated.synth_spec import synth_spec
    from genmech.tools.build_hand_urdf import urdf_path_for
    from genmech.tools.check_self_collision import _geometry_to_mesh

    if not hasattr(hand_hulls, "_pop"):
        hand_hulls._pop = {h.name: h for h in load_population(3)}
    hand = hand_hulls._pop.get(design)
    if hand is None:
        return None
    path = urdf_path_for(hand)
    if not path.is_file():
        return None

    urdf = yourdfpy.URDF.load(str(path), load_meshes=False,
                              load_collision_meshes=True, build_scene_graph=True,
                              build_collision_scene_graph=True)

    # HOME POSE. Only the hand joints matter -- everything here is expressed in
    # the palm frame, which the arm joints cannot move relative to the fingers.
    spec = synth_spec(hand)
    home = dict(spec.hand_default_joint_pos)
    limits = {j.name: j.limit for j in urdf.robot.joints if j.limit is not None}
    cfg = []
    for jname in urdf.actuated_joint_names:
        v = float(home.get(jname, 0.0))
        lim = limits.get(jname)
        if lim is not None and lim.lower is not None and lim.upper is not None:
            v = float(np.clip(v, lim.lower, lim.upper))
        cfg.append(v)
    urdf.update_cfg(np.array(cfg))

    palm_inv = np.linalg.inv(urdf.get_transform("gen_palm"))

    # An ACTIVE joint is one that can actually move. Ghosting pins a joint's
    # limits to a ~0 range rather than deleting it, so the joint list alone
    # over-counts; the range is what distinguishes them, and it reproduces
    # spec.n_active_joints exactly.
    # An ACTIVE joint can actually move; ghosting pins limits to a ~0 range
    # rather than deleting the joint, so the joint list alone over-counts.
    #
    # AA and FE at the same knuckle are EXACTLY co-located -- same child origin
    # to the last decimal -- so plotting one dot per joint silently draws 12
    # joints as 8 dots. Group by position instead and mark the knuckle once,
    # ringed when it also abducts. The dot count then matches the "Nj" label.
    knuckles: dict = {}
    for joint in urdf.robot.joints:
        if joint.type == "fixed" or not joint.child.startswith("gen_"):
            continue
        lim = joint.limit
        if lim is None or lim.lower is None or lim.upper is None:
            continue
        if (lim.upper - lim.lower) <= 1e-4:
            continue
        pos = (palm_inv @ urdf.get_transform(joint.child))[:3, 3]
        key = tuple(np.round(pos, 6))
        entry = knuckles.setdefault(key, {"pos": pos, "aa": False, "fe": False,
                                          "n": 0})
        entry["n"] += 1
        # Both CMC and MCP come in FE/AA pairs; PIP and DIP are flexion only.
        # Either half can be ghosted independently, so a knuckle can end up with
        # abduction and no flexion.
        if joint.name.endswith("_AA"):
            entry["aa"] = True
        else:
            entry["fe"] = True

    meshes = []
    for name, link in urdf.link_map.items():
        if not name.startswith("gen_"):
            continue
        for coll in link.collisions:
            mesh = _geometry_to_mesh(coll.geometry, path.parent, hull=False)
            if mesh is None:
                continue
            mesh = mesh.copy()
            origin = coll.origin if coll.origin is not None else np.eye(4)
            mesh.apply_transform(palm_inv @ urdf.get_transform(name) @ origin)
            meshes.append((name == "gen_palm", np.asarray(mesh.vertices)))
    if not meshes:
        return None

    allpts = np.vstack([v for _, v in meshes])
    centre = allpts.mean(axis=0)
    # Right-singular vectors, ordered by decreasing spread. u[0] is the長 axis
    # (fingers), u[1] the width, u[2] the flat direction we look along.
    _, _, vt = np.linalg.svd(allpts - centre, full_matrices=True)
    up, across = vt[0], vt[1]
    # Fingers should point UP the page, not down: orient by where the non-palm
    # geometry sits relative to the palm.
    finger_pts = [v for is_palm, v in meshes if not is_palm]
    if finger_pts:
        offset = np.vstack(finger_pts).mean(axis=0) - centre
        if float(offset @ up) < 0:
            up = -up
    basis = np.stack([across, up], axis=1)      # (3, 2)

    out = []
    for is_palm, verts in meshes:
        pts = (verts - centre) @ basis
        if len(pts) < 3:
            continue
        try:
            out.append((is_palm, pts[ConvexHull(pts).vertices]))
        except Exception:                        # degenerate (edge-on) link
            continue
    if not out:
        return None
    def project(entries):
        if not entries:
            return np.empty((0, 2))
        return (np.array([e["pos"] for e in entries]) - centre) @ basis

    vals = list(knuckles.values())
    return {
        "hulls": out,
        "fe_only": project([e for e in vals if e["fe"] and not e["aa"]]),
        "fe_and_aa": project([e for e in vals if e["fe"] and e["aa"]]),
        "aa_only": project([e for e in vals if e["aa"] and not e["fe"]]),
        "n_active": sum(e["n"] for e in vals),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", required=True, help="A population_eval output dir")
    p.add_argument("--out", default="figures/best_worst_embodiments.png")
    p.add_argument("--n", type=int, default=5)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    results = Path(args.results)
    rows = load_rows(results)
    z = object_controlled_z(rows)
    best, worst = pick(rows, args.n)
    order = best + worst
    labels = ["best"] * len(best) + ["worst"] * len(worst)

    hulls = {i: hand_hulls(rows[i]["design"]) for i in order}
    missing = [rows[i]["design"] for i in order if hulls[i] is None]
    if missing:
        print(f"[plot] no hand geometry for {missing}")
    # One half-extent for all ten, so a hand that reaches twice as far is drawn
    # twice as large. Per-hand autoscaling would make every hand look the same
    # size, which is the opposite of what this figure is for.
    half = max((np.abs(np.vstack([h for _, h in hs["hulls"]])).max()
                for hs in hulls.values() if hs), default=0.05) * 1.06
    print(f"[plot] hand pictograms at shared half-extent {half * 100:.1f} cm")

    goals = np.array([float(rows[i]["mean_goals"]) for i in order])

    fig = plt.figure(figsize=(13.2, 7.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0.075, 0.40, 0.90, 0.44])
    ax.set_facecolor(SURFACE)

    x = np.arange(len(order), dtype=float)
    x[len(best):] += 0.7          # a gap the eye reads as two groups, not colour
    colors = [COLOR_BEST if lab == "best" else COLOR_WORST for lab in labels]

    bars = ax.bar(x, goals, width=0.74, color=colors, zorder=3, linewidth=0)
    for bar in bars:
        bar.set_joinstyle("round")
    # A 0.00 design draws no bar, which reads as missing data and leaves the
    # legend's second swatch matching nothing on the canvas. Draw the zero as a
    # rule at the baseline: not a short bar, the actual value.
    for xi, g, c in zip(x, goals, colors):
        if g <= 1e-9:
            ax.plot([xi - 0.37, xi + 0.37], [0, 0], color=c, linewidth=3.4,
                    solid_capstyle="round", zorder=4)

    for xi, g in zip(x, goals):
        ax.text(xi, g + 0.16, f"{g:.2f}", ha="center", va="bottom", fontsize=12.5,
                color=INK_PRIMARY, fontweight="bold", zorder=5)

    ax.set_ylim(0, goals.max() * 1.20)
    ax.set_xlim(x[0] - 0.72, x[-1] + 0.72)
    ax.set_ylabel("mean goals per episode", fontsize=10.5, color=INK_SECONDARY,
                  labelpad=8)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.grid(axis="y", color="#e8e7e2", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#dcdbd5")
    ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9.5, length=0)
    ax.set_xticks([])

    # Group headers: identity carried by position and words, not colour alone.
    for span, text, colour in ((best, f"BEST {args.n}", COLOR_BEST),
                               (worst, f"WORST {args.n}   all 0.00 goals",
                                COLOR_WORST)):
        idx = [order.index(i) for i in span]
        ax.text(float(np.mean(x[idx])), goals.max() * 1.13, text, ha="center",
                fontsize=10.5, color=colour, fontweight="bold")

    fig.text(0.075, 0.945, "Best and worst embodiments of 24,576",
             fontsize=17, color=INK_PRIMARY, fontweight="bold", ha="left")
    fig.text(0.075, 0.902,
             "10 episodes each at 3 cm success tolerance. Hands drawn top-down "
             "onto the palm at home pose, one shared scale.",
             fontsize=10, color=INK_SECONDARY, ha="left")

    # Icons as real axes, positioned under each bar in FIGURE coordinates.
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    icon_w, icon_h, icon_y = 0.088, 0.215, 0.145
    for xi, i in zip(x, order):
        fx = inv.transform(ax.transData.transform((xi, 0)))[0]
        if hulls[i]:
            iax = fig.add_axes([fx - icon_w / 2, icon_y, icon_w, icon_h])
            draw_hand(iax, hulls[i], half)
        r = rows[i]
        fig.text(fx, icon_y - 0.030, r["design"].replace("gen_0003_", "#"),
                 ha="center", fontsize=10, color=INK_PRIMARY, fontweight="bold")
        fig.text(fx, icon_y - 0.068,
                 f"{r['n_active_fingers']}f / {r['n_active_joints']}j",
                 ha="center", fontsize=9.5, color=INK_SECONDARY)

    fig.legend(handles=[Patch(facecolor=PALM_FACE, edgecolor=PALM_EDGE, label="palm"),
                        Patch(facecolor=FINGER_FACE, edgecolor=FINGER_EDGE,
                              label="finger links"),
                        Line2D([], [], marker="o", linestyle="none",
                               markerfacecolor=JOINT_FACE, markeredgecolor=JOINT_FACE,
                               markersize=6, label="flexion"),
                        Line2D([], [], marker="o", linestyle="none",
                               markerfacecolor=JOINT_FACE, markeredgecolor=JOINT_RING,
                               markeredgewidth=1.6, markersize=7,
                               label="flexion + abduction"),
                        Line2D([], [], marker="o", linestyle="none",
                               markerfacecolor=JOINT_RING, markeredgecolor=JOINT_FACE,
                               markeredgewidth=1.1, markersize=6,
                               label="abduction only")],
               loc="lower right", bbox_to_anchor=(0.978, 0.012), ncol=5,
               frameon=False, fontsize=9, labelcolor=INK_MUTED)
    fig.text(0.075, 0.022, f"scale: each icon spans {2 * half * 100:.0f} cm",
             fontsize=9, color=INK_MUTED, ha="left")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] wrote {out}")

    csv_out = out.with_suffix(".csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "design", "mean_goals", "std_goals", "z_within_object",
                    "n_active_fingers", "n_active_joints", "object_type", "pool_index"])
        for lab, i in zip(labels, order):
            r = rows[i]
            w.writerow([lab, r["design"], r["mean_goals"], r["std_goals"],
                        f"{z[i]:+.3f}", r["n_active_fingers"], r["n_active_joints"],
                        r["object_type"], r["pool_index"]])
    print(f"[plot] wrote {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
