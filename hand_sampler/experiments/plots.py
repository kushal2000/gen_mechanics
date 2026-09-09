"""Figures from one or more runs of ``experiments.run``."""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import matplotlib.patheffects as pe                                # noqa: E402

from hand_sampler import design_space                                  # noqa: E402

# Validated categorical slots 1-3 (safe for all pairs), plus chart chrome.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, GRID, AXIS, INK, MUTED = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#0b0b0b", "#898781"
ORD = ("#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b")

JOINT_FLOOR = design_space.MIN_FINGERS
JOINT_CEIL = design_space.MAX_FINGERS * design_space.MAX_JOINTS_PER_FINGER


# --- loading ----------------------------------------------------------------

def load(path: str) -> list[dict]:
    """A run directory, or a .jsonl / .json file."""
    if os.path.isdir(path):
        path = os.path.join(path, "stats.jsonl")
    text = open(path).read().strip()
    if text.startswith("["):
        return json.loads(text)                    # legacy single-array output
    return [json.loads(line) for line in text.splitlines() if line]


def load_all(patterns: list[str]) -> list[list[dict]]:
    paths = sorted({p for pat in patterns for p in (glob.glob(pat) or [pat])})
    if not paths:
        raise SystemExit(f"no runs matched {patterns}")
    return [load(p) for p in paths]


# --- drawing ----------------------------------------------------------------

def frame(ax, title, sub="", ylab=""):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=13 if sub else 6)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED, fontsize=7.5)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("generation", color=MUTED, fontsize=8)
    if ylab:
        ax.set_ylabel(ylab, color=MUTED, fontsize=8)


def summarise(runs, f, n):
    """(median, min, max) across runs at each of the first n generations."""
    med, lo, hi = [], [], []
    for i in range(n):
        v = sorted(f(r[i]) for r in runs)
        med.append(v[len(v) // 2]); lo.append(v[0]); hi.append(v[-1])
    return med, lo, hi


def band(ax, g, runs, f, colour, label=None, at=0.5, dy=0.0):
    med, lo, hi = summarise(runs, f, len(g))
    if len(runs) > 1:
        ax.fill_between(g, lo, hi, color=colour, alpha=0.16, linewidth=0)
    ax.plot(g, med, color=colour, linewidth=1.9)
    if label:                                       # direct label: also the relief
        j = int(at * len(g))                        # some slots fail 3:1 on light
        ax.text(g[j], med[j] + dy, label, color=colour, fontsize=8, ha="center",
                path_effects=[pe.withStroke(linewidth=2.5, foreground=SURFACE)])
    return med


def unavailable(ax, field):
    """A figure is newer than the run it is drawing."""
    ax.text(0.5, 0.5, f"{field}\nnot recorded in these runs", color=MUTED,
            transform=ax.transAxes, fontsize=8.5, ha="center", va="center")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_xlabel(""); ax.set_ylabel("")


def recorded(runs, field) -> bool:
    return all(field in r[0] for r in runs)


def ref(ax, y, text, g, above=True):
    ax.axhline(y, color=AXIS, linewidth=1.1, linestyle=(0, (4, 3)))
    ax.text(g[-1], y, " " + text, color=MUTED, fontsize=7.5,
            va="bottom" if above else "top", ha="right")


def finish(fig, out, title):
    fig.suptitle(title, color=INK, fontsize=12, x=0.006, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"wrote {out}")


def new_fig(rows, cols, w, h):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h), dpi=130)
    fig.patch.set_facecolor("#f9f9f7")
    return fig, axes


def share(row, key, ks):
    h = row[key]
    return sum(h.get(str(k), 0) for k in ks) / sum(h.values())


# --- figures ----------------------------------------------------------------

def fig_single(runs, out):
    r = runs[0]
    g = [x["gen"] for x in r]
    fig, ax = new_fig(2, 3, 15, 7.6)

    frame(ax[0][0], "Mean fingers per hand", ylab="fingers")
    band(ax[0][0], g, [r], lambda x: x["n_fingers"], BLUE)
    ref(ax[0][0], design_space.MAX_FINGERS, "MAX_FINGERS", g, above=False)

    frame(ax[0][1], "Mean joints per hand", "band: p10-p90 of the population")
    ax[0][1].fill_between(g, [x["joints_p10"] for x in r], [x["joints_p90"] for x in r],
                          color=BLUE, alpha=0.16, linewidth=0)
    band(ax[0][1], g, [r], lambda x: x["n_joints"], BLUE)

    frame(ax[0][2], "Mean link length (mm)")
    band(ax[0][2], g, [r], lambda x: x["link_mm"], BLUE)
    ref(ax[0][2], 1000 * design_space.MIN_LINK_LENGTH, "MIN_LINK_LENGTH", g)
    ref(ax[0][2], 1000 * design_space.MAX_LINK_LENGTH, "MAX_LINK_LENGTH", g, above=False)

    a = ax[1][0]
    frame(a, "Finger-count composition", "share of population", "share")
    ks = sorted({int(k) for x in r for k in x["fingers_hist"]})
    w = 9

    def smooth(y):                       # raw, a per-band edge reads as hatching
        return [sum(y[max(0, i - w // 2):i + w // 2 + 1])
                / len(y[max(0, i - w // 2):i + w // 2 + 1]) for i in range(len(y))]

    a.stackplot(g, *[smooth([share(x, "fingers_hist", [k]) for x in r]) for k in ks],
                colors=ORD[:len(ks)], linewidth=0)
    a.set_ylim(0, 1); a.set_xlim(g[0], g[-1])
    for k, c in zip(ks, ORD):
        a.plot([], [], color=c, linewidth=6, label=f"{k}f")
    a.legend(frameon=False, fontsize=7.5, labelcolor=MUTED, ncol=len(ks),
             loc="upper center", bbox_to_anchor=(0.5, -0.22), handlelength=1.2,
             columnspacing=1.2)

    a = ax[1][1]
    frame(a, "Structural operator success rate",
          "zero = pinned against a constraint, not an equilibrium", "rate")
    for (op, at, dy), c in zip((("split_link", 0.30, +0.05), ("merge_links", 0.72, -0.10),
                                ("add_finger", 0.44, +0.05), ("remove_finger", 0.60, -0.10)),
                               (BLUE, ORANGE, AQUA, "#eda100")):
        band(a, g, [r], lambda x, o=op: x["rates"][o], c, op, at, dy)
    a.set_ylim(0, 1.09); a.set_xlim(g[0], g[-1])

    frame(ax[1][2], "Null-mutation rate", "child identical to parent", "rate")
    band(ax[1][2], g, [r], lambda x: x["null_rate"], BLUE)
    ax[1][2].set_ylim(bottom=0)
    finish(fig, out, "One run: population statistics against generation")


def fig_envelope(runs, out):
    n = min(len(r) for r in runs)
    g = [runs[0][i]["gen"] for i in range(n)]
    sub = f"median of {len(runs)} runs; band = min-max across runs"
    fig, ax = new_fig(2, 3, 15, 7.6)

    frame(ax[0][0], "Mean fingers per hand", sub, "fingers")
    band(ax[0][0], g, runs, lambda x: x["n_fingers"], BLUE)
    ref(ax[0][0], design_space.MAX_FINGERS, "MAX_FINGERS", g, above=False)

    frame(ax[0][1], "Mean joints per hand", sub, "joints")
    band(ax[0][1], g, runs, lambda x: x["n_joints"], BLUE)

    frame(ax[0][2], "Mean link length (mm)", sub)
    band(ax[0][2], g, runs, lambda x: x["link_mm"], BLUE)
    ref(ax[0][2], 1000 * design_space.MIN_LINK_LENGTH, "MIN_LINK_LENGTH", g)
    ref(ax[0][2], 1000 * design_space.MAX_LINK_LENGTH, "MAX_LINK_LENGTH", g, above=False)

    frame(ax[1][0], "Topology diversity", "distinct skeletons / population", "share")
    if recorded(runs, "topology_diversity"):
        band(ax[1][0], g, runs, lambda x: x["topology_diversity"], BLUE)
        ax[1][0].set_ylim(0, 1)
    else:
        unavailable(ax[1][0], "topology_diversity")

    a = ax[1][1]
    frame(a, "add_finger vs remove_finger success",
          "add at zero = palm full: a wall, not an equilibrium", "rate")
    band(a, g, runs, lambda x: x["rates"]["add_finger"], BLUE, "add_finger", 0.42, +0.06)
    band(a, g, runs, lambda x: x["rates"]["remove_finger"], ORANGE,
         "remove_finger", 0.66, -0.10)
    a.set_ylim(0, 1.12); a.set_xlim(g[0], g[-1])

    frame(ax[1][2], "Null-mutation rate", "child identical to parent", "rate")
    band(ax[1][2], g, runs, lambda x: x["null_rate"], BLUE)
    ax[1][2].set_ylim(bottom=0)
    finish(fig, out, "Across replicates: is it equilibrated, or still wandering?")


def fig_starvation(runs, out):
    n = min(len(r) for r in runs)
    g = [runs[0][i]["gen"] for i in range(n)]
    fig, (a1, a2) = new_fig(1, 2, 13.5, 4.9)

    frame(a1, "The sampled joint range slides off the floor",
          f"p10-p90 of the population; median of {len(runs)} runs", "joints per hand")
    p10 = summarise(runs, lambda x: x["joints_p10"], n)[0]
    p90 = summarise(runs, lambda x: x["joints_p90"], n)[0]
    a1.fill_between(g, p10, p90, color=BLUE, alpha=0.16, linewidth=0)
    a1.plot(g, p90, color=BLUE, linewidth=1.4, alpha=0.55)
    a1.plot(g, p10, color=BLUE, linewidth=2.0)
    j = int(0.30 * n)
    for y, lab, al in ((p10, "p10", 1.0), (p90, "p90", 0.7)):
        a1.text(g[j], y[j] + 0.9, lab, color=BLUE, alpha=al, fontsize=8, ha="center",
                path_effects=[pe.withStroke(linewidth=2.5, foreground=SURFACE)])
    a1.axhline(JOINT_FLOOR, color=ORANGE, linewidth=1.2, linestyle=(0, (4, 3)))
    # short, because on a run whose p10 is still at the floor a long label lies straight across...
    a1.text(g[-1], JOINT_FLOOR, f"floor = {JOINT_FLOOR} motors ", color=ORANGE,
            fontsize=8, ha="right", va="bottom")

    frame(a2, "The low-motor population empties out",
          "band = min-max across runs", "share of population")
    band(a2, g, runs, lambda x: share(x, "fingers_hist", [design_space.MIN_FINGERS]), BLUE,
         f"exactly {design_space.MIN_FINGERS} fingers", 0.30, +0.05)
    band(a2, g, runs, lambda x: share(x, "fingers_hist",
                                      [design_space.MIN_FINGERS, design_space.MIN_FINGERS + 1]), ORANGE,
         f"{design_space.MIN_FINGERS} or {design_space.MIN_FINGERS + 1} fingers", 0.55, +0.05)
    if recorded(runs, "joints_hist"):        # the motor axis of the headline plot
        lo = list(range(JOINT_FLOOR, JOINT_FLOOR + 3))
        band(a2, g, runs, lambda x: share(x, "joints_hist", lo), AQUA,
             f"{lo[0]}-{lo[-1]} motors", 0.80, +0.05)
    a2.set_ylim(0, 1)
    finish(fig, out, "Does the cheap end of the design space keep getting sampled?")


def fig_arms(null, mx, mn, out):
    arms = [("max_joints", mx, ORANGE, 0.30), ("random (null)", null, BLUE, 0.62),
            ("min_joints", mn, AQUA, 0.45)]
    arms = [(l, r, c, a) for l, r, c, a in arms if r]
    n = min(len(r) for _, rs, _, _ in arms for r in rs)
    g = [arms[0][1][0][i]["gen"] for i in range(n)]

    # Computed, never asserted in a literal: this number is a property of the data being plotted,...
    k = min(50, n - 1)
    rate = lambda rs: sum((r[k]["n_joints"] - r[0]["n_joints"]) / k for r in rs) / len(rs)
    ratio = abs(rate(mx) / rate(null)) if null and mx and rate(null) else None
    head = ("Selection moves n_joints "
            + (f"~{ratio:.0f}x faster than drift" if ratio else "against drift"))

    fig, (a1, a2) = new_fig(1, 2, 13.5, 4.9)
    frame(a1, head, f"median of runs; band = min-max. First {k} generations set the rate",
          "mean joints per hand")
    for lab, runs, c, at in arms:
        band(a1, g, runs, lambda x: x["n_joints"], c, lab, at, +2.0)
    ref(a1, JOINT_CEIL, f"{JOINT_CEIL} = MAX_FINGERS x MAX_JOINTS_PER_FINGER", g, False)
    ref(a1, JOINT_FLOOR, f"{JOINT_FLOOR} = the floor", g)
    a1.set_ylim(0, JOINT_CEIL * 1.1)

    frame(a2, "and it holds the cheap end that drift vacates",
          f"share of hands with exactly {design_space.MIN_FINGERS} fingers", "share of population")
    for lab, runs, c, at in arms:
        band(a2, g, runs, lambda x: share(x, "fingers_hist", [design_space.MIN_FINGERS]),
             c, lab, at, +0.05)
    a2.set_ylim(0, 1)
    finish(fig, out, "Selection strength against the grammar's drift, at matched truncation")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figure", choices=("single", "envelope", "starvation", "arms"))
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs or stats files")
    ap.add_argument("--max", nargs="+", default=[], help="arms: max_joints runs")
    ap.add_argument("--min", nargs="+", default=[], help="arms: min_joints runs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = load_all(args.runs)
    if args.figure == "single":
        fig_single(runs, args.out)
    elif args.figure == "envelope":
        fig_envelope(runs, args.out)
    elif args.figure == "starvation":
        fig_starvation(runs, args.out)
    else:
        fig_arms(runs, load_all(args.max) if args.max else [],
                 load_all(args.min) if args.min else [], args.out)


if __name__ == "__main__":
    main()
