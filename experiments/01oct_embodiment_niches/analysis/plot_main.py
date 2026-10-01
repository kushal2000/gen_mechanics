"""The three main phase-1 figures (the per-metric bar charts live in plots/details/).

    .venv_isaacsim/bin/python experiments/01oct_embodiment_niches/analysis/plot_main.py

  niche_map.png       hands x metrics, cell = the hand's rank on that metric (1 = best, darker)
  speed_vs_energy.png goals/min against joint work per goal; non-dominated hands joined
  robustness.png      hands x perturbations, cell = goals/min relative to the hand's own nominal
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from plot_bars import BAR, BEST, CONDS, GRID, INK, INK2, METRICS, NAMES, SURFACE  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3]
D = REPO / "debug_outputs/embodiment_niches"
OUT = pathlib.Path(__file__).resolve().parent.parent / "plots"
# Sequential blue ramp from the reference palette, light (worse) -> dark (better).
RAMP = LinearSegmentedColormap.from_list("blue", ["#f0f6fe", "#cde2fb", "#86b6ef", "#3987e5",
                                                  "#256abf", "#184f95", "#0d366b"])


def _cell_text(ax, x, y, s, dark):
    ax.text(x, y, s, ha="center", va="center", fontsize=9, color="#ffffff" if dark else INK)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


# Niche-map columns: each is the hand's mean rank over its component metrics, re-ranked.
# Cube motion is PER GOAL, so a slow policy does not read as a gentle one.
def _per_goal(key):
    return lambda d: d[key] * 60.0 / d["goals_per_min"]


def _robust(R, h):
    import numpy as np
    return float(np.mean([R[f"{h}__{c}"]["goals_per_min"] / R[f"{h}__nominal"]["goals_per_min"] for c in CONDS]))


# One metric per column: (column, metric and unit, [(getter, higher is better, value format)]).
COMPOSITES = [
    ("Speed", "goals / min", [(lambda d: d["goals_per_min"], True, "{:.0f}")]),
    # From the 5-minute nominal runs; the literal minimum is 0 for 8 of 9 hands.
    ("Reliability", "goals before drop,\nworst 5% of episodes", [(lambda d: d["_gbd"]["p5"], True, "{:.0f}")]),
    ("Energy", "joint work per goal (J)", [(lambda d: d["work_per_goal_J"], False, "{:.1f}")]),
    ("Motion economy", "path efficiency\n(goal angle / cube rotation)",
     [(lambda d: d["path_efficiency"], True, "{:.2f}")]),
    ("Smooth control", "action rate\n(rad/s per joint)", [(lambda d: d["target_rate_rad_s_per_joint"], False, "{:.0f}")]),
    ("Calm cube", "cube travel per goal (m)", [(_per_goal("obj_speed_mean"), False, "{:.2f}")]),
]


def _rank(vals, higher):
    import numpy as np
    v = np.asarray(vals, dtype=float)
    order = np.argsort(-v if higher else v, kind="stable")
    r = np.empty(len(v))
    r[order] = np.arange(1, len(v) + 1)
    return r


def niche_map(R, hands):
    import numpy as np
    for h in hands:
        R[f"{h}__nominal"]["_gbd"] = json.loads((D / f"{h}__nominal_5min.json").read_text())["goals_before_drop"]
    cols, ranks, labels = [], [], []
    for name, sub, comps in COMPOSITES:
        raw = [[g(R[f"{h}__nominal"]) for h in hands] for g, _, _ in comps]
        parts = [_rank(v, hi) for v, (_, hi, _) in zip(raw, comps)]
        ranks.append(_rank(np.mean(parts, axis=0), False))
        labels.append([", ".join(f.format(v[i]) for v, (_, _, f) in zip(raw, comps)) for i in range(len(hands))])
        cols.append((name, sub))
    ranks = np.stack(ranks, axis=1)
    fig, ax = plt.subplots(figsize=(11.5, 6.4), facecolor=SURFACE)
    ax.imshow((ranks == 1).astype(float), cmap=LinearSegmentedColormap.from_list("hl", ["#f3f2ef", BEST]),
              vmin=0, vmax=1, aspect="auto")
    for i in range(len(hands)):
        for j in range(len(cols)):
            win = ranks[i, j] == 1
            ax.text(j, i - 0.13, f"{int(ranks[i, j])}", ha="center", va="center", fontsize=11 if win else 10,
                    color="#ffffff" if win else INK, fontweight="semibold" if win else "normal")
            ax.text(j, i + 0.2, f"({labels[j][i]})", ha="center", va="center", fontsize=8,
                    color="#dbe8f8" if win else INK2)
    ax.set_xticks(range(len(cols)), [n for n, _ in cols], fontsize=10, color=INK, fontweight="semibold")
    ax.xaxis.tick_top()
    for j, (_, sub) in enumerate(cols):
        ax.text(j, len(hands) - 0.35, sub, ha="center", va="top", fontsize=8, color=INK2)
    ax.set_yticks(range(len(hands)), [NAMES[h] for h in hands], fontsize=10, color=INK)
    ax.set_xticks([x - 0.5 for x in range(1, len(cols))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(hands))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    _style(ax)
    ax.set_title("Niche map: who is best at what", loc="left", fontsize=13, color=INK,
                 fontweight="semibold", pad=40)
    ax.text(0, 1.075, "rank of each hand on one metric per column (1 = best, filled) · value in brackets · "
            "nominal conditions", transform=ax.transAxes,
            fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "niche_map.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def speed_vs_energy(R, hands, front):
    xs = {h: R[f"{h}__nominal"]["work_per_goal_J"] for h in hands}
    ys = {h: R[f"{h}__nominal"]["goals_per_min"] for h in hands}
    fig, ax = plt.subplots(figsize=(8.5, 6), facecolor=SURFACE)
    f = sorted(front, key=lambda h: xs[h])
    ax.plot([xs[h] for h in f], [ys[h] for h in f], color=BEST, linewidth=2, zorder=1)
    for h in hands:
        on = h in front
        ax.scatter(xs[h], ys[h], s=90, color=BEST if on else BAR, edgecolor=SURFACE, linewidth=2, zorder=2)
        ax.annotate(NAMES[h], (xs[h], ys[h]), xytext=(8, 6), textcoords="offset points",
                    fontsize=10, color=INK if on else INK2)
    ax.set_xlabel("joint work per goal (J) · lower is better", fontsize=10, color=INK2)
    ax.set_ylabel("goals per minute · higher is better", fontsize=10, color=INK2)
    ax.set_xlim(0, max(xs.values()) * 1.15)
    ax.set_ylim(0, max(ys.values()) * 1.12)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_title("Speed vs energy: 5 of 9 hands are non-dominated", loc="left", fontsize=13,
                 color=INK, fontweight="semibold", pad=26)
    ax.text(0, 1.015, "dark dots joined = no other hand is both faster and cheaper · nominal conditions",
            transform=ax.transAxes, fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "speed_vs_energy.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def robustness(R, hands):
    import numpy as np
    cs = list(CONDS)
    rel = np.array([[R[f"{h}__{c}"]["goals_per_min"] / R[f"{h}__nominal"]["goals_per_min"] for c in cs]
                    for h in hands])
    ab = np.array([[R[f"{h}__{c}"]["goals_per_min"] for c in cs] for h in hands])
    fig, ax = plt.subplots(figsize=(12, 6.2), facecolor=SURFACE)
    ax.imshow(rel, cmap=RAMP, vmin=0, vmax=1, aspect="auto")
    for i in range(len(hands)):
        for j in range(len(cs)):
            _cell_text(ax, j, i, f"{100 * rel[i, j]:.0f}%\n{ab[i, j]:.0f}/min" if ab[i, j] >= 1
                       else f"{100 * rel[i, j]:.0f}%\n{ab[i, j]:.1f}/min", rel[i, j] >= 0.55)
    best = {hands[i] for i in ab.argmax(0)}
    ax.set_xticks(range(len(cs)), [CONDS[c] for c in cs], rotation=25, ha="right", fontsize=9, color=INK)
    ax.set_yticks(range(len(hands)), [NAMES[h] for h in hands], fontsize=10, color=INK)
    ax.set_xticks([x - 0.5 for x in range(1, len(cs))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(hands))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    _style(ax)
    ax.set_title("Robustness: throughput kept under each perturbation", loc="left", fontsize=13,
                 color=INK, fontweight="semibold", pad=26)
    lead = (f"{NAMES[next(iter(best))]} has the highest absolute rate in every column" if len(best) == 1
            else "highest absolute rate: " + ", ".join(NAMES[h] for h in best))
    ax.text(0, 1.015, f"cell = % of the hand's own nominal goals/min, with the absolute rate · {lead}",
            transform=ax.transAxes, fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "robustness.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    s = json.loads((D / "summary.json").read_text())
    R, hands = s["raw"], s["hands"]
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    niche_map(R, hands)
    speed_vs_energy(R, hands, s["fronts"]["goals_per_min|work_per_goal"])
    robustness(R, hands)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
