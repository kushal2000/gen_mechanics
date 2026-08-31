"""Fitness against iteration for the training-free design loop.

    .venv_isaacsim/bin/python -m coevolution.loop.plot_loop_fitness

Reads results/loop/<arm>/summary.json and writes into
figures/frozen_controller_design_evolution/:

    loop_fitness_<arm>_top4k.png   the selected half's mean, per arm
    loop_fitness_<arm>_top1.png    the single best design, per arm
    loop_fitness_by_iteration.png  both arms, both aggregations, 2x2 combined

UNITS. Scores are stored as goals inside a 6,000-step budget. At 60 Hz policy
control that is 100 s, so the plots divide by 10 and report throughput in goals
per 10 s -- a rate the reader can hold, rather than a number that only means
something once you know the budget.

WHY EACH ARM GETS ITS OWN FIGURE rather than one axis with four lines: the two
arms are scored under DIFFERENT conditions (DR off vs disturbance at p=0.3), so
their heights are not comparable and putting them on one axis invites reading
"blue is above orange" as "nominal designs are better". Only the SHAPES compare.
The combined figure keeps them in separate panels for the same reason.

THE TOP-1 PANELS CARRY A WARNING and should not be read for level. The best
score is the maximum of 24,576 noisy measurements: two identical replicate runs
of one population share only ~1.3% of their top 10, so a dip between iterations
is a different draw from the tail, not the search regressing. The top-4,096 mean
pools enough designs for the noise to average out, which is why those curves are
smooth and these are not.
"""

from __future__ import annotations

import json
from pathlib import Path

from hand_sampler.paths import resolve as resolve_repo_path

PER_10S = 10.0                      # 6,000 steps @ 60 Hz = 100 s
OUT_DIR = "figures/frozen_controller_design_evolution"
SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"

# Colour follows the ENTITY (the arm), so it is identical in every figure here.
# Slots 1-2 of the reference categorical palette, documented as clearing the
# all-pairs CVD gates in both modes; markers differ too, so identity never rests
# on colour alone.
ARMS = {
    "nominal": ("#2a78d6", "o", "Nominal fitness", "DR off"),
    "wrench": ("#eb6834", "s", "Wrench fitness", "disturbance, p=0.3"),
}
AGGS = {
    "elite_mean": ("top 4,096 mean", "the selected half of each generation", "top4k",
                   "Throughput = goals in a 6,000-step budget (100 s at 60 Hz), / 10."),
    "best": ("single best design", "max of 24,576 scores", "top1",
             "Throughput = goals in a 6,000-step budget (100 s at 60 Hz), / 10.  "
             "NOTE: a noise maximum --\ntwo identical replicate runs share only ~1.3% "
             "of their top 10, so the LEVEL here is unreliable."),
}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color="#e6e5e0", lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d5d4cf")
    ax.tick_params(colors=INK2, labelsize=9.5, length=0)
    ax.set_xlabel("evolution iteration", fontsize=9.5, color=INK2)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = resolve_repo_path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    data = {a: json.loads((resolve_repo_path(f"results/loop/{a}/summary.json")).read_text())
            for a in ARMS}
    n_iter = max(len(v) for v in data.values())

    def draw(ax, arm, key, col, mk):
        it = [r["iteration"] for r in data[arm]]
        y = [r[key] / PER_10S for r in data[arm]]
        ax.plot(it, y, color=col, lw=2, marker=mk, ms=6, mec=SURFACE, mew=1.4,
                zorder=3, clip_on=False, label=ARMS[arm][2].split()[0].lower())
        ax.annotate(f"{y[-1]:.2f}", (it[-1], y[-1]), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=10, color=INK2)
        ax.set_xticks(range(n_iter))
        ax.set_xlim(-0.35, n_iter - 0.1)
        ax.margins(y=0.16)
        return y

    # --- one figure per (arm, aggregation) --------------------------------
    for arm, (col, mk, arm_t, arm_s) in ARMS.items():
        for key, (agg_t, agg_s, slug, note) in AGGS.items():
            fig, ax = plt.subplots(figsize=(6.4, 4.3), facecolor=SURFACE)
            _style(ax)
            y = draw(ax, arm, key, col, mk)
            # A single series needs no legend: the title names it.
            ax.set_title(f"{arm_t} -- {agg_t}", fontsize=12, color=INK, pad=26,
                         loc="left", fontweight="medium")
            ax.text(0, 1.012, f"{arm_s}; {agg_s}", transform=ax.transAxes,
                    fontsize=8.8, color=INK3, va="bottom")
            ax.set_ylabel("throughput (goals hit per 10 s)", fontsize=9.5, color=INK2)
            fig.text(0.012, 0.015, note, fontsize=7.8, color=INK3, va="bottom")
            fig.tight_layout(rect=[0, 0.10 if key == "elite_mean" else 0.135, 1, 1])
            p = out / f"loop_fitness_{arm}_{slug}.png"
            fig.savefig(p, dpi=200, facecolor=SURFACE)
            plt.close(fig)
            print(f"{str(p):72s} {y[0]:.2f} -> {y[-1]:.2f}")

    # --- combined 2x2 ------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), facecolor=SURFACE)
    for r, (key, (agg_t, agg_s, _slug, _note)) in enumerate(AGGS.items()):
        for c, (arm, (col, mk, arm_t, arm_s)) in enumerate(ARMS.items()):
            ax = axes[r][c]
            _style(ax)
            draw(ax, arm, key, col, mk)
            ax.set_title(f"{arm_t} -- {agg_t}", fontsize=11, color=INK, pad=24,
                         loc="left", fontweight="medium")
            ax.text(0, 1.012, f"{arm_s}; {agg_s}", transform=ax.transAxes,
                    fontsize=8.4, color=INK3, va="bottom")
            if c == 0:
                ax.set_ylabel("throughput (goals hit per 10 s)", fontsize=9.5, color=INK2)
    fig.suptitle("Training-free design loop: fitness by iteration", fontsize=13,
                 color=INK, x=0.008, ha="left", y=0.99, fontweight="medium")
    fig.text(0.008, 0.012,
             "Each arm is scored under its OWN condition, so heights are not comparable across columns -- compare the SHAPES. "
             "Top-1 (bottom row) is a\nnoise maximum: two identical replicate runs share only ~1.3% of their top 10, so its level "
             "is unreliable even where the trend is not.",
             fontsize=8.2, color=INK3, va="bottom")
    fig.tight_layout(rect=[0, 0.055, 1, 0.965])
    p = out / "loop_fitness_by_iteration.png"
    fig.savefig(p, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"{str(p):72s} combined 2x2")


if __name__ == "__main__":
    main()
