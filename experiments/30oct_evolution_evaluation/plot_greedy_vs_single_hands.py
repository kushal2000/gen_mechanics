"""Greedy evolution (baseline) against RL on one hand: a random hand and the Claude designer's picks. Training goals per
episode against training epochs, the same 8B-step budget.

    .venv_isaacsim/bin/python experiments/30oct_evolution_evaluation/plot_greedy_vs_single_hands.py

Single-hand runs: experiments/29oct_evolution_from_scratch/long_*.sub, one continuous run each (the random draw
seed_f5_cut_j1; Claude's pick seed_f6_nocut_j2 and runner-up seed_f3_cut_j2), trainer's ``episode_final/successes``.
Greedy evolution: its iterations stitched (plot_training_curves.py), two lines -- the population mean (what the
population-level training metric shows) and the best hand in the population at each point (each hand's
``per_hand_successes`` 50-epoch rolling mean, the max over hands; a single hand is what the other runs train). All
lines 50-epoch rolling means; markers at the first epoch each reaches 1 goal per episode.
Writes experiments/30oct_evolution_evaluation/plots/greedy_vs_single_hands.png.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_training_curves import (FRAMES_PER_EPOCH, INK, INK2, MAX_TRIM, RECOVER, SMOOTH,  # noqa: E402
                                  SURFACE, YMIN, _read, curve, first_at, iterations, rolling, stitch, style)

LONG = Path("debug_outputs/train_logs/29oct_evolution_from_scratch/long_runs")
OUT = Path(__file__).resolve().parent / "plots" / "greedy_vs_single_hands.png"
# fixed categorical order (dataviz reference palette, light): slot 1 blue, 2 orange, 4 yellow, 5 magenta, 7 violet
GREEDY_MEAN, GREEDY_BEST = "#eb6834", "#eb6834"
SINGLE = (("random_pick_long", "RL on a random hand (seed_f5_cut_j1)", "#2a78d6"),
          ("claude_pick_long", "RL on Claude's pick (seed_f6_nocut_j2)", "#1baf7a"))


def single(name: str):
    d = glob.glob(str(LONG / f"*_evo_{name}_*" / "*" / "summaries"))
    if not d:
        return None
    st, v = _read(d[0])["episode_final/successes"]
    return st / FRAMES_PER_EPOCH, rolling(v, SMOOTH)          # one continuous run: no restarts to trim


def greedy_best():
    """Per iteration, the best hand at each logged step (each hand's SMOOTH-epoch rolling mean, the max over hands),
    then stitched like every evolution curve (restart dips trimmed, smoothed)."""
    segs = []
    for g, ser in iterations("baseline"):
        tags = [t for t in ser if t.startswith("per_hand_successes/")]
        if not tags:
            continue
        steps = ser[tags[0]][0]
        best = np.max([np.interp(steps, ser[t][0], rolling(ser[t][1], SMOOTH)) for t in tags], axis=0)
        e = steps / FRAMES_PER_EPOCH
        keep = e >= (SMOOTH if g == 0 else 0)   # from scratch, a max over 30 hands of few-point means is all noise
        segs.append((g, e[keep], best[keep]))
    return stitch(segs)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6.6), facecolor=SURFACE)
    style(ax)
    series = []
    x, y, last = curve("baseline")
    series.append((x, y, f"Greedy evolution (iteration {last})", GREEDY_MEAN, "-"))
    for name, label, color in SINGLE:
        r = single(name)
        if r is not None:
            series.append((r[0], r[1], label, color, "-"))
    xmax = max(s[0][-1] for s in series)
    for xs, ys, label, color, ls in series:
        hit = first_at(xs, ys)
        lab = f"{label}: {ys[-1]:.1f} now; " + (f"1 goal/ep at epoch {hit[0]:,.0f}" if hit else "never 1 goal/ep")
        ax.plot(xs, np.where(ys > YMIN, ys, np.nan), color=color, lw=2.0, ls=ls, label=lab, zorder=3)
        if hit:
            ax.plot([hit[0]], [hit[1]], "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(YMIN, 200)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
    ax.axhline(50, color=INK2, lw=0.8, ls=":")
    ax.text(xmax * 1.075, 54, "episode cap: 50 goals", ha="right", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xlim(0, xmax * 1.08)
    ax.set_xlabel("training epochs (24,576 envs x 16 steps each)", color=INK2, fontsize=11)
    ax.set_ylabel("training goals per episode (log scale)", color=INK2, fontsize=11)
    ax.set_title("Greedy evolution vs RL on a single chosen hand (same budget)", color=INK, fontsize=14, loc="left",
                 pad=14)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK)
    fig.text(0.01, 0.012, "One run each (single seed). Training goals per episode (every SAPG block). Greedy evolution's "
             f"restart dips trimmed (back to {RECOVER:.0%} of the previous level, at most {MAX_TRIM} epochs);\n"
             f"{SMOOTH}-epoch rolling means. Markers: first epoch at 1 goal/episode. The single-hand runs are still "
             "training.", fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.15)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
