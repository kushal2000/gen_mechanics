"""The solo retraining (solo_retrain/): the three best designs, each trained ALONE from scratch, 3 seeds each.

    .venv_isaacsim/bin/python experiments/oct_2026_experiments/30oct_evolution_evaluation/plot_solo_retrain.py

Left: training goals per episode against epochs (the trainer's episode_final/successes, every SAPG block; a 50-epoch
rolling mean per seed), the mean over the 3 seeds as a line and the slowest-to-fastest seed as a band. Markers:
first epoch the seed mean reaches 1 goal/episode. Right: the final evaluation (eval_solo.sh: eval_hands_cap.py, the episode
cap raised to 200, 20 envs per seed), one dot per seed and the mean. Writes plots/solo_retrain.png.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_training_curves import FRAMES_PER_EPOCH, INK, INK2, SMOOTH, SURFACE, _read, rolling, style  # noqa: E402

LOGS = Path("debug_outputs/train_logs/30oct_evolution_evaluation/solo_retrain")
EVAL = Path("debug_outputs/evolution/solo_eval")
OUT = Path(__file__).resolve().parent / "plots" / "solo_retrain.png"
SEEDS = (100, 101, 102)
# (hand, label, colour: greedy orange and MAP-Elites aqua as in every evolution plot; Claude violet)
HANDS = (("seed_f3_cut_j2", "Claude's blind pick (seed_f3_cut_j2)", "#6250d6"),
         ("g0018_c00", "MAP-Elites' best (g0018_c00)", "#1baf7a"),
         ("g0029_c02", "Greedy evolution's best (g0029_c02)", "#eb6834"))
GRID = np.arange(0, 6361, 10.0)


def seed_curve(hand: str, seed: int):
    d = glob.glob(str(LOGS / f"*_solo_{hand}_s{seed}_*" / "*" / "summaries"))
    st, v = _read(d[0])["episode_final/successes"]
    e = st / FRAMES_PER_EPOCH
    y = rolling(v, SMOOTH)
    return np.where(GRID <= e.max(), np.interp(GRID, e, y), np.nan)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 6), facecolor=SURFACE, gridspec_kw={"width_ratios": [3.2, 1]})
    style(ax)
    style(bx)
    for k, (hand, label, color) in enumerate(HANDS):
        curves = np.array([seed_curve(hand, s) for s in SEEDS])
        mean, lo, hi = np.nanmean(curves, 0), np.nanmin(curves, 0), np.nanmax(curves, 0)
        m = np.isfinite(mean)
        ax.fill_between(GRID[m], lo[m], hi[m], color=color, alpha=0.18, lw=0)
        hit = np.nonzero(mean >= 1.0)[0]
        lab = label + (f": 1 goal/ep at epoch {GRID[hit[0]]:,.0f}" if len(hit) else "")
        ax.plot(GRID[m], mean[m], color=color, lw=2.0, label=lab, zorder=3)
        if len(hit):
            ax.plot([GRID[hit[0]]], [mean[hit[0]]], "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=4)
        final = []
        for s in SEEDS:
            f = EVAL / f"solo_{hand}_s{s}.json"
            if f.exists():
                final.append(list(json.loads(f.read_text())["hands"].values())[0]["goals_per_episode"])
        if final:
            bx.scatter([k] * len(final), final, s=60, color=color, edgecolor=SURFACE, lw=1.5, zorder=3)
            bx.plot([k - 0.25, k + 0.25], [np.mean(final)] * 2, color=color, lw=3, zorder=4)
            bx.text(k + 0.3, np.mean(final), f"{np.mean(final):.0f}", va="center", fontsize=10, color=INK)
    ax.set_ylim(0, 45)
    ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
    ax.set_xlim(0, 6360)
    ax.set_xlabel("training epochs (24,576 envs x 16 steps each)", color=INK2, fontsize=11)
    ax.set_ylabel("training goals per episode", color=INK2, fontsize=11)
    ax.set_title("Each design trained alone from scratch: mean of 3 seeds (band: slowest to fastest seed)",
                 color=INK, fontsize=12.5, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=INK)
    bx.set_xlim(-0.6, 2.9)
    bx.set_ylim(0, 200)
    bx.set_xticks(range(3))
    bx.set_xticklabels(["Claude", "MAP-Elites", "Greedy"], color=INK2, fontsize=10)
    bx.set_ylabel("final evaluation: goals per episode (cap 200)", color=INK2, fontsize=11)
    bx.set_title("After 6,360 epochs", color=INK, fontsize=12.5, loc="left")
    fig.text(0.01, 0.01, f"Training: the trainer's episode_final/successes (every SAPG block), {SMOOTH}-epoch rolling mean per "
             "seed.\nFinal: 20 envs per seed, first episode each, sampled actions, episode cap raised from 50 to 200; dots "
             "are seeds, bars means.", fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.15, wspace=0.22)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
