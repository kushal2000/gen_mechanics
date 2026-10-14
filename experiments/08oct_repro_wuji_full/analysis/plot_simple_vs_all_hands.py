"""Train on simple hands vs train on all of them: per-hand goals per episode on all 52 Wuji v2 finger-drop hands of
two joint-transformer policies with the identical recipe (v5, EMA 0.1 + action delay, 24k envs, 5 mini-epochs,
epoch 5000, seed 100), differing only in the hands they trained on:

    E4 (127128)   the 40 hands with 2 or 3 fingers (4- and 5-finger hands never seen)
    all (127127)  all 52 hands, the 4- and 5-finger ones included

Reads results/eval_per_hand_2to3_me5_ep5000_on_2plus_stochastic.json and
results/eval_per_hand_2plus_me5_ep5000_on_2plus_stochastic.json (08oct_multi_embodiment_evolution/analysis/
eval_per_hand.py: 64 envs per hand, goals in each env's first episode, sampled actions; the all-hands eval ran as 4
parallel parts of 13 hands, merged). One dot per hand, a bar for the group mean.
Writes plots/h4_multi_embodiment_robustness/simple_vs_all_hands.png.

    python experiments/08oct_repro_wuji_full/analysis/plot_simple_vs_all_hands.py
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import paper_style as ps  # noqa: E402
from plot_embodiment_generalization import n_fingers  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
RUNS = [  # (results file, legend, colour, x offset)
    ("eval_per_hand_2to3_me5_ep5000_on_2plus_stochastic.json", "Trained on 2-3-finger hands only (Ours)",
     ps.COLORS["hero"], -0.18),
    ("eval_per_hand_2plus_me5_ep5000_on_2plus_stochastic.json", "Trained on all 52 hands", ps.COLORS["foil"], 0.18),
]


def main():
    ps.configure()
    fig, ax = plt.subplots(figsize=(ps.TEXT_WIDTH * 0.6, 2.6))
    rng = np.random.default_rng(0)
    for fname, label, color, dx in RUNS:
        hands = json.loads((HERE / "results" / fname).read_text())["hands"]
        for k in (2, 3, 4, 5):
            y = np.array([v["goals_per_episode_first"] for n, v in hands.items() if n_fingers(n) == k])
            x = k + dx + rng.uniform(-0.07, 0.07, len(y))
            ax.scatter(x, y, s=12, color=color, alpha=0.7, linewidths=0, zorder=3)
            ax.hlines(y.mean(), k + dx - 0.15, k + dx + 0.15, color=color, linewidth=2.2, zorder=4)
            if dx < 0:      # beside the bar (the dots sit on it); the other policy's are all near 0: above
                ax.text(k + dx + 0.17, y.mean(), f"{y.mean():.1f}", ha="left", va="center", fontsize=7,
                        color=ps.COLORS["ink"])
            else:
                ax.text(k + dx + 0.1, y.mean() + 0.8, f"{y.mean():.1f}", ha="center", va="bottom", fontsize=7,
                        color=ps.COLORS["ink"])
            print(f"{label:42s} {k} fingers: {y.mean():.2f} ({len(y)} hands)")
    ax.axvline(3.5, color=ps.COLORS["neutral"], linestyle=":", linewidth=1)
    ax.text(3.58, 22.5, "Never seen by\nthe 2-3-finger policy", ha="left", va="top", fontsize=7,
            color=ps.COLORS["muted_ink"])
    ax.set_xticks([2, 3, 4, 5])
    ax.set_xlim(1.5, 5.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Fingers on the hand")
    ax.set_ylabel("Goals in first episode")
    ax.set_title("Train on Simple Hands vs All Hands", fontweight="bold")
    ps.style_axis(ax)
    fig.tight_layout()
    handles = [Line2D([], [], marker="o", linestyle="", color=c, markersize=5) for _, _, c, _ in RUNS] + \
              [Line2D([], [], color=ps.COLORS["muted_ink"], linewidth=2.2)]
    labels = [lab for _, lab, _, _ in RUNS] + ["Group mean"]
    ps.bottom_legend(fig, handles, labels, y=-0.01, loc="upper center", ncol=2, handlelength=1.6)
    ps.save_figure(fig, HERE / "plots/h4_multi_embodiment_robustness/simple_vs_all_hands", formats=("png",))
    plt.close(fig)


if __name__ == "__main__":
    main()
