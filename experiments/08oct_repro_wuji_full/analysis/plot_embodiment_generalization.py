"""Per-hand goals per episode of the 2-3-finger multi-hand policy (E4, 127128) on all 52 Wuji v2 finger-drop hands:
its 40 training hands (2 or 3 fingers) and 12 hands it never saw (4 and 5 fingers), grouped by finger count.

Reads results/eval_per_hand_2to3_me5_ep<N>_on_2plus.json (08oct_multi_embodiment_evolution/analysis/eval_per_hand.py:
64 envs per hand, each env's first episode, greedy). One dot per hand (left and right hands both drawn), a bar
for the group mean. Writes plots/h4_multi_embodiment_robustness/embodiment_generalization.png.

    python experiments/08oct_repro_wuji_full/analysis/plot_embodiment_generalization.py [--epoch 5000]
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import paper_style as ps  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def n_fingers(name: str) -> int:
    """wuji2[_right] = 5, ..._no_<f> = 4, ..._only_<a>_<b>[_<c>] = 2 or 3."""
    if "_only_" in name:
        return len(name.split("_only_")[1].split("_"))
    if "_no_" in name:
        return 4
    return 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", default="5000")
    args = ap.parse_args()
    res = json.loads((HERE / f"results/eval_per_hand_2to3_me5_ep{args.epoch}_on_2plus.json").read_text())
    hands = res["hands"]

    ps.configure()
    fig, ax = plt.subplots(figsize=(ps.TEXT_WIDTH * 0.55, 2.6))
    rng = np.random.default_rng(0)
    for k in (2, 3, 4, 5):
        group = [(n, v) for n, v in hands.items() if n_fingers(n) == k]
        held_out = group[0][1]["held_out"]
        assert all(v["held_out"] == held_out for _, v in group), k
        color = ps.COLORS["foil"] if held_out else ps.COLORS["hero"]
        y = np.array([v["goals_per_episode_first"] for _, v in group])
        x = k + rng.uniform(-0.18, 0.18, len(y))
        ax.scatter(x, y, s=16, color=color, alpha=0.75, linewidths=0, zorder=3)
        ax.hlines(y.mean(), k - 0.3, k + 0.3, color=color, linewidth=2.2, zorder=4)
        ax.text(k + 0.33, y.mean(), f"{y.mean():.1f}", va="center", ha="left", fontsize=7,
                color=ps.COLORS["ink"])
        print(f"{k} fingers ({'held out' if held_out else 'trained'}): {len(y)} hands, mean {y.mean():.2f}, "
              f"range {y.min():.2f}-{y.max():.2f}")
    ax.axvline(3.5, color=ps.COLORS["neutral"], linestyle=":", linewidth=1)
    ax.set_xticks([2, 3, 4, 5])
    ax.set_xlim(1.5, 5.7)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Fingers on the hand")
    ax.set_ylabel("Goals / episode")
    ax.set_title("2-3-Finger Policy on All 52 Hands", fontweight="bold")
    ps.style_axis(ax)
    fig.tight_layout()
    handles = [Line2D([], [], marker="o", linestyle="", color=ps.COLORS["hero"], markersize=5),
               Line2D([], [], marker="o", linestyle="", color=ps.COLORS["foil"], markersize=5),
               Line2D([], [], color=ps.COLORS["muted_ink"], linewidth=2.2)]
    labels = ["Training hands (40)", "Unseen hands (12)", "Group mean"]
    ps.bottom_legend(fig, handles, labels, y=-0.01, loc="upper center", ncol=3, bold=())
    out = HERE / "plots/h4_multi_embodiment_robustness/embodiment_generalization"
    ps.save_figure(fig, out, formats=("png",))
    plt.close(fig)


if __name__ == "__main__":
    main()
