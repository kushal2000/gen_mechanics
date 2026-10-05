"""Missing-finger ablation on Wuji v2: goals per episode with the full hand and with each finger removed.

Grouped bars, unified multi-embodiment policy vs the Wuji-only policy; one figure per comparison, PDF + PNG,
paper style. Data: results/missing_finger/<label>/finger_<finger>.json from run_embodiment_sweep.sub.

    .venv_isaacsim/bin/python experiments/05oct_embodiment_generalization/analysis/plot_missing_finger.py
    -> plots/missing_finger_<comparison>.{pdf,png}
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import paper_style as ps  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
R = HERE / "results/missing_finger"
OUT = HERE / "plots"
FINGERS = [("none", "None"), ("thumb", "Thumb"), ("index", "Index"), ("middle", "Middle"), ("ring", "Ring"),
           ("pinky", "Pinky")]
COMPARISONS = {   # file suffix: (unified label, Wuji-only label)
    "vs_wuji_32k": ("unified_40k", "wuji_only_32k"),
    "vs_wuji_40k": ("unified_40k", "wuji_only_40k"),
}


def gpe(label, finger):
    f = R / label / f"finger_{finger}.json"
    return json.loads(f.read_text())["goals_per_episode_first"] if f.exists() else np.nan


def plot(comp):
    uni, wuji = COMPARISONS[comp]
    pols = [("Unified Multi-Embodiment", uni, ps.COLORS["hero"]), ("Wuji Only", wuji, ps.COLORS["foil"])]
    x = np.arange(len(FINGERS)) + np.where(np.arange(len(FINGERS)) > 0, 0.4, 0.0)   # gap after the full hand
    w = 0.38
    fig, ax = plt.subplots(figsize=(ps.COLUMN_WIDTH * 1.25, 2.4))
    for i, (_, label, color) in enumerate(pols):
        y = np.array([gpe(label, k) for k, _ in FINGERS])
        xs = x + (i - 0.5) * w
        ax.bar(xs, np.nan_to_num(y), w * 0.92, color=color, alpha=ps.BAR_ALPHA, linewidth=0)
        for xi, yi in zip(xs, y):
            if np.isfinite(yi):
                ax.text(xi, yi + 0.6, f"{yi:.0f}", ha="center", va="bottom", fontsize=6, color=ps.COLORS["ink"])
    ax.axvline((x[0] + x[1]) / 2, color=ps.COLORS["neutral"], linewidth=0.7, linestyle=(0, (3, 2)))
    ax.axhline(50, color=ps.COLORS["neutral"], linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
    ax.set_ylim(0, 56)
    ps.style_axis(ax)
    ax.tick_params(axis="x", length=0)
    ax.set_xticks(x, [lab for _, lab in FINGERS])
    ax.set_xlabel("Missing finger")
    ax.set_ylabel("Goals / episode")
    ax.set_title("Robustness to Missing Fingers", fontweight="bold")
    ps.bottom_legend(fig, [Patch(facecolor=c, alpha=ps.BAR_ALPHA) for _, _, c in pols], [t for t, *_ in pols],
                     y=-0.14, columnspacing=1.6, bold=("Unified Multi-Embodiment",))
    paths = ps.save_figure(fig, OUT / f"missing_finger_{comp}")
    plt.close(fig)
    return paths


def main():
    ps.configure()
    for comp in COMPARISONS:
        plot(comp)
    for label in ("unified_40k", "wuji_only_32k", "wuji_only_40k"):
        print(f"  {label:14s} " + " ".join(f"{k} {gpe(label, k):.1f}" for k, _ in FINGERS))


if __name__ == "__main__":
    main()
