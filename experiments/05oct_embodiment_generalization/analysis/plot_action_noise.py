"""Action-noise sweep on Wuji v2: goals per episode as Gaussian noise is added to every action, every step.

Line plot, unified multi-embodiment policy vs the Wuji-only policy; one figure per comparison, PDF + PNG, paper
style. Data: results/action_noise/<label>/actnoise_<std>.json from run_embodiment_sweep.sub AXIS=actnoise.
Noise is in the normalized action space ([-1, 1] over each joint's range), then clipped.

    .venv_isaacsim/bin/python experiments/05oct_embodiment_generalization/analysis/plot_action_noise.py
    -> plots/action_noise_<comparison>.{pdf,png}
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
R = HERE / "results/action_noise"
OUT = HERE / "plots"
STDS = ["0", "0.1", "0.2", "0.3", "0.5", "0.75", "1.0"]
COMPARISONS = {   # file suffix: (unified label, Wuji-only label)
    "vs_wuji_32k": ("unified_40k", "wuji_only_32k"),
    "vs_wuji_40k": ("unified_40k", "wuji_only_40k"),
}


def gpe(label, std):
    f = R / label / f"actnoise_{std}.json"
    return json.loads(f.read_text())["goals_per_episode_first"] if f.exists() else np.nan


def plot(comp):
    uni, wuji = COMPARISONS[comp]
    pols = [("Unified Multi-Embodiment", uni, ps.COLORS["hero"], "o"), ("Wuji Only", wuji, ps.COLORS["foil"], "s")]
    fig, ax = plt.subplots(figsize=(ps.COLUMN_WIDTH, 2.4))
    x = np.array([float(s) for s in STDS])
    for _, label, color, marker in pols:
        y = np.array([gpe(label, s) for s in STDS])
        ok = np.isfinite(y)
        ax.plot(x[ok], y[ok], marker=marker, color=color, markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xlim(-0.04, 1.04)
    ax.axhline(50, color=ps.COLORS["neutral"], linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
    ax.text(ax.get_xlim()[1], 50.8, "max", ha="right", va="bottom", fontsize=7, color=ps.COLORS["muted_ink"])
    ax.set_ylim(0, 56)
    ps.style_axis(ax)
    ax.set_xlabel("Action noise (std)")
    ax.set_ylabel("Goals / episode")
    ax.set_title("Robustness to Action Noise", fontweight="bold")
    ps.bottom_legend(fig, [Patch(facecolor=c, alpha=ps.BAR_ALPHA) for _, _, c, _ in pols], [t for t, *_ in pols],
                     y=-0.14, columnspacing=1.6, bold=("Unified Multi-Embodiment",))
    paths = ps.save_figure(fig, OUT / f"action_noise_{comp}")
    plt.close(fig)
    return paths


def main():
    ps.configure()
    for comp in COMPARISONS:
        plot(comp)
    for label in ("unified_40k", "wuji_only_32k", "wuji_only_40k"):
        print(f"  {label:14s} " + " ".join(f"{s}:{gpe(label, s):.1f}" for s in STDS))


if __name__ == "__main__":
    main()
