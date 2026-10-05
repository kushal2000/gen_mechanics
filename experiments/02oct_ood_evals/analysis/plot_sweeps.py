"""OOD severity sweeps on Wuji v2, in the training metric (goals per episode, 50-goal max, each env's first episode):
push force in N, cube mass in g, friction coefficient. One figure per axis and comparison, PDF + PNG, paper style.

    .venv_isaacsim/bin/python experiments/02oct_ood_evals/analysis/plot_sweeps.py
    -> plots/sweeps_cap50/<comparison>/{force,mass,friction}.{pdf,png}

Data: results/sweeps_cap50/<label>/<axis>_<value>.json from run_ood_sweep.sub.
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
R = HERE / "results/sweeps_cap50"
OUT = HERE / "plots/sweeps_cap50"
AXES = {    # name: (result prefix, values as written in file names, x values, x label, title, reverse x)
    "force": ("newton", ["0", "5", "10", "15", "20", "25"], [0, 5, 10, 15, 20, 25],
              "Push force (N)", "Robustness to Force Perturbations", False),
    "mass": ("mass", ["45.6", "40", "30", "20", "10", "5"], [45.6, 40, 30, 20, 10, 5],
             "Cube mass (g)", "Robustness to Lighter Objects", True),
    "friction": ("friction", ["0.5", "0.4", "0.3", "0.2", "0.1", "0.05", "0"], [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0],
                 "Friction coefficient", "Robustness to Lower Friction", True),
}
COMPARISONS = {   # folder: (unified label, Wuji-only label)
    "vs_wuji_32k": ("unified_40k", "wuji_only_32k"),
    "vs_wuji_40k": ("unified_40k", "wuji_only_40k"),
}


def gpe(label, prefix, v):
    f = R / label / f"{prefix}_{v}.json"
    return json.loads(f.read_text())["goals_per_episode_first"] if f.exists() else np.nan


def plot(comp, name):
    prefix, keys, xs, xlabel, title, reverse = AXES[name]
    uni, wuji = COMPARISONS[comp]
    pols = [("Unified Multi-Embodiment", uni, ps.COLORS["hero"], "o"), ("Wuji Only", wuji, ps.COLORS["foil"], "s")]
    fig, ax = plt.subplots(figsize=(ps.COLUMN_WIDTH, 2.4))
    for txt, label, color, marker in pols:
        y = np.array([gpe(label, prefix, k) for k in keys])
        x = np.array(xs, float)
        ok = np.isfinite(y)
        ax.plot(x[ok], y[ok], marker=marker, color=color, markeredgecolor="white", markeredgewidth=0.6, label=txt)
    lo, hi = min(xs), max(xs)
    pad = 0.04 * (hi - lo)
    ax.set_xlim((hi + pad, lo - pad) if reverse else (lo - pad, hi + pad))   # severity increases to the right
    ax.axhline(50, color=ps.COLORS["neutral"], linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
    ax.text(ax.get_xlim()[1], 50.8, "max", ha="right", va="bottom", fontsize=7, color=ps.COLORS["muted_ink"])
    ax.set_ylim(0, 56)
    ps.style_axis(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Goals / episode")
    ax.set_title(title, fontweight="bold")
    ps.bottom_legend(fig, [Patch(facecolor=c, alpha=ps.BAR_ALPHA) for _, _, c, _ in pols], [t for t, *_ in pols],
                     y=-0.14, columnspacing=1.6, bold=("Unified Multi-Embodiment",))
    paths = ps.save_figure(fig, OUT / comp / name)
    plt.close(fig)
    return paths


FINGERS = [("none", "None"), ("thumb", "Thumb"), ("index", "Index"), ("middle", "Middle"), ("ring", "Ring"),
           ("pinky", "Pinky")]


def plot_fingers(comp):
    """Grouped bars: goals/episode with no finger missing, then with each finger removed."""
    uni, wuji = COMPARISONS[comp]
    pols = [("Unified Multi-Embodiment", uni, ps.COLORS["hero"]), ("Wuji Only", wuji, ps.COLORS["foil"])]
    x = np.arange(len(FINGERS)) + np.where(np.arange(len(FINGERS)) > 0, 0.4, 0.0)   # gap after the full hand
    w = 0.38
    fig, ax = plt.subplots(figsize=(ps.COLUMN_WIDTH * 1.25, 2.4))
    for i, (txt, label, color) in enumerate(pols):
        y = np.array([gpe(label, "finger", k) for k, _ in FINGERS])
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
    paths = ps.save_figure(fig, OUT / comp / "missing_finger")
    plt.close(fig)
    return paths


def main():
    ps.configure()
    for comp in COMPARISONS:
        for name in AXES:
            plot(comp, name)
        plot_fingers(comp)
    for label in ("unified_40k", "wuji_only_32k", "wuji_only_40k"):
        print(f"  {label:14s} " + " | ".join(
            f"{name}: " + " ".join(f"{gpe(label, a[0], k):.1f}" for k in a[1]) for name, a in AXES.items())
              + " | fingers: " + " ".join(f"{k} {gpe(label, 'finger', k):.1f}" for k, _ in FINGERS))


if __name__ == "__main__":
    main()
