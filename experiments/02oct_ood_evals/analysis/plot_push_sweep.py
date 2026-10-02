"""Push robustness on Wuji v2: goals/min as random pushes get stronger and more frequent, one PNG per plot.

Pushes are the env's own wrench DR: per step, with probability p, the cube gets a force
randn(3) x cube mass x force_scale for one step (force_decay 0); no torque. Neither policy trained with any.
The "relative" plots divide by each policy's own unpushed speed (force_scale 0), so robustness compares
independently of how skilled the policy is.

    .venv_isaacsim/bin/python experiments/02oct_ood_evals/analysis/plot_push_sweep.py
    -> plots/push_sweep/{strength,rate}_{goals,relative}.png
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
R = HERE / "results/push_sweep"
OUT = HERE / "plots/push_sweep"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
POLICIES = [("Unified (8 hands), ep 7827", "unified_ep_7827", "#2a78d6"),
            ("Wuji v2 only, ep 1600", "wuji_only_ep_1600", "#eb6834"),
            ("Wuji v2 only, ep 3600 (converged)", "wuji_only_ep_3600", "#1baf7a"),
            ("Unified (8 hands), final ep 10000", "unified_final", "#eda100")]
SCALES = [0, 10, 20, 35, 50, 75, 100, 150]
PROBS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
AXES = {
    "strength": ("scale", SCALES, "push strength: force_scale  (≈ scale / 10 g per axis), at 1.2 pushes / s",
                 "Wuji v2 under stronger pushes"),
    "rate": ("prob", PROBS, "pushes per second (log scale), at force_scale 50 (≈ 5 g)",
             "Wuji v2 under more frequent pushes"),
}


def gpm(label, axis, v):
    f = R / label / f"{axis}_{v}.json"
    return json.loads(f.read_text())["goals_per_min"] if f.exists() else np.nan


def plot(name, relative):
    axis, xs, xlabel, title = AXES[name]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for label_txt, label, color in POLICIES:
        if not (R / label).exists():
            continue                                    # a policy whose sweep has not started yet
        y = np.array([gpm(label, axis, v) for v in xs])
        if relative:
            y = y / gpm(label, "scale", 0)
        x = np.array(xs, float) if axis == "scale" else 60 * np.array(xs)
        ok = np.isfinite(y)
        ax.plot(x[ok], y[ok], "-o", color=color, linewidth=2, markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label_txt)
    if axis == "prob":
        ax.set_xscale("log")
        ax.set_xlim(60 * PROBS[0] / 1.3, 60 * PROBS[-1] * 1.3)    # fixed, so partial results read right
    else:
        ax.set_xlim(-5, SCALES[-1] + 5)
    if relative:
        ax.axhline(1.0, color=INK2, linewidth=0.8)
    ax.set_ylim(bottom=0)
    ax.grid(color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel("fraction of own unpushed speed" if relative else "goals / min", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    ax.set_title(title + (" (relative to own unpushed speed)" if relative else ""), loc="left", fontsize=12,
                 fontweight="bold", color=INK, pad=24)
    ax.text(0, 1.02, "1024 envs × 60 s per point, greedy actions · push = randn(3) × cube mass × force_scale "
            "for one 1/60 s step", transform=ax.transAxes, fontsize=8, color=INK2)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}_{'relative' if relative else 'goals'}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return p


def main():
    for name in AXES:
        for rel in (False, True):
            print("->", plot(name, rel))
    for txt, label, _ in POLICIES:
        print(f"  {txt:36s} scale: " + " ".join(f"{gpm(label, 'scale', v):.1f}" for v in SCALES)
              + " | prob: " + " ".join(f"{gpm(label, 'prob', v):.1f}" for v in PROBS))


if __name__ == "__main__":
    main()
