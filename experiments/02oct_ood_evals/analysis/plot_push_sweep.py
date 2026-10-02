"""Push robustness on Wuji v2: goals/min as random pushes get stronger (left) and more frequent (right).

Pushes are the env's own wrench DR: per step, with probability p, the cube gets a force
randn(3) x cube mass x force_scale for one step (force_decay 0); no torque. Neither policy trained with any.
Bottom row: each policy relative to its own unpushed speed (force_scale 0), so robustness compares
independently of how skilled the policy is.

    .venv_isaacsim/bin/python experiments/02oct_ood_evals/analysis/plot_push_sweep.py
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
R = HERE / "results/push_sweep"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
POLICIES = [("Unified (8 hands), ep 7827", "unified_ep_7827", "#2a78d6"),
            ("Wuji v2 only, ep 1600", "wuji_only_ep_1600", "#eb6834"),
            ("Wuji v2 only, ep 3600 (converged)", "wuji_only_ep_3600", "#1baf7a")]
SCALES = [0, 10, 20, 35, 50, 75, 100, 150]
PROBS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def gpm(label, axis, v):
    f = R / label / f"{axis}_{v}.json"
    return json.loads(f.read_text())["goals_per_min"] if f.exists() else np.nan


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), facecolor=SURFACE, sharex="col",
                             gridspec_kw={"hspace": 0.12, "wspace": 0.18})
    for name, label, color in POLICIES:
        base = gpm(label, "scale", 0)
        for col, (axis, xs) in enumerate((("scale", SCALES), ("prob", PROBS))):
            y = np.array([gpm(label, axis, v) for v in xs])
            x = np.array(xs, float) if axis == "scale" else 60 * np.array(xs)      # pushes per second
            for row, yy in enumerate((y, y / base)):
                ax = axes[row, col]
                ok = np.isfinite(yy)
                ax.plot(x[ok], yy[ok], "-o", color=color, linewidth=2, markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=name if (row, col) == (0, 0) else None)
    for (row, col), ax in np.ndenumerate(axes):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
        ax.tick_params(colors=INK2)
        if col == 1:
            ax.set_xscale("log")
        if row == 1:
            ax.axhline(1.0, color=INK2, linewidth=0.8)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("push strength: force_scale  (≈ scale / 10 g per axis), 1.2 pushes / s" if col == 0
                          else "pushes per second (log), force_scale 50 (≈ 5 g)", color=INK, fontsize=10)
        else:
            ax.set_ylim(bottom=0)
    axes[0, 0].set_ylabel("goals / min", color=INK, fontsize=10)
    axes[1, 0].set_ylabel("fraction of own unpushed speed", color=INK, fontsize=10)
    axes[0, 0].set_title("stronger pushes", loc="left", fontsize=11, color=INK)
    axes[0, 1].set_title("more frequent pushes", loc="left", fontsize=11, color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, ncol=3, loc="upper left", bbox_to_anchor=(0.055, 0.985),
               labelcolor=INK)
    fig.suptitle("Wuji v2 under random pushes: unified vs Wuji-only policy", x=0.06, ha="left", fontsize=13,
                 fontweight="bold", color=INK, y=1.06)
    fig.text(0.06, 1.015, "1024 envs × 60 s per point, greedy actions · push = randn(3) × cube mass × force_scale "
             "for one 1/60 s step · neither policy trained with pushes", fontsize=9, color=INK2)
    out = HERE / "plots/push_sweep_wuji2.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print("->", out)
    for name, label, _ in POLICIES:
        print(f"  {name:36s} scale: " + " ".join(f"{gpm(label, 'scale', v):.1f}" for v in SCALES)
              + " | prob: " + " ".join(f"{gpm(label, 'prob', v):.1f}" for v in PROBS))


if __name__ == "__main__":
    main()
