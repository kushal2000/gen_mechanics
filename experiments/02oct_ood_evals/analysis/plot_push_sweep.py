"""Push robustness on Wuji v2, in the TRAINING metric: goals per episode, episodes capped at 50 goals.

Two comparisons, each the latest unified policy against one Wuji-only checkpoint:
  vs_wuji_ep1600   the Wuji-only checkpoint nearest the unified policy in skill
  vs_wuji_ep3600   the converged Wuji-only policy
and for each, push strength (force_scale at 0.03 pushes/step) and push rate (at force_scale 50), absolute
and relative to the policy's own unpushed value: one PNG per plot.

Pushes are the env's own wrench DR: per step, with probability p, the cube gets randn(3) x cube mass x
force_scale for one 1/60 s step; no torque. Neither policy trained with any. goals/episode is each env's FIRST
episode (every env starts at reset together, so short episodes are not over-sampled), ended by a drop, 10 s
without a goal, or the 50-goal cap.

    .venv_isaacsim/bin/python experiments/02oct_ood_evals/analysis/plot_push_sweep.py
    -> plots/push_sweep_cap50/<comparison>/{strength,rate}_{goals,relative}.png
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
R = HERE / "results/push_sweep_cap50"
OUT = HERE / "plots/push_sweep_cap50"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
UNIFIED, WUJI = "#2a78d6", "#eb6834"                    # categorical slots 1, 2: unified blue, Wuji-only orange
SCALES = [0, 10, 20, 35, 50, 75, 100, 150]
PROBS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]
AXES = {
    "strength": ("scale", SCALES, "push strength: force_scale  (≈ scale / 10 g per axis), at 0.03 pushes / step",
                 "stronger pushes"),
    "rate": ("prob", PROBS, "pushes per second (log scale), at force_scale 50 (≈ 5 g)", "more frequent pushes"),
}
COMPARISONS = {
    "vs_wuji_ep1600": ("Wuji v2 only, ep 1600", "wuji_only_ep_1600"),
    "vs_wuji_ep3600": ("Wuji v2 only, ep 3600 (converged)", "wuji_only_ep_3600"),
}


def latest_unified():
    """The final checkpoint once its sweep has results, else ep 7827."""
    d = R / "unified_final"
    if d.exists() and any(d.iterdir()):
        return "Unified (8 hands), final", "unified_final"
    return "Unified (8 hands), ep 7827", "unified_ep_7827"


def gpe(label, axis, v):
    f = R / label / f"{axis}_{v}.json"
    if not f.exists():
        return np.nan
    return json.loads(f.read_text())["goals_per_episode_first"]


def plot(comp, name, relative):
    axis, xs, xlabel, what = AXES[name]
    pols = [(*latest_unified(), UNIFIED), (*COMPARISONS[comp], WUJI)]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for txt, label, color in pols:
        y = np.array([gpe(label, axis, v) for v in xs])
        if relative:
            y = y / gpe(label, "scale", 0)
        x = np.array(xs, float) if axis == "scale" else 60 * np.array(xs)
        ok = np.isfinite(y)
        ax.plot(x[ok], y[ok], "-o", color=color, linewidth=2, markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=txt)
    if axis == "prob":
        ax.set_xscale("log")
        ax.set_xlim(60 * PROBS[0] / 1.3, 60 * PROBS[-1] * 1.3)    # fixed, so partial results read right
    else:
        ax.set_xlim(-5, SCALES[-1] + 5)
    if relative:
        ax.axhline(1.0, color=INK2, linewidth=0.8)
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(0, 53)
        ax.axhline(50, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.text(ax.get_xlim()[1], 50.5, "cap: 50 goals / episode ", fontsize=8, color=INK2, va="bottom",
                ha="right")
    ax.grid(color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel("fraction of own unpushed goals / episode" if relative else "goals / episode", color=INK,
                  fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    ax.set_title(f"Wuji v2 under {what}" + (", relative to unpushed" if relative else ""), loc="left",
                 fontsize=12, fontweight="bold", color=INK, pad=24)
    ax.text(0, 1.02, "first episode per env, ended by a drop, 10 s without a goal, or 50 goals · "
            "1024 envs × 240 s per point", transform=ax.transAxes, fontsize=8, color=INK2)
    out = OUT / comp
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}_{'relative' if relative else 'goals'}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return p


def main():
    for comp in COMPARISONS:
        for name in AXES:
            for rel in (False, True):
                print("->", plot(comp, name, rel))
    for txt, label in [latest_unified(), *COMPARISONS.values()]:
        print(f"  {txt:36s} scale: " + " ".join(f"{gpe(label, 'scale', v):.1f}" for v in SCALES)
              + " | prob: " + " ".join(f"{gpe(label, 'prob', v):.1f}" for v in PROBS))


if __name__ == "__main__":
    main()
