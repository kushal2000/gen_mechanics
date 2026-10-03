"""Push robustness on Wuji v2, in the training metric: goals per episode, episodes capped at 50 goals.

Two comparisons, each the latest unified policy against one Wuji-only checkpoint:
  vs_wuji_ep1600   the Wuji-only checkpoint nearest the unified policy in skill
  vs_wuji_ep3600   the converged Wuji-only policy
For each: push strength (force_scale at 0.03 pushes/step) and push rate (at force_scale 50), absolute and
relative to the policy's own unpushed value. One figure per plot, PDF + PNG, in the paper style
(paper_style.py, copied from one-phase-paper).

Pushes are the env's own wrench DR: per step, with probability p, the cube gets randn(3) x cube mass x
force_scale for one 1/60 s step; no torque. Neither policy trained with any. Goals/episode is each env's
first episode, ended by a drop, 10 s without a goal, or the 50-goal cap; 1024 envs x 240 s per point.

    .venv_isaacsim/bin/python experiments/02oct_ood_evals/analysis/plot_push_sweep.py
    -> plots/push_sweep_cap50/<comparison>/{strength,rate}_{goals,relative}.{pdf,png}
"""
import json
import pathlib

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from matplotlib.patches import Patch  # noqa: E402

import paper_style as ps  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
R = HERE / "results/push_sweep_cap50"
OUT = HERE / "plots/push_sweep_cap50"
SCALES = [0, 10, 20, 35, 50, 75, 100, 150]
PROBS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]
AXES = {    # name: (result prefix, values, x label, title)
    "strength": ("scale", SCALES, "Push strength (force scale)", "Stronger pushes"),
    "rate": ("prob", PROBS, "Pushes per second", "More frequent pushes"),
}
COMPARISONS = {
    "vs_wuji_ep1600": ("Wuji Only", "wuji_only_ep_1600"),
    "vs_wuji_ep3600": ("Wuji Only", "wuji_only_ep_3600"),
    # Matched compute: both policies after ~10k epochs of 12288 envs (Wuji-only 908714 ep 6400 = 10162 total).
    "vs_wuji_10k": ("Wuji Only", "wuji_only_10k", "unified_final"),
}


def latest_unified():
    """The newest unified checkpoint whose evals are complete: the 20k-epoch fine-tune, else the 10k final,
    else ep 7827."""
    for label in ("unified_20k", "unified_final"):
        d = R / label
        if d.exists() and len(list(d.glob("*.json"))) >= len(SCALES) + len(PROBS):
            return "Unified Multi-Embodiment", label
    return "Unified Multi-Embodiment", "unified_ep_7827"


def gpe(label, axis, v):
    f = R / label / f"{axis}_{v}.json"
    return json.loads(f.read_text())["goals_per_episode_first"] if f.exists() else np.nan


def plot(comp, name, relative):
    axis, xs, xlabel, title = AXES[name]
    uni = ("Unified Multi-Embodiment", COMPARISONS[comp][2]) if len(COMPARISONS[comp]) > 2 else latest_unified()
    pols = [(*uni, ps.COLORS["hero"], "o"), (*COMPARISONS[comp][:2], ps.COLORS["foil"], "s")]
    fig, ax = plt.subplots(figsize=(ps.COLUMN_WIDTH, 2.4))
    handles = []
    for txt, label, color, marker in pols:
        y = np.array([gpe(label, axis, v) for v in xs])
        if relative:
            y = y / gpe(label, "scale", 0)
        x = np.array(xs, float) if axis == "scale" else 60 * np.array(xs)
        ok = np.isfinite(y)
        (h,) = ax.plot(x[ok], y[ok], marker=marker, color=color, markeredgecolor="white",
                       markeredgewidth=0.6, label=txt)
        handles.append(h)
    if axis == "prob":
        ax.set_xscale("log")
        ax.set_xlim(60 * PROBS[0] / 1.3, 60 * PROBS[-1] * 1.3)    # fixed, so partial results read right
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.3, 1, 3, 10]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    else:
        ax.set_xlim(-5, SCALES[-1] + 5)
    if relative:
        ax.axhline(1.0, color=ps.COLORS["neutral"], linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
        ax.set_ylim(0, 1.15)
    else:
        ax.axhline(50, color=ps.COLORS["neutral"], linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
        ax.text(ax.get_xlim()[1], 50.8, "max", ha="right", va="bottom", fontsize=7,
                color=ps.COLORS["muted_ink"])
        ax.set_ylim(0, 56)
    ps.style_axis(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Relative goals / episode" if relative else "Goals / episode")
    ax.set_title("Robustness to Force Perturbations", fontweight="bold")
    ps.bottom_legend(fig, [Patch(facecolor=h.get_color(), alpha=ps.BAR_ALPHA) for h in handles],
                     [h.get_label() for h in handles], y=-0.14, columnspacing=1.6, bold=("Unified Multi-Embodiment",))
    stem = OUT / comp / f"{name}_{'relative' if relative else 'goals'}"
    paths = ps.save_figure(fig, stem)
    plt.close(fig)
    return paths


def main():
    ps.configure()
    for comp in COMPARISONS:
        for name in AXES:
            for rel in (False, True):
                plot(comp, name, rel)
    for txt, label, *_ in [latest_unified(), *COMPARISONS.values()]:
        print(f"  {txt:30s} scale: " + " ".join(f"{gpe(label, 'scale', v):.1f}" for v in SCALES)
              + " | prob: " + " ".join(f"{gpe(label, 'prob', v):.1f}" for v in PROBS))


if __name__ == "__main__":
    main()
