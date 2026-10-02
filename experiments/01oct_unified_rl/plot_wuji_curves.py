"""Wuji v2 learning curves: the unified (8-hand) run vs the Wuji-only run, goals/episode on Wuji v2.

Left: against training epoch -- what each run costs in wall-clock and compute (every epoch is 12288 envs x
16 steps in both). Right: against the Wuji v2 experience each run has had -- the unified run gives Wuji v2
1/8 of its envs, so one of its epochs holds 1/8 the Wuji samples of a Wuji-only epoch.

    .venv_isaacsim/bin/python experiments/01oct_unified_rl/plot_wuji_curves.py
"""
import glob
import os
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
L = REPO / "debug_outputs/train_logs"
OUT = pathlib.Path(__file__).resolve().parent / "plots"
F = 196608                                             # frames per epoch: 12288 envs x 16 steps
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
UNIFIED, ONLY = "#2a78d6", "#eb6834"                  # categorical slots 1, 2


def curve(pattern, tag):
    ev = max(glob.glob(pattern), key=os.path.getmtime)
    ea = EventAccumulator(ev, size_guidance={"scalars": 0})
    ea.Reload()
    s = ea.Scalars(tag)
    ep = np.array([x.step / F for x in s])
    v = np.array([x.value for x in s])
    k = 15                                              # light smoothing, centred
    vs = np.convolve(v, np.ones(k) / k, mode="same")
    vs[: k // 2], vs[-(k // 2):] = v[: k // 2], v[-(k // 2):]
    return ep, vs


def first_reach(ep, v, level=1.0):
    i = np.argmax(v >= level)
    return ep[i] if v[i] >= level else None


def main():
    uni = curve(f"{L}/01oct_unified_rl/0_scale_train_left_multi_uniform_canon_c01_*866192*/rank_0/*/summaries/events*",
                "per_hand/wuji2_successes")
    only = curve(f"{L}/01oct_uniform_dynamics/0_scale_train_left_wuji2_uniform_canon_c01_*/rank_0/*/summaries/events*",
                 "episode_final/successes")
    runs = [("Unified policy, 8 hands (Wuji v2 = 1/8 of envs)", uni, UNIFIED, 1 / 8),
            ("Wuji v2-only policy", only, ONLY, 1.0)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE, sharey=True,
                             gridspec_kw={"wspace": 0.06})
    for ax, per_hand in zip(axes, (False, True)):
        ax.set_facecolor(SURFACE)
        for label, (ep, v), color, share in runs:
            x = ep * share if per_hand else ep
            ax.plot(x, v, color=color, linewidth=2, label=label)
            r = first_reach(ep, v)
            if r is not None:
                xr = r * share if per_hand else r
                ax.plot([xr], [1.0], "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                        zorder=5)
            ax.annotate(f"{v[-1]:.1f}", (x[-1], v[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        ax.grid(color=GRID, linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
        ax.tick_params(colors=INK2)
        ax.set_xlabel("Wuji v2 experience (Wuji-only-epoch equivalents)" if per_hand else "training epoch",
                      color=INK, fontsize=10)
        ax.set_title("per Wuji v2 sample" if per_hand else "per training epoch (same compute per epoch)",
                     loc="left", fontsize=10, color=INK2)
    axes[0].set_ylabel("goals / episode on Wuji v2 (training)", color=INK, fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, loc="center right", bbox_to_anchor=(1.0, 0.62), labelcolor=INK)
    fig.suptitle("Wuji v2: joint training vs Wuji-only training", x=0.07, ha="left", fontsize=13,
                 fontweight="bold", color=INK, y=1.01)
    fig.text(0.07, 0.945, "dots: first epoch at 1 goal/episode · both runs observe the canonical palm frame + "
             "palm extents · 12288 envs per epoch in both", fontsize=9, color=INK2)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "wuji_learning_curves.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print("->", p)
    for label, (ep, v), _, share in runs:
        r = first_reach(ep, v)
        print(f"  {label:50s} 1 goal/ep at epoch {r:.0f} ({r * share:.0f} Wuji-equivalent); "
              f"latest {v[-1]:.1f} @ {ep[-1]:.0f}" if r else f"  {label}: never 1 goal/ep")


if __name__ == "__main__":
    main()
