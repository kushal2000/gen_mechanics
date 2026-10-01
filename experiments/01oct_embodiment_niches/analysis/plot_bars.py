"""Bar charts of the phase-1 metrics: one chart per metric, one bar per hand.

    .venv_isaacsim/bin/python experiments/01oct_embodiment_niches/analysis/plot_bars.py

Writes ../plots/details/: <metric>.png each, nominal_grid.png (every nominal
metric), conditions_grid.png (goals/min under each perturbation, absolute).
Hands keep one fixed order in every chart so a hand sits in the same row everywhere; the best
hand per metric is the darker bar.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3]
D = REPO / "debug_outputs/embodiment_niches"
OUT = pathlib.Path(__file__).resolve().parent.parent / "plots" / "details"

NAMES = {"sharpa": "SHARPA", "gen_sharpa": "gen-SHARPA", "allegro": "Allegro", "leap": "LEAP",
         "shadow": "Shadow", "tesollo": "Tesollo", "dex3": "Dex3", "xhand": "XHAND", "wuji2": "Wuji v2"}
# Reference palette: one blue; the best bar takes a darker step of the same ramp.
BAR, BEST = "#86b6ef", "#1c5cab"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

# metric -> (title, unit/format, higher is better, getter on a run's JSON)
METRICS = {
    "goals_per_min":   ("Throughput", "goals / min", True, lambda d: d["goals_per_min"]),
    "time_per_goal":   ("Time per goal", "median s", False, lambda d: d["time_per_goal_s"]["median"]),
    "goals_per_drop":  ("Reliability", "goals per drop", True, lambda d: d["goals_per_drop"]),
    "path_efficiency": ("Path efficiency", "goal angle / cube rotation", True, lambda d: d["path_efficiency"]),
    "work_per_goal":   ("Energy per goal", "joint work, J", False, lambda d: d["work_per_goal_J"]),
    "power":           ("Mean power", "W per hand", False, lambda d: d["mean_power_W_per_env"]),
    "target_rate":     ("Action rate", "rad/s per joint", False, lambda d: d["target_rate_rad_s_per_joint"]),
    "speed_capped":    ("Time at joint speed cap", "% of steps", False, lambda d: 100 * d["joint_speed_saturated_frac"]),
    "obj_speed":       ("Cube speed", "m/s", False, lambda d: d["obj_speed_mean"]),
    "obj_acc":         ("Cube acceleration", "m/s²", False, lambda d: d["obj_acc_mean"]),
    "palm_dist_std":   ("Cube wander", "mm (std of palm distance)", False, lambda d: 1000 * d["palm_dist_std"]),
    "participation":   ("Joints used", "participation ratio / joints", True, lambda d: d["participation_frac"]),
}
CONDS = {"cube40": "40 mm cube", "cube55": "55 mm cube", "cube65": "65 mm cube",
         "light": "Cube mass ×0.5", "heavy": "Cube mass ×2", "slippery": "Cube friction ×0.5",
         "push": "Pushes ~2 g", "push_hard": "Pushes ~5 g"}


def _fmt(v: float) -> str:
    a = abs(v)
    return f"{v:.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.2f}"


def bars(ax, hands, vals, title, unit, higher):
    best = (max if higher else min)(range(len(vals)), key=lambda i: vals[i])
    y = list(range(len(hands)))[::-1]                      # first hand on top
    colors = [BEST if i == best else BAR for i in range(len(vals))]
    ax.barh(y, vals, height=0.62, color=colors, edgecolor=SURFACE, linewidth=1.0)
    vmax = max(vals) if max(vals) > 0 else 1.0
    for yi, v in zip(y, vals):
        ax.text(v + vmax * 0.015, yi, _fmt(v), va="center", ha="left", fontsize=8, color=INK2)
    ax.set_yticks(y, [NAMES[h] for h in hands], fontsize=9, color=INK)
    ax.set_xlim(0, vmax * 1.18)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="semibold", pad=16)
    ax.text(0, 1.02, f"{unit} · {'higher' if higher else 'lower'} is better", transform=ax.transAxes,
            fontsize=8, color=INK2, va="bottom")
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=INK2, labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    s = json.loads((D / "summary.json").read_text())
    R, hands = s["raw"], s["hands"]
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"

    nominal = {m: [g(R[f"{h}__nominal"]) for h in hands] for m, (_, _, _, g) in METRICS.items()}
    for m, (title, unit, hi, _) in METRICS.items():
        fig, ax = plt.subplots(figsize=(5.2, 3.6), facecolor=SURFACE)
        bars(ax, hands, nominal[m], title, unit, hi)
        fig.tight_layout()
        fig.savefig(OUT / f"{m}.png", dpi=160, facecolor=SURFACE)
        plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(20, 11.5), facecolor=SURFACE)
    for ax, (m, (title, unit, hi, _)) in zip(axes.flat, METRICS.items()):
        bars(ax, hands, nominal[m], title, unit, hi)
    fig.suptitle("Nine left-hand policies, nominal conditions (1024 envs × 60 s, greedy)",
                 x=0.01, ha="left", fontsize=14, color=INK, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.5, w_pad=2.0)
    fig.savefig(OUT / "nominal_grid.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(20, 7.8), facecolor=SURFACE)
    for ax, (c, label) in zip(axes.flat, CONDS.items()):
        vals = [R[f"{h}__{c}"]["goals_per_min"] for h in hands]
        bars(ax, hands, vals, label, "goals / min", True)
    for c, label in CONDS.items():
        f2, a2 = plt.subplots(figsize=(5.2, 3.6), facecolor=SURFACE)
        bars(a2, hands, [R[f"{h}__{c}"]["goals_per_min"] for h in hands], f"Throughput: {label}",
             "goals / min", True)
        f2.tight_layout()
        f2.savefig(OUT / f"goals_per_min__{c}.png", dpi=160, facecolor=SURFACE)
        plt.close(f2)
    fig.suptitle("Throughput under perturbation (absolute goals / min)", x=0.01, ha="left",
                 fontsize=14, color=INK, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.5, w_pad=2.0)
    fig.savefig(OUT / "conditions_grid.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
