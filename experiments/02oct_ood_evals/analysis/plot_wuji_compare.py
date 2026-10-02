"""Two policies on one hand: in-distribution (nominal) vs out-of-distribution evals, as grouped bars.

Top: goals/min per condition. Bottom: the same as a fraction of each policy's OWN nominal -- how much of
its skill survives the shift, which is the robustness comparison independent of how skilled it is.

    .venv_isaacsim/bin/python experiments/02oct_ood_evals/analysis/plot_wuji_compare.py \
        "Unified (8 hands), ep 6400=unified_ep_6400" "Wuji v2 only, ep 1600=wuji_only_ep_1600"
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
RES = HERE / "results/wuji_compare"
OUT = HERE / "plots/wuji_compare"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # categorical slots 1-4 of the validated default palette
CONDS = [("nominal", "nominal\n(train)"), ("cube40", "cube\n40 mm"), ("cube55", "cube\n55 mm"),
         ("cube65", "cube\n65 mm"), ("light", "light\ncube"), ("heavy", "heavy\ncube"),
         ("slippery", "slippery\n(×0.5 μ)"), ("push", "push"), ("push_hard", "hard\npush")]


def load(d):
    out = {}
    for c, _ in CONDS:
        f = RES / d / f"wuji2__{c}.json"
        out[c] = json.loads(f.read_text())["goals_per_min"] if f.exists() else np.nan
    return out


def main():
    pols = [a.split("=", 1) for a in sys.argv[1:]] or [
        ("Unified (8 hands), ep 6400", "unified_ep_6400"), ("Wuji v2 only, ep 1600", "wuji_only_ep_1600")]
    data = [(label, load(d)) for label, d in pols]
    n, k = len(CONDS), len(data)
    x = np.arange(n) + np.where(np.arange(n) > 0, 0.6, 0.0)      # a gap after the in-distribution group
    w = 0.8 / k
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.6), facecolor=SURFACE, sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.12})
    for ax, rel in zip(axes, (False, True)):
        ax.set_facecolor(SURFACE)
        for i, (label, v) in enumerate(data):
            y = np.array([v[c] for c, _ in CONDS])
            if rel:
                y = y / v["nominal"]
            xs = x + (i - (k - 1) / 2) * w
            ax.bar(xs, y, w * 0.92, color=SERIES[i], label=label, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            for xi, yi in zip(xs, y):
                if np.isfinite(yi):
                    ax.text(xi, yi, f"{yi:.2f}" if rel else f"{yi:.0f}", ha="center", va="bottom",
                            fontsize=7.5, color=INK2)
                else:
                    ax.text(xi, 0, "pending", ha="center", va="bottom", fontsize=7, color=INK2, rotation=90)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(INK2)
        ax.tick_params(colors=INK2, length=0)
        ax.axvline((x[0] + x[1]) / 2, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
        if rel:
            ax.axhline(1.0, color=INK2, linewidth=0.8)
            ax.set_ylabel("fraction of own nominal", color=INK, fontsize=10)
        else:
            ax.set_ylabel("goals / min", color=INK, fontsize=10)
            top = ax.get_ylim()[1]
            ax.text(x[0], top * 1.02, "in-distribution", ha="center", va="bottom", fontsize=9, color=INK)
            ax.text(x[1:].mean(), top * 1.02, "out-of-distribution", ha="center", va="bottom", fontsize=9,
                    color=INK)
    axes[1].set_xticks(x, [lab for _, lab in CONDS], fontsize=9, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, ncol=len(l), loc="upper left", bbox_to_anchor=(0.055, 0.995),
               labelcolor=INK)
    fig.suptitle("Wuji v2: unified policy vs Wuji-only policy (nearest saved checkpoint in skill)",
                 x=0.06, ha="left", fontsize=13, fontweight="bold", color=INK, y=1.075)
    fig.text(0.06, 1.03, "goals per minute, 1024 envs × 60 s per condition, greedy actions · bottom: each "
             "policy relative to its own nominal (1.0 = no loss)", fontsize=9, color=INK2)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "id_vs_ood.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print("->", p)
    for label, v in data:
        print(f"  {label:32s} " + " ".join(f"{c} {v[c]:.1f}" for c, _ in CONDS))


if __name__ == "__main__":
    main()
