"""Image tables of the best goals per episode each algorithm found in each niche: greedy evolution (baseline) and
MAP-Elites, side by side.

    .venv_isaacsim/bin/python experiments/oct_2026_experiments/30oct_evolution_evaluation/plot_niche_tables.py

The numbers are best_per_niche_tables.py's: per niche, the highest goals per episode any hand the run made there scored
in a training evaluation (10 envs, first episode each; under that iteration's policy), the hand's id under it. Rows:
fingers; columns: palm cut x finger length (mean joints per finger < 2.5, 2.5-3.5, > 3.5). One shared colour scale (a
single blue ramp, square-root so small values still read); a niche whose best scored 0 is grey.
Writes experiments/oct_2026_experiments/30oct_evolution_evaluation/plots/best_per_niche_tables.png.
"""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np

from coevolution.evolution.archive import Archive

RUNS = Path("debug_outputs/evolution")
OUT = Path(__file__).resolve().parent / "plots" / "best_per_niche_tables.png"
COLS = [(c, j) for c in ("nocut", "cut") for j in ("j0", "j1", "j2")]
LENGTH = {"j0": "short", "j1": "medium", "j2": "long"}
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]   # blue 100 -> 700
ZERO, INK, INK2, SURFACE = "#ecebe7", "#0b0b0b", "#52514e", "#fcfcfb"
TITLES = {"baseline": "Greedy evolution", "map_elites": "MAP-Elites"}


def grid(run: str):
    arch = Archive.load(RUNS / run / "archive.json")
    by = collections.defaultdict(list)
    for r in arch.records.values():
        by[r.niche].append(r)
    last = max(e["generation"] for r in arch.records.values() for e in r.fitness if not e["screen"])
    val = np.zeros((5, 6))
    ids = [["" for _ in range(6)] for _ in range(5)]
    for i, f in enumerate(range(2, 7)):
        for k, (cut, j) in enumerate(COLS):
            best = None
            for r in by.get(f"f{f}_{cut}_{j}", []):
                for e in r.fitness:
                    if not e["screen"] and (best is None or e["goals_per_episode"] > best[1]):
                        best = (r.id, e["goals_per_episode"])
            if best:
                ids[i][k], val[i, k] = best
    return val, ids, last


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm

    data = {run: grid(run) for run in ("baseline", "map_elites")}
    vmax = max(v.max() for v, _, _ in data.values())
    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    norm = PowerNorm(0.5, vmin=0.0, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8.2), facecolor=SURFACE)
    for ax, (run, (val, ids, last)) in zip(axes, data.items()):
        ax.set_facecolor(SURFACE)
        for i in range(5):
            for k in range(6):
                v = val[i, k]
                fill = ZERO if v <= 0 else cmap(norm(v))
                ax.add_patch(plt.Rectangle((k + 0.03, i + 0.03), 0.94, 0.94, facecolor=fill, edgecolor=SURFACE, lw=2))
                dark = v > 0 and norm(v) > 0.45
                ax.text(k + 0.5, i + 0.44, f"{v:g}", ha="center", va="center", fontsize=20, fontweight="bold",
                        color=SURFACE if dark else INK)
                ax.text(k + 0.5, i + 0.78, ids[i][k], ha="center", va="center", fontsize=10,
                        color=SURFACE if dark else INK2)
        ax.set_xlim(0, 6)
        ax.set_ylim(5, 0)
        ax.set_xticks([k + 0.5 for k in range(6)])
        ax.set_xticklabels([f"{'palm cut' if c == 'cut' else 'no cut'}\n{LENGTH[j]} fingers" for c, j in COLS],
                           fontsize=13, color=INK2)
        ax.xaxis.tick_top()
        ax.set_yticks([i + 0.5 for i in range(5)])
        ax.set_yticklabels([f"{f} fingers" for f in range(2, 7)], fontsize=14, color=INK2)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(TITLES[run], fontsize=20, fontweight="bold", color=INK, pad=58)
    fig.text(0.01, 0.015, "Cell: the best training-evaluation goals per episode any hand found in that niche scored "
             "(10 envs, first episode each, under that iteration's policy), and that hand. Grey: never above 0. "
             "One colour scale for both (square root). One run per algorithm.", fontsize=11, color=INK2)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.07, wspace=0.22)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
