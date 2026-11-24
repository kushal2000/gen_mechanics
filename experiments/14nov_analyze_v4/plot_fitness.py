"""Peak and mean fitness across iterations for the three v4 methods (14 Nov 2026): at every generation, the best and the
mean goals / episode over that generation's 4,096 training designs (gen_XXXX/labels.json: the training-log label, mean
goals over the design's episodes in the last 25% of the generation's training).

    House of Dextra   debug_outputs/coevolve/v4/hod         (K 0: the gen-0 designs throughout)
    MAP-Elites        debug_outputs/coevolve/v4/k5_none     (K 5, 16 per cell)
    Greedy            debug_outputs/coevolve/v4/k5_global   (K 5, one pool)

All three share gen 0 (the two searches branch from hod's trained gen 0). An episode stops at 50 goals
(max_successes), so 50 is the ceiling; both plots use the same 0-50 scale. A label is a mean over ~100 episodes, so the
maximum over 4,096 of them sits a little above the best design's true rate. The mean is over each run's own training
sample: greedy's is ~6k near-copies in a few dozen cells, MAP-Elites' an equal share of every cell.

    .venv_isaacsim/bin/python experiments/14nov_analyze_v4/plot_fitness.py [--out experiments/14nov_analyze_v4/plots]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUNS = [("hod", "House of Dextra"), ("k5_none", "MAP-Elites"), ("k5_global", "Greedy")]
COLORS = {"hod": "#2a78d6", "k5_none": "#eb6834", "k5_global": "#1baf7a"}   # categorical slots 1-3 (dataviz palette)
CAP = 50


def fitness_by_gen(run: Path, min_episodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(generations, peak, mean) of the training designs' labels."""
    gens, peak, mean = [], [], []
    for f in sorted(run.glob("gen_*/labels.json")):
        lab = json.loads(f.read_text())
        v = np.array([x[0] for x in lab.values() if x[1] >= min_episodes])
        if not len(v):
            continue
        gens.append(int(f.parent.name.split("_")[1]))
        peak.append(v.max())
        mean.append(v.mean())
    return np.array(gens), np.array(peak), np.array(mean)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="debug_outputs/coevolve/v4")
    ap.add_argument("--out", default="experiments/14nov_analyze_v4/plots")
    ap.add_argument("--min-episodes", type=int, default=20, help="ignore designs with fewer logged episodes")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    data = {k: fitness_by_gen(REPO / a.campaign / k, a.min_episodes) for k, _ in RUNS}

    ink, muted, grid = "#1f1f1e", "#6b6a64", "#e4e3df"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False})
    plots = [  # (column of fitness_by_gen, file, title, subtitle, y label)
        (1, "peak_fitness.png", "v4: peak design fitness per generation",
         "Max over each generation's 4,096 training designs (training-log labels); shared gen 0",
         "Best design, goals / episode"),
        (2, "mean_fitness.png", "v4: mean design fitness per generation",
         "Mean over each generation's 4,096 training designs (training-log labels); shared gen 0",
         "Mean design, goals / episode"),
    ]
    for col, fname, title, sub, ylab in plots:
        fig, ax = plt.subplots(figsize=(8.0, 4.6), facecolor="white")
        ax.axhline(CAP, color=muted, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(0.5, CAP + 0.8, "episode cap: 50 goals", color=muted, fontsize=8.5, va="bottom")
        ends = []
        for k, name in RUNS:
            g, y = data[k][0], data[k][col]
            ax.plot(g, y, color=COLORS[k], lw=2, zorder=3, label=name, solid_capstyle="round")
            ends.append([float(y[-1]), float(g[-1]), f"{name} {y[-1]:.1f}"])
        ends.sort()                                       # end labels nudged apart (>= 2.4 goals) so none collide
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 2.4)
        for y, g, txt in ends:
            ax.text(g + 0.8, y, txt, color=ink, fontsize=9, va="center")
        ax.set_xlim(0, max(d[0].max() for d in data.values()) + 12)
        ax.set_ylim(0, CAP + 4)
        ax.set_xlabel("Generation (1B training steps each)")
        ax.set_ylabel(ylab)
        ax.grid(axis="y", color=grid, lw=0.8, zorder=0)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, fontsize=9)
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold", color=ink)
        ax.set_title(sub, loc="left", fontsize=9, color=muted)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150, facecolor="white")
        plt.close(fig)
        print("[fitness]", out / fname)

    with open(out / "fitness.csv", "w", newline="") as fh:          # the table view
        w = csv.writer(fh)
        w.writerow(["generation"] + [f"{k}_peak" for k, _ in RUNS] + [f"{k}_mean" for k, _ in RUNS])
        gens = sorted(set().union(*(set(d[0].tolist()) for d in data.values())))
        look = {k: dict(zip(d[0].tolist(), zip(d[1].tolist(), d[2].tolist()))) for k, d in data.items()}
        for gi in gens:
            w.writerow([gi] + [f"{look[k][gi][0]:.3f}" if gi in look[k] else "" for k, _ in RUNS]
                       + [f"{look[k][gi][1]:.3f}" if gi in look[k] else "" for k, _ in RUNS])
    for k, name in RUNS:
        g, p, m = data[k]
        i, j = int(p.argmax()), int(m.argmax())
        print(f"{name:16s} peak {p.max():5.1f} at gen {g[i]:2d}, final {p[-1]:5.1f} | "
              f"mean: best {m.max():5.1f} at gen {g[j]:2d}, final {m[-1]:5.1f}")


if __name__ == "__main__":
    main()
