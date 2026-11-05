"""The value-net MAP-Elites archive as a grid per round (study 7): rows = finger count, columns = motor bins, the mean
goals/episode per cell -- the value net's prediction (top) and the 100-env truth (bottom, rounds scored so far).
Cells that are not niches are greyed out. One colour scale for every panel and round.

    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/plot_vf_archive_grid.py [--iteration 3]

Writes experiments/05nov_noevolution/plots/vf_archive/archive_niche_means.png: one column per round ("Iter"), the
rounds with 100-env truth so far (--all: every round).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
INK, INK2, SURF, EMPTY = "#1f1f1e", "#6b6a66", "#ffffff", "#f3f2ef"
FINGERS = [2, 3, 4, 5, 6]
VMAX = 1.25
FS = 9.5          # cell value font size (set per figure)
# one sequential hue, light -> dark (the reference palette's blue), starting off-white so 0 still reads as a cell
CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fc", "#9cc3ee", "#2a78d6", "#0d3d7a"])


def parse(n: str):
    f, m = n.split("_m")
    lo, hi = m.split("-")
    return int(f[1:]), (int(lo), int(hi))


def draw(ax, A, bins, value, title):
    col = {b: k for k, b in enumerate(bins)}
    row = {f: k for k, f in enumerate(FINGERS)}
    nr, nc = len(FINGERS), len(bins)
    ax.set_xlim(-0.5, nc - 0.5)
    ax.set_ylim(nr - 0.5, -0.5)
    ax.set_aspect("equal")
    gap = 0.06
    for fi in range(nr):
        for bi in range(nc):
            name = f"f{FINGERS[fi]}_m{bins[bi][0]}-{bins[bi][1]}"
            v = value(A[name]) if name in A else None
            face = EMPTY if v is None else CMAP(min(v, VMAX) / VMAX)
            ax.add_patch(plt.Rectangle((bi - 0.5 + gap, fi - 0.5 + gap), 1 - 2 * gap, 1 - 2 * gap, facecolor=face,
                                       edgecolor="none"))
            if v is not None:
                ax.text(bi, fi, f"{v:.2f}", ha="center", va="center", fontsize=FS,
                        color="white" if v > 0.55 * VMAX else INK)
            elif name in A:                                       # a niche, not scored yet
                ax.text(bi, fi, "N/A", ha="center", va="center", fontsize=FS - 1, color=INK2)
    ax.set_xticks(range(nc))
    ax.set_xticklabels([f"{lo}–{hi}" for lo, hi in bins], fontsize=FS - 1.5, color=INK2, rotation=90 if FS < 8 else 0)
    ax.set_yticks(range(nr))
    ax.set_yticklabels([str(f) for f in FINGERS], fontsize=9.5, color=INK2)
    ax.set_ylabel("fingers", color=INK2, fontsize=9.5)
    ax.tick_params(length=0, pad=4)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11 if FS > 8 else 9.5, color=INK, pad=6, fontweight="semibold")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--tag", default="", help="the search run's tag (study_vf_archive.py --tag)")
    ap.add_argument("--all", action="store_true", help="every round, unscored ones marked (default: scored rounds only)")
    args = ap.parse_args()
    od = REPO / "debug_outputs/noevolution/studies" / (f"it{args.iteration}_vf_archive" + (f"_{args.tag}" if args.tag else ""))
    S = json.loads((od / "state.json").read_text())
    D = S["designs"]
    T = {i: D[i]["truth"][0] for i in D if D[i]["truth"]}
    for f in od.glob("truth_r*.json"):
        T.update({i: v["scores"][0] for i, v in json.loads(f.read_text()).items() if v["scores"]})
    bins = sorted({parse(n)[1] for n in S["rounds"][0]["archive"]})
    out = REPO / "experiments/05nov_noevolution/plots/vf_archive"
    out.mkdir(parents=True, exist_ok=True)
    rounds = list(range(len(S["rounds"])))                   # every iteration: predicted for all, actual where scored
    global FS
    FS = 9.5 if len(rounds) <= 2 else 7.2
    w = 6.0 if len(rounds) <= 2 else 4.3
    fig, axs = plt.subplots(2, len(rounds), figsize=(w * len(rounds) + 1.0, 5.6 if len(rounds) > 2 else 6.4),
                            facecolor=SURF, squeeze=False)

    def actual(ids):
        return st.mean(T[i] for i in ids) if all(i in T for i in ids) else None
    for c, r in enumerate(rounds):
        A = S["rounds"][r]["archive"]
        draw(axs[0, c], A, bins, lambda ids: st.mean(D[i]["pred"] for i in ids), f"Iter {r} – Predicted Niche Means")
        draw(axs[1, c], A, bins, actual, f"Iter {r} – Actual Niche Means")
        axs[1, c].set_xlabel("motors", color=INK2, fontsize=9.5)
        if c:
            for a in axs[:, c]:
                a.set_ylabel("")
    fig.subplots_adjust(left=0.03 if len(rounds) > 2 else 0.06, right=0.955 if len(rounds) > 2 else 0.92, top=0.93,
                        bottom=0.12, hspace=0.38, wspace=0.08)
    cax = fig.add_axes([0.965 if len(rounds) > 2 else 0.935, 0.2, 0.005 if len(rounds) > 2 else 0.009, 0.6])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, VMAX)), cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8.5, labelcolor=INK2)
    cb.set_label("goals / episode", color=INK2, fontsize=9)
    fig.savefig(out / f"archive_niche_means{'_' + args.tag if args.tag else ''}.png", dpi=170, facecolor=SURF)
    plt.close(fig)
    print(f"[plots] wrote archive_niche_means{'_' + args.tag if args.tag else ''}.png (iterations {rounds})")


if __name__ == "__main__":
    main()
