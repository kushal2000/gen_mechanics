"""Plots of study 7 (study_vf_archive.py): the archive per niche, round 0 against round 1, and the value net's
predictions against the 100-env truth.

    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/plot_vf_archive.py [--iteration 3]

Writes experiments/05nov_noevolution/plots/vf_archive/: archive_per_niche.png, predicted_vs_true.png, per_niche.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
# the reference palette's first three categorical slots (validated all-pairs), text in ink, grid recessive
C_R0, C_R1, C_PRED = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def niche_key(n: str):
    return int(n[1]), int(n.split("_m")[1].split("-")[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", type=int, default=3)
    args = ap.parse_args()
    od = REPO / "debug_outputs/noevolution/studies" / f"it{args.iteration}_vf_archive"
    S = json.loads((od / "state.json").read_text())
    D = S["designs"]
    T = {i: D[i]["truth"][0] for i in D if D[i]["truth"]}
    for f in od.glob("truth_r*.json"):
        T.update({i: v["scores"][0] for i, v in json.loads(f.read_text()).items() if v["scores"]})
    A0, A1 = S["rounds"][0]["archive"], S["rounds"][1]["archive"]
    niches = sorted(A0, key=niche_key)
    rows = []
    for n in niches:
        t0 = [T[i] for i in A0[n]]
        t1 = [T[i] for i in A1[n]]
        p1 = [D[i]["pred"] for i in A1[n]]
        rows.append({"niche": n, **{f"r0_true_{k}": f(t0) for k, f in (("mean", st.mean), ("max", max), ("min", min))},
                     **{f"r1_true_{k}": f(t1) for k, f in (("mean", st.mean), ("max", max), ("min", min))},
                     **{f"r1_pred_{k}": f(p1) for k, f in (("mean", st.mean), ("max", max), ("min", min))}})
    out = REPO / "experiments/05nov_noevolution/plots/vf_archive"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_niche.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK, "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})

    # 1. per niche: min-max range and mean, round 0 true / round 1 true / round 1 predicted
    fig, ax = plt.subplots(figsize=(8.5, 11))
    y = np.arange(len(rows))[::-1]
    series = [("r0_true", "round 0 (the population), true", C_R0, -0.25, "o"),
              ("r1_true", "round 1 (after one round of search), true", C_R1, 0.0, "o"),
              ("r1_pred", "round 1, value net's prediction", C_PRED, 0.25, "D")]
    for key, label, col, dy, mk in series:
        lo = [r[f"{key}_min"] for r in rows]
        hi = [r[f"{key}_max"] for r in rows]
        mu = [r[f"{key}_mean"] for r in rows]
        ax.hlines(y + dy, lo, hi, color=col, lw=2, alpha=0.55, zorder=2)
        ax.scatter(mu, y + dy, s=36, color=col, marker=mk, edgecolor=SURF, linewidth=1.2, zorder=3, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels([r["niche"] for r in rows])
    ax.set_xlabel("goals per episode (100 envs; the line spans min to max, the marker is the mean)")
    ax.set_xlim(-0.03, 1.3)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    m0, m1, mp = (st.mean(r[f"{k}_mean"] for r in rows) for k in ("r0_true", "r1_true", "r1_pred"))
    ax.set_title(f"The archive per niche before and after one round of value-net search (iteration {args.iteration} "
                 f"policy)\nmean over niches: round 0 {m0:.2f} -> round 1 {m1:.2f} true, {mp:.2f} predicted",
                 loc="left", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.045), ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "archive_per_niche.png", dpi=160)
    plt.close(fig)

    # 2. predicted vs true: per niche (means) and per design (the round-1 archive)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    ax = axs[0]
    tm = [r["r1_true_mean"] for r in rows]
    pm = [r["r1_pred_mean"] for r in rows]
    ax.plot([0, 1.0], [0, 1.0], color=INK2, lw=1, ls="--", zorder=1)
    ax.scatter(tm, pm, s=40, color=C_R1, edgecolor=SURF, linewidth=1.2, zorder=3)
    for r, a, b in zip(rows, tm, pm):
        if r["niche"] in ("f3_m4-5", "f5_m20-21"):
            ax.annotate(r["niche"], (a, b), xytext=(6, -12), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_xlabel("true mean goals/episode (100 envs)")
    ax.set_ylabel("predicted mean goals/episode")
    ax.set_title(f"Per niche: the round-1 archive's mean\n(over-predicted by {st.mean(np.array(pm) - np.array(tm)):+.3f} "
                 f"on average)", loc="left", fontsize=10)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax = axs[1]
    ids = [i for v in A1.values() for i in v if D[i]["round"] == 1]
    tt = np.array([T[i] for i in ids])
    pp = np.array([D[i]["pred"] for i in ids])
    hb = ax.hexbin(tt, pp, gridsize=34, cmap="Oranges", mincnt=1, linewidths=0.2)
    ax.plot([0, 1.4], [0, 1.4], color=INK2, lw=1, ls="--")
    r = np.corrcoef(tt, pp)[0, 1]
    ax.set_xlabel("true goals/episode (100 envs)")
    ax.set_ylabel("predicted goals/episode")
    ax.set_title(f"Per design: the {len(ids)} children admitted in round 1\ncorrelation {r:.2f}; "
                 f"{(tt == 0).mean():.0%} score 0 goals", loc="left", fontsize=10)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("designs", color=INK2)
    for a in axs:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "predicted_vs_true.png", dpi=160)
    plt.close(fig)
    print(f"[plots] wrote {out}/archive_per_niche.png, predicted_vs_true.png, per_niche.csv")


if __name__ == "__main__":
    main()
