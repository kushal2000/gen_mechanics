"""Two figures for choosing K (design-search rounds between policy iterations).

  k_drift.png   the K=50 study (study_k50.py): one value-net search from v1 gen 0, archive niche mean predicted by the
                value net (mapped to the 100-env scale) vs the actual 100-env niche mean under the gen-0 policy, by K
  k_budget.png  peak fitness (best design's 100-env goals/episode with the run's own policy, on its training archive)
                vs environment interactions for the K=1/5/10 campaign runs

    .venv_isaacsim/bin/python experiments/05nov_coevolve/plot_k_choice.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_search_check as psc  # noqa: E402

CAMP = REPO / "debug_outputs/coevolve/v1"
INK, INK2, GRID, AXIS = "#1f1f1e", "#6b6a66", "#e6e5e1", "#c9c8c3"
BLUE_LIGHT, BLUE = "#9cc3ee", "#2a78d6"
K_COL = {1: "#9cc3ee", 5: "#2a78d6", 10: "#0d3d7a"}


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, length=0)


def drift(plt, out: Path) -> None:
    R = json.loads((REPO / "debug_outputs/coevolve/studies/k50_none/report.json").read_text())
    KMAX = 10
    rows = [r for r in R["rounds"] if r["round"] <= KMAX]
    s0 = R["seeds"]["actual"]
    Kp = [0] + [r["round"] for r in rows]                   # the value net's prediction exists for every round
    pred = [s0] + [r["pred"] for r in rows]
    Ka = [0] + [r["round"] for r in rows if "actual" in r]  # actual: the rounds scored so far
    act = [s0] + [r["actual"] for r in rows if "actual" in r]
    fig, ax = plt.subplots(figsize=(6.6, 4.1), facecolor="white")
    ax.fill_between(Ka, act, np.interp(Ka, Kp, pred), color=BLUE_LIGHT, alpha=0.18, lw=0)
    ax.plot(Kp, pred, color=BLUE_LIGHT, lw=2, marker="o", ms=6, mec="white", mew=1.2, label="predicted (value net)")
    ax.plot(Ka, act, color=BLUE, lw=2, marker="o", ms=6, mec="white", mew=1.2, label="actual (100-env eval)")
    kb = int(np.argmax(act))
    ax.annotate(f"peak at K = {Ka[kb]}: {act[kb]:.2f}", (Ka[kb], act[kb]), xytext=(8, -26), textcoords="offset points",
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xticks(range(0, KMAX + 1))
    ax.set_xlim(-0.4, KMAX + 0.4)
    ax.set_xlabel("design-search rounds K", color=INK2)
    ax.set_ylabel("archive niche mean  (goals / episode)", color=INK2)
    ax.set_title("Predicted vs. actual archive quality as search runs longer", loc="left", fontsize=11.5,
                 fontweight="semibold", color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "k_drift.png", dpi=180, facecolor="white")
    print("[k_choice] K  actual  predicted  gap")
    A = dict(zip(Ka, act))
    for k, p in zip(Kp, pred):
        print(f"           {k:2d}  " + (f"{A[k]:.3f}   {p:.3f}   {p - A[k]:+.3f}" if k in A else f"  —     {p:.3f}"))


def peak(gd: Path) -> float | None:
    if not (gd / "eval100").exists():
        return None
    A, T = psc.archive_of(gd), psc.scores(gd, "eval100")
    ids = [i for v in A.values() for i in v]
    if not ids or any(i not in T for i in ids):
        return None
    return float(max(T[i] for i in ids))


def budget(plt, out: Path) -> None:
    seeds = peak(CAMP / "hod/gen_0000")
    fig, ax = plt.subplots(figsize=(6.6, 4.1), facecolor="white")
    print("[k_choice] peak goals/episode by budget (training archive, own policy)")
    for k in (1, 5, 10):
        xs, ys = [1], [seeds]
        for g in range(1, 12):
            v = peak(CAMP / f"k{k}_none" / f"gen_{g:04d}")
            if v is None:
                break
            xs.append(g + 1)
            ys.append(v)
        print(f"           K={k:<2d} " + "  ".join(f"{x}B {y:.2f}" for x, y in zip(xs, ys)))
        ax.plot(xs, ys, color=K_COL[k], lw=2, marker="o", ms=7, mec="white", mew=1.5, label=f"K = {k}", zorder=3)
        ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), xytext=(7, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_ylim(0.5, 70)
    xmax = max(max(l.get_xdata()) for l in ax.get_lines())
    ax.set_xticks(range(1, int(xmax) + 1))
    ax.set_xticklabels([f"{b}B" for b in range(1, int(xmax) + 1)])
    ax.set_xlim(0.7, xmax + 0.6)
    ax.set_xlabel("environment interactions", color=INK2)
    ax.set_ylabel("peak goals / episode", color=INK2)
    ax.set_title("Best design found vs. interaction budget, by K", loc="left", fontsize=11.5, fontweight="semibold",
                 color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "k_budget.png", dpi=180, facecolor="white")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(__file__).resolve().parent / "plots"
    out.mkdir(exist_ok=True)
    drift(plt, out)
    budget(plt, out)
    print(f"[k_choice] wrote {out / 'k_drift.png'} and {out / 'k_budget.png'}")


if __name__ == "__main__":
    main()
