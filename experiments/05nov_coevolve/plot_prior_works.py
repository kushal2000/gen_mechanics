"""Does CoEvolve outperform prior works? Peak fitness (the best design's 100-env goals/episode, scored with the
method's own policy) against environment interactions, for

  Locoformer       one multi-embodiment policy trained on the fixed seed population (v1/hod), scored on that population
  House of Dextra  the same policy after B steps, then one K=5 value-net design search on its training logs (no extra env
                   steps), scored with that policy (studies/hod_search_<B>b; 1B = v1/k5_none's first search, which is
                   identical: same labels, net and seeds)
  CoEvolve         K=5 co-evolution (v1/<run>), each generation's archive scored with its own policy

    .venv_isaacsim/bin/python experiments/05nov_coevolve/plot_prior_works.py [--ours k5_none] [--metric peak|top10]

Writes experiments/05nov_coevolve/plots/prior_works_<metric>.png and prints the table. Points whose eval is not
complete are left out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_search_check as psc  # noqa: E402

CAMP = REPO / "debug_outputs/coevolve/v1"
STUDY = REPO / "debug_outputs/coevolve/studies"
COL = {"Locoformer": "#8a8984", "House of Dextra": "#d9822b", "CoEvolve (ours)": "#2a78d6"}


INTERIM: set = set()                                   # (method, budget) drawn from the top-256 quick eval


def point(gd: Path, sub: str, metric: str, key=None) -> float | None:
    """The metric over the full archive eval; if that is not complete, over the quick top-256 eval (quick_peak.py)."""
    if not (gd / "manifest.json").exists():
        return None
    if (gd / sub).exists():
        A, T = psc.archive_of(gd), psc.scores(gd, sub)
        ids = [i for v in A.values() for i in v]
        if ids and all(i in T for i in ids):
            v = np.sort([T[i] for i in ids])[::-1]
            return float(v[0] if metric == "peak" else v[:10].mean())
    quick = gd / sub.replace("eval100_policy", "eval100_top256_policy")
    if key and (quick / "c00_fitness.json").exists():
        v = np.sort([x["goals_per_episode"] for x in json.loads((quick / "c00_fitness.json").read_text())["hands"].values()])[::-1]
        INTERIM.add(key)
        return float(v[0] if metric == "peak" else v[:10].mean())
    return None


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="k5_none")
    ap.add_argument("--metric", choices=("peak", "top10"), default="peak")
    ap.add_argument("--prelim", action="store_true",
                    help="before the post-search evals: CoEvolve = its training archive (solid), skip empty lines")
    args = ap.parse_args()
    B = [1, 2, 3, 4, 5]
    rows = {
        "Locoformer": [point(CAMP / "hod" / f"gen_{b - 1:04d}", "eval100", args.metric) for b in B],
        "House of Dextra": [point(CAMP / "k5_none/gen_0001", "eval100_policy_g0000", args.metric)] + [
            point(STUDY / f"hod_search_{b}b" / f"gen_{b:04d}", f"eval100_policy_g{b - 1:04d}", args.metric,
                  ("House of Dextra", b)) for b in B[1:]],
        # policy after B steps (gen B-1), then the search's proposal (its own gen-B archive) scored by that policy
        "CoEvolve (ours)": [point(CAMP / args.ours / f"gen_{b:04d}", f"eval100_policy_g{b - 1:04d}", args.metric,
                                  ("CoEvolve (ours)", b)) for b in B],
    }
    within = [point(CAMP / "hod/gen_0000", "eval100", args.metric)] + [
        point(CAMP / args.ours / f"gen_{b - 1:04d}", "eval100", args.metric) for b in B[1:]]
    print(f"{args.metric} goals/episode (100 envs, own policy)")
    print(f"{'':16s}" + "".join(f"{b:>7d}B" for b in B))
    for k, v in rows.items():
        print(f"{k:16s}" + "".join((f"{x:7.2f}*" if (k, b) in INTERIM else f"{x:8.2f}") if x is not None
                                   else f"{'—':>8s}" for b, x in zip(B, v)))
    if INTERIM:
        print("  * interim: best of the value net's top 256 designs (quick_peak.py); full eval pending")

    print(f"{'CoEvolve, within':16s}" + "".join(f"{x:8.2f}" if x is not None else f"{'—':>8s}" for x in within)
          + "   (its training archive, no final search)")

    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor="white")
    if args.prelim:                                       # preliminary: CoEvolve measured on its training archive
        rows["CoEvolve (ours)"] = within
    else:
        xs = [b for b, x in zip(B, within) if x is not None]
        ax.plot(xs, [x for x in within if x is not None], color=COL["CoEvolve (ours)"], lw=1.2, ls="--", alpha=0.45,
                zorder=2, label="CoEvolve, training archive (no final search)")
    for k, v in rows.items():
        if args.prelim and sum(x is not None for x in v) < 2:
            continue
        xs = [b for b, x in zip(B, v) if x is not None]
        ys = [x for x in v if x is not None]
        ax.plot(xs, ys, color=COL[k], lw=2, marker="o", ms=7, mec="white", mew=1.5, label=k, zorder=3)
        hol = [(b, x) for b, x in zip(B, v) if x is not None and (k, b) in INTERIM]
        if hol:                                           # interim points: hollow
            ax.plot(*zip(*hol), ls="none", marker="o", ms=7, mfc="white", mec=COL[k], mew=1.8, zorder=4)
        if ys:
            dy = {"Locoformer": -7, "House of Dextra": 7}.get(k, 0)    # the two baselines end at ~1: keep labels apart
            ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), xytext=(7, dy), textcoords="offset points", va="center",
                        fontsize=9, color="#1f1f1e")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_ylim(0.5, 70)
    ax.set_xticks(B)
    ax.set_xticklabels([f"{b}B" for b in B])
    ax.set_xlim(0.7, 5.6)
    ax.set_xlabel("environment interactions", color="#6b6a66")
    ax.set_ylabel(("peak" if args.metric == "peak" else "top-10 mean") + " goals / episode", color="#6b6a66")
    ax.set_title("Best design found vs. interaction budget", loc="left", fontsize=11.5, fontweight="semibold",
                 color="#1f1f1e")
    ax.grid(color="#e6e5e1", lw=0.8, which="major")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors="#6b6a66", length=0)
    ax.legend(frameon=False, fontsize=9.5, labelcolor="#1f1f1e", loc="upper left")
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "plots" / f"prior_works_{args.metric}{'_prelim' if args.prelim else ''}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=180, facecolor="white")
    print(f"[prior_works] wrote {out}")


if __name__ == "__main__":
    main()
