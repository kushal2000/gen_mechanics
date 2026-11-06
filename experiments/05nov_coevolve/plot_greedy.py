"""Is MAP-Elites diversity maintenance necessary? CoEvolve with niches (v1/k<K>_none) vs greedy evolution
(v1/k<K>_global: the same search, labels and seeds, but selection keeps the population's global top 4,096).

  greedy_grid.png       best design per niche of the gen-1 archive (the first search), both scored by the same gen-0
                        policy with 100 envs; empty niches marked
  greedy_diversity.png  niches occupied and distinct seed lineages per generation

    .venv_isaacsim/bin/python experiments/05nov_coevolve/plot_greedy.py [--k 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "experiments/05nov_noevolution/studies"))
import diversity as D  # noqa: E402
import plot_search_check as psc  # noqa: E402
import plot_vf_archive_grid as G  # noqa: E402

CAMP = REPO / "debug_outputs/coevolve/v1"
INK, INK2 = "#1f1f1e", "#6b6a66"
COL = {"MAP-Elites (ours)": "#2a78d6", "Greedy": "#d9822b"}


def grid(plt, k: int, out: Path) -> None:
    runs = {"MAP-Elites (ours)": f"k{k}_none", "Greedy": f"k{k}_global"}
    seeds = psc.archive_of(CAMP / "hod/gen_0000")
    bins = sorted({G.parse(n)[1] for n in seeds})
    G.FS = 7.6
    G.VMAX = 0.9
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 3.5), facecolor="white")
    print(f"[greedy] K={k}, gen-1 archive scored by the gen-0 policy (100 envs)")
    for ax, (lab, r) in zip(axs, runs.items()):
        gd = CAMP / r / "gen_0001"
        A, T = psc.archive_of(gd), psc.scores(gd, "eval100_policy_g0000")
        full = {n: (A.get(n, [])) for n in seeds}             # every niche of the grid; empty ones get no value
        draw_A = {n: v for n, v in full.items() if v}          # emptied niches are drawn as blank cells + a label
        G.draw(ax, draw_A, bins, lambda ids: max(T[i] for i in ids),
               f"{lab}: best design per niche ({len(A)}/32 niches occupied)")
        for n, v in full.items():                            # mark the emptied niches
            if not v:
                f, (lo, hi) = G.parse(n)
                x, y = bins.index((lo, hi)), G.FINGERS.index(f)
                ax.add_patch(plt.Rectangle((x - 0.44, y - 0.44), 0.88, 0.88, facecolor="#fbeee9", edgecolor="none"))
                ax.text(x, y, "empty", ha="center", va="center", fontsize=6.5, color="#b4472f")
        v = np.sort([T[i] for x in A.values() for i in x])[::-1]
        print(f"  {lab:18s} niches {len(A):2d}  pop mean {v.mean():.3f}  top128 {v[:128].mean():.3f}  "
              f"top10 {v[:10].mean():.3f}  best {v[0]:.2f}  best/niche (occupied) "
              f"{np.mean([max(T[i] for i in x) for x in A.values()]):.3f}")
        ax.set_xlabel("motors", color=INK2, fontsize=9)
    axs[1].set_ylabel("")
    fig.subplots_adjust(left=0.04, right=0.93, top=0.86, bottom=0.17, wspace=0.08)
    cax = fig.add_axes([0.945, 0.22, 0.008, 0.58])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=G.CMAP, norm=plt.Normalize(0, G.VMAX)), cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8, labelcolor=INK2)
    cb.set_label("goals / episode", color=INK2, fontsize=8.5)
    fig.savefig(out / f"greedy_grid_k{k}.png", dpi=180, facecolor="white")


def diversity(plt, out: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.6), facecolor="white")
    print("[greedy] diversity per generation")
    for k, ls in ((5, "-"),):                          # K = 5 only (the nominal setting)
        for lab, rule in (("MAP-Elites (ours)", "none"), ("Greedy", "global")):
            run = CAMP / f"k{k}_{rule}"
            reg = json.loads((run / "designs.json").read_text())
            par, memo = D.parents_of(run, reg), {}
            gens, niches, lineages = [], [], []
            for gd in sorted(run.glob("gen_[0-9][0-9][0-9][0-9]")):
                if not (gd / "manifest.json").exists():
                    continue
                hands = json.loads((gd / "manifest.json").read_text())["hands"]
                if any(h["id"] not in reg for h in hands):
                    continue
                gens.append(int(gd.name[4:]))
                niches.append(len({h["niche"] for h in hands}))
                lineages.append(len(Counter(D.root(h["id"], par, memo) for h in hands)))
            print(f"  K={k} {lab:18s} niches {niches}  lineages {lineages}")
            for ax, y in zip(axs, (niches, lineages)):
                ax.plot(gens, y, color=COL[lab], ls=ls, lw=2, marker="o", ms=6, mec="white", mew=1.2, label=lab)
    axs[0].set_ylabel("niches occupied (of 32)", color=INK2)
    axs[0].set_ylim(0, 34)
    axs[1].set_ylabel("distinct seed lineages", color=INK2)
    axs[1].set_yscale("log")
    axs[1].set_yticks([1, 10, 100, 1000, 4096])
    axs[1].set_yticklabels(["1", "10", "100", "1000", "4096"])
    axs[1].minorticks_off()
    for ax, t in zip(axs, ("Niche coverage (K = 5)", "Genetic diversity (K = 5)")):
        ax.set_title(t, loc="left", fontsize=11, fontweight="semibold", color=INK)
        ax.set_xlabel("generation", color=INK2)
        ax.grid(color="#e6e5e1", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c9c8c3")
        ax.tick_params(colors=INK2, length=0)
        ax.xaxis.get_major_locator().set_params(integer=True)
    axs[1].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "greedy_diversity.png", dpi=180, facecolor="white")


def train_peak(run: Path, g: int, min_ep: int = 20):
    """(peak, top-10 mean, partial): per-design training goals/episode over the last 25% of generation g's training
    (labels.json once the generation is done; else the per-episode dumps so far), designs with >= min_ep episodes."""
    lab = (CAMP / "hod/gen_0000/labels.json") if g == 0 else (run / f"gen_{g:04d}/labels.json")
    if lab.exists():
        v = np.sort([x[0] for x in json.loads(lab.read_text()).values() if x[1] >= min_ep])[::-1]
        return float(v[0]), float(v[:10].mean()), False
    if not (run / f"gen_{g:04d}/manifest.json").exists():
        return None, None, False
    R = sorted((REPO / "debug_outputs/train_logs/coevolve" / run.parent.name / run.name / f"gen_{g:04d}").glob(
        "*/design_stats"))
    fs = sorted(R[0].glob("step_*.npz")) if R else []
    if len(fs) < 13:                                      # in progress: shown once half of its 26 dumps exist
        return None, None, False
    use = fs[-max(1, int(np.ceil(0.25 * len(fs)))):]
    hoe = np.load(R[0] / "meta.npz")["hand_of_env"].astype(int)
    env = np.concatenate([np.load(f)["env"] for f in use]).astype(int)
    goals = np.concatenate([np.load(f)["goals"] for f in use])
    ep = np.bincount(hoe[env], minlength=hoe.max() + 1)
    gl = np.bincount(hoe[env], goals, minlength=hoe.max() + 1)
    v = np.sort(gl[ep >= min_ep] / ep[ep >= min_ep])[::-1]
    return float(v[0]), float(v[:10].mean()), True


MAX_B = 5                                                 # x-axis ends at 5B env interactions


def budget(plt, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.1), facecolor="white")
    print("[greedy] peak training goals/episode per design (last 25% of each generation; * = still training)")
    for k, ls in ((5, "-"),):                          # K = 5 only (the nominal setting)
        for lab, rule in (("MAP-Elites (ours)", "none"), ("Greedy", "global")):
            run = CAMP / f"k{k}_{rule}"
            xs, ys, hol = [], [], []
            for g in range(0, MAX_B):                      # budgets 1..MAX_B billion steps
                v, _, partial = train_peak(run, g)
                if v is None:
                    break
                xs.append(g + 1)
                ys.append(v)
                if partial:
                    hol.append((g + 1, v))
                    break
            print(f"  K={k} {lab:18s} " + "  ".join(f"{x}B {y:.2f}{'*' if (x, y) in hol else ''}" for x, y in zip(xs, ys)))
            ax.plot(xs, ys, color=COL[lab], ls=ls, lw=2, marker="o", ms=7, mec="white", mew=1.5, label=lab, zorder=3)
            if hol:
                ax.plot(*zip(*hol), ls="none", marker="o", ms=7, mfc="white", mec=COL[lab], mew=1.8, zorder=4)
            ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), xytext=(7, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_yticks([0.2, 0.5, 1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["0.2", "0.5", "1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_ylim(0.25, 60)
    xmax = int(max(max(l.get_xdata()) for l in ax.get_lines()))
    ax.set_xticks(range(1, xmax + 1))
    ax.set_xticklabels([f"{b}B" for b in range(1, xmax + 1)])
    ax.set_xlim(0.7, xmax + 0.6)
    ax.set_xlabel("environment interactions", color=INK2)
    ax.set_ylabel("peak training goals / episode\n(all SAPG blocks)", color=INK2)
    ax.set_title("Best design during training vs. interaction budget (K = 5)", loc="left", fontsize=11.5, fontweight="semibold",
                 color=INK)
    ax.grid(color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color("#c9c8c3")
    ax.tick_params(colors=INK2, length=0)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "greedy_budget.png", dpi=180, facecolor="white")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--budget-only", action="store_true", help="only the peak-vs-budget plot (fast)")
    args = ap.parse_args()
    out = Path(__file__).resolve().parent / "plots"
    out.mkdir(exist_ok=True)
    if not args.budget_only:
        grid(plt, args.k, out)
        diversity(plt, out)
    budget(plt, out)
    print(f"[greedy] wrote {out}/greedy_grid_k{args.k}.png and greedy_diversity.png")


if __name__ == "__main__":
    main()
