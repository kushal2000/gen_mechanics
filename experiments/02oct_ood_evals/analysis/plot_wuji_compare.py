"""Wuji v2, in-distribution vs out-of-distribution, in the training metric (goals per episode, cap 50).

Two comparisons, each the latest unified policy against one Wuji-only checkpoint (vs_wuji_ep1600: nearest in
skill; vs_wuji_ep3600: converged), as grouped bars over the nominal condition and the 8 OOD conditions;
absolute and relative to each policy's own nominal. One figure per plot, PDF + PNG, paper style.

Goals/episode is each env's first episode, ended by a drop, 10 s without a goal, or the 50-goal cap;
1024 envs x 240 s per condition (results/wuji_compare_cap50, run_hand_uniform.sub MAX_GOALS=50).

    .venv_isaacsim/bin/python experiments/02oct_ood_evals/analysis/plot_wuji_compare.py
    -> plots/wuji_compare_cap50/<comparison>/id_vs_ood_{goals,relative}.{pdf,png}
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from matplotlib.patches import Patch  # noqa: E402

import paper_style as ps  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
RES = HERE / "results/wuji_compare_cap50"
OUT = HERE / "plots/wuji_compare_cap50"
CONDS = [("nominal", "Nominal"), ("cube40", "40 mm"), ("cube55", "55 mm"), ("cube65", "65 mm"),
         ("light", "Light"), ("heavy", "Heavy"), ("slippery", "Slippery"), ("push", "Push"),
         ("push_hard", "Hard\npush")]
COMPARISONS = {
    "vs_wuji_ep1600": ("Wuji Only", "wuji_only_ep_1600"),
    "vs_wuji_ep3600": ("Wuji Only", "wuji_only_ep_3600"),
    # Matched compute: both policies after ~10k epochs of 12288 envs (Wuji-only 908714 ep 6400 = 10162 total).
    "vs_wuji_10k": ("Wuji Only", "wuji_only_10k", "unified_final"),
}


def latest_unified():
    """The newest unified checkpoint whose evals are complete: the 20k-epoch fine-tune, else the 10k final,
    else ep 7827."""
    for label in ("unified_20k", "unified_final"):
        d = RES / label
        if d.exists() and len(list(d.glob("*.json"))) >= len(CONDS):
            return "Unified Multi-Embodiment", label
    return "Unified Multi-Embodiment", "unified_ep_7827"


def load(label):
    out = {}
    for c, _ in CONDS:
        f = RES / label / f"wuji2__{c}.json"
        out[c] = json.loads(f.read_text())["goals_per_episode_first"] if f.exists() else np.nan
    return out


def plot(comp, relative):
    uni = ("Unified Multi-Embodiment", COMPARISONS[comp][2]) if len(COMPARISONS[comp]) > 2 else latest_unified()
    pols = [(*uni, ps.COLORS["hero"]), (*COMPARISONS[comp][:2], ps.COLORS["foil"])]
    n, k = len(CONDS), len(pols)
    x = np.arange(n) + np.where(np.arange(n) > 0, 0.5, 0.0)       # a gap after the in-distribution bar
    w = 0.8 / k
    fig, ax = plt.subplots(figsize=(ps.TEXT_WIDTH * 0.62, 2.3))
    handles = []
    for i, (txt, label, color) in enumerate(pols):
        v = load(label)
        y = np.array([v[c] for c, _ in CONDS])
        if relative:
            y = y / v["nominal"]
        xs = x + (i - (k - 1) / 2) * w
        h = ax.bar(xs, np.nan_to_num(y), w * 0.9, color=color, alpha=ps.BAR_ALPHA, linewidth=0, label=txt)
        handles.append(h)
        for xi, yi in zip(xs, y):                       # value on top of each bar, as the reference does
            if np.isfinite(yi):
                ax.text(xi, yi, f"{yi:.2f}" if relative else f"{yi:.0f}", ha="center", va="bottom", fontsize=6,
                        color=ps.COLORS["ink"])
    ax.axvline((x[0] + x[1]) / 2, color=ps.COLORS["neutral"], linewidth=0.7, linestyle=(0, (3, 2)))
    if relative:
        ax.axhline(1.0, color=ps.COLORS["neutral"], linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
        ax.set_ylim(0, 1.2)
    else:
        ax.set_ylim(0, 52)
    ps.style_axis(ax)
    ax.tick_params(axis="x", length=0)
    ax.set_xticks(x, [lab for _, lab in CONDS])
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_ylabel("Relative goals / episode" if relative else "Goals / episode")
    ax.set_title("In- vs Out-of-Distribution", fontweight="bold")
    ps.bottom_legend(fig, [Patch(facecolor=c, alpha=ps.BAR_ALPHA) for _, _, c in pols],
                     [t for t, _, _ in pols], y=-0.12, columnspacing=1.6, bold=("Unified Multi-Embodiment",))
    paths = ps.save_figure(fig, OUT / comp / f"id_vs_ood_{'relative' if relative else 'goals'}")
    plt.close(fig)
    return paths


def main():
    ps.configure()
    for comp in COMPARISONS:
        for rel in (False, True):
            plot(comp, rel)
    for txt, label, *_ in [latest_unified(), *COMPARISONS.values()]:
        v = load(label)
        print(f"  {txt:30s} " + " ".join(f"{c} {v[c]:.1f}" for c, _ in CONDS))


if __name__ == "__main__":
    main()
