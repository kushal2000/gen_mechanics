"""Mutation distance from the population to gen-SHARPA, generation by generation.

    .venv_isaacsim/bin/python experiments/old_experiments/17sep_coevolution/analysis/plot_edit_distance.py [label]
"""
import json, pathlib, sys
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parent), "/share/portal/kk837/depthbasedRL/plot_figures"]
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _style import configure_rcparams, style_axis, COLORS

configure_rcparams()
label = sys.argv[1] if len(sys.argv) > 1 else "coevolution_v2_gen2k"
rows = json.load(open(f"debug_outputs/17sep_coevo_analysis/edit_distance_{label}.json"))
OUT = pathlib.Path(__file__).resolve().parent.parent / "plots"; OUT.mkdir(exist_ok=True)
G = np.array([r["gen"] for r in rows]); lo = np.array([r["total"] for r in rows])
med = np.array([r["median"] for r in rows]); reach = np.array([r["reachable"] for r in rows]) / 1024
HERO, FOIL, GREY = COLORS["play2win"], COLORS["play_only"], COLORS["separator"]
closed = int(G[np.argmax(reach == 0)]) if (reach == 0).any() else None

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax = axes[0]
ax.plot(G, med, color=FOIL, ls="--", label="Population median")
ax.plot(G, lo, color=HERO, label="Closest design")
ax.set_ylim(0, max(med) * 1.15); ax.set_xlim(G[0], G[-1])
ax.set_xlabel("Generation"); ax.set_ylabel("Mutations to gen-SHARPA")
ax.set_title("Distance", fontsize=11, pad=6); style_axis(ax)
ax2 = axes[1]
ax2.plot(G, 100 * reach, color=HERO)
ax2.set_ylim(0, 100); ax2.set_xlim(G[0], G[-1])
ax2.set_xlabel("Generation"); ax2.set_ylabel("Designs able to reach it (%)")
ax2.set_title("Reachability (25 mm palm)", fontsize=11, pad=6); style_axis(ax2)
if closed is not None:
    for a in (ax, ax2):
        a.axvline(closed, color=GREY, lw=0.8, ls=":")
    ax2.text(closed + 0.7, 55, f"closed off\n(gen {closed})", fontsize=7.5, color="#555", va="center")
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=8.5)
fig.tight_layout(w_pad=2.0)
p = OUT / f"edit_distance_{label}.png"; fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.1, facecolor="white")
print("wrote", p)
print(f"closest: gen 0 {lo[0]} edits -> gen {G[-1]} {lo[-1]}; median {med[0]:.0f} -> {med[-1]:.0f}; "
      f"reachable {100*reach[0]:.0f}% -> {100*reach[-1]:.0f}%" + (f", zero from gen {closed}" if closed is not None else ""))
