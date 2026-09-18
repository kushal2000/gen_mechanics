"""Joints per hand across the co-evolved generations, paper style."""
import pathlib, sys; sys.path[:0] = [str(pathlib.Path(__file__).resolve().parent), "/share/portal/kk837/depthbasedRL/plot_figures"]
import json, pathlib, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _style import configure_rcparams, style_axis, COLORS
from hand_sampler import population_io

configure_rcparams()
ROOT = pathlib.Path("assets/populations/coevo_r500_v1"); OUT = pathlib.Path("debug_outputs/10sep_coevo_analysis")
CACHE = OUT / ".curve_cache" / "joints_by_gen.json"
gens = sorted(ROOT.glob("gen_*"), key=lambda p: int(p.name.split("_")[1]))
rows = json.load(open(CACHE)) if CACHE.exists() else {}
for g in gens:
    if g.name not in rows:
        hands = population_io.load_population(g / "population.json")
        rows[g.name] = {"joints": [h.n_joints for h in hands], "fingers": [h.n_fingers for h in hands]}
CACHE.write_text(json.dumps(rows))
G = np.array([int(g.name.split("_")[1]) for g in gens])
J = np.array([rows[g.name]["joints"] for g in gens], float)      # (gens, 1024)
F = np.array([rows[g.name]["fingers"] for g in gens], float)
mean, lo, hi = J.mean(1), np.percentile(J, 25, axis=1), np.percentile(J, 75, axis=1)
print("gen 0 joints mean %.1f  gen %d mean %.1f  IQR %.0f-%.0f  fingers %.2f -> %.2f" % (mean[0], G[-1], mean[-1], lo[-1], hi[-1], F[0].mean(), F[-1].mean()))

HERO, FOIL = COLORS["play2win"], COLORS["play_only"]
fig, ax = plt.subplots(figsize=(3.6, 2.7))
# The fixed population is generation 0 held constant, so its band is gen 0's IQR.
ax.fill_between(G, lo, hi, color=HERO, alpha=0.18, lw=0)
ax.fill_between([0, G[-1]], lo[0], hi[0], color=FOIL, alpha=0.18, lw=0)
ax.plot(G, mean, color=HERO, label="Co-evolution")
ax.axhline(mean[0], color=FOIL, ls="--", label="Fixed population")
ax.set_xlim(0, G[-1]); ax.set_xticks(range(0, G[-1] + 1, 10))
ax.set_ylim(5, 20); ax.set_yticks([5, 10, 15, 20])
ax.set_xlabel("Generation"); ax.set_ylabel("Joints per hand\n(mean, IQR band)")
style_axis(ax)
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False,
           handlelength=1.8, handletextpad=0.5, columnspacing=1.4, fontsize=8.5)
fig.tight_layout()
p = OUT / "coevo_clean_joints.png"; fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.1, facecolor="white"); print("wrote", p)
