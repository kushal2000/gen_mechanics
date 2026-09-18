"""Phylogeny of coevolution_v1: how one family took the population.

Three figures, paper style:
  phylo_muller.png     share of the population descending from each gen-0
                       ancestor, generation by generation (a Muller plot)
  phylo_genealogy.png  the genealogy of the final population -- every design
                       with a living descendant, as a tree over generations,
                       coloured by joints per hand; mutations marked
  phylo_diversity.png  lineages alive and the largest lineage's share

    .venv_isaacsim/bin/python experiments/10sep_coevolution/analysis/phylogeny.py
"""
import sys; sys.path[:0] = ["/share/portal/kk837/depthbasedRL/plot_figures"]
import collections, json, pathlib
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from _style import configure_rcparams, style_axis, COLORS
from hand_sampler import population_io

configure_rcparams()
R = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "assets/populations/coevo_r500_v1")
OUT = pathlib.Path("debug_outputs/10sep_coevo_analysis")
gens = sorted(int(p.name.split("_")[1]) for p in R.glob("gen_*") if (p / "population.json").exists())
G = gens[-1]; N = len(population_io.load_population(R / "gen_0/population.json"))

# --- parent pointers: parent[g][i] = index in generation g-1; op[g][i] = mutation or None
parent, op = {}, {}
for g in range(1, G + 1):
    s = json.load(open(R / f"gen_{g}/selection.json"))
    parent[g] = s["survivors"] + [c["parent"] for c in s["children"]]
    op[g] = [None] * len(s["survivors"]) + [c["op"] for c in s["children"]]
anc = {0: list(range(N))}                       # gen-0 ancestor of every (g, i)
for g in range(1, G + 1):
    anc[g] = [anc[g - 1][p] for p in parent[g]]
joints = {g: [h.n_joints for h in population_io.load_population(R / f"gen_{g}/population.json")] for g in gens}

HERO = COLORS["play2win"]

def save(fig, name):
    p = OUT / f"{name}.png"; fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.1, facecolor="white"); plt.close(fig); print("wrote", p)

# --- 1. Muller plot ------------------------------------------------------------
share = np.zeros((G + 1, N))
for g in range(G + 1):
    for a in anc[g]: share[g, a] += 1.0 / N
peak = share.max(0); order = np.argsort(-peak)
TOP = 12
top = list(order[:TOP])
# sort the shown lineages by when they die, so the survivor sits at the bottom
death = [next((g for g in range(G + 1) if share[g, a] == 0 and g > 0), G + 1) for a in top]
top = [a for _, a in sorted(zip(death, top), reverse=True)]
palette = ["#2C7BB6", "#E08214", "#1A9850", "#D73027", "#7B3294", "#4D9221", "#C51B7D", "#762A83",
           "#B35806", "#01665E", "#8C510A", "#5E3C99"]
fig, ax = plt.subplots(figsize=(7.0, 3.0))
x = np.arange(G + 1); base = np.zeros(G + 1)
other = share.sum(1) - share[:, top].sum(1)
ax.fill_between(x, base, base + other, color="#d9d9d9", lw=0, label=f"other {N - TOP} lineages")
base = base + other
for k, a in enumerate(top):
    ax.fill_between(x, base, base + share[:, a], color=palette[k % len(palette)], lw=0,
                    label=f"#{a}" if a == top[-1] else None)
    base = base + share[:, a]
ax.set_xlim(0, G); ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0", "50 %", "100 %"])
ax.set_xlabel("Generation"); ax.set_ylabel("Share of population")
style_axis(ax)
fix = next(g for g in range(G + 1) if share[g].max() >= 1.0 - 1e-9)
ax.annotate(f"lineage #{top[-1]} fixed at generation {fix}", (fix, 1.0), xytext=(fix + 1.5, 0.9), fontsize=8, ha="left",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="#333"))
ax.text(1, 0.05, f"{TOP} largest lineages coloured", fontsize=8, color="#333")
fig.tight_layout(); save(fig, "phylo_muller")

# --- 2. genealogy of the final population ----------------------------------------
alive = {G: set(range(N))}
for g in range(G, 0, -1):
    alive[g - 1] = {parent[g][i] for i in alive[g]}
children = {g: collections.defaultdict(list) for g in range(G + 1)}
for g in range(1, G + 1):
    for i in alive[g]:
        children[g - 1][parent[g][i]].append(i)
roots = sorted(alive[0])
y = {}; leaf_y = [0]
def place(g, i):
    kids = children[g].get(i, []) if g < G else []
    if not kids:
        y[(g, i)] = leaf_y[0]; leaf_y[0] += 1; return y[(g, i)]
    ys = [place(g + 1, k) for k in kids]
    y[(g, i)] = float(np.mean(ys)); return y[(g, i)]
sys.setrecursionlimit(10000)
for r in roots: place(0, r)
n_leaves = leaf_y[0]
segs, cols, mut_x, mut_y, mut_c = [], [], [], [], []
jmin, jmax = 2, 24; cmap = plt.get_cmap("viridis")
for g in range(1, G + 1):
    for i in alive[g]:
        p = parent[g][i]; y0, y1 = y[(g - 1, p)], y[(g, i)]
        segs.append([(g - 1, y0), (g - 1, y1)]); cols.append((0.6, 0.6, 0.6, 0.5))
        c = cmap((joints[g][i] - jmin) / (jmax - jmin))
        segs.append([(g - 1, y1), (g, y1)]); cols.append(c)
        if op[g][i] is not None:
            mut_x.append(g); mut_y.append(y1); mut_c.append(c)
fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.35))
ax.scatter(mut_x, mut_y, s=1.2, c=mut_c, linewidths=0, zorder=3)
ax.set_xlim(-0.5, G + 0.5); ax.set_ylim(-5, n_leaves + 5)
ax.set_xlabel("Generation"); ax.set_ylabel(f"{n_leaves} designs of generation {G}, by descent")
ax.set_yticks([]); style_axis(ax); ax.spines["left"].set_visible(False)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(jmin, jmax)); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02); cb.set_label("Joints per hand"); cb.outline.set_linewidth(0.6)
n_mut = len(mut_x); coal = min(g for g in range(G + 1) if len(alive[g]) == 1) if any(len(alive[g]) == 1 for g in range(G + 1)) else None
ax.set_title(f"Every design with a living descendant; dots are mutations ({n_mut:,} on surviving lines)"
             + (f"; all of generation {G} descends from one design of generation {coal}" if coal is not None else ""), fontsize=9, loc="left")
fig.tight_layout(); save(fig, "phylo_genealogy")

# --- 3. diversity ----------------------------------------------------------------
n_alive = [len(set(anc[g])) for g in range(G + 1)]; largest = [share[g].max() for g in range(G + 1)]
fig, ax = plt.subplots(figsize=(3.6, 2.7))
ax.plot(range(G + 1), n_alive, color=HERO, label="Gen-0 lineages alive")
ax.set_yscale("log"); ax.set_ylim(0.8, 2000); ax.set_yticks([1, 10, 100, 1000]); ax.set_yticklabels(["1", "10", "100", "1000"])
ax.set_xlim(0, G); ax.set_xlabel("Generation"); ax.set_ylabel("Lineages alive")
ax2 = ax.twinx(); ax2.plot(range(G + 1), largest, color=COLORS["play_only"], ls="--", label="Largest lineage's share")
ax2.set_ylim(0, 1.02); ax2.set_yticks([0, 0.5, 1]); ax2.set_yticklabels(["0", "50 %", "100 %"]); ax2.set_ylabel("Largest lineage")
style_axis(ax); ax2.spines["top"].set_visible(False); ax2.spines["left"].set_visible(False); ax2.spines["right"].set_color(COLORS["spine"]); ax2.tick_params(colors=COLORS["spine"], length=3, width=0.8)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False, fontsize=8.5)
fig.tight_layout(); save(fig, "phylo_diversity")
print(f"lineages alive by gen: {n_alive[:16]} ...; fixed at gen {fix}; mutations on surviving lines {n_mut}")
