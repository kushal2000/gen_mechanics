"""coevolution_v1 in the layout of DERL's Figure 2 (Gupta et al. 2021):

  a  fitness of the top 100 designs per generation
  b  each lineage that ever mattered: peak abundance vs its founder's initial
     rank, dot size = beneficial mutations accrued
  c  phylogenetic tree: one dot per distinct design, size = descendants,
     opacity = fitness rank in its generation; radial, born-at-generation rings
  d  Muller diagram of the 10 largest lineages, opacity = fitness, stars =
     topology-changing mutations that founded a sub-lineage of > 20 designs

    .venv_isaacsim/bin/python experiments/10sep_coevolution/analysis/derl_fig2.py
"""
import sys; sys.path[:0] = ["/share/portal/kk837/depthbasedRL/plot_figures"]
import collections, glob, json, pathlib
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from _style import configure_rcparams, style_axis, COLORS
from coevolution.design_rewards import merge_rank_files
from hand_sampler import population_io

configure_rcparams()
R = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "assets/populations/coevo_r500_v1")
OUT = pathlib.Path("debug_outputs/10sep_coevo_analysis"); TAG = R.name
HERO = COLORS["play2win"]; TOPO = {"add_finger", "remove_finger", "split_link", "merge_links"}

# --- data -------------------------------------------------------------------------
gens = sorted(int(p.name.split("_")[1]) for p in R.glob("gen_*") if glob.glob(str(p / "design_rewards_rank*.json")))
G = gens[-1]; N = 1024
ret = {g: np.array([merge_rank_files(sorted(glob.glob(str(R / f"gen_{g}/design_rewards_rank*.json"))))[i]["return_mean"] for i in range(N)]) for g in gens}
pct = {g: np.argsort(np.argsort(ret[g])) / (N - 1) for g in gens}            # fitness rank within generation, 0..1
parent, op = {}, {}
for g in range(1, G + 1):
    s = json.load(open(R / f"gen_{g}/selection.json")); parent[g] = s["survivors"] + [c["parent"] for c in s["children"]]
    op[g] = [None] * len(s["survivors"]) + [c["op"] for c in s["children"]]
anc = {0: list(range(N))}
for g in range(1, G + 1): anc[g] = [anc[g - 1][p] for p in parent[g]]
share = np.zeros((G + 1, N))
for g in range(G + 1):
    for a in anc[g]: share[g, a] += 1.0 / N

# distinct designs: a node is born by a mutation (or at gen 0) and lives while it survives
node_of = {0: list(range(N))}; birth = {n: (0, n) for n in range(N)}; node_parent = {n: None for n in range(N)}; node_op = {}
nxt = N
for g in range(1, G + 1):
    node_of[g] = []
    for i, p in enumerate(parent[g]):
        if op[g][i] is None: node_of[g].append(node_of[g - 1][p])
        else:
            node_of[g].append(nxt); birth[nxt] = (g, i); node_parent[nxt] = node_of[g - 1][p]; node_op[nxt] = op[g][i]; nxt += 1
n_nodes = nxt
kids = collections.defaultdict(list)
for n in range(N, n_nodes): kids[node_parent[n]].append(n)
desc = np.zeros(n_nodes, int)
for n in range(n_nodes - 1, -1, -1):
    desc[n] = sum(desc[k] + 1 for k in kids[n])
fit = np.array([pct[birth[n][0]][birth[n][1]] for n in range(n_nodes)])        # fitness at first evaluation
lineage = np.zeros(n_nodes, int)
for n in range(n_nodes): lineage[n] = n if n < N else lineage[node_parent[n]]

def save(fig, name):
    p = OUT / f"{name}.png"; fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.1, facecolor="white"); plt.close(fig); print("wrote", p)

# --- a. fitness of the top 100 --------------------------------------------------------
def panel_a(ax):
    top = [np.sort(ret[g])[-100:].mean() for g in gens]
    ax.plot(gens, top, color=HERO); ax.fill_between(gens, [np.sort(ret[g])[-100:].min() for g in gens], [ret[g].max() for g in gens], color=HERO, alpha=0.15, lw=0)
    ax.set_xlabel("Generation"); ax.set_ylabel("Return of top 100 designs"); ax.set_xlim(0, G); style_axis(ax)

# --- b. peak abundance vs initial rank ---------------------------------------------------
def panel_b(ax):
    peak = share.max(0); init_rank = 1 - pct[0]                                   # 0 = best at gen 0
    benef = np.zeros(N, int)
    for g in range(1, G + 1):
        for i in range(512, N):
            if ret[g][i] > ret[g][i - 512]: benef[anc[g][i]] += 1                # child i sits beside its parent i-512
    keep = peak > 1.5 / N                                                          # lineages that ever grew past a founder + child
    sz = lambda b: 5 + 4.0 * np.sqrt(b)
    ax.scatter(init_rank[keep], peak[keep], s=sz(benef[keep]), color=HERO, alpha=0.55, linewidths=0)
    ax.set_yscale("log"); ax.set_xlim(0, 1); ax.set_ylim(1 / N, 1.5)
    ax.set_xlabel("Initial rank of founder (0 = best)"); ax.set_ylabel("Peak share of population"); style_axis(ax)
    for b in (1, 10, 100, 1000):
        ax.scatter([], [], s=sz(b), color=HERO, alpha=0.55, linewidths=0, label=str(b))
    ax.legend(title="beneficial\nmutations", frameon=False, fontsize=7, title_fontsize=7, loc="upper right", labelspacing=1.2)

# --- c. radial phylogenetic tree ----------------------------------------------------------
def panel_c(ax):
    # angle by DFS over the whole forest; roots get sectors proportional to their subtree
    sys.setrecursionlimit(100000)
    leaves = 0; ang = np.zeros(n_nodes)
    order = sorted(range(N), key=lambda n: -desc[n])
    stack = []
    def dfs(n):
        nonlocal leaves
        ch = kids[n]
        if not ch: ang[n] = leaves; leaves += 1; return
        for k in ch: dfs(k)
        ang[n] = np.mean([ang[k] for k in ch])
    for r in order: dfs(r)
    theta = 2 * np.pi * ang / max(leaves, 1)
    rad = np.array([birth[n][0] for n in range(n_nodes)], float) + 1.0
    x, y = rad * np.cos(theta), rad * np.sin(theta)
    segs = [[(x[node_parent[n]], y[node_parent[n]]), (x[n], y[n])] for n in range(N, n_nodes)]
    ax.add_collection(LineCollection(segs, colors=(0.5, 0.5, 0.5, 0.25), linewidths=0.25))
    base = np.array(to_rgb(HERO)); rgba = np.column_stack([np.tile(base, (n_nodes, 1)), 0.15 + 0.85 * fit])
    ax.scatter(x, y, s=1.0 + 0.6 * np.sqrt(desc), c=rgba, linewidths=0, zorder=3)
    ax.set_aspect("equal"); ax.axis("off"); lim = G + 2; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

# --- d. Muller diagram -------------------------------------------------------------------
def panel_d(ax):
    peak = share.max(0); top = list(np.argsort(-peak)[:10])
    death = [next((g for g in range(1, G + 1) if share[g, a] == 0), G + 1) for a in top]
    top = [a for _, a in sorted(zip(death, top), reverse=True)]
    palette = ["#2C7BB6", "#E08214", "#1A9850", "#D73027", "#7B3294", "#4D9221", "#C51B7D", "#B35806", "#01665E", "#8C510A"]
    # lineage fitness per generation: mean rank percentile of its members
    lfit = np.zeros((G + 1, N))
    for g in range(G + 1):
        acc = collections.defaultdict(list)
        for i, a in enumerate(anc[g]): acc[a].append(pct[g][i])
        for a, v in acc.items(): lfit[g, a] = np.mean(v)
    xs = np.arange(G + 1); base = np.zeros(G + 1)
    other = share.sum(1) - share[:, top].sum(1); base = base + other        # white space for the rest
    for k, a in enumerate(top):
        col = np.array(to_rgb(palette[k]))
        for g in range(G):
            alpha = 0.35 + 0.65 * lfit[g, a]
            ax.fill_between([g, g + 1], [base[g], base[g + 1]], [base[g] + share[g, a], base[g + 1] + share[g + 1, a]], color=(*col, alpha), lw=0)
        base = base + share[:, a]
    # stars: topology mutations that founded a sub-lineage of > 20 designs, placed at the birth generation inside the lineage band
    sx, sy = [], []
    cum = {}
    b = np.zeros(G + 1) + other
    for k, a in enumerate(top):
        cum[a] = (b.copy(), b + share[:, a]); b = b + share[:, a]
    STAR_MIN = 100
    for n in range(N, n_nodes):
        if node_op[n] in TOPO and desc[n] > STAR_MIN and lineage[n] in cum:
            g, i = birth[n]; lo, hi = cum[lineage[n]]
            members = [j for j, a in enumerate(anc[g]) if a == lineage[n]]      # the band's members this generation
            frac = (members.index(i) + 0.5) / len(members)
            sx.append(g); sy.append(lo[g] + frac * (hi[g] - lo[g]))
    ax.scatter(sx, sy, marker="*", s=18, color="white", edgecolors="#222", linewidths=0.4, zorder=4)
    ax.set_xlim(0, G); ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_xlabel("Generation"); ax.set_ylabel("Relative abundance"); style_axis(ax)
    print(f"[d] {len(sx)} topology mutations with > {STAR_MIN} descendants among the top-10 lineages")

fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], hspace=0.45, wspace=0.35)
axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]); axc, axd = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
panel_a(axa); panel_b(axb); panel_c(axc); panel_d(axd)
for ax, lab in ((axa, "a"), (axb, "b"), (axc, "c"), (axd, "d")):
    ax.text(-0.12, 1.06, lab, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
save(fig, f"derl_fig2_{TAG}")
for name, fn, size in (("a", panel_a, (3.4, 2.6)), ("b", panel_b, (3.4, 2.6)), ("c", panel_c, (4.0, 4.0)), ("d", panel_d, (3.6, 2.7))):
    f, ax = plt.subplots(figsize=size); fn(ax); f.tight_layout(); save(f, f"derl_fig2{name}_{TAG}")
print(f"{n_nodes} distinct designs over {G + 1} generations")
