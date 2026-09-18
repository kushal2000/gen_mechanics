"""Pictures of lineage 521: the direct ancestry of the best gen-39 design, the
best hands of the final population, and the rival lineages it displaced.

    .venv_isaacsim/bin/python experiments/10sep_coevolution/analysis/lineage_figure.py
"""
import collections, json, math, pathlib, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from coevolution.design_rewards import merge_rank_files
from hand_sampler import population_io
from hand_sampler.viewer import draw

R = pathlib.Path("assets/populations/coevo_r500_v1"); OUT = pathlib.Path("debug_outputs/10sep_coevo_analysis")
LAST = 39                                            # last generation with rewards
pop = {g: population_io.load_population(R / f"gen_{g}/population.json") for g in range(LAST + 1)}
sel = {g: json.load(open(R / f"gen_{g}/selection.json")) for g in range(1, LAST + 1)}
rew = {g: merge_rank_files(sorted((R / f"gen_{g}").glob("design_rewards_rank*.json"))) for g in range(LAST + 1)}
src = {g: sel[g]["survivors"] + [c["parent"] for c in sel[g]["children"]] for g in sel}   # new index -> old index
op_of = {g: [None] * len(sel[g]["survivors"]) + [c["op"] for c in sel[g]["children"]] for g in sel}
rank = {g: {d: r for r, d in enumerate(sorted(rew[g], key=lambda d: -rew[g][d]["return_mean"]))} for g in rew}

def grid(items, out, cols, flex=0.0, title=None):
    rows = math.ceil(len(items) / cols)
    fig = plt.figure(figsize=(2.6 * cols, 2.7 * rows), dpi=170)
    for i, (hand, label) in enumerate(items):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d"); draw(ax, hand, label, flex=flex)
        ax.set_xlim(-0.11, 0.11); ax.set_ylim(-0.11, 0.11); ax.set_zlim(0.0, 0.22)   # these hands outgrow the viewer's default 16 cm box
    if title: fig.suptitle(title, fontsize=10, y=0.995)
    fig.tight_layout(pad=0.2); fig.savefig(out, facecolor="white", bbox_inches="tight"); plt.close(fig); print("wrote", out)

def label(g, i):
    h = pop[g][i]; s = rew[g][i]["return_mean"]; return f"gen {g}  #{i}  rank {rank[g][i]+1}\n{h.n_fingers}f {h.n_joints}j  return {s:,.0f}"

# --- 1. direct ancestry of the best gen-39 design --------------------------------
best = min(rew[LAST], key=lambda d: rank[LAST][d]); chain = []; g, i = LAST, best
while g > 0:
    chain.append((g, i, op_of[g][i])); i = src[g][i]; g -= 1
chain.append((0, i, None)); chain.reverse()
print(f"gen-39 best is #{best}; gen-0 ancestor #{chain[0][1]}")
muts = [(g, i, op) for g, i, op in chain if op]
print(f"{len(muts)} mutations on the path:", collections.Counter(op for _, _, op in muts))
# every point where the design changed, thinned to 12 panels, first and last always in
changes = [chain[0]] + muts
pick = changes if len(changes) <= 12 else [changes[round(k * (len(changes) - 1) / 11)] for k in range(12)]
grid([(pop[g][i], (f"gen {g}  {op}\n" if op else f"gen {g}  ancestor #{i}\n") + f"{pop[g][i].n_fingers}f {pop[g][i].n_joints}j  return {rew[g][i]['return_mean']:,.0f}")
      for g, i, op in pick], OUT / "lineage_ancestry.png", cols=4,
     title=f"Direct ancestry of the best gen-{LAST} design: {len(muts)} mutations from gen-0 #{chain[0][1]}")

# --- 2. the best of the final population, open and closed ------------------------
top = sorted(rew[LAST], key=lambda d: rank[LAST][d])[:8]
grid([(pop[LAST][i], label(LAST, i)) for i in top], OUT / "lineage_final_top8.png", cols=4, title=f"Top 8 of generation {LAST} by mean return")
grid([(pop[LAST][i], label(LAST, i)) for i in top], OUT / "lineage_final_top8_flexed.png", cols=4, flex=math.radians(60), title=f"Top 8 of generation {LAST}, every joint at 60°")

# --- 3. the rivals: best member of each of the top lineages at gen 6 ------------
anc = list(range(len(pop[0])))
for g in range(1, 7): anc = [anc[j] for j in src[g]]
by_lineage = collections.defaultdict(list)
for d in range(len(anc)): by_lineage[anc[d]].append(d)
lineages = sorted(by_lineage, key=lambda a: -len(by_lineage[a]))[:8]
items = []
for a in lineages:
    d = min(by_lineage[a], key=lambda d: rank[6][d]); h = pop[6][d]
    items.append((h, f"lineage {a}  ({len(by_lineage[a])} hands)\nbest: rank {rank[6][d]+1}  {h.n_fingers}f {h.n_joints}j  {rew[6][d]['return_mean']:,.0f}"))
grid(items, OUT / "lineage_rivals_gen6.png", cols=4, title="Generation 6: the eight largest lineages, each shown by its best member")
