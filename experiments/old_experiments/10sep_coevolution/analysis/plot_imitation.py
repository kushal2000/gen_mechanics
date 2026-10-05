import glob, json, pathlib, warnings, numpy as np; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
def curve(ev):
    a = EventAccumulator(ev, size_guidance={"scalars": 0}); a.Reload(); return np.array([x.value for x in a.Scalars("rewards/iter")])
def smooth(v, w=50):
    c = np.cumsum(np.insert(v, 0, 0.0)); return np.array([(c[i+1]-c[max(0,i-w+1)])/(i+1-max(0,i-w+1)) for i in range(len(v))])
caps = curve(sorted(glob.glob("debug_outputs/train_logs/depth_d64/*_786519_*/rank_0/*/summaries/events*"))[-1])
R = pathlib.Path("assets/populations/coevo_r500_v1"); gens, bounds = [], []
for g in sorted(R.glob("gen_*"), key=lambda p: int(p.name.split("_")[1])):
    jid = (g/"job_id.txt").read_text().strip() if (g/"job_id.txt").exists() else None
    evs = sorted(glob.glob(f"debug_outputs/train_logs/coevo/*_{jid}_*/rank_0/*/summaries/events*")) if jid else []
    if evs: gens.append(curve(evs[-1])); bounds.append(sum(len(x) for x in gens))
coevo = np.concatenate(gens); sk, sc = smooth(caps), smooth(coevo); n = max(len(caps), len(coevo))
SURF, INK, INK2, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#eb6834"
fig, ax = plt.subplots(figsize=(11, 6.4), dpi=150, facecolor=SURF); ax.set_facecolor(SURF)
for s in ("top","right"): ax.spines[s].set_visible(False)
for s in ("left","bottom"): ax.spines[s].set_color(GRID)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(colors=INK2, labelsize=9, length=0)
ymax = max(sk.max(), sc.max()) * 1.10
for i, b in enumerate(bounds[:-1]):
    ax.axvline(b, color=GRID, lw=1.0, zorder=1); ax.text(b, ymax*0.985, f"gen {i+1}", ha="center", va="top", fontsize=8, color=INK2)
ax.plot(np.arange(1,len(caps)+1), sk, color=BLUE, lw=2.0, zorder=3,
        label="Imitation — one hand, SHARPA rebuilt from the design grammar (21 joints), no search")
ax.plot(np.arange(1,len(coevo)+1), sc, color=ORANGE, lw=2.0, zorder=3,
        label="Evolution — 1024 designs, rank + keep top 50% + mutate every 2000 epochs")
for b in bounds[:-1]: ax.plot([b],[sc[b-1]], marker="o", ms=5.5, mfc=SURF, mec=ORANGE, mew=1.6, zorder=4)
ax.annotate(f"capsule SHARPA  {sk[-1]:,.0f}", (len(caps), sk[-1]), xytext=(8,0), textcoords="offset points", va="center", fontsize=9.5, color=INK, fontweight="semibold")
ax.annotate(f"co-evolution  {sc[-1]:,.0f}", (len(coevo), sc[-1]), xytext=(8,0), textcoords="offset points", va="center", fontsize=9.5, color=INK, fontweight="semibold")
ax.set_ylim(0, ymax); ax.set_xlim(0, n*1.16)
ax.set_xlabel("training epochs   (1 epoch = 24,576 envs × 16 steps ≈ 0.39 M env steps)", color=INK2, fontsize=9.5)
ax.set_ylabel("mean episode return   (50-epoch rolling mean)", color=INK2, fontsize=9.5)
fig.suptitle("Imitation can sidestep the need for evolution", x=0.125, ha="left", color=INK, fontsize=13.5, fontweight="semibold", y=0.975)
ax.set_title("Same d64/L4 policy, same task, same hyperparameters. The imitated hand is one design authored through the same "
             "pipeline as every evolved one;\nit needs no population and no selection. Open circles are the evolution loop's selection events.",
             loc="left", color=INK2, fontsize=8.8, pad=10)
leg = ax.legend(loc="upper left", frameon=False, fontsize=9, bbox_to_anchor=(0.36, 0.60))
for t in leg.get_texts(): t.set_color(INK)
fig.tight_layout(rect=(0,0,1,0.95))
out = pathlib.Path("debug_outputs/10sep_coevo_analysis/imitation_vs_evolution.png"); fig.savefig(out, facecolor=SURF, bbox_inches="tight")
print(f"capsule {len(caps)} ep last {sk[-1]:.0f}; coevo {len(coevo)} ep last {sc[-1]:.0f}; wrote {out}")
