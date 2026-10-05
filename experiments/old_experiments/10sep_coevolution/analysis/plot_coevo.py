import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import pathlib, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from curves import baseline, coevo as load_coevo, blank_restarts, smooth, WARMUP, epoch_hour_ticks
base, _, bseg = baseline(); coevo, _, bounds, sel = load_coevo()
base = blank_restarts(base, bseg[:-1]); coevo = blank_restarts(coevo, bounds[:-1])
sb, sc = smooth(base), smooth(coevo); n = max(len(base), len(coevo))
SURF, INK, INK2, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#eb6834"
fig, ax = plt.subplots(figsize=(11, 6.4), dpi=150, facecolor=SURF); ax.set_facecolor(SURF)
for s in ("top","right"): ax.spines[s].set_visible(False)
for s in ("left","bottom"): ax.spines[s].set_color(GRID)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(colors=INK2, labelsize=9, length=0)
ymax = max(np.nanmax(sb), np.nanmax(sc)) * 1.10
for i, b in enumerate(bounds[:-1]):
    if (i+1) % 5 == 0:                       # a line every fifth generation; circles mark them all
        ax.axvline(b, color=GRID, lw=1.0, zorder=1); ax.text(b, ymax*0.985, f"gen {i+1}", ha="center", va="top", fontsize=8, color=INK2)
for b in bseg[:-1]: ax.axvline(b, color=BLUE, lw=0.8, ls=(0,(2,3)), alpha=0.5, zorder=1)
ax.plot(np.arange(1,len(base)+1), sb, color=BLUE, lw=2.0, zorder=3, label="Baseline — 1024 fixed designs, continued across 24 h links (dotted)")
ax.plot(np.arange(1,len(coevo)+1), sc, color=ORANGE, lw=2.0, zorder=3, label="Co-evolution V1 — every 2000 epochs: rank, keep top 50%, mutate to refill, carry the policy")
for b in bounds[:-1]: ax.plot([b],[sc[b-1]], marker="o", ms=4.5, mfc=SURF, mec=ORANGE, mew=1.4, zorder=4)
sbl = sb[np.isfinite(sb)][-1]; scl = sc[np.isfinite(sc)][-1]
ax.annotate(f"baseline  {sbl:,.0f}", (len(base), sbl), xytext=(8,0), textcoords="offset points", va="center", fontsize=9.5, color=INK, fontweight="semibold")
ax.annotate(f"co-evolution  {scl:,.0f}", (len(coevo), scl), xytext=(8,0), textcoords="offset points", va="center", fontsize=9.5, color=INK, fontweight="semibold")
ax.set_ylim(0, ymax); ax.set_xlim(0, n*1.14)
epoch_hour_ticks(ax, ink=INK2)
ax.set_ylabel("mean episode return   (50-epoch rolling mean)", color=INK2, fontsize=9.5)
fig.suptitle("Fixed population vs co-evolved — same 1024 round-500 designs, same d64/L4 policy",
             x=0.125, ha="left", color=INK, fontsize=12.5, fontweight="semibold", y=0.975)
ax.set_title(f"Open circles are selection events. Co-evolved joints/hand {sel[0]['mean_joints_before']:.1f} → {sel[-1]['mean_joints_after']:.1f} "
             f"over {len(sel)} selections. Co-evolution stopped at its planned 40 generations; the baseline is still running.\n"
             "Returns are not on one scale: the reward tightens with the success tolerance, and the two runs tightened at different times (see the tolerance chart).\n"
             f"The first {WARMUP} epochs after every restart are blanked: rl_games refills its 100-game reward average from the first, failed episodes of a fresh process.",
             loc="left", color=INK2, fontsize=8.8, pad=10)
leg = ax.legend(loc="upper left", frameon=False, fontsize=9, bbox_to_anchor=(0.0, 0.90))
for t in leg.get_texts(): t.set_color(INK)
fig.tight_layout(rect=(0,0,1,0.95))
out = pathlib.Path("debug_outputs/10sep_coevo_analysis/baseline_vs_coevo_v1.png"); fig.savefig(out, facecolor=SURF, bbox_inches="tight"); print("wrote", out)
