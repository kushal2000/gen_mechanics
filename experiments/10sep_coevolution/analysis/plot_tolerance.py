import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import pathlib, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from curves import baseline, coevo, epoch_hour_ticks
br, bt, bseg = baseline(); cr, ct, bounds, sel = coevo()
SURF, INK, INK2, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#eb6834"
fig, ax = plt.subplots(figsize=(11, 6.4), dpi=150, facecolor=SURF); ax.set_facecolor(SURF)
for s in ("top","right"): ax.spines[s].set_visible(False)
for s in ("left","bottom"): ax.spines[s].set_color(GRID)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(colors=INK2, labelsize=9, length=0)
n = max(len(bt), len(ct)); top = 0.08
for i, b in enumerate(bounds[:-1]):
    if (i+1) % 5 == 0:                # a line every fifth generation; the circles mark them all
        ax.axvline(b, color=GRID, lw=0.8, zorder=1); ax.text(b, top*0.99, f"{i+1}", ha="center", va="top", fontsize=7.5, color=INK2)
for b in bseg[:-1]: ax.axvline(b, color=BLUE, lw=0.8, ls=(0,(2,3)), alpha=0.5, zorder=1)
floor_at = int(np.argmax(ct <= 0.01 + 1e-9)) + 1 if (ct <= 0.01 + 1e-9).any() else None
floor_gen = next((i+1 for i, b in enumerate(bounds) if b >= floor_at), None) if floor_at else None
ax.text(-n*0.005, top*0.99, "gen", ha="right", va="top", fontsize=7.5, color=INK2)
ax.axhline(0.01, color=GRID, lw=1.2, ls=(0, (4, 3)), zorder=1)
ax.text(n*0.005, 0.0108, "curriculum floor 0.01", va="bottom", fontsize=8, color=INK2)
ax.plot(np.arange(1, len(bt)+1), bt, color=BLUE, lw=2.0, zorder=3, label="Baseline — 1024 fixed designs, continued across 24 h links (dotted)")
ax.plot(np.arange(1, len(ct)+1), ct, color=ORANGE, lw=2.0, zorder=3, label="Co-evolution V1 — every 2000 epochs: rank, keep top 50%, mutate to refill, carry policy + curriculum")
for b in bounds[:-1]: ax.plot([b], [ct[b-1]], marker="o", ms=4.5, mfc=SURF, mec=ORANGE, mew=1.4, zorder=4)
ax.annotate(f"baseline  {bt[-1]:.4f}", (len(bt), bt[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9.5, color=INK, fontweight="semibold")
ax.annotate(f"co-evolution  {ct[-1]:.4f}  (floor)", (len(ct), ct[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9.5, color=INK, fontweight="semibold")
ax.set_ylim(0, top); ax.set_xlim(0, n*1.16)
epoch_hour_ticks(ax, ink=INK2)
ax.set_ylabel("goal success tolerance  (m)  — lower is a harder task, cleared", color=INK2, fontsize=9.5)
fig.suptitle("Fixed population vs co-evolved — how far each has tightened the success curriculum",
             x=0.125, ha="left", color=INK, fontsize=12.5, fontweight="semibold", y=0.975)
ax.set_title("The tolerance only tightens when the population is succeeding at the current one, so it is a cumulative record of "
             "task mastery that\nthe reward scale cannot hide. Starts at 0.075; floor 0.01. Open circles are selection events. "
             f"Co-evolved joints/hand {sel[0]['mean_joints_before']:.1f} → {sel[-1]['mean_joints_after']:.1f}."
             + (f"\nCo-evolution reached the floor at epoch {floor_at:,} (generation {floor_gen}) and held it through generation {len(bounds)}; the baseline is still running." if floor_at else ""),
             loc="left", color=INK2, fontsize=8.6, pad=10)
leg = ax.legend(loc="upper right", frameon=False, fontsize=8.8, bbox_to_anchor=(0.86, 0.90))
for t in leg.get_texts(): t.set_color(INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = pathlib.Path("debug_outputs/10sep_coevo_analysis/tolerance_baseline_vs_coevo.png"); fig.savefig(out, facecolor=SURF, bbox_inches="tight")
print(f"baseline {len(bt)} ep tol {bt[0]:.4f}->{bt[-1]:.4f}; coevo {len(ct)} ep over {len(bounds)} gens tol {ct[0]:.4f}->{ct[-1]:.4f}; wrote {out}")
