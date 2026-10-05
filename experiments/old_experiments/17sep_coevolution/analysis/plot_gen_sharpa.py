"""gen-SHARPA against the population it was never part of.

For each generation the co-evolution run kept a checkpoint, this draws:
  * what the POPULATION scored at the end of that generation -- the mean of
    rewards/iter (and of successes, its goals per episode) over its last 100
    epochs, i.e. the same endpoint policy that the checkpoint holds, not the
    whole generation's average
  * what gen-SHARPA scored under that same checkpoint, zero-shot, from
    eval_one_hand.py

Both are at that generation's own success tolerance, which the second panel
shows, because a return is only comparable against another return measured at
the same tolerance.

THE TWO LINES COME FROM DIFFERENT INSTRUMENTS. The population is training's own
logged return; gen-SHARPA is eval_one_hand.py, which reads about 37 % low (see
its docstring). The shape of each line is sound and so is the divergence
between them, but the ratio is overstated. The one like-for-like measurement --
generation 16, both scored by the harness in the same run -- puts gen-SHARPA at
208 against a population mean of 1528, i.e. 14 %, and ranks it 1024th of 1024,
below the worst design in the population. Closing this properly means scoring
the population with the harness at every generation (eval_population.py).

    .venv_isaacsim/bin/python experiments/old_experiments/17sep_coevolution/analysis/plot_gen_sharpa.py [label]
"""
import json, pathlib, sys
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parent), "/share/portal/kk837/depthbasedRL/plot_figures"]
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from _style import configure_rcparams, style_axis, COLORS
import curves as C

configure_rcparams()
label = sys.argv[1] if len(sys.argv) > 1 else "coevolution_v2_gen2k"
rows = json.load(open(f"debug_outputs/17sep_coevo_analysis/one_hand_{label}_sharpa_capsule.json"))
OUT = pathlib.Path(__file__).resolve().parent.parent / "plots"; OUT.mkdir(exist_ok=True)
POP, IMIT, GREY = COLORS["play2win"], "#1A9850", COLORS["separator"]

r, t, bounds, _ = C.coevo(label)
JOBS = C.generation_jobs(label)
def endpoint(g, n=100):
    """The population's return over the last ``n`` epochs of generation g."""
    lo = 0 if g == 0 else bounds[g - 1]
    seg = r[lo:bounds[g]]
    return float(np.mean(seg[-n:])) if len(seg) else np.nan


def endpoint_goals(g, n=100):
    """The population's goals per episode at the end of generation g."""
    v = C.scalar(JOBS[g], "successes")
    return float(np.mean(v[-n:])) if len(v) else np.nan

G = np.array([x["gen"] for x in rows])
sharpa = np.array([x["ret"] for x in rows]); goals = np.array([x["goals"] for x in rows])
tol = np.array([x["tolerance"] for x in rows])
pop = np.array([endpoint(g) for g in G]); pop_goals = np.array([endpoint_goals(g) for g in G])

def ticks(a):
    """Label every 5th generation, tick every 1: the divergence sits between
    gens 3 and 9, which a 10-wide tick spacing hides."""
    a.xaxis.set_major_locator(MultipleLocator(5))
    a.xaxis.set_minor_locator(MultipleLocator(1))
    a.tick_params(axis="x", which="minor", length=2, width=0.6, colors=COLORS["spine"])

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax = axes[0]
ax.plot(G, pop, color=POP, marker="o", ms=2.6, mew=0, label="Population (end of generation)")
ax.plot(G, sharpa, color=IMIT, ls="-.", marker="o", ms=2.6, mew=0, label="gen-SHARPA, zero-shot")
ax.set_xlabel("Generation"); ax.set_ylabel("Mean episode return")
ax.set_title("Return under the same checkpoint", fontsize=11, pad=6)
ax.set_xlim(G[0], G[-1]); ax.set_ylim(0, None); style_axis(ax); ticks(ax)
ax2 = axes[1]
ax2.plot(G, pop_goals, color=POP, marker="o", ms=2.6, mew=0, label="Population")
ax2.plot(G, goals, color=IMIT, ls="-.", marker="o", ms=2.6, mew=0, label="gen-SHARPA, zero-shot")
ax2.set_xlabel("Generation"); ax2.set_ylabel("Goals per episode"); ax2.set_ylim(0, None)
ax2.set_title("Goals under the same checkpoint", fontsize=11, pad=6); style_axis(ax2)
ax2.set_xlim(G[0], G[-1]); ticks(ax2)   # same span as the left panel
ax3 = ax2.twinx()
ax3.plot(G, tol, color=GREY, ls=":", lw=1.2, label="Success tolerance")
ax3.set_ylabel("Tolerance (m)"); ax3.set_ylim(0, 0.08)
ax3.spines["top"].set_visible(False); ax3.spines["left"].set_visible(False)
ax3.spines["right"].set_color(COLORS["spine"]); ax3.tick_params(colors=COLORS["spine"], length=3, width=0.8)
h = [*axes[0].get_legend_handles_labels()[0], *ax3.get_legend_handles_labels()[0]]
l = [*axes[0].get_legend_handles_labels()[1], *ax3.get_legend_handles_labels()[1]]
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=8.5)
fig.tight_layout(w_pad=2.4)
p = OUT / f"gen_sharpa_vs_population_{label}.png"
fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.1, facecolor="white"); print("wrote", p)
for g, s, pp, gl, pg, tt in zip(G, sharpa, pop, goals, pop_goals, tol):
    print(f"  gen {g:3d}  tol {tt:.4f}  return {pp:7.0f} vs {s:6.0f} ({100*s/pp:4.0f}%)   "
          f"goals/ep {pg:5.2f} vs {gl:5.2f} ({100*gl/max(pg,1e-9):4.0f}%)")
