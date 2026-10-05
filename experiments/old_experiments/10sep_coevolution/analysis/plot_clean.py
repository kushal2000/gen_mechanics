"""Paper-style versions of the baseline-vs-coevolution charts, in the
depthbasedRL/plot_figures style: serif, no grid, no annotations, frameless
legend under the axes. One combined 1x2 figure plus each panel alone."""
import pathlib, sys; sys.path[:0] = [str(pathlib.Path(__file__).resolve().parent), "/share/portal/kk837/depthbasedRL/plot_figures"]
import pathlib, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker as mt
from _style import configure_rcparams, style_axis, COLORS
from curves import baseline, coevo, imitation, blank_restarts, bridge, smooth, SEC_PER_EPOCH

configure_rcparams()
OUT = pathlib.Path("debug_outputs/10sep_coevo_analysis")
HERO, FOIL = COLORS["play2win"], COLORS["play_only"]     # co-evolution is the hero
br, bt, bseg = baseline(); cr, ct, bounds, sel = coevo(); ir, it = imitation()
ir = smooth(ir, 200)
IMIT = "#1A9850"          # the style module's green (Play2Win in the fig-2 draft)
br = smooth(bridge(blank_restarts(br, bseg[:-1])), 200); cr = smooth(bridge(blank_restarts(cr, bounds[:-1])), 200)   # heavier smoothing than the working charts: this one carries no annotations to read against
XMAX = 80000

def xaxis(ax):
    ax.set_xlim(0, XMAX)
    ax.xaxis.set_major_locator(mt.MultipleLocator(20000))
    ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda e, _: f"{e/1000:.0f}k\n{e*SEC_PER_EPOCH/3600:.0f} h"))
    ax.set_xlabel("Training epochs / wall-clock hours")

def reward(ax, with_imitation=False, with_baseline=True):
    if with_baseline: ax.plot(np.arange(1, len(br)+1), br, color=FOIL, ls="--", label="Fixed population")
    ax.plot(np.arange(1, len(cr)+1), cr, color=HERO, label="Co-evolution")
    if with_imitation: ax.plot(np.arange(1, len(ir)+1), ir, color=IMIT, ls="-.", lw=1.6, label="Imitation (one SHARPA-like hand)")
    ax.set_ylim(0, 4500); ax.set_yticks([0, 1000, 2000, 3000, 4000])
    ax.set_ylabel("Mean episode return"); xaxis(ax); style_axis(ax)

def tolerance(ax, with_imitation=False, with_baseline=True):
    if with_baseline: ax.plot(np.arange(1, len(bt)+1), bt, color=FOIL, ls="--", label="Fixed population")
    ax.plot(np.arange(1, len(ct)+1), ct, color=HERO, label="Co-evolution")
    if with_imitation: ax.plot(np.arange(1, len(it)+1), it, color=IMIT, ls="-.", lw=1.6, label="Imitation (one SHARPA-like hand)")
    ax.axhline(0.01, color=COLORS["separator"], lw=0.7, ls=":")
    ax.text(XMAX * 0.99, 0.0115, "floor", ha="right", va="bottom", fontsize=8, color=COLORS["separator"])
    ax.set_ylim(0, 0.08); ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
    ax.set_ylabel("Success tolerance (m)"); xaxis(ax); style_axis(ax)

def legend(fig, ax, y, ncol=None):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol or len(h), frameon=False,
               handlelength=1.8, handletextpad=0.5, columnspacing=1.6, fontsize=9)

def save(fig, name):
    p = OUT / f"{name}.png"; fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.1, facecolor="white"); plt.close(fig); print("wrote", p)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
reward(axes[0]); tolerance(axes[1])
axes[0].set_title("Return"); axes[1].set_title("Curriculum")
legend(fig, axes[0], -0.06); fig.tight_layout(w_pad=1.5)
save(fig, "coevo_clean")
# baseline vs co-evolution (the two-line charts), and imitation vs co-evolution
for name, draw in (("coevo_clean_return", reward), ("coevo_clean_tolerance", tolerance)):
    fig, ax = plt.subplots(figsize=(3.6, 2.7)); draw(ax); legend(fig, ax, -0.08); fig.tight_layout(); save(fig, name)
    fig, ax = plt.subplots(figsize=(3.6, 2.7)); draw(ax, with_imitation=True, with_baseline=False); legend(fig, ax, -0.08); fig.tight_layout(); save(fig, name.replace("coevo_clean", "imitation_clean"))
