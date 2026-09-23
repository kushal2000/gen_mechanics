"""coevolution_v2: the two co-evolution arms against the fixed-population
baseline, and against imitation.

Writes to experiments/17sep_coevolution/plots/:
    v2_tolerance.png, v2_return.png, v2.png          baseline vs the two arms
    v2_imitation_tolerance.png, v2_imitation_return.png   imitation vs co-evolution

    .venv_isaacsim/bin/python experiments/17sep_coevolution/analysis/plot_v2.py
"""
import pathlib, sys
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parent), "/share/portal/kk837/depthbasedRL/plot_figures"]
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _style import configure_rcparams, style_axis, COLORS
import curves as C

configure_rcparams()
OUT = pathlib.Path(__file__).resolve().parent.parent / "plots"; OUT.mkdir(parents=True, exist_ok=True)
HERO, FOIL, ALT, IMIT = COLORS["play2win"], COLORS["play_only"], "#7B3294", "#1A9850"

cr, ct, cb, csel = C.coevo("coevolution_v2")            # a selection every 5000 epochs
gr, gt, gb, gsel = C.coevo("coevolution_v2_gen2k")      # a selection every 2000 epochs
br, bt, bb = C.baseline()                               # no selection, 24 h links
ir, it = C.imitation_v1()

def prep(r, starts):
    return C.smooth(C.bridge(C.blank_restarts(r, starts[:-1] if starts else [])), 200)
crs, grs, brs, irs = prep(cr, cb), prep(gr, gb), prep(br, bb), C.smooth(ir, 200)
XMAX = max(len(ct), len(gt), len(bt)) * 1.02
ARMS = [("Co-evolution (5000 ep/gen)", HERO, "-", crs, ct),
        ("Co-evolution (2000 ep/gen)", ALT, "-", grs, gt),
        ("Fixed population", FOIL, "--", brs, bt)]

def axis(ax):
    ax.set_xlim(0, XMAX); C.epoch_hour_ticks(ax); style_axis(ax)

def tolerance(ax, arms=ARMS, imitation=False):
    for label, col, ls, _, t in arms:
        ax.plot(np.arange(1, len(t) + 1), t, color=col, ls=ls, label=label)
    if imitation:
        ax.plot(np.arange(1, len(it) + 1), it, color=IMIT, ls="-.", lw=1.6, label="Imitation (one SHARPA-like hand)")
    ax.axhline(0.01, color=COLORS["separator"], lw=0.7, ls=":")
    ax.text(XMAX * 0.99, 0.0115, "floor", ha="right", va="bottom", fontsize=8, color=COLORS["separator"])
    ax.set_ylim(0, 0.08); ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
    ax.set_ylabel("Success tolerance (m)"); ax.set_title("Curriculum", fontsize=11, pad=6); axis(ax)

def ret(ax, arms=ARMS, imitation=False):
    for label, col, ls, r, _ in arms:
        ax.plot(np.arange(1, len(r) + 1), r, color=col, ls=ls, label=label)
    if imitation:
        ax.plot(np.arange(1, len(irs) + 1), irs, color=IMIT, ls="-.", lw=1.6, label="Imitation (one SHARPA-like hand)")
    ax.set_ylim(0, None); ax.set_ylabel("Mean episode return"); ax.set_title("Return", fontsize=11, pad=6); axis(ax)

def save(fig, ax, name, y=-0.1, ncol=2):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, fontsize=8.5)
    fig.tight_layout()
    p = OUT / f"{name}.png"; fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig); print("wrote", p)

for name, draw in (("v2_tolerance", tolerance), ("v2_return", ret)):
    fig, ax = plt.subplots(figsize=(3.9, 2.9)); draw(ax); save(fig, ax, name)
    # co-evolution vs imitation: the baseline is a different question, so it is not here
    fig, ax = plt.subplots(figsize=(3.9, 2.9)); draw(ax, arms=ARMS[:2], imitation=True)
    save(fig, ax, name.replace("v2_", "v2_imitation_"))
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9)); ret(axes[0]); tolerance(axes[1])
save(fig, axes[0], "v2", y=-0.06, ncol=3)

fmt = lambda t: f"{t[-1]:.4f}"
print(f"epochs  coevo5000 {len(ct):,}  coevo2000 {len(gt):,}  baseline {len(bt):,}  imitation {len(it):,}")
print(f"tolerance  coevo5000 {fmt(ct)}  coevo2000 {fmt(gt)}  baseline {fmt(bt)}  imitation {fmt(it)}")
n = min(len(ct), len(gt), len(bt))
print(f"at {n:,} epochs (the shortest arm): coevo5000 {ct[n-1]:.4f}  coevo2000 {gt[n-1]:.4f}  baseline {bt[n-1]:.4f}")
print(f"return at that point: coevo5000 {crs[n-1]:.0f}  coevo2000 {grs[n-1]:.0f}  baseline {brs[n-1]:.0f}")
