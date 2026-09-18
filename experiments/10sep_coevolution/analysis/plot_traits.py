"""Which design traits selection pushed: fingers, joints, finger length and
palm area per hand across generations, mean with IQR band, the fixed
population's value dashed, fixation marked.

    .venv_isaacsim/bin/python experiments/10sep_coevolution/analysis/plot_traits.py [population_dir] [fixation_gen]
"""
import sys; sys.path[:0] = ["/share/portal/kk837/depthbasedRL/plot_figures"]
import json, pathlib, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _style import configure_rcparams, style_axis, COLORS
from hand_sampler import population_io

configure_rcparams()
R = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "assets/populations/coevo_r500_v1")
FIX = int(sys.argv[2]) if len(sys.argv) > 2 else 13
OUT = pathlib.Path("debug_outputs/10sep_coevo_analysis"); TAG = R.name
gens = sorted(int(p.name.split("_")[1]) for p in R.glob("gen_*") if (p / "population.json").exists())
CACHE = OUT / ".curve_cache" / f"traits_{TAG}.json"
rows = json.load(open(CACHE)) if CACHE.exists() else {}
for g in gens:
    if str(g) in rows: continue
    hands = population_io.load_population(R / f"gen_{g}/population.json")
    rows[str(g)] = {
        "fingers": [h.n_fingers for h in hands],
        "joints": [h.n_joints for h in hands],
        # mean finger length per hand, cm: tip-to-mount along the finger at rest
        "finger_len": [100 * np.mean([sum(s.length for s in f.segments) for f in h.fingers]) for h in hands],
        "palm_area": [1e4 * h.palm.width * h.palm.length for h in hands],   # cm^2
    }
CACHE.write_text(json.dumps(rows))
G = np.array(gens)
# (key, panel title, y label)
TRAITS = [("fingers", "Finger count", "fingers per hand"), ("joints", "Joint count", "joints per hand"),
          ("finger_len", "Finger length", "mean per hand (cm)"), ("palm_area", "Palm size", "area (cm$^2$)")]
HERO, FOIL = COLORS["play2win"], COLORS["play_only"]

def draw(ax, key, title, label):
    v = np.array([rows[str(g)][key] for g in gens], float)
    mean, lo, hi = v.mean(1), np.percentile(v, 25, axis=1), np.percentile(v, 75, axis=1)
    ax.fill_between(G, lo, hi, color=HERO, alpha=0.18, lw=0, label="IQR")
    ax.plot(G, mean, color=HERO, label="Mean")
    ax.axvline(FIX, color=COLORS["separator"], lw=0.8, ls=":")
    ax.set_xlim(G[0], G[-1]); ax.set_xticks(range(0, int(G[-1]) + 1, 10)); ax.set_xlabel("Generation"); ax.set_ylabel(label)
    ax.set_title(title, fontsize=11, pad=6)
    style_axis(ax)
    y0, y1 = ax.get_ylim(); ax.text(FIX + 0.7, y1 - 0.04 * (y1 - y0), f"fixation\n(gen {FIX})", fontsize=7.5, color="#555", va="top")
    return mean, lo, hi

fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
stats = {}
for ax, (key, title, label) in zip(axes.flat, TRAITS):
    stats[key] = draw(ax, key, title, label)
h, l = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False)
fig.tight_layout(h_pad=1.6, w_pad=1.8)
p = OUT / f"traits_{TAG}.png"; fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.1, facecolor="white"); plt.close(fig); print("wrote", p)
for key, title, label in TRAITS:
    fig, ax = plt.subplots(figsize=(3.6, 2.7)); draw(ax, key, title, label)
    h, l = ax.get_legend_handles_labels(); fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout(); p = OUT / f"traits_{key}_{TAG}.png"; fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.1, facecolor="white"); plt.close(fig); print("wrote", p)
for key, title, label in TRAITS:
    m, lo, hi = stats[key]
    print(f"{title:26s} gen 0 {m[0]:6.2f} (IQR {lo[0]:.1f}-{hi[0]:.1f})   gen {FIX} {m[FIX]:6.2f}   gen {G[-1]} {m[-1]:6.2f} (IQR {lo[-1]:.1f}-{hi[-1]:.1f})")
