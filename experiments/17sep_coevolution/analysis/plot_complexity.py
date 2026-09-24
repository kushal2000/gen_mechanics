"""How expensive the population got: mass, actuation and inertia per generation.

The Sep 18 action item asked whether evolution is driving designs toward
complexity, and named mass and energy. Four measures, all computed from the
design alone -- no rollout, so every generation is comparable:

  mass        palm + real links, exactly as the simulator builds them
              (build.palm_mass_props and build.link_mass_props). Ghost padding
              links are 1e-6 kg and are excluded: they are envelope, not design.
  joints      actuated degrees of freedom, i.e. how much there is to control
  link length total structure, summed over every segment
  inertia     mean principal moment about the palm mount. This is the closest
              static stand-in for "energy": it is what the arm must accelerate
              to move the hand, and unlike mass it charges for reach. Each link
              is a solid cylinder about its own axis (local +x, verified against
              link_frames: |tip-base| equals the segment length), rotated into
              the palm frame and shifted to the mount by the parallel-axis
              theorem; the palm uses its own box tensor about its centre.

What this does NOT measure is actuation energy actually spent, which needs
rollouts under a policy. These are properties of the morphology.

    .venv_isaacsim/bin/python experiments/17sep_coevolution/analysis/plot_complexity.py [population_dir] [fixation_gen]
"""
import sys, json, pathlib
sys.path[:0] = ["/share/portal/kk837/depthbasedRL/plot_figures"]
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from _style import configure_rcparams, style_axis, COLORS
from hand_sampler import population_io, build

configure_rcparams()
R = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "assets/populations/coevolution_v2_gen2k")
FIX = int(sys.argv[2]) if len(sys.argv) > 2 else 12
OUT = pathlib.Path(__file__).resolve().parent.parent / "plots"; OUT.mkdir(exist_ok=True)
CACHE = pathlib.Path("debug_outputs/17sep_coevo_analysis/.curve_cache") / f"complexity_{R.name}.json"
CACHE.parent.mkdir(parents=True, exist_ok=True)
gens = sorted(int(p.name.split("_")[1]) for p in R.glob("gen_*") if (p / "population.json").exists())


def hand_stats(h):
    """(mass g, joints, total link length cm, mean principal moment g*cm^2)."""
    frames = build.link_frames(h)
    m_palm, I_palm = build.palm_mass_props(h)
    c_palm = np.asarray(build.palm_center_offset(h), float)
    # palm: its own tensor about its centre, shifted to the mount
    I = I_palm + m_palm * (float(c_palm @ c_palm) * np.eye(3) - np.outer(c_palm, c_palm))
    mass, length = m_palm, 0.0
    for f, finger in enumerate(h.fingers[:h.n_fingers]):
        for d, seg in enumerate(finger.segments):
            m, (ixx, iyy, _) = build.link_mass_props(seg.length)
            T = frames[(f, d)]; Rm = T[:3, :3]
            com = T[:3, 3] + Rm @ np.array([seg.length / 2.0, 0.0, 0.0])
            I_local = np.diag([ixx, iyy, iyy])          # +x is the link axis
            I += Rm @ I_local @ Rm.T + m * (float(com @ com) * np.eye(3) - np.outer(com, com))
            mass += m; length += seg.length
    moment = float(np.trace(I)) / 3.0                    # mean principal moment
    return 1000.0 * mass, h.n_joints, 100.0 * length, 1e7 * moment   # g, -, cm, g*cm^2


rows = json.load(open(CACHE)) if CACHE.exists() else {}
KEYS = ["mass", "joints", "link_len", "inertia"]
for g in gens:
    if str(g) in rows: continue
    hands = population_io.load_population(R / f"gen_{g}/population.json")
    vals = np.array([hand_stats(h) for h in hands], float)
    rows[str(g)] = {k: vals[:, i].tolist() for i, k in enumerate(KEYS)}
    print(f"  gen {g:3d}: mass {vals[:,0].mean():6.1f} g  joints {vals[:,1].mean():5.2f}  "
          f"len {vals[:,2].mean():5.2f} cm  inertia {vals[:,3].mean():8.1f} g cm^2", flush=True)
CACHE.write_text(json.dumps(rows))

G = np.array(gens)
PANELS = [("mass", "Hand mass", "g per hand"), ("joints", "Actuated joints", "joints per hand"),
          ("link_len", "Total link length", "cm per hand"), ("inertia", "Inertia about mount", "g cm$^2$ per hand")]
HERO = COLORS["play2win"]
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
for ax, (key, title, ylab) in zip(axes.flat, PANELS):
    v = np.array([rows[str(g)][key] for g in gens], float)
    ax.fill_between(G, np.percentile(v, 25, axis=1), np.percentile(v, 75, axis=1),
                    color=HERO, alpha=0.18, lw=0, label="IQR")
    ax.plot(G, v.mean(1), color=HERO, label="Mean")
    ax.axvline(FIX, color=COLORS["separator"], lw=0.8, ls=":")
    ax.set_xlim(G[0], G[-1]); ax.set_xlabel("Generation"); ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=11, pad=6)
    ax.xaxis.set_major_locator(MultipleLocator(10)); ax.xaxis.set_minor_locator(MultipleLocator(2))
    style_axis(ax)
    y0, y1 = ax.get_ylim()
    ax.text(FIX + 0.7, y1 - 0.04 * (y1 - y0), f"fixation\n(gen {FIX})", fontsize=7.5, color="#555", va="top")
h, l = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False)
fig.tight_layout(h_pad=1.6, w_pad=1.8)
p = OUT / f"complexity_{R.name}.png"
fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.1, facecolor="white"); print("wrote", p)

print(f"\n  gen |   mass g |  joints | link cm | inertia g cm^2")
for g in gens:
    r = rows[str(g)]
    print(f"  {g:3d} | {np.mean(r['mass']):8.1f} | {np.mean(r['joints']):7.2f} | "
          f"{np.mean(r['link_len']):7.2f} | {np.mean(r['inertia']):14.1f}")
