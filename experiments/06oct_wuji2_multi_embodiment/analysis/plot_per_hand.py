"""Plot eval_per_hand.py's JSON: goals/min and drops/min per Wuji v2 design, left and right, grouped by finger count.

    .venv_isaacsim/bin/python experiments/06oct_wuji2_multi_embodiment/analysis/plot_per_hand.py \
        experiments/06oct_wuji2_multi_embodiment/results/eval_per_hand_ep3600.json     # -> plots/eval_per_hand_ep3600.png
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FING = ["thumb", "index", "middle", "ring", "pinky"]


def fingers(hand: str) -> list[str]:
    core = hand.replace("wuji2_right", "").replace("wuji2", "").strip("_")
    if core.startswith("only_"):
        return [f for f in FING if f in core]
    if core.startswith("no_"):
        return [f for f in FING if f not in core]
    return FING


src = pathlib.Path(sys.argv[1])
d = json.loads(src.read_text())
rows = {}
for hand, m in d["hands"].items():
    f = fingers(hand)
    key = "+".join(x[0].upper() for x in f)
    rows.setdefault(key, {"n": len(f), "thumb": "thumb" in f})["R" if "_right" in hand else "L"] = m

# finger count descending, then mean goals/min descending within it
order = sorted(rows, key=lambda k: (-rows[k]["n"], -(rows[k]["L"]["goals_per_min"] + rows[k]["R"]["goals_per_min"])))
y, ys, prev = [], 0.0, None
for k in order:
    if prev is not None and rows[k]["n"] != prev:
        ys += 0.8                                    # gap between finger-count groups
    y.append(ys); ys += 1; prev = rows[k]["n"]

fig, axes = plt.subplots(1, 2, figsize=(11, 10), sharey=True, gridspec_kw={"width_ratios": [3, 2]})
for ax, metric, label in ((axes[0], "goals_per_min", "goals / min (higher is better)"),
                          (axes[1], "drops_per_min", "drops / min (lower is better)")):
    for yi, k in zip(y, order):
        r = rows[k]
        c = "#c44e52" if r["thumb"] else "#4c72b0"
        vl, vr = r["L"][metric], r["R"][metric]
        ax.plot([vl, vr], [yi, yi], color=c, alpha=0.35, lw=2)
        ax.scatter([vl], [yi], marker="o", color=c, s=36, zorder=3)
        ax.scatter([vr], [yi], marker="s", facecolor="white", edgecolor=c, s=36, zorder=3, lw=1.5)
    ax.set_xlabel(label)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(left=0)
axes[0].set_yticks(y)
axes[0].set_yticklabels(order, fontfamily="monospace")
axes[0].invert_yaxis()

# finger-count labels on the left margin
for n in (5, 4, 3, 2):
    ys_n = [yi for yi, k in zip(y, order) if rows[k]["n"] == n]
    axes[0].text(-0.21, (min(ys_n) + max(ys_n)) / 2, f"{n} fingers", transform=axes[0].get_yaxis_transform(),
                 rotation=90, va="center", ha="center", fontsize=10, fontweight="bold")

from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", ls="", color="gray", label="left hand"),
           Line2D([], [], marker="s", ls="", mfc="white", mec="gray", mew=1.5, label="right hand"),
           Line2D([], [], color="#c44e52", lw=4, label="has thumb"),
           Line2D([], [], color="#4c72b0", lw=4, label="no thumb")]
axes[1].legend(handles=handles, loc="upper right", frameon=True)
ep = d["checkpoint"].split("_ep_")[1].split("_")[0]
fig.suptitle(f"One policy, 52 Wuji v2 hands (31826, epoch {ep}): greedy eval, {d['envs_per_hand']} envs/hand x "
             f"{d['seconds']:.0f} s\nT thumb, I index, M middle, R ring, P pinky", fontsize=11)
fig.tight_layout(rect=(0.035, 0, 1, 1))
out = pathlib.Path(__file__).resolve().parent.parent / "plots" / f"{src.stem}.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(out)
