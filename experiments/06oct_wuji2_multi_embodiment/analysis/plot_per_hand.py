"""Per-design performance of the 52-hand Wuji v2 policy: goals per minute for each finger set, left vs right hand.

Grouped bars by finger count (5, 4, 3, 2), best first within a group; PDF + PNG, paper style. Data: one
results/eval_per_hand_ep<N>.json from eval_per_hand.py.

    .venv_isaacsim/bin/python experiments/06oct_wuji2_multi_embodiment/analysis/plot_per_hand.py \
        experiments/06oct_wuji2_multi_embodiment/results/eval_per_hand_ep3600.json
    -> plots/eval_per_hand_ep3600.{pdf,png}
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import paper_style as ps  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
OUT = HERE / "plots"
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def fingers(hand: str) -> list[str]:
    core = hand.replace("wuji2_right", "").replace("wuji2", "").strip("_")
    if core.startswith("only_"):
        return [f for f in FINGERS if f in core]
    if core.startswith("no_"):
        return [f for f in FINGERS if f not in core]
    return FINGERS


def plot(src: pathlib.Path):
    d = json.loads(src.read_text())
    rows = {}
    for hand, m in d["hands"].items():
        f = fingers(hand)
        rows.setdefault("+".join(x[0].upper() for x in f), {"n": len(f)})["R" if "_right" in hand else "L"] = \
            m["goals_per_min"]
    order = sorted(rows, key=lambda k: (-rows[k]["n"], -(rows[k]["L"] + rows[k]["R"])))
    # one slot per design, plus a gap between finger-count groups
    x, pos, prev = [], 0.0, None
    for k in order:
        if prev is not None and rows[k]["n"] != prev:
            pos += 0.6
        x.append(pos); pos += 1; prev = rows[k]["n"]
    x = np.array(x)
    w = 0.4
    sides = [("Left Hand", "L", ps.COLORS["hero"]), ("Right Hand", "R", ps.COLORS["foil"])]

    fig, ax = plt.subplots(figsize=(ps.TEXT_WIDTH, 2.6))
    for i, (_, side, color) in enumerate(sides):
        ax.bar(x + (i - 0.5) * w, [rows[k][side] for k in order], w * 0.92, color=color, alpha=ps.BAR_ALPHA,
               linewidth=0)
    top = max(max(r["L"], r["R"]) for r in rows.values())
    for n in (5, 4, 3, 2):
        xs = [xi for xi, k in zip(x, order) if rows[k]["n"] == n]
        ax.text((min(xs) + max(xs)) / 2, top * 1.08, "Full" if n == 5 else f"{n} Fingers", ha="center", va="bottom", fontsize=8,
                color=ps.COLORS["muted_ink"])
        if n != 2:
            ax.axvline(max(xs) + 0.8, color=ps.COLORS["neutral"], linewidth=0.7, linestyle=(0, (3, 2)))
    ax.set_ylim(0, top * 1.18)
    ax.set_yticks(np.arange(0, top + 1e-9, 5))
    ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
    ps.style_axis(ax)
    ax.tick_params(axis="x", length=0)
    ax.set_xticks(x, order, rotation=90, fontsize=7)
    ax.set_xlabel("Fingers present (T thumb, I index, M middle, R ring, P pinky)")
    ax.set_ylabel("Goals / min")
    ep = d["checkpoint"].split("_ep_")[1].split("_")[0]
    ax.set_title(f"One Policy, 52 Wuji Hands (epoch {ep})", fontweight="bold")
    ps.bottom_legend(fig, [Patch(facecolor=c, alpha=ps.BAR_ALPHA) for *_, c in sides], [t for t, *_ in sides],
                     y=-0.26, columnspacing=1.6, bold=())
    paths = ps.save_figure(fig, OUT / src.stem)
    plt.close(fig)
    for k in order:
        print(f"  {k:10s} L {rows[k]['L']:5.1f}  R {rows[k]['R']:5.1f}")
    return paths


def main():
    ps.configure()
    for arg in sys.argv[1:]:
        plot(pathlib.Path(arg))


if __name__ == "__main__":
    main()
