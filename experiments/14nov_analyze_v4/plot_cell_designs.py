"""The best design in each cell matching a filter (14 Nov 2026), from one v4 run: per cell, the training design with the
highest goals / episode label over generations >= --from-gen (designs with fewer than --min-episodes episodes skipped),
home pose, top down, cells ordered by that label. The filter is a regex on the cell name; layout labels are the code's
"gap" (largest empty angle between neighbouring finger facings) -- gap240-360 is a spread of 0-120 deg, the human-like,
one-sided layouts.

    .venv_isaacsim/bin/python experiments/14nov_analyze_v4/plot_cell_designs.py --cells 'f5_.*_gap240-360_' \
        [--run k5_none] [--from-gen 40] [--max 12]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments/06nov_coevolve_v2"), str(Path(__file__).parent)]
NAMES = {"k5_none": "MAP-Elites", "k5_global": "Greedy", "hod": "House of Dextra"}
SPREAD = {"gap240-360": "spread 0-120", "gap180-240": "spread 120-180", "gap120-180": "spread 180-240",
          "gap0-120": "spread 240-360"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="regex on the cell name, e.g. 'f5_.*_gap240-360_'")
    ap.add_argument("--run", default="k5_none")
    ap.add_argument("--campaign", default="debug_outputs/coevolve/v4")
    ap.add_argument("--from-gen", type=int, default=40)
    ap.add_argument("--min-episodes", type=int, default=20)
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from plot_morph_gallery import KIND_RGB, PALM_RGB, canon
    from plot_best_designs import draw, posed_tris
    from hand_sampler.grammar import v3
    v3.configure("v4")

    run = REPO / a.campaign / a.run
    reg = json.loads((run / "designs.json").read_text())
    pat = re.compile(a.cells)
    best = {}
    for f in sorted(run.glob("gen_*/labels.json")):
        g = int(f.parent.name.split("_")[1])
        if g < a.from_gen:
            continue
        for i, (v, n, *_) in json.loads(f.read_text()).items():
            c = reg[i]["niche"]
            if n >= a.min_episodes and pat.search(c) and (c not in best or v > best[c]["label"]):
                best[c] = {"id": i, "label": v, "episodes": n, "gen": g}
    if not best:
        raise SystemExit(f"no trained design in a cell matching {a.cells!r}")
    rows = sorted(best.items(), key=lambda kv: -kv[1]["label"])[: a.max]
    n = len(rows)
    ncol = min(a.cols, n)
    nrow = math.ceil(n / ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.9 * nrow + 0.6), squeeze=False, facecolor="white")
    for ax in axs.ravel():
        ax.axis("off")
    for k, (c, b) in enumerate(rows):
        ax = axs[k // ncol, k % ncol]
        ax.axis("on")
        hand = v3.hand_from_dict(reg[b["id"]]["design"])
        kinds = {f"f{i}_j{j}": v3.kind_of(f, jt) for i, f in enumerate(hand.fingers) for j, jt in enumerate(f.joints)}
        pats = " ".join("".join(v3.kind_of(f, jt)[0].upper() for jt in f.joints) for f in hand.fingers)
        rs, ctr, R = canon(reg[b["id"]]["spec"])
        urdf = REPO / rs.urdf_path if not Path(rs.urdf_path).is_absolute() else Path(rs.urdf_path)
        tris, cols = posed_tris(urdf, lambda jn: 0.0, lambda jn: kinds.get(jn, "flex"))
        draw(ax, (tris - ctr) @ R.T * 1000, cols, "face")
        f_, m_, gap, kn, reach = c.split("_")
        ax.set_title(f"{b['label']:.1f} goals / ep\n{m_[1:]} motors, {kn}, {reach}", fontsize=9, color="#1f1f1e")
        ax.set_xlabel(f"{pats}\nfacing gap {v3.design_gap_deg(hand):.0f}\N{DEGREE SIGN}, gen {b['gen']}, "
                      f"{b['episodes']} ep", fontsize=7.5, color="#6b6a64")
        print(f"{c:36s} {b['label']:5.1f} goals/ep  {b['id']}  fingers {pats}")
    fig.legend(handles=[Patch(color=KIND_RGB["flex"], label="flexion"), Patch(color=KIND_RGB["abd"], label="abduction"),
                        Patch(color=PALM_RGB, label="palm")], loc="upper right", ncol=3, frameon=False, fontsize=8.5)
    lay = {SPREAD.get(p, p) for p in re.findall(r"gap\d+-\d+", a.cells)}
    fig.suptitle(f"v4 {NAMES.get(a.run, a.run)}: best design per cell matching '{a.cells}'"
                 + (f" ({', '.join(sorted(lay))})" if lay else "") + f", gens {a.from_gen}+ (home pose, top down)",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold", color="#1f1f1e")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    tag = re.sub(r"[^A-Za-z0-9-]+", "_", a.cells).strip("_")
    out = REPO / (a.out or f"experiments/14nov_analyze_v4/plots/cells_{a.run}_{tag}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor="white")
    print("[cell designs]", out)


if __name__ == "__main__":
    main()
