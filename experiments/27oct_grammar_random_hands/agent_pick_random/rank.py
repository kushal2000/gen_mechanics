"""Combine stage 1 (screen.py -> metrics.json) and stage 2 (rotate.py -> rotate.json, rotate_res50.json) into one
table and the ranking; draw the summary figure.

    .venv_isaacsim/bin/python experiments/27oct_grammar_random_hands/agent_pick_random/rank.py

Score = ROTavg x pick2 x C3, times a joint-count factor min(1, (16 / J)^0.5) (16 = the 4 x 4 primitive hands that
reached ~30 goals on this task; more joints than that costs sample efficiency, fewer is not rewarded).
ROTavg = mean of the strict (30 % residual) and relaxed (50 %) rotation coverage.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from hand_sampler.grammar import derive as dv  # noqa: E402
from hand_sampler.grammar.hand import hand_from_dict  # noqa: E402

PLOTS = REPO / "experiments/27oct_grammar_random_hands/plots/agent_pick_random"
ASSETS = REPO / "assets/urdf/27oct_grammar_random_hands"
K = {"flexion": "B", "abduction": "S", "roll": "T"}


def structure(name):
    H = hand_from_dict(json.load(open(ASSETS / name / f"{name}.design.json")))
    fs = []
    for i, f in enumerate(H.fingers):
        ks = "".join(K[dv.joint_kind(H, i, j)] for j in range(len(f.joints)))
        fs.append({"kinds": ks, "len": sum(j.length for j in f.joints), "spatial": ("B" in ks and len(set(ks)) > 1)})
    return H, fs


def main():
    m = {r["name"]: r for r in json.load(open(HERE / "metrics.json"))}
    r30 = {r["name"]: r for r in json.load(open(HERE / "rotate.json"))}
    r50 = {r["name"]: r for r in json.load(open(HERE / "rotate_res50.json"))}
    pk = {r["name"]: r for r in json.load(open(HERE / "pickup.json"))}
    cl = {r["name"]: r for r in json.load(open(HERE / "clean.json"))}
    gcv = {r["name"]: r for r in json.load(open(HERE / "goal_cover.json"))}
    fine = {r["name"]: r for r in json.load(open(HERE / "fine_check.json"))}
    rows = []
    for nm, r in m.items():
        H, fs = structure(nm)
        J = r["joints"]
        rot = 0.5 * (r30[nm]["ROT"] + r50[nm]["ROT"])
        jf = min(1.0, (16.0 / J) ** 0.5)
        used = r["fingers_used_at_best"]
        deep = sum(1 for x in r["reach_x_p99"] if x >= 75.0)
        row = dict(name=nm, F=r["fingers"], J=J, cut=r["palm_cut"], c2=r["c2_ok"], C2=r["C2"], C3=r["C3"], C4=r["C4"],
                   C3n=r["C3_nbhd"], R3=r["R3"], zone=r["zone3_cm3"], zone75=r["zone3_75_cm3"],
                   pick2=pk[nm]["pick2"], pick3=pk[nm]["pick3"], pick2_low=pk[nm]["pick2_low"],
                   g_min=pk[nm]["g_min"], protrusion=max(pk[nm]["protrusion_mm"]), rot30=r30[nm]["ROT"], rot50=r50[nm]["ROT"],
                   rot30m=r30[nm]["ROT_mean"], rot50m=r50[nm]["ROT_mean"], rot=rot, jf=jf,
                   deep=deep, spatial=sum(f["spatial"] for f in fs), used=used,
                   J_used=sum(len(H.fingers[i].joints) for i in used) + len(H.palm_joints),
                   kinds="|".join(f"{f['kinds']}:{f['len']}" for f in fs), best=r["best_centre_mm"])
        row["screen_score"] = rot * row["pick2"] * row["C3"] * jf
        for key in ("CH2", "CH3", "ROTC", "CH3_mean", "ROTC_mean"):
            row[key] = cl[nm][key]
        row["clean_centre"] = cl[nm]["best_centre"]
        row["touching"] = max(cl[nm]["mean_touching"]) if cl[nm]["mean_touching"] else 0.0
        row["stage3_score"] = row["CH3"] * row["ROTC"] * row["pick2"] * jf
        row["GC3"] = gcv[nm]["GC3"] if nm in gcv else None
        row["GROT"] = gcv[nm]["GROT"] if nm in gcv else None
        row["fine_CH3"] = fine[nm]["CH3_best"] if nm in fine else None
        row["fine_ROTC"] = fine[nm]["ROTC_best"] if nm in fine else None
        if nm in gcv:
            row["score_nojf"] = row["GC3"] * row["GROT"] * row["pick2"]
            row["score"] = row["score_nojf"] * jf
        else:   # not in the top 20 of stage 3: ranked below them by the stage-3 score
            row["score_nojf"] = None
            row["score"] = -1.0 + row["stage3_score"]
        rows.append(row)
    rows.sort(key=lambda x: -x["score"])
    json.dump(rows, open(HERE / "ranking.json", "w"), indent=1)
    hdr = (f"{'#':>3} {'hand':14} F  J cut  C3   zone pick2 gmin rot30 rot50 scr   CH2  CH3  ROTC CH3m ROTCm touch "
           f"s3    GC3  GROT  score kinds")
    print(hdr)
    for k, x in enumerate(rows):
        print(f"{k + 1:3d} {x['name']:14} {x['F']} {x['J']:2d} {'C' if x['cut'] else ' '}  {x['C3']:.2f} "
              f"{x['zone']:5d} {x['pick2']:.2f} {x['g_min'] if x['g_min'] is not None else -1:4.0f} {x['rot30']:.2f}  "
              f"{x['rot50']:.2f}  {x['screen_score']:.2f}  {x['CH2']:.2f} {x['CH3']:.2f} {x['ROTC']:.2f} "
              f"{x['CH3_mean']:.2f} {x['ROTC_mean']:.2f} {x['touching']:.1f}  {x['stage3_score']:.2f}  "
              f"{(x['GC3'] if x['GC3'] is not None else float('nan')):.2f}  "
              f"{(x['GROT'] if x['GROT'] is not None else float('nan')):.2f}  {x['score']:.3f} "
              f"{x['kinds']}")

    # figure: rotation coverage vs joints, coloured by pickup, size by C3
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    J = np.array([x["J"] for x in rows]); rot = np.array([x["ROTC"] for x in rows])
    pk = np.array([x["pick2"] for x in rows]); c3 = np.array([x["CH3"] for x in rows])
    sc = axs[0].scatter(J, rot, c=pk, s=20 + 120 * c3, cmap="viridis", vmin=0, vmax=1, edgecolor="k", lw=0.4)
    for x in rows[:10]:
        axs[0].annotate(x["name"][-3:], (x["J"], x["ROTC"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    axs[0].set_xlabel("joints"); axs[0].set_ylabel("ROTC: collision-free rotation coverage")
    axs[0].set_title("Rotation coverage vs joint count\ncolour = pickup (frac. of pen yaws), size = CH3")
    fig.colorbar(sc, ax=axs[0], label="pick2")
    top = [x for x in rows if x["GC3"] is not None]
    for k, x in enumerate(top):
        col = "#c0392b" if k == 0 else ("#e67e22" if k == 1 else "#7f8c8d")
        axs[1].scatter(x["J"], x["score"], s=40, color=col, edgecolor="k", lw=0.4)
        axs[1].annotate(f"{x['name'][-3:]} ({x['F']}f)", (x["J"], x["score"]), fontsize=7, xytext=(3, 2),
                        textcoords="offset points")
    axs[1].set_xlabel("joints"); axs[1].set_ylabel("final score = GC3 x GROT x pick2 x joint factor")
    axs[1].set_title("Final score, the 20 hands that reached the goal-coverage stage")
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "ranking.png", dpi=120)
    plt.close(fig)
    # depth profile of the collision-free 3-finger hold, top 8
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for k, x in enumerate(top[:8]):
        g = gcv[x["name"]]
        ax.plot(g["depths"], g["hold_by_depth"], marker="o", ms=3, lw=2.2 if k < 2 else 1.0,
                label=f"{x['name'][-3:]} ({x['F']}f/{x['J']}j, protrusion {g['protrusion_face_mm']:.0f} mm)")
    ax.set_xlabel("pen centre depth below the palm face (mm)")
    ax.set_ylabel("fraction of axis directions held\n(collision-free, 3 fingers)")
    ax.set_title("Where under the palm each hand can hold the pen")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / "hold_by_depth.png", dpi=120)


if __name__ == "__main__":
    main()
