"""Morph the controller's best f5_m20-21 grammar design (g0007r04_19717) toward Wuji v2 one factor at a time, and
score every step zero-shot in the training (multi:template:) scene: where does the policy stop working?

Ladder (each a grammar design; facing theta points a finger along (y, z) = (sin, cos) in the palm plane):
  ref                 the design as evolved (fingers in five directions, ~210 mm)
  spread{S}           the four non-thumb fingers (0, 2, 3, 4) moved to radial bases 80 mm from the centre, their
                      facings spread over S degrees (S = 240 ~ the reference's spread, ..., 60); the 5-joint finger
                      (f1, facing 270, the one with an abduction) stays, as Wuji's thumb
  parallel            those four side by side on the +z edge (y = -45, -15, 15, 45; z = 80), all facing 0: Wuji's layout
  parallel_thumb315   + the thumb turned toward the fingers (facing 315, as Wuji's thumb points ~ -50 deg)
  parallel_wujijoints + the four fingers get Wuji's chain: flexion, abduction, flexion, flexion (30 / 70 / 60 / 50 mm)
  ref_x0p75, ref_x0p5, parallel_x0p5   sizes (positions and links x s, 10 mm grid); x0.5 ~ Wuji's 106 mm fingers

    .venv_isaacsim/bin/python experiments/06nov_coevolve_v2/grammar_morph.py [--envs 32] [--controllers k5_none:7 k1_none:8]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/05nov_noevolution/studies"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = REPO / "debug_outputs/coevolve/studies/grammar_morph"
REF = REPO / "debug_outputs/coevolve/v1/k5_none/hands/g0007r04_19717/g0007r04_19717.spec.json"
OTHERS, THUMB = [0, 2, 3, 4], 1
SPREADS = {240: [-60, 20, 100, 180], 180: [-45, 15, 75, 135], 120: [-30, 10, 50, 90], 60: [-15, 5, 25, 45]}


def ladder() -> dict[str, dict]:
    import wuji_zero_shot as W
    ref = json.loads(REF.read_text())["design"]
    cp = lambda d: json.loads(json.dumps(d))           # noqa: E731
    out = {"ref": cp(ref)}
    for S, faces in SPREADS.items():
        d = cp(ref)
        for i, th in zip(OTHERS, faces):
            f = d["fingers"][i]
            f["facing"] = th % 360
            f["y"], f["z"] = round(80 * math.sin(math.radians(th))), round(80 * math.cos(math.radians(th)))
        out[f"spread{S}"] = d
    par = cp(ref)
    for i, y in zip(OTHERS, (-45, -15, 15, 45)):
        par["fingers"][i].update(y=y, z=80, facing=0)
    out["parallel"] = par
    d = cp(par)
    d["fingers"][THUMB].update(facing=315, y=-63, z=20)
    out["parallel_thumb315"] = d
    d = cp(par)
    for i in OTHERS:
        d["fingers"][i]["joints"] = [{"type": "hinge", "axis": [0, 0], "length": 30},
                                     {"type": "hinge", "axis": [90, 0], "length": 70},
                                     {"type": "hinge", "axis": [0, 0], "length": 60},
                                     {"type": "hinge", "axis": [0, 0], "length": 50}]
    out["parallel_wujijoints"] = d
    # collision-free bunching (7 Nov: spread60 / parallel above intersect neighbouring fingers at spawn, and the
    # template scene collides fingers with each other): bases on the +z edge at a pitch the motors clear (C1), fanned
    for pitch, fans in ((60, (60, 30, 0)), (50, (60, 30, 0)), (40, (30, 0))):
        for S in fans:
            d = cp(ref)
            ys, fs = [-1.5 * pitch, -0.5 * pitch, 0.5 * pitch, 1.5 * pitch], [-S / 2, -S / 6, S / 6, S / 2]
            for i, y, f_ in zip(OTHERS, ys, fs):
                d["fingers"][i].update(y=round(y), z=80, facing=round(f_) % 360)
            out[f"line_p{pitch}_fan{S}"] = d
    d = cp(out["line_p40_fan0"])
    d["fingers"][THUMB].update(facing=300, y=-63, z=0)
    out["line_p40_fan0_thumb300"] = d
    # Wuji's finger chain on a working layout (7 Nov): flexion, abduction, flexion, flexion on the four non-thumb
    # fingers (lengths 0 / 70 / 50 / 70 mm), i.e. five abduction fingers -- 7 of 4096 gen-7 hands had 3+
    chain = [{"type": "hinge", "axis": [0, 0], "length": 0}, {"type": "hinge", "axis": [90, 0], "length": 70},
             {"type": "hinge", "axis": [0, 0], "length": 50}, {"type": "hinge", "axis": [0, 0], "length": 70}]
    for base in ("ref", "spread180"):
        d = cp(out[base])
        for i in OTHERS:
            d["fingers"][i]["joints"] = cp(chain)
        out[f"{base}_wujichain"] = d
        d = cp(out[base])
        for i in OTHERS:
            d["fingers"][i]["joints"] = cp(chain)
            d["fingers"][i]["joints"][1]["axis"] = [0, 0]            # same lengths, all flexion: the control
        out[f"{base}_wujilens"] = d
    # 7 Nov counterfactuals on the reference: Wuji's knuckle (a short flexion stub, then an abduction carrying the
    # rest of the base link) and a split flexion (short flexion, then a second flexion), on the 4 non-thumb fingers
    for stub in (28, 14):
        d = cp(ref)
        for i in OTHERS:
            j = d["fingers"][i]["joints"]
            L0 = j[0]["length"]
            d["fingers"][i]["joints"] = [{"type": "hinge", "axis": [0, 0], "length": stub},
                                         {"type": "hinge", "axis": [90, 0], "length": L0 - stub}] + j[1:]
        out[f"ref_knuckle{stub}"] = d
    d = cp(ref)
    for i in OTHERS:
        j = d["fingers"][i]["joints"]
        L0 = j[0]["length"]
        d["fingers"][i]["joints"] = [{"type": "hinge", "axis": [0, 0], "length": 28},
                                     {"type": "hinge", "axis": [0, 0], "length": L0 - 28}] + j[1:]
    out["ref_splitflex28"] = d
    # co-located (0 mm) 2-DOF base knuckles in both orders, the reference's own lengths (7 Nov): flexion then abduction
    # (Wuji, LEAP) vs abduction then flexion (Shadow, Tesollo); both legal (zero_link_ok)
    for order in ("FA", "AF"):
        d = cp(ref)
        for i in OTHERS:
            j = d["fingers"][i]["joints"]
            ax0, ax1 = ([0, 0], [90, 0]) if order == "FA" else ([90, 0], [0, 0])
            d["fingers"][i]["joints"] = [{"type": "hinge", "axis": ax0, "length": 0},
                                         {"type": "hinge", "axis": ax1, "length": j[0]["length"]}] + j[1:]
        out[f"ref_knuckle{order}0"] = d
    out["ref_x0p875"] = W.scaled(ref, 0.875)
    out["ref_x0p75"] = W.scaled(ref, 0.75)
    out["ref_x0p5"] = W.scaled(ref, 0.5)
    out["parallel_x0p5"] = W.scaled(par, 0.5)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--controllers", nargs="+", default=["v2/k5_none:latest"],
                    help="campaign/run:gen or :latest (v2 by default since 7 Nov); a bare run:gen means v1")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--wrist-z", type=float, default=None, help="wrist start height (m; training 0.15)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    import study_beam as sb
    from coevolution.evolution.descriptors import niche32
    from hand_sampler.grammar import hand as gh
    OUT.mkdir(parents=True, exist_ok=True)
    hands = []
    for hid, d in ladder().items():
        if args.only and hid not in args.only:
            continue
        viol = gh.check(gh.hand_from_dict(d), gh.mutation_rules(gh.FEETECH_COARSE_NOCUT_RULES))
        from hand_sampler.grammar import viability as gvb
        c1 = gvb.c1_self_overlap(gh.hand_from_dict(d))
        _, spec, fits = sb._build((hid, d, str(OUT / "hands")))
        hands.append({"id": hid, "spec": spec, "niche": niche32(d)})
        print(f"[morph] {hid:20s} niche {niche32(d)}  fits grid {fits}  grid-rule violations {len(viol)}  C1 clear {c1.ok} (worst {c1.worst_mm:.1f} mm)", flush=True)
    man = OUT / (f"{args.tag.strip('_')}_manifest.json" if args.tag else "manifest.json")  # loader: *manifest.json
    man.write_text(json.dumps({"generation": 0, "grid": {"n_rigid": 6, "n_moving": 0, "n_slots": 5}, "hands": hands}))
    res = {}
    for c in args.controllers:
        run, g = c.split(":")
        if "/" not in run:
            run = "v1/" + run
        rd = REPO / "debug_outputs/coevolve" / run
        if g == "latest":
            g = str(max(int(p_.parent.name[4:]) for p_ in rd.glob("gen_*/policy.pth")))
        run = run.replace("/", "_")
        tr = json.loads((rd / "state.json").read_text())["train_runs"][g]
        fo = OUT / f"fitness_{run}_g{g}_e{args.envs}{args.tag}.json"
        if not fo.exists():
            print(f"[morph] {run} gen {g}: {len(hands)} hands x {args.envs} envs ...", flush=True)
            subprocess.run([str(REPO / ".venv_isaacsim/bin/python"), str(REPO / "experiments/05nov_noevolution/studies/eval_values.py"),
                            "--manifest", str(man), "--checkpoint", str((rd / f"gen_{int(g):04d}/policy.pth").resolve()),
                            "--train-run", tr, "--out", str(fo), "--envs-per-hand", str(args.envs), "--seed", "7300",
                            "--table-clearance", "0.02"] + ([] if args.wrist_z is None else ["--wrist-z", str(args.wrist_z)]), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        res[c] = json.loads(fo.read_text())["hands"]
    print(f"{'hand':20s} " + " ".join(f"{c:>16s}" for c in res))
    for h in hands:
        print(f"{h['id']:20s} " + " ".join(f"{res[c][h['spec']]['goals_per_episode']:9.2f} g/ep  " for c in res), flush=True)


if __name__ == "__main__":
    main()
