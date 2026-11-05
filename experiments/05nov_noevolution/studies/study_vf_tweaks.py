"""Tuning the design value net (the per-finger transformer) for search: each configuration judged on the population's
validation split (in distribution) AND on the beam search's scored designs (out of distribution), every 500 steps;
early stopping at the step with the best population-validation rank.

    sbatch --array=0-10 experiments/05nov_noevolution/studies/study_vf_tweaks.sub     (one config per task)
    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/study_vf_tweaks.py --report

Data: iteration --iteration's 4,096 designs with 100-env labels, split 80/20 by niche (seed 0, as study_vf_hparams.py);
the out-of-distribution set: every design the study-5 beam search scored (100 envs), with its round.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import design_value_net as dvn  # noqa: E402
import study_value as sv  # noqa: E402

CONFIGS = [
    {"name": "baseline"},
    {"name": "dropout 0.1", "arch": {"dropout": 0.1}},
    {"name": "dropout 0.2", "arch": {"dropout": 0.2}},
    {"name": "wd 1e-2", "wd": 1e-2},
    {"name": "wd 1e-1", "wd": 1e-1},
    {"name": "d64 2 layers", "arch": {"d": 64, "ff": 256}},
    {"name": "d64 1 layer", "arch": {"d": 64, "ff": 256, "layers": 1}},
    {"name": "dropout 0.1 + wd 1e-2", "arch": {"dropout": 0.1}, "wd": 1e-2},
    {"name": "lr 1e-3", "lr": 1e-3},
    {"name": "ensemble of 5 (bootstrap)", "members": 5},
    {"name": "ensemble of 5, dropout 0.1 + wd 1e-2", "members": 5, "arch": {"dropout": 0.1}, "wd": 1e-2},
    # round 2 (6 Nov): dropout helps out of distribution, so around it
    {"name": "dropout 0.3", "arch": {"dropout": 0.3}},
    {"name": "dropout 0.4", "arch": {"dropout": 0.4}},
    {"name": "dropout 0.2, 30k steps", "arch": {"dropout": 0.2}, "steps": 30000},
    {"name": "dropout 0.2 + wd 1e-2", "arch": {"dropout": 0.2}, "wd": 1e-2},
    {"name": "dropout 0.2, d64 2 layers", "arch": {"dropout": 0.2, "d": 64, "ff": 256}},
    {"name": "dropout 0.3, 30k steps", "arch": {"dropout": 0.3}, "steps": 30000},
    {"name": "ensemble of 5, dropout 0.2", "members": 5, "arch": {"dropout": 0.2}},
]


def rank(a):
    return np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)


def spear(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.corrcoef(rank(a), rank(b))[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")


def grouped(p, y, groups):
    gs = {}
    for i, gname in enumerate(groups):
        gs.setdefault(gname, []).append(i)
    return float(np.nanmean([spear(p[idx], y[idx]) for idx in gs.values() if len(idx) > 4]))


def data(it: int, start: str):
    g = REPO / f"debug_outputs/evolution/noevolution_4096/gen_{it:04d}"
    man = json.loads((g / "manifest.json").read_text())
    lab = {}
    for f in sorted((REPO / "debug_outputs/noevolution/studies" / f"it{it}_env100").glob("c*_fitness.json")):
        lab.update({s: v["goals_per_episode"] for s, v in json.loads(f.read_text())["hands"].items()})
    hs = [h for h in man["hands"] if h["spec"] in lab]
    rng = np.random.default_rng(0)
    by, val = {}, set()
    for k, h in enumerate(hs):
        by.setdefault(h["niche"], []).append(k)
    for idx in by.values():
        val |= set(rng.choice(idx, int(round(0.2 * len(idx))), replace=False).tolist())
    tr = [k for k in range(len(hs)) if k not in val]
    va = sorted(val)
    D = [sv._load_design(h["spec"]) for h in hs]
    Y = np.array([lab[h["spec"]] for h in hs], np.float32)
    S = json.loads((REPO / "debug_outputs/noevolution/studies" / f"beam_{start}_it{it}" / "state.json").read_text())
    B = [(r["design"], st.mean(r["scores"]), r["round"]) for r in S["designs"].values() if r["scores"] and r["round"] > 0]
    return ([D[k] for k in tr], Y[tr], [D[k] for k in va], Y[va], [hs[k]["niche"] for k in va],
            [b[0] for b in B], np.array([b[1] for b in B], np.float32), [b[2] for b in B])


def run(args) -> None:
    c = CONFIGS[args.config]
    Dt, Yt, Dv, Yv, Gv, Db, Yb, Gb = data(args.iteration, args.start)
    preds = []
    for m in range(c.get("members", 1)):
        preds.append(dvn.train_preds(Dt, Yt, {"val": Dv, "beam": Db}, steps=c.get("steps", args.steps), lr=c.get("lr", 3e-4),
                                     wd=c.get("wd", 1e-4), seed=m, bootstrap=c.get("members", 1) > 1,
                                     arch=c.get("arch")))
    curve = []
    for k in range(len(preds[0]["val"])):
        step = preds[0]["val"][k][0]
        pv = np.mean([p["val"][k][1] for p in preds], axis=0)
        pb = np.mean([p["beam"][k][1] for p in preds], axis=0)
        curve.append({"step": step, "val_rank": spear(pv, Yv), "val_r": float(np.corrcoef(pv, Yv)[0, 1]),
                      "val_in_niche": grouped(pv, Yv, Gv), "beam_rank": spear(pb, Yb),
                      "beam_r": float(np.corrcoef(pb, Yb)[0, 1]), "beam_in_round": grouped(pb, Yb, Gb),
                      "beam_mean_pred": float(pb.mean()), "beam_mean_true": float(Yb.mean())})
    out = REPO / "debug_outputs/noevolution/studies" / f"it{args.iteration}_vf_tweaks"
    out.mkdir(exist_ok=True)
    (out / f"c{args.config:02d}.json").write_text(json.dumps({"config": c, "curve": curve}, indent=1))
    es = max(curve, key=lambda r: r["val_rank"])
    print(f"[tweaks] {c['name']}: early stop at {es['step']}: val rank {es['val_rank']:.3f}, beam rank "
          f"{es['beam_rank']:.3f}, beam in-round {es['beam_in_round']:.3f}", flush=True)


def report(args) -> None:
    out = REPO / "debug_outputs/noevolution/studies" / f"it{args.iteration}_vf_tweaks"
    print(f"  {'config':38s} {'stop':>6s} {'val rank':>8s} {'in niche':>8s} {'beam rank':>9s} {'in round':>8s} "
          f"{'beam pred/true mean':>19s} | {'at the end: val':>15s} {'beam':>6s} {'in round':>8s}")
    for f in sorted(out.glob("c*.json")):
        R = json.loads(f.read_text())
        c, cv = R["config"], R["curve"]
        es = max(cv, key=lambda r: r["val_rank"])
        e = cv[-1]
        print(f"  {c['name']:38s} {es['step']:6d} {es['val_rank']:8.3f} {es['val_in_niche']:8.3f} {es['beam_rank']:9.3f} "
              f"{es['beam_in_round']:8.3f} {es['beam_mean_pred']:9.3f}/{es['beam_mean_true']:.3f} | "
              f"{e['val_rank']:15.3f} {e['beam_rank']:6.3f} {e['beam_in_round']:8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=int, default=0)
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--start", default="f4_m12-13_126")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    (report if args.report else run)(args)


if __name__ == "__main__":
    main()
