"""How long and how fast to train the design value net: a train/val split of iteration --iteration's 4,096 designs
(100-env labels), the validation curve of design_value_net's per-finger transformer for each learning rate x weight
decay, every --eval-every steps.

    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/study_vf_hparams.py [--iteration 3]

Split: --val-frac of each niche to validation (seed 0, stratified). Writes it<N>_vf_hparams.json and prints, per run,
the best validation point and the curve.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import design_value_net as dvn  # noqa: E402
import study_value as sv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    ap.add_argument("--wds", type=float, nargs="+", default=[1e-4, 1e-2])
    args = ap.parse_args()
    g = REPO / f"debug_outputs/evolution/noevolution_4096/gen_{args.iteration:04d}"
    man = json.loads((g / "manifest.json").read_text())
    lab = {}
    for f in sorted((REPO / "debug_outputs/noevolution/studies" / f"it{args.iteration}_env100").glob("c*_fitness.json")):
        lab.update({s: v["goals_per_episode"] for s, v in json.loads(f.read_text())["hands"].items()})
    hs = [h for h in man["hands"] if h["spec"] in lab]
    rng = np.random.default_rng(0)
    val = set()
    by = {}
    for k, h in enumerate(hs):
        by.setdefault(h["niche"], []).append(k)
    for idx in by.values():
        val |= set(rng.choice(idx, int(round(args.val_frac * len(idx))), replace=False).tolist())
    tr = [k for k in range(len(hs)) if k not in val]
    va = sorted(val)
    D = [sv._load_design(h["spec"]) for h in hs]
    Y = [lab[h["spec"]] for h in hs]
    print(f"[hparams] {len(tr)} train / {len(va)} val designs (stratified by niche); val label sd {np.std([Y[k] for k in va]):.3f}",
          flush=True)
    runs = {}
    for lr in args.lrs:
        for wd in args.wds:
            t0 = time.time()
            c = dvn.train_curve([D[k] for k in tr], [Y[k] for k in tr], [D[k] for k in va], [Y[k] for k in va],
                                [hs[k]["niche"] for k in va], steps=args.steps, lr=lr, wd=wd, eval_every=args.eval_every)
            best = max(c, key=lambda r: r["val_rank"])
            runs[f"lr{lr:g}_wd{wd:g}"] = c
            print(f"[hparams] lr {lr:g} wd {wd:g} ({(time.time() - t0) / 60:.1f} min): best val rank {best['val_rank']:.3f} "
                  f"(r {best['val_r']:.3f}, mse {best['val_mse']:.4f}, in-niche {best['val_rank_in_group']:.3f}) at step "
                  f"{best['step']}; at the end {c[-1]['val_rank']:.3f} (train mse {c[-1]['train_mse']:.4f})", flush=True)
            print("    val rank by step: " + " ".join(f"{r['step'] // 1000 if r['step'] % 1000 == 0 else ''}"
                                                    f"{':' if r['step'] % 1000 == 0 else ''}{r['val_rank']:.2f}"
                                                    for r in c if r["step"] % 1000 == 0), flush=True)
    (REPO / "debug_outputs/noevolution/studies" / f"it{args.iteration}_vf_hparams.json").write_text(
        json.dumps({"train": len(tr), "val": len(va), "runs": runs}, indent=1))


if __name__ == "__main__":
    main()
