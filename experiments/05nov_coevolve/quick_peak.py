"""Quick peak-fitness evals: for an archive proposed by a search from policy gen G, score only the 256 designs the
value net ranks highest (one 100-env chunk instead of 16) with the gen-G policy. The best design is almost always
among them, so this gives an early peak; the full 4,096-design eval stays the reference.

    python quick_peak.py --run debug_outputs/coevolve/v1/k5_none --gen 2 [--top 256] [--submit "sbatch args"]

Writes <run>/gen_<gen>/eval100_top256_policy_g<G-1>/ (c00_manifest.json, jobs.txt) and submits it.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL_SUB = REPO / "experiments/05nov_noevolution/studies/study_beam_eval.sub"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gen", type=int, required=True, help="the proposed archive's generation (policy gen = gen - 1)")
    ap.add_argument("--top", type=int, default=256)
    ap.add_argument("--submit", default="", help="extra sbatch arguments (partition, exclude)")
    ap.add_argument("--own", action="store_true", help="gen --gen's training archive with its own policy, ranked by "
                    "its training-log labels (labels.json) instead of the search's predictions")
    args = ap.parse_args()
    run = REPO / args.run
    g, pg = (args.gen, args.gen) if args.own else (args.gen, args.gen - 1)
    cfg = json.loads((run / "config.json").read_text())
    S = json.loads((run / "state.json").read_text())
    if args.own:
        pred = {i: v[0] for i, v in json.loads((run / f"gen_{g:04d}/labels.json").read_text()).items()}
    else:
        pred = json.loads((run / f"gen_{pg:04d}/search/round_{cfg['k']:02d}_archive.json").read_text())["pred"]
    man = json.loads((run / f"gen_{g:04d}/manifest.json").read_text())
    spec = {h["id"]: h["spec"] for h in man["hands"]}
    top = sorted(spec, key=lambda i: -pred.get(i, -1e9))[:args.top]
    ed = run / f"gen_{g:04d}/eval100_top{args.top}_policy_g{pg:04d}"
    ed.mkdir(exist_ok=True)
    mp = ed / "c00_manifest.json"
    mp.write_text(json.dumps({"generation": g, "grid": man["grid"], "hands": [{"id": i, "spec": spec[i]} for i in top]}))
    tr = S.get("train_runs", {}).get(str(pg)) or S["train_run"]
    pol = (run / f"gen_{pg:04d}/policy.pth").resolve()
    (ed / "jobs.txt").write_text(f"{mp} 100 {ed / 'c00_fitness.json'} {9000 + g * 100} {pol} {tr} 0.02\n")
    out = subprocess.run(["sbatch", "--array=0", f"--job-name=QP_{run.name}_g{g}", *shlex.split(args.submit),
                          "--export", f"ALL,JOBS={ed / 'jobs.txt'}", str(EVAL_SUB)], capture_output=True, text=True)
    print(f"[quick_peak] {run.name} gen {g} (policy gen {pg}): top {len(top)} -> {out.stdout.strip()} {out.stderr.strip()}")


if __name__ == "__main__":
    main()
