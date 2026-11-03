"""The top hands of each run's final archive, re-evaluated with 100 envs under both reset modes.

    .venv_isaacsim/bin/python experiments/30oct_evolution_evaluation/top_hands_100env.py [--top 5] [--envs-per-hand 100]

The shortlist per run: the union of the --top niche winners of final_archive_eval.py's two evaluations (the runs'
own reset, finger_pos_fraction 1.0, and the table-safe 0.1), so neither reset mode picks it. Each hand is evaluated
under the run's final policy with eval_hands_cap.py at the loop's 50-goal cap, --envs-per-hand envs, once per reset
mode, reporting goals per episode +- its standard error and the drop rate. Writes
debug_outputs/evolution/final_archive_eval/top_hands_100env.json and prints the table.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "debug_outputs/evolution"
FINAL = RUNS / "final_archive_eval"
MODES = (("runs' reset (1.0)", 1.0), ("table-safe (0.1)", 0.1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--envs-per-hand", type=int, default=100)
    args = ap.parse_args()
    summaries = [json.loads((FINAL / "summary.json").read_text()), json.loads((FINAL / "summary_fpf0.1.json").read_text())]
    result = {}
    for run in ("map_elites", "baseline"):
        short = {}
        for s in summaries:
            winners = sorted(((d["best"]["final_goals"], n, d["best"]["hand"]) for n, d in s[run]["per_niche"].items()
                              if d["best"]), reverse=True)[:args.top]
            for _, n, h in winners:
                short[h] = n
        full = json.loads((FINAL / f"{run}_manifest.json").read_text())
        spec_of = {h["id"]: h["spec"] for h in full["hands"]}
        man = FINAL / f"top_{run}_manifest.json"
        man.write_text(json.dumps({"generation": full["generation"], "grid": full["grid"],
                                   "hands": [{"id": h, "spec": spec_of[h]} for h in sorted(short)]}, indent=1))
        last = summaries[0][run]["policy_iteration"]
        gdir = RUNS / run / f"gen_{last:04d}"
        train_run = sorted(glob.glob(str(REPO / "debug_outputs/train_logs/evolution" / run / f"gen_{last:04d}" / "*/")))[0]
        result[run] = {h: {"niche": n} for h, n in short.items()}
        for label, fpf in MODES:
            out = FINAL / f"top_{run}_{args.envs_per_hand}env_fpf{fpf:g}.json"
            if not out.exists():
                print(f"[top] {run}: {len(short)} hands x {args.envs_per_hand} envs, {label}", flush=True)
                subprocess.run([sys.executable, str(REPO / "experiments/30oct_evolution_evaluation/eval_hands_cap.py"),
                                "--manifest", str(man), "--checkpoint", str(gdir / "policy.pth"), "--train-run", train_run,
                                "--out", str(out), "--envs-per-hand", str(args.envs_per_hand), "--max-successes", "50",
                                "--finger-pos-fraction", str(fpf)],
                               stdout=open(out.with_suffix(".log"), "w"), stderr=subprocess.STDOUT, cwd=REPO)
            hands = json.loads(out.read_text())["hands"]
            for h in short:
                v = hands[spec_of[h]]
                result[run][h][label] = {"goals": v["goals_per_episode"], "sem": v.get("goals_sem"),
                                         "dropped": v.get("done_dropped")}
    (FINAL / "top_hands_100env.json").write_text(json.dumps(result, indent=1))
    for run, hands in result.items():
        print(f"\n{run} ({args.envs_per_hand} envs per hand, 50-goal cap; goals/ep +- s.e., dropped)")
        print("| hand | niche | " + " | ".join(l for l, _ in MODES) + " |\n|---|---|" + "---|" * len(MODES))
        for h, d in sorted(hands.items(), key=lambda kv: -max(kv[1][l]["goals"] for l, _ in MODES)):
            cells = [f"{d[l]['goals']:.1f} +- {d[l]['sem']:.1f} ({d[l]['dropped']:.0%})" for l, _ in MODES]
            print(f"| {h} | {d['niche']} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
