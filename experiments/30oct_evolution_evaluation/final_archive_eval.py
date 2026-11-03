"""The end-of-run archives re-scored under each run's FINAL policy (the README's to-do).

    .venv_isaacsim/bin/python experiments/30oct_evolution_evaluation/final_archive_eval.py [--runs baseline map_elites]
        [--top-k 3] [--envs-per-hand 20]      (needs a GPU: the interactive allocation)

The archive QD in state.json sums each niche's best hand by its LATEST training evaluation, which can be iterations old
(the policy has moved on; greedy also stops training most niches). Here, per run and niche: the top-k hands by that
latest value (always including the niche's current holder) are evaluated again with the loop's own evaluation
(coevolution/eval/eval_hands.py: sampled actions, first episode per env, the 50-goal cap, so comparable with the
loop's QD) under the run's last policy (gen_0063/policy.pth and its training run), with more envs per hand. A niche's
final score is its best candidate's; QD = their sum; a niche "has goals" above 0. Writes, under
debug_outputs/evolution/final_archive_eval/: <run>_manifest.json, <run>_fitness.json, and summary.json (+ printed table).
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from coevolution.evolution.archive import Archive  # noqa: E402

RUNS = REPO / "debug_outputs/evolution"
TRAIN_LOGS = REPO / "debug_outputs/train_logs/evolution"
OUT = RUNS / "final_archive_eval"


def candidates(run: str, k: int) -> dict[str, list]:
    arch = Archive.load(RUNS / run / "archive.json")
    by: dict[str, list] = {}
    for r in arch.records.values():
        e = r.latest(screen=False)
        if e is not None:
            by.setdefault(r.niche, []).append((float(e["value"]), float(e["goals_per_episode"]), int(e["generation"]), r))
    return {n: sorted(v, key=lambda t: -t[0])[:k] for n, v in by.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["baseline", "map_elites"])
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--envs-per-hand", type=int, default=20)
    ap.add_argument("--tag", default="", help="appended to every output file name (e.g. _it73_100env)")
    ap.add_argument("--finger-pos-fraction", type=float, default=None,
                    help="table-safe resets (e.g. 0.1): evaluate with eval_hands_cap.py at the same 50-goal cap; outputs get a _fpf<value> suffix")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = ("" if args.finger_pos_fraction is None else f"_fpf{args.finger_pos_fraction:g}") + args.tag
    summary = {}
    for run in args.runs:
        last = max(int(Path(p).name[4:]) for p in glob.glob(str(RUNS / run / "gen_*")) if (Path(p) / "policy.pth").exists())
        gdir = RUNS / run / f"gen_{last:04d}"
        grid = json.loads((gdir / "manifest.json").read_text())["grid"]
        cand = candidates(run, args.top_k)
        hands = {r.id: r.spec for v in cand.values() for (_, _, _, r) in v}
        man = OUT / f"{run}{tag}_manifest.json"          # template manifest refs must end in manifest.json
        man.write_text(json.dumps({"generation": last, "grid": grid,
                                   "hands": [{"id": i, "spec": s} for i, s in sorted(hands.items())]}, indent=1))
        train_run = sorted(glob.glob(str(TRAIN_LOGS / run / f"gen_{last:04d}" / "*/")))[0]
        fit = OUT / f"{run}_fitness{tag}.json"
        if not fit.exists():
            print(f"[final] {run}: {len(cand)} niches, {len(hands)} candidate hands under gen_{last:04d}'s policy", flush=True)
            cmd = [sys.executable, str(REPO / "coevolution/eval/eval_hands.py")]
            if args.finger_pos_fraction is not None:     # the copy with the reset option, at the loop's 50-goal cap
                cmd = [sys.executable, str(REPO / "experiments/30oct_evolution_evaluation/eval_hands_cap.py"),
                       "--max-successes", "50", "--finger-pos-fraction", str(args.finger_pos_fraction)]
            subprocess.run(cmd + ["--manifest", str(man), "--checkpoint", str(gdir / "policy.pth"), "--train-run", train_run,
                                  "--out", str(fit), "--envs-per-hand", str(args.envs_per_hand)],
                           stdout=open(OUT / f"{run}_eval{tag}.log", "w"), stderr=subprocess.STDOUT, cwd=REPO)
        scores = json.loads(fit.read_text())["hands"]
        by_spec = {v["spec"] if isinstance(v, dict) and "spec" in v else k: v for k, v in scores.items()}
        niches = {}
        for n, v in sorted(cand.items()):
            rows = []
            for (val, goals, gen, r) in v:
                s = scores.get(r.spec) or by_spec.get(r.spec)
                rows.append({"hand": r.id, "stale_goals": goals, "stale_iter": gen,
                             "final_goals": None if s is None else float(s["goals_per_episode"])})
            best = max((x for x in rows if x["final_goals"] is not None), key=lambda x: x["final_goals"], default=None)
            niches[n] = {"best": best, "candidates": rows}
        finals = [d["best"]["final_goals"] for d in niches.values() if d["best"]]
        stale = [d["candidates"][0]["stale_goals"] for d in niches.values()]
        summary[run] = {"policy_iteration": last, "niches": len(niches), "niches_with_goals": sum(1 for g in finals if g > 0),
                        "qd_final": float(sum(finals)), "qd_stale": float(sum(stale)),
                        "best_final": max(finals) if finals else None, "per_niche": niches}
        print(f"[final] {run} (policy of iteration {last}): QD {summary[run]['qd_final']:.1f} (stale {summary[run]['qd_stale']:.1f}), "
              f"{summary[run]['niches_with_goals']}/{len(niches)} niches with goals, best {summary[run]['best_final']}", flush=True)
    (OUT / f"summary{tag}.json").write_text(json.dumps(summary, indent=1, default=str))
    names = sorted({n for s in summary.values() for n in s["per_niche"]})
    print("\n| niche | " + " | ".join(summary) + " |\n|---|" + "---|" * len(summary))
    for n in names:
        cells = []
        for run, s in summary.items():
            d = s["per_niche"].get(n)
            cells.append("-" if not d or not d["best"] else f"{d['best']['final_goals']:.1f} ({d['best']['hand']})")
        print(f"| {n} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
