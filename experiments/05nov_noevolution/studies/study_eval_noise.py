"""Study 1: how noisy is the loop's evaluation, and how much does picking the best inflate it (the winner's curse)?

    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/study_eval_noise.py [--iteration 1] [--top 8]
        [--random 8] [--envs 100]

Per niche, the --top hands by the iteration's own evaluation (gen_NNNN/fitness.json: 4 envs per hand, first episode
each) and --random other hands, re-scored with --envs envs each under the same iteration's policy (eval_hands.py,
sampled actions, a different seed). For each group: the 4-env score against the 100-env score. The top group's gap is
selection on noise; the random group's is noise alone. Writes debug_outputs/noevolution/studies/it<N>_noise/
(manifests, eval logs, scores.json with every hand's two scores) and prints the summary.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import statistics as st
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUN = REPO / "debug_outputs/evolution/noevolution_4096"
FIT = "goals_per_episode"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", type=int, default=1)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--random", type=int, default=8)
    ap.add_argument("--envs", type=int, default=100)
    ap.add_argument("--chunk-hands", type=int, default=256, help="hands per eval scene (x --envs envs)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    g = RUN / f"gen_{args.iteration:04d}"
    man = json.loads((g / "manifest.json").read_text())
    fit = json.loads((g / "fitness.json").read_text())["hands"]
    train_run = sorted(glob.glob(str(REPO / "debug_outputs/train_logs/evolution/noevolution_4096" / g.name / "*/")))[0]
    out = REPO / "debug_outputs/noevolution/studies" / f"it{args.iteration}_noise"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    by = defaultdict(list)
    for h in man["hands"]:
        by[h["niche"]].append(h)
    chosen = []
    for n, hs in by.items():
        ranked = sorted(hs, key=lambda h: (-fit[h["spec"]][FIT], -fit[h["spec"]].get("reward_per_step", 0.0)))
        top, rest = ranked[:args.top], ranked[args.top:]
        chosen += [dict(h, group="top") for h in top] + [dict(h, group="random") for h in rng.sample(rest, args.random)]
    scores = {}
    for k in range(0, len(chosen), args.chunk_hands):
        part = chosen[k:k + args.chunk_hands]
        mp = out / f"chunk{k // args.chunk_hands:02d}_manifest.json"     # template manifests must end in manifest.json
        mp.write_text(json.dumps({"generation": man["generation"], "grid": man["grid"],
                                  "hands": [{"id": h["id"], "spec": h["spec"], "niche": h["niche"]} for h in part]}))
        fo = out / f"chunk{k // args.chunk_hands:02d}_fitness.json"
        if not fo.exists():
            print(f"[noise] scoring hands {k}-{k + len(part) - 1} x {args.envs} envs", flush=True)
            with open(out / f"chunk{k // args.chunk_hands:02d}.log", "w") as log:
                subprocess.run([sys.executable, str(REPO / "coevolution/eval/eval_hands.py"), "--manifest", str(mp),
                                "--checkpoint", str(g / "policy.pth"), "--train-run", train_run, "--out", str(fo),
                                "--envs-per-hand", str(args.envs), "--seed", str(args.seed)],
                               stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
        scores.update(json.loads(fo.read_text())["hands"])
    rows = [{"id": h["id"], "niche": h["niche"], "group": h["group"], "loop4": fit[h["spec"]][FIT],
             f"env{args.envs}": scores[h["spec"]][FIT]} for h in chosen if h["spec"] in scores]
    (out / "scores.json").write_text(json.dumps(rows, indent=1))
    hi = f"env{args.envs}"
    print(f"\n[noise] iteration {args.iteration}: {len(rows)} hands, 4-env loop score vs {args.envs}-env re-score")
    for grp in ("top", "random"):
        r = [x for x in rows if x["group"] == grp]
        a, b = [x["loop4"] for x in r], [x[hi] for x in r]
        print(f"  {grp:6s} ({len(r)} hands): loop 4-env mean {st.mean(a):.3f}, {args.envs}-env mean {st.mean(b):.3f}, "
              f"gap {st.mean(a) - st.mean(b):+.3f}")
    a, b = [x["loop4"] for x in rows], [x[hi] for x in rows]
    print(f"  correlation of the two scores over all chosen hands: {st.correlation(a, b):.2f}")
    print(f"\n  per niche: top-{args.top} 4-env mean -> {args.envs}-env mean | random-{args.random} 100-env mean | "
          f"best hand by {args.envs} envs")
    for n in by:
        t = [x for x in rows if x["niche"] == n and x["group"] == "top"]
        rnd = [x for x in rows if x["niche"] == n and x["group"] == "random"]
        best = max(t + rnd, key=lambda x: x[hi])
        print(f"  {n:10s} {st.mean(x['loop4'] for x in t):.2f} -> {st.mean(x[hi] for x in t):.2f} | "
              f"{st.mean(x[hi] for x in rnd):.2f} | {best['id']} {best[hi]:.2f}")


if __name__ == "__main__":
    main()
