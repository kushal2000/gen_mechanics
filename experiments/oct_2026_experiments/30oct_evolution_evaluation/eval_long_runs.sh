#!/bin/bash
# The long single runs (experiments/oct_2026_experiments/29oct_evolution_from_scratch/long_*.sub) at one matched checkpoint, with the
# evolution runs' fitness (eval_hands.py: 10 envs per hand, first episode each, sampled actions).
#   bash experiments/oct_2026_experiments/30oct_evolution_evaluation/eval_long_runs.sh 10176
# Kept checkpoints (rl_games keeps epoch/636 = perfect squares, and the last): 636 2544 5724 10176 15900 20352 =
# the ends of iterations 0 3 8 15 24 31. Writes debug_outputs/evolution/long_runs_eval/<run>_ep<E>.json and prints a
# summary. Needs a GPU (run on the interactive allocation).
set -euo pipefail
cd /share/portal/kk837/gen_mechanics
EP=${1:?epoch}
X=experiments/oct_2026_experiments/29oct_evolution_from_scratch
O=debug_outputs/evolution/long_runs_eval
LR=debug_outputs/train_logs/29oct_evolution_from_scratch/long_runs
mkdir -p $O
declare -A MAN=([rl_only_long]=assets/urdf/29oct_niche_seed/manifest.json [claude_pick_long]=$X/claude_pick/manifest.json
                [claude_runner_up_long]=$X/claude_pick/runner_up_manifest.json [random_pick_long]=$X/random_pick/manifest.json)
for run in rl_only_long claude_pick_long claude_runner_up_long random_pick_long; do
    TR=$(ls -d $LR/*_evo_${run}_*/ 2>/dev/null | head -1)
    CK=$(ls $TR*/nn/last_*_ep_${EP}_*.pth 2>/dev/null | head -1 || true)
    [[ -z "$CK" ]] && { echo "$run: no epoch $EP checkpoint yet"; continue; }
    [[ -s $O/${run}_ep$EP.json ]] && { echo "$run: already evaluated"; continue; }
    timeout 1200 .venv_isaacsim/bin/python coevolution/eval/eval_hands.py --manifest ${MAN[$run]} --checkpoint "$CK" \
        --train-run "$TR" --out $O/${run}_ep$EP.json --envs-per-hand 10 > $O/${run}_ep$EP.log 2>&1 || true
done
.venv_isaacsim/bin/python - "$EP" <<'PY'
import json, sys, numpy as np
from pathlib import Path
ep = sys.argv[1]
for f in sorted(Path("debug_outputs/evolution/long_runs_eval").glob(f"*_ep{ep}.json")):
    h = json.loads(f.read_text())["hands"]; g = [v["goals_per_episode"] for v in h.values()]
    cap = np.mean([v.get("done_max_successes", 0) for v in h.values()])
    print(f"{f.stem}: best {max(g)} mean {np.mean(g):.2f} hands with goals {sum(x > 0 for x in g)}/{len(g)} "
          f"QD {sum(g):.1f} episodes at the 50-goal cap {cap:.2f}")
PY
