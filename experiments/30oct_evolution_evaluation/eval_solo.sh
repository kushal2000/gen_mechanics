#!/bin/bash
# Final evaluation of the solo runs (solo_retrain/) and the single-hand long runs with the episode cap raised:
# eval_hands_cap.py, --max-successes 200, 20 envs per hand (first episode each, sampled actions). Skips checkpoints
# already evaluated and runs not finished yet; then prints each hand's mean and range over seeds.
#   bash experiments/30oct_evolution_evaluation/eval_solo.sh          (needs a GPU: the interactive allocation)
# Writes debug_outputs/evolution/solo_eval/<run>.json.
set -euo pipefail
cd /share/portal/kk837/gen_mechanics
E=experiments/30oct_evolution_evaluation; X=experiments/29oct_evolution_from_scratch
O=debug_outputs/evolution/solo_eval; mkdir -p $O
run() {   # name, train-run dir, checkpoint, manifest
    [[ -s $O/$1.json ]] && return 0
    timeout 2400 .venv_isaacsim/bin/python $E/eval_hands_cap.py --manifest "$4" --checkpoint "$3" --train-run "$2" \
        --out $O/$1.json --envs-per-hand 20 --max-successes 200 > $O/$1.log 2>&1 || true
    [[ -s $O/$1.json ]] && echo "$1: $(tail -1 $O/$1.log | cut -c1-120)" || echo "$1: eval failed, see $O/$1.log"
}
for h in g0018_c00 g0029_c02 seed_f3_cut_j2; do
  for s in 100 101 102; do
    TR=$(ls -d debug_outputs/train_logs/30oct_evolution_evaluation/solo_retrain/*_solo_${h}_s${s}_*/ 2>/dev/null | head -1)
    CK=$(ls $TR*/nn/last_*_ep_6360_*.pth 2>/dev/null | head -1 || true)
    [[ -n "$CK" ]] && run solo_${h}_s$s "$TR" "$CK" $E/solo_retrain/${h}_manifest.json
  done
done
declare -A MAN=([claude_pick_long]=$X/claude_pick/manifest.json [claude_runner_up_long]=$X/claude_pick/runner_up_manifest.json
                [random_pick_long]=$X/random_pick/manifest.json)
for n in claude_runner_up_long claude_pick_long random_pick_long; do
    TR=$(ls -d debug_outputs/train_logs/29oct_evolution_from_scratch/long_runs/*_evo_${n}_*/ | head -1)
    CK=$(ls $TR*/nn/last_*_ep_20352_*.pth 2>/dev/null | head -1 || true)
    [[ -n "$CK" ]] && run ${n}_ep20352 "$TR" "$CK" ${MAN[$n]}
done
.venv_isaacsim/bin/python - <<'PY'
import json, collections, numpy as np
from pathlib import Path
by = collections.defaultdict(list)
for f in sorted(Path("debug_outputs/evolution/solo_eval").glob("*.json")):
    v = list(json.loads(f.read_text())["hands"].values())[0]
    key = f.stem.rsplit("_s", 1)[0] if f.stem.startswith("solo_") else f.stem
    by[key].append((v["goals_per_episode"], v.get("done_max_successes", 0), v["done_dropped"]))
for k, rows in by.items():
    g = np.array([r[0] for r in rows])
    print(f"{k:32s} seeds {len(rows)}: goals/ep mean {g.mean():6.1f} (range {g.min():.1f}-{g.max():.1f}); "
          f"at the 200 cap {np.mean([r[1] for r in rows]):.2f}, dropped {np.mean([r[2] for r in rows]):.2f}")
PY
