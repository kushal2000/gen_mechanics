#!/bin/bash
# Grammar-solo comparison, step 2: one design's solo HORA training (its own
# controller, its own grasp cache, 4096 envs, a fixed seed). One job per
# design, indexed through job_manifest.csv (make_job_manifest.py) by
# SLURM_ARRAY_TASK_ID. Submitted as a throttled array (--array 0-N%2, the
# cluster's 2-GPU-at-a-time cap); see README.md in this directory for the
# exact `cluster submit` invocation.
#
# Required env (set via `cluster submit --env K=V` or exported before
# sourcing): GM_CODE_DIR (the gm_<sha> snapshot), GM_RUN_ROOT (holds
# job_manifest.csv and gets train/<group>/<design_index>/ written under it).
# Optional: SECS (wall-clock budget for train.py incl. boot; default 2100 =
# 35 min), SEED (agent.params.seed; default 42, fixed across every job in
# this comparison so only the design differs).
set +e; set -u -o pipefail  # the cluster wrapper's outer script runs under `set -e`; we want to
# log a timeout (expected: `timeout -k 30` returns 124) or a real failure, not die silently.

: "${GM_CODE_DIR:?set GM_CODE_DIR to the gm_<sha> snapshot dir}"
: "${GM_RUN_ROOT:?set GM_RUN_ROOT to the run's output root (holds job_manifest.csv)}"
G=$GM_CODE_DIR
PY=/data/pulkitag/users/mpeticco/code/gen_mechanics/.venv_isaacsim/bin/python3
export PYTHONPATH="$G:$G/third_party/rl_games"
export OMNI_KIT_ACCEPT_EULA=YES
export WANDB_MODE=disabled
export GEN_MECH_HAND_SOURCE_ROOT=/data/pulkitag/users/mpeticco/code/hand_models
export OMNI_KIT_CACHE_PATH=/tmp/mpeticco_kit_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
trap 'rm -rf "$OMNI_KIT_CACHE_PATH"' EXIT
cd "$G"

MANIFEST=$GM_RUN_ROOT/job_manifest.csv
T=${SLURM_ARRAY_TASK_ID:?this script is meant to run as a SLURM array task}
ROW=$(awk -F, -v idx="$T" 'NR==1{next} $1==idx{print; found=1} END{if(!found) exit 1}' "$MANIFEST")
if [ -z "$ROW" ]; then
  echo "no job_manifest.csv row for array index $T (manifest: $MANIFEST)"
  exit 2
fi
GROUP=$(echo "$ROW" | cut -d, -f2)
DIDX=$(echo "$ROW" | cut -d, -f3)
SOURCE=$(echo "$ROW" | cut -d, -f4)
SHA=$(echo "$ROW" | cut -d, -f5)
POP=$(echo "$ROW" | cut -d, -f6)
CACHE=$(echo "$ROW" | cut -d, -f7)
OUT=$GM_RUN_ROOT/train/$GROUP/$DIDX
mkdir -p "$OUT"
printf '{"group": "%s", "design_index": "%s", "source": "%s", "sha256": "%s"}\n' \
  "$GROUP" "$DIDX" "$SOURCE" "$SHA" > "$OUT/design_meta.json"
echo "array $T -> group=$GROUP design_index=$DIDX source=$SOURCE code=$(cat "$G/SYNCED_COMMIT")"

SECS=${SECS:-2100}
SEED=${SEED:-42}
timeout -k 30 "$SECS" "$PY" coevolution/train.py --task GenMech-InHandReorient-Direct-v0 \
  --agent rl_games_anyrotate_ppo_cfg_entry_point --headless \
  env.task_profile=hora env.scene.num_envs=4096 \
  env.anyrotate.grasp_cache="$CACHE" env.anyrotate.grasp_cache_generate=false \
  env.assets.hand_population="$POP" agent.params.seed="$SEED" \
  "hydra.run.dir=$OUT" > "$OUT/train.log" 2>&1 &
TP=$!
sleep 30
CHILD=$(pgrep -P "$TP" | head -1 || true)
"$PY" -m isaacsimenvs.inhand_reorient.tools.poll_design_scores \
  "$OUT/per_design_scores_rank0.json" "$OUT/windows.jsonl" "${CHILD:-$TP}" &
RC=0; wait "$TP" || RC=$?
wait || true
echo "train exit $RC (124 = the planned time limit) group=$GROUP design_index=$DIDX"
