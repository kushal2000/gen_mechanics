#!/bin/bash
# One generation of an embodiment-evolution run (coevolution/evolution; README.md), then the next one submitted.
# Called by a run's .sub (archive.sub, map_elites.sub, dry_*.sub), which sets every variable below and SUB_PATH
# (itself, to resubmit).
#
#   EVO_RUN            run directory (debug_outputs/evolution/<name>): config.json, archive.json, state.json, gen_NNNN/
#   EVO_CONFIG         the run's EvolutionConfig JSON (read at init only; the run dir keeps its own copy)
#   SEED_MANIFEST      generation 0's hands (init only): assets/urdf/29oct_niche_seed/manifest.json for both runs
#   MAX_GEN            stop after this many generations
#   EPOCHS_PER_GEN     training epochs per generation (NUM_ENVS x 16 steps each)
#   EVAL_ENVS_PER_HAND eval_hands.py's parallel envs per hand (each scored on its first episode)
#   TRAIN_ENV          the env/agent settings: what the trainer gets on top of the generation's hand set
#   WANDB_PREFIX       wandb run names <prefix>_iter_<generation> (the logger appends the date and time), group <prefix>
#   NUM_ENVS, SEED, GAMMA, ...   the trainer's own variables (experiments/12oct_inhand_olympics/olympics_sapg_mlp.sub)
#
# A generation: train gen_G/manifest.json (from the last checkpoint; generation 0 from scratch) -> evaluate it ->
# propose children (parents on this generation's fitness) -> [archive: screen the children zero-shot] -> admit ->
# resubmit. Restartable: a resubmitted generation re-trains, re-evaluates, and keeps any children already proposed.
set -euo pipefail
cd /share/portal/kk837/gen_mechanics
PY=.venv_isaacsim/bin/python
NAME=$(basename "$EVO_RUN")
state() { $PY -c "import json,sys; print(json.load(open('$EVO_RUN/state.json'))[sys.argv[1]])" "$1"; }

[[ -f "$EVO_RUN/state.json" ]] || $PY -m coevolution.evolution.generation init --run-dir "$EVO_RUN" \
    --config "$EVO_CONFIG" --seed-manifest "$SEED_MANIFEST"
SCREENED=$($PY -c "from coevolution.evolution.config import EvolutionConfig as C; print(int(C.load('$EVO_RUN/config.json').screened))")
G=$(state generation)
CKPT=$(state checkpoint)
GDIR=$(printf "%s/gen_%04d" "$EVO_RUN" "$G")
echo "[evolution] $NAME generation $G on $(hostname), from ${CKPT:-scratch}"

# 1. train the population (one shared policy) from the last checkpoint, or from scratch
LOG_GROUP="evolution/$NAME/$(basename "$GDIR")"
case "$LOG_GROUP" in                                     # a resubmitted generation trains afresh
    evolution/?*/gen_[0-9][0-9][0-9][0-9]) rm -rf "debug_outputs/train_logs/$LOG_GROUP" ;;
    *) echo "[evolution] refusing to clear unexpected log group '$LOG_GROUP'"; exit 1 ;;
esac
CK_VARS=()
[[ -n "$CKPT" ]] && CK_VARS=(CHECKPOINT="$CKPT" CHECKPOINT_MODE=weights)
env HAND="multi:template:$GDIR/manifest.json" HAND_TAG=evo RUN_TAG="${NAME}_g$G" LOG_GROUP="$LOG_GROUP" \
    EPOCHS="$EPOCHS_PER_GEN" EXTRA_HYDRA="$TRAIN_ENV" WANDB_NAME="${WANDB_PREFIX}_iter_$G" WANDB_GROUP="$WANDB_PREFIX" \
    ${CK_VARS[@]+"${CK_VARS[@]}"} \
    bash experiments/12oct_inhand_olympics/olympics_sapg_mlp.sub > "$GDIR/train.log" 2>&1 || true
grep -q "MAX EPOCHS" "$GDIR/train.log" || { echo "[evolution] training did not finish; see $GDIR/train.log"; tail -20 "$GDIR/train.log"; exit 1; }
TRAIN_RUN=$(ls -d "debug_outputs/train_logs/$LOG_GROUP"/*/ | head -1)
NEW=$(ls -t "$TRAIN_RUN"*/nn/last_*_ep_*.pth | head -1)
cp "$NEW" "$GDIR/policy.pth"                             # the generation's policy, kept with the run

evaluate() {   # manifest out
    $PY coevolution/eval/eval_hands.py --manifest "$1" --checkpoint "$GDIR/policy.pth" --train-run "$TRAIN_RUN" \
        --out "$2" --envs-per-hand "$EVAL_ENVS_PER_HAND" > "${2%.json}.log" 2>&1 || true   # Kit's exit status lies
    [[ -s "$2" ]] || { echo "[evolution] eval wrote nothing; see ${2%.json}.log"; tail -20 "${2%.json}.log"; exit 1; }
}
# 2. evaluate the trained population, 3. propose children on that fitness
evaluate "$GDIR/manifest.json" "$GDIR/fitness.json"
$PY -m coevolution.evolution.generation propose --run-dir "$EVO_RUN" --fitness "$GDIR/fitness.json"
# 4. screened strategies: the children zero-shot, 5. admit
ADMIT=(--run-dir "$EVO_RUN" --checkpoint "$GDIR/policy.pth")
if [[ "$SCREENED" == 1 ]]; then
    if $PY -c "import json,sys; sys.exit(0 if json.load(open('$GDIR/children_manifest.json'))['hands'] else 1)"; then
        evaluate "$GDIR/children_manifest.json" "$GDIR/screen.json"
    else
        echo '{"hands": {}}' > "$GDIR/screen.json"
    fi
    ADMIT+=(--screen-fitness "$GDIR/screen.json")
fi
$PY -m coevolution.evolution.generation admit "${ADMIT[@]}"
$PY -m coevolution.evolution.generation status --run-dir "$EVO_RUN" | head -14

# 6. the next generation
if (( G + 1 < MAX_GEN )); then
    sbatch "$SUB_PATH"
else
    echo "[evolution] $NAME: reached MAX_GEN=$MAX_GEN"
fi
