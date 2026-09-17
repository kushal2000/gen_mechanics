#!/bin/bash
# Start either arm of the experiment on a population file.
#
#   launch.sh coevo    <population.json> <label> [KEY=VALUE ...]
#   launch.sh baseline <population.json> <label> [KEY=VALUE ...]
#
# <label> names the study: wandb group, MODEL_TAG prefix, and for co-evolution
# the folder assets/populations/<label>/gen_<k>/ that accumulates the record.
# Extra KEY=VALUE pairs are exported to the job (EPOCHS_PER_GEN, KEEP, MAX_GEN,
# EPOCHS, MAX_CONT, SEED, CHECKPOINT, RESUME_TOL, ...).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ARM="${1:?coevo | baseline}" POP="${2:?population .json}" LABEL="${3:?label}"; shift 3
[[ -f "$POP" ]] || { echo "no such population: $POP"; exit 1; }
POP=$(readlink -f "$POP")
EXTRA=""; for kv in "$@"; do EXTRA+=",$kv"; done
mkdir -p /share/portal/kk837/gen_mechanics/debug_outputs/train_logs/coevolution
case "$ARM" in
    coevo)
        DIR=/share/portal/kk837/gen_mechanics/assets/populations/$LABEL
        [[ -e "$DIR/gen_0" ]] && { echo "$DIR/gen_0 exists; pick a new label or remove it"; exit 1; }
        mkdir -p "$DIR/gen_0"; cp "$POP" "$DIR/gen_0/population.json"
        sbatch --job-name="${LABEL}_g000" --export="ALL,COEVO_DIR=$DIR,GEN=0,STUDY_ID=$LABEL$EXTRA" "$HERE/coevo_gen.sub" ;;
    baseline)
        sbatch --job-name="${LABEL}_c01" --export="ALL,ROBOT_SPEC=$POP,STUDY_ID=$LABEL$EXTRA" "$HERE/baseline.sub" ;;
    *) echo "unknown arm: $ARM"; exit 1 ;;
esac
