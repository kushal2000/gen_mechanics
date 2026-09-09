#!/bin/bash
set -euo pipefail
cd /share/portal/kk837/gen_mechanics
source .venv_isaacsim/bin/activate
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PXR_WORK_THREAD_LIMIT=2
# Direct P2P timed out on portal-compute-02's B0/B1 GPU pair even for a
# 32-element broadcast. Disabling it passed on the same allocation.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PHASE="${PHASE:-tune}"
export STUDY_ID="${STUDY_ID:-scaling_laws_shared_privileged_v1}"
STUDY_SCRIPT=experiments/decentralized_control/scaling_laws/study.py
if [[ "$PHASE" == tune ]]; then
    PAIR=$(python "$STUDY_SCRIPT" trial "${SLURM_ARRAY_TASK_ID:-0}")
    read -r NUM_ENVS_PER_GPU GLOBAL_MINIBATCH <<< "$PAIR"
    export MAX_EPOCHS="${MAX_EPOCHS:-60}"
    export SEED="${SEED:-123}"
elif [[ "$PHASE" == train ]]; then
    if [[ -z "${NUM_ENVS_PER_GPU:-}" || -z "${GLOBAL_MINIBATCH:-}" ]]; then
        PAIR=$(python "$STUDY_SCRIPT" best --model "d${D_MODEL}_l${TRANSFORMER_LAYERS}" --study "$STUDY_ID")
        read -r NUM_ENVS_PER_GPU GLOBAL_MINIBATCH <<< "$PAIR"
    fi
    export MAX_EPOCHS="${MAX_EPOCHS:-1000000}"
    export SEED="${SEED:-$((100 + 100 * ${SLURM_ARRAY_TASK_ID:-0}))}"
else
    echo 'PHASE must be tune or train'; exit 1
fi
export NUM_ENVS_PER_GPU GLOBAL_MINIBATCH
export EXPL_BLOCK_SIZE=$((NUM_ENVS_PER_GPU / 6))
export WANDB_PROJECT="${WANDB_PROJECT:-gen_mechanics}"
export WANDB_ENTITY="${WANDB_ENTITY:-kk837}"
export WANDB_GROUP="${WANDB_GROUP:-$STUDY_ID}"
export WANDB_ACTIVATE="${WANDB_ACTIVATE:-0}"
export LEARNING_RATE="${LEARNING_RATE:-0.0001}"
export RUN_NAME="0_scale_${PHASE}_d${D_MODEL}_l${TRANSFORMER_LAYERS}_e${NUM_ENVS_PER_GPU}_b${GLOBAL_MINIBATCH}_s${SEED}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-0}"

(( GLOBAL_MINIBATCH % 2 == 0 )) || { echo 'GLOBAL_MINIBATCH must divide across 2 GPUs'; exit 1; }
(( NUM_ENVS_PER_GPU == 6 * EXPL_BLOCK_SIZE )) || { echo 'Keep six SAPG blocks per rank for evaluation compatibility'; exit 1; }
LOCAL_BATCH=$((GLOBAL_MINIBATCH / 2))
ROLLOUT=$((NUM_ENVS_PER_GPU * 16))
# central_value_config=null, so the SAPG-augmented actor rollout is the only
# dataset; the separate critic rollout this used to also check no longer
# exists. PPODataset._get_item hands any remainder to the LAST minibatch, so a
# non-dividing size makes one step much wider than the rest (global 98304 at
# 12288 envs/GPU: three of 49152 then one of 81920) and adds a compile shape.
AUG_ROLLOUT=$((ROLLOUT + ROLLOUT / 6))
(( AUG_ROLLOUT % LOCAL_BATCH == 0 )) || {
    echo "Minibatch $LOCAL_BATCH/rank must divide the SAPG-augmented rollout $AUG_ROLLOUT"; exit 1;
}
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    export SCALING_RUN_DIR="/tmp/scaling_dry_run/$RUN_NAME"
    export LOCAL_RANK=0
    export WORLD_SIZE=2
    bash experiments/decentralized_control/scaling_laws/run_rank.sh
    exit
fi
RUN_ROOT="${SCALING_RUN_ROOT:-$PWD/debug_outputs/scaling_laws}"
mkdir -p "$RUN_ROOT"
SCALING_RUN_DIR=$(mktemp -d "$RUN_ROOT/${RUN_NAME}_XXXXXX")
export SCALING_RUN_DIR
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/${USER}_scale_${SLURM_JOB_ID:-local}"
export OMNI_KIT_CACHE_PATH="${SLURM_TMPDIR:-/tmp}/${USER}_scale_kit_${SLURM_JOB_ID:-local}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$OMNI_KIT_CACHE_PATH"
echo "Run: $SCALING_RUN_DIR"
echo "shared_actor_critic=d${D_MODEL}/L${TRANSFORMER_LAYERS} heads=1 ff=2 seed=$SEED"
echo "global_envs=$((2 * NUM_ENVS_PER_GPU)) global_minibatch=$GLOBAL_MINIBATCH local_minibatch=$LOCAL_BATCH"
# Slurm resolves --output before the job starts, so it cannot be pointed
# inside a directory mktemp has not created yet. Link the job's streams in
# after the fact: everything for a run is then reachable from its own
# directory, while an early failure still lands in a file that exists.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    while read -r stream path; do
        [[ -n "$path" && "$path" != /dev/null ]] && ln -sfn "$path" "$SCALING_RUN_DIR/slurm.$stream"
    done < <(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
             | tr ' ' '\n' | sed -n 's|^StdOut=|out |p; s|^StdErr=|err |p')
fi
git rev-parse HEAD > "$SCALING_RUN_DIR/git_commit.txt"
git diff > "$SCALING_RUN_DIR/worktree.patch"
nvidia-smi > "$SCALING_RUN_DIR/gpus.txt"
python "$STUDY_SCRIPT" record --status running
experiments/monitor_usage.sh "$SCALING_RUN_DIR/usage" 30 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
set +e
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    --log-dir "$SCALING_RUN_DIR/torchrun" --redirects 3 --tee 3 \
    --no-python bash experiments/decentralized_control/scaling_laws/run_rank.sh
STATUS=$?
set -e
if (( STATUS == 0 )); then
    python "$STUDY_SCRIPT" record --status completed
else
    python "$STUDY_SCRIPT" record --status failed
fi
exit "$STATUS"
