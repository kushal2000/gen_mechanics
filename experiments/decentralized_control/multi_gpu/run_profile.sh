#!/bin/bash
set -euo pipefail
cd /share/portal/kk837/gen_mechanics
source .venv_isaacsim/bin/activate
export OMP_NUM_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/${USER}_hybrid_inductor_${SLURM_JOB_ID:-local}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" debug_outputs/bench_logs
RUN_DIR=$(mktemp -d "debug_outputs/bench_logs/hybrid_${PROFILE_GPUS}gpu_${SLURM_JOB_ID:-local}_XXXXXX")
echo "Results: $PWD/$RUN_DIR"
nvidia-smi
# Fixed GLOBAL batch: local batches are 8192 (2 GPUs) or 4096 (4 GPUs).
# Backward includes DDP gradient synchronization. No optimizer/PPO/simulator.
for attention in explicit sdpa; do
    python -m torch.distributed.run --standalone --nnodes=1 \
        --nproc_per_node="$PROFILE_GPUS" \
        experiments/decentralized_control/profile_hybrid_transformer.py \
        --batch "${GLOBAL_BATCH:-16384}" --compile 1 --attention "$attention" \
        --output "$RUN_DIR/${attention}.json"
done
