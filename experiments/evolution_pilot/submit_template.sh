#!/usr/bin/env bash
# Evolution pilot (E-R2', plan-rl-grammar-tuning.md's "Revision, 2026-09-27"):
# exact `cluster submit` command for a single-job MAP-Elites run of
# isaacsimenvs.inhand_reorient.evolution.driver. NOT YET SUBMITTED -- this is
# a template a human runs by hand after reading the settings below; nothing
# in this repo executes it automatically, and this branch's own rule is "no
# cluster use" for the worker that wrote it.
#
# One SLURM job runs the WHOLE multi-generation loop in one process (per
# coevolution-reuse-survey.md's own lesson: no self-resubmitting chains,
# --auto-requeue is also NOT used here since the driver's own resume path
# already handles a hard time-limit kill correctly -- rerun the identical
# command with the same --run-dir and it continues from the last completed
# generation).
#
# Recommended settings for one ~5 GPU-hour pilot run -- CAVEAT (see the
# worker report): this pass's own --designs 64 --num-envs 4096 timing run
# did not finish within the session (killed mid-generation-0's scene
# boot/author, itself informative: 64-design/4096-env boot alone ran past
# 4 minutes without reaching epoch 1, well past the tiny run's 3-27s boot
# at 16 designs/512 envs). The numbers below are therefore a REASONED
# ESTIMATE from a prior same-codebase benchmark
# (project-notes/grammar/LOG.md, "Pilot prep": "10-min 32-design run at
# 4096 envs: ~113k fps"), not a measurement from this pass -- rerun the
# timing run locally (see the worker report's exact command) before
# trusting these numbers for a real cluster allocation.
#
#   local fps (32 designs, 4096 envs, prior benchmark): ~113,000 env-steps/s
#   cluster slowdown factor (project-notes/grammar/LOG.md, "Cluster
#     throughput investigation": the env step is CPU-single-thread bound,
#     not GPU bound, so this factor is expected to transfer to the
#     population path too): ~3.4x  ->  cluster fps ~= 33,000
#   5 GPU-hours = 18,000 s -> ~594M env-steps total -> / (4096 envs * 16
#     horizon_length) ~= 9,050 epochs total budget, MINUS boot/select
#     overhead per generation (boot was <1 min warm-cache on ada6000-shared
#     per LOG.md, but budget more for a 64-design population's heavier
#     scene authoring -- reserve ~5 min/generation)
#   8 generations -> ~1,050 epochs/generation, ~35 min/generation at cluster
#     speed (30 min train + ~5 min boot/select), ~4.7 h total -> --time
#     5:30:00 leaves a safety margin without wasting the allocation.

set -euo pipefail

# --- one-time setup (skip if already done for this checkout) ---------------
#   cluster sync . gen_mechanics
#   cluster sync hand_sampler/grammar_bench/fixtures hand_models   # if the
#       probe/projected-hand fixtures aren't already on the cluster NFS
#   cluster submit --tier cpu --time 30 --name gm-venv -- \
#       "cd /data/pulkitag/users/mpeticco/code/gen_mechanics && bash experiments/csail_reorient/install_venv.sh"

REMOTE_DIR=/data/pulkitag/users/mpeticco/code/gen_mechanics
RUN_NAME=evolution_pilot_$(date +%Y%m%d_%H%M%S)

# --dry-run first (per this branch's own cluster-etiquette rule: "cluster
# submit --dry-run and --check for each job type, a 20-minute smoke, then
# the real run") -- drop --dry-run only after reading the printed sbatch
# script and running a short smoke (e.g. --generations 1 --epochs-per-gen 50).
cluster submit \
    --tier main --partition ada6000-shared \
    --gpus 1 --gpu-mem 40 \
    --time 5:30:00 \
    --name "$RUN_NAME" \
    --dir "$REMOTE_DIR" \
    --env OMNI_KIT_ACCEPT_EULA=YES \
    --env WANDB_MODE=disabled \
    --dry-run \
    -- \
    ".venv_isaacsim/bin/python3 -m isaacsimenvs.inhand_reorient.evolution.driver \
        --variant G_V1 --seed 0 \
        --generations 8 \
        --designs 64 \
        --probes allegro_right,dclaw,sharpa_left_on_iiwa14,leap_right \
        --num-envs 4096 \
        --epochs-per-gen 1050 \
        --fitness train_tail \
        --gen-timeout-s 2400 \
        --run-dir outputs/evolution_pilot/${RUN_NAME}"

# --- resume after a kill / hitting the --time limit -------------------------
# Rerun the EXACT SAME `cluster submit` command above (same --run-dir): the
# driver's own state.json (archive + RNG state + last checkpoint + last
# success tolerance) picks up from the last completed generation. Never
# --auto-requeue this job -- the driver is not written to be signalled
# mid-generation and resume the SAME generation; it always redoes an
# interrupted generation from scratch, which --auto-requeue's 2-minutes-
# before-limit requeue would otherwise waste (a generation that's 90%
# done gets discarded, not checkpointed mid-flight).

# --- fetch results back -------------------------------------------------
#   cluster fetch code/gen_mechanics/outputs/evolution_pilot/${RUN_NAME} outputs/evolution_pilot/
