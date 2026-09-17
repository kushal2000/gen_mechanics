# Shared training configuration for the co-evolution experiment. Sourced by
# baseline.sub and coevo_gen.sub so the two arms differ ONLY in whether the
# population is re-selected between links -- same model, same envs, same
# minibatch, same seed, same curriculum hand-off.
#
# Everything is overridable from the environment (sbatch --export=ALL,...), and
# the defaults are what coevolution_v1 ran.

REPO=/share/portal/kk837/gen_mechanics
export D_MODEL="${D_MODEL:-64}" TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-4}"
export NUM_ENVS_PER_GPU="${NUM_ENVS_PER_GPU:-12288}"     # x2 GPUs = 24576 envs = 24 per design at 1024
export GLOBAL_MINIBATCH="${GLOBAL_MINIBATCH:-114688}"
export SEED="${SEED:-100}"
export PHASE=train
export WANDB_ACTIVATE="${WANDB_ACTIVATE:-1}" WANDB_MODE=online CAPTURE_VIEWER="${CAPTURE_VIEWER:-1}"
export WANDB_PROJECT=gen_mechanics WANDB_ENTITY=kk837
export DESIGN_REWARDS=1                                   # per-design returns; selection and the tolerance hand-off read them
export SCALING_RUN_ROOT="${SCALING_RUN_ROOT:-$REPO/debug_outputs/train_logs/coevolution}"
mkdir -p "$SCALING_RUN_ROOT"

# The newest rl_games checkpoint under a run directory (its rolling autosave
# included), or nothing.
newest_checkpoint() { ls -t "$1"/rank_0/*/nn/*.pth 2>/dev/null | head -1; }
# The run directory this slurm job's run.sh wrote.
own_run_dir() { ls -dt "$SCALING_RUN_ROOT"/*_"${SLURM_JOB_ID}"_* 2>/dev/null | head -1; }
