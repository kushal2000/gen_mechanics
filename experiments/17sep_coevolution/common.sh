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
# Objects: the hand-picked 24 (scene_utils/objects/curated_pools.py DIVERSE_24,
# small and large of each of the twelve size distributions), dealt so every
# design meets every object exactly once per generation and the two ranks hold
# disjoint halves of the deal (robot_spec.object_index, design_cycle).
export OBJECT_POOL="${OBJECT_POOL:-diverse24}"
export OBJECT_ASSIGNMENT="${OBJECT_ASSIGNMENT:-design_cycle}"
export NUM_ASSETS_PER_TYPE="${NUM_ASSETS_PER_TYPE:-100}"    # unused while OBJECT_POOL is set
# Observation: the task YAML's defaults -- token geometry from the end effector
# in metres, the palm as four keypoints, keypoints_rel_ee in the ee frame.
# The newest rl_games checkpoint under a run directory (its rolling autosave
# included), or nothing.
newest_checkpoint() { ls -t "$1"/rank_0/*/nn/*.pth 2>/dev/null | head -1; }
# The run directory this slurm job's run.sh wrote.
own_run_dir() { ls -dt "$SCALING_RUN_ROOT"/*_"${SLURM_JOB_ID}"_* 2>/dev/null | head -1; }
