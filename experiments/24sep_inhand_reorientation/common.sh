# Shared configuration for the IN-HAND REORIENTATION co-evolution experiment:
# a fixed palm-up hand with no arm, one 45 mm cube already in the palm, goal a
# uniform random SO(3) orientation. Sourced by baseline.sub and coevo_gen.sub
# so the two arms differ ONLY in whether the population is re-selected.
#
# Two things differ structurally from 17sep_coevolution, and both ride on the
# ROBOT_SPEC string rather than on separate flags:
#   * TASK selects the registered env (task YAML, rewards, terminations).
#   * ROBOT_SPEC carries a "handonly:" prefix, which is what drops the 7 arm
#     joints. It has to be on the string because the agent YAML interpolates
#     the network's spec from env.assets.robot_spec -- a separate knob would
#     let the policy build itself for 37 joints against a 30-joint robot.

REPO=/share/portal/kk837/gen_mechanics
export D_MODEL="${D_MODEL:-64}" TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-4}"
export NUM_ENVS_PER_GPU="${NUM_ENVS_PER_GPU:-12288}"     # x2 GPUs = 24576 envs = 24 per design at 1024
export GLOBAL_MINIBATCH="${GLOBAL_MINIBATCH:-114688}"
export SEED="${SEED:-100}"
export PHASE=train
export TASK="${TASK:-GenMech-InHandReorient-Direct-v0}"
export WANDB_ACTIVATE="${WANDB_ACTIVATE:-1}" WANDB_MODE=online CAPTURE_VIEWER="${CAPTURE_VIEWER:-1}"
export WANDB_PROJECT=gen_mechanics WANDB_ENTITY=kk837
export DESIGN_REWARDS=1                                   # per-design returns; selection and the tolerance hand-off read them
export SCALING_RUN_ROOT="${SCALING_RUN_ROOT:-$REPO/debug_outputs/train_logs/24sep_inhand_reorientation}"
mkdir -p "$SCALING_RUN_ROOT"
# One object, so there is nothing to deal: every env gets the same 45 mm cube
# (curated_pools.CUBE1) and design_cycle would be a no-op.
export OBJECT_POOL="${OBJECT_POOL:-cube1}"
export OBJECT_ASSIGNMENT="${OBJECT_ASSIGNMENT:-env_modulo}"
export NUM_ASSETS_PER_TYPE="${NUM_ASSETS_PER_TYPE:-100}"    # unused while OBJECT_POOL is set
# ROBOT_SPEC must carry the handonly: prefix -- refuse rather than silently
# train an arm-bearing robot on a task whose palm is bolted in mid-air.
if [[ -n "${ROBOT_SPEC:-}" && "${ROBOT_SPEC}" != handonly:* ]]; then
    echo "[common] ROBOT_SPEC must start with 'handonly:' for this task; got ${ROBOT_SPEC}" >&2
    exit 1
fi
# sbatch against a controller that sometimes answers "Socket timed out" AFTER
# accepting the job (generation 0 of coevolution_v2_gen2k: the error, exit 1,
# and the generation-1 job queued anyway). Retrying blindly would submit twice
# and two jobs would write one gen_<k> directory, so on failure look the name
# up in the queue first and adopt the job if it is there.
submit_once() {   # submit_once <job-name> <sbatch args...>; prints the job id
    local name="$1"; shift; local j i
    for i in 1 2 3 4 5 6; do
        j=$(sbatch --parsable --job-name="$name" "$@" 2>/dev/null) && [[ "$j" =~ ^[0-9]+ ]] && { echo "${j%%;*}"; return 0; }
        sleep 20
        j=$(squeue -h -u "$USER" --name="$name" -o %i 2>/dev/null | head -1)
        [[ -n "$j" ]] && { echo "$j"; return 0; }
        echo "[submit] attempt $i for $name failed; retrying" >&2; sleep 40
    done
    echo "[submit] could not submit $name" >&2; return 1
}

# The newest rl_games checkpoint under a run directory (its rolling autosave
# included), or nothing.
newest_checkpoint() { ls -t "$1"/rank_0/*/nn/*.pth 2>/dev/null | head -1; }
# The run directory this slurm job's run.sh wrote.
own_run_dir() { ls -dt "$SCALING_RUN_ROOT"/*_"${SLURM_JOB_ID}"_* 2>/dev/null | head -1; }
