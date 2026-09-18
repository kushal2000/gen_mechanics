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
# What coevolution_v1 ran, pinned, because the code's defaults have moved on:
#   objects     a random 1200-entry pool (12 size distributions x 100), env i
#               holding entry i % 1200 -- each design met its own fixed dozen
#   observation token geometry measured from the design's palm centre, divided
#               by hand_scale; no palm keypoints (obs.geometry_origin default
#               is palm_center, but the task YAML now says ee, so say it here)
# experiments/17sep_coevolution changes both.
export OBJECT_POOL="" OBJECT_ASSIGNMENT=env_modulo NUM_ASSETS_PER_TYPE=100
V1_STATE_LIST='[joint_pos,joint_vel,prev_joint_pos,prev_joint_vel,prev_action_targets,joint_link_bbox,joint_lower,joint_upper,joint_enabled,object_keypoints_rel_joint,hand_scale,palm_pos,palm_rot,palm_vel,object_rot,object_vel,keypoints_rel_palm,keypoints_rel_goal,object_scales,closest_keypoint_max_dist,closest_fingertip_dist,lifted_object,progress,successes,reward]'
export EXTRA_HYDRA="${EXTRA_HYDRA:-} env.obs.geometry_origin=palm_center env.obs.state_list=$V1_STATE_LIST"
export SCALING_RUN_ROOT="${SCALING_RUN_ROOT:-$REPO/debug_outputs/train_logs/10sep_coevolution}"
mkdir -p "$SCALING_RUN_ROOT"

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
