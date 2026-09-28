# Shared configuration for the IN-HAND REORIENTATION co-evolution experiment:
# a fixed palm-up hand with no arm, one 45 mm cube already in the palm, goal a
# uniform random SO(3) orientation. Sourced by baseline.sub and coevo_gen.sub
# so the two arms differ ONLY in whether the population is re-selected.
#
# Two things differ structurally from 28sep_inhand_reorientation, and both ride on the
# ROBOT_SPEC string rather than on separate flags:
#   * TASK selects the registered env (task YAML, rewards, terminations).
#   * ROBOT_SPEC carries a "handonly:" prefix, which is what drops the 7 arm
#     joints. It has to be on the string because the agent YAML interpolates
#     the network's spec from env.assets.robot_spec -- a separate knob would
#     let the policy build itself for 37 joints against a 30-joint robot.

REPO=/share/portal/kk837/gen_mechanics
export D_MODEL="${D_MODEL:-64}" TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-4}"
# ONE gpu, all 24576 envs on it. The hand-only robot drops 7 arm joints and 7
# large arm links per env against pose reaching, which is what makes this fit
# where the arm version OOMed. GPUS is set HERE rather than at submit time
# because the link chaining's --export does not carry it, so a second link
# would otherwise revert to two.
export GPUS="${GPUS:-1}"
export NUM_ENVS_PER_GPU="${NUM_ENVS_PER_GPU:-24576}"
export GLOBAL_MINIBATCH="${GLOBAL_MINIBATCH:-114688}"
export SEED="${SEED:-100}"
export PHASE=train
export TASK="${TASK:-GenMech-InHandReorient-Direct-v0}"
# Viewer ON: pose_viewer now serves a hand-only robot. urdf_for_viewing is
# already rooted at ARM_TIP_LINK, the stub carries that link at the articulation
# root with an identity transform, and joint_names_canonical is exactly the 30
# hand slots the URDF declares -- so the arm graft is skipped rather than
# drawing an arm that is not in the scene. (With an arm it is still needed: the
# trajectory animates 37 joints against a 30-joint URDF.)
export WANDB_ACTIVATE="${WANDB_ACTIVATE:-1}" WANDB_MODE=online CAPTURE_VIEWER="${CAPTURE_VIEWER:-1}"
export CAPTURE_VIDEO="${CAPTURE_VIDEO:-0}"
# How often the INTERACTIVE pose viewer captures, in epochs. run_rank.sh
# defaults to 6000, which suits a 15000-epoch pose-reaching link and means a
# brand-new task shows nothing for hours. 250 puts a viewer on wandb within the
# first few minutes of training, which is when it is most worth looking at.
# CAPTURE_VIEWER_LEN stays at the default 600 frames: shorter and the capture
# ends mid-episode.
export CAPTURE_VIEWER_INTERVAL="${CAPTURE_VIEWER_INTERVAL:-250}"
# Its own project: this task shares no metric scale with pose reaching -- the
# return is a rotation residual, not a reach -- so mixing them in one project
# makes every cross-run chart misleading.
export WANDB_PROJECT="${WANDB_PROJECT:-gen_mechanics_inhandreorient}" WANDB_ENTITY=kk837
export DESIGN_REWARDS=1                                   # per-design returns; selection and the tolerance hand-off read them
export SCALING_RUN_ROOT="${SCALING_RUN_ROOT:-$REPO/debug_outputs/train_logs/28sep_inhand_reorientation}"
mkdir -p "$SCALING_RUN_ROOT"
# One object, so there is nothing to deal: every env gets the same 45 mm cube
# (curated_pools.CUBE1) and design_cycle would be a no-op.
# --- the success threshold, set in DEGREES ------------------------------------
# The env compares a keypoint residual in METRES, and the conversion is not
# obvious, so set the angle and let this do the arithmetic. For keypoints at
# radius r = 0.5 * edge * sqrt(3):
#
#     success_tolerance = edge * sqrt(3) * sin(theta / 2)
#
# keypoint_scale cancels -- it multiplies the tolerance (observations.py:269) and
# is already inside the keypoint radius (reset.py:233) -- so it must NOT appear
# here. Changing keypoint_scale looks like a difficulty knob and moves nothing;
# inhand_reorient/tests/test_orientation_metric.py pins that.
#
# The angle is the threshold on the MOST SENSITIVE rotation axis. The four reward
# keypoints are two antipodal pairs and coplanar, so the same angle about the
# least sensitive axis is 1/sin(35.2644 deg) = 1.732x looser. Verified in sim by
# bisecting near_goal over 32 envs: 5.7362 and 9.9437 deg for a 5.7362 setting.
export SUCCESS_TOLERANCE_DEG="${SUCCESS_TOLERANCE_DEG:-5.0}"
_EDGE_M=$(python3 -c "
import re, sys
t = open('$REPO/coevolution/cfg/task/InHandReorient.yaml').read()
m = re.search(r'^\s*object_base_size:\s*([0-9.]+)', t, re.M)
sys.exit('object_base_size not found in the task YAML') if not m else None
sys.stdout.write(m.group(1))
") || { echo "[common] could not read the cube edge from the task YAML" >&2; exit 1; }
_TOL_M=$(python3 -c "
import math
print('%.8f' % ($_EDGE_M * math.sqrt(3) * math.sin(math.radians($SUCCESS_TOLERANCE_DEG) / 2)))
")
echo "[common] success tolerance ${SUCCESS_TOLERANCE_DEG} deg -> ${_TOL_M} m (cube edge ${_EDGE_M} m)"
# Both, because the curriculum is inert for this task: it clamps into
# [target, success], so setting them equal pins the threshold. Setting only one
# would leave a curriculum running between the two.
export EXTRA_HYDRA="${EXTRA_HYDRA:-} env.termination.success_tolerance=$_TOL_M env.termination.target_success_tolerance=$_TOL_M"

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

# --- the run body, shared by every .sub in this folder ------------------------
# Each .sub sets STUDY_ID, SUCCESS_TOLERANCE_DEG and ROBOT_SPEC, sources this,
# then calls run_and_chain "$0". Everything below is plumbing that has no reason
# to differ between the runs: restoring from the previous link's checkpoint,
# handing the curriculum over, running the payload, and resubmitting 15 minutes
# before the wall clock so a 24 h limit does not cap the experiment.
prepare_link() {
    export WANDB_GROUP="$STUDY_ID" MAX_EPOCHS="$EPOCHS"
    export MODEL_TAG="${STUDY_ID}_c$(printf '%02d' "$CONT")_d${D_MODEL}_l${TRANSFORMER_LAYERS}"
    export CHECKPOINT="" RESUME_TOL=""
    if [[ -n "${SOURCE_RUN:-}" ]]; then
        CHECKPOINT=$(newest_checkpoint "$SOURCE_RUN")
        [[ -f "$CHECKPOINT" ]] || { echo "no checkpoint under $SOURCE_RUN"; exit 1; }
        # The tolerance does not survive an rl_games checkpoint, so it is read
        # back from the source run rather than restarting the curriculum. Inert
        # here (success == target), but wrong silently if it were ever dropped.
        RESUME_TOL=$(cd "$REPO" && .venv_isaacsim/bin/python -m coevolution.loop.final_tolerance "$SOURCE_RUN") \
            || { echo "no final success tolerance under $SOURCE_RUN; refusing to continue blind"; exit 1; }
    fi
    echo "[$STUDY_ID] link #$CONT of $MAX_CONT on $ROBOT_SPEC"
    echo "[$STUDY_ID] threshold ${SUCCESS_TOLERANCE_DEG} deg   policy from: ${CHECKPOINT:-scratch}"
}

CHAINED=0
chain_next() {   # chain_next <path to this .sub>
    (( CHAINED )) && return 0
    local self="$1" run_dir; run_dir=$(own_run_dir)
    if [[ -z "$run_dir" ]] || [[ ! -f "$(newest_checkpoint "$run_dir")" ]]; then
        echo "[$STUDY_ID] no checkpoint to chain from; not resubmitting"; return 1
    fi
    local next=$((CONT + 1))
    if (( next > MAX_CONT )); then echo "[$STUDY_ID] reached MAX_CONT=$MAX_CONT"; CHAINED=1; return 0; fi
    # Named explicitly rather than trusting --export=ALL, which has been observed
    # not to carry them: GPUS reverting to 2 on link #2 is what prompted this.
    local j; j=$(submit_once "${STUDY_ID}_c$(printf '%02d' "$next")" \
        --export=ALL,SOURCE_RUN="$run_dir",ROBOT_SPEC="$ROBOT_SPEC",STUDY_ID="$STUDY_ID",EPOCHS="$EPOCHS",MAX_CONT="$MAX_CONT",CONT="$next",NUM_ENVS_PER_GPU="$NUM_ENVS_PER_GPU",GLOBAL_MINIBATCH="$GLOBAL_MINIBATCH",MINI_EPOCHS="${MINI_EPOCHS:-2}",GPUS="$GPUS",SEED="$SEED",WANDB_ACTIVATE="$WANDB_ACTIVATE",SUCCESS_TOLERANCE_DEG="$SUCCESS_TOLERANCE_DEG" \
        "$self")
    echo "[$STUDY_ID] submitted link #$next as $j (from $run_dir)"; CHAINED=1
}

run_and_chain() {   # run_and_chain <path to this .sub>
    local self="$1"
    prepare_link
    trap "echo '[$STUDY_ID] USR1: 15 min to the limit, chaining now'; chain_next '$self'" USR1
    cd "$REPO"
    set +e
    bash "${RUN_SH:-experiments/decentralized_control/scaling_laws/run.sh}" &
    local payload=$!
    wait $payload; local status=$?
    # A signal interrupts wait, so wait again until the payload is really gone.
    while kill -0 $payload 2>/dev/null; do wait $payload; status=$?; done
    set -e
    local run_dir; run_dir=$(own_run_dir)
    echo "[$STUDY_ID] payload exited $status; run dir $run_dir"
    if (( status != 0 )) && [[ ! -f "$(newest_checkpoint "$run_dir")" ]]; then
        echo "[$STUDY_ID] failed with no checkpoint; stopping"; exit "$status"
    fi
    chain_next "$self"
}
