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
# 12288, NOT 24576. Measured: 24576 envs at minibatch 114688 dies with
# torch.OutOfMemoryError on the transformer update -- empty_strided_cuda
# ((114688, 31, 128), fp16) = 868 MiB against 47.28 of 47.38 GiB already in use,
# with 78 MiB unallocated, so it is real pressure and not fragmentation.
#
# The binding cost is the UPDATE, not the simulation: 31 tokens x d_model x the
# minibatch. That is why dropping the arm's 7 joints and 8 links per env did not
# buy the env count it looked like it should -- the saving is in the rollout
# buffer, not in the activations. 12288 with this minibatch is the configuration
# 15 completed runs used on 24 Sep.
#
# AUG_ROLLOUT = envs * 16 * 7/6 = 229376, and 114688 divides it exactly twice, so
# no minibatch is left wider than the others.
#
# SAPG's block size follows automatically: run.sh:39 sets
# EXPL_BLOCK_SIZE = NUM_ENVS_PER_GPU / 6, so 12288 gives 2048 and the six
# exploration blocks are preserved -- the same six the 24 Sep runs used. Nothing
# to set by hand, but the constraint num_envs % expl_coef_block_size == 0 is real,
# so an env count that is not a multiple of 6 would break it silently.
export NUM_ENVS_PER_GPU="${NUM_ENVS_PER_GPU:-12288}"
export GLOBAL_MINIBATCH="${GLOBAL_MINIBATCH:-114688}"
# Declared, not inherited. run_rank.sh:137 also defaults to 2, so this changes
# nothing today -- but the value would then be right only because two defaults in
# different files agree, and a chained link is exactly where that stops being true.
# The 24sep runs used 1; this pair uses 2, so they are not comparable on sample
# reuse and only against each other.
export MINI_EPOCHS="${MINI_EPOCHS:-2}"
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
# Per-design returns need a POPULATION to attribute them to. A registered fixed hand
# has one design and no population, and design_rewards.py:49 raises rather than
# degrading -- which is how the first Allegro submission died after authoring 12288
# envs. Derived from the spec form rather than left to each .sub to remember.
if [[ "${ROBOT_SPEC:-}" == handonly:* || "${ROBOT_SPEC:-}" == *.json ]]; then
    export DESIGN_REWARDS="${DESIGN_REWARDS:-1}"   # selection and the tolerance hand-off read these
else
    export DESIGN_REWARDS=0
fi
echo "[common] robot ${ROBOT_SPEC:-<default>}  design_rewards ${DESIGN_REWARDS}"
export SCALING_RUN_ROOT="${SCALING_RUN_ROOT:-$REPO/debug_outputs/train_logs/28sep_inhand_reorientation}"
mkdir -p "$SCALING_RUN_ROOT"
# One object, so there is nothing to deal: every env gets the same 45 mm cube
# (curated_pools.CUBE1) and design_cycle would be a no-op.
# --- palm geometry, declared rather than inherited -----------------------------
# The tilt is read from the ENVIRONMENT at import (robot_param_constants), so it
# never reaches the saved hydra config. Declaring it here puts it in the job log
# and, via the chain export, keeps it identical across links. Without that a
# chained run silently changes geometry when the source default moves: job 572107
# is link #5 of a chain started under a different tilt, and its own record cannot
# say which one it ran.
export HAND_ONLY_PALM_PITCH_DEG="${HAND_ONLY_PALM_PITCH_DEG:-5.0}"
export HAND_ONLY_PALM_ROLL_DEG="${HAND_ONLY_PALM_ROLL_DEG:-0.0}"
echo "[common] palm pitch ${HAND_ONLY_PALM_PITCH_DEG} deg, roll ${HAND_ONLY_PALM_ROLL_DEG} deg"

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
read -r _DEC _DT_S < <(python3 -c "
import re, sys
t = open('$REPO/coevolution/cfg/task/InHandReorient.yaml').read()
d = re.search(r'^decimation:\s*([0-9]+)', t, re.M)
s = re.search(r'^\s*dt:\s*([0-9.]+)', t, re.M)
sys.exit('decimation or dt not found in the task YAML') if not (d and s) else None
print(d.group(1), s.group(1))
") || { echo "[common] could not read decimation/dt from the task YAML" >&2; exit 1; }
_TOL_M=$(python3 -c "
import math
print('%.8f' % ($_EDGE_M * math.sqrt(3) * math.sin(math.radians($SUCCESS_TOLERANCE_DEG) / 2)))
")
echo "[common] success tolerance ${SUCCESS_TOLERANCE_DEG} deg -> ${_TOL_M} m (cube edge ${_EDGE_M} m)"
# --- the per-goal step budget ---------------------------------------------------
# WHICH FIELD ACTUALLY TRUNCATES: max_episode_length, which DirectRLEnv derives as
# ceil(episode_length_s / (sim.dt * decimation)) -- reward_utils/termination.py:77
# compares against that. termination.episode_length looks like the knob and is read
# by nothing but population/run_config.py's record, so setting it alone changes the
# recorded config and not the task. Both are set here, from one step count.
#
# The budget is PER GOAL, not per episode: termination.py:51 zeroes
# episode_length_buf on every goal hit, so a run with 600 lets the policy spend 10 s
# reaching each goal in turn and never truncates while it keeps scoring.
export EPISODE_STEPS="${EPISODE_STEPS:-600}"
_STEP_DT=$(python3 -c "print(repr($_DT_S * $_DEC))")
# Half a step short of the exact value, so the ceil lands ON the step count instead
# of one past it when the division comes out a hair above an integer.
_EP_S=$(python3 -c "print('%.8f' % (($EPISODE_STEPS - 0.5) * $_STEP_DT))")
_EP_CHECK=$(python3 -c "import math; print(math.ceil($_EP_S / $_STEP_DT))")
if [[ "$_EP_CHECK" != "$EPISODE_STEPS" ]]; then
    echo "[common] episode_length_s $_EP_S gives $_EP_CHECK steps, wanted $EPISODE_STEPS" >&2
    exit 1
fi
echo "[common] episode budget ${EPISODE_STEPS} steps per goal -> episode_length_s ${_EP_S} (verified ${_EP_CHECK} steps)"

# Composed FRESH, never appended to an inherited copy: a chained link gets the
# parent's whole environment through --export=ALL, so appending would pass every
# override twice on link #2.  Both tolerance fields, because the curriculum clamps
# into [target, success] and setting only one would leave it running between them.
# USER_HYDRA: per-.sub overrides, appended here. Safe on the chain for the same reason as
# above -- EXTRA_HYDRA is rebuilt from it on every link, never from its own inherited copy.
export EXTRA_HYDRA="env.termination.success_tolerance=$_TOL_M env.termination.target_success_tolerance=$_TOL_M env.episode_length_s=$_EP_S env.termination.episode_length=$EPISODE_STEPS ${USER_HYDRA:-}"

export OBJECT_POOL="${OBJECT_POOL:-cube1}"
export OBJECT_ASSIGNMENT="${OBJECT_ASSIGNMENT:-env_modulo}"
export NUM_ASSETS_PER_TYPE="${NUM_ASSETS_PER_TYPE:-100}"    # unused while OBJECT_POOL is set
# The robot must be hand-only -- refuse rather than silently train an arm-bearing
# robot on a task whose palm is bolted in mid-air. Two forms are hand-only:
#
#   handonly:<population>   a GENERATED population. The prefix is what drops the 7 arm
#                           joints, and it has to ride on the spec string because the
#                           agent YAML interpolates the network's shape from it.
#   *_handonly              a REGISTERED fixed hand that declares arm_joint_names=()
#                           in its own spec, so it needs no prefix and would in fact
#                           raise KeyError with one (is_population_ref only knows
#                           gen_s<seed>_n<size> and .json).
#
# This is a NAME check, not a spec check, because common.sh runs before Kit boots and
# resolving a spec imports scene_utils -> assembly -> pxr. The naming convention is
# the contract; assembly._verify_articulation_view is what actually catches a robot
# with the wrong joint count.
# multi:<spec>+<spec>+...  several registered hand-only specs in one scene (robots/multi_hand.py).
# *_handonly_<tag>           a reduced variant of a hand-only spec (make_missing_fingers.py: no_<finger>, only_<a>_<b>).
if [[ -n "${ROBOT_SPEC:-}" && "${ROBOT_SPEC}" != handonly:* && "${ROBOT_SPEC}" != *_handonly \
      && "${ROBOT_SPEC}" != *_handonly_* && "${ROBOT_SPEC}" != multi:* ]]; then
    echo "[common] ROBOT_SPEC must be 'handonly:<population>' or a registered" \
         "'*_handonly' spec for this task; got ${ROBOT_SPEC}" >&2
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
    # The arch belongs in the tag. run.sh:16-21 leaves D_MODEL/TRANSFORMER_LAYERS unset for
    # the MLP because they do not describe it, but this tag is built from them regardless --
    # so an ARCH=mlp run would be filed, named and charted as "d64_l4", which is a lie that
    # survives into wandb and the run directory.
    if [[ "${ARCH:-transformer}" == mlp ]]; then
        export MODEL_TAG="${STUDY_ID}_c$(printf '%02d' "$CONT")_mlp"
    else
        export MODEL_TAG="${STUDY_ID}_c$(printf '%02d' "$CONT")_d${D_MODEL}_l${TRANSFORMER_LAYERS}"
    fi
    export CHECKPOINT="" RESUME_TOL=""
    if [[ -n "${SOURCE_RUN:-}" ]]; then
        CHECKPOINT=$(newest_checkpoint "$SOURCE_RUN")
        [[ -f "$CHECKPOINT" ]] || { echo "no checkpoint under $SOURCE_RUN"; exit 1; }
        # The tolerance does not survive an rl_games checkpoint, so it is read back
        # from the source run rather than restarting the curriculum. Inert here
        # (success == target), but wrong silently if it were ever dropped.
        #
        # Only when design rewards exist: final_tolerance reads
        # design_rewards_rank0.json, which a fixed hand never writes because it has no
        # population. For those, EXTRA_HYDRA pins both tolerance fields from
        # SUCCESS_TOLERANCE_DEG on every link anyway, so there is nothing to hand over.
        if [[ "${DESIGN_REWARDS:-0}" == 1 ]]; then
            RESUME_TOL=$(cd "$REPO" && .venv_isaacsim/bin/python -m coevolution.loop.final_tolerance "$SOURCE_RUN") \
                || { echo "no final success tolerance under $SOURCE_RUN; refusing to continue blind"; exit 1; }
        else
            echo "[$STUDY_ID] fixed hand: tolerance comes from SUCCESS_TOLERANCE_DEG, not a hand-off"
        fi
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
        --export=ALL,ARCH="${ARCH:-transformer}",SOURCE_RUN="$run_dir",ROBOT_SPEC="$ROBOT_SPEC",STUDY_ID="$STUDY_ID",EPOCHS="$EPOCHS",MAX_CONT="$MAX_CONT",CONT="$next",NUM_ENVS_PER_GPU="$NUM_ENVS_PER_GPU",GLOBAL_MINIBATCH="$GLOBAL_MINIBATCH",MINI_EPOCHS="${MINI_EPOCHS:-2}",GPUS="$GPUS",SEED="$SEED",WANDB_ACTIVATE="$WANDB_ACTIVATE",SUCCESS_TOLERANCE_DEG="$SUCCESS_TOLERANCE_DEG",EPISODE_STEPS="$EPISODE_STEPS",HAND_ONLY_PALM_PITCH_DEG="$HAND_ONLY_PALM_PITCH_DEG",HAND_ONLY_PALM_ROLL_DEG="$HAND_ONLY_PALM_ROLL_DEG" \
        "$self")
    echo "[$STUDY_ID] submitted link #$next as $j (from $run_dir)"; CHAINED=1
}

run_and_chain() {   # run_and_chain <path to this .sub>
    local self="$1"
    prepare_link
    trap "echo '[$STUDY_ID] USR1: 15 min to the limit, chaining now'; chain_next '$self'" USR1
    cd "$REPO"
    set +e
    bash "${RUN_SH:-experiments/old_experiments/decentralized_control/scaling_laws/run.sh}" &
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
    # A NONZERO exit that has not already chained is a crash, not the wall clock: the
    # time-limit path chains from the USR1 trap above and sets CHAINED first. Resubmitting
    # after a crash that left a checkpoint is how an OOM at epoch 33 turned into links
    # c02..c08 of inhand_allegro_tf_big (2026-09-29), each resuming the same config and
    # dying the same way ~9 minutes in, with two copies of c06 and c07 running at once.
    if (( status != 0 )) && (( ! CHAINED )); then
        echo "[$STUDY_ID] payload crashed (exit $status); NOT resubmitting -- a deterministic" \
             "crash would only repeat. Fix the cause and resubmit this .sub by hand."
        exit "$status"
    fi
    chain_next "$self"
}
