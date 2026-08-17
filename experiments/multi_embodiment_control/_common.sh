#!/bin/bash
# Shared settings for the multi-embodiment control comparison.
#
# The three runs in this folder are a CONTROLLED COMPARISON:
#
#   01_sharpa.sub            one fixed hand          (SHARPA, 29 joints)
#   02_gen_sharpa_like.sub   one generated hand      (37-joint template)
#   03_population_24k.sub    24,576 distinct hands   (one per env)
#
# The only thing that may differ between them is the embodiment. Every other
# knob lives HERE, in one place, sourced by all three -- because a comparison
# whose settings are copy-pasted into three files is one edit away from being a
# comparison of settings instead of a comparison of embodiments.
#
# Not sourced by anything else in the repo; the production training path
# (experiments/train.sub) keeps its own defaults.

# --- 1. Domain randomization: OFF -------------------------------------------
#
# Every DR field in DomainRandomizationCfg, set to its identity value. Listed
# exhaustively rather than relying on defaults, so that a future change to a
# default cannot silently reintroduce randomization into this comparison.
#
# Why off: DR exists to make one policy robust to sim-to-real gaps on ONE hand.
# Here the axis of variation under study is the hand itself, and DR noise on top
# of morphology variation would confound "the policy cannot handle this
# morphology" with "the policy cannot handle this observation noise".
DR_OFF=(
    # observation and action delays
    env.domain_randomization.use_obs_delay=false
    env.domain_randomization.use_action_delay=false
    env.domain_randomization.use_object_state_delay_noise=false
    # observation noise
    env.domain_randomization.object_state_xyz_noise_std=0.0
    env.domain_randomization.object_state_rotation_noise_degrees=0.0
    env.domain_randomization.joint_velocity_obs_noise_std=0.0
    env.domain_randomization.object_scale_noise_multiplier_range=[1.0,1.0]
    # random wrenches on the object
    env.domain_randomization.force_scale=0.0
    env.domain_randomization.torque_scale=0.0
    env.domain_randomization.force_prob_range=[0.0,0.0]
    env.domain_randomization.torque_prob_range=[0.0,0.0]
    # per-env friction buckets (already identity by default; pinned anyway)
    env.domain_randomization.object_friction_scale_range=[1.0,1.0]
    env.domain_randomization.fingertip_friction_scale_range=[1.0,1.0]
)

# --- 2. Action moving average: 1.0 ------------------------------------------
#
# 1.0 means the commanded target is taken as-is, with no smoothing against the
# previous target. The default 0.1 is heavy low-pass filtering, which couples
# the effective control bandwidth to the policy's action rate -- and across a
# morphology population that filter interacts with each design's own dynamics,
# so identical actions produce different effective commands on different hands.
ACTION_RAW=(
    env.action.arm_moving_average=1.0
    env.action.hand_moving_average=1.0
)

# --- 3. Symmetric actor-critic on the privileged state ----------------------
#
# The env ships ASYMMETRIC: the critic sees obs.state_list, the actor the strict
# subset obs.obs_list. Both lists are already declared in ObsCfg, so making the
# actor symmetric is just pointing one at the other -- no new config field, and
# no literal field list duplicated into these sub files.
#
# OmegaConf interpolation is what makes this work for BOTH envs from one line:
# the multi-embodiment env's state_list appends `morphology`, and the
# interpolation resolves against whichever env is loaded, so the actor picks
# that up automatically. A hand-written literal would need one version per env
# and would go stale the moment ObsCfg changes.
#
# The actor therefore sees the full privileged state: object and palm velocity,
# the lift flag, the running closest-distance trackers, and the morphology
# descriptor where there is one.
SYMMETRIC_OBS=(
    'env.obs.obs_list=${env.obs.state_list}'
)

COMMON_OVERRIDES=("${DR_OFF[@]}" "${ACTION_RAW[@]}" "${SYMMETRIC_OBS[@]}")

# --- shared launcher ---------------------------------------------------------
# Expects: TASK ROBOT_OVERRIDES(array) NUM_ENVS SEED MAX_EPOCHS EXPERIMENT_TAG
# WANDB_GROUP, and optionally EXTRA_OVERRIDES(array).
#
# EXTRA_OVERRIDES is expanded with the ${arr[@]+...} idiom rather than
# "${arr[@]:-}": under `set -u` the latter expands an UNSET array to a single
# empty string, which hydra then rejects as a malformed override.
launch_training() {
    # wandb on by default; a smoke run sets WANDB_ACTIVATE= to skip it, so
    # throwaway config checks do not litter the project with runs.
    : "${WANDB_ACTIVATE=1}"
    # On by default for training; the smoke runs turn it off with the wandb run.
    : "${CAPTURE_VIEWER=1}"
    local REPO_ROOT="/share/portal/kk837/gen_mechanics"
    local HORIZON_LENGTH=16
    # Overridable so a smoke run can use a small env count: SAPG requires
    # num_envs %% expl_coef_block_size == 0, which otherwise pins the
    # smallest runnable scene at 4096 envs.
    local EXPL_COEF_BLOCK_SIZE="${EXPL_COEF_BLOCK_SIZE:-4096}"
    local WANDB_PROJECT="${WANDB_PROJECT:-gen_mechanics}"
    local WANDB_ENTITY="${WANDB_ENTITY:-kk837}"
    # The wandb pose viewer. experiments/train.sub passes these; this launcher
    # was written from scratch and dropped them, so the first three arms logged
    # no viewer. The viewer fetches robot meshes over raw.githubusercontent, so
    # it only renders for a robot whose URDF is COMMITTED -- true for
    # sharpa_iiwa14 and gen_sharpa_like, NOT for a 24k generated population
    # (those URDFs are generated per run and are not in git).
    local VIEWER_RAW_BASE="${VIEWER_RAW_BASE:-https://raw.githubusercontent.com/kushal2000/gen_mechanics/master/}"

    # SAPG requires num_envs % expl_coef_block_size == 0.
    if (( NUM_ENVS % EXPL_COEF_BLOCK_SIZE != 0 )); then
        echo "NUM_ENVS=$NUM_ENVS must be a multiple of $EXPL_COEF_BLOCK_SIZE" >&2
        return 1
    fi
    # rl_games floors num_minibatches to 0 if the batch is smaller than one
    # minibatch, then divides by it; scale with NUM_ENVS so small runs work.
    local BATCH=$(( HORIZON_LENGTH * NUM_ENVS ))
    local MINIBATCH_SIZE="${MINIBATCH_SIZE:-$(( BATCH / 4 ))}"
    if (( BATCH % MINIBATCH_SIZE != 0 )); then
        echo "minibatch_size=$MINIBATCH_SIZE must divide $BATCH" >&2
        return 1
    fi

    local EXPERIMENT_NAME="${EXPERIMENT_TAG}_seed${SEED}_$(date +%Y-%m-%d_%H-%M-%S)"
    [ -n "${SLURM_JOB_ID:-}" ] && scontrol update JobId="$SLURM_JOB_ID" \
        JobName="$EXPERIMENT_NAME" || true

    local RUN_DIR="${REPO_ROOT}/train_dir/${WANDB_PROJECT}/${WANDB_GROUP}/${EXPERIMENT_NAME}"
    mkdir -p "$RUN_DIR"
    exec > "${RUN_DIR}/slurm.log" 2> "${RUN_DIR}/slurm.err"

    cd "$REPO_ROOT"
    source .venv_isaacsim/bin/activate
    export OMNI_KIT_ACCEPT_EULA=YES
    export OMNI_KIT_CACHE_PATH="/tmp/${USER}_ov_cache_${EXPERIMENT_TAG}"
    mkdir -p "$OMNI_KIT_CACHE_PATH"

    echo "=== multi-embodiment control: ${EXPERIMENT_TAG} ==="
    echo "task=$TASK num_envs=$NUM_ENVS seed=$SEED max_epochs=$MAX_EPOCHS"
    echo "batch=$BATCH minibatch=$MINIBATCH_SIZE"
    echo "embodiment overrides: ${ROBOT_OVERRIDES[*]}"
    echo "common overrides:     ${COMMON_OVERRIDES[*]}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

    python -u genmech/train.py \
        --task "$TASK" \
        --agent rl_games_sapg_cfg_entry_point \
        --headless \
        ${CAPTURE_VIEWER:+--capture_viewer} \
        ${CAPTURE_VIEWER:+--capture_viewer_github_raw_base "$VIEWER_RAW_BASE"} \
        ${WANDB_ACTIVATE:+--wandb_activate} \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_group "$WANDB_GROUP" \
        --wandb_entity "$WANDB_ENTITY" \
        --wandb_tags "$EXPERIMENT_TAG" "seed$SEED" "no_dr" "symmetric_obs" \
        --wandb_name "$EXPERIMENT_NAME" \
        "${ROBOT_OVERRIDES[@]}" \
        "${COMMON_OVERRIDES[@]}" \
        ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"} \
        env.scene.num_envs="$NUM_ENVS" \
        agent.params.seed="$SEED" \
        agent.params.config.max_epochs="$MAX_EPOCHS" \
        agent.params.config.minibatch_size="$MINIBATCH_SIZE" \
        agent.params.config.central_value_config.minibatch_size="$MINIBATCH_SIZE" \
        agent.params.config.expl_coef_block_size="$EXPL_COEF_BLOCK_SIZE" \
        hydra.run.dir="$RUN_DIR"
}
