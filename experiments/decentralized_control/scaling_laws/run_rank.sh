#!/bin/bash
# Assemble one rank's argv and exec it. torchrun runs this once per GPU.
#
# WHY THIS IS PER RANK AND NOT PART OF run.sh: exactly one line needs the rank.
# Hydra resolves its run dir from argv at decorator time, and $LOCAL_RANK only
# exists inside the process torchrun spawns, so `hydra.run.dir=.../rank_$LOCAL_RANK`
# cannot be written before the spawn. Everything else here is per JOB.
# (`rank_${oc.env:LOCAL_RANK,0}` would let run.sh own the whole command; untested
# against hydra_task_config_with_yaml, so it stays as it is.)
#
# RANK-0 GATING IS NOT DONE HERE. train.py owns it -- it forces wandb_activate,
# capture_video and capture_viewer off when RANK != 0, picks cuda:$LOCAL_RANK,
# sets multi_gpu and offsets the seed by the rank. This file used to gate them a
# second time, which was worse than redundant: it tested LOCAL_RANK where
# train.py tests the GLOBAL rank, so on more than one node it would have enabled
# the viewer on every node's rank 0 and relied on train.py to take it back. Two
# predicates for one decision is how a change in one of them silently does
# nothing.
set -euo pipefail
LOCAL_BATCH=$((GLOBAL_MINIBATCH / 2))
# ROBOT_SPEC selects the hand: a registered name, gen_s<seed>_n<count> for a
# generated population, or a path to a population .json. It has to be ONE name
# -- the agent YAML interpolates the network's copy from it, so a second knob
# would let the two disagree. run.sh resolves the default and exports it; a
# second default here is how the two drift apart.
: "${ROBOT_SPEC:?run_rank.sh is launched by run.sh, which exports ROBOT_SPEC}"
VIEWER_ARGS=()
if [[ "${CAPTURE_VIEWER:-0}" == 1 ]]; then
    # LEN and INTERVAL are overridable so a short run can actually finish a
    # capture. The smoke ran the viewer at the real run's 600-frame length and
    # wrote nothing -- 3 epochs is 48 steps -- so the half of the path that
    # builds the HTML went untested by the very job that exists to test it.
    VIEWER_ARGS=(--capture_viewer
        --capture_viewer_len "${CAPTURE_VIEWER_LEN:-600}"
        --capture_viewer_interval "${CAPTURE_VIEWER_INTERVAL:-6000}")
fi
VIDEO_ARGS=()
if [[ "${CAPTURE_VIDEO:-0}" == 1 ]]; then
    # Isaac's own RTX render, not the pose viewer: the pose viewer draws what
    # the design SAYS, and this draws what the simulator actually built. The
    # two disagreeing is the whole class of bug this path exists to catch.
    VIDEO_ARGS=(--capture_video
        --video_interval "${VIDEO_INTERVAL:-10000}"
        --video_capture_frames "${VIDEO_FRAMES:-10}"
        --video_fps "${VIDEO_FPS:-5}")
fi
# Coevolution: per-design returns for selection, and the previous generation's
# policy to start from. `weights`, not `resume`: resume restores the epoch
# counter too, and a generation that resumes at epoch 2000 with max_epochs 2000
# trains for zero epochs.
COEVO_ARGS=()
if [[ "${DESIGN_REWARDS:-0}" == 1 ]]; then
    COEVO_ARGS+=(--design_rewards)
fi
if [[ -n "${CHECKPOINT:-}" ]]; then
    [[ -f "$CHECKPOINT" ]] || { echo "CHECKPOINT not found: $CHECKPOINT"; exit 1; }
    COEVO_ARGS+=(--checkpoint "$CHECKPOINT" --checkpoint_load_mode weights)
fi
# The success-tolerance curriculum is env state, not network state: weights
# mode does not carry it, and a continuation that omits this restarts at 0.075
# on a different reward scale (reward_utils/curriculum.py). The old generation
# loop passed it; this one did not, and got away with it only because nothing
# had tightened yet.
if [[ -n "${RESUME_TOL:-}" ]]; then
    COEVO_ARGS+=("env.termination.resume_success_tolerance=$RESUME_TOL")
fi
WANDB_ARGS=()
if [[ "$WANDB_ACTIVATE" == 1 ]]; then
    WANDB_ARGS=(--wandb_activate --wandb_project "$WANDB_PROJECT"
        --wandb_entity "$WANDB_ENTITY" --wandb_group "$WANDB_GROUP" --wandb_name "$RUN_NAME")
fi
# Split on whitespace deliberately -- several overrides must become several argv
# entries -- but through an array, so an unset EXTRA_HYDRA is empty and not "".
EXTRA_HYDRA_ARGS=()
if [[ -n "${EXTRA_HYDRA:-}" ]]; then
    read -r -a EXTRA_HYDRA_ARGS <<< "$EXTRA_HYDRA"
fi
# Both families share every env/PPO setting below; only the backbone and
# its entry point differ, so the comparison is the network and nothing else.
if [[ "${ARCH:-transformer}" == mlp ]]; then
    AGENT_ENTRY=rl_games_sapg_nolstm_cfg_entry_point
    # The YAML's own [1024,1024,512,512] trunk. separate=false makes the
    # value head share it, which is the symmetric actor-critic.
    NET_ARGS=(agent.params.network.separate=false)
else
    AGENT_ENTRY=rl_games_joint_transformer_cfg_entry_point
    NET_ARGS=(
        agent.params.network.separate=false
        "agent.params.network.d_model=$D_MODEL"
        "agent.params.network.n_layers=$TRANSFORMER_LAYERS"
        agent.params.network.n_heads=1 agent.params.network.ff_mult=2
        agent.params.network.compile_net=true
        '++agent.params.network.mu_head_units=[64]'
        'agent.params.network.arm_head_units=[256,128]'
        ++agent.params.network.parallel_block=false
        ++agent.params.network.affine_norm=true
        ++agent.params.network.final_norm=true
        'agent.params.network.value_head_units=[512,256]'
    )
fi
ARGS=(
    python -u coevolution/train.py
    --task GenMech-PoseReach-Direct-v0
    --agent "$AGENT_ENTRY" --headless
    "${WANDB_ARGS[@]}"
    "${VIEWER_ARGS[@]}"
    "${VIDEO_ARGS[@]}"
    ${COEVO_ARGS[@]+"${COEVO_ARGS[@]}"}
    "env.assets.robot_spec=$ROBOT_SPEC"
    "env.assets.num_assets_per_type=${NUM_ASSETS_PER_TYPE:-100}"
    "env.assets.object_assignment=${OBJECT_ASSIGNMENT:-env_modulo}"
    "env.assets.object_pool=${OBJECT_POOL:-}"
    "env.scene.num_envs=$NUM_ENVS_PER_GPU"
    'env.obs.obs_list=${env.obs.state_list}'
    env.action.arm_moving_average=1.0 env.action.hand_moving_average=1.0
    env.domain_randomization.use_obs_delay=false
    env.domain_randomization.use_action_delay=false
    env.domain_randomization.use_object_state_delay_noise=false
    env.domain_randomization.joint_velocity_obs_noise_std=0.0
    env.domain_randomization.force_scale=0.0 env.domain_randomization.torque_scale=0.0
    ${EXTRA_HYDRA_ARGS[@]+"${EXTRA_HYDRA_ARGS[@]}"}
    "agent.params.seed=$SEED"
    "agent.params.config.name=$RUN_NAME"
    "agent.params.config.full_experiment_name=$RUN_NAME"
    agent.params.config.multi_gpu=true agent.params.config.mixed_precision=true
    "agent.params.config.minibatch_size=$LOCAL_BATCH"
    agent.params.config.central_value_config=null
    "agent.params.config.expl_coef_block_size=$EXPL_BLOCK_SIZE"
    "agent.params.config.learning_rate=$LEARNING_RATE"
    agent.params.config.lr_schedule=adaptive
    agent.params.config.horizon_length=16 agent.params.config.mini_epochs=2
    "agent.params.config.max_epochs=$MAX_EPOCHS"
    agent.params.config.save_frequency=100 agent.params.config.save_best_after=0
    "${NET_ARGS[@]}"
    "hydra.run.dir=$SCALING_RUN_DIR/rank_$LOCAL_RANK"
)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${ARGS[@]}"
    printf '\n'
else
    export WANDB_DIR="$SCALING_RUN_DIR"
    exec "${ARGS[@]}"
fi
