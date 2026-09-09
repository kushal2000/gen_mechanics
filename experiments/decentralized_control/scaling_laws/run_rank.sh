#!/bin/bash
set -euo pipefail
LOCAL_BATCH=$((GLOBAL_MINIBATCH / 2))
WANDB_ARGS=()
VIEWER_ARGS=()
if [[ "$LOCAL_RANK" == 0 && "${CAPTURE_VIEWER:-0}" == 1 ]]; then
    VIEWER_ARGS=(--capture_viewer)
fi
if [[ "$LOCAL_RANK" == 0 && "$WANDB_ACTIVATE" == 1 ]]; then
    WANDB_ARGS=(--wandb_activate --wandb_project "$WANDB_PROJECT"
        --wandb_entity "$WANDB_ENTITY" --wandb_group "$WANDB_GROUP" --wandb_name "$RUN_NAME")
fi
ARGS=(
    python -u coevolution/train.py
    --task GenMech-PoseReach-Direct-v0
    --agent rl_games_joint_transformer_cfg_entry_point --headless
    "${WANDB_ARGS[@]}"
    "${VIEWER_ARGS[@]}"
    env.assets.robot_spec=sharpa_iiwa14
    env.assets.robot_population_path= env.assets.robot_population_seed=-1
    env.assets.robot_population_count=0 env.assets.num_assets_per_type=100
    "env.scene.num_envs=$NUM_ENVS_PER_GPU"
    'env.obs.obs_list=${env.obs.state_list}'
    env.action.arm_moving_average=1.0 env.action.hand_moving_average=1.0
    env.domain_randomization.use_obs_delay=false
    env.domain_randomization.use_action_delay=false
    env.domain_randomization.use_object_state_delay_noise=false
    env.domain_randomization.joint_velocity_obs_noise_std=0.0
    env.domain_randomization.force_scale=0.0 env.domain_randomization.torque_scale=0.0
    "agent.params.seed=$SEED"
    "agent.params.config.name=$RUN_NAME"
    "agent.params.config.full_experiment_name=$RUN_NAME"
    agent.params.config.multi_gpu=true agent.params.config.mixed_precision=true
    "agent.params.config.minibatch_size=$LOCAL_BATCH"
    agent.params.config.central_value_config=null
    agent.params.network.separate=false
    "agent.params.config.expl_coef_block_size=$EXPL_BLOCK_SIZE"
    "agent.params.config.learning_rate=$LEARNING_RATE"
    agent.params.config.lr_schedule=adaptive
    agent.params.config.horizon_length=16 agent.params.config.mini_epochs=2
    "agent.params.config.max_epochs=$MAX_EPOCHS"
    agent.params.config.save_frequency=100 agent.params.config.save_best_after=0
    "agent.params.network.d_model=$D_MODEL"
    "agent.params.network.n_layers=$TRANSFORMER_LAYERS"
    agent.params.network.n_heads=1 agent.params.network.ff_mult=2
    agent.params.network.compile_net=true
    '++agent.params.network.mu_head_units=[64]'
    'agent.params.network.arm_head_units=[256,128]'
    ++agent.params.network.parallel_block=false ++agent.params.network.affine_norm=true
    ++agent.params.network.final_norm=true
    'agent.params.network.value_head_units=[512,256]'
    "hydra.run.dir=$SCALING_RUN_DIR/rank_$LOCAL_RANK"
)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${ARGS[@]}"
    printf '\n'
else
    export WANDB_DIR="$SCALING_RUN_DIR"
    exec "${ARGS[@]}"
fi
