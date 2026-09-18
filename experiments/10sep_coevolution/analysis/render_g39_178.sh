#!/bin/bash
# Film gen-39 design #178 under the gen-39 policy on the curated objects (V1 observation pins).
cd /share/portal/kk837/gen_mechanics
export GENMECH_KEEP_VISUALS=1
.venv_isaacsim/bin/python -u experiments/10sep_coevolution/analysis/render_policy.py --headless --enable_cameras \
  --run_dir debug_outputs/train_logs/coevo/0_scale_train_coevo_g039_d64_l4_e12288_b114688_s100_300557_0_fDPVEh --checkpoint /share/portal/kk837/gen_mechanics/debug_outputs/train_logs/coevo/0_scale_train_coevo_g039_d64_l4_e12288_b114688_s100_300557_0_fDPVEh/rank_0/0_scale_train_coevo_g039_d64_l4_e12288_b114688_s100_300557_0/nn/last_0_scale_train_coevo_g039_d64_l4_e12288_b114688_s100_300557_0_ep_2000_rew__4009.389_.pth \
  --population $PWD/debug_outputs/10sep_coevo_analysis/videos/design_g39_178.json "$@" \
  env.scene.num_envs=24 env.assets.robot_spec=$PWD/debug_outputs/10sep_coevo_analysis/videos/design_g39_178.json \
  env.assets.object_pool=diverse24 env.assets.object_assignment=design_cycle env.obs.geometry_origin=palm_center \
  "env.obs.state_list=[joint_pos,joint_vel,prev_joint_pos,prev_joint_vel,prev_action_targets,joint_link_bbox,joint_lower,joint_upper,joint_enabled,object_keypoints_rel_joint,hand_scale,palm_pos,palm_rot,palm_vel,object_rot,object_vel,keypoints_rel_palm,keypoints_rel_goal,object_scales,closest_keypoint_max_dist,closest_fingertip_dist,lifted_object,progress,successes,reward]" 'env.obs.obs_list=${env.obs.state_list}' \
  env.action.arm_moving_average=1.0 env.action.hand_moving_average=1.0 \
  env.domain_randomization.use_obs_delay=false env.domain_randomization.use_action_delay=false \
  env.domain_randomization.use_object_state_delay_noise=false env.domain_randomization.joint_velocity_obs_noise_std=0.0 \
  env.domain_randomization.force_scale=0.0 env.domain_randomization.torque_scale=0.0
