"""Assert the aliased clean/noisy observation path is bit-for-bit unchanged.

``build_observations`` used to recompute every object-derived field a second
time for the "noisy" copy even when object-state DR was off, where the two
inputs are the same tensor. The builder now aliases instead. This checks the
claim the optimization rests on: with DR off, the fields the policy and the
critic receive are exactly what the recomputing formulas produce -- equal, not
merely close.

Small and fast on purpose; the physics does not matter here, only the math.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \
      experiments/decentralized_control/verify_obs_equivalence.py --num_envs 1024
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--num_assets_per_type", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    app = AppLauncher(args).app

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.pose_reaching_6d.env_cfg import PoseReachEnvCfg
    from isaacsimenvs.pose_reaching_6d.obs_utils import observations as O
    from isaacsimenvs.pose_reaching_6d.obs_utils.layout import field_offsets

    cfg = PoseReachEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type
    cfg.assets.robot_spec = "sharpa_iiwa14"
    cfg.action.arm_moving_average = 1.0
    cfg.action.hand_moving_average = 1.0
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_action_delay = False
    cfg.domain_randomization.use_object_state_delay_noise = False
    cfg.domain_randomization.joint_velocity_obs_noise_std = 0.0
    cfg.domain_randomization.force_scale = 0.0
    cfg.domain_randomization.torque_scale = 0.0

    env = gym.make("GenMech-PoseReach-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    env.reset()

    offsets = field_offsets(cfg.obs.obs_list, inner.scene_record.robot_spec)
    actions = torch.zeros(args.num_envs, inner.cfg.action_space, device=inner.device)

    for step in range(args.steps):
        obs, *_ = env.step(actions.uniform_(-1.0, 1.0))
        policy = obs["policy"]

        with torch.inference_mode():
            # The pre-optimization formulas, written out.
            env_origins = inner.scene.env_origins
            palm_state = inner.robot.data.body_state_w[:, inner._palm_body_id, :]
            palm_rot = palm_state[:, 3:7]
            palm_center = O._apply_local_offset(
                palm_state[:, 0:3], palm_rot, inner._palm_center_offset,
                (inner.num_envs,),
            )
            palm_pos = palm_center - env_origins
            obj_pos = inner.object.data.root_pos_w - env_origins
            obj_rot = inner.object.data.root_quat_w
            goal_pos = inner.goal_viz.data.root_pos_w - env_origins
            goal_rot = inner.goal_viz.data.root_quat_w

            kp = inner._keypoint_offsets * inner._object_scale_multiplier.unsqueeze(1)
            obj_kp = O._keypoints_world(obj_pos, obj_rot, kp)
            goal_kp = O._keypoints_world(goal_pos, goal_rot, kp)
            # What the old code did for the noisy copy: recompute from the
            # same inputs.
            noisy_obj_kp = O._keypoints_world(obj_pos, obj_rot, kp)

            _, joint_origins, valid = O._joint_link_geometry_obs(
                inner, palm_center, palm_rot, env_origins)
            expected = {
                "keypoints_rel_palm": noisy_obj_kp - palm_pos.unsqueeze(1),
                "keypoints_rel_goal": noisy_obj_kp - goal_kp,
                "object_keypoints_rel_joint": O._object_keypoints_rel_joint(
                    noisy_obj_kp, joint_origins, palm_rot, inner._hand_scale,
                    valid),
            }

        clip = cfg.obs.clamp_abs_observations
        for name, value in expected.items():
            start, end = offsets[name]
            got = policy[:, start:end]
            want = value.reshape(value.shape[0], -1).clamp(-clip, clip)
            if not torch.equal(got, want):
                diff = (got - want).abs().max().item()
                raise AssertionError(
                    f"step {step}: {name} differs from the recomputed "
                    f"reference by up to {diff:g}"
                )
        print(f"step {step}: {', '.join(expected)} bit-identical", flush=True)

    print(f"\nPASS: aliased noisy fields match recomputation exactly over "
          f"{args.steps} steps at {args.num_envs} envs", flush=True)
    env.close()
    app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
