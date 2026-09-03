"""Attribute the rollout-side FPS drop between physics and pure tensor work.

``profile_sharpa_joint_obs.py`` answers "is the joint-token path correct, and
what does one step cost".  This answers the follow-up: *which* of the changes
between the 140-wide single-embodiment reference run and the 778-wide
decentralized run cost the throughput, and how much of the remainder is tensor
bookkeeping that never touches PhysX.

It boots the scene once and then A/Bs, in-process:

* ``arm_moving_average`` 1.0 vs the reference 0.1 -- the action smoothing is
  the one env change that alters the *trajectories*, and therefore contacts.
* the 778-field observation list vs the reference 140-field one.
* the wrench-DR block, which still writes an all-zero external wrench to every
  object each step when the scales are 0.
* observation-builder components, one at a time.

Example::

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \
      experiments/decentralized_control/profile_env_step_breakdown.py \
      --num_envs 24576 --num_assets_per_type 100 --steps 40
"""

from __future__ import annotations

import argparse
import os
import time


# The reference run's lists, minus ``fingertip_pos_rel_palm``: the field is
# still sized in layout.py but build_observations no longer emits it, so the
# stand-in is 125/175 wide rather than 140/190.  The 653 columns that separate
# it from the decentralized list are the ones under test.
REFERENCE_OBS_LIST = (
    "joint_pos", "joint_vel", "prev_action_targets", "palm_pos", "palm_rot",
    "object_rot", "keypoints_rel_palm", "keypoints_rel_goal", "object_scales",
)
REFERENCE_STATE_LIST = (
    "joint_pos", "joint_vel", "prev_action_targets", "palm_pos", "palm_rot",
    "palm_vel", "object_rot", "object_vel", "keypoints_rel_palm",
    "keypoints_rel_goal", "object_scales", "closest_keypoint_max_dist",
    "closest_fingertip_dist", "lifted_object", "progress", "successes",
    "reward",
)


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=24576)
    parser.add_argument("--num_assets_per_type", type=int, default=100)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--obs_iters", type=int, default=100)
    parser.add_argument(
        "--position_iterations", type=int, default=0,
        help="Override PhysX min/max position iterations (task default: 8). "
             "Solver iterations are fixed at scene creation, so comparing 8 "
             "against 4 means two runs of this script.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    app = AppLauncher(args).app

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.pose_reaching_6d.env_cfg import PoseReachEnvCfg
    from isaacsimenvs.pose_reaching_6d.obs_utils import observations as O
    from isaacsimenvs.pose_reaching_6d.obs_utils import actions as A

    cfg = PoseReachEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type
    cfg.assets.robot_spec = "sharpa_iiwa14"
    # The decentralized launcher's env settings.
    cfg.action.arm_moving_average = 1.0
    cfg.action.hand_moving_average = 1.0
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_action_delay = False
    cfg.domain_randomization.use_object_state_delay_noise = False
    cfg.domain_randomization.joint_velocity_obs_noise_std = 0.0
    cfg.domain_randomization.force_scale = 0.0
    cfg.domain_randomization.torque_scale = 0.0
    if args.position_iterations:
        cfg.sim.physx.min_position_iteration_count = args.position_iterations
        cfg.sim.physx.max_position_iteration_count = args.position_iterations

    boot = time.perf_counter()
    env = gym.make("GenMech-PoseReach-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    env.reset()
    torch.cuda.synchronize(inner.device)
    print(f"\nboot {time.perf_counter() - boot:.1f} s  "
          f"envs={args.num_envs} assets_per_type={args.num_assets_per_type} "
          f"position_iterations={cfg.sim.physx.min_position_iteration_count}",
          flush=True)

    device = inner.device
    n_act = inner.cfg.action_space

    def random_actions() -> torch.Tensor:
        return torch.empty(args.num_envs, n_act, device=device).uniform_(-1.0, 1.0)

    def steady_ms(fn, iters: int, warmup: int) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize(device)
        return 1e3 * (time.perf_counter() - start) / iters

    def step_once() -> None:
        env.step(random_actions())

    # ---- 1. does action smoothing (i.e. contact load) explain the gap? -----
    print("\n1. env.step under random actions, by action smoothing", flush=True)
    smoothing_ms = {}
    for label, moving_average in (("1.0 (decentralized run)", 1.0),
                                  ("0.1 (reference run)", 0.1)):
        inner.cfg.action.arm_moving_average = moving_average
        inner.cfg.action.hand_moving_average = moving_average
        ms = steady_ms(step_once, args.steps, args.warmup)
        smoothing_ms[moving_average] = ms
        print(f"  arm/hand_moving_average {label:<24} {ms:8.3f} ms/step",
              flush=True)
    delta = smoothing_ms[1.0] - smoothing_ms[0.1]
    print(f"  attributable to smoothing {delta:+8.3f} ms/step "
          f"({100.0 * delta / max(smoothing_ms[1.0], 1e-9):+5.1f}%)", flush=True)
    inner.cfg.action.arm_moving_average = 1.0
    inner.cfg.action.hand_moving_average = 1.0

    # ---- 2. observation width -------------------------------------------
    print("\n2. build_observations, by field list", flush=True)
    wide_lists = (tuple(inner.cfg.obs.obs_list), tuple(inner.cfg.obs.state_list))
    with torch.no_grad():
        wide_ms = steady_ms(lambda: O.build_observations(inner),
                            args.obs_iters, args.warmup)
        widths = O.build_observations(inner)
        print(f"  778/800-wide decentralized  {wide_ms:8.3f} ms  "
              f"policy={tuple(widths['policy'].shape)} "
              f"state={tuple(widths['critic'].shape)}", flush=True)

        inner.cfg.obs.obs_list = REFERENCE_OBS_LIST
        inner.cfg.obs.state_list = REFERENCE_STATE_LIST
        narrow_ms = steady_ms(lambda: O.build_observations(inner),
                              args.obs_iters, args.warmup)
        narrow = O.build_observations(inner)
        print(f"  140-wide reference          {narrow_ms:8.3f} ms  "
              f"policy={tuple(narrow['policy'].shape)} "
              f"state={tuple(narrow['critic'].shape)}", flush=True)
        print(f"  attributable to obs width   {wide_ms - narrow_ms:+8.3f} ms/step",
              flush=True)
        inner.cfg.obs.obs_list, inner.cfg.obs.state_list = wide_lists

    # ---- 3. observation-builder components -------------------------------
    print("\n3. observation components (wide list)", flush=True)
    with torch.no_grad():
        env_origins = inner.scene.env_origins
        palm_state = inner.robot.data.body_state_w[:, inner._palm_body_id, :]
        palm_pos_w, palm_rot = palm_state[:, 0:3], palm_state[:, 3:7]
        palm_center = O._apply_local_offset(
            palm_pos_w, palm_rot, inner._palm_center_offset, (inner.num_envs,))
        obj_pos = inner.object.data.root_pos_w - env_origins
        obj_rot = inner.object.data.root_quat_w
        kp_off = inner._keypoint_offsets * inner._object_scale_multiplier.unsqueeze(1)
        obj_kp = O._keypoints_world(obj_pos, obj_rot, kp_off)
        bbox, joint_origins, valid = O._joint_link_geometry_obs(
            inner, palm_center, palm_rot, env_origins)
        policy_tensor = O.build_observations(inner)["policy"]
        stand_ins = _field_stand_ins(O, inner)

        components = {
            "_canonical_joint_obs": lambda: O._canonical_joint_obs(inner),
            "body_state_w[:, 22 links]": lambda: inner.robot.data.body_state_w[
                :, inner._joint_link_body_ids, :],
            "_keypoints_world (x3/step)": lambda: O._keypoints_world(
                obj_pos, obj_rot, kp_off),
            "_joint_link_geometry_obs": lambda: O._joint_link_geometry_obs(
                inner, palm_center, palm_rot, env_origins),
            "_object_keypoints_rel_joint (x2/step)": (
                lambda: O._object_keypoints_rel_joint(
                    obj_kp, joint_origins, palm_rot, inner._hand_scale, valid)),
            "_stack_obs_dict policy": lambda: O._stack_obs_dict(
                stand_ins, inner.cfg.obs.obs_list),
            "_stack_obs_dict state": lambda: O._stack_obs_dict(
                stand_ins, inner.cfg.obs.state_list),
            "clamp(policy)": lambda: policy_tensor.clamp(-10.0, 10.0),
        }
        for name, fn in components.items():
            try:
                ms = steady_ms(fn, args.obs_iters, args.warmup)
            except Exception as exc:  # noqa: BLE001 - diagnostic script
                print(f"  {name:<38} skipped ({exc})", flush=True)
                continue
            print(f"  {name:<38} {ms:8.3f} ms", flush=True)

    # ---- 4. the all-zero wrench write ------------------------------------
    print("\n4. wrench DR with force_scale=torque_scale=0", flush=True)
    with torch.no_grad():
        wrench_ms = steady_ms(lambda: A.apply_wrench_dr(inner),
                              args.obs_iters, args.warmup)
        print(f"  apply_wrench_dr                        {wrench_ms:8.3f} ms",
              flush=True)
        print(f"  object.has_external_wrench = "
              f"{getattr(inner.object, 'has_external_wrench', None)}", flush=True)
        write_ms = steady_ms(inner.scene.write_data_to_sim, args.obs_iters,
                             args.warmup)
        print(f"  scene.write_data_to_sim (x2/step)      {write_ms:8.3f} ms",
              flush=True)

    # Re-baseline here rather than reusing section 1: the scene has advanced.
    baseline_step = steady_ms(step_once, args.steps, args.warmup)
    original_wrench = A.apply_wrench_dr
    try:
        A.apply_wrench_dr = lambda _env: None
        for attr in ("_external_force_b", "_external_torque_b"):
            buf = getattr(inner.object, attr, None)
            if buf is not None:
                buf.zero_()
        inner.object.has_external_wrench = False
        no_wrench_ms = steady_ms(step_once, args.steps, args.warmup)
    finally:
        A.apply_wrench_dr = original_wrench
    print(f"  env.step as configured                 {baseline_step:8.3f} ms/step",
          flush=True)
    print(f"  env.step with the wrench block removed {no_wrench_ms:8.3f} ms/step "
          f"({no_wrench_ms - baseline_step:+.3f} ms)", flush=True)

    # ---- 5. reset cost ----------------------------------------------------
    print("\n5. _reset_idx cost by batch size", flush=True)
    for count in (16, 64, 256):
        ids = torch.arange(count, device=device, dtype=torch.long)
        ms = steady_ms(lambda ids=ids: inner._reset_idx(ids), 20, 5)
        print(f"  {count:>4} envs                              {ms:8.3f} ms",
              flush=True)

    env.close()
    app.close()
    os._exit(0)


def _field_stand_ins(module, env):
    """Same-shaped tensors per obs field, so the concatenation is timed alone."""
    import torch

    sizes = module.obs_field_sizes(env.scene_record.robot_spec)
    return {
        name: torch.empty(env.num_envs, sizes[name], device=env.device)
        for name in set(env.cfg.obs.obs_list) | set(env.cfg.obs.state_list)
    }


if __name__ == "__main__":
    main()
