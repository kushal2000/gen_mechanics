"""Validate and profile the SHARPA joint-token observation path.

This is intentionally separate from training: CUDA synchronization is required
for trustworthy timings and would destroy rollout throughput if left in the env.

Example::

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \
      experiments/decentralized_control/profile_sharpa_joint_obs.py \
      --headless --num_envs 4096 --steps 200 --policy both
"""

from __future__ import annotations

import argparse
import os
import time


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--detail_steps", type=int, default=10,
        help="Policy steps in the synchronized component-level breakdown.",
    )
    parser.add_argument(
        "--policy", choices=("mlp", "transformer", "both"), default="both",
        help="Policy forward path(s) to benchmark alongside the environment.",
    )
    parser.add_argument("--num_assets_per_type", type=int, default=2)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    app = AppLauncher(args).app

    import gymnasium as gym
    import torch

    import coevolution.networks  # noqa: F401
    import isaacsimenvs  # noqa: F401
    from coevolution.networks.joint_transformer import JointTransformerNet
    from rl_games.algos_torch.network_builder import A2CBuilder
    from isaacsimenvs.pose_reaching_6d.env_cfg import PoseReachEnvCfg
    from isaacsimenvs.pose_reaching_6d.obs_utils.layout import (
        build_token_layout, field_offsets,
    )
    from isaacsimenvs.pose_reaching_6d.obs_utils.observations import (
        _normalize_joint_pos, build_observations, compute_intermediate_values,
    )
    from isaacsimenvs.pose_reaching_6d.reset_utils.logging_utils import (
        log_step_metrics,
    )
    from isaacsimenvs.pose_reaching_6d.reward_utils.rewards import compute_rewards
    from isaacsimenvs.pose_reaching_6d.reward_utils.termination import (
        compute_terminations, update_tolerance_curriculum,
    )

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

    create_start = time.perf_counter()
    env = gym.make("GenMech-PoseReach-Direct-v0", cfg=cfg)
    create_ms = 1e3 * (time.perf_counter() - create_start)
    inner = env.unwrapped
    reset_start = time.perf_counter()
    obs, _ = env.reset()
    torch.cuda.synchronize(inner.device)
    initial_reset_ms = 1e3 * (time.perf_counter() - reset_start)
    policy = obs["policy"]
    assert policy.shape == (args.num_envs, 778), policy.shape

    # ---- ordered local-box invariants -------------------------------------
    box = inner._joint_link_bbox_local[0]
    e1, e2, e3 = box[:, 1] - box[:, 0], box[:, 2] - box[:, 0], box[:, 3] - box[:, 0]
    lengths = torch.stack(
        [torch.linalg.vector_norm(e1, dim=-1),
         torch.linalg.vector_norm(e2, dim=-1),
         torch.linalg.vector_norm(e3, dim=-1)], dim=-1
    )
    assert torch.all(lengths > 0), lengths
    unit = torch.stack([e1, e2, e3], dim=1) / lengths[:, :, None]
    gram = unit @ unit.transpose(1, 2)
    eye = torch.eye(3, device=gram.device).expand_as(gram)
    torch.testing.assert_close(gram, eye, atol=2e-5, rtol=0.0)
    proximal_center = box[:, 0] + 0.5 * e2 + 0.5 * e3
    torch.testing.assert_close(
        proximal_center, torch.zeros_like(proximal_center), atol=2e-6, rtol=0.0
    )

    # ---- exact flat-vector/token contract --------------------------------
    spec = inner.scene_record.robot_spec
    layout = build_token_layout(spec, cfg.obs.obs_list)
    gather = torch.tensor(layout["token_columns"], device=policy.device)
    tokens = policy[:, gather]
    assert tokens.shape == (args.num_envs, 22, 32), tokens.shape
    offsets = field_offsets(cfg.obs.obs_list, spec)
    bbox_start, bbox_end = offsets["joint_link_bbox"]
    bbox_from_flat = policy[:, bbox_start:bbox_end].reshape(args.num_envs, 22, 12)
    # Five proprioceptive scalars precede the twelve box coordinates.
    torch.testing.assert_close(tokens[:, :, 5:17], bbox_from_flat)

    # ---- reset and one-step history alignment -----------------------------
    pos_start, pos_end = offsets["joint_pos"]
    prev_start, prev_end = offsets["prev_joint_pos"]
    torch.testing.assert_close(
        policy[:, pos_start:pos_end], policy[:, prev_start:prev_end],
        atol=2e-5, rtol=0.0,
    )
    before = inner.robot.data.joint_pos[:, inner._perm_lab_to_canon].clone()
    actions = torch.zeros(
        args.num_envs, inner.cfg.action_space, device=inner.device
    )
    next_obs, *_ = env.step(actions)
    expected_prev = _normalize_joint_pos(
        before, inner._joint_lower_canon, inner._joint_upper_canon
    )
    torch.testing.assert_close(
        next_obs["policy"][:, prev_start:prev_end], expected_prev,
        atol=2e-5, rtol=0.0,
    )

    # ---- real policy backbones used by the matched experiments ------------
    net_params = {
        "robot_spec": "sharpa_iiwa14",
        "obs_list": list(cfg.obs.obs_list),
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 4,
        "ff_mult": 4,
        "dropout": 0.0,
        "arm_head_units": [1024, 512],
        "value_head_units": [512, 256],
        "space": {"continuous": {
            "mu_activation": "None",
            "sigma_activation": "None",
            "sigma_init": {"name": "const_initializer", "val": 0},
            "fixed_sigma": "fixed",
        }},
    }
    transformer_net = None
    if args.policy in ("transformer", "both"):
        transformer_net = JointTransformerNet(
            net_params, actions_num=inner.cfg.action_space,
            input_shape=(inner.cfg.observation_space,), value_size=1,
            num_seqs=1, type="simple",
        ).to(inner.device).eval()

    mlp_net = None
    if args.policy in ("mlp", "both"):
        mlp_params = {
            "separate": False,
            "space": {"continuous": {
                "mu_activation": "None",
                "sigma_activation": "None",
                "mu_init": {"name": "default"},
                "sigma_init": {"name": "const_initializer", "val": 0},
                # SAPG's coef_cond only changes the tiny sigma lookup.  A fixed
                # sigma isolates the configured [1024,1024,512,512] backbone.
                "fixed_sigma": "fixed",
            }},
            "mlp": {
                "units": [1024, 1024, 512, 512],
                "activation": "elu",
                "d2rl": False,
                "initializer": {"name": "default"},
                "regularizer": {"name": "None"},
            },
        }
        mlp_builder = A2CBuilder()
        mlp_builder.load(mlp_params)
        mlp_net = mlp_builder.build(
            "pose_reach_profile", actions_num=inner.cfg.action_space,
            input_shape=(inner.cfg.observation_space,), value_size=1,
            num_seqs=1, type="simple",
        ).to(inner.device).eval()

    with torch.inference_mode():
        for policy_name, policy_net in (
            ("transformer", transformer_net), ("mlp", mlp_net),
        ):
            if policy_net is None:
                continue
            mu, sigma, value, _ = policy_net({"obs": next_obs["policy"]})
            assert mu.shape == (args.num_envs, 29), (policy_name, mu.shape)
            assert sigma.shape in ((29,), (args.num_envs, 29)), (
                policy_name, sigma.shape
            )
            assert value.shape == (args.num_envs, 1), (policy_name, value.shape)
    print(
        f"validation: PASS  obs={tuple(policy.shape)} "
        f"tokens={tuple(tokens.shape)} action={(args.num_envs, 29)} "
        f"policy={args.policy}",
        flush=True,
    )
    print("\nEnvironment startup", flush=True)
    print(f"  construction (gym.make) {create_ms / 1e3:8.3f} s", flush=True)
    print(f"  initial reset           {initial_reset_ms / 1e3:8.3f} s", flush=True)
    print(
        f"  ready for policy        {(create_ms + initial_reset_ms) / 1e3:8.3f} s",
        flush=True,
    )

    def wall_ms(fn, iterations: int) -> float:
        """Amortized host+device latency with one sync around the full loop."""
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize(inner.device)
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize(inner.device)
        return 1e3 * (time.perf_counter() - start) / iterations

    def detailed_step_timing(iterations: int) -> tuple[dict[str, float], int]:
        """Run the DirectRLEnv step path with synchronization at each boundary.

        This intentionally mirrors Isaac Lab's ``DirectRLEnv.step``.  The
        synchronization makes attribution unambiguous, at the cost of a small
        perturbation relative to the uninterrupted throughput measurement.
        """
        totals: dict[str, float] = {}
        reset_count = 0

        def measured(name, fn):
            torch.cuda.synchronize(inner.device)
            start = time.perf_counter()
            result = fn()
            torch.cuda.synchronize(inner.device)
            totals[name] = totals.get(name, 0.0) + 1e3 * (
                time.perf_counter() - start
            )
            return result

        def update_counters():
            inner.episode_length_buf += 1
            inner.common_step_counter += 1

        def task_state_and_dones():
            update_tolerance_curriculum(inner)
            compute_intermediate_values(inner)
            terminated, timed_out = compute_terminations(inner)
            inner.reset_terminated[:], inner.reset_time_outs[:] = (
                terminated, timed_out
            )
            inner.reset_buf = inner.reset_terminated | inner.reset_time_outs

        def reward_math():
            inner.reward_buf = compute_rewards(inner)

        def reset_finished_envs():
            nonlocal reset_count
            reset_ids = inner.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            reset_count += int(reset_ids.numel())
            if reset_ids.numel() > 0:
                inner._reset_idx(reset_ids)

        def interval_events():
            if inner.cfg.events and "interval" in inner.event_manager.available_modes:
                inner.event_manager.apply(mode="interval", dt=inner.step_dt)

        def observations():
            inner.obs_buf = inner._get_observations()
            if inner.cfg.observation_noise_model:
                inner.obs_buf["policy"] = inner._observation_noise_model(
                    inner.obs_buf["policy"]
                )

        for _ in range(iterations):
            action = actions.to(inner.device)
            if inner.cfg.action_noise_model:
                action = inner._action_noise_model(action)
            measured("action preprocessing", lambda: inner._pre_physics_step(action))

            for _ in range(inner.cfg.decimation):
                inner._sim_step_counter += 1
                measured("apply target", inner._apply_action)
                measured("write commands", inner.scene.write_data_to_sim)
                measured("physics simulate", lambda: inner.sim.step(render=False))
                measured(
                    "state refresh",
                    lambda: inner.scene.update(dt=inner.physics_dt),
                )

            measured("step counters", update_counters)
            measured("task state + dones", task_state_and_dones)
            measured("reward math", reward_math)
            measured("metric logging", lambda: log_step_metrics(inner))
            measured("reset handling", reset_finished_envs)
            measured("interval events", interval_events)
            measured("observation creation", observations)

        return {name: value / iterations for name, value in totals.items()}, reset_count

    latest = next_obs["policy"]
    with torch.inference_mode():
        obs_ms = wall_ms(lambda: build_observations(inner), args.steps)
        policy_ms = {}
        if mlp_net is not None:
            policy_ms["MLP"] = wall_ms(
                lambda: mlp_net({"obs": latest}), args.steps
            )
        if transformer_net is not None:
            policy_ms["transformer"] = wall_ms(
                lambda: transformer_net({"obs": latest}), args.steps
            )
        step_ms = wall_ms(lambda: env.step(actions), args.steps)
        detail, reset_count = detailed_step_timing(args.detail_steps)

    non_obs_ms = max(0.0, step_ms - obs_ms)
    print("\nSteady-state latency per policy step", flush=True)
    print(f"  environment step       {step_ms:8.3f} ms", flush=True)
    print(f"  observation creation   {obs_ms:8.3f} ms  ({100*obs_ms/step_ms:5.1f}% env)", flush=True)
    print(f"  env excluding obs      {non_obs_ms:8.3f} ms  ({100*non_obs_ms/step_ms:5.1f}% env)", flush=True)
    for policy_name, net_ms in policy_ms.items():
        rollout_ms = step_ms + net_ms
        print(
            f"  {policy_name + ' forward':<23} {net_ms:8.3f} ms  "
            f"({100*net_ms/rollout_ms:5.1f}% rollout)",
            flush=True,
        )
        print(
            f"  {policy_name + ' rollout total':<23} {rollout_ms:8.3f} ms",
            flush=True,
        )
    print(
        "  note: env-excluding-obs contains physics, state refresh, rewards, "
        "terminations and action processing",
        flush=True,
    )

    detail_total = sum(detail.values())
    print("\nSynchronized environment-step breakdown", flush=True)
    for name, value in detail.items():
        print(
            f"  {name:<24} {value:8.3f} ms  "
            f"({100.0 * value / detail_total:5.1f}%)",
            flush=True,
        )
    print(f"  {'measured total':<24} {detail_total:8.3f} ms", flush=True)
    print(
        f"  reset envs processed      {reset_count} across "
        f"{args.detail_steps} policy steps",
        flush=True,
    )
    print(
        "  note: each detailed boundary synchronizes CUDA for attribution; "
        "use steady-state latency above for true uninterrupted throughput",
        flush=True,
    )

    env.close()
    app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
