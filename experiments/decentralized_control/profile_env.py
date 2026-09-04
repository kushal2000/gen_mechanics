"""Environment- and training-side profiling. Boots Isaac Sim.

The counterpart to profile_net.py: everything here needs the simulator or a
live rl_games loop.

    profile_env.py step      where a rollout step's time goes
    profile_env.py obs       observation correctness (tokens + aliased fields)
    profile_env.py update    instrument update_time inside a real training run

WHAT THIS ESTABLISHED (24576 envs, 100 assets/type, RTX 6000 Ada):

  * The rollout is physics, 81%. The whole 778-wide observation build is
    2.0 ms of a 238 ms step; the extra 653 columns over the reference cost
    0.62 ms. Tensor bookkeeping was never the problem.
  * arm_moving_average is the expensive setting, not the observation:
    1.0 costs 237.7 ms/step against 0.1's 108.4. That is why these runs trail
    the single-embodiment reference. It is a dynamics change, not a free win.
  * update_time was fully explained once instrumented: the central-value net
    was training in fp32 (66.86 ms/minibatch against the actor's 42.73, for a
    strictly smaller network) and every actor minibatch shipped its whole
    gradient to the host twice.

`step` and `obs` take --num_envs / --num_assets_per_type. `update` forwards
everything to coevolution/train.py, so pass that script's arguments:

    profile_env.py update -- --task GenMech-PoseReach-Direct-v0 \
        --agent rl_games_joint_transformer_cfg_entry_point --headless \
        agent.params.config.max_epochs=14 ...
"""

from __future__ import annotations

import argparse
import collections
import functools
import os
import pathlib
import sys
import time

TOTALS: dict[str, float] = collections.defaultdict(float)
COUNTS: dict[str, int] = collections.defaultdict(int)
EPOCH = {"n": 0}

# The reference run's lists, minus fingertip_pos_rel_palm: the field is still
# sized in layout.py but build_observations no longer emits it. The 653 columns
# that separate this from the decentralized list are what is under test.
REFERENCE_OBS_LIST = (
    "joint_pos", "joint_vel", "prev_action_targets", "palm_pos", "palm_rot",
    "object_rot", "keypoints_rel_palm", "keypoints_rel_goal", "object_scales",
)
REFERENCE_STATE_LIST = REFERENCE_OBS_LIST + (
    "palm_vel", "object_vel", "closest_keypoint_max_dist",
    "closest_fingertip_dist", "lifted_object", "progress", "successes", "reward",
)


# --------------------------------------------------------------------------
# update:  instrument a live training run
# --------------------------------------------------------------------------

def _install_update_instrumentation() -> None:
    """Time every seam inside update_time (a2c_common.train_epoch:1386-1455).

    Every boundary synchronizes CUDA, so attribution is unambiguous and the
    total runs slightly above an uninstrumented epoch. Compare shares.
    """
    import torch
    from rl_games.algos_torch import a2c_continuous, central_value
    from rl_games.common import a2c_common, datasets

    device = torch.device("cuda")

    def timed(label, unwrapped):
        @functools.wraps(unwrapped)
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            try:
                return unwrapped(*args, **kwargs)
            finally:
                torch.cuda.synchronize(device)
                TOTALS[label] += time.perf_counter() - start
                COUNTS[label] += 1
        return wrapper

    A, C, D = a2c_continuous.A2CAgent, central_value.CentralValueTrain, datasets.PPODataset
    A.prepare_dataset = timed("prepare_dataset", A.prepare_dataset)
    C.train_net = timed("central_value.train_net (total)", C.train_net)
    C.update_dataset = timed("  central_value.update_dataset", C.update_dataset)
    C.train_critic = timed("  central_value.train_critic", C.train_critic)
    A.train_actor_critic = timed("actor train_actor_critic (total)", A.train_actor_critic)
    A.calc_gradients = timed("  actor calc_gradients", A.calc_gradients)
    a2c_common.A2CBase._preproc_obs = timed("  _preproc_obs (actor)",
                                            a2c_common.A2CBase._preproc_obs)
    C._preproc_obs = timed("  _preproc_obs (critic)", C._preproc_obs)
    D.__getitem__ = timed("  dataset __getitem__", D.__getitem__)
    torch.nn.utils.clip_grad_norm_ = timed("  clip_grad_norm_",
                                           torch.nn.utils.clip_grad_norm_)
    a2c_common.A2CBase.augment_batch_for_mixed_expl = timed(
        "augment_batch_for_mixed_expl (play)",
        a2c_common.A2CBase.augment_batch_for_mixed_expl)

    original = A.train_epoch

    def train_epoch(self, *args, **kwargs):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = original(self, *args, **kwargs)
        torch.cuda.synchronize(device)
        wall = time.perf_counter() - start
        EPOCH["n"] += 1
        if EPOCH["n"] >= 4:   # skip allocator growth and lazy init
            print(f"\n=== epoch {EPOCH['n']}  wall {wall:.3f} s "
                  f"(peak {torch.cuda.max_memory_allocated() / 2**20:,.0f} MiB)",
                  flush=True)
            for key, seconds in TOTALS.items():
                print(f"  {key:<38} {seconds:7.3f} s  "
                      f"{100 * seconds / wall:5.1f}% wall  n={COUNTS[key]:<4} "
                      f"{1e3 * seconds / max(COUNTS[key], 1):7.2f} ms/call",
                      flush=True)
            torch.cuda.reset_peak_memory_stats()
        TOTALS.clear()
        COUNTS.clear()
        return result

    A.train_epoch = train_epoch


def cmd_update(passthrough) -> None:
    # Class-level patches only need to land before Runner.run builds the agent,
    # and rl_games is pure torch, so importing it ahead of AppLauncher is safe
    # (unlike anything under isaaclab.*).
    _install_update_instrumentation()
    sys.argv = [sys.argv[0]] + passthrough
    import coevolution.train as train_module
    train_module.main()


# --------------------------------------------------------------------------
# step / obs:  need a booted env
# --------------------------------------------------------------------------

def _make_env(args):
    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.pose_reaching_6d.env_cfg import PoseReachEnvCfg

    cfg = PoseReachEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type
    cfg.assets.robot_spec = "sharpa_iiwa14"
    cfg.action.arm_moving_average = 1.0
    cfg.action.hand_moving_average = 1.0
    dr = cfg.domain_randomization
    dr.use_obs_delay = dr.use_action_delay = dr.use_object_state_delay_noise = False
    dr.joint_velocity_obs_noise_std = dr.force_scale = dr.torque_scale = 0.0
    if args.position_iterations:
        cfg.sim.physx.min_position_iteration_count = args.position_iterations
        cfg.sim.physx.max_position_iteration_count = args.position_iterations

    boot = time.perf_counter()
    env = gym.make("GenMech-PoseReach-Direct-v0", cfg=cfg)
    env.reset()
    torch.cuda.synchronize(env.unwrapped.device)
    print(f"\nboot {time.perf_counter() - boot:.1f} s  envs={args.num_envs} "
          f"assets_per_type={args.num_assets_per_type} "
          f"position_iterations={cfg.sim.physx.min_position_iteration_count}",
          flush=True)
    return env


def cmd_step(args) -> None:
    import torch
    from isaacsimenvs.pose_reaching_6d.obs_utils import actions as A
    from isaacsimenvs.pose_reaching_6d.obs_utils import observations as O

    env = _make_env(args)
    inner = env.unwrapped
    device = inner.device

    def steady(fn, iters, warmup=None):
        for _ in range(warmup if warmup is not None else args.warmup):
            fn()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize(device)
        return 1e3 * (time.perf_counter() - start) / iters

    def step_once():
        env.step(torch.empty(args.num_envs, inner.cfg.action_space,
                             device=device).uniform_(-1.0, 1.0))

    print("\n1. env.step under random actions, by action smoothing", flush=True)
    smoothing = {}
    for ma in (1.0, 0.1):
        inner.cfg.action.arm_moving_average = ma
        inner.cfg.action.hand_moving_average = ma
        smoothing[ma] = steady(step_once, args.steps)
        print(f"  arm/hand_moving_average {ma:<5} {smoothing[ma]:8.3f} ms/step",
              flush=True)
    delta = smoothing[1.0] - smoothing[0.1]
    print(f"  attributable to smoothing {delta:+8.3f} ms/step "
          f"({100 * delta / smoothing[1.0]:+5.1f}%)  <- the real regression",
          flush=True)
    inner.cfg.action.arm_moving_average = 1.0
    inner.cfg.action.hand_moving_average = 1.0

    print("\n2. build_observations, by field list", flush=True)
    wide = (tuple(inner.cfg.obs.obs_list), tuple(inner.cfg.obs.state_list))
    with torch.no_grad():
        wide_ms = steady(lambda: O.build_observations(inner), args.obs_iters)
        shapes = O.build_observations(inner)
        print(f"  decentralized  {wide_ms:8.3f} ms  "
              f"policy={tuple(shapes['policy'].shape)} "
              f"state={tuple(shapes['critic'].shape)}", flush=True)
        inner.cfg.obs.obs_list, inner.cfg.obs.state_list = (
            REFERENCE_OBS_LIST, REFERENCE_STATE_LIST)
        narrow_ms = steady(lambda: O.build_observations(inner), args.obs_iters)
        print(f"  reference-ish  {narrow_ms:8.3f} ms", flush=True)
        print(f"  cost of the wider observation {wide_ms - narrow_ms:+8.3f} ms/step",
              flush=True)
        inner.cfg.obs.obs_list, inner.cfg.obs.state_list = wide

    print("\n3. observation components", flush=True)
    with torch.no_grad():
        origins = inner.scene.env_origins
        palm = inner.robot.data.body_state_w[:, inner._palm_body_id, :]
        palm_rot = palm[:, 3:7]
        centre = O._apply_local_offset(palm[:, 0:3], palm_rot,
                                       inner._palm_center_offset, (inner.num_envs,))
        obj_pos = inner.object.data.root_pos_w - origins
        obj_rot = inner.object.data.root_quat_w
        kp = inner._keypoint_offsets * inner._object_scale_multiplier.unsqueeze(1)
        obj_kp = O._keypoints_world(obj_pos, obj_rot, kp)
        _, joint_origins, valid = O._joint_link_geometry_obs(inner, centre, palm_rot, origins)
        for name, fn in (
            ("_canonical_joint_obs", lambda: O._canonical_joint_obs(inner)),
            ("_keypoints_world", lambda: O._keypoints_world(obj_pos, obj_rot, kp)),
            ("_joint_link_geometry_obs",
             lambda: O._joint_link_geometry_obs(inner, centre, palm_rot, origins)),
            ("_object_keypoints_rel_joint",
             lambda: O._object_keypoints_rel_joint(obj_kp, joint_origins, palm_rot,
                                                   inner._hand_scale, valid)),
        ):
            print(f"  {name:<32} {steady(fn, args.obs_iters):8.3f} ms", flush=True)

    print("\n4. the all-zero wrench write (force_scale = torque_scale = 0)", flush=True)
    with torch.no_grad():
        print(f"  apply_wrench_dr          {steady(lambda: A.apply_wrench_dr(inner), args.obs_iters):8.3f} ms",
              flush=True)
        print(f"  scene.write_data_to_sim  {steady(inner.scene.write_data_to_sim, args.obs_iters):8.3f} ms"
              "   (x decimation per step)", flush=True)

    print("\n5. _reset_idx by batch size", flush=True)
    for count in (16, 64, 256):
        ids = torch.arange(count, device=device, dtype=torch.long)
        print(f"  {count:>4} envs {steady(lambda ids=ids: inner._reset_idx(ids), 20, 5):8.3f} ms",
              flush=True)

    env.close()


def cmd_obs(args) -> None:
    """Token layout correctness, and that the aliased noisy fields are exact."""
    import torch
    from isaacsimenvs.pose_reaching_6d.obs_utils import observations as O
    from isaacsimenvs.pose_reaching_6d.obs_utils.layout import field_offsets

    env = _make_env(args)
    inner = env.unwrapped
    offsets = field_offsets(inner.cfg.obs.obs_list, inner.scene_record.robot_spec)
    actions = torch.zeros(args.num_envs, inner.cfg.action_space, device=inner.device)

    for step in range(args.steps):
        obs, *_ = env.step(actions.uniform_(-1.0, 1.0))
        policy = obs["policy"]
        with torch.no_grad():
            origins = inner.scene.env_origins
            palm = inner.robot.data.body_state_w[:, inner._palm_body_id, :]
            palm_rot = palm[:, 3:7]
            centre = O._apply_local_offset(palm[:, 0:3], palm_rot,
                                           inner._palm_center_offset, (inner.num_envs,))
            palm_pos = centre - origins
            obj_pos = inner.object.data.root_pos_w - origins
            obj_rot = inner.object.data.root_quat_w
            goal_pos = inner.goal_viz.data.root_pos_w - origins
            goal_rot = inner.goal_viz.data.root_quat_w
            kp = inner._keypoint_offsets * inner._object_scale_multiplier.unsqueeze(1)
            obj_kp = O._keypoints_world(obj_pos, obj_rot, kp)
            goal_kp = O._keypoints_world(goal_pos, goal_rot, kp)
            # What the pre-aliasing code did: recompute from the same inputs.
            noisy = O._keypoints_world(obj_pos, obj_rot, kp)
            _, joint_origins, valid = O._joint_link_geometry_obs(
                inner, centre, palm_rot, origins)
            expected = {
                "keypoints_rel_palm": noisy - palm_pos.unsqueeze(1),
                "keypoints_rel_goal": noisy - goal_kp,
                "object_keypoints_rel_joint": O._object_keypoints_rel_joint(
                    noisy, joint_origins, palm_rot, inner._hand_scale, valid),
            }
        clip = inner.cfg.obs.clamp_abs_observations
        for name, value in expected.items():
            start, end = offsets[name]
            want = value.reshape(value.shape[0], -1).clamp(-clip, clip)
            if not torch.equal(policy[:, start:end], want):
                raise AssertionError(
                    f"step {step}: {name} differs by up to "
                    f"{(policy[:, start:end] - want).abs().max().item():g}")
        print(f"  step {step}: {', '.join(expected)} bit-identical", flush=True)

    print(f"\nPASS: aliased noisy fields match recomputation exactly over "
          f"{args.steps} steps at {args.num_envs} envs", flush=True)

    # ---- per-field statistics ------------------------------------------
    # A field pinned against clamp_abs_observations carries no gradient and
    # no information. joint_link_bbox and object_keypoints_rel_joint are both
    # divided by hand_scale and together are 528 of the 778 columns, so a bad
    # scale there silently blanks two thirds of the observation.
    clip = inner.cfg.obs.clamp_abs_observations
    print(f"\nper-field statistics over {args.steps} steps "
          f"(clamp +/-{clip}, hand_scale={float(inner._hand_scale[0, 0]):.4f})",
          flush=True)
    print(f"  {'field':<30} {'cols':>5} {'mean':>9} {'std':>9} {'min':>9} "
          f"{'max':>9} {'%|x|>=clip':>11} {'%zero-var':>10}", flush=True)
    acc = []
    for _ in range(args.steps):
        o, *_ = env.step(actions.uniform_(-1.0, 1.0))
        acc.append(o["policy"].clone())
    allobs = torch.cat(acc, dim=0)
    for field, (start, end) in offsets.items():
        block = allobs[:, start:end]
        pinned = (block.abs() >= clip - 1e-4).float().mean().item() * 100
        percol = block.std(dim=0)
        deadcol = (percol < 1e-6).float().mean().item() * 100
        print(f"  {field:<30} {end-start:>5} {block.mean():>9.3f} "
              f"{block.std():>9.3f} {block.min():>9.3f} {block.max():>9.3f} "
              f"{pinned:>10.1f}% {deadcol:>9.1f}%", flush=True)
    pinned_all = (allobs.abs() >= clip - 1e-4).float().mean().item() * 100
    dead_all = (allobs.std(dim=0) < 1e-6).float().mean().item() * 100
    print(f"  {'-- WHOLE OBSERVATION':<30} {allobs.shape[1]:>5} "
          f"{'':>39} {pinned_all:>10.1f}% {dead_all:>9.1f}%", flush=True)
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("step", "obs", "update"))
    parser.add_argument("--num_envs", type=int, default=24576)
    parser.add_argument("--num_assets_per_type", type=int, default=100)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--obs_iters", type=int, default=100)
    parser.add_argument("--position_iterations", type=int, default=0,
                        help="override PhysX min/max position iterations "
                             "(task default 8); fixed at scene creation")
    args, passthrough = parser.parse_known_args()

    sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
    if args.command == "update":
        cmd_update([a for a in passthrough if a != "--"])
        return

    from isaaclab.app import AppLauncher
    app_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(app_parser)
    app_args = app_parser.parse_args([])
    app_args.headless = True
    app = AppLauncher(app_args).app

    {"step": cmd_step, "obs": cmd_obs}[args.command](args)
    app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
