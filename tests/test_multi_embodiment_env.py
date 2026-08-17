"""Tier-1 contract test: one distinct hand per env, in the real task env.

The multi-embodiment claim is easy to make and easy to get silently wrong. Two
failure modes have already been paid for in this repo, both of which look like
success from the outside:

  * every env receives the SAME design, because the spawner collapsed the pool
    or the USDs were identical -- the scene runs, the numbers look fine, and no
    morphology diversity exists at all;
  * envs receive DIFFERENT designs than their observation buffers describe,
    which is what happened to the object pool: 510 of 512 envs held the wrong
    asset and the policy score fell 5.07 -> 3.00 while every asset-level
    comparison passed.

So this asserts diversity actually reached the simulator, and that it reached
the same env the observation was built for:

  * N Robot prims, one articulation view (a second view means the joint
    template broke and designs stopped sharing dof_count)
  * joint limits differ across envs -- read from the sim, not the config, so a
    collapsed pool cannot pass
  * the fingertip validity mask varies across envs, and matches the design each
    env was assigned
  * observations are finite through reset and stepping, with the padded
    fingertip slots reading exactly zero where a finger is ghosted

    .venv_isaacsim/bin/python tests/test_multi_embodiment_env.py \\
      --num_envs 16 --population_count 8 --steps 10
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--population_seed", type=int, default=0)
    parser.add_argument("--population_count", type=int, default=8)
    parser.add_argument("--num_assets_per_type", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    app = AppLauncher(args).app

    import gymnasium as gym
    import torch

    import genmech.tasks  # noqa: F401  registers the env
    from isaaclab.sim.utils import find_matching_prim_paths
    from genmech.tasks.pose_reach.env_multi_cfg import PoseReachMultiEnvCfg
    from genmech.tasks.pose_reach.utils.morphology import DESCRIPTOR_DIM

    cfg = PoseReachMultiEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type
    cfg.assets.robot_population_seed = args.population_seed
    cfg.assets.robot_population_count = args.population_count

    env = gym.make("GenMech-PoseReachMulti-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    failures: list[str] = []

    def check(ok: bool, msg: str) -> None:
        print(f"[multi] {'ok  ' if ok else 'FAIL'} {msg}")
        if not ok:
            failures.append(msg)

    specs = inner._robot_population_specs
    check(specs is not None and len(specs) == args.population_count,
          f"population resolved: {0 if specs is None else len(specs)} designs "
          f"(expected {args.population_count})")

    robot_prims = find_matching_prim_paths("/World/envs/env_.*/Robot")
    # BASE PLACEMENT. The arm is fixed-base, so its root_joint pins it wherever
    # the prim sits and init_state cannot move it afterwards. Authoring prims
    # with spawn=None meant nothing applied the spawner's translation: the robot
    # sat at the env origin, on top of the table instead of 0.8 m behind it, and
    # the whole task geometry was wrong. It reached a 24k training run, and
    # nothing in this suite would have caught it.
    import torch as _torch

    _origins = env.scene.env_origins
    _base = (env.robot.data.root_pos_w - _origins)[0]
    _want = _torch.tensor(env.robot_spec.base_pos, device=_base.device,
                          dtype=_base.dtype)
    check(bool(_torch.allclose(_base, _want, atol=1e-3)),
          f"robot base at {[round(v, 4) for v in _base.tolist()]}, "
          f"spec.base_pos says {list(env.robot_spec.base_pos)}")

    check(len(robot_prims) == args.num_envs,
          f"{len(robot_prims)} Robot prims (expected {args.num_envs})")

    # One articulation view holding every design is the whole point: ghosting
    # pads all designs to one joint count so they can share it.
    check(len(inner.scene.articulations) == 1,
          f"one articulation view (got {len(inner.scene.articulations)})")
    check(inner.robot.num_instances == args.num_envs,
          f"view holds all {args.num_envs} envs "
          f"(got {inner.robot.num_instances})")

    # --- diversity actually reached the simulator --------------------------
    limits = inner.robot.data.joint_pos_limits.detach().cpu()
    sigs = {limits[e].numpy().tobytes() for e in range(args.num_envs)}
    check(len(sigs) > 1,
          f"joint limits differ across envs: {len(sigs)} distinct signatures "
          f"(1 would mean every env got the same hand)")

    # Designs repeat when num_envs > population_count, so the count of distinct
    # signatures should be the number of distinct designs actually placed.
    idx = inner._robot_design_index_per_env.detach().cpu()
    n_designs_used = len(set(idx.tolist()))
    check(len(sigs) <= n_designs_used,
          f"{len(sigs)} limit signatures across {n_designs_used} designs used "
          f"(more signatures than designs would mean envs were mixed up)")

    # --- the mask matches the design the env was assigned ------------------
    mask = inner._fingertip_mask.detach().cpu()
    check(mask.shape == (args.num_envs, inner._num_fingertips),
          f"fingertip mask is per-env: {tuple(mask.shape)} "
          f"(expected {(args.num_envs, inner._num_fingertips)})")
    mismatched = [
        e for e in range(args.num_envs)
        if tuple(mask[e].tolist()) != tuple(specs[int(idx[e])].fingertip_slot_mask)
    ]
    check(not mismatched,
          f"every env's mask matches its design "
          f"({len(mismatched)} mismatched)")
    distinct_masks = {tuple(mask[e].tolist()) for e in range(args.num_envs)}
    print(f"[multi]      {len(distinct_masks)} distinct fingertip masks over "
          f"{args.num_envs} envs: "
          f"{sorted(''.join('A' if b else '.' for b in m) for m in distinct_masks)}")

    offs = inner._fingertip_offsets.detach().cpu()
    check(offs.shape == (args.num_envs, inner._num_fingertips, 3),
          f"fingertip offsets are per-env: {tuple(offs.shape)}")

    # --- morphology descriptor ---------------------------------------------
    morph = inner._morphology_per_env.detach().cpu()
    check(morph.shape == (args.num_envs, DESCRIPTOR_DIM),
          f"morphology is per-env: {tuple(morph.shape)} "
          f"(expected {(args.num_envs, DESCRIPTOR_DIM)})")
    check(bool(torch.isfinite(morph).all()), "morphology descriptor is finite")

    # It must DISTINGUISH designs -- an all-constant descriptor would train a
    # policy that cannot tell its hands apart while looking perfectly healthy.
    distinct_morph = {morph[e].numpy().tobytes() for e in range(args.num_envs)}
    check(len(distinct_morph) == n_designs_used,
          f"{len(distinct_morph)} distinct descriptors for {n_designs_used} "
          f"designs (must be one-to-one)")

    # And it must describe the design THIS env actually holds. Envs sharing a
    # design must share a descriptor, and envs with different designs must not.
    by_design: dict[int, set] = {}
    for e in range(args.num_envs):
        by_design.setdefault(int(idx[e]), set()).add(morph[e].numpy().tobytes())
    check(all(len(v) == 1 for v in by_design.values()),
          "envs sharing a design share a descriptor")
    check(len({next(iter(v)) for v in by_design.values()}) == len(by_design),
          "envs with different designs have different descriptors")

    # The descriptor is a property of the hand, not of the episode: it must not
    # move when the sim does.
    morph_before = morph.clone()

    # Observation width must account for it.
    from genmech.tasks.pose_reach.utils.obs_utils import compute_obs_dim
    expected_obs = compute_obs_dim(cfg.obs.obs_list, inner.robot_spec)
    check("morphology" in cfg.obs.obs_list,
          "morphology is in the actor obs list")
    check(inner.cfg.observation_space == expected_obs,
          f"observation_space {inner.cfg.observation_space} == "
          f"computed {expected_obs}")

    # --- observations survive reset and stepping ---------------------------
    obs, _ = env.reset(seed=0)
    def _flat(o):
        t = o["policy"] if isinstance(o, dict) and "policy" in o else o
        return t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
    check(bool(torch.isfinite(_flat(obs)).all()), "observations finite after reset")

    act = torch.zeros(inner.num_envs, inner.cfg.action_space, device=inner.device)
    for _ in range(args.steps):
        obs, _, _, _, _ = env.step(act)
    check(bool(torch.isfinite(_flat(obs)).all()),
          f"observations finite after {args.steps} steps")

    # Ghosted slots must read exactly zero -- a ghost finger's link still has a
    # pose, and letting it into the observation is the silent failure the mask
    # exists to prevent.
    ghost = ~inner._fingertip_mask
    if bool(ghost.any()):
        d = inner._curr_fingertip_distances
        check(bool((d[ghost] == 0.0).all()),
              f"all {int(ghost.sum())} ghosted fingertip distances are zero")
        cf = inner._closest_fingertip_dist
        check(bool((cf[ghost] == 0.0).all()),
              "ghosted slots stay zero in closest_fingertip_dist")
    else:
        print("[multi] note: population has no ghosted fingers; mask untested")

    check(bool(torch.equal(inner._morphology_per_env.detach().cpu(), morph_before)),
          f"morphology unchanged after reset and {args.steps} steps")

    if failures:
        print(f"\n[multi] {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("\nmulti-embodiment env test OK")

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1 if failures else 0)


if __name__ == "__main__":
    main()
