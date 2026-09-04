"""Zero-shot finger dropout: does one checkpoint drive a hand it never saw?

Runs a policy trained on the intact 22-DoF SHARPA hand, unchanged, on six
morphologies -- intact plus one per removed finger -- and reports goals hit
over a fixed horizon.

    python experiments/decentralized_control/eval/eval_finger_dropout.py \
        --checkpoint debug_outputs/train_logs/<run>/0_pose_reach_jt_sapg/nn/<x>.pth \
        --config     debug_outputs/train_logs/<run>/.hydra/config.yaml

WHY THE SAME WEIGHTS FIT A SMALLER HAND. Every parameter of
JointTransformerNet is independent of the hand joint count: token_dim is 32 and
global_dim is 74 for any hand, mu_head is shared across tokens, arm_head and
value_head read only the global vector, and attention is length-agnostic.
Removing a finger changes only the NUMBER of tokens (22 -> 18 or 17). The two
exceptions -- ``sigma`` and the observation normalizer -- are sliced by
canonical index in finger_specs.remap_checkpoint. sigma's values are
irrelevant under deterministic playback but its shape must satisfy
load_state_dict; running_mean_std's values matter a great deal, so its columns
are selected rather than reinitialised.

The MLP baseline cannot be run this way at all -- its first layer is
Linear(obs_dim + 32, 1024), so a different hand is a different network. That
asymmetry is the result, not an oversight.

THE METRIC. ``env._successes`` counts goals in the CURRENT episode and is
zeroed on reset, so reading it once at the end would undercount every env that
reset. Each env's count is banked at the reset boundary and added to whatever
still stands at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help=".pth trained on the intact hand")
    p.add_argument("--config", default="",
                   help="the run's .hydra/config.yaml; defaults to searching "
                        "two levels above the checkpoint")
    p.add_argument("--variants", default="intact,thumb,index,middle,ring,pinky")
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--num_assets_per_type", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sapg_expl_coef", type=float, default=50.0,
                   help="trailing exploration column a SAPG checkpoint expects; "
                        "pass a negative value for a plain PPO checkpoint")
    p.add_argument("--sampled", action="store_true",
                   help="sample actions instead of using mu")
    p.add_argument("--out", default="debug_outputs/eval_logs/finger_dropout.json")
    return p.parse_known_args()[0]


def find_config(args) -> str:
    if args.config:
        return args.config
    # .../<run>/<experiment>/nn/<ckpt>.pth  ->  .../<run>/.hydra/config.yaml
    for parent in pathlib.Path(args.checkpoint).resolve().parents:
        candidate = parent / ".hydra" / "config.yaml"
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "could not find the run's .hydra/config.yaml above "
        f"{args.checkpoint}; pass --config explicitly")


def main() -> None:
    args = parse_args()
    repo = pathlib.Path(__file__).parents[3]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    os.chdir(repo)   # urdf_path and the config path are repo-relative

    run_config = find_config(args)
    if not pathlib.Path(args.checkpoint).is_file():
        # rl_games' restore silently keeps random weights for a missing path.
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    from isaaclab.app import AppLauncher
    app_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(app_parser)
    app_args = app_parser.parse_args([])
    app_args.headless = True
    app = AppLauncher(app_args).app

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401
    import finger_specs
    from coevolution.eval.rl_player import RlPlayer
    from coevolution.population.run_config import (
        load_run_config, synthesise_policy_config,
    )
    from isaacsimenvs.pose_reaching_6d.env_cfg import PoseReachEnvCfg
    from isaacsimenvs.pose_reaching_6d.scene_utils.robots import SHARPA_IIWA14

    # RlPlayer reads cfg["train"]; a run stores the identical rl_games block
    # under "agent". The repo already synthesises one from the other, and it
    # resolves against the WHOLE config because num_actors interpolates out to
    # env.scene.num_envs -- an agent-only save leaves a dangling key that fails
    # minutes later, inside a Kit boot.
    config_path = synthesise_policy_config(
        load_run_config(pathlib.Path(run_config).parent.parent),
        pathlib.Path(args.out).parent / "policy_config.yaml", args.num_envs)
    print(f"synthesised policy config from {run_config} -> {config_path}",
          flush=True)

    # PoseReachEnvCfg() carries the CURRENT default obs/state lists, which
    # happen to match the joint_transformer runs. A checkpoint trained on any
    # other list (the 140-d MLP, say) would then be fed a differently-shaped
    # and differently-ordered observation with no error anywhere -- the widths
    # are only checked against the network. Take the lists from the run that
    # produced the checkpoint.
    import yaml as _yaml
    _run = _yaml.safe_load(open(run_config))["env"]["obs"]
    ckpt_obs_list = tuple(_run["obs_list"])
    ckpt_state_list = tuple(_run.get("state_list", ()))

    coef = None if args.sapg_expl_coef < 0 else args.sapg_expl_coef
    results = {}

    for variant in (v.strip() for v in args.variants.split(",") if v.strip()):
        spec = finger_specs.register(variant)
        print(f"\n{'=' * 74}\n{variant}: {spec.num_hand_joints} hand joints, "
              f"{spec.num_joints} actions, {spec.num_fingertip_slots} fingertips"
              f"\n  urdf {spec.urdf_path}\n{'=' * 74}", flush=True)

        # A fresh cfg per variant: derive_spaces refuses to overwrite a
        # non-zero action_space, so a reused cfg carries the previous hand's.
        cfg = PoseReachEnvCfg()
        cfg.obs.obs_list = ckpt_obs_list
        if ckpt_state_list:
            cfg.obs.state_list = ckpt_state_list
        cfg.scene.num_envs = args.num_envs
        cfg.assets.num_assets_per_type = args.num_assets_per_type
        cfg.assets.robot_spec = spec.name
        cfg.assets.robot_population_path = ""
        cfg.assets.robot_population_seed = -1
        cfg.assets.robot_population_count = 0
        cfg.action.arm_moving_average = 1.0
        cfg.action.hand_moving_average = 1.0
        dr = cfg.domain_randomization
        dr.use_obs_delay = dr.use_action_delay = dr.use_object_state_delay_noise = False
        dr.joint_velocity_obs_noise_std = dr.force_scale = dr.torque_scale = 0.0
        cfg.seed = args.seed

        boot = time.perf_counter()
        env = gym.make("GenMech-PoseReach-Direct-v0", cfg=cfg)
        inner = env.unwrapped
        obs, _ = env.reset()
        n_obs = obs["policy"].shape[1]
        print(f"  booted in {time.perf_counter() - boot:.0f} s; "
              f"obs {n_obs} actions {inner.cfg.action_space}", flush=True)

        # Built with no checkpoint, then loaded through the remap: rl_games'
        # restore is a strict load and would reject the resized sigma.
        player = RlPlayer(
            num_observations=n_obs, num_actions=inner.cfg.action_space,
            config_path=config_path, checkpoint_path=None,
            device=str(inner.device), sapg_expl_coef=coef,
            num_envs=args.num_envs)
        _load_remapped(player, args.checkpoint, SHARPA_IIWA14, spec,
                       list(cfg.obs.obs_list), finger_specs)

        results[variant] = rollout(env, inner, player, args, spec)
        env.close()
        del env, player
        torch.cuda.empty_cache()

    report(results, args, run_config)
    app.close()
    os._exit(0)


def _load_remapped(player, checkpoint, full_spec, spec, obs_list, finger_specs):
    """Slice the intact-hand checkpoint onto this hand and load it."""
    from rl_games.algos_torch import torch_ext

    ckpt = torch_ext.load_checkpoint(checkpoint)
    if 0 in ckpt:
        ckpt = ckpt[0]
    ckpt = finger_specs.remap_checkpoint(ckpt, full_spec, spec, obs_list)

    model = player.player.model
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    resized = [k for k in ckpt["model"]
               if k in dict(model.named_parameters()) or k in dict(model.named_buffers())]
    print(f"  loaded {len(resized)} tensors; {len(missing)} missing, "
          f"{len(unexpected)} unexpected", flush=True)
    for k in list(missing)[:5]:
        print(f"      missing:    {k}", flush=True)
    for k in list(unexpected)[:5]:
        print(f"      unexpected: {k}", flush=True)
    if player.player.normalize_input and "running_mean_std" in ckpt:
        player.player.model.running_mean_std.load_state_dict(
            ckpt["running_mean_std"])


def rollout(env, inner, player, args, spec) -> dict:
    import torch

    device = inner.device
    obs, _ = env.reset()
    player.reset()
    banked = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    episodes = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    prev = inner._successes.clone()

    t0 = time.perf_counter()
    for step in range(args.steps):
        action = player.get_normalized_action(
            obs=obs["policy"], deterministic_actions=not args.sampled)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        if bool(done.any()):
            banked += torch.where(done, prev, torch.zeros_like(prev))
            episodes += done.long()
        prev = inner._successes.clone()
        if (step + 1) % 1000 == 0:
            print(f"    step {step + 1:>5}/{args.steps}  "
                  f"goals/env {(banked + prev).float().mean():6.3f}  "
                  f"{(step + 1) / (time.perf_counter() - t0):5.0f} steps/s",
                  flush=True)
    total = (banked + prev).float()
    out = {
        "hand_joints": spec.num_hand_joints, "actions": spec.num_joints,
        "fingertips": spec.num_fingertip_slots,
        "goals_per_env_mean": float(total.mean()),
        "goals_per_env_std": float(total.std()),
        "goals_total": float(total.sum()),
        "frac_env_with_a_goal": float((total > 0).float().mean()),
        "episodes_per_env": float(episodes.float().mean()),
        "envs": args.num_envs, "steps": args.steps,
    }
    print(f"  -> {out['goals_per_env_mean']:.3f} goals/env "
          f"({out['frac_env_with_a_goal'] * 100:.0f}% of envs scored at least one)",
          flush=True)
    return out


def report(results, args, config_path) -> None:
    print(f"\n{'=' * 88}")
    print(f"ZERO-SHOT FINGER DROPOUT   {args.num_envs} envs x {args.steps} steps, "
          f"{'sampled' if args.sampled else 'deterministic'}")
    print(f"  checkpoint {args.checkpoint}")
    print(f"  config     {config_path}")
    print(f"{'=' * 88}")
    hdr = (f"{'variant':<9} {'hand':>5} {'act':>4} {'tips':>5} {'goals/env':>10} "
           f"{'std':>7} {'vs intact':>10} {'envs scoring':>13} {'episodes':>9}")
    print(hdr + "\n" + "-" * len(hdr))
    base = results.get("intact", {}).get("goals_per_env_mean")
    for name, r in results.items():
        rel = f"{r['goals_per_env_mean'] / base:9.2f}x" if base else f"{'-':>10}"
        print(f"{name:<9} {r['hand_joints']:>5} {r['actions']:>4} "
              f"{r['fingertips']:>5} {r['goals_per_env_mean']:>10.3f} "
              f"{r['goals_per_env_std']:>7.3f} {rel:>10} "
              f"{r['frac_env_with_a_goal'] * 100:>12.0f}% "
              f"{r['episodes_per_env']:>9.1f}")
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"checkpoint": args.checkpoint, "config": config_path,
         "num_envs": args.num_envs, "steps": args.steps,
         "deterministic": not args.sampled, "results": results}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
