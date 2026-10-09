"""Smoke test of the OOD-axis DR (DomainRandomizationCfg push / contact friction / object mass / hand tilt).

Builds the full-hand MLP run's env (118744's saved config) with every OOD-axis DR on, steps it with the policy's
own checkpoint, and reports: the friction values PhysX holds per env, the object masses, the push sizes, the tilt
forces, and the step time. Then the tilt check: one process with the hand really tilted (HAND_ONLY_PALM_PITCH_DEG)
and one with the tilt emulated by the object force should move a cube resting on a still hand the same way.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python experiments/08oct_repro_wuji_full/analysis/smoke_ood_dr.py \\
        --mode dr            # DR on, stats + timing
        --mode tilt_force --tilt 20     # emulated tilt, zero actions: cube drift
        --mode tilt_real --tilt 20      # real tilt, zero actions: cube drift
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
TASK = "GenMech-InHandReorient-Direct-v0"
RUN = "debug_outputs/train_logs/08oct_repro_wuji_full/0_scale_train_left_wuji2_full_v5_ema0.1_dact_mlp_e24k_c01_mlp_e24576_b114688_s100_118744_0_yz4RjI"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("dr", "dr_off", "tilt_force", "tilt_real"), required=True)
    ap.add_argument("--tilt", type=float, default=20.0)
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    if args.mode == "tilt_real":
        os.environ["HAND_ONLY_PALM_PITCH_DEG"] = str(args.tilt)
        os.environ["HAND_ONLY_ALLOW_ANY_SLOPE"] = "1"

    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    app_args, _ = lp.parse_known_args([])
    app_args.headless = True
    app = AppLauncher(app_args).app

    import gymnasium as gym
    import torch
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.utils.hydra_utils import hydra_task_config_with_yaml

    saved = OmegaConf.load(REPO / RUN / "rank_0/.hydra/config.yaml")
    sys.argv = [sys.argv[0]]

    @hydra_task_config_with_yaml(TASK, "rl_games_joint_transformer_cfg_entry_point")
    def _run(env_cfg, agent_cfg):
        env_d = OmegaConf.to_container(saved.env, resolve=True)
        for k in ("observation_space", "state_space", "action_space"):
            env_d.pop(k, None)
        env_cfg.from_dict(env_d)
        env_cfg.scene.num_envs = args.num_envs
        dr = env_cfg.domain_randomization
        dr.use_action_delay = False
        if args.mode == "dr":
            dr.push_magnitude_range_n = (0.0, 1.5)
            dr.force_prob_range = (1 / 18, 1 / 18)
            dr.force_only_when_lifted = False
            dr.contact_friction_range = (0.25, 1.0)
            dr.object_mass_scale_range = (15 / 45.5625, 250 / 45.5625)
            dr.tilt_range_deg = (-5.0, 15.0)
        if args.mode == "tilt_force":
            dr.tilt_range_deg = (args.tilt, args.tilt)
        env_cfg.termination.max_consecutive_successes = 1000
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        env.reset()
        N, dev = env.num_envs, env.device
        act = torch.zeros(N, env.action_space.shape[-1], device=dev)
        if args.mode.startswith("tilt"):
            # Hold the reset pose (joint_units "range": target = lower + (a + 1) / 2 (upper - lower)) so the cube
            # stays on the hand and only the slope moves it.
            ids = env._hand_joint_ids
            q = env.robot.data.joint_pos[:, ids]
            act[:, ids] = (2 * (q - env._hand_lower) / (env._hand_upper - env._hand_lower) - 1).clamp(-1, 1)
        start = env.object.data.root_pos_w[:, :3].clone() - env.scene.env_origins
        t0 = time.time()
        for k in range(args.steps):
            env.step(act)
            if k == 14 and args.mode.startswith("tilt"):
                pos = env.object.data.root_pos_w[:, :3] - env.scene.env_origins
                d = (pos - start).mean(0)
                if args.mode == "tilt_real":     # into the untilted hand's axes: rotate back about x by tilt - 5 deg
                    import math
                    a = -math.radians(args.tilt - 5.0)
                    d = torch.stack([d[0], math.cos(a) * d[1] - math.sin(a) * d[2],
                                     math.sin(a) * d[1] + math.cos(a) * d[2]])
                print(f"[smoke] {args.mode} tilt {args.tilt:g}: mean cube displacement after 0.25 s, hand axes "
                      f"x {d[0] * 1000:+.1f} y {d[1] * 1000:+.1f} z {d[2] * 1000:+.1f} mm", flush=True)
                break
        dt = (time.time() - t0) / (k + 1)
        print(f"[smoke] {args.mode}: {dt * 1000:.1f} ms per env step at {N} envs", flush=True)
        if args.mode == "dr":
            rob = env.robot.root_physx_view.get_material_properties()
            obj = env.object.root_physx_view.get_material_properties()
            per_env = rob[:, :, 0]
            same = bool((per_env == per_env[:, :1]).all())
            print(f"[smoke] robot friction: {len(per_env[:, 0].unique())} distinct values "
                  f"{per_env[:, 0].min():.3f}..{per_env[:, 0].max():.3f}; all shapes of an env equal: {same}; "
                  f"object == robot per env: {bool(torch.allclose(obj[:, 0, 0], rob[:, 0, 0]))}", flush=True)
            m = env._object_mass[:, 0] * 1000
            print(f"[smoke] object mass g: min {m.min():.1f} median {m.median():.1f} max {m.max():.1f}", flush=True)
            p = env._push_magnitude
            print(f"[smoke] push N: min {p.min():.2f} mean {p.mean():.2f} max {p.max():.2f}", flush=True)
            f = env._tilt_force[:, 0] / (env._object_mass * 9.81)
            print(f"[smoke] tilt force / mg: y {f[:, 1].min():+.3f}..{f[:, 1].max():+.3f} "
                  f"z {f[:, 2].min():+.4f}..{f[:, 2].max():+.4f}", flush=True)
        env.close()

    _run()
    app.close()


if __name__ == "__main__":
    main()
