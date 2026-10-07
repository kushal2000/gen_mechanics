"""Can a hand's joints follow commands in the in-hand task's sim? Drive every joint with a slow sine over its range
(normalized action), and report how far each one lags its target -- a jammed joint (a finger root pressed into
the palm, say) shows as a large steady error.

One Isaac Sim process per hand, same config as capture_spawn.py (12491's):

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python experiments/07oct_commercial_vs_generated/analysis/track_joints.py \
        --spec handonly:assets/urdf/unified_dynamics_commercial_hands_replicas/wuji2/wuji2_left_replica_clone.json
"""
from __future__ import annotations

import argparse
import glob
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
TASK = "GenMech-InHandReorient-Direct-v0"
RUN = "debug_outputs/train_logs/06oct_wuji_5rads/0_scale_train_left_wuji2_full_v5_ema0.1_c01_*_12491_0_*"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--num-envs", type=int, default=4)
    args = ap.parse_args()
    spec = args.spec
    if spec.startswith("handonly:") and not spec[9:].startswith("/"):
        spec = "handonly:" + str(REPO / spec[9:])

    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    app_args, _ = lp.parse_known_args([])
    app_args.headless = True
    app = AppLauncher(app_args).app

    import gymnasium as gym
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.utils.hydra_utils import hydra_task_config_with_yaml

    saved = OmegaConf.load(pathlib.Path(sorted(glob.glob(str(REPO / RUN)))[0]) / "rank_0/.hydra/config.yaml")
    saved.env.assets.robot_spec = spec
    sys.argv = [sys.argv[0]]

    @hydra_task_config_with_yaml(TASK, "rl_games_joint_transformer_cfg_entry_point")
    def _run(env_cfg, agent_cfg):
        env_d = OmegaConf.to_container(saved.env, resolve=True)
        for k in ("observation_space", "state_space", "action_space"):
            env_d.pop(k, None)
        env_cfg.from_dict(env_d)
        env_cfg.scene.num_envs = args.num_envs
        env = gym.make(TASK, cfg=env_cfg)
        inner = env.unwrapped
        env.reset()
        hid = torch.as_tensor(inner._hand_joint_ids, device=inner.device)
        lim = inner.robot.data.joint_pos_limits[0, hid]
        real = ((lim[:, 1] - lim[:, 0]) > 1e-6).cpu().numpy()
        names = [inner.robot.joint_names[i] for i in hid.tolist()]
        A = inner.action_space.shape[-1]
        dt = float(inner.step_dt)
        steps = int(args.seconds / dt)
        err = np.zeros(len(names)); span = np.zeros(len(names))
        lo_seen = np.full(len(names), np.inf); hi_seen = np.full(len(names), -np.inf)
        for k in range(steps):
            a = 0.8 * math.sin(2 * math.pi * k * dt / 3.0)              # a 3 s period, 80% of the range
            act = torch.full((inner.num_envs, A), a, device=inner.device)
            env.step(act)
            q = inner.robot.data.joint_pos[0, hid].cpu().numpy()
            tgt = inner._cur_targets[0, hid].cpu().numpy()
            if k > steps // 3:
                err += np.abs(q - tgt)
            lo_seen = np.minimum(lo_seen, q); hi_seen = np.maximum(hi_seen, q)
        n = steps - steps // 3 - 1
        rng = (lim[:, 1] - lim[:, 0]).cpu().numpy()
        print(f"[track] {args.spec}")
        for i, nm in enumerate(names):
            if real[i]:
                print(f"  {nm:26s} mean |q - target| {np.degrees(err[i] / n):6.1f} deg   travelled "
                      f"{100 * (hi_seen[i] - lo_seen[i]) / rng[i]:5.1f}% of range")
        print(f"[track] mean over joints {np.degrees((err[real] / n).mean()):.1f} deg", flush=True)
        env.close()

    _run()
    app.close()


if __name__ == "__main__":
    main()
