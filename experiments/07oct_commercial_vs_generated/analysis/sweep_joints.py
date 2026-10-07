"""Collision check in free space: sweep each joint of a hand alone in Isaac Sim and see whether it tracks.

The env is built as the policy runs it (the 52-hand policy's config) but driven DIRECTLY: PD targets are written to
the articulation and the physics is stepped, bypassing the action pipeline (no moving average, no policy). The cube
is parked far away. Every hand joint is held at 0 (clamped into its limits); then one joint at a time goes lower ->
upper -> lower limit while the rest hold. An unfiltered self-collision shows as a joint that stops short of its
range (low coverage, large error) and/or shoves the other joints off their targets.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python experiments/07oct_commercial_vs_generated/analysis/sweep_joints.py \
        --spec handonly:assets/urdf/unified_dynamics_commercial_hands_replicas/wuji2/wuji2_left_replica_clone.json \
        --name wuji2_replica_clone
    -> results/sweeps/<name>.json
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
OUT = pathlib.Path(__file__).resolve().parent.parent / "results/sweeps"
sys.path.insert(0, str(REPO))
TASK = "GenMech-InHandReorient-Direct-v0"
CKPT = ("debug_outputs/train_logs/06oct_wuji2_multi_embodiment/0_scale_train_wuji2_lr_2plus_ema0.1_c01_*_31826_0_*"
        "/rank_0/*/nn/last_*_ep_5000_*.pth")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--sweep-seconds", type=float, default=2.0, help="one way, lower -> upper")
    args = ap.parse_args()
    spec = args.spec
    if spec.startswith("handonly:") and not spec[9:].startswith("/"):
        spec = "handonly:" + str(REPO / spec[9:])

    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    app = AppLauncher(a).app

    import gymnasium as gym
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.utils.hydra_utils import hydra_task_config_with_yaml

    ckpt = pathlib.Path(sorted(glob.glob(str(REPO / CKPT)))[0])
    saved = OmegaConf.load(ckpt.parents[3] / "rank_0/.hydra/config.yaml")
    saved.env.assets.robot_spec = spec
    sys.argv = [sys.argv[0]]

    @hydra_task_config_with_yaml(TASK, "rl_games_joint_transformer_cfg_entry_point")
    def _run(env_cfg, agent_cfg):
        env_d = OmegaConf.to_container(saved.env, resolve=True)
        for k in ("observation_space", "state_space", "action_space"):
            env_d.pop(k, None)
        env_cfg.from_dict(env_d)
        env_cfg.scene.num_envs = 1
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        env.reset()
        rob, obj, sim = env.robot, env.object, env.sim
        dt = float(env.physics_dt)
        hid = [int(i) for i in env._hand_joint_ids]
        lim = rob.data.joint_pos_limits[0].cpu().numpy()
        real = [i for i in hid if lim[i, 1] - lim[i, 0] > 1e-6]
        names = rob.joint_names
        far = obj.data.root_state_w.clone()
        far[:, 0:3] = env.scene.env_origins + torch.tensor([0.0, 0.0, -5.0], device=env.device)
        far[:, 7:] = 0.0

        home = torch.zeros_like(rob.data.joint_pos)
        for i in range(home.shape[1]):
            home[0, i] = float(np.clip(0.0, lim[i, 0], lim[i, 1]))

        def step(target):
            obj.write_root_state_to_sim(far)
            rob.set_joint_position_target(target)
            rob.write_data_to_sim()
            sim.step(render=False)
            rob.update(dt)
            obj.update(dt)

        def settle(n=240):
            rob.write_joint_state_to_sim(home, torch.zeros_like(home))
            for _ in range(n):
                step(home)

        n_way = int(args.sweep_seconds / dt)
        results = {}
        for j in real:
            settle()
            lo, hi = float(lim[j, 0]), float(lim[j, 1])
            path = np.concatenate([np.linspace(home[0, j].item(), lo, n_way // 2), np.linspace(lo, hi, n_way),
                                   np.linspace(hi, lo, n_way)])
            q_j, err, others = [], [], 0.0
            for v in path:
                t = home.clone()
                t[0, j] = float(v)
                step(t)
                q = rob.data.joint_pos[0]
                q_j.append(float(q[j]))
                err.append(abs(float(q[j]) - float(v)))
                dev = (q - home[0]).abs()
                dev[j] = 0.0
                others = max(others, float(dev[real].max()))
            q_j = np.array(q_j)
            cov = (q_j.max() - q_j.min()) / (hi - lo)
            results[names[j]] = {"range": [lo, hi], "reached": [float(q_j.min()), float(q_j.max())],
                                 "coverage": float(cov), "max_err_deg": float(np.degrees(max(err))),
                                 "mean_err_deg": float(np.degrees(np.mean(err))),
                                 "max_other_deviation_deg": float(np.degrees(others))}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{args.name}.json").write_text(json.dumps({"spec": args.spec, "physics_dt": dt, "joints": results},
                                                          indent=1))
        print(f"[sweep] {args.name}: {len(results)} joints", flush=True)
        for n, r in results.items():
            flag = "  <-- " if r["coverage"] < 0.95 or r["max_other_deviation_deg"] > 3.0 else ""
            print(f"  {n:26s} coverage {100 * r['coverage']:5.1f}%  max err {r['max_err_deg']:6.1f} deg  "
                  f"others pushed {r['max_other_deviation_deg']:5.1f} deg{flag}", flush=True)
        env.close()

    _run()
    app.close()


if __name__ == "__main__":
    main()
