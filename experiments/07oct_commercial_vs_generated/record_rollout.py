"""Record a policy acting on one hand, for viser_play.py: the zero-shot evaluation's setup, step by step.

Builds GenMech-InHandReorient-Direct-v0 with the POLICY's run config (default: the 52-hand Wuji v2 policy, 31826,
epoch 5000), swaps in the hand to watch (env.assets.robot_spec, as eval_niches.py --target does: the network is
built for the target's joints, the action noise resized), runs it greedily, and saves the first --envs envs:
every joint angle, the hand's root pose, the cube, the goal, the palm frame, and each step's goal / drop / reset
flags. 50-goal episodes and timeouts as in training.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python experiments/07oct_commercial_vs_generated/record_rollout.py \
        --spec handonly:assets/urdf/unified_dynamics_commercial_hands_replicas/wuji2/wuji2_left_replica_clone.json \
        --name wuji2_replica_clone
    -> results/rollouts/<name>.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
TASK = "GenMech-InHandReorient-Direct-v0"
DEFAULT_CKPT = ("debug_outputs/train_logs/06oct_wuji2_multi_embodiment/0_scale_train_wuji2_lr_2plus_ema0.1_c01_*_31826_0_*"
                "/rank_0/*/nn/last_*_ep_5000_*.pth")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="registry name, or handonly:<population .json>")
    ap.add_argument("--name", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--num-envs", type=int, default=16, help="simulated")
    ap.add_argument("--envs", type=int, default=4, help="recorded (the first ones)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    spec = args.spec
    if spec.startswith("handonly:") and not spec[9:].startswith("/"):
        spec = "handonly:" + str(REPO / spec[9:])
    ckpt = pathlib.Path(args.checkpoint or sorted(glob.glob(str(REPO / DEFAULT_CKPT)))[0])

    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    app_args, _ = lp.parse_known_args([])
    app_args.headless, app_args.device = True, args.device
    app = AppLauncher(app_args).app

    import gymnasium as gym
    import numpy as np
    import torch
    import yaml
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.eval.rl_player import RlPlayer
    from coevolution.eval._play_worker import _resize_sigma
    from coevolution.utils.hydra_utils import hydra_task_config_with_yaml
    from isaaclab.utils.math import quat_mul

    saved = OmegaConf.load(ckpt.parents[3] / "rank_0/.hydra/config.yaml")
    saved.env.assets.robot_spec = spec
    sys.argv = [sys.argv[0]]

    @hydra_task_config_with_yaml(TASK, "rl_games_joint_transformer_cfg_entry_point")
    def _run(env_cfg, agent_cfg):
        env_d = OmegaConf.to_container(saved.env, resolve=True)
        for k in ("observation_space", "state_space", "action_space"):
            env_d.pop(k, None)
        env_cfg.from_dict(env_d)
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        env = gym.make(TASK, cfg=env_cfg)
        inner = env.unwrapped
        obs, _ = env.reset()
        acfg = OmegaConf.to_container(saved, resolve=True)["agent"]
        acfg["params"]["config"]["multi_gpu"] = False
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump({"train": acfg}, f)
        f.close()
        player = RlPlayer(obs["policy"].shape[-1], inner.action_space.shape[-1], f.name,
                          _resize_sigma(str(ckpt), int(inner.action_space.shape[-1])),
                          device=args.device, sapg_expl_coef=0.0, num_envs=inner.num_envs)

        E, dt = args.envs, float(inner.step_dt)
        steps = int(round(args.seconds / dt))
        org = inner.scene.env_origins[:E]
        R = inner.robot.data
        rec = {k: [] for k in ("q", "root", "object", "goal", "palm", "success", "fall", "reset")}
        for _ in range(steps):
            with torch.no_grad():
                act = player.get_normalized_action(obs["policy"], deterministic_actions=True)
            obs, _, _, _, _ = env.step(act)
            ps = R.body_state_w[:E, inner._palm_body_id, :]
            canon = quat_mul(ps[:, 3:7], inner._palm_frame_quat[:E])
            rec["q"].append(R.joint_pos[:E].cpu().numpy())
            rec["root"].append(torch.cat([R.root_state_w[:E, 0:3] - org, R.root_state_w[:E, 3:7]], -1).cpu().numpy())
            rec["object"].append(torch.cat([inner.object.data.root_pos_w[:E] - org,
                                            inner.object.data.root_quat_w[:E]], -1).cpu().numpy())
            rec["goal"].append(torch.cat([inner.goal_viz.data.root_pos_w[:E] - org,
                                          inner.goal_viz.data.root_quat_w[:E]], -1).cpu().numpy())
            rec["palm"].append(torch.cat([ps[:, 0:3] - org, canon], -1).cpu().numpy())
            rec["success"].append(inner._is_success[:E].cpu().numpy().astype(bool))
            r = inner._termination_reasons
            rec["fall"].append(r["fall"][:E].cpu().numpy().astype(bool))
            rec["reset"].append((r["fall"] | r["timeout"] | r["hand_far"] | r["max_successes"])[:E].cpu().numpy())
        out = HERE / "results/rollouts" / f"{args.name}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = {"name": args.name, "spec": args.spec, "checkpoint": str(ckpt.relative_to(REPO)), "dt": dt,
                "joint_names": list(inner.robot.joint_names), "object_size": list(inner.cfg.reward.fixed_size)}
        np.savez_compressed(out, meta=json.dumps(meta), **{k: np.stack(v) for k, v in rec.items()})
        goals = np.stack(rec["success"]).sum(0)
        print(f"[rollout] {args.name}: {steps} steps x {E} envs, goals per env {goals.tolist()}, "
              f"drops {np.stack(rec['fall']).sum(0).tolist()} -> {out}", flush=True)
        env.close()

    _run()
    app.close()


if __name__ == "__main__":
    main()
