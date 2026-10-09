"""Physics blow-up probe: drive a hand with random actions (no policy) and look for states PhysX handles badly.

Hypothesis (8 Oct): the runs whose PPO updates exploded (KL in the thousands) all include ring/pinky fingers, while
thumb+index+middle and index+middle trained stably. Random actions explore the reachable joint space quickly; if
some of it makes PhysX produce huge contact/joint forces, the observations show it: joint or object speeds far beyond
what the 5 rad/s drives can produce, joints far outside their limits, non-finite values.

Same env config as a training run (its saved .hydra config: EMA 0.1, action delay, 5 rad/s, friction), only the
hand and the env count changed. Per hand, over --steps control steps of uniform random actions in [-1, 1]
(re-drawn every --hold steps): max |joint vel|, max object speed / angular speed, max joint-limit violation,
non-finite counts, and the fraction of envs that ever exceed a blow-up threshold.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python experiments/08oct_multi_embodiment_evolution/analysis/\\
        probe_physics.py --hands full only_thumb_index_middle only_index_middle
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
TASK = "GenMech-InHandReorient-Direct-v0"
RUN_CFG = (REPO / "debug_outputs/train_logs/08oct_multi_embodiment_evolution").glob(
    "0_scale_train_left_wuji2_full_ema0.1_dact_*_113950_0_*/rank_0/.hydra/config.yaml")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", nargs="+", default=["full", "only_thumb_index_middle", "only_index_middle"])
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1800, help="control steps (60 per s)")
    ap.add_argument("--hold", type=int, default=6, help="steps each random action is held")
    ap.add_argument("--zero", action="store_true", help="zero actions instead of random (rest-state check)")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent.parent / "results/probe_physics.json"))
    args = ap.parse_args()

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

    saved = OmegaConf.load(next(iter(RUN_CFG)))
    results = json.loads(pathlib.Path(args.out).read_text()) if pathlib.Path(args.out).exists() else {}
    sys.argv = [sys.argv[0]]
    hand = args.hands[0]            # one Kit scene per process: the .sub loops over hands
    spec = "wuji2_left_uniform_handonly" + ("" if hand == "full" else f"_{hand}")
    saved.env.assets.robot_spec = spec

    @hydra_task_config_with_yaml(TASK, "rl_games_joint_transformer_cfg_entry_point")
    def _run(env_cfg, agent_cfg):
        env_d = OmegaConf.to_container(saved.env, resolve=True)
        for k in ("observation_space", "state_space", "action_space"):
            env_d.pop(k, None)
        env_cfg.from_dict(env_d)
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = 0
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        obs, _ = env.reset()
        N, A, dev = env.num_envs, env.action_space.shape[-1], env.device
        lo, hi = env._joint_lower_hand, env._joint_upper_hand
        g = torch.Generator(device=dev).manual_seed(0)
        stats = {k: torch.zeros(N, device=dev) for k in ("jvel", "ovel", "oang", "limit", "nonfinite_obs")}
        jv_series, ov_series = [], []
        act = torch.zeros(N, A, device=dev)
        for k in range(args.steps):
            if not args.zero and k % args.hold == 0:
                act = torch.rand(N, A, device=dev, generator=g) * 2 - 1
            obs, _, _, _, _ = env.step(act)
            q = env.robot.data.joint_pos[:, env._perm_lab_to_canon]
            qd = env.robot.data.joint_vel[:, env._perm_lab_to_canon]
            jv = qd.abs().max(dim=1).values
            ov = env.object.data.root_lin_vel_w.norm(dim=1)
            oa = env.object.data.root_ang_vel_w.norm(dim=1)
            lim = torch.maximum(lo - q, q - hi).clamp(min=0).max(dim=1).values
            nf = (~torch.isfinite(obs["policy"])).any(dim=1).float()
            for name, v in (("jvel", jv), ("ovel", ov), ("oang", oa), ("limit", lim), ("nonfinite_obs", nf)):
                stats[name] = torch.maximum(stats[name], torch.nan_to_num(v, nan=1e9))
            jv_series.append(float(jv.max())); ov_series.append(float(ov.max()))
        q = lambda t, p: float(torch.quantile(t, p))
        res = {
            "spec": spec, "num_envs": N, "steps": args.steps, "zero_actions": args.zero,
            "max_joint_vel": float(stats["jvel"].max()), "p99_env_max_joint_vel": q(stats["jvel"], 0.99),
            "median_env_max_joint_vel": q(stats["jvel"], 0.5),
            "frac_envs_joint_vel_gt_20": float((stats["jvel"] > 20).float().mean()),
            "frac_envs_joint_vel_gt_100": float((stats["jvel"] > 100).float().mean()),
            "max_object_speed": float(stats["ovel"].max()), "frac_envs_object_speed_gt_3": float((stats["ovel"] > 3).float().mean()),
            "max_object_ang_speed": float(stats["oang"].max()), "frac_envs_object_ang_gt_50": float((stats["oang"] > 50).float().mean()),
            "max_limit_violation_rad": float(stats["limit"].max()),
            "frac_envs_limit_violation_gt_0.2": float((stats["limit"] > 0.2).float().mean()),
            "frac_envs_nonfinite_obs": float(stats["nonfinite_obs"].mean()),
            "worst_step_joint_vel": int(max(range(len(jv_series)), key=jv_series.__getitem__)),
        }
        print("[probe] " + json.dumps(res), flush=True)
        results[f"{hand}{'_zero' if args.zero else ''}"] = res
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(results, indent=1))
        env.close()

    _run()
    app.close()


if __name__ == "__main__":
    main()
