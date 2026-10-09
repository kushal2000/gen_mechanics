"""OOD sweep of one single-hand policy: object mass, friction (object + hand together) or push force, first-episode
protocol.

Protocol: --num-envs envs (64), all reset together; each env's FIRST episode is scored (goals before it ends: drop,
timeout or the 50-goal cap), as 06oct_wuji2_multi_embodiment/analysis/eval_per_hand.py. Greedy actions (SAPG
coefficient 0). Everything else as the policy's own run trained -- its saved env config (EMA, action delay, 5 rad/s,
5 deg tolerance) -- except the swept quantity:

  mass      object mass in grams, via assets.object_density_scale = g / 45.5625 (the 45 mm cube at 500 kg/m^3 every
            run trained with; mass and inertia scale together). Set at scene build: ONE value per process.
  friction  robot_friction = finger_tip_friction = object_friction = mu (trained 0.5), applied live through the
            scene's own material pass, so all values run in one process.
  push      a push of exactly N newtons in a uniformly random direction (domain_randomization.force_magnitude_n,
            independent of the mass), lasting one 1/60 s control step, every 0.3 s (18 steps) -- the viewer's
            cadence. force_only_when_lifted is turned OFF: "lifted" means above the cube's starting height, which
            palm-up rarely is, so the default gating would suppress almost every push.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python experiments/08oct_repro_wuji_full/analysis/eval_ood.py \\
        --checkpoint <pth> --axis friction --values 0.1 0.25 0.5 0.75 1.0 1.5 --out results/ood/<label>/friction.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
TASK = "GenMech-InHandReorient-Direct-v0"
NOMINAL_G = 45.5625
PUSH_PERIOD = 18


def run_dir_of(ckpt: pathlib.Path) -> pathlib.Path:
    """<run>/rank_0/<name>/nn/x.pth or <run>/rank_0/<name>/last/model.pth (or a snapshot next to either)."""
    for p in ckpt.parents:
        if (p / "rank_0/.hydra/config.yaml").exists():
            return p
    raise SystemExit(f"no run config above {ckpt}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--axis", choices=("mass", "friction", "push"), required=True)
    ap.add_argument("--values", nargs="+", type=float, required=True)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--max-seconds", type=float, default=300.0, help="sim-time cap per value")
    ap.add_argument("--max-goals", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.axis == "mass" and len(args.values) != 1:
        raise SystemExit("--axis mass: one value per process (the mass is fixed when the scene is built)")

    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    app_args, _ = lp.parse_known_args([])
    app_args.headless = True
    app = AppLauncher(app_args).app

    import gymnasium as gym
    import tempfile
    import torch
    import yaml
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.eval.rl_player import RlPlayer
    from coevolution.utils.hydra_utils import hydra_task_config_with_yaml
    from isaacsimenvs.pose_reaching_6d.scene_utils.materials import apply_physx_material_properties

    ckpt = pathlib.Path(args.checkpoint).resolve()
    saved = OmegaConf.load(run_dir_of(ckpt) / "rank_0/.hydra/config.yaml")
    sys.argv = [sys.argv[0]]
    out_path = pathlib.Path(args.out)
    # Appends: the mass sweep is one process per value, all writing the same file.
    results = json.loads(out_path.read_text())["results"] if out_path.exists() else {}

    @hydra_task_config_with_yaml(TASK, "rl_games_joint_transformer_cfg_entry_point")
    def _run(env_cfg, agent_cfg):
        env_d = OmegaConf.to_container(saved.env, resolve=True)
        for k in ("observation_space", "state_space", "action_space"):
            env_d.pop(k, None)
        env_cfg.from_dict(env_d)
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        env_cfg.termination.max_consecutive_successes = args.max_goals
        dr = env_cfg.domain_randomization
        dr.force_scale, dr.torque_scale = 0.0, 0.0
        if args.axis == "mass":
            env_cfg.assets.object_density_scale = args.values[0] / NOMINAL_G
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        obs, _ = env.reset()
        acfg = OmegaConf.to_container(saved, resolve=True)["agent"]
        acfg["params"]["config"]["multi_gpu"] = False
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump({"train": acfg}, f)
        f.close()
        player = RlPlayer(obs["policy"].shape[-1], env.action_space.shape[-1], f.name, str(ckpt),
                          device=env.device, sapg_expl_coef=0.0, num_envs=env.num_envs)
        N, dev, dt = env.num_envs, env.device, float(env.step_dt)
        mass_g = float(env._object_mass[0, 0]) * 1000.0

        for v in args.values:
            dr_live = env.cfg.domain_randomization
            if args.axis == "friction":
                for key in ("robot_friction", "finger_tip_friction", "object_friction"):
                    setattr(env.cfg.assets, key, float(v))
                apply_physx_material_properties(env)
            pushing = args.axis == "push" and v > 0
            dr_live.force_magnitude_n = float(v) if pushing else 0.0
            dr_live.force_only_when_lifted = False
            obs, _ = env.reset()
            goals, drops, streak = (torch.zeros(N, device=dev) for _ in range(3))
            first_goals = torch.full((N,), -1.0, device=dev)
            first_drop = torch.zeros(N, dtype=torch.bool, device=dev)
            t0, k = time.time(), 0
            while k < int(round(args.max_seconds / dt)) and bool((first_goals < 0).any()):
                if pushing:
                    env._random_force_prob[:] = 1.0 if k % PUSH_PERIOD == 0 else 0.0
                with torch.no_grad():
                    act = player.get_normalized_action(obs["policy"], deterministic_actions=True)
                obs, *_ = env.step(act)
                r = env._termination_reasons
                end = (r["fall"] | r["timeout"] | r["hand_far"] | r["max_successes"]).float()
                succ = env._is_success.float()
                goals += succ
                streak += succ
                drops += r["fall"].float()
                fresh = (end > 0) & (first_goals < 0)
                first_goals[fresh] = streak[fresh]
                first_drop[fresh] = r["fall"].bool()[fresh]
                streak *= 1.0 - end
                k += 1
            done = first_goals >= 0
            minutes = k * dt / 60.0
            res = {
                "value": v, "axis": args.axis, "object_mass_g": mass_g, "envs": N, "sim_seconds": k * dt,
                "goals_per_episode_first": float(first_goals[done].mean()) if done.any() else float("nan"),
                "goals_per_episode_first_std": float(first_goals[done].std()) if int(done.sum()) > 1 else float("nan"),
                "first_episode_drop_frac": float(first_drop[done].float().mean()) if done.any() else float("nan"),
                "first_episode_unfinished_frac": 1.0 - float(done.float().mean()),
                "goals_per_min": float(goals.sum()) / (N * minutes),
                "drops_per_min": float(drops.sum()) / (N * minutes),
                # Over the whole window (every env, every episode until the last first episode ended):
                "goals_total": float(goals.sum()), "drops_total": float(drops.sum()),
                "goals_per_drop": float(goals.sum()) / max(float(drops.sum()), 1.0),
            }
            results[f"{args.axis}={v:g}"] = res
            print(f"[ood] {args.axis}={v:g}: goals/ep (first) {res['goals_per_episode_first']:.2f}  goals/drop "
                  f"{res['goals_per_drop']:.1f}  drop frac "
                  f"{res['first_episode_drop_frac']:.2f}  goals/min {res['goals_per_min']:.1f}  "
                  f"({time.time() - t0:.0f} s wall)", flush=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"checkpoint": str(ckpt), "results": results}, indent=1))
        env.close()

    _run()
    app.close()


if __name__ == "__main__":
    main()
