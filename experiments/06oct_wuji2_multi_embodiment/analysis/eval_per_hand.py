"""Per-hand performance of ONE multi-embodiment policy, measured fresh rather than read off training's logs.

The training run's per_hand/* curves average each hand's last 3000 episodes (rlgames_utils.py), which at
~236 envs per hand is a ~1000-epoch trailing window. This instead builds the run's own 52-hand scene (its saved
config, every hand dealt round-robin as in training), runs a checkpoint greedily and splits the results by hand.

Headline: goals in each env's FIRST episode, averaged over the hand's envs (goals_per_episode_first, as in
05oct's missing-finger plots). Every env starts at reset together, so this is not biased toward short (dropped)
episodes the way an average over episodes that ended inside a window is. Runs until every env's first episode
has ended, or --max-seconds of sim time; envs still in their first episode then are reported, not averaged.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \
        experiments/06oct_wuji2_multi_embodiment/analysis/eval_per_hand.py               # -> results/eval_per_hand_ep<N>.json
    .venv_isaacsim/bin/python experiments/06oct_wuji2_multi_embodiment/analysis/plot_per_hand.py \
        experiments/06oct_wuji2_multi_embodiment/results/eval_per_hand_ep<N>.json     # -> plots/eval_per_hand_ep<N>.png

Everything as trained (episode length, 50-goal cap, physics) except greedy actions (SAPG coefficient 0).
Also goals/min and drops/min per env over the whole window.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[3]
RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
sys.path.insert(0, str(REPO))

TASK = "GenMech-InHandReorient-Direct-v0"
LOGS = REPO / "debug_outputs/train_logs/06oct_wuji2_multi_embodiment"


def newest_checkpoint(job: str) -> pathlib.Path:
    pths = glob.glob(f"{LOGS}/0_scale_train_*_{job}_0_*/rank_0/*/nn/last_*_ep_*.pth")
    if not pths:
        raise SystemExit(f"no checkpoints for job {job} under {LOGS}")
    return pathlib.Path(max(pths, key=lambda p: int(re.search(r"_ep_(\d+)_", p).group(1))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="31826")
    ap.add_argument("--checkpoint", default="", help="default: the job's highest-epoch checkpoint")
    ap.add_argument("--envs-per-hand", type=int, default=64)
    ap.add_argument("--max-seconds", type=float, default=600.0,
                    help="sim-time cap; stops earlier once every env's first episode has ended")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="default: results/eval_per_hand_ep<N>.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    app_args, _ = lp.parse_known_args([])
    app_args.headless, app_args.device = True, args.device
    app = AppLauncher(app_args).app

    import gymnasium as gym
    import tempfile
    import torch
    import yaml
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.eval.rl_player import RlPlayer
    from coevolution.utils.hydra_utils import hydra_task_config_with_yaml

    ckpt = pathlib.Path(args.checkpoint) if args.checkpoint else newest_checkpoint(args.job)
    # The run's config, not the checkpoint's folder: a copied checkpoint (e.g. a snapshot of the best-reward
    # one, which training keeps overwriting) lives outside the run.
    run = glob.glob(f"{LOGS}/0_scale_train_*_{args.job}_0_*/rank_0/.hydra/config.yaml")
    saved = OmegaConf.load(run[0] if run else ckpt.parents[2] / ".hydra/config.yaml")
    sys.argv = [sys.argv[0]]
    print(f"[per_hand] checkpoint {ckpt.name}", flush=True)
    if not args.out:
        epoch = re.search(r"_ep_(\d+)_", ckpt.name).group(1)
        args.out = str(RESULTS / f"eval_per_hand_ep{epoch}.json")

    @hydra_task_config_with_yaml(TASK, "rl_games_joint_transformer_cfg_entry_point")
    def _run(env_cfg, agent_cfg):
        env_d = OmegaConf.to_container(saved.env, resolve=True)
        for k in ("observation_space", "state_space", "action_space"):
            env_d.pop(k, None)
        env_cfg.from_dict(env_d)
        env_cfg.seed = args.seed

        # The hand count is the scene's; build once at a placeholder size to learn it would cost a second
        # Kit scene, so size from the spec instead: round-robin dealing needs a multiple of the hand count.
        from isaacsimenvs.common.robots.multi_hand import spec_names
        n_hands = len(spec_names(saved.env.assets.robot_spec))
        env_cfg.scene.num_envs = n_hands * args.envs_per_hand

        env = gym.make(TASK, cfg=env_cfg)
        inner = env.unwrapped
        obs, _ = env.reset()
        hs = inner.scene_record.hand_set
        hands = [s.hand_name for s in hs.specs]
        idx = inner.scene_record.robot_design_index.to(inner.device).long()
        assert len(hands) == n_hands, (len(hands), n_hands)

        acfg = OmegaConf.to_container(saved, resolve=True)["agent"]
        acfg["params"]["config"]["multi_gpu"] = False
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump({"train": acfg}, f)
        f.close()
        player = RlPlayer(obs["policy"].shape[-1], inner.action_space.shape[-1], f.name, str(ckpt),
                          device=args.device, sapg_expl_coef=0.0, num_envs=inner.num_envs)

        N, dev, dt = inner.num_envs, inner.device, float(inner.step_dt)
        max_steps = int(round(args.max_seconds / dt))
        goals, drops, streak = (torch.zeros(N, device=dev) for _ in range(3))
        first_goals = torch.full((N,), -1.0, device=dev)       # goals in the first episode; -1 = still running
        first_drop = torch.zeros(N, dtype=torch.bool, device=dev)
        t0 = time.time()
        k = 0
        while k < max_steps and bool((first_goals < 0).any()):
            with torch.no_grad():
                act = player.get_normalized_action(obs["policy"], deterministic_actions=True)
            obs, _, _, _, _ = env.step(act)
            succ = inner._is_success.float()
            r = inner._termination_reasons
            fall = r["fall"].float()
            end = (r["fall"] | r["timeout"] | r["hand_far"] | r["max_successes"]).float()
            goals += succ
            streak += succ
            drops += fall
            fresh = (end > 0) & (first_goals < 0)
            first_goals[fresh] = streak[fresh]
            first_drop[fresh] = r["fall"].bool()[fresh]
            streak *= 1.0 - end
            k += 1
            if k % 600 == 0:
                print(f"[per_hand] step {k} ({k * dt:.0f} s sim, {time.time() - t0:.0f} s wall): "
                      f"{int((first_goals < 0).sum())} of {N} envs still in their first episode", flush=True)

        minutes = k * dt / 60.0
        out = {"checkpoint": str(ckpt), "seconds": k * dt, "envs_per_hand": args.envs_per_hand, "hands": {}}
        for h, name in enumerate(hands):
            m = idx == h
            n = int(m.sum())
            done = m & (first_goals >= 0)
            out["hands"][name] = {
                "envs": n,
                "goals_per_episode_first": float(first_goals[done].mean()) if done.any() else float("nan"),
                "goals_per_episode_first_std": float(first_goals[done].std()) if int(done.sum()) > 1 else float("nan"),
                "first_episode_drop_frac": float(first_drop[done].float().mean()) if done.any() else float("nan"),
                "first_episode_unfinished_frac": 1.0 - int(done.sum()) / n,
                "goals_per_min": float(goals[m].sum()) / (n * minutes),
                "drops_per_min": float(drops[m].sum()) / (n * minutes),
            }
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"[per_hand] wrote {args.out} ({time.time() - t0:.0f} s)", flush=True)
        env.close()

    _run()
    app.close()


if __name__ == "__main__":
    main()
