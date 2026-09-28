"""Train a rung of the IsaacLab in-hand ladder, IsaacLab's way.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python -m isaacsimenvs.inhand_isaaclab.train \
        --task GenMech-InHandIsaacLab-step0-reference-v0 --num-envs 8192 --max-epochs 5000

WHY NOT coevolution/train.py. Our training stack is built around a RobotSpec and the
joint transformer: it derives the observation layout from a spec, resolves a population,
banks per-design returns, and cross-checks the network's width against the env's. A
manager-based env has none of those. Bending our stack around it would make the
reference no longer a reference -- the whole value of this folder is that the baseline
is THEIRS, learner included -- so this uses IsaacLab's own rl_games wrapper and the
agent config their task ships.

That means numbers here are NOT directly comparable to our runs' wandb charts: different
learner, different observation, different reward scale. What IS comparable across the
rungs is each other, because the agent config is held fixed at every rung.
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", default="GenMech-InHandIsaacLab-step0-reference-v0")
parser.add_argument("--num-envs", type=int, default=8192)
parser.add_argument("--max-epochs", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--minibatch-size", type=int, default=0,
                    help="0 keeps the agent config's own value")
parser.add_argument("--run-dir", default="",
                    help="where checkpoints and logs go; defaults under debug_outputs")
parser.add_argument("--checkpoint", default="", help="restore and continue from this .pth")
parser.add_argument("--wandb", action="store_true", help="log to the inhand wandb project")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
if not hasattr(args, "headless") or args.headless is None:
    args.headless = True

app = AppLauncher(args).app

import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

from isaacsimenvs.inhand_isaaclab import register

register()   # after AppLauncher, never at import time -- see register()'s docstring

REPO = "/share/portal/kk837/gen_mechanics"


def main() -> None:
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rung = args.task.replace("GenMech-InHandIsaacLab-", "").replace("-v0", "")
    # The leading "0_" is NOT decoration. Our vendored rl_games reads the policy index
    # straight out of the experiment name -- a2c_common.py:84,
    # `int(self.experiment_name.split('_')[0])` -- for its multi-policy support, and
    # raises ValueError on a name that does not start with an integer and an underscore.
    # It is why every run in this repo is called 0_scale_train_...
    run_dir = args.run_dir or os.path.join(
        REPO, "debug_outputs/train_logs/inhand_isaaclab", f"0_{rung}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    agent_cfg["params"]["seed"] = args.seed
    cfg = agent_cfg["params"]["config"]
    cfg["max_epochs"] = args.max_epochs
    cfg["num_actors"] = env_cfg.scene.num_envs
    cfg["train_dir"] = run_dir
    cfg["full_experiment_name"] = os.path.basename(run_dir)
    cfg["name"] = os.path.basename(run_dir)
    if args.minibatch_size:
        cfg["minibatch_size"] = args.minibatch_size

    # OUR rl_games IS A SAPG FORK, and IsaacLab's stock agent config predates it. The
    # fork reads these keys unconditionally in places: a2c_common.py:329 defaults
    # expl_type to 'none', but :400 calls .startswith() on config.get('expl_type') with
    # no default, so an absent key is an AttributeError rather than plain PPO. Setting
    # them here rather than patching the fork keeps the reference's learner theirs in
    # every respect that matters, and records the requirement where it bites.
    cfg.setdefault("expl_type", "none")            # 'none' = vanilla PPO in this fork
    cfg.setdefault("use_others_experience", "none")
    cfg.setdefault("expl_coef_block_size", cfg["num_actors"])
    cfg.setdefault("expl_reward_coef_scale", 0.0)

    # And the fork's sigma is not stock either: it takes a STRING, one of 'fixed',
    # 'coef_cond' or 'obs_cond', where stock rl_games takes a bool. IsaacLab's True
    # matches none of them and falls through to the branch that builds sigma as a
    # Linear, after which const_initializer tries to .fill_() a module -- "'Linear'
    # object has no attribute 'fill_'".
    #
    # 'fixed' is the plain-PPO path and the right translation of their True. NOT
    # 'coef_cond', which our own SAPG configs use: that conditions sigma on the
    # exploration coefficient and needs per-env coef_ids from the env side, which a
    # manager-based env does not supply -- it fails with KeyError: 'coef_ids'.
    space = agent_cfg["params"]["network"].get("space", {}).get("continuous")
    if space is not None and not isinstance(space.get("fixed_sigma"), str):
        space["fixed_sigma"] = "fixed"
    # rl_games requires horizon * actors to be divisible by the minibatch. Their config
    # is written for their default env count, so a different --num-envs silently gives
    # the LAST minibatch a different width unless this is checked.
    rollout = cfg["horizon_length"] * cfg["num_actors"]
    if rollout % cfg["minibatch_size"]:
        raise SystemExit(
            f"minibatch {cfg['minibatch_size']} does not divide horizon*actors = "
            f"{cfg['horizon_length']}*{cfg['num_actors']} = {rollout}. "
            f"Pass --minibatch-size with a divisor, e.g. {math.gcd(rollout, 32768)}.")

    print(f"[train] task        {args.task}")
    print(f"[train] envs        {env_cfg.scene.num_envs}  decimation {env_cfg.decimation}  "
          f"episode {env_cfg.episode_length_s} s")
    thr = env_cfg.commands.object_pose.orientation_success_threshold
    print(f"[train] threshold   {thr:.4f} rad ({math.degrees(thr):.2f} deg)")
    print(f"[train] minibatch   {cfg['minibatch_size']} of horizon*actors {rollout} "
          f"({rollout // cfg['minibatch_size']} per epoch)")
    print(f"[train] run dir     {run_dir}", flush=True)

    if args.wandb:
        import wandb

        wandb.init(project="gen_mechanics_inhandreorient", entity="kk837",
                   group="inhand_isaaclab", name=os.path.basename(run_dir),
                   config={"task": args.task, "num_envs": env_cfg.scene.num_envs,
                           "decimation": env_cfg.decimation,
                           "episode_length_s": env_cfg.episode_length_s,
                           "success_threshold_rad": float(thr)},
                   dir=run_dir)

    env = gym.make(args.task, cfg=env_cfg)
    env = RlGamesVecEnvWrapper(env, agent_cfg["params"]["config"]["device"],
                               agent_cfg["params"]["env"]["clip_observations"],
                               agent_cfg["params"]["env"]["clip_actions"])
    vecenv.register("IsaacRlgWrapper",
                    lambda cfg_name, num_actors, **kw: RlGamesGpuEnv(cfg_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: env})

    runner = Runner()
    runner.load(agent_cfg)
    runner.run({"train": True, "play": False, "sigma": None,
                **({"checkpoint": args.checkpoint} if args.checkpoint else {})})
    env.close()


if __name__ == "__main__":
    main()
    app.close()
