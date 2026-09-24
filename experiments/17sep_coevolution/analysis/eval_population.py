"""Score a population under a trained policy, without training anything.

The policy is restored through ``coevolution.eval.RlPlayer``, whose rl_games
player runs in eval mode: no optimiser, and -- the reason this exists rather
than a training job with the learning rate set to zero -- no running_mean_std
or value normaliser updates either. Those are buffer updates, not optimiser
steps, so a "frozen" training run is not frozen: its input normalisation drifts
as it sees the new design's observations.

Returns are banked by ``coevolution.design_rewards.DesignRewardWrapper``, the
same code selection ranked on, so the numbers are comparable with a generation's
own table. Actions are SAMPLED by default, because selection ranked designs on
stochastic training rollouts; --deterministic scores the mean action instead.

    .venv_isaacsim/bin/python experiments/17sep_coevolution/analysis/eval_population.py \
        --headless --meta <...meta.json> --steps 4800 [--deterministic]
"""
import argparse, json, os, pathlib, sys, tempfile, time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="GenMech-PoseReach-Direct-v0")
parser.add_argument("--agent", default="rl_games_cfg_entry_point")
parser.add_argument("--meta", required=True, help="a prep_gen_sharpa_eval meta.json")
parser.add_argument("--steps", type=int, default=4800, help="env steps (300 epochs x 16)")
parser.add_argument("--deterministic", action="store_true")
parser.add_argument("--rank", type=int, default=0, help="which half of the object deal to run")
parser.add_argument("--world_size", type=int, default=2)
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
META = json.load(open(args.meta))
# design_cycle deals object (global_env // n_designs) % pool, so ONE rank of
# 12288 envs only ever reaches objects 0..11. Training ran two ranks and dealt
# all 24; setting RANK here makes this process the same slice of that deal, and
# merging the two tables reproduces it exactly.
import os as _os
_os.environ["RANK"] = str(args.rank); _os.environ["WORLD_SIZE"] = str(args.world_size)
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym, torch, yaml
from omegaconf import OmegaConf
import isaacsimenvs  # noqa: F401
from coevolution.utils.hydra_utils import hydra_task_config_with_yaml
from coevolution.eval.rl_player import RlPlayer
from coevolution.design_rewards import DesignRewardWrapper


def player_config(run_dir: str, population: str) -> str:
    """The training run's agent config, as the ``{'train': ...}`` yaml RlPlayer
    reads, with robot_spec pointed at the population being scored."""
    cfg = OmegaConf.load(pathlib.Path(run_dir) / "rank_0" / ".hydra" / "config.yaml")
    cfg.env.assets.robot_spec = population
    agent = OmegaConf.to_container(cfg, resolve=True)["agent"]
    agent["params"]["config"]["multi_gpu"] = False
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump({"train": agent}, f); f.close()
    return f.name


@hydra_task_config_with_yaml(args.task, args.agent)
def main(env_cfg, agent_cfg):
    env_cfg.sim.device = args.device
    env_cfg.assets.robot_spec = os.path.abspath(META["population"])
    # PIN, not resume: resume_success_tolerance only sets the starting value and
    # the curriculum is free to tighten it during the evaluation.
    env_cfg.termination.eval_success_tolerance = META["tolerance"]
    out = pathlib.Path(args.out or (pathlib.Path(args.meta).with_suffix("").with_suffix("")
                                    .as_posix() + ".scores.json"))
    env = gym.make(args.task, cfg=env_cfg)
    table = out.with_suffix(f".rank{args.rank}.table.json")
    banked = DesignRewardWrapper(env, output_path=table, rank=args.rank, flush_every=500)
    obs, _ = banked.reset()
    # the run the checkpoint came from: four levels up (run/rank_0/<name>/nn/x.pth),
    # so take the recorded path rather than counting
    run_dir = META.get("run_dir") or str(pathlib.Path(META["checkpoint"]).parents[3])
    player = RlPlayer(obs["policy"].shape[-1], env.unwrapped.action_space.shape[-1],
                      player_config(run_dir, os.path.abspath(META["population"])),
                      META["checkpoint"], device=args.device, sapg_expl_coef=50.0,
                      num_envs=env.unwrapped.num_envs)
    print(f"[eval] rank {args.rank} of {args.world_size}: {env.unwrapped.num_envs} envs, tolerance "
          f"{float(env.unwrapped._current_success_tolerance):.4f}, "
          f"{'deterministic' if args.deterministic else 'sampled'} actions, {args.steps} steps", flush=True)
    t0 = time.time()
    for t in range(args.steps):
        with torch.no_grad():
            act = player.get_normalized_action(obs["policy"], deterministic_actions=args.deterministic)
        obs, _, _, _, _ = banked.step(act)
        if (t + 1) % 1000 == 0:
            print(f"[eval] {t+1}/{args.steps} steps, {banked.count.sum().item():.0f} episodes, "
                  f"{time.time()-t0:.0f} s", flush=True)
    banked.flush()
    json.dump({**META, "steps": args.steps, "deterministic": args.deterministic,
               "rank": args.rank, "world_size": args.world_size, "table": str(table)},
              open(out.with_suffix(f".rank{args.rank}.json"), "w"), indent=1)
    print(f"[eval] wrote {table}", flush=True)


main()
sys.stdout.flush(); os._exit(0)
