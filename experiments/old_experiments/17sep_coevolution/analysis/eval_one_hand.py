"""One hand, every checkpoint, one scene build.

The quick signal: how does a single design score under each checkpoint of a
co-evolution run? The scene is built ONCE (one design, a few hundred envs) and
each checkpoint is restored into the same player, so the whole curve costs one
build plus a few seconds per checkpoint -- minutes, not hours. It cannot say
where the design would RANK (that needs the cohort scored under the same
policy, which is eval_population.py); it says whether the policy can drive it
at all, and whether that improves over the run.


KNOWN DISCREPANCY -- READ BEFORE QUOTING AN ABSOLUTE NUMBER FROM THIS.

Scored against training's own record, this harness reads about 37 % low: the
generation-16 population of coevolution_v2_gen2k averages 2443 here and 1528
under RlPlayer, and per-design Spearman between the two is only 0.474. What
that was NOT (each checked, 2026-09-23):

  cold start            training's own rollout, restarted from the same
                        checkpoint with the learning rate at zero, climbs to
                        2305 by epoch 544; this harness plateaus near 1824 over
                        a longer horizon, so a warm env population is not it
  exploration coef      50 (most exploratory) and 0 (greedy) give 1780 and
                        1623 -- the SAPG block conditioning barely moves it
  sampled vs mean       deterministic actions give 1780 against 1796 sampled
  design subsample      every-8th of 1024 is unbiased to +0.4 %
  episode window        real and large early (0.45 goals/ep at 1200 steps,
                        1.35 at 14400) but converged by ~14000 steps
  design vs episode     weighting the mean either way differs by 1 %
  weighting
  object pool           WAS a real bug: without the overrides below the env
                        falls back to the random 1200-object pool, not the
                        curated 24 the policy trained on
  tolerance drift       WAS a real bug: the curriculum tightens every 3000 env
                        steps, so the tolerance has to be PINNED, not merely
                        initialised -- see eval_success_tolerance below
  env config            the hydra override lists are identical

So the difference is somewhere between RlPlayer and rl_games' training rollout
and has not been found. Every design in a given run is measured identically, so
comparisons WITHIN a run are sound and were the basis for the gen-SHARPA
result; absolute returns and fine-grained rankings are not.

    .venv_isaacsim/bin/python experiments/old_experiments/17sep_coevolution/analysis/eval_one_hand.py --headless \
        --population assets/populations/sharpa_capsule.json --label coevolution_v2_gen2k --steps 1200
"""
import argparse, glob, json, os, pathlib, sys, tempfile, time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="GenMech-PoseReach-Direct-v0")
parser.add_argument("--agent", default="rl_games_cfg_entry_point")
parser.add_argument("--population", required=True, help="a one-design population .json")
parser.add_argument("--label", default="coevolution_v2_gen2k", help="which arm's checkpoints")
parser.add_argument("--gens", default="", help="comma-separated; default every generation with a checkpoint")
parser.add_argument("--steps", type=int, default=1200, help="env steps per checkpoint, AFTER warmup")
parser.add_argument("--warmup", type=int, default=0,
                    help="steps to run before banking anything. A fresh scene resets every env at "
                         "once, so at first only SHORT episodes have finished -- successful ones "
                         "chain goals and each goal restarts the 600-step clock, so they are still "
                         "in flight. Measuring from step 0 therefore reads far below steady state: "
                         "training's own rollout from a cold start reports 607 where the same "
                         "policy's steady state is 2416.")
parser.add_argument("--window", type=int, default=0,
                    help="if set, report each window of this many steps separately, so the "
                         "approach to steady state is visible in one run")
parser.add_argument("--tolerance", default="own", help="'own' (each generation's) or a number")
parser.add_argument("--expl_coef", type=float, default=50.0,
                    help="the SAPG exploration coefficient fed to the policy. Training spans "
                         "50 -> 0 across env blocks, so one fixed value is never exactly what "
                         "the training curve averaged; 50 is the most exploratory.")
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym, torch, yaml
from omegaconf import OmegaConf
import isaacsimenvs  # noqa: F401
from coevolution.utils.hydra_utils import hydra_task_config_with_yaml
from coevolution.eval.rl_player import RlPlayer
from coevolution.design_rewards import DesignRewardWrapper

P = pathlib.Path("assets/populations") / args.label
gens = [int(g) for g in args.gens.split(",") if g] or sorted(
    int(p.name.split("_")[1]) for p in P.glob("gen_*") if (p / "checkpoint.txt").exists())
POP = os.path.abspath(args.population)


def run_dir(gen: int) -> str:
    """The training run a generation's checkpoint came out of. Read from the
    generation's own run_dir.txt rather than counted back from the checkpoint
    path: the checkpoint sits four levels down (run/rank_0/<name>/nn/x.pth)."""
    return (P / f"gen_{gen}/run_dir.txt").read_text().strip()


def player_config(run_dir: str) -> str:
    cfg = OmegaConf.load(pathlib.Path(run_dir) / "rank_0" / ".hydra" / "config.yaml")
    cfg.env.assets.robot_spec = POP
    agent = OmegaConf.to_container(cfg, resolve=True)["agent"]
    agent["params"]["config"]["multi_gpu"] = False
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump({"train": agent}, f); f.close()
    return f.name


@hydra_task_config_with_yaml(args.task, args.agent)
def main(env_cfg, agent_cfg):
    env_cfg.sim.device = args.device
    env_cfg.assets.robot_spec = POP
    env = gym.make(args.task, cfg=env_cfg)
    banked = DesignRewardWrapper(env, output_path="/dev/null", rank=0, flush_every=10**9)
    obs, _ = banked.reset()
    inner = env.unwrapped
    first = (P / f"gen_{gens[0]}/checkpoint.txt").read_text().strip()
    player = RlPlayer(obs["policy"].shape[-1], inner.action_space.shape[-1],
                      player_config(run_dir(gens[0])), first,
                      device=args.device, sapg_expl_coef=args.expl_coef, num_envs=inner.num_envs)
    rows = []
    print(f"[one_hand] {pathlib.Path(POP).name} in {inner.num_envs} envs, {len(gens)} checkpoints, "
          f"{args.steps} steps each, exploration coefficient {args.expl_coef:g}\n  gen |     tol |  return | goals/ep | success | lift rate | episodes", flush=True)
    for g in gens:
        d = P / f"gen_{g}"
        player.player.restore(d.joinpath("checkpoint.txt").read_text().strip())
        tol = float((d / "success_tolerance.txt").read_text()) if args.tolerance == "own" else float(args.tolerance)
        # PIN it: update_tolerance_curriculum runs every step and tightens the
        # tolerance once _frame_counter - _last_curriculum_update >= 3000 env
        # steps. That counter does not reset between checkpoints, so simply
        # assigning _current_success_tolerance lets the curriculum move it out
        # from under the evaluation after the first few thousand steps.
        # eval_success_tolerance is the documented pin and is re-applied every
        # step by the curriculum itself.
        inner.cfg.termination.eval_success_tolerance = tol
        inner._current_success_tolerance = tol
        for t in (banked.sum, banked.count, banked.goals_sum, banked.succeeded, banked.running):
            t.zero_()
        obs, _ = banked.reset()
        # what the policy's action distribution looks like, to compare against
        # training: a sigma far from what training used would explain a gap.
        with torch.no_grad():
            a_det = player.get_normalized_action(obs["policy"], deterministic_actions=True)
            a_smp = player.get_normalized_action(obs["policy"], deterministic_actions=False)
        print(f"      actions: |mu| mean {a_det.abs().mean():.3f}, sampled-minus-mu sd "
              f"{(a_smp - a_det).std():.3f}", flush=True)
        lifted = 0.0; n = 0
        t0 = time.time()

        def act_and_step(o):
            with torch.no_grad():
                a = player.get_normalized_action(o["policy"], deterministic_actions=args.deterministic)
            return banked.step(a)[0]

        for _ in range(args.warmup):
            obs = act_and_step(obs)
        if args.warmup:
            for t_ in (banked.sum, banked.count, banked.goals_sum, banked.succeeded):
                t_.zero_()
            print(f"      warmup {args.warmup} steps ({time.time()-t0:.0f} s)", flush=True)
        for step in range(args.steps):
            obs = act_and_step(obs)
            lifted += float(inner._lifted_object.float().mean()); n += 1
            if args.window and (step + 1) % args.window == 0:
                e = float(banked.count.sum())
                print(f"      to step {step+1}: return {float(banked.sum.sum())/max(e,1):7.0f}  "
                      f"goals/ep {float(banked.goals_sum.sum())/max(e,1):5.2f}  ({int(e)} episodes)", flush=True)
                for t_ in (banked.sum, banked.count, banked.goals_sum, banked.succeeded):
                    t_.zero_()
        assert abs(float(inner._current_success_tolerance) - tol) < 1e-9, \
            f"tolerance drifted to {float(inner._current_success_tolerance)} from {tol}"
        ep = float(banked.count.sum()); ret = float(banked.sum.sum()) / max(ep, 1)
        goals = float(banked.goals_sum.sum()) / max(ep, 1); succ = float(banked.succeeded.sum()) / max(ep, 1)
        rows.append(dict(gen=g, tolerance=tol, ret=ret, goals=goals, success=succ, lift=lifted / n, episodes=int(ep)))
        print(f"  {g:3d} | {tol:7.4f} | {ret:7.0f} | {goals:8.2f} | {succ:7.2f} | {lifted/n:9.2f} | {int(ep):8d}"
              f"   ({time.time()-t0:.0f} s)", flush=True)
    suffix = "" if args.expl_coef == 50.0 else f"_coef{args.expl_coef:g}"
    out = pathlib.Path(args.out or f"debug_outputs/17sep_coevo_analysis/one_hand_{args.label}_"
                                   f"{pathlib.Path(POP).stem}{suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rows, indent=1))
    print(f"[one_hand] wrote {out}", flush=True)


main()
sys.stdout.flush(); os._exit(0)
