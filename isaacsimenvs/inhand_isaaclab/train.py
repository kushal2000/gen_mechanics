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
parser.add_argument("--wandb-project", default="gen_mechanics_inhandreorient")
parser.add_argument("--wandb-group", default="inhand_isaaclab")
parser.add_argument("--gamma", type=float, default=0.0,
                    help="override the agent's discount; 0 keeps IsaacLab's 0.998. Ours is "
                         "0.99, which at 30 Hz is a 1.7 s horizon against their 10.5 s")
parser.add_argument("--mixed-precision", action="store_true",
                    help="rl_games mixed precision on, as our runs (theirs: off)")
parser.add_argument("--our-hparams", action="store_true",
                    help="swap in OUR PPO hyperparameters (lr 1e-4, 2 mini_epochs, horizon 16, "
                         "entropy 0, reward_shaper 0.01, bounds 1e-4); minibatch = half the "
                         "(augmented) rollout, the same 2-per-epoch our runs use")
parser.add_argument("--sapg", action="store_true",
                    help="run OUR SAPG settings instead of vanilla PPO (ladder rung: does "
                         "SAPG itself cost anything on a task known to learn?)")
parser.add_argument("--sapg-block-size", type=int, default=2048,
                    help="envs per exploration block; must divide --num-envs")
parser.add_argument("--viewer-every", type=int, default=200,
                    help="epochs between viewer pages; 0 disables the viewer")
parser.add_argument("--viewer-frames", type=int, default=400,
                    help="rolling frame buffer a viewer page is built from")
parser.add_argument("--viewer-stride", type=int, default=1,
                    help="capture one frame every N env steps")
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
from isaacsimenvs.inhand_isaaclab.observer import InHandObserver

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
    # A --run-dir whose basename does not start with "0_" reaches the fork as
    # int(name.split('_')[0]) and dies with `invalid literal for int(): 'obs'` from deep
    # inside the algo, minutes after Kit booted. Caught here, where it is legible.
    if not os.path.basename(run_dir).startswith("0_"):
        raise SystemExit(
            f"run dir basename must start with '0_' (the fork parses the policy index "
            f"from it): {os.path.basename(run_dir)!r}")

    agent_cfg["params"]["seed"] = args.seed
    cfg = agent_cfg["params"]["config"]
    cfg["max_epochs"] = args.max_epochs
    cfg["num_actors"] = env_cfg.scene.num_envs
    cfg["train_dir"] = run_dir
    cfg["full_experiment_name"] = os.path.basename(run_dir)
    cfg["name"] = os.path.basename(run_dir)
    if args.our_hparams:
        # The learner rung the SAPG rung did NOT cover: that one toggled SAPG but kept
        # IsaacLab's lr 5e-4 / 5 mini_epochs / horizon 24. Ours, from
        # coevolution/cfg/train/PoseReachSAPG*.yaml plus run_rank.sh's overrides:
        # lr 1e-4 adaptive (kl 0.016, same), 2 mini_epochs, horizon 16, entropy 0,
        # reward_shaper 0.01, bounds_loss 1e-4. Gradient work per sample
        # (mini_epochs x lr) is ~12x lower than theirs. Minibatch is set below, once the
        # augmented width is known, to 2 per epoch as ours runs (114688 of 229376).
        cfg["learning_rate"] = 1e-4
        cfg["mini_epochs"] = 2
        cfg["horizon_length"] = 16
        cfg["entropy_coef"] = 0.0
        cfg["bounds_loss_coef"] = 1e-4
        cfg.setdefault("reward_shaper", {})["scale_value"] = 0.01
        print("[train] hparams     OURS: lr 1e-4, mini_epochs 2, horizon 16, entropy 0, "
              "reward_shaper 0.01, bounds 1e-4")
    if args.mixed_precision:
        cfg["mixed_precision"] = True
        print("[train] mixed_precision on")
    if args.minibatch_size:
        cfg["minibatch_size"] = args.minibatch_size

    # OUR rl_games IS A SAPG FORK, and IsaacLab's stock agent config predates it. The
    # fork reads these keys unconditionally in places: a2c_common.py:329 defaults
    # expl_type to 'none', but :400 calls .startswith() on config.get('expl_type') with
    # no default, so an absent key is an AttributeError rather than plain PPO. Setting
    # them here rather than patching the fork keeps the reference's learner theirs in
    # every respect that matters, and records the requirement where it bites.
    if args.sapg:
        # The rung: the reference ENV and their network, driven by OUR learner. Values are
        # copied from coevolution/cfg/train/PoseReachSAPG*.yaml so this is the same SAPG the
        # in-hand jobs run, not an approximation of it.
        #
        # Nothing has to change on the env side. a2c_continuous.py:20 grows the network's
        # input by the embedding width on its own, and :34 supplies coef_ids from the ALGO,
        # so their stock actor_critic MLP takes 'extra_param' (the scalar block id becomes a
        # learned 32-d embedding, 72 -> 104) and 'coef_cond' sigma unchanged -- the fork
        # patched the stock A2CBuilder.Network in place.
        blocks, rem = divmod(cfg["num_actors"], args.sapg_block_size)
        if rem or blocks < 2:
            raise SystemExit(
                f"--sapg-block-size {args.sapg_block_size} must divide --num-envs "
                f"{cfg['num_actors']} into at least 2 blocks (got {blocks} remainder {rem})")
        cfg["expl_type"] = "mixed_expl_learn_param"
        cfg["expl_reward_type"] = "entropy"
        cfg["expl_reward_coef_scale"] = 0.002
        cfg["expl_reward_coef_embd_size"] = 32
        cfg["expl_coef_block_size"] = args.sapg_block_size
        cfg["use_others_experience"] = "lf"
        cfg["off_policy_ratio"] = 1.0
        print(f"[train] SAPG        {blocks} blocks of {args.sapg_block_size}, "
              f"use_others_experience=lf, fixed_sigma=coef_cond")
    else:
        cfg.setdefault("expl_type", "none")        # 'none' = vanilla PPO in this fork
        cfg.setdefault("use_others_experience", "none")
        cfg.setdefault("expl_coef_block_size", cfg["num_actors"])
        cfg.setdefault("expl_reward_coef_scale", 0.0)

    # And the fork's sigma is not stock either: it takes a STRING, one of 'fixed',
    # 'coef_cond' or 'obs_cond', where stock rl_games takes a bool. IsaacLab's True
    # matches none of them and falls through to the branch that builds sigma as a
    # Linear, after which const_initializer tries to .fill_() a module -- "'Linear'
    # object has no attribute 'fill_'".
    #
    # 'fixed' is the plain-PPO path and the right translation of their True. 'coef_cond',
    # which our own SAPG configs use, conditions sigma on the exploration coefficient and
    # is only legal WITH SAPG on: coef_ids is put in build_config by a2c_continuous.py:34
    # under expl_type mixed_expl, so asking for it while expl_type is 'none' is what raises
    # KeyError: 'coef_ids'. It is not an env-side requirement.
    space = agent_cfg["params"]["network"].get("space", {}).get("continuous")
    if space is not None:
        space["fixed_sigma"] = "coef_cond" if args.sapg else "fixed"
    # rl_games requires horizon * actors to be divisible by the minibatch. Their config
    # is written for their default env count, so a different --num-envs silently gives
    # the LAST minibatch a different width unless this is checked.
    rollout = cfg["horizon_length"] * cfg["num_actors"]
    # Under SAPG the dataset the minibatches are cut from is WIDER than the rollout.
    # augment_batch_for_mixed_expl repeats the batch num_repeat = min(blocks,
    # off_policy_ratio + 1) times (a2c_common.py:984) and filter_leader then keeps only the
    # leading block from each extra copy, so the length is rollout * (1 + (num_repeat-1)/
    # blocks) -- 5/4 at four blocks with off_policy_ratio 1.0. Checking the un-augmented
    # rollout instead would pass and then hand one ragged minibatch to PPODataset, which
    # gives the remainder to the LAST minibatch rather than failing.
    effective = rollout
    if args.sapg:
        blocks = cfg["num_actors"] // args.sapg_block_size
        num_repeat = min(blocks, int(cfg["off_policy_ratio"]) + 1)
        effective = rollout + (num_repeat - 1) * (rollout // blocks)
    if args.our_hparams and not args.minibatch_size:
        cfg["minibatch_size"] = effective // 2
    if effective % cfg["minibatch_size"]:
        what = "the SAPG-augmented rollout" if args.sapg else "horizon*actors"
        raise SystemExit(
            f"minibatch {cfg['minibatch_size']} does not divide {what} = {effective} "
            f"(horizon {cfg['horizon_length']} x actors {cfg['num_actors']}"
            f"{' x augmentation' if args.sapg else ''}). "
            f"Pass --minibatch-size with a divisor, e.g. {math.gcd(effective, 32768)}.")

    if args.gamma:
        # A learner rung, like --sapg: gamma is the cheapest test of the mechanism that
        # keeps the cube in the hand. Their 0.998 at 30 Hz values 316 steps (10.5 s) of an
        # 18.7 s episode; our 0.99 at 60 Hz values 99 steps (1.7 s). A drop forfeits the
        # rest of the episode in both envs -- this sets how much of that forfeit is
        # visible to the value function.
        print(f"[train] gamma       {cfg['gamma']} -> {args.gamma}")
        cfg["gamma"] = args.gamma

    print(f"[train] task        {args.task}")
    print(f"[train] envs        {env_cfg.scene.num_envs}  decimation {env_cfg.decimation}  "
          f"episode {env_cfg.episode_length_s} s")
    thr = env_cfg.commands.object_pose.orientation_success_threshold
    print(f"[train] threshold   {thr:.4f} rad ({math.degrees(thr):.2f} deg)")
    print(f"[train] minibatch   {cfg['minibatch_size']} of {effective} "
          f"({effective // cfg['minibatch_size']} per epoch)")
    print(f"[train] run dir     {run_dir}", flush=True)

    if args.wandb:
        import wandb

        # sync_tensorboard IS the logging. rl_games writes every scalar through its
        # own SummaryWriter and never calls wandb.log, so without this the run appears
        # in the project with config and system metrics and NO training charts -- which
        # is exactly what the first submission of this did. It works only because this
        # init runs before Runner builds the writer; see wandb_utils.WandbAlgoObserver,
        # which exists for the same reason on the coevolution path.
        wandb.init(project=args.wandb_project, entity="kk837",
                   sync_tensorboard=True,
                   group=args.wandb_group, name=os.path.basename(run_dir),
                   config={"task": args.task, "num_envs": env_cfg.scene.num_envs,
                           "decimation": env_cfg.decimation,
                           "episode_length_s": env_cfg.episode_length_s,
                           "success_threshold_rad": float(thr)},
                   dir=run_dir)

    env = gym.make(args.task, cfg=env_cfg)
    # Held before wrapping. The observer needs the ManagerBasedRLEnv, and digging it back
    # out of algo.vec_env means depending on rl_games' wrapper nesting, which is an
    # implementation detail; this is the same object either way.
    base_env = env.unwrapped
    # The object AS BUILT. The cfg's density is not authoritative -- the USD carries its own
    # mass and wins over it -- so a rung that changes the cube must be verified here rather
    # than read off the config. Turns "I set scale 0.75" into a measurement.
    try:
        # MEAN OVER ENVS, not env 0. The startup event object_scale_mass multiplies each
        # env's object mass by U(0.4, 1.6) (inhand_env_cfg.py EventCfg), so any single
        # env's mass says nothing about the nominal value -- reading env 0 is how "0.264
        # kg" got quoted for a cube whose nominal mass is something else. E[U] = 1, so the
        # mean recovers the nominal and the spread should span roughly 0.4x-1.6x of it.
        _ms = base_env.scene["object"].root_physx_view.get_masses().float()
        _m = float(_ms.mean().item())
        _lo, _hi = float(_ms.min().item()), float(_ms.max().item())
        _s = getattr(env_cfg.scene.object.spawn, "scale", None)
        _edge = 0.06 * (float(_s[0]) if _s else 1.0)
        print(f"[train] object      edge ~{_edge * 1000:.1f} mm  mass {_m:.5f} kg mean "
              f"[{_lo:.5f}, {_hi:.5f}]  nominal density {_m / max(_edge ** 3, 1e-12):.0f} "
              f"kg/m3  (spawn scale {_s})", flush=True)
    except Exception as _exc:
        print(f"[train] object      could not be measured: {_exc}", flush=True)
    env = RlGamesVecEnvWrapper(env, agent_cfg["params"]["config"]["device"],
                               agent_cfg["params"]["env"]["clip_observations"],
                               agent_cfg["params"]["env"]["clip_actions"])
    vecenv.register("IsaacRlgWrapper",
                    lambda cfg_name, num_actors, **kw: RlGamesGpuEnv(cfg_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: env})

    # Per-episode reward breakdowns, successes/episode and the viewer. Isaac Lab already
    # computes the first two into extras["log"]; the fork's DefaultAlgoObserver drops
    # them, so without this observer the run logs rewards/step and nothing else.
    def _log_html(key: str, html: str) -> None:
        if not args.wandb:
            return
        import wandb

        if wandb.run is not None:
            # Same call shape as coevolution/pose_viewer.py:578, which is the version
            # known to render in this project's wandb.
            wandb.log({key: wandb.Html(html)})

    observer = InHandObserver(base_env, viewer_every=args.viewer_every,
                              viewer_frames=args.viewer_frames,
                              viewer_stride=args.viewer_stride,
                              out_dir=run_dir, log_html=_log_html)

    runner = Runner(algo_observer=observer)
    runner.load(agent_cfg)
    runner.run({"train": True, "play": False, "sigma": None,
                **({"checkpoint": args.checkpoint} if args.checkpoint else {})})
    env.close()


if __name__ == "__main__":
    main()
    app.close()
