r"""Frozen-policy evaluation of an rl_games checkpoint on a grammar
population (plan revision, step 1: "Add a fixed-horizon evaluation of a
checkpoint on a population, including projected commercial hands").

Loads the checkpoint deterministically (no learning, no exploration noise:
``player.get_action(obs, is_deterministic=True)`` returns the policy's mean
action) and runs ``--episodes-per-design`` (default 8) episodes for every
design in ``--population``, envs split evenly across designs
(``hand_sampler.robot_spec.design_index``, the same assignment
``author_grammar.setup_grammar_robot`` uses), at a FIXED, frozen success
tolerance (``--eval-success-tolerance``, default 0.4) -- both curricula
(tolerance and goal difficulty) are frozen for the whole eval, since a
"frozen-policy" measurement should not itself keep progressing a
curriculum while it runs.

"Check first whether coevolution/train.py has a test or play mode you can
reuse": yes, ``--test`` (``rl_games.torch_runner.Runner.run_play``) -- but
it drives an OPEN-ENDED ``player.run()`` loop (``games_num`` "games", no
notion of "K episodes per DESIGN", no tolerance override) with no output
this task needs. This script reuses the SAME setup train.py's ``--test``
path does -- ``Runner``, ``register_rlgames_env``, checkpoint restore via
``player.restore()`` (which also restores the checkpoint's own saved
curriculum state via ``env.set_env_state``, immediately overridden here to
the fixed eval tolerance) -- but drives its OWN step loop instead of
``player.run()``, mirroring ``BasePlayer.run()``'s RNN-state-reset-on-done
logic exactly (rl_games' vendored source, ``rl_games/common/player.py``).

Per-design graded scoring comes for free from Part C: ``design_scoring.py``
banks every episode inside the env's own ``_get_rewards``/``_get_dones``
hooks on every ``env.step()`` call, regardless of who is driving it -- this
script just drives the env with the loaded policy's actions and reads the
accumulated per-design tallies off with ``design_scoring.read_snapshot``.

Run (single Kit process; ``timeout -k 30 <cap>`` per this branch's Kit-run
rule; ``env.assets.hand_population=<file>`` works exactly as train.py's own
Hydra CLI overrides do, but --population is a plain argparse flag here,
matching --checkpoint, since a fixed-horizon eval script has no notion of
"training config overrides"):

    OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_CACHE_PATH=/tmp/$USER/ov_cache \
    WANDB_MODE=disabled timeout -k 30 600 \
    .venv_isaacsim/bin/python isaacsimenvs/inhand_reorient/evaluate_population.py \
        --checkpoint <run_dir>/nn/<name>.pth \
        --population outputs/grammar_populations/test16.json \
        --num-envs 128 --episodes-per-design 8 \
        --out <run_dir>/eval_scores.json
"""

from __future__ import annotations

import argparse
import math
import os
import sys


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="GenMech-InHandReorient-Direct-v0")
    parser.add_argument(
        "--agent", default="rl_games_sapg_cfg_entry_point",
        help="Key in gym.register kwargs for the rl_games YAML -- must match the "
             "network architecture the checkpoint was trained with.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the .pth to evaluate")
    parser.add_argument("--population", required=True, help="Path to a grammar population JSON")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--episodes-per-design", type=int, default=8)
    parser.add_argument("--eval-success-tolerance", type=float, default=0.4)
    parser.add_argument(
        "--max-extra-passes", type=int, default=3,
        help="Extra rollout passes (each episodes-per-design * max_episode_length steps) if some "
             "design still hasn't reached episodes-per-design episodes after the first pass.",
    )
    parser.add_argument("--out", default=None, help="eval_scores.json path; defaults next to --checkpoint")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--sim_device", default="cuda:0")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    app = AppLauncher(args_cli).app

    # 2. Safe to import isaaclab-backed modules now.
    import json
    from pathlib import Path

    import gymnasium as gym

    import isaacsimenvs  # noqa: F401  registers GenMech-InHandReorient-Direct-v0
    import coevolution.networks  # noqa: F401  registers the joint_transformer net
    import isaacsimenvs.inhand_reorient.design_scoring as design_scoring
    import isaacsimenvs.inhand_reorient.env as inhand_env_module
    from coevolution.utils.hydra_utils import hydra_task_config_with_yaml
    from coevolution.utils.rlgames_utils import register_rlgames_env
    from rl_games.torch_runner import Runner

    # Freeze BOTH curricula for the whole eval: a frozen-policy measurement
    # should not itself keep advancing a curriculum while it runs. Applied
    # by replacing the name in env.py's OWN module namespace -- env.py's
    # `_get_dones` resolves `update_tolerance_curriculum`/
    # `update_goal_curriculum` via its module globals at CALL time, so this
    # takes effect without editing env.py or pose_reaching_6d (out of
    # scope for this branch).
    inhand_env_module.update_tolerance_curriculum = lambda env: None
    inhand_env_module.update_goal_curriculum = lambda env: None

    @hydra_task_config_with_yaml(args_cli.task, args_cli.agent)
    def run(env_cfg, agent_cfg: dict) -> None:
        env_cfg.sim.device = args_cli.sim_device
        env_cfg.assets.hand_population = args_cli.population
        env_cfg.scene.num_envs = args_cli.num_envs

        env = gym.make(args_cli.task, cfg=env_cfg)
        inner_env = env.unwrapped

        clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
        clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
        register_rlgames_env(env, rl_device=args_cli.rl_device, clip_obs=clip_obs, clip_actions=clip_actions)

        runner = Runner()
        runner.load(agent_cfg)
        runner.reset()
        player = runner.create_player()
        player.restore(args_cli.checkpoint)  # may also restore the checkpoint's own env_state (curriculum)
        player.model.eval()

        # Fixed eval tolerance, applied AFTER restore (which may have
        # loaded the checkpoint's OWN trained-to tolerance via env_state) --
        # everything after this point runs with update_tolerance_curriculum
        # no-op'd above, so nothing changes it again.
        inner_env._current_success_tolerance = float(args_cli.eval_success_tolerance)

        n_designs = int(getattr(inner_env, "hand_tables", None).n_designs) if getattr(inner_env, "hand_tables", None) is not None else 1
        target_episodes = int(args_cli.episodes_per_design)
        max_episode_steps = int(inner_env.max_episode_length)
        print(f"[evaluate_population] {n_designs} designs, {inner_env.num_envs} envs, "
              f"target {target_episodes} episodes/design, eval_success_tolerance="
              f"{args_cli.eval_success_tolerance}, checkpoint={args_cli.checkpoint}", flush=True)

        obs = player.env_reset(player.env)
        if player.is_rnn:
            player.init_rnn()

        max_passes = 1 + int(args_cli.max_extra_passes)
        for pass_i in range(max_passes):
            for _ in range(target_episodes * max_episode_steps):
                action = player.get_action(obs, is_deterministic=True)
                obs, _reward, done, _info = player.env_step(player.env, action)
                if player.is_rnn:
                    all_done_indices = done.nonzero(as_tuple=False)
                    for s in player.states:
                        s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

            episodes = inner_env._score_episodes.detach().cpu()
            short = int((episodes < target_episodes).sum())
            if short == 0:
                break
            print(f"[evaluate_population] pass {pass_i + 1}/{max_passes}: {short}/{n_designs} designs "
                  f"still short of {target_episodes} episodes; running another pass", flush=True)
        else:
            print(f"[evaluate_population] WARNING: {short}/{n_designs} designs never reached "
                  f"{target_episodes} episodes after {max_passes} passes; reporting what was achieved", flush=True)

        snapshot = design_scoring.read_snapshot(inner_env)
        out_path = Path(args_cli.out) if args_cli.out else Path(args_cli.checkpoint).with_name("eval_scores.json")
        payload = {
            "checkpoint": str(args_cli.checkpoint),
            "population": str(args_cli.population),
            "num_envs": int(inner_env.num_envs),
            "target_episodes_per_design": target_episodes,
            "eval_success_tolerance": float(args_cli.eval_success_tolerance),
            "n_designs": n_designs,
            "designs": snapshot["designs"],
            "graded_fitness_formula": snapshot["graded_fitness_formula"],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(out_path)
        print(f"[evaluate_population] wrote {out_path}", flush=True)
        for d_str, row in sorted(snapshot["designs"].items(), key=lambda kv: int(kv[0])):
            print(f"  design {d_str} ({row['source']}): {row['episodes']} episodes, "
                  f"graded_fitness={row['graded_fitness']:.3f}, goals/ep={row['goals_per_episode']:.2f}, "
                  f"success_rate={row['success_rate']:.2f}", flush=True)

    run()

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
