"""Play a checkpoint in one env and stream Isaac's render into viser.

Shared engine behind ``play_pose_reach.py`` and ``play_inhand.py``. Boots Kit,
builds ONE env (a few, if you want to watch several designs), restores a policy
and steps it, pushing the RTX frame into a viser page with controls for pausing,
resetting, resampling the goal and pinning the tolerance.

TWO ORDERING RULES, both load-bearing:

  1. Kit boots BEFORE viser is imported. viser drags in trimesh/embreex, and
     Kit segfaults at boot when those C extensions are already loaded -- which
     is why dextoolbench's eval_interactive_isaacsim.py pays for a whole
     subprocess. Booting first makes one process enough. Do not hoist the
     viser import to the top of this file.
  2. ``enable_cameras`` is set before AppLauncher. render_mode="rgb_array"
     silently yields None without it.

THE ENV CONFIG COMES FROM THE RUN, NOT FROM DEFAULTS. A checkpoint is only
meaningful against the env it trained in, and reconstructing that env from task
defaults has already gone wrong twice here -- once silently evaluating against
the random 1200-object pool instead of the curated 24, once with an obs layout
1067 wide against a policy built for 1045. So the run's saved
``.hydra/config.yaml`` is replayed onto the env config, and anything you pass on
the command line lands on top of that.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile
import time


def build_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--checkpoint", required=True, help="an rl_games .pth")
    p.add_argument("--run-dir", default="", help="the training run dir; inferred from --checkpoint if omitted")
    p.add_argument("--population", default="", help="override the run's robot_spec (e.g. one design to watch)")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--tolerance", type=float, default=0.0, help="pin the success tolerance; 0 = the run's own")
    p.add_argument("--expl-coef", type=float, default=0.0,
                   help="SAPG exploration coefficient. Training spans 50->0 across env blocks; "
                        "0 is the greedy end and the right one for watching a policy.")
    p.add_argument("--deterministic", action="store_true", help="use the mean action, not a sample")
    p.add_argument("--fps", type=float, default=30.0, help="cap on frames pushed per second")
    return p


def infer_run_dir(checkpoint: str) -> pathlib.Path:
    """The training run a checkpoint belongs to.

    Layout is ``<run>/rank_0/<name>/nn/<ckpt>.pth``, so the run is four levels
    up. Counting wrong lands on rank_0 and the config load fails confusingly.
    """
    return pathlib.Path(checkpoint).resolve().parents[3]


def launch_app(args):
    """Boot Kit. Nothing heavy may be imported before this returns."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    app_args, _ = parser.parse_known_args([])
    app_args.headless = True
    app_args.enable_cameras = True          # rgb_array yields None without it
    app_args.device = getattr(args, "device", "cuda:0")
    return AppLauncher(app_args).app


def agent_yaml_from_run(run_dir: pathlib.Path, population: str) -> str:
    """The run's own agent config, written out for RlPlayer."""
    import yaml
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(run_dir / "rank_0" / ".hydra" / "config.yaml")
    if population:
        cfg.env.assets.robot_spec = population
    agent = OmegaConf.to_container(cfg, resolve=True)["agent"]
    agent["params"]["config"]["multi_gpu"] = False
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump({"train": agent}, f)
    f.close()
    return f.name


def env_cfg_from_run(env_cfg, run_dir: pathlib.Path, args):
    """Replay the run's env config, then apply the command line on top."""
    from omegaconf import OmegaConf

    saved = OmegaConf.load(run_dir / "rank_0" / ".hydra" / "config.yaml")
    env_cfg.from_dict(OmegaConf.to_container(saved.env, resolve=True))
    env_cfg.scene.num_envs = args.num_envs
    if args.population:
        env_cfg.assets.robot_spec = args.population
    if args.tolerance > 0:
        # eval_success_tolerance PINS it. Merely setting the current value lets
        # the curriculum tighten it out from under a long session.
        env_cfg.termination.eval_success_tolerance = args.tolerance
    return env_cfg


def play(task: str, args) -> None:
    """Build the env, restore the policy, serve the render in viser."""
    app = launch_app(args)                                   # rule 1: Kit first

    import gymnasium as gym
    import numpy as np
    import torch

    import isaacsimenvs  # noqa: F401  registers the tasks
    from coevolution.eval.rl_player import RlPlayer
    from coevolution.utils.hydra_utils import hydra_task_config_with_yaml

    run_dir = pathlib.Path(args.run_dir) if args.run_dir else infer_run_dir(args.checkpoint)
    if not (run_dir / "rank_0" / ".hydra" / "config.yaml").is_file():
        raise SystemExit(f"no saved config under {run_dir}; pass --run-dir explicitly")
    print(f"[play] task {task}\n[play] run  {run_dir}\n[play] ckpt {args.checkpoint}", flush=True)

    @hydra_task_config_with_yaml(task, "rl_games_joint_transformer_cfg_entry_point")
    def _main(env_cfg, agent_cfg):
        env_cfg = env_cfg_from_run(env_cfg, run_dir, args)
        env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")
        inner = env.unwrapped
        obs, _ = env.reset()

        player = RlPlayer(
            obs["policy"].shape[-1], inner.action_space.shape[-1],
            agent_yaml_from_run(run_dir, args.population), args.checkpoint,
            device=env_cfg.sim.device, sapg_expl_coef=args.expl_coef,
            num_envs=inner.num_envs)

        import viser                                          # rule 1: after Kit
        server = viser.ViserServer(port=args.port)
        print(f"\n[play] viser on http://localhost:{args.port}\n", flush=True)

        state = {"running": True, "step": 0, "reset": False, "resample": False}
        with server.gui.add_folder("playback"):
            b_pause = server.gui.add_button("pause / resume")
            b_step = server.gui.add_button("single step")
            b_reset = server.gui.add_button("reset episode")
            b_goal = server.gui.add_button("resample goal")
            g_det = server.gui.add_checkbox("deterministic action", args.deterministic)
        with server.gui.add_folder("status"):
            md = server.gui.add_markdown("")

        @b_pause.on_click
        def _(_) -> None:
            state["running"] = not state["running"]

        @b_step.on_click
        def _(_) -> None:
            state["running"], state["step"] = False, state["step"] + 1

        @b_reset.on_click
        def _(_) -> None:
            state["reset"] = True

        @b_goal.on_click
        def _(_) -> None:
            state["resample"] = True

        period = 1.0 / max(args.fps, 1e-3)
        t_last = 0.0
        try:
            while app.is_running():
                if state["reset"]:
                    obs, _ = env.reset()
                    state["reset"] = False
                if state["resample"]:
                    from isaacsimenvs.pose_reaching_6d.reset_utils.reset import reset_goal_trackers

                    reset_goal_trackers(inner, torch.arange(inner.num_envs, device=inner.device))
                    state["resample"] = False

                if state["running"] or state["step"] > 0:
                    with torch.no_grad():
                        act = player.get_normalized_action(
                            obs["policy"], deterministic_actions=bool(g_det.value))
                    obs, _, _, _, _ = env.step(act)
                    state["step"] = max(0, state["step"] - 1)
                else:
                    inner.sim.render()    # keep the viewport live while paused

                now = time.time()
                if now - t_last >= period:
                    frame = env.render()
                    if frame is not None:
                        server.scene.set_background_image(np.asarray(frame, dtype=np.uint8))
                    md.content = _status(inner)
                    t_last = now
        finally:
            env.close()

    _main()
    import os

    sys.stdout.flush()
    os._exit(0)      # Kit does not tear down cleanly


def _status(inner) -> str:
    """The numbers worth reading while watching, for env 0."""
    def one(name, fmt="{:.4f}"):
        v = getattr(inner, name, None)
        if v is None:
            return "-"
        try:
            return fmt.format(float(v[0]) if hasattr(v, "__len__") else float(v))
        except Exception:
            return "-"
    return (
        f"**tolerance** {one('_current_success_tolerance')}  \n"
        f"**keypoint dist** {one('_keypoints_max_dist')} m  \n"
        f"**best so far** {one('_closest_keypoint_max_dist')} m  \n"
        f"**near goal** {one('_near_goal', '{:.0f}')}  |  "
        f"**successes** {one('_successes', '{:.0f}')}  \n"
        f"**lifted** {one('_lifted_object', '{:.0f}')}  |  "
        f"**step** {one('episode_length_buf', '{:.0f}')}"
    )
