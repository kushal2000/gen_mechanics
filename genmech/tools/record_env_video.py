"""Record an mp4 of the env with NO policy, to look at the scene geometry.

play_video.py needs a trained checkpoint, which is the wrong tool for "is the
arm in the right place relative to the table". This drives zero actions and just
films the scene, so it works before a single gradient step exists and on any
embodiment.

It exists because a placement bug got as far as a training run: authoring robot
prims with spawn=None meant Isaac Lab never applied the spawner's translation,
and the arm is FIXED-BASE, so init_state could not move it afterwards. The robot
sat at the env origin, on top of the table instead of 0.8 m behind it. Numbers
caught it in the end, but it was visible immediately in the wandb viewer, and a
20-second clip at scene-build time would have caught it before the run.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.record_env_video --author --steps 200
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parents[1] / "videos"


def _parse_args():
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--author", action="store_true",
                   help="authored robot prims (spawn=None) instead of converted")
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--population_count", type=int, default=4)
    p.add_argument("--population_seed", type=int, default=2)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--single", action="store_true",
                   help="single-embodiment env (sharpa) as a reference clip")
    p.add_argument("--out", default=None)
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    a.headless = True
    a.enable_cameras = True
    return a


def main() -> None:
    args = _parse_args()
    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app

    import gymnasium as gym
    import torch

    import genmech.tasks  # noqa: F401

    tag = "single" if args.single else ("authored" if args.author else "converted")
    out = Path(args.out) if args.out else VIDEO_DIR / f"env_{tag}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.single:
        from genmech.tasks.pose_reach.env_cfg import PoseReachEnvCfg

        cfg = PoseReachEnvCfg()
        cfg.assets.robot_spec = "sharpa_iiwa14"
        task = "GenMech-PoseReach-Direct-v0"
    else:
        from genmech.tasks.pose_reach.env_multi_cfg import PoseReachMultiEnvCfg

        cfg = PoseReachMultiEnvCfg()
        cfg.assets.robot_population_count = args.population_count
        cfg.assets.robot_population_seed = args.population_seed
        cfg.assets.author_robot_usds = bool(args.author)
        task = "GenMech-PoseReachMulti-Direct-v0"
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = 2

    # render_mode="rgb_array" makes DirectRLEnv build a tiled camera lazily;
    # RecordVideo then pulls a frame per step.
    env = gym.make(task, cfg=cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env, video_folder=str(out.parent), name_prefix=out.stem,
        step_trigger=lambda s: s == 0, video_length=args.steps,
        disable_logger=True)

    inner = env.unwrapped
    env.reset()
    origins = inner.scene.env_origins
    print(f"[video:{tag}] robot base = "
          f"{(inner.robot.data.root_pos_w - origins)[0].tolist()}")
    print(f"[video:{tag}] table      = "
          f"{(inner.table.data.root_pos_w - origins)[0].tolist()}")

    # Zero actions: the point is the scene, not the behaviour. Joint targets are
    # randomised at reset, so the arm still moves enough to read its pose.
    act = torch.zeros((inner.num_envs, inner.cfg.action_space), device=inner.device)
    for _ in range(args.steps):
        env.step(act)
    env.close()

    made = sorted(out.parent.glob(f"{out.stem}*.mp4"))
    print(f"[video:{tag}] wrote {[str(m) for m in made]}")

    del app
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
