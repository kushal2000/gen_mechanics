"""Capture a rollout of the IsaacLab in-hand reference into a viewer page.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python -m isaacsimenvs.inhand_isaaclab.capture_viewer \
        --task GenMech-InHandIsaacLab-step0-reference-v0 --frames 600 \
        --checkpoint debug_outputs/train_logs/inhand_isaaclab/<run>/.../nn/<ckpt>.pth

With no checkpoint the policy is random, which is still worth looking at: it shows the
reset pose, the cube's start placement and the goal marker, which is most of what goes
wrong before a policy is involved.

Writes a self-contained-ish HTML (the hand's URDF and meshes are fetched from master)
under debug_outputs/inhand_isaaclab_viewers/.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", default="GenMech-InHandIsaacLab-step0-reference-v0")
parser.add_argument("--frames", type=int, default=600, help="policy steps to record")
parser.add_argument("--checkpoint", default="", help="rl_games .pth; random policy if absent")
parser.add_argument("--out", default="", help="output .html; defaults under debug_outputs")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
if not getattr(args, "headless", None):
    args.headless = True
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

from isaacsimenvs.inhand_isaaclab import register
from isaacsimenvs.inhand_isaaclab.viewer import build_html, capture_frame

register()

REPO = "/share/portal/kk837/gen_mechanics"


def _policy(env, checkpoint: str, agent_cfg):
    """A callable ``obs -> action``, restored from a checkpoint or random.

    Built through rl_games' own player so the network is exactly the one that trained,
    rather than a reimplementation of it that could differ in normalisation.
    """
    if not checkpoint:
        n = env.action_space.shape[-1]
        dev = env.unwrapped.device
        print("[capture] no checkpoint: random policy", flush=True)
        return lambda obs: torch.zeros((env.unwrapped.num_envs, n), device=dev).uniform_(-1, 1)

    from rl_games.torch_runner import Runner

    runner = Runner()
    runner.load(agent_cfg)
    player = runner.create_player()
    player.restore(checkpoint)
    player.has_batch_dimension = True
    print(f"[capture] restored {os.path.basename(checkpoint)}", flush=True)

    def act(obs):
        with torch.no_grad():
            return player.get_action(obs["policy"] if isinstance(obs, dict) else obs,
                                     is_deterministic=True)

    return act


def main() -> None:
    env_cfg = parse_env_cfg(args.task, num_envs=1)
    agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")
    env = gym.make(args.task, cfg=env_cfg)
    obs, _ = env.reset()
    act = _policy(env, args.checkpoint, agent_cfg)

    frames = []
    for i in range(args.frames):
        frames.append(capture_frame(env, 0))
        obs, _, _, _, _ = env.step(act(obs))
    print(f"[capture] {len(frames)} frames", flush=True)

    out = args.out or os.path.join(
        REPO, "debug_outputs/inhand_isaaclab_viewers",
        f"{args.task.replace('GenMech-InHandIsaacLab-', '').replace('-v0', '')}_"
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_html(frames))
    print(f"[capture] wrote {out} ({os.path.getsize(out) / 1024:.0f} KiB)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
