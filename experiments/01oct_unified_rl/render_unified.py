"""Film the unified policy on every hand of its scene: one clip per hand, tiled into one grid video.

One env per hand (multi:uniform deals env i to hand i mod 8), the run's own env config, the checkpoint
driven greedily (exploration block 0). Each hand is filmed separately with the camera on its palm centre,
so every tile is the same shot of a different hand; the tiles are captioned with the hand and the goals it
reaches in the clip, then laid out in a grid.

    GENMECH_KEEP_VISUALS=1 OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \
        experiments/01oct_unified_rl/render_unified.py --headless --enable_cameras \
        --run_dir <run dir> --checkpoint <.pth> --out debug_outputs/01oct_unified_rl/videos/<tag> [--frame_only]
"""
import argparse
import os
import sys
import tempfile
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="GenMech-InHandReorient-Direct-v0")
parser.add_argument("--run_dir", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--eye", default="0.22,-0.30,0.20", help="camera offset from the palm centre (world)")
parser.add_argument("--look", default="0.0,-0.03,0.04", help="look-at offset from the palm centre (world)")
parser.add_argument("--res", default="640,480")
parser.add_argument("--cols", type=int, default=4)
parser.add_argument("--out", required=True)
parser.add_argument("--frame_only", action="store_true", help="one still per hand + the grid, for framing")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]
os.environ.setdefault("GENMECH_KEEP_VISUALS", "1")      # the robot's visual meshes, stripped in training
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import isaacsimenvs  # noqa: E402,F401
from coevolution.eval.rl_player import RlPlayer  # noqa: E402
from coevolution.utils.hydra_utils import hydra_task_config_with_yaml  # noqa: E402

EYE = np.array([float(v) for v in args.eye.split(",")])
LOOK = np.array([float(v) for v in args.look.split(",")])
W, H = (int(v) for v in args.res.split(","))
NAMES = {"sharpa": "SHARPA", "allegro": "Allegro", "leap": "LEAP", "shadow": "Shadow", "dex3": "Dex3",
         "tesollo": "Tesollo", "wuji2": "Wuji v2", "xhand": "XHAND"}


def caption(frame: np.ndarray, title: str, sub: str) -> np.ndarray:
    im = Image.fromarray(frame[..., :3])
    d = ImageDraw.Draw(im)
    try:
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        small = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        big = small = ImageFont.load_default()
    d.rectangle([0, 0, W, 66], fill=(20, 20, 20))
    d.text((14, 6), title, font=big, fill=(255, 255, 255))
    d.text((14, 40), sub, font=small, fill=(200, 200, 200))
    return np.asarray(im)


def grid(tiles: list[np.ndarray]) -> np.ndarray:
    rows = [np.concatenate(tiles[i:i + args.cols], axis=1) for i in range(0, len(tiles), args.cols)]
    return np.concatenate(rows, axis=0)


@hydra_task_config_with_yaml(args.task, "rl_games_joint_transformer_cfg_entry_point")
def main(env_cfg, agent_cfg):
    saved = OmegaConf.load(os.path.join(args.run_dir, "rank_0", ".hydra", "config.yaml"))
    env_d = OmegaConf.to_container(saved.env, resolve=True)
    for k in ("observation_space", "state_space", "action_space"):
        env_d.pop(k, None)
    env_cfg.from_dict(env_d)
    hands = env_cfg.assets.robot_spec
    env_cfg.scene.num_envs = 8
    env_cfg.seed = 0
    env_cfg.termination.max_consecutive_successes = 0
    env_cfg.viewer.resolution = (W, H)
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array").unwrapped
    obs, _ = env.reset()

    acfg = OmegaConf.to_container(saved, resolve=True)["agent"]
    acfg["params"]["config"]["multi_gpu"] = False
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump({"train": acfg}, f)
    f.close()
    player = RlPlayer(obs["policy"].shape[-1], env.action_space.shape[-1], f.name, args.checkpoint,
                      device=env.device, sapg_expl_coef=0.0, num_envs=env.num_envs)

    hs = env.scene_record.hand_set
    hand_of_env = env.scene_record.robot_design_index.tolist()
    epoch = os.path.basename(args.checkpoint).split("_ep_")[-1].split("_")[0] if "_ep_" in args.checkpoint else "?"
    steps = 1 if args.frame_only else int(round(args.seconds / float(env.step_dt)))
    stride = max(1, int(round(1.0 / (args.fps * float(env.step_dt)))))
    os.makedirs(args.out, exist_ok=True)
    print(f"[render] {hands}: {env.num_envs} envs, {steps} steps, every {stride}th frame, epoch {epoch}", flush=True)

    clips = {}
    for h, spec in enumerate(hs.specs):
        e = hand_of_env.index(h)
        name = NAMES.get(spec.hand_name, spec.hand_name)
        obs, _ = env.reset()
        goals, frames, t0 = 0, [], time.time()
        for t in range(steps):
            with torch.no_grad():
                act = player.get_normalized_action(obs["policy"], deterministic_actions=True)
            obs, *_ = env.step(act)
            goals += int(env._is_success[e])
            if t % stride == 0 or args.frame_only:
                c = env._palm_center_pos_w[e].cpu().numpy()
                env.sim.set_camera_view(eye=(c + EYE).tolist(), target=(c + LOOK).tolist())
                frames.append(caption(env.render(), name, f"{spec.num_hand_joints} joints   goals: {goals}"))
        clips[name] = frames
        if args.frame_only:
            imageio.imwrite(os.path.join(args.out, f"{spec.hand_name}.png"), frames[-1])
        else:
            imageio.mimwrite(os.path.join(args.out, f"{spec.hand_name}.mp4"), frames, fps=args.fps,
                             quality=8, macro_block_size=None)
        print(f"[render] {name}: {goals} goals in {args.seconds:.0f} s ({time.time() - t0:.0f} s wall)", flush=True)

    n = min(len(v) for v in clips.values())
    tiles = list(clips.values())
    if args.frame_only:
        imageio.imwrite(os.path.join(args.out, "grid.png"), grid([v[-1] for v in tiles]))
    else:
        path = os.path.join(args.out, "grid.mp4")
        imageio.mimwrite(path, [grid([v[i] for v in tiles]) for i in range(n)], fps=args.fps, quality=8,
                         macro_block_size=None)
        print(f"[render] wrote {path}", flush=True)


main()
sys.stdout.flush()
os._exit(0)
