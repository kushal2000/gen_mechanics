"""Film a trained policy driving one design on several objects.

Builds a small scene with the design in every env and the curated pool dealt
one object per env (design_cycle with one design: env e holds object e), loads
an rl_games checkpoint through RlPlayer, and records one clip per chosen env
with the camera on that env. Frames come from DirectRLEnv.render() at the
viewer resolution; the camera is moved with sim.set_camera_view.

    .venv_isaacsim/bin/python experiments/10sep_coevolution/analysis/render_policy.py --headless --enable_cameras \
        --run_dir <run dir with rank_0/.hydra/config.yaml> --checkpoint <.pth> --population <one-design .json> \
        --objects 1,6,9,11,15,21 --steps 600 --out debug_outputs/10sep_coevo_analysis/videos/<tag> [hydra overrides...]
"""
import argparse, os, sys, tempfile, time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="GenMech-PoseReach-Direct-v0")
parser.add_argument("--agent", default="rl_games_cfg_entry_point")
parser.add_argument("--run_dir", required=True); parser.add_argument("--checkpoint", required=True)
parser.add_argument("--population", required=True, help="a one-design population .json")
parser.add_argument("--objects", default="1,6,9,11,15,21", help="curated-pool indices = env ids to film")
parser.add_argument("--steps", type=int, default=600); parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--tolerance", type=float, default=0.01)
parser.add_argument("--eye", default="0.9,-1.1,1.15"); parser.add_argument("--lookat", default="0.0,0.25,0.6")
parser.add_argument("--out", required=True); parser.add_argument("--frame_only", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym, numpy as np, torch, yaml, imageio
from omegaconf import OmegaConf
import isaacsimenvs  # noqa
from coevolution.utils.hydra_utils import hydra_task_config_with_yaml
from coevolution.eval.rl_player import RlPlayer
from isaacsimenvs.pose_reaching_6d.scene_utils.objects.curated_pools import CURATED_POOLS

OBJ = [int(v) for v in args.objects.split(",")]
EYE = np.array([float(v) for v in args.eye.split(",")]); LOOK = np.array([float(v) for v in args.lookat.split(",")])
POOL = CURATED_POOLS["diverse24"]


def player_config(run_dir: str, population: str) -> str:
    """The run's agent config as the {'train': ...} yaml RlPlayer reads, with
    robot_spec pointed at the one-design file (the layout is the same 5x6 envelope)."""
    cfg = OmegaConf.load(os.path.join(run_dir, "rank_0", ".hydra", "config.yaml"))
    cfg.env.assets.robot_spec = population
    agent = OmegaConf.to_container(cfg, resolve=True)["agent"]
    agent["params"]["config"]["multi_gpu"] = False
    p = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False); yaml.safe_dump({"train": agent}, p); p.close()
    return p.name


@hydra_task_config_with_yaml(args.task, args.agent)
def main(env_cfg, agent_cfg):
    env_cfg.sim.device = args.device
    env_cfg.viewer.resolution = (1280, 960)
    env_cfg.termination.resume_success_tolerance = args.tolerance
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array").unwrapped
    obs, _ = env.reset()
    n_obs = obs["policy"].shape[-1]; n_act = env.action_space.shape[-1]
    player = RlPlayer(n_obs, n_act, player_config(args.run_dir, args.population), args.checkpoint,
                      device=args.device, sapg_expl_coef=50.0, num_envs=env.num_envs)
    origins = env.scene.env_origins.cpu().numpy()
    os.makedirs(args.out, exist_ok=True)
    print(f"[render] {env.num_envs} envs, obs {n_obs}, act {n_act}, tolerance {float(env._current_success_tolerance):.3f}", flush=True)
    for k in OBJ:
        typ, handle, head, *_ = POOL[k]
        name = f"{k:02d}_{typ}_{'box' if len(handle) == 3 else 'cyl'}"
        o = origins[k]; env.sim.set_camera_view(eye=(o + EYE).tolist(), target=(o + LOOK).tolist())
        obs, _ = env.reset(); frames = []; goals0 = int(env._successes[k]); t0 = time.time()
        for t in range(args.steps):
            with torch.no_grad():
                act = player.get_normalized_action(obs["policy"], deterministic_actions=True)
            obs, rew, term, trunc, info = env.step(act)
            if not args.frame_only or t == args.steps - 1:
                frames.append(env.render())
        goals = int(env._successes[k]) - goals0
        if args.frame_only:
            imageio.imwrite(os.path.join(args.out, name + ".png"), frames[-1]); print(f"[render] wrote {name}.png", flush=True); continue
        path = os.path.join(args.out, name + ".mp4")
        imageio.mimwrite(path, frames, fps=args.fps, quality=8, macro_block_size=None)
        print(f"[render] {name}: {len(frames)} frames, env {k} reached {goals} goals in the clip ({time.time()-t0:.0f} s)", flush=True)


main()
sys.stdout.flush(); os._exit(0)
