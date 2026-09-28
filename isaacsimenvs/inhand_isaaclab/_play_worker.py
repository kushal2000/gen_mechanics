"""Kit-side worker for the IsaacLab in-hand reference player. Stdlib-only at module level.

Boots Isaac Sim, builds a rung of the ladder, optionally restores an rl_games policy,
steps it, and streams joint angles and poses to the viser parent over a localhost
Connection. It renders NOTHING: the parent draws the hand itself with ViserUrdf.

WHY A SEPARATE PROCESS, and why this file is a sibling of ours rather than a branch in
it. viser and Kit cannot share one interpreter -- Kit's pip_prebundle carries an old
``websockets`` and wins the import regardless of sys.path order, the package's own
``__path__``, or an explicit spec pin. That is the same reason
``coevolution/eval/_play_worker.py`` exists. This is a separate worker because that one
speaks our env: it resolves a RobotSpec, reads ``_keypoints_max_dist``, permutes joints
into canonical order and banks per-design state, none of which a manager-based env has.
Bending it would put two envs' quirks in one file and risk the player people are
currently using to debug live runs.

Keep this module's top-level imports stdlib-only so nothing heavy loads before Kit.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from multiprocessing.connection import Client

FRAME_PERIOD_S = 1.0 / 30.0      # viser cannot use more than this


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--authkey", required=True)
    return p.parse_known_args()


def main() -> None:
    args, _rest = _args()
    # The parent creates its Listener with the RAW bytes, so decode rather than encode.
    conn = Client((args.host, args.port), authkey=bytes.fromhex(args.authkey))

    def send(**kw):
        try:
            conn.send(kw)
        except Exception:
            pass

    try:
        from isaaclab.app import AppLauncher

        ap = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(ap)
        app_args, _ = ap.parse_known_args([])
        app_args.headless = True
        app_args.device = args.device
        app = AppLauncher(app_args).app
        send(kind="status", text="Kit booted; building the scene")

        import pathlib
        import tempfile

        import gymnasium as gym
        import torch

        from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

        from isaacsimenvs.inhand_isaaclab import register
        from isaacsimenvs.inhand_isaaclab.viewer import (
            DEX_CUBE_EDGE_M,
            ROBOT_URDF_RELPATH,
            capture_frame,
            object_urdf_text,
        )

        register()
        env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped
        obs, _ = env.reset()

        # The cube the parent should draw, written where it can read it. Generated from
        # the task's own object generator at the size measured off their stage, so the
        # faces match every other viewer in this repo.
        cube = pathlib.Path(tempfile.mkdtemp(prefix="isaaclab_play_")) / "cube.urdf"
        cube.write_text(object_urdf_text(DEX_CUBE_EDGE_M), encoding="utf-8")

        policy = None
        if args.checkpoint:
            from rl_games.torch_runner import Runner

            agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")
            # The same three dialect translations train.py applies; see its comments.
            cfg = agent_cfg["params"]["config"]
            cfg["num_actors"] = inner.num_envs
            cfg.setdefault("expl_type", "none")
            cfg.setdefault("use_others_experience", "none")
            cfg.setdefault("expl_coef_block_size", inner.num_envs)
            cfg.setdefault("expl_reward_coef_scale", 0.0)
            space = agent_cfg["params"]["network"].get("space", {}).get("continuous")
            if space is not None and not isinstance(space.get("fixed_sigma"), str):
                space["fixed_sigma"] = "fixed"
            runner = Runner()
            runner.load(agent_cfg)
            policy = runner.create_player()
            policy.restore(args.checkpoint)
            policy.has_batch_dimension = True
            send(kind="status", text=f"restored {pathlib.Path(args.checkpoint).name}")

        n_act = env.action_space.shape[-1]
        robot = inner.scene["robot"]
        lim = robot.data.joint_pos_limits[0]
        send(kind="ready",
             # CANONICAL here means the robot's OWN order, which is what the parent zips
             # against. Their order is breadth-first (all _joint_0, then all _joint_1),
             # not the finger-major order our URDF declares -- so names must travel with
             # the angles, never be assumed.
             joint_names=list(robot.data.joint_names),
             joint_lower=[float(v) for v in lim[:, 0]],
             joint_upper=[float(v) for v in lim[:, 1]],
             robot_urdf=str(pathlib.Path("/share/portal/kk837/gen_mechanics") / ROBOT_URDF_RELPATH),
             cube_urdf=str(cube),
             num_envs=int(inner.num_envs))

        running, pending, manual = True, 0, None
        last_frame = [0.0]
        while app.is_running():
            while conn.poll():
                m = conn.recv()
                c = m.get("cmd")
                if c == "quit":
                    env.close()
                    return
                if c == "toggle":
                    running = not running
                    send(kind="running", value=bool(running))
                elif c == "step":
                    running, pending = False, pending + 1
                elif c == "reset":
                    obs, _ = env.reset()
                elif c == "joints":
                    vals = m.get("values")
                    manual = None if not vals else torch.tensor(
                        vals, device=inner.device, dtype=torch.float32
                    ).unsqueeze(0).expand(inner.num_envs, -1).clamp(lim[:, 0], lim[:, 1])
                    if manual is not None:
                        running = True
                        send(kind="running", value=True)

            if manual is not None:
                # Position targets, written straight to the articulation. Their env drives
                # joints through an action term, so bypassing it is the only way to hold a
                # pose; physics keeps running so the cube still reacts.
                robot.set_joint_position_target(manual)
                robot.write_data_to_sim()
                inner.sim.step(render=False)
                inner.scene.update(inner.physics_dt)
            elif running or pending:
                if policy is None:
                    a = torch.zeros((inner.num_envs, n_act), device=inner.device).uniform_(-1, 1)
                else:
                    with torch.no_grad():
                        o = obs["policy"] if isinstance(obs, dict) else obs
                        a = policy.get_action(o, is_deterministic=True)
                obs, _, _, _, _ = env.step(a)
                pending = max(0, pending - 1)

            now = time.time()
            if now - last_frame[0] < FRAME_PERIOD_S:
                if not running and not pending and manual is None:
                    time.sleep(0.005)
                continue
            last_frame[0] = now

            f = capture_frame(env, 0)
            send(kind="frame", **f)
    except Exception:
        send(kind="error", text=traceback.format_exc())
    finally:
        try:
            conn.close()
        except Exception:
            pass
        import os

        os._exit(0)          # Kit does not tear down cleanly


if __name__ == "__main__":
    main()
