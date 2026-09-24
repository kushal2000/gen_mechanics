"""Kit-side worker for the interactive players. Stdlib-only at module level.

Boots Isaac Sim, builds the env, restores a policy, steps it, and streams joint
positions and object poses back to the viser parent over a localhost
Connection. It renders NOTHING: the parent draws the hand itself from these
numbers with ViserUrdf, which needs no Kit.

WHY A SEPARATE PROCESS. viser and Kit cannot share one interpreter. Kit
prepends a ``pip_prebundle`` carrying an old ``websockets`` and installs import
machinery that wins over sys.path order, the package's own ``__path__`` and an
explicit spec pin -- all three were tried, and ``websockets.http11`` still
resolved inside the Isaac Sim tree, so ``import viser`` died every time. Keep
this module's top-level imports stdlib-only so nothing heavy is loaded before
Kit boots.

Not launched directly; ``viser_play.py`` spawns it.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from multiprocessing.connection import Client


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--population", required=True, help="already narrowed by the parent")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--expl-coef", type=float, default=0.0)
    p.add_argument("--tolerance", type=float, default=0.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--authkey", required=True)
    return p.parse_known_args()


def main() -> None:
    args, hydra_args = _args()
    # The parent passes the key as hex; it creates its Listener with the RAW
    # bytes, so decode rather than encode or the digests never match.
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

        import gymnasium as gym
        import torch
        from omegaconf import OmegaConf

        import isaacsimenvs  # noqa: F401
        from coevolution.eval.rl_player import RlPlayer
        from coevolution.utils.hydra_utils import hydra_task_config_with_yaml

        run_dir = args.run_dir
        sys.argv = [sys.argv[0]] + list(hydra_args)

        @hydra_task_config_with_yaml(args.task, "rl_games_joint_transformer_cfg_entry_point")
        def _run(env_cfg, agent_cfg):
            saved = OmegaConf.load(f"{run_dir}/rank_0/.hydra/config.yaml")
            env_d = OmegaConf.to_container(saved.env, resolve=True)
            # Derived at scene build, and stored as STRINGS in the saved config.
            for k in ("observation_space", "state_space", "action_space"):
                env_d.pop(k, None)
            env_cfg.from_dict(env_d)
            env_cfg.scene.num_envs = args.num_envs
            env_cfg.assets.robot_spec = args.population
            if args.tolerance > 0:
                env_cfg.termination.eval_success_tolerance = args.tolerance

            env = gym.make(args.task, cfg=env_cfg)
            inner = env.unwrapped
            obs, _ = env.reset()

            import tempfile

            import yaml

            acfg = OmegaConf.to_container(saved, resolve=True)["agent"]
            acfg["params"]["config"]["multi_gpu"] = False
            f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
            yaml.safe_dump({"train": acfg}, f)
            f.close()
            player = RlPlayer(obs["policy"].shape[-1], inner.action_space.shape[-1], f.name,
                              args.checkpoint, device=args.device,
                              sapg_expl_coef=args.expl_coef, num_envs=inner.num_envs)

            spec = inner.scene_record.robot_spec
            # The real object per env, not a stand-in cube: each env draws one
            # of the curated pool, and which one is per-env bookkeeping only the
            # scene_record knows.
            from coevolution.pose_viewer import object_urdf_for_env

            obj_urdfs = []
            for e in range(int(inner.num_envs)):
                try:
                    obj_urdfs.append(str(object_urdf_for_env(inner, e)[1]))
                except Exception:
                    obj_urdfs.append(None)
            send(kind="ready",
                 joint_names=list(spec.joint_names_canonical),
                 num_envs=int(inner.num_envs),
                 num_arm_joints=int(spec.num_arm_joints),
                 object_urdfs=obj_urdfs,
                 base_pos=[float(v) for v in spec.base_pos],
                 base_rot=[float(v) for v in spec.base_rot])

            running, pending, det = True, 0, False
            while app.is_running():
                while conn.poll():
                    m = conn.recv()
                    c = m.get("cmd")
                    if c == "quit":
                        env.close()
                        return
                    if c == "pause":
                        running = False
                    elif c == "resume":
                        running = True
                    elif c == "step":
                        running, pending = False, pending + 1
                    elif c == "deterministic":
                        det = bool(m.get("value"))
                    elif c == "reset":
                        obs, _ = env.reset()
                    elif c == "resample":
                        from isaacsimenvs.pose_reaching_6d.reset_utils.reset import (
                            reset_goal_trackers,
                        )

                        reset_goal_trackers(
                            inner, torch.arange(inner.num_envs, device=inner.device))

                if running or pending:
                    with torch.no_grad():
                        act = player.get_normalized_action(obs["policy"], deterministic_actions=det)
                    obs, _, _, _, _ = env.step(act)
                    pending = max(0, pending - 1)

                origins = inner.scene.env_origins
                send(kind="frame",
                     joint_pos=inner.robot.data.joint_pos.detach().cpu().numpy().tolist(),
                     object_pos=(inner.object.data.root_pos_w - origins).cpu().numpy().tolist(),
                     object_quat=inner.object.data.root_quat_w.cpu().numpy().tolist(),
                     goal_pos=(inner.goal_viz.data.root_pos_w - origins).cpu().numpy().tolist(),
                     goal_quat=inner.goal_viz.data.root_quat_w.cpu().numpy().tolist(),
                     table_pos=((inner.table.data.root_pos_w - origins).cpu().numpy().tolist()
                                if getattr(inner, "table", None) is not None else None),
                     table_quat=(inner.table.data.root_quat_w.cpu().numpy().tolist()
                                 if getattr(inner, "table", None) is not None else None),
                     stats=_stats(inner))
            env.close()

        _run()
    except Exception:
        send(kind="error", text=traceback.format_exc())
    finally:
        try:
            conn.close()
        except Exception:
            pass
        import os

        os._exit(0)          # Kit does not tear down cleanly


def _stats(inner) -> dict:
    def g(name):
        v = getattr(inner, name, None)
        if v is None:
            return None
        try:
            return [float(x) for x in (v if hasattr(v, "__len__") else [v])]
        except Exception:
            return None
    return {k: g(k) for k in ("_current_success_tolerance", "_keypoints_max_dist",
                              "_closest_keypoint_max_dist", "_near_goal", "_successes",
                              "_lifted_object", "episode_length_buf")}


if __name__ == "__main__":
    main()
