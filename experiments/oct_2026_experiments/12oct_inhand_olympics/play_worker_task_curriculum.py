"""viser_play.py's worker for task_curriculum's Play-family runs (Kuka/Flexiv + SHARPA, Flexiv + Wuji, human-like or
not): the run's env built as task_curriculum's own viewer builds it (isaacsimenvs/eval/viser_play.py there: the
run's saved config, one env per object -- 12, one per size distribution -- the eval protocol's randomization off),
its policy stepped, and the watched env streamed back in viser_play's protocol. task_curriculum is only read.

Runs under task_curriculum's venv with that repo first on the path (its package is also called isaacsimenvs), so
nothing from gen_mechanics is imported here; stdlib only at module level, as Kit must boot first.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path

TC = Path("/share/portal/kk837/task_curriculum")
FRAME_PERIOD_S = 1.0 / 30.0


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tolerance", type=float, default=0.0, help="success tolerance (m); 0 = the curriculum floor")
    p.add_argument("--tolerance-note", default="")
    p.add_argument("--host", required=True)
    p.add_argument("--wport", type=int, required=True)
    p.add_argument("--authkey", required=True)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def _pose7(data, e, origins):
    return [*(data.root_pos_w[e] - origins[e]).tolist(), *data.root_quat_w[e].tolist()]


def main() -> None:
    args = _args()
    sys.path.insert(0, str(TC))
    os.chdir(TC)
    conn = Client((args.host, args.wport), authkey=bytes.fromhex(args.authkey))
    try:
        _work(args, conn)
    except Exception:
        import traceback
        try:
            conn.send({"kind": "error", "text": traceback.format_exc()})
        except Exception:
            pass
    os._exit(0)


def _work(args, conn) -> None:
    from isaaclab.app import AppLauncher
    ap = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(ap)
    a = ap.parse_args([])
    a.headless, a.device = True, args.device
    app = AppLauncher(a).app  # noqa: F841

    import tempfile

    import gymnasium as gym
    import numpy as np
    import torch
    import yaml

    import isaacsimenvs  # noqa: F401  (task_curriculum's) gym.register side effects
    from isaacsimenvs.eval.player import PretrainedPlayer
    from isaacsimenvs.eval.protocol import disable_randomization
    from isaacsimenvs.eval.viser_play import SAPG_MAX_COEF, Run, object_type
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    run = Run(Path(args.run_dir))
    if run.task is None:
        raise RuntimeError(f"{run.name}: no task for robot {run.robot!r}")
    conn.send({"kind": "status", "text": f"building {run.task}"})
    sys.argv = [sys.argv[0]]

    @hydra_task_config_with_yaml(run.task, "")
    def _run(env_cfg, agent_cfg) -> None:
        if run.config_is_own:
            saved = dict(run.config["env"])
            for k in ("observation_space", "state_space", "action_space", "num_observations", "num_states",
                      "num_actions"):
                saved.pop(k, None)
            env_cfg.from_dict(saved)
        from isaacsimenvs.tasks.play.utils.object_size_distributions import OBJECT_SIZE_DISTRIBUTIONS
        env_cfg.assets.num_assets_per_type = 1
        types_cfg = set(env_cfg.assets.handle_head_types)
        env_cfg.scene.num_envs = sum(d.type in types_cfg for d in OBJECT_SIZE_DISTRIBUTIONS)
        env_cfg.seed = 0
        floor = float(env_cfg.termination.target_success_tolerance)
        env_cfg.termination.eval_success_tolerance = args.tolerance or floor
        disable_randomization(env_cfg)

        env = gym.make(run.task, cfg=env_cfg)
        inner = env.unwrapped
        n = inner.num_envs
        n_obs, n_act = int(inner.cfg.observation_space), int(inner.cfg.action_space)
        origins = inner.scene.env_origins
        urdfs = [str(Path(p).resolve()) for p in inner._object_urdf_paths]
        asset_of_env = inner._object_asset_index_per_env.cpu().numpy().tolist()
        type_of_env = [object_type(urdfs[a]) for a in asset_of_env]
        types = sorted(set(type_of_env))

        def make_player(ckpt: str):
            agent = run.config["agent"]
            agent["params"]["config"]["multi_gpu"] = False
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                yaml.safe_dump({"train": agent}, f)
            blocks = run.num_blocks()
            coef = float(np.linspace(SAPG_MAX_COEF, 0.0, blocks)[-1]) if blocks > 1 else 0.0   # the leader, 0
            return PretrainedPlayer(f.name, Path(ckpt), num_envs=n, device=args.device, num_observations=n_obs,
                                    num_actions=n_act, expl_coef=coef)

        player = make_player(args.checkpoint)
        rs = inner.cfg.reset
        start_xyz = (0.0, 0.0, float(rs.table_reset_z) + float(rs.table_object_z_offset))
        conn.send({"kind": "ready", "controls": "play",
                   "robots": [{"urdf": str((TC / inner.cfg.assets.robot_urdf).resolve()),
                               "joints": list(inner.robot.joint_names)}],
                   "table": {"urdf": str((TC / inner.cfg.assets.table_urdf).resolve())},
                   "objects": {"urdfs": urdfs},
                   "env_names": [f"{type_of_env[e]} · {Path(urdfs[asset_of_env[e]]).stem[:40]}" for e in range(n)],
                   "num_envs": n, "tolerance": float(inner.cfg.termination.eval_success_tolerance),
                   "tolerance_note": args.tolerance_note or f"the curriculum floor ({floor * 100:.1f} cm)",
                   "start": {"random": rs.fixed_start_pose is None},
                   "summary": f"{run.task} · {n} envs, one object each ({', '.join(types)}) · eval protocol "
                              f"(randomization off)"})

        running, pending, det, watch, speed = True, 0, False, 0, 1.0
        stats = {t: [] for t in types}
        ep_goals = np.zeros(n, dtype=np.int64)
        obs, _ = env.reset()
        t_next, t_frame, steps, last_end = time.time(), 0.0, 0, ""
        while True:
            while conn.poll():
                m = conn.recv()
                c = m["cmd"]
                if c == "quit":
                    return
                if c == "toggle":
                    running = not running
                elif c == "step":
                    running, pending = False, pending + 1
                elif c == "det":
                    det = bool(m["value"])
                elif c == "watch":
                    watch = max(0, min(n - 1, int(m["value"])))
                elif c == "speed":
                    speed = float(m["value"])
                elif c == "tolerance":
                    inner.cfg.termination.eval_success_tolerance = float(m["value"])
                elif c in ("reset", "ckpt", "init"):
                    if c == "ckpt":
                        conn.send({"kind": "status", "text": "loading the checkpoint"})
                        player = make_player(m["path"])
                        if m.get("tolerance"):
                            inner.cfg.termination.eval_success_tolerance = float(m["tolerance"])
                    if c == "init":
                        if m.get("random", True):
                            rs.fixed_start_pose = None
                        else:
                            from scipy.spatial.transform import Rotation as Rot
                            x, y, z, w = Rot.from_euler("xyz", m["rpy"], degrees=True).as_quat()
                            rs.fixed_start_pose = (*start_xyz, float(w), float(x), float(y), float(z))
                    obs, _ = env.reset()
                    player.reset_rnn()
                    ep_goals[:] = 0
                    for v in stats.values():
                        v.clear()
                    steps, last_end = 0, ""
            if running or pending:
                pending = max(0, pending - 1)
                with torch.no_grad():
                    action = player.get_action(obs["policy"], deterministic=det)
                obs, _, term, trunc, _ = env.step(action)
                ep_goals += inner._is_success.cpu().numpy().astype(np.int64)
                ended = term | trunc
                if bool(ended.any()):
                    ids = ended.nonzero(as_tuple=True)[0]
                    player.reset_rnn(ids)
                    reasons = getattr(inner, "_termination_reasons", {}) or {}
                    for e in ids.tolist():
                        stats[type_of_env[e]].append(int(ep_goals[e]))
                        if e == watch:
                            why = next((r for r, v in reasons.items() if bool(v[e])), "ended")
                            last_end = f"{why} after {int(ep_goals[e])} goals"
                        ep_goals[e] = 0
                steps += 1
                if speed > 0:
                    t_next += float(inner.step_dt) / speed
                    time.sleep(max(0.0, t_next - time.time()))
                    t_next = max(t_next, time.time() - 0.1)
                else:
                    t_next = time.time()
            else:
                time.sleep(0.02)
                t_next = time.time()
            now = time.time()
            if now - t_frame < FRAME_PERIOD_S:
                continue
            t_frame = now
            e = watch
            rows = ["| object type | finished episodes | goals / episode | ≥ 1 goal |", "|---|---|---|---|"]
            for t, g in stats.items():
                rows.append(f"| {t} | {len(g)} | {sum(g) / len(g):.1f} | {100 * sum(x >= 1 for x in g) / len(g):.0f}% |"
                            if g else f"| {t} | 0 | – | – |")
            allg = [x for g in stats.values() for x in g]
            if allg:
                rows.append(f"| **all** | {len(allg)} | {sum(allg) / len(allg):.1f} | "
                            f"{100 * sum(x >= 1 for x in allg) / len(allg):.0f}% |")
            tol = float(inner.cfg.termination.eval_success_tolerance)
            conn.send({"kind": "frame", "robot": 0, "q": inner.robot.data.joint_pos[e].tolist(),
                       "robot_pose": _pose7(inner.robot.data, e, origins),
                       "table_pose": _pose7(inner.table.data, e, origins),
                       "object": asset_of_env[e], "object_pose": _pose7(inner.object.data, e, origins),
                       "goal_pose": _pose7(inner.goal_viz.data, e, origins), "push": None,
                       "status": (f"env {e} ({type_of_env[e]}) · goals this episode **{int(ep_goals[e])}** · "
                                  f"tolerance {tol * 100:.1f} cm"
                                  + (f"  \nlast episode: {last_end}" if last_end else "")
                                  + f"  \n{'running' if running else 'paused'} · {steps} steps"),
                       "stats": "\n".join(rows), "steps": steps, "running": running})

    _run()


if __name__ == "__main__":
    main()
