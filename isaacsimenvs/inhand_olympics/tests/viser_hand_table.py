"""Viser test of the in-hand olympics scene: the Wuji v2 left hand on its 6-DOF dummy wrist, palm down above the
table, with the pen (a capsule, its tip red) lying on it and its current goal drawn as a ghost. Every joint gets a slider spanning its limits in sim, grouped in collapsible folders (wrist, then one per
finger); a slider sets where that joint should go, the env's own action pipeline steers it there (fingers:
absolute targets through the EMA; wrist: speed-limited deltas through the EMA), and viser draws the simulated
joint positions. The status line shows the worst target-tracking error.

Kit and viser cannot share an interpreter, so the sim runs in a child worker (as in
experiments/01oct_unified_rl/viser_compare.py) and streams joint positions back.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python isaacsimenvs/inhand_olympics/tests/viser_hand_table.py \\
        --port 8090                     # then open http://<node>:8090
    ... viser_hand_table.py --smoke     # no viewer: the pen settles on the table, joints track random targets,
                                        # the palm faces down
"""
from __future__ import annotations

import argparse
import os
import pathlib
import queue
import secrets
import socket
import subprocess
import sys
import time
from multiprocessing.connection import Client, Listener

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--hand", default="wuji2_left_uniform_handonly")
    ap.add_argument("--smoke", action="store_true", help="headless check, no viewer")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--host", default="", help=argparse.SUPPRESS)
    ap.add_argument("--wport", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--authkey", default="", help=argparse.SUPPRESS)
    return ap.parse_args()


def _boot():
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    return AppLauncher(a).app


def _make_env(hand: str):
    import isaacsimenvs.inhand_olympics  # noqa: F401
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg
    cfg = InHandOlympicsEnvCfg()
    cfg.hand.robot_spec = hand
    cfg.scene.num_envs = 1
    cfg.goal.seconds_per_goal = 1e6       # no timeouts while someone is driving the sliders
    cfg.task.names = ("pen_spin", "pen_lift")
    env = InHandOlympicsEnv(cfg)
    env.reset()
    return env


def _palm_normal_world(env):
    """The palm normal in world, from the root body's orientation and the hand's palm frame."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    from isaacsimenvs.inhand_olympics.wrist import palm_frame
    body = env.robot.body_names.index(env.robot_spec.palm_body_name)
    w, x, y, z = env.robot.data.body_quat_w[0, body].tolist()
    return Rotation.from_quat([x, y, z, w]).as_matrix() @ palm_frame(env.robot_spec)[:, 0]


def smoke(args):
    app = _boot()
    import torch

    from isaacsimenvs.inhand_olympics.wrist import WRIST_JOINTS
    env = _make_env(args.hand)
    names = env.robot.joint_names
    n, palm, pen_ok, lat_ok, task_ok = None, None, False, False, False
    try:
        print(f"[smoke] {len(names)} joints: {names}", flush=True)
        palm = _palm_normal_world(env)
        print(f"[smoke] palm normal in world at reset: {palm.round(3).tolist()} (want [0, 0, -1])", flush=True)
        root = env.robot.data.body_pos_w[0, env.robot.body_names.index(env.robot_spec.palm_body_name)]
        print(f"[smoke] hand root at {root.tolist()} (table top {env.cfg.table.top_height})", flush=True)
        default = env.robot.data.default_joint_pos.clone()
        for _ in range(120):                        # 2 s at the reset pose: the pen should lie still on the table
            env.step(env.action_toward(default))
        pen_z = float(env.pen.data.root_pos_w[0, 2] - env.scene.env_origins[0, 2])
        pen_ok = abs(pen_z - (env.cfg.table.top_height + env.cfg.pen.diameter / 2)) < 0.003
        print(f"[smoke] pen centre at z {pen_z:.4f} after 2 s (want table top + radius "
              f"{env.cfg.table.top_height + env.cfg.pen.diameter / 2:.4f}); speed "
              f"{float(env.pen.data.root_lin_vel_w[0].norm()):.4f} m/s", flush=True)
        # Action latency: step t sends the action t / 1000 everywhere; the env must apply one of t-3..t
        # (clipped at the episode's first step).
        env.reset()
        delays = []
        for t in range(1, 61):
            env.step(torch.full((1, len(names)), t / 1000.0, device=env.device))
            delays.append(t - round(float(env.applied_actions[0, 0]) * 1000.0))
        lat_ok = all(0 <= d <= env.cfg.action.action_delay_max for d in delays)
        print(f"[smoke] action latency over 60 steps: delays seen {sorted(set(delays))} "
              f"(want within 0..{env.cfg.action.action_delay_max}); {'ok' if lat_ok else 'WRONG'}", flush=True)
        task_ok = _check_pen_spin(env) & _check_pen_lift(env)
        # The joint-tracking trials below may knock the pen about; keep that from resetting the robot.
        env.cfg.termination.max_goal_distance = env.cfg.termination.fall_below_table = 1e3
        env.reset()
        g = torch.Generator(device="cpu").manual_seed(0)
        worst, w = {}, env.wrist_ids
        top_speed = torch.zeros(len(w), device=env.device)
        for trial in range(3):
            # Desired poses inside the middle 60% of each range, so contact with the table rarely blocks them.
            u = (torch.rand(len(names), generator=g) * 1.2 - 0.6).to(env.device).unsqueeze(0)
            desired = env.action_to_target(u)
            for _ in range(180):                    # 3 s
                env.step(env.action_toward(desired))
                top_speed = torch.maximum(top_speed, env.robot.data.joint_vel[0, w].abs())
            err = (env.robot.data.joint_pos - desired)[0].abs()
            moved = (env.robot.data.joint_pos - env.robot.data.default_joint_pos)[0].abs()
            print(f"[smoke] trial {trial}: joints moved from default by {float(moved.min()):.3f}.."
                  f"{float(moved.max()):.3f}", flush=True)
            for i, nm in enumerate(names):
                worst[nm] = max(worst.get(nm, 0.0), float(err[i]))
        for nm in names:
            print(f"[smoke]   {nm:28s} worst |q - target| {worst[nm]:.4f}", flush=True)
        n = sum(v > 0.05 for v in worst.values())
        print(f"[smoke] {n} of {len(names)} joints off target by more than 0.05 after 3 s", flush=True)
        # The PD lags the target, so allow 10% over the commanded speed.
        over = top_speed > 1.1 * env.wrist_speed
        for i, nm in enumerate(WRIST_JOINTS):
            print(f"[smoke]   {nm:12s} top speed {float(top_speed[i]):.3f} (limit {float(env.wrist_speed[i]):.3f})",
                  flush=True)
        n += int(over.sum())
    finally:
        # Printed before closing: app.close() can end the process.
        ok = palm is not None and palm[2] < -0.99 and n == 0 and pen_ok and lat_ok and task_ok
        print(f"[smoke] {'PASS' if ok else 'FAIL'}: palm faces down, the pen rests on the table, every joint "
              f"tracks its target, the wrist keeps to its speed limits, actions arrive 0-{env.cfg.action.action_delay_max} "
              f"steps late, pen_spin and pen_lift goals, holds and pushes behave as they should", flush=True)
        env.close()
        app.close()
    os._exit(0 if ok else 1)


def _force_task(env, name: str):
    """Reset, then put every env on task ``name``."""
    import math

    import torch
    env.reset()
    k = [t.name for t in env.tasks].index(name)
    ids = torch.arange(env.num_envs, device=env.device)
    env.task_id[:] = k
    env.tasks[k].reset(ids)
    env.successes[:] = 0.0
    env.near_goal_steps[:] = 0
    env.closest_goal_dist[:] = math.nan
    return ids


def _check_pen_lift(env) -> bool:
    """The goal is the pen's own flat pose lift_height up; a goal needs hold_steps in tolerance; the lifted pen is
    pushed, the resting one not."""
    import torch
    ids = _force_task(env, "pen_lift")
    hold = lambda: env.action_toward(env.robot.data.joint_pos.clone())        # noqa: E731
    env.step(hold())
    lift = env.cfg.pen_lift.lift_height
    d0 = float(env.goal_dist[0])
    print(f"[smoke] pen_lift: goal distance at reset {d0 * 1000:.1f} mm (lift height {lift * 1000:.0f} mm)", flush=True)
    ok = abs(d0 - lift) < 0.003
    env.force_prob[:] = 0.0
    env.torque_prob[:] = 0.0
    env.place_pen(ids, env.goal_pos[ids].clone(), env.goal_quat[ids].clone())
    env.step(hold())
    held, hits = int(env.near_goal_steps[0]), int(env.successes[0])
    print(f"[smoke] pen_lift: pen put on its goal -> steps in tolerance {held}, successes {hits} "
          f"(a goal needs {env.cfg.pen_lift.hold_steps})", flush=True)
    ok &= held == 1 and hits == 0
    env.force_prob[:] = 1.0
    env.torque_prob[:] = 1.0
    ids = _force_task(env, "pen_lift")
    env.force_prob[:] = 1.0
    env.torque_prob[:] = 1.0
    env.place_pen(ids, env.goal_pos[ids].clone(), env.goal_quat[ids].clone())
    env.step(hold())
    before_hit = float(env.pen_forces.norm())
    env.successes[:] = 1.0                         # as if the goal had been hit once
    env.place_pen(ids, env.goal_pos[ids].clone(), env.goal_quat[ids].clone())
    pos = env.goal_pos[ids].clone()
    pos[:, 2] = env.pen_rest_height
    env.step(hold())
    pushed = float(env.pen_forces.norm()), float(env.pen_torques.norm())
    env.place_pen(ids, pos, env.goal_quat[ids].clone())
    env.step(hold())
    resting = float(env.pen_forces.norm())
    print(f"[smoke] pen_lift: push on the lifted pen before the first hit {before_hit:.3f} N; after it "
          f"{pushed[0]:.3f} N, twist {pushed[1]:.4f} N m; on the pen back on the table {resting:.3f} N", flush=True)
    ok &= before_hit == 0.0 and pushed[0] > 0.0 and pushed[1] > 0.0 and resting == 0.0
    print(f"[smoke] pen_lift checks {'ok' if ok else 'WRONG'}", flush=True)
    return ok


def _check_pen_spin(env) -> bool:
    """Goals one step round from the pen; a hit counts and moves the goal on; a knocked-away pen ends the episode."""
    import math

    import torch
    ids = _force_task(env, "pen_spin")[:1]
    hold = lambda: env.action_toward(env.robot.data.joint_pos.clone())        # noqa: E731
    env.step(hold())
    step = math.radians(env.cfg.pen_spin.step_deg)
    want = 2 * (env.cfg.pen.length / 2) * math.sin(step / 2)               # tip (and tail) displacement
    d0 = float(env.goal_dist[0])
    print(f"[smoke] pen_spin: goal distance at reset {d0 * 1000:.1f} mm (one {env.cfg.pen_spin.step_deg:g} deg step: "
          f"{want * 1000:.1f} mm)", flush=True)
    ok = abs(d0 - want) < 0.003
    env.place_pen(ids, env.goal_pos[ids].clone(), env.goal_quat[ids].clone())
    env.step(hold())
    bonus = float(env.reward_terms["goal_bonus"][0])
    env.step(hold())
    d1 = float(env.goal_dist[0])
    print(f"[smoke] pen_spin: pen put on its goal -> successes {int(env.successes[0])}, bonus {bonus:g}, "
          f"next goal {d1 * 1000:.1f} mm away", flush=True)
    ok &= int(env.successes[0]) == 1 and bonus == env.cfg.reward.goal_bonus and abs(d1 - want) < 0.003
    pos = env.goal_pos[ids].clone()
    pos[:, 0] += 0.2
    env.place_pen(ids, pos, env.goal_quat[ids].clone())
    env.step(hold())
    away = bool(env.termination_reasons["away"][0])
    print(f"[smoke] pen_spin: pen moved 0.2 m away -> terminated {away}, successes after the reset "
          f"{int(env.successes[0])}", flush=True)
    ok &= away and int(env.successes[0]) == 0
    obs = env._get_observations()
    print(f"[smoke] pen_spin checks {'ok' if ok else 'WRONG'}; actor obs {tuple(obs['policy'].shape)}, critic obs "
          f"{tuple(obs['critic'].shape)} (spaces {env.cfg.observation_space}, {env.cfg.state_space})", flush=True)
    ok &= obs["policy"].shape[-1] == env.cfg.observation_space and obs["critic"].shape[-1] == env.cfg.state_space
    ok &= bool(torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all())
    return ok


def worker(args):
    app = _boot()
    conn = Client((args.host, args.wport), authkey=bytes.fromhex(args.authkey))
    try:
        _loop(args, conn)
    except Exception:
        import traceback
        conn.send({"kind": "error", "text": traceback.format_exc()})
    conn.close()
    app.close()
    os._exit(0)


def _loop(args, conn):
    import torch
    env = _make_env(args.hand)
    names = list(env.robot.joint_names)
    w = env.cfg.table
    conn.send({"kind": "ready", "urdf": str(env.wrist_urdf_path), "joints": names,
               "lower": env.joint_lower.tolist(), "upper": env.joint_upper.tolist(),
               "default": env.robot.data.default_joint_pos[0].tolist(),
               "base_pos": [0.0, 0.0, w.top_height], "table_size": list(w.size), "table_top": w.top_height,
               "pen": [env.cfg.pen.length, env.cfg.pen.diameter]})
    targets = env.robot.data.default_joint_pos.clone()
    t_next = time.time()
    while True:
        while conn.poll():
            m = conn.recv()
            if m["cmd"] == "quit":
                env.close()
                return
            if m["cmd"] == "target":
                targets[0, names.index(m["joint"])] = float(m["value"])
            elif m["cmd"] == "reset":
                env.reset()
                targets = env.robot.data.default_joint_pos.clone()
            elif m["cmd"] == "pen_to_goal":
                ids = torch.zeros(1, dtype=torch.long, device=env.device)
                env.place_pen(ids, env.goal_pos[ids].clone(), env.goal_quat[ids].clone())
        env.step(env.action_toward(targets))
        q = env.robot.data.joint_pos[0]
        conn.send({"kind": "frame", "q": q.tolist(), "err": float((q - targets[0]).abs().max()),
                   "pen_pos": (env.pen.data.root_pos_w[0] - env.scene.env_origins[0]).tolist(),
                   "pen_quat": env.pen.data.root_quat_w[0].tolist(),
                   "goal_pos": env.goal_pos[0].tolist(), "goal_quat": env.goal_quat[0].tolist(),
                   "goal_dist": float(env.goal_dist[0]), "successes": int(env.successes[0]),
                   "task": env.tasks[int(env.task_id[0])].name,
                   "ended": [k for k, v in env.termination_reasons.items() if bool(v[0])]})
        t_next += float(env.step_dt)
        time.sleep(max(0.0, t_next - time.time()))
        t_next = max(t_next, time.time() - 0.1)


def _group(name: str) -> str:
    if name.startswith("wrist_"):
        return "wrist"
    return next((f for f in FINGERS if f in name), "other")


def _add_pen(server, name: str, length: float, diameter: float, opacity: float = 1.0):
    """A pen as drawn: a capsule along its local x, tip to tip ``length``, its tip (+x end) marked red."""
    import numpy as np
    import trimesh
    mesh = trimesh.creation.capsule(height=length - diameter, radius=diameter / 2, count=[24, 24])   # along z
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))            # z -> x
    node = server.scene.add_mesh_simple(name, vertices=mesh.vertices, faces=mesh.faces, color=(38, 89, 204),
                                        opacity=opacity)
    server.scene.add_icosphere(f"{name}/tip", radius=0.4 * diameter, color=(220, 40, 40),
                               position=(length / 2 - 0.3 * diameter, 0.0, 0.0), opacity=opacity)
    return node


def main(args):
    import viser
    from viser.extras import ViserUrdf

    authkey = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    host, port = listener.address
    child = subprocess.Popen([sys.executable, __file__, "--worker", "--hand", args.hand, "--host", host,
                              "--wport", str(port), "--authkey", authkey.hex()],
                             env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
    print("[olympics-viser] Kit is booting in the child (about a minute)", flush=True)
    conn = listener.accept()
    server = viser.ViserServer(port=args.port)
    print(f"\n[olympics-viser] open http://{socket.gethostname()}:{args.port}\n", flush=True)
    outbox: "queue.Queue[dict]" = queue.Queue()
    md = server.gui.add_markdown("waiting for Kit...")
    urdf = info = pen = goal = None
    sliders = {}
    try:
        while child.poll() is None:
            while True:
                try:
                    conn.send(outbox.get_nowait())
                except queue.Empty:
                    break
            if not conn.poll(timeout=0.05):
                continue
            batch = [conn.recv()]
            while conn.poll():
                batch.append(conn.recv())
            frame = None
            for m in batch:
                if m["kind"] == "error":
                    md.content = "**worker failed**\n\n```\n" + m["text"][-1500:] + "\n```"
                    print(m["text"], flush=True)
                    raise SystemExit(1)
                if m["kind"] == "frame":
                    frame = m
                elif m["kind"] == "ready":
                    info = m
                    sx, sy, sz = m["table_size"]
                    server.scene.add_box("/table", dimensions=(sx, sy, sz), color=(140, 107, 77),
                                         position=(0.0, 0.0, m["table_top"] - sz / 2))
                    server.scene.add_grid("/ground", width=2.0, height=2.0, position=(0.0, 0.0, 0.0))
                    server.scene.add_frame("/base", position=tuple(m["base_pos"]), show_axes=False)
                    urdf = ViserUrdf(server, pathlib.Path(m["urdf"]), root_node_name="/base/robot")
                    pen = _add_pen(server, "/pen", *m["pen"])
                    goal = _add_pen(server, "/goal", *m["pen"], opacity=0.3)
                    b_reset = server.gui.add_button("reset (sim and sliders)")
                    b_goal = server.gui.add_button("put the pen on its goal")
                    b_goal.on_click(lambda _: outbox.put({"cmd": "pen_to_goal"}))
                    groups = ["wrist"] + [f for f in FINGERS if any(_group(n) == f for n in m["joints"])]
                    for g in groups + (["other"] if any(_group(n) == "other" for n in m["joints"]) else []):
                        with server.gui.add_folder(g, expand_by_default=(g == "wrist")):
                            for i, n in enumerate(m["joints"]):
                                if _group(n) != g:
                                    continue
                                lo, hi = m["lower"][i], m["upper"][i]
                                s = server.gui.add_slider(n, lo, hi, (hi - lo) / 200, m["default"][i])
                                s.on_update(lambda _, n=n, s=s: outbox.put(
                                    {"cmd": "target", "joint": n, "value": float(s.value)}))
                                sliders[n] = (s, m["default"][i])

                    def _reset(_):
                        for s, d in sliders.values():
                            s.value = d
                        outbox.put({"cmd": "reset"})
                    b_reset.on_click(_reset)
            if frame is None or info is None:
                continue
            urdf.update_cfg(dict(zip(info["joints"], frame["q"])))
            pen.position, pen.wxyz = tuple(frame["pen_pos"]), tuple(frame["pen_quat"])
            goal.position, goal.wxyz = tuple(frame["goal_pos"]), tuple(frame["goal_quat"])
            ended = f"  \n**episode ended: {', '.join(frame['ended'])}**" if frame["ended"] else ""
            md.content = (f"task **{frame['task']}** · goals hit {frame['successes']} · goal distance "
                          f"{frame['goal_dist'] * 1000:.0f} mm  \nworst |q - target| {frame['err']:.3f}{ended}")
    except (EOFError, ConnectionResetError):
        print("[olympics-viser] worker closed the connection", flush=True)
    finally:
        try:
            conn.send({"cmd": "quit"})
        except Exception:
            pass
        time.sleep(1.0)
        if child.poll() is None:
            child.kill()


if __name__ == "__main__":
    a = parse()
    if a.smoke:
        smoke(a)
    elif a.worker:
        worker(a)
    else:
        main(a)
