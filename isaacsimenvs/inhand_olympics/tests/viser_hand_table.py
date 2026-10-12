"""Viser test of the in-hand olympics scene: the Wuji v2 left hand on its 6-DOF dummy wrist, palm down above the
table. Every joint gets a slider spanning its limits in sim, grouped in collapsible folders (wrist, then one per
finger); a slider sets that joint's position target, the sim tracks it, and viser draws the simulated joint
positions. The status line shows the worst target-tracking error.

Kit and viser cannot share an interpreter, so the sim runs in a child worker (as in
experiments/01oct_unified_rl/viser_compare.py) and streams joint positions back.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python isaacsimenvs/inhand_olympics/tests/viser_hand_table.py \\
        --port 8090                     # then open http://<node>:8090
    ... viser_hand_table.py --smoke     # no viewer: random targets, check tracking and that the palm faces down
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
    cfg.episode_length_s = 1e6            # the viewer never resets on its own
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
    env = _make_env(args.hand)
    names = env.robot.joint_names
    n, palm = None, None
    try:
        print(f"[smoke] {len(names)} joints: {names}", flush=True)
        palm = _palm_normal_world(env)
        print(f"[smoke] palm normal in world at reset: {palm.round(3).tolist()} (want [0, 0, -1])", flush=True)
        root = env.robot.data.body_pos_w[0, env.robot.body_names.index(env.robot_spec.palm_body_name)]
        print(f"[smoke] hand root at {root.tolist()} (table top {env.cfg.table.top_height})", flush=True)
        g = torch.Generator(device="cpu").manual_seed(0)
        worst = {}
        for trial in range(3):
            # Targets inside the middle 60% of each range, so contact with the table rarely blocks them.
            a = (torch.rand(len(names), generator=g) * 1.2 - 0.6).to(env.device).unsqueeze(0)
            for _ in range(180):                    # 3 s
                env.step(a)
            err = (env.robot.data.joint_pos - env.targets)[0].abs()
            moved = (env.robot.data.joint_pos - env.robot.data.default_joint_pos)[0].abs()
            print(f"[smoke] trial {trial}: joints moved from default by {float(moved.min()):.3f}.."
                  f"{float(moved.max()):.3f}", flush=True)
            for i, nm in enumerate(names):
                worst[nm] = max(worst.get(nm, 0.0), float(err[i]))
        for nm in names:
            print(f"[smoke]   {nm:28s} worst |q - target| {worst[nm]:.4f}", flush=True)
        n = sum(v > 0.05 for v in worst.values())
        print(f"[smoke] {n} of {len(names)} joints off target by more than 0.05 after 3 s", flush=True)
    finally:
        # Printed before closing: app.close() can end the process.
        ok = palm is not None and palm[2] < -0.99 and n == 0
        print(f"[smoke] {'PASS' if ok else 'FAIL'}: palm faces down and every joint tracks its target", flush=True)
        env.close()
        app.close()
    os._exit(0 if ok else 1)


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
               "base_pos": [0.0, 0.0, w.top_height], "table_size": list(w.size), "table_top": w.top_height})
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
        env.step(env.target_to_action(targets))
        q = env.robot.data.joint_pos[0]
        conn.send({"kind": "frame", "q": q.tolist(), "err": float((q - targets[0]).abs().max())})
        t_next += float(env.step_dt)
        time.sleep(max(0.0, t_next - time.time()))
        t_next = max(t_next, time.time() - 0.1)


def _group(name: str) -> str:
    if name.startswith("wrist_"):
        return "wrist"
    return next((f for f in FINGERS if f in name), "other")


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
    urdf = info = None
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
                    b_reset = server.gui.add_button("reset (sim and sliders)")
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
            md.content = f"worst |q - target|: {frame['err']:.3f}"
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
