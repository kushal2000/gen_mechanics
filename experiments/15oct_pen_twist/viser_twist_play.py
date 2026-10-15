"""Prototype of the twist-apart pen task, live in viser: a hand on the dummy wrist and the twist pen (barrel + cap on a
friction joint about the long axis, isaacsimenvs/inhand_olympics/twist_pen.py) floating at the palm, gravity off.
Drive every joint with a slider and try to twist the cap against the barrel; the page shows the twist angle.

    .venv_isaacsim/bin/python experiments/15oct_pen_twist/viser_twist_play.py [--port 8095] [--hand <spec>]

Panels:
  pen      place the pen at the palm (its axis across the palm, along the fingers, or out of the palm; how far out;
           sideways offset), the cap's friction torque (live), and a twist test: equal and opposite torques on cap and
           barrel for a second, to feel the friction (below it the cap holds, above it it turns)
  tether   a soft spring holding the barrel at its placed pose (on at placement). With gravity off and nothing else
           holding it, the first finger to touch the pen pushes it away before an opposing finger arrives; close the
           fingers with the tether on, then turn it off to see whether the grip holds on its own
  joints   a slider per joint, wrist first (the env's own action pipeline steers each joint to its slider)

Kit and viser cannot share an interpreter: the sim runs in a child (this file with --worker) and streams frames back.
Loading another hand restarts it (about a minute).
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
import threading
import time
from multiprocessing.connection import Client, Listener

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
PRIMS = REPO / "assets/urdf/generated/14oct_design_search"
AXES = ("across the palm", "along the fingers", "out of the palm")


def hands() -> dict[str, str]:
    """label -> robot spec for the loader: Wuji full and thumb + index, then every built primitive hand."""
    out = {"wuji2_left_uniform_handonly": "wuji2_left_uniform_handonly",
           "wuji2_left_uniform_handonly_only_thumb_index": "wuji2_left_uniform_handonly_only_thumb_index"}
    for p in sorted(PRIMS.glob("*.spec.json")):
        out[p.name.removesuffix(".spec.json")] = str(p.relative_to(REPO))
    return out


# --- the Kit child ------------------------------------------------------------------------------------------------
def worker(args) -> None:
    sys.path.insert(0, str(REPO))
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    app = AppLauncher(a).app
    conn = Client((args.host, args.wport), authkey=bytes.fromhex(args.authkey))
    try:
        _loop(args, conn)
    except Exception:
        import traceback
        conn.send({"kind": "error", "text": traceback.format_exc()})
    conn.close()
    app.close()
    os._exit(0)


def _loop(args, conn) -> None:
    import math

    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation

    import isaacsimenvs.inhand_olympics  # noqa: F401
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg
    from isaacsimenvs.inhand_olympics.twist_pen import link_extents
    from isaacsimenvs.inhand_olympics.wrist import palm_frame

    cfg = InHandOlympicsEnvCfg()
    cfg.hand.robot_spec = args.hand
    cfg.scene.num_envs = 1
    cfg.task.names = ("pen_spin",)                 # no pushes, not carried: the task only places the pen at reset
    cfg.goal.seconds_per_goal = 1e6
    cfg.goal.max_successes = 10 ** 6
    cfg.termination.max_goal_distance = 0.0        # no "away": the pen lives at the palm here, not near a goal
    cfg.reset.stagger_initial_episodes = False
    cfg.reset.finger_pos_fraction = 0.0            # fingers start at the hand's default pose
    cfg.reset.finger_vel = 0.0
    cfg.pen.shape = args.pen_shape
    cfg.pen.twist = True
    cfg.pen.gravity = False
    cfg.pen.twist_friction_torque = args.friction
    env = InHandOlympicsEnv(cfg)
    env.reset()
    dev, ids = env.device, torch.zeros(1, dtype=torch.long, device=env.device)
    names = list(env.robot.joint_names)
    barrel, cap = env.pen.body_names.index("barrel"), env.pen.body_names.index("cap")
    frame = palm_frame(env.robot_spec)                 # columns: palm normal, width, wrist-to-fingertip (root body)
    p = cfg.pen
    conn.send({"kind": "ready", "urdf": str(env.wrist_urdf_path), "joints": names,
               "lower": env.joint_lower[0].tolist(), "upper": env.joint_upper[0].tolist(),
               "default": env.robot.data.default_joint_pos[0].tolist(), "base_pos": [0.0, 0.0, cfg.table.top_height],
               "table_size": list(cfg.table.size), "table_top": cfg.table.top_height,
               "pen": {"length": p.length, "diameter": p.diameter, "shape": p.shape,
                       "links": link_extents(p.length, p.cap_length, 0.001)}, "friction": p.twist_friction_torque})

    def palm_axes():
        q = env.robot.data.body_quat_w[0, env.palm_body].tolist()
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix() @ frame
        pos = (env.robot.data.body_pos_w[0, env.palm_body] - env.scene.env_origins[0]).cpu().numpy()
        return pos, R[:, 0], R[:, 1], R[:, 2]

    def place(m: dict) -> None:
        pos, normal, width, fingers = palm_axes()
        centre = pos + m["out"] * normal + m["side"] * width + m["along"] * fingers
        x = {AXES[0]: width, AXES[1]: fingers, AXES[2]: normal}[m["axis"]]
        other = fingers if m["axis"] != AXES[1] else width
        y = np.cross(other, x)
        y /= np.linalg.norm(y)
        R = np.stack([x, y, np.cross(x, y)], 1)
        w, xx, yy, zz = np.roll(Rotation.from_matrix(R).as_quat(), 1)
        env.place_pen(ids, torch.tensor([centre], dtype=torch.float32, device=dev),
                      torch.tensor([[w, xx, yy, zz]], dtype=torch.float32, device=dev))
        tether.update(on=True, pos=centre, quat=np.array([w, xx, yy, zz]))
        twist.update(turned=0.0, prev=0.0)

    targets = env.robot.data.default_joint_pos.clone()
    test = {"until": 0.0, "torque": 0.0}
    tether = {"on": True, "pos": None, "quat": None}
    twist = {"turned": 0.0, "prev": 0.0}                # PhysX wraps the joint angle to +-pi: accumulate the change
    last_place = {"axis": AXES[0], "out": 0.04, "side": 0.0, "along": 0.04}
    place(last_place)
    t_next = time.time()
    while True:
        while conn.poll():
            m = conn.recv()
            c = m["cmd"]
            if c == "quit":
                env.close()
                return
            if c == "target":
                targets[0, names.index(m["joint"])] = float(m["value"])
            elif c == "reset":
                env.reset()
                targets = env.robot.data.default_joint_pos.clone()
                place(last_place)
            elif c == "place":
                last_place = m
                place(m)
            elif c == "friction":
                env.pen.write_joint_effort_limit_to_sim(torch.tensor([[float(m["value"])]], device=dev), joint_ids=[0])
            elif c == "twist_test":
                test = {"until": time.time() + float(m["seconds"]), "torque": float(m["torque"])}
            elif c == "zero_twist":
                twist["turned"] = 0.0
            elif c == "tether":
                tether["on"] = bool(m["on"])
        # The twist test: +torque on the cap and -torque on the barrel, about the barrel's axis (an internal twist:
        # the pen as a whole feels none).
        torques = torch.zeros(1, 2, 3, device=dev)
        forces = torch.zeros(1, 2, 3, device=dev)
        if tether["on"]:
            # A spring-damper on the barrel toward its placed pose: about 30 rad/s for the 30 g pen, critically damped.
            o = env.scene.env_origins[0].cpu().numpy()
            x = env.pen.data.body_pos_w[0, barrel].cpu().numpy() - o
            v = env.pen.data.body_lin_vel_w[0, barrel].cpu().numpy()
            wv = env.pen.data.body_ang_vel_w[0, barrel].cpu().numpy()
            q = env.pen.data.body_quat_w[0, barrel].tolist()
            err = (Rotation.from_quat(np.roll(tether["quat"], -1)) * Rotation.from_quat([q[1], q[2], q[3], q[0]]).inv())
            forces[0, barrel] = torch.tensor(25.0 * (tether["pos"] - x) - 1.7 * v, dtype=torch.float32, device=dev)
            torques[0, barrel] = torch.tensor(0.008 * err.as_rotvec() - 4e-4 * wv, dtype=torch.float32, device=dev)
        if time.time() < test["until"]:
            q = env.pen.data.body_quat_w[0, barrel].tolist()
            axis = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()[:, 0]
            t = torch.tensor(axis, dtype=torch.float32, device=dev) * test["torque"]
            torques[0, cap] += t
            torques[0, barrel] -= t
        env.pen.set_external_force_and_torque(forces=forces, torques=torques, is_global=True, body_ids=[barrel, cap])
        env.step(env.action_toward(targets))
        qj = env.robot.data.joint_pos[0]
        o = env.scene.env_origins[0]
        palm, *_ = palm_axes()
        bp = env.pen.data.body_pos_w[0] - o
        now = float(env.pen_twist()[0])
        twist["turned"] += (now - twist["prev"] + math.pi) % (2 * math.pi) - math.pi
        twist["prev"] = now
        conn.send({"kind": "frame", "q": qj.tolist(), "err": float((qj - targets[0]).abs().max()),
                   "barrel": [bp[barrel].tolist(), env.pen.data.body_quat_w[0, barrel].tolist()],
                   "cap": [bp[cap].tolist(), env.pen.data.body_quat_w[0, cap].tolist()],
                   "twist": twist["turned"], "twist_vel": float(env.pen.data.joint_vel[0, 0]), "tether": tether["on"],
                   "palm_dist": float(np.linalg.norm(bp[barrel].cpu().numpy() - palm)),
                   "testing": time.time() < test["until"],
                   "ended": [k for k, v in env.termination_reasons.items() if bool(v[0])]})
        t_next += float(env.step_dt)
        time.sleep(max(0.0, t_next - time.time()))
        t_next = max(t_next, time.time() - 0.1)


# --- the page -----------------------------------------------------------------------------------------------------
class Sim:
    def __init__(self, listener, authkey: bytes, hand: str, pen_shape: str, friction: float):
        host, port = listener.address
        self.proc = subprocess.Popen(
            [sys.executable, __file__, "--worker", "--hand", hand, "--pen-shape", pen_shape, "--friction",
             str(friction), "--host", host, "--wport", str(port), "--authkey", authkey.hex()],
            cwd=REPO, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
        self.conn = None
        self._q: "queue.Queue" = queue.Queue()
        threading.Thread(target=lambda: self._q.put(listener.accept()), daemon=True).start()

    def connected(self) -> bool:
        if self.conn is None:
            try:
                self.conn = self._q.get_nowait()
            except queue.Empty:
                return False
        return True

    def stop(self) -> None:
        try:
            if self.conn is not None:
                self.conn.send({"cmd": "quit"})
        except Exception:
            pass
        try:
            self.proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _link_mesh(server, name: str, x0: float, x1: float, diameter: float, color):
    """One link of the pen as drawn: a box from x0 to x1 along the pen (the link frame at the pen's centre), with a
    stripe along one face so its turning shows."""
    import numpy as np
    import trimesh
    box = trimesh.creation.box(extents=(x1 - x0, diameter, diameter))
    box.apply_translation(((x0 + x1) / 2, 0.0, 0.0))
    node = server.scene.add_mesh_simple(name, vertices=box.vertices, faces=box.faces, color=color)
    stripe = trimesh.creation.box(extents=(x1 - x0, diameter * 0.25, diameter * 0.08))
    stripe.apply_translation(((x0 + x1) / 2, 0.0, diameter / 2 + diameter * 0.04))
    server.scene.add_mesh_simple(f"{name}/stripe", vertices=stripe.vertices, faces=stripe.faces, color=(250, 250, 250))
    return node


def main() -> None:
    sys.path.insert(0, str(REPO / "isaacsimenvs/inhand_olympics/tests"))
    from viser_hand_table import _group      # Kit-free

    import signal

    import viser
    from viser.extras import ViserUrdf

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    authkey = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    server = viser.ViserServer(port=ARGS.port)
    server.scene.set_up_direction("+z")
    print(f"\n[twist-play] open http://{socket.gethostname()}:{ARGS.port}\n", flush=True)
    outbox: "queue.Queue[dict]" = queue.Queue()
    requests: "queue.Queue[str]" = queue.Queue()
    hs = hands()
    first = next((k for k, v in hs.items() if v == ARGS.hand or k == ARGS.hand), "wuji2_left_uniform_handonly")

    with server.gui.add_folder("hand"):
        g_hand = server.gui.add_dropdown("hand", tuple(hs), initial_value=first)
        b_load = server.gui.add_button("load (restarts the sim, about a minute)")
    md = server.gui.add_markdown("")
    with server.gui.add_folder("pen"):
        g_axis = server.gui.add_dropdown("pen axis", AXES, initial_value=AXES[0])
        g_out = server.gui.add_slider("out from the palm (cm)", 0.0, 15.0, 0.1, 4.0)
        g_along = server.gui.add_slider("toward the fingertips (cm)", -5.0, 15.0, 0.1, 4.0)
        g_side = server.gui.add_slider("sideways (cm)", -8.0, 8.0, 0.1, 0.0)
        b_place = server.gui.add_button("place the pen at the palm (at rest, untwisted, tethered)")
        g_tether = server.gui.add_checkbox("tether the barrel (a soft spring to its placed pose)", True)
        g_fric = server.gui.add_slider("cap friction torque (N m)", 0.0, 0.1, 0.001, ARGS.friction)
        g_tq = server.gui.add_slider("twist test torque (N m)", 0.0, 0.1, 0.001, 0.02)
        b_test = server.gui.add_button("twist test: opposite torques on cap and barrel for 1 s")
        b_zero = server.gui.add_button("zero the twist counter")
        b_reset = server.gui.add_button("reset (sim and sliders)")

    def place_msg() -> dict:
        return {"cmd": "place", "axis": str(g_axis.value), "out": g_out.value / 100, "side": g_side.value / 100,
                "along": g_along.value / 100}

    def _place(_) -> None:
        g_tether.value = True
        outbox.put(place_msg())

    b_place.on_click(_place)
    g_tether.on_update(lambda _: outbox.put({"cmd": "tether", "on": bool(g_tether.value)}))
    g_fric.on_update(lambda _: outbox.put({"cmd": "friction", "value": float(g_fric.value)}))
    b_test.on_click(lambda _: outbox.put({"cmd": "twist_test", "torque": float(g_tq.value), "seconds": 1.0}))
    b_zero.on_click(lambda _: outbox.put({"cmd": "zero_twist"}))
    b_load.on_click(lambda _: requests.put(str(g_hand.value)))
    requests.put(first)

    sim: Sim | None = None
    view: dict = {}
    load_id = 0
    t_boot = 0.0

    def clear_view() -> None:
        if "urdf" in view:
            view["urdf"].remove()
        for key in ("nodes", "folders"):
            for h in view.get(key, []):
                try:
                    h.remove()
                except Exception:
                    pass
        view.clear()

    def build_view(m: dict) -> None:
        sx, sy, sz = m["table_size"]
        nodes = [server.scene.add_box(f"/table{load_id}", dimensions=(sx, sy, sz), color=(140, 107, 77),
                                      position=(0.0, 0.0, m["table_top"] - sz / 2)),
                 server.scene.add_frame(f"/base{load_id}", position=tuple(m["base_pos"]), show_axes=False)]
        urdf = ViserUrdf(server, pathlib.Path(m["urdf"]), root_node_name=f"/base{load_id}/robot")
        pen = m["pen"]
        links = {"barrel": _link_mesh(server, f"/barrel{load_id}", *pen["links"]["barrel"], pen["diameter"],
                                      (38, 89, 204)),
                 "cap": _link_mesh(server, f"/cap{load_id}", *pen["links"]["cap"], pen["diameter"], (220, 60, 40))}
        nodes += list(links.values())
        sliders, folders = {}, []
        names = m["joints"]
        for g in sorted({_group(n) for n in names}, key=lambda g: (g != "wrist", g)):
            with server.gui.add_folder(g, expand_by_default=(g != "wrist")) as fold:
                folders.append(fold)
                for i, n in enumerate(names):
                    if _group(n) != g:
                        continue
                    lo, hi = m["lower"][i], m["upper"][i]
                    s = server.gui.add_slider(n, lo, hi, (hi - lo) / 200, m["default"][i])
                    s.on_update(lambda _, n=n, s=s: outbox.put({"cmd": "target", "joint": n, "value": float(s.value)}))
                    sliders[n] = (s, m["default"][i])
        view.update(urdf=urdf, links=links, nodes=nodes, folders=folders, sliders=sliders, info=m)
        outbox.put(place_msg())

    @b_reset.on_click
    def _(_) -> None:
        for s, d in view.get("sliders", {}).values():
            s.value = d
        g_tether.value = True
        outbox.put({"cmd": "reset"})
        outbox.put(place_msg())

    try:
        while True:
            try:
                name = requests.get_nowait()
                if sim is not None:
                    sim.stop()
                clear_view()
                while not outbox.empty():
                    outbox.get_nowait()
                load_id += 1
                sim = Sim(listener, authkey, hs[name], ARGS.pen_shape, float(g_fric.value))
                t_boot = time.time()
                md.content = f"loading **{name}** -- Kit boots in a child (about a minute)"
            except queue.Empty:
                pass
            if sim is None or sim.proc.poll() is not None:
                time.sleep(0.1)
                continue
            if not sim.connected():
                md.content = f"Kit booting... {time.time() - t_boot:.0f} s"
                time.sleep(0.1)
                continue
            try:
                while True:
                    try:
                        sim.conn.send(outbox.get_nowait())
                    except queue.Empty:
                        break
                if not sim.conn.poll(timeout=0.05):
                    continue
                batch = [sim.conn.recv()]
                while sim.conn.poll():
                    batch.append(sim.conn.recv())
            except (EOFError, ConnectionResetError, BrokenPipeError):
                md.content = "**the sim closed the connection**"
                sim = None
                continue
            frame = None
            for m in batch:
                if m["kind"] == "error":
                    md.content = "**sim failed**\n\n```\n" + m["text"][-1500:] + "\n```"
                    print(m["text"], flush=True)
                elif m["kind"] == "ready":
                    build_view(m)
                elif m["kind"] == "frame":
                    frame = m
            if frame is None or "info" not in view:
                continue
            view["urdf"].update_cfg(dict(zip(view["info"]["joints"], frame["q"])))
            for k in ("barrel", "cap"):
                view["links"][k].position, view["links"][k].wxyz = tuple(frame[k][0]), tuple(frame[k][1])
            import math
            tw = frame["twist"]
            md.content = (f"**cap twist {math.degrees(tw):+.0f}°** ({tw / (2 * math.pi):+.2f} turns) · "
                          f"{math.degrees(frame['twist_vel']):+.0f}°/s"
                          f"{' · **twist test on**' if frame['testing'] else ''}"
                          f"{' · tethered' if frame['tether'] else ' · **free**'}  \n"
                          f"pen centre {frame['palm_dist'] * 100:.1f} cm from the palm · worst |q - target| "
                          f"{frame['err']:.3f}"
                          + (f"  \nepisode ended: {', '.join(frame['ended'])}" if frame["ended"] else ""))
    except KeyboardInterrupt:
        pass
    finally:
        if sim is not None:
            sim.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--hand", default="wuji2_left_uniform_handonly")
    ap.add_argument("--pen-shape", default="box", choices=("box",))
    ap.add_argument("--friction", type=float, default=0.01, help="the cap's friction torque at start (N m)")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--host", default="", help=argparse.SUPPRESS)
    ap.add_argument("--wport", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--authkey", default="", help=argparse.SUPPRESS)
    ARGS = ap.parse_args()
    if ARGS.worker:
        worker(ARGS)
    else:
        main()
