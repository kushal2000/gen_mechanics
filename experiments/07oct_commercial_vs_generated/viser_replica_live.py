"""Live viewer: the 52-hand Wuji v2 policy on the vendor hand and on a replica, side by side, in viser.

Two panels, each with its own hand dropdown -- the vendor Wuji v2, or any replica in
assets/urdf/unified_dynamics_commercial_hands_replicas/wuji2 (the grammar's own, each fidelity level, the clone).
Both run the SAME policy (06oct_wuji2_multi_embodiment, 31826, epoch 5000) live in Isaac Sim, zero-shot on a
replica: the joint transformer is rebuilt for the hand's joints with the same weights.

Built on experiments/01oct_unified_rl/viser_compare.py (its --worker) and 05oct's viser_missing_finger.py: Kit and
viser cannot share an interpreter (coevolution/eval/viser_play.py), so each panel's simulation is a child worker
streaming joint angles and object poses back; changing a panel's hand restarts its worker (about a minute).

    .venv_isaacsim/bin/python experiments/07oct_commercial_vs_generated/viser_replica_live.py --port 8084
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import queue
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from multiprocessing.connection import Listener

REPO = pathlib.Path(__file__).resolve().parents[2]
WORKER = REPO / "experiments/01oct_unified_rl/viser_compare.py"
sys.path.insert(0, str(WORKER.parent))
REPLICAS = REPO / "assets/urdf/unified_dynamics_commercial_hands_replicas/wuji2"
CKPT = ("debug_outputs/train_logs/06oct_wuji2_multi_embodiment/0_scale_train_wuji2_lr_2plus_ema0.1_c01_*_31826_0_*"
        "/rank_0/*/nn/last_*_ep_5000_*.pth")


def hands() -> dict[str, str]:
    """Dropdown label -> spec the worker builds."""
    out = {"Wuji v2 (vendor)": "wuji2"}
    for p in sorted(REPLICAS.glob("wuji2_left_replica*.json")):
        if p.name.endswith(".joint_map.json"):
            continue
        tag = p.stem[len("wuji2_left_replica"):].lstrip("_") or "grammar"
        out[f"replica: {tag}"] = f"handonly:{p.relative_to(REPO)}"
    return out


class Panel:
    """One side: a hand dropdown, its worker, and its drawing under /p<i>."""

    def __init__(self, server, i: int, x: float, opts: dict, initial: str, policy: str):
        self.server, self.i, self.x, self.opts, self.policy = server, i, x, opts, policy
        self.authkey = secrets.token_bytes(16)
        self.listener = Listener(("127.0.0.1", 0), authkey=self.authkey)
        with server.gui.add_folder(f"{'left' if i == 0 else 'right'} hand"):
            self.dd = server.gui.add_dropdown("hand", tuple(opts), initial_value=initial)
            self.md = server.gui.add_markdown("")
        self.wanted, self.current = initial, None
        self.dd.on_update(lambda _: setattr(self, "wanted", str(self.dd.value)))
        self.child = self.conn = self.info = None
        self.urdf = self.obj = self.goal = self.label = None
        self.dead = False

    def stop(self):
        if self.conn is not None:
            try:
                self.conn.send({"cmd": "quit"})
            except Exception:
                pass
        if self.child is not None:
            time.sleep(0.5)
            if self.child.poll() is None:
                self.child.kill()
            self.child.wait()
        try:
            self.server.scene.remove_by_name(f"/p{self.i}")
        except Exception:
            pass
        self.child = self.conn = self.info = None
        self.urdf = self.obj = self.goal = self.label = None

    def start(self):
        self.stop()
        self.current, self.dead = self.wanted, False
        self.md.content = f"**loading {self.current}** (Kit boots in about a minute)"
        host, port = self.listener.address
        cmd = [sys.executable, str(WORKER), "--worker", "--hand", self.opts[self.current], "--host", host,
               "--wport", str(port), "--authkey", self.authkey.hex(), "--policy", self.policy]
        self.child = subprocess.Popen(cmd, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
        self.conn = self.listener.accept()

    def on_ready(self, m):
        from viser.extras import ViserUrdf
        from viser_unified import _face_cube
        spec = self.opts[self.current]
        if not m["urdf"]:                       # a generated design: its viewing URDF, as viser_compare does
            from coevolution.eval import viser_play as vp
            from hand_sampler import population_io
            from viser_compare import spec_ref
            hand = population_io.load_population(spec_ref(spec)[len("handonly:"):])[0]
            m["urdf"] = str(vp._viewing_urdf(hand, spec_ref(spec), pathlib.Path(tempfile.mkdtemp()) / "hand.urdf"))
        s = self.server
        s.scene.add_frame(f"/p{self.i}", position=(self.x, 0.0, 0.0), show_axes=False)
        s.scene.add_frame(f"/p{self.i}/base", position=tuple(m["base_pos"]), wxyz=tuple(m["base_rot"]),
                          show_axes=False)
        self.urdf = ViserUrdf(s, pathlib.Path(m["urdf"]), root_node_name=f"/p{self.i}/base/robot")
        self.obj = _face_cube(s, f"/p{self.i}/obj", m["cube"], opacity=1.0)
        self.goal = _face_cube(s, f"/p{self.i}/goal", m["cube"], opacity=0.35)
        self.label = s.scene.add_label(f"/p{self.i}/label", self.current)
        self.info = m
        self.md.content = f"**{self.current}**: {len(m['joints'])} joint slots, policy {m['labels'][0]}"

    def pump(self, outbox: list):
        """Send queued commands, draw the newest frame. Returns False when nothing came."""
        if self.wanted != self.current:
            self.start()
        if self.dead or self.conn is None:
            return False
        for m in outbox:
            self.conn.send(m)
        if not self.conn.poll(timeout=0.01):
            if self.child.poll() is not None:
                self.md.content = "**worker exited** -- pick a hand to restart"
                self.dead = True
            return False
        batch = [self.conn.recv()]
        while self.conn.poll():
            batch.append(self.conn.recv())
        frame = None
        for m in batch:
            if m["kind"] == "frame":
                frame = m
            elif m["kind"] == "ready":
                self.on_ready(m)
            elif m["kind"] == "error":
                self.md.content = "**worker failed**\n\n```\n" + m["text"][-1500:] + "\n```"
                print(m["text"], flush=True)
                self.dead = True
        if frame is None or self.info is None:
            return True
        self.urdf.update_cfg(dict(zip(self.info["joints"], frame["q"][0])))
        self.obj.position, self.obj.wxyz = tuple(frame["obj_pos"][0]), tuple(frame["obj_quat"][0])
        self.goal.position, self.goal.wxyz = tuple(frame["goal_pos"][0]), tuple(frame["goal_quat"][0])
        gp = frame["goal_pos"][0]
        self.label.position = (gp[0], gp[1], gp[2] + 0.09)
        self.label.text = f"{self.current}  ·  {frame['goals'][0]} goals  ·  {frame['drops'][0]} drops"
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8084)
    ap.add_argument("--spacing", type=float, default=0.32)
    ap.add_argument("--right", default="replica: clone", help="the right panel's starting hand")
    args = ap.parse_args()

    import viser

    ckpt = sorted(glob.glob(str(REPO / CKPT)))[0]
    policy = f"Multi-52 Wuji v2={ckpt}"
    opts = hands()
    server = viser.ViserServer(port=args.port)
    print(f"\n[replica-live] viser on http://{socket.gethostname()}:{args.port}  ({len(opts)} hands)\n", flush=True)
    panels = [Panel(server, 0, -args.spacing / 2, opts, "Wuji v2 (vendor)", policy),
              Panel(server, 1, args.spacing / 2, opts, args.right if args.right in opts else list(opts)[-1], policy)]
    with server.gui.add_folder("playback (both)"):
        b_pause = server.gui.add_button("pause / resume")
        b_reset = server.gui.add_button("reset")
        s_speed = server.gui.add_slider("speed (x real time)", 0.1, 2.0, 0.05, 1.0)
    outbox: "queue.Queue[dict]" = queue.Queue()
    b_pause.on_click(lambda _: outbox.put({"cmd": "toggle"}))
    b_reset.on_click(lambda _: outbox.put({"cmd": "reset"}))
    s_speed.on_update(lambda _: outbox.put({"cmd": "speed", "value": float(s_speed.value)}))
    try:
        while True:
            msgs = []
            while True:
                try:
                    msgs.append(outbox.get_nowait())
                except queue.Empty:
                    break
            busy = [p.pump(msgs) for p in panels]
            if not any(busy):
                time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        for p in panels:
            p.stop()


if __name__ == "__main__":
    main()
