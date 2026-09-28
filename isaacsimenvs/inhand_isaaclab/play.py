"""Watch a rung of the IsaacLab in-hand ladder in viser.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python -m isaacsimenvs.inhand_isaaclab.play \
        --task GenMech-InHandIsaacLab-step0-reference-v0 --port 8083 \
        --checkpoint debug_outputs/train_logs/inhand_isaaclab/<run>/.../nn/<ckpt>.pth

With no checkpoint the policy is random, which still shows the reset pose, the cube's
start placement and the goal -- most of what goes wrong before a policy exists.

Kit runs in a CHILD process (``_play_worker.py``); this process owns viser and does all
the drawing. They cannot share an interpreter: Kit's pip_prebundle carries an old
``websockets`` and wins the import however you order sys.path.

What you see:
  * the hand, drawn from our unified Allegro URDF -- the reference's USD cannot be
    loaded by a browser, and its joint names are identical to ours (checked as a set),
    so the same file serves both
  * the cube, Rubik-faced at the reference's own 60 mm, so its orientation is readable
  * the goal, the same cube translucent, so "are the faces lined up" is answerable
  * sliders to drive the 16 joints by hand, ranged by the robot's own limits
"""

from __future__ import annotations

import argparse
import pathlib
import secrets
import socket
import subprocess
import sys
from multiprocessing.connection import Listener

import numpy as np
import viser
from viser.extras import ViserUrdf

# Parent-side and Kit-free: turns a box-visual URDF into (dims, rgb, xyz, wxyz) tuples.
from coevolution.eval.viser_play import _goal_boxes_from_urdf

_WORKER = pathlib.Path(__file__).resolve().parent / "_play_worker.py"
GOAL_OPACITY = 0.45


def _draw_cube(server, urdf: pathlib.Path, root: str, opacity: float | None) -> None:
    """The Rubik cube under ``root``, one viser box per URDF visual.

    Rebuilt box by box rather than handed to ViserUrdf, because ViserUrdf's only colour
    control is mesh_color_override, which repaints every visual ONE colour -- and the
    six face colours are the entire reason the cube is drawn this way.
    """
    boxes = _goal_boxes_from_urdf(urdf)
    if not boxes:
        return
    for i, (dims, rgb, xyz, wxyz) in enumerate(boxes):
        server.scene.add_box(f"{root}/v{i}", color=rgb, dimensions=dims,
                             opacity=opacity, position=xyz, wxyz=wxyz, flat_shading=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", default="GenMech-InHandIsaacLab-step0-reference-v0")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--port", type=int, default=8083)
    p.add_argument("--device", default="cuda:0")
    args, rest = p.parse_known_args()

    authkey = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    host, port = listener.address

    child = subprocess.Popen(
        [sys.executable, str(_WORKER), "--task", args.task, "--checkpoint", args.checkpoint,
         "--num-envs", "1", "--device", args.device, "--host", host, "--port", str(port),
         "--authkey", authkey.hex()] + list(rest),
        env={**__import__("os").environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
    print("[play] Kit is booting in the child; this takes a minute", flush=True)
    conn = listener.accept()

    server = viser.ViserServer(port=args.port)
    print(f"\n[play] viser on http://{socket.gethostname()}:{args.port}\n", flush=True)

    with server.gui.add_folder("playback"):
        b_pause = server.gui.add_button("pause / resume")
        b_step = server.gui.add_button("single step")
        b_reset = server.gui.add_button("reset episode")
    with server.gui.add_folder("status"):
        md = server.gui.add_markdown("waiting for Kit...")

    # GUI callbacks run on viser's threads while the main loop sits in conn.recv(), and
    # multiprocessing.Connection is NOT thread-safe: sending from a callback races the
    # read and the command is silently lost, which is exactly what "pause does nothing"
    # looks like. Queue instead, and let the one thread that owns the connection send.
    import queue

    outbox: "queue.Queue[dict]" = queue.Queue()
    b_pause.on_click(lambda _: outbox.put({"cmd": "toggle"}))
    b_step.on_click(lambda _: outbox.put({"cmd": "step"}))
    b_reset.on_click(lambda _: outbox.put({"cmd": "reset"}))

    robot_frame = server.scene.add_frame("/robot", show_axes=False)
    obj_frame = server.scene.add_frame("/object", show_axes=False)
    goal_frame = server.scene.add_frame("/goal", show_axes=False)
    urdf: ViserUrdf | None = None
    joint_names: list[str] | None = None
    sliders: dict = {}

    def build_sliders(names, lower, upper) -> None:
        if sliders or not names:
            return
        with server.gui.add_folder("joints (manual)", expand_by_default=False):
            on = server.gui.add_checkbox("drive joints by hand", False)
            zero = server.gui.add_button("all to 0")
            rows = []
            for k, nm in enumerate(names):
                lo, hi = float(lower[k]), float(upper[k])
                if hi - lo < 1e-6:
                    hi = lo + 1e-6
                rows.append((k, server.gui.add_slider(
                    nm, lo, hi, max((hi - lo) / 200.0, 1e-4), min(max(0.0, lo), hi)), lo, hi))
        sliders["on"], sliders["rows"] = on, rows

        def push(_=None, arm=False):
            # Touching a slider arms the override: a control that silently does nothing
            # is indistinguishable from a broken one.
            if arm and not on.value:
                on.value = True
                return
            if not on.value:
                outbox.put({"cmd": "joints", "values": None})
                return
            vals = [0.0] * len(names)
            for k, sl, _lo, _hi in rows:
                vals[k] = float(sl.value)
            outbox.put({"cmd": "joints", "values": vals})

        on.on_update(lambda _=None: push())
        for row in rows:
            row[1].on_update(lambda _=None: push(arm=True))

        @zero.on_click
        def _(_):
            for _k, sl, lo, hi in rows:
                sl.value = min(max(0.0, lo), hi)
            push(arm=True)

    try:
        while child.poll() is None:
            while True:                       # every send happens on this thread
                try:
                    conn.send(outbox.get_nowait())
                except queue.Empty:
                    break
            if not conn.poll(timeout=0.05):
                continue
            # Drain: keep only the LAST frame but handle every other message in order.
            # The child outruns viser by orders of magnitude, so rendering each frame
            # shows a backlog minutes old, and an earlier version that overwrote one
            # slot ate the one-shot ready handshake whenever frames arrived with it.
            batch = [conn.recv()]
            while conn.poll():
                batch.append(conn.recv())
            newest = None
            for m in batch:
                kind = m.get("kind")
                if kind == "frame":
                    newest = m
                elif kind == "status":
                    md.content = m["text"]
                elif kind == "running":
                    try:
                        b_pause.label = "resume" if not m["value"] else "pause"
                    except Exception:
                        pass              # older viser: label is read-only
                elif kind == "ready":
                    joint_names = m["joint_names"]
                    urdf = ViserUrdf(server, pathlib.Path(m["robot_urdf"]),
                                     root_node_name="/robot/hand")
                    cube = pathlib.Path(m["cube_urdf"])
                    _draw_cube(server, cube, "/object/mesh", None)
                    _draw_cube(server, cube, "/goal/mesh", GOAL_OPACITY)
                    build_sliders(joint_names, m["joint_lower"], m["joint_upper"])
                    md.content = "running"
                elif kind == "error":
                    md.content = "**worker failed**\n\n```\n" + m["text"][-1500:] + "\n```"
                    print(m["text"], flush=True)
                    raise SystemExit(1)
            if newest is None or urdf is None:
                continue

            # Zip by NAME. Their joint order is breadth-first and our URDF's is
            # finger-major, so positional zipping would give every joint another
            # joint's angle -- the bug that once made a finger look frozen.
            urdf.update_cfg(dict(zip(newest["robot_joint_names"], newest["robot_joint_pos"])))
            for frame, key in ((robot_frame, "robot_base_pose"), (obj_frame, "object_pose"),
                               (goal_frame, "goal_pose")):
                pose = newest[key]
                frame.position = tuple(pose[:3])
                frame.wxyz = (pose[6], pose[3], pose[4], pose[5])   # xyzw -> wxyz
    finally:
        try:
            conn.send({"cmd": "quit"})
        except Exception:
            pass
        child.terminate()


if __name__ == "__main__":
    main()
