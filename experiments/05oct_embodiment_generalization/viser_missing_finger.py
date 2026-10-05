"""Live viewer: the unified policy vs the Wuji-only policy on Wuji v2 with a finger missing, picked in a dropdown.

Left: Unified Multi-Embodiment (40k epochs). Right: Wuji Only (~32k epochs). Both drive the same variant -- the
full hand or the hand with one finger removed (make_missing_fingers.py) -- zero-shot: the joint transformer is
rebuilt for the variant's joints with the same weights. Changing the variant rebuilds the scene, which Kit
cannot do in place, so the simulation worker (viser_compare.py --worker) is restarted for it (about a minute).

    .venv_isaacsim/bin/python experiments/05oct_embodiment_generalization/viser_missing_finger.py --port 8081
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
import time
from multiprocessing.connection import Listener

REPO = pathlib.Path(__file__).resolve().parents[2]
WORKER = REPO / "experiments/01oct_unified_rl/viser_compare.py"
sys.path.insert(0, str(WORKER.parent))
from viser_compare import _face_cube  # noqa: E402

L = REPO / "debug_outputs/train_logs"
FINGERS = ("none", "thumb", "index", "middle", "ring", "pinky")


def first(pattern: str) -> str:
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(f"no checkpoint matches {pattern}")
    return hits[0]


def policies() -> list[str]:
    uni = first(f"{L}/01oct_unified_rl/0_scale*_c04_*948509*/rank_0/*/nn/last_*ep_10000_*.pth")
    wuji = first(f"{L}/01oct_uniform_dynamics/*969180*/rank_0/*/nn/last_*ep_8050_*.pth")
    return [f"Unified Multi-Embodiment (40k)={uni}", f"Wuji Only (~32k)={wuji}"]


def spec_for(finger: str) -> str:
    return "wuji2" if finger == "none" else f"wuji2_left_uniform_handonly_no_{finger}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--spacing", type=float, default=0.32)
    ap.add_argument("--finger", default="none", choices=FINGERS, help="variant to start with")
    args = ap.parse_args()

    import viser
    from viser.extras import ViserUrdf

    pols = policies()
    for p in pols:
        print(f"[fingers] {p}", flush=True)
    authkey = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    host, port = listener.address

    server = viser.ViserServer(port=args.port)
    print(f"\n[fingers] viser on http://{socket.gethostname()}:{args.port}\n", flush=True)
    outbox: "queue.Queue[dict]" = queue.Queue()
    with server.gui.add_folder("hand"):
        dd = server.gui.add_dropdown("missing finger", FINGERS, initial_value=args.finger)
        md_hand = server.gui.add_markdown("")
    with server.gui.add_folder("playback"):
        b_pause = server.gui.add_button("pause / resume")
        b_reset = server.gui.add_button("reset all")
        s_speed = server.gui.add_slider("speed (x real time)", 0.1, 2.0, 0.05, 1.0)
        g_labels = server.gui.add_checkbox("show labels", True)
    md = server.gui.add_markdown("")
    b_pause.on_click(lambda _: outbox.put({"cmd": "toggle"}))
    b_reset.on_click(lambda _: outbox.put({"cmd": "reset"}))
    s_speed.on_update(lambda _: outbox.put({"cmd": "speed", "value": float(s_speed.value)}))
    wanted = [args.finger]
    dd.on_update(lambda _: wanted.__setitem__(0, str(dd.value)))

    state = {"child": None, "conn": None, "finger": None, "info": None, "nodes": [], "urdfs": [], "objs": [],
             "goals": [], "labels": []}

    def stop():
        if state["conn"] is not None:
            try:
                state["conn"].send({"cmd": "quit"})
            except Exception:
                pass
        if state["child"] is not None:
            time.sleep(0.5)
            if state["child"].poll() is None:
                state["child"].kill()
            state["child"].wait()
        for name in state["nodes"]:
            try:
                server.scene.remove_by_name(name)
            except Exception:
                pass
        state.update(child=None, conn=None, info=None, nodes=[], urdfs=[], objs=[], goals=[], labels=[])

    def start(finger):
        stop()
        md_hand.content = f"**loading {finger}** (Kit boots in about a minute)"
        cmd = [sys.executable, str(WORKER), "--worker", "--hand", spec_for(finger), "--host", host,
               "--wport", str(port), "--authkey", authkey.hex()]
        for p in pols:
            cmd += ["--policy", p]
        state["child"] = subprocess.Popen(cmd, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
        state["conn"] = listener.accept()
        state["finger"] = finger
        print(f"[fingers] worker up for {finger}", flush=True)

    try:
        while True:
            if wanted[0] != state["finger"]:
                start(wanted[0])
                state["dead"] = False
            if state.get("dead"):
                time.sleep(0.2)
                continue
            conn = state["conn"]
            while True:
                try:
                    conn.send(outbox.get_nowait())
                except queue.Empty:
                    break
            if not conn.poll(timeout=0.05):
                if state["child"].poll() is not None:     # crashed: wait for the user to pick a variant
                    md_hand.content = "**worker exited** -- pick a variant in the dropdown to restart"
                    state["dead"] = True
                continue
            batch = [conn.recv()]
            while conn.poll():
                batch.append(conn.recv())
            frame = None
            for m in batch:
                if m["kind"] == "frame":
                    frame = m
                elif m["kind"] == "error":
                    md.content = "**worker failed**\n\n```\n" + m["text"][-1500:] + "\n```"
                    print(m["text"], flush=True)
                elif m["kind"] == "ready":
                    state["info"] = m
                    n = len(m["labels"])
                    for i in range(n):
                        off = (args.spacing * (i - (n - 1) / 2), 0.0, 0.0)
                        server.scene.add_frame(f"/p{i}", position=off, show_axes=False)
                        server.scene.add_frame(f"/p{i}/base", position=tuple(m["base_pos"]),
                                               wxyz=tuple(m["base_rot"]), show_axes=False)
                        state["urdfs"].append(ViserUrdf(server, pathlib.Path(m["urdf"]),
                                                        root_node_name=f"/p{i}/base/robot"))
                        state["objs"].append(_face_cube(server, f"/p{i}/obj", m["cube"], opacity=1.0))
                        state["goals"].append(_face_cube(server, f"/p{i}/goal", m["cube"], opacity=0.35))
                        state["labels"].append(server.scene.add_label(f"/p{i}/label", m["labels"][i]))
                        state["nodes"].append(f"/p{i}")
                    md_hand.content = (f"Wuji v2, **{'full hand' if state['finger'] == 'none' else 'no ' + state['finger']}**"
                                       f" ({len(m['joints'])} joints)")
                    md.content = " vs ".join(m["labels"])
            info = state["info"]
            if frame is None or info is None:
                continue
            for i in range(len(state["urdfs"])):
                state["urdfs"][i].update_cfg(dict(zip(info["joints"], frame["q"][i])))
                state["objs"][i].position = tuple(frame["obj_pos"][i])
                state["objs"][i].wxyz = tuple(frame["obj_quat"][i])
                state["goals"][i].position = tuple(frame["goal_pos"][i])
                state["goals"][i].wxyz = tuple(frame["goal_quat"][i])
                gp = frame["goal_pos"][i]
                state["labels"][i].position = (gp[0], gp[1], gp[2] + 0.09)
                state["labels"][i].text = (f"{info['labels'][i]}  ·  {frame['goals'][i]} goals  ·  "
                                           f"{frame['drops'][i]} drops")
                state["labels"][i].visible = bool(g_labels.value)
    except KeyboardInterrupt:
        pass
    finally:
        stop()


if __name__ == "__main__":
    main()
