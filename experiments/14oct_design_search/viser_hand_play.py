"""Play with primitive hands (primitive_hands.py) in the in-hand olympics env, in viser: pick a built hand (or build
one from bearings, or at random) and it loads on the 6-DOF dummy wrist, palm down above the table, with the pen.
Every joint has a slider (grouped: wrist, then finger 0, 1, ...) that the env's own action pipeline steers it to,
and every joint's position limits can be edited live, in the sim.

    .venv_isaacsim/bin/python experiments/14oct_design_search/viser_hand_play.py [--port 8093] [--hand <name>]
    # then open http://<node>:8093

The simulation is the olympics viewer's worker (isaacsimenvs/inhand_olympics/tests/viser_hand_table.py --worker):
Kit and viser cannot share an interpreter, so the sim runs in a child and streams joint positions back. Loading
another hand restarts the child (about a minute).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import queue
import secrets
import socket
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Listener

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKER = REPO / "isaacsimenvs/inhand_olympics/tests/viser_hand_table.py"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WORKER.parent))
import primitive_hands as ph  # noqa: E402
from viser_hand_table import _add_pen, _group  # noqa: E402  (Kit-free helpers)


def built(out_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """name -> spec JSON of every built primitive hand."""
    return {p.name[: -len(".spec.json")]: p for p in sorted(out_dir.glob("*.spec.json"))}


class Sim:
    """The worker for one hand, connecting back on ``listener`` (accepted on a thread, so a child that dies while
    booting cannot hang the page)."""

    def __init__(self, listener, authkey: bytes, spec: pathlib.Path, pen_shape: str):
        host, port = listener.address
        self.proc = subprocess.Popen(
            [sys.executable, str(WORKER), "--worker", "--hand", str(spec.relative_to(REPO)), "--pen-shape", pen_shape,
             "--host", host, "--wport", str(port), "--authkey", authkey.hex()],
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--hand", default="", help="the built hand to load first (default: the first one)")
    ap.add_argument("--pen-shape", default="box", choices=("box", "capsule"))
    ap.add_argument("--out", default=str(ph.OUT), help="the folder of built hands")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out)
    if not built(out_dir):
        ph.build(ph.PrimitiveHand(bearings_deg=(0.0, 90.0, 180.0, 270.0)), out_dir)

    import signal

    import viser
    from viser.extras import ViserUrdf

    # A kill unwinds like Ctrl-C, so the finally below stops the Kit child instead of orphaning it.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    authkey = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    server = viser.ViserServer(port=args.port)
    print(f"\n[hand-play] open http://{socket.gethostname()}:{args.port}\n", flush=True)
    outbox: "queue.Queue[dict]" = queue.Queue()
    requests: "queue.Queue[str]" = queue.Queue()

    hands = built(out_dir)
    first = args.hand if args.hand in hands else next(iter(hands))
    with server.gui.add_folder("hand"):
        g_hand = server.gui.add_dropdown("hand", tuple(hands), initial_value=first)
        b_load = server.gui.add_button("load (restarts the sim, about a minute)")
        md_hand = server.gui.add_markdown("")
    with server.gui.add_folder("build a hand", expand_by_default=False):
        g_bear = server.gui.add_text("bearings (deg)", "0,90,180,270")
        b_build = server.gui.add_button("build from bearings")
        g_gap = server.gui.add_number("min gap (deg)", 25.0, step=5.0)
        b_rand = server.gui.add_button("build a random hand")
        md_build = server.gui.add_markdown("")
    md = server.gui.add_markdown("")
    b_reset = server.gui.add_button("reset (sim and sliders)")
    b_goal = server.gui.add_button("put the pen on its goal")
    b_urdf = server.gui.add_button("restore the URDF's joint limits")

    def describe(name: str) -> None:
        d = json.loads((out_dir / f"{name}.design.json").read_text())
        h = ph.PrimitiveHand.from_json(d)
        md_hand.content = (f"**{h.name}**  \nbearings {', '.join(f'{b:g}' for b in h.bearings_deg)} deg · gaps "
                           f"{', '.join(f'{g:g}' for g in h.gaps_deg())} deg  \n{len(h.bearings_deg)} fingers x "
                           f"{h.modules_per_finger} capsules ({h.module_length * 100:g} cm long, "
                           f"{2 * h.module_radius * 100:g} cm wide) · palm {2 * h.palm_radius * 100:g} cm across, "
                           f"{h.palm_thickness * 100:g} cm thick")

    def add_hand(h: ph.PrimitiveHand) -> None:
        h = h.canonical()
        ph.build(h, out_dir)
        hands.clear()
        hands.update(built(out_dir))
        g_hand.options = tuple(hands)
        g_hand.value = h.name
        md_build.content = f"built **{h.name}** -- press load"

    @b_build.on_click
    def _(_) -> None:
        try:
            add_hand(ph.PrimitiveHand(bearings_deg=tuple(float(x) for x in str(g_bear.value).split(",") if x.strip())))
        except Exception as e:
            md_build.content = f"**not built:** {e}"

    rand = {"seed": 1000}

    @b_rand.on_click
    def _(_) -> None:
        for _ in range(50):
            rand["seed"] += 1
            h = ph.sample(1, seed=rand["seed"], min_gap_deg=float(g_gap.value))[0]
            if h.name not in hands:
                add_hand(h)
                return
        md_build.content = "**no new hand found** -- try a smaller min gap"

    g_hand.on_update(lambda _: describe(str(g_hand.value)))
    b_load.on_click(lambda _: requests.put(str(g_hand.value)))
    b_goal.on_click(lambda _: outbox.put({"cmd": "pen_to_goal"}))
    describe(first)
    requests.put(first)

    sim: Sim | None = None
    view: dict = {}          # per loaded hand: scene nodes, GUI folders, sliders, limit boxes
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
        """The scene and the joint panels for a freshly loaded hand (the worker's ready message)."""
        sx, sy, sz = m["table_size"]
        nodes = [server.scene.add_box(f"/table{load_id}", dimensions=(sx, sy, sz), color=(140, 107, 77),
                                      position=(0.0, 0.0, m["table_top"] - sz / 2)),
                 server.scene.add_frame(f"/base{load_id}", position=tuple(m["base_pos"]), show_axes=False)]
        urdf = ViserUrdf(server, pathlib.Path(m["urdf"]), root_node_name=f"/base{load_id}/robot")
        pen = _add_pen(server, f"/pen{load_id}", *m["pen"], shape=m["pen_shape"])
        goal = _add_pen(server, f"/goal{load_id}", *m["pen"], opacity=0.3, shape=m["pen_shape"])
        nodes += [pen, goal]
        sliders, limit_boxes, urdf_limits, folders = {}, {}, {}, []
        names = m["joints"]
        groups = sorted({_group(n) for n in names}, key=lambda g: (g != "wrist", g))
        for g in groups:
            with server.gui.add_folder(g, expand_by_default=(g == "wrist")) as fold:
                folders.append(fold)
                for i, n in enumerate(names):
                    if _group(n) != g:
                        continue
                    lo, hi = m["lower"][i], m["upper"][i]
                    s = server.gui.add_slider(n, lo, hi, (hi - lo) / 200, m["default"][i])
                    s.on_update(lambda _, n=n, s=s: outbox.put({"cmd": "target", "joint": n, "value": float(s.value)}))
                    sliders[n] = (s, m["default"][i])
                    urdf_limits[n] = (lo, hi)
                with server.gui.add_folder(f"{g} limits", expand_by_default=False):
                    for i, n in enumerate(names):
                        if _group(n) != g:
                            continue
                        lo, hi = m["lower"][i], m["upper"][i]
                        b_lo = server.gui.add_number(f"{n} lower", round(lo, 4), step=0.01)
                        b_hi = server.gui.add_number(f"{n} upper", round(hi, 4), step=0.01)
                        limit_boxes[n] = (b_lo, b_hi)

                        def _limits(_, n=n, b_lo=b_lo, b_hi=b_hi):
                            lo, hi = float(b_lo.value), float(b_hi.value)
                            if hi <= lo:
                                md.content = f"**{n}: upper must be above lower**"
                                return
                            s = sliders[n][0]
                            s.min, s.max, s.step = lo, hi, (hi - lo) / 200
                            s.value = min(max(float(s.value), lo), hi)
                            outbox.put({"cmd": "limit", "joint": n, "lower": lo, "upper": hi})
                        b_lo.on_update(_limits)
                        b_hi.on_update(_limits)
        view.update(urdf=urdf, pen=pen, goal=goal, nodes=nodes, folders=folders, sliders=sliders,
                    limit_boxes=limit_boxes, urdf_limits=urdf_limits, info=m)

    @b_reset.on_click
    def _(_) -> None:
        for s, d in view.get("sliders", {}).values():
            s.value = d
        outbox.put({"cmd": "reset"})

    @b_urdf.on_click
    def _(_) -> None:
        for n, (lo, hi) in view.get("urdf_limits", {}).items():
            b_lo, b_hi = view["limit_boxes"][n]
            b_lo.value, b_hi.value = round(lo, 4), round(hi, 4)       # each fires _limits

    t_scan = 0.0
    try:
        while True:
            if time.time() - t_scan > 5.0:          # hands built elsewhere (primitive_hands.py) join the dropdown
                t_scan = time.time()
                now = built(out_dir)
                if set(now) != set(hands):
                    cur = str(g_hand.value)
                    hands.clear()
                    hands.update(now)
                    g_hand.options = tuple(hands)
                    if cur in hands:
                        g_hand.value = cur
            try:
                name = requests.get_nowait()
                if sim is not None:
                    sim.stop()
                clear_view()
                while not outbox.empty():
                    outbox.get_nowait()
                load_id += 1
                sim = Sim(listener, authkey, hands[name], args.pen_shape)
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
            view["pen"].position, view["pen"].wxyz = tuple(frame["pen_pos"]), tuple(frame["pen_quat"])
            view["goal"].position, view["goal"].wxyz = tuple(frame["goal_pos"]), tuple(frame["goal_quat"])
            ended = f"  \n**episode ended: {', '.join(frame['ended'])}**" if frame["ended"] else ""
            md.content = (f"task **{frame['task']}** · goals hit {frame['successes']} · goal distance "
                          f"{frame['goal_dist'] * 1000:.0f} mm  \nworst |q - target| {frame['err']:.3f}{ended}")
    except KeyboardInterrupt:
        pass
    finally:
        if sim is not None:
            sim.stop()


if __name__ == "__main__":
    main()
