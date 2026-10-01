"""Drive any commercial hand's joints with PD-target sliders in the real simulator, and watch for self-collisions.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python assets/urdf/unified_dynamics_commercial_hands/viser_joints.py --port 8082

Pick a hand (uniform or vendor dynamics) and press "Load hand": Isaac boots in a child process with that hand
and no policy. Every slider sets one joint's PD TARGET; physics keeps running, so a joint that a self-collision
(or its torque limit) stops short of its target shows up as tracking error.

Self-collision check, on every frame the simulator sends: the links' collision hulls are posed from the
joint positions PhysX reports, and every pair PhysX actually collides -- not a jointed parent/child pair, not
in the spec's filtered pairs -- is tested for overlap. Overlapping links are outlined in red and listed with
their penetration. The cube is spawned out of the hand and drop / timeout resets are disabled so the hand can
be posed freely.

Reuses coevolution/eval/_play_worker.py (Kit and viser cannot share an interpreter).
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
import tempfile
import time
import xml.etree.ElementTree as ET
from multiprocessing.connection import Listener

import numpy as np
import trimesh

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
WORKER = REPO / "coevolution/eval/_play_worker.py"
TASK = "GenMech-InHandReorient-Direct-v0"
HANDS = ("sharpa", "allegro", "leap", "shadow", "dex3", "tesollo", "wuji2", "xhand")
VENDOR_SPEC = {"sharpa": "sharpa_handonly", **{h: f"{h}_left_handonly" for h in HANDS if h != "sharpa"}}
# Free posing: cube out of the hand, no drop / timeout resets.
POSE_FREELY = ["env.reset.object_in_hand=false", "env.termination.drop_distance_m=1000.0",
               "env.episode_length_s=100000.0", "env.termination.episode_length=6000000"]
UNIFORM_FRICTION = ["env.assets.robot_friction=0.5", "env.assets.finger_tip_friction=0.5",
                    "env.assets.object_friction=0.5"]
MIN_OVERLAP_M = 0.5e-3


# ----------------------------------------------------------------------------- collision geometry

def _filtered_pairs(hand: str, uniform: bool) -> set[frozenset]:
    """Pairs PhysX does NOT collide: the spec's adjacency filter (jointed pairs are added separately)."""
    if hand == "sharpa":
        import importlib.util
        import types
        pkg = "isaacsimenvs.pose_reaching_6d.scene_utils.robots"
        for n in ("isaacsimenvs.pose_reaching_6d.scene_utils", pkg, pkg + ".adjacency"):
            if n not in sys.modules:
                m = types.ModuleType(n); m.__path__ = []; sys.modules[n] = m
        for mod in ("sharpa_iiwa14", "sharpa_handonly"):
            sp = importlib.util.spec_from_file_location(
                f"{pkg}.adjacency.{mod}", REPO / f"isaacsimenvs/pose_reaching_6d/scene_utils/robots/adjacency/{mod}.py")
            m = importlib.util.module_from_spec(sp); sys.modules[sp.name] = m; sp.loader.exec_module(m)
        adj = m.SHARPA_HANDONLY_ADJACENT_LINKS
    else:
        d = HERE if uniform else REPO / "assets/urdf/unified_commercial_hands"
        adj = json.loads((d / hand / f"{hand}_left.spec.json").read_text())["adjacent_links"]
    return {frozenset((a, b)) for a, bs in adj.items() for b in bs}


class Collider:
    """Collision hulls per moving body (fixed children merged in, as Isaac's importer does)."""

    def __init__(self, urdf_path: pathlib.Path, filtered: set[frozenset]):
        import yourdfpy
        self.robot = yourdfpy.URDF.load(str(urdf_path), build_scene_graph=True, load_meshes=False)
        joints = list(self.robot.joint_map.values())
        parent = {j.child: (j.parent, j.type) for j in joints}

        def body(link):
            while link in parent and parent[link][1] == "fixed":
                link = parent[link][0]
            return link
        self.body = {l: body(l) for l in self.robot.link_map}
        root = ET.parse(urdf_path).getroot()
        base = urdf_path.parent
        self.hulls: dict[str, list[trimesh.Trimesh]] = {}
        for link in root.findall("link"):
            for col in link.findall("collision"):
                g = col.find("geometry")
                if g is None or g.find("mesh") is None and g.find("box") is None:
                    continue
                if g.find("mesh") is not None:
                    m = trimesh.load(base / g.find("mesh").get("filename"), force="mesh")
                    if g.find("mesh").get("scale"):
                        m.apply_scale([float(v) for v in g.find("mesh").get("scale").split()])
                else:
                    m = trimesh.creation.box([float(v) for v in g.find("box").get("size").split()])
                o = col.find("origin")
                T = np.eye(4)
                if o is not None:
                    T = trimesh.transformations.euler_matrix(*[float(v) for v in o.get("rpy", "0 0 0").split()])
                    T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
                h = m.convex_hull
                h.apply_transform(T)
                self.hulls.setdefault(link.get("name"), []).append(h)
        jointed = {frozenset((self.body[j.parent], self.body[j.child])) for j in joints if j.type != "fixed"}
        bodies = sorted({self.body[l] for l in self.hulls})
        skip = jointed | {frozenset((self.body[a], self.body[b])) for a, b in (tuple(p) for p in filtered if len(p) == 2)}
        self.pairs = [(a, b) for i, a in enumerate(bodies) for b in bodies[i + 1:] if frozenset((a, b)) not in skip]

    def world_hulls(self, cfg: dict) -> dict[str, list[trimesh.Trimesh]]:
        self.robot.update_cfg({k: v for k, v in cfg.items() if k in self.robot.joint_map})
        out = {}
        for link, hs in self.hulls.items():
            T = self.robot.get_transform(link)
            out.setdefault(self.body[link], []).extend(h.copy().apply_transform(T) for h in hs)
        return out

    @staticmethod
    def _depth(A, B) -> float:
        if not (np.all(A.bounds[0] <= B.bounds[1]) and np.all(B.bounds[0] <= A.bounds[1])):
            return 0.0
        d = 0.0
        for P, Q in ((A, B), (B, A)):
            n, o = P.face_normals, P.triangles[:, 0]
            s = Q.vertices @ n.T - np.einsum("fk,fk->f", o, n)[None]
            inside = s.max(1) < 0
            if inside.any():
                d = max(d, float((-s.max(1))[inside].max()))
        return d

    def overlaps(self, world) -> list[tuple[str, str, float]]:
        hits = []
        for a, b in self.pairs:
            d = max((self._depth(A, B) for A in world.get(a, []) for B in world.get(b, [])), default=0.0)
            if d > MIN_OVERLAP_M:
                hits.append((a, b, d))
        return sorted(hits, key=lambda t: -t[2])


# ----------------------------------------------------------------------------- Kit child

class Session:
    def __init__(self, hand: str, uniform: bool, device: str):
        self.hand, self.uniform = hand, uniform
        spec = f"{hand}_left_uniform_handonly" if uniform else VENDOR_SPEC[hand]
        authkey = secrets.token_bytes(16)
        self.listener = Listener(("127.0.0.1", 0), authkey=authkey)
        host, port = self.listener.address
        extra = POSE_FREELY + (UNIFORM_FRICTION if uniform else [])
        self.child = subprocess.Popen(
            [sys.executable, str(WORKER), "--task", TASK, "--checkpoint", "", "--run-dir", "",
             "--population", spec, "--num-envs", "1", "--expl-coef", "0", "--device", device,
             "--host", host, "--port", str(port), "--authkey", authkey.hex(), *extra],
            env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}, cwd=str(REPO))
        self.conn = None

    def accept(self, timeout_s=600.0) -> bool:
        self.listener._listener._socket.settimeout(2.0)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.child.poll() is not None:
                return False
            try:
                self.conn = self.listener.accept()
                return True
            except (socket.timeout, TimeoutError):
                continue
        return False

    def close(self):
        try:
            if self.conn is not None:
                self.conn.send({"cmd": "quit"})
        except Exception:
            pass
        for _ in range(20):
            if self.child.poll() is not None:
                break
            time.sleep(0.25)
        if self.child.poll() is None:
            self.child.kill(); self.child.wait()
        try:
            self.listener.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------- viewer

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--hand", default="allegro", choices=HANDS)
    args = ap.parse_args()

    import viser
    from viser.extras import ViserUrdf

    server = viser.ViserServer(port=args.port)
    print(f"\n[joints] viser on http://{socket.gethostname()}:{args.port}\n", flush=True)
    with server.gui.add_folder("hand"):
        g_hand = server.gui.add_dropdown("hand", HANDS, initial_value=args.hand)
        g_uniform = server.gui.add_checkbox("uniform dynamics", True)
        b_load = server.gui.add_button("Load hand")
    with server.gui.add_folder("collisions"):
        g_check = server.gui.add_checkbox("check self-collisions", True)
        md_col = server.gui.add_markdown("")
    with server.gui.add_folder("status"):
        md = server.gui.add_markdown("pick a hand, then Load hand")

    outbox: "queue.Queue[dict]" = queue.Queue()
    load_req: "queue.Queue[tuple]" = queue.Queue()
    b_load.on_click(lambda _: load_req.put((str(g_hand.value), bool(g_uniform.value))))
    robot_root = server.scene.add_frame("/robot_root", show_axes=False)
    st: dict = {"gui": [], "sliders": [], "red": []}
    sess: Session | None = None

    def clear():
        for h in st["gui"] + st["red"]:
            try:
                h.remove()
            except Exception:
                pass
        for n in ("/robot_root/robot",):
            try:
                server.scene.remove_by_name(n)
            except Exception:
                pass
        st.update({"gui": [], "sliders": [], "red": [], "robot": None, "collider": None, "names": None})

    def push_targets(_=None):
        if not st["sliders"]:
            return
        vals = [0.0] * len(st["names"])
        for k, sl in st["sliders"]:
            vals[k] = float(sl.value)
        outbox.put({"cmd": "joints", "values": vals})

    def build_sliders(names, lo, hi):
        folder = server.gui.add_folder("PD targets (rad)")
        st["gui"].append(folder)
        with folder:
            b_zero = server.gui.add_button("all to home (0, clamped)")
            b_open = server.gui.add_button("all to lower limit")
            b_close = server.gui.add_button("all to upper limit")
            st["gui"] += [b_zero, b_open, b_close]
            for k, n in enumerate(names):
                a, b = float(lo[k]), max(float(hi[k]), float(lo[k]) + 1e-4)
                sl = server.gui.add_slider(n, a, b, (b - a) / 200.0, min(max(0.0, a), b))
                sl.on_update(push_targets)
                st["sliders"].append((k, sl)); st["gui"].append(sl)

        def set_all(f):
            for k, sl in st["sliders"]:
                sl.value = f(float(lo[k]), float(hi[k]))
            push_targets()
        b_zero.on_click(lambda _: set_all(lambda a, b: min(max(0.0, a), b)))
        b_open.on_click(lambda _: set_all(lambda a, b: a))
        b_close.on_click(lambda _: set_all(lambda a, b: b))

    def load(hand, uniform):
        nonlocal sess
        if sess is not None:
            md.content = "stopping the previous hand..."
            sess.close(); sess = None
        clear()
        md.content = f"loading **{hand}** ({'uniform' if uniform else 'vendor'} dynamics): Kit boots in ~1 min"
        sess = Session(hand, uniform, args.device)
        st["filtered"] = _filtered_pairs(hand, uniform)
        if not sess.accept():
            md.content = "**worker died before connecting** -- see the terminal"
            sess.close(); sess = None

    load_req.put((args.hand, True))
    last_check = 0.0
    try:
        while True:
            try:
                req = load_req.get_nowait()
                while not load_req.empty():
                    req = load_req.get_nowait()
                load(*req)
                while not outbox.empty():
                    outbox.get_nowait()
            except queue.Empty:
                pass
            if sess is None or sess.conn is None:
                time.sleep(0.1); continue
            if sess.child.poll() is not None:
                md.content = "**worker exited** -- see the terminal; load again"
                sess.close(); sess = None; continue
            try:
                while not outbox.empty():
                    sess.conn.send(outbox.get_nowait())
                if not sess.conn.poll(timeout=0.05):
                    continue
                batch = [sess.conn.recv()]
                while sess.conn.poll():
                    batch.append(sess.conn.recv())
            except (EOFError, OSError):
                md.content = "**worker closed the connection** -- load again"
                sess.close(); sess = None; continue
            frame = None
            for m in batch:
                k = m.get("kind")
                if k == "frame":
                    frame = m
                elif k == "error":
                    md.content = "**worker failed**\n\n```\n" + m["text"][-1500:] + "\n```"
                    print(m["text"], flush=True)
                elif k == "ready":
                    st["names"] = m["joint_names"]
                    urdf = pathlib.Path(m["urdf_path"])
                    robot_root.position = tuple(m["base_pos"]); robot_root.wxyz = tuple(m["base_rot"])
                    st["robot"] = ViserUrdf(server, urdf, root_node_name="/robot_root/robot")
                    st["collider"] = Collider(urdf, st["filtered"])
                    st["lo"], st["hi"] = m["joint_lower"], m["joint_upper"]
                    build_sliders(st["names"], st["lo"], st["hi"])
                    push_targets()
                    md.content = (f"**{sess.hand}** ready: {len(st['names'])} joints, "
                                  f"{len(st['collider'].pairs)} body pairs that PhysX collides")
            if frame is None or st.get("robot") is None:
                continue
            q = frame["joint_pos"][0]
            cfg = dict(zip(st["names"], q))
            st["robot"].update_cfg(cfg)
            if time.time() - last_check < 0.2:
                continue
            last_check = time.time()
            # Tracking: a joint that sits well short of its target is blocked (contact or torque limit).
            tgt = {st["names"][k]: float(sl.value) for k, sl in st["sliders"]}
            err = sorted(((n, tgt[n] - cfg[n]) for n in tgt), key=lambda t: -abs(t[1]))
            lines = [f"**max tracking error** {err[0][0]} {np.degrees(err[0][1]):+.1f} deg"] if err else []
            for h in st["red"]:
                try:
                    h.remove()
                except Exception:
                    pass
            st["red"] = []
            if g_check.value and st.get("collider") is not None:
                world = st["collider"].world_hulls(cfg)
                hits = st["collider"].overlaps(world)
                for a, b, d in hits[:12]:
                    for name in (a, b):
                        for j, h in enumerate(world.get(name, [])):
                            st["red"].append(server.scene.add_mesh_simple(
                                f"/robot_root/overlap/{name}_{j}_{len(st['red'])}", vertices=h.vertices,
                                faces=h.faces, color=(230, 60, 60), opacity=0.45))
                md_col.content = ("**no self-collisions**" if not hits else
                                  f"**{len(hits)} colliding pair(s)**  \n" +
                                  "  \n".join(f"{a} × {b}: {1000 * d:.1f} mm" for a, b, d in hits[:12]))
            blocked = [f"{n} {np.degrees(e):+.0f}°" for n, e in err if abs(e) > np.radians(5)][:8]
            md.content = "  \n".join(lines + (["**short of target (>5°):** " + ", ".join(blocked)] if blocked else []))
    except KeyboardInterrupt:
        pass
    finally:
        if sess is not None:
            sess.close()


if __name__ == "__main__":
    main()
