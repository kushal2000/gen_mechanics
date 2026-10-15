"""See the pen_repose goal space in viser, with a hand in context.

Goals come from sample_goals.py (the task's own sampling code), so they are exactly what training draws:
  first goals   one per reset, as a cloud of goal centres coloured by the pen's tilt (blue flat, red vertical),
                plus a number of them drawn as translucent pens
  random walks  each episode's goals in order: its centres joined by a line, and the pens of one chosen walk up to
                a chosen step
The table, the region goal centres can occupy (the box sideways, the ceiling, and the floor for a flat and for an
upright pen), and a hand on the env's dummy wrist (palm down, as the env mounts it) with a slider per joint spanning
its limits -- wrist and fingers -- to pose it among the goals.

    .venv_isaacsim/bin/python experiments/15oct_pen_repose/viser_goals.py [--port 8094]
    # resample from the page (runs sample_goals.py under Kit, about a minute)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
SAMPLES = REPO / "debug_outputs/train_logs/15oct_pen_repose/goal_samples.npz"
FLAT, UPRIGHT = np.array([0.25, 0.45, 0.85]), np.array([0.90, 0.30, 0.15])


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz -> (N, 3, 3)."""
    w, x, y, z = (q[:, i] for i in range(4))
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


def tilt_colors(quat: np.ndarray) -> np.ndarray:
    """(N, 3) colours: blue for a flat pen, red for an upright one (|z| of its long axis)."""
    t = np.abs(quat_to_mat(quat)[:, 2, 0])[:, None]
    return (1 - t) * FLAT + t * UPRIGHT


def pens_mesh(pos: np.ndarray, quat: np.ndarray, length: float, diameter: float, shape: str):
    """One trimesh of every pen (box or capsule along local x), each face coloured by its pen's tilt."""
    import trimesh
    if shape == "box":
        base = trimesh.creation.box(extents=(length, diameter, diameter))
    else:
        base = trimesh.creation.capsule(height=length - diameter, radius=diameter / 2, count=[8, 8])
        base.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    R, cols = quat_to_mat(quat), tilt_colors(quat)
    verts = np.einsum("nij,vj->nvi", R, base.vertices) + pos[:, None, :]
    faces = base.faces[None] + (np.arange(len(pos)) * len(base.vertices))[:, None, None]
    mesh = trimesh.Trimesh(verts.reshape(-1, 3), faces.reshape(-1, 3), process=False)
    fc = np.repeat((cols * 255).astype(np.uint8), len(base.faces), axis=0)
    mesh.visual.face_colors = np.concatenate([fc, np.full((len(fc), 1), 255, np.uint8)], 1)
    return mesh


def reach_below(quat: np.ndarray, length: float, diameter: float, shape: str) -> np.ndarray:
    R = quat_to_mat(quat)
    h, r = length / 2, diameter / 2
    if shape == "box":
        return h * np.abs(R[:, 2, 0]) + r * (np.abs(R[:, 2, 1]) + np.abs(R[:, 2, 2]))
    return (h - r) * np.abs(R[:, 2, 0]) + r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--samples", default=str(SAMPLES))
    args = ap.parse_args()
    import viser
    from viser.extras import ViserUrdf

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    print(f"\n[viser_goals] open http://{socket.gethostname()}:{args.port}\n", flush=True)
    state = {"dirty": True, "data": None, "nodes": {}, "hand": None, "hand_nodes": [], "sliders": {}, "busy": False}

    with server.gui.add_folder("goals"):
        g_mode = server.gui.add_dropdown("show", ("first goals", "random walks"), initial_value="first goals")
        g_n = server.gui.add_slider("first goals drawn as pens", 0, 400, 10, 60)
        g_cloud = server.gui.add_checkbox("first-goal centres (cloud)", True)
        g_walk = server.gui.add_slider("walk", 0, 23, 1, 0)
        g_step = server.gui.add_slider("step (goals so far)", 1, 61, 1, 15)
        g_bounds = server.gui.add_checkbox("goal-centre bounds", True)
        b_resample = server.gui.add_button("resample (runs the task's sampler, ~1 min)")
        md = server.gui.add_markdown("")
    with server.gui.add_folder("hand"):
        g_hand = server.gui.add_dropdown("hand", ("-",))
        g_show_hand = server.gui.add_checkbox("show the hand", True)
        b_pose = server.gui.add_button("reset the pose (wrist default, fingers 0)")
    joint_folder = {"handle": None}

    def load() -> None:
        d = np.load(args.samples)
        meta = json.loads(str(d["meta"]))
        state["data"] = {k: d[k] for k in d.files if k != "meta"} | {"meta": meta}
        names = tuple(h["name"] for h in meta["hands"]) or ("-",)
        g_hand.options = names
        if g_hand.value not in names:
            g_hand.value = "wuji2_left_uniform_handonly" if "wuji2_left_uniform_handonly" in names else names[0]
        g_walk.max = max(len(d["walk_pos"]) - 1, 1)
        g_step.max = d["walk_pos"].shape[1]

    def drop(key: str) -> None:
        for node in state["nodes"].pop(key, []):
            try:
                node.remove()
            except Exception:
                pass

    def put(key: str, node) -> None:
        state["nodes"].setdefault(key, []).append(node)

    def draw_static() -> None:
        m = state["data"]["meta"]
        drop("static")
        sx, sy, sz = m["table_size"]
        put("static", server.scene.add_box("/table", dimensions=(sx, sy, sz), color=(140, 107, 77),
                                           position=(0.0, 0.0, m["table_top"] - sz / 2)))
        if g_bounds.value:
            cx, cy = m["centre_xy"]
            w = m["half_width_xy"]
            L, D, shape = m["pen"]
            top = m["pen_rest_height"] + m["max_height"]
            flat_floor = m["table_top"] + m["min_clearance"] + reach_below(np.array([[1.0, 0, 0, 0]]), L, D, shape)[0]
            upright = np.array([[np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]])     # long axis vertical
            up_floor = m["table_top"] + m["min_clearance"] + reach_below(upright, L, D, shape)[0]
            for name, z, col in (("ceiling", top, (90, 90, 90)), ("floor_flat", flat_floor, FLAT * 255),
                                 ("floor_upright", up_floor, UPRIGHT * 255)):
                c = np.array([[cx - w, cy - w, z], [cx + w, cy - w, z], [cx + w, cy + w, z], [cx - w, cy + w, z]])
                seg = np.stack([c, np.roll(c, -1, 0)], 1)
                put("static", server.scene.add_line_segments(f"/bounds/{name}", points=seg.astype(np.float32),
                                                             colors=tuple(int(v) for v in col), line_width=2.5))
            for x, y in ((cx - w, cy - w), (cx + w, cy - w), (cx + w, cy + w), (cx - w, cy + w)):
                put("static", server.scene.add_line_segments(
                    f"/bounds/edge_{x:+.2f}_{y:+.2f}", points=np.array([[[x, y, flat_floor], [x, y, top]]], np.float32),
                    colors=(120, 120, 120), line_width=1.5))

    def draw_goals() -> None:
        d, m = state["data"], state["data"]["meta"]
        L, D, shape = m["pen"]
        drop("goals")
        if g_mode.value == "first goals":
            pos, quat = d["first_pos"], d["first_quat"]
            if g_cloud.value:
                put("goals", server.scene.add_point_cloud("/goals/cloud", points=pos.astype(np.float32),
                                                          colors=(tilt_colors(quat) * 255).astype(np.uint8),
                                                          point_size=0.003))
            k = int(g_n.value)
            if k:
                put("goals", server.scene.add_mesh_trimesh("/goals/pens", pens_mesh(pos[:k], quat[:k], L, D, shape)))
            z = pos[:, 2] - m["table_top"]
            md.content = (f"**{len(pos)} first goals** (random_first_goal {m['random_first_goal']}): centre "
                          f"{z.min() * 100:.1f}-{z.max() * 100:.1f} cm above the table, within "
                          f"±{m['half_width_xy'] * 100:.0f} cm sideways. Colour: the pen's tilt (blue flat, red "
                          f"upright). Lines: the ceiling (grey) and the lowest a centre can go for a flat (blue) and an "
                          f"upright (red) pen.")
        else:
            wp, wq = d["walk_pos"], d["walk_quat"]
            for i in range(len(wp)):
                put("goals", server.scene.add_spline_catmull_rom(f"/goals/walk{i}", points=wp[i].astype(np.float32),
                                                                 color=(70, 70, 70) if i != int(g_walk.value)
                                                                 else (20, 120, 60), line_width=1.0
                                                                 if i != int(g_walk.value) else 3.0))
            i, k = int(g_walk.value), int(g_step.value)
            put("goals", server.scene.add_mesh_trimesh("/goals/walk_pens", pens_mesh(wp[i, :k], wq[i, :k], L, D, shape)))
            put("goals", server.scene.add_icosphere("/goals/current", radius=0.004, color=(20, 120, 60),
                                                    position=tuple(wp[i, k - 1])))
            steps = np.linalg.norm(np.diff(wp[i], axis=0), axis=-1)
            md.content = (f"**{len(wp)} random walks** of {wp.shape[1]} goals (step <= {m['step_pos'] * 100:.0f} cm, "
                          f"{m['step_deg'][0]:.0f}-{m['step_deg'][1]:.0f}°). Walk {i}: goals 1-{k} drawn as pens; its "
                          f"centre steps {steps.mean() * 100:.1f} cm on average, max {steps.max() * 100:.1f} cm "
                          f"(a turn toward upright can lift it past the 5 cm).")

    def build_hand() -> None:
        for n in state["hand_nodes"]:
            try:
                n.remove()
            except Exception:
                pass
        state["hand_nodes"] = []
        if state["hand"] is not None:
            state["hand"].remove()
            state["hand"] = None
        if joint_folder["handle"] is not None:
            joint_folder["handle"].remove()
            joint_folder["handle"] = None
        state["sliders"] = {}
        m = state["data"]["meta"]
        hand = next((h for h in m["hands"] if h["name"] == g_hand.value), None)
        if hand is None or not g_show_hand.value:
            return
        frame = server.scene.add_frame("/hand", position=(0.0, 0.0, m["table_top"]), show_axes=False)
        state["hand_nodes"].append(frame)
        urdf = ViserUrdf(server, pathlib.Path(hand["urdf"]), root_node_name="/hand/robot")
        state["hand"] = urdf
        limits = urdf.get_actuated_joint_limits()
        defaults = m["wrist_default"]
        fold = server.gui.add_folder("joints (each slider spans the joint's limits)")
        joint_folder["handle"] = fold
        with fold:
            for name in urdf.get_actuated_joint_names():
                lo, hi = limits[name]
                lo = -3.14 if lo is None else lo
                hi = 3.14 if hi is None else hi
                v = float(np.clip(defaults.get(name, 0.0), lo, hi))
                s = server.gui.add_slider(name, float(lo), float(hi), (float(hi) - float(lo)) / 200, v)
                s.on_update(lambda _: pose())
                state["sliders"][name] = (s, v)
        pose()

    def pose() -> None:
        if state["hand"] is not None:
            state["hand"].update_cfg({n: float(s.value) for n, (s, _) in state["sliders"].items()})

    def refresh(_=None) -> None:
        state["dirty"] = True

    for g in (g_mode, g_n, g_cloud, g_walk, g_step, g_bounds):
        g.on_update(refresh)
    g_hand.on_update(lambda _: state.update(rebuild=True))
    g_show_hand.on_update(lambda _: state.update(rebuild=True))

    @b_pose.on_click
    def _(_) -> None:
        for s, v in state["sliders"].values():
            s.value = v

    @b_resample.on_click
    def _(_) -> None:
        if state["busy"]:
            return
        state["busy"] = True
        md.content = "resampling (Kit boots, about a minute)..."

        def run():
            seed = int(time.time()) % 100000
            subprocess.run([sys.executable, str(HERE / "sample_goals.py"), "--seed", str(seed), "--out", args.samples],
                           cwd=REPO, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"},
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            state.update(reload=True, busy=False)
        threading.Thread(target=run, daemon=True).start()

    load()
    state["rebuild"] = True
    while True:
        if state.pop("reload", False):
            load()
            state["dirty"] = True
        if state.pop("rebuild", False):
            build_hand()
        if state["dirty"]:
            state["dirty"] = False
            draw_static()
            draw_goals()
        time.sleep(0.05)


if __name__ == "__main__":
    main()
