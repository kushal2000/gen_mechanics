"""One pen_repose goal you pose with sliders, inside the limits the task samples goals in, with a hand in context.

The goal: a centre and an orientation. Sideways the centre stays in the task's box (centre_xy +- half_width_xy); its
height runs from the floor for the CURRENT orientation (the pen's lowest point min_clearance above the table -- a flat
pen can sit low, an upright one stands higher; the slider's lower end moves as you turn the pen) up to the ceiling
(pen_rest_height + max_height). Orientation: the long axis's heading and tilt (any direction a goal can take), and roll
about it, which the keypoint success test can't see. The pen is coloured by tilt (blue flat, red upright); the box
outline is drawn at the current floor. A hand on the env's dummy wrist (mounted at the table top, as scene.py does), a
slider per joint spanning its limits -- wrist and fingers.

Limits and hands come from sample_goals.py's npz (the task's config and the wrist URDFs it builds):
    .venv_isaacsim/bin/python experiments/15oct_pen_repose/viser_goals.py [--port 8094]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import socket
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
SAMPLES = REPO / "debug_outputs/train_logs/15oct_pen_repose/goal_samples.npz"
FLAT, UPRIGHT = np.array([0.25, 0.45, 0.85]), np.array([0.90, 0.30, 0.15])


def axis_angle_quat(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, float)
    return np.array([np.cos(angle / 2), *(np.sin(angle / 2) * a)])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def goal_quat(heading_deg: float, tilt_deg: float, roll_deg: float) -> np.ndarray:
    """The pen's long axis (local x) at ``heading`` about z and ``tilt`` up from the table, rolled about itself."""
    return quat_mul(quat_mul(axis_angle_quat((0, 0, 1), np.radians(heading_deg)),
                             axis_angle_quat((0, 1, 0), -np.radians(tilt_deg))),
                    axis_angle_quat((1, 0, 0), np.radians(roll_deg)))


def reach_below(q: np.ndarray, length: float, diameter: float, shape: str) -> float:
    """As PenRepose.reach_below: how far the pen reaches below its centre at orientation q."""
    R = quat_to_mat(q)
    h, r = length / 2, diameter / 2
    if shape == "box":
        return h * abs(R[2, 0]) + r * (abs(R[2, 1]) + abs(R[2, 2]))
    return (h - r) * abs(R[2, 0]) + r


def pen_mesh(length: float, diameter: float, shape: str, color: np.ndarray):
    import trimesh
    if shape == "box":
        m = trimesh.creation.box(extents=(length, diameter, diameter))
    else:
        m = trimesh.creation.capsule(height=length - diameter, radius=diameter / 2, count=[16, 16])
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    m.visual.face_colors = np.append((color * 255).astype(np.uint8), 255)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--samples", default=str(SAMPLES))
    args = ap.parse_args()
    import viser
    from viser.extras import ViserUrdf

    m = json.loads(str(np.load(args.samples)["meta"]))
    L, D, shape = m["pen"]
    top, (cx, cy), w = m["table_top"], m["centre_xy"], m["half_width_xy"]
    ceiling = m["pen_rest_height"] + m["max_height"]
    flat_floor = top + m["min_clearance"] + reach_below(goal_quat(0, 0, 0), L, D, shape)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    print(f"\n[viser_goals] open http://{socket.gethostname()}:{args.port}\n", flush=True)
    sx, sy, sz = m["table_size"]
    server.scene.add_box("/table", dimensions=(sx, sy, sz), color=(140, 107, 77), position=(0.0, 0.0, top - sz / 2))
    state = {"dirty": True, "rebuild": True, "hand": None, "sliders": {}, "folder": None, "nodes": []}

    with server.gui.add_folder("goal (sliders span the task's limits)"):
        g_x = server.gui.add_slider("x (cm)", (cx - w) * 100, (cx + w) * 100, 0.1, cx * 100)
        g_y = server.gui.add_slider("y (cm)", (cy - w) * 100, (cy + w) * 100, 0.1, cy * 100)
        g_head = server.gui.add_slider("heading (deg)", -180.0, 180.0, 1.0, 0.0)
        g_tilt = server.gui.add_slider("tilt up from flat (deg)", -90.0, 90.0, 1.0, 0.0)
        g_roll = server.gui.add_slider("roll about the axis (deg)", -180.0, 180.0, 1.0, 0.0)
        g_z = server.gui.add_slider("centre height above table (cm)", (flat_floor - top) * 100, (ceiling - top) * 100,
                                    0.1, (m["pen_rest_height"] + 0.08 - top) * 100)
        g_box = server.gui.add_checkbox("show the bounds", True)
        md = server.gui.add_markdown("")
    with server.gui.add_folder("hand"):
        names = tuple(h["name"] for h in m["hands"])
        g_hand = server.gui.add_dropdown("hand", names, initial_value="wuji2_left_uniform_handonly"
                                         if "wuji2_left_uniform_handonly" in names else names[0])
        g_show = server.gui.add_checkbox("show the hand", True)
        b_pose = server.gui.add_button("reset the pose (wrist default, fingers 0)")

    def draw() -> None:
        for n in state["nodes"]:
            n.remove()
        state["nodes"] = []
        q = goal_quat(g_head.value, g_tilt.value, g_roll.value)
        floor = top + m["min_clearance"] + reach_below(q, L, D, shape)
        g_z.min = (floor - top) * 100                         # the lowest this orientation's centre may go
        if g_z.value < g_z.min:
            g_z.value = g_z.min
        pos = np.array([g_x.value / 100, g_y.value / 100, top + g_z.value / 100])
        t = abs(np.sin(np.radians(g_tilt.value)))
        state["nodes"].append(server.scene.add_mesh_trimesh(
            "/goal/pen", pen_mesh(L, D, shape, (1 - t) * FLAT + t * UPRIGHT), wxyz=tuple(q), position=tuple(pos)))
        axis = quat_to_mat(q)[:, 0]
        state["nodes"].append(server.scene.add_point_cloud(    # the 4 success keypoints, tip to tail
            "/goal/keypoints", points=(pos + np.linspace(-L / 2, L / 2, 4)[:, None] * axis).astype(np.float32),
            colors=(20, 20, 20), point_size=0.006, point_shape="circle"))
        if g_box.value:
            for name, z, col in (("ceiling", ceiling, (90, 90, 90)), ("floor", floor, (20, 120, 60))):
                c = np.array([[cx - w, cy - w, z], [cx + w, cy - w, z], [cx + w, cy + w, z], [cx - w, cy + w, z]])
                state["nodes"].append(server.scene.add_line_segments(
                    f"/bounds/{name}", points=np.stack([c, np.roll(c, -1, 0)], 1).astype(np.float32), colors=col,
                    line_width=2.5))
            edges = np.array([[[x, y, floor], [x, y, ceiling]] for x in (cx - w, cx + w) for y in (cy - w, cy + w)])
            state["nodes"].append(server.scene.add_line_segments("/bounds/edges", points=edges.astype(np.float32),
                                                                 colors=(120, 120, 120), line_width=1.5))
        lowest = pos[2] - reach_below(q, L, D, shape) - top
        md.content = (f"Centre {g_z.value:.1f} cm above the table; this orientation's floor is "
                      f"{g_z.min:.1f} cm (pen's lowest point {lowest * 100:.1f} cm up, min "
                      f"{m['min_clearance'] * 100:.0f} cm), ceiling {(ceiling - top) * 100:.1f} cm. Green outline: the "
                      f"floor, grey: the ceiling; sideways ±{w * 100:.0f} cm. Dots: the 4 success keypoints (each "
                      f"within 1 cm counts). Roll is invisible to them.")

    def build_hand() -> None:
        if state["hand"] is not None:
            state["hand"].remove()
            state["hand"] = None
        if state["folder"] is not None:
            state["folder"].remove()
            state["folder"] = None
        state["sliders"] = {}
        hand = next(h for h in m["hands"] if h["name"] == g_hand.value)
        if not g_show.value:
            return
        server.scene.add_frame("/hand", position=(0.0, 0.0, top), show_axes=False)
        urdf = ViserUrdf(server, pathlib.Path(hand["urdf"]), root_node_name="/hand/robot")
        state["hand"] = urdf
        limits = urdf.get_actuated_joint_limits()
        state["folder"] = server.gui.add_folder("joints (each slider spans the joint's limits)")
        with state["folder"]:
            for name in urdf.get_actuated_joint_names():
                lo, hi = limits[name]
                lo, hi = (-3.14 if lo is None else float(lo)), (3.14 if hi is None else float(hi))
                v = float(np.clip(m["wrist_default"].get(name, 0.0), lo, hi))
                s = server.gui.add_slider(name, lo, hi, (hi - lo) / 200, v)
                s.on_update(lambda _: pose())
                state["sliders"][name] = (s, v)
        pose()

    def pose() -> None:
        if state["hand"] is not None:
            state["hand"].update_cfg({n: float(s.value) for n, (s, _) in state["sliders"].items()})

    for g in (g_x, g_y, g_head, g_tilt, g_roll, g_z, g_box):
        g.on_update(lambda _: state.update(dirty=True))
    g_hand.on_update(lambda _: state.update(rebuild=True))
    g_show.on_update(lambda _: state.update(rebuild=True))

    @b_pose.on_click
    def _(_) -> None:
        for s, v in state["sliders"].values():
            s.value = v

    while True:
        if state.pop("rebuild", False):
            build_hand()
        if state["dirty"]:
            state["dirty"] = False
            draw()
        time.sleep(0.03)


if __name__ == "__main__":
    main()
