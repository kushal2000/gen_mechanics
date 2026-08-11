"""Interactive reachability viewer — drive the arm by hand and see what it reaches.

The goal volume and table height are held **identical across hands**; if a hand
cannot comfortably reach the workspace, the fix is its mounting transform or arm
home pose, never the workspace. Otherwise a hand that simply cannot reach part
of the goal box looks worse at *generalizing* for a purely kinematic reason
(docs/methodology.md §7).

This is a pure-kinematics viewer: ``yourdfpy`` for FK, ``viser`` for display, no
Isaac Sim. It starts in seconds and holds no GPU, so it can run alongside
training.

    .venv_isaacsim/bin/python -m genmech.tools.reachability_viewer
    .venv_isaacsim/bin/python -m genmech.tools.reachability_viewer \\
        --robot_spec sharpa_iiwa14,allegro_iiwa14

With two specs the robots render side by side, offset in x, sharing one set of
arm sliders so the same arm pose can be compared directly.

Then open the printed URL. "Print pose" dumps the current joint vector in a form
that can be pasted into a spec's ``arm_default_joint_pos``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from genmech.robots import REGISTRY, get_robot_spec
from genmech.utils.paths import resolve as resolve_repo_path


# Task geometry the viewer draws, from ResetCfg. Held constant across hands.
GOAL_VOLUME_MINS = (-0.35, -0.2, 0.6)
GOAL_VOLUME_MAXS = (0.35, 0.2, 0.95)
TABLE_Z = 0.38
TABLE_SIZE = (1.2, 0.8)
SIDE_BY_SIDE_OFFSET_X = 1.4


def _quat_wxyz_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def _mat_to_wxyz(T: np.ndarray) -> np.ndarray:
    """Rotation matrix -> wxyz quaternion (Shepperd's method, branch on trace)."""
    R = T[:3, :3]
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([w, x, y, z])


class RobotView:
    """One robot in the scene: its URDF, sliders' target, and pose readouts."""

    def __init__(self, server, spec, base_offset_x: float, show_meshes: bool = True):
        import yourdfpy
        from viser.extras import ViserUrdf

        self.spec = spec
        self.base_offset_x = base_offset_x

        urdf_path = resolve_repo_path(spec.urdf_path)
        if not urdf_path.exists():
            raise SystemExit(f"{spec.name}: URDF not found at {urdf_path}")
        self.urdf = yourdfpy.URDF.load(str(urdf_path))

        # The robot sits at the spec's base pose, shifted in x when several
        # robots share the scene.
        base_pos = np.array(spec.base_pos, dtype=float) + np.array([base_offset_x, 0, 0])
        self.root = server.scene.add_frame(
            f"/{spec.name}", show_axes=False, position=tuple(base_pos),
            wxyz=tuple(np.array(spec.base_rot, dtype=float)),
        )
        self.viser_urdf = ViserUrdf(
            server, self.urdf, root_node_name=f"/{spec.name}", load_meshes=show_meshes
        )
        self.joint_names = list(self.viser_urdf.get_actuated_joint_names())
        self.limits = self.viser_urdf.get_actuated_joint_limits()

        server.scene.add_label(f"/{spec.name}/label", spec.name,
                               position=(0.0, 0.0, 1.5))

        # Palm center and fingertip pads, the frames the observation is built
        # in. Drawn so a wrong offset is visible rather than inferred.
        self.palm_frame = server.scene.add_frame(
            f"/{spec.name}/palm_center", axes_length=0.06, axes_radius=0.004
        )
        self.tip_frames = [
            server.scene.add_frame(f"/{spec.name}/tip_{i}",
                                   axes_length=0.03, axes_radius=0.002)
            for i in range(spec.num_fingertips)
        ]

    def joint_index(self, name: str) -> int | None:
        return self.joint_names.index(name) if name in self.joint_names else None

    def default_cfg(self, *, start_arm_higher: bool = False,
                    hand_closed_frac: float = 0.0) -> np.ndarray:
        """Joint vector for the spec's home pose, in this URDF's joint order."""
        arm = self.spec.arm_default_joint_pos_resolved(start_arm_higher=start_arm_higher)
        hand = dict(self.spec.hand_default_joint_pos)
        cfg = np.zeros(len(self.joint_names))
        for i, name in enumerate(self.joint_names):
            if name in arm:
                cfg[i] = arm[name]
            elif name in hand:
                lo, hi = self.limits[name]
                # Interpolate toward the limit that closes the finger. Both
                # bounds exist for every hand joint in both URDFs.
                target = hi if hi is not None else hand[name]
                cfg[i] = (1 - hand_closed_frac) * hand[name] + hand_closed_frac * target
        return cfg

    def update(self, cfg: np.ndarray) -> None:
        self.viser_urdf.update_cfg(cfg)
        self._refresh_markers(cfg)

    def _refresh_markers(self, cfg: np.ndarray) -> None:
        """Move the palm-center and fingertip-pad frames to match the pose."""
        self.urdf.update_cfg(
            {n: float(v) for n, v in zip(self.joint_names, cfg)}
        )
        spec = self.spec
        try:
            T_palm = self.urdf.get_transform(spec.palm_body_name, self.urdf.base_link)
        except Exception:
            return
        offset = np.array(spec.palm_center_offset, dtype=float)
        pos = T_palm[:3, 3] + T_palm[:3, :3] @ offset
        self.palm_frame.position = tuple(pos)
        self.palm_frame.wxyz = tuple(_mat_to_wxyz(T_palm))

        for i, (tip_name, tip_off) in enumerate(
            zip(spec.fingertip_body_names, spec.fingertip_offsets)
        ):
            try:
                T = self.urdf.get_transform(tip_name, self.urdf.base_link)
            except Exception:
                continue
            p = T[:3, 3] + T[:3, :3] @ np.array(tip_off, dtype=float)
            self.tip_frames[i].position = tuple(p)
            self.tip_frames[i].wxyz = tuple(_mat_to_wxyz(T))

    def report(self, cfg: np.ndarray) -> str:
        """Human-readable pose summary, pasteable into a spec."""
        spec = self.spec
        self.urdf.update_cfg({n: float(v) for n, v in zip(self.joint_names, cfg)})
        lines = [f"--- {spec.name} ---", "arm_default_joint_pos = {"]
        for name in spec.arm_joint_names:
            i = self.joint_index(name)
            if i is not None:
                lines.append(f'    "{name}": {cfg[i]:.4f},')
        lines.append("}")

        T_palm = self.urdf.get_transform(spec.palm_body_name, self.urdf.base_link)
        base = np.array(spec.base_pos, dtype=float)
        palm_world = base + T_palm[:3, 3] + T_palm[:3, :3] @ np.array(spec.palm_center_offset)
        lines.append(f"palm center (world, incl. base offset): "
                     f"{np.round(palm_world, 4).tolist()}")
        inside = all(
            GOAL_VOLUME_MINS[k] <= palm_world[k] <= GOAL_VOLUME_MAXS[k] for k in range(3)
        )
        lines.append(f"palm inside goal volume: {inside}")
        return "\n".join(lines)


def _draw_task_geometry(server) -> None:
    """Goal volume, table, and world origin — the frame every hand shares."""
    import trimesh

    mins = np.array(GOAL_VOLUME_MINS)
    maxs = np.array(GOAL_VOLUME_MAXS)
    center = (mins + maxs) / 2
    size = maxs - mins

    server.scene.add_box(
        "/task/goal_volume", dimensions=tuple(size), position=tuple(center),
        color=(80, 170, 255), opacity=0.15, wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    # Wireframe edges, so the box reads as a volume rather than a solid.
    corners = np.array(list(trimesh.creation.box(extents=size).vertices)) + center
    edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    segs = np.array([[corners[a], corners[b]] for a, b in edges])
    server.scene.add_line_segments("/task/goal_volume_edges", points=segs,
                                   colors=(40, 120, 220), line_width=2.0)

    server.scene.add_box(
        "/task/table", dimensions=(TABLE_SIZE[0], TABLE_SIZE[1], 0.02),
        position=(0.0, 0.0, TABLE_Z - 0.01), color=(160, 130, 100),
    )
    server.scene.add_frame("/task/world", axes_length=0.15, axes_radius=0.005)
    server.scene.add_grid("/task/grid", width=3.0, height=3.0, position=(0, 0, 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_spec", default=",".join(sorted(REGISTRY)),
                        help="Comma-separated spec names; several render side by side.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no_meshes", action="store_true",
                        help="Skip visual meshes (much faster to load).")
    args = parser.parse_args()

    import viser

    names = [n.strip() for n in args.robot_spec.split(",") if n.strip()]
    specs = [get_robot_spec(n) for n in names]

    server = viser.ViserServer(port=args.port)
    _draw_task_geometry(server)

    views: list[RobotView] = []
    for i, spec in enumerate(specs):
        offset = i * SIDE_BY_SIDE_OFFSET_X
        print(f"[viewer] loading {spec.name} from {spec.urdf_path}")
        views.append(RobotView(server, spec, offset, show_meshes=not args.no_meshes))

    # One slider per joint name across all robots. Shared arm joints get one
    # slider so the same arm pose can be compared directly between hands.
    all_joints: list[str] = []
    for v in views:
        for n in v.joint_names:
            if n not in all_joints:
                all_joints.append(n)

    arm_joints = [n for n in all_joints
                  if any(n in v.spec.arm_joint_names for v in views)]
    hand_joints = [n for n in all_joints if n not in arm_joints]

    sliders: dict[str, object] = {}

    def apply() -> None:
        for v in views:
            cfg = np.array([
                float(sliders[n].value) if n in sliders else 0.0
                for n in v.joint_names
            ])
            v.update(cfg)

    def add_slider(name: str, folder_default: float) -> None:
        lo, hi = next(
            (v.limits[name] for v in views if name in v.limits), (-3.14, 3.14)
        )
        lo = -3.14 if lo is None else lo
        hi = 3.14 if hi is None else hi
        s = server.gui.add_slider(name, min=float(lo), max=float(hi),
                                  step=1e-3, initial_value=float(folder_default))
        s.on_update(lambda _: apply())
        sliders[name] = s

    defaults = {}
    for v in views:
        cfg = v.default_cfg()
        for n, val in zip(v.joint_names, cfg):
            defaults.setdefault(n, float(val))

    with server.gui.add_folder("Arm"):
        for n in arm_joints:
            add_slider(n, defaults.get(n, 0.0))
    with server.gui.add_folder("Hand"):
        for n in hand_joints:
            add_slider(n, defaults.get(n, 0.0))

    with server.gui.add_folder("Poses"):
        btn_home = server.gui.add_button("Home pose")
        btn_higher = server.gui.add_button("start_arm_higher pose")
        btn_open = server.gui.add_button("Open hand")
        btn_close = server.gui.add_button("Close hand")
        btn_print = server.gui.add_button("Print pose")

    def set_from(fn) -> None:
        for v in views:
            cfg = fn(v)
            for n, val in zip(v.joint_names, cfg):
                if n in sliders:
                    sliders[n].value = float(val)
        apply()

    btn_home.on_click(lambda _: set_from(lambda v: v.default_cfg()))
    btn_higher.on_click(
        lambda _: set_from(lambda v: v.default_cfg(start_arm_higher=True))
    )
    btn_open.on_click(lambda _: set_from(lambda v: v.default_cfg(hand_closed_frac=0.0)))
    btn_close.on_click(lambda _: set_from(lambda v: v.default_cfg(hand_closed_frac=1.0)))

    def do_print(_) -> None:
        print("\n" + "=" * 60)
        for v in views:
            cfg = np.array([
                float(sliders[n].value) if n in sliders else 0.0
                for n in v.joint_names
            ])
            print(v.report(cfg))
        print("=" * 60, flush=True)

    btn_print.on_click(do_print)

    # A draggable target inside the goal volume: park it at a corner and drive
    # the arm to it to judge whether that corner is comfortably reachable.
    target = server.scene.add_transform_controls(
        "/task/target", scale=0.2,
        position=tuple((np.array(GOAL_VOLUME_MINS) + np.array(GOAL_VOLUME_MAXS)) / 2),
    )
    readout = server.gui.add_text("target", initial_value="—", disabled=True)

    @target.on_update
    def _(_) -> None:
        p = np.array(target.position)
        inside = all(GOAL_VOLUME_MINS[k] <= p[k] <= GOAL_VOLUME_MAXS[k] for k in range(3))
        readout.value = f"{np.round(p, 3).tolist()} {'in' if inside else 'OUTSIDE'}"

    apply()

    print(f"\n[viewer] serving on http://localhost:{args.port}")
    print(f"[viewer] robots: {[s.name for s in specs]}")
    print(f"[viewer] goal volume {GOAL_VOLUME_MINS} .. {GOAL_VOLUME_MAXS}, "
          f"table top z={TABLE_Z}")
    print("[viewer] drag the sliders; 'Print pose' dumps a pasteable "
          "arm_default_joint_pos. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[viewer] stopped")


if __name__ == "__main__":
    main()
