"""Interactive reachability viewer — drive each hand by hand and see what it reaches.

The goal volume and table height are held **identical across hands**; if a hand
cannot comfortably reach the workspace, the fix is its mounting transform or arm
home pose, never the workspace. Otherwise a hand that simply cannot reach part
of the goal box looks worse at *generalizing* for a purely kinematic reason
(docs/methodology.md §7).

Pure kinematics: ``yourdfpy`` for FK, ``viser`` for display, no Isaac Sim. It
starts in seconds and holds no GPU, so it runs happily alongside training.

    .venv_isaacsim/bin/python -m genmech.tools.reachability_viewer
    .venv_isaacsim/bin/python -m genmech.tools.reachability_viewer \\
        --robot_spec sharpa_iiwa14,allegro_iiwa14

Each robot gets its **own complete scene clone** — its own table, goal volume,
and grid at the same relative placement — so the two are compared like for like
rather than sharing one table. Each also gets its own GUI section with
independent sliders, so the hands can be posed separately.

Sliders are in **degrees**, labelled with short joint names and grouped per
finger. "Copy arm pose to ..." transfers the arm joints between robots, which is
the usual way to compare: pose one arm, copy it, then look at how the two hands
present themselves at the identical arm configuration.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from genmech.robots import REGISTRY, get_robot_spec
from genmech.utils.paths import resolve as resolve_repo_path


# Task geometry, from ResetCfg. Identical for every hand — that is the point.
GOAL_VOLUME_MINS = (-0.35, -0.2, 0.6)
GOAL_VOLUME_MAXS = (0.35, 0.2, 0.95)
TABLE_Z = 0.38
TABLE_SIZE = (1.2, 0.8)
CLONE_SPACING_X = 1.8

RAD = np.pi / 180.0


def _mat_to_wxyz(T: np.ndarray) -> np.ndarray:
    """Rotation matrix -> wxyz quaternion (branch on trace for stability)."""
    R = T[:3, :3]
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)
    return np.array(q)


def _short(name: str) -> str:
    """Readable slider label: drop the arm prefix, keep the finger/joint part."""
    n = name
    for prefix in ("iiwa14_joint_", "left_"):
        if n.startswith(prefix):
            n = n[len(prefix):]
    # SHARPA's left_1_thumb_... digits exist only to force a sort order.
    if len(n) > 2 and n[0].isdigit() and n[1] == "_":
        n = n[2:]
    return n


def _finger_of(spec, joint: str) -> str:
    """Group key for a hand joint, so sliders sit under the right finger."""
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        if finger in joint:
            return finger
    return "other"


def _draw_scene_clone(server, prefix: str, offset_x: float) -> None:
    """One full task setup: table, goal volume, grid, world frame."""
    import trimesh

    mins = np.array(GOAL_VOLUME_MINS) + np.array([offset_x, 0, 0])
    maxs = np.array(GOAL_VOLUME_MAXS) + np.array([offset_x, 0, 0])
    center = (mins + maxs) / 2
    size = maxs - mins

    server.scene.add_box(f"{prefix}/goal_volume", dimensions=tuple(size),
                         position=tuple(center), color=(80, 170, 255), opacity=0.12,
                         wxyz=(1.0, 0.0, 0.0, 0.0))
    corners = np.array(trimesh.creation.box(extents=size).vertices) + center
    edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    server.scene.add_line_segments(
        f"{prefix}/goal_volume_edges",
        points=np.array([[corners[a], corners[b]] for a, b in edges]),
        colors=(40, 120, 220), line_width=2.0,
    )
    server.scene.add_box(f"{prefix}/table",
                         dimensions=(TABLE_SIZE[0], TABLE_SIZE[1], 0.02),
                         position=(offset_x, 0.0, TABLE_Z - 0.01),
                         color=(160, 130, 100))
    server.scene.add_frame(f"{prefix}/origin", axes_length=0.15, axes_radius=0.005,
                           position=(offset_x, 0.0, 0.0))
    server.scene.add_grid(f"{prefix}/grid", width=2.4, height=2.4,
                          position=(offset_x, 0.0, 0.0))


class RobotView:
    """One robot: its URDF, its scene clone, its sliders, its readouts."""

    def __init__(self, server, spec, offset_x: float, show_meshes: bool = True):
        import yourdfpy
        from viser.extras import ViserUrdf

        self.server = server
        self.spec = spec
        self.offset_x = offset_x

        urdf_path = resolve_repo_path(spec.urdf_path)
        if not urdf_path.exists():
            raise SystemExit(f"{spec.name}: URDF not found at {urdf_path}")
        self.urdf = yourdfpy.URDF.load(str(urdf_path))

        _draw_scene_clone(server, f"/{spec.name}_scene", offset_x)

        base = np.array(spec.base_pos, dtype=float) + np.array([offset_x, 0, 0])
        server.scene.add_frame(f"/{spec.name}", show_axes=False, position=tuple(base),
                               wxyz=tuple(np.array(spec.base_rot, dtype=float)))
        self.viser_urdf = ViserUrdf(server, self.urdf, root_node_name=f"/{spec.name}",
                                    load_meshes=show_meshes)
        self.joint_names = list(self.viser_urdf.get_actuated_joint_names())
        self.limits = self.viser_urdf.get_actuated_joint_limits()

        server.scene.add_label(f"/{spec.name}/label", spec.name, position=(0.0, 0.0, 1.6))
        self.palm_frame = server.scene.add_frame(f"/{spec.name}/palm_center",
                                                 axes_length=0.07, axes_radius=0.004)
        self.tip_frames = [
            server.scene.add_frame(f"/{spec.name}/tip_{i}", axes_length=0.03,
                                   axes_radius=0.002)
            for i in range(spec.num_fingertips)
        ]
        self.sliders: dict[str, object] = {}

    # --- poses -------------------------------------------------------------
    def pose_dict(self, *, start_arm_higher: bool = False,
                  hand_closed_frac: float = 0.0) -> dict[str, float]:
        arm = self.spec.arm_default_joint_pos_resolved(start_arm_higher=start_arm_higher)
        hand = dict(self.spec.hand_default_joint_pos)
        out = {}
        for name in self.joint_names:
            if name in arm:
                out[name] = arm[name]
            elif name in hand:
                lo, hi = self.limits[name]
                hi = hand[name] if hi is None else hi
                out[name] = (1 - hand_closed_frac) * hand[name] + hand_closed_frac * hi
            else:
                out[name] = 0.0
        return out

    def current(self) -> dict[str, float]:
        return {n: float(self.sliders[n].value) * RAD for n in self.joint_names}

    def set_pose(self, pose: dict[str, float]) -> None:
        for name, val in pose.items():
            if name in self.sliders:
                self.sliders[name].value = float(val) / RAD

    # --- rendering ---------------------------------------------------------
    def refresh(self) -> None:
        pose = self.current()
        cfg = np.array([pose[n] for n in self.joint_names])
        self.viser_urdf.update_cfg(cfg)

        self.urdf.update_cfg(pose)
        spec = self.spec
        off = np.array([self.offset_x, 0.0, 0.0]) + np.array(spec.base_pos, dtype=float)
        try:
            T = self.urdf.get_transform(spec.palm_body_name, self.urdf.base_link)
        except Exception:
            return
        p = T[:3, 3] + T[:3, :3] @ np.array(spec.palm_center_offset, dtype=float)
        self.palm_frame.position = tuple(p + off)
        self.palm_frame.wxyz = tuple(_mat_to_wxyz(T))
        for i, (tip, tip_off) in enumerate(
            zip(spec.fingertip_body_names, spec.fingertip_offsets)
        ):
            Tt = self.urdf.get_transform(tip, self.urdf.base_link)
            pt = Tt[:3, 3] + Tt[:3, :3] @ np.array(tip_off, dtype=float)
            self.tip_frames[i].position = tuple(pt + off)
            self.tip_frames[i].wxyz = tuple(_mat_to_wxyz(Tt))

    def report(self) -> str:
        spec = self.spec
        pose = self.current()
        self.urdf.update_cfg(pose)
        lines = [f"--- {spec.name} ---", "arm_default_joint_pos = {"]
        for name in spec.arm_joint_names:
            if name in pose:
                lines.append(f'    "{name}": {pose[name]:.4f},')
        lines.append("}")
        lines.append("hand_default_joint_pos = {")
        for name in spec.hand_joint_names:
            if name in pose:
                lines.append(f'    "{name}": {pose[name]:.4f},')
        lines.append("}")

        T = self.urdf.get_transform(spec.palm_body_name, self.urdf.base_link)
        palm = (np.array(spec.base_pos, dtype=float) + T[:3, 3]
                + T[:3, :3] @ np.array(spec.palm_center_offset))
        inside = all(GOAL_VOLUME_MINS[k] <= palm[k] <= GOAL_VOLUME_MAXS[k] for k in range(3))
        lines.append(f"palm center (own scene frame): {np.round(palm, 4).tolist()}")
        lines.append(f"palm inside goal volume: {inside}")
        return "\n".join(lines)


def _build_gui(server, view: RobotView, views: list[RobotView]) -> None:
    """One GUI section per robot: its own sliders, poses, and copy targets."""
    spec = view.spec

    def on_change(_=None) -> None:
        view.refresh()

    with server.gui.add_folder(spec.name):
        home = view.pose_dict()

        with server.gui.add_folder("Arm (deg)"):
            for name in spec.arm_joint_names:
                if name not in view.limits:
                    continue
                lo, hi = view.limits[name]
                lo = -np.pi if lo is None else lo
                hi = np.pi if hi is None else hi
                s = server.gui.add_slider(
                    _short(name), min=round(lo / RAD, 1), max=round(hi / RAD, 1),
                    step=0.5, initial_value=round(home[name] / RAD, 1),
                )
                s.on_update(on_change)
                view.sliders[name] = s

        # Hand sliders grouped per finger, so a finger can be curled as a unit.
        by_finger: dict[str, list[str]] = {}
        for name in spec.hand_joint_names:
            by_finger.setdefault(_finger_of(spec, name), []).append(name)
        with server.gui.add_folder("Hand (deg)", expand_by_default=False):
            for finger, names in by_finger.items():
                with server.gui.add_folder(finger, expand_by_default=False):
                    for name in names:
                        if name not in view.limits:
                            continue
                        lo, hi = view.limits[name]
                        lo = -np.pi if lo is None else lo
                        hi = np.pi if hi is None else hi
                        s = server.gui.add_slider(
                            _short(name), min=round(lo / RAD, 1), max=round(hi / RAD, 1),
                            step=0.5, initial_value=round(home[name] / RAD, 1),
                        )
                        s.on_update(on_change)
                        view.sliders[name] = s

        with server.gui.add_folder("Pose"):
            b_home = server.gui.add_button("Home")
            b_high = server.gui.add_button("start_arm_higher")
            b_open = server.gui.add_button("Open hand")
            b_close = server.gui.add_button("Close hand")
            b_print = server.gui.add_button("Print pose")

            def apply(fn):
                def _(_):
                    view.set_pose(fn())
                    view.refresh()
                return _

            b_home.on_click(apply(lambda: view.pose_dict()))
            b_high.on_click(apply(lambda: view.pose_dict(start_arm_higher=True)))
            b_open.on_click(apply(lambda: view.pose_dict(hand_closed_frac=0.0)))
            b_close.on_click(apply(lambda: view.pose_dict(hand_closed_frac=1.0)))
            b_print.on_click(lambda _: print("\n" + view.report(), flush=True))

        # The arm is identical across hands by construction, so its joint values
        # transfer verbatim. Posing one arm and copying it is the natural way to
        # see how the two hands present themselves at the same configuration.
        others = [v for v in views if v is not view]
        if others:
            with server.gui.add_folder("Copy arm pose"):
                for other in others:
                    btn = server.gui.add_button(f"-> {other.spec.name}")

                    def _copy(_, src=view, dst=other):
                        pose = src.current()
                        dst.set_pose({n: v for n, v in pose.items()
                                      if n in dst.spec.arm_joint_names})
                        dst.refresh()

                    btn.on_click(_copy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_spec", default=",".join(sorted(REGISTRY)),
                        help="Comma-separated spec names; each gets its own scene clone.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no_meshes", action="store_true",
                        help="Skip visual meshes (much faster to load).")
    args = parser.parse_args()

    import viser

    names = [n.strip() for n in args.robot_spec.split(",") if n.strip()]
    specs = [get_robot_spec(n) for n in names]

    server = viser.ViserServer(port=args.port)

    views: list[RobotView] = []
    for i, spec in enumerate(specs):
        print(f"[viewer] loading {spec.name} from {spec.urdf_path}")
        views.append(RobotView(server, spec, i * CLONE_SPACING_X,
                               show_meshes=not args.no_meshes))

    for view in views:
        _build_gui(server, view, views)
    for view in views:
        view.refresh()

    print(f"\n[viewer] serving on http://localhost:{args.port}")
    print(f"[viewer] robots: {[s.name for s in specs]} "
          f"(each in its own scene clone, {CLONE_SPACING_X} m apart in x)")
    print(f"[viewer] goal volume {GOAL_VOLUME_MINS} .. {GOAL_VOLUME_MAXS}, "
          f"table top z={TABLE_Z}")
    print("[viewer] sliders are in DEGREES. 'Print pose' dumps a pasteable spec "
          "block. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[viewer] stopped")


if __name__ == "__main__":
    main()
