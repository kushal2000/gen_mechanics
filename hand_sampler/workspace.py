"""Task geometry shared by every viewer, and the collision-hull swap.

These are facts about the scene a hand is judged in -- where the table top
sits, how big the table actually is, and what geometry PhysX really uses for
contacts -- not facts about any one viewer. They lived in the reachability
viewer, which meant the design-space and mutation viewers had to reach into
``isaacsimenvs`` to draw a table, inverting the dependency rule. Pure numpy and
trimesh, so ``hand_sampler`` stays importable without a simulator.
"""

from __future__ import annotations

import numpy as np

from hand_sampler.paths import resolve as resolve_repo_path


# Task geometry, from ResetCfg. Identical for every hand — that is the point.
GOAL_VOLUME_MINS = (-0.35, -0.2, 0.6)
GOAL_VOLUME_MAXS = (0.35, 0.2, 0.95)
TABLE_Z = 0.38
TABLE_URDF = "assets/urdf/table_narrow.urdf"


def table_extents() -> tuple[float, float, float]:
    """Box dimensions read from the actual table asset.

    This used to be a hardcoded TABLE_SIZE = (1.2, 0.8), which is 2.5x too wide
    in x and 2x in y against the asset's 0.475 x 0.4 x 0.3. An oversized table
    swallows the arm's link_3 and link_4, which looks exactly like the robot
    colliding with the table when nothing is wrong with the robot.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(resolve_repo_path(TABLE_URDF)).getroot()
    for link in root.findall("link"):
        for coll in link.findall("collision"):
            box = coll.find("geometry/box")
            if box is not None:
                return tuple(float(v) for v in box.get("size").split())
    raise RuntimeError(f"no box collision geometry in {TABLE_URDF}")


def _hull_collision_scene(urdf) -> int:
    """Replace each collision mesh with its convex hull, in place.

    Isaac Lab stamps ``approximation="convexHull"`` on every mesh collider
    (verified in the baked USD: 34/34 on SHARPA, 26/26 on Allegro), so PhysX
    never resolves contacts against the triangle meshes the URDF declares -- it
    uses their hulls, with every concavity between phalanges filled in.

    Showing the declared mesh therefore overstates the fidelity of the
    simulation. Hulling the collision scene before ViserUrdf reads it means the
    "collision" view is the geometry that actually decides contacts, and it
    still follows the joint sliders because only the geometry is swapped, not
    the scene graph.

    Generated hands are unaffected -- their capsules and palm box are analytic
    primitives with approximation "None", i.e. simulated exactly as declared.
    """
    import trimesh

    scene = getattr(urdf, "collision_scene", None)
    if scene is None:
        return 0
    n = 0
    for key, geom in list(scene.geometry.items()):
        if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 3:
            try:
                scene.geometry[key] = geom.convex_hull
                n += 1
            except Exception:
                pass
    return n


class RobotView:
    """One robot: its URDF, its scene clone, its sliders, its readouts."""

    def __init__(self, server, spec, offset_x: float, show_meshes: bool = True,
                 joint_overrides: dict[str, float] | None = None,
                 hull_collision: bool = True):
        import yourdfpy
        from viser.extras import ViserUrdf

        self.server = server
        self.spec = spec
        self.offset_x = offset_x
        self.joint_overrides = dict(joint_overrides or {})

        urdf_path = resolve_repo_path(spec.urdf_path)
        if not urdf_path.exists():
            raise SystemExit(f"{spec.name}: URDF not found at {urdf_path}")
        # Both scene graphs, so the viewer can switch between what is RENDERED
        # (visual) and what is SIMULATED (collision). They are not the same
        # geometry: SHARPA's collision meshes are coarser than its visual ones,
        # and a generated hand's capsule is one collision cylinder but a
        # cylinder plus two spheres in visual, since URDF has no capsule.
        self.urdf = yourdfpy.URDF.load(
            str(urdf_path),
            load_meshes=True, load_collision_meshes=True,
            build_scene_graph=True, build_collision_scene_graph=True,
        )
        if hull_collision:
            _hull_collision_scene(self.urdf)

        _draw_scene_clone(server, f"/{spec.name}_scene", offset_x)

        base = np.array(spec.base_pos, dtype=float) + np.array([offset_x, 0, 0])
        server.scene.add_frame(f"/{spec.name}", show_axes=False, position=tuple(base),
                               wxyz=tuple(np.array(spec.base_rot, dtype=float)))
        self.viser_urdf = ViserUrdf(server, self.urdf, root_node_name=f"/{spec.name}",
                                    load_meshes=show_meshes,
                                    load_collision_meshes=True)
        # show_visual / show_collision are boolean PROPERTIES here, not methods.
        self.viser_urdf.show_visual = True
        self.viser_urdf.show_collision = False
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
        # Applied last so a --joint_override wins over the spec's home pose.
        for name, val in self.joint_overrides.items():
            if name in out:
                out[name] = val
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


# A joint with less travel than this is ghosted, not steerable. Generated hands
# lock unused joints to [0, 1e-8] so the articulation keeps one shape across
# designs (hand_sampler/params.py); a slider whose min and max both
# round to 0.0 degrees is a broken widget, so they get a disabled one instead --
# which also makes it visible at a glance WHICH joints a design ghosted.
GHOST_TRAVEL_RAD = 1e-6


__all__ = ["GOAL_VOLUME_MINS", "GOAL_VOLUME_MAXS", "TABLE_Z", "TABLE_URDF",
           "table_extents", "_hull_collision_scene"]
