"""Interactive visualization of the actual SHARPA policy observation.

The overlay is decoded from the flat observation returned by the environment,
not rebuilt from the URDF.  Select a hand joint in the browser to inspect:

* its four ordered controlled-link box points and complete wireframe;
* the joint origin derived from the proximal face;
* distal, positive-bend, and signed-axis directions; and
* the four object keypoints stored relative to that joint.

Run from the repository root::

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \
      isaacsimenvs/pose_reaching_6d/tests/visualize_observations.py \
      --headless --port 8085

Then open ``http://localhost:8085``.  The simulator remains paused until Step
or Reset is pressed.
"""

from __future__ import annotations

import argparse
import os
import time


def main() -> None:
    # Isaac Lab imports must follow AppLauncher construction.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--num_assets_per_type", type=int, default=2)
    parser.add_argument(
        "--smoke", action="store_true",
        help="build and draw one frame, validate reconstruction, then exit",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    app = AppLauncher(args).app

    import numpy as np
    import torch
    import trimesh
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf

    import gymnasium as gym
    import isaacsimenvs  # noqa: F401
    from hand_sampler.gates.mesh import _geometry_to_mesh
    from hand_sampler.paths import resolve
    from isaacsimenvs.pose_reaching_6d.env_cfg import PoseReachEnvCfg
    from isaacsimenvs.pose_reaching_6d.obs_utils.layout import field_offsets

    cfg = PoseReachEnvCfg()
    cfg.scene.num_envs = 1
    cfg.assets.robot_spec = "sharpa_iiwa14"
    cfg.assets.num_assets_per_type = args.num_assets_per_type
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_action_delay = False
    cfg.domain_randomization.use_object_state_delay_noise = False
    cfg.domain_randomization.joint_velocity_obs_noise_std = 0.0
    cfg.domain_randomization.force_scale = 0.0
    cfg.domain_randomization.torque_scale = 0.0

    env = gym.make("GenMech-PoseReach-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    obs, _ = env.reset()
    spec = inner.scene_record.robot_spec
    offsets = field_offsets(cfg.obs.obs_list, spec)

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.add_grid("/ground", width=2.0, height=2.0, cell_size=0.1)
    server.gui.add_markdown(
        "# SHARPA token observations\n"
        "White = joint origin, red = distal, green = positive bend, blue = axis. "
        "Magenta object points are reconstructed from the selected joint token; "
        "cyan points come from the global observation and should overlap."
    )
    joint_select = server.gui.add_dropdown(
        "joint", tuple(spec.hand_joint_names), initial_value=spec.hand_joint_names[0]
    )
    step_button = server.gui.add_button("Step zero action")
    reset_button = server.gui.add_button("Reset")
    status = server.gui.add_markdown("starting")

    # Robot mesh is only background.  Every colored overlay below comes from
    # ``obs['policy']``.
    urdf = yourdfpy.URDF.load(
        str(resolve(spec.urdf_path)), load_meshes=True,
        load_collision_meshes=False, build_scene_graph=True,
        build_collision_scene_graph=False,
    )
    robot_root = server.scene.add_frame(
        "/robot", show_axes=False, position=tuple(float(v) for v in spec.base_pos),
        wxyz=tuple(float(v) for v in spec.base_rot),
    )
    del robot_root
    robot = ViserUrdf(server, urdf, root_node_name="/robot", load_meshes=True)
    viser_joints = list(robot.get_actuated_joint_names())
    sim_names = list(inner.robot.data.joint_names)
    sim_index = {name: i for i, name in enumerate(sim_names)}
    viser_from_sim = [sim_index.get(name, -1) for name in viser_joints]

    def collision_mesh(path) -> trimesh.Trimesh:
        """Whole fixed-joint URDF collision shape in its root-link frame."""
        path = resolve(path)
        model = yourdfpy.URDF.load(
            str(path), load_meshes=False, load_collision_meshes=False,
            build_scene_graph=True, build_collision_scene_graph=True,
        )
        pieces = []
        for name, link in model.link_map.items():
            for collision in link.collisions:
                mesh = _geometry_to_mesh(
                    collision.geometry, path.parent, hull=False
                )
                if mesh is None:
                    continue
                mesh = mesh.copy()
                origin = (
                    collision.origin
                    if collision.origin is not None else np.eye(4)
                )
                mesh.apply_transform(model.get_transform(name) @ origin)
                pieces.append(mesh)
        if not pieces:
            raise RuntimeError(f"no collision geometry in {path}")
        return trimesh.util.concatenate(pieces)

    def colored(mesh: trimesh.Trimesh, rgb) -> trimesh.Trimesh:
        mesh = mesh.copy()
        mesh.visual.face_colors = (*rgb, 255)
        return mesh

    # Physical scene context.  Geometry is local under live-pose frames, so it
    # continues to agree after stepping or resetting the simulator.
    table_frame = server.scene.add_frame("/table", show_axes=False)
    server.scene.add_mesh_trimesh(
        "/table/collision",
        colored(collision_mesh(cfg.assets.table_urdf), (145, 105, 70)),
    )
    object_frame = server.scene.add_frame("/object", show_axes=False)

    state = {
        "step": False,
        "reset": False,
        "dirty": True,
        "handles": [],
        "object_asset": None,
        "object_mesh": None,
    }
    step_button.on_click(lambda _: state.update(step=True))
    reset_button.on_click(lambda _: state.update(reset=True))
    joint_select.on_update(lambda _: state.update(dirty=True))

    def take(vector: np.ndarray, field: str) -> np.ndarray:
        start, end = offsets[field]
        return vector[start:end]

    def rotation_xyzw(q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        return np.asarray([
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ])

    def world_from_palm(points: np.ndarray, palm_pos, palm_rot, scale=1.0):
        return palm_pos + (points * scale) @ palm_rot.T

    def clear_overlay() -> None:
        for handle in state["handles"]:
            handle.remove()
        state["handles"].clear()

    def sphere(name: str, point, color, radius=0.004) -> None:
        state["handles"].append(server.scene.add_icosphere(
            name, radius=radius, color=color, position=tuple(float(v) for v in point)
        ))

    def draw() -> None:
        clear_overlay()
        vector = obs["policy"][0].detach().cpu().numpy()
        selected = list(spec.hand_joint_names).index(joint_select.value)
        hand_scale = float(take(vector, "hand_scale")[0])
        palm_pos = take(vector, "palm_pos")
        palm_rot = rotation_xyzw(take(vector, "palm_rot"))

        boxes = take(vector, "joint_link_bbox").reshape(22, 4, 3)
        p0, p1, p2, p3 = boxes[selected]
        e1, e2, e3 = p1 - p0, p2 - p0, p3 - p0
        corners_palm = np.asarray([
            p0,
            p0 + e1,
            p0 + e2,
            p0 + e3,
            p0 + e1 + e2,
            p0 + e1 + e3,
            p0 + e2 + e3,
            p0 + e1 + e2 + e3,
        ])
        corners = world_from_palm(corners_palm, palm_pos, palm_rot, hand_scale)
        edge_pairs = (
            (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4),
            (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7),
        )
        state["handles"].append(server.scene.add_line_segments(
            "/obs/link_box", points=np.asarray(
                [[corners[a], corners[b]] for a, b in edge_pairs]
            ), colors=(245, 190, 60), line_width=3.0,
        ))

        ordered = world_from_palm(
            np.asarray([p0, p1, p2, p3]), palm_pos, palm_rot, hand_scale
        )
        colors = ((230, 230, 230), (235, 70, 60), (70, 220, 90), (65, 120, 245))
        for i, (point, color) in enumerate(zip(ordered, colors)):
            sphere(f"/obs/ordered_p{i}", point, color)
        for i, color in zip((1, 2, 3), colors[1:]):
            state["handles"].append(server.scene.add_line_segments(
                f"/obs/basis_{i}", points=np.asarray([[ordered[0], ordered[i]]]),
                colors=color, line_width=6.0,
            ))

        # The proximal face is spanned by e2/e3, so its centre is the joint.
        joint_palm = p0 + 0.5 * e2 + 0.5 * e3
        joint_world = world_from_palm(joint_palm, palm_pos, palm_rot, hand_scale)
        sphere("/obs/joint_origin", joint_world, (255, 255, 255), radius=0.0055)

        rel = take(vector, "object_keypoints_rel_joint").reshape(22, 4, 3)[selected]
        object_from_joint = joint_world + (rel * hand_scale) @ palm_rot.T
        global_object = palm_pos + take(vector, "keypoints_rel_palm").reshape(4, 3)
        for i, point in enumerate(object_from_joint):
            sphere(f"/obs/object_from_joint_{i}", point, (235, 60, 220), radius=0.0045)
        for i, point in enumerate(global_object):
            sphere(f"/obs/object_global_{i}", point, (40, 220, 230), radius=0.0025)
        state["handles"].append(server.scene.add_line_segments(
            "/obs/joint_to_object", points=np.asarray(
                [[joint_world, point] for point in object_from_joint]
            ), colors=(235, 60, 220), line_width=1.5,
        ))

        q = inner.robot.data.joint_pos[0].detach().cpu().numpy()
        robot.update_cfg(np.asarray([q[i] if i >= 0 else 0.0 for i in viser_from_sim]))

        env_origin = inner.scene.env_origins[0].detach().cpu().numpy()
        table_frame.position = tuple(float(v) for v in (
            inner.table.data.root_pos_w[0].detach().cpu().numpy() - env_origin
        ))
        table_frame.wxyz = tuple(float(v) for v in (
            inner.table.data.root_quat_w[0].detach().cpu().numpy()
        ))
        object_frame.position = tuple(float(v) for v in (
            inner.object.data.root_pos_w[0].detach().cpu().numpy() - env_origin
        ))
        object_frame.wxyz = tuple(float(v) for v in (
            inner.object.data.root_quat_w[0].detach().cpu().numpy()
        ))
        asset_index = int(inner.scene_record.object_pool_index[0].item())
        if state["object_asset"] != asset_index:
            if state["object_mesh"] is not None:
                state["object_mesh"].remove()
            mesh = collision_mesh(inner.scene_record.object_urdf_paths[asset_index])
            state["object_mesh"] = server.scene.add_mesh_trimesh(
                "/object/collision", colored(mesh, (215, 140, 50))
            )
            state["object_asset"] = asset_index

        enabled = take(vector, "joint_enabled")[selected]
        lower = take(vector, "joint_lower")[selected]
        upper = take(vector, "joint_upper")[selected]
        overlap_mm = np.linalg.norm(object_from_joint - global_object, axis=1) * 1000.0
        status.content = (
            f"**{joint_select.value}** — token {selected}\n\n"
            f"enabled `{enabled:.0f}`, limits `[{lower:.3f}, {upper:.3f}] rad`  \n"
            f"box edges `{np.linalg.norm(e1)*hand_scale*1000:.1f}`, "
            f"`{np.linalg.norm(e2)*hand_scale*1000:.1f}`, "
            f"`{np.linalg.norm(e3)*hand_scale*1000:.1f} mm`  \n"
            f"object-keypoint reconstruction error: max `{overlap_mm.max():.4f} mm`"
        )
        state["dirty"] = False

    print(f"\n[obs-viz] open http://localhost:{args.port}", flush=True)
    print("[obs-viz] Ctrl-C to stop", flush=True)
    period = 1.0 / max(args.hz, 1e-3)
    actions = torch.zeros(1, inner.cfg.action_space, device=inner.device)
    if args.smoke:
        draw()
        print("[obs-viz] smoke PASS", flush=True)
        env.close()
        app.close()
        os._exit(0)
    try:
        while True:
            t0 = time.time()
            if state["reset"]:
                obs, _ = env.reset()
                state.update(reset=False, dirty=True)
            if state["step"]:
                obs, *_ = env.step(actions)
                state.update(step=False, dirty=True)
            if state["dirty"]:
                draw()
            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("\n[obs-viz] stopped", flush=True)

    env.close()
    app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
