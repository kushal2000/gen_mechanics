"""Interactive policy demo — watch a rollout in viser, with episode controls.

Modelled on simtoolreal's ``dextoolbench/eval_interactive_isaacgym.py``, which
is the version that works: a ``ViserUrdf`` driven by ``update_cfg`` each step, a
table and object drawn in place, and a default camera set on client connect so
you land looking at the robot instead of at empty space.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.interactive_eval_viewer \\
        --robot_spec gen_sharpa_like \\
        --checkpoint train_dir/.../nn/0_pose_reach_sapg.pth \\
        --policy_config /tmp/gen_policy_config.yaml --port 8084

``--policy_spec sharpa_iiwa14`` drives a generated hand with SHARPA's policy
(genmech/eval/crossbody.py).

**Geometry.** The toggle picks what is drawn: ``visual`` is the URDF's render
geometry, ``collision`` is what PhysX resolves contacts against. They are not
the same -- Isaac Lab stamps ``approximation="convexHull"`` on every mesh
collider, and a generated hand's capsule is one collision cylinder but a
cylinder plus two spheres in visual. Judge penetration on ``collision``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


TABLE_Z = 0.38
TABLE_DIMS = (0.475, 0.4, 0.3)


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_spec", default="gen_sharpa_like")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy_config", required=True)
    parser.add_argument("--policy_spec", default=None)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_assets_per_type", type=int, default=2)
    parser.add_argument("--port", type=int, default=8084)
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--sapg_expl_coef", type=float, default=50.0)
    parser.add_argument("--hz", type=float, default=30.0)
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
    import genmech.tasks  # noqa: F401
    from genmech.eval.rl_player import RlPlayer
    from genmech.tasks.pose_reach.env_cfg import PoseReachEnvCfg
    from genmech.tools.check_self_collision import _geometry_to_mesh
    from genmech.utils.paths import resolve as resolve_repo_path

    cfg = PoseReachEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type
    cfg.assets.robot_spec = args.robot_spec

    env = gym.make("GenMech-PoseReach-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    inner._replay_target_lab_order = None
    spec = inner.robot_spec

    # --- policy -----------------------------------------------------------
    adapter = None
    n_act = int(inner.cfg.action_space)
    p_obs, p_act = inner.cfg.observation_space, n_act
    if args.policy_spec and args.policy_spec != args.robot_spec:
        from genmech.eval.crossbody import CrossBodyAdapter
        from genmech.robots import get_robot_spec
        from genmech.tasks.pose_reach.utils.obs_utils import compute_obs_dim

        pol = get_robot_spec(args.policy_spec)
        adapter = CrossBodyAdapter(pol, spec, cfg.obs.obs_list, args.rl_device)
        print(f"[viz] {adapter.describe()}", flush=True)
        p_obs, p_act = compute_obs_dim(cfg.obs.obs_list, pol), pol.num_joints

    player = RlPlayer(
        num_observations=p_obs, num_actions=p_act,
        config_path=args.policy_config, checkpoint_path=args.checkpoint,
        device=args.rl_device, sapg_expl_coef=args.sapg_expl_coef,
        num_envs=args.num_envs,
    )
    player.player.init_rnn()

    # Step once before building the scene: RigidObject root poses are not
    # populated until the sim has run, so reading the table during construction
    # gave (0, 0, 0) and drew it 230 mm below where it actually is.
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(torch.zeros((args.num_envs, n_act), device=inner.device))

    # --- viser ------------------------------------------------------------
    server = viser.ViserServer(host="0.0.0.0", port=args.port)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # Without this you land on viser's default view, which for a scene this
        # size is a grey void. This looks at the hand over the table.
        client.camera.position = (0.9, -0.9, 1.15)
        client.camera.look_at = (0.0, 0.15, 0.62)

    server.scene.add_grid("/ground", width=2.4, height=2.4, cell_size=0.1)
    # The table box is CENTRED on its rigid-body root, so its surface is at
    # root + half the box height: 0.3754 + 0.15 = 0.525 at the nominal setting.
    #
    # `table_reset_z` (0.38) is therefore the table's CENTRE, not its surface,
    # despite the comment at env_cfg.py:69 saying "the table surface height
    # stays at table_reset_z". Measured: objects reset at 0.38 + 0.25 = 0.63,
    # fall, and settle at 0.535 with zero velocity -- consistent with a surface
    # at 0.525 and ~10 mm of object half-thickness, not with a surface at 0.38.
    #
    # The root is also randomized per episode by +-table_reset_z_range, so this
    # is read every step rather than fixed at construction.
    table_handle = server.scene.add_box(
        "/table", dimensions=TABLE_DIMS, position=(0.0, 0.0, TABLE_Z),
        color=(180, 130, 70))
    print(f"[viz] table surface: cfg {cfg.reset.table_reset_z} "
          f"+-{cfg.reset.table_reset_z_range} m, read per-env each step",
          flush=True)

    urdf_path = resolve_repo_path(spec.urdf_path)
    robot_urdf = yourdfpy.URDF.load(
        str(urdf_path), load_meshes=True, load_collision_meshes=True,
        build_scene_graph=True, build_collision_scene_graph=True)

    server.scene.add_frame("/robot", show_axes=False,
                           position=tuple(float(v) for v in spec.base_pos),
                           wxyz=tuple(float(v) for v in spec.base_rot))
    robot = ViserUrdf(server, robot_urdf, root_node_name="/robot",
                      load_meshes=True, load_collision_meshes=True)
    robot.show_visual = True
    robot.show_collision = False

    # --- object -----------------------------------------------------------
    # One mesh per pool asset, since each env draws a different one. Their
    # geometry is used AS WRITTEN: _object_scale_per_env holds
    # object_scales_normalized, which is dimensions divided by object_base_size
    # for the reward and observation, NOT a mesh multiplier. Applying it stretches
    # the tool several-fold -- the URDF already carries the true dimensions.
    def _asset_mesh(path: str):
        u = yourdfpy.URDF.load(path, load_meshes=False, load_collision_meshes=False,
                               build_scene_graph=True,
                               build_collision_scene_graph=True)
        out = None
        for name, link in u.link_map.items():
            for coll in link.collisions:
                m = _geometry_to_mesh(coll.geometry, urdf_path.parent, hull=False)
                if m is None:
                    continue
                m = m.copy()
                origin = coll.origin if coll.origin is not None else np.eye(4)
                m.apply_transform(u.get_transform(name) @ origin)
                out = m if out is None else trimesh.util.concatenate([out, m])
        return out

    obj_frame = server.scene.add_frame("/object", show_axes=False)
    asset_meshes = {}
    for e in range(args.num_envs):
        idx = int(inner._object_asset_index_per_env[e].item())
        if idx in asset_meshes:
            continue
        m = _asset_mesh(inner._object_urdf_paths[idx])
        if m is None:
            continue
        asset_meshes[idx] = server.scene.add_mesh_simple(
            f"/object/asset{idx}", vertices=m.vertices, faces=m.faces,
            color=(215, 140, 50))
        print(f"[viz] object asset {idx}: extents "
              f"{np.round(m.extents * 1000, 1)} mm", flush=True)
    goal_frame = server.scene.add_frame("/goal", show_axes=False)
    goal_meshes = {}
    for idx, _h in list(asset_meshes.items()):
        gm = _asset_mesh(inner._object_urdf_paths[idx])
        if gm is None:
            continue
        goal_meshes[idx] = server.scene.add_mesh_simple(
            f"/goal/asset{idx}", vertices=gm.vertices, faces=gm.faces,
            color=(90, 190, 120), opacity=0.45)

    # --- GUI --------------------------------------------------------------
    server.gui.add_markdown(f"# {args.robot_spec}\n### Interactive policy demo")

    with server.gui.add_folder("Episode Controls", expand_by_default=True):
        b_run = server.gui.add_button("Run")
        b_pause = server.gui.add_button("Pause")
        b_step = server.gui.add_button("Step once")
        b_reset = server.gui.add_button("Reset")

    with server.gui.add_folder("View", expand_by_default=True):
        g_env = server.gui.add_slider("env", min=0, max=args.num_envs - 1,
                                      step=1, initial_value=0)
        g_geom = server.gui.add_dropdown("geometry", ("visual", "collision"),
                                         initial_value="visual")
        server.gui.add_markdown(
            "_judge penetration on **collision** — PhysX simulates mesh "
            "colliders as convex hulls, coarser than the visual_")

    md_status = server.gui.add_markdown("**Status:** ready")
    md_stats = server.gui.add_markdown("**Episodes:** none yet")

    state = {"running": False, "once": False, "reset": False, "step": 0,
             "eps": [], "cur_goals": 0}
    b_run.on_click(lambda _: state.update(running=True))
    b_pause.on_click(lambda _: state.update(running=False))
    b_step.on_click(lambda _: state.update(once=True))
    b_reset.on_click(lambda _: state.update(reset=True))

    def _on_geom(_=None) -> None:
        coll = g_geom.value == "collision"
        robot.show_collision = coll
        robot.show_visual = not coll
    g_geom.on_update(_on_geom)

    # --- joint plumbing ---------------------------------------------------
    # ViserUrdf wants values in the URDF's actuated-joint order; the sim reports
    # them in Isaac Lab's parser order. Map by name rather than assuming.
    viser_joints = list(robot.get_actuated_joint_names())
    sim_joints = list(inner.robot.data.joint_names)
    sim_index = {n: i for i, n in enumerate(sim_joints)}
    missing = [n for n in viser_joints if n not in sim_index]
    if missing:
        print(f"[viz] WARNING: {len(missing)} URDF joints absent from the sim "
              f"(held at 0): {missing[:4]}", flush=True)
    order = [sim_index.get(n, -1) for n in viser_joints]

    def push(e: int) -> None:
        origin = inner.scene.env_origins[e].cpu().numpy()
        tp = inner.table.data.root_pos_w[e].cpu().numpy() - origin
        table_handle.position = (float(tp[0]), float(tp[1]), float(tp[2]))
        table_handle.wxyz = tuple(
            float(v) for v in inner.table.data.root_quat_w[e].cpu().numpy())
        q = inner.robot.data.joint_pos[e].cpu().numpy()
        robot.update_cfg(np.array([q[i] if i >= 0 else 0.0 for i in order]))
        op = inner.object.data.root_pos_w[e].cpu().numpy() - origin
        oq = inner.object.data.root_quat_w[e].cpu().numpy()
        obj_frame.position = tuple(float(v) for v in op)
        obj_frame.wxyz = tuple(float(v) for v in oq)
        shown = int(inner._object_asset_index_per_env[e].item())
        for idx, h in asset_meshes.items():
            h.visible = (idx == shown)
        # The goal is carried by the goal_viz RigidObject, not by a pair of
        # tensors on the env.
        gp = inner.goal_viz.data.root_pos_w[e].cpu().numpy() - origin
        gq = inner.goal_viz.data.root_quat_w[e].cpu().numpy()
        goal_frame.position = tuple(float(v) for v in gp)
        goal_frame.wxyz = tuple(float(v) for v in gq)
        for idx, h in goal_meshes.items():
            h.visible = (idx == shown)

    print(f"\n[viz] serving on http://localhost:{args.port}")
    print(f"[viz] press Run. Toggle geometry to 'collision' to judge contact.\n")

    period = 1.0 / max(args.hz, 1e-3)
    try:
        while True:
            t0 = time.time()
            if state["reset"]:
                obs, _ = env.reset()
                player.player.init_rnn()
                state.update(reset=False, step=0, cur_goals=0)

            if state["running"] or state["once"]:
                state["once"] = False
                po = obs["policy"].to(args.rl_device)
                if adapter is not None:
                    po = adapter.observation(po)
                act = player.get_normalized_action(po, deterministic_actions=True)
                if adapter is not None:
                    act = adapter.action(act)
                obs, _, term, trunc, _ = env.step(act.to(inner.device))
                state["step"] += 1
                e0 = int(g_env.value)
                if bool((term | trunc)[e0].item()):
                    state["eps"].append(state["cur_goals"])
                    state["step"] = 0

            e = int(g_env.value)
            state["cur_goals"] = int(inner._successes[e].item())
            push(e)
            md_status.content = (
                f"**{'running' if state['running'] else 'paused'}** — "
                f"step {state['step']}, env {e}\n\n"
                f"goals **{state['cur_goals']}**, "
                f"lifted {bool(inner._lifted_object[e].item())}\n\n"
                f"table surface z = "
                f"{float(inner._table_z_per_env[e].item()) + TABLE_DIMS[2] / 2.0:.4f} m "
                f"(centre {float(inner._table_z_per_env[e].item()):.4f})")
            if state["eps"]:
                md_stats.content = (
                    f"**Episodes:** {len(state['eps'])}, "
                    f"mean goals {sum(state['eps']) / len(state['eps']):.2f}")
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[viz] stopped")

    del app
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
