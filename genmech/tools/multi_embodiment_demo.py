"""One Isaac scene holding several distinct robot embodiments at once.

The morphology search this repo is building towards needs many hands evaluated
per outer iteration. One embodiment per simulation means one Kit boot per design
-- roughly a minute of process startup to amortise over a rollout lasting
seconds. The alternative is a single scene where env i holds morphology m_i,
stepped in one batch. This tool demonstrates that, and records it.

Embodiments vary along two axes, and the axes have very different costs:

  TOPOLOGY (different joint counts) costs one Articulation VIEW each.
      An Isaac Lab ``Articulation`` reads ``num_joints`` from
      ``root_physx_view.shared_metatype.dof_count`` -- one scalar for the whole
      view -- so robots with different joint counts cannot share one. They can
      still share a SCENE, over disjoint environment subsets. Established by
      ``probe_multi_articulation.py``; here it runs on the production asset
      pipeline (self-collision filters, capsule replacement, the training bake
      props) rather than a simplified one.

  GEOMETRY (same joint count, different link lengths / finger placement) is
      FREE. ``MultiUsdFileCfg`` hands each env of a single view its own USD.
      Established by ``probe_heterogeneous_envs.py`` (5.23 cm reach spread).

Generated hands are all one topology by construction: ghosting pads every design
to the same 37-joint template. So an arbitrary number of *sampled* hands costs
exactly one view -- the view count is bounded by the number of hand FAMILIES,
not by the population size. That is what makes a morphology sweep affordable.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.multi_embodiment_demo \\
        --variants 12 --video videos/multi_embodiment.mp4

Reports, per view: instance and joint counts against the spec, whether every
instance actuates, and -- for the multi-variant view -- the spread in fingertip
reach that proves the envs really did receive different geometry.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Bake props for the robot articulation, copied from the training scene builder
# (scene_utils.build_scene) rather than invented, so this demo exercises the
# same assets the policy trains on.
# Table asset and placement, both from the task config rather than eyeballed:
# assets.table_urdf, and reset.table_reset_z which is the table's CENTRE (the
# surface sits 0.15 m above it, the box being 0.30 m tall).
TABLE_URDF = "assets/urdf/table_narrow.urdf"
TABLE_CENTRE_Z = 0.38

# One colour per embodiment. Chosen to stay legible against BOTH the dark ground
# plane and the light sky band -- a single dark scheme reads well against one and
# disappears into the other. Yellows and oranges are omitted: the iiwa14 is KUKA
# yellow, and a hand in that family stops reading as a separate object.
PALETTE = (
    (0.20, 0.45, 0.85),   # blue
    (0.85, 0.25, 0.30),   # red
    (0.15, 0.65, 0.45),   # green
    (0.60, 0.30, 0.75),   # purple
    (0.10, 0.65, 0.75),   # teal
    (0.90, 0.40, 0.65),   # pink
    (0.35, 0.55, 0.20),   # olive
    (0.75, 0.35, 0.20),   # rust
    (0.25, 0.30, 0.65),   # indigo
    (0.55, 0.70, 0.25),   # lime
    (0.80, 0.20, 0.55),   # magenta
    (0.30, 0.60, 0.90),   # sky
)

_ROBOT_BAKE_PROPS = dict(
    disable_gravity=True,
    max_depenetration_velocity=1000.0,
    enabled_self_collisions=True,
    solver_position_iterations=8,
    solver_velocity_iterations=0,
)


def _parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topologies", default="sharpa_iiwa14,allegro_iiwa14,gen_sharpa_like",
        help="comma-separated robot specs, one Articulation view each")
    parser.add_argument(
        "--variants", type=int, default=8,
        help="sampled hands sharing the generated view; they add embodiments "
             "without adding views")
    parser.add_argument("--envs_per_topology", type=int, default=4)
    parser.add_argument("--no_table", action="store_true",
                        help="omit the task table (it is what makes the arm's "
                             "home pose read as a workspace rather than a pose "
                             "in mid-air)")
    parser.add_argument("--env_spacing", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", default=None, help="write an mp4 here")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=12.0,
                        help="duration of the recorded clip")
    parser.add_argument("--sweep", action="store_true",
                        help="record the hands opening and closing instead of "
                             "holding the home pose")
    parser.add_argument("--no_self_collision", action="store_true",
                        help="convert without self-collision, to test whether "
                             "finger-vs-finger contact is what lifts the tips")
    parser.add_argument("--diag_tip_z", action="store_true",
                        help="report, per env, whether any fingertip RISES "
                             "during the recorded sweep, measured from the sim")
    parser.add_argument("--sweep_frac", type=float, default=0.45,
                        help="how far into each flexion joint's range to close. "
                             "Measured over the cached population, every "
                             "fingertip descends up to ~0.5; past that the "
                             "fingers curl under into a fist and the tips come "
                             "back UP toward the palm (13 of 27 rise at full "
                             "range), which reads as bending the wrong way.")
    parser.add_argument("--hand_color", type=float, nargs=3, default=None,
                        help="one diffuse colour for every hand. Omit to give "
                             "each EMBODIMENT its own colour from the palette, "
                             "which is usually what you want: the point of the "
                             "scene is that the hands differ.")
    parser.add_argument("--no_recolor", action="store_true",
                        help="leave the stock near-white hands alone")
    parser.add_argument("--orbit_deg", type=float, default=70.0,
                        help="degrees the camera orbits over the clip; gives a "
                             "static-pose recording motion without moving the "
                             "robots (0 pins the camera)")
    parser.add_argument("--still", default=None,
                        help="write a single framing PNG here and exit")
    # Framing is over the whole env grid rather than one env, so the camera is
    # placed relative to the grid centroid, not to an env origin as in run_eval.
    parser.add_argument("--camera_eye", type=float, nargs=3, default=(3.2, -4.2, 2.6))
    parser.add_argument("--camera_target", type=float, nargs=3, default=(0.0, 0.0, 1.0))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if args.video or args.still:
        args.enable_cameras = True
    return args


def _prepare_robot_usd(spec, work: Path, tag: str, self_collision: bool = True) -> str:
    """Convert and bake one robot exactly as the training scene builder does."""

    from genmech.tasks.pose_reach.utils.scene_utils import (
        _apply_self_collision_filters,
        _bake_usd,
        _convert_urdf_to_usd,
        _robot_joint_drive_cfg,
    )

    raw = _convert_urdf_to_usd(
        spec.urdf_path, work / "usd",
        fix_base=True, self_collision=self_collision,
        joint_drive=_robot_joint_drive_cfg(),
        replace_cylinders_with_capsules=spec.replace_cylinders_with_capsules,
    )
    _apply_self_collision_filters(raw, spec.adjacent_links)
    return _bake_usd(raw, work / "baked", tag,
                     props=dict(_ROBOT_BAKE_PROPS), apply_physx_articulation=True)


def _sample_variant_specs(count: int, seed: int, work: Path) -> list:
    """Sampled hands sharing the generated topology, so they share a view.

    Loaded from the cached population rather than resampled. The geometric gate
    rejects ~93% of draws and each rejection costs a URDF write, a load and a
    penetration check, so sampling inline burned minutes per run and threw the
    result away. Build the cache once with:

        python -m genmech.tools.build_hand_population --seed 0 --count 64
    """

    from genmech.robots.generated.population import build_population, population_specs

    try:
        specs = population_specs(seed, count)
        if len(specs) >= count:
            print(f"[demo] loaded {count} hands from the cached population "
                  f"(seed {seed})")
            return specs[:count]
        print(f"[demo] cached population for seed {seed} holds only "
              f"{len(specs)} hands; extending to {count}")
    except (FileNotFoundError, KeyError) as exc:
        print(f"[demo] no usable cached population for seed {seed} ({exc}); "
              f"generating one now -- this is slow, and is what the cache exists "
              f"to avoid")

    build_population(seed, count)
    return population_specs(seed, count)


def _articulation_cfg(spec, prim_path: str, usd_paths: list[str]):
    """One view. ``usd_paths`` longer than 1 gives each env its own geometry."""

    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.from_files import UsdFileCfg
    from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg

    spawn = (UsdFileCfg(usd_path=usd_paths[0]) if len(usd_paths) == 1
             else MultiUsdFileCfg(usd_path=list(usd_paths), random_choice=False))
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=spec.base_pos, rot=spec.base_rot,
            joint_pos={**spec.arm_default_joint_pos_resolved(start_arm_higher=False),
                       **spec.hand_default_joint_pos},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(spec.arm_joint_names),
                stiffness=dict(spec.arm_stiffness),
                damping=dict(spec.arm_damping),
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=list(spec.hand_joint_names),
                stiffness=dict(spec.hand_stiffness),
                damping=dict(spec.hand_damping),
                armature=dict(spec.hand_armature),
            ),
        },
    )


def _reach_per_env(art, spec):
    """Palm-to-fingertip distance in each instance of a view.

    Read from body poses rather than from the parameters that generated them,
    so a variant that failed to spawn distinctly shows up as a duplicate
    reading instead of being taken on trust.
    """

    import torch

    names = list(art.data.body_names)
    tips = [n for n in spec.fingertip_body_names if n in names]
    if spec.palm_body_name not in names or not tips:
        return None
    palm = art.data.body_pos_w[:, names.index(spec.palm_body_name), :]
    tip = art.data.body_pos_w[:, names.index(tips[0]), :]
    return torch.linalg.norm(tip - palm, dim=-1)


def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher

    from genmech.robots import get_robot_spec

    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    print(f"[demo] topologies: {topologies}")
    print(f"[demo] sampled variants inside the generated view: {args.variants}")

    app = AppLauncher(args).app

    import tempfile

    import torch
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.utils import configclass

    work = Path(tempfile.mkdtemp(prefix="genmech_multiembodiment_"))
    (work / "usd").mkdir(parents=True, exist_ok=True)

    # ---- Assets -----------------------------------------------------------
    t0 = time.perf_counter()
    views: list[dict] = []
    for name in topologies:
        spec = get_robot_spec(name)
        usds = [_prepare_robot_usd(spec, work, spec.name,
                                   self_collision=not args.no_self_collision)]
        labels = [spec.name]

        # Sampled hands share the generated template's joint count, so they ride
        # in this same view at no extra cost.
        if args.variants > 0 and name.startswith("gen_"):
            for vspec in _sample_variant_specs(args.variants, args.seed, work):
                usds.append(_prepare_robot_usd(
                    vspec, work, vspec.name,
                    self_collision=not args.no_self_collision))
                labels.append(vspec.name)

        views.append(dict(name=name, spec=spec, usds=usds, labels=labels))
        print(f"[demo] {name}: {spec.num_joints} joints, {len(usds)} embodiment(s)")
    convert_s = time.perf_counter() - t0

    # ---- Env allocation ---------------------------------------------------
    # Each view owns a contiguous block wide enough for every one of its USD
    # variants to land somewhere: MultiUsdFileCfg cycles the list in prim order.
    cursor, total = 0, 0
    for v in views:
        width = max(args.envs_per_topology, len(v["usds"]))
        v["envs"] = list(range(cursor, cursor + width))
        cursor += width
        total += width
    print(f"[demo] {total} envs across {len(views)} views: "
          + ", ".join(f"{v['name']}[{v['envs'][0]}..{v['envs'][-1]}]" for v in views))

    # ---- Scene ------------------------------------------------------------
    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=args.device))
    fields = {
        f"robot_{i}": _articulation_cfg(
            v["spec"],
            "/World/envs/env_(" + "|".join(str(e) for e in v["envs"]) + ")/Robot",
            v["usds"])
        for i, v in enumerate(views)
    }

    # The task table, converted and baked exactly as the training scene builder
    # does it (scene_utils.build_scene): fix_base=False, then kinematic so it
    # cannot be shoved around. Without it the arm's home pose reads as a pose in
    # mid-air rather than a hand placed over a workspace.
    if not args.no_table:
        from genmech.tasks.pose_reach.utils.scene_utils import (
            _bake_usd,
            _convert_urdf_to_usd,
            build_rigid_object_cfg,
        )

        table_usd = _bake_usd(
            _convert_urdf_to_usd(TABLE_URDF, work / "usd", fix_base=False),
            work / "baked", "table",
            props=dict(kinematic_enabled=True, disable_gravity=True,
                       articulation_enabled=False),
        )
        fields["table"] = build_rigid_object_cfg("/World/envs/env_.*/Table",
                                                 [table_usd])
    # replicate_physics and clone_in_fabric must both be off: the first clones
    # one env's physics onto the rest, erasing the per-env differences that are
    # the entire point; the second keeps distinct USDs out of the USD stage.
    Cfg = configclass(type(
        "MultiEmbodimentSceneCfg", (InteractiveSceneCfg,),
        {"__annotations__": {k: type(c) for k, c in fields.items()}, **fields}))

    t0 = time.perf_counter()
    scene = InteractiveScene(Cfg(num_envs=total, env_spacing=args.env_spacing,
                                 replicate_physics=False, clone_in_fabric=False))

    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

    # Darken the hands. Done as a USD material bound after spawn rather than by
    # editing URDFs: the generated hands carry SHARPA's material palette, but
    # SHARPA's and Allegro's own colours live in their meshes, so only a
    # stage-level override reaches all three families uniformly.
    #
    # Hand links are "everything under Robot that is not an arm link". The arm
    # link list is the same constant the URDF builder splices the arm in with,
    # so the two cannot drift apart; all three robots share this arm.
    if not args.no_recolor:
        import omni.usd
        from pxr import UsdShade

        from genmech.tools.build_hand_urdf import ARM_LINKS

        stage = omni.usd.get_context().get_stage()
        arm_links = set(ARM_LINKS)

        def _material(name: str, rgb):
            cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(rgb),
                                              roughness=0.5, metallic=0.0)
            path = f"/World/Looks/{name}"
            cfg.func(path, cfg)
            return UsdShade.Material.Get(stage, path)

        # One material per EMBODIMENT, keyed by USD, so two envs holding the same
        # design share a colour and different designs never do.
        if args.hand_color is not None:
            shades = {}
            single = _material("Hand", args.hand_color)
        else:
            shades = {}
            all_usds = [u for v in views for u in v["usds"]]
            for i, usd in enumerate(dict.fromkeys(all_usds)):
                shades[usd] = _material(f"Hand{i:03d}", PALETTE[i % len(PALETTE)])
            single = None

        painted = 0
        for v in views:
            for slot, e in enumerate(v["envs"]):
                robot_prim = stage.GetPrimAtPath(f"/World/envs/env_{e}/Robot")
                if not robot_prim.IsValid():
                    continue
                # MultiUsdFileCfg cycles its list in prim order, so env slot k of
                # this view holds usds[k % len(usds)] -- the same rule the
                # spawner used, not a guess about which env got which design.
                usd = v["usds"][slot % len(v["usds"])]
                shade = single if single is not None else shades[usd]
                for link in robot_prim.GetChildren():
                    if link.GetName() in arm_links:
                        continue
                    UsdShade.MaterialBindingAPI.Apply(link).Bind(shade)
                    painted += 1
        n_colors = 1 if single is not None else len(shades)
        print(f"[demo] recoloured {painted} hand links using {n_colors} colour(s)")

    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light)

    sim.reset()
    build_s = time.perf_counter() - t0
    for i, v in enumerate(views):
        v["art"] = scene[f"robot_{i}"]

    def _apply_home_pose():
        """Put every articulation INTO its home pose and command it to stay.

        ``sim.reset()`` does not do this: it initializes the physics views but
        leaves each implicit actuator's position target at zero, so the PD
        controller drives the robot to the all-zeros configuration -- an iiwa14
        standing bolt upright -- silently overriding the init_state home pose.
        The state write says where the robot IS, the target write says where the
        controller wants it; both are needed or it snaps straight back.

        Called after EVERY sim.reset(), because each one re-zeroes the targets.
        """
        for v in views:
            art = v["art"]
            q0 = art.data.default_joint_pos.clone()
            art.write_joint_state_to_sim(q0, torch.zeros_like(q0))
            art.set_joint_position_target(q0)
            art.write_data_to_sim()
            v["q_home"] = q0

    _apply_home_pose()

    # Place the table the way reset_utils._reset_table_pose does: local
    # (0, 0, table_reset_z) plus the env origin, identity rotation. The env
    # randomizes xy/yaw/z around this; here it stays nominal so every env shows
    # the same workspace and only the hand differs.
    def _place_table():
        """Put the table where reset_utils._reset_table_pose puts it.

        Must be re-applied after EVERY sim.reset(), for the same reason the home
        pose must be: a reset restores each asset's init_state, and this table's
        init_state is the origin. Applying it once and then resetting again
        leaves the table sitting on the floor -- 38 cm below where the policy
        expects the work surface.
        """
        if args.no_table:
            return
        table = scene["table"]
        pose = torch.zeros(total, 7, device=sim.device)
        pose[:, :3] = scene.env_origins
        pose[:, 2] += TABLE_CENTRE_Z
        pose[:, 3] = 1.0
        table.write_root_pose_to_sim(pose)

    _place_table()
    if not args.no_table:
        print(f"[demo] table centre z={TABLE_CENTRE_Z} (surface {TABLE_CENTRE_Z + 0.15})")

    # ---- Camera over the whole grid ---------------------------------------
    camera = None
    if args.video or args.still:
        from isaaclab.sensors import Camera, CameraCfg

        camera = Camera(cfg=CameraCfg(
            prim_path="/World/DemoCamera", update_period=0,
            height=720, width=1280, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0,
                horizontal_aperture=20.955, clipping_range=(0.1, 200.0)),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 10.0), rot=(1.0, 0.0, 0.0, 0.0),
                                       convention="opengl"),
        ))
        sim.reset()
        # This reset re-zeroed the joint targets AND restored the table to its
        # init_state at the origin; both have to be re-applied.
        _apply_home_pose()
        _place_table()
        grid_centre = scene.env_origins.mean(dim=0)
        cam_target = grid_centre + torch.tensor(args.camera_target, device=sim.device)
        eye = grid_centre + torch.tensor(args.camera_eye, device=sim.device)
        camera.set_world_poses_from_view(eye.unsqueeze(0), cam_target.unsqueeze(0))
        print(f"[demo] camera eye={eye.tolist()} target={cam_target.tolist()}")

    for _ in range(10):
        sim.step()
        scene.update(1 / 120.0)

    # Is the arm actually in the task's home pose? Checked against the spec's
    # own numbers rather than judged by eye -- an all-zeros arm looks like a
    # plausible pose in a render, which is exactly how it went unnoticed.
    for v in views:
        art, spec = v["art"], v["spec"]
        names = list(art.data.joint_names)
        want = spec.arm_default_joint_pos_resolved(start_arm_higher=False)
        err = max(abs(float(art.data.joint_pos[0, names.index(j)]) - q)
                  for j, q in want.items())
        flag = "" if err < 1e-2 else "   <-- NOT AT HOME POSE"
        print(f"[demo] {v['name']:<18} arm home-pose error {err * 1000:7.3f} mrad{flag}")

    # And check the table ended up where it was put. It is kinematic with gravity
    # disabled so it cannot fall -- but a sim.reset() teleports it back to its
    # init_state at the origin, which looks identical to falling and is what
    # actually happened here.
    if not args.no_table:
        z = scene["table"].data.root_pos_w[:, 2] - scene.env_origins[:, 2]
        err = float((z - TABLE_CENTRE_Z).abs().max())
        flag = "" if err < 1e-3 else f"   <-- WRONG, expected centre {TABLE_CENTRE_Z}"
        print(f"[demo] table centre z={float(z.min()):.4f}..{float(z.max()):.4f} "
              f"(surface {float(z.min()) + 0.15:.4f}), error {err * 1000:.2f} mm{flag}")

    for v in views:
        art = v["art"]
        v["q_before"] = art.data.joint_pos.clone()
        v["lower"] = art.data.joint_pos_limits[..., 0]
        v["upper"] = art.data.joint_pos_limits[..., 1]
        # Sweep the HAND only. Driving the arm to its limits too folds it into a
        # crumple that hides the hand -- and the hand is the subject here. The
        # arm holds the settled default pose, presenting the hand to camera.
        names = list(art.data.joint_names)
        hand = {n for n in v["spec"].hand_joint_names}
        # FLEXION joints only. Sweeping the abduction joints too drives them to
        # the end of their range as well, which SPLAYS the fingers apart -- the
        # opposite of closing, and it reads on camera as fingers bending
        # outwards. A grasp flexes FE/PIP/DIP and leaves abduction near neutral.
        v["hand_cols"] = torch.tensor(
            [i for i, n in enumerate(names)
             if n in hand and n.rsplit("_", 1)[-1] in ("FE", "PIP", "DIP")],
            device=art.data.joint_pos.device)
        v["all_hand_cols"] = torch.tensor(
            [i for i, n in enumerate(names) if n in hand],
            device=art.data.joint_pos.device)
        # Hold the DEFAULT pose, not the settled one. Reading back the settled
        # pose would silently bake in any drift away from home.
        v["q_hold"] = v["q_home"].clone()

    if args.still:
        # Several candidate framings in ONE Kit boot. Booting Kit and converting
        # the robots costs minutes; moving a camera costs nothing, so guessing
        # one angle per run is the expensive way to do this.
        import imageio

        centroid = scene.env_origins.mean(dim=0)
        candidates = {
            "wide": ((3.2, -4.2, 2.6), (0.0, 0.0, 1.0)),
            "mid": ((2.2, -2.8, 1.9), (0.0, 0.0, 1.25)),
            "close": ((1.4, -1.8, 1.5), (0.0, 0.0, 1.30)),
            "hands": ((0.9, -1.2, 1.35), (0.0, 0.0, 1.32)),
        }
        out_dir = Path(args.still).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.still).stem
        for label, (eye_l, tgt_l) in candidates.items():
            eye = centroid + torch.tensor(eye_l, device=sim.device)
            tgt = centroid + torch.tensor(tgt_l, device=sim.device)
            camera.set_world_poses_from_view(eye.unsqueeze(0), tgt.unsqueeze(0))
            sim.step()
            camera.update(0.0)
            path = out_dir / f"{stem}_{label}.png"
            imageio.imwrite(path, camera.data.output["rgb"][0].cpu().numpy()[:, :, :3])
            print(f"[demo] framing '{label}' eye={eye_l} target={tgt_l} -> {path}")
        del app
        os._exit(0)

    import math

    def _hand_target(v, phase, cols_key="hand_cols"):
        """Home pose, with the HAND joints driven to a fraction of their range.

        A fraction rather than a fixed angle, because the same absolute angle is
        meaningless across a 90-degree finger joint and a 2-degree abduction.
        The arm holds its home pose throughout -- driving the arm to its limits
        folds it into a crumple that buries the hand.
        """
        target = v["q_hold"].clone()
        cols = v[cols_key]
        # HOME -> upper limit, not lower -> upper. The lower limit is maximum
        # EXTENSION, so sweeping the full range makes every finger travel
        # backwards past its rest pose first -- on camera the hand opens further
        # before it closes, which reads as the fingers bending the wrong way.
        # Interpolating from the home pose means the only motion is closing.
        home = v["q_hold"][:, cols]
        target[:, cols] = home + phase * (v["upper"][:, cols] - home)
        return target

    def _advance(n):
        for _ in range(n):
            for v in views:
                v["art"].write_data_to_sim()
            sim.step()
            scene.update(1 / 120.0)

    # ---- Actuation probe --------------------------------------------------
    # Run this BEFORE recording, whatever the clip shows. The verification below
    # asks whether every instance actually actuates, and a static recording
    # would otherwise report every joint as motionless and call it a failure.
    # The probe drives EVERY hand joint, abduction included -- the check is that
    # each joint actuates, which is a different question from what the recorded
    # clip should show.
    for v in views:
        v["art"].set_joint_position_target(_hand_target(v, 0.75, "all_hand_cols"))
    _advance(90)
    for v in views:
        v["q_probe"] = v["art"].data.joint_pos.clone()

    # Back to the home pose, and let it settle before the camera rolls.
    for v in views:
        v["art"].set_joint_position_target(v["q_hold"])
    _advance(120)

    # ---- Record -----------------------------------------------------------
    # Default is every robot holding the SAME home pose, so the only thing that
    # differs across envs is the hand itself. The camera orbits instead, which
    # gives the clip motion without implying the robots are doing a task.
    steps_per_frame = max(1, round((1.0 / args.video_fps) / (1 / 120.0)))
    n_frames = int(args.seconds * args.video_fps) if args.video else 120
    frames = []

    t0 = time.perf_counter()
    for f in range(n_frames):
        frac = f / max(n_frames - 1, 1)
        if args.sweep:
            phase = args.sweep_frac * (0.5 - 0.5 * math.cos(2 * math.pi * frac * 2))
            for v in views:
                v["art"].set_joint_position_target(_hand_target(v, phase))
        else:
            for v in views:
                v["art"].set_joint_position_target(v["q_hold"])

        if camera is not None and args.orbit_deg:
            theta = math.radians(args.orbit_deg) * (frac - 0.5)
            ex, ey, ez = args.camera_eye
            eye = grid_centre + torch.tensor(
                (ex * math.cos(theta) - ey * math.sin(theta),
                 ex * math.sin(theta) + ey * math.cos(theta), ez),
                device=sim.device)
            camera.set_world_poses_from_view(eye.unsqueeze(0), cam_target.unsqueeze(0))

        _advance(steps_per_frame)

        # Per-env fingertip height, read from the SIM rather than predicted from
        # URDF kinematics. The two can disagree: a joint that cannot reach its
        # target (contact, torque limit, a ghosted joint being dragged) moves
        # differently than the kinematics say it should, and the camera shows
        # the sim's answer.
        if args.diag_tip_z:
            for v in views:
                art = v["art"]
                if "tip_rows" not in v:
                    names = list(art.data.body_names)
                    v["tip_idx"] = [names.index(b) for b in v["spec"].fingertip_body_names
                                    if b in names]
                    v["tip_rows"] = []
                if v["tip_idx"]:
                    z = art.data.body_pos_w[:, v["tip_idx"], 2]
                    v["tip_rows"].append(z.clone())

        if camera is not None:
            camera.update(0.0)
            rgb = camera.data.output["rgb"]
            if rgb is not None and rgb.shape[0] > 0:
                frames.append(rgb[0].cpu().numpy()[:, :, :3].copy())
    sweep_s = time.perf_counter() - t0

    if frames:
        import imageio
        out = Path(args.video)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(out), frames, fps=args.video_fps)
        print(f"[demo] wrote {len(frames)} frames -> {out}")

    if args.diag_tip_z:
        # Do the per-env joint LIMITS actually differ? MultiUsdFileCfg gives each
        # env its own USD, but a single Articulation view reports one buffer --
        # if limits were broadcast from env 0, every hand would be driven to the
        # reference hand's limits and move in ways its own URDF never allows.
        for v in views:
            lim = v["art"].data.joint_pos_limits
            spread = (lim.max(dim=0).values - lim.min(dim=0).values).max()
            print(f"[diag] {v['name']}: joint-limit spread across envs = "
                  f"{float(spread):.6f} rad "
                  f"({'PER-ENV' if float(spread) > 1e-6 else 'IDENTICAL -- broadcast!'})")
        print(f"\n[diag] fingertip height during the sweep, per env "
              f"(sim, not kinematics)")
        for v in views:
            rows = v.get("tip_rows")
            if not rows:
                continue
            z = torch.stack(rows)                    # (frames, envs, tips)
            rise = (z - z[0:1]).max(dim=0).values    # worst rise per env,tip
            tips = [b for b in v["spec"].fingertip_body_names
                    if b in list(v["art"].data.body_names)]
            for e in range(z.shape[1]):
                worst = float(rise[e].max())
                usd = Path(v["usds"][e % len(v["usds"])]).stem
                if worst <= 0.005:
                    continue
                per_tip = "  ".join(
                    f"{tips[t].replace('gen_', '')}:{float(rise[e, t]) * 100:+.1f}"
                    for t in range(rise.shape[1]))
                print(f"[diag]   env {v['envs'][e]:>3} {usd:<20} "
                      f"worst {worst * 100:+6.2f} cm   [{per_tip}]")

    # ---- Report -----------------------------------------------------------
    print(f"\n{'=' * 78}")
    print(f"[demo] convert+bake {convert_s:6.1f}s   scene build {build_s:6.1f}s   "
          f"sweep {sweep_s:6.1f}s")
    print(f"{'=' * 78}")
    print(f"{'view':<20} {'envs':>5} {'joints':>7} {'spec':>5} {'moved':>8} "
          f"{'max dq':>8}  {'reach':>16}")

    failures = []
    for v in views:
        art, spec = v["art"], v["spec"]
        # From the probe, not from the current pose: the recorded clip may have
        # ended back at the home pose, where nothing has moved by definition.
        dq = (v["q_probe"] - v["q_before"]).abs()
        # Every INSTANCE must move, not just the view on average: a variant that
        # silently failed to spawn its actuators would hide behind its
        # neighbours in a mean.
        moved = int((dq.max(dim=1).values > 1e-3).sum())

        reach = _reach_per_env(art, spec)
        if reach is None:
            desc = "-"
        elif len(v["usds"]) > 1:
            desc = f"{(reach.max() - reach.min()).item() * 100:.2f} cm spread"
        else:
            desc = f"{reach.mean().item() * 100:.1f} cm"

        ok = art.num_joints == spec.num_joints and moved == art.num_instances
        if not ok:
            failures.append(v["name"])
        print(f"{v['name']:<20} {art.num_instances:>5} {art.num_joints:>7} "
              f"{spec.num_joints:>5} {moved:>5}/{art.num_instances:<2} "
              f"{dq.max().item():>7.3f}  {desc:>16}" + ("" if ok else "   <-- FAILED"))

        # A multi-variant view whose envs all read the same reach means the
        # variants collapsed onto one template -- what replicate_physics=True
        # would silently do.
        if len(v["usds"]) > 1 and reach is not None:
            if (reach.max() - reach.min()).item() < 1e-4:
                failures.append(f"{v['name']} (variants collapsed)")

    distinct = sorted({v["art"].num_joints for v in views})
    n_embodiments = sum(len(v["usds"]) for v in views)
    print(f"\n[demo] {len(distinct)} distinct topologies coexisting: {distinct} joints")
    print(f"[demo] {n_embodiments} distinct embodiments in {len(views)} views "
          f"over {total} envs")

    if failures:
        print(f"\n[demo] FAILED: {failures}")
    else:
        print("\n[demo] OK: every view reports its spec's joint count, every "
              "instance actuates, and each multi-variant view's envs differ.")

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1 if failures else 0)


if __name__ == "__main__":
    main()
