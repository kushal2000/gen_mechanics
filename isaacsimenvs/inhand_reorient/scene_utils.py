"""Scene: one fixed, hand-only robot, palm-up, no arm and no table; a cube to
reorient; a non-colliding goal marker.

Uses Isaac Lab's ordinary regex spawner (``replicate_physics=True``) rather
than ``pose_reaching_6d.scene_utils.assembly``'s manual per-env Sdf-reference
authoring: that manual path exists there to support per-env DISTINCT designs
and objects cheaply, which Phase 1 does not need (one hand, one cube size).
The tradeoff is a slower one-time scene boot for a fixed hand (the
``assembly.py`` docstring measures ~110 s) against much less new code; worth
it here, and worth revisiting if boot time becomes the bottleneck once a
population is involved (Phase 2).
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from isaacsimenvs.pose_reaching_6d.scene_utils.assembly import _convert_fixed_robot

from .hand_only import build_hand_only_spec
from .obs_utils import derive_spaces

ROBOT_PATH = "/World/envs/env_.*/Robot"
OBJECT_PATH = "/World/envs/env_.*/Object"
GOALVIZ_PATH = "/World/envs/env_.*/GoalViz"

__all__ = ["setup_scene", "finalize_scene"]


def _hand_articulation_cfg(spec, usd_path: str) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=ROBOT_PATH,
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=spec.base_pos, rot=spec.base_rot,
            joint_pos=dict(spec.hand_default_joint_pos),
            joint_vel={".*": 0.0},
        ),
        actuators={
            "hand": ImplicitActuatorCfg(
                joint_names_expr=list(spec.hand_joint_names),
                stiffness=dict(spec.hand_stiffness),
                damping=dict(spec.hand_damping),
                armature=dict(spec.hand_armature),
                friction=0.0,
            ),
        },
    )


def _object_cfg(prim_path: str, size: float, offsets: dict, *, kinematic: bool,
                density: float = 500.0, color: tuple[float, float, float] = (0.2, 0.4, 0.9)
                ) -> RigidObjectCfg:
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=kinematic, disable_gravity=kinematic)
    collision_props = None if kinematic else sim_utils.CollisionPropertiesCfg(**offsets)
    mass_props = None if kinematic else sim_utils.MassPropertiesCfg(density=density)
    spawn = sim_utils.CuboidCfg(
        size=(size, size, size),
        rigid_props=rigid_props, mass_props=mass_props, collision_props=collision_props,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )
    return RigidObjectCfg(prim_path=prim_path, spawn=spawn)


def setup_scene(env) -> None:
    t0 = time.perf_counter()
    asset_dir = Path(tempfile.mkdtemp(prefix="inhand_reorient_"))

    spec, cut = build_hand_only_spec(env.cfg.assets.hand_id, out_dir=asset_dir)
    derive_spaces(env.cfg, spec)
    print(f"[inhand_reorient] hand={env.cfg.assets.hand_id} joints={spec.num_hand_joints} "
          f"root={spec.hand_root} unresolved_meshes={cut.unresolved_meshes} "
          f"({time.perf_counter() - t0:.1f}s)", flush=True)

    offsets = dict(contact_offset=env.cfg.physics.contact_offset,
                   rest_offset=env.cfg.physics.rest_offset)
    robot_usd, _robot_root = _convert_fixed_robot(spec, spec.urdf_path, asset_dir / "usd", offsets)
    print(f"[inhand_reorient] converted hand USD ({time.perf_counter() - t0:.1f}s)", flush=True)

    env.robot = Articulation(_hand_articulation_cfg(spec, robot_usd))
    env.object = RigidObject(_object_cfg(
        OBJECT_PATH, env.cfg.assets.object_size_m, offsets, kinematic=False,
        density=env.cfg.assets.object_density))
    env.goal_viz = RigidObject(_object_cfg(
        GOALVIZ_PATH, env.cfg.assets.object_size_m, offsets, kinematic=True,
        color=(0.9, 0.3, 0.2)))

    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # Clone env 0's prim subtree into every other env (the standard Isaac Lab
    # regex-spawner pattern; pose_reaching_6d skips this because it authors
    # every env's prims itself instead -- see this module's docstring).
    env.scene.clone_environments(copy_from_source=False)
    if env.device == "cpu":
        env.scene.filter_collisions(global_prim_paths=[])

    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["object"] = env.object
    env.scene.rigid_objects["goal_viz"] = env.goal_viz

    env.hand_spec = spec
    env.hand_cut = cut
    print(f"[inhand_reorient] scene ready ({time.perf_counter() - t0:.1f}s)", flush=True)


def finalize_scene(env) -> None:
    """Needs the started sim: resolve the palm body index off the live
    articulation view (fix_base means it is the articulation's own root
    body, but its index in body_names is whatever PhysX assigned)."""
    body_names = env.robot.data.body_names
    if env.hand_spec.palm_body_name not in body_names:
        raise RuntimeError(
            f"palm body {env.hand_spec.palm_body_name!r} not among the articulation's "
            f"bodies {body_names}")
    env.palm_body_idx = body_names.index(env.hand_spec.palm_body_name)
    if env.robot.data.joint_pos.shape[0] != env.num_envs:
        raise RuntimeError(
            f"articulation view has {env.robot.data.joint_pos.shape[0]} envs, "
            f"expected {env.num_envs}")
    if env.robot.data.joint_pos.shape[1] != env.hand_spec.num_hand_joints:
        raise RuntimeError(
            f"articulation view has {env.robot.data.joint_pos.shape[1]} joints, "
            f"the spec has {env.hand_spec.num_hand_joints}")
    print(f"[inhand_reorient] articulation view: {env.robot.data.joint_pos.shape[0]} envs x "
          f"{env.robot.data.joint_pos.shape[1]} joints, palm_body_idx={env.palm_body_idx}",
          flush=True)
