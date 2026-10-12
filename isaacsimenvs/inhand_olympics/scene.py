"""Scene: the hand on its dummy wrist (one articulation) above a kinematic table, a ground plane and a light."""
from __future__ import annotations

import tempfile
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from ..pose_reaching_6d.common_utils.urdf_to_usd import (
    _apply_self_collision_filters, _convert_urdf_to_usd, _robot_joint_drive_cfg,
)
from ..pose_reaching_6d.scene_utils.author_robot import flatten_robot_usd
from ..pose_reaching_6d.scene_utils.robots import get_robot_spec
from .wrist import WRIST_JOINTS, build_wrist_urdf

ROBOT_PATH = "/World/envs/env_.*/Robot"
TABLE_PATH = "/World/envs/env_.*/Table"


def _actuators(spec, cfg) -> dict:
    w = cfg.wrist
    hand = list(spec.hand_joint_names)
    return {
        "wrist_prismatic": ImplicitActuatorCfg(
            joint_names_expr=list(WRIST_JOINTS[:3]), stiffness=w.prismatic_stiffness, damping=w.prismatic_damping,
            effort_limit_sim=w.prismatic_effort_limit, velocity_limit_sim=w.prismatic_velocity_limit, friction=0.0),
        "wrist_revolute": ImplicitActuatorCfg(
            joint_names_expr=list(WRIST_JOINTS[3:]), stiffness=w.revolute_stiffness, damping=w.revolute_damping,
            effort_limit_sim=w.revolute_effort_limit, velocity_limit_sim=w.revolute_velocity_limit, friction=0.0),
        # The hand's own gains, as every hand-only run uses them (pose_reaching_6d assembly._actuator_groups).
        "hand": ImplicitActuatorCfg(
            joint_names_expr=hand, stiffness=dict(spec.hand_stiffness), damping=dict(spec.hand_damping),
            armature=dict(spec.hand_armature), friction=0.0,
            velocity_limit_sim=cfg.hand.velocity_limit or None),
    }


def setup_scene(env) -> None:
    cfg = env.cfg
    spec = get_robot_spec(cfg.hand.robot_spec)
    n_joints = len(WRIST_JOINTS) + len(spec.hand_joint_names)
    if cfg.action_space not in (0, n_joints):
        raise ValueError(f"action_space={cfg.action_space}, but the robot has {n_joints} joints")
    # DirectRLEnv reads the spaces after this hook. Placeholder observation: joint positions and velocities.
    cfg.action_space = n_joints
    cfg.observation_space = 2 * n_joints

    work = Path(tempfile.mkdtemp(prefix="olympics_assets_"))
    urdf = build_wrist_urdf(spec, cfg.wrist, work / "hand_on_wrist.urdf")
    converted = _convert_urdf_to_usd(
        str(urdf), work, fix_base=True, self_collision=True, joint_drive=_robot_joint_drive_cfg(),
        replace_cylinders_with_capsules=spec.replace_cylinders_with_capsules)
    _apply_self_collision_filters(converted, spec.adjacent_links)
    flat, _ = flatten_robot_usd(converted, work / "robot_flat.usd",
                                contact_offset=cfg.contact_offset, rest_offset=cfg.rest_offset)

    top = cfg.table.top_height
    pose = dict(zip(WRIST_JOINTS, cfg.wrist.default_pose))
    env.robot = Articulation(ArticulationCfg(
        prim_path=ROBOT_PATH,
        spawn=sim_utils.UsdFileCfg(usd_path=flat),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, top), rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={**pose, **spec.hand_default_joint_pos}, joint_vel={".*": 0.0}),
        actuators=_actuators(spec, cfg),
    ))
    sx, sy, sz = cfg.table.size
    env.table = RigidObject(RigidObjectCfg(
        prim_path=TABLE_PATH,
        spawn=sim_utils.CuboidCfg(
            size=(sx, sy, sz),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=cfg.contact_offset,
                                                             rest_offset=cfg.rest_offset),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=cfg.table.friction,
                                                            dynamic_friction=cfg.table.friction),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.30)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, top - sz / 2)),
    ))
    spawn_ground_plane("/World/ground", GroundPlaneCfg())
    env.scene.clone_environments(copy_from_source=False)
    env.scene.filter_collisions(global_prim_paths=["/World/ground"])
    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["table"] = env.table
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    env.robot_spec = spec
    env.wrist_urdf_path = urdf
