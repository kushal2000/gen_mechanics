"""Scene: the hand on its dummy wrist (one articulation) above a kinematic table, the pen lying on the table, a
ground plane and a light. With ``hand.robot_spec = "multi:<spec>+<spec>+..."`` several hands, each on its own envs
(round-robin) and its own articulation, which ``env.robot`` presents as one (multi_hand.py)."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils import get_current_stage
import numpy as np
import torch
from pxr import Sdf

from isaacsimenvs.common.urdf_to_usd import (
    _apply_self_collision_filters, _convert_urdf_to_usd, _robot_joint_drive_cfg,
)
from hand_sampler.design_space import compute_mass_and_inertia
from isaacsimenvs.common.author_objects import author_handle_head, author_physics_material
from isaacsimenvs.common.usd_flatten import flatten_robot_usd
from isaacsimenvs.common.sdf import attr, define, set_xform
from isaacsimenvs.common.robots import get_robot_spec
from isaacsimenvs.common.robots.multi_hand import hand_set, is_multi_ref
from .wrist import WRIST_JOINTS, build_wrist_urdf

ROBOT_PATH = "/World/envs/env_.*/Robot"
TABLE_PATH = "/World/envs/env_.*/Table"
PEN_PATH = "/World/envs/env_.*/Pen"


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


def _hand_usd(spec, cfg, work: Path) -> tuple[Path, str, str]:
    """The hand on its wrist: (URDF, flattened USD, its root prim path)."""
    work.mkdir(parents=True, exist_ok=True)
    urdf = build_wrist_urdf(spec, cfg.wrist, work / "hand_on_wrist.urdf")
    converted = _convert_urdf_to_usd(
        str(urdf), work, fix_base=True, self_collision=True, joint_drive=_robot_joint_drive_cfg(),
        replace_cylinders_with_capsules=spec.replace_cylinders_with_capsules, collider_type=cfg.hand.collider_type)
    _apply_self_collision_filters(converted, spec.adjacent_links)
    flat, root = flatten_robot_usd(converted, work / "robot_flat.usd",
                                   contact_offset=cfg.contact_offset, rest_offset=cfg.rest_offset)
    return urdf, flat, root


def _robot_cfg(spec, cfg, prim_path: str, usd: str | None) -> ArticulationCfg:
    pose = dict(zip(WRIST_JOINTS, cfg.wrist.default_pose))
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(usd_path=usd) if usd else None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, cfg.table.top_height), rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={**pose, **spec.hand_default_joint_pos}, joint_vel={".*": 0.0}),
        actuators=_actuators(spec, cfg),
    )


def _block_regex(env_ids) -> str:
    """``/World/envs/env_(i|j|...)/Robot`` over one hand's envs (pose_reaching_6d assembly._block_regex)."""
    ids = [str(int(i)) for i in env_ids]
    return f"/World/envs/env_({'|'.join(ids)})/Robot" if len(ids) > 1 else f"/World/envs/env_{ids[0]}/Robot"


def setup_scene(env) -> None:
    cfg = env.cfg
    multi = is_multi_ref(cfg.hand.robot_spec)
    hs = hand_set(cfg.hand.robot_spec) if multi else None
    spec = hs.template if multi else get_robot_spec(cfg.hand.robot_spec)
    n_joints = len(WRIST_JOINTS) + len(spec.hand_joint_names)
    if cfg.action_space not in (0, n_joints):
        raise ValueError(f"action_space={cfg.action_space}, but the robot has {n_joints} joints")
    # DirectRLEnv reads the spaces after this hook (the env sets observation_space from this).
    cfg.action_space = n_joints

    work = Path(tempfile.mkdtemp(prefix="olympics_assets_"))
    top = cfg.table.top_height
    if multi:
        # Each hand converted once; referenced into its envs once they exist (below, after cloning).
        hands = [_hand_usd(s, cfg, work / f"hand{h}") for h, s in enumerate(hs.specs)]
        hand_of_env = hs.hand_of_env(env.num_envs)
        urdf = hands[int(hand_of_env[0])][0]
    else:
        urdf, flat, _ = _hand_usd(spec, cfg, work)
        hand_of_env = np.zeros(env.num_envs, dtype=np.int64)
        env.robot = Articulation(_robot_cfg(spec, cfg, ROBOT_PATH, flat))
    sx, sy, sz = cfg.table.size
    env.table = RigidObject(RigidObjectCfg(
        prim_path=TABLE_PATH,
        spawn=sim_utils.CuboidCfg(
            size=(sx, sy, sz),
            # As in-hand's table (assembly._table_props): kinematic, no gravity.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=cfg.contact_offset,
                                                             rest_offset=cfg.rest_offset),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=cfg.table.friction,
                                                            dynamic_friction=cfg.table.friction),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.30)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, top - sz / 2)),
    ))
    # The pen, authored exactly as in-hand reorientation authors its objects (author_objects.author_handle_head: a
    # handle-only object, i.e. one capsule or box along +x, its rigid-body and collider properties, inertia from
    # density),
    # into env_0; clone_environments copies it.
    pen = cfg.pen
    handle = density = None
    if pen.twist:                                         # twist_pen.py checks its own shapes (box/capsule/cylinder)
        pass
    elif pen.shape == "capsule":
        handle = (pen.length - pen.diameter, pen.diameter)    # (cylinder length, diameter): a capsule along +x
    elif pen.shape == "box":
        handle = (pen.length, pen.diameter, pen.diameter)     # (x, y, z): a box, its long axis along +x
    else:
        raise ValueError(f"pen.shape must be 'capsule' or 'box', got {pen.shape!r}")
    if handle is not None:
        density = pen.mass / compute_mass_and_inertia(handle, 1.0)[0]
    layer = get_current_stage().GetRootLayer()
    pen_init = dict(pos=(pen.init_xy[0], pen.init_xy[1], top + pen.diameter / 2),
                    rot=(math.cos(pen.init_yaw / 2), 0.0, 0.0, math.sin(pen.init_yaw / 2)))
    if pen.twist:
        _twist_pen(env, cfg, layer, pen_init)
    else:
        _rigid_pen(env, cfg, layer, pen_init, handle, density)
    spawn_ground_plane("/World/ground", GroundPlaneCfg())
    # A multi-hand scene clones by COPY: inheriting clones (copy_from_source=False) compose env_0's own hand
    # reference into every env, and an inherited opinion beats an env's own reference, so every env got env_0's hand
    # merged over its own (found 15 Oct: in a 6-primitive-hand scene every env had hand 0's finger bearings).
    env.scene.clone_environments(copy_from_source=multi)
    if multi:
        with Sdf.ChangeBlock():
            for env_path in env.scene.env_prim_paths:
                h = int(hand_of_env[int(env_path.rsplit("_", 1)[1])])
                _, usd, root_path = hands[h]
                prim = define(layer, f"{env_path}/Robot", "Xform")
                prim.referenceList.explicitItems.append(Sdf.Reference(usd, Sdf.Path(root_path)))
                set_xform(layer.GetPrimAtPath(f"{env_path}/Robot"), (0.0, 0.0, float(top)))
        from .multi_hand import OlympicsMultiHand
        arts = [Articulation(_robot_cfg(s, cfg, _block_regex(np.nonzero(hand_of_env == h)[0]), None))
                for h, s in enumerate(hs.specs)]
        env.robot = OlympicsMultiHand(hs, arts, torch.as_tensor(hand_of_env), env.device)
        counts = {s.name: int((hand_of_env == h).sum()) for h, s in enumerate(hs.specs)}
        print(f"[olympics] hands dealt round-robin: {counts}", flush=True)
    env.scene.filter_collisions(global_prim_paths=["/World/ground"])
    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["table"] = env.table
    if cfg.pen.twist:
        env.scene.articulations["pen"] = env.pen
    else:
        env.scene.rigid_objects["pen"] = env.pen
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    env.robot_spec = spec                      # a multi-hand scene: the padded template (hs.template)
    env.hand_set = hs
    env.hand_of_env = torch.as_tensor(hand_of_env, device=env.device)
    env.wrist_urdf_path = urdf                 # env 0's hand (the pose viewer's)
    env.hand_urdf_paths = [h[0] for h in hands] if multi else [urdf]   # per hand, env i's is [hand_of_env[i]]


def _float(layer, bodies: list[str]) -> None:
    """Gravity off on these bodies (PenCfg.gravity False)."""
    for b in bodies:
        attr(layer.GetPrimAtPath(b), "physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool, True)


def _twist_pen(env, cfg, layer, pen_init: dict) -> None:
    """The twist-apart pen (twist_pen.py): barrel and cap on a revolute joint, one articulation, into env_0
    (clone_environments copies it)."""
    from .twist_pen import JOINT, LINKS, author_twist_pen
    pen = cfg.pen
    with Sdf.ChangeBlock():
        mat = author_physics_material(layer, "/World/PhysicsMaterials/pen",
                                      static_friction=pen.friction, dynamic_friction=pen.friction)
        author_twist_pen(layer, "/World/envs/env_0/Pen", length=pen.length, diameter=pen.diameter,
                         cap_length=pen.cap_length, mass=pen.mass, friction=pen.friction, shape=pen.shape,
                         material_path=mat, damping=pen.twist_damping, friction_torque=pen.twist_friction_torque,
                         fixed_barrel=pen.twist_fixed_barrel)
        if not pen.gravity:
            _float(layer, [f"/World/envs/env_0/Pen/{n}" for n in LINKS])
    env.pen = Articulation(ArticulationCfg(
        prim_path=PEN_PATH, spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(**pen_init, joint_pos={JOINT: 0.0}, joint_vel={JOINT: 0.0}),
        actuators={"twist": ImplicitActuatorCfg(joint_names_expr=[JOINT], stiffness=0.0, damping=pen.twist_damping,
                                                effort_limit_sim=pen.twist_friction_torque)},
    ))


def _rigid_pen(env, cfg, layer, pen_init: dict, handle, density) -> None:
    """The one-piece pen, into env_0 (clone_environments copies it)."""
    pen = cfg.pen
    with Sdf.ChangeBlock():
        mat = author_physics_material(layer, "/World/PhysicsMaterials/pen",
                                      static_friction=pen.friction, dynamic_friction=pen.friction)
        define(layer, "/World/envs/env_0/Pen", "Xform")
        author_handle_head(layer, "/World/envs/env_0/Pen", handle, None, density, None,
                           collision=True, material_path=mat)
        # PhysX's rolling/torsion knobs (PenCfg), over author_handle_head's values when set.
        def _set(path: str, name: str, value: float) -> None:
            spec = layer.GetPrimAtPath(path)
            if name in spec.attributes:                 # authored already (author_handle_head)
                spec.attributes[name].default = value
            else:
                attr(spec, name, Sdf.ValueTypeNames.Float, value)
        body = "/World/envs/env_0/Pen/object_root"
        if pen.contact_slop > 0:
            _set(body, "physxRigidBody:contactSlopCoefficient", float(pen.contact_slop))
        if pen.torsional_patch_radius > 0:
            for name in ("torsionalPatchRadius", "minTorsionalPatchRadius"):
                _set(f"{body}/collisions/mesh_0/{'box' if pen.shape == 'box' else 'capsule'}",
                     f"physxCollision:{name}", float(pen.torsional_patch_radius))
        if not pen.gravity:
            _float(layer, [body])
    env.pen = RigidObject(RigidObjectCfg(prim_path=PEN_PATH, spawn=None,
                                         init_state=RigidObjectCfg.InitialStateCfg(**pen_init)))
