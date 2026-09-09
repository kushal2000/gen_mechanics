"""Build the scene and decide which design and object each env gets.

Meshes are converted from URDF once: the arm, a fixed hand, the table. Every
env's Robot, Object and GoalViz prims are authored straight onto the stage (a
fixed hand as one reference to its converted file); only the table goes
through the spawner. Every assignment is recorded, then checked against the
live sim.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pxr import Sdf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage

from ..common_utils.physx import _log_scene_step
from ..common_utils.urdf_to_usd import (
    _apply_self_collision_filters, _convert_urdf_to_usd, _robot_joint_drive_cfg,
)
from ..obs_utils import derive_spaces
from .author_objects import author_handle_head, author_physics_material
from .author_robot import flatten_robot_usd
from .materials import apply_physx_material_properties
from .sdf import define, set_xform
from .objects.generate_objects import generate_handle_head_urdfs
from .robots import get_robot_spec

ROBOT_PATH = "/World/envs/env_.*/Robot"
TABLE_PATH = "/World/envs/env_.*/Table"
OBJECT_PATH = "/World/envs/env_.*/Object"
GOALVIZ_PATH = "/World/envs/env_.*/GoalViz"

def _table_props(offsets: dict) -> dict:
    """PhysX properties the URDF converter leaves unset, applied at spawn."""
    return dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(**offsets),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
    )





@dataclass(frozen=True)
class SceneRecord:
    """What setup_scene decided, stored as ``env.scene_record``."""

    robot_spec: object  # defines the joint and observation layout
    object_urdf_paths: list[str]  # the pool in final order; viewers read meshes from these
    object_scale: torch.Tensor  # (N, 3) dimensions / object_base_size
    object_pool_index: torch.Tensor  # (N,) long
    asset_dir: str  # temp dir holding the URDFs and converted USDs




def _resolve_spec(cfg):
    """The spec defining the action and observation layout."""
    return get_robot_spec(cfg.assets.robot_spec)


# --- spawn configs ------------------------------------------------------------

def build_robot_articulation_cfg(spec, *, start_arm_higher: bool = False) -> ArticulationCfg:
    """The robot articulation over prims already on the stage."""
    return ArticulationCfg(
        prim_path=ROBOT_PATH,
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=spec.base_pos,
            rot=spec.base_rot,
            joint_pos={
                **spec.arm_default_joint_pos_resolved(start_arm_higher=start_arm_higher),
                **spec.hand_default_joint_pos,
            },
            joint_vel={".*": 0.0},
        ),
        # Keyed by joint name.
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
                friction=dict(spec.hand_friction),
            ),
        },
    )


def build_rigid_object_cfg(prim_path: str, usd_path: str, props: dict) -> RigidObjectCfg:
    """RigidObject spawned from one converted USD."""
    return RigidObjectCfg(prim_path=prim_path, spawn=sim_utils.UsdFileCfg(usd_path=usd_path, **props))


# --- env prims ------------------------------------------------------------------

def _materialize_env_prims(env) -> None:
    """Pre-create the env roots so regex spawns and in-place authoring see every env."""
    stage = get_current_stage()
    for env_path in env.scene.env_prim_paths:
        if not stage.GetPrimAtPath(env_path).IsValid():
            stage.DefinePrim(env_path, "Xform")


def _env_id_of(prim_path: str) -> int:
    for token in prim_path.split("/"):
        if token.startswith("env_"):
            return int(token.removeprefix("env_"))
    raise ValueError(f"no env_<i> component in {prim_path!r}")


def _env_paths_in_order(env) -> list[str]:
    return sorted(env.scene.env_prim_paths, key=_env_id_of)


def _check_numeric_env_order(env, prim_paths: list[str], what: str) -> None:
    """View row i is env i only if the prims come back in numeric env order."""
    if len(prim_paths) != env.num_envs:
        raise RuntimeError(f"Expected {env.num_envs} {what} prims, got {len(prim_paths)}.")
    observed = [_env_id_of(p) for p in prim_paths]
    if observed != sorted(observed):
        raise RuntimeError(
            f"{what} prims are not in numeric env order (first 12: {observed[:12]})")


# --- objects --------------------------------------------------------------------

def _resolve_object_pool(assets_cfg, out_dir: str):
    """``(urdf_paths, scales_normalized, params)`` for the procedural pool."""
    urdf_paths, scales, params = generate_handle_head_urdfs(
        handle_head_types=tuple(assets_cfg.handle_head_types),
        num_per_type=assets_cfg.num_assets_per_type,
        out_dir=out_dir,
        shuffle=assets_cfg.shuffle_assets,
        seed=assets_cfg.object_seed,
        density_scale=assets_cfg.object_density_scale,
    )
    if not urdf_paths:
        raise ValueError("no object URDFs generated; check handle_head_types and num_assets_per_type")
    return urdf_paths, scales, params


def _author_objects_into_envs(env, object_params) -> dict[int, int]:
    """Author Object and GoalViz into every env; returns ``{env_id: pool_index}``
    with env i on entry ``i % pool``."""
    n_pool = len(object_params)
    assets_cfg = env.cfg.assets
    layer = get_current_stage().GetRootLayer()
    t0 = time.perf_counter()
    asset_index: dict[int, int] = {}
    with Sdf.ChangeBlock():
        # One shared material, bound on the shapes as they are authored; the
        # friction pass overwrites its values per env.
        mat_path = author_physics_material(
            layer, "/World/PhysicsMaterials/object",
            static_friction=float(assets_cfg.object_friction),
            dynamic_friction=float(assets_cfg.object_friction),
            restitution=float(assets_cfg.object_restitution))
        for env_path in _env_paths_in_order(env):
            env_id = _env_id_of(env_path)
            asset_index[env_id] = env_id % n_pool
            handle_scale, head_scale, handle_density, head_density = \
                object_params[env_id % n_pool]
            # GoalViz: no collider and no motion.
            for name, collision, kinematic in (("Object", True, False),
                                               ("GoalViz", False, True)):
                define(layer, f"{env_path}/{name}", "Xform")
                author_handle_head(
                    layer, f"{env_path}/{name}",
                    handle_scale, head_scale, handle_density, head_density,
                    collision=collision,
                    material_path=mat_path, kinematic=kinematic)
    _log_scene_step(
        t0, f"authored {env.num_envs} Object + GoalViz prims from a {n_pool}-entry pool")
    return asset_index


def _object_tensors(env, object_scales_normalized, authored_map) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-env ``(scale (N, 3), pool index (N,))`` from the authoring record."""
    _check_numeric_env_order(env, find_matching_prim_paths(OBJECT_PATH), "Object")
    scale = torch.zeros(env.num_envs, 3, device=env.device)
    pool_index = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    for env_id, asset_index in authored_map.items():
        scale[env_id] = torch.tensor(object_scales_normalized[asset_index], device=env.device)
        pool_index[env_id] = asset_index
    return scale, pool_index


# --- robots ---------------------------------------------------------------------

def _convert_fixed_robot(spec, urdf: str, usd_work_dir: Path, offsets: dict) -> tuple[str, str]:
    """URDF -> one flattened USD with self-collision filters and PhysX props; ``(path, root)``."""
    converted = _convert_urdf_to_usd(
        urdf, usd_work_dir, fix_base=True, self_collision=True,
        joint_drive=_robot_joint_drive_cfg(),
        replace_cylinders_with_capsules=spec.replace_cylinders_with_capsules,
    )
    _apply_self_collision_filters(converted, spec.adjacent_links)
    return flatten_robot_usd(converted, usd_work_dir / "robot_flat.usd", **offsets)




def _author_robots_into_envs(env, spec, asset_dir: Path, offsets: dict,
                             t0: float) -> None:
    """One Robot prim per env, no spawner clone: one reference to the converted
    file. Spawning through Isaac Lab instead copies the prim tree into every
    env, ~110 s."""
    robot_usd, robot_root = _convert_fixed_robot(
        spec, env.cfg.assets.robot_urdf or spec.urdf_path, asset_dir / "usd", offsets)
    _log_scene_step(t0, "converted the robot")

    base_pos = tuple(float(v) for v in spec.base_pos)
    base_rot = tuple(float(v) for v in spec.base_rot)
    layer = get_current_stage().GetRootLayer()
    t_auth = time.perf_counter()
    with Sdf.ChangeBlock():
        for env_path in _env_paths_in_order(env):
            root = f"{env_path}/Robot"
            prim = define(layer, root, "Xform")
            prim.referenceList.explicitItems.append(
                Sdf.Reference(robot_usd, Sdf.Path(robot_root)))
            # spawn=None places nothing and a fixed base ignores init_state.pos.
            set_xform(layer.GetPrimAtPath(root), base_pos, base_rot)
        # Inside the block only specs are written; the stage recomposes on exit.
        t_written = time.perf_counter()
    t_composed = time.perf_counter()
    per_robot_ms = (t_composed - t_auth) / max(env.num_envs, 1) * 1000
    _log_scene_step(t0, f"authored {env.num_envs} robots into env prims "
                        f"({per_robot_ms:.2f} ms each = {t_written - t_auth:.1f}s writing "
                        f"specs + {t_composed - t_written:.1f}s recomposing)")






# --- entry points ---------------------------------------------------------------

def setup_scene(env) -> None:
    """Build and register robot, table, object, goal, ground, and light;
    leaves the decisions in ``env.scene_record``."""
    # Spaces first: DirectRLEnv reads them in _configure_gym_env_spaces, after this hook.
    spec = _resolve_spec(env.cfg)
    derive_spaces(env.cfg, spec)

    assets_cfg = env.cfg.assets
    offsets = dict(contact_offset=env.cfg.physics.contact_offset,
                   rest_offset=env.cfg.physics.rest_offset)
    t0 = time.perf_counter()
    _log_scene_step(t0, f"setup start num_envs={env.num_envs} "
                        f"num_assets_per_type={assets_cfg.num_assets_per_type}")

    asset_dir = Path(tempfile.mkdtemp(prefix="genmech_assets_"))
    (asset_dir / "usd").mkdir()
    _materialize_env_prims(env)

    # 1. Object pool.
    urdf_paths, object_scales, object_params = _resolve_object_pool(assets_cfg, str(asset_dir))
    _log_scene_step(t0, f"generated {len(urdf_paths)} object URDFs")

    # 2. Robots, authored into every env.
    _author_robots_into_envs(env, spec, asset_dir, offsets, t0)

    # 3. Table, converted and spawned.
    table_usd = _convert_urdf_to_usd(assets_cfg.table_urdf, asset_dir / "usd", fix_base=False)

    # 4. Spawn.
    env.robot = Articulation(build_robot_articulation_cfg(
        spec, start_arm_higher=env.cfg.reset.start_arm_higher))
    env.table = RigidObject(build_rigid_object_cfg(TABLE_PATH, table_usd, _table_props(offsets)))
    authored_map = _author_objects_into_envs(env, object_params)
    env.object = RigidObject(RigidObjectCfg(prim_path=OBJECT_PATH, spawn=None))
    env.goal_viz = RigidObject(RigidObjectCfg(prim_path=GOALVIZ_PATH, spawn=None))
    _log_scene_step(t0, "spawned robot/table/object/goalviz")

    # 5. Ground plane and dome light.
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # 6. Which pool entry and which design each env holds.
    object_scale, object_pool_index = _object_tensors(env, object_scales, authored_map)
    env.scene_record = SceneRecord(
        robot_spec=spec,
        object_urdf_paths=[str(p) for p in urdf_paths],
        object_scale=object_scale, object_pool_index=object_pool_index,
        asset_dir=str(asset_dir))

    # 7. Register so DirectRLEnv refreshes their tensors each step.
    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["table"] = env.table
    env.scene.rigid_objects["object"] = env.object
    env.scene.rigid_objects["goal_viz"] = env.goal_viz
    _log_scene_step(t0, "registered assets with scene")


def _verify_articulation_view(env) -> None:
    """The articulation view must cover every env with a full joint set.
    Instancing is the reason this can silently come up short."""
    data = env.robot.data
    if data.joint_pos.shape[0] != env.num_envs:
        raise RuntimeError(
            f"articulation view has {data.joint_pos.shape[0]} envs, expected {env.num_envs}")
    n_joints = len(env.scene_record.robot_spec.joint_names_canonical)
    if data.joint_pos.shape[1] != n_joints:
        raise RuntimeError(
            f"articulation view has {data.joint_pos.shape[1]} joints, "
            f"the spec has {n_joints}")
    shapes = env.robot.root_physx_view.max_shapes
    if shapes < 1:
        raise RuntimeError("articulation view reports no collision shapes")
    print(f"[scene] articulation view: {data.joint_pos.shape[0]} envs x "
          f"{data.joint_pos.shape[1]} joints, {shapes} shapes", flush=True)


def finalize_scene(env) -> None:
    """Needs the started sim: materials bind through PhysX views and the
    design check reads the live articulation. Runs before the first reset."""
    _verify_articulation_view(env)
    apply_physx_material_properties(env)


__all__ = ["SceneRecord", "finalize_scene", "setup_scene"]