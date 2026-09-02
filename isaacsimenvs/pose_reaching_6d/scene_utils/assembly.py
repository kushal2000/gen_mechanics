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
from ..obs_utils import build_morphology_obs, derive_spaces, force_morphology_field
from .author_objects import author_handle_head, author_physics_material
from .author_robot import arm_only_urdf, author_robot_prims, flatten_robot_usd
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


# --- robot population --------------------------------------------------------

@dataclass(frozen=True)
class RobotPopulation:
    """A hand population: the parameter vectors and the specs built from them."""

    hands: list
    specs: list


@dataclass(frozen=True)
class SceneRecord:
    """What setup_scene decided, stored as ``env.scene_record``."""

    robot_spec: object  # defines the joint and observation layout
    population: RobotPopulation | None  # None for one fixed hand
    object_urdf_paths: list[str]  # the pool in final order; viewers read meshes from these
    object_scale: torch.Tensor  # (N, 3) dimensions / object_base_size
    object_pool_index: torch.Tensor  # (N,) long
    robot_design_index: torch.Tensor | None  # (N,) long, population only
    robot_collider_links: dict[int, dict[str, int]] | None  # per design, population only
    asset_dir: str  # temp dir holding the URDFs and converted USDs


def _load_robot_population(assets_cfg) -> RobotPopulation | None:
    """The configured population, or None for one fixed hand."""
    seed = assets_cfg.robot_population_seed
    path = assets_cfg.robot_population_path
    if seed < 0 and not path:
        return None
    from hand_sampler.population import load_population_any
    from hand_sampler.synth_spec import synth_spec

    hands = load_population_any(None if seed < 0 else seed, path)
    count = assets_cfg.robot_population_count
    if count > len(hands):
        raise ValueError(
            f"robot_population_count={count} exceeds the cached population "
            f"for seed {seed} ({len(hands)} hands).")
    if count:
        hands = hands[:count]
    specs = [synth_spec(h, ensure_urdf=False) for h in hands]  # authored, no URDF needed
    # One articulation view: one joint set and one observation layout.
    ref = specs[0]
    for s in specs[1:]:
        if s.joint_names_canonical != ref.joint_names_canonical:
            raise RuntimeError(
                f"design {s.name} does not share the joint template with {ref.name}")
        if s.num_fingertip_slots != ref.num_fingertip_slots:
            raise RuntimeError(
                f"design {s.name} has {s.num_fingertip_slots} fingertip slots, "
                f"{ref.name} has {ref.num_fingertip_slots}")
    return RobotPopulation(hands=hands, specs=specs)


def _resolve_population_and_spec(cfg):
    """The population (injected via ``assets.robot_population`` or loaded) and
    the spec defining the action and observation layout. A population shares
    one joint template, so its first member defines it and ``robot_spec`` is ignored."""
    population = cfg.assets.robot_population
    if population is None:
        population = _load_robot_population(cfg.assets)
    if population is None:
        return None, get_robot_spec(cfg.assets.robot_spec)
    force_morphology_field(cfg, len(population.specs))
    return population, population.specs[0]


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
        # Keyed by joint name, so a population must share one joint set.
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


def _convert_arm(tmp_dir: str, offsets: dict):
    """The shared iiwa14 arm: ``(usd, root, link7_world)``."""
    arm_dir = Path(tmp_dir) / "arm"
    raw = _convert_urdf_to_usd(
        str(arm_only_urdf(arm_dir / "iiwa14_arm_only.urdf")), arm_dir,
        fix_base=True, self_collision=True, joint_drive=_robot_joint_drive_cfg())
    arm_usd, arm_root = flatten_robot_usd(raw, arm_dir / "arm_flat.usd", **offsets)
    stage = Usd.Stage.Open(arm_usd)
    link7_world = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(
        stage.GetPrimAtPath(f"{arm_root}/iiwa14_link_7"))).T
    return arm_usd, arm_root, link7_world


def _author_robots_into_envs(env, spec, population, asset_dir: Path, offsets: dict,
                             t0: float) -> dict[int, dict[str, int]] | None:
    """One Robot prim per env, no spawner clone. A fixed hand is one reference
    to its converted file; a population design references the shared arm and
    authors its hand. Spawning through Isaac Lab instead copies the prim tree
    into every env, ~110 s for a fixed hand and hours for a population.
    Returns each design's collider record, for the friction pass."""
    if population is None:
        robot_usd, robot_root = _convert_fixed_robot(
            spec, env.cfg.assets.robot_urdf or spec.urdf_path, asset_dir / "usd", offsets)
        _log_scene_step(t0, "converted the robot")
    else:
        arm_usd, arm_root, link7_world = _convert_arm(asset_dir, offsets)
        hands, specs = population.hands, population.specs
        _log_scene_step(t0, "converted the shared arm once")

    base_pos = tuple(float(v) for v in spec.base_pos)
    base_rot = tuple(float(v) for v in spec.base_rot)
    layer = get_current_stage().GetRootLayer()
    collider_links = None if population is None else {}
    t_auth = time.perf_counter()
    with Sdf.ChangeBlock():
        for env_path in _env_paths_in_order(env):
            root = f"{env_path}/Robot"
            if population is None:
                prim = define(layer, root, "Xform")
                prim.referenceList.explicitItems.append(
                    Sdf.Reference(robot_usd, Sdf.Path(robot_root)))
            else:
                idx = _env_id_of(env_path) % len(specs)
                colliders = author_robot_prims(
                    layer, root, hands[idx],
                    arm_usd=arm_usd, arm_root_prim=arm_root, link7_world=link7_world,
                    adjacency=specs[idx].adjacent_links, **offsets)
                collider_links[idx] = colliders
            # spawn=None places nothing and a fixed base ignores init_state.pos.
            set_xform(layer.GetPrimAtPath(root), base_pos, base_rot)
    per_robot_ms = (time.perf_counter() - t_auth) / max(env.num_envs, 1) * 1000
    _log_scene_step(t0, f"authored {env.num_envs} robots into env prims "
                        f"({per_robot_ms:.2f} ms each)")
    return collider_links


def _robot_design_index(env, num_designs: int) -> torch.Tensor:
    """design_index[i] = i % num_designs, the assignment the authoring made."""
    _check_numeric_env_order(env, find_matching_prim_paths(ROBOT_PATH), "Robot")
    return torch.tensor([i % num_designs for i in range(env.num_envs)],
                        device=env.device, dtype=torch.long)


def _verify_robot_design_assignment(env, record: SceneRecord) -> None:
    """Envs sharing a design must share joint limits read back from PhysX.
    Limits are drawn per design, so they fingerprint what was actually loaded."""
    limits = env.robot.data.joint_pos_limits.detach().cpu().numpy()  # (N, J, 2)
    idx = record.robot_design_index.detach().cpu().numpy()
    by_design: dict[int, list] = {}
    for e in range(env.num_envs):
        by_design.setdefault(int(idx[e]), []).append(limits[e])
    bad = sorted(
        d for d, mats in by_design.items()
        if any(not np.allclose(mats[0], m, atol=1e-9) for m in mats[1:]))
    if bad:
        raise RuntimeError(
            f"envs sharing design {bad[:5]} do not share joint limits; "
            f"env i should hold design i % {len(record.population.specs)}")
    n_unique = len({np.round(mats[0], 9).tobytes() for mats in by_design.values()})
    print(f"[scene] robot design assignment verified: {len(by_design)} designs over "
          f"{env.num_envs} envs, {n_unique} distinguishable by joint limits")


# --- entry points ---------------------------------------------------------------

def setup_scene(env) -> None:
    """Build and register robot, table, object, goal, ground, and light;
    leaves the decisions in ``env.scene_record``."""
    # Spaces first: DirectRLEnv reads them in _configure_gym_env_spaces, after this hook.
    population, spec = _resolve_population_and_spec(env.cfg)
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
    collider_links = _author_robots_into_envs(env, spec, population, asset_dir, offsets, t0)

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
        robot_spec=spec, population=population,
        object_urdf_paths=[str(p) for p in urdf_paths],
        object_scale=object_scale, object_pool_index=object_pool_index,
        robot_design_index=(None if population is None
                            else _robot_design_index(env, len(population.specs))),
        robot_collider_links=collider_links, asset_dir=str(asset_dir))

    # 7. Register so DirectRLEnv refreshes their tensors each step.
    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["table"] = env.table
    env.scene.rigid_objects["object"] = env.object
    env.scene.rigid_objects["goal_viz"] = env.goal_viz
    _log_scene_step(t0, "registered assets with scene")


def finalize_scene(env) -> None:
    """Needs the started sim: materials bind through PhysX views and the
    design check reads the live articulation. Runs before the first reset."""
    apply_physx_material_properties(env)
    if env.scene_record.population is None:
        return
    _verify_robot_design_assignment(env, env.scene_record)
    build_morphology_obs(env)


__all__ = ["RobotPopulation", "SceneRecord", "finalize_scene", "setup_scene"]