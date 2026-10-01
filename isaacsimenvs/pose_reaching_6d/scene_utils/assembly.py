"""Build the scene and decide which design and object each env gets.

Meshes are converted from URDF once: the arm, a fixed hand, the table. Every
env's Robot, Object and GoalViz prims are authored straight onto the stage (a
fixed hand as one reference to its converted file); only the table goes
through the spawner. Every assignment is recorded, then checked against the
live sim.
"""

from __future__ import annotations

import tempfile
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pxr import Sdf, Usd, UsdGeom
from scipy.spatial.transform import Rotation

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
from hand_sampler import build
from hand_sampler.robot_param_constants import ARM_ADJACENT_LINKS, ARM_TIP_LINK
from hand_sampler.robot_spec import design_index, object_index
from .author_robot import ARM_PRIM, arm_only_urdf, flatten_robot_usd, stub_urdf
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
    # None for one fixed hand; a HandPopulation when every env holds a design.
    population: object | None
    robot_design_index: torch.Tensor | None  # (N,) long, population only
    robot_collider_links: dict | None  # per design {link: n_colliders}
    object_urdf_paths: list[str]  # the pool in final order; viewers read meshes from these
    object_scale: torch.Tensor  # (N, 3) dimensions / object_base_size
    object_pool_index: torch.Tensor  # (N,) long
    asset_dir: str  # temp dir holding the URDFs and converted USDs
    # Several hands in one scene (robots/multi_hand.py): the HandSet, and robot_design_index is then
    # each env's HAND index. None otherwise.
    hand_set: object | None = None


def _resolve_spec(cfg):
    """``(population, spec)``. A population defines the layout through its
    template, so ``assets.robot_spec`` is ignored when one is supplied."""
    population = getattr(cfg.assets, "robot_population", None)
    if population is None:
        from hand_sampler.robot_spec import is_population_ref, population_from_ref
        if is_population_ref(cfg.assets.robot_spec):
            # "handonly:" on the front selects the stub mount. It rides on the
            # reference so the network resolves the same spec -- see
            # hand_sampler.robot_spec.HANDONLY_PREFIX.
            population = population_from_ref(cfg.assets.robot_spec)
    if population is None:
        return None, get_robot_spec(cfg.assets.robot_spec)
    return population, population.spec


# --- spawn configs ------------------------------------------------------------

def build_robot_articulation_cfg(spec, *, start_arm_higher: bool = False,
                                 hand_velocity_limit: float | None = None,
                                 hand_overrides: dict | None = None,
                                 prim_path: str = ROBOT_PATH) -> ArticulationCfg:
    """The robot articulation over prims already on the stage."""
    return ArticulationCfg(
        prim_path=prim_path,
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
        actuators=_actuator_groups(spec, hand_velocity_limit=hand_velocity_limit,
                                   hand_overrides=hand_overrides),
    )


def _actuator_groups(spec, *, hand_velocity_limit: float | None = None,
                     hand_overrides: dict | None = None) -> dict:
    """Actuator groups, keyed by joint name.

    The arm group is OMITTED when the spec has no arm joints: Isaac Lab raises
    "No joints found for actuator group: arm" (articulation.py:1725) on a group
    whose expression matches nothing, so a hand-only robot cannot carry an empty
    one. With an arm present this builds exactly the dict it always did.
    """
    # PhysicsCfg overrides: one value for every hand joint where set (> 0), else the spec's own.
    ov = {k: v for k, v in (hand_overrides or {}).items() if v}
    joints = list(spec.hand_joint_names)
    groups = {
        "hand": ImplicitActuatorCfg(
            joint_names_expr=joints,
            stiffness={n: ov["stiffness"] for n in joints} if "stiffness" in ov else dict(spec.hand_stiffness),
            damping={n: ov["damping"] for n in joints} if "damping" in ov else dict(spec.hand_damping),
            armature={n: ov["armature"] for n in joints} if "armature" in ov else dict(spec.hand_armature),
            effort_limit_sim=ov.get("effort"),
            # Zero everywhere, deliberately. 0.0 rather than None: None takes
            # whatever the USD carries, which is not uniformity.
            friction=0.0,
            # None: the USD's (vendor URDF's) per-joint speed cap. See PhysicsCfg.hand_velocity_limit.
            velocity_limit_sim=hand_velocity_limit,
        ),
    }
    if spec.arm_joint_names:
        groups["arm"] = ImplicitActuatorCfg(
            joint_names_expr=list(spec.arm_joint_names),
            stiffness=dict(spec.arm_stiffness),
            damping=dict(spec.arm_damping),
            friction=0.0,
        )
    return groups


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
        curated=assets_cfg.object_pool or None,
    )
    if not urdf_paths:
        raise ValueError("no object URDFs generated; check handle_head_types and num_assets_per_type")
    return urdf_paths, scales, params


def _author_objects_into_envs(env, object_params, design_idx) -> dict[int, int]:
    """Author Object and GoalViz into every env; returns ``{env_id: pool_index}``
    by the configured ``object_assignment`` rule. ``design_idx`` is the per-env
    design (None for a single robot)."""
    n_pool = len(object_params)
    assets_cfg = env.cfg.assets
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    n_designs = 1 if design_idx is None else int(design_idx.max()) + 1
    which = object_index(env.num_envs, n_pool, assets_cfg.object_assignment,
                         rank=rank, world_size=world, n_designs=n_designs)
    if assets_cfg.object_assignment == "design_cycle":
        per_design = env.num_envs * world // max(n_designs, 1)
        print(f"[scene] object assignment design_cycle: pool {n_pool}, {per_design} envs per design"
              + ("" if per_design == n_pool else
                 f" -- pool and envs-per-design differ, so designs meet {min(per_design, n_pool)} distinct objects"),
              flush=True)
        if design_idx is not None:
            # What THIS rank actually deals, from the arrays that author it: the
            # other rank must hold the complementary set, and the log of both
            # ranks together is the record that every design met every object.
            held = {}
            for d, o in zip(design_idx.tolist(), which.tolist()):
                held.setdefault(int(d), set()).add(int(o))
            counts = sorted(len(v) for v in held.values())
            print(f"[scene] rank {rank}: design 0 holds objects {sorted(held.get(0, ()))}; "
                  f"distinct objects per design on this rank min {counts[0]} max {counts[-1]} "
                  f"over {len(held)} designs", flush=True)
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
            asset_index[env_id] = int(which[env_id])
            handle_scale, head_scale, handle_density, head_density = \
                object_params[asset_index[env_id]]
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


def _convert_arm(tmp_dir, offsets: dict, hand_only: bool = False):
    """The shared mount every authored design attaches to.

    ``(usd, root, link7_world, link7_mass_props)``. Converted ONCE: a population
    references this one file and authors only its own hand bodies on top.

    ``hand_only`` swaps the 7-DOF iiwa14 for a single fixed link that keeps the
    name ``ARM_TIP_LINK``. Everything downstream is untouched -- the hand still
    attaches to ``{root}/arm/iiwa14_link_7``, the palm still merges into it --
    the articulation just has no arm joints.
    """
    arm_dir = Path(tmp_dir) / "arm"
    urdf = (stub_urdf(arm_dir / "iiwa14_tip_stub.urdf") if hand_only
            else arm_only_urdf(arm_dir / "iiwa14_arm_only.urdf"))
    raw = _convert_urdf_to_usd(
        str(urdf), arm_dir,
        fix_base=True, self_collision=True, joint_drive=_robot_joint_drive_cfg())
    # The arm's OWN consecutive links, filtered here for the same reason the
    # fixed robot filters its whole map: self_collision=True above turns the
    # articulation's self-collisions on, and nothing else on this path masks
    # anything. Convert -> filter -> flatten, the order _convert_fixed_robot
    # uses; the hand's pairs are authored per env by build.author_hand, which
    # has no file to edit.
    # One link has no consecutive-link pairs to filter.
    if not hand_only:
        _apply_self_collision_filters(raw, dict(ARM_ADJACENT_LINKS))
    arm_usd, arm_root = flatten_robot_usd(raw, arm_dir / "arm_flat.usd", **offsets)
    stage = Usd.Stage.Open(arm_usd)
    link7 = stage.GetPrimAtPath(f"{arm_root}/{ARM_TIP_LINK}")
    link7_world = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(link7)).T
    return arm_usd, arm_root, link7_world, _rigid_body_mass_props(link7)


def _rigid_body_mass_props(prim):
    """``(mass, centre of mass, 3x3 inertia)`` as the arm USD declares them.

    The palm merges into this body, so its mass has to be ADDED to what is
    already there rather than replacing it -- and the flange's own numbers are
    only readable off the converted stage.
    """
    from pxr import Gf

    def _get(name, default):
        a = prim.GetAttribute(name)
        return a.Get() if a and a.Get() is not None else default

    mass = float(_get("physics:mass", 0.0))
    com = np.asarray([float(v) for v in _get("physics:centerOfMass", Gf.Vec3f(0.0))], float)
    diag = np.asarray([float(v) for v in _get("physics:diagonalInertia", Gf.Vec3f(0.0))], float)
    axes = _get("physics:principalAxes", Gf.Quatf(1.0))
    imag = axes.GetImaginary()
    rot = Rotation.from_quat(
        [float(imag[0]), float(imag[1]), float(imag[2]), float(axes.GetReal())]).as_matrix()
    return mass, com, rot @ np.diag(diag) @ rot.T


def _author_robots_into_envs(env, spec, population, design_idx, asset_dir: Path,
                             offsets: dict, t0: float) -> dict[int, dict[str, int]] | None:
    """One Robot prim per env, no spawner clone. A fixed hand is one reference
    to its converted file; a design references the shared arm and authors its
    own hand. Spawning through Isaac Lab instead copies the prim tree into every
    env, ~110 s for a fixed hand and hours for a population.
    Returns each design's collider record, for the friction pass."""
    if population is None:
        robot_usd, robot_root = _convert_fixed_robot(
            spec, env.cfg.assets.robot_urdf or spec.urdf_path, asset_dir / "usd", offsets)
        _log_scene_step(t0, "converted the robot")
    else:
        # The spec is the single source of truth: no arm joints means the hand
        # mounts on the stub. Nothing else can disagree with it.
        hand_only = spec.num_arm_joints == 0
        arm_usd, arm_root, link7_world, link7_mass = _convert_arm(
            asset_dir, offsets, hand_only=hand_only)
        _log_scene_step(t0, f"converted the shared {'tip stub' if hand_only else 'arm'} once")

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
                # From the SAME array the tables are gathered with: computing it
                # twice is how rank 1 authored designs 0..n while its tables
                # described n..2n, which the reset invariant caught.
                idx = int(design_idx[_env_id_of(env_path)])
                define(layer, root, "Xform")
                arm = define(layer, f"{root}{ARM_PRIM}", "Xform")
                arm.referenceList.explicitItems.append(
                    Sdf.Reference(str(arm_usd), Sdf.Path(arm_root)))
                collider_links[idx] = build.author_hand(
                    layer, root, population.hands[idx],
                    palm_body_path=f"{root}{ARM_PRIM}/{ARM_TIP_LINK}",
                    link7_world=link7_world, link7_mass_props=link7_mass)
            # spawn=None places nothing and a fixed base ignores init_state.pos.
            set_xform(layer.GetPrimAtPath(root), base_pos, base_rot)
        # Inside the block only specs are written; the stage recomposes on exit.
        t_written = time.perf_counter()
    t_composed = time.perf_counter()
    per_robot_ms = (t_composed - t_auth) / max(env.num_envs, 1) * 1000
    _log_scene_step(t0, f"authored {env.num_envs} robots into env prims "
                        f"({per_robot_ms:.2f} ms each = {t_written - t_auth:.1f}s writing "
                        f"specs + {t_composed - t_written:.1f}s recomposing)")
    return collider_links


def _block_regex(env_ids) -> str:
    """``/World/envs/env_(i|j|...)/Robot`` over one hand's envs (as UHAS does)."""
    ids = [str(int(i)) for i in env_ids]
    return f"/World/envs/env_({'|'.join(ids)})/Robot" if len(ids) > 1 else f"/World/envs/env_{ids[0]}/Robot"


def _author_multi_hands(env, hs, hand_idx, asset_dir: Path, offsets: dict, t0: float) -> None:
    """Every hand converted ONCE, then referenced into its own envs at its own base pose."""
    converted = []
    for h, spec in enumerate(hs.specs):
        work = asset_dir / "usd" / f"hand{h}_{spec.hand_name}"
        work.mkdir(parents=True, exist_ok=True)
        converted.append(_convert_fixed_robot(spec, spec.urdf_path, work, offsets))
        _log_scene_step(t0, f"converted hand {h}: {spec.name}")
    layer = get_current_stage().GetRootLayer()
    with Sdf.ChangeBlock():
        for env_path in _env_paths_in_order(env):
            h = int(hand_idx[_env_id_of(env_path)])
            spec, (usd, root_path) = hs.specs[h], converted[h]
            root = f"{env_path}/Robot"
            prim = define(layer, root, "Xform")
            prim.referenceList.explicitItems.append(Sdf.Reference(usd, Sdf.Path(root_path)))
            set_xform(layer.GetPrimAtPath(root), tuple(map(float, spec.base_pos)), tuple(map(float, spec.base_rot)))
    counts = {s.hand_name: int((hand_idx == h).sum()) for h, s in enumerate(hs.specs)}
    _log_scene_step(t0, f"authored {env.num_envs} robots, hands dealt round-robin: {counts}")


# --- entry points ---------------------------------------------------------------

def setup_scene(env) -> None:
    """Build and register robot, table, object, goal, ground, and light;
    leaves the decisions in ``env.scene_record``."""
    # Spaces first: DirectRLEnv reads them in _configure_gym_env_spaces, after this hook.
    from .robots.multi_hand import hand_set as _hand_set, is_multi_ref
    hs = _hand_set(env.cfg.assets.robot_spec) if is_multi_ref(env.cfg.assets.robot_spec) else None
    population, spec = (None, hs.template) if hs is not None else _resolve_spec(env.cfg)
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
    design_idx = None if population is None else design_index(
        env.num_envs, population.n_designs,
        rank=int(os.environ.get("RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")))
    if hs is not None:
        design_idx = hs.hand_of_env(env.num_envs)     # each env's HAND, in contiguous blocks
    if design_idx is not None and population is not None:
        # In the log, because "which designs did this rank hold" is otherwise
        # only answerable by re-deriving it.
        print(f"[scene] rank {int(os.environ.get('RANK', '0'))} of "
              f"{os.environ.get('WORLD_SIZE', '1')} holds designs "
              f"{int(design_idx.min())}..{int(design_idx.max())} "
              f"({len(set(design_idx.tolist()))} distinct of {population.n_designs})",
              flush=True)
    if hs is not None:
        _author_multi_hands(env, hs, design_idx, asset_dir, offsets, t0)
        collider_links = None
    else:
        collider_links = _author_robots_into_envs(
            env, spec, population, design_idx, asset_dir, offsets, t0)

    # 3. Table, converted and spawned -- unless the task has no use for one.
    # An in-hand task holds the object on a fixed palm, so nothing ever rests on
    # a table and the ground plane already catches a dropped object. Parking it
    # below the scene still spawns a rigid body per env and still cooks and
    # broadphases it, for nothing.
    want_table = not env.cfg.reset.object_in_hand
    table_usd = (_convert_urdf_to_usd(assets_cfg.table_urdf, asset_dir / "usd", fix_base=False)
                 if want_table else None)

    # 4. Spawn.
    overrides = {"effort": env.cfg.physics.hand_effort_limit,
                 "stiffness": env.cfg.physics.hand_stiffness,
                 "damping": env.cfg.physics.hand_damping,
                 "armature": env.cfg.physics.hand_armature}
    if hs is not None:
        # One articulation per hand over its own env block; env.robot presents them as one.
        from .multi_articulation import MultiHandArticulation
        arts = [Articulation(build_robot_articulation_cfg(
                    s, hand_velocity_limit=env.cfg.physics.hand_velocity_limit or None,
                    hand_overrides=overrides, prim_path=_block_regex(np.nonzero(design_idx == h)[0])))
                for h, s in enumerate(hs.specs)]
        env.robot = MultiHandArticulation(hs, arts, torch.as_tensor(design_idx), env.device)
    else:
        env.robot = Articulation(build_robot_articulation_cfg(
            spec, start_arm_higher=env.cfg.reset.start_arm_higher,
            hand_velocity_limit=env.cfg.physics.hand_velocity_limit or None,
            hand_overrides=overrides))
    env.table = (RigidObject(build_rigid_object_cfg(TABLE_PATH, table_usd, _table_props(offsets)))
                 if want_table else None)
    authored_map = _author_objects_into_envs(env, object_params, design_idx)
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
        robot_design_index=(None if design_idx is None
                            else torch.as_tensor(design_idx, device=env.device)),
        robot_collider_links=collider_links,
        object_urdf_paths=[str(p) for p in urdf_paths],
        object_scale=object_scale, object_pool_index=object_pool_index,
        asset_dir=str(asset_dir), hand_set=hs)

    # 7. Register so DirectRLEnv refreshes their tensors each step.
    env.scene.articulations["robot"] = env.robot
    if env.table is not None:
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