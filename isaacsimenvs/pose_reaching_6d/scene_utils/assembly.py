"""Building the scene, and deciding which design and object each env gets.

``setup_scene`` is the entry point. The assignment helpers below record what
each env actually received rather than re-deriving it later: design_index[i]
= i % num_designs and object_index[i] = i % pool_size hold only if nothing
reorders the lists, and a re-derivation bug once put 510 of 512 envs on the
wrong object.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg, spawn_ground_plane
from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg
from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage

from hand_sampler.paths import resolve as resolve_repo_path

from .objects.generate_objects import generate_handle_head_urdfs
from ..obs_utils import derive_spaces
from .robots import get_robot_spec
from isaacsimenvs.pose_reaching_6d.common_utils.physx import _log_scene_step
from isaacsimenvs.pose_reaching_6d.common_utils.urdf_to_usd import (
    _apply_self_collision_filters,
    _bake_usd,
    _convert_urdf_to_usd,
    _generate_scaled_table_urdfs,
    _robot_joint_drive_cfg,
)


def build_robot_articulation_usd_cfg(
    usd_path: str | list[str], spec, *, start_arm_higher: bool = False
) -> ArticulationCfg:
    """Build the robot articulation for ``spec`` from its baked USD(s).

    Actuator groups are keyed by exact joint name rather than a regex, so a hand
    whose joints do not share a common prefix still gets its gains applied.

    A LIST of USDs spawns one design per env, round-robin, exactly as the object
    pool does -- env i gets ``usds[i % len(usds)]``. All of them must share the
    joint names and count the actuator tables and ``spec`` describe, which is
    what ghosting guarantees for a generated population: every design is padded
    to the same 37-joint template, so an arbitrary number of morphologies costs
    one articulation view. Designs with different joint COUNTS cannot share a
    view at all -- PhysX reads ``shared_metatype.dof_count`` once per view.
    """
    # The sentinel means the robot prims were authored straight onto the stage
    # (one design per env), so there is nothing to spawn. This is what avoids
    # MultiUsdFileCfg's per-design composition at k = n.
    if usd_path == "__authored_in_place__":
        spawn = None
    else:
        many = not isinstance(usd_path, str)
        spawn = (MultiUsdFileCfg(usd_path=list(usd_path), random_choice=False)
                 if many else UsdFileCfg(usd_path=usd_path))
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=spec.base_pos,
            rot=spec.base_rot,
            joint_pos={
                **spec.arm_default_joint_pos_resolved(start_arm_higher=start_arm_higher),
                **spec.hand_default_joint_pos,
            },
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


def build_rigid_object_cfg(prim_path: str, usd_paths: list[str]) -> RigidObjectCfg:
    """Spawn a RigidObject from one or more pre-baked USDs (round-robin)."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=MultiUsdFileCfg(usd_path=list(usd_paths), random_choice=False),
    )


def _materialize_env_prims(env) -> None:
    stage = get_current_stage()
    for env_path in env.scene.env_prim_paths:
        if not stage.GetPrimAtPath(env_path).IsValid():
            stage.DefinePrim(env_path, "Xform")


def _env_id_of(prim_path: str) -> int:
    """Numeric env id from a `/World/envs/env_<i>` path or any child of one."""
    for token in prim_path.split("/"):
        if token.startswith("env_"):
            return int(token.removeprefix("env_"))
    raise ValueError(f"no env_<i> component in {prim_path!r}")


def _author_objects_into_envs(env, object_params, n_pool: int,
                              which=("object", "goalviz")) -> None:
    """Author Object and GoalViz prims directly into every env.

    Env i takes pool entry ``i % n_pool`` -- the same assignment MultiUsdFileCfg
    makes, so ``_build_object_scale_tensor``'s per-env scale lookup stays correct
    and the policy sees the object it expects.

    The goal marker is the same geometry with collision omitted, matching how the
    converted path bakes goalviz with ``collision_enabled=False``.
    """
    from pxr import Sdf

    from isaacsimenvs.pose_reaching_6d.scene_utils.author_usd import define
    from isaacsimenvs.pose_reaching_6d.scene_utils.author_objects import (
        author_handle_head,
        author_physics_material,
    )

    stage = get_current_stage()
    layer = stage.GetRootLayer()
    t0 = time.perf_counter()
    with Sdf.ChangeBlock():
        # One shared material; the env overwrites its values per env through
        # root_physx_view.set_material_properties, so per-asset materials would
        # buy nothing and cost a prim each.
        mat_path = author_physics_material(
            layer, "/World/PhysicsMaterials/object",
            static_friction=float(env.cfg.assets.object_friction),
            dynamic_friction=float(env.cfg.assets.object_friction),
            restitution=float(env.cfg.assets.object_restitution))
        # Assign pool entries by NUMERIC env id, and record the map so that
        # _build_object_scale_tensor can consume it rather than re-deriving it.
        #
        # Re-deriving is what went wrong before. That function reads
        # find_matching_prim_paths("/World/envs/env_.*/Object") and takes
        # asset_index = position % pool, and this loop used to walk
        # sorted(env_prim_paths) on the belief that the two orders agree.
        # They do not: find_matching_prim_paths returns NUMERIC order
        # (env_0, env_1, env_2, ...) while sorted() returns lexicographic
        # (env_0, env_1, env_10, env_100, ...). Measured on 512 envs, the two
        # disagreed in 510 of them -- so nearly every env held an object whose
        # geometry did not match the object_scales in its own observation, and
        # the policy scored 3.00 goals against 5.07 for the converted path.
        #
        # Note the assignment is deliberately identical to MultiUsdFileCfg's,
        # which walks that same numeric list: env i takes pool entry i % n_pool.
        # Recording the map means a future change to either ordering cannot
        # silently desynchronise the two again.
        env._authored_asset_index = {}
        for env_path in sorted(env.scene.env_prim_paths, key=_env_id_of):
            source_idx = _env_id_of(env_path)
            env._authored_asset_index[source_idx] = source_idx % n_pool
            handle_scale, head_scale, handle_density, head_density = \
                object_params[source_idx % n_pool]
            # GoalViz: no collider AND no motion. The converted path bakes it
            # kinematic with gravity disabled; both are required.
            for name, collision, kinematic in (("Object", True, False),
                                               ("GoalViz", False, True)):
                if name.lower() not in which:
                    continue
                # The parent Xform the converter's USD root maps onto.
                define(layer, f"{env_path}/{name}", "Xform")
                author_handle_head(
                    layer, f"{env_path}/{name}",
                    handle_scale, head_scale, handle_density, head_density,
                    body_at_root=False, collision=collision,
                    material_path=mat_path, kinematic=kinematic)
    # The physics material is bound by _shape_prim, on the collision shapes,
    # inside the ChangeBlock above.
    #
    # It used to be bound here instead, by isaaclab's bind_physics_material,
    # twice per env and outside the ChangeBlock. That helper is decorated with
    # apply_nested so every call traverses the subtree of a live stage, and the
    # authored 24k population makes that stage ~5.8M prims: 49,152 traversals
    # that grow with the robots, for a binding whose target prims we authored
    # ourselves and therefore already know. Without a bound material the env's
    # friction pass fails loudly with "Failed to get rigid body material
    # properties from backend", so a regression here is not silent.

    _log_scene_step(
        t0, f"authored {env.num_envs} Object + GoalViz prims from a "
            f"{n_pool}-entry pool")


@dataclass(frozen=True)
class RobotPopulation:
    """A resolved hand population: the parameter vectors and the specs built
    from them, loaded together.

    Both halves are kept because both are needed and the manifest is expensive:
    a 24,576-hand manifest is 266 MB of JSON. Four call sites used to load it
    independently -- the spec build, the USD build, the morphology descriptor,
    and setup_scene's own fallback -- each re-deriving the -1 sentinel and the
    robot_population_count truncation. Four copies of that logic is four chances
    for the descriptor to describe a different hand than the env holds, which is
    the failure _verify_robot_design_assignment exists to catch.
    """

    hands: list
    specs: list


def _resolve_robot_population(assets_cfg) -> RobotPopulation | None:
    """Load a cached hand population, or None for the single-robot env."""
    seed = getattr(assets_cfg, "robot_population_seed", None)
    path = getattr(assets_cfg, "robot_population_path", None)
    if seed is not None and int(seed) < 0:
        seed = None                      # -1 is the "no population" sentinel
    if seed is None and not path:
        return None
    from hand_sampler.population import load_population_any
    from hand_sampler.synth_spec import synth_spec

    hands = load_population_any(seed, path)
    # The AUTHORED path builds every robot's physics from HandParams and never
    # opens a design URDF -- only the converter branch reads design_spec.urdf_path
    # (see _build_robot_usds below). So writing one URDF per design costs 13 min
    # of every scene build and 24,576 files into a directory already holding 172k,
    # for artifacts nothing reads. It is also a race: five concurrent seed-jobs on
    # one population all find the files absent and all write the same paths.
    authored = bool(getattr(assets_cfg, "author_robot_usds", False))
    count = int(getattr(assets_cfg, "robot_population_count", 0) or 0)
    if count:
        if count > len(hands):
            raise ValueError(
                f"robot_population_count={count} exceeds the cached population "
                f"for seed {seed} ({len(hands)} hands). Build a larger one with "
                f"hand_sampler.population.build_population.")
        hands = hands[:count]
    specs = [synth_spec(h, ensure_urdf=not authored) for h in hands]

    # Every design must present the same joint vector: one articulation view has
    # one dof_count, and the actuator tables are keyed by name. Ghosting is what
    # makes this hold; check it rather than trust it, because a mismatch here
    # surfaces as an opaque PhysX view error much later.
    ref = specs[0]
    for s in specs[1:]:
        if s.joint_names_canonical != ref.joint_names_canonical:
            raise RuntimeError(
                f"design {s.name} does not share the joint template with "
                f"{ref.name}: a population must be padded to one joint set "
                f"before it can share an articulation view")
        if s.num_fingertip_slots != ref.num_fingertip_slots:
            raise RuntimeError(
                f"design {s.name} has {s.num_fingertip_slots} fingertip slots, "
                f"{ref.name} has {ref.num_fingertip_slots}: the observation "
                f"layout must be identical across a population")
    return RobotPopulation(hands=hands, specs=specs)


def finalize_population(env) -> None:
    """After the scene, materials and buffers exist: check the sim matches."""
    if env._robot_population is None:
        return
    from ..obs_utils import build_morphology_obs

    # Against the live sim: unchecked, the same assumption put 510 of 512 envs
    # on the wrong object.
    _verify_robot_design_assignment(env, env._robot_population.specs)
    build_morphology_obs(env)


def resolve_spec(env, cfg):
    """The RobotSpec defining the action and observation layout.

    A population's designs share one joint template, so any member defines it
    and ``robot_spec`` is ignored. ``assets.robot_population`` injects designs
    directly; otherwise they come from the manifest.

    Called before ``super().__init__`` so ``derive_spaces`` can write the widths
    DirectRLEnv reads in ``_configure_gym_env_spaces``. That happens after
    ``_setup_scene``, so this could equally run there; it is here so the spec is
    settled before the scene starts using it.
    """
    from ..obs_utils import force_morphology_field

    population = cfg.assets.robot_population
    if population is not None:
        env._robot_population = population
        env._robot_population_specs = population.specs
    else:
        population = _ensure_robot_population(env, cfg.assets)
    if population is None:
        return get_robot_spec(cfg.assets.robot_spec)
    force_morphology_field(cfg, len(population.specs))
    return population.specs[0]


def _ensure_robot_population(env, assets_cfg) -> RobotPopulation | None:
    """Resolve the population once per env, then hand back the same object."""
    if not hasattr(env, "_robot_population"):
        env._robot_population = _resolve_robot_population(assets_cfg)
        env._robot_population_specs = (
            env._robot_population.specs if env._robot_population else None)
    return env._robot_population


def _build_robot_design_tensor(env, num_designs: int) -> None:
    """Record which design each env holds, and CHECK it against the sim.

    MultiUsdFileCfg assigns proto i to the i-th Robot prim that
    ``find_matching_prim_paths`` returns, so env i holds design i % k. That is an
    assumption about someone else's iteration order, and the identical
    assumption about the object pool silently gave 510 of 512 envs the wrong
    asset -- costing a 5.07 -> 3.00 goals/episode regression that every
    asset-level comparison passed (docs/multi_embodiment.md §4).

    So this does not merely record the map; it verifies it against a quantity
    that actually differs per design and is read back out of PhysX. Joint limits
    are that quantity: the sampler draws each design's abduction range
    independently, so the limit vector is effectively a fingerprint.
    """
    num_envs = env.num_envs
    robot_prim_paths = find_matching_prim_paths("/World/envs/env_.*/Robot")
    if len(robot_prim_paths) != num_envs:
        raise RuntimeError(
            f"Expected {num_envs} Robot prims, got {len(robot_prim_paths)}.")
    observed = [_env_id_of(p) for p in robot_prim_paths]
    if observed != sorted(observed):
        raise RuntimeError(
            "find_matching_prim_paths no longer returns Robot prims in numeric "
            f"env order (first 12: {observed[:12]}); the design assignment "
            "below assumes it does.")

    env._robot_design_index_per_env = torch.tensor(
        [i % num_designs for i in range(num_envs)],
        device=env.device, dtype=torch.long,
    )


def _verify_robot_design_assignment(env, specs) -> None:
    """Confirm env i really holds design i % k, using per-design joint limits.

    Runs after the sim exists, so it reads what PhysX loaded rather than what
    the config asked for. Cheap, and it closes the exact hole that produced the
    object-assignment regression.
    """
    import numpy as np

    limits = env.robot.data.joint_pos_limits.detach().cpu().numpy()  # (N, J, 2)
    idx = env._robot_design_index_per_env.detach().cpu().numpy()
    # Designs whose limit vectors are identical cannot distinguish an assignment
    # error, so report how much signal the check actually has.
    sigs = {}
    for e in range(env.num_envs):
        sigs.setdefault(idx[e], []).append(limits[e])
    distinct = {d: np.round(v[0], 9).tobytes() for d, v in sigs.items()}
    n_unique = len(set(distinct.values()))

    bad = []
    for d, mats in sigs.items():
        ref = mats[0]
        for m in mats[1:]:
            if not np.allclose(ref, m, atol=1e-9):
                bad.append(d)
                break
    if bad:
        raise RuntimeError(
            f"robot design assignment is inconsistent: envs sharing design "
            f"{sorted(bad)[:5]} do not share joint limits. env i is supposed to "
            f"hold design i % {len(specs)}.")
    print(f"[scene] robot design assignment verified: {len(sigs)} designs over "
          f"{env.num_envs} envs, {n_unique} distinguishable by joint limits")


def _build_object_scale_tensor(env, object_scales_normalized, num_object_usds: int) -> None:
    num_envs = env.num_envs
    object_prim_paths = find_matching_prim_paths("/World/envs/env_.*/Object")
    if len(object_prim_paths) != num_envs:
        raise RuntimeError(
            f"Expected {num_envs} Object prims after MultiUsdFileCfg spawn, "
            f"got {len(object_prim_paths)}. Cloner-drop bug may have returned."
        )

    env._object_scale_per_env = torch.zeros(num_envs, 3, device=env.device, dtype=torch.float32)
    env._object_asset_index_per_env = torch.zeros(num_envs, device=env.device, dtype=torch.long)
    # When the objects were authored, that pass already decided which pool entry
    # each env holds; take its map rather than re-deriving one here. Two
    # independent derivations of the same mapping is exactly how the paths
    # desynchronised before -- see the note in _author_objects_into_envs.
    authored_map = getattr(env, "_authored_asset_index", None)
    if authored_map is None:
        # The converted path pairs env i with the i-th spawned proto by relying
        # on MultiUsdFileCfg walking this same list in this same order. That
        # holds only while the order is numeric; if Isaac Lab ever returns
        # lexicographic order instead, every env past env_9 silently gets an
        # object its observation does not describe. Fail loudly instead.
        observed = [_env_id_of(p) for p in object_prim_paths]
        if observed != sorted(observed):
            raise RuntimeError(
                "find_matching_prim_paths no longer returns Object prims in "
                f"numeric env order (first 12: {observed[:12]}). "
                "_build_object_scale_tensor's asset assignment assumes it does; "
                "fix the assignment before training on this."
            )
    for source_idx, obj_path in enumerate(object_prim_paths):
        env_id = _env_id_of(obj_path)
        asset_index = (authored_map[env_id] if authored_map is not None
                       else source_idx % num_object_usds)
        env._object_scale_per_env[env_id] = torch.tensor(
            object_scales_normalized[asset_index], device=env.device, dtype=torch.float32,
        )
        env._object_asset_index_per_env[env_id] = asset_index


def setup_scene(env) -> None:
    """Build and register robot, table, object, goal, ground, and light."""
    # Both before anything reads them: setup_scene needs robot_spec, and
    # DirectRLEnv reads the widths in _configure_gym_env_spaces, which runs
    # after this hook.
    env.robot_spec = resolve_spec(env, env.cfg)
    derive_spaces(env.cfg, env.robot_spec)


    # One source for both asset backends: the converter path bakes these onto
    # every collision prim, the authoring path writes them directly.
    contact_offset = env.cfg.physics.contact_offset
    rest_offset = env.cfg.physics.rest_offset
    assets_cfg = env.cfg.assets
    setup_t0 = time.perf_counter()
    _log_scene_step(
        setup_t0,
        f"setup start num_envs={env.num_envs} "
        f"num_assets_per_type={assets_cfg.num_assets_per_type}",
    )

    # 1. Resolve the object pool: a single named URDF (DexToolBench eval) or
    #    procedural URDFs generated in a per-launch temp dir.
    env._tmp_asset_dir = tempfile.mkdtemp(prefix="genmech_assets_")
    if assets_cfg.object_urdf:
        if assets_cfg.object_scale is None:
            raise ValueError(
                "cfg.assets.object_scale must be set when object_urdf is given "
                "(policy-normalized grasp-bbox scale, NAME_TO_OBJECT convention)."
            )
        urdf_paths = [resolve_repo_path(assets_cfg.object_urdf)]
        if not urdf_paths[0].exists():
            raise FileNotFoundError(f"object_urdf not found: {urdf_paths[0]}")
        object_scales_normalized = [tuple(assets_cfg.object_scale)]
        # A caller-supplied URDF is not generated from parameters we hold, so it
        # cannot be authored -- only converted.
        object_params = None
    else:
        urdf_paths, object_scales_normalized, object_params = generate_handle_head_urdfs(
            handle_head_types=tuple(assets_cfg.handle_head_types),
            num_per_type=assets_cfg.num_assets_per_type,
            out_dir=env._tmp_asset_dir,
            shuffle=assets_cfg.shuffle_assets,
            seed=assets_cfg.object_seed,
            density_scale=assets_cfg.object_density_scale,
        )
    if not urdf_paths:
        raise ValueError(
            "No procedural object URDFs were generated. "
            "Check cfg.assets.handle_head_types and num_assets_per_type."
        )
    env._object_urdf_paths = [str(path) for path in urdf_paths]
    _log_scene_step(setup_t0, f"generated {len(urdf_paths)} object URDFs")

    # 2. Convert URDFs -> raw USDs -> role-specific baked USDs.
    usd_work_dir = Path(env._tmp_asset_dir) / "usd"
    bake_root = Path(env._tmp_asset_dir) / "baked_usd"
    usd_work_dir.mkdir(parents=True, exist_ok=True)

    # Objects can be AUTHORED instead of converted. Each is one rigid body with
    # two analytic shapes and a mass/inertia generate_objects already computes in
    # closed form, so the URDF round-trip through Kit's importer recovers numbers
    # we started with -- measured at 0.2 s per object, ~240 s for the 1200-object
    # pool. Authoring is ~5 prims each.
    #
    # Verified equivalent before being offered: mass, inertia and centre of mass
    # agree to ~1e-8 and dropped-object resting poses to ~2e-7 m
    # (genmech.tools.compare_object_assets, compare_object_physics). The capsule
    # detail matters and is handled in author_objects: this conversion passes
    # replace_cylinders_with_capsules=True, so a URDF cylinder becomes a CAPSULE
    # whose height is the cylindrical section.
    # "object" / "goalviz" author just one of the pair; used to bisect which
    # asset carries a behavioural difference rather than guessing attributes.
    author_objects = bool(getattr(assets_cfg, "author_object_usds", False))
    _which_cfg = str(getattr(assets_cfg, "author_which", "both")).lower()
    _author_which = ({"object", "goalviz"} if _which_cfg == "both"
                     else {_which_cfg}) if author_objects else set()
    if author_objects and object_params is None:
        raise ValueError(
            "assets.author_object_usds=True requires the procedural object pool; "
            "assets.object_urdf is set, and a caller-supplied URDF carries no "
            "parameters to author from."
        )

    object_raw_usds = [
        _convert_urdf_to_usd(
            str(urdf), usd_work_dir, fix_base=False, replace_cylinders_with_capsules=True,
        )
        for urdf in urdf_paths
    ] if not (author_objects and _author_which == {"object", "goalviz"}) else []
    object_usd_paths = [
        _bake_usd(usd, bake_root, "object", props=dict(
            kinematic_enabled=False, disable_gravity=False,
            max_depenetration_velocity=1000.0, articulation_enabled=False,
        ), contact_offset=contact_offset, rest_offset=rest_offset)
        for usd in object_raw_usds
    ]
    goalviz_usd_paths = [
        _bake_usd(usd, bake_root, "goalviz", props=dict(
            kinematic_enabled=True, disable_gravity=True, articulation_enabled=False,
        ), collision_enabled=False, contact_offset=contact_offset, rest_offset=rest_offset)
        for usd in object_raw_usds
    ]

    spec = env.robot_spec

    def _prepare_robot_usd(design_spec, tag: str) -> str:
        """URDF -> USD -> self-collision filters -> baked USD, for one design."""
        converted = _convert_urdf_to_usd(
            assets_cfg.robot_urdf or design_spec.urdf_path, usd_work_dir,
            fix_base=True, self_collision=True,
            joint_drive=_robot_joint_drive_cfg(),
            replace_cylinders_with_capsules=design_spec.replace_cylinders_with_capsules,
        )
        # Isaac Gym enables all robot self-collisions then masks adjacent links;
        # mirror that by authoring FilteredPairsAPI for the spec's adjacency
        # pairs before the bake (PhysX additionally auto-filters directly-jointed
        # parent/child links). The adjacency is PER DESIGN -- geometry-less
        # ghosted links are transparent, so which pairs need filtering differs.
        _apply_self_collision_filters(converted, design_spec.adjacent_links)
        return _bake_usd(
            converted, bake_root, tag,
            props=dict(
                disable_gravity=True, max_depenetration_velocity=1000.0,
                enabled_self_collisions=True,
                solver_position_iterations=8, solver_velocity_iterations=0,
            ),
            apply_physx_articulation=True,
            contact_offset=contact_offset, rest_offset=rest_offset,
        )

    # Resolved by the env before super().__init__ (the observation spaces are
    # derived from it); re-resolve only when setup_scene is driven directly, as
    # the benchmarking tools do.
    def _author_population_usds(specs) -> list[str]:
        """Author one USD per design, with the arm converted ONCE and referenced.

        Measured against the converter on this cluster: 1.17 s per design (~8 h
        for 24,576) becomes ~29 ms (~12 min), because ~90% of a conversion is
        re-importing the SAME arm's 16 STL meshes.

        The authored asset is verified equivalent to the converted one by
        genmech.tools.compare_authored_robot: masses, inertias, joint limits,
        colliders and every actuation property agree exactly, and driven
        identically the fingertips land within 0.0013 mm.
        """
        import numpy as np

        from isaacsimenvs.pose_reaching_6d.scene_utils.author_robot import (
            arm_only_urdf, author_robot_usd, flatten_arm_usd,
        )
        from pxr import Sdf, Usd, UsdGeom

        arm_dir = Path(env._tmp_asset_dir) / "arm"
        arm_urdf = arm_only_urdf(arm_dir / "iiwa14_arm_only.urdf")
        arm_raw = _convert_urdf_to_usd(
            str(arm_urdf), arm_dir, fix_base=True, self_collision=True,
            joint_drive=_robot_joint_drive_cfg())
        # Flatten: the converter's output references configuration/*_base.usd,
        # and those do NOT resolve through a second level of nesting -- a
        # referenced arm otherwise composes with NO collision geometry.
        arm_usd = flatten_arm_usd(arm_raw, arm_dir / "arm_flat.usd",
                                  contact_offset=contact_offset,
                                  rest_offset=rest_offset)
        arm_stage = Usd.Stage.Open(arm_usd)
        arm_root = str(next(c for c in arm_stage.GetPseudoRoot().GetChildren()).GetPath())
        link7_world = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(
            arm_stage.GetPrimAtPath(f"{arm_root}/iiwa14_link_7"))).T
        _log_scene_step(setup_t0, "converted the shared arm once")

        layer_for_envs = get_current_stage().GetRootLayer()
        hands = _ensure_robot_population(env, assets_cfg).hands

        if bool(getattr(assets_cfg, "author_robots_into_envs", True)):
            # DIRECTLY INTO THE ENV PRIMS, no files and no MultiUsdFileCfg.
            #
            # Writing a USD per design and spawning it makes Isaac Lab compose
            # each design into each env, which is the 0.00018*k*n term in
            # docs/multi_embodiment.md 3 -- about 30 h at k = n = 24,576, and
            # the reason a 24k run sat in the scene build for hours AFTER
            # authoring completed in 28 min. Authoring into the env prims is
            # what bench_authored_sps measured at 362 s for k = n = 24,576, and
            # is exactly the pattern _author_objects_into_envs already uses.
            from isaacsimenvs.pose_reaching_6d.scene_utils.author_robot import author_robot_prims
            from isaacsimenvs.pose_reaching_6d.scene_utils.author_usd import _set_xform

            env._robot_design_index_per_env = torch.tensor(
                [i % len(specs) for i in range(env.num_envs)],
                device=env.device, dtype=torch.long)
            env._robot_collider_links = {}
            # The same pose build_robot_articulation_usd_cfg puts in init_state,
            # applied to the prim instead because a fixed base ignores the other.
            base_pos = env.robot_spec.base_pos
            base_rot = env.robot_spec.base_rot
            t_auth = time.perf_counter()
            with Sdf.ChangeBlock():
                for env_path in sorted(env.scene.env_prim_paths, key=_env_id_of):
                    idx = _env_id_of(env_path) % len(specs)
                    _, _cl = author_robot_prims(
                        layer_for_envs, f"{env_path}/Robot", hands[idx],
                        specs[idx], arm_usd=arm_usd, arm_root_prim=arm_root,
                        link7_world=link7_world,
                        adjacency=specs[idx].adjacent_links,
                        in_change_block=True, contact_offset=contact_offset, rest_offset=rest_offset)
                    # Recorded HERE, where the geometry is created, so the
                    # friction pass does not have to rediscover it per link.
                    env._robot_collider_links[idx] = _cl
                    # PLACE THE BASE. The arm is FIXED-BASE: its root_joint pins
                    # it to the world wherever the prim sits, so init_state.pos
                    # cannot move it afterwards -- writing a root pose to a
                    # fixed articulation does nothing. Isaac Lab's spawner
                    # normally applies translation/orientation when it CREATES
                    # the prim, but spawn=None means nothing creates it, so
                    # nothing applied it: the robot sat at the env origin, on
                    # top of the table instead of 0.8 m behind it, with link_7
                    # reaching to y = -0.575 instead of +0.181.
                    _root_spec = layer_for_envs.GetPrimAtPath(f"{env_path}/Robot")
                    _set_xform(_root_spec, tuple(float(v) for v in base_pos),
                               tuple(float(v) for v in base_rot))
            _log_scene_step(
                setup_t0,
                f"authored {env.num_envs} robots into env prims from "
                f"{len(specs)} designs "
                f"({(time.perf_counter() - t_auth) / max(env.num_envs, 1) * 1000:.2f} ms each)")
            return None          # signals: nothing to spawn

        out = []
        design_dir = Path(env._tmp_asset_dir) / "designs"
        for i, (hand, design_spec) in enumerate(zip(hands, specs)):
            # ONE DIRECTORY PER DESIGN. _bake_usd copies the source file's
            # sibling directories (shutil.copytree over the parent's children),
            # so putting every authored design in one folder makes baking design
            # i copy the i-1 designs before it -- O(n^2), and measured at 39
            # designs/min falling, i.e. SLOWER than the 51/min converter this
            # path exists to beat.
            raw = author_robot_usd(
                hand, design_spec, design_dir / hand.name / f"{hand.name}.usd",
                arm_usd=arm_usd, arm_root_prim=arm_root, link7_world=link7_world, contact_offset=contact_offset, rest_offset=rest_offset)
            # The same finish the converted path gets, so self-collision
            # filtering and the articulation/solver properties are identical.
            _apply_self_collision_filters(raw, design_spec.adjacent_links)
            out.append(_bake_usd(
                raw, bake_root, f"robot_{i:05d}",
                props=dict(disable_gravity=True, max_depenetration_velocity=1000.0,
                           enabled_self_collisions=True,
                           solver_position_iterations=8, solver_velocity_iterations=0),
                apply_physx_articulation=True, contact_offset=contact_offset, rest_offset=rest_offset))
        return out

    population = _ensure_robot_population(env, assets_cfg)
    population_specs = population.specs if population is not None else None
    if population_specs is None:
        robot_usd_arg: str | list[str] = _prepare_robot_usd(spec, "robot")
    else:
        # One asset per DESIGN, not per env: n envs share k designs, and
        # preparing per env is what turned a 24,576-env build into hours
        # (docs/multi_embodiment.md §3).
        if bool(getattr(assets_cfg, "author_robot_usds", False)):
            robot_usd_arg = _author_population_usds(population_specs)
            # None means the prims are already on the stage: spawn nothing.
            if robot_usd_arg is None:
                robot_usd_arg = "__authored_in_place__"
        else:
            robot_usd_arg = [
                _prepare_robot_usd(s, f"robot_{i:05d}")
                for i, s in enumerate(population_specs)
            ]
        if isinstance(robot_usd_arg, str):
            # The sentinel, not a path list. len() on it reported "21 distinct
            # robot USDs" -- the length of "__authored_in_place__" -- for a run
            # that wrote no USDs at all.
            _log_scene_step(setup_t0, "robot prims already on the stage, "
                                      "nothing to spawn")
        else:
            _log_scene_step(
                setup_t0, f"prepared {len(robot_usd_arg)} distinct robot USDs")
    robot_usd_path = robot_usd_arg
    # Table USD(s). When table_scale_range_x/y are non-trivial and
    # table_scale_num_variants > 1, pre-bake N scaled URDF variants and pass
    # them as a list to RigidObject — Isaac Lab's MultiUsdFileCfg cycles
    # through the list, giving each env one of the variants. Z scale is held
    # at 1.0 so the table surface height matches the policy's expectation.
    scale_range_x = tuple(float(v) for v in getattr(assets_cfg, "table_scale_range_x", (1.0, 1.0)))
    scale_range_y = tuple(float(v) for v in getattr(assets_cfg, "table_scale_range_y", (1.0, 1.0)))
    n_table_variants = int(getattr(assets_cfg, "table_scale_num_variants", 1))
    table_scale_is_trivial = (
        scale_range_x == (1.0, 1.0) and scale_range_y == (1.0, 1.0)
    ) or n_table_variants <= 1
    if table_scale_is_trivial:
        table_usd_paths = [_bake_usd(
            _convert_urdf_to_usd(assets_cfg.table_urdf, usd_work_dir, fix_base=False),
            bake_root, "table",
            props=dict(
                kinematic_enabled=True, disable_gravity=True, articulation_enabled=False,
            ),
            contact_offset=contact_offset, rest_offset=rest_offset,
        )]
        # Single (sx, sy) = (1.0, 1.0) for downstream consumers (eval viz).
        env._table_variant_scales = [(1.0, 1.0)]
    else:
        variant_urdf_dir = Path(env._tmp_asset_dir) / "table_variants"
        variant_urdf_paths, variant_scales = _generate_scaled_table_urdfs(
            base_urdf_path=assets_cfg.table_urdf,
            num_variants=n_table_variants,
            scale_range_x=scale_range_x,
            scale_range_y=scale_range_y,
            out_dir=variant_urdf_dir,
            # Deterministic across runs so the on-disk variants are stable.
            # The env-level seed governs which variant lands in which env via
            # Isaac Lab's round-robin spawn ordering.
            seed=0,
        )
        env._table_variant_scales = list(variant_scales)
        table_usd_paths = [
            _bake_usd(
                _convert_urdf_to_usd(p, usd_work_dir, fix_base=False),
                bake_root, f"table_variant_{idx:03d}",
                props=dict(
                    kinematic_enabled=True, disable_gravity=True, articulation_enabled=False,
                ),
                contact_offset=contact_offset, rest_offset=rest_offset,
            )
            for idx, p in enumerate(variant_urdf_paths)
        ]
        # variant_scales already stashed above for downstream consumers.
        _log_scene_step(
            setup_t0,
            f"baked {len(table_usd_paths)} scaled table USD variants "
            f"x_range={scale_range_x} y_range={scale_range_y}",
        )
    _log_scene_step(setup_t0, "resolved baked USDs")

    # 3. Pre-create env roots so regex spawns resolve to every env.
    _materialize_env_prims(env)
    # These three steps were a single silent block, and at 24,576 authored
    # robots (~5.8M prims on the stage) it stayed silent for over 20 minutes
    # with no way to tell which step was responsible -- or whether it was
    # progressing at all. Anything that traverses or recomposes the stage per
    # env costs O(n x stage) here, which is the shape of every scaling bug found
    # on this path so far, so each step now reports its own time.
    _log_scene_step(setup_t0, "materialized env prims")

    # 4. Spawn assets.
    env.robot = Articulation(build_robot_articulation_usd_cfg(
        robot_usd_path,
        env.robot_spec,
        start_arm_higher=getattr(env.cfg.reset, "start_arm_higher", False),
    ))
    _log_scene_step(setup_t0, "built robot articulation")
    env.table = RigidObject(build_rigid_object_cfg("/World/envs/env_.*/Table", table_usd_paths))
    _log_scene_step(setup_t0, "spawned table")
    if author_objects:
        # Author one object and one goal marker INTO each env, cycling the pool
        # the same way MultiUsdFileCfg would. This replaces both the per-variant
        # conversion (~0.2 s x 1200) and the proto/copy spawn machinery: there is
        # no USD file, no template prim, and no Sdf.CopySpec.
        n_pool = len(object_params)
        _author_objects_into_envs(env, object_params, n_pool, which=_author_which)
        env.object = (RigidObject(RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object", spawn=None))
            if "object" in _author_which
            else RigidObject(build_rigid_object_cfg(
                "/World/envs/env_.*/Object", object_usd_paths)))
        env.goal_viz = (RigidObject(RigidObjectCfg(
            prim_path="/World/envs/env_.*/GoalViz", spawn=None))
            if "goalviz" in _author_which
            else RigidObject(build_rigid_object_cfg(
                "/World/envs/env_.*/GoalViz", goalviz_usd_paths)))
    else:
        env.object = RigidObject(build_rigid_object_cfg("/World/envs/env_.*/Object", object_usd_paths))
        env.goal_viz = RigidObject(build_rigid_object_cfg("/World/envs/env_.*/GoalViz", goalviz_usd_paths))
    _log_scene_step(setup_t0, "spawned robot/table/object/goalviz")

    # 5. Ground plane + dome light (global, outside env_*).
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # 6. Per-env scale tensor for spawned Objects.
    _build_object_scale_tensor(
        env, object_scales_normalized,
        len(object_params) if author_objects else len(object_usd_paths))
    if env._robot_population_specs is not None:
        _build_robot_design_tensor(env, len(env._robot_population_specs))

    # 7. Register with scene so DirectRLEnv refreshes their tensors each step.
    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["table"] = env.table
    env.scene.rigid_objects["object"] = env.object
    env.scene.rigid_objects["goal_viz"] = env.goal_viz
    _log_scene_step(setup_t0, "registered assets with scene")
