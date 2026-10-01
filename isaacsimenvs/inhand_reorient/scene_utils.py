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

import dataclasses
import math
import tempfile
import time
from pathlib import Path

import isaaclab.sim as sim_utils
from hand_sampler import robot_param_constants as rpc
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from isaacsimenvs.pose_reaching_6d.scene_utils.assembly import _convert_fixed_robot

from . import hand_only
from .hand_only import build_hand_only_spec
from .obs_utils import derive_spaces
from .palm_calibration import load_calibration
from .repose_profile import is_repose

ROBOT_PATH = "/World/envs/env_.*/Robot"
OBJECT_PATH = "/World/envs/env_.*/Object"
GOALVIZ_PATH = "/World/envs/env_.*/GoalViz"

__all__ = ["setup_scene", "finalize_scene", "apply_palm_calibration"]

_LIMIT_MARGIN_RAD = 1e-3
"""How far a calibrated default_joint_pos gets nudged inside its joint's own
limits in ``apply_palm_calibration`` -- see the comment at its call site."""


def apply_palm_calibration(env, spec):
    """Overrides ``spec.base_rot`` and ``spec.hand_default_joint_pos`` (and
    ``env.cfg.reset.object_spawn_offset``, IN/OUT like ``derive_spaces``) from
    ``hand_calibration.json``'s entry for ``env.cfg.assets.hand_id``, written
    by ``calibrate_palm_up.py``.

    The default-joint-pos override (I24 priority check) matters beyond
    calibration itself: ``env.robot.data.default_joint_pos`` -- seeded from
    ``spec.hand_default_joint_pos`` via ``_hand_articulation_cfg``'s
    ``InitialStateCfg`` -- is also what every episode resets TO
    (``reset_utils.reset_env_state``) and what "zero action" targets
    (``obs_utils.pre_physics_step``), so a calibration that found a genuinely
    CUPPED hold (not the flat, uncurled ``clamp(0.0, lower, upper)`` every
    hand got before I24) needs to reach training the same way ``base_rot``
    already does, or training still starts from -- and keeps drifting back
    toward -- an open hand that never touched the object.

    Fallback (no entry for this hand, or no file yet): identity ``base_rot``
    and the spec's own (flat) default pose, with a logged warning --
    calibration is data, not a hard requirement, so an uncalibrated hand
    still boots.
    """
    hand_id = env.cfg.assets.hand_id
    entry = load_calibration().get(hand_id)
    if entry is None:
        print(f"[inhand_reorient] WARNING: no palm-up calibration for hand_id={hand_id!r} "
              f"in hand_calibration.json; falling back to identity base_rot and the spec's own "
              f"(flat) default joint pose. Run calibrate_palm_up.py --hand {hand_id} to add one.",
              flush=True)
        return spec
    base_rot = tuple(float(v) for v in entry["base_rot"])
    spawn_offset = tuple(float(v) for v in entry["spawn_offset_local"])
    env.cfg.reset.object_spawn_offset = spawn_offset

    limit_of = dict(zip(spec.hand_joint_names, spec.hand_joint_limits))
    hand_default_joint_pos = dict(spec.hand_default_joint_pos)
    calibrated_pose = entry.get("hand_default_joint_pos")
    n_overridden = 0
    if calibrated_pose:
        for j, v in calibrated_pose.items():
            if j in hand_default_joint_pos:
                # I25 retry crash (wuji_right): a calibration winner with
                # curl_frac/curl_profile position 0.0 saves the RAW lower
                # bound as that joint's default -- exactly on the limit.
                # The saved value round-trips through float32 (PhysX
                # soft_joint_pos_limits) -> JSON -> float64, which lands a
                # few ULPs outside the hard limit Isaac Lab's own
                # Articulation._validate_cfg() re-derives from the USD, so a
                # boundary-exact default can hard-error at scene boot ("...
                # default positions out of the limits") even though it was
                # in range when it was measured. _LIMIT_MARGIN_RAD nudges
                # any calibrated default a hair inside its own joint's
                # limits -- physically negligible (0.057 deg), just enough
                # to absorb that rounding.
                lo, hi = limit_of.get(j, (-float("inf"), float("inf")))
                margin = min(_LIMIT_MARGIN_RAD, max(0.0, (hi - lo) / 2.0 - 1e-9))
                hand_default_joint_pos[j] = min(max(float(v), lo + margin), hi - margin)
                n_overridden += 1

    print(f"[inhand_reorient] hand={hand_id} palm-up calibration: base_rot={base_rot} "
          f"palm_normal_axis={entry.get('axis')} curl_frac={entry.get('curl_frac')} "
          f"rest_score={entry.get('score')} (stability={entry.get('stability_score')} "
          f"reach={entry.get('reach_score')}) spawn_offset_local={spawn_offset} "
          f"default_joint_pos_overridden={n_overridden}/{len(hand_default_joint_pos)} "
          f"(dated {entry.get('date')}, sha {entry.get('git_sha', 'unknown')[:12]})", flush=True)
    return dataclasses.replace(
        spec, base_rot=base_rot, hand_default_joint_pos=hand_default_joint_pos)


# Grammar envelope only (never present in a single-hand HandOnlySpec):
# pc0_j/pc1_j are mechanically DIFFERENT from every other joint slot even
# when they are themselves a GHOST -- a padding-only carrier still often
# carries an entire REAL finger chain (envelope slot 3/4's root-mounted
# digit, see grammar_envelope.canonicalize). A Kit smoke found the hand's
# ordinary (SHARPA-derived) gains too weak to hold that load against
# gravity through a near-massless ghost body: a bounded creep of the ghost
# past its (0, 1e-8) limit (up to ~2.3 rad over 200 steps on wuji_right,
# whose finger 3 hangs off a ghost pc0) -- distinct from, and discovered
# only after fixing, the far more severe (100+ rad in 1-2 steps)
# rest-self-penetration instability the same population had before
# make_grammar_population.py started filtering on rest_overlap_pairs. This
# carrier-only actuator group (500/20/0.01 vs the hand's own SHARPA-derived
# ~3.9/0.15) brought the residual down to ~0.0125 rad max over 200 steps --
# bounded and stable, but still above the design note's 1e-4 target; a 6x
# stiffness increase (3000/80) made no further difference (bit-identical
# result), so the residual is bounded by `physxJoint:maxJointVelocity`
# (10 rad/s) during a brief initial transient, not by drive weakness --
# left as a known gap, see the Phase 2 report.
_CARRIER_JOINT_NAMES = ("pc0_j", "pc1_j")
_CARRIER_STIFFNESS = 500.0
_CARRIER_DAMPING = 20.0
_CARRIER_ARMATURE = 0.01
_CARRIER_EFFORT_LIMIT_NM = 5.0
"""Review risk 8, second half: the AUTHORED USD joint drive's own
`maxForce` (`author_grammar._author_joint`, `rpc.GEN_JOINT_EFFORT_NM` = 1.0
Nm, the SAME for every joint including carriers) saturates before the
stiff carrier gains above can do their job -- a Kit smoke found a 6x
stiffness increase (3000/80) bit-identical to the 500/20 baseline, because
the torque output was capped at 1 Nm either way, not because 500/20 was
already sufficient. `ImplicitActuatorCfg.effort_limit_sim` overrides that
authored cap for this actuator group ONLY (the "hand" group keeps the
authored 1 Nm), scene-wide default; `_apply_per_env_carrier_gains` below
then restores the REAL-carrier envs' whole group (stiffness/damping/
armature/effort) back to the hand's own normal per-joint values, since a
REAL carrier is an actuated design joint the policy controls, not a
mechanical support -- it should not get a stronger-than-normal torque
budget or 128x stiffer gains (review risk 8's "SHARPA and carrier designs
get a palm DOF 128x stiffer than their fingers")."""


def _repose_hand_props(r):
    """``ALLEGRO_HAND_CFG``'s rigid-body and articulation-root properties
    (isaaclab_assets/robots/allegro.py), from the ``repose`` profile block."""
    rigid = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=bool(r.hand_disable_gravity),
        retain_accelerations=False,
        enable_gyroscopic_forces=False,
        angular_damping=float(r.hand_angular_damping),
        max_linear_velocity=1000.0,
        max_angular_velocity=64 / math.pi * 180.0,
        max_depenetration_velocity=float(r.hand_max_depenetration_velocity),
        max_contact_impulse=1e32,
    )
    articulation = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=int(r.hand_solver_position_iterations),
        solver_velocity_iteration_count=int(r.hand_solver_velocity_iterations),
        sleep_threshold=float(r.hand_sleep_threshold),
        stabilization_threshold=float(r.hand_stabilization_threshold),
    )
    return rigid, articulation


def _hand_articulation_cfg(spec, usd_path: str | None, repose=None) -> ArticulationCfg:
    """`usd_path=None` (the grammar-population path): the prims were already
    authored directly into the stage (`scene/author_grammar.py`), so this
    Articulation only needs to ATTACH to them, not spawn anything. Scene-
    wide defaults only (one `ImplicitActuatorCfg` per named group, applied
    identically to every env at construction time) -- a population's PER-ENV
    override (ghost vs real carrier) happens after the articulation view is
    live, in `_apply_per_env_carrier_gains`, the same pattern
    `_resolve_population_joint_permutation` already uses for
    `default_joint_pos`."""
    carrier_names = [n for n in _CARRIER_JOINT_NAMES if n in spec.hand_joint_names]
    hand_names = [n for n in spec.hand_joint_names if n not in carrier_names]
    actuators = {
        "hand": ImplicitActuatorCfg(
            joint_names_expr=hand_names,
            stiffness={n: spec.hand_stiffness[n] for n in hand_names},
            damping={n: spec.hand_damping[n] for n in hand_names},
            armature={n: spec.hand_armature[n] for n in hand_names},
            friction=0.0,
        ),
    }
    if carrier_names:
        actuators["carrier"] = ImplicitActuatorCfg(
            joint_names_expr=carrier_names,
            stiffness={n: _CARRIER_STIFFNESS for n in carrier_names},
            damping={n: _CARRIER_DAMPING for n in carrier_names},
            armature={n: _CARRIER_ARMATURE for n in carrier_names},
            effort_limit_sim={n: _CARRIER_EFFORT_LIMIT_NM for n in carrier_names},
            friction=0.0,
        )
    spawn = None
    if usd_path is not None:
        if repose is None:
            spawn = sim_utils.UsdFileCfg(usd_path=usd_path)
        else:
            rigid, articulation = _repose_hand_props(repose)
            spawn = sim_utils.UsdFileCfg(
                usd_path=usd_path, rigid_props=rigid, articulation_props=articulation)
    return ArticulationCfg(
        prim_path=ROBOT_PATH,
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=spec.base_pos, rot=spec.base_rot,
            joint_pos=dict(spec.hand_default_joint_pos),
            joint_vel={".*": 0.0},
        ),
        actuators=actuators,
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


def _repose_object_cfg(prim_path: str, r, *, kinematic: bool,
                       color: tuple[float, float, float] = (0.2, 0.4, 0.9)) -> RigidObjectCfg:
    """The ``isaaclab_repose`` cube: NVIDIA's DexCube (0.06 m collision cube
    at scale 1.2) rigid-body, mass and collision properties, as a plain
    cuboid (no download at boot). The kinematic goal marker gets no
    collision or mass."""
    if kinematic:
        rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
        mass_props = None
        collision_props = None
    else:
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
            enable_gyroscopic_forces=bool(r.object_enable_gyroscopic_forces),
            solver_position_iteration_count=int(r.object_solver_position_iterations),
            solver_velocity_iteration_count=int(r.object_solver_velocity_iterations),
            sleep_threshold=float(r.object_sleep_threshold),
            stabilization_threshold=float(r.object_stabilization_threshold),
            max_depenetration_velocity=float(r.object_max_depenetration_velocity),
        )
        if float(r.object_mass_kg) > 0.0:
            mass_props = sim_utils.MassPropertiesCfg(mass=float(r.object_mass_kg))
        else:
            mass_props = sim_utils.MassPropertiesCfg(density=float(r.object_density))
        collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=float(r.object_contact_offset),
            rest_offset=float(r.object_rest_offset),
            torsional_patch_radius=float(r.object_torsional_patch_radius),
            min_torsional_patch_radius=float(r.object_min_torsional_patch_radius),
        )
    size = float(r.object_size_m)
    spawn = sim_utils.CuboidCfg(
        size=(size, size, size),
        rigid_props=rigid_props, mass_props=mass_props, collision_props=collision_props,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )
    return RigidObjectCfg(prim_path=prim_path, spawn=spawn)


def _scene_objects(env, offsets: dict) -> tuple[RigidObject, RigidObject]:
    """``(object, goal marker)`` for the active task profile."""
    if is_repose(env.cfg):
        r = env.cfg.repose
        return (RigidObject(_repose_object_cfg(OBJECT_PATH, r, kinematic=False)),
                RigidObject(_repose_object_cfg(GOALVIZ_PATH, r, kinematic=True, color=(0.9, 0.3, 0.2))))
    return (
        RigidObject(_object_cfg(
            OBJECT_PATH, env.cfg.assets.object_size_m, offsets, kinematic=False,
            density=env.cfg.assets.object_density)),
        RigidObject(_object_cfg(
            GOALVIZ_PATH, env.cfg.assets.object_size_m, offsets, kinematic=True,
            color=(0.9, 0.3, 0.2))),
    )


def _apply_repose_props_to_population(env) -> None:
    """The population path authors its robots itself (no spawner), so the
    profile's hand properties are written onto every env's Robot afterwards."""
    t0 = time.perf_counter()
    rigid, articulation = _repose_hand_props(env.cfg.repose)
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Sdf

    stage = get_current_stage()
    with Sdf.ChangeBlock():
        for env_path in env.scene.env_prim_paths:
            robot_path = f"{env_path}/Robot"
            sim_utils.modify_rigid_body_properties(robot_path, rigid, stage=stage)
            sim_utils.modify_articulation_root_properties(robot_path, articulation, stage=stage)
    print(f"[inhand_reorient] isaaclab_repose hand properties written to {len(env.scene.env_prim_paths)} "
          f"robots ({time.perf_counter() - t0:.1f}s)", flush=True)


def setup_scene(env) -> None:
    if env.cfg.assets.hand_population:
        _setup_scene_population(env)
        return
    _setup_scene_single_hand(env)


def _setup_scene_single_hand(env) -> None:
    t0 = time.perf_counter()
    asset_dir = Path(tempfile.mkdtemp(prefix="inhand_reorient_"))

    spec, cut = build_hand_only_spec(env.cfg.assets.hand_id, out_dir=asset_dir)
    spec = apply_palm_calibration(env, spec)
    derive_spaces(env.cfg, spec)
    print(f"[inhand_reorient] hand={env.cfg.assets.hand_id} joints={spec.num_hand_joints} "
          f"root={spec.hand_root} unresolved_meshes={cut.unresolved_meshes} "
          f"({time.perf_counter() - t0:.1f}s)", flush=True)

    offsets = dict(contact_offset=env.cfg.physics.contact_offset,
                   rest_offset=env.cfg.physics.rest_offset)
    robot_usd, _robot_root = _convert_fixed_robot(spec, spec.urdf_path, asset_dir / "usd", offsets)
    print(f"[inhand_reorient] converted hand USD ({time.perf_counter() - t0:.1f}s)", flush=True)

    repose = env.cfg.repose if is_repose(env.cfg) else None
    env.robot = Articulation(_hand_articulation_cfg(spec, robot_usd, repose=repose))
    env.object, env.goal_viz = _scene_objects(env, offsets)

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


def _setup_scene_population(env) -> None:
    """Phase 2: one design per env, authored directly (no regex-spawner
    clone) -- see `scene/author_grammar.py`."""
    from .scene import author_grammar

    assert env.cfg.scene.replicate_physics is False, (
        "assets.hand_population is set: pass env.scene.replicate_physics=False "
        "(a population cannot be homogenized by cloning env 0's physics)")
    assert env.cfg.scene.clone_in_fabric is False, (
        "assets.hand_population is set: pass env.scene.clone_in_fabric=False")

    t0 = time.perf_counter()
    _materialize_env_prims_population(env)
    author_grammar.setup_grammar_robot(env)
    spec = env.hand_spec
    derive_spaces(env.cfg, spec)
    print(f"[inhand_reorient] grammar population: {env.hand_tables.n_designs} designs, "
          f"{env.num_envs} envs, action_space={env.cfg.action_space} "
          f"({time.perf_counter() - t0:.1f}s)", flush=True)

    offsets = dict(contact_offset=env.cfg.physics.contact_offset, rest_offset=env.cfg.physics.rest_offset)
    if is_repose(env.cfg):
        _apply_repose_props_to_population(env)
    env.robot = Articulation(_hand_articulation_cfg(spec, usd_path=None))
    env.object, env.goal_viz = _scene_objects(env, offsets)

    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # No clone_environments(): every env's Robot prim was authored directly
    # above, distinct per design. Object/GoalViz still spawn per env through
    # their own regex prim_path (independent of clone_environments/
    # replicate_physics, which only govern the manually-authored robot).
    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["object"] = env.object
    env.scene.rigid_objects["goal_viz"] = env.goal_viz
    print(f"[inhand_reorient] population scene ready ({time.perf_counter() - t0:.1f}s)", flush=True)


def _materialize_env_prims_population(env) -> None:
    from isaacsim.core.utils.stage import get_current_stage

    stage = get_current_stage()
    for env_path in env.scene.env_prim_paths:
        if not stage.GetPrimAtPath(env_path).IsValid():
            stage.DefinePrim(env_path, "Xform")


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
    missing_tips = [t for t in env.hand_spec.fingertip_body_names if t not in body_names]
    if missing_tips:
        raise RuntimeError(
            f"fingertip bodies {missing_tips} not among the articulation's bodies {body_names}")
    env.fingertip_body_idx = [body_names.index(t) for t in env.hand_spec.fingertip_body_names]
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

    # Design note risk 1 ("topology drift: assert homogeneity") -- every env's
    # design is authored directly into its own USD subtree (author_grammar.py),
    # not cloned from one template like the single-hand path, so nothing but
    # this assertion catches a design whose AUTHORED topology silently
    # diverged from the fixed 32-slot envelope (a body/joint count or type
    # mismatch PhysX's own tensor API would otherwise either reject outright
    # or, worse, silently fall back to a per-env Python path for). `omni.
    # physics.tensors.ArticulationView.is_homogeneous`: "whether all the
    # articulations in the view are of the same type."
    if not env.robot.root_physx_view.is_homogeneous:
        raise RuntimeError(
            "articulation view is NOT homogeneous across envs -- some env's authored "
            "topology diverged from the fixed 32-slot envelope (see grammar_envelope.py's "
            "module docstring); this must never happen for any admitted population")

    if getattr(env, "hand_tables", None) is not None:
        _resolve_population_joint_permutation(env)


def _resolve_population_joint_permutation(env) -> None:
    """The articulation view's own joint order is not guaranteed to match
    `grammar_envelope.SLOT_NAMES` (PhysX/articulation discovery order need
    not equal Sdf-authoring order -- confirmed different for the
    single-hand SHARPA path too, see `diagnostics.py`'s own note). Every
    `env.hand_tables` table (`joint_valid`, `joint_limits`, ...) is indexed
    by `SLOT_NAMES`; `env.scene_record["slot_of_phys_col"]` is the
    permutation from an articulation-view COLUMN to its `SLOT_NAMES` index,
    so `table[..., perm]` reindexes a `SLOT_NAMES`-ordered row into
    articulation-view column order. Fatal (not a silent best-effort) if any
    name is missing -- exactly the silent-mispairing bug `diagnostics.py`
    warns about, now checked instead of risked."""
    import torch

    from .scene import grammar_envelope as ge

    phys_names = list(env.robot.data.joint_names)
    missing = [n for n in phys_names if n not in ge.SLOT_NAMES]
    if missing:
        raise RuntimeError(f"articulation view has joint(s) {missing} not in grammar_envelope.SLOT_NAMES")
    if len(phys_names) != len(ge.SLOT_NAMES):
        raise RuntimeError(
            f"articulation view has {len(phys_names)} joints, SLOT_NAMES has {len(ge.SLOT_NAMES)}")
    perm = [ge.SLOT_NAMES.index(n) for n in phys_names]
    if perm == list(range(len(ge.SLOT_NAMES))):
        print("[inhand_reorient] articulation joint order matches SLOT_NAMES (identity permutation)",
              flush=True)
    else:
        print(f"[inhand_reorient] articulation joint order DIFFERS from SLOT_NAMES; "
              f"permutation resolved and stored in env.scene_record", flush=True)
    perm_t = torch.as_tensor(perm, device=env.device, dtype=torch.long)
    env.scene_record["slot_of_phys_col"] = perm_t

    # I27/risk 3 (design note): every env's articulation initialised its own
    # `default_joint_pos` from env 0's design's `hand_default_joint_pos`
    # (`_hand_articulation_cfg`'s `InitialStateCfg` is one template dict for
    # the whole scene) -- WRONG for every env whose design differs. Overwrite
    # it per env, in phys-column order, now that the articulation view (and
    # this permutation) are live.
    population = env.hand_tables
    design_idx = env.scene_record["design_idx"]
    default_pos = torch.as_tensor(population.default_joint_pos, device=env.device, dtype=torch.float32)
    default_pos = default_pos[design_idx][:, perm_t]  # (num_envs, 32), phys column order
    env.robot.data.default_joint_pos[:] = default_pos
    env.robot.write_joint_state_to_sim(default_pos, torch.zeros_like(default_pos))
    # Review risk 11: cache it OUTSIDE Isaac Lab's own mutable buffer too --
    # `reset_utils._population_default_joint_pos`/`obs_utils.pre_physics_
    # step` read this instead of `env.robot.data.default_joint_pos` on the
    # population path, so a later articulation re-initialization (which
    # rebuilds that buffer from the scene-wide CFG template, env 0's design)
    # cannot silently revert every other env's default pose.
    env.scene_record["default_joint_pos"] = default_pos.clone()
    print(f"[inhand_reorient] wrote per-env default_joint_pos for {env.num_envs} envs "
          f"from their own design's palm_up calibration", flush=True)

    _apply_per_env_carrier_gains(env, phys_names, design_idx)


def _apply_per_env_carrier_gains(env, phys_names: list[str], design_idx) -> None:
    """Review risk 8: the scene-wide "carrier" `ImplicitActuatorCfg`
    (`_hand_articulation_cfg`, stiff gains + a raised effort limit) is
    meant for a GHOST pc0/pc1 (mechanically supporting a root-mounted
    digit through a near-massless body, see this module's own comment
    above `_CARRIER_JOINT_NAMES`) -- applied scene-wide, it ALSO lands on
    every env whose pc0/pc1 is instead a REAL, policy-actuated design
    joint (SHARPA, "carrier" designs), 128x stiffer and 5x the torque
    budget of that same design's own fingers. Overwrite those envs' whole
    carrier group back to the hand's normal per-joint gains, now that the
    articulation view (and its own joint-order permutation) are live --
    same per-env-override pattern as `default_joint_pos` just above."""
    import torch

    from .scene import grammar_envelope as ge

    carrier_cols = [i for i, n in enumerate(phys_names) if n in _CARRIER_JOINT_NAMES]
    if not carrier_cols:
        return
    carrier_slots = [ge.SLOT_NAMES.index(phys_names[i]) for i in carrier_cols]

    population = env.hand_tables
    joint_valid = torch.as_tensor(population.joint_valid, device=env.device, dtype=torch.bool)
    real = joint_valid[design_idx][:, carrier_slots]  # (num_envs, len(carrier_cols)) bool

    def _per_env(real_value: float, ghost_value: float) -> torch.Tensor:
        ghost = torch.full_like(real, ghost_value, dtype=torch.float32)
        real_t = torch.full_like(real, real_value, dtype=torch.float32)
        return torch.where(real, real_t, ghost)

    stiffness = _per_env(hand_only.DEFAULT_HAND_STIFFNESS, _CARRIER_STIFFNESS)
    damping = _per_env(hand_only.DEFAULT_HAND_DAMPING, _CARRIER_DAMPING)
    armature = _per_env(hand_only.DEFAULT_HAND_ARMATURE, _CARRIER_ARMATURE)
    effort = _per_env(rpc.GEN_JOINT_EFFORT_NM, _CARRIER_EFFORT_LIMIT_NM)

    joint_ids = carrier_cols
    env.robot.write_joint_stiffness_to_sim(stiffness, joint_ids=joint_ids)
    env.robot.write_joint_damping_to_sim(damping, joint_ids=joint_ids)
    env.robot.write_joint_armature_to_sim(armature, joint_ids=joint_ids)
    env.robot.write_joint_effort_limit_to_sim(effort, joint_ids=joint_ids)
    n_real_envs = int(real.any(dim=-1).sum())
    print(f"[inhand_reorient] carrier gains: {n_real_envs}/{env.num_envs} envs have a REAL "
          f"pc0/pc1 carrier and got the hand's normal gains restored; the rest keep the "
          f"stiff ghost-carrier defaults", flush=True)
