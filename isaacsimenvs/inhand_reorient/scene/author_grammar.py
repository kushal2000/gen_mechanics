"""Author a `GrammarPopulation` (or a single `EnvelopeDesign`) directly into
USD prims, one design per env, no spawner clone -- mirrors
`isaacsimenvs/pose_reaching_6d/scene_utils/assembly.py`'s
`_author_robots_into_envs` (manual per-env `Sdf.ChangeBlock` authoring
instead of Isaac Lab's regex spawner) and reuses
`hand_sampler/build.py::author_hand`'s USD-authoring conventions (body/joint
naming, ghost ` (0, 1e-8)` locked limits, no ghost collider, capsule
colliders, `physics:axis = "Z"` with the real per-joint axis baked into
`localRot0`/`localRot1`). `pxr`/`isaaclab` imports are all lazy (inside
functions), so this module is importable outside Kit (only actually CALLING
its authoring functions needs a running Kit process).

World anchoring (no arm here, unlike the old sampler, whose hand merges into
an already-fixed arm flange): follows Isaac Lab's own documented "make a
floating robot fixed" pattern (`isaaclab/test/deps/isaacsim/
check_floating_base_made_fixed.py`) -- `ArticulationRootAPI`/
`PhysxArticulationAPI` go on the PARENT prim (`root_path`, i.e. the env's
`.../Robot` Xform), and a `PhysicsFixedJoint` with only `physics:body1` set
(no `body0` => anchored to world) attaches to the design's own root/palm
rigid body, one level below.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from hand_sampler import robot_param_constants as rpc
from hand_sampler.design_space import mat_to_pos_quat
from hand_sampler.robot_spec import design_index

from hand_sampler.grammar import hand as gh

from . import grammar_envelope as ge
from . import population_file as pf

ROOT_BODY_NAME = "root"

MIMIC_GEARING = -1.0
"""PhysX mimic joints enforce `q + gearing * q_reference + offset = 0`, so a
follower carrier that turns exactly with its leader has gearing -1, offset 0
(Phase 0 of the 36-slot layout: tie error under 1.3e-3 rad with different
ties per env in one articulation view)."""


def _env_id_of(prim_path: str) -> int:
    for token in prim_path.split("/"):
        if token.startswith("env_"):
            return int(token[len("env_"):])
    raise ValueError(f"no env_<i> component in {prim_path!r}")


def _env_paths_in_order(env) -> List[str]:
    return sorted(env.scene.env_prim_paths, key=_env_id_of)


# --------------------------------------------------------------------------
# Per-slot body/joint authoring (one design)
# --------------------------------------------------------------------------


def _author_body(layer, path: str, *, mass: float, com: Sequence[float], inertia_diag: Sequence[float],
                 pos: Sequence[float], quat_wxyz: Sequence[float], filtered_pair_targets: Sequence[str] = ()) -> None:
    """A rigid body with its mass, centre of mass and diagonal inertia
    authored explicitly."""
    from pxr import Gf, Sdf

    from isaacsimenvs.pose_reaching_6d.scene_utils.sdf import attr, define, rel, set_xform

    apis = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"] + (["PhysicsFilteredPairsAPI"] if filtered_pair_targets else [])
    body = define(layer, path, "Xform", apis)
    attr(body, "physics:mass", Sdf.ValueTypeNames.Float, float(mass))
    attr(body, "physics:diagonalInertia", Sdf.ValueTypeNames.Float3, Gf.Vec3f(*[float(v) for v in inertia_diag]))
    attr(body, "physics:centerOfMass", Sdf.ValueTypeNames.Float3, Gf.Vec3f(*[float(v) for v in com]))
    set_xform(body, pos, quat_wxyz)
    if filtered_pair_targets:
        rel(body, "physics:filteredPairs", list(filtered_pair_targets))


def _author_convex_hull(layer, path: str, points: np.ndarray, *, contact_offset: Optional[float],
                        rest_offset: Optional[float], material_path: Optional[str] = None,
                        vertex_limit: int = 64) -> None:
    """A convex-hull mesh collider at `path` (points in its parent body's
    frame): the hull's triangles, PhysX `convexHull` approximation. A link's
    8 core-box corners with `rest_offset` = the corner radius give the
    rounded box; a palm plate's prism has at most 64 vertices and 34 faces
    (GPU hull limits)."""
    from pxr import Gf, Sdf, Vt
    from scipy.spatial import ConvexHull

    from isaacsimenvs.pose_reaching_6d.scene_utils.sdf import attr, define

    hull = ConvexHull(np.asarray(points, dtype=float))
    used = np.unique(hull.simplices)
    remap = {int(v): i for i, v in enumerate(used)}
    verts = hull.points[used]
    tris = [[remap[int(i)] for i in tri] for tri in hull.simplices]
    centroid = verts.mean(axis=0)
    for t in tris:
        a, b, c = verts[t[0]], verts[t[1]], verts[t[2]]
        if np.dot(np.cross(b - a, c - a), a - centroid) < 0:
            t[1], t[2] = t[2], t[1]
    apis = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI", "PhysxConvexHullCollisionAPI"]
    if material_path:
        apis.append("MaterialBindingAPI")
    mesh = define(layer, path, "Mesh", apis)
    attr(mesh, "points", Sdf.ValueTypeNames.Point3fArray,
         Vt.Vec3fArray([Gf.Vec3f(*[float(x) for x in v]) for v in verts]))
    attr(mesh, "faceVertexCounts", Sdf.ValueTypeNames.IntArray, Vt.IntArray([3] * len(tris)))
    attr(mesh, "faceVertexIndices", Sdf.ValueTypeNames.IntArray, Vt.IntArray([int(i) for t in tris for i in t]))
    attr(mesh, "physics:approximation", Sdf.ValueTypeNames.Token, "convexHull")
    attr(mesh, "physxConvexHullCollision:hullVertexLimit", Sdf.ValueTypeNames.Int, int(vertex_limit))
    if contact_offset is not None:
        attr(mesh, "physxCollision:contactOffset", Sdf.ValueTypeNames.Float, float(contact_offset))
    if rest_offset is not None:
        attr(mesh, "physxCollision:restOffset", Sdf.ValueTypeNames.Float, float(rest_offset))
    if material_path:
        binding = Sdf.RelationshipSpec(mesh, "material:binding:physics", custom=False)
        binding.targetPathList.explicitItems.append(Sdf.Path(material_path))


def _author_joint(layer, joint_path: str, *, body0_path: str, body1_path: str,
                  frame0: np.ndarray, frame1: np.ndarray, limits: Tuple[float, float],
                  prismatic: bool = False, mimic_of: Optional[str] = None, gearing: float = -1.0,
                  drive: bool = True) -> None:
    """A revolute (or prismatic) joint about (along) its local z, with the
    grammar's uniform drive; `mimic_of` (a joint path) ties it to that joint
    with a PhysX mimic joint (`q + gearing * q_ref = 0`) and no drive."""
    from pxr import Gf, Sdf

    from isaacsimenvs.pose_reaching_6d.scene_utils.sdf import attr, define, rel

    kind = "linear" if prismatic else "angular"
    apis = [f"PhysicsDriveAPI:{kind}", "PhysxJointAPI"]
    if mimic_of is not None:
        apis.append("PhysxMimicJointAPI:rotZ")
    j = define(layer, joint_path, "PhysicsPrismaticJoint" if prismatic else "PhysicsRevoluteJoint", apis)
    rel(j, "physics:body0", body0_path)
    rel(j, "physics:body1", body1_path)
    jpos, jquat = mat_to_pos_quat(frame0)
    _, j1quat = mat_to_pos_quat(frame1)
    attr(j, "physics:localPos0", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(*[float(v) for v in jpos]))
    attr(j, "physics:localRot0", Sdf.ValueTypeNames.Quatf,
         Gf.Quatf(float(jquat[0]), Gf.Vec3f(*[float(v) for v in jquat[1:]])))
    attr(j, "physics:localPos1", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(0.0, 0.0, 0.0))
    attr(j, "physics:localRot1", Sdf.ValueTypeNames.Quatf,
         Gf.Quatf(float(j1quat[0]), Gf.Vec3f(*[float(v) for v in j1quat[1:]])))
    attr(j, "physics:axis", Sdf.ValueTypeNames.Token, "Z")
    lo, hi = limits
    scale = 1.0 if prismatic else 180.0 / math.pi
    attr(j, "physics:lowerLimit", Sdf.ValueTypeNames.Float, float(lo * scale))
    attr(j, "physics:upperLimit", Sdf.ValueTypeNames.Float, float(hi * scale))
    attr(j, "physics:jointEnabled", Sdf.ValueTypeNames.Bool, True)
    attr(j, "physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool, False)
    on = drive and mimic_of is None
    attr(j, f"drive:{kind}:physics:stiffness", Sdf.ValueTypeNames.Float, float(gh.JOINT_STIFFNESS) if on else 0.0)
    attr(j, f"drive:{kind}:physics:damping", Sdf.ValueTypeNames.Float, float(gh.JOINT_DAMPING) if on else 0.0)
    attr(j, f"drive:{kind}:physics:maxForce", Sdf.ValueTypeNames.Float, float(gh.JOINT_EFFORT_NM))
    attr(j, f"drive:{kind}:physics:targetPosition", Sdf.ValueTypeNames.Float, 0.0)
    attr(j, "physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float,
         float(gh.JOINT_VELOCITY_RAD_S * (1.0 if prismatic else 180.0 / math.pi)))
    attr(j, "physxJoint:armature", Sdf.ValueTypeNames.Float, float(gh.JOINT_ARMATURE))
    if mimic_of is not None:
        rel(j, "physxMimicJoint:rotZ:referenceJoint", mimic_of)
        attr(j, "physxMimicJoint:rotZ:gearing", Sdf.ValueTypeNames.Float, float(gearing))
        attr(j, "physxMimicJoint:rotZ:offset", Sdf.ValueTypeNames.Float, 0.0)


def _author_fixed_joint(layer, joint_path: str, *, body0_path: str, body1_path: str,
                        pos0: Sequence[float]) -> None:
    """A fixed joint inside the articulation: body1's origin at `pos0` in
    body0's frame, same orientation."""
    from pxr import Gf, Sdf

    from isaacsimenvs.pose_reaching_6d.scene_utils.sdf import attr, define, rel

    j = define(layer, joint_path, "PhysicsFixedJoint")
    rel(j, "physics:body0", body0_path)
    rel(j, "physics:body1", body1_path)
    attr(j, "physics:localPos0", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(*[float(v) for v in pos0]))
    attr(j, "physics:localRot0", Sdf.ValueTypeNames.Quatf, Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    attr(j, "physics:localPos1", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(0.0, 0.0, 0.0))
    attr(j, "physics:localRot1", Sdf.ValueTypeNames.Quatf, Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))


LINK_CONTACT_EXTRA_M = 0.002
"""A link's contact offset is its corner radius plus this (`restOffset` = r)."""
CARRIER_MASS_PROPS = (0.01, (2e-6, 2e-6, 2e-6))
"""A locked or follower carrier (no collider) gets a small real mass: a
near-massless body between the palm and a real finger destabilised the
solver (a Kit diagnostic: 114 rad/s within 2 steps on a ghost carrier)."""


def author_design(layer, root_path: str, design: ge.EnvelopeDesign, *,
                  base_pos: Sequence[float] = (0.0, 0.0, 0.0),
                  base_rot_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
                  world_anchor_pos: Optional[Sequence[float]] = None,
                  contact_offset: Optional[float] = None,
                  rest_offset: Optional[float] = None) -> Dict[str, bool]:
    """Author one design under `root_path` (an env's `Robot` Xform, the
    articulation root): the root body (the palm frame, carrying the main
    palm plate), 6 finger slots of a carrier (a palm section on a leader),
    5 finger links (rounded-box hulls) and a fixed fingertip body each. Every
    body is a flat sibling under `root_path`, authored at its rest pose
    (`authored_fk` at q = 0 composed with the base pose). `world_anchor_pos`
    is the base position in stage coordinates (the fixed joint anchoring the
    root to the world reads its anchor in world space). Returns
    `{body_name: has_collider}`."""
    from pxr import Gf, Sdf

    from isaacsimenvs.pose_reaching_6d.scene_utils.author_objects import author_physics_material
    from isaacsimenvs.pose_reaching_6d.scene_utils.sdf import attr, define, rel

    define(layer, root_path, "Xform", ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"])
    define(layer, f"{root_path}/joints", "Scope")
    material = author_physics_material(layer, f"{root_path}/PhysicsMaterial", gh.FRICTION, gh.FRICTION, 0.0)
    frames = ge.joint_local_frames(design)
    colliders: Dict[str, bool] = {}
    base_pos = tuple(float(v) for v in base_pos)
    base_rot_wxyz = tuple(float(v) for v in base_rot_wxyz)
    world_anchor = tuple(float(v) for v in (world_anchor_pos if world_anchor_pos is not None else base_pos))
    link_rest = ge.LINK_RADIUS_M
    link_contact = ge.LINK_RADIUS_M + LINK_CONTACT_EXTRA_M

    T0 = ge.authored_fk(design, np.zeros(ge.N_SLOTS))
    T_base = np.eye(4)
    T_base[:3, :3] = Rotation.from_quat([base_rot_wxyz[1], base_rot_wxyz[2], base_rot_wxyz[3],
                                         base_rot_wxyz[0]]).as_matrix()
    T_base[:3, 3] = base_pos
    T_in_root = T_base @ T0

    def node_path(node: int) -> str:
        return f"{root_path}/{ROOT_BODY_NAME}" if node == ge.ROOT_NODE else f"{root_path}/{ge.slot_body(node)}"

    filtered_targets_of: Dict[str, List[str]] = {}
    for i, j in design.filtered_pairs:
        pi, pj = node_path(i), node_path(j)
        filtered_targets_of.setdefault(pi, []).append(pj)
        filtered_targets_of.setdefault(pj, []).append(pi)

    # --- the root: the palm frame, with the main palm plate --------------------
    root_body_path = node_path(ge.ROOT_NODE)
    plate = ge.plate_points(design, ge.ROOT_NODE)
    m, com, inertia = ge.plate_mass_props(plate)
    _author_body(layer, root_body_path, mass=m, com=com, inertia_diag=inertia, pos=base_pos, quat_wxyz=base_rot_wxyz,
                 filtered_pair_targets=filtered_targets_of.get(root_body_path, ()))
    define(layer, f"{root_body_path}/collisions", "Xform")
    _author_convex_hull(layer, f"{root_body_path}/collisions/palm_plate", plate, contact_offset=contact_offset,
                        rest_offset=rest_offset, material_path=material)
    colliders[ROOT_BODY_NAME] = True
    fixed = define(layer, f"{root_path}/root_fixed_joint", "PhysicsFixedJoint")
    rel(fixed, "physics:body1", root_body_path)
    attr(fixed, "physics:localPos0", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(*world_anchor))
    attr(fixed, "physics:localRot0", Sdf.ValueTypeNames.Quatf, Gf.Quatf(base_rot_wxyz[0], Gf.Vec3f(*base_rot_wxyz[1:])))
    attr(fixed, "physics:localPos1", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(0.0, 0.0, 0.0))
    attr(fixed, "physics:localRot1", Sdf.ValueTypeNames.Quatf, Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

    roles = ge.carrier_roles(design)
    tips = ge.tip_offsets(design)
    for f in range(ge.N_FINGERS):
        c = ge.carrier_slot(f)
        role = roles[f]
        body_path = node_path(c)
        pos, quat = mat_to_pos_quat(T_in_root[c])
        if role == ge.LEADER:
            pts = ge.plate_points(design, c)
            m, com, inertia = ge.plate_mass_props(pts)
        else:
            m, (com, inertia) = CARRIER_MASS_PROPS[0], ((0.0, 0.0, 0.0), CARRIER_MASS_PROPS[1])
        _author_body(layer, body_path, mass=m, com=com, inertia_diag=inertia, pos=pos, quat_wxyz=quat,
                     filtered_pair_targets=filtered_targets_of.get(body_path, ()))
        if role == ge.LEADER:
            define(layer, f"{body_path}/collisions", "Xform")
            _author_convex_hull(layer, f"{body_path}/collisions/palm_plate", pts, contact_offset=contact_offset,
                                rest_offset=rest_offset, material_path=material)
        colliders[ge.slot_body(c)] = role == ge.LEADER
        limits = ge.GHOST_LIMITS if role == ge.LOCKED else tuple(float(v) for v in design.slot_limits[c])
        mimic_of = f"{root_path}/joints/{ge.SLOT_NAMES[int(design.slot_tie[c])]}" if role == ge.FOLLOWER else None
        _author_joint(layer, f"{root_path}/joints/{ge.SLOT_NAMES[c]}", body0_path=root_body_path, body1_path=body_path,
                      frame0=frames[c, 0], frame1=frames[c, 1], limits=limits, mimic_of=mimic_of, gearing=-1.0,
                      drive=role == ge.LEADER)

        for d in range(ge.N_JOINTS_PER_FINGER):
            slot = ge.finger_slot(f, d)
            real = bool(design.slot_real[slot])
            body_path = node_path(slot)
            pos, quat = mat_to_pos_quat(T_in_root[slot])
            if real:
                m, com, inertia = ge.link_mass_props(design, slot)
            else:
                m, com, inertia = rpc.VIRTUAL_LINK_MASS_KG, (0.0, 0.0, 0.0), (rpc.VIRTUAL_LINK_INERTIA,) * 3
            _author_body(layer, body_path, mass=m, com=com, inertia_diag=inertia, pos=pos, quat_wxyz=quat,
                         filtered_pair_targets=filtered_targets_of.get(body_path, ()))
            if real:
                define(layer, f"{body_path}/collisions", "Xform")
                _author_convex_hull(layer, f"{body_path}/collisions/rounded_box", ge.link_core_points(design, slot),
                                    contact_offset=link_contact, rest_offset=link_rest, material_path=material)
            colliders[ge.slot_body(slot)] = real
            tie = int(design.slot_tie[slot])
            _author_joint(
                layer, f"{root_path}/joints/{ge.SLOT_NAMES[slot]}", body0_path=node_path(ge.SLOT_PARENT[slot]),
                body1_path=body_path, frame0=frames[slot, 0], frame1=frames[slot, 1],
                limits=tuple(float(v) for v in design.slot_limits[slot]) if real else ge.GHOST_LIMITS,
                prismatic=bool(design.slot_prismatic[slot]),
                mimic_of=f"{root_path}/joints/{ge.SLOT_NAMES[tie]}" if real and tie >= 0 else None,
                gearing=-float(design.slot_gear[slot]), drive=real)

        last = ge.LAST_FINGER_SLOTS[f]
        tip_path = f"{root_path}/{ge.tip_body(f)}"
        offset = np.eye(4)
        offset[0, 3] = float(tips[f])
        pos, quat = mat_to_pos_quat(T_in_root[last] @ offset)
        _author_body(layer, tip_path, mass=rpc.VIRTUAL_LINK_MASS_KG, com=(0.0, 0.0, 0.0),
                     inertia_diag=(rpc.VIRTUAL_LINK_INERTIA,) * 3, pos=pos, quat_wxyz=quat)
        colliders[ge.tip_body(f)] = False
        _author_fixed_joint(layer, f"{root_path}/joints/{ge.tip_body(f)}_fixed", body0_path=node_path(last),
                            body1_path=tip_path, pos0=(float(tips[f]), 0.0, 0.0))
    return colliders


# --------------------------------------------------------------------------
# Population authoring
# --------------------------------------------------------------------------


HAND_BASE_POS_M: Tuple[float, float, float] = (0.0, 0.0, 0.5)
"""Same convention as `hand_only.HandOnlySpec.base_pos`: a fixed height
above the ground plane, identical for every env/design (the design only
varies `base_rot`, from its own `palm_up` calibration)."""


def author_population(env, population: ge.GrammarPopulation, design_idx: np.ndarray,
                      base_pos_by_design: Optional[np.ndarray] = None) -> Dict[int, Dict[str, bool]]:
    """One `Robot` prim per env, authored from `population`, `design_idx[env_id]`
    picking which design. Hard-asserts `replicate_physics=False` and
    `clone_in_fabric=False` (the caller must have set these; this function
    only asserts, per the design note's env-interface contract)."""
    assert env.cfg.scene.replicate_physics is False, "grammar populations require replicate_physics=False"
    assert env.cfg.scene.clone_in_fabric is False, "grammar populations require clone_in_fabric=False"

    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Sdf

    layer = get_current_stage().GetRootLayer()
    collider_links: Dict[int, Dict[str, bool]] = {}
    env_paths = _env_paths_in_order(env)
    assert len(env_paths) == env.num_envs, (
        f"expected {env.num_envs} env prim paths, found {len(env_paths)}"
    )
    # The fixed joint's world anchor (`author_design`'s `world_anchor_pos`)
    # is in GLOBAL stage coordinates, not env-local -- each env sits at its
    # own `env_origins[env_id]` on the grid cloner's layout (nonzero for
    # every env but the one at the stage origin).
    env_origins = env.scene.env_origins.detach().cpu().numpy()
    base_pos_np = np.asarray(HAND_BASE_POS_M, dtype=float)
    # Review item 11: give population robot colliders the SAME contact/rest
    # offset as the single-hand path's own (`scene_utils._setup_scene_
    # single_hand`'s `_convert_fixed_robot`, which threads these same
    # `env.cfg.physics.*` values through `author_robot.flatten_robot_usd`),
    # instead of silently falling back to PhysX's own per-shape defaults.
    contact_offset = float(env.cfg.physics.contact_offset)
    rest_offset = float(env.cfg.physics.rest_offset)
    if base_pos_by_design is None:
        base_pos_by_design = np.tile(np.asarray(HAND_BASE_POS_M, dtype=float), (population.n_designs, 1))

    with Sdf.ChangeBlock():
        for env_path in env_paths:
            env_id = _env_id_of(env_path)
            idx = int(design_idx[env_id])
            root_path = f"{env_path}/Robot"
            design = population.designs[idx]
            base_rot = tuple(float(v) for v in population.base_rot[idx])
            base_pos = tuple(float(v) for v in base_pos_by_design[idx])
            world_anchor = tuple(float(v) for v in (env_origins[env_id] + np.asarray(base_pos)))
            authored = author_design(
                layer, root_path, design, base_pos=base_pos, base_rot_wxyz=base_rot,
                world_anchor_pos=world_anchor, contact_offset=contact_offset, rest_offset=rest_offset,
            )
            collider_links.setdefault(idx, authored)

    return collider_links


def _adjacent_links_template() -> Dict[str, List[str]]:
    """Immediate parent/child body-name adjacency over the FIXED envelope
    topology (same for every design; see `grammar_envelope.SLOT_PARENT`),
    fingertip bodies included -- enough for `HandOnlySpec.validate()`'s
    "self-collision would be unfiltered" check. Per-design pairs beyond
    parent and child are collision-filtered at authoring time
    (`EnvelopeDesign.filtered_pairs`)."""
    adjacency: Dict[str, List[str]] = {ROOT_BODY_NAME: []}

    def _link(a: str, b: str) -> None:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    for slot in range(ge.N_SLOTS):
        parent = ge.SLOT_PARENT[slot]
        _link(ge.slot_body(slot), ROOT_BODY_NAME if parent == ge.ROOT_SENTINEL else ge.slot_body(parent))
    for f in range(ge.N_FINGERS):
        _link(ge.tip_body(f), ge.slot_body(ge.LAST_FINGER_SLOTS[f]))
    return adjacency


def build_hand_population_spec(population: ge.GrammarPopulation, template_idx: int, base_rot: Sequence[float]):
    """A `hand_only.HandOnlySpec`-shaped template describing the FIXED
    36-joint envelope (same joint/body names and action-space size for
    every design) -- lets `obs_utils.derive_spaces`/`reward_utils`/
    `reset_utils` (all spec-generic, keyed by field name) run unmodified for
    the population path. Per-design specifics (limits, validity, default
    pose) live in `env.hand_tables` (the `GrammarPopulation`), not here --
    `hand_joint_limits`/`hand_default_joint_pos` below are only
    `template_idx`'s own values, a representative default, NOT authoritative
    (the actually-authored PhysX joint limits differ per env; see the
    module docstring and the Phase 2 report's known gaps around per-env
    ghost-joint masking in reward/obs code, which is not yet wired)."""
    from .. import hand_only  # local import: hand_only pulls in the same package's __init__ chain

    stiffness = {n: gh.JOINT_STIFFNESS for n in ge.SLOT_NAMES}
    damping = {n: gh.JOINT_DAMPING for n in ge.SLOT_NAMES}
    armature = {n: gh.JOINT_ARMATURE for n in ge.SLOT_NAMES}
    default_pos = dict(zip(ge.SLOT_NAMES, (float(v) for v in population.default_joint_pos[template_idx])))
    limits = tuple(
        (float(lo), float(hi)) for lo, hi in population.joint_limits[template_idx]
    )
    fingertip_names = ge.TIP_BODY_NAMES

    return hand_only.HandOnlySpec(
        name="grammar_population", hand_name="grammar_population", urdf_path="",
        hand_root=ROOT_BODY_NAME, hand_joint_names=ge.SLOT_NAMES, hand_joint_limits=limits,
        palm_body_name=ROOT_BODY_NAME, fingertip_body_names=fingertip_names,
        hand_stiffness=stiffness, hand_damping=damping, hand_armature=armature,
        hand_default_joint_pos=default_pos, palm_center_offset=(0.0, 0.0, 0.0),
        adjacent_links=_adjacent_links_template(), base_pos=HAND_BASE_POS_M,
        base_rot=tuple(float(v) for v in base_rot),
    )


def _apply_projected_poses(env, population: ge.GrammarPopulation) -> Optional[np.ndarray]:
    """``anyrotate.population_projected_pose``: projected commercial hands
    with a ``hand_pose_file`` entry stand where their single-hand URDF asset
    does (``projected_hands.urdf_equivalent_placement``); edits
    ``population`` in place and returns the per-design base positions (None
    when off)."""
    a = getattr(env.cfg, "anyrotate", None)
    if a is None or not getattr(a, "population_projected_pose", False):
        return None
    from ..palm_calibration import load_repose_hand_poses, resolve_repose_hand_pose_path
    from .projected_hands import hand_id_of, urdf_equivalent_placement

    poses = load_repose_hand_poses(resolve_repose_hand_pose_path(a.hand_pose_file))
    base_pos = np.tile(np.asarray(HAND_BASE_POS_M, dtype=float), (population.n_designs, 1))
    for i, d in enumerate(population.designs):
        hand_id = hand_id_of(d.source)
        if hand_id is None or hand_id not in poses:
            continue
        pl = urdf_equivalent_placement(d, hand_id, poses[hand_id])
        population.base_rot[i] = np.asarray(pl["base_rot_wxyz"])
        population.spawn_offset[i] = np.asarray(pl["spawn_offset"])
        population.default_joint_pos[i] = pl["default_q"]
        base_pos[i] = np.asarray(pl["base_pos"])
        print(f"[inhand_reorient] {d.source}: URDF-equivalent pose base_pos={pl['base_pos']} "
              f"base_rot={tuple(round(v, 6) for v in pl['base_rot_wxyz'])} spawn={pl['spawn_offset']}", flush=True)
    return base_pos


def setup_grammar_robot(env) -> None:
    """Env-hook entry point: load `env.cfg.assets.hand_population`, author it
    into every env, and set `env.hand_spec`/`env.scene_record`/
    `env.hand_tables` (design note's "Env interface needed"; `env.
    palm_body_idx` is resolved later by `scene_utils.finalize_scene`, once
    the articulation view is live, same as the single-hand path). Called
    from `scene_utils.setup_scene` instead of the single-hand path when
    `hand_population` is non-empty."""
    import torch

    assert env.cfg.scene.replicate_physics is False, "hand_population requires scene.replicate_physics=False"
    assert env.cfg.scene.clone_in_fabric is False, "hand_population requires scene.clone_in_fabric=False"

    designs = pf.load_population(env.cfg.assets.hand_population)
    population = ge.build_population(designs)
    print(f"[inhand_reorient] loaded grammar population: {population.n_designs} designs from "
          f"{env.cfg.assets.hand_population}", flush=True)

    base_pos_by_design = _apply_projected_poses(env, population)
    idx = design_index(
        env.num_envs, population.n_designs,
        rank=int(os.environ.get("RANK", "0")), world_size=int(os.environ.get("WORLD_SIZE", "1")),
    )
    # Review item 10 (risk): a previous version of this function swapped
    # env 0's design assignment so a "limits contain 0" design landed there
    # (Isaac Lab validates `InitialStateCfg.joint_pos` at construction time
    # using ONE scene-wide template, built from THIS index below). Removed:
    # it could silently DROP a design from the population outright (when the
    # chosen candidate design held zero envs, `out[0] = candidate` overwrote
    # env 0's previous design with no swap-back, so that previous design's
    # only representative env vanished with no error) for a benefit that no
    # longer exists -- `scene_utils._resolve_population_joint_permutation`
    # now overwrites EVERY env's own `default_joint_pos` (including env 0's)
    # from ITS OWN design right after the articulation view goes live, so
    # the scene-wide CFG template's own limits/defaults (always mutually
    # consistent, since both come from the SAME `template_idx`) no longer
    # need env 0 to be any particular design.
    collider_links = author_population(env, population, idx, base_pos_by_design)

    env.hand_spec = build_hand_population_spec(population, int(idx[0]), population.base_rot[int(idx[0])])
    env.hand_tables = population
    env.scene_record = {
        "population": population, "design_idx": torch.as_tensor(idx, device=env.device),
        "collider_links": collider_links, "population_path": env.cfg.assets.hand_population,
    }
