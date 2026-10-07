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

from . import grammar_envelope as ge
from . import population_file as pf

ROOT_BODY_NAME = "root"
PC_BODY_NAMES = ("pc0", "pc1")


def _finger_body_name(f: int, d: int) -> str:
    return f"f{f}_link{d}"


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


def _author_body_and_collider(layer, path: str, *, length: float, radius: float, mass: float,
                               inertia_diag: Tuple[float, float, float], com_z: float,
                               pos: Sequence[float], quat_wxyz: Sequence[float],
                               real: bool, filtered_pair_targets: Sequence[str] = (),
                               contact_offset: Optional[float] = None,
                               rest_offset: Optional[float] = None) -> None:
    from pxr import Gf, Sdf

    from isaacsimenvs.pose_reaching_6d.scene_utils.sdf import attr, define, rel, set_xform

    apis = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    if filtered_pair_targets:
        # Review item 4's exemption: this body has a rest-pose overlap with
        # one it is not kinematically adjacent to (a PROJECTED commercial
        # hand's capsule-projection artifact -- see
        # `grammar_envelope.EnvelopeDesign.filtered_pairs`) -- collision-
        # filter it explicitly instead of letting PhysX's depenetration
        # impulse resolve a real interpenetration (which blows up the
        # ghost/carrier joints on step 0 for a SAMPLED design with the same
        # symptom; sampled designs are instead REJECTED on this, never
        # exempted, so they never reach this branch).
        apis.append("PhysicsFilteredPairsAPI")
    body = define(layer, path, "Xform", apis)
    attr(body, "physics:mass", Sdf.ValueTypeNames.Float, float(mass))
    attr(body, "physics:diagonalInertia", Sdf.ValueTypeNames.Float3,
         Gf.Vec3f(*[float(v) for v in inertia_diag]))
    attr(body, "physics:centerOfMass", Sdf.ValueTypeNames.Float3, Gf.Vec3f(0.0, 0.0, float(com_z)))
    set_xform(body, pos, quat_wxyz)
    if filtered_pair_targets:
        rel(body, "physics:filteredPairs", list(filtered_pair_targets))

    if real:
        r = float(radius)
        define(layer, f"{path}/collisions", "Xform")
        mesh = define(layer, f"{path}/collisions/mesh_0", "Xform")
        # Capsule collider along the body's own local +z (grammar's segment
        # convention), so it is rotated 90deg off the collider prim's own
        # native +Z-is-the-capsule-axis default onto our +z... the capsule
        # geometry schema's own axis token handles this directly (below),
        # so mesh_0 needs no extra rotation -- unlike build.py's URDF-derived
        # links (which run along local +x and rotate the capsule mesh to
        # match), a grammar body already runs along +z.
        cap_apis = ["PhysicsCollisionAPI"]
        if contact_offset is not None or rest_offset is not None:
            # Parity with the single-hand path's own colliders (review
            # item 11's "robot colliders have no contact offsets or
            # friction"): the single-hand path's URDF-converted USD gets
            # these from `author_robot.flatten_robot_usd` (same
            # `PhysxCollisionAPI` attrs, same `env.cfg.physics.*` values);
            # a grammar body is authored directly here instead, so it needs
            # the same two attrs set explicitly, or it silently falls back
            # to PhysX's own per-shape defaults instead of this task's
            # configured 2mm contact / 0mm rest offset.
            cap_apis.append("PhysxCollisionAPI")
        cap = define(layer, f"{path}/collisions/mesh_0/capsule", "Capsule", cap_apis)
        attr(cap, "radius", Sdf.ValueTypeNames.Double, r)
        attr(cap, "height", Sdf.ValueTypeNames.Double, float(rpc.cylinder_part(max(length, 1e-6), r)))
        attr(cap, "axis", Sdf.ValueTypeNames.Token, "Z")
        if contact_offset is not None:
            attr(cap, "physxCollision:contactOffset", Sdf.ValueTypeNames.Float, float(contact_offset))
        if rest_offset is not None:
            attr(cap, "physxCollision:restOffset", Sdf.ValueTypeNames.Float, float(rest_offset))
        set_xform(mesh, (0.0, 0.0, length / 2.0), (1.0, 0.0, 0.0, 0.0))


def _author_convex_hull(layer, path: str, points: np.ndarray, *, contact_offset: Optional[float],
                        rest_offset: Optional[float], vertex_limit: int = 64) -> None:
    """A convex-hull mesh collider at ``path`` (points in its parent body's
    frame): the hull's triangles, PhysX ``convexHull`` approximation."""
    from pxr import Gf, Sdf, Vt
    from scipy.spatial import ConvexHull

    from isaacsimenvs.pose_reaching_6d.scene_utils.sdf import attr, define

    hull = ConvexHull(np.asarray(points, dtype=float))
    used = np.unique(hull.simplices)
    remap = {int(v): i for i, v in enumerate(used)}
    verts = hull.points[used]
    tris = [[remap[int(i)] for i in tri] for tri in hull.simplices]
    # Outward winding: flip any triangle whose normal points to the centroid.
    centroid = verts.mean(axis=0)
    for t in tris:
        a, b, c = verts[t[0]], verts[t[1]], verts[t[2]]
        if np.dot(np.cross(b - a, c - a), a - centroid) < 0:
            t[1], t[2] = t[2], t[1]
    apis = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI", "PhysxConvexHullCollisionAPI"]
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


def _author_joint(layer, joint_path: str, *, body0_path: str, body1_path: str,
                   frame0: np.ndarray, frame1: np.ndarray, limits_rad: Tuple[float, float],
                   valid: bool) -> None:
    from pxr import Gf, Sdf

    from isaacsimenvs.pose_reaching_6d.scene_utils.sdf import attr, define, rel

    j = define(layer, joint_path, "PhysicsRevoluteJoint", ["PhysicsDriveAPI:angular", "PhysxJointAPI"])
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
    lo, hi = limits_rad
    attr(j, "physics:lowerLimit", Sdf.ValueTypeNames.Float, float(math.degrees(lo)))
    attr(j, "physics:upperLimit", Sdf.ValueTypeNames.Float, float(math.degrees(hi)))
    attr(j, "physics:jointEnabled", Sdf.ValueTypeNames.Bool, True)
    attr(j, "physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool, False)
    attr(j, "drive:angular:physics:stiffness", Sdf.ValueTypeNames.Float, rpc.CONVERTER_DRIVE_STIFFNESS)
    attr(j, "drive:angular:physics:damping", Sdf.ValueTypeNames.Float, rpc.CONVERTER_DRIVE_DAMPING)
    attr(j, "drive:angular:physics:maxForce", Sdf.ValueTypeNames.Float, float(rpc.GEN_JOINT_EFFORT_NM))
    attr(j, "drive:angular:physics:targetPosition", Sdf.ValueTypeNames.Float, 0.0)
    attr(j, "physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float,
         float(math.degrees(rpc.GEN_JOINT_VELOCITY_RAD_S)))


def _link_mass_props(length: float, radius: float, real: bool) -> Tuple[float, Tuple[float, float, float]]:
    """Mass and diagonal inertia of a real link: a solid cylinder of the
    link's length, floored at a solid sphere of the link's radius. Without
    the floor a 0 mm link (two joints at one point) would get about 1e-6 kg
    and 5e-11 kg m^2 between two real links, a PhysX stability risk. The
    floor only acts on links shorter than 4/3 of the radius (16 mm at the
    largest sampled radius, 12 mm)."""
    if not real:
        return rpc.VIRTUAL_LINK_MASS_KG, (rpc.VIRTUAL_LINK_INERTIA,) * 3
    mass = max(math.pi * radius * radius * length * rpc.GEN_LINK_DENSITY_KG_M3, 1e-6)
    izz = 0.5 * mass * radius * radius
    ixx = iyy = mass * (3.0 * radius * radius + length * length) / 12.0
    sphere = 4.0 / 3.0 * math.pi * radius ** 3 * rpc.GEN_LINK_DENSITY_KG_M3
    if sphere > mass:
        i_sphere = 0.4 * sphere * radius * radius
        mass, ixx, iyy, izz = sphere, max(ixx, i_sphere), max(iyy, i_sphere), max(izz, i_sphere)
    return mass, (ixx, iyy, izz)


def author_design(layer, root_path: str, design: ge.EnvelopeDesign, *,
                   base_pos: Sequence[float] = (0.0, 0.0, 0.0),
                   base_rot_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
                   world_anchor_pos: Optional[Sequence[float]] = None,
                   contact_offset: Optional[float] = None,
                   rest_offset: Optional[float] = None,
                   collider_radius: Optional[float] = None,
                   palm_hull_points: Optional[np.ndarray] = None,
                   palm_filter_slots: Sequence[int] = ()) -> Dict[str, bool]:
    """Author one design's root/palm, palm carriers and 30 finger-joint
    slots under `root_path` (an already-`define`-d Xform). `base_pos`/
    `base_rot_wxyz` place the design's root BODY relative to `root_path`
    (its own env's Xform -- USD composes this with the env's own origin
    automatically, so this stays env-LOCAL). `world_anchor_pos` is the SAME
    point but in GLOBAL STAGE coordinates (defaults to `base_pos` unchanged,
    correct only for an env whose own origin is the stage origin): the fixed
    joint anchoring root to world (`body0` unset) reads its own
    `localPos0`/`localRot0` as an ABSOLUTE world-frame anchor, INDEPENDENT
    of any USD xformOp hierarchy above it and thus INDEPENDENT of the env's
    own grid offset -- confirmed by two Kit smokes: leaving `localPos0` at
    its identity default fought a world-origin anchor against the authored
    pose every step (FK error up to 0.24 m, ghost joints spun to 150+ rad);
    setting it to `base_pos` alone (no env-origin offset) reproduced the
    IDENTICAL error, because most sampled envs sit away from the stage
    origin on the grid cloner's layout. The caller (`author_population`)
    passes `world_anchor_pos = env_origin + base_pos`. `collider_radius`
    (> 0) replaces the design's capsule radius in every collider (masses
    keep the design's); `palm_hull_points` (root frame) adds a convex palm
    collider to the root body. Returns
    `{body_name: has_collider}` for every authored body -- a friction pass
    may use it (see `AssetsCfg.modify_asset_frictions`, not yet wired for
    populations here; see the Phase 2 report's known gaps)."""
    from pxr import Gf, Sdf

    from isaacsimenvs.pose_reaching_6d.scene_utils.sdf import attr, define, rel

    # `root_path` (e.g. ".../Robot") is the ArticulationRoot Xform -- the
    # PARENT of the design's own root/palm rigid body, one level below (see
    # module docstring's "World anchoring" note). Left at IDENTITY: the
    # design's world placement lives entirely on the root BODY + fixed
    # joint below, not on this parent xform.
    define(layer, root_path, "Xform", ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"])
    define(layer, f"{root_path}/joints", "Scope")
    frames = ge.joint_local_frames(design)
    colliders: Dict[str, bool] = {}
    base_pos = tuple(float(v) for v in base_pos)
    base_rot_wxyz = tuple(float(v) for v in base_rot_wxyz)
    world_anchor = tuple(float(v) for v in (world_anchor_pos if world_anchor_pos is not None else base_pos))

    # Every body is authored as a FLAT SIBLING directly under `root_path`
    # (not nested under its own kinematic parent body), so each one's own
    # initial xform must be its ABSOLUTE (root_path-relative) rest pose, NOT
    # `design.slot_origin[slot]` (which is relative to the slot's KINEMATIC
    # parent body -- only equal to the root_path-relative pose for a d==0
    # finger slot mounted directly on root; wrong for every continuation
    # joint d>=1, and for a carrier-mounted finger's own d==0 slot, whose
    # origin is relative to pc0/pc1, not root_path). Confirmed by a Kit
    # diagnostic: d==0 root-mounted slots read back with ~0 error, every
    # continuation slot (d>=1) was off starting at its own joint, compounding
    # down the chain. `T0[slot]` (`authored_fk` at q=0, ROOT-relative by
    # construction) composed with the design's own base transform gives the
    # correct root_path-relative pose for every slot uniformly.
    T0 = ge.authored_fk(design, np.zeros(ge.N_SLOTS))
    T_base = np.eye(4)
    T_base[:3, :3] = Rotation.from_quat(
        [base_rot_wxyz[1], base_rot_wxyz[2], base_rot_wxyz[3], base_rot_wxyz[0]]
    ).as_matrix()
    T_base[:3, 3] = base_pos
    T_slot_in_root_path = T_base @ T0  # (32,4,4), broadcasting T_base over all 32 slots

    def _slot_pos_quat(slot: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        return mat_to_pos_quat(T_slot_in_root_path[slot])

    def _node_body_path(node: int) -> str:
        """`grammar_envelope.rest_overlap_pairs`/`filtered_pairs` node index
        (a joint slot, or `ge.ROOT_NODE` for the root/palm capsule) -> this
        design's own authored body path."""
        if node == ge.ROOT_NODE:
            return f"{root_path}/{ROOT_BODY_NAME}"
        if node == ge.PC0_SLOT:
            return f"{root_path}/{PC_BODY_NAMES[0]}"
        if node == ge.PC1_SLOT:
            return f"{root_path}/{PC_BODY_NAMES[1]}"
        f, d = divmod(node, ge.N_JOINTS_PER_FINGER)
        return f"{root_path}/{_finger_body_name(f, d)}"

    # Review item 4's exemption (projected commercial hands only -- see
    # `_author_body_and_collider`'s own comment): which authored body path
    # must collision-filter which other authored body path(s), from this
    # design's `filtered_pairs` (empty for every sampled design).
    filtered_targets_of: Dict[str, List[str]] = {}
    for i, j in design.filtered_pairs:
        pi, pj = _node_body_path(i), _node_body_path(j)
        filtered_targets_of.setdefault(pi, []).append(pj)
        filtered_targets_of.setdefault(pj, []).append(pi)
    for s in palm_filter_slots:  # the palm hull ignores each finger's first links
        pi, pj = _node_body_path(ge.ROOT_NODE), _node_body_path(int(s))
        if pj not in filtered_targets_of.get(pi, []):
            filtered_targets_of.setdefault(pi, []).append(pj)
            filtered_targets_of.setdefault(pj, []).append(pi)

    # --- root/palm body: the design's own fixed-base anchor, authored
    # DIRECTLY at its world pose (base_pos/base_rot_wxyz) -------------------
    root_body_path = f"{root_path}/{ROOT_BODY_NAME}"
    root_mass = max(math.pi * design.capsule_radius_m ** 2 * max(design.root_length_m, 1e-6)
                     * rpc.GEN_PALM_DENSITY_KG_M3, 1e-6)
    col_r = float(collider_radius) if collider_radius is not None and collider_radius > 0 else design.capsule_radius_m
    _author_body_and_collider(
        layer, root_body_path, length=design.root_length_m, radius=col_r,
        mass=root_mass,
        inertia_diag=_link_mass_props(design.root_length_m, design.capsule_radius_m, True)[1],
        com_z=design.root_length_m / 2.0, pos=base_pos, quat_wxyz=base_rot_wxyz, real=True,
        filtered_pair_targets=filtered_targets_of.get(root_body_path, ()),
        contact_offset=contact_offset, rest_offset=rest_offset,
    )
    if palm_hull_points is not None and len(palm_hull_points) >= 4:
        _author_convex_hull(layer, f"{root_body_path}/collisions/palm_hull", palm_hull_points,
                            contact_offset=contact_offset, rest_offset=rest_offset)
    colliders[ROOT_BODY_NAME] = True

    # A `PhysicsFixedJoint` (body1 = root, body0 UNSET = world) anchors the
    # root rigid body to world. `localPos1`/`localRot1` (body1 = root's own
    # frame) stay identity -- root's own origin. `localPos0`/`localRot0`
    # (the WORLD anchor, since body0 is unset) are set to the SAME
    # base_pos/base_rot_wxyz as the body itself, so the constraint holds
    # root exactly where it was authored, not at the world origin.
    fixed = define(layer, f"{root_path}/root_fixed_joint", "PhysicsFixedJoint")
    rel(fixed, "physics:body1", root_body_path)
    attr(fixed, "physics:localPos0", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(*world_anchor))
    attr(fixed, "physics:localRot0", Sdf.ValueTypeNames.Quatf,
         Gf.Quatf(base_rot_wxyz[0], Gf.Vec3f(*base_rot_wxyz[1:])))
    attr(fixed, "physics:localPos1", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(0.0, 0.0, 0.0))
    attr(fixed, "physics:localRot1", Sdf.ValueTypeNames.Quatf, Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

    # --- palm-carrier bodies + joints (pc0_j, pc1_j) -----------------------
    # A GHOST pc body attaches DIRECTLY to root, but -- unlike the old
    # sampler's ghosts, which only ever carry OTHER ghosts at a finger's
    # unused tail -- a padding-only pc0/pc1 (no real carrier in the source
    # design) still carries an entire REAL finger chain when a root-mounted
    # digit fills envelope slot 3/4 (`canonicalize`'s free-slot assignment).
    # The mass ratio that actually destabilises the solver is ghost-vs-that-
    # REAL-FINGER, not ghost-vs-root: the coordinator's diagnosis, confirmed
    # by a Kit diagnostic (pc1_j on a G_SERIAL design, carrying finger 4,
    # reached 114 rad/s within 2 steps) -- and unchanged by an earlier fix
    # that only scaled the ghost mass against ROOT (1% of root was still
    # negligible next to the real finger 4 chain hanging off it). Fixed by
    # giving the ghost a REAL phalanx's mass/inertia (same density formula,
    # using this SPECIFIC finger's own first real segment length as the
    # reference -- "realistic", not "negligible", exactly because it is
    # mechanically carrying that finger).
    for pc in range(2):
        slot = ge.PC0_SLOT + pc
        body_path = f"{root_path}/{PC_BODY_NAMES[pc]}"
        valid = bool(design.slot_valid[slot])
        length = float(design.slot_length[slot])
        if valid:
            mass, inertia = _link_mass_props(length, design.capsule_radius_m, True)
        else:
            finger_f = 3 + pc
            ref_length = float(design.slot_length[finger_f * ge.N_JOINTS_PER_FINGER])
            if not design.slot_valid[finger_f * ge.N_JOINTS_PER_FINGER]:
                ref_length = 0.0  # that finger slot is itself unused padding: no load to buffer
            mass, inertia = _link_mass_props(max(ref_length, ge.GHOST_LENGTH_M), design.capsule_radius_m, True)
        pos, quat = _slot_pos_quat(slot)
        _author_body_and_collider(
            layer, body_path, length=length, radius=col_r, mass=mass,
            inertia_diag=inertia, com_z=length / 2.0, pos=pos, quat_wxyz=quat, real=valid,
            filtered_pair_targets=filtered_targets_of.get(body_path, ()),
            contact_offset=contact_offset, rest_offset=rest_offset,
        )
        colliders[PC_BODY_NAMES[pc]] = valid
        limits = tuple(float(v) for v in design.slot_limits[slot]) if valid else ge.GHOST_LIMITS
        _author_joint(
            layer, f"{root_path}/joints/{ge.SLOT_NAMES[slot]}", body0_path=root_body_path,
            body1_path=body_path, frame0=frames[slot, 0], frame1=frames[slot, 1],
            limits_rad=limits, valid=valid,
        )

    # --- finger joint slots (f0_j0..f4_j5) ----------------------------------
    for f in range(ge.N_FINGERS):
        for d in range(ge.N_JOINTS_PER_FINGER):
            slot = f * ge.N_JOINTS_PER_FINGER + d
            valid = bool(design.slot_valid[slot])
            length = float(design.slot_length[slot])
            body_path = f"{root_path}/{_finger_body_name(f, d)}"
            mass, inertia = _link_mass_props(length, design.capsule_radius_m, valid)
            pos, quat = _slot_pos_quat(slot)
            _author_body_and_collider(
                layer, body_path, length=length, radius=col_r, mass=mass,
                inertia_diag=inertia, com_z=length / 2.0, pos=pos, quat_wxyz=quat, real=valid,
                filtered_pair_targets=filtered_targets_of.get(body_path, ()),
                contact_offset=contact_offset, rest_offset=rest_offset,
            )
            colliders[_finger_body_name(f, d)] = valid

            if d == 0:
                body0_path = root_body_path if f in (0, 1, 2) else f"{root_path}/{PC_BODY_NAMES[f - 3]}"
            else:
                body0_path = f"{root_path}/{_finger_body_name(f, d - 1)}"
            limits = tuple(float(v) for v in design.slot_limits[slot]) if valid else ge.GHOST_LIMITS
            _author_joint(
                layer, f"{root_path}/joints/{ge.SLOT_NAMES[slot]}", body0_path=body0_path,
                body1_path=body_path, frame0=frames[slot, 0], frame1=frames[slot, 1],
                limits_rad=limits, valid=valid,
            )

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
    a = getattr(env.cfg, "anyrotate", None)
    radius_override = float(getattr(a, "population_capsule_radius", -1.0)) if a is not None else -1.0
    palm_mode = getattr(a, "population_palm_collider", "capsule") if a is not None else "capsule"
    if palm_mode not in ("capsule", "mount_hull", "mount_hull_filtered"):
        raise ValueError(f"anyrotate.population_palm_collider={palm_mode!r}; expected capsule, mount_hull or "
                         f"mount_hull_filtered")
    hull_by_design: Dict[int, np.ndarray] = {}
    palm_filter_by_design: Dict[int, List[int]] = {}
    if palm_mode in ("mount_hull", "mount_hull_filtered"):
        from .projected_hands import palm_filter_slots, palm_hull_points

        for i, d in enumerate(population.designs):
            r = radius_override if radius_override > 0 else d.capsule_radius_m
            hull_by_design[i] = palm_hull_points(d, r)
            if palm_mode == "mount_hull_filtered":
                palm_filter_by_design[i] = palm_filter_slots(d)
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
                collider_radius=radius_override if radius_override > 0 else None,
                palm_hull_points=hull_by_design.get(idx), palm_filter_slots=palm_filter_by_design.get(idx, ()),
            )
            collider_links.setdefault(idx, authored)

    return collider_links


def _adjacent_links_template() -> Dict[str, List[str]]:
    """Immediate parent/child body-name adjacency over the FIXED envelope
    topology (same for every design; see `grammar_envelope.SLOT_PARENT`) --
    enough for `HandOnlySpec.validate()`'s "self-collision would be
    unfiltered" check. Real per-design self-collision filtering between
    non-adjacent bodies is not attempted here (see the Phase 2 report's
    known gaps)."""
    def _body_name(slot: int) -> str:
        if slot == ge.PC0_SLOT:
            return PC_BODY_NAMES[0]
        if slot == ge.PC1_SLOT:
            return PC_BODY_NAMES[1]
        f, d = divmod(slot, ge.N_JOINTS_PER_FINGER)
        return _finger_body_name(f, d)

    adjacency: Dict[str, List[str]] = {ROOT_BODY_NAME: []}
    for slot in range(ge.N_SLOTS):
        name = _body_name(slot)
        parent = ge.SLOT_PARENT[slot]
        parent_name = ROOT_BODY_NAME if parent == ge.ROOT_SENTINEL else _body_name(parent)
        adjacency.setdefault(name, []).append(parent_name)
        adjacency.setdefault(parent_name, []).append(name)
    return adjacency


def build_hand_population_spec(population: ge.GrammarPopulation, template_idx: int, base_rot: Sequence[float]):
    """A `hand_only.HandOnlySpec`-shaped template describing the FIXED
    32-joint envelope (same joint/body names and action-space size for
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

    stiffness = {n: hand_only.DEFAULT_HAND_STIFFNESS for n in ge.SLOT_NAMES}
    damping = {n: hand_only.DEFAULT_HAND_DAMPING for n in ge.SLOT_NAMES}
    armature = {n: hand_only.DEFAULT_HAND_ARMATURE for n in ge.SLOT_NAMES}
    default_pos = dict(zip(ge.SLOT_NAMES, (float(v) for v in population.default_joint_pos[template_idx])))
    limits = tuple(
        (float(lo), float(hi)) for lo, hi in population.joint_limits[template_idx]
    )
    fingertip_names = tuple(_finger_body_name(f, ge.N_JOINTS_PER_FINGER - 1) for f in range(ge.N_FINGERS))

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
