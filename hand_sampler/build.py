"""Author a ``Hand`` into USD prims: the tree's path into a simulator.

Generated designs have no URDF and never will, so nothing converts for them --
this writes the bodies, capsule colliders and revolute joints directly into a
layer, which is also the only way to reach 24k designs (spawning through Isaac
Lab copies the prim tree per env: ~110 s for one hand, hours for a population).

Every design is authored into the SAME envelope, ``MAX_FINGERS`` x
``MAX_JOINTS_PER_FINGER``, because one scene is one articulation with one DOF
count. Slots a design does not use get a zero-length link and a joint locked at
its rest angle, so ``joint_enabled`` reads 0 and the policy's key mask drops the
token. Names match ``population_spec``: body ``f{f}_link{d}``, joint
``f{f}_j{d}``.

``link_frames`` and ``link_mass_props`` are pure numpy and carry the geometry;
``author_hand`` needs ``pxr`` and can only run inside Kit.
"""

from __future__ import annotations

import math

import numpy as np

from hand_sampler import design_space
from hand_sampler import robot_param_constants as rpc

GHOST_LENGTH_M: float = 1e-4
"""A ghost link still needs a body; zero length gives PhysX a degenerate inertia
tensor, so it gets the smallest length that is not zero. It carries no collider
and the virtual-link mass, not a mass derived from this length."""


def flange_to_palm() -> np.ndarray:
    """``link_7 -> palm``, the transform merge_fixed_joints collapses."""
    m = design_space.rpy_to_mat((0.0, 0.0, rpc.FLANGE_TO_PALM_YAW_RAD))
    out = np.eye(4)
    out[:3, :3] = m
    out[:3, 3] = (0.0, 0.0, rpc.LINK7_TO_FLANGE_Z_M + rpc.FLANGE_TO_PALM_Z_M)
    return out


def hinge_to_z(axis) -> np.ndarray:
    """3x3 taking +z onto ``axis``: the joint frame a USD revolute hinge needs.

    ``physics:axis`` is a token, so an arbitrary hinge cannot be named there --
    it has to be rotated into one of the axes the token allows. WHICH rotation
    does not matter: both of a joint's frames get this same one, so they still
    coincide at rest and the zero angle is unmoved. Only the third column is
    load-bearing.

    Built from an orthonormal frame rather than the shortest arc, which is
    ill-conditioned for a hinge near -z -- and ``wrap_theta`` makes that a
    reachable design, since a hinge and its negation are the same joint.
    """
    a = np.asarray(axis, dtype=float)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    ref = np.eye(3)[int(np.argmin(np.abs(a)))]      # least parallel to a
    x = np.cross(ref, a)
    x /= max(float(np.linalg.norm(x)), 1e-12)
    return np.column_stack([x, np.cross(a, x), a])


def palm_center_offset(hand: design_space.Hand) -> tuple[float, float, float]:
    """Where ``palm_pos`` should be measured, in ``iiwa14_link_7``'s frame.

    The palm merges into the arm's last link, so the observation needs the walk
    from that link out to the middle of the palm slab. Derived HERE because the
    flange->palm transform is chosen here: computing it anywhere else lets the
    two disagree, and the observation would silently report a palm that is not
    where the hand is.
    """
    centre = np.append(np.asarray(design_space.palm_center(hand.palm), float), 1.0)
    return tuple(float(v) for v in (flange_to_palm() @ centre)[:3])


def palm_box(hand: design_space.Hand) -> tuple[tuple, np.ndarray]:
    """``(extents, pose)`` of the design's palm slab, in ``iiwa14_link_7``'s frame.

    The palm is a box the design names and the simulator never had: nothing
    authored one, so ``hand.palm`` decided where fingers mounted and nothing
    else. An object passed straight through it, the hand flew 0.72 kg light,
    and ``palm_center_offset`` pointed the policy at a centre with no body at it.
    """
    pose = flange_to_palm().copy()
    centre = np.append(np.asarray(design_space.palm_center(hand.palm), float), 1.0)
    pose[:3, 3] = (flange_to_palm() @ centre)[:3]
    return tuple(float(v) for v in hand.palm.extents), pose


def palm_mass_props(hand: design_space.Hand) -> tuple[float, np.ndarray]:
    """``(mass, 3x3 inertia about the palm's own centre)`` for the palm box.

    A box at SHARPA's palm density, which is what ``GEN_PALM_DENSITY_KG_M3``
    was defined for and never used by: a bigger palm weighs more, and a palm
    the size of SHARPA's weighs what SHARPA's does.
    """
    tx, ty, tz = hand.palm.extents
    mass = rpc.GEN_PALM_DENSITY_KG_M3 * tx * ty * tz
    return mass, np.diag([mass * (ty * ty + tz * tz) / 12.0,
                          mass * (tx * tx + tz * tz) / 12.0,
                          mass * (tx * tx + ty * ty) / 12.0])


def merge_rigid_bodies(parts) -> tuple[float, np.ndarray, np.ndarray]:
    """Combine ``(mass, centre, 3x3 inertia about that centre, rotation)`` parts.

    Returns ``(mass, centre of mass, inertia about it)`` in the shared frame.
    The palm merges INTO the arm's last link -- ``rpc.ARM_TIP_LINK`` says so,
    and so do ``flange_to_palm`` and ``palm_center_offset`` -- so the two bodies
    become one and their inertias have to be added about the joint centre of
    mass, not just their masses.
    """
    parts = list(parts)
    mass = sum(float(m) for m, _, _, _ in parts)
    if mass <= 0.0:
        raise ValueError("a merged body needs positive mass")
    com = sum(float(m) * np.asarray(c, float) for m, c, _, _ in parts) / mass

    inertia = np.zeros((3, 3))
    for m, c, i_local, rot in parts:
        rot = np.asarray(rot, float)
        # Into the shared frame, then the parallel-axis shift to the new centre.
        i_shared = rot @ np.asarray(i_local, float) @ rot.T
        d = np.asarray(c, float) - com
        inertia += i_shared + float(m) * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return mass, com, inertia


def principal_axes(inertia) -> tuple[np.ndarray, np.ndarray]:
    """``(diagonal, wxyz quaternion)`` -- USD wants a diagonal plus its rotation."""
    values, vectors = np.linalg.eigh(np.asarray(inertia, float))
    if np.linalg.det(vectors) < 0:            # a reflection is not a rotation
        vectors[:, 0] = -vectors[:, 0]
    _, quat = design_space.mat_to_pos_quat(
        np.block([[vectors, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]))
    return values, np.asarray(quat, float)


def adjacent_links() -> dict[str, list[str]]:
    """Link pairs to exclude from self-collision: a joint's own two bodies.

    Consecutive links touch by construction, so leaving them in makes a hand
    start every episode in self-contact and be pushed apart by the solver
    instead of reaching. Depends on the ENVELOPE, not on any one design: every
    slot's body exists in every env, ghost or not, so the map is shared.
    """
    F, D = design_space.MAX_FINGERS, design_space.MAX_JOINTS_PER_FINGER
    out: dict[str, list[str]] = {rpc.ARM_TIP_LINK: []}
    for f in range(F):
        out[rpc.ARM_TIP_LINK].append(f"f{f}_link0")
        for d in range(D):
            out.setdefault(f"f{f}_link{d}", [])
            if d == 0:
                out[f"f{f}_link{d}"].append(rpc.ARM_TIP_LINK)
            else:
                out[f"f{f}_link{d}"].append(f"f{f}_link{d - 1}")
                out[f"f{f}_link{d - 1}"].append(f"f{f}_link{d}")
    return {k: sorted(set(v)) for k, v in out.items()}


def filtered_pairs(root_path: str, palm_body_path: str) -> dict[str, list[str]]:
    """``link -> prim paths it must NOT collide with``, for one authored design.

    ``adjacent_links`` is names; PhysX wants paths. The fixed-robot path gets
    this for free -- ``_apply_self_collision_filters`` edits a converted USD
    file -- but an authored design has no file to edit, so its pairs have to go
    in as it is written. Without them a generated hand runs with self-collisions
    fully enabled and no masking: consecutive capsules are built to touch, so
    every joint is in contact at rest and interpenetrating the moment it bends.

    Keyed on the ENVELOPE, like ``adjacent_links`` itself: a ghost carries no
    collider and cannot collide, but its slot exists in every env either way.
    """
    path_of = {rpc.ARM_TIP_LINK: palm_body_path}
    F, D = design_space.MAX_FINGERS, design_space.MAX_JOINTS_PER_FINGER
    for f in range(F):
        for d in range(D):
            path_of[f"f{f}_link{d}"] = f"{root_path}/f{f}_link{d}"
    return {
        link: [path_of[n] for n in neighbours if n in path_of]
        for link, neighbours in adjacent_links().items()
    }


def link_frames(hand: design_space.Hand) -> dict[tuple[int, int], np.ndarray]:
    """``(finger, depth) -> 4x4`` pose of every link body, in the palm frame.

    Covers the whole envelope, not just the design. A ghost slot is placed at
    the TIP of the last real link, not at its base: the template's fingertip
    body is always slot ``D-1``, and ``reset.py`` gives a generated design no pad
    offset because "its capsule tip IS the pad". Stacked at the base instead,
    that body sat one whole link short -- 30 to 50 mm, on every finger of every
    design -- and ``fingertip_pos_rel_palm`` was wrong by exactly that.
    """
    F, D = design_space.MAX_FINGERS, design_space.MAX_JOINTS_PER_FINGER
    frames: dict[tuple[int, int], np.ndarray] = {}
    for f in range(F):
        finger = hand.fingers[f] if f < hand.n_fingers else None
        if finger is None:
            # A ghost FINGER stacks at the palm origin; fingertip_valid masks it.
            for d in range(D):
                frames[(f, d)] = np.eye(4)
            continue
        pos, rot = design_space.mount_frame(finger.mount, hand.palm)
        acc = np.eye(4)
        acc[:3, :3], acc[:3, 3] = rot, pos
        for d in range(D):
            if d < finger.n_joints:
                seg = finger.segments[d]
                step = np.eye(4)
                step[:3, :3] = design_space.rodrigues(
                    design_space.axis_of(seg.joint), seg.joint.offset)
                if d:
                    step[:3, 3] = (finger.segments[d - 1].length, 0.0, 0.0)
                acc = acc @ step
            elif d == finger.n_joints:
                # The first ghost steps out to the real tip and every ghost
                # after it stays there, so slot D-1 IS the fingertip.
                step = np.eye(4)
                step[:3, 3] = (finger.segments[d - 1].length, 0.0, 0.0)
                acc = acc @ step
            frames[(f, d)] = acc.copy()
    return frames


def joint_local_frames(hand: design_space.Hand
                       ) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """``(finger, depth) -> (frame in body0, frame in body1)`` for every slot.

    A USD revolute joint is two local frames plus a token axis. The pair has to
    satisfy two things, and neither was true of what this used to author:

      * They must COINCIDE once each body sits at its authored pose, or the
        solver relocates the link to satisfy the constraint and the hand the sim
        runs is not the hand ``link_frames`` placed. ``localPos0``/``localRot0``
        live in BODY0's frame, so a child past the first has to be walked back
        through its parent; the palm-frame pose went in instead, which left 32%
        of a generation's joints disagreeing by up to 2 m.
      * The hinge must land on the token axis. ``theta`` is what makes a joint
        flexion (+z) or abduction (+y) and half of a sampled generation is
        abduction, so hardcoding ``Z`` bent those designs in a direction the
        design space never described. Rotating BOTH frames by the same
        ``hinge_to_z`` moves the hinge onto +z without moving the rest pose.

    Pure numpy, because ``author_hand`` needs Kit and so went unchecked.
    """
    F, D = design_space.MAX_FINGERS, design_space.MAX_JOINTS_PER_FINGER
    frames = link_frames(hand)
    mount = flange_to_palm()
    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for f in range(F):
        finger = hand.fingers[f] if f < hand.n_fingers else None
        for d in range(D):
            real = finger is not None and d < finger.n_joints
            if d:
                origin = np.linalg.inv(frames[(f, d - 1)]) @ frames[(f, d)]
            else:
                origin = mount @ frames[(f, 0)]
            # A ghost is locked, so its hinge is arbitrary; keep it on +z.
            hinge = (design_space.axis_of(finger.segments[d].joint) if real
                     else np.array([0.0, 0.0, 1.0]))
            r_hinge = hinge_to_z(hinge)
            frame0 = origin.copy()
            frame0[:3, :3] = origin[:3, :3] @ r_hinge
            frame1 = np.eye(4)
            frame1[:3, :3] = r_hinge
            out[(f, d)] = (frame0, frame1)
    return out


def link_mass_props(length: float, *, real: bool = True
                    ) -> tuple[float, tuple[float, float, float]]:
    """``(mass, diagonal inertia)`` for one link.

    A real link is a solid-cylinder equivalent at the generated density, so a
    longer link weighs proportionally more rather than every design weighing the
    same.

    A GHOST link takes the virtual-link mass instead -- 1e-6 kg, the value the
    reference setup uses for exactly this. Padding must not change the physics
    of the hand it pads: at the density-derived mass a ghost is 0.29% of a real
    link, and 27 of them add 2.4 g to a hand whose links weigh 90 g. At 1e-6 they
    add 27 ug, which is 0.03% of ONE link. Stacked at their parent, they also
    have no lever arm.
    """
    if not real:
        return rpc.VIRTUAL_LINK_MASS_KG, (rpc.VIRTUAL_LINK_INERTIA,) * 3
    r = rpc.GEN_LINK_RADIUS_M
    mass = max(math.pi * r * r * length * rpc.GEN_LINK_DENSITY_KG_M3, 1e-6)
    ixx = 0.5 * mass * r * r                              # about the link axis
    iyy = mass * (3.0 * r * r + length * length) / 12.0   # transverse
    return mass, (ixx, iyy, iyy)


def author_hand(layer, root_path: str, hand: design_space.Hand, *,
                palm_body_path: str, link7_world, link7_mass_props=None) -> dict[str, int]:
    """Write one design's bodies, colliders and joints under ``root_path``.

    Returns ``{link: n_colliders}`` for the friction pass. Needs ``pxr``, so it
    only runs inside Kit -- the geometry it places is in ``link_frames``, which
    does not.

    ``link7_mass_props`` is the arm flange's own ``(mass, centre, 3x3 inertia)``.
    Given it, the design's palm is merged into that body: a collider for it, and
    the combined mass. Without it the palm is authored with geometry only, which
    is what a caller that cannot read the arm gets.
    """
    from pxr import Gf, Sdf

    from isaacsimenvs.pose_reaching_6d.scene_utils.sdf import attr, define, rel, set_xform

    F, D = design_space.MAX_FINGERS, design_space.MAX_JOINTS_PER_FINGER
    palm_mat = np.asarray(link7_world, float) @ flange_to_palm()
    frames = link_frames(hand)
    joint_frames = joint_local_frames(hand)
    pairs = filtered_pairs(root_path, palm_body_path)
    colliders: dict[str, int] = {}
    define(layer, f"{root_path}/joints", "Scope")

    for f in range(F):
        finger = hand.fingers[f] if f < hand.n_fingers else None
        for d in range(D):
            real = finger is not None and d < finger.n_joints
            seg = finger.segments[d] if real else None
            length = seg.length if real else GHOST_LENGTH_M
            name = f"f{f}_link{d}"

            body = define(layer, f"{root_path}/{name}", "Xform",
                          ["PhysicsRigidBodyAPI", "PhysicsMassAPI",
                           "PhysicsFilteredPairsAPI"])
            if pairs.get(name):
                rel(body, "physics:filteredPairs", pairs[name])
            mass, inertia = link_mass_props(length, real=real)
            attr(body, "physics:mass", Sdf.ValueTypeNames.Float, float(mass))
            attr(body, "physics:diagonalInertia", Sdf.ValueTypeNames.Float3,
                 Gf.Vec3f(*[float(v) for v in inertia]))
            attr(body, "physics:centerOfMass", Sdf.ValueTypeNames.Float3,
                 Gf.Vec3f(float(length / 2.0), 0.0, 0.0))
            pos, quat = design_space.mat_to_pos_quat(palm_mat @ frames[(f, d)])
            set_xform(body, pos, quat)

            # A ghost link carries no collider: it must not touch anything.
            if real:
                r = rpc.GEN_LINK_RADIUS_M
                define(layer, f"{root_path}/{name}/collisions", "Xform")
                mesh = define(layer, f"{root_path}/{name}/collisions/mesh_0", "Xform")
                set_xform(mesh, (length / 2.0, 0.0, 0.0),
                          design_space.rpy_to_quat_wxyz((0.0, math.pi / 2.0, 0.0)))
                cap = define(layer, f"{root_path}/{name}/collisions/mesh_0/capsule",
                             "Capsule", ["PhysicsCollisionAPI"])
                attr(cap, "radius", Sdf.ValueTypeNames.Double, float(r))
                attr(cap, "height", Sdf.ValueTypeNames.Double,
                     float(rpc.cylinder_part(length, r)))
                attr(cap, "axis", Sdf.ValueTypeNames.Token, "Z")
                colliders[name] = colliders.get(name, 0) + 1

            j = define(layer, f"{root_path}/joints/f{f}_j{d}", "PhysicsRevoluteJoint",
                       ["PhysicsDriveAPI:angular", "PhysxJointAPI"])
            rel(j, "physics:body0",
                palm_body_path if d == 0 else f"{root_path}/f{f}_link{d - 1}")
            rel(j, "physics:body1", f"{root_path}/{name}")
            frame0, frame1 = joint_frames[(f, d)]
            jpos, jquat = design_space.mat_to_pos_quat(frame0)
            _, j1quat = design_space.mat_to_pos_quat(frame1)
            attr(j, "physics:localPos0", Sdf.ValueTypeNames.Point3f,
                 Gf.Vec3f(*[float(v) for v in jpos]))
            attr(j, "physics:localRot0", Sdf.ValueTypeNames.Quatf,
                 Gf.Quatf(float(jquat[0]), Gf.Vec3f(*[float(v) for v in jquat[1:]])))
            attr(j, "physics:localPos1", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(0.0, 0.0, 0.0))
            attr(j, "physics:localRot1", Sdf.ValueTypeNames.Quatf,
                 Gf.Quatf(float(j1quat[0]), Gf.Vec3f(*[float(v) for v in j1quat[1:]])))
            attr(j, "physics:axis", Sdf.ValueTypeNames.Token, "Z")
            # A ghost is locked by coincident limits, which is exactly what
            # reset.py reads back as joint_enabled = 0.
            # (0, 1e-8), not (0, 0): the old multi-embodiment path locked ghosts
            # this way, and exactly coincident limits are a degenerate
            # constraint. Still far below reset.py's 1e-6 enabled threshold.
            lo, hi = (seg.joint.limits or design_space.JOINT_LIMIT) if real else (0.0, 1e-8)
            attr(j, "physics:lowerLimit", Sdf.ValueTypeNames.Float, float(math.degrees(lo)))
            attr(j, "physics:upperLimit", Sdf.ValueTypeNames.Float, float(math.degrees(hi)))
            attr(j, "physics:jointEnabled", Sdf.ValueTypeNames.Bool, True)
            attr(j, "physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool, False)
            attr(j, "drive:angular:physics:stiffness", Sdf.ValueTypeNames.Float,
                 rpc.CONVERTER_DRIVE_STIFFNESS)
            attr(j, "drive:angular:physics:damping", Sdf.ValueTypeNames.Float,
                 rpc.CONVERTER_DRIVE_DAMPING)
            # Full effort for ghosts too: the drive is what holds a locked joint shut.
            attr(j, "drive:angular:physics:maxForce", Sdf.ValueTypeNames.Float,
                 float(rpc.GEN_JOINT_EFFORT_NM))
            attr(j, "drive:angular:physics:targetPosition", Sdf.ValueTypeNames.Float, 0.0)
            # Unset, this is no limit at all (5.9e36), and the reaction moves the arm.
            attr(j, "physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float,
                 float(math.degrees(rpc.GEN_JOINT_VELOCITY_RAD_S)))

    _author_palm(layer, root_path, hand, palm_body_path, link7_mass_props, colliders)
    return colliders


def _author_palm(layer, root_path, hand, palm_body_path, link7_mass_props, colliders):
    """The palm slab, ON the flange body rather than beside it.

    ``ARM_TIP_LINK`` is "the link the hand's palm merges into under
    merge_fixed_joints", and ``flange_to_palm`` and ``palm_center_offset`` say
    the same -- so the palm is a collider and a mass on link_7, not a body of
    its own. That also leaves the articulation's link count and every joint name
    untouched, so a checkpoint still loads.
    """
    from pxr import Gf, Sdf

    from isaacsimenvs.pose_reaching_6d.scene_utils.sdf import attr, define, set_xform

    extents, pose = palm_box(hand)
    mass, inertia_local = palm_mass_props(hand)
    pos, quat = design_space.mat_to_pos_quat(pose)

    wrap = define(layer, f"{palm_body_path}/hand_palm", "Xform")
    set_xform(wrap, pos, quat)
    shape = define(layer, f"{palm_body_path}/hand_palm/box", "Cube",
                   ["PhysicsCollisionAPI", "PhysxCollisionAPI"])
    lx, ly, lz = (float(v) for v in extents)
    attr(shape, "size", Sdf.ValueTypeNames.Double, 1.0)
    attr(shape, "xformOp:scale", Sdf.ValueTypeNames.Double3, Gf.Vec3d(lx, ly, lz))
    attr(shape, "xformOpOrder", Sdf.ValueTypeNames.TokenArray, ["xformOp:scale"])
    attr(shape, "extent", Sdf.ValueTypeNames.Float3Array,
         [Gf.Vec3f(-0.5 * lx, -0.5 * ly, -0.5 * lz),
          Gf.Vec3f(0.5 * lx, 0.5 * ly, 0.5 * lz)])
    colliders[rpc.ARM_TIP_LINK] = colliders.get(rpc.ARM_TIP_LINK, 0) + 1

    if link7_mass_props is None:
        return
    # One body now, so the inertias add about the JOINT centre of mass.
    arm_mass, arm_com, arm_inertia = link7_mass_props
    total, com, inertia = merge_rigid_bodies([
        (float(arm_mass), np.asarray(arm_com, float), np.asarray(arm_inertia, float), np.eye(3)),
        (mass, pose[:3, 3], inertia_local, pose[:3, :3]),
    ])
    diag, principal = principal_axes(inertia)
    body = Sdf.CreatePrimInLayer(layer, Sdf.Path(palm_body_path))   # an OVER: the
    attr(body, "physics:mass", Sdf.ValueTypeNames.Float, float(total))  # arm defines it
    attr(body, "physics:centerOfMass", Sdf.ValueTypeNames.Float3,
         Gf.Vec3f(*[float(v) for v in com]))
    attr(body, "physics:diagonalInertia", Sdf.ValueTypeNames.Float3,
         Gf.Vec3f(*[float(v) for v in diag]))
    attr(body, "physics:principalAxes", Sdf.ValueTypeNames.Quatf,
         Gf.Quatf(float(principal[0]), Gf.Vec3f(*[float(v) for v in principal[1:]])))


def urdf_for_viewing(hand: design_space.Hand, out_path) -> "pathlib.Path":
    """Write a URDF of ``hand`` for the browser viewer. NOT for simulation.

    The simulator gets authored USD prims; nothing here feeds physics. But the
    Three.js viewer is URDF-driven, and a generated spec has no urdf_path -- so
    without this the viewer falls back to SHARPA's and silently shows the wrong
    hand while the sim runs another one.

    The WHOLE envelope, not just the design's links. The viewer animates
    ``joint_names_canonical``, which is the template's F x D slot names, and it
    throws on the first name the URDF does not carry -- so every slot needs its
    joint even when the design does not use it. A ghost slot gets a body and a
    joint but no geometry, which is what keeps it from drawing as a speck at the
    palm; ``author_hand`` locks it the same way, at ``(0, 1e-8)``.

    Rooted at ``ARM_TIP_LINK``, not at a palm of its own: the palm merges into
    the arm's last link, so ``d == 0`` hangs off the flange through
    ``flange_to_palm`` exactly as ``author_hand`` places it. The caller grafts
    this onto the arm chain.
    """
    import pathlib
    import xml.etree.ElementTree as ET

    root = ET.Element("robot", {"name": "generated_hand"})
    root.append(ET.Comment(" GENERATED by hand_sampler/build.py for VIEWING only. "
                           "The simulator authors USD prims directly. "))
    F, D = design_space.MAX_FINGERS, design_space.MAX_JOINTS_PER_FINGER
    frames = link_frames(hand)
    mount = flange_to_palm()
    r = rpc.GEN_LINK_RADIUS_M
    # The palm rides on the flange, the way it is authored: one link, and the
    # viewer showing what the simulator has rather than a hand floating free.
    flange = ET.SubElement(root, "link", {"name": rpc.ARM_TIP_LINK})
    extents, palm_pose = palm_box(hand)
    palm_rpy = design_space.mat_to_rpy(palm_pose[:3, :3])
    for tag in ("visual", "collision"):
        node = ET.SubElement(flange, tag)
        ET.SubElement(node, "origin", {
            "xyz": " ".join(f"{v}" for v in palm_pose[:3, 3]),
            "rpy": f"{palm_rpy[0]} {palm_rpy[1]} {palm_rpy[2]}"})
        ET.SubElement(ET.SubElement(node, "geometry"),
                      "box", {"size": " ".join(f"{v}" for v in extents)})

    for f in range(F):
        finger = hand.fingers[f] if f < hand.n_fingers else None
        for d in range(D):
            real = finger is not None and d < finger.n_joints
            seg = finger.segments[d] if real else None
            name = f"f{f}_link{d}"
            link = ET.SubElement(root, "link", {"name": name})
            # A ghost carries no geometry, the way it carries no collider.
            if real:
                # A CAPSULE, which URDF cannot name: the collider the sim
                # authors is a cylinder plus two hemispherical caps, and on a
                # median link the caps are HALF its length. Drawn as one
                # cylinder the fingers came out flat-ended and too stubby,
                # which is most misleading exactly at the tip that does the
                # touching. Same decomposition as the USD Capsule prim --
                # cylinder_part long, centred, with a sphere at each end.
                barrel = rpc.cylinder_part(seg.length, r)
                centre = seg.length / 2.0
                for tag in ("visual", "collision"):
                    node = ET.SubElement(link, tag)
                    # The capsule lies along +x; a URDF cylinder is along +z.
                    ET.SubElement(node, "origin", {
                        "xyz": f"{centre} 0 0", "rpy": f"0 {math.pi / 2.0} 0"})
                    geom = ET.SubElement(node, "geometry")
                    ET.SubElement(geom, "cylinder", {
                        "radius": f"{r}", "length": f"{max(barrel, 1e-6)}"})
                    for end in (centre - barrel / 2.0, centre + barrel / 2.0):
                        cap = ET.SubElement(link, tag)
                        ET.SubElement(cap, "origin", {"xyz": f"{end} 0 0", "rpy": "0 0 0"})
                        ET.SubElement(ET.SubElement(cap, "geometry"),
                                      "sphere", {"radius": f"{r}"})

            # A joint origin is the CHILD's frame in the PARENT's, so walk back
            # through the parent rather than reusing the palm-frame pose: the
            # absolute rotation put every link past the first at the wrong
            # angle, which is how a two-segment finger drew as a bent stub.
            if d:
                parent = f"f{f}_link{d - 1}"
                local = np.linalg.inv(frames[(f, d - 1)]) @ frames[(f, d)]
            else:
                parent = rpc.ARM_TIP_LINK
                local = mount @ frames[(f, 0)]
            pos = tuple(float(v) for v in local[:3, 3])
            rpy = design_space.mat_to_rpy(local[:3, :3])

            joint = ET.SubElement(root, "joint",
                                  {"name": f"f{f}_j{d}", "type": "revolute"})
            ET.SubElement(joint, "parent", {"link": parent})
            ET.SubElement(joint, "child", {"link": name})
            ET.SubElement(joint, "origin", {
                "xyz": f"{pos[0]} {pos[1]} {pos[2]}",
                "rpy": f"{rpy[0]} {rpy[1]} {rpy[2]}"})
            # A URDF axis is a vector, so the hinge goes in as-is. Hardcoding z
            # drew every abduction joint as a flexion joint.
            hinge = design_space.axis_of(seg.joint) if real else (0.0, 0.0, 1.0)
            ET.SubElement(joint, "axis", {
                "xyz": f"{float(hinge[0])} {float(hinge[1])} {float(hinge[2])}"})
            lo, hi = (seg.joint.limits or design_space.JOINT_LIMIT) if real else (0.0, 1e-8)
            ET.SubElement(joint, "limit", {
                "lower": f"{lo}", "upper": f"{hi}",
                "effort": f"{rpc.GEN_JOINT_EFFORT_NM}",
                "velocity": f"{rpc.GEN_JOINT_VELOCITY_RAD_S}"})

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path
