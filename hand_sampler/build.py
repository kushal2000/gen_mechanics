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


def link_frames(hand: design_space.Hand) -> dict[tuple[int, int], np.ndarray]:
    """``(finger, depth) -> 4x4`` pose of every link body, in the palm frame.

    Covers the whole envelope, not just the design: a ghost slot is placed at
    its parent so its body sits on top of the last real link rather than at the
    palm origin, which keeps a padded finger's template tip where its real tip
    is.
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
            frames[(f, d)] = acc.copy()
    return frames


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
                palm_body_path: str, link7_world) -> dict[str, int]:
    """Write one design's bodies, colliders and joints under ``root_path``.

    Returns ``{link: n_colliders}`` for the friction pass. Needs ``pxr``, so it
    only runs inside Kit -- the geometry it places is in ``link_frames``, which
    does not.
    """
    from pxr import Gf, Sdf

    from isaacsimenvs.pose_reaching_6d.scene_utils.sdf import attr, define, rel, set_xform

    F, D = design_space.MAX_FINGERS, design_space.MAX_JOINTS_PER_FINGER
    palm_mat = np.asarray(link7_world, float) @ flange_to_palm()
    frames = link_frames(hand)
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
                          ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
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
            origin = flange_to_palm() @ frames[(f, d)] if d == 0 else frames[(f, d)]
            if d == 0 and finger is not None:
                origin = flange_to_palm() @ frames[(f, 0)]
            jpos, jquat = design_space.mat_to_pos_quat(origin)
            attr(j, "physics:localPos0", Sdf.ValueTypeNames.Point3f,
                 Gf.Vec3f(*[float(v) for v in jpos]))
            attr(j, "physics:localRot0", Sdf.ValueTypeNames.Quatf,
                 Gf.Quatf(float(jquat[0]), Gf.Vec3f(*[float(v) for v in jquat[1:]])))
            attr(j, "physics:localPos1", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(0.0, 0.0, 0.0))
            attr(j, "physics:localRot1", Sdf.ValueTypeNames.Quatf,
                 Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
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

    return colliders
