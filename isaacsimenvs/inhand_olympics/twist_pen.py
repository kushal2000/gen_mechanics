"""The twist-apart pen: the pen split into a barrel and a cap, joined by a revolute joint about its long axis.

One articulation, authored straight into a layer as scene.py authors the rigid pen (author_objects conventions):

    <path>                ArticulationRoot
    <path>/barrel         the root link: x from -length/2 to length/2 - cap_length - gap
    <path>/cap            x from length/2 - cap_length to length/2
    <path>/twist          revolute about x, barrel -> cap, no limits

Both links' frames sit at the pen's centre (each shape and centre of mass offset along x), so the articulation's root
pose is the pen's centre and orientation, as the rigid pen's root pose is, and the joint angle is the cap's twist
relative to the barrel. Masses split by length, the pen's total mass kept.

The cap resists twisting like a friction fit: the joint's drive has no stiffness, a large damping and its effort capped
at the friction torque (env TwistPenCfg / the actuator cfg), so below that torque the cap barely creeps (velocity =
torque / damping) and above it the drive saturates and the cap turns, resisted by the friction torque.
"""
from __future__ import annotations

import math

from isaacsimenvs.pose_reaching_6d.scene_utils.author_objects import (_rigid_body_defaults, _shape_prim,
                                                                      author_physics_material)
from isaacsimenvs.pose_reaching_6d.scene_utils.sdf import attr, define, rel

LINKS = ("barrel", "cap")
JOINT = "twist"


def link_extents(length: float, cap_length: float, gap: float) -> dict[str, tuple[float, float]]:
    """Each link's (x start, x end) along the pen, the pen's centre at 0."""
    return {"barrel": (-length / 2, length / 2 - cap_length - gap / 2), "cap": (length / 2 - cap_length + gap / 2,
                                                                                length / 2)}


def author_twist_pen(layer, path: str, *, length: float, diameter: float, cap_length: float, mass: float,
                     friction: float, shape: str = "box", gap: float = 0.001, material_path: str | None = None,
                     damping: float = 10.0, friction_torque: float = 0.02) -> None:
    """Author the two-link pen at ``path`` (an Xform made here). ``shape`` box or capsule (a capsule's caps make the
    two links rounded at the pen's ends and at the split). damping (N m s/rad) and friction_torque (N m) are the
    authored drive; the env's actuator cfg sets them again at runtime."""
    from pxr import Gf, Sdf

    if material_path is None:
        material_path = author_physics_material(layer, f"{path}_material", static_friction=friction,
                                                dynamic_friction=friction)
    root = define(layer, path, "Xform", ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"])
    attr(root, "physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool, False)
    attr(root, "physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int, 8)
    attr(root, "physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int, 1)
    ext = link_extents(length, cap_length, gap)
    total = sum(b - a for a, b in ext.values())
    for name, (a, b) in ext.items():
        seg, mid = b - a, (a + b) / 2
        m = mass * seg / total
        body = define(layer, f"{path}/{name}", "Xform", ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"])
        _rigid_body_defaults(body)
        r = diameter / 2
        if shape == "box":
            scale = (seg, diameter, diameter)
            ixx, iyy = m * diameter ** 2 / 6, m * (seg ** 2 + diameter ** 2) / 12
            rpy = (0.0, 0.0, 0.0)
        elif shape == "capsule":
            scale = (max(seg - diameter, 1e-4), diameter)
            ixx, iyy = m * r * r / 2, m * (3 * r * r + seg ** 2) / 12     # as a solid cylinder; close enough
            rpy = (0.0, -math.pi / 2, 0.0)
        else:
            raise ValueError(f"twist pen shape must be 'box' or 'capsule', got {shape!r}")
        attr(body, "physics:mass", Sdf.ValueTypeNames.Float, float(m))
        attr(body, "physics:diagonalInertia", Sdf.ValueTypeNames.Float3, Gf.Vec3f(float(ixx), float(iyy), float(iyy)))
        attr(body, "physics:centerOfMass", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(float(mid), 0.0, 0.0))
        define(layer, f"{path}/{name}/collisions", "Xform")
        _shape_prim(layer, f"{path}/{name}/collisions/mesh_0", scale, (mid, 0.0, 0.0), rpy, collision=True,
                    material_path=material_path)
    joint = define(layer, f"{path}/{JOINT}", "PhysicsRevoluteJoint", ["PhysicsDriveAPI:angular", "PhysxJointAPI"])
    rel(joint, "physics:body0", f"{path}/barrel")
    rel(joint, "physics:body1", f"{path}/cap")
    attr(joint, "physics:axis", Sdf.ValueTypeNames.Token, "X")
    for k in ("0", "1"):
        attr(joint, f"physics:localPos{k}", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(0.0, 0.0, 0.0))
        attr(joint, f"physics:localRot{k}", Sdf.ValueTypeNames.Quatf, Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    # USD angular drive gains are per degree; maxForce is N m.
    attr(joint, "drive:angular:physics:type", Sdf.ValueTypeNames.Token, "force")
    attr(joint, "drive:angular:physics:stiffness", Sdf.ValueTypeNames.Float, 0.0)
    attr(joint, "drive:angular:physics:damping", Sdf.ValueTypeNames.Float, float(damping) * math.pi / 180)
    attr(joint, "drive:angular:physics:maxForce", Sdf.ValueTypeNames.Float, float(friction_torque))
    attr(joint, "drive:angular:physics:targetVelocity", Sdf.ValueTypeNames.Float, 0.0)


__all__ = ["JOINT", "LINKS", "author_twist_pen", "link_extents"]
