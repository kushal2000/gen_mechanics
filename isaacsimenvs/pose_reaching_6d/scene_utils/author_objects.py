"""Author the task objects' USD directly, instead of converting URDFs.

Each object is one rigid body with two analytic shapes (handle and head, each
a box or a capsule) and a closed-form mass and inertia, so the URDF round-trip
through Kit's importer (~0.2 s each, ~4 min for a 1200-object pool) only
recovers numbers we started with.

Conventions, shared with ``generate_objects``: a 3-tuple scale is a BOX
(lx, ly, lz), a 2-tuple a CYLINDER (height, diameter); the handle is rotated
rpy=(0, -pi/2, 0) so its axis lies along +x (ixx <-> izz); the head sits at
x_offset along +x, a cylinder head rotated rpy=(-pi/2, 0, 0) (iyy <-> izz);
the inertial frame sits at the composite COM.
"""

from __future__ import annotations

import math

from hand_sampler.inertia import compute_mass_and_inertia
from hand_sampler.rotations import rpy_to_quat_wxyz

from .author_usd import MAX_DEPEN_VELOCITY, attr, define
from .objects.generate_objects import OBJECT_ROOT_LINK


def _rigid_body_defaults(body_spec) -> None:
    """Properties whose authored defaults differ from the converter's: principalAxes
    defaults to a zero quaternion, maxDepenetrationVelocity to 3.0."""
    from pxr import Gf, Sdf

    attr(body_spec, "physics:principalAxes", Sdf.ValueTypeNames.Quatf, Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    attr(body_spec, "physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float,
         MAX_DEPEN_VELOCITY)


def _hold_in_place(body_spec, kinematic: bool) -> None:
    """Kinematic with gravity off, as the goal marker must be; a non-colliding
    dynamic body still falls."""
    from pxr import Sdf

    if kinematic:
        attr(body_spec, "physics:kinematicEnabled", Sdf.ValueTypeNames.Bool, True)
        attr(body_spec, "physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool, True)


def author_physics_material(layer, path: str, static_friction: float = 0.5,
                            dynamic_friction: float = 0.5, restitution: float = 0.0) -> str:
    """One physics material for authored colliders to bind against."""
    from pxr import Sdf

    mat = define(layer, path, "Material", ["PhysicsMaterialAPI"])
    attr(mat, "physics:staticFriction", Sdf.ValueTypeNames.Float, float(static_friction))
    attr(mat, "physics:dynamicFriction", Sdf.ValueTypeNames.Float, float(dynamic_friction))
    attr(mat, "physics:restitution", Sdf.ValueTypeNames.Float, float(restitution))
    return path


def _shape_prim(layer, path: str, scale, xyz, rpy, collision: bool = True,
                material_path: str | None = None):
    """One handle/head shape, structured as the converter emits it:
    mesh_N (Xform: translate, orient, scale) > unscaled shape with a scaled extent."""
    from pxr import Gf, Sdf

    is_box = len(scale) == 3
    wrap = define(layer, path, "Xform")
    attr(wrap, "xformOp:translate", Sdf.ValueTypeNames.Double3, Gf.Vec3d(*[float(v) for v in xyz]))
    ops = ["xformOp:translate"]
    if any(abs(float(v)) > 1e-12 for v in rpy):
        w, x, y, z = rpy_to_quat_wxyz(rpy)
        attr(wrap, "xformOp:orient", Sdf.ValueTypeNames.Quatd, Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        ops.append("xformOp:orient")
    if is_box:
        lx, ly, lz = (float(v) for v in scale)
        attr(wrap, "xformOp:scale", Sdf.ValueTypeNames.Double3, Gf.Vec3d(lx, ly, lz))
        ops.append("xformOp:scale")
    attr(wrap, "xformOpOrder", Sdf.ValueTypeNames.TokenArray, ops)

    # Collision APIs always: a non-colliding shape is collisionEnabled=False, not
    # a body with no shapes, which PhysX cannot report materials for. The
    # material is bound here on the shape, inside the caller's ChangeBlock.
    apis = ["PhysicsCollisionAPI", "PhysxCollisionAPI"] + (["MaterialBindingAPI"] if material_path else [])
    if is_box:
        shape = define(layer, f"{path}/box", "Cube", apis)
        attr(shape, "size", Sdf.ValueTypeNames.Double, 1.0)
        attr(shape, "extent", Sdf.ValueTypeNames.Float3Array,
             [Gf.Vec3f(-0.5 * lx, -0.5 * ly, -0.5 * lz), Gf.Vec3f(0.5 * lx, 0.5 * ly, 0.5 * lz)])
    else:
        h, d = (float(v) for v in scale)
        r = d / 2.0
        shape = define(layer, f"{path}/capsule", "Capsule", apis)
        attr(shape, "height", Sdf.ValueTypeNames.Double, h)
        attr(shape, "radius", Sdf.ValueTypeNames.Double, r)
        attr(shape, "axis", Sdf.ValueTypeNames.Token, "Z")
        attr(shape, "extent", Sdf.ValueTypeNames.Float3Array,
             [Gf.Vec3f(-r, -r, -(h / 2 + r)), Gf.Vec3f(r, r, h / 2 + r)])
    if collision:
        for name, value in (("contactOffset", 0.002), ("restOffset", 0.0),
                            ("torsionalPatchRadius", 0.0), ("minTorsionalPatchRadius", 0.0)):
            attr(shape, f"physxCollision:{name}", Sdf.ValueTypeNames.Float, value)
    else:
        attr(shape, "physics:collisionEnabled", Sdf.ValueTypeNames.Bool, False)
    if material_path:
        binding = Sdf.RelationshipSpec(shape, "material:binding:physics", custom=False)
        binding.targetPathList.explicitItems.append(Sdf.Path(material_path))
    return shape


def author_handle_head(layer, prim_path: str, handle_scale, head_scale,
                       handle_density: float, head_density: float,
                       collision: bool = True, material_path: str | None = None,
                       kinematic: bool = False) -> None:
    """Author one handle(+head) object as a single rigid body under ``prim_path``.
    ``head_scale`` None is a handle-only type."""
    from pxr import Gf, Sdf

    if len(handle_scale) == 3:
        handle_rpy = (0.0, 0.0, 0.0)
        handle_mass, handle_ixx, handle_iyy, handle_izz = compute_mass_and_inertia(
            handle_scale, handle_density)
    else:
        handle_rpy = (0.0, -math.pi / 2.0, 0.0)  # axis along +x: ixx <-> izz
        handle_mass, handle_izz, handle_iyy, handle_ixx = compute_mass_and_inertia(
            handle_scale, handle_density)

    body_path = f"{prim_path}/{OBJECT_ROOT_LINK}"
    body = define(layer, body_path, "Xform",
                  ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"])
    _rigid_body_defaults(body)
    _hold_in_place(body, kinematic)

    if head_scale is None:
        attr(body, "physics:mass", Sdf.ValueTypeNames.Float, float(handle_mass))
        attr(body, "physics:diagonalInertia", Sdf.ValueTypeNames.Float3,
             Gf.Vec3f(float(handle_ixx), float(handle_iyy), float(handle_izz)))
        define(layer, f"{body_path}/collisions", "Xform")
        _shape_prim(layer, f"{body_path}/collisions/mesh_0", handle_scale, (0.0, 0.0, 0.0),
                    handle_rpy, collision=collision, material_path=material_path)
        return

    if len(head_scale) == 3:
        x_offset = float(handle_scale[0]) / 2.0 + float(head_scale[0]) / 2.0
        head_rpy = (0.0, 0.0, 0.0)
        head_mass, head_ixx, head_iyy, head_izz = compute_mass_and_inertia(
            head_scale, head_density)
    else:
        x_offset = float(handle_scale[0]) / 2.0 + float(head_scale[1]) / 2.0
        head_rpy = (-math.pi / 2.0, 0.0, 0.0)  # axis along +y: iyy <-> izz
        head_mass, head_ixx, head_izz, head_iyy = compute_mass_and_inertia(
            head_scale, head_density)

    # Composite inertia about the joint COM (parallel axis).
    total_mass = handle_mass + head_mass
    com_x = (head_mass * x_offset) / total_mass
    d_handle, d_head = -com_x, x_offset - com_x
    ixx = handle_ixx + head_ixx
    iyy = (handle_iyy + handle_mass * d_handle ** 2) + (head_iyy + head_mass * d_head ** 2)
    izz = (handle_izz + handle_mass * d_handle ** 2) + (head_izz + head_mass * d_head ** 2)

    attr(body, "physics:mass", Sdf.ValueTypeNames.Float, float(total_mass))
    attr(body, "physics:diagonalInertia", Sdf.ValueTypeNames.Float3,
         Gf.Vec3f(float(ixx), float(iyy), float(izz)))
    attr(body, "physics:centerOfMass", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(float(com_x), 0.0, 0.0))
    define(layer, f"{body_path}/collisions", "Xform")
    _shape_prim(layer, f"{body_path}/collisions/mesh_0", handle_scale, (0.0, 0.0, 0.0),
                handle_rpy, collision=collision, material_path=material_path)
    _shape_prim(layer, f"{body_path}/collisions/mesh_1", head_scale, (x_offset, 0.0, 0.0),
                head_rpy, collision=collision, material_path=material_path)


__all__ = ["author_handle_head", "author_physics_material"]
