"""Author the task objects' USD directly, instead of converting URDFs.

Measured on a live training launch at 24,576 envs:

    [+240.52s]  resolved baked USDs                  <- 100 object variants
    [+756.59s]  spawned robot/table/object/goalviz

Conversion is ~4 minutes of the ~12.6-minute scene build, and it is almost
entirely the objects: the robot is a single conversion, the objects are 100.

Nothing about an object needs a URDF importer. Each is ONE rigid body with two
analytic shapes -- a handle and a head, each a box or a cylinder -- and a
composite mass/inertia that ``generate_objects`` already computes in closed form
("No trimesh / vhacd code"). Round-tripping that through URDF text and Kit's
importer to recover numbers we started with is pure overhead.

So this authors the same body with Sdf specs: ~5 prims per object against the
~400 of a robot. Authoring measured at ~8 ms per 44-link robot, so a 5-prim
object is well under a millisecond.

The geometry conventions are copied from ``_handle_head_urdf_variable_density``
and must stay in step with it:

  * a 3-tuple scale is a BOX (lx, ly, lz); a 2-tuple is a CYLINDER (height,
    diameter), radius = diameter / 2
  * the handle cylinder is rotated rpy=(0, -pi/2, 0) so its axis lies along +x,
    and its inertia is flipped ixx <-> izz to match
  * the head sits at x_offset along +x, and the head cylinder is rotated
    rpy=(-pi/2, 0, 0) with iyy <-> izz flipped
  * the inertial frame sits at the composite COM, not the origin

Correctness is checked against the converter, not asserted: see
``genmech.tools.compare_object_assets``, which builds both and compares mass,
inertia, centre of mass and collision extents.
"""

from __future__ import annotations

import math

# Mirrors generate_objects._OBJECT_ROOT_LINK. Downstream code (the pose viewer,
# the env's object handling) looks this name up, so the authored prim must carry
# it too.
OBJECT_ROOT_LINK = "object_root"


def _rigid_body_defaults(body_spec) -> None:
    """Properties PhysX needs that do NOT default usefully when authored.

    Found by diffing a converted object body against an authored one:

      physics:principalAxes   converted (1,0,0,0), authored (0,0,0,0). This is
          the ORIENTATION OF THE INERTIA TENSOR, and the unset default is a zero
          quaternion -- not a rotation at all. Colliders and mass were correct,
          so the hand still lifted the object (72% vs 77%), but its rotational
          dynamics were wrong and the task is to reach a POSE goal: 2.92 goals
          against the converted path's 5.07.

      physxRigidBody:maxDepenetrationVelocity   the scene builder bakes 1000.0
          for objects; the authored default is 3.0, which resolves deep
          interpenetration far more slowly.

    Also authors the identity transform ops the converter writes, so the prim
    carries the same xformOpOrder the rest of the stack sees.
    """
    from pxr import Gf, Sdf

    from genmech.robots.generated.author_usd import attr

    attr(body_spec, "physics:principalAxes", Sdf.ValueTypeNames.Quatf,
         Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    attr(body_spec, "physxRigidBody:maxDepenetrationVelocity",
         Sdf.ValueTypeNames.Float, 1000.0)


def _hold_in_place(body_spec, kinematic: bool) -> None:
    """Make a body kinematic with gravity off, as the goal marker must be.

    The converted path bakes goalviz with kinematic_enabled=True and
    disable_gravity=True. Authoring it as a plain dynamic body let every goal
    marker free-fall 31 metres in 150 steps, and since the policy steers on
    keypoints_rel_goal it was chasing a target accelerating toward the floor:
    0.00 goals, 3% lift, against 5.07 goals and 77% lift on the converted path.

    Disabling the COLLIDER is not enough and is a different property -- a goal
    marker must not collide AND must not move.
    """
    from pxr import Sdf

    from genmech.robots.generated.author_usd import attr

    if not kinematic:
        return
    attr(body_spec, "physics:kinematicEnabled", Sdf.ValueTypeNames.Bool, True)
    attr(body_spec, "physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool, True)


def author_physics_material(layer, path: str, static_friction: float = 0.5,
                            dynamic_friction: float = 0.5,
                            restitution: float = 0.0) -> str:
    """One physics material for authored colliders to bind against."""
    from pxr import Sdf

    from genmech.robots.generated.author_usd import attr, define

    mat = define(layer, path, "Material", ["PhysicsMaterialAPI"])
    attr(mat, "physics:staticFriction", Sdf.ValueTypeNames.Float, float(static_friction))
    attr(mat, "physics:dynamicFriction", Sdf.ValueTypeNames.Float, float(dynamic_friction))
    attr(mat, "physics:restitution", Sdf.ValueTypeNames.Float, float(restitution))
    return path


def _shape_prim(layer, path: str, scale, xyz, rpy, collision: bool = True,
                material_path: str | None = None):
    """One handle/head shape, structured exactly as the converter emits it.

    The converter writes  mesh_N (Xform: translate, orient, scale) > shape,
    with the shape itself unscaled and carrying a scaled `extent`. Authoring the
    scale onto the shape prim instead composes to the same transform but is not
    the same USD, and the object was the asset that carried a 5.07 -> 3.00 drop
    in policy score once every physics attribute already matched. Mirroring the
    structure removes that as a variable rather than arguing about it.
    """
    import math

    from pxr import Gf, Sdf

    from genmech.robots.generated.author_usd import attr, define

    is_box = len(scale) == 3

    # --- mesh_N wrapper carries the placement (and the scale, for boxes) ---
    wrap = define(layer, path, "Xform")
    ops = []
    attr(wrap, "xformOp:translate", Sdf.ValueTypeNames.Double3,
         Gf.Vec3d(*[float(v) for v in xyz]))
    ops.append("xformOp:translate")
    if any(abs(float(v)) > 1e-12 for v in rpy):
        # The converter stores orientation as a quaternion, not Euler.
        cr, cp, cy = (math.cos(float(v) / 2) for v in rpy)
        sr, sp, sy = (math.sin(float(v) / 2) for v in rpy)
        q = Gf.Quatd(cr * cp * cy + sr * sp * sy,
                     Gf.Vec3d(sr * cp * cy - cr * sp * sy,
                              cr * sp * cy + sr * cp * sy,
                              cr * cp * sy - sr * sp * cy))
        attr(wrap, "xformOp:orient", Sdf.ValueTypeNames.Quatd, q)
        ops.append("xformOp:orient")
    if is_box:
        lx, ly, lz = (float(v) for v in scale)
        attr(wrap, "xformOp:scale", Sdf.ValueTypeNames.Double3, Gf.Vec3d(lx, ly, lz))
        ops.append("xformOp:scale")
    attr(wrap, "xformOpOrder", Sdf.ValueTypeNames.TokenArray, ops)

    # --- the shape itself: unscaled, with a scaled extent ------------------
    # ALWAYS apply the collision APIs. A non-colliding shape is expressed as
    # collisionEnabled=False, not as a shape with no API: a body with zero
    # shapes gives PhysX a view it cannot report materials for, and the env's
    # friction pass dies with "Failed to get rigid body material properties".
    # The converter does the same -- goalviz keeps its colliders, disabled.
    apis = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]
    # Bind the physics material HERE, on the shape prim, inside the caller's
    # ChangeBlock.
    #
    # This used to be done afterwards by isaaclab's bind_physics_material, two
    # calls per env, outside the ChangeBlock. That helper is decorated with
    # apply_nested, so each call traverses the subtree, and each authors into a
    # live stage -- at 24,576 envs that is 49,152 traversals of a stage the
    # authored population has grown to ~5.8M prims. We author these shapes
    # ourselves and know exactly which prims need the binding, so there is
    # nothing to discover by traversal.
    if material_path:
        apis.append("MaterialBindingAPI")
    if is_box:
        shape = define(layer, f"{path}/box", "Cube", apis)
        attr(shape, "size", Sdf.ValueTypeNames.Double, 1.0)
        attr(shape, "extent", Sdf.ValueTypeNames.Float3Array,
             [Gf.Vec3f(-0.5 * lx, -0.5 * ly, -0.5 * lz),
              Gf.Vec3f(0.5 * lx, 0.5 * ly, 0.5 * lz)])
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
        attr(shape, "physxCollision:contactOffset", Sdf.ValueTypeNames.Float, 0.002)
        attr(shape, "physxCollision:restOffset", Sdf.ValueTypeNames.Float, 0.0)
        attr(shape, "physxCollision:torsionalPatchRadius", Sdf.ValueTypeNames.Float, 0.0)
        attr(shape, "physxCollision:minTorsionalPatchRadius", Sdf.ValueTypeNames.Float, 0.0)
    else:
        attr(shape, "physics:collisionEnabled", Sdf.ValueTypeNames.Bool, False)

    # material:binding:physics is the relationship UsdShade's MaterialBindingAPI
    # writes for materialPurpose="physics" -- the same one bind_physics_material
    # produces, authored directly instead of through a stage-level helper.
    if material_path:
        rel = Sdf.RelationshipSpec(shape, "material:binding:physics",
                                   custom=False)
        rel.targetPathList.explicitItems.append(Sdf.Path(material_path))
    return shape


def author_handle_head(layer, prim_path: str, handle_scale, head_scale,
                       handle_density: float, head_density: float,
                       body_at_root: bool = False, collision: bool = True,
                       material_path: str | None = None,
                       kinematic: bool = False):
    """Author one handle+head object as a single rigid body.

    Returns the composite ``(mass, ixx, iyy, izz, com_x)`` actually authored, so
    a caller can compare them against the URDF path's numbers rather than
    trusting that two implementations of the same arithmetic agree.
    """
    from pxr import Gf, Sdf

    from genmech.robots.generated.author_usd import attr, define
    from genmech.tasks.pose_reach.utils.generate_objects import (
        _compute_mass_and_inertia,
    )

    # --- handle -----------------------------------------------------------
    if len(handle_scale) == 3:
        handle_rpy = (0.0, 0.0, 0.0)
        handle_mass, handle_ixx, handle_iyy, handle_izz = _compute_mass_and_inertia(
            handle_scale, handle_density)
    else:
        handle_rpy = (0.0, -math.pi / 2.0, 0.0)
        # Rotated so the handle axis is along +x: ixx <-> izz, matching
        # _handle_head_urdf_variable_density.
        handle_mass, handle_izz, handle_iyy, handle_ixx = _compute_mass_and_inertia(
            handle_scale, handle_density)

    # --- handle-only objects ----------------------------------------------
    # head_scale is None for types the generator emits without a head (its
    # generate_handle_head_urdf delegates to generate_handle_urdf). The inertia
    # is then the handle's alone, with the inertial frame at the origin rather
    # than a composite COM, and no head shape is authored.
    if head_scale is None:
        body_path = prim_path if body_at_root else f"{prim_path}/{OBJECT_ROOT_LINK}"
        body = define(layer, body_path, "Xform",
                      ["PhysicsRigidBodyAPI", "PhysicsMassAPI",
                       "PhysxRigidBodyAPI"])
        _rigid_body_defaults(body)
        _hold_in_place(body, kinematic)
        attr(body, "physics:mass", Sdf.ValueTypeNames.Float, float(handle_mass))
        attr(body, "physics:diagonalInertia", Sdf.ValueTypeNames.Float3,
             Gf.Vec3f(float(handle_ixx), float(handle_iyy), float(handle_izz)))
        define(layer, f"{body_path}/collisions", "Xform")
        _shape_prim(layer, f"{body_path}/collisions/mesh_0", handle_scale,
                    (0.0, 0.0, 0.0), handle_rpy, collision=collision,
                    material_path=material_path)
        return (float(handle_mass), float(handle_ixx), float(handle_iyy),
                float(handle_izz), 0.0)

    # --- head -------------------------------------------------------------
    if len(head_scale) == 3:
        hlx = float(head_scale[0])
        x_offset = float(handle_scale[0]) / 2.0 + hlx / 2.0
        head_rpy = (0.0, 0.0, 0.0)
        head_mass, head_ixx, head_iyy, head_izz = _compute_mass_and_inertia(
            head_scale, head_density)
    else:
        hh, hd = (float(v) for v in head_scale)
        hr = hd / 2.0
        x_offset = float(handle_scale[0]) / 2.0 + hr
        head_rpy = (-math.pi / 2.0, 0.0, 0.0)
        # Rotated so the head axis is along +y: iyy <-> izz.
        head_mass, head_ixx, head_izz, head_iyy = _compute_mass_and_inertia(
            head_scale, head_density)

    # --- composite inertia about the joint COM ----------------------------
    total_mass = handle_mass + head_mass
    com_x = (head_mass * x_offset) / total_mass
    d_handle = -com_x
    d_head = x_offset - com_x
    ixx = handle_ixx + head_ixx
    iyy = (handle_iyy + handle_mass * d_handle ** 2) + (
        head_iyy + head_mass * d_head ** 2)
    izz = (handle_izz + handle_mass * d_handle ** 2) + (
        head_izz + head_mass * d_head ** 2)

    # body_at_root puts the rigid body ON prim_path instead of a child link.
    # The env addresses objects as /World/envs/env_N/Object, and an extra nesting
    # level would change what the RigidObject view resolves to.
    body_path = prim_path if body_at_root else f"{prim_path}/{OBJECT_ROOT_LINK}"
    body = define(layer, body_path, "Xform",
                  ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"])
    _rigid_body_defaults(body)
    _hold_in_place(body, kinematic)
    attr(body, "physics:mass", Sdf.ValueTypeNames.Float, float(total_mass))
    attr(body, "physics:diagonalInertia", Sdf.ValueTypeNames.Float3,
         Gf.Vec3f(float(ixx), float(iyy), float(izz)))
    attr(body, "physics:centerOfMass", Sdf.ValueTypeNames.Point3f,
         Gf.Vec3f(float(com_x), 0.0, 0.0))

    define(layer, f"{body_path}/collisions", "Xform")
    _shape_prim(layer, f"{body_path}/collisions/mesh_0", handle_scale,
                (0.0, 0.0, 0.0), handle_rpy, collision=collision,
                material_path=material_path)
    _shape_prim(layer, f"{body_path}/collisions/mesh_1", head_scale,
                (x_offset, 0.0, 0.0), head_rpy, collision=collision,
                material_path=material_path)

    return float(total_mass), float(ixx), float(iyy), float(izz), float(com_x)


__all__ = ["OBJECT_ROOT_LINK", "author_handle_head"]
