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


def _shape_prim(layer, path: str, scale, xyz, rpy, collision: bool = True):
    """One handle/head shape: Cube for a 3-tuple, Cylinder for a 2-tuple."""
    from pxr import Gf, Sdf

    from genmech.robots.generated.author_usd import attr, define

    is_box = len(scale) == 3
    # CAPSULE, not Cylinder. The training env converts objects with
    # replace_cylinders_with_capsules=True, so a URDF cylinder becomes a capsule
    # whose `height` is the CYLINDRICAL SECTION with a hemisphere added at each
    # end -- total extent height + 2*radius. generate_objects already computes
    # the inertia on that basis ("Capsule-approximation for cylinders"), so the
    # capsule is the shape the whole pipeline means. Emitting a plain cylinder
    # here made objects rest up to 0.53 mm differently from the converted ones.
    # The goal marker is the same shape with no collider, matching how the
    # converted path bakes goalviz with collision_enabled=False.
    prim = define(layer, path, "Cube" if is_box else "Capsule",
                  ["PhysicsCollisionAPI"] if collision else None)
    if is_box:
        lx, ly, lz = (float(v) for v in scale)
        # UsdGeom.Cube has a single `size`; non-uniform box dimensions have to
        # come from a scale op, which is how the converter expresses them too.
        attr(prim, "size", Sdf.ValueTypeNames.Double, 1.0)
        ops = [("xformOp:scale", Sdf.ValueTypeNames.Double3, Gf.Vec3d(lx, ly, lz))]
    else:
        h, d = (float(v) for v in scale)
        attr(prim, "height", Sdf.ValueTypeNames.Double, h)
        attr(prim, "radius", Sdf.ValueTypeNames.Double, d / 2.0)
        # URDF cylinders are z-axis; the rpy below rotates them into place, so
        # the prim keeps the same local axis the URDF geometry had.
        attr(prim, "axis", Sdf.ValueTypeNames.Token, "Z")
        ops = []

    if any(abs(float(v)) > 1e-12 for v in xyz):
        ops.insert(0, ("xformOp:translate", Sdf.ValueTypeNames.Double3,
                       Gf.Vec3d(*[float(v) for v in xyz])))
    if any(abs(float(v)) > 1e-12 for v in rpy):
        idx = 1 if ops and ops[0][0] == "xformOp:translate" else 0
        ops.insert(idx, ("xformOp:rotateXYZ", Sdf.ValueTypeNames.Double3,
                         Gf.Vec3d(*[math.degrees(float(v)) for v in rpy])))
    for name, tn, value in ops:
        attr(prim, name, tn, value)
    if ops:
        attr(prim, "xformOpOrder", Sdf.ValueTypeNames.TokenArray,
             [name for name, _, _ in ops])
    return prim


def author_handle_head(layer, prim_path: str, handle_scale, head_scale,
                       handle_density: float, head_density: float,
                       body_at_root: bool = False, collision: bool = True):
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
                  ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
    attr(body, "physics:mass", Sdf.ValueTypeNames.Float, float(total_mass))
    attr(body, "physics:diagonalInertia", Sdf.ValueTypeNames.Float3,
         Gf.Vec3f(float(ixx), float(iyy), float(izz)))
    attr(body, "physics:centerOfMass", Sdf.ValueTypeNames.Point3f,
         Gf.Vec3f(float(com_x), 0.0, 0.0))

    _shape_prim(layer, f"{body_path}/handle", handle_scale, (0.0, 0.0, 0.0),
                handle_rpy, collision=collision)
    _shape_prim(layer, f"{body_path}/head", head_scale, (x_offset, 0.0, 0.0),
                head_rpy, collision=collision)

    return float(total_mass), float(ixx), float(iyy), float(izz), float(com_x)


__all__ = ["OBJECT_ROOT_LINK", "author_handle_head"]
