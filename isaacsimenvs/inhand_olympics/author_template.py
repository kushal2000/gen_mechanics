"""Author one env's template robot (common/robots/template_robot.py) as USD prims. Needs ``pxr``.

Written to match the URDF converter's assets prim for prim (read off one, 29 Oct; see template_robot.py): bodies as
Xforms with the rigid-body, mass and PhysX APIs and their pose in the robot root frame; colliders under
``<body>/collisions/mesh_<i>``, a Cube or Capsule under a posed (and, for a box, scaled) Xform, or a convex Mesh;
joints under ``<root>/joints`` as revolute / prismatic joints with a force drive at zero gains (Isaac Lab's actuators
set the gains at runtime) and the URDF's effort and speed; the fixed base as ``root_joint`` carrying the
articulation root. Gravity off and the depenetration cap as usd_flatten.py sets them.
"""
from __future__ import annotations

import math

import numpy as np

from hand_sampler.build import principal_axes
from hand_sampler.design_space import mat_to_pos_quat
from isaacsimenvs.common.robots.template_robot import TemplateRobot
from isaacsimenvs.common.sdf import MAX_DEPEN_VELOCITY, attr, define, rel, set_xform


def _quat(sdf_mod, gf_mod, wxyz, single=True):
    w, x, y, z = (float(v) for v in wxyz)
    return gf_mod.Quatf(w, gf_mod.Vec3f(x, y, z)) if single else gf_mod.Quatd(w, gf_mod.Vec3d(x, y, z))


def _collision_offsets(shape, Sdf, contact_offset: float, rest_offset: float) -> None:
    attr(shape, "physxCollision:contactOffset", Sdf.ValueTypeNames.Float, float(contact_offset))
    attr(shape, "physxCollision:restOffset", Sdf.ValueTypeNames.Float, float(rest_offset))


def _author_collider(layer, path: str, c, contact_offset: float, rest_offset: float) -> None:
    from pxr import Gf, Sdf

    apis = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]
    if c.kind == "mesh":
        shape = define(layer, path, "Mesh", apis + ["PhysicsMeshCollisionAPI"])
        pts = np.asarray(c.points, float)
        attr(shape, "points", Sdf.ValueTypeNames.Point3fArray, [Gf.Vec3f(*(float(v) for v in p)) for p in pts])
        attr(shape, "faceVertexCounts", Sdf.ValueTypeNames.IntArray, [3] * len(c.faces))
        attr(shape, "faceVertexIndices", Sdf.ValueTypeNames.IntArray, [int(i) for f in c.faces for i in f])
        attr(shape, "physics:approximation", Sdf.ValueTypeNames.Token, "convexHull")
        attr(shape, "subdivisionScheme", Sdf.ValueTypeNames.Token, "none")
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        attr(shape, "extent", Sdf.ValueTypeNames.Float3Array,
             [Gf.Vec3f(*(float(v) for v in lo)), Gf.Vec3f(*(float(v) for v in hi))])
        _collision_offsets(shape, Sdf, contact_offset, rest_offset)
        return
    pos, quat = mat_to_pos_quat(c.pose)
    wrap = define(layer, path, "Xform")
    if c.kind == "box":
        lx, ly, lz = (float(v) for v in c.dims)
        set_xform(wrap, pos, quat, scale=(lx, ly, lz))
        shape = define(layer, f"{path}/box", "Cube", apis)
        attr(shape, "size", Sdf.ValueTypeNames.Double, 1.0)
        attr(shape, "extent", Sdf.ValueTypeNames.Float3Array,
             [Gf.Vec3f(-0.5 * lx, -0.5 * ly, -0.5 * lz), Gf.Vec3f(0.5 * lx, 0.5 * ly, 0.5 * lz)])
    elif c.kind == "capsule":
        h, r = (float(v) for v in c.dims)
        set_xform(wrap, pos, quat)
        shape = define(layer, f"{path}/capsule", "Capsule", apis)
        attr(shape, "height", Sdf.ValueTypeNames.Double, h)
        attr(shape, "radius", Sdf.ValueTypeNames.Double, r)
        attr(shape, "axis", Sdf.ValueTypeNames.Token, "Z")
        attr(shape, "extent", Sdf.ValueTypeNames.Float3Array,
             [Gf.Vec3f(-r, -r, -(h / 2 + r)), Gf.Vec3f(r, r, h / 2 + r)])
    else:
        raise ValueError(f"collider kind {c.kind!r}")
    _collision_offsets(shape, Sdf, contact_offset, rest_offset)


def author_template_robot(layer, root_path: str, robot: TemplateRobot, *, contact_offset: float,
                          rest_offset: float) -> None:
    """Write ``robot`` under ``root_path`` (an Xform made here; the caller places it)."""
    from pxr import Gf, Sdf

    define(layer, root_path, "Xform")
    define(layer, f"{root_path}/joints", "Scope")
    for b in robot.bodies:
        apis = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
        pairs = robot.filtered_pairs.get(b.name)
        if pairs:
            apis.append("PhysicsFilteredPairsAPI")
        body = define(layer, f"{root_path}/{b.name}", "Xform", apis)
        if pairs:
            rel(body, "physics:filteredPairs", [f"{root_path}/{p}" for p in pairs])
        diag, pquat = principal_axes(b.inertia)
        attr(body, "physics:mass", Sdf.ValueTypeNames.Float, float(b.mass))
        attr(body, "physics:diagonalInertia", Sdf.ValueTypeNames.Float3,
             Gf.Vec3f(*[float(max(v, 0.0)) for v in diag]))
        attr(body, "physics:principalAxes", Sdf.ValueTypeNames.Quatf, _quat(Sdf, Gf, pquat))
        attr(body, "physics:centerOfMass", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(*[float(v) for v in b.com]))
        attr(body, "physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool, True)
        attr(body, "physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float, MAX_DEPEN_VELOCITY)
        pos, quat = mat_to_pos_quat(b.pose)
        set_xform(body, pos, quat)
        if b.colliders:
            define(layer, f"{root_path}/{b.name}/collisions", "Xform")
            for i, c in enumerate(b.colliders):
                _author_collider(layer, f"{root_path}/{b.name}/collisions/mesh_{i}", c, contact_offset, rest_offset)
    for j in robot.joints:
        revolute = j.type == "revolute"
        drive = "angular" if revolute else "linear"
        prim = define(layer, f"{root_path}/joints/{j.name}",
                      "PhysicsRevoluteJoint" if revolute else "PhysicsPrismaticJoint",
                      [f"PhysicsDriveAPI:{drive}", "PhysxJointAPI"])
        rel(prim, "physics:body0", f"{root_path}/{j.body0}")
        rel(prim, "physics:body1", f"{root_path}/{j.body1}")
        pos0, quat0 = mat_to_pos_quat(j.frame0)
        rot1 = np.eye(4)
        rot1[:3, :3] = j.rot1
        _, quat1 = mat_to_pos_quat(rot1)
        attr(prim, "physics:localPos0", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(*[float(v) for v in pos0]))
        attr(prim, "physics:localRot0", Sdf.ValueTypeNames.Quatf, _quat(Sdf, Gf, quat0))
        attr(prim, "physics:localPos1", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(0.0, 0.0, 0.0))
        attr(prim, "physics:localRot1", Sdf.ValueTypeNames.Quatf, _quat(Sdf, Gf, quat1))
        attr(prim, "physics:axis", Sdf.ValueTypeNames.Token, "Z")
        scale = 180.0 / math.pi if revolute else 1.0          # USD: degrees for a hinge, metres for a slide
        attr(prim, "physics:lowerLimit", Sdf.ValueTypeNames.Float, float(j.lower * scale))
        attr(prim, "physics:upperLimit", Sdf.ValueTypeNames.Float, float(j.upper * scale))
        attr(prim, "physics:jointEnabled", Sdf.ValueTypeNames.Bool, True)
        attr(prim, "physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool, False)
        attr(prim, f"drive:{drive}:physics:type", Sdf.ValueTypeNames.Token, "force")
        attr(prim, f"drive:{drive}:physics:stiffness", Sdf.ValueTypeNames.Float, 0.0)
        attr(prim, f"drive:{drive}:physics:damping", Sdf.ValueTypeNames.Float, 0.0)
        attr(prim, f"drive:{drive}:physics:maxForce", Sdf.ValueTypeNames.Float, float(j.effort))
        attr(prim, f"drive:{drive}:physics:targetPosition", Sdf.ValueTypeNames.Float, 0.0)
        attr(prim, "physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float, float(j.velocity * scale))
    # the fixed base, carrying the articulation root (the converter's fix_base layout)
    base = define(layer, f"{root_path}/root_joint", "PhysicsFixedJoint",
                  ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"])
    rel(base, "physics:body1", f"{root_path}/{robot.bodies[0].name}")
    for k in ("0", "1"):
        attr(base, f"physics:localPos{k}", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(0.0, 0.0, 0.0))
        attr(base, f"physics:localRot{k}", Sdf.ValueTypeNames.Quatf, Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    attr(base, "physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool, True)
    attr(base, "physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int, 8)
    attr(base, "physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int, 0)


__all__ = ["author_template_robot"]
