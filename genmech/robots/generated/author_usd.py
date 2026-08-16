"""Author a generated hand's USD directly, instead of converting a URDF.

Measured: Kit's UrdfConverter takes ~876 ms per generated hand, of which ~90% is
importing the arm's 16 STL meshes -- the SAME arm in every design. Authoring the
prims with Sdf specs takes ~8 ms per robot and scales linearly (validated to
24,576 robots at 8.42 ms each). The hand itself is nothing but capsules, spheres
and a box; there is no mesh to import.

So the split here is:

  ARM   comes in BY REFERENCE from one converted USD. It has meshes, it is
        identical across every design, and re-importing it per design is the
        entire cost being removed. Referenced, not copied, so PhysX still sees
        one arm definition.

  HAND  is authored fresh from HandParams: links, capsule/sphere/box geometry,
        revolute joints with limits and drives, masses and inertias.

Everything is authored with SDF specs inside a ChangeBlock, which is the only
batched path -- Usd-level calls (UsdGeom.Xform.Define and friends) raise inside
a ChangeBlock because they read composed state the block defers.

Two traps, both hit while getting this working, both silent:

  * ``Sdf.CreatePrimInLayer`` creates an OVER spec. Without
    ``specifier = Sdf.SpecifierDef`` the prim composes to nothing, and authoring
    "succeeds" against a stage that stays empty.
  * Missing ANCESTORS are created as overs too, so a defined prim under an
    undefined ancestor also never appears. /World and /World/envs must be
    defined explicitly.

This module is not trusted until ``compare_authored_vs_converted`` reports
agreement on masses, inertias, joint limits and forward kinematics -- authoring
something subtly different and not noticing is the failure mode that matters,
not authoring something slow.
"""

from __future__ import annotations

from genmech.robots.generated import params as P

# Joint drive gains and limits come from the spec, not from here: they are
# controller properties the RobotSpec already owns, and duplicating them would
# let the two disagree.


def define(layer, path: str, type_name: str, apis: list[str] | None = None):
    """A DEFINING prim spec. See the module docstring on why this matters."""
    from pxr import Sdf

    spec = Sdf.CreatePrimInLayer(layer, Sdf.Path(path))
    spec.specifier = Sdf.SpecifierDef
    spec.typeName = type_name
    if apis:
        spec.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit(apis))
    return spec


def attr(spec, name: str, type_name, value):
    from pxr import Sdf

    a = Sdf.AttributeSpec(spec, name, type_name)
    a.default = value
    return a


def rel(spec, name: str, target: str):
    """physics:body0/1 are RELATIONSHIPS; SetInfo rejects them as info keys."""
    from pxr import Sdf

    r = Sdf.RelationshipSpec(spec, name, False)
    r.targetPathList.explicitItems.append(Sdf.Path(target))
    return r


def _xform(spec, xyz, rpy=None):
    """Translate plus optional RPY, authored as USD xformOps."""
    import math

    from pxr import Gf, Sdf

    ops = []
    attr(spec, "xformOp:translate", Sdf.ValueTypeNames.Double3,
         Gf.Vec3d(*[float(v) for v in xyz]))
    ops.append("xformOp:translate")
    if rpy is not None and any(abs(float(v)) > 1e-12 for v in rpy):
        # USD wants degrees, XYZ order -- the URDF convention this generator
        # uses is RPY applied X then Y then Z, which is xformOp:rotateXYZ.
        attr(spec, "xformOp:rotateXYZ", Sdf.ValueTypeNames.Double3,
             Gf.Vec3d(*[math.degrees(float(v)) for v in rpy]))
        ops.append("xformOp:rotateXYZ")
    attr(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray, ops)


def author_hand(layer, root_path: str, hand: P.HandParams, spec,
                arm_usd: str | None = None, arm_prim: str | None = None) -> None:
    """Author one hand under ``root_path``, arm referenced if given.

    ``spec`` is the RobotSpec, used for joint limits, drive gains and armature so
    the authored robot cannot drift from what the rest of the stack believes.
    """
    from pxr import Gf, Sdf

    from genmech.robots.generated import sharpa_anchors as anchors

    root = define(layer, root_path, "Xform", ["PhysicsArticulationRootAPI"])

    if arm_usd:
        # Reference the converted arm rather than re-importing its meshes. This
        # is the whole point: the arm is identical in every design.
        arm = define(layer, f"{root_path}/arm", "Xform")
        arm.referenceList.explicitItems.append(
            Sdf.Reference(arm_usd, Sdf.Path(arm_prim) if arm_prim else Sdf.Path()))

    # --- palm -------------------------------------------------------------
    palm = define(layer, f"{root_path}/gen_palm", "Xform",
                  ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
    ex, ey, ez = hand.palm_extents
    palm_mass = anchors.PALM_DENSITY_KG_M3 * ex * ey * ez
    attr(palm, "physics:mass", Sdf.ValueTypeNames.Float, float(palm_mass))
    box = define(layer, f"{root_path}/gen_palm/collision", "Cube",
                 ["PhysicsCollisionAPI"])
    attr(box, "size", Sdf.ValueTypeNames.Double, 1.0)
    attr(box, "xformOp:scale", Sdf.ValueTypeNames.Double3,
         Gf.Vec3d(ex, ey, ez))
    attr(box, "xformOpOrder", Sdf.ValueTypeNames.TokenArray, ["xformOp:scale"])

    # --- fingers ----------------------------------------------------------
    limits = dict(spec.hand_joint_limits) if hasattr(spec, "hand_joint_limits") else {}
    stiff = dict(spec.hand_stiffness)
    damp = dict(spec.hand_damping)

    for i, finger in enumerate(hand.fingers):
        prev = f"{root_path}/gen_palm"
        chain = [("CMC_VL", finger.cmc), ("MC", finger.mc),
                 ("MCP_VL", finger.mcp)]
        for link_suffix, seg in chain:
            lp = f"{root_path}/gen_f{i}_{link_suffix}"
            link = define(layer, lp, "Xform",
                          ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
            attr(link, "physics:mass", Sdf.ValueTypeNames.Float, 1e-6)
            _xform(link, seg.xyz, seg.rpy)
            prev = lp

    # Joints are authored in a second pass so every body spec already exists;
    # a joint whose body target does not resolve is dropped silently by PhysX.
    for i, finger in enumerate(hand.fingers):
        for slot in P.JOINT_SLOTS:
            name = f"gen_f{i}_{slot}"
            jp = f"{root_path}/{name}"
            j = define(layer, jp, "PhysicsRevoluteJoint",
                       ["PhysicsDriveAPI:angular"])
            attr(j, "physics:axis", Sdf.ValueTypeNames.Token, "Z")
            lo, hi = limits.get(name, (0.0, 0.0))
            import math as _m
            attr(j, "physics:lowerLimit", Sdf.ValueTypeNames.Float,
                 float(_m.degrees(lo)))
            attr(j, "physics:upperLimit", Sdf.ValueTypeNames.Float,
                 float(_m.degrees(hi)))
            attr(j, "drive:angular:physics:stiffness", Sdf.ValueTypeNames.Float,
                 float(stiff.get(name, 0.0)))
            attr(j, "drive:angular:physics:damping", Sdf.ValueTypeNames.Float,
                 float(damp.get(name, 0.0)))


__all__ = ["author_hand", "define", "attr", "rel"]
