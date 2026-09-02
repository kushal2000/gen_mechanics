"""Sdf spec helpers for authoring prims straight into a layer.

Trap: ``Sdf.CreatePrimInLayer`` creates OVERs, for the prim and for any
missing ancestor. Without ``specifier = SpecifierDef`` the prim composes to
nothing and authoring silently succeeds against an empty stage.
"""

from __future__ import annotations

# What the converter sets at spawn (PhysX default is 3.0).
MAX_DEPEN_VELOCITY: float = 1000.0


def define(layer, path: str, type_name: str, apis: list[str] | None = None):
    """A DEFINING prim spec (see the module docstring)."""
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
    """A relationship (physics:body0/1 are relationships, not info keys)."""
    from pxr import Sdf

    r = Sdf.RelationshipSpec(spec, name, False)
    r.targetPathList.explicitItems.append(Sdf.Path(target))
    return r


def set_xform(spec, xyz, quat_wxyz=None, scale=(1.0, 1.0, 1.0)):
    """translate + orient + scale, in the converter's op order."""
    from pxr import Gf, Sdf

    attr(spec, "xformOp:translate", Sdf.ValueTypeNames.Double3,
         Gf.Vec3d(*[float(v) for v in xyz]))
    ops = ["xformOp:translate"]
    if quat_wxyz is not None:
        w, x, y, z = (float(v) for v in quat_wxyz)
        attr(spec, "xformOp:orient", Sdf.ValueTypeNames.Quatd, Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        ops.append("xformOp:orient")
    attr(spec, "xformOp:scale", Sdf.ValueTypeNames.Double3,
         Gf.Vec3d(*[float(v) for v in scale]))
    ops.append("xformOp:scale")
    attr(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray, ops)


__all__ = ["MAX_DEPEN_VELOCITY", "attr", "define", "rel", "set_xform"]
