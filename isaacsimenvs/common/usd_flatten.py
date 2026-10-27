"""Flatten a converted robot USD into one layer, PhysX props set once and the render-only prims dropped.

Every robot the tasks spawn from a URDF goes through ``flatten_robot_usd`` (pose reaching's arm and hands, the
olympics hand). Needs Kit (pxr).
"""

from __future__ import annotations

from pathlib import Path

from isaacsimenvs.common.sdf import MAX_DEPEN_VELOCITY


def _strip_render_only(stage, root) -> tuple[int, int]:
    """Drop the render-only subtrees (per-link ``visuals``, the ``Looks`` scope).

    Every env references this file, and composition costs ~1.5 us per prim per
    env, so the visual meshes and their shaders cost minutes at 24k envs and
    PhysX never reads them. A subtree goes only if nothing under it carries
    physics, which keeps collision geometry and physics materials. The robot
    is invisible to a camera or the GUI afterwards.
    """
    from pxr import Usd, UsdPhysics

    def walk(prim):
        # The converter puts geometry behind instance proxies; a plain PrimRange
        # stops at the instance root and reports a subtree as empty.
        return Usd.PrimRange(prim, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))

    def physical(prim) -> bool:
        return any(p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI)
                   or p.HasAPI(UsdPhysics.ArticulationRootAPI)
                   or p.HasAPI(UsdPhysics.MaterialAPI) or p.IsA(UsdPhysics.Joint)
                   for p in walk(prim))

    def colliders() -> int:
        return sum(1 for p in walk(root) if p.HasAPI(UsdPhysics.CollisionAPI))

    before, colliders_before = sum(1 for _ in walk(root)), colliders()
    doomed, it = [], iter(walk(root))
    for prim in it:
        if prim.GetName() in ("visuals", "Looks") and not physical(prim):
            doomed.append(prim.GetPath())
            it.PruneChildren()
    for path in doomed:
        stage.RemovePrim(path)
    # The whole point is that only rendering is lost. A collider going with it
    # would be silent: the robot still loads, it just stops touching anything.
    if colliders() != colliders_before:
        raise RuntimeError(
            f"stripping render-only prims changed the collider count "
            f"({colliders_before} -> {colliders()}); refusing to build the scene")
    return before - sum(1 for _ in walk(root)), before


def keep_visuals_requested() -> bool:
    """``GENMECH_KEEP_VISUALS=1`` keeps the robot renderable.

    Stripping the render-only prims is a 24k-env throughput measure and it
    makes the robot invisible to any camera. A handful of envs being recorded
    to video wants the opposite trade, and at that scale the composition cost
    it buys back is irrelevant.
    """
    import os
    return os.environ.get("GENMECH_KEEP_VISUALS", "0") not in ("0", "", "false")


def flatten_robot_usd(usd_path: str, out_path: Path, *,
                      contact_offset: float, rest_offset: float,
                      strip_render_only: bool | None = None) -> tuple[str, str]:
    """Inline the converter's sub-layers into one file with the robot PhysX
    props set once and the render-only prims dropped; returns
    ``(path, root_prim_path)``. Nested references do not resolve a second level
    down, so an unflattened file composes with no colliders when an env prim
    references it.

    ``strip_render_only=False`` (or ``GENMECH_KEEP_VISUALS=1``) keeps the
    visual meshes so the robot can be rendered. The output then goes to a
    DIFFERENT filename, because a stripped and an unstripped flatten are not
    interchangeable and silently reusing one for the other would either cost
    throughput or produce an empty video.
    """
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

    if strip_render_only is None:
        strip_render_only = not keep_visuals_requested()
    if not strip_render_only:
        out_path = out_path.with_name(
            out_path.stem + "_vis" + out_path.suffix)

    stage = Usd.Stage.Open(usd_path, Usd.Stage.LoadAll)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stage.Flatten().Export(str(out_path))

    flat = Usd.Stage.Open(str(out_path))
    root = flat.GetDefaultPrim() or next(
        p for p in flat.GetPseudoRoot().GetChildren() if p.IsValid())
    removed = _strip_render_only(flat, root)[0] if strip_render_only else 0
    # Every remaining prim is composed once per env, so the breakdown says
    # where the next cut is: wrapper Xforms, colliders, joints.
    # Only prims outside a prototype are composed per env; instanced geometry
    # is shared, so it is free however many envs reference it.
    kinds: dict[str, int] = {}
    shared = 0
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        if prim.IsInstanceProxy():
            shared += 1
            continue
        kinds[prim.GetTypeName() or "(none)"] = kinds.get(prim.GetTypeName() or "(none)", 0) + 1
    breakdown = ", ".join(f"{k} x{v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
    print(f"[scene_utils] {out_path.name}: "
          f"{'dropped %d render-only prims' % removed if strip_render_only else 'KEPT render-only prims (renderable)'}; "
          f"{sum(kinds.values())} composed per env ({breakdown}), "
          f"{shared} more shared in prototypes", flush=True)
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            prim.CreateAttribute("physxRigidBody:disableGravity",
                                 Sdf.ValueTypeNames.Bool).Set(True)
            prim.CreateAttribute("physxRigidBody:maxDepenetrationVelocity",
                                 Sdf.ValueTypeNames.Float).Set(MAX_DEPEN_VELOCITY)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(prim)
            prim.CreateAttribute("physxArticulation:enabledSelfCollisions",
                                 Sdf.ValueTypeNames.Bool).Set(True)
            prim.CreateAttribute("physxArticulation:solverPositionIterationCount",
                                 Sdf.ValueTypeNames.Int).Set(8)
            prim.CreateAttribute("physxArticulation:solverVelocityIterationCount",
                                 Sdf.ValueTypeNames.Int).Set(0)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            PhysxSchema.PhysxCollisionAPI.Apply(prim)
            prim.CreateAttribute("physxCollision:contactOffset",
                                 Sdf.ValueTypeNames.Float).Set(contact_offset)
            prim.CreateAttribute("physxCollision:restOffset",
                                 Sdf.ValueTypeNames.Float).Set(rest_offset)
    flat.GetRootLayer().Save()
    return str(out_path), str(root.GetPath())


__all__ = ["flatten_robot_usd", "keep_visuals_requested"]
