"""What does the converter actually produce for a generated hand?

The authored path has to reproduce the CONVERTED asset, and the converter is
free to transform what the URDF said -- ``merge_fixed_joints=True`` folds
gen_palm into the arm's last link and each fingertip frame into its distal
phalanx, so the authored USD must describe the post-merge robot, not the URDF's
own link list.

Guessing that structure is how the object authoring lost hours: every attribute
matched and the assets still behaved differently, because the comparison was
against an assumed layout rather than the real one. So this dumps the real one.

Prints, for one converted generated hand:

  * every prim with a physics API, its type and its API schemas
  * rigid bodies: mass, diagonal inertia, centre of mass, principal axes
  * collision shapes: type, size/radius/height, and local transform
  * joints: type, body0/body1, local poses, axis, limits, drive gains
  * the articulation root

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.probe_hand_usd --hand gen_sharpa_like
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_args():
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hand", default="gen_sharpa_like",
                   help="generated hand name, or a population entry")
    p.add_argument("--max_prims", type=int, default=400)
    p.add_argument("--finger", type=int, default=0,
                   help="dump joints/bodies for this finger index in detail")
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    a.headless = True
    return a


def main() -> None:
    args = _parse_args()
    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app

    import tempfile
    from pathlib import Path

    from pxr import Usd

    from genmech.robots import get_robot_spec
    from genmech.tasks.pose_reach.utils.scene_utils import _convert_urdf_to_usd

    spec = get_robot_spec(args.hand)
    work = Path(tempfile.mkdtemp(prefix="genmech_handusd_"))
    print(f"[probe] hand={spec.name} urdf={spec.urdf_path}")

    usd = _convert_urdf_to_usd(
        spec.urdf_path, work, fix_base=True, self_collision=True,
        replace_cylinders_with_capsules=spec.replace_cylinders_with_capsules,
    )
    print(f"[probe] converted -> {usd}\n")

    stage = Usd.Stage.Open(usd)

    def apis(prim):
        info = prim.GetMetadata("apiSchemas")
        if not info:
            return []
        return [str(x) for x in info.GetAppliedItems()]

    def get(prim, name):
        a = prim.GetAttribute(name)
        return a.Get() if a and a.IsValid() else None

    bodies, joints, colliders, others = [], [], [], []
    for prim in stage.Traverse():
        ap = apis(prim)
        t = str(prim.GetTypeName())
        path = str(prim.GetPath())
        if any("RigidBodyAPI" in s for s in ap):
            bodies.append((path, t, ap))
        if "Joint" in t:
            joints.append((path, t))
        if any("CollisionAPI" in s for s in ap):
            colliders.append((path, t))
        if any("ArticulationRootAPI" in s for s in ap):
            others.append((path, t, ap))

    print(f"[probe] === STRUCTURE ===")
    print(f"[probe] articulation roots : {[p for p, _, _ in others]}")
    print(f"[probe] rigid bodies       : {len(bodies)}")
    print(f"[probe] joints             : {len(joints)}")
    print(f"[probe] collision shapes   : {len(colliders)}")

    fpref = f"gen_f{args.finger}_"
    print(f"\n[probe] === RIGID BODIES matching {fpref!r} ===")
    for path, t, ap in bodies:
        if fpref not in path:
            continue
        prim = stage.GetPrimAtPath(path)
        print(f"[probe] {path}")
        print(f"[probe]   type={t} apis={ap}")
        for k in ("physics:mass", "physics:diagonalInertia",
                  "physics:centerOfMass", "physics:principalAxes",
                  "xformOp:translate", "xformOp:orient", "xformOpOrder"):
            v = get(prim, k)
            if v is not None:
                print(f"[probe]   {k} = {v}")

    print(f"\n[probe] === COLLISION SHAPES matching {fpref!r} ===")
    for path, t in colliders:
        if fpref not in path:
            continue
        prim = stage.GetPrimAtPath(path)
        vals = {k: get(prim, k) for k in
                ("radius", "height", "axis", "size", "extent",
                 "xformOp:translate", "xformOp:orient", "xformOp:scale",
                 "xformOpOrder", "physics:approximation")}
        print(f"[probe] {path}  type={t}")
        for k, v in vals.items():
            if v is not None:
                print(f"[probe]   {k} = {v}")

    print(f"\n[probe] === JOINTS matching {fpref!r} ===")
    for path, t in joints:
        if fpref not in path:
            continue
        prim = stage.GetPrimAtPath(path)
        b0 = prim.GetRelationship("physics:body0")
        b1 = prim.GetRelationship("physics:body1")
        print(f"[probe] {path}  type={t}")
        print(f"[probe]   body0 = {b0.GetTargets() if b0 else None}")
        print(f"[probe]   body1 = {b1.GetTargets() if b1 else None}")
        for k in ("physics:axis", "physics:lowerLimit", "physics:upperLimit",
                  "physics:localPos0", "physics:localRot0",
                  "physics:localPos1", "physics:localRot1",
                  "physics:jointEnabled", "physics:excludeFromArticulation",
                  "drive:angular:physics:stiffness",
                  "drive:angular:physics:damping",
                  "drive:angular:physics:maxForce",
                  "drive:angular:physics:targetPosition",
                  "physxJoint:maxJointVelocity",
                  "physxArticulation:armature"):
            v = get(prim, k)
            if v is not None:
                print(f"[probe]   {k} = {v}")
        print(f"[probe]   apis = {apis(prim)}")

    print(f"\n[probe] === ALL BODY PATHS (first {args.max_prims}) ===")
    for path, _, _ in bodies[:args.max_prims]:
        print(f"[probe]   {path}")

    print(f"\n[probe] === ALL JOINT PATHS (first {args.max_prims}) ===")
    for path, t in joints[:args.max_prims]:
        print(f"[probe]   {path}  ({t})")

    del app
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
