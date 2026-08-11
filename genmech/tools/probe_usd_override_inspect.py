"""Step 1 of the USD-override question: what does UrdfConverter actually emit?

Before asking whether PhysX honours override layers, we need the real prim
paths and attribute names to override. Converts one robot and dumps the finger
chain: prim types, xform ops, joint local frames, mass/inertia attrs.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.probe_usd_override_inspect
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_spec", default="allegro_iiwa14")
    parser.add_argument("--chain", default="index",
                        help="finger name prefix to dump")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    from genmech.robots import get_robot_spec
    from genmech.utils.paths import resolve as resolve_repo_path

    spec = get_robot_spec(args.robot_spec)
    app = AppLauncher(args).app

    import tempfile

    from pxr import Usd, UsdGeom, UsdPhysics

    from genmech.tasks.pose_reach.utils.scene_utils import (
        _convert_urdf_to_usd,
        _robot_joint_drive_cfg,
    )

    work = Path(tempfile.mkdtemp(prefix="genmech_ovr_inspect_"))
    t0 = time.perf_counter()
    raw = _convert_urdf_to_usd(
        str(resolve_repo_path(spec.urdf_path)), work / "usd",
        fix_base=True, self_collision=True,
        joint_drive=_robot_joint_drive_cfg(),
    )
    convert_s = time.perf_counter() - t0
    print(f"\n[inspect] converted in {convert_s:.2f} s -> {raw}")

    stage = Usd.Stage.Open(raw)
    print(f"[inspect] default prim: {stage.GetDefaultPrim().GetPath()}")
    print(f"[inspect] root layer  : {stage.GetRootLayer().identifier}")
    print(f"[inspect] sublayers   : {stage.GetRootLayer().subLayerPaths}")

    print(f"\n[inspect] ---- prims matching '{args.chain}' ----")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if args.chain not in path:
            continue
        print(f"\n  {path}")
        print(f"    type      : {prim.GetTypeName()}")
        schemas = prim.GetAppliedSchemas()
        if schemas:
            print(f"    schemas   : {list(schemas)}")
        if prim.IsA(UsdGeom.Xformable):
            ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
            if ops:
                print(f"    xform ops : "
                      f"{[(o.GetOpName(), o.Get()) for o in ops]}")
        interesting = ("physics:localPos", "physics:localRot", "physics:mass",
                       "physics:diagonalInertia", "physics:principalAxes",
                       "physics:body0", "physics:body1", "physics:axis",
                       "xformOp:scale", "drive:")
        for attr in prim.GetAttributes():
            n = attr.GetName()
            if any(k in n for k in interesting):
                print(f"    {n:34s} = {attr.Get()}")
        for rel in prim.GetRelationships():
            if "body" in rel.GetName():
                print(f"    {rel.GetName():34s} -> {rel.GetTargets()}")

    print(f"\n[inspect] ---- articulation / instancing flags ----")
    for prim in stage.Traverse():
        if prim.IsInstanceable():
            print(f"  instanceable: {prim.GetPath()}")
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print(f"  articulation root: {prim.GetPath()}")

    del app
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
