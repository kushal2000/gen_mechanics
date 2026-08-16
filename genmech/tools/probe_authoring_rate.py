"""How fast can we author N DISTINCT robots straight into the stage?

Target: k = n = 24,576, every env a different hand. The current path costs ~30
hours, and almost all of it is machinery that is pointless when each design is
used exactly once:

  * convert each design to a USD FILE            (1.1 s each -> 7.5 h)
  * spawn each file into a template proto        (113 ms each -> 46 min)
  * Sdf.CopySpec the proto into its one env      (a copy of a thing used once)

None of that is needed to put a robot in a stage. The irreducible costs are
authoring the prims and letting PhysX parse them (~13 ms/env, and with every env
distinct there is nothing to replicate).

So: what is the authoring floor? Two APIs, and the difference matters:

  USD-LEVEL   UsdGeom.Xform.Define / UsdPhysics.*.Apply. Convenient, and what a
              from-scratch authoring pass would reach for first. Cannot be used
              inside an Sdf.ChangeBlock -- Usd-level calls read composed state
              that a change block defers, which is exactly why
              spawn_multi_asset batches only its CopySpec loop.

  SDF-LEVEL   Sdf.CreatePrimInLayer + PrimSpec/AttributeSpec, inside a
              ChangeBlock. Legal to batch, so change notifications are processed
              once instead of per edit. This is the path Isaac Lab's own fast
              loop uses.

Measures prims/second for both, at several robot counts, and reports whether the
rate is flat in n (authoring is local) or degrades (something scans the stage).

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.probe_authoring_rate --counts 64 256 1024
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# A generated hand+arm is ~47 links / 44 joints / ~400 prims once visuals and
# collision geometry are counted (measured by probe_stage_census).
LINKS_PER_ROBOT = 47
PRIMS_PER_LINK = 4          # link Xform + collision + 2 visual


def _parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--out", default="authoring_rate.json")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    return args


def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app

    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    def fresh_stage():
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World")
        return stage

    def author_usd_level(stage, n_robots: int) -> None:
        """The convenient API. Cannot be batched."""
        for r in range(n_robots):
            base = f"/World/envs/env_{r}/Robot"
            root = UsdGeom.Xform.Define(stage, base)
            UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
            for i in range(LINKS_PER_ROBOT):
                link = UsdGeom.Xform.Define(stage, f"{base}/link_{i}")
                UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
                m = UsdPhysics.MassAPI.Apply(link.GetPrim())
                m.CreateMassAttr(0.04)
                cap = UsdGeom.Capsule.Define(stage, f"{base}/link_{i}/col")
                cap.CreateRadiusAttr(0.01 + 0.0001 * r)   # distinct per robot
                cap.CreateHeightAttr(0.03)
                UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
                UsdGeom.Sphere.Define(stage, f"{base}/link_{i}/v0")
                UsdGeom.Sphere.Define(stage, f"{base}/link_{i}/v1")
                if i:
                    j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/j_{i}")
                    j.CreateBody0Rel().SetTargets([f"{base}/link_{i-1}"])
                    j.CreateBody1Rel().SetTargets([f"{base}/link_{i}"])
                    j.CreateAxisAttr("Z")
                    j.CreateLocalPos0Attr(Gf.Vec3f(0.03 + 0.0001 * r, 0, 0))

    def author_sdf_level(stage, n_robots: int) -> None:
        """Sdf specs inside one ChangeBlock: batched change notification."""
        layer = stage.GetRootLayer()
        with Sdf.ChangeBlock():
            for r in range(n_robots):
                base = f"/World/envs/env_{r}/Robot"
                Sdf.CreatePrimInLayer(layer, Sdf.Path(base))
                root = layer.GetPrimAtPath(Sdf.Path(base))
                root.typeName = "Xform"
                root.SetInfo("apiSchemas",
                             Sdf.TokenListOp.CreateExplicit(["PhysicsArticulationRootAPI"]))
                for i in range(LINKS_PER_ROBOT):
                    lp = Sdf.Path(f"{base}/link_{i}")
                    Sdf.CreatePrimInLayer(layer, lp)
                    link = layer.GetPrimAtPath(lp)
                    link.typeName = "Xform"
                    link.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit(
                        ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]))
                    mass = Sdf.AttributeSpec(link, "physics:mass",
                                             Sdf.ValueTypeNames.Float)
                    mass.default = 0.04

                    cp = Sdf.Path(f"{base}/link_{i}/col")
                    Sdf.CreatePrimInLayer(layer, cp)
                    col = layer.GetPrimAtPath(cp)
                    col.typeName = "Capsule"
                    col.SetInfo("apiSchemas",
                                Sdf.TokenListOp.CreateExplicit(["PhysicsCollisionAPI"]))
                    rad = Sdf.AttributeSpec(col, "radius", Sdf.ValueTypeNames.Double)
                    rad.default = 0.01 + 0.0001 * r
                    hgt = Sdf.AttributeSpec(col, "height", Sdf.ValueTypeNames.Double)
                    hgt.default = 0.03

                    for v in ("v0", "v1"):
                        vp = Sdf.Path(f"{base}/link_{i}/{v}")
                        Sdf.CreatePrimInLayer(layer, vp)
                        layer.GetPrimAtPath(vp).typeName = "Sphere"

                    if i:
                        jp = Sdf.Path(f"{base}/j_{i}")
                        Sdf.CreatePrimInLayer(layer, jp)
                        jt = layer.GetPrimAtPath(jp)
                        jt.typeName = "PhysicsRevoluteJoint"
                        ax = Sdf.AttributeSpec(jt, "physics:axis",
                                               Sdf.ValueTypeNames.Token)
                        ax.default = "Z"
                        lp0 = Sdf.AttributeSpec(jt, "physics:localPos0",
                                                Sdf.ValueTypeNames.Point3f)
                        lp0.default = Gf.Vec3f(0.03 + 0.0001 * r, 0, 0)

    rows = []
    print(f"[auth] {LINKS_PER_ROBOT} links/robot, ~{LINKS_PER_ROBOT * PRIMS_PER_LINK} prims/robot")
    print(f"[auth] {'robots':>8} {'usd-level':>12} {'sdf-level':>12} "
          f"{'usd prims/s':>13} {'sdf prims/s':>13} {'speedup':>8}")
    for n in args.counts:
        stage = fresh_stage()
        t0 = time.perf_counter()
        author_usd_level(stage, n)
        usd_s = time.perf_counter() - t0
        n_prims_usd = len(list(stage.Traverse()))

        stage = fresh_stage()
        t0 = time.perf_counter()
        author_sdf_level(stage, n)
        sdf_s = time.perf_counter() - t0
        n_prims_sdf = len(list(stage.Traverse()))

        rows.append(dict(robots=n, usd_s=usd_s, sdf_s=sdf_s,
                         prims_usd=n_prims_usd, prims_sdf=n_prims_sdf))
        print(f"[auth] {n:>8} {usd_s:>11.2f}s {sdf_s:>11.2f}s "
              f"{n_prims_usd / usd_s:>13,.0f} {n_prims_sdf / sdf_s:>13,.0f} "
              f"{usd_s / max(sdf_s, 1e-9):>7.1f}x")

    # Extrapolate the winner to the target, and say plainly which it is.
    best = min(rows[-1]["usd_s"], rows[-1]["sdf_s"])
    per_robot = best / rows[-1]["robots"]
    print(f"\n[auth] best rate: {per_robot * 1000:.2f} ms/robot")
    print(f"[auth] authoring 24,576 distinct robots -> {per_robot * 24576:.0f}s "
          f"({per_robot * 24576 / 60:.1f} min)")
    print(f"[auth] for comparison: current path is ~1.1s conversion + 0.113s "
          f"template spawn per design = {(1.1 + 0.113) * 24576 / 3600:.1f} hours")

    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
