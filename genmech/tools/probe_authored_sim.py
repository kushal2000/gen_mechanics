"""Do directly-authored robots actually simulate, and how fast can we make them?

probe_authoring_rate measured ~5 ms per 47-link robot for Sdf-level authoring
inside a ChangeBlock -- 250x faster than the convert-to-file path -- but its
prim count came back near zero, which means the specs may not have defined
anything. ``Sdf.CreatePrimInLayer`` creates an OVER spec by default; without
``specifier = Sdf.SpecifierDef`` the prim never materialises on the stage, and
timing the authoring of nothing is easy and meaningless.

So this checks the whole chain rather than the clock:

  1. author k robots, each with DISTINCT geometry, via Sdf specs in one
     ChangeBlock
  2. confirm the prims exist on the composed stage
  3. hand them to PhysX as an Articulation view and confirm the joint count
  4. drive the joints and confirm they move

Only if all four hold is the timing meaningful. A fast build that yields no
articulation, or an articulation whose joints do not move, is not a speed-up --
it is the same mistake as a scene that builds quickly because the assets landed
in env_0.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.probe_authored_sim --robots 64
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Small enough to read at a glance, large enough that PhysX must build a real
# articulation: a base plus a 3-link finger chain.
N_LINKS = 4


def _parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robots", type=int, default=64)
    parser.add_argument("--spacing", type=float, default=1.0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    return args


def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app

    import omni.usd
    import torch
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    layer = stage.GetRootLayer()

    def define(path: str, type_name: str, apis: list[str] | None = None):
        """A DEFINING spec. The default from CreatePrimInLayer is an 'over',
        which composes to nothing -- the bug this probe exists to rule out."""
        spec = Sdf.CreatePrimInLayer(layer, Sdf.Path(path))
        spec.specifier = Sdf.SpecifierDef
        spec.typeName = type_name
        if apis:
            spec.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit(apis))
        return spec

    def attr(spec, name, type_name, value):
        a = Sdf.AttributeSpec(spec, name, type_name)
        a.default = value
        return a

    def rel(spec, name, target):
        """physics:body0/1 are RELATIONSHIPS. SetInfo rejects them as info keys;
        they need a RelationshipSpec with explicit targets."""
        r = Sdf.RelationshipSpec(spec, name, False)
        r.targetPathList.explicitItems.append(Sdf.Path(target))
        return r

    t0 = time.perf_counter()
    with Sdf.ChangeBlock():
        # Every ancestor must be DEFINED, not an over. CreatePrimInLayer creates
        # missing ancestors as overs, and a prim whose ancestor is undefined does
        # not compose onto the stage -- which is why the first attempt authored
        # 64 robots into a stage that ended up holding one prim.
        for ancestor in ("/World", "/World/envs"):
            spec = Sdf.CreatePrimInLayer(layer, Sdf.Path(ancestor))
            spec.specifier = Sdf.SpecifierDef
            spec.typeName = "Xform"
        for r in range(args.robots):
            base = f"/World/envs/env_{r}/Robot"
            define(f"/World/envs/env_{r}", "Xform")
            root = define(base, "Xform", ["PhysicsArticulationRootAPI"])
            attr(root, "xformOp:translate", Sdf.ValueTypeNames.Double3,
                 Gf.Vec3d(r * args.spacing, 0.0, 1.0))
            attr(root, "xformOpOrder", Sdf.ValueTypeNames.TokenArray,
                 ["xformOp:translate"])

            # Geometry differs per robot, so nothing can be shared or replicated
            # -- this is the k = n case.
            radius = 0.02 + 0.0005 * (r % 8)
            length = 0.10 + 0.01 * (r % 5)

            for i in range(N_LINKS):
                lp = f"{base}/link_{i}"
                link = define(lp, "Xform",
                              ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
                attr(link, "physics:mass", Sdf.ValueTypeNames.Float, 0.05)
                attr(link, "physics:diagonalInertia", Sdf.ValueTypeNames.Float3,
                     Gf.Vec3f(1e-4, 1e-4, 1e-4))
                attr(link, "xformOp:translate", Sdf.ValueTypeNames.Double3,
                     Gf.Vec3d(0.0, 0.0, -i * length))
                attr(link, "xformOpOrder", Sdf.ValueTypeNames.TokenArray,
                     ["xformOp:translate"])

                col = define(f"{lp}/collision", "Capsule",
                             ["PhysicsCollisionAPI"])
                attr(col, "radius", Sdf.ValueTypeNames.Double, radius)
                attr(col, "height", Sdf.ValueTypeNames.Double, length)
                attr(col, "axis", Sdf.ValueTypeNames.Token, "Z")

                if i == 0:
                    j = define(f"{base}/joint_root", "PhysicsFixedJoint")
                    rel(j, "physics:body1", lp)
                else:
                    jp = f"{base}/joint_{i}"
                    j = define(jp, "PhysicsRevoluteJoint",
                               ["PhysicsDriveAPI:angular"])
                    rel(j, "physics:body0", f"{base}/link_{i-1}")
                    rel(j, "physics:body1", lp)
                    attr(j, "physics:axis", Sdf.ValueTypeNames.Token, "X")
                    attr(j, "physics:localPos0", Sdf.ValueTypeNames.Point3f,
                         Gf.Vec3f(0.0, 0.0, -length))
                    attr(j, "physics:localPos1", Sdf.ValueTypeNames.Point3f,
                         Gf.Vec3f(0.0, 0.0, 0.0))
                    attr(j, "physics:lowerLimit", Sdf.ValueTypeNames.Float, -60.0)
                    attr(j, "physics:upperLimit", Sdf.ValueTypeNames.Float, 60.0)
                    attr(j, "drive:angular:physics:stiffness",
                         Sdf.ValueTypeNames.Float, 100.0)
                    attr(j, "drive:angular:physics:damping",
                         Sdf.ValueTypeNames.Float, 10.0)
    author_s = time.perf_counter() - t0

    # ---- 2. did the prims materialise? ------------------------------------
    n_prims = len(list(stage.Traverse()))
    robots_found = len([p for p in stage.Traverse()
                        if p.GetPath().pathString.endswith("/Robot")])
    joints_found = len([p for p in stage.Traverse()
                        if p.GetTypeName() == "PhysicsRevoluteJoint"])
    print(f"[authsim] authored {args.robots} robots in {author_s * 1000:.0f} ms "
          f"({author_s / args.robots * 1000:.2f} ms/robot)")
    print(f"[authsim] stage: {n_prims} prims, {robots_found} Robot prims, "
          f"{joints_found} revolute joints "
          f"(expected {args.robots} and {args.robots * (N_LINKS - 1)})")
    if robots_found != args.robots:
        print("[authsim] FAIL: prims did not materialise -- specs composed to nothing")
        del app
        os._exit(1)

    # ---- 3/4. does PhysX build it, and do the joints move? ----------------
    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=args.device))
    robot = Articulation(ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=None,          # already on the stage; nothing to spawn
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"],
                                              stiffness=100.0, damping=10.0)},
    ))
    t0 = time.perf_counter()
    sim.reset()
    reset_s = time.perf_counter() - t0
    print(f"[authsim] PhysX reset {reset_s:.2f}s "
          f"({reset_s / args.robots * 1000:.2f} ms/robot)")
    print(f"[authsim] articulation view: {robot.num_instances} instances x "
          f"{robot.num_joints} joints")

    q0 = robot.data.joint_pos.clone()
    target = q0 + 0.4
    robot.set_joint_position_target(target)
    for _ in range(60):
        robot.write_data_to_sim()
        sim.step()
        robot.update(1 / 120.0)
    moved = float((robot.data.joint_pos - q0).abs().max())
    print(f"[authsim] max joint motion under target: {moved:.4f} rad "
          f"({'MOVES' if moved > 0.05 else 'DID NOT MOVE'})")

    ok = (robots_found == args.robots
          and robot.num_instances == args.robots
          and robot.num_joints == N_LINKS - 1
          and moved > 0.05)
    print(f"[authsim] {'PASS' if ok else 'FAIL'}: authored robots "
          f"{'simulate correctly' if ok else 'are not usable'}")

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
