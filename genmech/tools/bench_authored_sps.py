"""k = n: every env a DIFFERENT robot, authored straight into the stage.

The convert-to-file path cannot reach this point: at k = n = 24,576 it needs
~7.5 h of URDF->USD conversion plus ~46 min of template spawning, before a
quadratic-looking copy/parse term. All of that machinery exists to share assets
between envs, and at k = n nothing is shared -- each design is used exactly once.

probe_authored_sim established the alternative works: author each robot's prims
directly with Sdf specs inside one ChangeBlock, no URDF, no USD file, no
converter, no template proto, no CopySpec. 64 distinct 4-link robots authored in
46 ms, PhysX built one articulation view over all of them, and the joints
tracked their targets.

This scales that to a robot of comparable complexity to a generated hand+arm
(~44 joints) and reports STEP RATE, so the number is comparable with the
convert-path benchmarks: 42.9 steps/s for 64 designs over 24,576 envs.

Two honest differences from the real thing, both of which affect the physics:

  * the chain is SERIAL, where a hand branches into fingers. Same body and joint
    count, different articulation topology, so PhysX's solve is not identical.
  * no meshes, no self-collision filters, no arm. This measures what authoring
    plus PhysX cost, not a drop-in replacement for the production asset.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.bench_authored_sps --robots 24576 --links 44
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robots", type=int, default=24576,
                        help="k = n: this many envs, each a distinct robot")
    parser.add_argument("--links", type=int, default=44,
                        help="links per robot; a generated hand+arm is ~47")
    parser.add_argument("--distinct", type=int, default=0,
                        help="number of DISTINCT geometries, cycled across envs "
                             "(0 = every env unique, i.e. k=n). Holding topology "
                             "and env count fixed while varying only this "
                             "separates the cost of diversity from the cost of "
                             "the robot itself.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--spacing", type=float, default=1.3)
    parser.add_argument("--out", default="authored_sps.json")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    return args


def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher

    print(f"[auth-sps] robots=k=n={args.robots} links={args.links}")
    app = AppLauncher(args).app

    import math

    import omni.usd
    import torch
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from pxr import Gf, Sdf, UsdGeom

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    layer = stage.GetRootLayer()

    def define(path, type_name, apis=None):
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
        r = Sdf.RelationshipSpec(spec, name, False)
        r.targetPathList.explicitItems.append(Sdf.Path(target))
        return r

    # Envs on a square grid, matching how InteractiveScene lays them out, so the
    # robots are spread over the ground rather than stacked at one point.
    side = int(math.ceil(math.sqrt(args.robots)))

    t0 = time.perf_counter()
    with Sdf.ChangeBlock():
        # Ancestors must be DEFINED: CreatePrimInLayer makes missing ancestors
        # 'over' specs, and a prim under an undefined ancestor never composes.
        for ancestor in ("/World", "/World/envs"):
            define(ancestor, "Xform")

        for r in range(args.robots):
            base = f"/World/envs/env_{r}/Robot"
            env = define(f"/World/envs/env_{r}", "Xform")
            attr(env, "xformOp:translate", Sdf.ValueTypeNames.Double3,
                 Gf.Vec3d((r % side) * args.spacing,
                          (r // side) * args.spacing, 0.0))
            attr(env, "xformOpOrder", Sdf.ValueTypeNames.TokenArray,
                 ["xformOp:translate"])

            root = define(base, "Xform", ["PhysicsArticulationRootAPI"])
            attr(root, "xformOp:translate", Sdf.ValueTypeNames.Double3,
                 Gf.Vec3d(0.0, 0.0, 1.5))
            attr(root, "xformOpOrder", Sdf.ValueTypeNames.TokenArray,
                 ["xformOp:translate"])

            # Distinct geometry per robot: this is the k = n case, so nothing
            # may be shared, replicated or instanced between envs.
            # With --distinct D, only D geometries exist and they repeat.
            g = r if args.distinct <= 0 else (r % args.distinct)
            radius = 0.015 + 0.0005 * (g % 11) + 1e-6 * (g // 11)
            length = 0.05 + 0.004 * (g % 7) + 1e-6 * (g // 7)

            for i in range(args.links):
                lp = f"{base}/link_{i}"
                link = define(lp, "Xform", ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
                attr(link, "physics:mass", Sdf.ValueTypeNames.Float, 0.04)
                attr(link, "physics:diagonalInertia", Sdf.ValueTypeNames.Float3,
                     Gf.Vec3f(1e-5, 1e-5, 1e-5))
                attr(link, "xformOp:translate", Sdf.ValueTypeNames.Double3,
                     Gf.Vec3d(0.0, 0.0, -i * length))
                attr(link, "xformOpOrder", Sdf.ValueTypeNames.TokenArray,
                     ["xformOp:translate"])

                col = define(f"{lp}/collision", "Capsule", ["PhysicsCollisionAPI"])
                attr(col, "radius", Sdf.ValueTypeNames.Double, radius)
                attr(col, "height", Sdf.ValueTypeNames.Double, length)
                attr(col, "axis", Sdf.ValueTypeNames.Token, "Z")

                if i == 0:
                    j = define(f"{base}/joint_root", "PhysicsFixedJoint")
                    rel(j, "physics:body1", lp)
                else:
                    j = define(f"{base}/joint_{i}", "PhysicsRevoluteJoint",
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
                         Sdf.ValueTypeNames.Float, 50.0)
                    attr(j, "drive:angular:physics:damping",
                         Sdf.ValueTypeNames.Float, 5.0)
    author_s = time.perf_counter() - t0
    print(f"[auth-sps] authored in {author_s:.1f}s "
          f"({author_s / args.robots * 1000:.2f} ms/robot)")

    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=args.device))
    robot = Articulation(ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=None,          # already authored onto the stage
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"],
                                              stiffness=50.0, damping=5.0)},
    ))
    t0 = time.perf_counter()
    sim.reset()
    reset_s = time.perf_counter() - t0
    print(f"[auth-sps] PhysX reset {reset_s:.1f}s "
          f"({reset_s / args.robots * 1000:.2f} ms/robot)")
    print(f"[auth-sps] view: {robot.num_instances} instances x "
          f"{robot.num_joints} joints")
    if robot.num_instances != args.robots:
        print(f"[auth-sps] FAIL: expected {args.robots} instances")
        del app
        os._exit(1)

    q0 = robot.data.joint_pos.clone()
    robot.set_joint_position_target(q0 + 0.2)
    for _ in range(args.warmup):
        robot.write_data_to_sim()
        sim.step()
        robot.update(1 / 120.0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        robot.write_data_to_sim()
        sim.step()
        robot.update(1 / 120.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    moved = float((robot.data.joint_pos - q0).abs().max())
    result = dict(robots=args.robots, links=args.links,
                  distinct=args.distinct or args.robots,
                  joints=int(robot.num_joints), instances=int(robot.num_instances),
                  author_s=author_s, reset_s=reset_s, setup_s=author_s + reset_s,
                  sps=args.steps / dt, env_sps=args.steps * args.robots / dt,
                  max_joint_motion=moved)
    print(f"\n[auth-sps] TOTAL SETUP {result['setup_s']:.1f}s "
          f"(author {author_s:.1f}s + reset {reset_s:.1f}s)")
    print(f"[auth-sps] {result['sps']:.1f} steps/s, "
          f"{result['env_sps']:,.0f} env-steps/s")
    print(f"[auth-sps] joints moved {moved:.4f} rad "
          f"({'live' if moved > 0.02 else 'STATIC -- suspect'})")

    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
