"""Can morphology be authored as USD overrides instead of N converted USDs?

The co-design proposal (docs/proposal_codesign.md §5) costs 1.2 s of URDF->USD
conversion per unique embodiment -- 15.3 hours for a 24,576-robot population,
which dominates the RL cost. If a variant can instead be a thin USD layer that
*references* one converted base and overrides a few attributes, that cost
collapses to a single conversion plus N cheap attribute writes.

The question is not whether USD composes -- it does -- but whether **PhysX**
honours the composed result. Three things are tested independently:

  A. Link length via ``physics:localPos0`` on the joint prim. The converter
     emits links as flat siblings, so all kinematics live on the joints:
     localPos0 is the joint frame in the parent body, i.e. exactly the URDF
     joint origin, i.e. exactly the link length. Nothing is rescaled, so
     collision shapes are reused unchanged and no recooking is implied.
  B. Inertial properties via ``physics:mass`` on the link prim.
  C. Geometry size via ``xformOp:scale`` on the link prim. This is the risky
     one: the visual will follow, but if PhysX cooks collision from the base
     layer the robot silently collides at the wrong size.

Test A has a ground truth. probe_heterogeneous_envs.py produced the same
morphology change through the honest path -- editing the URDF and running a
full conversion -- and measured palm->fingertip 0.3022 m (nominal) vs 0.3545 m
(finger origins x1.35). If the override route reproduces those numbers, the two
routes agree and the cheap one is sound.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.probe_usd_override
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ground truth from probe_heterogeneous_envs.py --robot_spec allegro_iiwa14
# --finger_scale 1.35, which reached these via full per-variant conversion.
GROUND_TRUTH_NOMINAL = 0.3022
GROUND_TRUTH_SCALED = 0.3545


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_spec", default="allegro_iiwa14")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--finger_scale", type=float, default=1.35)
    parser.add_argument("--mass_scale", type=float, default=3.0)
    parser.add_argument("--geom_scale", type=float, default=1.5,
                        help="xformOp:scale factor for test C")
    parser.add_argument("--n_timing", type=int, default=64,
                        help="variants to author when timing the override path")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    from genmech.robots import get_robot_spec
    from genmech.utils.paths import resolve as resolve_repo_path

    spec = get_robot_spec(args.robot_spec)
    app = AppLauncher(args).app

    import tempfile

    import torch
    from pxr import Gf, Sdf, Usd

    from genmech.tasks.pose_reach.utils.scene_utils import (
        _bake_usd,
        _convert_urdf_to_usd,
        _robot_joint_drive_cfg,
    )

    work = Path(tempfile.mkdtemp(prefix="genmech_ovr_"))
    FINGERS = ("index", "middle", "ring", "thumb")

    # ---- convert + bake the base ONCE -------------------------------------
    t0 = time.perf_counter()
    raw = _convert_urdf_to_usd(
        str(resolve_repo_path(spec.urdf_path)), work / "usd",
        fix_base=True, self_collision=True,
        joint_drive=_robot_joint_drive_cfg(),
    )
    base = _bake_usd(
        raw, work / "baked", "base",
        props=dict(disable_gravity=True, max_depenetration_velocity=1000.0,
                   enabled_self_collisions=False,
                   solver_position_iterations=8, solver_velocity_iterations=0),
        apply_physx_articulation=True,
    )
    convert_s = time.perf_counter() - t0
    print(f"\n[ovr] base convert+bake: {convert_s:.2f} s")

    base_stage = Usd.Stage.Open(base)
    root_path = base_stage.GetDefaultPrim().GetPath()
    root_name = root_path.name
    print(f"[ovr] base default prim: {root_path}")

    # Read the values we are about to override, so scaling is relative to truth
    # rather than to a hardcoded guess.
    joint_localpos: dict[str, Gf.Vec3f] = {}
    for prim in base_stage.Traverse():
        name = prim.GetPath().name
        if not any(f in name for f in FINGERS) or "joint" not in name:
            continue
        attr = prim.GetAttribute("physics:localPos0")
        if attr and attr.Get() is not None:
            joint_localpos[str(prim.GetPath())] = attr.Get()
    print(f"[ovr] found {len(joint_localpos)} finger joint localPos0 attrs")
    # merge_fixed_joints=True composes the palm/mount fixed transforms into the
    # FIRST joint of each finger, so its localPos0 is not the URDF origin --
    # it is iiwa14_link_7 -> knuckle. Scaling it therefore scales the mount
    # offset too. Print the magnitudes so that is visible, not inferred.
    for jpath in sorted(joint_localpos):
        v = joint_localpos[jpath]
        mag = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
        print(f"[ovr]   {Sdf.Path(jpath).name:18s} localPos0 = "
              f"({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})  |v| = {mag:.4f} m")

    tip_link = spec.fingertip_body_names[0]
    tip_prim_path = f"{root_path}/{tip_link}"
    base_mass = base_stage.GetPrimAtPath(tip_prim_path).GetAttribute(
        "physics:mass").Get()
    print(f"[ovr] base {tip_link} mass = {base_mass:.6f} kg")

    # ---- author a variant as a reference + overrides -----------------------
    def author_variant(path: Path, *, finger_scale: float,
                       mass_scale: float, geom_scale: float) -> str:
        stage = Usd.Stage.CreateNew(str(path))
        prim = stage.DefinePrim(str(root_path))
        # Reference, not sublayer: the base is a separate asset whose default
        # prim composes in under ours, so its meshes are loaded once and shared.
        prim.GetReferences().AddReference(base)
        stage.SetDefaultPrim(prim)

        if finger_scale != 1.0:
            for jpath, v in joint_localpos.items():
                over = stage.OverridePrim(jpath)
                over.CreateAttribute(
                    "physics:localPos0", Sdf.ValueTypeNames.Point3f
                ).Set(Gf.Vec3f(v[0] * finger_scale, v[1] * finger_scale,
                               v[2] * finger_scale))
        if mass_scale != 1.0:
            over = stage.OverridePrim(tip_prim_path)
            over.CreateAttribute(
                "physics:mass", Sdf.ValueTypeNames.Float
            ).Set(float(base_mass * mass_scale))
        if geom_scale != 1.0:
            over = stage.OverridePrim(tip_prim_path)
            over.CreateAttribute(
                "xformOp:scale", Sdf.ValueTypeNames.Double3
            ).Set(Gf.Vec3d(geom_scale, geom_scale, geom_scale))
        stage.GetRootLayer().Save()
        return str(path)

    var_dir = work / "variants"
    var_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    variant = author_variant(var_dir / "v_scaled.usda",
                             finger_scale=args.finger_scale,
                             mass_scale=args.mass_scale,
                             geom_scale=args.geom_scale)
    author_s = time.perf_counter() - t0
    print(f"[ovr] authored one variant in {author_s * 1000:.2f} ms")

    t0 = time.perf_counter()
    for i in range(args.n_timing):
        author_variant(var_dir / f"t_{i}.usda",
                       finger_scale=1.0 + 0.01 * i,
                       mass_scale=1.0, geom_scale=1.0)
    bulk_s = time.perf_counter() - t0
    print(f"[ovr] authored {args.n_timing} variants in {bulk_s:.3f} s "
          f"({bulk_s / args.n_timing * 1000:.2f} ms each)")
    print(f"[ovr] extrapolated to 24576: override {bulk_s / args.n_timing * 24576 / 60:.1f} min "
          f"vs conversion {convert_s * 24576 / 3600:.1f} h")

    # ---- spawn base and variant into alternating envs ---------------------
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg
    from isaaclab.utils import configclass

    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device="cuda:0"))

    @configclass
    class Cfg(InteractiveSceneCfg):
        robot: ArticulationCfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=MultiUsdFileCfg(usd_path=[base, variant], random_choice=False),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=spec.base_pos, rot=spec.base_rot,
                joint_pos={**spec.arm_default_joint_pos,
                           **spec.hand_default_joint_pos},
                joint_vel={".*": 0.0},
            ),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=list(spec.arm_joint_names),
                    stiffness=dict(spec.arm_stiffness),
                    damping=dict(spec.arm_damping)),
                "hand": ImplicitActuatorCfg(
                    joint_names_expr=list(spec.hand_joint_names),
                    stiffness=dict(spec.hand_stiffness),
                    damping=dict(spec.hand_damping),
                    armature=dict(spec.hand_armature)),
            },
        )

    scene = InteractiveScene(Cfg(num_envs=args.num_envs, env_spacing=2.0,
                                 replicate_physics=False, clone_in_fabric=False))
    sim.reset()
    robot: Articulation = scene["robot"]
    for _ in range(3):
        sim.step()
        scene.update(1 / 120.0)

    print(f"\n[ovr] view: {robot.num_instances} instances x "
          f"{robot.num_joints} joints")

    # ---- A: did link length change? ---------------------------------------
    bid = robot.find_bodies(tip_link)[0][0]
    palm_id = robot.find_bodies(spec.palm_body_name)[0][0]
    d = robot.data.body_pos_w[:, bid, :] - robot.data.body_pos_w[:, palm_id, :]
    reach = torch.linalg.norm(d, dim=-1)
    print(f"\n[ovr] TEST A -- link length via joint localPos0 override")
    for i, r in enumerate(reach.tolist()):
        kind = "base" if i % 2 == 0 else f"override x{args.finger_scale}"
        print(f"[ovr]   env {i}: palm->{tip_link} = {r:.4f} m   ({kind})")
    spread = float(reach.max() - reach.min())
    a_works = spread > 1e-3
    print(f"[ovr]   spread {spread * 100:.2f} cm -> "
          f"{'OVERRIDE TOOK EFFECT' if a_works else 'NO EFFECT (collapsed)'}")
    if a_works:
        nom = float(reach[0])
        scl = float(reach[1])
        print(f"[ovr]   vs full-conversion ground truth: "
              f"nominal {nom:.4f} vs {GROUND_TRUTH_NOMINAL:.4f} "
              f"(d={abs(nom - GROUND_TRUTH_NOMINAL) * 1000:.1f} mm), "
              f"scaled {scl:.4f} vs {GROUND_TRUTH_SCALED:.4f} "
              f"(d={abs(scl - GROUND_TRUTH_SCALED) * 1000:.1f} mm)")

    # ---- B: did mass change? ----------------------------------------------
    masses = robot.root_physx_view.get_masses()
    print(f"\n[ovr] TEST B -- mass via physics:mass override")
    for i in range(robot.num_instances):
        kind = "base" if i % 2 == 0 else f"override x{args.mass_scale}"
        print(f"[ovr]   env {i}: {tip_link} mass = "
              f"{float(masses[i, bid]):.6f} kg   ({kind})")
    b_works = abs(float(masses[1, bid]) - float(masses[0, bid])) > 1e-9
    print(f"[ovr]   -> {'OVERRIDE TOOK EFFECT' if b_works else 'NO EFFECT'}")

    # ---- C: did geometry scale reach the collision shape? ------------------
    # A scaled link keeps its body origin, so body_pos_w cannot detect this.
    # The physx view's link incoming-joint force is no help either; the honest
    # check is the cooked collision extent, read back off the scene graph.
    print(f"\n[ovr] TEST C -- geometry size via xformOp:scale override")
    # Collision meshes are authored with purpose 'guide', so a default
    # BBoxCache returns an empty range (FLT_MAX sentinels). Include every
    # purpose, and measure visuals and collisions separately -- the whole point
    # is to detect the case where the visual scales and the collision does not.
    from pxr import UsdGeom

    from isaacsim.core.utils.stage import get_current_stage

    live = get_current_stage()
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                          UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
        useExtentsHint=False,
    )

    def extent(prim_path: str) -> tuple[float, float, float] | None:
        prim = live.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            return None
        s = rng.GetSize()
        return (float(s[0]), float(s[1]), float(s[2]))

    c_vis, c_col = [], []
    for i in range(robot.num_instances):
        kind = "base" if i % 2 == 0 else f"override x{args.geom_scale}"
        vis = extent(f"/World/envs/env_{i}/Robot/{tip_link}/visuals")
        col = extent(f"/World/envs/env_{i}/Robot/{tip_link}/collisions")
        c_vis.append(vis)
        c_col.append(col)
        fmt = lambda e: ("empty" if e is None
                         else f"({e[0]:.4f}, {e[1]:.4f}, {e[2]:.4f})")
        print(f"[ovr]   env {i} ({kind:16s}) visual {fmt(vis)}  "
              f"collision {fmt(col)}")

    def ratio(vals) -> float | None:
        if vals[0] is None or vals[1] is None or max(vals[0]) <= 0:
            return None
        return max(vals[1]) / max(vals[0])

    r_vis, r_col = ratio(c_vis), ratio(c_col)
    print(f"[ovr]   visual size ratio    = "
          f"{'n/a' if r_vis is None else f'{r_vis:.4f}'} "
          f"(authored {args.geom_scale})")
    print(f"[ovr]   collision size ratio = "
          f"{'n/a' if r_col is None else f'{r_col:.4f}'} "
          f"(authored {args.geom_scale})")
    tol = 0.02
    if r_vis is None or r_col is None:
        c_verdict = "INCONCLUSIVE (empty bounds)"
    elif (abs(r_col - args.geom_scale) < tol
          and abs(r_vis - args.geom_scale) < tol):
        c_verdict = "WORKS -- visual and collision both scaled"
    elif abs(r_vis - args.geom_scale) < tol and abs(r_col - 1.0) < tol:
        c_verdict = "DANGEROUS -- visual scaled, collision did NOT"
    else:
        c_verdict = "UNEXPECTED -- see ratios above"
    print(f"[ovr]   -> {c_verdict}")

    print(f"\n[ovr] {'=' * 58}")
    print(f"[ovr] A link length : {'WORKS' if a_works else 'FAILS'}")
    print(f"[ovr] B mass        : {'WORKS' if b_works else 'FAILS'}")
    print(f"[ovr] C geom scale  : {c_verdict}")
    print(f"[ovr] PROBE COMPLETE")

    del app
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
