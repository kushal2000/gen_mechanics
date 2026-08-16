"""Do authored and converted objects BEHAVE the same under physics?

compare_object_assets showed mass and inertia agree to ~1e-8, but that is only
half the question. The two paths express geometry differently: the converter
nests shapes under mesh_N Xforms with size=1.0 plus an extent, while the
authored path scales the Cube prim directly. Those should compose to the same
collider, and reading USD carefully is a poor way to find out -- the capsule
bug earlier today was exactly a case where the declared geometry and the
simulated geometry differed, and no amount of staring at the asset revealed it.

So this drops both onto a table in the same scene, from identical initial poses
with identical velocities, and compares where they end up.

  * converted objects occupy the EVEN envs, authored the ODD ones
  * the same (handle, head, densities) parameters drive both
  * identical initial pose, zero velocity, same gravity and dt

If the colliders differ in size, the objects settle at different heights or
tip differently, and the resting-pose comparison shows it. If they match, the
authored path is a physical substitute, which is what the training env needs
before it can adopt it.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.compare_object_physics --variants 8 --steps 400
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--tol_m", type=float, default=1e-3,
                        help="position agreement required, metres")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    return args


def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app

    import tempfile
    from pathlib import Path

    import numpy as np
    import omni.usd
    import torch
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
    from pxr import Sdf, UsdGeom

    from genmech.tasks.pose_reach.utils.author_objects import author_handle_head
    from genmech.tasks.pose_reach.utils.generate_objects import (
        generate_handle_head_urdf,
    )
    from genmech.tasks.pose_reach.utils.scene_utils import (
        _bake_usd,
        _convert_urdf_to_usd,
    )

    work = Path(tempfile.mkdtemp(prefix="genmech_objphys_"))
    (work / "urdf").mkdir(parents=True, exist_ok=True)
    (work / "usd").mkdir(parents=True, exist_ok=True)

    cases = [
        ((0.12, 0.03, 0.03), (0.05, 0.05, 0.05), 700.0, 900.0),
        ((0.14, 0.025, 0.025), (0.06, 0.04), 700.0, 900.0),
        ((0.10, 0.028), (0.05, 0.05, 0.04), 800.0, 850.0),
        ((0.12, 0.03), (0.05, 0.045), 750.0, 950.0),
        ((0.16, 0.035, 0.030), (0.07, 0.06, 0.05), 600.0, 1100.0),
        ((0.09, 0.022), (0.04, 0.035), 900.0, 700.0),
        ((0.11, 0.026, 0.026), (0.045, 0.038), 820.0, 780.0),
        ((0.13, 0.031), (0.055, 0.05, 0.045), 680.0, 1020.0),
    ][: args.variants]

    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=args.device))
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    layer = stage.GetRootLayer()

    # --- authored objects: odd slots ---------------------------------------
    from genmech.robots.generated.author_usd import define

    with Sdf.ChangeBlock():
        for anc in ("/World", "/World/authored"):
            spec = Sdf.CreatePrimInLayer(layer, Sdf.Path(anc))
            spec.specifier = Sdf.SpecifierDef
            spec.typeName = "Xform"
        for i, (hs, hd_, dh, dhead) in enumerate(cases):
            define(layer, f"/World/authored/obj_{i}", "Xform")
            author_handle_head(layer, f"/World/authored/obj_{i}", hs, hd_, dh, dhead)

    # --- converted objects --------------------------------------------------
    conv_usds = []
    for i, (hs, hd_, dh, dhead) in enumerate(cases):
        urdf = generate_handle_head_urdf(work / "urdf" / f"o{i}.urdf",
                                         handle_scale=hs, head_scale=hd_,
                                         handle_density=dh, head_density=dhead)
        # MATCH THE TRAINING ENV: it converts objects with capsule replacement
        # (scene_utils.build_scene), which turns every cylinder into a capsule
        # and reads the URDF `length` as the CYLINDRICAL SECTION -- adding a
        # hemisphere at each end. Comparing against a plain cylinder conversion
        # would test an asset the task never uses.
        raw = _convert_urdf_to_usd(str(urdf), work / "usd", fix_base=False,
                                   replace_cylinders_with_capsules=True)
        conv_usds.append(_bake_usd(raw, work / "baked", f"o{i}", props=dict(
            kinematic_enabled=False, disable_gravity=False,
            max_depenetration_velocity=1000.0, articulation_enabled=False)))

    # --- converted objects spawned alongside, same scene, same physics -----
    from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg

    with Sdf.ChangeBlock():
        spec = Sdf.CreatePrimInLayer(layer, Sdf.Path("/World/converted"))
        spec.specifier = Sdf.SpecifierDef
        spec.typeName = "Xform"
        for i in range(len(cases)):
            define(layer, f"/World/converted/obj_{i}", "Xform")

    # The rigid body sits at .../object_root in BOTH paths -- the converter
    # nests it under the robot-name prim, and author_handle_head uses the same
    # link name so downstream code can address either identically.
    converted = RigidObject(RigidObjectCfg(
        prim_path="/World/converted/obj_.*/object_root",
        spawn=MultiUsdFileCfg(usd_path=conv_usds, random_choice=False),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.30))))

    authored = RigidObject(RigidObjectCfg(
        prim_path="/World/authored/obj_.*/object_root", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.30))))

    sim.reset()

    n = authored.num_instances
    print(f"[objphys] authored view: {n} instances, "
          f"converted view: {converted.num_instances} "
          f"(expected {len(cases)} each)")
    if n != len(cases) or converted.num_instances != len(cases):
        print("[objphys] FAIL: views did not resolve to one instance per case")
        del app
        os._exit(1)

    # Drop them and record where they settle.
    # Identical initial state, except the two SETS are separated in y. Placing
    # both at the same coordinates spawns each authored object inside its
    # converted twin, and they shove each other apart -- which looks exactly
    # like a physics disagreement and is not one.
    for y_offset, view in ((0.0, authored), (5.0, converted)):
        state = view.data.default_root_state.clone()
        state[:, 0] = torch.arange(n, device=state.device) * 0.5
        state[:, 1] = y_offset
        state[:, 2] = 0.30
        state[:, 7:] = 0.0            # zero linear and angular velocity
        view.write_root_state_to_sim(state)

    for _ in range(args.steps):
        for view in (authored, converted):
            view.write_data_to_sim()
        sim.step()
        for view in (authored, converted):
            view.update(1 / 120.0)

    za = authored.data.root_pos_w.cpu().numpy()
    zc = converted.data.root_pos_w.cpu().numpy()
    zc = zc.copy()
    zc[:, 1] -= 5.0                   # undo the separation before comparing
    dz = np.abs(za - zc).max(axis=1)
    print(f"[objphys] {'case':>5} {'authored z':>11} {'converted z':>12} "
          f"{'|delta| (m)':>12}")
    for i in range(n):
        flag = "" if dz[i] < args.tol_m else "   <-- DIFFERS"
        print(f"[objphys] {i:>5} {za[i, 2]:>11.5f} {zc[i, 2]:>12.5f} "
              f"{dz[i]:>12.2e}{flag}")
    worst = float(dz.max())
    ok = worst < args.tol_m
    print(f"\n[objphys] worst position disagreement {worst * 1000:.3f} mm "
          f"(tolerance {args.tol_m * 1000:.1f} mm)")
    print(f"[objphys] {'PASS' if ok else 'FAIL'}: authored objects "
          f"{'behave identically to converted' if ok else 'diverge from converted'}")
    settled = ok

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if settled else 1)


if __name__ == "__main__":
    main()
