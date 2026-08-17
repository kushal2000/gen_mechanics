"""Does an AUTHORED robot simulate like the CONVERTED one?

compare_authored_hand checks the asset: masses, inertias, joint wiring, collider
geometry. It passes at 0 mismatches, and that is still not sufficient -- the
object work proved exactly this. Every asset-level comparison passed while the
policy scored 3.00 against 5.07, because equality of assets is not equality of
behaviour once the asset is loaded, articulated and stepped.

So this loads both robots into ONE scene and drives them identically:

  * both resolve to an articulation with the same joint count and joint NAMES
  * PhysX reports the same per-body masses (which is where the merged palm shows
    up -- it is 37% of link_7's mass and would be silently missing)
  * the same joint limits
  * driven to the same targets for N steps from the same initial state, the
    fingertip bodies end up in the same place

The last one is the real test. Anything wrong with a joint frame, a limit, an
axis or the palm merge moves the fingertips, and fingertip position is what the
task's reward and observation are built on.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.compare_authored_robot --hand gen_sharpa_like --steps 120
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_args():
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hand", default="gen_sharpa_like")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--null_control", action="store_true",
                   help="spawn the CONVERTED usd as both robots. Any difference\n"
                        "that survives this is an artefact of the harness, not\n"
                        "of the authored asset.")
    p.add_argument("--target", choices=("home", "midrange"), default="midrange",
                   help="midrange jams the fingers together and exercises "
                        "self-collision; home isolates kinematics and drives")
    p.add_argument("--tol_m", type=float, default=1e-3,
                   help="fingertip agreement required, metres")
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

    import numpy as np
    import torch
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab.assets import Articulation
    from isaaclab.sim import SimulationCfg, SimulationContext
    from genmech.robots import get_robot_spec
    from genmech.robots.generated.author_robot import (
        arm_only_urdf, author_robot_usd, flatten_arm_usd)
    from genmech.robots.generated.synth_spec import params_for_name, synth_spec
    from genmech.tasks.pose_reach.utils.scene_utils import (
        _apply_self_collision_filters,
        _bake_usd,
        _convert_urdf_to_usd,
        _robot_joint_drive_cfg,
        build_robot_articulation_usd_cfg,
    )

    # The PRODUCTION finish, applied identically to both paths. Skipping it is
    # what made an earlier run of this tool look like a broken authored robot:
    # without enabled_self_collisions the authored fingers passed through each
    # other and tracked their targets perfectly, while the converted robot's
    # fingers collided and could not reach -- so the authored one looked wrong
    # precisely because it was under-constrained.
    BAKE = dict(disable_gravity=True, max_depenetration_velocity=1000.0,
                enabled_self_collisions=True,
                solver_position_iterations=8, solver_velocity_iterations=0)

    def finish(raw_usd, tag, adjacency):
        _apply_self_collision_filters(raw_usd, adjacency)
        return _bake_usd(raw_usd, work / "baked", tag, props=BAKE,
                         apply_physx_articulation=True)

    spec = get_robot_spec(args.hand)
    hand = params_for_name(args.hand)
    work = Path(tempfile.mkdtemp(prefix="genmech_robotcmp_"))

    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=args.device))

    # --- converted ----------------------------------------------------------
    conv_raw = _convert_urdf_to_usd(
        spec.urdf_path, work / "conv", fix_base=True, self_collision=True,
        joint_drive=_robot_joint_drive_cfg(),
        replace_cylinders_with_capsules=spec.replace_cylinders_with_capsules)
    conv_usd = finish(conv_raw, "conv", spec.adjacent_links)

    # --- authored (arm converted once, referenced) --------------------------
    arm_urdf = arm_only_urdf(work / "urdf" / "arm.urdf")
    # joint_drive is REQUIRED: DriveAPI prims must exist for ImplicitActuator's
    # runtime gains to land on them. Without it the arm's joints have no drive,
    # the spec's arm gains never attach, and the arm sags under gravity while
    # the authored hand tracks perfectly -- which reads as an authoring bug and
    # is a missing converter argument.
    arm_usd = _convert_urdf_to_usd(str(arm_urdf), work / "arm",
                                   fix_base=True, self_collision=True,
                                   joint_drive=_robot_joint_drive_cfg())
    # Flatten before referencing: the converter's output is not self-contained.
    arm_usd = flatten_arm_usd(arm_usd, work / "arm" / "arm_flat.usd")
    st = Usd.Stage.Open(arm_usd)
    arm_root = str(next(c for c in st.GetPseudoRoot().GetChildren()).GetPath())
    link7_world = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(
        st.GetPrimAtPath(f"{arm_root}/iiwa14_link_7"))).T
    auth_spec = synth_spec(hand, ensure_urdf=False)
    # NO external finish: the authored path now writes the self-collision
    # filters and the physics properties in its single pass, which is the
    # optimisation under test. If that inlining is wrong, this comparison is
    # what catches it.
    auth_usd = author_robot_usd(hand, auth_spec,
                                work / "auth" / f"{hand.name}.usd",
                                arm_usd=arm_usd, arm_root_prim=arm_root,
                                link7_world=link7_world,
                                adjacency=auth_spec.adjacent_links)

    # --- spawn both, side by side ------------------------------------------
    import omni.usd
    from pxr import Sdf

    stage = omni.usd.get_context().get_stage()
    with Sdf.ChangeBlock():
        for path in ("/World", "/World/conv", "/World/auth"):
            s = Sdf.CreatePrimInLayer(stage.GetRootLayer(), Sdf.Path(path))
            s.specifier = Sdf.SpecifierDef
            s.typeName = "Xform"

    def make(prim_path, usd, y_offset=0.0):
        cfg = build_robot_articulation_usd_cfg(usd, spec)
        cfg.prim_path = prim_path
        # SEPARATE THEM. Both default to spec.base_pos, so spawning both there
        # puts each robot inside its twin: they interpenetrate and shove each
        # other apart, symmetrically. That reads as a physics disagreement and
        # is nothing of the kind -- the giveaway is joints deviating from target
        # by EQUAL AND OPPOSITE amounts (joint_2: conv +0.0268, auth -0.0268).
        # compare_object_physics already carries this exact warning; I hit it
        # again here.
        pos = list(cfg.init_state.pos)
        pos[1] += y_offset
        cfg.init_state.pos = tuple(pos)
        return Articulation(cfg)

    conv = make("/World/conv/Robot", conv_usd, y_offset=0.0)
    if args.null_control:
        print("[robotcmp] NULL CONTROL: both robots are the CONVERTED asset")
        auth_usd = conv_usd
    auth = make("/World/auth/Robot", auth_usd, y_offset=4.0)
    sim.reset()

    fails: list[str] = []

    def check(ok, msg):
        print(f"[robotcmp] {'ok  ' if ok else 'FAIL'} {msg}")
        if not ok:
            fails.append(msg)

    check(conv.num_joints == auth.num_joints,
          f"joint count {conv.num_joints} vs {auth.num_joints}")
    cj, aj = list(conv.data.joint_names), list(auth.data.joint_names)
    check(cj == aj, f"joint NAMES identical ({len(cj)} vs {len(aj)})")
    cb, ab = list(conv.data.body_names), list(auth.data.body_names)
    check(sorted(cb) == sorted(ab),
          f"body name sets identical ({len(cb)} vs {len(ab)})")

    # masses as PhysX holds them -- where a missing merged palm would show
    cm = conv.root_physx_view.get_masses()[0].cpu().numpy()
    am = auth.root_physx_view.get_masses()[0].cpu().numpy()
    order = [ab.index(n) for n in cb] if sorted(cb) == sorted(ab) else None
    if order is not None:
        dm = np.abs(cm - am[order])
        worst = int(np.argmax(dm))
        check(bool(np.allclose(cm, am[order], rtol=1e-4, atol=1e-9)),
              f"per-body masses agree (worst {cb[worst]}: "
              f"{cm[worst]:.6f} vs {am[order][worst]:.6f})")
        print(f"[robotcmp]      total mass {cm.sum():.6f} vs {am.sum():.6f} kg")

    # Inertias as PhysX holds them. The merged palm's inertia is authored as an
    # OVER on a REFERENCED prim; the mass override provably lands (masses agree),
    # but nothing has confirmed the inertia does, and a link_7 carrying the
    # arm-only 0.001 diagonal instead of the merged tensor would show up as
    # exactly what we see: the arm settling differently while the hand agrees.
    ci = conv.root_physx_view.get_inertias()[0].cpu().numpy().reshape(len(cb), -1)
    ai = auth.root_physx_view.get_inertias()[0].cpu().numpy().reshape(len(ab), -1)
    if order is not None:
        di = np.abs(ci - ai[order]).max(axis=1)
        w = int(np.argmax(di))
        check(bool(di.max() < 1e-6),
              f"per-body inertias agree (worst {cb[w]}: {di.max():.3e})")
        for i in np.argsort(-di)[:3]:
            print(f"[robotcmp]   {cb[i]:<18} conv={np.round(ci[i],6)}")
            print(f"[robotcmp]   {'':<18} auth={np.round(ai[order][i],6)}")

    # Gravity and solver settings, read off the BAKED stages. _bake_usd applies
    # these, and the authored robot's arm arrives through a REFERENCE -- if the
    # bake only edits locally-defined prims, the arm keeps gravity on while the
    # hand does not, and the arm settles differently while the hand agrees.
    for tag, path in (("conv", conv_usd), ("auth", auth_usd)):
        st2 = Usd.Stage.Open(path)
        vals = {}
        for prim in st2.Traverse():
            a = prim.GetAttribute("physxRigidBody:disableGravity")
            if a and a.IsValid() and a.Get() is not None:
                vals.setdefault(bool(a.Get()), []).append(prim.GetPath().name)
        art = {}
        for prim in st2.Traverse():
            for k in ("physxArticulation:enabledSelfCollisions",
                      "physxArticulation:solverPositionIterationCount"):
                a = prim.GetAttribute(k)
                if a and a.IsValid() and a.Get() is not None:
                    art[k] = a.Get()
        print(f"[robotcmp] {tag}: disableGravity True on {len(vals.get(True, []))} "
              f"bodies, False on {len(vals.get(False, []))}; articulation {art}")
        if vals.get(False):
            print(f"[robotcmp]     gravity STILL ON: {sorted(vals[False])[:8]}")

    cl = conv.data.joint_pos_limits[0].cpu().numpy()
    al = auth.data.joint_pos_limits[0].cpu().numpy()
    check(bool(np.allclose(cl, al, atol=1e-6)),
          f"joint limits agree (max diff {np.abs(cl - al).max():.2e} rad)")

    # --- collider census, on the LIVE stage ---------------------------------
    # compare_authored_hand only ever compared Capsule prims, so the PALM BOX
    # -- the one collider authored by a different route, as a Cube on a
    # referenced prim -- was never checked. At the home pose the fingers rest
    # against the palm, so a misplaced palm collider shows up as contact torque
    # in the ARM while the hand's own joints look fine.
    cache2 = UsdGeom.XformCache()

    def colliders_by_body(root_path):
        out = {}
        # TraverseInstanceProxies: the ARM's collision meshes live in flattened
        # instancing prototypes (/Flattened_Prototype_N/link_N/...), and a bare
        # Traverse walks past every one of them. Both sides then report zero arm
        # colliders, the sets compare equal, and the check passes -- which is
        # what it did while the authored arm genuinely had NO geometry at all
        # (0.01 MB vs 10.57 MB, 0 meshes vs 16), because the arm-only URDF wrote
        # mesh paths relative to a directory it does not live in.
        for prim in stage.Traverse(Usd.TraverseInstanceProxies(
                Usd.PrimAllPrimsPredicate)):
            if not str(prim.GetPath()).startswith(root_path):
                continue
            # COLLIDERS only -- and CollisionAPI alone is the test. There used
            # to be a type allow-list here (Capsule/Cube/Sphere/Cylinder/Mesh)
            # to skip the converted asset's visual prims, but CollisionAPI
            # already does that, and the allow-list silently dropped the arm:
            # its collision prims are the importer's node_STL_BINARY_, whose
            # type is not in the list. Both sides lost their arm colliders
            # identically, so the census compared equal and reported agreement
            # on geometry neither side was being checked for.
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            t = str(prim.GetTypeName())
            body = prim
            while body and body.GetPath().pathString != root_path:
                nm = body.GetPath().name
                if nm.startswith("gen_f") or nm.startswith("iiwa14_link"):
                    break
                body = body.GetParent()
            else:
                continue
            m = np.asarray(cache2.GetLocalToWorldTransform(prim)).T
            mb = np.asarray(cache2.GetLocalToWorldTransform(body)).T
            local = np.linalg.inv(mb) @ m
            g = lambda k: (prim.GetAttribute(k).Get()
                           if prim.GetAttribute(k) and prim.GetAttribute(k).IsValid()
                           else None)
            dims = tuple(round(float(x), 6) for x in
                         (g("radius") or 0.0, g("height") or 0.0, g("size") or 0.0))
            scale = np.linalg.norm(local[:3, :3], axis=0)
            out.setdefault(body.GetPath().name, []).append(
                (t, dims, tuple(np.round(local[:3, 3], 6)),
                 tuple(np.round(scale, 6))))
        return {k: sorted(v) for k, v in out.items()}

    cc2 = colliders_by_body("/World/conv/Robot")
    ac2 = colliders_by_body("/World/auth/Robot")
    check(set(cc2) == set(ac2),
          f"same bodies carry colliders ({len(cc2)} vs {len(ac2)}) "
          f"only-conv={sorted(set(cc2)-set(ac2))[:4]} "
          f"only-auth={sorted(set(ac2)-set(cc2))[:4]}")
    for name in sorted(set(cc2) & set(ac2)):
        if cc2[name] != ac2[name]:
            check(False, f"colliders differ on {name}")
            print(f"[robotcmp]     conv {cc2[name]}")
            print(f"[robotcmp]     auth {ac2[name]}")

    # An ABSOLUTE floor, not just agreement. Every check above is a comparison,
    # and a comparison cannot distinguish "both correct" from "both empty" --
    # that is exactly how an arm with no collision geometry passed. So assert
    # the arm actually carries colliders, on each side independently.
    for tag, census in (("conv", cc2), ("auth", ac2)):
        arm = sorted(b for b in census if b.startswith("iiwa14_link"))
        check(len(arm) >= 7,
              f"{tag} arm carries colliders on {len(arm)} links "
              f"(expect >=7 of iiwa14_link_0..7): {arm}")

    # Runtime actuation, as Isaac Lab resolved it. Everything above is USD-level;
    # these are what PhysX actually drives with, and a mismatch here explains a
    # steady-state offset from target that no geometry check would ever show.
    for field in ("joint_stiffness", "joint_damping", "joint_armature",
                  "joint_friction", "joint_effort_limits", "joint_velocity_limits"):
        gc = getattr(conv.data, field, None)
        ga = getattr(auth.data, field, None)
        if gc is None or ga is None:
            continue
        vc = gc[0].cpu().numpy()
        va = ga[0].cpu().numpy()
        d = np.abs(vc - va)
        ok = bool(np.allclose(vc, va, rtol=1e-4, atol=1e-9))
        check(ok, f"{field} agrees (max diff {d.max():.3e})")
        if not ok:
            for i in np.argsort(-d)[:5]:
                print(f"[robotcmp]     {cj[i]:<18} conv {vc[i]:.6f}  auth {va[i]:.6f}")

    # data.joint_velocity_limits and what PhysX holds can disagree: the former
    # is parsed from USD, the latter is what the solver uses. Check both, so a
    # reporting artefact is not mistaken for a physics difference (or vice versa).
    try:
        pc = conv.root_physx_view.get_dof_max_velocities()[0].cpu().numpy()
        pa = auth.root_physx_view.get_dof_max_velocities()[0].cpu().numpy()
        d = np.abs(pc - pa)
        check(bool(np.allclose(pc, pa, rtol=1e-4)),
              f"PhysX dof max velocities agree (max diff {d.max():.3e})")
        print(f"[robotcmp]      physx conv[0]={pc[0]:.6f} auth[0]={pa[0]:.6f}")
    except Exception as exc:                                   # noqa: BLE001
        print(f"[robotcmp] (physx max-velocity readback unavailable: {exc})")

    # --- drive both identically and compare fingertips ----------------------
    tip_ids_c = conv.find_bodies(list(spec.fingertip_slots), preserve_order=True)[0]
    tip_ids_a = auth.find_bodies(list(spec.fingertip_slots), preserve_order=True)[0]

    q0 = conv.data.default_joint_pos.clone()
    for view in (conv, auth):
        view.write_joint_state_to_sim(q0.clone(), torch.zeros_like(q0))
        view.set_joint_position_target(q0.clone())
        view.write_data_to_sim()

    torch.manual_seed(0)
    lo = conv.data.joint_pos_limits[..., 0]
    hi = conv.data.joint_pos_limits[..., 1]
    # midrange drives every joint to the middle of its travel, which for a hand
    # means driving the fingers INTO each other -- a contact stress test. home
    # holds the default pose, where nothing should touch, and so separates a
    # kinematic/drive difference from a contact one.
    target = q0.clone() if args.target == "home" else lo + 0.5 * (hi - lo)
    for _ in range(args.steps):
        for view in (conv, auth):
            view.set_joint_position_target(target.clone())
            view.write_data_to_sim()
        sim.step()
        for view in (conv, auth):
            view.update(1 / 120.0)

    # Compare in each robot's OWN base frame, since they are spawned apart.
    def tips(view, ids):
        root = view.data.root_pos_w[0]
        return (view.data.body_pos_w[0, ids, :] - root).cpu().numpy()

    # Every body, base-relative. Five fingertips off by the SAME ~105 mm is a
    # rigid offset of the whole hand, not a kinematic error, and comparing only
    # the tips cannot tell those apart. Walking the chain shows where it enters.
    rc = conv.data.root_pos_w[0]
    ra = auth.data.root_pos_w[0]
    print("[robotcmp] per-body offset, base-relative (worst 10):")
    rows = []
    for n in cb:
        pc = (conv.data.body_pos_w[0, cb.index(n), :] - rc).cpu().numpy()
        pa = (auth.data.body_pos_w[0, ab.index(n), :] - ra).cpu().numpy()
        rows.append((float(np.linalg.norm(pc - pa)), n, pc, pa))
    for dist, n, pc, pa in sorted(rows, reverse=True)[:10]:
        print(f"[robotcmp]   {n:<18} |d|={dist*1000:8.3f} mm  "
              f"conv={np.round(pc,4)} auth={np.round(pa,4)}")
    chain = ["iiwa14_link_0", "iiwa14_link_7", "gen_f0_CMC_VL", "gen_f0_MC",
             "gen_f0_MCP_VL", "gen_f0_PP", "gen_f0_MP", "gen_f0_DP"]
    print("[robotcmp] along one finger's chain:")
    for n in chain:
        if n in cb and n in ab:
            pc = (conv.data.body_pos_w[0, cb.index(n), :] - rc).cpu().numpy()
            pa = (auth.data.body_pos_w[0, ab.index(n), :] - ra).cpu().numpy()
            print(f"[robotcmp]   {n:<18} |d|={np.linalg.norm(pc-pa)*1000:8.3f} mm")

    tc, ta = tips(conv, tip_ids_c), tips(auth, tip_ids_a)
    d = np.linalg.norm(tc - ta, axis=-1)
    print(f"[robotcmp] fingertip positions after {args.steps} steps, "
          f"base-relative:")
    for i, name in enumerate(spec.fingertip_slots):
        print(f"[robotcmp]   {name:<16} |delta| = {d[i] * 1000:8.4f} mm")
    check(bool(d.max() < args.tol_m),
          f"fingertips agree within {args.tol_m * 1000:.1f} mm "
          f"(worst {d.max() * 1000:.4f} mm)")

    qc = conv.data.joint_pos[0].cpu().numpy()
    qa = auth.data.joint_pos[0].cpu().numpy()
    tg = target[0].cpu().numpy()
    qd = np.abs(qc - qa)
    check(bool(qd.max() < 1e-3),
          f"joint angles agree (worst {qd.max():.2e} rad)")
    # Which joints, and is each robot even REACHING its target? A joint that
    # tracks in one robot and not the other says the drive did not attach;
    # both missing by the same amount says the target is simply unreachable.
    print(f"[robotcmp] worst joints (target / converted / authored, rad):")
    for i in np.argsort(-qd)[:8]:
        print(f"[robotcmp]   {cj[i]:<18} tgt {tg[i]:+.4f}  conv {qc[i]:+.4f}  "
              f"auth {qa[i]:+.4f}  |d| {qd[i]:.4f}")
    arm_idx = [i for i, n in enumerate(cj) if n.startswith("iiwa14_")]
    hand_idx = [i for i, n in enumerate(cj) if not n.startswith("iiwa14_")]
    print(f"[robotcmp] max |delta|: arm joints {qd[arm_idx].max():.4f} rad, "
          f"hand joints {qd[hand_idx].max():.4f} rad")

    print(f"\n[robotcmp] {'PASS' if not fails else 'FAIL'}: {len(fails)} problem(s)")
    del app
    sys.stdout.flush()
    os._exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
