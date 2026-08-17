"""Do robots authored INTO env prims match robots authored to FILES?

The in-env path (spawn=None, prims written straight into /World/envs/env_N/Robot)
removed the k*n composition cost -- 552 s -> 52 s at k = n = 2048. But it also
changed how the arm composes: the file path references arm_flat.usd into a design
USD and then references THAT into the env, two levels deep, while the in-env path
references the arm directly into the env prim, one level.

Two levels of nesting is exactly what silently stripped the arm's collision
geometry once already (docs/multi_embodiment.md, the flatten_arm_usd fix). A
cheaper composition route is not automatically a correct one, and "the arm looks
fine" is not a measurement -- a robot with no arm colliders still builds, still
steps, and still trains, just wrongly.

So this builds the SAME designs both ways in one stage and diffs the composed
result prim by prim:

  * every prim under the robot, by relative path, with type and API schemas
  * every authored attribute, as the UNION of both sides, so an attribute the
    file path sets and the in-env path never writes shows up as missing rather
    than being invisible
  * collision-shape counts called out separately, since that is the failure mode
    with precedent here

Anything that differs is a difference between the two authoring routes, because
the hand params, the spec and the arm USD are identical by construction.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.check_inenv_robot --count 3
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_args():
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count", type=int, default=3, help="designs to compare")
    p.add_argument("--seed", type=int, default=2, help="population seed")
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
    import omni.usd
    from pxr import Sdf, Usd, UsdGeom

    from genmech.robots.generated.author_robot import (
        arm_only_urdf,
        author_robot_prims,
        author_robot_usd,
        flatten_arm_usd,
    )
    from genmech.robots.generated.population import load_population
    from genmech.robots.generated.synth_spec import synth_spec
    from genmech.tasks.pose_reach.utils.scene_utils import (
        _convert_urdf_to_usd,
        _robot_joint_drive_cfg,
    )

    work = Path(tempfile.mkdtemp(prefix="genmech_inenv_"))

    # --- the shared arm, exactly as scene_utils prepares it ------------------
    arm_dir = work / "arm"
    arm_urdf = arm_only_urdf(arm_dir / "iiwa14_arm_only.urdf")
    arm_raw = _convert_urdf_to_usd(str(arm_urdf), arm_dir, fix_base=True,
                                   self_collision=True,
                                   joint_drive=_robot_joint_drive_cfg())
    arm_usd = flatten_arm_usd(arm_raw, arm_dir / "arm_flat.usd")
    arm_stage = Usd.Stage.Open(arm_usd)
    arm_root = str(next(c for c in arm_stage.GetPseudoRoot().GetChildren()).GetPath())
    link7_world = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(
        arm_stage.GetPrimAtPath(f"{arm_root}/iiwa14_link_7"))).T

    hands = load_population(args.seed)[:args.count]
    print(f"[inenv] comparing {len(hands)} designs from population seed {args.seed}")

    stage = omni.usd.get_context().get_stage()
    layer = stage.GetRootLayer()

    def dump(root_path):
        """{relative prim path: (type, apis, {attr: value})} for a subtree."""
        out = {}
        root = stage.GetPrimAtPath(root_path)
        # TraverseInstanceProxies, NOT a bare PrimRange. The converted arm's
        # collision meshes live in flattened instancing prototypes, and a bare
        # range walks straight past them -- it reports 0 arm colliders for an
        # arm that has 8, and reports the same 0 for an arm that genuinely has
        # none. That false equality is exactly the kind of blind spot this tool
        # exists to remove.
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            rel = str(prim.GetPath())[len(root_path):] or "/"
            md = prim.GetMetadata("apiSchemas")
            apis = tuple(sorted(str(x) for x in md.GetAppliedItems())) if md else ()
            attrs = {}
            for a in prim.GetAttributes():
                if not a.HasAuthoredValue():
                    continue
                try:
                    attrs[a.GetName()] = a.Get()
                except Exception:                                  # noqa: BLE001
                    attrs[a.GetName()] = "<unreadable>"
            for r in prim.GetRelationships():
                t = r.GetTargets()
                if t:
                    attrs["rel:" + r.GetName()] = tuple(x.name for x in t)
            out[rel] = (str(prim.GetTypeName()), apis, attrs)
        return out

    def norm(v):
        if v is None:
            return None
        if hasattr(v, "GetReal") and hasattr(v, "GetImaginary"):
            im = v.GetImaginary()
            return tuple(round(float(x), 6) for x in
                         (v.GetReal(), im[0], im[1], im[2]))
        if hasattr(v, "__len__") and not isinstance(v, str):
            try:
                return tuple(round(float(x), 6) for x in np.asarray(v).ravel())
            except Exception:                                      # noqa: BLE001
                return tuple(str(x) for x in v)
        if isinstance(v, float):
            return round(v, 6)
        return v

    def n_colliders(d):
        return sum(1 for _, (_, apis, _) in d.items()
                   if "PhysicsCollisionAPI" in apis)

    total_problems = 0
    for i, hand in enumerate(hands):
        spec = synth_spec(hand)

        # route A: author to a FILE, then reference it in (production path today)
        f = author_robot_usd(hand, spec, work / f"d{i}" / f"{hand.name}.usd",
                             arm_usd=arm_usd, arm_root_prim=arm_root,
                             link7_world=link7_world,
                             adjacency=spec.adjacent_links)
        a_path = f"/Cmp{i}/FromFile"
        with Sdf.ChangeBlock():
            for anc in (f"/Cmp{i}", a_path):
                s = Sdf.CreatePrimInLayer(layer, Sdf.Path(anc))
                s.specifier = Sdf.SpecifierDef
                s.typeName = "Xform"
        stage.GetPrimAtPath(a_path).GetReferences().AddReference(f)

        # route B: author the prims STRAIGHT into the target prim (new path)
        b_path = f"/Cmp{i}/InEnv"
        with Sdf.ChangeBlock():
            author_robot_prims(layer, b_path, hand, spec, arm_usd=arm_usd,
                               arm_root_prim=arm_root, link7_world=link7_world,
                               adjacency=spec.adjacent_links,
                               in_change_block=True)

        A, B = dump(a_path), dump(b_path)
        ca, cb = n_colliders(A), n_colliders(B)
        print(f"\n[inenv] {hand.name}: prims file={len(A)} inenv={len(B)}  "
              f"colliders file={ca} inenv={cb}")
        if cb == 0:
            print("[inenv]   *** IN-ENV ROBOT HAS NO COLLIDERS ***")
        # A matching collider COUNT proves the two routes agree; it does not
        # prove either one is right. The arm is the part with a history of
        # losing its colliders to nested references, so name them.
        n_mesh = sum(1 for _, (t, _, _) in B.items() if t == "Mesh")
        print(f"[inenv]   meshes: {n_mesh}")
        if n_mesh == 0:
            print("[inenv]   *** ROBOT HAS NO MESHES -- arm geometry lost ***")

        problems = 0
        for p in sorted(set(A) - set(B)):
            print(f"[inenv]   ONLY IN FILE : {p} type={A[p][0]}")
            problems += 1
        for p in sorted(set(B) - set(A)):
            print(f"[inenv]   ONLY IN INENV: {p} type={B[p][0]}")
            problems += 1
        for p in sorted(set(A) & set(B)):
            ta, apisA, attrA = A[p]
            tb, apisB, attrB = B[p]
            if ta != tb:
                print(f"[inenv]   TYPE {p}: file={ta} inenv={tb}")
                problems += 1
            if set(apisA) != set(apisB):
                print(f"[inenv]   APIS {p}: only-file={sorted(set(apisA)-set(apisB))} "
                      f"only-inenv={sorted(set(apisB)-set(apisA))}")
                problems += 1
            for k in sorted(set(attrA) | set(attrB)):
                va, vb = norm(attrA.get(k)), norm(attrB.get(k))
                if k not in attrB:
                    print(f"[inenv]   ATTR MISSING IN INENV {p}.{k} = {va}")
                    problems += 1
                elif k not in attrA:
                    print(f"[inenv]   ATTR EXTRA IN INENV   {p}.{k} = {vb}")
                    problems += 1
                elif va != vb:
                    print(f"[inenv]   ATTR DIFFERS {p}.{k}: file={va} inenv={vb}")
                    problems += 1
        print(f"[inenv]   {problems} difference(s)")
        total_problems += problems

    print(f"\n[inenv] TOTAL {total_problems} difference(s) over {len(hands)} designs")
    del app
    sys.stdout.flush()
    os._exit(0 if total_problems == 0 else 1)


if __name__ == "__main__":
    main()
