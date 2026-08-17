"""EXHAUSTIVE prim/attribute diff between a converted and an authored object.

Every earlier object comparison enumerated attributes I thought to check -- 36
physics properties, collider geometry, mass and inertia -- and they all matched.
That method cannot see the one thing most likely to be wrong: an attribute the
CONVERTER sets that the authoring never writes. It is absent from the authored
prim, so it is absent from the diff, so the diff is green.

Training then diverged (reward 4000 -> 251 with everything else identical),
which is the failure such a blind spot would produce.

So this enumerates instead of assuming:

  * every prim under each object, by relative path, with type and API schemas
  * every AUTHORED attribute on every prim, as the union of both sides
  * reports prims only-in-one, attributes only-in-one, and value mismatches

Both objects are built with the SAME parameters in the SAME stage, so anything
that differs is a difference between the two paths, not between two objects.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.diff_object_prims
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_args():
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=int, default=0, help="pool entry to compare")
    p.add_argument("--show_all", action="store_true")
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
    from pxr import Sdf, Usd

    from genmech.tasks.pose_reach.utils.author_objects import (
        author_handle_head,
        author_physics_material,
    )
    from genmech.tasks.pose_reach.utils.generate_objects import (
        generate_handle_head_urdfs,
    )
    from genmech.tasks.pose_reach.utils.scene_utils import (
        _bake_usd,
        _convert_urdf_to_usd,
    )

    work = Path(tempfile.mkdtemp(prefix="genmech_objdiff_"))
    paths, scales, params = generate_handle_head_urdfs(
        handle_head_types=("hammer", "screwdriver", "marker", "spatula",
                           "eraser", "brush"),
        num_per_type=2, out_dir=work / "urdf", shuffle=False, seed=42,
        density_scale=1.0)
    hs, head, hd, headd = params[args.index]
    print(f"[objdiff] pool[{args.index}] handle={hs} head={head} "
          f"densities=({hd}, {headd})")

    # --- converted, through the production pipeline -------------------------
    raw = _convert_urdf_to_usd(str(paths[args.index]), work / "usd",
                               fix_base=False,
                               replace_cylinders_with_capsules=True)
    conv_usd = _bake_usd(raw, work / "baked", "obj", props=dict(
        kinematic_enabled=False, disable_gravity=False,
        max_depenetration_velocity=1000.0, articulation_enabled=False))

    stage = omni.usd.get_context().get_stage()
    layer = stage.GetRootLayer()
    with Sdf.ChangeBlock():
        for anc in ("/World", "/World/conv", "/World/auth"):
            s = Sdf.CreatePrimInLayer(layer, Sdf.Path(anc))
            s.specifier = Sdf.SpecifierDef
            s.typeName = "Xform"

    # converted: reference the baked asset exactly as the spawner would
    ref = stage.GetPrimAtPath("/World/conv")
    ref.GetReferences().AddReference(conv_usd)

    # authored: exactly what _author_objects_into_envs does
    with Sdf.ChangeBlock():
        mat = author_physics_material(layer, "/World/PhysicsMaterials/object",
                                      static_friction=0.5, dynamic_friction=0.5,
                                      restitution=0.0)
        author_handle_head(layer, "/World/auth", hs, head, hd, headd,
                           body_at_root=False, collision=True,
                           material_path=mat, kinematic=False)

    def dump(root_path):
        """{relative prim path: (type, apis, {attr: value})} for a subtree."""
        out = {}
        root = stage.GetPrimAtPath(root_path)
        for prim in Usd.PrimRange(root):
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
            rels = {}
            for r in prim.GetRelationships():
                t = r.GetTargets()
                if t:
                    rels["rel:" + r.GetName()] = tuple(x.name for x in t)
            attrs.update(rels)
            out[rel] = (str(prim.GetTypeName()), apis, attrs)
        return out

    c, a = dump("/World/conv"), dump("/World/auth")

    def norm(v):
        if v is None:
            return None
        if hasattr(v, "GetReal") and hasattr(v, "GetImaginary"):
            im = v.GetImaginary()
            return tuple(round(float(x), 7) for x in
                         (v.GetReal(), im[0], im[1], im[2]))
        if isinstance(v, (list, tuple)) or hasattr(v, "__len__") and not isinstance(v, str):
            try:
                return tuple(round(float(x), 7) for x in np.asarray(v).ravel())
            except Exception:                                      # noqa: BLE001
                return tuple(str(x) for x in v)
        if isinstance(v, float):
            return round(v, 7)
        return v

    print(f"\n[objdiff] prims: converted {len(c)}, authored {len(a)}")
    only_c = sorted(set(c) - set(a))
    only_a = sorted(set(a) - set(c))
    for p in only_c:
        print(f"[objdiff] ONLY IN CONVERTED: {p}  type={c[p][0]} apis={c[p][1]}")
    for p in only_a:
        print(f"[objdiff] ONLY IN AUTHORED : {p}  type={a[p][0]} apis={a[p][1]}")

    problems = len(only_c) + len(only_a)
    for p in sorted(set(c) & set(a)):
        tc, ac_, attrs_c = c[p]
        ta, aa_, attrs_a = a[p]
        if tc != ta:
            print(f"[objdiff] TYPE {p}: converted={tc} authored={ta}")
            problems += 1
        if set(ac_) != set(aa_):
            print(f"[objdiff] APIS {p}:\n    only-conv={sorted(set(ac_)-set(aa_))}"
                  f"\n    only-auth={sorted(set(aa_)-set(ac_))}")
            problems += 1
        for k in sorted(set(attrs_c) | set(attrs_a)):
            vc, va = norm(attrs_c.get(k)), norm(attrs_a.get(k))
            if k not in attrs_a:
                print(f"[objdiff] ATTR MISSING IN AUTHORED  {p}.{k} = {vc}")
                problems += 1
            elif k not in attrs_c:
                print(f"[objdiff] ATTR EXTRA IN AUTHORED    {p}.{k} = {va}")
                problems += 1
            elif vc != va:
                print(f"[objdiff] ATTR DIFFERS {p}.{k}:\n    conv={vc}\n    auth={va}")
                problems += 1

    print(f"\n[objdiff] {problems} difference(s)")
    del app
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
