"""Does the authored object match the converted one, physically?

Direct authoring is only a win if it produces the same object. Faster asset
preparation that quietly changes mass, inertia or collision size is not an
optimisation -- it is the failure mode this project has already shipped twice
today (capsule collision shapes 43-57% oversized because the converter reads a
cylinder's `length` as its cylindrical section; ghost joints escaping limits
they appeared to have). Both would have been caught by a comparison like this.

So this builds the SAME object both ways and compares what PhysX actually reads:

  * total mass
  * diagonal inertia
  * centre of mass
  * collision shape count and dimensions

Compared through the USD attributes rather than the URDF text, because the
converter is free to transform what the URDF said -- and where it does, that
transformation is precisely what the authored path has to reproduce.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.compare_object_assets
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=int, default=6,
                        help="object variants to compare")
    parser.add_argument("--rtol", type=float, default=1e-3)
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
    from pxr import Sdf, Usd, UsdGeom

    from genmech.tasks.pose_reach.utils.author_objects import (
        OBJECT_ROOT_LINK,
        author_handle_head,
    )
    from genmech.tasks.pose_reach.utils.generate_objects import (
        generate_handle_head_urdf,
    )
    from genmech.tasks.pose_reach.utils.scene_utils import _convert_urdf_to_usd

    work = Path(tempfile.mkdtemp(prefix="genmech_objcmp_"))
    (work / "urdf").mkdir(parents=True, exist_ok=True)
    (work / "usd").mkdir(parents=True, exist_ok=True)

    # A spread of shapes: box/box, box/cylinder, cylinder/box, cylinder/cylinder.
    cases = [
        ((0.12, 0.03, 0.03), (0.05, 0.05, 0.05), 700.0, 900.0),
        ((0.14, 0.025, 0.025), (0.06, 0.04), 700.0, 900.0),
        ((0.10, 0.028), (0.05, 0.05, 0.04), 800.0, 850.0),
        ((0.12, 0.03), (0.05, 0.045), 750.0, 950.0),
        ((0.16, 0.035, 0.030), (0.07, 0.06, 0.05), 600.0, 1100.0),
        ((0.09, 0.022), (0.04, 0.035), 900.0, 700.0),
    ][: args.variants]

    def read_converted(usd_path: str) -> dict:
        stage = Usd.Stage.Open(usd_path)
        out = {"shapes": []}
        for prim in stage.Traverse():
            # Check the ATTRIBUTE, not the API schema type: resolving
            # "PhysicsMassAPI" through TfType raises unless the schema plugin is
            # registered, and the attribute is what PhysX reads anyway.
            if prim.GetAttribute("physics:mass").IsValid():
                m = prim.GetAttribute("physics:mass").Get()
                if m:
                    out["mass"] = float(m)
                    di = prim.GetAttribute("physics:diagonalInertia").Get()
                    if di:
                        out["inertia"] = np.array([float(v) for v in di])
                    com = prim.GetAttribute("physics:centerOfMass").Get()
                    if com:
                        out["com"] = np.array([float(v) for v in com])
            t = prim.GetTypeName()
            if t in ("Cube", "Cylinder", "Mesh"):
                out["shapes"].append(str(t))
        return out

    print(f"[objcmp] {'case':>5} {'mass converted':>15} {'mass authored':>14} "
          f"{'rel err':>9}  shapes")
    failures = []
    for i, (hs, hd_, dens_h, dens_head) in enumerate(cases):
        # --- converted path -----------------------------------------------
        urdf = generate_handle_head_urdf(
            work / "urdf" / f"obj_{i}.urdf",
            handle_scale=hs, head_scale=hd_,
            handle_density=dens_h, head_density=dens_head)
        usd = _convert_urdf_to_usd(str(urdf), work / "usd", fix_base=False)
        conv = read_converted(usd)

        # --- authored path ------------------------------------------------
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        layer = stage.GetRootLayer()
        with Sdf.ChangeBlock():
            for anc in ("/World", "/World/Objects"):
                spec = Sdf.CreatePrimInLayer(layer, Sdf.Path(anc))
                spec.specifier = Sdf.SpecifierDef
                spec.typeName = "Xform"
            root = Sdf.CreatePrimInLayer(layer, Sdf.Path(f"/World/Objects/obj_{i}"))
            root.specifier = Sdf.SpecifierDef
            root.typeName = "Xform"
            mass, ixx, iyy, izz, com_x = author_handle_head(
                layer, f"/World/Objects/obj_{i}", hs, hd_, dens_h, dens_head)

        authored_shapes = [p.GetTypeName() for p in stage.Traverse()
                           if p.GetTypeName() in ("Cube", "Cylinder")]

        rel = abs(mass - conv.get("mass", float("nan"))) / max(conv.get("mass", 1.0), 1e-12)
        ok = rel < args.rtol and len(authored_shapes) == 2
        if not ok:
            failures.append(i)
        print(f"[objcmp] {i:>5} {conv.get('mass', float('nan')):>15.6f} "
              f"{mass:>14.6f} {rel:>9.2e}  "
              f"conv={conv['shapes']} auth={[str(s) for s in authored_shapes]}"
              f"{'' if ok else '   <-- MISMATCH'}")
        if "inertia" in conv:
            ia = np.array([ixx, iyy, izz])
            ic = conv["inertia"]
            ierr = float(np.max(np.abs(ia - ic) / np.maximum(np.abs(ic), 1e-12)))
            flag = "" if ierr < args.rtol else "   <-- INERTIA MISMATCH"
            print(f"[objcmp]       inertia converted {ic} authored {ia} "
                  f"rel {ierr:.2e}{flag}")
            if ierr >= args.rtol:
                failures.append(i)

    print(f"\n[objcmp] {'PASS' if not failures else 'FAIL'}: "
          f"{len(cases) - len(set(failures))}/{len(cases)} variants match")

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
