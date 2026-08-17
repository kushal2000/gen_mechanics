"""Does the authored hand match the converted one?

The authoring path is only a win if it produces the same robot. Faster asset
preparation that quietly changes a mass, a joint limit or a collider size is not
an optimisation -- it is the failure mode this project has already shipped three
times: capsule colliders 43-57% oversized, ghost joints escaping limits, and
1,000 objects assigned to the wrong environments.

So this builds the SAME hand both ways and compares what PhysX will read:

  * rigid bodies      -- name set, mass, diagonal inertia, centre of mass,
                         principal axes
  * collision shapes  -- type, radius, height, and the COMPOSED transform in the
                         body's own frame (not the authored-time transform: the
                         object work proved those differ while composing the
                         same, and the composed one is what collides)
  * joints            -- body0/body1, local poses, axis, limits, drive gains

Compared through USD attributes rather than the URDF, because the converter is
free to transform what the URDF said, and where it does, that transformation is
precisely what the authored path has to reproduce.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.compare_authored_hand --hand gen_sharpa_like
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_args():
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hand", default="gen_sharpa_like")
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--atol", type=float, default=1e-6)
    p.add_argument("--show", action="store_true", help="print per-body detail")
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

    from genmech.robots import get_robot_spec
    from genmech.robots.generated.author_usd import author_hand, define
    from genmech.robots.generated.synth_spec import params_for_name
    from genmech.tasks.pose_reach.utils.scene_utils import _convert_urdf_to_usd

    spec = get_robot_spec(args.hand)
    hand = params_for_name(args.hand)
    work = Path(tempfile.mkdtemp(prefix="genmech_handcmp_"))

    # --- converted reference ------------------------------------------------
    conv_usd = _convert_urdf_to_usd(
        spec.urdf_path, work, fix_base=True, self_collision=True,
        replace_cylinders_with_capsules=spec.replace_cylinders_with_capsules)
    conv = Usd.Stage.Open(conv_usd)
    conv_root = next(c for c in conv.GetPseudoRoot().GetChildren())
    conv_root_path = str(conv_root.GetPath())

    # The authored hand is placed relative to the merged palm body, so read that
    # body's world transform out of the converted asset -- it is the same for
    # every design (the arm is identical), which is the whole premise.
    cache = UsdGeom.XformCache()
    link7 = conv.GetPrimAtPath(f"{conv_root_path}/iiwa14_link_7")
    link7_world = np.asarray(cache.GetLocalToWorldTransform(link7)).T

    # --- authored ------------------------------------------------------------
    omni.usd.get_context().new_stage()
    auth = omni.usd.get_context().get_stage()
    layer = auth.GetRootLayer()
    with Sdf.ChangeBlock():
        define(layer, "/robot", "Xform")
        define(layer, "/robot/iiwa14_link_7", "Xform",
               ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
        summary = author_hand(layer, "/robot", hand, spec,
                              palm_body_path="/robot/iiwa14_link_7",
                              link7_world=link7_world)
    print(f"[handcmp] authored {summary}")

    def bodies(stage, root):
        out = {}
        for prim in stage.Traverse():
            md = prim.GetMetadata("apiSchemas")
            ap = [str(x) for x in md.GetAppliedItems()] if md else []
            if not any("RigidBodyAPI" in s for s in ap):
                continue
            name = prim.GetPath().name
            if not name.startswith("gen_f"):
                continue
            g = lambda k: (prim.GetAttribute(k).Get()
                           if prim.GetAttribute(k) and prim.GetAttribute(k).IsValid()
                           else None)
            out[name] = {
                "mass": g("physics:mass"),
                "inertia": g("physics:diagonalInertia"),
                "com": g("physics:centerOfMass"),
                "axes": g("physics:principalAxes"),
            }
        return out

    def colliders(stage, base_usd=None):
        """Capsules keyed by body, with the transform in the BODY's frame.

        The converter does not put colliders inline: each link carries a
        ``collisions`` Xform that REFERENCES ``/colliders/<link>`` in
        ``configuration/<name>_base.usd``, and those references do not resolve
        when the top-level asset is opened on its own -- which made an earlier
        run of this tool report "converted 0 capsules" and compare nothing at
        all. Read the base layer directly when given, so the comparison has
        something to compare.
        """
        c = UsdGeom.XformCache()
        out = {}
        stages = [(stage, False)] + ([(Usd.Stage.Open(str(base_usd)), True)]
                                     if base_usd else [])
        for st, is_base in stages:
            for prim in st.Traverse():
                if str(prim.GetTypeName()) != "Capsule":
                    continue
                path = str(prim.GetPath())
                if is_base and not path.startswith("/colliders/"):
                    continue
                body = prim
                while body and not body.GetPath().name.startswith("gen_f"):
                    body = body.GetParent()
                    if not body or body.GetPath().pathString == "/":
                        body = None
                        break
                if body is None:
                    continue
                name = body.GetPath().name
                if name in out:
                    continue
                m_shape = np.asarray(c.GetLocalToWorldTransform(prim)).T
                m_body = np.asarray(c.GetLocalToWorldTransform(body)).T
                local = np.linalg.inv(m_body) @ m_shape
                g = lambda k: (prim.GetAttribute(k).Get()
                               if prim.GetAttribute(k) and prim.GetAttribute(k).IsValid()
                               else None)
                out[name] = {
                    "radius": float(g("radius")), "height": float(g("height")),
                    "axis": str(g("axis")), "pos": local[:3, 3].copy(),
                }
        return out

    def joints(stage):
        out = {}
        for prim in stage.Traverse():
            if "RevoluteJoint" not in str(prim.GetTypeName()):
                continue
            name = prim.GetPath().name
            if not name.startswith("gen_f"):
                continue
            g = lambda k: (prim.GetAttribute(k).Get()
                           if prim.GetAttribute(k) and prim.GetAttribute(k).IsValid()
                           else None)
            b0 = prim.GetRelationship("physics:body0")
            b1 = prim.GetRelationship("physics:body1")
            out[name] = {
                "body0": b0.GetTargets()[0].name if b0 and b0.GetTargets() else None,
                "body1": b1.GetTargets()[0].name if b1 and b1.GetTargets() else None,
                "axis": str(g("physics:axis")),
                "lower": g("physics:lowerLimit"), "upper": g("physics:upperLimit"),
                "localPos0": g("physics:localPos0"), "localRot0": g("physics:localRot0"),
                "localPos1": g("physics:localPos1"),
                "stiffness": g("drive:angular:physics:stiffness"),
                "damping": g("drive:angular:physics:damping"),
                "maxForce": g("drive:angular:physics:maxForce"),
            }
        return out

    fails: list[str] = []

    def _vals(v):
        """Gf.Quat*/Vec* -> flat float array."""
        if hasattr(v, "GetReal") and hasattr(v, "GetImaginary"):
            im = v.GetImaginary()
            q = np.asarray([float(v.GetReal()), float(im[0]), float(im[1]),
                            float(im[2])])
            # q and -q are the same rotation; pin the sign so they compare equal.
            return q if (q[np.argmax(np.abs(q))] >= 0) else -q
        return np.asarray(v, dtype=float).ravel()

    def close(a, b) -> bool:
        if a is None or b is None:
            return a is None and b is None
        return bool(np.allclose(_vals(a), _vals(b),
                                rtol=args.rtol, atol=args.atol))

    cb, ab = bodies(conv, conv_root_path), bodies(auth, "/robot")
    print(f"\n[handcmp] bodies: converted {len(cb)}, authored {len(ab)}")
    if set(cb) != set(ab):
        fails.append(f"body name sets differ: only-conv={sorted(set(cb)-set(ab))[:5]} "
                     f"only-auth={sorted(set(ab)-set(cb))[:5]}")
    for name in sorted(set(cb) & set(ab)):
        for key in ("mass", "inertia", "com", "axes"):
            if not close(cb[name][key], ab[name][key]):
                fails.append(f"{name}.{key}: conv={cb[name][key]} auth={ab[name][key]}")
            elif args.show and key == "mass":
                print(f"[handcmp]   {name}.mass ok ({cb[name][key]:.8f})")

    base_usd = Path(conv_usd).parent / 'configuration' / \
        f"{Path(conv_usd).stem}_base.usd"
    cc = colliders(conv, base_usd if base_usd.exists() else None)
    ac = colliders(auth)
    print(f"[handcmp] capsules: converted {len(cc)}, authored {len(ac)}")
    if set(cc) != set(ac):
        fails.append(f"capsule body sets differ: only-conv={sorted(set(cc)-set(ac))[:5]} "
                     f"only-auth={sorted(set(ac)-set(cc))[:5]}")
    for name in sorted(set(cc) & set(ac)):
        for key in ("radius", "height", "pos"):
            if not close(cc[name][key], ac[name][key]):
                fails.append(f"{name}.capsule.{key}: conv={cc[name][key]} "
                             f"auth={ac[name][key]}")
        if cc[name]["axis"] != ac[name]["axis"]:
            fails.append(f"{name}.capsule.axis: {cc[name]['axis']} vs {ac[name]['axis']}")

    cj, aj = joints(conv), joints(auth)
    print(f"[handcmp] joints: converted {len(cj)}, authored {len(aj)}")
    if set(cj) != set(aj):
        fails.append(f"joint name sets differ: only-conv={sorted(set(cj)-set(aj))[:5]} "
                     f"only-auth={sorted(set(aj)-set(cj))[:5]}")
    for name in sorted(set(cj) & set(aj)):
        for key in ("body0", "body1", "axis"):
            if cj[name][key] != aj[name][key]:
                fails.append(f"{name}.{key}: conv={cj[name][key]} auth={aj[name][key]}")
        for key in ("lower", "upper", "localPos0", "localRot0", "localPos1",
                    "stiffness", "damping", "maxForce"):
            if not close(cj[name][key], aj[name][key]):
                fails.append(f"{name}.{key}: conv={cj[name][key]} auth={aj[name][key]}")

    print()
    for f in fails[:40]:
        print(f"[handcmp] MISMATCH {f}")
    if len(fails) > 40:
        print(f"[handcmp] ... and {len(fails) - 40} more")
    ok = not fails
    print(f"\n[handcmp] {'PASS' if ok else 'FAIL'}: {len(fails)} mismatch(es); "
          f"authored hand {'matches' if ok else 'DIFFERS FROM'} the converter")

    del app
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
