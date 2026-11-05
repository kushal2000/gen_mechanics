"""Colour a template scene's hands by joint kind, in the live stage (renders only; physics untouched).

The template scene (`author_template.py`) authors only colliders, so the renderer shows those, in default grey, and
URDF visual colours never reach it. `colour_template_hands` sets each collider's displayColor from the kind of the joint
that moves its body (flexion blue, abduction red, roll green, as in the audit figures; palm and wrist grey) and wraps
every roll tip's capsule in visual-only strips, light to dark green round it, so its spin shows. Needs Kit (pxr, omni);
call it after the env is built, e.g. from render_archive_videos.py --colour-joints.
"""
from __future__ import annotations

import math

import numpy as np

KIND_RGB = {(0, 0): (0.16, 0.42, 0.85), (90, 0): (0.86, 0.20, 0.18), (0, 90): (0.15, 0.65, 0.30)}
GREY = (0.62, 0.62, 0.64)
STRIPES = 8


def _frame(axis: str):
    a = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}[axis]
    u = np.array([0, 0, 1.0]) if axis != "Z" else np.array([1.0, 0, 0])
    u = u - a * (u @ a)
    u /= np.linalg.norm(u)
    return a, u, np.cross(a, u)


def colour_template_hands(env, designs: list, n_slots: int, stripes: bool = True) -> dict:
    """Colour every env's hand. `designs`: the grammar design dict of each hand, in `env.hand_set.specs` order;
    `n_slots`: joint slots per finger slot (the manifest grid). Returns counts of what was coloured."""
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, Vt
    from scipy.spatial.transform import Rotation
    from isaacsimenvs.common.robots.multi_hand import slot_link

    stage = omni.usd.get_context().get_stage()
    hs = env.hand_set
    counts = {"colliders": 0, "stripes": 0}
    for e, h in enumerate(env.hand_of_env.long().tolist()):
        spec, lay, des = hs.specs[h], hs.layouts[h], designs[h]
        names = list(spec.hand_joint_names)
        body_rgb, roll_tips = {}, set()
        for f, slot in enumerate(lay.finger_slot_of_finger):
            for k, jidx in enumerate(lay.fingers[f]):
                i, j = (int(x) for x in names[jidx][1:].split("_j"))
                joints = des["fingers"][i]["joints"]
                ax = tuple(joints[j]["axis"])
                body = slot_link(slot * n_slots + k)
                body_rgb[body] = KIND_RGB[ax]
                if j == len(joints) - 1 and ax == (0, 90):
                    roll_tips.add(body)
        root = stage.GetPrimAtPath(f"/World/envs/env_{e}/Robot")
        if not root.IsValid():
            continue
        caps = []
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            body = prim.GetPath().MakeRelativePath(root.GetPath()).pathString.split("/")[0]
            rgb = body_rgb.get(body, GREY)
            if body in roll_tips and prim.IsA(UsdGeom.Capsule):
                rgb = (0.60, 0.92, 0.60)                       # the roll tip itself light green; strips go round it
                caps.append(prim)
            UsdGeom.Gprim(prim).CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*rgb)]))
            counts["colliders"] += 1
        if not stripes:
            continue
        for cap_prim in caps:
            cap = UsdGeom.Capsule(cap_prim)
            r, hgt = float(cap.GetRadiusAttr().Get()), float(cap.GetHeightAttr().Get())
            a, u, v = _frame(str(cap.GetAxisAttr().Get()))
            parent = cap_prim.GetParent().GetPath()
            for k in range(STRIPES):
                th = 2 * math.pi * k / STRIPES
                d = math.cos(th) * u + math.sin(th) * v
                t = np.cross(a, d)
                q = Rotation.from_matrix(np.column_stack([d, t, a])).as_quat()         # x y z w
                cube = UsdGeom.Cube.Define(stage, parent.AppendChild(f"rolltip_stripe{k}"))
                cube.CreateSizeAttr(1.0)
                xf = UsdGeom.Xformable(cube)
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(*((r + 0.0006) * d)))
                xf.AddOrientOp().Set(Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2])))
                xf.AddScaleOp().Set(Gf.Vec3f(0.0012, 2 * r * math.sin(math.pi / STRIPES) * 0.98, max(hgt, 1e-3) * 0.98))
                s = k / (STRIPES - 1)
                cube.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.62 - 0.57 * s, 0.95 - 0.60 * s, 0.62 - 0.52 * s)]))
                counts["stripes"] += 1
    return counts
