"""A grammar hand (``hand_sampler.grammar.hand.Hand``) as the files the simulator loads: URDF, RobotSpec JSON, design
JSON and the plate meshes. Promoted 29 Oct from experiments/27oct_grammar_random_hands/make_random.py (which built
the 100 random hands with it, and now imports it) so embodiment evolution can build its children with the same code.
Pure Python (numpy, trimesh): no simulator.

    from hand_sampler.grammar.urdf_export import build
    row = build(hand, "my_hand", out_dir)       # -> out_dir/my_hand.{urdf,spec.json,design.json}, *.obj

Geometry is the grammar's own solids at the zero pose (``derive.solids_at``), each on the body that owns it
(``derive.solid_owner``): the plates and the mount as convex meshes, the base and hinge motors as boxes on their
plate, each link its box, the fingertip a capsule (a URDF cylinder the importer turns into a capsule:
``replace_cylinders_with_capsules``; nothing else is a cylinder). Bodies: ``palm`` (the root), ``palm0`` (the part
beyond the cut, hinged by ``palm0_joint``), ``f<i>_link<j>`` (joint ``f<i>_j<j>``). The palm frame is the grammar's:
x the grasp normal, the plate in y-z, the mount behind the back face (-x) -- so ``palm_frame`` is the identity.

Dynamics: the uniform commercial hands' (``assets/urdf/unified_dynamics_commercial_hands/make_uniform.py``), NOT the
grammar's own Feetech effort: effort 0.5 N.m, 5 rad/s, stiffness 3.0, damping 0.0775, armature 0.00058, link density
1750 kg/m^3, friction 0.5 (set by the run config, not the URDF). Self-collision: filtered for the pairs that touch by
construction (``viability.touching_pairs``).
"""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from hand_sampler import REPO_ROOT as REPO
from hand_sampler.grammar import derive as dv
from hand_sampler.grammar import hand as gh
from hand_sampler.grammar import viability as gvb

# The uniform commercial hands' actuation (make_uniform.py: EFFORT, VELOCITY, STIFFNESS, DAMPING, ARMATURE, DENSITY,
# FRICTION); check_uniform() asserts they still match.
EFFORT, VELOCITY = 0.5, 5.0
STIFFNESS, ARMATURE = 3.0, 0.00116 * 0.5
DAMPING = 2 * 0.929 * (STIFFNESS * ARMATURE) ** 0.5
DENSITY, FRICTION = 1750.0, 0.5
RULES, RULES_NAME = gh.FEETECH_COARSE_RULES, "FEETECH_COARSE_RULES"

RGBA = {"palm": "0.55 0.57 0.62 1", "motor": "0.15 0.15 0.17 1", "link": "0.80 0.62 0.30 1",
        "tip": "0.85 0.35 0.25 1"}


def _fmt(v) -> str:
    return " ".join(f"{float(x):.9g}" for x in v)


def _rpy(R: np.ndarray) -> tuple[float, float, float]:
    from hand_sampler.grammar.fk import matrix_to_rpy
    return tuple(float(v) for v in matrix_to_rpy(R))


def body_name(owner: tuple) -> str:
    if owner[0] == "palm":
        return "palm" if owner[1] < 0 else f"palm{owner[1]}"
    return f"f{owner[1]}_link{owner[2]}"


def body_frames(hand: gh.Hand) -> dict[str, np.ndarray]:
    """Each body's frame at the zero pose, in the palm frame (4x4, metres): the URDF frames ``derive.derive`` uses."""
    pose = dv.fk(hand)
    T = {"palm": np.eye(4)}
    for k in range(len(hand.palm_joints)):
        Tk = np.eye(4)
        Tk[:3, 3] = dv.palm_hinge_m(hand, k)
        T[f"palm{k}"] = Tk
    for i, f in enumerate(hand.fingers):
        for j in range(len(f.joints)):
            T[f"f{i}_link{j}"] = pose.links[i][j]
    return T


def _shape(link: ET.Element, kind: str, geom: tuple[str, dict], xyz, rpy=(0.0, 0.0, 0.0), rgba=None, name=None):
    el = ET.SubElement(link, kind, **({"name": name} if name else {}))
    ET.SubElement(el, "origin", xyz=_fmt(xyz), rpy=_fmt(rpy))
    g = ET.SubElement(el, "geometry")
    ET.SubElement(g, geom[0], **{k: (v if isinstance(v, str) else _fmt(np.atleast_1d(v))) for k, v in geom[1].items()})
    if rgba is not None:
        mat = ET.SubElement(el, "material", name=f"{name}_mat" if name else "mat")
        ET.SubElement(mat, "color", rgba=rgba)
    return el


def _inertial(link: ET.Element, mass: float, com, inertia_diag, rpy=(0.0, 0.0, 0.0)) -> None:
    el = ET.SubElement(link, "inertial")
    ET.SubElement(el, "origin", xyz=_fmt(com), rpy=_fmt(rpy))
    ET.SubElement(el, "mass", value=f"{mass:.9g}")
    ix, iy, iz = (float(v) for v in inertia_diag)
    ET.SubElement(el, "inertia", ixx=f"{ix:.9g}", ixy="0", ixz="0", iyy=f"{iy:.9g}", iyz="0", izz=f"{iz:.9g}")


def _plate_mass(points_m: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """(mass, centre of mass, principal inertia, principal axes) of a convex prism at DENSITY."""
    m = trimesh.convex.convex_hull(points_m)
    m.density = DENSITY
    vals, vecs = trimesh.inertia.principal_axis(m.moment_inertia)
    if np.linalg.det(vecs) < 0:
        vecs = -vecs
    return float(m.mass), m.center_mass, vals, vecs.T      # principal_axis gives the axes as rows


def build(hand: gh.Hand, name: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    T = body_frames(hand)
    links = {b: ET.Element("link", name=b) for b in T}
    counts: dict[str, int] = {}
    for s in dv.solids_at(hand):
        b = body_name(dv.solid_owner(hand, s.key))
        Ti = np.linalg.inv(T[b])
        k = counts[b] = counts.get(b, 0) + 1
        label = f"{b}_{s.key[0]}{k}"
        if s.box is not None:                          # a motor or a link box (palm frame, mm)
            R = Ti[:3, :3] @ s.box.axes
            c = Ti[:3, :3] @ (s.box.centre * dv.MM) + Ti[:3, 3]
            geom = ("box", {"size": 2.0 * s.box.half * dv.MM})
            rgba = RGBA["motor" if s.key[0] in ("motor", "hinge") else "link"]
            rpy = _rpy(R)
        elif s.radius > 0:                             # the fingertip capsule: a cylinder along its segment
            p0, p1 = (Ti[:3, :3] @ p + Ti[:3, 3] for p in s.points)
            d = p1 - p0
            L = float(np.linalg.norm(d))
            z = d / L if L > 1e-12 else np.array([1.0, 0.0, 0.0])
            x = np.cross(z, [0.0, 0.0, 1.0] if abs(z[2]) < 0.9 else [0.0, 1.0, 0.0])
            x /= np.linalg.norm(x)
            R = np.column_stack([x, np.cross(z, x), z])
            c, rpy = (p0 + p1) / 2.0, _rpy(R)
            geom = ("cylinder", {"radius": s.radius, "length": max(L, 1e-6)})
            rgba = RGBA["tip"]
        else:                                          # a plate or the mount: a convex prism, as a mesh
            pts = s.points @ Ti[:3, :3].T + Ti[:3, 3]
            fn = f"{label}.obj"
            trimesh.convex.convex_hull(pts).export(out_dir / fn)
            geom, c, rpy, rgba = ("mesh", {"filename": fn}), np.zeros(3), (0.0, 0.0, 0.0), RGBA["palm"]
        _shape(links[b], "visual", geom, c, rpy, rgba=rgba, name=label)
        _shape(links[b], "collision", geom, c, rpy)

    # mass: each link its box / capsule (the grammar's), each plate its prism
    g = dv.hand_geometry(hand)
    for b, link in links.items():
        if b.startswith("palm"):
            sec = -1 if b == "palm" else int(b[4:])
            pts = dv.plate_points_m(g.outlines[sec])
            pts = pts @ np.linalg.inv(T[b])[:3, :3].T + np.linalg.inv(T[b])[:3, 3]
            m, com, I, axes = _plate_mass(pts)
            _inertial(link, m, com, I, _rpy(axes))
        else:
            i, j = (int(v) for v in b[1:].split("_link"))
            m, com, I = dv.link_mass_props(hand, i, j)
            _inertial(link, m, com, I)

    model = dv.derive(hand, name=name)
    root = ET.Element("robot", name=name)
    for b in links:
        root.append(links[b])
    joints, limits = [], {}
    for jt in model.joints:
        el = ET.SubElement(root, "joint", name=jt.name, type=jt.type)
        ET.SubElement(el, "parent", link=jt.parent)
        ET.SubElement(el, "child", link=jt.child)
        ET.SubElement(el, "origin", xyz=_fmt(jt.origin.xyz), rpy=_fmt(jt.origin.rpy))
        ET.SubElement(el, "axis", xyz=_fmt(jt.axis))
        lo, hi = (float(v) for v in jt.limits)
        ET.SubElement(el, "limit", effort=f"{EFFORT:g}", lower=f"{lo:.9g}", upper=f"{hi:.9g}", velocity=f"{VELOCITY:g}")
        ET.SubElement(el, "dynamics", damping=f"{DAMPING:.6g}", friction="0")
        joints.append(jt.name)
        limits[jt.name] = [lo, hi]
    assert not model.couplings, "the coarse rules allow hinges only"

    adj: dict[str, set[str]] = {b: set() for b in links}
    for a, c in gvb.touching_pairs(hand):
        ba, bc = body_name(dv.solid_owner(hand, a)), body_name(dv.solid_owner(hand, c))
        if ba != bc:
            adj[ba].add(bc)

    tree = ET.ElementTree(root)
    ET.indent(tree)
    urdf = out_dir / f"{name}.urdf"
    tree.write(urdf, encoding="unicode", xml_declaration=False)

    outline = g.outline                                 # the uncut palm, (y, z) mm
    lo_yz, hi_yz = outline.min(0) * dv.MM, outline.max(0) * dv.MM
    centre = [0.0, *((lo_yz + hi_yz) / 2.0)]
    n_tips = len(hand.fingers)
    spec = {
        "hand": name, "side": "left", "urdf": str(urdf.relative_to(REPO)) if urdf.is_relative_to(REPO) else str(urdf), "palm_body": "palm",
        "hand_joint_names": joints, "joint_limits": limits,
        "hand_default_joint_pos": {j: 0.0 for j in joints},          # the grammar's start pose (fingers open)
        "fingertip_body_names": [f"f{i}_link{len(f.joints) - 1}" for i, f in enumerate(hand.fingers)],
        "fingertip_offsets": [[f.joints[-1].length * dv.MM, 0.0, 0.0] for f in hand.fingers],
        "palm_frame": np.eye(3).tolist(), "palm_frame_det": 1.0,
        "palm_centre_in_palm_body": centre,
        "palm_extents_ours": [gh.PALM_THICKNESS_MM * dv.MM, *(hi_yz - lo_yz).tolist()],
        "palm_centre_ours": centre,
        "adjacent_links": {b: sorted(v) for b, v in adj.items()},
        "stiffness": STIFFNESS, "damping": DAMPING, "armature": ARMATURE,
        "effort_limit": EFFORT, "velocity_limit": VELOCITY,
        "link_density_kg_m3": DENSITY, "friction": FRICTION,
        "replace_cylinders_with_capsules": True,
        "rules": RULES_NAME, "design": gh.hand_to_dict(hand),
    }
    (out_dir / f"{name}.spec.json").write_text(json.dumps(spec, indent=1))
    (out_dir / f"{name}.design.json").write_text(json.dumps(gh.hand_to_dict(hand), indent=1))
    c2 = gvb.c2_workspace_overlap(hand)
    return {"name": name, "spec": spec["urdf"].replace(".urdf", ".spec.json"), "fingers": n_tips,
            "joints": len(joints), "palm_cut": bool(hand.palm_joints),
            "joints_per_finger": [len(f.joints) for f in hand.fingers],
            "c2_ok": bool(c2.ok), "c2_best_mm": None if math.isinf(c2.best_mm) else round(float(c2.best_mm), 1)}


__all__ = ["ARMATURE", "DAMPING", "DENSITY", "EFFORT", "FRICTION", "RULES", "RULES_NAME", "STIFFNESS", "VELOCITY",
           "body_frames", "body_name", "build"]
