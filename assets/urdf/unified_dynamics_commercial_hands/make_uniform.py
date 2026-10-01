"""Uniform-dynamics copies of the commercial hands: same kinematics and geometry, gen-SHARPA's dynamics.

    .venv_isaacsim/bin/python assets/urdf/unified_dynamics_commercial_hands/make_uniform.py

For each hand, reads its unified URDF (``../unified_commercial_hands/<hand>/``) and writes
``<hand>/<hand>_left.urdf`` here plus a ``<hand>_left.spec.json`` for the RobotSpec, changing ONLY
dynamics, following the generated hands' rules (``hand_sampler/robot_param_constants.py``) at 0.5 N.m:

  every joint     <limit effort=0.5 N.m velocity=10 rad/s>, <dynamics damping=DAMPING friction=0>;
                  lower/upper (kinematics) untouched
  spec gains      stiffness 3.0, damping 0.078, armature 0.00058 (see EFFORT below) -- gains are the
                  actuator config, not the URDF
  finger links    mass, centre of mass and inertia recomputed from the link's collision hulls at
                  LINK_DENSITY_KG_M3 = 1750 kg/m^3 -- NOT gen-SHARPA's 2712: dividing each vendor's own
                  finger-link masses by their hull volumes gives a median of 1749 kg/m^3 (IQR 1190-2510)
                  across all 8 hands, so 1750 keeps real finger masses while making them uniform; a link
                  with no collision geometry keeps a 1 g / tiny-inertia placeholder
  friction        not a URDF property: uniform runs set every contact to 0.5 in the env config
                  (assets.robot_friction, finger_tip_friction, object_friction)
  palm (root)     vendor inertial kept: the base is fixed, so the palm never moves

Meshes are not copied: every mesh path is rewritten to point into ../unified_commercial_hands, so the
geometry stays single-sourced. Joint names, ranges, axes and origins are byte-for-byte the source's.
"""
from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SRC = REPO / "assets/urdf/unified_commercial_hands"
sys.path.insert(0, str(REPO))
from hand_sampler import robot_param_constants as rpc  # noqa: E402

# Torque limit 0.5 N.m (user, 1 Oct): 1.0 is a stall-torque figure; 0.5 sits nearer real continuous
# ratings (Tesollo 0.4 rated, Wuji v2 0.3 on its outer joints). Everything else follows gen-SHARPA's RULES
# rather than its 1.0 N.m values, so the actuator stays self-consistent:
#   armature  = 0.00116 x torque limit                   (SHARPA's armature-per-torque, as gen-SHARPA)
#   stiffness = 3.0 N.m/rad -- NOT gen-SHARPA's "limit / 1 rad" (0.5 here): that tracks with a 2-step lag
#               (natural frequency 29 rad/s); 3.0 is Isaac Lab's Allegro reorientation setting at this same
#               0.5 N.m limit and what all 9 converged ten-hands runs used (saturates at 0.17 rad, ~72 rad/s)
#   damping   = 2 x 0.929 x sqrt(stiffness x armature)   (critically damped, SHARPA's ratio)
# Peak joint acceleration (limit / armature ~ 860 rad/s^2) is gen-SHARPA's; damping 0.0775 is the same
# damping ratio (~0.93) the converged runs had at stiffness 3.0 / armature 0.001.
EFFORT = 0.5
VELOCITY = rpc.GEN_JOINT_VELOCITY_RAD_S                  # 10 rad/s
ARMATURE = 0.00116 * EFFORT
STIFFNESS = 3.0
DAMPING = 2 * 0.929 * (STIFFNESS * ARMATURE) ** 0.5
LINK_DENSITY_KG_M3 = 1750.0          # see docstring: the vendors' own effective density
DENSITY = LINK_DENSITY_KG_M3
FRICTION = 0.5                       # every contact, set by the run config (not the URDF)
PLACEHOLDER_MASS, PLACEHOLDER_I = 1e-3, 1e-8

# hand -> source URDF (relative to SRC). SHARPA's unified URDF predates onboard.py's naming.
HANDS = {h: f"{h}/{h}_left.urdf" for h in ("allegro", "leap", "shadow", "dex3", "tesollo", "wuji2", "xhand")}
HANDS["sharpa"] = "sharpa/sharpa.urdf"


def _origin(el) -> np.ndarray:
    T = np.eye(4)
    o = el.find("origin") if el is not None else None
    if o is not None:
        xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
        T = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
        T[:3, 3] = xyz
    return T


def _collision_parts(link, base: Path) -> list[trimesh.Trimesh]:
    """The link's collision geometry as convex hulls in the LINK frame (what PhysX collides)."""
    parts = []
    for col in link.findall("collision"):
        g = col.find("geometry")
        if g is None:
            continue
        if g.find("mesh") is not None:
            m = g.find("mesh")
            mesh = trimesh.load(base / m.get("filename"), force="mesh")
            if m.get("scale"):
                mesh.apply_scale([float(v) for v in m.get("scale").split()])
        elif g.find("box") is not None:
            mesh = trimesh.creation.box([float(v) for v in g.find("box").get("size").split()])
        elif g.find("sphere") is not None:
            mesh = trimesh.creation.icosphere(radius=float(g.find("sphere").get("radius")), subdivisions=2)
        elif g.find("cylinder") is not None:
            c = g.find("cylinder")
            mesh = trimesh.creation.cylinder(float(c.get("radius")), float(c.get("length")))
        else:
            continue
        hull = mesh.convex_hull
        hull.apply_transform(_origin(col))
        if hull.volume > 1e-12:
            parts.append(hull)
    return parts


def _mass_props(parts) -> tuple[float, np.ndarray, np.ndarray]:
    """(mass, centre of mass, inertia about it in link axes) of solid hulls at DENSITY."""
    if not parts:
        return PLACEHOLDER_MASS, np.zeros(3), np.eye(3) * PLACEHOLDER_I
    masses, coms, inertias = [], [], []
    for p in parts:
        p.density = DENSITY
        masses.append(p.mass)
        coms.append(p.center_mass)
        inertias.append(p.moment_inertia)          # about the part's own centre of mass
    M = float(sum(masses))
    com = sum(m * c for m, c in zip(masses, coms)) / M
    I = np.zeros((3, 3))
    for m, c, Ii in zip(masses, coms, inertias):
        d = c - com
        I += Ii + m * (d @ d * np.eye(3) - np.outer(d, d))   # parallel axis
    return M, com, 0.5 * (I + I.T)


def make(hand: str, rel: str) -> dict:
    src = SRC / rel
    out_dir = HERE / hand
    out_dir.mkdir(exist_ok=True)
    root = ET.parse(src).getroot()
    child_links = {j.find("child").get("link") for j in root.findall("joint")}
    palm = next(l.get("name") for l in root.findall("link") if l.get("name") not in child_links)

    summary = {"links": {}, "palm": palm}
    for link in root.findall("link"):
        if link.get("name") == palm:
            continue
        M, com, I = _mass_props(_collision_parts(link, src.parent))
        if M < PLACEHOLDER_MASS:            # a sliver of a hull: floor it (scaling inertia with it)
            I, M = I * (PLACEHOLDER_MASS / M), PLACEHOLDER_MASS
        old = link.find("inertial")
        if old is not None:
            link.remove(old)
        inertial = ET.Element("inertial")
        ET.SubElement(inertial, "origin", {"xyz": " ".join(f"{v:.6g}" for v in com), "rpy": "0 0 0"})
        ET.SubElement(inertial, "mass", {"value": f"{M:.6g}"})
        ET.SubElement(inertial, "inertia", {
            "ixx": f"{I[0, 0]:.6g}", "ixy": f"{I[0, 1]:.6g}", "ixz": f"{I[0, 2]:.6g}",
            "iyy": f"{I[1, 1]:.6g}", "iyz": f"{I[1, 2]:.6g}", "izz": f"{I[2, 2]:.6g}"})
        link.insert(0, inertial)
        summary["links"][link.get("name")] = round(M, 5)

    n_joints = 0
    for j in root.findall("joint"):
        if j.get("type") not in ("revolute", "continuous", "prismatic"):
            continue
        n_joints += 1
        lim = j.find("limit")
        if lim is None:
            lim = ET.SubElement(j, "limit")
        lim.set("effort", f"{EFFORT}")
        lim.set("velocity", f"{VELOCITY}")
        for d in j.findall("dynamics"):
            j.remove(d)
        ET.SubElement(j, "dynamics", {"damping": f"{DAMPING:.6g}", "friction": "0"})

    for m in root.iter("mesh"):                     # geometry stays in unified_commercial_hands
        m.set("filename", os.path.relpath(src.parent / m.get("filename"), out_dir))

    root.insert(0, ET.Comment(
        f" GENERATED by unified_dynamics_commercial_hands/make_uniform.py from unified_commercial_hands/{rel}: "
        f"kinematics and geometry unchanged; every joint effort {EFFORT} N.m, velocity {VELOCITY} rad/s, "
        f"damping {DAMPING:.4g}; link inertials from collision hulls at {DENSITY:.0f} kg/m^3 (palm kept). "
        "Do not hand-edit. "))
    dst = out_dir / f"{hand}_left.urdf"
    ET.ElementTree(root).write(dst, xml_declaration=True, encoding="utf-8")

    spec_src = SRC / hand / f"{hand}_left.spec.json"
    if spec_src.is_file():
        spec = json.loads(spec_src.read_text())
        spec.update({"urdf": str(dst.relative_to(REPO)), "stiffness": STIFFNESS, "damping": DAMPING,
                     "armature": ARMATURE, "effort_limit": EFFORT, "velocity_limit": VELOCITY,
                     "link_density_kg_m3": DENSITY, "friction": FRICTION})
        (out_dir / f"{hand}_left.spec.json").write_text(json.dumps(spec, indent=1))
    summary.update({"joints": n_joints, "finger_mass_kg": round(sum(summary["links"].values()), 4)})
    return summary


if __name__ == "__main__":
    report = {}
    for hand, rel in HANDS.items():
        s = make(hand, rel)
        report[hand] = s
        ms = sorted(s["links"].values())
        print(f"{hand:8s} {s['joints']:2d} joints  finger links {len(ms):2d}  total {s['finger_mass_kg']:.3f} kg  "
              f"median {1000 * np.median(ms):5.1f} g  min {1000 * ms[0]:6.2f} g  palm '{s['palm']}' kept")
    (HERE / "masses.json").write_text(json.dumps(report, indent=1))
