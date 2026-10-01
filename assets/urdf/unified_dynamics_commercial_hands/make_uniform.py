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

JOINT CONVENTIONS are canonicalised too (kinematics unchanged -- every link lands where it did): for
each joint, the direction its link moves at +q is taken at the home pose in our palm frame (x = grasp
normal, y = width, z = wrist -> fingertip). Mostly sideways (along y) -> a SPREAD joint, and +q must move
toward +y (all hands are left hands in the same palm convention, so +y is the same anatomical side);
otherwise a FLEXION joint, and +q must move the fingertip toward the grasp point 5 cm in front of the palm
centre (curling, for a finger; it also gives the thumb a direction). A joint that points the other way is flipped (axis and
limits negated), and a joint whose 0 lies outside its range is re-zeroed at its home pose (the joint origin
rotated by that angle). Recorded per joint in canonical_map.json as q_new = sign * q_vendor - offset, so
the same normalised target means the same motion on every hand.

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
SHARPA_PALM = "left_hand_C_MC"


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


def _palm_frame(hand: str) -> np.ndarray:
    """Columns: our (normal, width, finger) axes in the palm body's frame (onboard.py's measurement)."""
    spec = SRC / hand / f"{hand}_left.spec.json"
    if spec.is_file():
        return np.asarray(json.loads(spec.read_text())["palm_frame"], float)
    sys.path.insert(0, str(SRC))
    import onboard
    cfg = dict(vendor=HANDS[hand], palm=SHARPA_PALM, tips=list(rpc.FINGERTIP_BODY_NAMES), thumb="thumb")
    return np.asarray(onboard.measure(hand, cfg, (SRC / HANDS[hand]).resolve(), "left")["palm_frame"], float)


def _tips(hand: str):
    """The spec's fingertip bodies and pad offsets (in each tip body's frame)."""
    spec = SRC / hand / f"{hand}_left.spec.json"
    if spec.is_file():
        s = json.loads(spec.read_text())
        return list(s["fingertip_body_names"]), [np.asarray(o, float) for o in s["fingertip_offsets"]]
    off = np.asarray(rpc.FINGERTIP_OFFSET, float)
    return list(rpc.FINGERTIP_BODY_NAMES), [off] * len(rpc.FINGERTIP_BODY_NAMES)


GRASP_OFFSET_M = 0.05   # the grasp point: 5 cm in front of the palm centre, where the object sits


def _palm_centre_ours(hand: str) -> np.ndarray:
    spec = SRC / hand / f"{hand}_left.spec.json"
    if spec.is_file():
        return np.asarray(json.loads(spec.read_text())["palm_centre_ours"], float)
    sys.path.insert(0, str(SRC))
    import onboard
    cfg = dict(vendor=HANDS[hand], palm=SHARPA_PALM, tips=list(rpc.FINGERTIP_BODY_NAMES), thumb="thumb")
    return np.asarray(onboard.measure(hand, cfg, (SRC / HANDS[hand]).resolve(), "left")["palm_centre_ours"], float)


def canonical_map(src: Path, F: np.ndarray, tips, pads, centre_ours: np.ndarray) -> dict:
    """Per actuated joint: kind (flex / spread), sign, offset (radians), so q_new = sign*q_vendor - offset.

    The direction is where the joint's FINGERTIP pad moves at +q (axis x (pad - pivot)), at the home pose
    in our palm axes -- not the child link's own box, which for the first joint of a two-joint knuckle is a
    stub sitting on the pivot and says nothing about which way the finger goes.
    """
    import yourdfpy
    robot = yourdfpy.URDF.load(str(src), build_scene_graph=True, load_meshes=False)
    names = [j.name for j in robot.joint_map.values() if j.type in ("revolute", "continuous")]
    home = {n: float(np.clip(0.0, robot.joint_map[n].limit.lower, robot.joint_map[n].limit.upper)) for n in names}
    robot.update_cfg(home)
    palm = robot.base_link
    pad_pos = {t: (robot.get_transform(frame_to=t, frame_from=palm) @ np.append(o, 1.0))[:3]
               for t, o in zip(tips, pads)}
    children: dict[str, list[str]] = {}
    for j in robot.joint_map.values():
        children.setdefault(j.parent, []).append(j.child)

    def subtree(link):
        out, stack = set(), [link]
        while stack:
            l = stack.pop(); out.add(l); stack += children.get(l, [])
        return out

    def tip_motion(n, cfg):
        robot.update_cfg(cfg)
        j = robot.joint_map[n]
        below = subtree(j.child)
        mine = [(robot.get_transform(frame_to=t, frame_from=palm) @ np.append(o, 1.0))[:3]
                for t, o in zip(tips, pads) if t in below]
        if not mine:
            raise ValueError(f"{src.name}: joint {n} has no fingertip below it")
        T = robot.get_transform(frame_to=j.child, frame_from=palm)
        axis_palm = T[:3, :3] @ (np.asarray(j.axis, float) / np.linalg.norm(j.axis))
        lever = np.mean(mine, axis=0) - T[:3, 3]
        raw = F.T @ np.cross(axis_palm, lever)
        tip_ours = F.T @ np.mean(mine, axis=0)
        return raw, float(np.linalg.norm(raw) / max(np.linalg.norm(lever), 1e-12)), tip_ours

    out = {}
    for n in names:
        j = robot.joint_map[n]
        raw, frac, tip = tip_motion(n, home)
        via = "home"
        if frac < 0.3:
            # The axis points along the straight finger (Allegro's finger bases, outstretched thumbs): at
            # home the joint only twists the finger. Curl everything below it to mid-range and look again,
            # as UHAS does -- that is the motion the joint produces once the finger is in use.
            below = subtree(j.child)
            curled = dict(home)
            for m in names:
                jm = robot.joint_map[m]
                if m != n and jm.child in below:
                    curled[m] = 0.5 * (jm.limit.lower + jm.limit.upper)
            raw, frac, tip = tip_motion(n, curled)
            via = "curled"
        bend = raw / max(np.linalg.norm(raw), 1e-12)
        # SPREAD only if the tip moves mostly sideways (along the palm width); anything else is FLEX, and +q
        # must move the tip toward the grasp point in front of the palm -- for a finger that is curling
        # toward the palm, and it gives the thumb (whose joints move its tip in every direction) one too.
        grasp = centre_ours + np.array([GRASP_OFFSET_M, 0.0, 0.0])
        to_grasp = (grasp - tip) / max(np.linalg.norm(grasp - tip), 1e-12)
        kind = "spread" if abs(bend[1]) >= 0.7 else "flex"
        sign = 1.0 if (bend @ to_grasp if kind == "flex" else bend[1]) >= 0 else -1.0
        lo, hi = (j.limit.lower, j.limit.upper) if sign > 0 else (-j.limit.upper, -j.limit.lower)
        out[n] = {"kind": kind, "sign": sign, "offset": float(np.clip(0.0, lo, hi)),
                  "tip_motion_ours": [round(float(v), 3) for v in bend], "measured_at": via,
                  "toward_grasp": round(float(bend @ to_grasp), 3),
                  "motion_per_lever": round(frac, 3)}
    return out


def _apply_canonical(root, cmap: dict) -> None:
    """Rewrite joint axes, origins and limits in place so q_new = sign*q_vendor - offset."""
    for j in root.findall("joint"):
        c = cmap.get(j.get("name"))
        if c is None:
            continue
        ax = j.find("axis")
        axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()])
        lim = j.find("limit")
        lo, hi = float(lim.get("lower")), float(lim.get("upper"))
        if c["sign"] < 0:
            axis, lo, hi = -axis, -hi, -lo
        if c["offset"] != 0.0:
            T = _origin(j) @ trimesh.transformations.rotation_matrix(c["offset"], axis)
            o = j.find("origin")
            if o is None:
                o = ET.SubElement(j, "origin")
            o.set("xyz", " ".join(f"{v:.9g}" for v in T[:3, 3]))
            o.set("rpy", " ".join(f"{v:.9g}" for v in trimesh.transformations.euler_from_matrix(T, axes="sxyz")))
            lo, hi = lo - c["offset"], hi - c["offset"]
        if ax is None:
            ax = ET.SubElement(j, "axis")
        ax.set("xyz", " ".join(f"{v:.9g}" for v in axis))
        lim.set("lower", f"{lo:.9g}")
        lim.set("upper", f"{hi:.9g}")


def make(hand: str, rel: str) -> dict:
    src = SRC / rel
    out_dir = HERE / hand
    out_dir.mkdir(exist_ok=True)
    root = ET.parse(src).getroot()
    cmap = canonical_map(src, _palm_frame(hand), *_tips(hand), _palm_centre_ours(hand))
    _apply_canonical(root, cmap)
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
        spec["hand_default_joint_pos"] = {n: 0.0 for n in spec["hand_default_joint_pos"]}   # home is 0 now
        if "joint_limits" in spec:
            spec["joint_limits"] = {n: sorted([cmap[n]["sign"] * v - cmap[n]["offset"] for v in lim])
                                    if n in cmap else lim for n, lim in spec["joint_limits"].items()}
        spec.update({"urdf": str(dst.relative_to(REPO)), "stiffness": STIFFNESS, "damping": DAMPING,
                     "armature": ARMATURE, "effort_limit": EFFORT, "velocity_limit": VELOCITY,
                     "link_density_kg_m3": DENSITY, "friction": FRICTION})
        (out_dir / f"{hand}_left.spec.json").write_text(json.dumps(spec, indent=1))
    summary.update({"joints": n_joints, "finger_mass_kg": round(sum(summary["links"].values()), 4),
                    "canonical": cmap})
    return summary


if __name__ == "__main__":
    report = {}
    for hand, rel in HANDS.items():
        s = make(hand, rel)
        report[hand] = s
        ms = sorted(s["links"].values())
        print(f"{hand:8s} {s['joints']:2d} joints  finger links {len(ms):2d}  total {s['finger_mass_kg']:.3f} kg  "
              f"median {1000 * np.median(ms):5.1f} g  min {1000 * ms[0]:6.2f} g  palm '{s['palm']}' kept")
    (HERE / "masses.json").write_text(json.dumps({h: {k: v for k, v in r.items() if k != "canonical"}
                                                 for h, r in report.items()}, indent=1))
    (HERE / "canonical_map.json").write_text(json.dumps({h: r["canonical"] for h, r in report.items()}, indent=1))
    for h, r in report.items():
        c = r["canonical"]
        flips = [n for n, v in c.items() if v["sign"] < 0]
        zeros = {n: round(v["offset"], 3) for n, v in c.items() if v["offset"] != 0.0}
        kinds = sum(v["kind"] == "flex" for v in c.values())
        print(f"  {h:8s} {kinds} flex / {len(c) - kinds} spread; flipped {len(flips)}: {flips}; re-zeroed {zeros}")
