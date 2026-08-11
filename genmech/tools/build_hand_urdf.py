"""Turn a ``HandParams`` into a URDF: the procedural hand generator.

Companion to ``build_allegro_urdf.py``. That script splices one specific hand
onto the iiwa14; this one synthesises an arbitrary hand from a parameter vector,
which is what the co-design search needs (docs/proposal_codesign.md).

    .venv_isaacsim/bin/python -m genmech.tools.build_hand_urdf --params sharpa_like
    .venv_isaacsim/bin/python -m genmech.tools.build_hand_urdf --seed 3 --n_fingers 4
    .venv_isaacsim/bin/python -m genmech.tools.build_hand_urdf --population 12 --seed 0

**The arm is copied verbatim** from the SHARPA URDF, reusing
``build_allegro_urdf``'s own link and joint lists, so every robot in the registry
shares one arm and docs/methodology.md §1 keeps holding.

**Geometry is capsules, not meshes.** Arbitrary segment lengths cannot reuse
fixed-length meshes, and scaling a mesh non-uniformly distorts exactly the
knuckle geometry that matters for contact. Capsules are exact at any length,
are the cheapest shape PhysX has, and remove any mesh-fidelity difference
between two generated hands. URDF has no capsule primitive, so this emits
``<cylinder>`` and the spec sets ``replace_cylinders_with_capsules=True``.

**Ghosting keeps the articulation one shape.** Every hand emits 30 hand joints
whatever it looks like. A ghosted joint keeps its place in the chain with limits
locked to ``[0, 1e-8]``; a ghosted link carries no collision or visual geometry
and ``mass = 1e-6``. That is SHARPA's own convention for its zero-length virtual
links, and it is what lets one Isaac Lab ``Articulation`` view hold designs with
different finger counts -- see ``genmech/tools/probe_multi_articulation.py``.

Mass and inertia come from ``generate_objects._compute_mass_and_inertia``, the
same helper the object pipeline uses, with per-tier densities calibrated against
SHARPA's measured link masses in ``sharpa_anchors.py``. Nothing here invents a
mass.
"""

from __future__ import annotations

import argparse
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

from genmech.robots.generated import params as P
from genmech.robots.generated import sharpa_anchors as A
from genmech.tasks.pose_reach.utils.generate_objects import _compute_mass_and_inertia
from genmech.tools.build_allegro_urdf import (
    ARM_JOINTS,
    ARM_LINKS,
    ARM_MESH_PREFIX_TO,
    SHARPA_URDF,
    _rewrite_meshes,
)
from genmech.utils.paths import resolve as resolve_repo_path


OUT_DIR = "assets/urdf/generated"

PALM_LINK = "gen_palm"
FLANGE_LINK = "iiwa14_link_ee"

# A segment shorter than this gets no geometry: below roughly one radius a
# capsule stops being a model of a segment and becomes a sphere whose volume
# barely responds to length (see MC_MIN_LENGTH_M in sharpa_anchors).
MIN_SEGMENT_M = A.MC_MIN_LENGTH_M

GHOST_LIMIT = (0.0, 1e-8)


def _f(x: float) -> str:
    return f"{x:.9g}"


def _vec(v) -> str:
    return " ".join(_f(c) for c in v)


def _origin(parent: ET.Element, xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> None:
    ET.SubElement(parent, "origin", {"xyz": _vec(xyz), "rpy": _vec(rpy)})


def _add_inertial(link: ET.Element, mass: float, inertia: tuple[float, float, float],
                  xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> None:
    el = ET.SubElement(link, "inertial")
    _origin(el, xyz, rpy)
    ET.SubElement(el, "mass", {"value": _f(mass)})
    ixx, iyy, izz = inertia
    ET.SubElement(el, "inertia", {
        "ixx": _f(ixx), "iyy": _f(iyy), "izz": _f(izz),
        "ixy": "0", "ixz": "0", "iyz": "0",
    })


def _virtual_link(root: ET.Element, name: str) -> ET.Element:
    """A massless, geometry-free body.

    Used for the zero-length bodies between coincident joints, for segments
    below MIN_SEGMENT_M, and for every link of a ghosted finger. SHARPA already
    represents its own virtual links this way (mass 1e-6).
    """
    link = ET.SubElement(root, "link", {"name": name})
    _add_inertial(link, A.VIRTUAL_LINK_MASS_KG,
                  (A.VIRTUAL_LINK_INERTIA,) * 3)
    return link


def _capsule_link(root: ET.Element, name: str, *, length: float, radius: float,
                  density: float) -> ET.Element:
    """A capsule of ``length`` along the link's +x axis, starting at the origin.

    The +x convention follows SHARPA, whose phalanx joints translate along local
    x. URDF cylinders are z-aligned, so the geometry carries rpy=(0, pi/2, 0) and
    is shifted to its midpoint. The inertial origin takes the *same* rotation, so
    the tensor returned by ``_compute_mass_and_inertia`` (which is expressed
    about the capsule's own z axis) stays correct without re-deriving it -- the
    trick generate_objects.py already uses for its cylinder objects.
    """
    link = ET.SubElement(root, "link", {"name": name})
    mass, ixx, iyy, izz = _compute_mass_and_inertia((length, 2.0 * radius), density)

    geom_xyz = (length / 2.0, 0.0, 0.0)
    geom_rpy = (0.0, math.pi / 2.0, 0.0)

    for tag in ("visual", "collision"):
        el = ET.SubElement(link, tag)
        _origin(el, geom_xyz, geom_rpy)
        g = ET.SubElement(el, "geometry")
        ET.SubElement(g, "cylinder", {"length": _f(length), "radius": _f(radius)})

    _add_inertial(link, mass, (ixx, iyy, izz), geom_xyz, geom_rpy)
    return link


def _box_link(root: ET.Element, name: str, *, extents, center, density: float) -> ET.Element:
    link = ET.SubElement(root, "link", {"name": name})
    mass, ixx, iyy, izz = _compute_mass_and_inertia(tuple(extents), density)
    for tag in ("visual", "collision"):
        el = ET.SubElement(link, tag)
        _origin(el, center)
        g = ET.SubElement(el, "geometry")
        ET.SubElement(g, "box", {"size": _vec(extents)})
    _add_inertial(link, mass, (ixx, iyy, izz), center)
    return link


def _joint(root: ET.Element, name: str, *, parent: str, child: str,
           seg: P.Segment, limits: tuple[float, float], slot: str,
           ghost: bool) -> ET.Element:
    """One revolute joint. Axis is always [0 0 1], following SHARPA.

    A ghosted joint keeps its origin transform -- that rotation is part of the
    chain's geometry and must still apply -- but its travel is locked and its
    actuator is given effectively no authority.
    """
    j = ET.SubElement(root, "joint", {"name": name, "type": "revolute"})
    ET.SubElement(j, "parent", {"link": parent})
    ET.SubElement(j, "child", {"link": child})
    _origin(j, seg.xyz, seg.rpy)
    ET.SubElement(j, "axis", {"xyz": "0 0 1"})
    lo, hi = GHOST_LIMIT if ghost else limits
    effort = 1e-3 if ghost else A.SLOT_EFFORT_NM[slot]
    ET.SubElement(j, "limit", {
        "lower": _f(lo), "upper": _f(hi),
        "effort": _f(effort), "velocity": _f(A.SLOT_VELOCITY_RAD_S[slot]),
    })
    return j


def joint_name(finger_index: int, slot: str) -> str:
    return f"gen_f{finger_index}_{slot}"


def link_name(finger_index: int, part: str) -> str:
    return f"gen_f{finger_index}_{part}"


# Link parts in chain order, and which tier (if any) gives them geometry.
LINK_PARTS: tuple[tuple[str, str | None], ...] = (
    ("CMC_VL", None),   # between CMC_FE and CMC_AA
    ("MC", "mc"),       # metacarpal
    ("MCP_VL", None),   # between MCP_FE and MCP_AA
    ("PP", "pp"),
    ("MP", "mp"),
    ("DP", "dp"),
    ("tip", None),      # fingertip reference frame
)


def _build_finger(root: ET.Element, index: int, fp: P.FingerParams) -> None:
    parts = [link_name(index, p) for p, _ in LINK_PARTS]

    # --- links ---
    for (part, tier) in LINK_PARTS:
        name = link_name(index, part)
        if tier is None or not fp.active:
            _virtual_link(root, name)
            continue
        length = fp.segment_length(tier)
        if length < MIN_SEGMENT_M:
            _virtual_link(root, name)
            continue
        _capsule_link(
            root, name,
            length=length,
            radius=A.TIER_RADIUS_M[tier] * fp.radius_scale,
            density=A.TIER_DENSITY_KG_M3[tier],
        )

    # --- joints ---
    limits = dict(zip(P.JOINT_SLOTS, fp.limits))
    enabled = dict(zip(P.JOINT_SLOTS, fp.enabled))

    def ghost(slot: str) -> bool:
        return (not fp.active) or (not enabled[slot])

    chain: tuple[tuple[str, str, str, P.Segment], ...] = (
        # slot,      parent,        child,       origin
        ("CMC_FE", PALM_LINK, parts[0], fp.mount),
        ("CMC_AA", parts[0], parts[1], fp.cmc),
        ("MCP_FE", parts[1], parts[2], fp.mc),
        ("MCP_AA", parts[2], parts[3], fp.mcp),
        ("PIP", parts[3], parts[4],
         P.Segment(xyz=(fp.pp_length, 0.0, 0.0), rpy=P.ROLL_AA_TO_FE.rpy)),
        ("DIP", parts[4], parts[5],
         P.Segment(xyz=(fp.mp_length, 0.0, 0.0))),
    )
    for slot, parent, child, seg in chain:
        _joint(root, joint_name(index, slot), parent=parent, child=child,
               seg=seg, limits=limits[slot], slot=slot, ghost=ghost(slot))

    # Fingertip frame: fixed, so merge_fixed_joints folds it into DP and the
    # fingertip *body* the task tracks is gen_f{i}_DP.
    tip = ET.SubElement(root, "joint",
                        {"name": f"gen_f{index}_tip_fix", "type": "fixed"})
    ET.SubElement(tip, "parent", {"link": parts[5]})
    ET.SubElement(tip, "child", {"link": parts[6]})
    _origin(tip, (fp.dp_length if fp.active else 0.0, 0.0, 0.0))


def build_urdf(hand: P.HandParams, *, mount_yaw: float | None = None) -> ET.Element:
    """Assemble the complete robot: iiwa14 arm + generated hand."""
    yaw = A.FLANGE_TO_PALM_YAW_RAD if mount_yaw is None else mount_yaw
    sharpa = ET.parse(resolve_repo_path(SHARPA_URDF)).getroot()

    root = ET.Element("robot", {"name": f"iiwa14_{hand.name}"})
    root.append(ET.Comment(
        f" GENERATED by genmech/tools/build_hand_urdf.py; do not hand-edit.\n"
        f"     params: {hand.name}\n"
        f"     arm:    iiwa14 chain copied verbatim from the SHARPA URDF, so the\n"
        f"             arm is identical across every robot (docs/methodology.md 1).\n"
        f"     hand:   {hand.n_active_fingers} active finger(s), "
        f"{hand.n_active_joints} active joint(s) of "
        f"{P.N_FINGER_SLOTS * P.N_JOINT_SLOTS} emitted;\n"
        f"             the rest are ghosted so the articulation shape is fixed.\n"
        f"     mount:  {FLANGE_LINK} -> {PALM_LINK} at z={A.FLANGE_TO_PALM_Z_M},\n"
        f"             yaw={yaw:.6f} rad ({math.degrees(yaw):.1f} deg), which is\n"
        f"             SHARPA's composed flange-to-palm transform.\n"
        f"     geometry: capsules; densities calibrated against SHARPA's measured\n"
        f"             link masses (genmech/robots/generated/sharpa_anchors.py).\n"
    ))

    for mat in sharpa.findall("material"):
        root.append(mat)

    # --- arm, verbatim ---
    for link in sharpa.findall("link"):
        if link.get("name") in ARM_LINKS:
            _rewrite_meshes(link, "", ARM_MESH_PREFIX_TO)
            root.append(link)
    for joint in sharpa.findall("joint"):
        if joint.get("name") in ARM_JOINTS:
            root.append(joint)

    # --- palm ---
    _box_link(root, PALM_LINK, extents=hand.palm_extents,
              center=A.PALM_BOX_CENTER_M, density=A.PALM_DENSITY_KG_M3)
    mount = ET.SubElement(root, "joint",
                          {"name": "iiwa14_gen_palm", "type": "fixed"})
    ET.SubElement(mount, "parent", {"link": FLANGE_LINK})
    ET.SubElement(mount, "child", {"link": PALM_LINK})
    _origin(mount, (0.0, 0.0, A.FLANGE_TO_PALM_Z_M), (0.0, 0.0, yaw))

    # --- fingers ---
    for i, fp in enumerate(hand.fingers):
        _build_finger(root, i, fp)

    return root


def write_urdf(hand: P.HandParams, out_path: Path, *,
               mount_yaw: float | None = None) -> Path:
    root = build_urdf(hand, mount_yaw=mount_yaw)
    xml = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(
        indent="  ", encoding="utf-8").decode("utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml)
    return out_path


def urdf_path_for(hand: P.HandParams) -> Path:
    return resolve_repo_path(OUT_DIR) / f"{hand.name}.urdf"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default=None,
                        help="'sharpa_like' for the reference vector")
    parser.add_argument("--seed", type=int, default=None,
                        help="sample a hand from the design space")
    parser.add_argument("--n_fingers", type=int, default=None)
    parser.add_argument("--population", type=int, default=None,
                        help="write N sampled hands (requires --seed)")
    parser.add_argument("--out", default=None, help="output path (single hand)")
    parser.add_argument("--mount_yaw", type=float, default=None)
    args = parser.parse_args()

    if args.population is not None:
        if args.seed is None:
            raise SystemExit("--population requires --seed")
        hands = P.sample_population(args.seed, args.population,
                                    n_fingers=args.n_fingers)
    elif args.seed is not None:
        rng = random.Random(args.seed)
        hands = [P.sample_valid(rng, name=f"gen_{args.seed:04d}_000",
                                n_fingers=args.n_fingers)]
    else:
        if args.params not in (None, "sharpa_like"):
            raise SystemExit(f"unknown --params {args.params!r}")
        hands = [P.SHARPA_LIKE]

    for hand in hands:
        out = Path(args.out) if (args.out and len(hands) == 1) else urdf_path_for(hand)
        write_urdf(hand, out, mount_yaw=args.mount_yaw)
        print(f"[build_hand_urdf] {hand.name}: "
              f"{hand.n_active_fingers} fingers, "
              f"{hand.n_active_joints}/{P.N_FINGER_SLOTS * P.N_JOINT_SLOTS} "
              f"active joints -> {out}")


if __name__ == "__main__":
    main()
