"""Author a generated hand's USD directly, instead of converting a URDF.

Kit's UrdfConverter costs ~0.9 s per hand, ~90% of it re-importing the arm's
16 STL meshes. The hand is capsules and a box, so it is authored with Sdf
specs (~8 ms) and the arm comes in by reference from one converted USD.

The target is the converter's POST-MERGE robot: ``merge_fixed_joints`` folds
``gen_palm`` into ``iiwa14_link_7`` and each fingertip into its distal
phalanx, so the authored asset is

    /<root>                                 Xform
      /arm                                  reference: iiwa14_link_0..7 + joints
      /gen_f{i}_{CMC_VL,MC,MCP_VL,PP,MP,DP}  30 Xform bodies
        /collisions/mesh_0/capsule          Capsule, axis Z
      /joints/gen_f{i}_{slot}               30 PhysicsRevoluteJoint
      /root_joint                           PhysicsFixedJoint + ArticulationRootAPI

Conventions, read off a converted asset: a capsule is ``radius=r,
height=length-2r, axis=Z`` under an Xform at ``(length/2, 0, 0)`` rotated 90
deg about Y; joint limits and velocities are in DEGREES; ``localPos0/Rot0``
is the joint origin in the parent frame, ``localPos1/Rot1`` identity; inertia
is ``diagonalInertia`` plus a ``principalAxes`` quaternion; a virtual link has
mass and inertia 1e-6.

Sdf trap: ``Sdf.CreatePrimInLayer`` creates OVERs, for the prim and for any
missing ancestor. Without ``specifier = SpecifierDef`` the prim composes to
nothing and authoring silently succeeds against an empty stage.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

from hand_sampler import params as P
from hand_sampler import sharpa_anchors as A
from hand_sampler.allegro_urdf import SHARPA_URDF
from hand_sampler.inertia import compute_mass_and_inertia
from hand_sampler.paths import resolve as resolve_repo_path
from hand_sampler.rotations import mat_to_pos_quat, rpy_to_mat, rpy_to_quat_wxyz
from hand_sampler.urdf import has_collision_geometry

PALM_BODY = "iiwa14_link_7"  # post-merge palm body
# Chain order and the tier that gives each link its geometry
# (build_hand_urdf.LINK_PARTS minus `tip`, which merges into DP).
LINK_PARTS: tuple[tuple[str, str | None], ...] = (
    ("CMC_VL", None), ("MC", "mc"), ("MCP_VL", None),
    ("PP", "pp"), ("MP", "mp"), ("DP", "dp"),
)
# link_7 -> link_ee (iiwa14_joint_ee); merge_fixed_joints folds it into the palm mount.
LINK7_TO_FLANGE_Z_M: float = 0.045
# What the converter writes for drives whatever gains are asked for. Isaac Lab
# overwrites them at runtime; matching keeps authored and converted assets comparable.
CONVERTER_DRIVE_STIFFNESS: float = 625.0
CONVERTER_DRIVE_DAMPING: float = 0.0


# --- Sdf helpers -----------------------------------------------------------------

def define(layer, path: str, type_name: str, apis: list[str] | None = None):
    """A DEFINING prim spec (see the module docstring)."""
    from pxr import Sdf

    spec = Sdf.CreatePrimInLayer(layer, Sdf.Path(path))
    spec.specifier = Sdf.SpecifierDef
    spec.typeName = type_name
    if apis:
        spec.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit(apis))
    return spec


def attr(spec, name: str, type_name, value):
    from pxr import Sdf

    a = Sdf.AttributeSpec(spec, name, type_name)
    a.default = value
    return a


def rel(spec, name: str, target: str):
    """A relationship (physics:body0/1 are relationships, not info keys)."""
    from pxr import Sdf

    r = Sdf.RelationshipSpec(spec, name, False)
    r.targetPathList.explicitItems.append(Sdf.Path(target))
    return r


def _set_xform(spec, xyz, quat_wxyz=None, scale=(1.0, 1.0, 1.0)):
    """translate + orient + scale, in the converter's op order."""
    from pxr import Gf, Sdf

    attr(spec, "xformOp:translate", Sdf.ValueTypeNames.Double3,
         Gf.Vec3d(*[float(v) for v in xyz]))
    ops = ["xformOp:translate"]
    if quat_wxyz is not None:
        w, x, y, z = (float(v) for v in quat_wxyz)
        attr(spec, "xformOp:orient", Sdf.ValueTypeNames.Quatd, Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        ops.append("xformOp:orient")
    attr(spec, "xformOp:scale", Sdf.ValueTypeNames.Double3,
         Gf.Vec3d(*[float(v) for v in scale]))
    ops.append("xformOp:scale")
    attr(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray, ops)


# --- geometry and mass, shared with the URDF builder ---------------------------

def _link_shape(fp: P.FingerParams, tier: str):
    """``(length, radius, density)``, or None if this link carries no shape."""
    if not has_collision_geometry(fp, tier):
        return None
    return (fp.segment_length(tier), A.TIER_RADIUS_M[tier] * fp.radius_scale,
            A.TIER_DENSITY_KG_M3[tier])


def _link_mass_props(fp: P.FingerParams, tier: str | None):
    """``(mass, (ixx,iyy,izz), com, principal_axes)`` for one link, as build_hand_urdf has it."""
    virtual = (A.VIRTUAL_LINK_MASS_KG, (A.VIRTUAL_LINK_INERTIA,) * 3,
               (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    if tier is None or not fp.active or fp.segment_length(tier) < A.MC_MIN_LENGTH_M:
        props = virtual
    else:
        length = fp.segment_length(tier)
        radius = A.TIER_RADIUS_M[tier] * fp.radius_scale
        mass, ixx, iyy, izz = compute_mass_and_inertia(
            (A.cylinder_part(length, radius), 2.0 * radius), A.TIER_DENSITY_KG_M3[tier])
        # Inertial frame at the capsule midpoint, rotated 90 deg about Y.
        props = (mass, (ixx, iyy, izz), (length / 2.0, 0.0, 0.0),
                 rpy_to_quat_wxyz((0.0, math.pi / 2.0, 0.0)))
    return _merge_tip(fp, props) if tier == "dp" else props


def _merge_tip(fp: P.FingerParams, props):
    """Fold the fixed-jointed `tip` link into DP, as merge_fixed_joints does.

    The tip sits on the capsule axis at ``(dp_length, 0, 0)``, so the shift
    adds ``m*d^2`` to the two perpendicular moments and nothing to the axial one.
    """
    mass, (ixx, iyy, izz), com, axes = props
    m_tip, i_tip = A.VIRTUAL_LINK_MASS_KG, A.VIRTUAL_LINK_INERTIA
    x_tip = float(fp.dp_length) if fp.active else 0.0  # a ghosted tip sits at the origin
    x_dp = float(com[0])
    total = mass + m_tip
    x_com = (mass * x_dp + m_tip * x_tip) / total
    d_dp, d_tip = x_dp - x_com, x_tip - x_com
    return (total,
            (ixx + mass * d_dp ** 2 + i_tip + m_tip * d_tip ** 2,
             iyy + mass * d_dp ** 2 + i_tip + m_tip * d_tip ** 2,
             izz + i_tip),
            (x_com, 0.0, 0.0), axes)


def _mat_from_seg(seg: P.Segment) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rpy_to_mat(seg.rpy)
    m[:3, 3] = [float(v) for v in seg.xyz]
    return m


def finger_chain(fp: P.FingerParams):
    """``[(slot, part, parent_part_or_None, Segment), ...]``, as build_hand_urdf chains them."""
    return [
        ("CMC_FE", "CMC_VL", None, fp.mount),
        ("CMC_AA", "MC", "CMC_VL", fp.cmc),
        ("MCP_FE", "MCP_VL", "MC", fp.mc),
        ("MCP_AA", "PP", "MCP_VL", fp.mcp),
        ("PIP", "MP", "PP", P.Segment(xyz=(fp.pp_length, 0.0, 0.0), rpy=P.ROLL_AA_TO_FE.rpy)),
        ("DIP", "DP", "MP", P.Segment(xyz=(fp.mp_length, 0.0, 0.0))),
    ]


def flange_to_palm() -> P.Segment:
    """link_7 -> gen_palm, the transform merge_fixed_joints collapses."""
    return P.Segment(xyz=(0.0, 0.0, LINK7_TO_FLANGE_Z_M + A.FLANGE_TO_PALM_Z_M),
                     rpy=(0.0, 0.0, A.FLANGE_TO_PALM_YAW_RAD))


# --- the merged palm body ---------------------------------------------------------

_LINK7_INERTIAL: tuple | None = None


def arm_link7_inertial() -> tuple[float, tuple, tuple]:
    """``(mass, com, (ixx,iyy,izz))`` of iiwa14_link_7 alone, parsed once from the SHARPA URDF."""
    global _LINK7_INERTIAL
    if _LINK7_INERTIAL is None:
        root = ET.parse(resolve_repo_path(SHARPA_URDF)).getroot()
        link = next((l for l in root.findall("link") if l.get("name") == PALM_BODY), None)
        if link is None:
            raise RuntimeError(f"{PALM_BODY} not found in {SHARPA_URDF}")
        inertial = link.find("inertial")
        origin = inertial.find("origin")
        com = (tuple(float(v) for v in origin.get("xyz").split())
               if origin is not None else (0.0, 0.0, 0.0))
        it = inertial.find("inertia")
        _LINK7_INERTIAL = (float(inertial.find("mass").get("value")), com,
                           (float(it.get("ixx")), float(it.get("iyy")), float(it.get("izz"))))
    return _LINK7_INERTIAL


def merged_palm_body_props(hand: P.HandParams):
    """``(mass, (ixx,iyy,izz), com, principal_axes)`` of iiwa14_link_7 with the
    palm box merged in, as the converter has it. The palm is 37% of the merged
    mass and yawed 75 deg, so the summed tensor is off-diagonal and is diagonalised."""
    m_arm, com_arm, diag_arm = arm_link7_inertial()
    com_arm = np.asarray(com_arm, dtype=float)
    ex, ey, ez = (float(v) for v in hand.palm_extents)
    m_palm = A.PALM_DENSITY_KG_M3 * ex * ey * ez
    i_palm_local = np.diag([m_palm * (ey * ey + ez * ez) / 12.0,
                            m_palm * (ex * ex + ez * ez) / 12.0,
                            m_palm * (ex * ex + ey * ey) / 12.0])
    t_palm = _mat_from_seg(flange_to_palm())  # link_7 <- palm
    r_palm = t_palm[:3, :3]
    com_palm = (t_palm @ np.append(
        np.asarray([float(v) for v in P.palm_center(hand.palm_extents)]), 1.0))[:3]
    i_palm = r_palm @ i_palm_local @ r_palm.T
    total = m_arm + m_palm
    com = (m_arm * com_arm + m_palm * com_palm) / total

    def shift(inertia, mass, centre):
        d = centre - com
        return inertia + mass * (float(d @ d) * np.eye(3) - np.outer(d, d))

    i_total = shift(np.diag(diag_arm), m_arm, com_arm) + shift(i_palm, m_palm, com_palm)
    vals, vecs = np.linalg.eigh(i_total)
    if np.linalg.det(vecs) < 0:  # keep it a rotation
        vecs[:, 0] = -vecs[:, 0]
    axes = mat_to_pos_quat(np.block([[vecs, np.zeros((3, 1))], [np.zeros((1, 3)), 1.0]]))[1]
    return float(total), tuple(float(v) for v in vals), tuple(float(v) for v in com), axes


# --- authoring ---------------------------------------------------------------------

def author_hand(layer, root_path: str, hand: P.HandParams, spec, *,
                palm_body_path: str | None = None, link7_world=None) -> dict:
    """Author one hand's bodies, colliders and joints under ``root_path``.

    ``palm_body_path`` is the merged palm body every finger's first joint
    attaches to; ``link7_world`` places the link prims (the joints define the
    kinematics). Returns ``{"bodies", "capsules", "joints", "collider_links"}``.
    """
    from pxr import Gf, Sdf

    palm_path = palm_body_path or f"{root_path}/{PALM_BODY}"
    world0 = np.eye(4) if link7_world is None else np.asarray(link7_world, float)
    palm_mat = world0 @ _mat_from_seg(flange_to_palm())

    limits_of = {}
    for i, fp in enumerate(hand.fingers):
        enabled = dict(zip(P.JOINT_SLOTS, fp.enabled))
        for slot, (lo, hi) in zip(P.JOINT_SLOTS, fp.limits):
            ghost = not fp.active or not enabled[slot]
            limits_of[(i, slot)] = (0.0, 1e-8) if ghost else (float(lo), float(hi))

    n_bodies = n_caps = 0
    collider_links: dict[str, int] = {}
    define(layer, f"{root_path}/joints", "Scope")

    for i, fp in enumerate(hand.fingers):
        acc = palm_mat.copy()
        part_world: dict[str, np.ndarray] = {}
        for _slot, part, _parent, seg in finger_chain(fp):
            acc = acc @ _mat_from_seg(seg)
            part_world[part] = acc.copy()

        for part, tier in LINK_PARTS:
            name = f"gen_f{i}_{part}"
            body = define(layer, f"{root_path}/{name}", "Xform",
                          ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
            mass, inertia, com, axes = _link_mass_props(fp, tier)
            attr(body, "physics:mass", Sdf.ValueTypeNames.Float, float(mass))
            attr(body, "physics:diagonalInertia", Sdf.ValueTypeNames.Float3,
                 Gf.Vec3f(*[float(v) for v in inertia]))
            attr(body, "physics:centerOfMass", Sdf.ValueTypeNames.Float3,
                 Gf.Vec3f(*[float(v) for v in com]))
            attr(body, "physics:principalAxes", Sdf.ValueTypeNames.Quatf,
                 Gf.Quatf(float(axes[0]), Gf.Vec3f(*[float(v) for v in axes[1:]])))
            pos, quat = mat_to_pos_quat(part_world[part])
            _set_xform(body, pos, quat)
            n_bodies += 1

            shape = _link_shape(fp, tier) if (tier and fp.active) else None
            if shape is None:
                continue
            length, radius, _ = shape
            define(layer, f"{root_path}/{name}/collisions", "Xform")
            mesh = define(layer, f"{root_path}/{name}/collisions/mesh_0", "Xform")
            _set_xform(mesh, (length / 2.0, 0.0, 0.0), rpy_to_quat_wxyz((0.0, math.pi / 2.0, 0.0)))
            cap = define(layer, f"{root_path}/{name}/collisions/mesh_0/capsule",
                         "Capsule", ["PhysicsCollisionAPI"])
            attr(cap, "radius", Sdf.ValueTypeNames.Double, float(radius))
            attr(cap, "height", Sdf.ValueTypeNames.Double, float(A.cylinder_part(length, radius)))
            attr(cap, "axis", Sdf.ValueTypeNames.Token, "Z")
            n_caps += 1
            # Recorded where the collider is created; the friction pass needs each
            # link's shape count, which depends on segment length, not just fp.active.
            collider_links[name] = collider_links.get(name, 0) + 1

        for slot, part, parent_part, seg in finger_chain(fp):
            # PhysxJointAPI is required for physxJoint:* attributes to be read at all.
            j = define(layer, f"{root_path}/joints/gen_f{i}_{slot}", "PhysicsRevoluteJoint",
                       ["PhysicsDriveAPI:angular", "PhysxJointAPI"])
            rel(j, "physics:body0",
                palm_path if parent_part is None else f"{root_path}/gen_f{i}_{parent_part}")
            rel(j, "physics:body1", f"{root_path}/gen_f{i}_{part}")
            # The first joint hangs off the MERGED palm, so its origin includes flange->palm.
            origin = (_mat_from_seg(flange_to_palm()) @ _mat_from_seg(seg)
                      if parent_part is None else _mat_from_seg(seg))
            pos, quat = mat_to_pos_quat(origin)
            attr(j, "physics:localPos0", Sdf.ValueTypeNames.Point3f,
                 Gf.Vec3f(*[float(v) for v in pos]))
            attr(j, "physics:localRot0", Sdf.ValueTypeNames.Quatf,
                 Gf.Quatf(float(quat[0]), Gf.Vec3f(*[float(v) for v in quat[1:]])))
            attr(j, "physics:localPos1", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(0.0, 0.0, 0.0))
            attr(j, "physics:localRot1", Sdf.ValueTypeNames.Quatf,
                 Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            attr(j, "physics:axis", Sdf.ValueTypeNames.Token, "Z")
            lo, hi = limits_of[(i, slot)]
            attr(j, "physics:lowerLimit", Sdf.ValueTypeNames.Float, float(math.degrees(lo)))
            attr(j, "physics:upperLimit", Sdf.ValueTypeNames.Float, float(math.degrees(hi)))
            attr(j, "physics:jointEnabled", Sdf.ValueTypeNames.Bool, True)
            attr(j, "physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool, False)
            attr(j, "drive:angular:physics:stiffness", Sdf.ValueTypeNames.Float,
                 CONVERTER_DRIVE_STIFFNESS)
            attr(j, "drive:angular:physics:damping", Sdf.ValueTypeNames.Float,
                 CONVERTER_DRIVE_DAMPING)
            # Full effort for ghosts too: the drive is what holds a locked joint shut.
            attr(j, "drive:angular:physics:maxForce", Sdf.ValueTypeNames.Float,
                 float(A.SLOT_EFFORT_NM[slot]))
            attr(j, "drive:angular:physics:targetPosition", Sdf.ValueTypeNames.Float, 0.0)
            # Unset, this is no limit at all (5.9e36), and the reaction torque moves the arm.
            attr(j, "physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float,
                 float(math.degrees(A.SLOT_VELOCITY_RAD_S[slot])))

    return {"bodies": n_bodies, "capsules": n_caps,
            "joints": len(hand.fingers) * P.N_JOINT_SLOTS,
            "collider_links": dict(collider_links)}


__all__ = ["author_hand", "define", "attr", "rel", "finger_chain",
           "flange_to_palm", "PALM_BODY", "LINK_PARTS"]
