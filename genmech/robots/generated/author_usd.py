"""Author a generated hand's USD directly, instead of converting a URDF.

Measured: Kit's UrdfConverter takes ~876 ms per generated hand, of which ~90% is
importing the arm's 16 STL meshes -- the SAME arm in every design. Authoring the
prims with Sdf specs takes ~8 ms per robot and scales linearly (validated to
24,576 robots at 8.42 ms each). The hand itself is nothing but capsules and a
box; there is no mesh to import.

So the split here is:

  ARM   comes in BY REFERENCE from one converted USD, built once from an
        arm-only URDF. It has meshes, it is identical across every design, and
        re-importing it per design is the entire cost being removed.

  HAND  is authored fresh from HandParams: links, capsule geometry, revolute
        joints with limits and drives, masses and inertias.

THE TARGET IS THE CONVERTER'S POST-MERGE ROBOT, NOT THE URDF.
``merge_fixed_joints=True`` folds ``gen_palm`` into ``iiwa14_link_7`` and each
fingertip frame into its distal phalanx, so the authored asset must describe:

    /<root>                            Xform
      /arm                             (reference; iiwa14_link_0..7 + 7 joints)
      /gen_f{i}_{CMC_VL,MC,MCP_VL,PP,MP,DP}   30 Xform bodies
        /collisions/mesh_0/capsule     Capsule, axis Z
      /joints/gen_f{i}_{slot}          30 PhysicsRevoluteJoint
      /root_joint                      PhysicsFixedJoint + ArticulationRootAPI

Every convention below was read off a converted asset with
``genmech.tools.probe_hand_usd`` rather than assumed, because the object
authoring lost hours to exactly that: every attribute matched an ASSUMED layout
while the real one differed.

  * a capsule is ``radius=r, height=length-2r, axis=Z`` under an Xform at
    ``(length/2, 0, 0)`` rotated 90 deg about Y -- the URDF emits a cylinder of
    ``cylinder_part`` and the importer caps it, so the capsule spans exactly
    ``length``;
  * joint limits are in DEGREES;
  * ``physics:localPos0/localRot0`` are the joint origin in the PARENT body's
    frame, ``localPos1/localRot1`` are identity;
  * inertia is ``diagonalInertia`` plus a ``principalAxes`` quaternion, which
    for a +x capsule is the 90-deg-about-Y rotation (0.7071, 0, 0.7071, 0);
  * a virtual link is mass 1e-6, inertia 1e-6, identity principal axes.

Two Sdf traps, both hit while getting this working, both silent:

  * ``Sdf.CreatePrimInLayer`` creates an OVER spec. Without
    ``specifier = Sdf.SpecifierDef`` the prim composes to nothing, and authoring
    "succeeds" against a stage that stays empty.
  * Missing ANCESTORS are created as overs too, so a defined prim under an
    undefined ancestor also never appears.

This module is not trusted until ``genmech.tools.compare_authored_hand`` reports
agreement with the converter on masses, inertias, joint limits, drive gains and
composed collider geometry. Authoring something subtly different and not
noticing is the failure mode that matters, not authoring something slow.
"""

from __future__ import annotations

import math

from genmech.robots.generated import params as P
from genmech.robots.generated import sharpa_anchors as A

# Post-merge palm body: gen_palm is fixed-jointed to the flange, so the importer
# folds it into the arm's last link.
PALM_BODY = "iiwa14_link_7"

# Chain order and which tier gives each link its geometry. Mirrors
# build_hand_urdf.LINK_PARTS minus `tip`, which merges into DP.
LINK_PARTS: tuple[tuple[str, str | None], ...] = (
    ("CMC_VL", None), ("MC", "mc"), ("MCP_VL", None),
    ("PP", "pp"), ("MP", "mp"), ("DP", "dp"),
)


# ---------------------------------------------------------------------------
# Sdf helpers
# ---------------------------------------------------------------------------

def define(layer, path: str, type_name: str, apis: list[str] | None = None):
    """A DEFINING prim spec. See the module docstring on why this matters."""
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
    """physics:body0/1 are RELATIONSHIPS; SetInfo rejects them as info keys."""
    from pxr import Sdf

    r = Sdf.RelationshipSpec(spec, name, False)
    r.targetPathList.explicitItems.append(Sdf.Path(target))
    return r


def _quat_from_rpy(rpy) -> tuple[float, float, float, float]:
    """URDF RPY -> (w, x, y, z). Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = (float(v) / 2.0 for v in rpy)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return (cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)


def _set_xform(spec, xyz, quat_wxyz=None, scale=(1.0, 1.0, 1.0)):
    """translate + orient + scale, in the converter's op order."""
    from pxr import Gf, Sdf

    attr(spec, "xformOp:translate", Sdf.ValueTypeNames.Double3,
         Gf.Vec3d(*[float(v) for v in xyz]))
    ops = ["xformOp:translate"]
    if quat_wxyz is not None:
        w, x, y, z = (float(v) for v in quat_wxyz)
        attr(spec, "xformOp:orient", Sdf.ValueTypeNames.Quatd,
             Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        ops.append("xformOp:orient")
    attr(spec, "xformOp:scale", Sdf.ValueTypeNames.Double3,
         Gf.Vec3d(*[float(v) for v in scale]))
    ops.append("xformOp:scale")
    attr(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray, ops)


# ---------------------------------------------------------------------------
# geometry and mass, shared with the URDF builder so the two cannot drift
# ---------------------------------------------------------------------------

def _link_shape(fp: P.FingerParams, tier: str):
    """``(length, radius, density)`` or None if this link carries no shape."""
    from genmech.tools.build_hand_urdf import has_collision_geometry

    if not has_collision_geometry(fp, tier):
        return None
    return (fp.segment_length(tier),
            A.TIER_RADIUS_M[tier] * fp.radius_scale,
            A.TIER_DENSITY_KG_M3[tier])


def _link_mass_props(fp: P.FingerParams, tier: str | None):
    """``(mass, (ixx,iyy,izz), com, principal_axes)`` for one link.

    Mass comes from the same ``_compute_mass_and_inertia`` the URDF builder uses
    -- one mass model, not two. A link below MIN_SEGMENT_M, or shorter than its
    own diameter, is handled by the callers exactly as build_hand_urdf does:
    the first is virtual, the second keeps its mass but drops its geometry.
    """
    from genmech.tasks.pose_reach.utils.generate_objects import (
        _compute_mass_and_inertia,
    )

    virtual = (A.VIRTUAL_LINK_MASS_KG,
               (A.VIRTUAL_LINK_INERTIA,) * 3,
               (0.0, 0.0, 0.0),
               (1.0, 0.0, 0.0, 0.0))
    if tier is None or not fp.active:
        return _merge_tip(fp, virtual) if tier == "dp" else virtual
    length = fp.segment_length(tier)
    if length < A.MC_MIN_LENGTH_M:
        return _merge_tip(fp, virtual) if tier == "dp" else virtual
    radius = A.TIER_RADIUS_M[tier] * fp.radius_scale
    density = A.TIER_DENSITY_KG_M3[tier]
    mass, ixx, iyy, izz = _compute_mass_and_inertia(
        (A.cylinder_part(length, radius), 2.0 * radius), density)
    # The URDF puts the inertial frame at the capsule midpoint, rotated 90 deg
    # about Y so the tensor (expressed about the capsule's own z axis) stays
    # correct. The converter turns that rpy into principalAxes.
    props = (mass, (ixx, iyy, izz), (length / 2.0, 0.0, 0.0),
             _quat_from_rpy((0.0, math.pi / 2.0, 0.0)))
    return _merge_tip(fp, props) if tier == "dp" else props


def _merge_tip(fp: P.FingerParams, props):
    """Fold the fixed-jointed `tip` link into DP, as merge_fixed_joints does.

    The fingertip frame is a virtual link fixed to DP at ``(dp_length, 0, 0)``.
    The importer collapses it, so the converted DP carries the tip's mass and
    inertia too -- and omitting that made every authored DP's inertia low by
    exactly the tip's 1e-6, which the comparison against the converter caught.

    Everything lies on the capsule's own axis, so the principal frame does not
    rotate: a shift along that axis adds ``m*d^2`` to the two perpendicular
    moments and nothing to the axial one.
    """
    mass, (ixx, iyy, izz), com, axes = props
    m_tip = A.VIRTUAL_LINK_MASS_KG
    i_tip = A.VIRTUAL_LINK_INERTIA
    # A ghosted finger's tip sits at the origin (build_hand_urdf emits 0.0).
    x_tip = float(fp.dp_length) if fp.active else 0.0
    x_dp = float(com[0])

    total = mass + m_tip
    x_com = (mass * x_dp + m_tip * x_tip) / total
    d_dp, d_tip = x_dp - x_com, x_tip - x_com
    return (total,
            (ixx + mass * d_dp ** 2 + i_tip + m_tip * d_tip ** 2,
             iyy + mass * d_dp ** 2 + i_tip + m_tip * d_tip ** 2,
             izz + i_tip),
            (x_com, 0.0, 0.0),
            axes)


def _mat_from_seg(seg: P.Segment):
    import numpy as np

    r, p, y = (float(v) for v in seg.rpy)
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    m = np.eye(4)
    m[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    m[:3, 3] = [float(v) for v in seg.xyz]
    return m


def _mat_to_pos_quat(m):
    """4x4 -> (translation, (w,x,y,z))."""
    import numpy as np

    r = np.asarray(m[:3, :3], dtype=float)
    tr = float(np.trace(r))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w, x = 0.25 * s, (r[2, 1] - r[1, 2]) / s
        y, z = (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w, x = (r[2, 1] - r[1, 2]) / s, 0.25 * s
        y, z = (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w, x = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s
        y, z = 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w, x = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s
        y, z = (r[1, 2] + r[2, 1]) / s, 0.25 * s
    return tuple(float(v) for v in m[:3, 3]), (w, x, y, z)


def finger_chain(fp: P.FingerParams):
    """``[(slot, part, parent_part_or_None, Segment), ...]`` in chain order.

    Mirrors ``build_hand_urdf._build_finger``'s chain exactly. ``parent_part``
    is None for the first joint, whose parent is the merged palm body.
    """
    return [
        ("CMC_FE", "CMC_VL", None, fp.mount),
        ("CMC_AA", "MC", "CMC_VL", fp.cmc),
        ("MCP_FE", "MCP_VL", "MC", fp.mc),
        ("MCP_AA", "PP", "MCP_VL", fp.mcp),
        ("PIP", "MP", "PP",
         P.Segment(xyz=(fp.pp_length, 0.0, 0.0), rpy=P.ROLL_AA_TO_FE.rpy)),
        ("DIP", "DP", "MP", P.Segment(xyz=(fp.mp_length, 0.0, 0.0))),
    ]


# link_7 -> link_ee, from the SHARPA URDF's `iiwa14_joint_ee`
# (xyz="0 0 0.045", rpy="0 0 0"). The URDF mounts the palm on link_ee, but
# merge_fixed_joints collapses BOTH fixed joints, so the palm's transform
# relative to the surviving body (link_7) is this composed with the mount.
# Omitting it put every finger 45 mm too low -- caught by comparing localPos0
# against the converter (0.1162 vs 0.0712).
LINK7_TO_FLANGE_Z_M: float = 0.045

# What Kit's UrdfConverter writes for drive stiffness, regardless of the
# JointDriveCfg gains we pass (we ask for 0/0 and get this). The value is
# functionally irrelevant -- Isaac Lab overwrites gains at runtime from the
# spec's tables via ImplicitActuatorCfg, and the DriveAPI prim only has to
# EXIST for those to land -- but matching it keeps the authored asset
# byte-comparable to a converted one.
CONVERTER_DRIVE_STIFFNESS: float = 625.0
CONVERTER_DRIVE_DAMPING: float = 0.0


def flange_to_palm() -> P.Segment:
    """link_7 -> gen_palm, the transform merge_fixed_joints collapses."""
    return P.Segment(
        xyz=(0.0, 0.0, LINK7_TO_FLANGE_Z_M + A.FLANGE_TO_PALM_Z_M),
        rpy=(0.0, 0.0, A.FLANGE_TO_PALM_YAW_RAD))


# ---------------------------------------------------------------------------
# the merged palm body
# ---------------------------------------------------------------------------

_LINK7_INERTIAL: tuple | None = None


def arm_link7_inertial() -> tuple[float, tuple, tuple]:
    """``(mass, com, (ixx,iyy,izz))`` for iiwa14_link_7 ALONE, from the URDF.

    Parsed rather than hardcoded so it cannot drift from the arm the rest of the
    stack loads. The arm is identical across every design, so this is read once.
    """
    global _LINK7_INERTIAL
    if _LINK7_INERTIAL is not None:
        return _LINK7_INERTIAL
    import xml.etree.ElementTree as ET

    from genmech.tools.build_allegro_urdf import SHARPA_URDF
    from genmech.utils.paths import resolve as resolve_repo_path

    root = ET.parse(resolve_repo_path(SHARPA_URDF)).getroot()
    for link in root.findall("link"):
        if link.get("name") != PALM_BODY:
            continue
        inertial = link.find("inertial")
        origin = inertial.find("origin")
        com = tuple(float(v) for v in origin.get("xyz").split()) if origin is not None \
            else (0.0, 0.0, 0.0)
        it = inertial.find("inertia")
        _LINK7_INERTIAL = (
            float(inertial.find("mass").get("value")),
            com,
            (float(it.get("ixx")), float(it.get("iyy")), float(it.get("izz"))),
        )
        return _LINK7_INERTIAL
    raise RuntimeError(f"{PALM_BODY} not found in {SHARPA_URDF}")


def merged_palm_body_props(hand: P.HandParams):
    """Mass properties of iiwa14_link_7 AFTER the palm is merged into it.

    merge_fixed_joints collapses gen_palm (and the massless link_ee) into the
    arm's last link, so the converted robot has no gen_palm body at all -- its
    720 g box is part of link_7. Authoring link_7 with the ARM's inertial alone
    would lose that, and the error is not small: the palm is 37% of the merged
    mass and sits 95 mm off the link_7 origin.

    Both tensors are moved to the merged centre of mass by the parallel-axis
    theorem and summed in the link_7 frame, then diagonalised -- the palm's box
    axes are rotated 75 deg about z relative to link_7, so the sum is genuinely
    off-diagonal and a diagonal-only treatment would be wrong.

    Returns ``(mass, (ixx,iyy,izz), com, principal_axes_quat)``.
    """
    import numpy as np

    m_arm, com_arm, diag_arm = arm_link7_inertial()
    com_arm = np.asarray(com_arm, dtype=float)

    ex, ey, ez = (float(v) for v in hand.palm_extents)
    m_palm = A.PALM_DENSITY_KG_M3 * ex * ey * ez
    # Box inertia about its own centre, in the PALM frame.
    i_palm_local = np.diag([
        m_palm * (ey * ey + ez * ez) / 12.0,
        m_palm * (ex * ex + ez * ez) / 12.0,
        m_palm * (ex * ex + ey * ey) / 12.0,
    ])

    t_palm = _mat_from_seg(flange_to_palm())          # link_7 <- palm
    r_palm = t_palm[:3, :3]
    com_palm = (t_palm @ np.append(
        np.asarray([float(v) for v in P.palm_center(hand.palm_extents)]), 1.0))[:3]
    i_palm = r_palm @ i_palm_local @ r_palm.T          # into the link_7 frame

    total = m_arm + m_palm
    com = (m_arm * com_arm + m_palm * com_palm) / total

    def shift(inertia, mass, centre):
        d = centre - com
        return inertia + mass * (float(d @ d) * np.eye(3) - np.outer(d, d))

    i_total = shift(np.diag(diag_arm), m_arm, com_arm) + shift(i_palm, m_palm, com_palm)

    vals, vecs = np.linalg.eigh(i_total)
    if np.linalg.det(vecs) < 0:                       # keep it a rotation
        vecs[:, 0] = -vecs[:, 0]
    return (float(total), tuple(float(v) for v in vals),
            tuple(float(v) for v in com), _mat_to_pos_quat(
                np.block([[vecs, np.zeros((3, 1))], [np.zeros((1, 3)), 1.0]]))[1])


# ---------------------------------------------------------------------------
# authoring
# ---------------------------------------------------------------------------

def author_hand(layer, root_path: str, hand: P.HandParams, spec,
                *, palm_body_path: str | None = None,
                link7_world=None) -> dict:
    """Author one hand's bodies, colliders and joints under ``root_path``.

    ``spec`` supplies joint limits, drive gains and effort so the authored robot
    cannot drift from what the rest of the stack believes. ``palm_body_path`` is
    the prim the first joint of every finger attaches to -- the merged palm body,
    which normally lives inside the referenced arm. ``link7_world`` is that
    body's world transform, used to place the finger link prims; the joints
    define the kinematics, so this only sets the initial pose.

    Returns a summary dict for the caller to log or assert on.
    """
    import numpy as np
    from pxr import Sdf

    palm_path = palm_body_path or f"{root_path}/{PALM_BODY}"
    world0 = np.eye(4) if link7_world is None else np.asarray(link7_world, float)
    palm_mat = world0 @ _mat_from_seg(flange_to_palm())

    limits_of = {}
    for i, fp in enumerate(hand.fingers):
        for slot, (lo, hi) in zip(P.JOINT_SLOTS, fp.limits):
            ghost = (not fp.active) or (not dict(zip(P.JOINT_SLOTS, fp.enabled))[slot])
            limits_of[(i, slot)] = ((0.0, 1e-8) if ghost else (float(lo), float(hi)))

    # Gains come from CONVERTER_DRIVE_* below, not the spec -- see the note there.

    n_bodies = n_caps = 0
    collider_links: dict[str, int] = {}
    define(layer, f"{root_path}/joints", "Scope")

    for i, fp in enumerate(hand.fingers):
        acc = palm_mat.copy()
        part_world: dict[str, np.ndarray] = {}
        for slot, part, _parent, seg in finger_chain(fp):
            acc = acc @ _mat_from_seg(seg)
            part_world[part] = acc.copy()

        # --- bodies ---
        for part, tier in LINK_PARTS:
            name = f"gen_f{i}_{part}"
            body = define(layer, f"{root_path}/{name}", "Xform",
                          ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
            mass, inertia, com, axes = _link_mass_props(fp, tier)
            attr(body, "physics:mass", Sdf.ValueTypeNames.Float, float(mass))
            from pxr import Gf
            attr(body, "physics:diagonalInertia", Sdf.ValueTypeNames.Float3,
                 Gf.Vec3f(*[float(v) for v in inertia]))
            attr(body, "physics:centerOfMass", Sdf.ValueTypeNames.Float3,
                 Gf.Vec3f(*[float(v) for v in com]))
            attr(body, "physics:principalAxes", Sdf.ValueTypeNames.Quatf,
                 Gf.Quatf(float(axes[0]), Gf.Vec3f(*[float(v) for v in axes[1:]])))
            pos, quat = _mat_to_pos_quat(part_world[part])
            _set_xform(body, pos, quat)
            n_bodies += 1

            # --- collider ---
            shape = _link_shape(fp, tier) if (tier and fp.active) else None
            if shape is None:
                continue
            length, radius, _ = shape
            define(layer, f"{root_path}/{name}/collisions", "Xform")
            mesh = define(layer, f"{root_path}/{name}/collisions/mesh_0", "Xform")
            _set_xform(mesh, (length / 2.0, 0.0, 0.0),
                       _quat_from_rpy((0.0, math.pi / 2.0, 0.0)))
            cap = define(layer, f"{root_path}/{name}/collisions/mesh_0/capsule",
                         "Capsule", ["PhysicsCollisionAPI"])
            attr(cap, "radius", Sdf.ValueTypeNames.Double, float(radius))
            attr(cap, "height", Sdf.ValueTypeNames.Double,
                 float(A.cylinder_part(length, radius)))
            attr(cap, "axis", Sdf.ValueTypeNames.Token, "Z")
            n_caps += 1
            # Record WHICH link got a collider, not just how many.
            #
            # The env's friction pass needs each link's shape-index range, and
            # rediscovering that afterwards is what costs ~96 min at 24,576
            # designs -- one create_rigid_body_view per link per design. Here it
            # is free: this line runs exactly when a collider is created, so the
            # map cannot disagree with the geometry. Note the condition above
            # depends on segment LENGTH (via _link_shape), not just fp.active --
            # inferring this map from the active-finger mask alone is precisely
            # the bug that mis-assigned design 5120's fingertips.
            collider_links[name] = collider_links.get(name, 0) + 1

        # --- joints ---
        for slot, part, parent_part, seg in finger_chain(fp):
            jname = f"gen_f{i}_{slot}"
            j = define(layer, f"{root_path}/joints/{jname}",
                       "PhysicsRevoluteJoint",
                       # PhysxJointAPI is REQUIRED for the physxJoint:* attributes
                       # below to be read at all. Authoring maxJointVelocity
                       # without it leaves the joint unlimited, and both Isaac
                       # Lab and PhysX report it as unlimited -- the attribute is
                       # present in the USD and simply never parsed.
                       ["PhysicsDriveAPI:angular", "PhysxJointAPI"])
            parent_path = (palm_path if parent_part is None
                           else f"{root_path}/gen_f{i}_{parent_part}")
            rel(j, "physics:body0", parent_path)
            rel(j, "physics:body1", f"{root_path}/gen_f{i}_{part}")

            # The first joint's parent is the MERGED palm body, so its origin is
            # the flange->palm transform composed with the mount -- exactly what
            # merge_fixed_joints folds together.
            origin = (_mat_from_seg(flange_to_palm()) @ _mat_from_seg(seg)
                      if parent_part is None else _mat_from_seg(seg))
            pos, quat = _mat_to_pos_quat(origin)
            from pxr import Gf
            attr(j, "physics:localPos0", Sdf.ValueTypeNames.Point3f,
                 Gf.Vec3f(*[float(v) for v in pos]))
            attr(j, "physics:localRot0", Sdf.ValueTypeNames.Quatf,
                 Gf.Quatf(float(quat[0]), Gf.Vec3f(*[float(v) for v in quat[1:]])))
            attr(j, "physics:localPos1", Sdf.ValueTypeNames.Point3f,
                 Gf.Vec3f(0.0, 0.0, 0.0))
            attr(j, "physics:localRot1", Sdf.ValueTypeNames.Quatf,
                 Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

            attr(j, "physics:axis", Sdf.ValueTypeNames.Token, "Z")
            lo, hi = limits_of[(i, slot)]
            # DEGREES: the converter writes 5.73e-7 for the 1e-8 rad ghost limit.
            attr(j, "physics:lowerLimit", Sdf.ValueTypeNames.Float,
                 float(math.degrees(lo)))
            attr(j, "physics:upperLimit", Sdf.ValueTypeNames.Float,
                 float(math.degrees(hi)))
            attr(j, "physics:jointEnabled", Sdf.ValueTypeNames.Bool, True)
            attr(j, "physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool, False)

            # Match the converter, not the spec: Kit writes a uniform 625/0 and
            # Isaac Lab overwrites gains at runtime from spec.hand_stiffness /
            # hand_damping via ImplicitActuatorCfg. Authoring the spec values
            # here would make the asset differ from a converted one while
            # changing nothing at runtime.
            attr(j, "drive:angular:physics:stiffness", Sdf.ValueTypeNames.Float,
                 CONVERTER_DRIVE_STIFFNESS)
            attr(j, "drive:angular:physics:damping", Sdf.ValueTypeNames.Float,
                 CONVERTER_DRIVE_DAMPING)
            # Effort is per SLOT and must NOT be throttled for ghosts: the
            # actuator is what holds a locked joint shut, and 1e-3 N.m could not
            # -- ghosted joints were pushed 21 deg open in training.
            attr(j, "drive:angular:physics:maxForce", Sdf.ValueTypeNames.Float,
                 float(A.SLOT_EFFORT_NM[slot]))
            attr(j, "drive:angular:physics:targetPosition",
                 Sdf.ValueTypeNames.Float, 0.0)
            # The URDF's <limit velocity=...>. Omitting it does NOT leave the
            # joint at some sane default -- it leaves maxJointVelocity
            # uninitialised, which Isaac Lab reports as 5.9e36 rad/s, i.e. no
            # limit at all. The hand still looked correct in every static
            # comparison (masses, inertias, limits, colliders, gains all exact)
            # while its joints could move arbitrarily fast, and the reaction
            # torque that produced settled the ARM into a different pose.
            # DEGREES per second, like the joint limits above -- the converter
            # writes 678.426 for an 11.8408 rad/s limit. Writing rad/s here is a
            # 57x underestimate that silently throttles every hand joint.
            attr(j, "physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float,
                 float(math.degrees(A.SLOT_VELOCITY_RAD_S[slot])))

    return {"bodies": n_bodies, "capsules": n_caps,
            "joints": len(hand.fingers) * P.N_JOINT_SLOTS,
            "collider_links": dict(collider_links)}


__all__ = ["author_hand", "define", "attr", "rel", "finger_chain",
           "flange_to_palm", "PALM_BODY", "LINK_PARTS"]
