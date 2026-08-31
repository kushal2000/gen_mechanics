"""Measured SHARPA constants that anchor the generated design space.

Every number here was measured from the SHARPA asset by
``genmech/tools/measure_sharpa_anchors.py``, which can re-derive and check them::

    .venv_isaacsim/bin/python -m genmech.tools.measure_sharpa_anchors --verify

They are committed rather than measured at import time so that the design space
is reproducible without loading meshes, and so an asset change shows up as a
failing verify rather than as a silently different set of hands.

**Why anchor at all.** The generator builds hands from capsules, and a capsule
needs a radius and a density. Chosen by eye, both would be inventions, and mass
is not free -- it sets how hard a finger hits the object and what the arm carries.
Instead the radius is read off SHARPA's own collision meshes and the density is
solved so a capsule at SHARPA's nominal length and radius weighs what SHARPA
weighs. The SHARPA-like hand then reproduces SHARPA's masses by construction.

Kinematic anchors (mount poses, segment lengths, joint limits) live in
``params.py`` next to the ranges they centre; this module is the *geometry and
mass* model only.
"""

from __future__ import annotations


# --- capsule tiers ---------------------------------------------------------
#
# Radius is half the mean of a mesh's two SHORT AABB axes. The long axis
# describes segment length, not thickness, so it is excluded.
TIER_RADIUS_M: dict[str, float] = {
    "mc": 0.019348576781339943,
    "pp": 0.010165722109377384,
    "mp": 0.00893329898826778,
    "dp": 0.007338061812333763,
}

# Nominal length is the URDF *joint spacing*, not the mesh extent. The meshes
# overhang their joints (MP's mesh spans 36.5 mm across a 31.5 mm joint spacing),
# but kinematics follow the joint spacing, so the capsule must too.
TIER_NOMINAL_LENGTH_M: dict[str, float] = {
    "mc": 0.0661,   # thumb CMC_AA -> MCP_FE
    "pp": 0.0470,   # MCP_AA -> PIP
    "mp": 0.0315,   # PIP -> DIP
    "dp": 0.0260,   # DIP -> fingertip
}

# Measured link masses the densities are fitted to. DP carries three links
# because merge_fixed_joints folds the elastomer pad (1.100 g) and the fingertip
# frame (0.001 g) into it; a capsule fitted to the 3.002 g bare DP would come out
# 27% light.
TIER_MASS_KG: dict[str, float] = {
    "mc": 0.116,      # left_thumb_MC
    "pp": 0.04138,    # left_index_PP
    "mp": 0.02509,    # left_index_MP
    "dp": 0.004103,   # left_index_DP + _elastomer + _fingertip
}

# rho = mass / capsule_volume(nominal_length, radius), where the capsule's TOTAL
# length -- cylindrical section plus both hemispherical caps -- equals the joint
# spacing. Getting that wrong is easy: Isaac Lab's converter reads a URDF
# cylinder's `length` as the cylindrical section only, so emitting the segment
# length directly yields a capsule 2r too long (67.3 mm for a 47.0 mm phalanx).
#
# The 3.4x spread between MP and DP is real and worth preserving: proximal links
# house actuators (linear density ~0.85 g/mm) while the distal is a passive shell
# (~0.12 g/mm). A single global density would be wrong by 7x on the fingertip,
# which is the link that actually touches the object.
#
# Sanity: these land near the densities implied by the real mesh volumes
# (pp 3032, mp 3131, dp 1644 kg/m^3), which the earlier over-long capsules did
# not -- the corrected shape is a better model of the link it stands in for.
TIER_DENSITY_KG_M3: dict[str, float] = {
    "mc": 1853.918713613545,
    "pp": 3168.7741471413146,
    "mp": 3917.691174722447,
    "dp": 1149.0599166926079,
}

# The metacarpal density comes from the THUMB, not the pinky. SHARPA's pinky MC
# is 9.6 mm long with a 17.3 mm radius, so as a capsule it is a sphere with a
# sliver in the middle: volume is dominated by the end caps and barely responds
# to length, and the fitted density blows up to 4108 kg/m^3 against the thumb's
# 1073. Below this length a capsule stops being a meaningful model of a segment,
# so the generator emits a massless jointless link instead of geometry.
MC_MIN_LENGTH_M: float = 0.005

# --- palm ------------------------------------------------------------------
#
# Modelled as a box on the mesh AABB. The real palm fills 57% of that box, so the
# fitted density is correspondingly below the material density -- the box is a
# collision proxy, not a claim about what the palm is made of.
PALM_EXTENTS_M: tuple[float, float, float] = (
    0.04989549145102501,
    0.08517111465334892,
    0.08640973269939425,
)
PALM_MASS_KG: float = 0.72045
PALM_DENSITY_KG_M3: float = 1961.9482523719728

# Where the palm box sits in the palm link frame. The mesh runs from z=0 up to
# z=86.4 mm, so a box centred on the link origin would put half the palm inside
# the wrist.
PALM_BOX_CENTER_M: tuple[float, float, float] = (0.00034, -0.00109, 0.04320)

# --- actuation, per joint slot ---------------------------------------------
#
# URDF effort/velocity limits, read off SHARPA. These are motor properties, so
# they are held fixed in v1 rather than searched -- a design that won by drawing
# a stronger actuator would not be telling us about geometry.
#
# Two slots are approximations, because the template's 6 slots do not map
# one-to-one onto SHARPA's heterogeneous fingers:
#   * the pinky's CMC (effort 0.5285, vel 35.07) lands in the CMC_FE slot, which
#     takes the thumb's much stronger 3.3 / 11.84;
#   * the thumb's IP (effort 0.638) lands in the DIP slot, which takes 0.189.
SLOT_EFFORT_NM: dict[str, float] = {
    "CMC_FE": 3.3,
    "CMC_AA": 3.3,
    "MCP_FE": 1.864,
    "MCP_AA": 1.864,
    "PIP": 0.638,
    "DIP": 0.189369,
}
SLOT_VELOCITY_RAD_S: dict[str, float] = {
    "CMC_FE": 11.84076833,
    "CMC_AA": 11.84076833,
    "MCP_FE": 16.07692878,
    "MCP_AA": 16.07692878,
    "PIP": 11.61831603,
    "DIP": 14.66594768,
}

# Per-slot PD gains, damping and armature, from the SHARPA spec's per-tier
# values (isaacsimenvs/robots/sharpa_iiwa14.py). Same reasoning: controller and motor
# properties, held fixed so the search sees geometry only.
SLOT_STIFFNESS: dict[str, float] = {
    "CMC_FE": 1.38, "CMC_AA": 1.38,
    "MCP_FE": 4.76, "MCP_AA": 6.62,
    "PIP": 0.9, "DIP": 0.9,
}
SLOT_DAMPING: dict[str, float] = {
    "CMC_FE": 0.02782345, "CMC_AA": 0.02782345,
    "MCP_FE": 0.20859232, "MCP_AA": 0.24595532,
    "PIP": 0.04243185, "DIP": 0.03504461,
}
SLOT_ARMATURE: dict[str, float] = {
    "CMC_FE": 0.0032, "CMC_AA": 0.0032,
    "MCP_FE": 0.00265, "MCP_AA": 0.00265,
    "PIP": 0.0006, "DIP": 0.00042,
}

# --- virtual / ghost links -------------------------------------------------
#
# SHARPA already represents its zero-length bodies (between coincident FE and AA
# joints) with mass 1e-6 kg. Ghosted joints therefore reuse the reference hand's
# own convention rather than importing a foreign one.
VIRTUAL_LINK_MASS_KG: float = 1e-6
VIRTUAL_LINK_INERTIA: float = 1e-6


# --- flange -> palm --------------------------------------------------------
#
# SHARPA reaches its palm through two fixed joints:
#   iiwa14_link_ee --(rpy 0,0,+15 deg)--> sharpa_mount --(z=0.05, rpy 0,0,-90 deg)--> palm
# Composed, that is a pure z translation of 50 mm and a net -75 deg yaw (the
# 15 deg rotation is about z, so it leaves the z offset alone). Generated hands
# mount at the same place, so palm_center_offset keeps meaning the same physical
# point across every robot in the registry (docs/methodology.md).
FLANGE_TO_PALM_Z_M: float = 0.05
FLANGE_TO_PALM_YAW_RAD: float = -1.3089969389957472   # -75 deg


def cylinder_part(total_length: float, radius: float) -> float:
    """Cylindrical section of a capsule whose TOTAL length is ``total_length``.

    One definition, shared by the URDF builder and the self-collision checker,
    so the shape they assume cannot drift from the shape PhysX simulates.
    """
    return max(total_length - 2.0 * radius, 0.0)


__all__ = [
    "cylinder_part",
    "TIER_RADIUS_M",
    "TIER_NOMINAL_LENGTH_M",
    "TIER_MASS_KG",
    "TIER_DENSITY_KG_M3",
    "MC_MIN_LENGTH_M",
    "PALM_EXTENTS_M",
    "PALM_MASS_KG",
    "PALM_DENSITY_KG_M3",
    "PALM_BOX_CENTER_M",
    "SLOT_EFFORT_NM",
    "SLOT_VELOCITY_RAD_S",
    "SLOT_STIFFNESS",
    "SLOT_DAMPING",
    "SLOT_ARMATURE",
    "VIRTUAL_LINK_MASS_KG",
    "VIRTUAL_LINK_INERTIA",
    "FLANGE_TO_PALM_Z_M",
    "FLANGE_TO_PALM_YAW_RAD",
]
