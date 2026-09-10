"""Every fixed number in the project, in three groups.

IIWA14 and SHARPA are measured: they describe hardware that exists, and
re-deriving them needs genmech/tools/measure_sharpa_anchors.py against the
asset. The generated-population defaults are chosen, and they are anchored to
the SHARPA measurements so a sampled hand lands in the same physical regime as
the one the task was built around.

Design-space BOUNDS -- the grids, the finger and joint limits, what counts as a
legal hand -- are not here. They live in design_space and validate_design,
because they define the search rather than the hardware.
"""

from __future__ import annotations

import math


# =============================================================================
# IIWA14 ARM -- measured
# =============================================================================


ARM_NAME = "iiwa14"

# --- joints --------------------------------------------------------------------
ARM_JOINT_NAMES: tuple[str, ...] = (
    "iiwa14_joint_1",
    "iiwa14_joint_2",
    "iiwa14_joint_3",
    "iiwa14_joint_4",
    "iiwa14_joint_5",
    "iiwa14_joint_6",
    "iiwa14_joint_7",
)

# --- gains and home pose -------------------------------------------------------
ARM_STIFFNESS: dict[str, float] = {
    "iiwa14_joint_1": 600.0,
    "iiwa14_joint_2": 600.0,
    "iiwa14_joint_3": 500.0,
    "iiwa14_joint_4": 400.0,
    "iiwa14_joint_5": 200.0,
    "iiwa14_joint_6": 200.0,
    "iiwa14_joint_7": 200.0,
}

ARM_DAMPING: dict[str, float] = {
    "iiwa14_joint_1": 27.027026473513512,
    "iiwa14_joint_2": 27.027026473513512,
    "iiwa14_joint_3": 24.672186769721083,
    "iiwa14_joint_4": 22.067474708266914,
    "iiwa14_joint_5": 9.752538131173853,
    "iiwa14_joint_6": 9.147747263670984,
    "iiwa14_joint_7": 9.147747263670984,
}

ARM_DEFAULT_JOINT_POS: dict[str, float] = {
    "iiwa14_joint_1": -1.571,
    "iiwa14_joint_2": 1.571,
    "iiwa14_joint_3": 0.0,
    "iiwa14_joint_4": 1.376,
    "iiwa14_joint_5": 0.0,
    "iiwa14_joint_6": 1.485,
    "iiwa14_joint_7": 1.308,
}

# Applied on top of the home pose when reset.start_arm_higher is set.
START_ARM_HIGHER_DELTAS: dict[str, float] = {
    "iiwa14_joint_2": -math.radians(10.0),
    "iiwa14_joint_4": +math.radians(10.0),
}

# --- placement and self-collision ----------------------------------------------
# Base placement on the table.
BASE_POS: tuple[float, float, float] = (0.0, 0.8, 0.0)
BASE_ROT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

# Serial chain, so only consecutive links need filtering.
ARM_ADJACENT_LINKS: dict[str, list[str]] = {
    "iiwa14_link_0": ["iiwa14_link_1"],
    "iiwa14_link_1": ["iiwa14_link_0", "iiwa14_link_2"],
    "iiwa14_link_2": ["iiwa14_link_1", "iiwa14_link_3"],
    "iiwa14_link_3": ["iiwa14_link_2", "iiwa14_link_4"],
    "iiwa14_link_4": ["iiwa14_link_3", "iiwa14_link_5"],
    "iiwa14_link_5": ["iiwa14_link_4", "iiwa14_link_6"],
    "iiwa14_link_6": ["iiwa14_link_5", "iiwa14_link_7"],
}

# The link the hand's palm merges into under merge_fixed_joints.
ARM_TIP_LINK = "iiwa14_link_7"

ARM_LINK_PRIM_REGEX = "/World/envs/env_.*/Robot/iiwa14_link_.*/visuals"


assert len(ARM_JOINT_NAMES) == 7
assert set(ARM_STIFFNESS) == set(ARM_JOINT_NAMES)
assert set(ARM_DAMPING) == set(ARM_JOINT_NAMES)
assert set(ARM_DEFAULT_JOINT_POS) == set(ARM_JOINT_NAMES)


# =============================================================================
# SHARPA HAND -- measured
#
# The SLOT_* and TIER_* tables are keyed by the slot vocabulary the tree
# representation replaced, so they describe the old parameterisation. The palm,
# flange and virtual-link scalars are representation-neutral.
# =============================================================================

# --- the asset -----------------------------------------------------------------
SHARPA_URDF = "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
ARM_LINKS = tuple(f"iiwa14_link_{i}" for i in range(8)) + ("iiwa14_link_ee",)
ARM_JOINTS = tuple(f"iiwa14_joint_{i}" for i in range(1, 8)) + ("iiwa14_joint_ee",)


# --- capsule tiers -------------------------------------------------------------
# Radius is half the measured cross-section; one capsule per phalanx tier.
TIER_RADIUS_M: dict[str, float] = {
    "mc": 0.019348576781339943,
    "pp": 0.010165722109377384,
    "mp": 0.00893329898826778,
    "dp": 0.007338061812333763,
}

# Nominal length is the URDF *joint spacing*, not the mesh extent.
TIER_NOMINAL_LENGTH_M: dict[str, float] = {
    "mc": 0.0661,   # thumb CMC_AA -> MCP_FE
    "pp": 0.0470,   # MCP_AA -> PIP
    "mp": 0.0315,   # PIP -> DIP
    "dp": 0.0260,   # DIP -> fingertip
}

# Measured link masses the densities are fitted to.
TIER_MASS_KG: dict[str, float] = {
    "mc": 0.116,      # left_thumb_MC
    "pp": 0.04138,    # left_index_PP
    "mp": 0.02509,    # left_index_MP
    "dp": 0.004103,   # left_index_DP + _elastomer + _fingertip
}

# rho = mass / capsule_volume(nominal_length, radius), on the TOTAL capsule length.
TIER_DENSITY_KG_M3: dict[str, float] = {
    "mc": 1853.918713613545,
    "pp": 3168.7741471413146,
    "mp": 3917.691174722447,
    "dp": 1149.0599166926079,
}

# Below this a metacarpal is a virtual link rather than a capsule.
MC_MIN_LENGTH_M: float = 0.005

# --- palm ----------------------------------------------------------------------
# Modelled as a box, not a mesh.
PALM_EXTENTS_M: tuple[float, float, float] = (
    0.04989549145102501,
    0.08517111465334892,
    0.08640973269939425,
)
PALM_MASS_KG: float = 0.72045
PALM_DENSITY_KG_M3: float = 1961.9482523719728

# Where the palm box sits in the palm link frame.
PALM_BOX_CENTER_M: tuple[float, float, float] = (0.00034, -0.00109, 0.04320)

# PROVENANCE: simtoolreal trained and shipped a checkpoint for this robot, so
# the joint order, gains, armature and home pose below are TRANSCRIBED from its
# scene constants rather than chosen or derived -- the URDF carries no gains and
# no armature at all. The transcription was checked by a bitwise rollout parity
# test against that setup.
#
# The left_1_ / left_2_ / ... numeric infixes are NOT cosmetic: they force Isaac
# Gym's alphabetical-within-depth joint sort into this order, and the pretrained
# checkpoint's action layout depends on it. Do not tidy them away.

# --- controlled joints ---------------------------------------------------------
HAND_JOINT_NAMES: tuple[str, ...] = (
    "left_1_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE",
    "left_thumb_MCP_AA", "left_thumb_IP",
    "left_2_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
    "left_3_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
    "left_4_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
    "left_5_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA",
    "left_pinky_PIP", "left_pinky_DIP",
)

# --- per-joint gains, damping and armature -------------------------------------
# Authoritative: what the articulation is actually driven with.
HAND_STIFFNESS: dict[str, float] = {
    "left_1_thumb_CMC_FE": 6.95, "left_thumb_CMC_AA": 13.2, "left_thumb_MCP_FE": 4.76,
    "left_thumb_MCP_AA": 6.62, "left_thumb_IP": 0.9,
    "left_2_index_MCP_FE": 4.76, "left_index_MCP_AA": 6.62,
    "left_index_PIP": 0.9, "left_index_DIP": 0.9,
    "left_3_middle_MCP_FE": 4.76, "left_middle_MCP_AA": 6.62,
    "left_middle_PIP": 0.9, "left_middle_DIP": 0.9,
    "left_4_ring_MCP_FE": 4.76, "left_ring_MCP_AA": 6.62,
    "left_ring_PIP": 0.9, "left_ring_DIP": 0.9,
    "left_5_pinky_CMC": 1.38, "left_pinky_MCP_FE": 4.76, "left_pinky_MCP_AA": 6.62,
    "left_pinky_PIP": 0.9, "left_pinky_DIP": 0.9,
}

HAND_DAMPING: dict[str, float] = {
    "left_1_thumb_CMC_FE": 0.28676845, "left_thumb_CMC_AA": 0.40845109,
    "left_thumb_MCP_FE": 0.20394083, "left_thumb_MCP_AA": 0.24044435,
    "left_thumb_IP": 0.04190723,
    "left_2_index_MCP_FE": 0.20859232, "left_index_MCP_AA": 0.24595532,
    "left_index_PIP": 0.04243185, "left_index_DIP": 0.03504461,
    "left_3_middle_MCP_FE": 0.2085923, "left_middle_MCP_AA": 0.24595532,
    "left_middle_PIP": 0.04243185, "left_middle_DIP": 0.03504461,
    "left_4_ring_MCP_FE": 0.20859226, "left_ring_MCP_AA": 0.24595528,
    "left_ring_PIP": 0.04243183, "left_ring_DIP": 0.0350446,
    "left_5_pinky_CMC": 0.02782345, "left_pinky_MCP_FE": 0.20859229,
    "left_pinky_MCP_AA": 0.24595528, "left_pinky_PIP": 0.04243183,
    "left_pinky_DIP": 0.0350446,
}

HAND_ARMATURE: dict[str, float] = {
    "left_1_thumb_CMC_FE": 0.0032, "left_thumb_CMC_AA": 0.0032,
    "left_thumb_MCP_FE": 0.00265, "left_thumb_MCP_AA": 0.00265, "left_thumb_IP": 0.0006,
    "left_2_index_MCP_FE": 0.00265, "left_index_MCP_AA": 0.00265,
    "left_index_PIP": 0.0006, "left_index_DIP": 0.00042,
    "left_3_middle_MCP_FE": 0.00265, "left_middle_MCP_AA": 0.00265,
    "left_middle_PIP": 0.0006, "left_middle_DIP": 0.00042,
    "left_4_ring_MCP_FE": 0.00265, "left_ring_MCP_AA": 0.00265,
    "left_ring_PIP": 0.0006, "left_ring_DIP": 0.00042,
    "left_5_pinky_CMC": 0.00012, "left_pinky_MCP_FE": 0.00265,
    "left_pinky_MCP_AA": 0.00265, "left_pinky_PIP": 0.0006, "left_pinky_DIP": 0.00042,
}

# MEASURED, NOT APPLIED. simtoolreal's isaacgym env sets these; nothing here
# does, because joint friction is zero on every robot. Kept as the record of
# what the hardware transcription said.
#
# These are torques in N.m: 21 of the 22 are exactly 4% (proximal) or 2%
# (distal) of that joint's effort limit in SHARPA_URDF, to six figures. The URDF
# itself declares no friction for any hand joint -- no <dynamics> element at
# all -- and no armature or transmission either, so these and HAND_ARMATURE
# could only have come from simtoolreal's scene constants.
# left_5_pinky_CMC is the exception at 2.27%, a round 0.012 where 2% of its
# 0.5285 effort would be 0.01057; it is the armature outlier too.
HAND_FRICTION: dict[str, float] = {
    "left_1_thumb_CMC_FE": 0.132,
    "left_thumb_CMC_AA": 0.132,
    "left_thumb_MCP_FE": 0.07456,
    "left_thumb_MCP_AA": 0.07456,
    "left_thumb_IP": 0.01276,
    "left_2_index_MCP_FE": 0.07456,
    "left_index_MCP_AA": 0.07456,
    "left_index_PIP": 0.01276,
    "left_index_DIP": 0.00378738,
    "left_3_middle_MCP_FE": 0.07456,
    "left_middle_MCP_AA": 0.07456,
    "left_middle_PIP": 0.01276,
    "left_middle_DIP": 0.00378738,
    "left_4_ring_MCP_FE": 0.07456,
    "left_ring_MCP_AA": 0.07456,
    "left_ring_PIP": 0.01276,
    "left_ring_DIP": 0.00378738,
    "left_5_pinky_CMC": 0.012,
    "left_pinky_MCP_FE": 0.07456,
    "left_pinky_MCP_AA": 0.07456,
    "left_pinky_PIP": 0.01276,
    "left_pinky_DIP": 0.00378738,
}

# --- fingertips ----------------------------------------------------------------
FINGERTIP_BODY_NAMES: tuple[str, ...] = (
    "left_index_DP", "left_middle_DP", "left_ring_DP", "left_thumb_DP", "left_pinky_DP",
)

FINGERTIP_OFFSET: Vec3 = (0.02, 0.002, 0.0)


# --- actuation limits, per joint slot ------------------------------------------
# Effort and velocity ceilings as the URDF declares them.
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


# --- virtual / ghost links -----------------------------------------------------
# Near-zero so a disabled slot keeps its link without adding mass.
VIRTUAL_LINK_MASS_KG: float = 1e-6
VIRTUAL_LINK_INERTIA: float = 1e-6


# --- drive gains the URDF converter writes onto every joint prim ---
# Not the actuator's gains: Isaac Lab's ImplicitActuator overrides these at
# runtime. They exist so the DriveAPI prim is there for it to write into.
CONVERTER_DRIVE_STIFFNESS: float = 625.0
CONVERTER_DRIVE_DAMPING: float = 0.0

# --- flange -> palm ------------------------------------------------------------
# Where the hand attaches to the arm's last link.
# link_7 -> flange, the piece merge_fixed_joints collapses along with the next.
LINK7_TO_FLANGE_Z_M: float = 0.045
FLANGE_TO_PALM_Z_M: float = 0.05
FLANGE_TO_PALM_YAW_RAD: float = -1.3089969389957472   # -75 deg


def cylinder_part(total_length: float, radius: float) -> float:
    """Cylindrical section of a capsule whose TOTAL length is ``total_length``."""
    return max(total_length - 2.0 * radius, 0.0)


# =============================================================================
# GENERATED POPULATION -- chosen, anchored to the measurements above
#
# What a sampled hand gets when it is authored, where the design space does not
# say. Every value here is one SHARPA number or a simple function of one, so a
# generated hand is heavier or lighter than SHARPA for a reason we can name.
# =============================================================================

# --- links ---------------------------------------------------------------------
# The proximal tier: a generated link is one capsule, not a three-tier chain.
GEN_LINK_RADIUS_M: float = TIER_RADIUS_M["pp"]

# Solid-cylinder equivalent, so a link's mass follows its length.
GEN_LINK_DENSITY_KG_M3: float = TIER_MASS_KG["pp"] / (
    math.pi * TIER_RADIUS_M["pp"] ** 2 * TIER_NOMINAL_LENGTH_M["pp"])

# --- actuation ----------------------------------------------------------------
# ONE ACTUATOR EVERYWHERE, so a morphology comparison is not also a comparison
# of who was given the stronger motors. SHARPA does the opposite -- 3.3 N.m at
# the thumb base down to 0.189 at a DIP, a 17x fall-off matching how humanoids
# are actuated (Unitree G1: knee 120, hip 88, ankle 50, wrist 8) -- so a
# generated hand is NOT actuated like SHARPA, deliberately.

# Hardware ceilings. 1.0 N.m sits between SHARPA's PIP (0.638) and MCP (1.864).
GEN_JOINT_EFFORT_NM: float = 1.0
GEN_JOINT_VELOCITY_RAD_S: float = 10.0

# Control, not hardware: these say how the joint tracks a target, and are ours
# to tune. kp = 1.0 is matched to the torque ceiling -- the actuator saturates
# at ~1 rad of error, about the joint's full travel, so it can use its whole
# range without sitting permanently clipped. kd keeps SHARPA's damping ratio,
# which is kd/kp ~ 0.045 at every one of its joints.
GEN_JOINT_STIFFNESS: float = 1.0

# Hardware again, and it scales with the actuator's torque in SHARPA:
# armature/effort averages 0.00116 across its five tiers. Joint friction is not
# modelled at all -- see HAND_FRICTION below.
GEN_JOINT_ARMATURE: float = 0.00116 * GEN_JOINT_EFFORT_NM


# Critically damped against the joint's own inertia, which is what SHARPA is:
# kd/(2*sqrt(kp*armature)) is 0.90 to 1.08 across all 22 of its joints.
GEN_JOINT_DAMPING: float = 2 * 0.929 * math.sqrt(GEN_JOINT_STIFFNESS * GEN_JOINT_ARMATURE)


def gen_joint_drive(depth: int = 0, theta: float = 0.0):
    """``(effort, velocity, stiffness, damping, armature)`` for a joint.

    Takes depth and theta so a caller need not know they are ignored; the whole
    point is that every generated joint is identical.
    """
    return (GEN_JOINT_EFFORT_NM, GEN_JOINT_VELOCITY_RAD_S, GEN_JOINT_STIFFNESS,
            GEN_JOINT_DAMPING, GEN_JOINT_ARMATURE)


# --- palm ----------------------------------------------------------------------
# A box at SHARPA's density, so a bigger palm weighs more.
GEN_PALM_DENSITY_KG_M3: float = PALM_MASS_KG / (
    PALM_EXTENTS_M[0] * PALM_EXTENTS_M[1] * PALM_EXTENTS_M[2])
