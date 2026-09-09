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

ARM_JOINT_NAMES: tuple[str, ...] = (
    "iiwa14_joint_1",
    "iiwa14_joint_2",
    "iiwa14_joint_3",
    "iiwa14_joint_4",
    "iiwa14_joint_5",
    "iiwa14_joint_6",
    "iiwa14_joint_7",
)

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

# Applied on top of the home pose when reset.start_arm_higher is set, which the DexToolBench...
START_ARM_HIGHER_DELTAS: dict[str, float] = {
    "iiwa14_joint_2": -math.radians(10.0),
    "iiwa14_joint_4": +math.radians(10.0),
}

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

SHARPA_URDF = "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
ARM_LINKS = tuple(f"iiwa14_link_{i}" for i in range(8)) + ("iiwa14_link_ee",)
ARM_JOINTS = tuple(f"iiwa14_joint_{i}" for i in range(1, 8)) + ("iiwa14_joint_ee",)


# --- capsule tiers --------------------------------------------------------- Radius is half...
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

# rho = mass / capsule_volume(nominal_length, radius), where the capsule's TOTAL length --...
TIER_DENSITY_KG_M3: dict[str, float] = {
    "mc": 1853.918713613545,
    "pp": 3168.7741471413146,
    "mp": 3917.691174722447,
    "dp": 1149.0599166926079,
}

# The metacarpal density comes from the THUMB, not the pinky.
MC_MIN_LENGTH_M: float = 0.005

# --- palm ------------------------------------------------------------------ Modelled as a...
PALM_EXTENTS_M: tuple[float, float, float] = (
    0.04989549145102501,
    0.08517111465334892,
    0.08640973269939425,
)
PALM_MASS_KG: float = 0.72045
PALM_DENSITY_KG_M3: float = 1961.9482523719728

# Where the palm box sits in the palm link frame.
PALM_BOX_CENTER_M: tuple[float, float, float] = (0.00034, -0.00109, 0.04320)

# --- actuation, per joint slot --------------------------------------------- URDF...
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

# Per-slot PD gains, damping and armature, from the SHARPA spec's per-tier values...
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

# --- virtual / ghost links ------------------------------------------------- SHARPA already...
VIRTUAL_LINK_MASS_KG: float = 1e-6
VIRTUAL_LINK_INERTIA: float = 1e-6


# --- flange -> palm -------------------------------------------------------- SHARPA reaches...
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

# The proximal tier: a generated link is one capsule, not a three-tier chain.
GEN_LINK_RADIUS_M: float = TIER_RADIUS_M["pp"]

# Solid-cylinder equivalent, so a link's mass follows its length.
GEN_LINK_DENSITY_KG_M3: float = TIER_MASS_KG["pp"] / (
    math.pi * TIER_RADIUS_M["pp"] ** 2 * TIER_NOMINAL_LENGTH_M["pp"])

# One actuator type for every generated joint: SHARPA's PIP, its middle joint.
GEN_JOINT_EFFORT_NM: float = SLOT_EFFORT_NM["PIP"]
GEN_JOINT_VELOCITY_RAD_S: float = SLOT_VELOCITY_RAD_S["PIP"]
GEN_JOINT_STIFFNESS: float = SLOT_STIFFNESS["PIP"]
GEN_JOINT_DAMPING: float = SLOT_DAMPING["PIP"]
GEN_JOINT_ARMATURE: float = SLOT_ARMATURE["PIP"]

# The palm is a box; SHARPA's density, so a bigger palm weighs more.
GEN_PALM_DENSITY_KG_M3: float = PALM_MASS_KG / (
    PALM_EXTENTS_M[0] * PALM_EXTENTS_M[1] * PALM_EXTENTS_M[2])
