"""Left SHARPA hand (22 DoF, 5 fingers) on a KUKA iiwa14.

The reference robot: simtoolreal trained and shipped a checkpoint for it, so
every value here is transcribed from that setup rather than chosen (joint
order, gains, armature and home pose from its scene constants; one shared
palm and fingertip offset; adjacency from its LEFT map), and the transcription
was checked by a bitwise rollout parity test against it.
"""

from __future__ import annotations

from isaacsimenvs.pose_reaching_6d.scene_utils.robots.adjacency.sharpa_iiwa14 import SHARPA_IIWA14_ADJACENT_LINKS
from hand_sampler.iiwa14_arm import (
    ARM_DAMPING,
    ARM_DEFAULT_JOINT_POS,
    ARM_JOINT_NAMES,
    ARM_LINK_PRIM_REGEX,
    ARM_NAME,
    ARM_STIFFNESS,
    ARM_TIP_LINK,
    BASE_POS,
    BASE_ROT,
    START_ARM_HIGHER_DELTAS,
)
from hand_sampler.spec import RobotSpec, Vec3


# Thumb has 5 DoF, index/middle/ring 4 each, pinky 5 => 22.
#
# The left_1_ / left_2_ / ... numeric infixes are not cosmetic: they force Isaac
# Gym's alphabetical-within-depth joint sort into this order. They are preserved
# because the pretrained checkpoint's action layout depends on it.
HAND_JOINT_NAMES: tuple[str, ...] = (
    "left_1_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE",
    "left_thumb_MCP_AA", "left_thumb_IP",
    "left_2_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
    "left_3_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
    "left_4_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
    "left_5_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA",
    "left_pinky_PIP", "left_pinky_DIP",
)

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

# Fingertip bodies, post-merge: the *_elastomer and *_fingertip links are
# fixed-jointed onto the distal phalanges, so they collapse into the DP links.
FINGERTIP_BODY_NAMES: tuple[str, ...] = (
    "left_index_DP", "left_middle_DP", "left_ring_DP", "left_thumb_DP", "left_pinky_DP",
)

# simtoolreal used one shared offset for all five pads. Kept identical here;
# the per-fingertip field exists for hands with asymmetric distal geometry.
_SHARPA_FINGERTIP_OFFSET: Vec3 = (0.02, 0.002, 0.0)


SHARPA_IIWA14 = RobotSpec(
    name="sharpa_iiwa14",
    arm_name=ARM_NAME,
    hand_name="sharpa",
    urdf_path="assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf",

    arm_joint_names=ARM_JOINT_NAMES,
    hand_joint_names=HAND_JOINT_NAMES,

    # The palm merges into the arm's last link under merge_fixed_joints.
    palm_body_name=ARM_TIP_LINK,
    fingertip_body_names=FINGERTIP_BODY_NAMES,

    arm_stiffness=ARM_STIFFNESS,
    arm_damping=ARM_DAMPING,
    hand_stiffness=HAND_STIFFNESS,
    hand_damping=HAND_DAMPING,
    hand_armature=HAND_ARMATURE,

    arm_default_joint_pos=ARM_DEFAULT_JOINT_POS,
    hand_default_joint_pos={name: 0.0 for name in HAND_JOINT_NAMES},
    start_arm_higher_deltas=START_ARM_HIGHER_DELTAS,

    # Grasp center, ~16 cm out along the flange axis from iiwa14_link_7.
    palm_center_offset=(-0.0, -0.02, 0.16),
    fingertip_offsets=tuple(_SHARPA_FINGERTIP_OFFSET for _ in FINGERTIP_BODY_NAMES),

    adjacent_links=SHARPA_IIWA14_ADJACENT_LINKS,
    link_prim_regexes=(
        ARM_LINK_PRIM_REGEX,
        "/World/envs/env_.*/Robot/left_.*/visuals",
    ),

    base_pos=BASE_POS,
    base_rot=BASE_ROT,

    notes=(
        "Reference robot. All values transcribed from simtoolreal's validated "
        "Isaac Sim setup and checked by bitwise rollout parity against it."
    ),
)


__all__ = ["SHARPA_IIWA14"]
