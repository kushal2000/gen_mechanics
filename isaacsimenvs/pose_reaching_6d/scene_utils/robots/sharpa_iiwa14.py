"""Left SHARPA hand (22 DoF, 5 fingers) on a KUKA iiwa14.

The reference robot: simtoolreal trained and shipped a checkpoint for it, so
every value here is transcribed from that setup rather than chosen (joint
order, gains, armature and home pose from its scene constants; one shared
palm and fingertip offset; adjacency from its LEFT map), and the transcription
was checked by a bitwise rollout parity test against it.
"""

from __future__ import annotations

from isaacsimenvs.pose_reaching_6d.scene_utils.robots.adjacency.sharpa_iiwa14 import SHARPA_IIWA14_ADJACENT_LINKS
from hand_sampler.robot_param_constants import (
    FINGERTIP_BODY_NAMES,
    FINGERTIP_OFFSET,
    HAND_ARMATURE,
    HAND_DAMPING,
    HAND_JOINT_NAMES,
    HAND_STIFFNESS,
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
from hand_sampler.robot_spec import RobotSpec, Vec3


# Thumb has 5 DoF, index/middle/ring 4 each, pinky 5 => 22.
#
# The left_1_ / left_2_ / ... numeric infixes are not cosmetic: they force Isaac
# Gym's alphabetical-within-depth joint sort into this order. They are preserved
# because the pretrained checkpoint's action layout depends on it.


# Fingertip bodies, post-merge: the *_elastomer and *_fingertip links are
# fixed-jointed onto the distal phalanges, so they collapse into the DP links.

# simtoolreal used one shared offset for all five pads. Kept identical here;
# the per-fingertip field exists for hands with asymmetric distal geometry.


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
    fingertip_offsets=tuple(FINGERTIP_OFFSET for _ in FINGERTIP_BODY_NAMES),

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
