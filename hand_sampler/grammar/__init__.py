"""Hand-kinematics grammar: frozen data model + forward kinematics + sampling.

Stdlib + numpy only. No dependency on scipy or hand_sampler.design_space.
"""

from .kinematics import (
    Approximation,
    AffineCoupling,
    Body,
    Frame,
    Joint,
    JointGroup,
    KinematicModel,
    LoopClosure,
    ModelError,
    Pose,
    UnsupportedConstruct,
    validate,
)
from .fk import (
    forward_kinematics,
    pose_to_matrix,
    position_error,
    rodrigues,
    rotation_error,
    rpy_to_matrix,
)
from .coords import (
    LimitConflict,
    admissible_box,
    check_limits,
    independent_joints,
    movable_joints,
    q_from_u,
    sample_configurations,
)

__all__ = [
    "Approximation",
    "AffineCoupling",
    "Body",
    "Frame",
    "Joint",
    "JointGroup",
    "KinematicModel",
    "LoopClosure",
    "ModelError",
    "Pose",
    "UnsupportedConstruct",
    "validate",
    "forward_kinematics",
    "pose_to_matrix",
    "position_error",
    "rodrigues",
    "rotation_error",
    "rpy_to_matrix",
    "LimitConflict",
    "admissible_box",
    "check_limits",
    "independent_joints",
    "movable_joints",
    "q_from_u",
    "sample_configurations",
]
