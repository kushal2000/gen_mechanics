"""The policy interface: what it sees, what its body is, what it does.

``observations`` builds the state vector, ``morphology`` the fixed-width design
descriptor appended to it -- which is what lets one policy condition on 24,576
different hands -- and ``actions`` turns the policy's output into joint targets.
"""

from .morphology import (
    DESCRIPTOR_DIM, FIELD_LAYOUT, PER_FINGER_DIM,
    build_morphology_obs, describe_layout, finger_descriptor, hand_descriptor,
    population_descriptors,
)
from .observations import (
    KEYPOINT_CORNERS, build_observations, compute_intermediate_values,
    compute_obs_dim, derive_spaces, force_morphology_field, obs_field_sizes,
)
from .actions import (
    apply_action_pipeline, apply_wrench_dr, pre_physics_step, sample_log_uniform,
)

__all__ = [
    "DESCRIPTOR_DIM",
    "FIELD_LAYOUT",
    "KEYPOINT_CORNERS",
    "PER_FINGER_DIM",
    "apply_action_pipeline",
    "apply_wrench_dr",
    "build_morphology_obs",
    "build_observations",
    "compute_intermediate_values",
    "compute_obs_dim",
    "derive_spaces",
    "describe_layout",
    "finger_descriptor",
    "force_morphology_field",
    "hand_descriptor",
    "obs_field_sizes",
    "population_descriptors",
    "pre_physics_step",
    "sample_log_uniform",
]
