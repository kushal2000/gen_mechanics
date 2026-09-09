"""The policy interface: what it sees, what its body is, what it does.

``observations`` builds the state vector and ``actions`` turns the policy's
output into joint targets. Morphology is not a separate field: a hand is
described to the policy by its per-joint link geometry, so the same tokens carry
both the design and its state.
"""

from .observations import (
    KEYPOINT_CORNERS, build_observations, compute_intermediate_values,
    compute_obs_dim, derive_spaces, obs_field_sizes,
)
from .actions import (
    apply_action_pipeline, apply_wrench_dr, pre_physics_step, sample_log_uniform,
)

__all__ = [
    "KEYPOINT_CORNERS",
    "apply_action_pipeline",
    "apply_wrench_dr",
    "build_observations",
    "compute_intermediate_values",
    "compute_obs_dim",
    "derive_spaces",
    "obs_field_sizes",
    "pre_physics_step",
    "sample_log_uniform",
]
