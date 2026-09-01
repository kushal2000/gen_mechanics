"""The policy interface: what it sees, what its body is, what it does.

``observations`` builds the state vector, ``morphology`` the fixed-width design
descriptor appended to it -- which is what lets one policy condition on 24,576
different hands -- and ``actions`` turns the policy's output into joint targets.
"""

from .morphology import (  # noqa: F401
    DESCRIPTOR_DIM, FIELD_LAYOUT, PER_FINGER_DIM,
    describe_layout, finger_descriptor, hand_descriptor, population_descriptors,
)
from .observations import (  # noqa: F401
    KEYPOINT_CORNERS, build_observations, compute_intermediate_values,
    compute_obs_dim, derive_spaces, force_morphology_field, obs_field_sizes,
)
from .actions import (  # noqa: F401
    apply_action_pipeline, apply_wrench_dr, sample_log_uniform,
)
