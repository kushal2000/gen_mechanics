"""Kit-free flat-observation and per-joint-token layouts.

The environment emits every token feature explicitly.  The transformer only
gathers columns from the same flat vector consumed by the MLP; it neither parses
joint names nor reaches into a URDF to invent architecture-only features.
"""

from __future__ import annotations


NUM_KEYPOINTS: int = 4
JOINT_BOX_POINTS: int = 4
JOINT_BOX_DIM: int = 3 * JOINT_BOX_POINTS


def obs_field_sizes(spec) -> dict[str, int]:
    """Per-field widths for ``spec``."""
    n_joints = spec.num_joints
    n_hand = spec.num_hand_joints
    n_tips = spec.num_fingertip_slots
    return {
        # Current and one-step-old proprioception, in canonical arm+hand order.
        "joint_pos": n_joints,
        "joint_vel": n_joints,
        "prev_joint_pos": n_joints,
        "prev_joint_vel": n_joints,
        "prev_action_targets": n_joints,
        # Explicit hand-token data.  These fields never include the arm.
        "joint_link_bbox": JOINT_BOX_DIM * n_hand,
        "joint_lower": n_hand,
        "joint_upper": n_hand,
        "joint_enabled": n_hand,
        "object_keypoints_rel_joint": 3 * NUM_KEYPOINTS * n_hand,
        "hand_scale": 1,
        # Global task state.
        "palm_pos": 3,
        "palm_rot": 4,
        "palm_vel": 6,
        "object_rot": 4,
        "object_vel": 6,
        "keypoints_rel_palm": 3 * NUM_KEYPOINTS,
        "keypoints_rel_goal": 3 * NUM_KEYPOINTS,
        "object_scales": 3,
        # Privileged critic/training state.
        "closest_keypoint_max_dist": 1,
        "closest_fingertip_dist": n_tips,
        "lifted_object": 1,
        "progress": 1,
        "successes": 1,
        "reward": 1,
        # Legacy fields remain sizeable so old saved configs fail at checkpoint
        # width rather than at config parsing.  New configs do not request them.
        "fingertip_pos_rel_palm": 3 * n_tips,
        "morphology": 143,
    }


def compute_obs_dim(field_list, spec) -> int:
    sizes = obs_field_sizes(spec)
    unknown = [f for f in field_list if f not in sizes]
    if unknown:
        raise KeyError(f"unknown observation field(s) {unknown}; valid: {sorted(sizes)}")
    return sum(sizes[f] for f in field_list)


def field_offsets(field_list, spec) -> dict[str, tuple[int, int]]:
    sizes = obs_field_sizes(spec)
    unknown = [f for f in field_list if f not in sizes]
    if unknown:
        raise KeyError(f"unknown observation field(s) {unknown}; valid: {sorted(sizes)}")
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for field in field_list:
        if field in offsets:
            raise ValueError(f"field {field!r} appears twice in {list(field_list)}")
        offsets[field] = (cursor, cursor + sizes[field])
        cursor += sizes[field]
    return offsets


# One scalar per canonical arm+hand joint.  Hand entries become token features;
# arm entries become global context.
JOINT_WIDTH_FIELDS: tuple[str, ...] = (
    "joint_pos",
    "joint_vel",
    "prev_joint_pos",
    "prev_joint_vel",
    "prev_action_targets",
)

# Explicit hand-only fields and their per-token stride.
HAND_TOKEN_FIELDS: dict[str, int] = {
    "joint_link_bbox": JOINT_BOX_DIM,
    "joint_lower": 1,
    "joint_upper": 1,
    "joint_enabled": 1,
    "object_keypoints_rel_joint": 3 * NUM_KEYPOINTS,
}


def build_token_layout(spec, field_list) -> dict:
    """Describe exact gathers from a flat observation into hand-joint tokens."""
    offsets = field_offsets(field_list, spec)
    n_arm = spec.num_arm_joints
    n_hand = spec.num_hand_joints
    missing = [f for f in (*JOINT_WIDTH_FIELDS, *HAND_TOKEN_FIELDS) if f not in offsets]
    if missing:
        raise KeyError(
            "joint_transformer requires explicit token fields; missing "
            f"{missing} from {list(field_list)}"
        )

    token_columns: list[list[int]] = [[] for _ in range(n_hand)]
    global_slices: list[list[int]] = []

    for field in JOINT_WIDTH_FIELDS:
        start, _ = offsets[field]
        if n_arm:
            global_slices.append([start, start + n_arm])
        for joint in range(n_hand):
            token_columns[joint].append(start + n_arm + joint)

    for field, stride in HAND_TOKEN_FIELDS.items():
        start, end = offsets[field]
        if end - start != stride * n_hand:
            raise ValueError(
                f"{field} is {end - start} wide, expected {stride} x {n_hand}"
            )
        for joint in range(n_hand):
            token_columns[joint].extend(
                range(start + joint * stride, start + (joint + 1) * stride)
            )

    token_only = set(JOINT_WIDTH_FIELDS) | set(HAND_TOKEN_FIELDS)
    for field in field_list:
        if field in token_only:
            continue
        start, end = offsets[field]
        global_slices.append([start, end])

    widths = {len(row) for row in token_columns}
    if len(widths) != 1:
        raise RuntimeError(f"ragged token columns: {sorted(widths)}")
    token_dim = widths.pop() if widths else 0

    return {
        "obs_dim": compute_obs_dim(field_list, spec),
        "n_arm": n_arm,
        "n_hand": n_hand,
        "n_joints": spec.num_joints,
        "hand_joint_names": list(spec.hand_joint_names),
        "token_columns": token_columns,
        "token_dim": token_dim,
        "global_slices": global_slices,
        "global_dim": sum(end - start for start, end in global_slices),
    }


__all__ = [
    "HAND_TOKEN_FIELDS",
    "JOINT_BOX_DIM",
    "JOINT_BOX_POINTS",
    "JOINT_WIDTH_FIELDS",
    "NUM_KEYPOINTS",
    "build_token_layout",
    "compute_obs_dim",
    "field_offsets",
    "obs_field_sizes",
]
