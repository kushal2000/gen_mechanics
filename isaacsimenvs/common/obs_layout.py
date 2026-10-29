"""Kit-free flat-observation and per-joint-token layouts.

The environment emits every token feature explicitly.  The transformer only
gathers columns from the same flat vector consumed by the MLP; it neither parses
joint names nor reaches into a URDF to invent architecture-only features.
"""

from __future__ import annotations


NUM_KEYPOINTS: int = 4
JOINT_BOX_POINTS: int = 4
JOINT_BOX_DIM: int = 3 * JOINT_BOX_POINTS


def obs_field_sizes(spec, extra_sizes: dict[str, int] | None = None) -> dict[str, int]:
    """Per-field widths for ``spec``; ``extra_sizes`` adds global fields another env emits (e.g. in-hand
    olympics' task_onehot)."""
    n_joints = spec.num_joints
    n_hand = spec.num_hand_joints
    n_tips = spec.num_fingertips
    return {
        # Current and one-step-old proprioception, in canonical arm+hand order.
        "joint_pos": n_joints,
        "joint_vel": n_joints,
        "prev_joint_pos": n_joints,
        "prev_joint_vel": n_joints,
        "prev_action_targets": n_joints,
        # Explicit hand-token data.  These fields never include the arm.
        "joint_link_bbox": JOINT_BOX_DIM * n_hand,
        # The joint's own position (its child body's origin), palm frame, from the palm centre. Needed with the
        # "extent" link boxes, which no longer start at the joint (inhand_olympics ObsCfg.joint_box).
        "joint_origin": 3 * n_hand,
        "joint_lower": n_hand,
        "joint_upper": n_hand,
        "joint_enabled": n_hand,
        # 1 on a joint behind the hand's palm cut (its finger moves with the palm joint), else 0; static per hand.
        "joint_moving": n_hand,
        "object_keypoints_rel_joint": 3 * NUM_KEYPOINTS * n_hand,
        "hand_scale": 1,
        # The palm slab as four points in the end-effector frame; static per design.
        "palm_keypoints": 3 * JOINT_BOX_POINTS,
        # The slab's thickness, width and length, metres; static per hand.
        "palm_extents": 3,
        # Global task state. palm_* is measured at the design's palm centre,
        # ee_* at link_7's own origin; same body, different point.
        "palm_pos": 3,
        "palm_rot": 4,
        "palm_vel": 6,
        "ee_pos": 3,
        "ee_rot": 4,
        "ee_vel": 6,
        "keypoints_rel_ee": 3 * NUM_KEYPOINTS,
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
        "fingertip_pos_rel_palm": 3 * n_tips,
        "fingertip_pos_rel_ee": 3 * n_tips,
        **(extra_sizes or {}),
    }


def compute_obs_dim(field_list, spec, extra_sizes: dict[str, int] | None = None) -> int:
    sizes = obs_field_sizes(spec, extra_sizes)
    unknown = [f for f in field_list if f not in sizes]
    if unknown:
        raise KeyError(f"unknown observation field(s) {unknown}; valid: {sorted(sizes)}")
    return sum(sizes[f] for f in field_list)


def field_offsets(field_list, spec, extra_sizes: dict[str, int] | None = None) -> dict[str, tuple[int, int]]:
    sizes = obs_field_sizes(spec, extra_sizes)
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
    "joint_origin": 3,
    "joint_lower": 1,
    "joint_upper": 1,
    "joint_enabled": 1,
    "joint_moving": 1,
    "object_keypoints_rel_joint": 3 * NUM_KEYPOINTS,
}

# Token fields a policy may leave out of its observation list; every other token field is
# required, so an accidental omission still raises. Without object_keypoints_rel_joint a joint
# token carries only the joint's own state and geometry, and the object reaches it through the
# global token by attention.
OPTIONAL_HAND_TOKEN_FIELDS: frozenset[str] = frozenset({"object_keypoints_rel_joint", "joint_origin", "joint_moving"})

# Gathered per joint but never a token feature: only the attention mask reads it, raw.
MASK_FIELD = "joint_enabled"


def build_token_layout(spec, field_list, extra_sizes: dict[str, int] | None = None) -> dict:
    """Describe exact gathers from a flat observation into hand-joint tokens.

    ``joint_enabled`` is NOT a token feature: it is gathered separately
    (``enabled_columns``) and used only to build the attention mask, raw. A per-joint
    existence flag carries no information the policy should act on, and a column that is a
    constant 1 on a fixed hand is exactly what an input normaliser turns into noise.
    """
    offsets = field_offsets(field_list, spec, extra_sizes)
    n_arm = spec.num_arm_joints
    n_hand = spec.num_hand_joints
    missing = [f for f in (*JOINT_WIDTH_FIELDS, *HAND_TOKEN_FIELDS)
               if f not in offsets and f not in OPTIONAL_HAND_TOKEN_FIELDS]
    if missing:
        raise KeyError(
            "joint_transformer requires explicit token fields; missing "
            f"{missing} from {list(field_list)}"
        )
    token_fields = {f: w for f, w in HAND_TOKEN_FIELDS.items()
                    if f in offsets and f != MASK_FIELD}

    token_columns: list[list[int]] = [[] for _ in range(n_hand)]
    global_slices: list[list[int]] = []
    # Where each field sits inside a token, so the network can treat fields differently (e.g. keep the
    # physical-unit joint fields out of the running normaliser).
    token_field_slices: dict[str, list[int]] = {}
    _pos = 0

    for field in JOINT_WIDTH_FIELDS:
        start, _ = offsets[field]
        if n_arm:
            global_slices.append([start, start + n_arm])
        for joint in range(n_hand):
            token_columns[joint].append(start + n_arm + joint)
        token_field_slices[field] = [_pos, _pos + 1]
        _pos += 1

    for field, stride in token_fields.items():
        start, end = offsets[field]
        if end - start != stride * n_hand:
            raise ValueError(
                f"{field} is {end - start} wide, expected {stride} x {n_hand}"
            )
        for joint in range(n_hand):
            token_columns[joint].extend(
                range(start + joint * stride, start + (joint + 1) * stride)
            )
        token_field_slices[field] = [_pos, _pos + stride]
        _pos += stride

    start, end = offsets[MASK_FIELD]
    if end - start != n_hand:
        raise ValueError(f"{MASK_FIELD} is {end - start} wide, expected {n_hand}")
    enabled_columns = list(range(start, end))

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
        "obs_dim": compute_obs_dim(field_list, spec, extra_sizes),
        "enabled_columns": enabled_columns,
        "n_arm": n_arm,
        "n_hand": n_hand,
        "n_joints": spec.num_joints,
        "hand_joint_names": list(spec.hand_joint_names),
        "token_columns": token_columns,
        "token_dim": token_dim,
        "token_field_slices": token_field_slices,
        "global_slices": global_slices,
        "global_dim": sum(end - start for start, end in global_slices),
    }


def build_finger_layout(spec, field_list, finger_layout, extra_sizes: dict[str, int] | None = None) -> dict:
    """``build_token_layout`` plus the static maps from its joint tokens onto the finger-major slot grid of
    robots/finger_layout.py (``finger_layout``: the spec's FingerLayout, from ``robots.get_finger_layout``):

      slot_gather    (N_HAND_SLOTS,) the hand joint behind each slot, or ``n_hand`` for a ghost slot (the network
                     appends one all-zero joint row, so the same index_select fills ghosts)
      slot_of_joint  (n_hand,) each hand joint's slot: the network's per-slot actions back to canonical order
      finger_dim     n_slots x (token_dim + 1): a finger's joint tokens palm -> tip, then its n_slots validity bits
    (``n_fingers`` and ``n_slots`` from the layout's grid: 6 x 5 by default, a ``multi:template:`` set's own grid.)
    """
    layout = build_token_layout(spec, field_list, extra_sizes)
    if finger_layout.n_joints != layout["n_hand"]:
        raise ValueError(f"finger layout covers {finger_layout.n_joints} hand joints, the observation {layout['n_hand']}")
    grid = finger_layout.grid
    layout["slot_gather"] = [j if j >= 0 else layout["n_hand"] for j in finger_layout.joint_of_slot]
    layout["slot_of_joint"] = list(finger_layout.slot_of_joint)
    layout["n_fingers"], layout["n_slots"] = grid.n_fingers, grid.n_slots
    layout["finger_dim"] = grid.n_slots * (layout["token_dim"] + 1)
    return layout


__all__ = [
    "HAND_TOKEN_FIELDS",
    "OPTIONAL_HAND_TOKEN_FIELDS",
    "JOINT_BOX_DIM",
    "JOINT_BOX_POINTS",
    "JOINT_WIDTH_FIELDS",
    "MASK_FIELD",
    "NUM_KEYPOINTS",
    "build_finger_layout",
    "build_token_layout",
    "compute_obs_dim",
    "field_offsets",
    "obs_field_sizes",
]
