"""RobotSpec variants for the finger-dropout eval, and checkpoint remapping.

A variant is SHARPA_IIWA14 with one finger's joints, fingertip and actuator
entries filtered out, pointing at the matching URDF from
make_finger_variants.py. Nothing is renamed and no joint is reordered, so the
retained joints keep the names, order, limits and geometry the checkpoint was
trained with.

WHY THE POLICY TRANSFERS UNCHANGED. Every weight in JointTransformerNet is
independent of the hand joint count:

    token_proj   Linear(32 -> d)          token_dim is 32 for any hand
    global_proj  Linear(74 -> d)          global_dim is arm + task only
    encoder      shared, attention        length-agnostic
    mu_head      Linear(d -> 1), shared   applied per token
    arm_head     Linear(d+74 -> 7)        arm is untouched
    value_head   Linear(2d+74 -> 1)       mean-pools over tokens

Only ``sigma`` is (actions_num,) or (blocks, actions_num), and the observation
normalizer is (obs_dim,). Both are remapped here by canonical joint index --
sigma's VALUES are irrelevant under deterministic evaluation, but its SHAPE
still has to satisfy a strict load_state_dict, and running_mean_std's values
very much do matter.
"""

from __future__ import annotations

import dataclasses
import pathlib

from isaacsimenvs.pose_reaching_6d.scene_utils.robots import (
    REGISTRY, SHARPA_IIWA14,
)

ASSETS = "experiments/decentralized_control/eval/assets"

# Which fingertip body each finger owns. fingertip_body_names is ordered
# [index, middle, ring, thumb, pinky] while hand_joint_names is thumb-first,
# so these two orderings do NOT line up -- dropping the thumb removes
# fingertip slot 3, not slot 0.
FINGER_TIP = {
    "thumb": "left_thumb_DP",
    "index": "left_index_DP",
    "middle": "left_middle_DP",
    "ring": "left_ring_DP",
    "pinky": "left_pinky_DP",
}
FINGERS = tuple(FINGER_TIP)


def _finger_joints(finger: str) -> tuple[str, ...]:
    """The hand joints belonging to one finger, in spec order."""
    return tuple(j for j in SHARPA_IIWA14.hand_joint_names if finger in j)


def make_variant(finger: str):
    """SHARPA_IIWA14 with ``finger`` removed, matching sharpa_no_<finger>.urdf."""
    if finger not in FINGER_TIP:
        raise KeyError(f"unknown finger {finger!r}; have {FINGERS}")
    drop_joints = set(_finger_joints(finger))
    drop_tip = FINGER_TIP[finger]

    keep_joints = tuple(j for j in SHARPA_IIWA14.hand_joint_names
                        if j not in drop_joints)
    tips = SHARPA_IIWA14.fingertip_body_names
    keep_tip_idx = [i for i, t in enumerate(tips) if t != drop_tip]

    return dataclasses.replace(
        SHARPA_IIWA14,
        name=f"sharpa_iiwa14_no_{finger}",
        urdf_path=f"{ASSETS}/sharpa_no_{finger}.urdf",
        hand_joint_names=keep_joints,
        fingertip_body_names=tuple(tips[i] for i in keep_tip_idx),
        fingertip_offsets=tuple(
            SHARPA_IIWA14.fingertip_offsets[i] for i in keep_tip_idx),
        # Actuator tables are keyed by joint name; a stale key for a joint the
        # articulation no longer has is a silent no-op at best.
        hand_stiffness={k: v for k, v in SHARPA_IIWA14.hand_stiffness.items()
                        if k not in drop_joints},
        hand_damping={k: v for k, v in SHARPA_IIWA14.hand_damping.items()
                      if k not in drop_joints},
        hand_armature={k: v for k, v in SHARPA_IIWA14.hand_armature.items()
                       if k not in drop_joints},
        hand_default_joint_pos={
            k: v for k, v in SHARPA_IIWA14.hand_default_joint_pos.items()
            if k not in drop_joints},
        # Self-collision filtering is by post-merge body name; entries naming a
        # removed link would not resolve.
        adjacent_links={
            k: [v for v in vs if finger not in v]
            for k, vs in SHARPA_IIWA14.adjacent_links.items()
            if finger not in k},
    )


def intact_variant():
    """SHARPA_IIWA14 pointed at the copied URDF, so the control is not special."""
    return dataclasses.replace(
        SHARPA_IIWA14, name="sharpa_iiwa14_intact",
        urdf_path=f"{ASSETS}/sharpa_intact.urdf")


VARIANTS = {"intact": intact_variant, **{f: (lambda f=f: make_variant(f))
                                         for f in FINGERS}}


def register(name: str):
    """Put a variant in the registry so cfg.assets.robot_spec can name it."""
    spec = VARIANTS[name]()
    REGISTRY[spec.name] = spec
    return spec


# --------------------------------------------------------------------------
# checkpoint remapping
# --------------------------------------------------------------------------

def obs_column_map(full_spec, sub_spec, field_list) -> list[int]:
    """Columns of the FULL observation that the REDUCED observation keeps.

    Field widths that scale with the hand shrink; everything else is copied
    across. Returned in reduced-observation order, so
    ``full_tensor[..., cols]`` is the reduced tensor.
    """
    from isaacsimenvs.pose_reaching_6d.obs_utils.layout import (
        JOINT_WIDTH_FIELDS, HAND_TOKEN_FIELDS, field_offsets, obs_field_sizes,
    )
    full_off = field_offsets(field_list, full_spec)
    full_sizes = obs_field_sizes(full_spec)
    keep_hand = [full_spec.hand_joint_names.index(j)
                 for j in sub_spec.hand_joint_names]
    keep_tip = [full_spec.fingertip_body_names.index(t)
                for t in sub_spec.fingertip_body_names]
    n_arm = full_spec.num_arm_joints

    cols: list[int] = []
    for field in field_list:
        start = full_off[field][0]
        if field in JOINT_WIDTH_FIELDS:           # arm block, then hand joints
            cols += [start + i for i in range(n_arm)]
            cols += [start + n_arm + j for j in keep_hand]
        elif field in HAND_TOKEN_FIELDS:          # stride per hand joint
            stride = HAND_TOKEN_FIELDS[field]
            for j in keep_hand:
                cols += [start + j * stride + k for k in range(stride)]
        elif field == "closest_fingertip_dist":
            cols += [start + i for i in keep_tip]
        elif field == "fingertip_pos_rel_palm":
            cols += [start + 3 * i + k for i in keep_tip for k in range(3)]
        else:
            cols += list(range(start, start + full_sizes[field]))
    return cols


def remap_checkpoint(ckpt: dict, full_spec, sub_spec, obs_list) -> dict:
    """Slice a full-hand checkpoint down to a reduced hand, in place.

    running_mean_std carries the statistics the policy was trained under, so
    its columns are selected rather than reinitialised. sigma is sliced by
    canonical joint index purely so a strict load succeeds.
    """
    import torch

    cols = torch.tensor(obs_column_map(full_spec, sub_spec, obs_list),
                        dtype=torch.long)
    keep_act = ([i for i in range(full_spec.num_arm_joints)]
                + [full_spec.num_arm_joints + full_spec.hand_joint_names.index(j)
                   for j in sub_spec.hand_joint_names])
    keep_act = torch.tensor(keep_act, dtype=torch.long)

    model = ckpt["model"]
    for key, tensor in list(model.items()):
        if not hasattr(tensor, "shape"):
            continue
        if "running_mean_std" in key and tensor.ndim == 1 and \
                tensor.shape[0] == len(obs_column_map(full_spec, full_spec, obs_list)):
            model[key] = tensor[cols].clone()
        elif key.endswith("sigma") and tensor.shape[-1] == full_spec.num_joints:
            model[key] = tensor[..., keep_act].clone()
    return ckpt


__all__ = ["FINGERS", "VARIANTS", "make_variant", "intact_variant", "register",
           "obs_column_map", "remap_checkpoint"]
