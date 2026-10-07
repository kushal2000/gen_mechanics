"""Several hands in ONE scene: each on its own block of envs, its own articulation, its real topology.

Selected by ``robot_spec = "multi:<spec>+<spec>+..."`` (registered RobotSpec names; "+", because hydra
reads a comma in an override as a sweep), or the shorthand
``"multi:uniform"`` for every uniform-dynamics commercial hand, or ``"multi:wuji2_lr_2plus"`` for every Wuji v2
with two or more fingers, left and right (2 x 26 = 52: the full hand, 5 no_<finger>, 10 three- and 10
two-finger only_<...>, from make_missing_fingers.py --all). Kit-free, so the network resolves the same
layout as the env (``get_robot_spec`` returns :attr:`HandSet.template`).

Nothing is padded in the SIMULATOR -- every hand keeps its own links and joints (a padded topology made
the sim much slower). The padding lives only in torch: the policy sees ``J = max joints`` slots and
``F = max fingertips`` tips, and a hand with fewer leaves the rest as ghosts:

  joint slot k < n_h     the hand's k-th joint in its spec's canonical order; a ghost slot has zero range,
                         so ``joint_enabled`` (read from the limits) masks it in attention, exactly as for a
                         generated population
  tip slot i < F_h       the hand's i-th fingertip; ``fingertip_valid`` masks the rest
  per-env tables         joint-link boxes, palm keypoints, palm centre, fingertip pad offsets, hand scale --
                         gathered per env by :meth:`HandSet.per_env`, the interface ``allocate_state_buffers``
                         already reads a population through

The slot a joint lands in carries no meaning across hands: the joint transformer has no positional
embedding, so tokens are a set. Envs are dealt to hands ROUND-ROBIN (env i -> hand i mod H): SAPG gives
each contiguous block of envs its own exploration coefficient, and contiguous hand blocks would tie a hand's
identity to how much it explores.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from hand_sampler.robot_spec import RobotSpec

MULTI_PREFIX = "multi:"
UNIFORM_HANDS = ("sharpa", "allegro", "leap", "shadow", "dex3", "tesollo", "wuji2", "xhand")
PALM = "palm"
WUJI2_FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _subset_tags(fingers, min_fingers: int) -> list[str]:
    """make_missing_fingers.make_all's tags, in its order: no_<f> for one short, only_<a>_<b>... below."""
    import itertools
    tags = []
    for k in range(len(fingers) - 1, min_fingers - 1, -1):
        for keep in itertools.combinations(fingers, k):
            tags.append("no_" + next(f for f in fingers if f not in keep) if k == len(fingers) - 1
                        else "only_" + "_".join(keep))
    return tags


WUJI2_LR_2PLUS = tuple(f"wuji2_{side}_uniform_handonly{t}" for side in ("left", "right")
                       for t in ["", *(f"_{g}" for g in _subset_tags(WUJI2_FINGERS, 2))])


def is_multi_ref(ref) -> bool:
    return isinstance(ref, str) and ref.startswith(MULTI_PREFIX)


def spec_names(ref: str) -> tuple[str, ...]:
    body = ref[len(MULTI_PREFIX):]
    if body == "uniform":
        return tuple(f"{h}_left_uniform_handonly" for h in UNIFORM_HANDS)
    if body == "wuji2_lr_2plus":
        return WUJI2_LR_2PLUS
    names = tuple(n.strip() for n in body.split("+") if n.strip())
    if len(names) < 1 or len(set(names)) != len(names):
        raise ValueError(f"{ref!r}: need one or more distinct robot specs")
    return names


def slot_joint(k: int) -> str:
    return f"s{k}"


def slot_tip(i: int) -> str:
    return f"tip{i}"


def slot_link(k: int) -> str:
    return f"link_s{k}"


@dataclass(frozen=True)
class HandSet:
    ref: str
    specs: tuple[RobotSpec, ...]
    template: RobotSpec
    n_joints: int            # J, slots
    n_tips: int              # F, slots
    # (H, ...) tables, one row per hand, slot-padded
    joint_link_boxes: np.ndarray       # (H, J, 4, 3)
    joint_geometry_valid: np.ndarray   # (H, J) bool -- a real joint whose link box is usable
    joint_valid: np.ndarray            # (H, J) bool -- a real joint at all
    fingertip_valid: np.ndarray        # (H, F) bool
    fingertip_offsets: np.ndarray      # (H, F, 3)
    palm_center_offset: np.ndarray     # (H, 3)
    palm_keypoints: np.ndarray         # (H, 4, 3)
    hand_scale: np.ndarray             # (H,)

    @property
    def n_hands(self) -> int:
        return len(self.specs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.specs)

    def hand_of_env(self, num_envs: int) -> np.ndarray:
        """(N,) hand index per env, round-robin: every SAPG block holds every hand equally."""
        if num_envs < self.n_hands:
            raise ValueError(f"{num_envs} envs cannot hold {self.n_hands} hands")
        return np.arange(num_envs) % self.n_hands

    def per_env(self, hand_idx: np.ndarray) -> dict:
        """The population-style per-env tables (allocate_state_buffers) plus the pad offsets."""
        i = np.asarray(hand_idx, dtype=np.int64)
        return {
            "joint_link_bbox_local": self.joint_link_boxes[i].astype(np.float32),
            "joint_geometry_valid": self.joint_geometry_valid[i],
            "hand_scale": self.hand_scale[i].astype(np.float32)[:, None],
            "fingertip_valid": self.fingertip_valid[i],
            "palm_center_offset": self.palm_center_offset[i].astype(np.float32),
            "palm_keypoints": self.palm_keypoints[i].astype(np.float32),
            "fingertip_offsets": self.fingertip_offsets[i].astype(np.float32),
            "joint_valid": self.joint_valid[i],
        }


@functools.lru_cache(maxsize=None)
def hand_set(ref: str) -> HandSet:
    from . import REGISTRY                       # the per-hand specs, by name
    missing = [n for n in spec_names(ref) if n not in REGISTRY]
    if missing:
        raise KeyError(f"{ref!r}: unknown robot specs {missing}")
    specs = tuple(REGISTRY[n] for n in spec_names(ref))
    for s in specs:
        if s.num_arm_joints:
            raise ValueError(f"{ref!r}: {s.name} has an arm; a multi-hand scene is hand-only")
    H = len(specs)
    J = max(s.num_hand_joints for s in specs)
    F = max(s.num_fingertips for s in specs)
    boxes = np.zeros((H, J, 4, 3), np.float32)
    geom_valid = np.zeros((H, J), bool)
    joint_valid = np.zeros((H, J), bool)
    tip_valid = np.zeros((H, F), bool)
    tip_off = np.zeros((H, F, 3), np.float32)
    for h, s in enumerate(specs):
        n = s.num_hand_joints
        boxes[h, :n] = np.asarray(s.joint_link_boxes, np.float32)
        geom_valid[h, :n] = np.asarray(s.joint_geometry_valid, bool)
        joint_valid[h, :n] = True
        tip_valid[h, :s.num_fingertips] = True
        if s.fingertip_offsets:
            tip_off[h, :s.num_fingertips] = np.asarray(s.fingertip_offsets, np.float32)
    joints = tuple(slot_joint(k) for k in range(J))
    s0 = specs[0]
    template = RobotSpec(
        name=ref, arm_name="", hand_name="multi", urdf_path="",
        arm_joint_names=(), hand_joint_names=joints,
        palm_body_name=PALM, fingertip_body_names=tuple(slot_tip(i) for i in range(F)),
        arm_stiffness={}, arm_damping={},
        # Per-hand articulations take their own gains; these only satisfy the spec's tables.
        hand_stiffness={j: float(next(iter(s0.hand_stiffness.values()))) for j in joints},
        hand_damping={j: float(next(iter(s0.hand_damping.values()))) for j in joints},
        hand_armature={j: float(next(iter(s0.hand_armature.values()))) for j in joints},
        arm_default_joint_pos={}, hand_default_joint_pos={j: 0.0 for j in joints},
        start_arm_higher_deltas={},
        palm_center_offset=tuple(float(v) for v in s0.palm_center_offset),
        # Each hand's USD carries its own filters; the template's map only satisfies validation.
        adjacent_links={PALM: [slot_link(0)]},
        link_prim_regexes=("/World/envs/env_.*/Robot/.*/visuals",),
        joint_link_bodies=tuple(slot_link(k) for k in range(J)),
        joint_link_boxes=tuple(tuple(tuple(map(float, p)) for p in box) for box in np.zeros((J, 4, 3))),
        joint_geometry_valid=tuple(True for _ in range(J)),
        hand_scale=float(max(s.hand_scale for s in specs)),
        palm_keypoints=tuple(tuple(map(float, p)) for p in s0.palm_keypoints),
        fingertip_offsets=tuple((0.0, 0.0, 0.0) for _ in range(F)),
        notes=f"multi-hand template over {', '.join(s.name for s in specs)}; per-env tables in HandSet.",
    )
    return HandSet(
        ref=ref, specs=specs, template=template, n_joints=J, n_tips=F,
        joint_link_boxes=boxes, joint_geometry_valid=geom_valid, joint_valid=joint_valid,
        fingertip_valid=tip_valid, fingertip_offsets=tip_off,
        palm_center_offset=np.asarray([s.palm_center_offset for s in specs], np.float32),
        palm_keypoints=np.asarray([s.palm_keypoints for s in specs], np.float32),
        hand_scale=np.asarray([s.hand_scale for s in specs], np.float32),
    )


__all__ = ["HandSet", "MULTI_PREFIX", "PALM", "UNIFORM_HANDS", "hand_set", "is_multi_ref",
           "slot_joint", "slot_link", "slot_tip", "spec_names"]
