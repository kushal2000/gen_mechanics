"""Commercial hands onboarded from their vendor URDFs: one RobotSpec per <hand>_<side>.spec.json.

Every value comes from ``assets/urdf/unified_commercial_hands/onboard.py``, which measures it from the
URDF and writes a report (``<hand>_<side>.report.md``) with the checks it ran. This module only turns the
JSON into a RobotSpec -- the same fields ``allegro_handonly.py`` fills in by hand, derived the same way:
the palm keypoints from the palm box in our convention, the mount from the palm frame and centre, the
joint tokens from ``joint_link_boxes`` on the unified URDF. Registered as ``<hand>_<side>_handonly``.

Validated: built from ``allegro/allegro_right.spec.json`` this reproduces the hand-written
ALLEGRO_HANDONLY's palm frame, palm box, mount pose and palm keypoints (tests/test_unified_hands.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hand_sampler.build import palm_keypoints
from hand_sampler.design_space import joint_link_boxes
from hand_sampler.robot_param_constants import hand_only_base_pos, hand_only_base_rot
from hand_sampler.robot_spec import RobotSpec

REPO = Path(__file__).resolve().parents[4]
HANDS_DIR = REPO / "assets/urdf/unified_commercial_hands"
# Onboarded and accepted for training. (allegro right stays the hand-written allegro_handonly.)
LEFT_HANDS = ("allegro", "leap", "shadow", "dex3", "tesollo", "wuji2", "xhand")


def spec_from_json(path: Path, name: str | None = None) -> RobotSpec:
    s = json.loads(Path(path).read_text())
    joints = tuple(s["hand_joint_names"])
    F = np.asarray(s["palm_frame"], float)
    F_t = tuple(tuple(float(v) for v in row) for row in F)     # hashable: the placement helpers are cached
    centre_palm = tuple(float(v) for v in s["palm_centre_in_palm_body"])
    bodies, boxes, valid, scale = joint_link_boxes(s["urdf"] if Path(s["urdf"]).is_absolute()
                                                   else str(REPO / s["urdf"]), joints)
    return RobotSpec(
        name=name or f"{s['hand']}_{s['side']}_handonly",
        arm_name="",
        hand_name=s["hand"],
        urdf_path=s["urdf"],
        arm_joint_names=(),
        hand_joint_names=joints,
        palm_body_name=s["palm_body"],
        fingertip_body_names=tuple(s["fingertip_body_names"]),
        arm_stiffness={}, arm_damping={},
        hand_stiffness={n: float(s["stiffness"]) for n in joints},
        hand_damping={n: float(s["damping"]) for n in joints},
        hand_armature={n: float(s["armature"]) for n in joints},
        fingertip_offsets=tuple(tuple(map(float, p)) for p in s["fingertip_offsets"]),
        joint_link_bodies=tuple(bodies),
        joint_link_boxes=boxes,
        joint_geometry_valid=valid,
        hand_scale=float(scale),
        arm_default_joint_pos={},
        hand_default_joint_pos={n: float(v) for n, v in s["hand_default_joint_pos"].items()},
        start_arm_higher_deltas={},
        palm_center_offset=centre_palm,
        palm_keypoints=tuple(tuple(map(float, p)) for p in
                             palm_keypoints(tuple(s["palm_centre_ours"]), tuple(s["palm_extents_ours"]), frame=F)),
        adjacent_links={k: list(v) for k, v in s["adjacent_links"].items()},
        link_prim_regexes=("/World/envs/env_.*/Robot/.*/visuals",),
        base_pos=hand_only_base_pos(F_t, centre_palm),
        base_rot=hand_only_base_rot(F_t),
        notes=(f"{s['hand']} {s['side']} hand, {len(joints)} DoF, hand-only; onboarded by "
               f"assets/urdf/unified_commercial_hands/onboard.py -- see {s['hand']}_{s['side']}.report.md and "
               f"vendor_{s['side']}/SOURCE.md. Gains uniform (stiffness {s['stiffness']}, damping {s['damping']})."),
    )


UNIFIED_LEFT = {h: spec_from_json(HANDS_DIR / h / f"{h}_left.spec.json") for h in LEFT_HANDS}

__all__ = ["UNIFIED_LEFT", "spec_from_json"]
