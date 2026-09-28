"""Wonik Allegro, right hand, 16 DoF, mounted with no arm.

The first commercial hand in the design space. Everything here is measured from
``assets/urdf/unified_commercial_hands/allegro/allegro.urdf`` or transcribed from a
named source; nothing is chosen to make a number look right.

WHY THERE IS NO ARM AND NO ``handonly:`` PREFIX. The prefix exists to turn a
generated POPULATION hand-only, because the procedural path grafts a hand onto
``ARM_TIP_LINK`` and needs a flag to know the arm is a stub. A fixed spec has no such
indirection: it declares ``arm_joint_names=()``, names its own palm body, and
``assembly._convert_fixed_robot`` converts its URDF directly. Asking for
``handonly:allegro_handonly`` would in fact raise ``KeyError`` --
``is_population_ref`` only recognises ``gen_s<seed>_n<size>`` and ``.json``, so the
prefixed string falls through to the registry where it is not a key.

WHY THE PALM BODY IS ``palm_link`` AND NOT ``iiwa14_link_7``. Every other spec's
palm merges into the arm's flange under ``merge_fixed_joints``. The unified URDF is
rooted at the palm itself and its only joints are the 16 revolute ones, so nothing
collapses and the palm is a body in its own right. That also means this spec never
touches the arm stub, ``build.author_hand`` or ``flange_to_palm``.
"""

from __future__ import annotations

import numpy as np

from hand_sampler.build import palm_keypoints
from hand_sampler.design_space import joint_link_boxes
from hand_sampler.robot_param_constants import hand_only_base_pos, hand_only_base_rot
from hand_sampler.robot_spec import RobotSpec, Vec3
from isaacsimenvs.pose_reaching_6d.scene_utils.robots.adjacency.allegro_handonly import (
    ALLEGRO_HANDONLY_ADJACENT_LINKS,
)

URDF = "assets/urdf/unified_commercial_hands/allegro/allegro.urdf"
PALM_BODY = "palm_link"
FINGERS = ("index", "middle", "ring", "thumb")

# Proximal to distal within each finger, fingers in the vendor's own order. Nothing
# forces this order for a hand trained from scratch -- unlike SHARPA, whose infixes
# are pinned by a pretrained checkpoint's action layout -- but it is the order the
# URDF declares, so a reader comparing the two files sees the same sequence.
HAND_JOINT_NAMES: tuple[str, ...] = tuple(
    f"{finger}_joint_{i}" for finger in FINGERS for i in range(4)
)

# The distal phalanx, which carries the biotac mesh. NOT *_biotac_tip: those links
# are massful frames with no geometry at all, and the unified URDF drops them (the
# vendor file had lost index_biotac_tip's joint entirely).
FINGERTIP_BODY_NAMES: tuple[str, ...] = tuple(f"{finger}_link_3" for finger in FINGERS)

# --- actuation ----------------------------------------------------------------
# Isaac Lab's own Allegro configuration: ALLEGRO_HAND_CFG in isaaclab_assets gives
# stiffness 3.0, damping 0.1, effort_limit_sim 0.5, friction 0.01. Armature is not
# in that config; 0.001 is authored, and sits inside SHARPA's measured 0.00042 to
# 0.0032 range. The URDF's own <dynamics> is asymmetric nonsense -- only the index
# chain declares damping -- so it is overridden uniformly here rather than imported.
HAND_STIFFNESS = {n: 3.0 for n in HAND_JOINT_NAMES}
HAND_DAMPING = {n: 0.1 for n in HAND_JOINT_NAMES}
HAND_ARMATURE = {n: 0.001 for n in HAND_JOINT_NAMES}

# --- home pose ----------------------------------------------------------------
# Open, EXCEPT the thumb. thumb_joint_0's limit is [+0.2792, +1.5708], so q = 0 is
# outside the range: a zero home pose starts the joint in violation and PhysX pushes
# it out, which looks like the thumb twitching on reset. The lower limit is the
# open-most legal value. IsaacLab's inhand config sets the same joint to 0.28 for
# what is presumably the same reason.
# 0.28, not the limit itself. The URDF's lower bound is 0.279244444444, and sitting
# exactly on a limit is a degenerate place to start -- writing it out to fewer digits
# lands 4e-10 rad OUTSIDE the bound, which is how a "safe" literal becomes a violation.
# IsaacLab's inhand config uses 0.28 for the same joint, 0.76 mrad clear of the bound.
# reset.py:357 clamps the reset pose into the limits anyway, so this is belt and
# braces rather than the only guard.
THUMB_ROLL_HOME_RAD = 0.28
HAND_DEFAULT_JOINT_POS = {
    n: (THUMB_ROLL_HOME_RAD if n == "thumb_joint_0" else 0.0) for n in HAND_JOINT_NAMES
}

# --- palm geometry, measured off base_link.obj --------------------------------
# The MAIN slab only. The mesh also contains a stray 88 x 117 x 3 mm plate floating
# 54.5 mm behind the palm; unify_from_vendor.py excludes it and replaces the collider
# with this box, because a convex hull over both is 98 mm thick instead of 40.8.
PALM_EXTENTS_PALMLINK_M: Vec3 = (0.100200, 0.117288, 0.040800)   # along palm_link x,y,z
PALM_CENTRE_PALMLINK_M: Vec3 = (-0.007763, 0.008777, -0.020400)

# palm_link is +x out the fingers, +y index->ring, +z out of the palm toward the
# object -- measured from the finger roots (which differ only in y, at 45.1 mm pitch),
# from the slab being thin in z with all material at z <= 0, and from flexion carrying
# the fingertips into +z.
#
# Our palm convention is x = thickness and grasp normal, y = width, z = fingers. So
# ours <- palm_link is (x,y,z) <- (z,y,x), whose determinant is -1: a MIRROR, because
# this convention was built around SHARPA, a left hand. Flipping the width axis gives
# (z,-y,x) and determinant +1, so the width runs ring->index instead of index->ring.
# Nothing observes or rewards the width's sign, and the alternative -- the mirrored
# meshes the integration deleted in e4eb91a carried -- discards the vendor geometry
# that is the whole point of importing a real hand.
PALM_FRAME = ((0.0, 0.0, 1.0),
              (0.0, -1.0, 0.0),
              (1.0, 0.0, 0.0))

_F = np.asarray(PALM_FRAME, float)
# The same box, re-expressed in our convention, for palm_keypoints.
_EXT_OURS = (PALM_EXTENTS_PALMLINK_M[2], PALM_EXTENTS_PALMLINK_M[1], PALM_EXTENTS_PALMLINK_M[0])
_CENTRE_OURS = tuple(_F.T @ np.asarray(PALM_CENTRE_PALMLINK_M, float))

# The palm centre as a VECTOR in palm_link's frame, so hand_only_base_pos places it
# exactly. Passing a scalar reach instead -- which is all a generated population can
# offer, its palm centre varying per design -- assumes the offset runs along the root's
# +z, and put this palm 42 mm off target: Allegro's offset is 11.7 mm lateral and
# points INTO the slab, not out of it.

# Read once, here, rather than at every run start: the env takes its tokens from the
# spec and never learns that this hand happens to have a URDF.
_BODIES, _BOXES, _VALID, _SCALE = joint_link_boxes(URDF, HAND_JOINT_NAMES)

# The pad centre in each fingertip body's frame. The biotac's tactile surface faces
# the palm: link_3's +y maps to palm +z, and the vendor's own (dropped) tip frames sat
# at (0.055, 0.015, 0) for the fingers and (0.070, 0.010, 0) for the thumb -- a point
# just off the +y face, at the centre of the pad rather than at the mesh apex. Those
# offsets are kept, since they are the vendor's statement of where the pad is.
_FINGER_PAD: Vec3 = (0.055, 0.015, 0.0)
_THUMB_PAD: Vec3 = (0.070, 0.010, 0.0)
FINGERTIP_OFFSETS = tuple(
    _THUMB_PAD if finger == "thumb" else _FINGER_PAD for finger in FINGERS
)

ALLEGRO_HANDONLY = RobotSpec(
    name="allegro_handonly",
    arm_name="",
    hand_name="allegro",
    urdf_path=URDF,
    arm_joint_names=(),
    hand_joint_names=HAND_JOINT_NAMES,
    palm_body_name=PALM_BODY,
    fingertip_body_names=FINGERTIP_BODY_NAMES,
    arm_stiffness={},
    arm_damping={},
    hand_stiffness=HAND_STIFFNESS,
    hand_damping=HAND_DAMPING,
    hand_armature=HAND_ARMATURE,
    fingertip_offsets=FINGERTIP_OFFSETS,
    joint_link_bodies=tuple(_BODIES),
    joint_link_boxes=_BOXES,
    joint_geometry_valid=_VALID,
    hand_scale=float(_SCALE),
    arm_default_joint_pos={},
    hand_default_joint_pos=HAND_DEFAULT_JOINT_POS,
    start_arm_higher_deltas={},
    palm_center_offset=tuple(float(v) for v in PALM_CENTRE_PALMLINK_M),
    palm_keypoints=tuple(
        tuple(map(float, p)) for p in palm_keypoints(_CENTRE_OURS, _EXT_OURS, frame=_F)
    ),
    adjacent_links=ALLEGRO_HANDONLY_ADJACENT_LINKS,
    link_prim_regexes=("/World/envs/env_.*/Robot/.*/visuals",),
    base_pos=hand_only_base_pos(PALM_FRAME, PALM_CENTRE_PALMLINK_M),
    base_rot=hand_only_base_rot(PALM_FRAME),
    notes=(
        "Wonik Allegro right hand, 16 DoF, hand-only. Geometry measured from "
        "assets/urdf/unified_commercial_hands/allegro/allegro.urdf, which is generated "
        "by that folder's unify_from_vendor.py from the Isaac Gym vendor description "
        "(simtoolreal/isaacgym/assets/urdf/kuka_allegro_description) plus the inertials "
        "recovered from this repo at e4eb91a^. Gains from Isaac Lab's ALLEGRO_HAND_CFG; "
        "armature authored. Pad offsets are the vendor's own biotac tip frames. The palm "
        "frame adapts the vendor's axes to ours by flipping the width, which avoids "
        "mirroring the meshes."
    ),
)

__all__ = ["ALLEGRO_HANDONLY", "HAND_JOINT_NAMES", "PALM_FRAME", "URDF"]
