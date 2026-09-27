"""Pure-python (no isaaclab) drop-termination geometry.

Kept out of ``reward_utils.py`` for the same reason ``goal_curriculum.py`` is
(see that module's docstring): ``reward_utils.py`` imports
``isaacsimenvs.pose_reaching_6d`` modules that bootstrap Kit on import, so
nothing that has to be testable under plain pytest can live there. This
module imports nothing at all -- it works with any object exposing the usual
torch tensor comparison operators (a real torch tensor in production, or one
built in a CPU test; torch itself needs no Kit, only ``isaaclab.*`` does).
"""

from __future__ import annotations

__all__ = ["object_below_palm"]


def object_below_palm(obj_pos_w_z, palm_pos_w_z, drop_distance_m: float):
    """True where the object's WORLD z has fallen more than half the drop
    distance below the palm's own WORLD z.

    Uses WORLD z deliberately, not the palm's own local/body frame. Every
    hand's ``base_rot`` (``hand_calibration.json``, ``calibrate_palm_up.py``)
    rotates it so ITS OWN chosen "up" axis maps to WORLD +z -- the palm
    BODY's local frame axes stay whatever the source URDF originally
    declared (e.g. local +x is "up" for a hand calibrated on that axis), so
    a hand-LOCAL z-coordinate (``env._obj_pos_palm[:, 2]``, as this check
    used before a 2026-09-27 coordinator review) is only "how far above the
    palm" for a hand that happens to calibrate to local-z-up. For any other
    calibrated axis it measures the wrong, non-vertical direction: e.g. a
    hand calibrated on local +x (base_rot maps local x -> world z) with the
    object sitting 0.15 m to one WORLD-horizontal side of the palm, at
    exactly the palm's own height (not dropped at all), lands at local
    z = -0.15 under that rotation -- below the old check's -0.12 m threshold
    (drop_distance_m=0.24) even though nothing fell. The population path
    already drop-terminated some designs on step 1 from this (worker report,
    2026-09-27). WORLD z has no such dependency: it is "up" for every hand,
    unconditionally, once base_rot has done its job.
    """
    return (obj_pos_w_z - palm_pos_w_z) < -0.5 * drop_distance_m
