"""CPU tests for the drop-termination "below the palm" check (2026-09-27
coordinator review).

``drop_detection.py`` imports nothing (not even isaaclab-triggering
pose_reaching_6d modules), so it is reachable under plain pytest -- no Kit,
only ``torch`` itself (safe standalone; only ``isaaclab.*`` needs Kit -- see
that module's docstring and ``tests/test_config_sanity.py``'s for the
general rule this package follows).
"""

from __future__ import annotations

import torch

from isaacsimenvs.inhand_reorient.drop_detection import object_below_palm

DROP_DISTANCE_M = 0.24  # matches env_cfg.ResetCfg.drop_distance_m's default


def test_object_exactly_at_palm_height_is_not_dropped():
    obj_z = torch.tensor([0.5])
    palm_z = torch.tensor([0.5])
    assert not bool(object_below_palm(obj_z, palm_z, DROP_DISTANCE_M).item())


def test_object_far_below_palm_in_world_z_is_dropped():
    obj_z = torch.tensor([0.1])
    palm_z = torch.tensor([0.5])  # 0.4 m below, past the 0.12 m half-threshold
    assert bool(object_below_palm(obj_z, palm_z, DROP_DISTANCE_M).item())


def test_object_above_the_palm_is_not_dropped():
    obj_z = torch.tensor([0.9])
    palm_z = torch.tensor([0.5])
    assert not bool(object_below_palm(obj_z, palm_z, DROP_DISTANCE_M).item())


def test_object_just_inside_the_half_threshold_is_not_dropped():
    obj_z = torch.tensor([0.5 - 0.5 * DROP_DISTANCE_M + 0.01])
    palm_z = torch.tensor([0.5])
    assert not bool(object_below_palm(obj_z, palm_z, DROP_DISTANCE_M).item())


def test_object_just_past_the_half_threshold_is_dropped():
    obj_z = torch.tensor([0.5 - 0.5 * DROP_DISTANCE_M - 0.01])
    palm_z = torch.tensor([0.5])
    assert bool(object_below_palm(obj_z, palm_z, DROP_DISTANCE_M).item())


def test_batched_envs_independently():
    obj_z = torch.tensor([0.5, 0.1, 0.9])
    palm_z = torch.tensor([0.5, 0.5, 0.5])
    result = object_below_palm(obj_z, palm_z, DROP_DISTANCE_M)
    assert result.tolist() == [False, True, False]


def test_non_z_calibrated_axis_worked_example():
    """The regression this check exists for: a hand calibrated with its
    LOCAL +x axis as "up" (base_rot maps local +x -> world +z). Its palm
    BODY frame's own z-axis then points along world -x -- e.g. under
    base_rot = R with columns (local x,y,z expressed in world) =
    ((0,0,1), (0,1,0), (-1,0,0)).

    Object sits 0.15 m to one WORLD-horizontal side of the palm, at exactly
    the palm's own height -- world_offset = (0.15, 0, 0), i.e. NOT dropped
    (obj world z == palm world z). Rotating that offset into the palm's own
    local frame (v_local = R^T @ v_world) gives local z = -0.15: past the
    OLD check's threshold (-0.5 * 0.24 = -0.12 m), so the pre-fix code
    (``env._obj_pos_palm[:, 2] < -0.12``) would have wrongly terminated this
    episode as a drop. The new WORLD-z check does not, because it never
    looks at the palm's local/body frame at all.
    """
    world_offset = torch.tensor([0.15, 0.0, 0.0])
    R = torch.tensor([
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ])  # columns (0,0,1)/(0,1,0)/(-1,0,0): local x/y/z expressed in world
    local_offset = R.T @ world_offset
    assert torch.allclose(local_offset, torch.tensor([0.0, 0.0, -0.15]), atol=1e-6)
    old_buggy_below_palm = bool((local_offset[2] < -0.5 * DROP_DISTANCE_M).item())
    assert old_buggy_below_palm, "sanity check: the pre-fix formula does mis-fire here"

    palm_z = torch.tensor([0.5])
    obj_z = palm_z + world_offset[2]  # same height: world_offset's z is 0
    assert not bool(object_below_palm(obj_z, palm_z, DROP_DISTANCE_M).item())
