"""Pure-python tests for palm_calibration.py -- no Kit needed (see its
module docstring for why this has to stay isaaclab-free)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from isaacsimenvs.inhand_reorient import palm_calibration as pc

_LOCAL_AXIS = {
    "+x": np.array([1.0, 0.0, 0.0]), "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]), "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]), "-z": np.array([0.0, 0.0, -1.0]),
}


def test_quat_apply_matches_scipy():
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_euler("xyz", (12.0, -34.0, 56.0), degrees=True)
    x, y, z, w = rot.as_quat()
    q = np.array([w, x, y, z])
    v = np.array([0.3, -0.7, 1.1])
    got = pc.quat_apply(q, v)
    want = rot.apply(v)
    assert np.allclose(got, want, atol=1e-6)


def test_quat_inv_undoes_quat_mul():
    q = np.array([0.5, 0.5, 0.5, 0.5])
    identity = pc.quat_mul(q, pc.quat_inv(q))
    assert np.allclose(identity, [1.0, 0.0, 0.0, 0.0], atol=1e-6)


@pytest.mark.parametrize("axis", pc.AXIS_NAMES)
def test_each_axis_candidate_maps_its_local_axis_to_world_up(axis):
    candidates = {c["axis"]: c for c in pc.candidate_rotations(full=False)}
    q = np.array(candidates[axis]["quat_wxyz"])
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)
    mapped = pc.quat_apply(q, _LOCAL_AXIS[axis])
    assert np.allclose(mapped, [0.0, 0.0, 1.0], atol=1e-6), (
        f"{axis} candidate does not send local {axis} to world +z: {mapped}")


def test_candidate_rotations_default_scope_is_six():
    cands = pc.candidate_rotations(full=False)
    assert len(cands) == 6
    assert {c["axis"] for c in cands} == set(pc.AXIS_NAMES)


def test_candidate_rotations_full_scope_is_twenty_four_and_still_maps_up():
    cands = pc.candidate_rotations(full=True)
    assert len(cands) == 24
    for c in cands:
        q = np.array(c["quat_wxyz"])
        mapped = pc.quat_apply(q, _LOCAL_AXIS[c["axis"]])
        assert np.allclose(mapped, [0.0, 0.0, 1.0], atol=1e-6)


def test_load_calibration_missing_file_returns_empty_dict(tmp_path):
    assert pc.load_calibration(tmp_path / "nope.json") == {}


def test_save_then_load_calibration_round_trips_and_merges(tmp_path):
    path = tmp_path / "hand_calibration.json"
    pc.save_calibration({"sharpa": {"base_rot": [1.0, 0.0, 0.0, 0.0], "score": 0.9}}, path)
    pc.save_calibration({"dclaw": {"base_rot": [0.0, 1.0, 0.0, 0.0], "score": 0.5}}, path)
    data = pc.load_calibration(path)
    assert set(data) == {"sharpa", "dclaw"}
    assert data["sharpa"]["score"] == 0.9
    # File is valid, indented JSON (readable for review, not just round-trippable).
    assert json.loads(path.read_text()) == data


def test_git_sha_returns_a_hex_string_in_this_repo():
    sha = pc.git_sha()
    assert sha != "unknown"
    assert len(sha) == 40
    int(sha, 16)  # raises if not hex


# --------------------------------------------------------------------------
# I34 (2026-09-27): the ONE spawn-offset convention calibration and the env
# share -- quat_apply(base_rot, spawn_offset_local) is the WORLD-frame offset
# reset_utils.reset_env_state places the cube at (see
# MIN_SPAWN_HEIGHT_ABOVE_PALM_M's docstring for why this is exact, not
# approximate, for a fixed-base hand-only articulation). Every hand actually
# committed to hand_calibration.json must clear the same margin above the
# palm that drop_detection.object_below_palm requires at runtime, or its
# very first reset drop-terminates.
# --------------------------------------------------------------------------

ALL_CALIBRATED_HANDS = (
    "sharpa", "allegro_right", "dclaw", "xhand_right", "wuji_right", "tesollo_dg5f_right",
)


@pytest.mark.parametrize("hand_id", ALL_CALIBRATED_HANDS)
def test_every_calibrated_hand_clears_the_min_spawn_height_above_the_palm(hand_id):
    entry = pc.load_calibration().get(hand_id)
    if entry is None:
        pytest.skip(f"no committed calibration entry for {hand_id!r}")
    base_rot = np.array(entry["base_rot"], dtype=float)
    local_offset = np.array(entry["spawn_offset_local"], dtype=float)
    world_offset = pc.quat_apply(base_rot, local_offset)
    assert world_offset[2] > pc.MIN_SPAWN_HEIGHT_ABOVE_PALM_M, (
        f"{hand_id}: spawn point is {world_offset[2]:.4f} m above the palm in world z "
        f"(need > {pc.MIN_SPAWN_HEIGHT_ABOVE_PALM_M} m); base_rot={entry['base_rot']} "
        f"spawn_offset_local={entry['spawn_offset_local']} -- re-run calibrate_palm_up.py "
        f"--hand {hand_id}")
