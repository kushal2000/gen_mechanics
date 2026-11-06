"""The motor model (`hand_sampler/grammar/motor.py`): the Feetech HL-3915 box,
the link and fingertip floors, crossing slides (E15)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from hand_sampler.grammar import motor as mt
from hand_sampler.grammar.derive import direction
from hand_sampler.grammar.hand import KIND_RANGE_DEG

BEND, SPREAD, TWIST = mt.Z, mt.Y, mt.X
KINDS = {"bend": BEND, "spread": SPREAD, "twist": TWIST}
COARSE_AXES = [direction(az, el) for az in range(0, 180, 30) for el in (-60, -30, 0, 30, 60)] + [direction(0, 90)]


@pytest.mark.parametrize("a,b,floor", [
    ("bend", "bend", 42), ("bend", "spread", 42), ("bend", "twist", 32),
    ("spread", "bend", 41), ("spread", "spread", 41), ("spread", "twist", 31),
    ("twist", "bend", 42), ("twist", "spread", 42), ("twist", "twist", 32),
])
def test_on_kind_link_floors(a, b, floor):
    """The next whole mm past `BUILD_CLEARANCE_MM` (Martin, 2026-11-04: no
    longer up to the 10 mm grid, which made 40.29 mm 50)."""
    assert mt.link_floor_mm(KINDS[a], KINDS[b]) == floor


def test_floor_rounding_and_lengths():
    assert [mt.floor_mm(v) for v in (40.29, 39.73, 39.0, 41.0, 49.5, 30.0)] == [42, 41, 40, 42, 51, 31]
    assert mt.lengths_from(41, 90, 10) == [41, 50, 60, 70, 80, 90]
    assert mt.lengths_from(41, 90, 20) == [41, 60, 80]
    assert mt.lengths_from(30, 90, 20) == [30, 50, 70, 90]
    assert mt.lengths_from(100, 90, 10) == []


def test_link_floors_match_e15():
    """E15's exact numbers before rounding: 40.29 for two square axes, 39.73
    after a spread (its range never swings a corner fully forward), 41 after
    a twist."""
    assert mt.plane_floor_mm(BEND, BEND) == pytest.approx(40.293, abs=1e-3)
    assert mt.plane_floor_mm(SPREAD, BEND) == pytest.approx(39.727, abs=1e-3)
    assert mt.plane_floor_mm(TWIST, BEND) == pytest.approx(41.0, abs=1e-9)


def test_floors_over_the_coarse_axes():
    assert len(COARSE_AXES) == 31
    floors = {mt.link_floor_mm(a, b) for a in COARSE_AXES for b in COARSE_AXES}
    assert floors == {31, 32, 37, 41, 42, 43, 45, 46, 47, 48, 51}
    # the fingertip: the swing, a 5 mm gap and the dome (Martin, 11/06): 33-34 mm after every kind, 39 at most
    assert {mt.tip_floor_mm(a) for a in KINDS.values()} == {33, 34}
    assert max(mt.tip_floor_mm(a) for a in COARSE_AXES) == 39


def test_a_turned_bend_motor():
    """A turned bend motor lies with its body on the back of the finger: 10
    mm of it behind its axis along the link instead of 24.5, but its back
    corner swings 20.9 mm forward at the -30 degree hyperextension (13.8 for
    a motor lying along the link)."""
    M = mt.motor_axes(BEND, turned=True)
    assert np.allclose(M[:, 1], -mt.Y) and np.allclose(M[:, 2], mt.X)
    assert mt.back_mm(BEND, True) == pytest.approx(10.0) and mt.back_mm(BEND) == pytest.approx(24.5)
    assert mt.swept_front_mm(BEND, True) == pytest.approx(10 * math.cos(math.radians(30)) + 24.5 * 0.5)
    assert mt.swept_front_mm(BEND) == pytest.approx(math.hypot(9.5, 10.0))
    assert [mt.link_floor_mm(BEND, BEND, ta, tb) for ta, tb in ((False, False), (False, True), (True, True), (True, False))] \
        == [42, 27, 34, 49]
    assert mt.link_floor_mm(SPREAD, BEND, False, True) == 27 and mt.link_floor_mm(TWIST, BEND, False, True) == 28
    assert mt.tip_floor_mm(BEND, True) == 40 and mt.tip_floor_mm(BEND) == 33
    assert mt.crossing_slide_mm(SPREAD, BEND, turned_b=True) == pytest.approx(42.5)
    assert mt.BEND_RANGE_DEG == tuple(float(v) for v in KIND_RANGE_DEG["flexion"])


def test_a_flipped_bend_motor():
    """A flipped bend motor lies along the link after its joint (Martin,
    2026-10-29): its back is the finger direction, 24.5 mm of it ahead of
    its axis; the link before holds its horns in a yoke round its front,
    which swings back as far as a lying motor's swings forward."""
    M = mt.motor_axes(BEND, flipped=True)
    assert np.allclose(M[:, 1], mt.X)
    assert mt.reach_ahead_mm(BEND, "flipped") == pytest.approx(24.5)
    assert mt.reach_behind_mm(BEND, "flipped") == pytest.approx(math.hypot(9.5, 10.0))
    for pa in ("lying", "turned"):
        for pb in ("lying", "turned"):
            assert mt.pose_floor_mm(BEND, BEND, pa, pb) == mt.link_floor_mm(BEND, BEND, pa == "turned", pb == "turned")
    assert [mt.pose_floor_mm(BEND, BEND, *pp) for pp in (("lying", "flipped"), ("turned", "flipped"),
                                                         ("flipped", "flipped"), ("flipped", "turned"),
                                                         ("flipped", "lying"))] == [31, 38, 42, 38, 52]
    assert mt.pose_floor_mm(TWIST, BEND, "lying", "flipped") == 32
    assert mt.tip_pose_floor_mm(BEND, "flipped") == 44 and mt.tip_floor_mm(BEND, True) == 40


def test_motor_box():
    """The 34 x 20 x 29 mm servo: 24.5 mm behind its axis, 9.5 ahead, 29
    along it, 20 wide; its back is the finger direction reversed (for a twist
    motor, the closing direction reversed)."""
    for a in KINDS.values():
        M = mt.motor_axes(a)
        assert np.allclose(M.T @ M, np.eye(3))
        C = mt.box_corners(mt.motor_centre(a), M, mt.MOTOR_HALF_MM)
        along_back = C @ M[:, 1]
        assert along_back.max() == pytest.approx(24.5) and along_back.min() == pytest.approx(-9.5)
        assert np.ptp(C @ M[:, 0]) == pytest.approx(29.0) and np.ptp(C @ M[:, 2]) == pytest.approx(20.0)
    assert np.allclose(mt.motor_axes(BEND)[:, 1], -mt.X)
    assert np.allclose(mt.motor_axes(TWIST)[:, 1], -mt.Y)


def test_crossing_rules_and_slides():
    assert mt.crossing_ok(BEND, SPREAD) and mt.crossing_ok(SPREAD, BEND)
    assert not mt.crossing_ok(BEND, BEND)                 # the same line
    # into a twist, its servo standing across the first one's axis as at the base of LEAP's thumb (2026-11-04);
    # a twist's servo cannot slide aside along its own axis
    assert mt.crossing_ok(BEND, TWIST) and mt.crossing_ok(SPREAD, TWIST)
    assert not mt.crossing_ok(TWIST, SPREAD) and not mt.crossing_ok(TWIST, BEND)
    # a wall plus 1.5 mm for the near horn's screw head (Martin, 2026-10-23)
    assert mt.crossing_slide_mm(SPREAD, BEND) == pytest.approx(28.0)
    assert mt.crossing_slide_mm(BEND, SPREAD) == pytest.approx(28.0)
    assert mt.crossing_slide_mm(BEND, BEND) == pytest.approx(32.5)
    assert mt.crossing_slide_mm(BEND, direction(30, 0)) == pytest.approx(35.557, abs=1e-3)
    assert mt.crossing_slide_mm(BEND, TWIST) == pytest.approx(28.0)        # past the twist servo's width
    assert mt.crossing_slide_mm(SPREAD, TWIST) == pytest.approx(42.5)      # past its body on the finger's back


def test_crossed_motors_stay_a_wall_apart_over_the_range():
    """The earlier motor, slid along its axis, clears the next one by a wall
    whatever the earlier joint's angle (the separating plane is square to its
    axis)."""
    from hand_sampler.grammar.derive import _rot_batch, gjk_distance

    pairs = list(itertools.permutations([BEND, SPREAD, direction(30, 0)], 2))
    pairs += [(a, TWIST) for a in (BEND, SPREAD, direction(30, 0))]
    for a, b in pairs:
        S = mt.crossing_slide_mm(a, b)
        A = mt.box_corners(mt.motor_centre(a, -S), mt.motor_axes(a), mt.MOTOR_HALF_MM)
        B0 = mt.box_corners(mt.motor_centre(b), mt.motor_axes(b), mt.MOTOR_HALF_MM)
        for R in _rot_batch(a, np.linspace(-np.pi / 2, np.pi / 2, 13)):
            assert gjk_distance(A, B0 @ R.T) >= mt.WALL_MM - 1e-6


def test_a_flipped_spread_motor():
    """A spread servo past a link too short for it to lie along flips, its
    case along the link after its joint, as LEAP's thumb servos sit
    (2026-11-04; a bend servo turns there instead): 31 mm after a twist, 30
    after a spread, 41 to another flipped spread, 52 to a lying servo, a 44
    mm fingertip."""
    assert mt.pose_floor_mm(TWIST, SPREAD, "lying", "flipped") == 31
    assert mt.pose_floor_mm(SPREAD, SPREAD, "lying", "flipped") == 30
    assert mt.pose_floor_mm(SPREAD, SPREAD, "flipped", "flipped") == 41
    assert mt.pose_floor_mm(SPREAD, BEND, "flipped", "lying") == 52
    assert mt.tip_pose_floor_mm(SPREAD, "flipped") == 44
    assert mt.pose_floor_mm(TWIST, SPREAD, "lying", "lying") == 42      # lying, as before


def test_a_link_before_a_crossing_fits_the_servo_after_it():
    """A crossing's second servo shares the crossing's point, lies along the
    finger (24.5 mm behind it) and sits in the next part: the link before the
    crossing keeps its housing wall and a gap past the plane floor
    (2026-10-23): 45 mm after a bend, 44 after a spread, 52 after a turned
    bend."""
    assert mt.crossing_floor_mm(BEND) == 45 and mt.crossing_floor_mm(SPREAD) == 44
    assert mt.crossing_floor_mm(BEND, True) == 52
