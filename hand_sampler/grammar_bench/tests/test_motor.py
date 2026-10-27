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
    ("bend", "bend", 50), ("bend", "spread", 50), ("bend", "twist", 40),
    ("spread", "bend", 40), ("spread", "spread", 40), ("spread", "twist", 30),
    ("twist", "bend", 50), ("twist", "spread", 50), ("twist", "twist", 40),
])
def test_on_kind_link_floors(a, b, floor):
    assert mt.link_floor_mm(KINDS[a], KINDS[b]) == floor


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
    assert max(floors) == 50 and min(floors) == 30
    assert {mt.tip_floor_mm(a) for a in KINDS.values()} == {30}
    assert max(mt.tip_floor_mm(a) for a in COARSE_AXES) == 40


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
        == [50, 30, 40, 50]
    assert mt.link_floor_mm(SPREAD, BEND, False, True) == 30 and mt.link_floor_mm(TWIST, BEND, False, True) == 30
    assert mt.tip_floor_mm(BEND, True) == 40 and mt.tip_floor_mm(BEND) == 30
    assert mt.crossing_slide_mm(SPREAD, BEND, turned_b=True) == pytest.approx(41.0)
    assert mt.BEND_RANGE_DEG == tuple(float(v) for v in KIND_RANGE_DEG["flexion"])


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
    assert not mt.crossing_ok(BEND, TWIST) and not mt.crossing_ok(TWIST, SPREAD)
    assert mt.crossing_slide_mm(SPREAD, BEND) == pytest.approx(26.5)
    assert mt.crossing_slide_mm(BEND, SPREAD) == pytest.approx(26.5)
    assert mt.crossing_slide_mm(BEND, BEND) == pytest.approx(31.0)
    assert mt.crossing_slide_mm(BEND, direction(30, 0)) == pytest.approx(34.057, abs=1e-3)


def test_crossed_motors_stay_a_wall_apart_over_the_range():
    """The earlier motor, slid along its axis, clears the next one by a wall
    whatever the earlier joint's angle (the separating plane is square to its
    axis)."""
    from hand_sampler.grammar.derive import _rot_batch, gjk_distance

    for a, b in itertools.permutations([BEND, SPREAD, direction(30, 0)], 2):
        S = mt.crossing_slide_mm(a, b)
        A = mt.box_corners(mt.motor_centre(a, -S), mt.motor_axes(a), mt.MOTOR_HALF_MM)
        B0 = mt.box_corners(mt.motor_centre(b), mt.motor_axes(b), mt.MOTOR_HALF_MM)
        for R in _rot_batch(a, np.linspace(-np.pi / 2, np.pi / 2, 13)):
            assert gjk_distance(A, B0 @ R.T) >= mt.WALL_MM - 1e-6
