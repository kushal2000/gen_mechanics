"""The capsule SHARPA must be a design the grammar would produce."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hand_sampler import design_space as ds, sharpa_capsule, validate_design


@pytest.fixture(scope="module")
def hand():
    return sharpa_capsule.sharpa_capsule()


def test_it_is_a_legal_design(hand):
    assert validate_design.check(hand) == []


def test_it_has_sharpas_structure(hand):
    """Five fingers; the thumb 5 joints, the rest 4. 21, not 22: the pinky's
    roll joint is the one thing the grammar cannot say."""
    assert hand.n_fingers == 5
    assert [f.n_joints for f in hand.fingers] == [5, 4, 4, 4, 4]
    assert hand.n_joints == 21


def test_every_joint_is_pure_flexion_or_abduction(hand):
    """theta on {0, 90}, phi = 90: exactly what the sampler draws."""
    for finger in hand.fingers:
        for seg in finger.segments:
            assert math.isclose(seg.joint.phi, math.pi / 2)
            assert min(abs(seg.joint.theta), abs(seg.joint.theta - math.pi / 2)) < 1e-9


def test_the_mcp_pattern_is_flexion_then_abduction(hand):
    """SHARPA's MCP_FE is the parent of MCP_AA on every finger; same here."""
    for finger in hand.fingers[1:]:
        thetas = [round(math.degrees(s.joint.theta)) for s in finger.segments]
        assert thetas == [0, 90, 0, 0]
    thumb = [round(math.degrees(s.joint.theta)) for s in hand.fingers[0].segments]
    assert thumb == [0, 90, 0, 90, 0]


def test_everything_sits_on_the_grammars_grid(hand):
    for finger in hand.fingers:
        for seg in finger.segments:
            assert math.isclose(seg.length / ds.LINK_QUANTUM, round(seg.length / ds.LINK_QUANTUM))
            assert math.isclose(seg.joint.offset / ds.ANGLE_QUANTUM,
                                round(seg.joint.offset / ds.ANGLE_QUANTUM), abs_tol=1e-9)
    for v in hand.palm.extents:
        assert math.isclose(v / ds.PALM_QUANTUM, round(v / ds.PALM_QUANTUM))


def test_fingertips_land_near_sharpas(hand):
    """Within 20 mm on the four fingers, ~35 on the thumb -- the thumb is the
    deviation the grammar forces, mounted on a thin face 24 mm outboard of where
    SHARPA's sits."""
    names = ("thumb", "index", "middle", "ring", "pinky")
    for name, finger in zip(names, hand.fingers):
        tip = ds.fingertip(finger, hand.palm) * 1000
        err = np.linalg.norm(tip - np.array(sharpa_capsule.SHARPA_TIPS_MM[name]))
        assert err < (36.0 if name == "thumb" else 20.0), f"{name}: {err:.1f} mm"


def test_it_round_trips_through_the_population_file(hand, tmp_path):
    from hand_sampler import population_io

    path = population_io.save_population([hand], tmp_path / "s.json", name="s")
    assert population_io.load_population(path) == [hand]
