"""The capsule SHARPA must be a design the grammar would produce."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hand_sampler import design_space as ds, sharpa_capsule, validate_design


@pytest.fixture(scope="module")
def hand():
    return sharpa_capsule.sharpa_capsule()


@pytest.mark.xfail(strict=True, reason=(
    "The capsule SHARPA was built against the pre-motor constants and is no "
    "longer a design the grammar would produce: 15 mm links against a 20 mm "
    "floor, a mount at u = 0.6 when every mount is now on the midplane, and a "
    "25 mm palm against a thickness tied to the capsule diameter. It is left "
    "alone deliberately -- assets/populations/sharpa_capsule.json came from "
    "this function and the uniform-dynamics training runs are trained on it, so "
    "regenerating the hand would break comparability with results already "
    "collected. hand_sampler.commercial.fit('sharpa') is the in-grammar SHARPA "
    "under the current constants. Decide which of the two is the baseline "
    "before deleting this."))
def test_it_is_a_legal_design(hand):
    assert validate_design.check(hand) == []


def test_it_has_sharpas_structure(hand):
    """Five fingers; the thumb 5 joints, the rest 4. 21, not 22: the pinky's
    roll joint is the one thing the grammar cannot say."""
    assert hand.n_fingers == 5
    assert [f.n_joints for f in hand.fingers] == [5, 4, 4, 4, 4]
    assert hand.n_joints == 21


def test_every_joint_is_pure_flexion_or_abduction(hand):
    """Hinges only, which is exactly what the sampler draws."""
    for finger in hand.fingers:
        for seg in finger.segments:
            assert seg.joint.kind in (ds.FLEXION, ds.ABDUCTION)
            assert seg.lean == 0, "the capsule's links are all straight on"


def test_the_mcp_pattern_is_flexion_then_abduction(hand):
    """SHARPA's MCP_FE is the parent of MCP_AA on every finger; same here."""
    F, A = ds.FLEXION, ds.ABDUCTION
    for finger in hand.fingers[1:]:
        assert [s.joint.kind for s in finger.segments] == [F, A, F, F]
    assert [s.joint.kind for s in hand.fingers[0].segments] == [F, A, F, A, F]


def test_everything_sits_on_the_grammars_grid(hand):
    for finger in hand.fingers:
        for seg in finger.segments:
            assert math.isclose(seg.length / ds.LINK_QUANTUM, round(seg.length / ds.LINK_QUANTUM))
            assert math.isclose(seg.joint.offset / ds.ANGLE_QUANTUM,
                                round(seg.joint.offset / ds.ANGLE_QUANTUM), abs_tol=1e-9)
    # the palm has no extents of its own to check: it is the hull of the mounts
    # grown by PALM_MIN_RADIUS, so its size falls out of where they are


def test_fingertips_land_near_sharpas(hand):
    """Within 20 mm on the four fingers, ~35 on the thumb -- the thumb is the
    deviation the grammar forces, mounted on a thin face 24 mm outboard of where
    SHARPA's sits."""
    names = ("thumb", "index", "middle", "ring", "pinky")
    # SHARPA_TIPS_MM were measured with the WRIST at z = 0; a radial palm puts
    # its own centre there instead, so the whole hand sits one half-length
    # further back. Compare shapes, not absolute positions.
    shift = np.array([0.0, 0.0, 42.5])
    for name, finger in zip(names, hand.fingers):
        tip = ds.fingertip(finger, hand.palm) * 1000 + shift
        err = np.linalg.norm(tip - np.array(sharpa_capsule.SHARPA_TIPS_MM[name]))
        assert err < (36.0 if name == "thumb" else 20.0), f"{name}: {err:.1f} mm"


def test_it_round_trips_through_the_population_file(hand, tmp_path):
    from hand_sampler import population_io

    path = population_io.save_population([hand], tmp_path / "s.json", name="s")
    assert population_io.load_population(path) == [hand]
