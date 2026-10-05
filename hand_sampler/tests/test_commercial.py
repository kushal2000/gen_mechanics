"""The vendor hands, fitted into this grammar.

LEAP is the one that fits almost exactly: its own geometry is already what the
grammar says a hand is. wuji2 and SHARPA are harder in two specific ways that
are properties of THOSE HANDS rather than failures of the fit, and both are
asserted here so they stay visible:

* their knuckles sit closer together than a 30 mm motor capsule allows, so the
  row has to be spread to make them buildable at all, and
* their joint axes are genuinely oblique, which three kinds of joint cannot say.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hand_sampler import commercial, design_space as D, validate_design


@pytest.fixture(scope="module")
def fits():
    return {name: commercial.fit(name) for name in commercial.HANDS}


def _vendor(name):
    ds = [commercial._straighten(d) for d in commercial.digits(name)]
    row, thumb = commercial._split(ds)
    return commercial._palm_axes(row), row + [thumb]


# --- all three are designs the grammar would produce -------------------------

@pytest.mark.parametrize("name", commercial.HANDS)
def test_every_hand_fits_and_is_legal(name, fits):
    hand, _ = fits[name]
    assert validate_design.check(hand) == [], validate_design.check(hand)
    assert hand.n_fingers >= 4


@pytest.mark.parametrize("name", commercial.HANDS)
def test_every_hand_can_actually_close(name, fits):
    """A fit that validates but cannot grasp would be a fit of the wrong thing."""
    hand, _ = fits[name]
    assert D.curl_score(hand) > 0.9, D.curl_score(hand)


# --- the thumb, and the tip, both of which were read wrong at first ----------

@pytest.mark.parametrize("name", commercial.HANDS)
def test_the_thumb_is_the_digit_that_opposes(name):
    """SHARPA's MCP_FE and MCP_AA are coincident, so its first link has no
    length and no direction; dividing by it handed every digit a NaN and the
    thumb search picked whichever came first. _first_dir skips degenerate links.
    """
    ds = commercial.digits(name)
    _, thumb = commercial._split(ds)
    dirs = [commercial._first_dir(d) for d in ds]
    assert all(np.all(np.isfinite(v)) for v in dirs), "a digit has no direction"
    u = commercial._first_dir(thumb)
    others = [v for d, v in zip(ds, dirs) if d is not thumb]
    assert float(np.mean([u @ v for v in others])) < 0.4, \
        f"{name}: the chosen thumb points the same way as the row"
    if "thumb" in "".join(d.name for d in ds):
        assert "thumb" in thumb.name, f"{name}: picked {thumb.name} as the thumb"


@pytest.mark.parametrize("name", commercial.HANDS)
def test_every_digit_has_a_real_fingertip(name):
    """The tip can be several fixed joints out -- SHARPA goes DP -> elastomer ->
    fingertip -- and a link can carry more than one, wuji2 hanging a zero-offset
    sensor frame beside its tip. Taking the first fixed child gave a tip ON the
    last joint, and so a zero-length distal link."""
    for d in commercial.digits(name):
        last = float(np.linalg.norm(d.tip - d.pos[-1]))
        assert last > 0.015, f"{name} {d.name}: distal link {last*1000:.1f} mm"


# --- what the fit costs, per hand -------------------------------------------

def test_leap_needs_no_spreading_and_the_others_do():
    """wuji2 packs its row at 19-24 mm centres and SHARPA at 17-20, against a 35
    mm floor set by two 30 mm capsules plus clearance. LEAP's own 45 mm spacing
    already clears it, which is why LEAP alone comes through undistorted.
    """
    _, leap_notes = commercial.fit("leap")
    assert not any("motor floor" in n for n in leap_notes), leap_notes
    for name in ("wuji2", "sharpa"):
        _, notes = commercial.fit(name)
        moved = [n for n in notes if "motor floor" in n]
        assert moved, f"{name} should have needed spreading"


@pytest.mark.parametrize("name,worst_mm", [("leap", 9.0), ("wuji2", 26.0), ("sharpa", 36.0)])
def test_per_digit_tip_error(name, worst_mm, fits):
    """Each digit measured from its OWN base, so this is the shape of the finger
    rather than where the palm put it."""
    hand, _ = fits[name]
    M, order = _vendor(name)
    for f, d in zip(hand.fingers, order):
        want = M.T @ (d.tip - d.pos[0])
        base, _ = D.mount_frame(f.mount, hand.palm)
        got = D.fingertip(f, hand.palm) - base
        err = float(np.linalg.norm(got - want)) * 1000
        assert err <= worst_mm, f"{name} {d.name}: {err:.1f} mm"


@pytest.mark.parametrize("name", commercial.HANDS)
def test_the_axis_error_is_the_vendors_own_obliquity(name, fits):
    """The fit snaps each hinge to the nearest of three kinds, so what is left
    over is exactly how far the vendor's axis sat from a coordinate direction in
    the first place -- nothing is lost in between.

    LEAP's axes are all coordinate directions and it fits them to 0 degrees.
    wuji2's are 9.7 degrees off at the median and SHARPA's 7.7, and that is what
    three kinds of joint cost on those hands. A finer grid would not help; only
    allowing oblique axes would, which is the simplification being bought.
    """
    hand, _ = fits[name]
    M, order = _vendor(name)
    fitted, vendor = [], []
    for f, d in zip(hand.fingers, order):
        for got, raw in zip(D.joint_axes(f, hand.palm), d.axis):
            want = M.T @ raw
            fitted.append(math.degrees(math.acos(float(np.clip(abs(got @ want), -1, 1)))))
            vendor.append(math.degrees(math.acos(min(1.0, float(np.max(np.abs(want)))))))
    assert np.median(fitted) == pytest.approx(np.median(vendor), abs=1.0), \
        f"{name}: fit adds error beyond the vendor's own obliquity"


def test_only_leap_has_axis_aligned_joints():
    """Recorded because it is why LEAP was the right hand to build the grammar
    around, and why the other two cannot be fitted as closely."""
    M, order = _vendor("leap")
    off = [math.acos(min(1.0, float(np.max(np.abs(M.T @ a)))))
           for d in order for a in d.axis]
    assert max(math.degrees(x) for x in off) < 2.0

    for name in ("wuji2", "sharpa"):
        M, order = _vendor(name)
        off = [math.degrees(math.acos(min(1.0, float(np.max(np.abs(M.T @ a))))))
               for d in order for a in d.axis]
        assert max(off) > 30.0, f"{name} looks axis-aligned after all: {max(off):.1f}d"
