"""The vendor hands, fitted into this grammar.

LEAP and MIDAS fit closely -- their own geometry is already close to what the
grammar says a hand is. wuji2 is harder, in two specific ways that are
properties of THAT HAND rather than failures of the fit, and both are asserted
here so they stay visible:

* its knuckles sit 22 mm apart, closer than a 30 mm motor capsule allows, so the
  row has to be spread 13 mm wider than the vendor's to clear at all -- and it
  then clears by exactly nothing, which is its own test, and
* its joint axes are genuinely oblique, which three kinds of joint cannot say.

SHARPA was fitted and dropped; see commercial.HANDS for why.
"""

from __future__ import annotations

import itertools
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

@pytest.mark.parametrize("name", ("leap", "midas"))
def test_these_hands_fit_and_are_legal(name, fits):
    hand, _ = fits[name]
    assert validate_design.check(hand) == [], validate_design.check(hand)
    assert hand.n_fingers >= 4


def test_wuji2_clears_its_own_links_by_nothing_at_all(fits):
    """Legal, and only just: its closest two links sit EXACTLY on the 30 mm
    floor, with zero microns to spare.

    Its own links pass within 22.4 mm of each other and two 30 mm capsules need
    30, so the fit has to spread the bases until they clear -- and the separator
    stops the moment they do, which on a 5 mm cartesian grid means landing on
    the boundary exactly. It could not land there while a mount was polar: a
    radius on one grid and a bearing on another put the bases at distances that
    were never round numbers, so the hand missed the floor by 2.9 mm and was
    refused outright.

    So this is not a hand that became buildable. It is the same hand, now
    describable to the last micron of the rule, and nobody should build two
    capsules that touch. Asserted here so the zero is on the record rather than
    reading as a pass.

    Generated hands do not do this -- over 60 drifted designs none came within
    0.1 mm of the floor and the median clearance was 13.6 mm -- because nothing
    is pushing them to the boundary the way the fit's separator pushes wuji2.
    """
    hand, _ = fits["wuji2"]
    assert not validate_design.check(hand)
    links = D.rest_capsules(hand)
    gap = min(D.segment_distance(p0, p1, q0, q1)
              for (fi, si, p0, p1), (fj, sj, q0, q1) in itertools.combinations(links, 2)
              if not (fi == fj and abs(si - sj) <= 1))
    assert gap == pytest.approx(2.0 * D.CAPSULE_RADIUS, abs=1e-6), (
        f"wuji2's closest links are {gap * 1000:.3f} mm apart, not on the "
        f"{2 * D.CAPSULE_RADIUS * 1000:.0f} mm floor -- if this has gained real "
        f"clearance the fit has changed, and if it has lost any it is illegal")


@pytest.mark.parametrize("name", commercial.HANDS)
def test_every_hand_can_actually_close(name, fits):
    """A fit that validates but cannot grasp would be a fit of the wrong thing.

    MIDAS scores lower than the other two because it leads each finger with
    abduction -- its first joint is mcp_abad -- so the joints that close it sit
    further down the chain with a shorter moment arm to the tip.
    """
    hand, _ = fits[name]
    assert D.curl_score(hand) > 0.7, D.curl_score(hand)


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

@pytest.mark.parametrize("name", commercial.HANDS)
def test_every_vendor_hand_clears_the_arm(name, fits):
    """These are the hands the clearance rule has to admit.

    Every vendor hand reaches back: each has a thumb 30 degrees off the wrist
    bearing, and before WRIST_STANDOFF LEAP's went 17 mm inside the arm. They
    are the tightest cases the grammar has to accept, so they are what pins the
    standoff -- LEAP clears by 8 mm, wuji2 by 14, MIDAS by 21.
    """
    hand, _ = fits[name]
    z, who = D.rearmost(hand)
    assert z >= D.ARM_FACE_Z, (
        f"{name}: {'the palm' if who is None else f'finger {who}'} reaches "
        f"{(D.ARM_FACE_Z - z) * 1000:.0f} mm into the arm")
    assert not validate_design.check_arm_clearance(hand)


def test_every_hand_needs_some_spreading_on_a_radial_palm():
    """A radial palm places a base at its OWN polar coordinate, snapped to a
    5 mm radius grid and a 15 deg bearing grid. The snap alone can move a base
    several millimetres -- a bearing quantum is 6.5 mm of arc at r = 50 -- so
    even LEAP, whose knuckles are a comfortable 45 mm apart, needs a nudge after
    snapping. That is the grid, not the hand being too tightly packed.
    """
    for name in commercial.HANDS:
        _, notes = commercial.fit(name)
        assert [n for n in notes if "motor floor" in n], name


@pytest.mark.parametrize("name,worst_mm", [("leap", 10.0), ("wuji2", 20.0),
                                           ("midas", 30.0), ("allegro", 30.0)])
def test_per_digit_tip_error(name, worst_mm, fits):
    """Each digit measured from its OWN base, so this is the shape of the finger
    rather than where the palm put it.

    Measured: LEAP 8.0 mm at worst, wuji2 17.5, MIDAS 27.8, Allegro 27.4 -- and
    in the last two it is the thumb alone, the fingers being 2.6 and 7.5 to
    15.2. The bounds carry about 15% of headroom. wuji2's was 46 mm, set when
    its row was being fitted parallel; it is 17.5 now that the row keeps its own
    splay, and leaving the old number there would have tested nothing.
    """
    hand, _ = fits[name]
    M, order = _vendor(name)
    for f, d in zip(hand.fingers, order):
        want = M.T @ (d.tip - d.pos[0])
        base, _ = D.mount_frame(f.mount)
        got = D.fingertip(f, hand.palm) - base
        err = float(np.linalg.norm(got - want)) * 1000
        assert err <= worst_mm, f"{name} {d.name}: {err:.1f} mm"


def test_allegros_base_joint_is_a_roll_at_rest_and_a_spread_when_bent(fits):
    """Why Allegro's row reads roll -> flexion -> flexion -> flexion.

    Its base joint's axis is (0, 0, 1) and the whole finger then runs along +z,
    so at full extension the joint twists the finger about its own axis and
    moves the fingertip NOWHERE. It becomes a spread only once the finger bends:
    driven +/-20 degrees it sweeps the tip 0.0 mm extended, 41 mm at 15 degrees
    of flex, 71 at 45, and 11 again by 90.

    The grammar names a joint by its axis at the REST pose, so `roll` is the
    honest reading and the curl score is right to give it nothing. LEAP's
    equivalent joint is the contrast: a true abduction, sweeping 94 mm with the
    finger straight out. Pinned because `roll` on three identical fingers of a
    well-known hand looks like the mistake it would be on any other hand.
    """
    ds = {d.name: d for d in commercial.digits("allegro")}
    d = ds["joint_0_0"]
    straight = [p for p in d.pos] + [d.tip]
    axis = np.asarray(d.axis[0], float)
    axis = axis / np.linalg.norm(axis)
    for p in straight[1:]:
        along = p - straight[0]
        off = along - float(along @ axis) * axis
        assert float(np.linalg.norm(off)) < 1e-6, (
            "the finger is not collinear with its own base axis, so that joint "
            "is not a roll at rest and the fitted kind needs re-deriving")

    hand, _ = fits["allegro"]
    for f in hand.fingers[:3]:
        kinds = [s.joint.kind for s in f.segments]
        assert kinds == [D.ROLL, D.FLEXION, D.FLEXION, D.FLEXION], kinds


@pytest.mark.parametrize("name", commercial.HANDS)
def test_the_axis_error_is_the_vendors_own_obliquity(name, fits):
    """The fit snaps each hinge to the nearest of three kinds, so what is left
    over is at most how far the vendor's axis sat from a coordinate direction in
    the first place -- nothing is lost in between.

    LEAP's axes are all coordinate directions and it fits them to 0 degrees.
    MIDAS's it matches exactly too. That is what three kinds of joint cost on
    those hands, and a finer grid would not help; only allowing oblique axes
    would, which is the simplification being bought.

    An UPPER bound, not an equality, because wuji2 now comes in under it: its
    obliquity is 10.2 degrees at the median and the fit lands at 6.4. A mount
    carries a facing, and wuji2's row is mounted at -15, 0, 0 and +15 degrees --
    its own splay -- which turns each mount frame and lets an axis that is
    frame-aligned IN THAT FRAME sit closer to an oblique one than any palm axis
    can. The equality held while the row was being fitted parallel.
    """
    hand, _ = fits[name]
    M, order = _vendor(name)
    fitted, vendor = [], []
    for f, d in zip(hand.fingers, order):
        for got, raw in zip(D.joint_axes(f, hand.palm), d.axis):
            want = M.T @ raw
            fitted.append(math.degrees(math.acos(float(np.clip(abs(got @ want), -1, 1)))))
            vendor.append(math.degrees(math.acos(min(1.0, float(np.max(np.abs(want)))))))
    assert np.median(fitted) <= np.median(vendor) + 1.0, \
        f"{name}: fit adds error beyond the vendor's own obliquity "\
        f"({np.median(fitted):.1f} deg against {np.median(vendor):.1f})"


def test_leap_is_the_only_exactly_axis_aligned_hand():
    """Recorded because it is why LEAP was the right hand to build the grammar
    around. MIDAS is close but not exact; wuji2 is not close."""
    M, order = _vendor("leap")
    off = [math.degrees(math.acos(min(1.0, float(np.max(np.abs(M.T @ a))))))
           for d in order for a in d.axis]
    assert max(off) < 2.0

    M, order = _vendor("wuji2")
    off = [math.degrees(math.acos(min(1.0, float(np.max(np.abs(M.T @ a))))))
           for d in order for a in d.axis]
    assert max(off) > 30.0, f"wuji2 looks axis-aligned after all: {max(off):.1f}d"


# --- MIDAS, whose URDF needed two things the others did not -----------------

def test_midas_ignores_its_four_bar_linkages():
    """Each MCP pitch link carries TWO revolute children: the PIP, which is the
    finger, and a linkage joint that closes a four-bar and goes nowhere. Taking
    the first child followed whichever the file listed first; the finger is the
    longest way down.

    The distal joint is treated as fully actuated, which is what the URDF
    already says -- it carries no <mimic>, and the coupling lives only in the
    vendor's MuJoCo model as an equality constraint.
    """
    for d in commercial.digits("midas"):
        assert len(d.joints) == 4, (d.name, d.joints)
        assert not any("linkage" in j for j in d.joints), d.joints


def test_midas_digits_end_at_a_fingertip():
    """Its URDF has no tip frames at all, so without TIP_FALLBACK every finger
    would come out with a zero-length distal link."""
    assert "midas" in commercial.TIP_FALLBACK
    for d in commercial.digits("midas"):
        last = float(np.linalg.norm(d.tip - d.pos[-1])) * 1000
        assert 15.0 < last < 60.0, f"{d.name}: distal link {last:.1f} mm"


def test_the_vendored_midas_urdf_is_untouched():
    """The tip offsets live in TIP_FALLBACK precisely so this file does not have
    to be edited. If it ever grows a tip frame, move them out of the table."""
    import xml.etree.ElementTree as ET
    root = ET.parse(commercial.urdf_of("midas")).getroot()
    assert not root.findall(".//mimic"), "upstream has no mimic tags"
    tips = [j.get("name") for j in root.findall("joint")
            if j.get("type") == "fixed" and "tip" in (j.get("name") or "")]
    assert not tips, f"the URDF grew tip frames: {tips}"
