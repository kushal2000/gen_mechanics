"""The two viability checks: C1 (a wall between solids that do not touch by
construction, exact box distances) and C2 (fingertip workspaces meet above
the palm)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from hand_sampler.grammar import derive as dv
from hand_sampler.grammar import operators as ops
from hand_sampler.grammar import viability as V
from hand_sampler.grammar.hand import (
    FEETECH_RULES,
    KIND_AXIS,
    NO_RULES,
    PLATE_HALF_MM,
    Finger,
    Hand,
    Joint,
    PalmJoint,
    check,
    finger_section,
)

BEND, SPREAD, TWIST = KIND_AXIS["flexion"], KIND_AXIS["abduction"], KIND_AXIS["roll"]


def _finger(y, z, facing=0, tilt=0, joints=((BEND, 50), (BEND, 30))) -> Finger:
    return Finger(y=y, z=z, facing=facing, tilt=tilt, joints=tuple(Joint("hinge", a, L) for a, L in joints))


def _box(rng, centre_scale=0.03):
    c = rng.normal(size=3) * centre_scale
    h = rng.uniform(0.001, 0.02, 3)
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return dv.box_corners(np.zeros(3), h) @ R.T + c


def _brute(A, B, n=1000, rng=None):
    """Upper bound on the distance from random convex combinations (for a
    sanity check of GJK)."""
    la = rng.dirichlet(np.ones(len(A)), n)
    lb = rng.dirichlet(np.ones(len(B)), n)
    pa, pb = la @ A, lb @ B
    return float(np.min(np.linalg.norm(pa[:, None] - pb[None], axis=-1)))


def test_gjk_is_exact_on_axis_aligned_boxes():
    A = dv.box_corners(np.zeros(3), np.array([0.01, 0.01, 0.01]))
    for d in (0.025, 0.05, 0.1):
        B = dv.box_corners(np.array([d, 0.0, 0.0]), np.array([0.01, 0.01, 0.01]))
        assert V.gjk_distance(A, B) == pytest.approx(d - 0.02, abs=1e-12)
    B = dv.box_corners(np.array([0.03, 0.03, 0.0]), np.array([0.01, 0.01, 0.01]))
    assert V.gjk_distance(A, B) == pytest.approx(math.hypot(0.01, 0.01), abs=1e-12)
    B = dv.box_corners(np.array([0.015, 0.0, 0.0]), np.array([0.01, 0.01, 0.01]))
    assert V.gjk_distance(A, B) == 0.0


def test_gjk_is_never_above_a_sampled_bound():
    rng = np.random.default_rng(0)
    for _ in range(100):
        A, B = _box(rng), _box(rng)
        d = V.gjk_distance(A, B)
        assert d <= _brute(A, B, rng=rng) + 1e-9
        assert d >= 0.0


def _side_by_side(gap_y: int) -> Hand:
    """Two flat bending fingers side by side, `gap_y` mm apart, both along +z
    (no rules: any 10 mm spacing)."""
    return Hand(fingers=(_finger(0, 80), _finger(gap_y, 80)))


def test_c1_keeps_a_wall_between_neighbouring_fingers():
    """Bending fingers are 33 mm wide (the motor along its axis plus a wall
    each side), so neighbours need 35 mm: 40 mm leaves 7, 30 mm overlaps by
    3 plus the wall."""
    for gap, ok, worst in ((40, True, 0.0), (30, False, 5.0)):
        h = _side_by_side(gap)
        r = V.c1_self_overlap(h)
        assert r.ok == ok
        if not ok:
            assert {("link", 0, 0), ("link", 1, 0)} <= {k for a, b, _ in r.pairs for k in (a, b)}
            assert r.worst_mm == pytest.approx(min(worst, 2.0))     # intersecting solids count 2 mm


def test_c1_measures_the_gap():
    """Bending fingers 40 mm apart: 7 mm between their 33 mm links, 11 mm
    between their 29 mm base motors."""
    h = _side_by_side(40)
    solids = {s.key: s for s in dv.solids_at(h)}
    d = V.gjk_distance(solids[("link", 0, 0)].points, solids[("link", 1, 0)].points) * 1e3
    assert d == pytest.approx(7.0)
    d = V.gjk_distance(solids[("motor", 0)].points, solids[("motor", 1)].points) * 1e3
    assert d == pytest.approx(11.0)


def test_c1_ignores_solids_that_touch_by_construction():
    """A plate with its base motors and its fingers' first links, consecutive
    links, and links across a 0 mm link."""
    f = _finger(0, 80, joints=((SPREAD, 0), (BEND, 0), (BEND, 40), (BEND, 30)))
    h = Hand(fingers=(f, _finger(60, 80)))
    pairs = V.touching_pairs(h)
    for a, b in ((("palm", -1), ("motor", 0)), (("link", 0, 0), ("palm", -1)), (("link", 0, 1), ("palm", -1)),
                 (("link", 0, 2), ("palm", -1)), (("link", 0, 2), ("link", 0, 0)), (("link", 0, 1), ("link", 0, 0)),
                 (("link", 0, 0), ("motor", 0)), (("palm", -1), ("mount",))):
        assert (a, b) in pairs and (b, a) in pairs
    assert (("link", 0, 3), ("palm", -1)) not in pairs
    assert (("motor", 0), ("motor", 1)) not in pairs


def test_a_link_cannot_lie_in_the_palm():
    """A finger running along the palm's edge over the plate fails (its far
    end is checked against the plate it is hinged to); the same finger
    pointing outward passes."""
    out = Hand(fingers=(_finger(0, 80), _finger(80, 0, facing=90)))
    assert V.c1_self_overlap(out).ok
    along = Hand(fingers=(_finger(0, 80, joints=((BEND, 50), (BEND, 30))),
                          _finger(80, 0, facing=0, joints=((BEND, 50), (BEND, 30)))))
    r = V.c1_self_overlap(along)
    assert not r.ok and any({a, b} == {("palm", -1), ("link", 1, 0)} for a, b, _ in r.pairs)


def test_the_mount_blocks_motors_behind_the_palm():
    """A spread-first finger's motor sticks 9 mm out of the back: on the inner
    ring, pointing outward, it hits the mount; a bend-first one (flush) clears
    it by exactly a wall."""
    for axis, ok in ((BEND, True), (SPREAD, False)):
        h = Hand(fingers=(_finger(0, 40, joints=((axis, 50), (BEND, 30))), _finger(0, -80, facing=180)))
        r = V.c1_self_overlap(h)
        assert r.ok == ok, r.pairs
        if not ok:
            assert any({a, b} == {("mount",), ("motor", 0)} for a, b, _ in r.pairs)


def test_a_cut_through_a_base_motor_fails():
    """The line z = 40 leaves the finger at (0, 80) beyond it with its motor's
    housing clear of the seam; at z = 60 the seam runs through that housing."""
    h = Hand(fingers=(_finger(0, 80), _finger(80, 0, facing=90), _finger(-80, 0, facing=270)))
    good = replace(h, palm_joints=(PalmJoint(facing=90, offset=40),))
    assert check(good, FEETECH_RULES) == [] and V.c1_self_overlap(good).ok
    bad = replace(h, palm_joints=(PalmJoint(facing=90, offset=60),))
    assert check(bad, FEETECH_RULES) == []
    r = V.c1_self_overlap(bad)
    assert not r.ok and any(b == ("seam",) for _, b, _ in r.pairs)


def test_random_hands_with_a_cut_keep_their_seam():
    rng = np.random.default_rng(3)
    n = 0
    while n < 20:
        h = ops.random_hand(rng, FEETECH_RULES)
        if not h.palm_joints:
            continue
        n += 1
        assert V.seam_overlaps(h) == [] and dv.hand_geometry(h).hinge_clear


def _opposed(gap_mm: int) -> Hand:
    """Two fingers facing each other across the palm (no rules)."""
    return Hand(fingers=(_finger(-gap_mm, 60, facing=90, tilt=30, joints=((BEND, 40),)),
                         _finger(gap_mm, 60, facing=270, tilt=30, joints=((BEND, 40),))))


def test_c2_passes_opposed_fingers_and_fails_far_ones():
    assert V.c2_workspace_overlap(_opposed(40)).ok
    far = Hand(fingers=(_finger(-150, 60, facing=270, joints=((BEND, 30),)),
                        _finger(150, 60, facing=90, joints=((BEND, 30),))))
    assert not V.c2_workspace_overlap(far).ok


def test_c2_keeps_only_samples_above_the_plate():
    r = V.c2_workspace_overlap(_opposed(40))
    for k in r.kept:
        assert (k[:, 0] > PLATE_HALF_MM * 1e-3).all()


def test_c2_fingers_beyond_the_cut_share_the_hinge():
    """Two twisting fingers on the part beyond the cut: their tips are fixed
    on that part, so their distance (36.9 mm) is the same at every hinge
    angle. With one hinge angle per comparison C2 finds exactly that distance
    (independent angles would bring them to 24 mm, a pose the hand cannot
    take)."""
    fingers = (_finger(0, -80, facing=0, tilt=90, joints=((TWIST, 60),)),
               _finger(40, -80, facing=270, tilt=30, joints=((TWIST, 50),)),
               _finger(0, 80, joints=((TWIST, 30),)))
    h = Hand(fingers=fingers, palm_joints=(PalmJoint(facing=270, offset=40),))
    assert [finger_section(h, i) for i in range(3)] == [0, 0, -1]
    r = V.c2_workspace_overlap(h, tol_mm=1000.0)
    tips = dv.fk(h).tips * 1e3
    assert r.pair == (0, 1)
    assert r.best_mm == pytest.approx(float(np.linalg.norm(tips[0] - tips[1])), abs=1e-6)


def test_c2_is_deterministic():
    h = ops.random_hand(np.random.default_rng(1), FEETECH_RULES)
    a, b = V.c2_workspace_overlap(h), V.c2_workspace_overlap(h)
    assert a.best_mm == b.best_mm and a.pair == b.pair


def test_object_starts_above_the_palm():
    """Fingers pointing away from the centre: the cube sits 5 mm above the
    front face over the outline's centroid; fingers curling over the centre
    raise it."""
    h = Hand(fingers=(_finger(0, 80), _finger(0, -80, facing=180)))
    p = dv.object_start_m(h, object_half_size_m=0.03)
    assert p[0] == pytest.approx((PLATE_HALF_MM + 35) * 1e-3)
    assert dv.point_in_polygon(p[None, 1:] * 1e3, dv.palm_outlines_mm(h)[-1])[0]
    big = dv.object_start_m(h, object_half_size_m=0.036)
    assert big[0] == pytest.approx((PLATE_HALF_MM + 41) * 1e-3)
    assert dv.object_start_m(_opposed(40), 0.03)[0] > (PLATE_HALF_MM + 35) * 1e-3


def test_object_starts_clear_of_a_finger_standing_by_it():
    """A finger tilted straight up next to the palm centre: the object starts
    where it clears every solid by 5 mm at the start pose."""
    up = _finger(0, 40, facing=0, tilt=90, joints=((BEND, 50), (BEND, 30)))
    h = Hand(fingers=(up, _finger(0, -80, facing=180)))
    p = dv.object_start_m(h, 0.03) * 1e3
    assert p[0] > PLATE_HALF_MM + 35
    cube = dv.box_corners(p, np.full(3, 30.0)) * 1e-3
    assert min(V.gjk_distance(cube, s.points) for s in dv.solids_at(h, dv.start_q(h))) * 1e3 >= 5.0 - 1e-6
