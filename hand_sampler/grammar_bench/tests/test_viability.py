"""The two viability checks: C1 (self-overlap, rounded-box distance) and C2
(fingertip workspaces meet above the palm)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hand_sampler.grammar import derive as dv
from hand_sampler.grammar import operators as ops
from hand_sampler.grammar import viability as V
from hand_sampler.grammar.hand import EVOLUTION_RULES, Finger, Hand, Joint


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


def _two_fingers(gap_mm: float) -> Hand:
    """Two parallel straight fingers; their rounded links (19 mm wide) are
    `gap_mm` apart side by side if positive, overlapping if negative."""
    y = int(round(19 + gap_mm))
    return Hand(fingers=(Finger(y=0, z=80, joints=(Joint("hinge", (0, 0), 50),)),
                         Finger(y=y, z=80, joints=(Joint("hinge", (0, 0), 50),))))


def test_c1_measures_rounded_box_overlap():
    for gap in (2, 0, -2, -3):
        ov = V.overlaps(_two_fingers(gap))
        assert ov == [] or max(o for _, _, o in ov) == pytest.approx(-gap, abs=1e-6)


def test_c1_needs_more_than_3_mm():
    assert V.c1_self_overlap(_two_fingers(-3)).ok
    h = Hand(fingers=(Finger(y=0, z=80, joints=(Joint("hinge", (0, 0), 50),)),
                      Finger(y=19, z=80, facing=355, joints=(Joint("hinge", (0, 0), 50),))))
    r = V.c1_self_overlap(h)
    assert not r.ok and r.worst_mm > 3.0


def test_c1_ignores_bodies_that_touch_by_construction():
    """Consecutive links, links across a 0 mm link and a finger's first link
    against its plate always touch, at any pose."""
    f = Finger(y=0, z=80, joints=(Joint("hinge", (90, 0), 0), Joint("hinge", (0, 0), 0), Joint("hinge", (0, 0), 40)))
    h = Hand(fingers=(f, Finger(y=60, z=80, joints=(Joint("hinge", (0, 0), 40),))))
    pairs = V.touching_pairs(h)
    for a, b in ((("link", 0, 0), ("palm", -1)), (("link", 0, 1), ("palm", -1)), (("link", 0, 2), ("palm", -1)),
                 (("link", 0, 2), ("link", 0, 0)), (("link", 0, 1), ("link", 0, 0))):
        assert (a, b) in pairs
    assert V.c1_self_overlap(h).ok


def test_c1_palm_counts_as_a_link():
    """A finger tilted down into the plate overlaps the main palm."""
    h = Hand(fingers=(Finger(y=0, z=80, joints=(Joint("hinge", (0, 0), 20), Joint("hinge", (0, 0), 40))),
                      Finger(y=40, z=80, tilt=-30, joints=(Joint("hinge", (0, 0), 15), Joint("hinge", (0, 0), 40)))))
    keys = {(a, b) for a, b, _ in V.c1_self_overlap(h).pairs}
    assert (("palm", -1), ("link", 1, 1)) in keys


def _opposed(gap_mm: int) -> Hand:
    """Two fingers facing each other across the palm."""
    return Hand(fingers=(Finger(y=-gap_mm, z=60, facing=90, tilt=30, joints=(Joint("hinge", (0, 0), 40),)),
                         Finger(y=gap_mm, z=60, facing=270, tilt=30, joints=(Joint("hinge", (0, 0), 40),))))


def test_c2_passes_opposed_fingers_and_fails_far_ones():
    assert V.c2_workspace_overlap(_opposed(40)).ok
    far = Hand(fingers=(Finger(y=-150, z=60, facing=270, joints=(Joint("hinge", (0, 0), 30),)),
                        Finger(y=150, z=60, facing=90, joints=(Joint("hinge", (0, 0), 30),))))
    r = V.c2_workspace_overlap(far)
    assert not r.ok


def test_c2_keeps_only_samples_above_the_plate():
    r = V.c2_workspace_overlap(_opposed(40))
    for k in r.kept:
        assert (k[:, 0] > dv.PALM_THICKNESS_MM / 2 * 1e-3).all()


def test_c2_is_deterministic():
    h = ops.random_hand(np.random.default_rng(1), EVOLUTION_RULES)
    a, b = V.c2_workspace_overlap(h), V.c2_workspace_overlap(h)
    assert a.best_mm == b.best_mm and a.pair == b.pair


def test_object_starts_above_the_palm():
    h = _opposed(40)
    p = dv.object_start_m(h, object_half_size_m=0.03)
    assert p[0] == pytest.approx(dv.PALM_THICKNESS_MM / 2 * 1e-3 + 0.035)
    assert dv.point_in_polygon(p[None, 1:] * 1e3, dv.palm_outlines_mm(h)[-1])[0]
