"""Curl authority: can a finger close toward the palm at all.

A cheap filter for the downstream search, which otherwise spends its budget
training hands that physically cannot grasp -- every joint a roll, or every
joint an abduction, or a finger tipped so far over that nothing it does brings
its tip toward an object.

What it is NOT: a measure of how GOOD a hand is. It only separates hands that
can close from hands that cannot, and it is deliberately generous in between.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hand_sampler import design_space as D
from hand_sampler import gen_init_pop, commercial, sharpa_capsule

F, A, R = D.FLEXION, D.ABDUCTION, D.ROLL


BEARINGS = (0.0, 45.0, 315.0, 90.0, 270.0, 135.0)
"""Spread round the palm and clear of the arm, in the order they are
used, so a two-finger hand gets the two most opposed of them."""


def _hand(kinds, n_fingers=3, lean=0, radius=0.040):
    palm = D.Palm(D.PALM_THICKNESS)
    return D.Hand(palm, tuple(
        D.Finger(D.Mount.polar(radius, math.radians(b) % (2 * math.pi),
                         math.radians(b) % (2 * math.pi)),
                 tuple(D.Segment(D.Joint(k), 0.040, lean=lean) for k in kinds))
        for b in BEARINGS[:n_fingers]))


# --- what it must reject ----------------------------------------------------

def test_a_finger_of_roll_joints_cannot_close():
    """A roll axis lies along its own link, so turning it moves no tip anywhere."""
    assert D.curl_score(_hand([R, R, R])) == pytest.approx(0.0, abs=1e-12)


def test_a_finger_of_abduction_joints_cannot_close():
    """Abduction's axis IS the grasp direction, so it spreads a hand and never
    closes it. This is exact, not approximate, and it holds whichever way round
    the palm the finger sits."""
    assert D.curl_score(_hand([A, A, A])) == pytest.approx(0.0, abs=1e-12)
    palm = D.Palm(D.PALM_THICKNESS)
    for b in range(0, 360, 15):
        f = D.Finger(D.Mount.polar(0.040, math.radians(b), math.radians(b)),
                     (D.Segment(D.Joint(A), 0.040),))
        assert D.curl_authority(f, palm) == pytest.approx(0.0, abs=1e-12), b


def test_leaning_a_finger_over_degrades_it_rather_than_killing_it():
    """A 45 degree lean tips the hinge with the link, so the finger still closes,
    just less directly. The filter should see that as weak, not as broken."""
    upright = D.curl_score(_hand([F, F, F]))
    leaning = D.curl_score(_hand([F, F, F], lean=1))
    assert leaning < upright / 2.0, (upright, leaning)
    assert leaning > 0.1, f"a leaning finger still closes somewhat, got {leaning}"


# --- what it must NOT reject ------------------------------------------------

def test_the_simplest_possible_hand_passes():
    """Two fingers, one flexion joint each. If the filter fails this it is
    rejecting hands for being simple, which is the one thing it must not do."""
    assert D.curl_score(_hand([F], n_fingers=2)) == pytest.approx(1.0)


def test_a_flexion_finger_scores_full_marks_whichever_way_it_faces():
    """A palm is a disc, so there is no privileged direction to sit in: a
    flexion joint closes toward the grasp volume from anywhere on it."""
    palm = D.Palm(D.PALM_THICKNESS)
    for b in range(0, 360, 15):
        f = D.Finger(D.Mount.polar(0.040, math.radians(b), math.radians(b)),
                     (D.Segment(D.Joint(F), 0.040),))
        assert D.curl_authority(f, palm) == pytest.approx(1.0), b


def test_leap_passes():
    """The one hand actually fitted into this grammar, and so the only one whose
    score calibrates anything."""
    assert D.curl_score(commercial.fit()[0]) == pytest.approx(1.0)


def test_the_sharpa_capsule_is_not_a_reference_but_still_scores():
    """A smoke test, not a calibration. The capsule was hand-built against older
    constants and has never been fitted to the real SHARPA under this grammar,
    so what its score SHOULD be is unknown -- only that the measure returns
    something sane for a five-fingered design with flexion in it.
    """
    assert D.curl_score(sharpa_capsule.sharpa_capsule()) > 0.0


# --- the shape of the number ------------------------------------------------

def test_more_joints_does_not_inflate_the_score():
    """Taken as a MAX over joints, not a norm. The norm would rank a 4-joint
    finger above a 3-joint one, which is a difference no grasp can use."""
    three = D.curl_score(_hand([F, F, F]))
    four = D.curl_score(_hand([F, F, F, F]))
    assert four == pytest.approx(three), (three, four)


def test_a_longer_finger_gets_no_credit_for_being_long():
    """Divided by its own reach, so the score is about direction, not size."""
    palm = D.Palm(D.PALM_THICKNESS)
    short = D.Finger(D.Mount.polar(0.035, math.radians(0), math.radians(0)), (D.Segment(D.Joint(F), 0.020),))
    long_ = D.Finger(D.Mount.polar(0.035, math.radians(0), math.radians(0)), (D.Segment(D.Joint(F), 0.080),))
    assert D.curl_authority(short, palm) == pytest.approx(D.curl_authority(long_, palm))


def test_the_score_is_the_second_finger_not_the_best():
    """One good finger beside a dead one is not a hand that grasps."""
    palm = D.Palm(D.PALM_THICKNESS)
    good = D.Finger(D.Mount.polar(0.031, math.radians(0), math.radians(0)), (D.Segment(D.Joint(F), 0.040),))
    dead = D.Finger(D.Mount.polar(0.039, math.radians(0), math.radians(0)), (D.Segment(D.Joint(R), 0.040),))
    assert D.curl_authority(good, palm) == pytest.approx(1.0)
    assert D.curl_authority(dead, palm) == pytest.approx(0.0, abs=1e-12)
    assert D.curl_score(D.Hand(palm, (good, dead))) == pytest.approx(0.0, abs=1e-12)
    assert D.curl_score(D.Hand(palm, (good, good))) == pytest.approx(1.0)


def test_it_is_cheap_enough_to_filter_with():
    """One forward-kinematics call per finger. If this ever needs a sweep over
    joint angles it stops being a filter and becomes an evaluation."""
    import time
    pop = gen_init_pop.seed_population(0, 100)
    t0 = time.perf_counter()
    for h in pop:
        D.curl_score(h)
    per_hand = (time.perf_counter() - t0) / len(pop)
    assert per_hand < 5e-3, f"{per_hand*1e6:.0f} us per hand is too slow to filter with"


# --- what it says about generation 0 ----------------------------------------

def test_most_of_generation_zero_can_close():
    """This filter found a seeding fault and the fault is now fixed.

    Drawing flexion and abduction evenly, only 38 percent of generation 0 had
    the two closing fingers a grasp needs, because abduction has no curl
    authority at all. SEED_KIND_WEIGHTS favours flexion 3 to 1 and it is 74
    percent. Not 100: seeding flexion alone would get there, and generation 0 is
    meant to start somewhere plain rather than somewhere already solved.
    """
    pop = gen_init_pop.seed_population(0, 200)
    fraction = sum(1 for h in pop if D.curl_score(h) > 0.05) / len(pop)
    assert fraction > 0.65, f"only {fraction:.0%} of generation 0 can close"


def test_seeding_never_draws_a_roll_joint():
    """A roll spins its own link about its own axis. On a one-joint finger that
    is a motor that does nothing a capsule can see."""
    pop = gen_init_pop.seed_population(0, 200)
    assert all(s.joint.kind != R for h in pop for f in h.fingers for s in f.segments)
