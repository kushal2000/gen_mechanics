"""The two knobs the minimal set added: a mounting tilt, and roll.

Everything else about the grammar is unchanged, so what is checked here is that
the tilt is a well-formed alphabet, that it reaches the authored robot and not
just the forward kinematics, that roll is reachable by mutation, and that LEAP
-- the one hand this grammar targets -- comes out of it a legal design.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from hand_sampler import design_space as D
from hand_sampler import build, gen_init_pop, leap_fit, mutate_design, population_io
from hand_sampler import validate_design


# --- the tilt alphabet ------------------------------------------------------

def test_the_tilt_table_is_the_45_degree_grid():
    """26 directions: 6 through faces, 12 through edges, 8 through corners."""
    assert D.N_TILTS == 26
    assert D.TILTS[0] == (1.0, 0.0, 0.0), "index 0 must be straight on"
    assert len({tuple(np.round(t, 9)) for t in D.TILTS}) == 26, "a direction twice"
    for t in D.TILTS:
        assert abs(np.linalg.norm(t) - 1.0) < 1e-12
    kinds = [sum(1 for c in v if c) for v in D.TILT_LATTICE]
    assert (kinds.count(1), kinds.count(2), kinds.count(3)) == (6, 12, 8)


def test_tilt_zero_changes_nothing():
    """The default has to be free, or every hand written before tilts existed
    would move the moment the field was added."""
    assert np.allclose(D.tilt_rot(0), np.eye(3))


def test_every_tilt_points_where_it_says():
    for i in range(D.N_TILTS):
        assert np.allclose(D.tilt_rot(i)[:, 0], D.TILTS[i], atol=1e-9), i


def test_a_tilt_step_is_reversible_and_reaches_everything():
    """The operator's inverse has to be a single step back, or the walk is a
    one-way street and the ratchet argument goes with it."""
    for i, near in enumerate(D.TILT_NEIGHBOURS):
        for j in near:
            assert i in D.TILT_NEIGHBOURS[j], f"{i}->{j} does not come back"
    seen, stack = {0}, [0]
    while stack:
        for j in D.TILT_NEIGHBOURS[stack.pop()]:
            if j not in seen:
                seen.add(j)
                stack.append(j)
    assert len(seen) == D.N_TILTS, "some tilt cannot be reached from straight on"
    turns = [math.degrees(math.acos(np.clip(np.dot(D.TILTS[i], D.TILTS[j]), -1, 1)))
             for i, near in enumerate(D.TILT_NEIGHBOURS) for j in near]
    assert 35.0 < min(turns) and max(turns) < 45.1, (min(turns), max(turns))


# --- it has to reach the robot, not just the kinematics ---------------------

def _tilted_hand():
    def seg(th, phi, tilt, L):
        return D.Segment(D.Joint(th, phi, 0.0), L, tilt=tilt)
    return D.Hand(D.Palm(D.PALM_THICKNESS, 0.08, 0.08), (
        D.Finger(D.Mount("+z", 0.5, 0.5),
                 (seg(math.pi / 2, math.pi / 2, 0, 0.040),
                  seg(0.0, math.pi / 2, 7, 0.030),
                  seg(0.0, 0.0, 0, 0.025))),          # the last one is a roll joint
        D.Finger(D.Mount("-y", 0.5, 0.5),
                 (seg(0.0, math.pi / 2, 13, 0.035),
                  seg(math.pi / 2, math.pi / 2, 2, 0.020)))))


def test_a_tilted_hand_is_legal():
    assert validate_design.check(_tilted_hand()) == []


def test_the_authored_robot_agrees_with_the_kinematics_under_tilt():
    """build.link_frames composes the chain a second time, for the simulator. A
    tilt applied in one and not the other is a hand that moves differently in
    simulation than the design space says it does.
    """
    hand = _tilted_hand()
    frames = build.link_frames(hand)
    worst = 0.0
    for fi, finger in enumerate(hand.fingers):
        pts, _ = D.forward_kinematics(finger, hand.palm)
        axes = D.joint_axes(finger, hand.palm)
        for d, sg in enumerate(finger.segments):
            worst = max(worst, float(np.linalg.norm(frames[(fi, d)][:3, 3] - pts[d])))
            world = frames[(fi, d)][:3, :3] @ D.axis_of(sg.joint)
            worst = max(worst, float(np.linalg.norm(world - axes[d])))
    assert worst < 1e-12, f"authored frames drift from the kinematics by {worst}"


def test_tilt_survives_the_file():
    hand = _tilted_hand()
    assert population_io.hand_from_dict(population_io.hand_to_dict(hand)) == hand
    plain = population_io.hand_to_dict(gen_init_pop.seed_population(0, 1)[0])
    assert all("tilt" not in s for f in plain["fingers"] for s in f["segments"]), \
        "straight on is the default and should stay out of the file"


# --- the operators ----------------------------------------------------------

def test_perturb_tilt_acts_and_only_moves_tilt():
    rng = random.Random(0)
    hand = gen_init_pop.seed_population(0, 1)[0]
    acted = 0
    for _ in range(60):
        child = mutate_design.mutate(rng, hand, "perturb_tilt")
        if child is None:
            continue
        acted += 1
        moved = [(a.tilt, b.tilt)
                 for fa, fb in zip(hand.fingers, child.fingers)
                 for a, b in zip(fa.segments, fb.segments) if a.tilt != b.tilt]
        assert len(moved) == 1, f"one segment per step, not {len(moved)}"
        for fa, fb in zip(hand.fingers, child.fingers):
            assert fa.mount == fb.mount
            for a, b in zip(fa.segments, fb.segments):
                assert a.joint == b.joint and a.length == b.length
        hand = child
    assert acted > 30, f"perturb_tilt acted only {acted} times in 60"


def test_mutation_reaches_roll():
    """phi is unpinned, so a walk must be able to find a roll joint. Without it
    LEAP's thumb is unreachable."""
    rng = random.Random(3)
    hand = gen_init_pop.seed_population(0, 1)[0]
    seen_roll = seen_tilt = False
    for _ in range(4000):
        child = mutate_design.mutate(rng, hand)
        if child is None:
            continue
        hand = child
        seen_roll |= any(abs(s.joint.phi) < 1e-9
                         for f in hand.fingers for s in f.segments)
        seen_tilt |= any(s.tilt for f in hand.fingers for s in f.segments)
        if seen_roll and seen_tilt:
            break
    assert seen_roll, "4000 mutations never produced a roll joint"
    assert seen_tilt, "4000 mutations never tilted a segment"


def test_mutation_stays_closed_with_the_new_knobs():
    rng = random.Random(5)
    hand = gen_init_pop.seed_population(0, 1)[0]
    for _ in range(3000):
        child = mutate_design.mutate(rng, hand)
        if child is not None:
            assert validate_design.check(child) == [], validate_design.check(child)
            hand = child


# --- LEAP -------------------------------------------------------------------

@pytest.fixture(scope="module")
def leap():
    return leap_fit.fit()


def test_leap_is_a_legal_design(leap):
    hand, _ = leap
    assert validate_design.check(hand) == []
    assert hand.n_fingers == 4
    assert [f.n_joints for f in hand.fingers] == [4, 4, 4, 4]


def test_leap_keeps_its_own_kinematics(leap):
    """Per-digit fingertip, each digit measured from its OWN base, against the
    vendor URDF. 8.0 mm on the three fingers and 7.1 on the thumb; what is left
    is the 5 mm length grid and the 15 mm link the 20 mm floor rounds up."""
    hand, _ = leap
    ds = [leap_fit._straighten(d) for d in leap_fit.digits()]
    row, thumb = leap_fit._split(ds)
    M = leap_fit._palm_axes(row)
    for f, d in zip(hand.fingers, row + [thumb]):
        want = M.T @ (d.tip - d.pos[0])
        base, _ = D.mount_frame(f.mount, hand.palm)
        got = D.fingertip(f, hand.palm) - base
        assert float(np.linalg.norm(got - want)) * 1000 < 10.0, d.name


def test_leap_keeps_its_own_joint_axes(leap):
    """Within a degree. LEAP's axes are all coordinate directions and the
    grammar's 15 degree grid contains them exactly, so the residual is rounding
    in the vendor's own numbers rather than anything the grammar imposes."""
    hand, _ = leap
    ds = [leap_fit._straighten(d) for d in leap_fit.digits()]
    row, thumb = leap_fit._split(ds)
    M = leap_fit._palm_axes(row)
    for f, d in zip(hand.fingers, row + [thumb]):
        for got, raw in zip(D.joint_axes(f, hand.palm), d.axis):
            want = M.T @ raw
            gap = math.degrees(math.acos(float(np.clip(abs(got @ want), -1, 1))))
            assert gap < 1.5, f"{d.name}: axis {gap:.1f} deg off"


def test_leaps_links_need_no_tilt(leap):
    """The finding that made straight links the right call: once the free slides
    along each joint axis are taken out, LEAP is a collinear chain. If this ever
    fails, the hand needed a bend after all and the minimal set is wrong."""
    hand, _ = leap
    assert all(s.tilt == 0 for f in hand.fingers for s in f.segments), \
        [[s.tilt for s in f.segments] for f in hand.fingers]


def test_leaps_thumb_has_the_one_roll_joint(leap):
    hand, _ = leap
    rolls = [sum(1 for s in f.segments if abs(s.joint.phi) < 1e-9) for f in hand.fingers]
    assert rolls[:3] == [0, 0, 0], f"a row finger got a roll joint: {rolls}"
    assert rolls[3] == 1, f"the thumb should carry exactly one roll joint, got {rolls[3]}"


def test_the_fit_says_what_it_cost(leap):
    """The two prices of a BOX palm, both reported rather than hidden: the row's
    bases sit 6 mm off the midplane that every mount is pinned to, and the thumb
    lands 27 mm from where LEAP puts it, because a mount has to be on a face and
    the thumb's base is not on one."""
    _, notes = leap
    assert any("27 mm off" in n for n in notes), notes
    assert any("20 mm floor" in n for n in notes), notes
    assert not any(n.startswith("NOT A LEGAL DESIGN") for n in notes), notes
