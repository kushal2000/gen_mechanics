"""The minimal grammar: three kinds of joint and five mounting directions.

A segment is a motor bolted to the one before it. Where its shaft points is the
joint's KIND -- roll along the link, or one of the two hinges across it -- and
which way the link leaves is the LEAN, straight on or tipped 45 degrees one of
four ways. That is the whole rotational alphabet: 3 x 5 against the 44,954
spellings the same 3 degrees of freedom used to have.

There is no assembly angle. Travel is symmetric about the rest pose, and a right
angle between two links is not a bracket but an abduction joint next to a
flexion one, which is the same corner and an actuated one.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from hand_sampler import design_space as D
from hand_sampler import build, gen_init_pop, commercial, mutate_design, population_io
from hand_sampler import validate_design


# --- the alphabet -----------------------------------------------------------

def test_there_are_three_kinds_and_they_are_perpendicular():
    assert D.JOINT_KIND_NAMES == ("roll", "flexion", "abduction")
    axes = [D.axis_of(D.Joint(k)) for k in D.JOINT_KINDS]
    for a in axes:
        assert abs(np.linalg.norm(a) - 1.0) < 1e-12
    for i in range(3):
        for j in range(i + 1, 3):
            assert abs(float(axes[i] @ axes[j])) < 1e-12, "two kinds share an axis"
    # roll is the one along the link, which is +x in a joint's own frame
    assert np.allclose(D.axis_of(D.Joint(D.ROLL)), [1, 0, 0])


def test_the_lean_table_is_straight_on_plus_four_tips():
    assert D.N_LEANS == 5
    assert D.LEANS[0] == (1.0, 0.0, 0.0), "index 0 must be straight on"
    for v in D.LEANS[1:]:
        ang = math.degrees(math.acos(np.clip(np.dot(v, D.LEANS[0]), -1, 1)))
        assert abs(ang - 45.0) < 1e-9, f"a lean should tip 45 deg, not {ang}"
    assert len({tuple(np.round(v, 9)) for v in D.LEANS}) == 5


def test_lean_zero_changes_nothing():
    """The default has to be free, or every hand written before leans existed
    would move the moment the field was added."""
    assert np.allclose(D.lean_rot(0), np.eye(3))


def test_every_lean_points_where_it_says():
    for i in range(D.N_LEANS):
        assert np.allclose(D.lean_rot(i)[:, 0], D.LEANS[i], atol=1e-9), i


def test_a_lean_step_is_reversible_and_reaches_everything():
    for i, near in enumerate(D.LEAN_NEIGHBOURS):
        for j in near:
            assert i in D.LEAN_NEIGHBOURS[j], f"{i}->{j} does not come back"
    seen, stack = {0}, [0]
    while stack:
        for j in D.LEAN_NEIGHBOURS[stack.pop()]:
            if j not in seen:
                seen.add(j)
                stack.append(j)
    assert len(seen) == D.N_LEANS, "some lean cannot be reached from straight on"
    # opposite tips are 90 deg apart, which is two steps through straight on
    assert 2 not in D.LEAN_NEIGHBOURS[1] and 4 not in D.LEAN_NEIGHBOURS[3]


# --- it has to reach the robot, not just the kinematics ---------------------

def _leaning_hand():
    def seg(kind, lean, L):
        return D.Segment(D.Joint(kind), L, lean=lean)
    return D.Hand(D.Palm(D.PALM_THICKNESS), (
        D.Finger(D.Mount(0.035, math.radians(0), math.radians(0)),
                 (seg(D.FLEXION, 0, 0.040),
                  seg(D.ABDUCTION, 3, 0.030),
                  seg(D.ROLL, 0, 0.025))),
        D.Finger(D.Mount(0.035, math.radians(270), math.radians(270)),
                 (seg(D.ABDUCTION, 1, 0.035),
                  seg(D.FLEXION, 2, 0.020)))))


def test_a_leaning_hand_is_legal():
    assert validate_design.check(_leaning_hand()) == []


def test_the_authored_robot_agrees_with_the_kinematics_under_lean():
    """build.link_frames composes the chain a second time, for the simulator. A
    lean applied in one and not the other is a hand that moves differently in
    simulation than the design space says it does.
    """
    hand = _leaning_hand()
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


def test_a_hand_survives_the_file():
    hand = _leaning_hand()
    assert population_io.hand_from_dict(population_io.hand_to_dict(hand)) == hand
    plain = population_io.hand_to_dict(gen_init_pop.seed_population(0, 1)[0])
    assert all("lean" not in s for f in plain["fingers"] for s in f["segments"]), \
        "straight on is the default and should stay out of the file"
    assert plain["fingers"][0]["segments"][0]["joint"]["kind"] in D.JOINT_KIND_NAMES, \
        "a joint should read as a word, not a number"


def test_a_file_written_before_this_is_refused_with_the_conversion():
    """Silently reading theta as a kind would be a population that looks fine
    and is a different set of hands."""
    stale = {"palm": {"thickness": 0.025},
             "fingers": [{"mount": {"radius": 0.03, "bearing": 0.0, "facing": 0.0},
                          "segments": [{"joint": {"theta": 0.0, "phi": math.pi / 2},
                                        "length": 0.04}]}]}
    with pytest.raises(ValueError, match="theta"):
        population_io.hand_from_dict(stale)

    boxed = {"palm": {"thickness": 0.025, "width": 0.06, "length": 0.06},
             "fingers": []}
    with pytest.raises(ValueError, match="width"):
        population_io.hand_from_dict(boxed)

    faced = {"palm": {"thickness": 0.025},
             "fingers": [{"mount": {"face": "+z", "u": 0.5, "v": 0.5},
                          "segments": [{"joint": {"kind": "flexion"},
                                        "length": 0.04}]}]}
    with pytest.raises(ValueError, match="face"):
        population_io.hand_from_dict(faced)


# --- the operators ----------------------------------------------------------

def test_the_operator_set_is_the_minimal_one():
    assert mutate_design.OPERATORS == (
        "split_link", "merge_links", "add_finger", "remove_finger",
        "perturb_kind", "perturb_length", "move_mount", "aim_mount",
        "perturb_lean")
    assert "perturb_palm" not in mutate_design.OPERATORS, \
        "a palm has nothing left to perturb -- its outline is derived"


def test_perturb_kind_acts_and_only_moves_one_kind():
    rng = random.Random(0)
    hand = gen_init_pop.seed_population(0, 1)[0]
    acted = 0
    for _ in range(60):
        child = mutate_design.mutate(rng, hand, "perturb_kind")
        if child is None:
            continue
        acted += 1
        moved = [(a.joint.kind, b.joint.kind)
                 for fa, fb in zip(hand.fingers, child.fingers)
                 for a, b in zip(fa.segments, fb.segments) if a.joint.kind != b.joint.kind]
        assert len(moved) == 1, f"one joint per step, not {len(moved)}"
        for fa, fb in zip(hand.fingers, child.fingers):
            assert fa.mount == fb.mount
            for a, b in zip(fa.segments, fb.segments):
                assert a.length == b.length and a.lean == b.lean
        hand = child
    assert acted > 30, f"perturb_kind acted only {acted} times in 60"


def test_perturb_lean_acts_and_only_moves_one_lean():
    rng = random.Random(1)
    hand = gen_init_pop.seed_population(0, 1)[0]
    acted = 0
    for _ in range(60):
        child = mutate_design.mutate(rng, hand, "perturb_lean")
        if child is None:
            continue
        acted += 1
        moved = [(a.lean, b.lean)
                 for fa, fb in zip(hand.fingers, child.fingers)
                 for a, b in zip(fa.segments, fb.segments) if a.lean != b.lean]
        assert len(moved) == 1, f"one segment per step, not {len(moved)}"
        for fa, fb in zip(hand.fingers, child.fingers):
            for a, b in zip(fa.segments, fb.segments):
                assert a.joint == b.joint and a.length == b.length
        hand = child
    assert acted > 30, f"perturb_lean acted only {acted} times in 60"


def test_mutation_reaches_roll_and_lean():
    rng = random.Random(3)
    hand = gen_init_pop.seed_population(0, 1)[0]
    seen_roll = seen_lean = False
    for _ in range(4000):
        child = mutate_design.mutate(rng, hand)
        if child is None:
            continue
        hand = child
        seen_roll |= any(s.joint.kind == D.ROLL for f in hand.fingers for s in f.segments)
        seen_lean |= any(s.lean for f in hand.fingers for s in f.segments)
        if seen_roll and seen_lean:
            break
    assert seen_roll, "4000 mutations never produced a roll joint"
    assert seen_lean, "4000 mutations never leaned a segment"


def test_mutation_stays_closed():
    rng = random.Random(5)
    hand = gen_init_pop.seed_population(0, 1)[0]
    for _ in range(3000):
        child = mutate_design.mutate(rng, hand)
        if child is not None:
            assert validate_design.check(child) == [], validate_design.check(child)
            hand = child


def test_a_generated_hand_has_no_assembly_angle():
    """The offset survives only on MEASURED hands. If a generated one ever grows
    one, the link is being aimed by two knobs again."""
    rng = random.Random(7)
    hand = gen_init_pop.seed_population(0, 1)[0]
    for _ in range(1500):
        child = mutate_design.mutate(rng, hand)
        if child is not None:
            hand = child
    assert all(s.joint.offset == 0.0 for f in hand.fingers for s in f.segments)


# --- LEAP -------------------------------------------------------------------

@pytest.fixture(scope="module")
def leap():
    return commercial.fit()


def test_leap_is_a_legal_design(leap):
    hand, _ = leap
    assert validate_design.check(hand) == []
    assert hand.n_fingers == 4
    assert [f.n_joints for f in hand.fingers] == [4, 4, 4, 4]


def test_leaps_fingers_are_flexion_abduction_flexion_flexion(leap):
    """What LEAP is, straight off its own URDF, now readable in the genotype."""
    hand, _ = leap
    names = [[D.JOINT_KIND_NAMES[s.joint.kind] for s in f.segments] for f in hand.fingers]
    assert names[:3] == [["flexion", "abduction", "flexion", "flexion"]] * 3, names
    assert names[3].count("roll") == 1, f"the thumb should carry one roll: {names[3]}"
    assert sum(r.count("roll") for r in names[:3]) == 0, "a row finger got a roll joint"


def test_leaps_links_need_no_lean(leap):
    """The finding that made straight links the right call: once the free slides
    along each joint axis are taken out, LEAP is a collinear chain. If this ever
    fails, the hand needed a corner after all and the minimal set is wrong."""
    hand, _ = leap
    assert all(s.lean == 0 for f in hand.fingers for s in f.segments), \
        [[s.lean for s in f.segments] for f in hand.fingers]


def test_leap_keeps_its_own_kinematics(leap):
    """Per-digit fingertip, each measured from its OWN base, against the vendor."""
    hand, _ = leap
    ds = [commercial._straighten(d) for d in commercial.digits()]
    row, thumb = commercial._split(ds)
    M = commercial._palm_axes(row)
    for f, d in zip(hand.fingers, row + [thumb]):
        want = M.T @ (d.tip - d.pos[0])
        base, _ = D.mount_frame(f.mount)
        got = D.fingertip(f, hand.palm) - base
        assert float(np.linalg.norm(got - want)) * 1000 < 10.0, d.name


def test_leap_keeps_its_own_joint_axes(leap):
    hand, _ = leap
    ds = [commercial._straighten(d) for d in commercial.digits()]
    row, thumb = commercial._split(ds)
    M = commercial._palm_axes(row)
    for f, d in zip(hand.fingers, row + [thumb]):
        for got, raw in zip(D.joint_axes(f, hand.palm), d.axis):
            want = M.T @ raw
            gap = math.degrees(math.acos(float(np.clip(abs(got @ want), -1, 1))))
            assert gap < 1.5, f"{d.name}: axis {gap:.1f} deg off"


def test_the_fit_says_what_it_cost(leap):
    """The two prices of a BOX palm, both reported rather than hidden."""
    _, notes = leap
    assert any("20 mm floor" in n for n in notes), notes
    assert not any(n.startswith("NOT A LEGAL DESIGN") for n in notes), notes


def test_perturb_kind_favours_flexion():
    """The operator is biased on PURPOSE: flexion is the only kind that closes a
    hand, so it is easier to enter than to leave.

    Left to drift with no selection the kind mix settles near 20 / 45 / 35 roll
    / flexion / abduction, against 33 each if the draw were uniform. The ceiling
    is structural -- the operator must change the kind it lands on, so a flexion
    joint always leaves flexion and no weight pushes it past about 47 percent.
    """
    from collections import Counter

    rng = random.Random(0)
    hand = gen_init_pop.seed_population(0, 1)[0]
    for _ in range(300):                       # grow it so there is something to stir
        child = mutate_design.mutate(rng, hand, "split_link")
        if child:
            hand = child
        if hand.n_joints >= 12:
            break

    seen = Counter()
    for _ in range(4000):
        child = mutate_design.mutate(rng, hand, "perturb_kind")
        if child:
            hand = child
        for f in hand.fingers:
            for s in f.segments:
                seen[s.joint.kind] += 1
    total = sum(seen.values())
    share = {k: seen[k] / total for k in D.JOINT_KINDS}

    assert share[D.FLEXION] > 0.40, share
    assert share[D.FLEXION] > share[D.ABDUCTION] > share[D.ROLL], share
    # and every kind stays reachable -- a weight of zero would strand designs
    assert share[D.ROLL] > 0.10, f"roll has become unreachable in practice: {share}"


def test_the_weights_leave_nothing_unreachable():
    assert set(mutate_design.KIND_WEIGHTS) == set(D.JOINT_KINDS)
    assert all(w > 0 for w in mutate_design.KIND_WEIGHTS.values()), \
        "a zero weight would make that kind unreachable and strand any design holding one"
