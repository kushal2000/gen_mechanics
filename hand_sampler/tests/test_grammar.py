"""Properties the grammar must hold, not examples of it working."""

from __future__ import annotations

import dataclasses
import itertools
import math
import random

import numpy as np
import pytest

from hand_sampler import design_space
from hand_sampler import mutate_design
from hand_sampler import gen_init_pop
from hand_sampler import validate_design


@pytest.fixture(scope="module")
def pop():
    return gen_init_pop.seed_population(0, 120)


# --- kinematics -------------------------------------------------------------

def test_axis_endpoints():
    """There are three axes and no fourth: a joint is a kind, not an angle."""
    assert np.allclose(design_space.axis_of(design_space.Joint(design_space.FLEXION)),
                       [0, 0, 1])
    assert np.allclose(design_space.axis_of(design_space.Joint(design_space.ABDUCTION)),
                       [0, 1, 0])
    assert np.allclose(design_space.axis_of(design_space.Joint(design_space.ROLL)),
                       [1, 0, 0])


def test_axis_is_unit():
    for k in design_space.JOINT_KINDS:
        assert abs(np.linalg.norm(design_space.axis_of(design_space.Joint(k))) - 1) < 1e-12


def test_mount_frame_orthonormal_in_every_direction():
    """A palm is a disc, so a finger can leave in any of the 24 directions on
    the angle grid -- including the two where it runs along GRASP_DIR and the
    old face-tangent construction degenerated."""
    for b in range(0, 360, 15):
        _, R = design_space.mount_frame(
            design_space.Mount.polar(0.040, math.radians(b), math.radians(b)))
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9), b
        assert abs(np.linalg.det(R) - 1.0) < 1e-9, b
def test_seeds_valid(pop):
    assert all(validate_design.is_valid(h) for h in pop)


def test_seeds_are_simple(pop):
    """Generation 0 starts at the conventional corner."""
    assert max(h.n_joints for h in pop) <= 4
    assert all(h.n_fingers == 2 for h in pop)
    assert all(s.joint.kind in (design_space.FLEXION, design_space.ABDUCTION)
               for h in pop for f in h.fingers for s in f.segments), \
        "generation 0 draws hinges; a roll joint on a seed is a wasted motor"
    assert all(s.lean == 0 for h in pop for f in h.fingers for s in f.segments)


# --- operators --------------------------------------------------------------

def test_every_operator_can_act(pop):
    rng = random.Random(1)
    # remove_finger needs a hand above MIN_FINGERS carrying a single-joint finger; a seed is at...
    grown = [c for h in pop[:40] if (c := mutate_design.mutate(rng, h, "add_finger"))]
    for op in mutate_design.OPERATORS:
        source = grown if op == "remove_finger" else pop[:40]
        assert any(mutate_design.mutate(rng, h, op) for h in source), op


def test_mutation_is_closed(pop):
    """Every operator returns a valid hand or raises."""
    rng = random.Random(2)
    h = pop[0]
    for _ in range(3000):
        c = mutate_design.mutate(rng, h)
        if c is not None:
            assert validate_design.is_valid(c), validate_design.check(c)
            h = c


def test_split_link_preserves_reach(pop):
    """Splitting divides a link and merging restores it, so reach is unchanged -- which is..."""
    rng = random.Random(3)
    for h in pop[:60]:
        c = mutate_design.mutate(rng, h, "split_link")
        if c is not None:
            assert sum(f.reach for f in c.fingers) == pytest.approx(
                sum(f.reach for f in h.fingers), abs=1e-12)


@pytest.mark.parametrize("add,remove", [("split_link", "merge_links"),
                                        ("add_finger", "remove_finger")])
def test_exact_inverse(pop, add, remove):
    """Add then remove must return the ORIGINAL hand, not a nearby one."""
    rng = random.Random(4)
    tried = recovered = 0
    for h in pop[:60]:
        child = mutate_design.mutate(rng, h, add)
        if child is None:
            continue
        tried += 1
        back = random.Random(99)
        if any(mutate_design.mutate(back, child, remove) == h for _ in range(60)):
            recovered += 1
    assert tried > 0
    assert recovered == tried, f"{recovered}/{tried} recovered"


def _grow(h):
    """Room for the fingers the test is about to add."""
    from dataclasses import replace
    return replace(h, palm=design_space.Palm(design_space.PALM_THICKNESS))


def _balance(pop, target, seed=0):
    """(P(up), add_finger success rate) over structural moves from n >= target."""
    def at_least(h, t, s):
        rng = random.Random(s)
        for _ in range(8000):
            if h.n_joints >= t:
                return h
            c = mutate_design.mutate(
                rng, h, mutate_design.STRUCTURAL[rng.randrange(len(mutate_design.STRUCTURAL))])
            if c:
                h = c
        return h

    rng = random.Random(seed)
    st = mutate_design.Stats()
    up = down = 0
    for i in range(25):
        h = at_least(pop[i], target, i)
        assert h.n_joints >= target, f"could not reach n={target}"
        for _ in range(80):
            c = mutate_design.mutate(
                rng, h,
                mutate_design.STRUCTURAL[rng.randrange(len(mutate_design.STRUCTURAL))],
                stats=st)
            if c is None:
                continue
            up, down = (up + 1, down) if c.n_joints > h.n_joints else (up, down + 1)
    assert up + down > 200, f"too few moves at n={target}"
    return up / (up + down), st.rate("add_finger")


def test_operators_are_unbiased(pop):
    """Per move, complexity must be as likely to fall as to rise.

    Measured on a palm with room to spare. This test only applies STRUCTURAL
    operators, so the palm never grows -- and on a seed-sized palm the finger row
    fills at around three fingers, after which ``add_finger`` simply cannot act
    and the balance reads 33% at n=6 and 21% at n=10. That is the palm being
    full, not the operators being biased, and the next test asserts it directly.
    """
    for target in (4, 6):
        p_up, _ = _balance([_grow(h) for h in pop], target)
        assert 0.40 < p_up < 0.60, f"n={target}: P(up) = {p_up:.1%}"


def test_the_finger_cap_is_what_stops_complexity(pop):
    """Complexity stops rising at the envelope cap, not for want of room.

    A stored palm had a width and a length, so a seed-sized one filled at about
    three fingers and MIN_MOUNT_SEPARATION was what stopped a hand growing. A
    radial palm has no size of its own -- it is the hull of wherever the fingers
    went -- so what bounds complexity is the ANNULUS a mount may live in, and
    that holds twelve mounts at the separation floor against a cap of six.
    MAX_FINGERS binds first and geometry never gets the chance.
    """
    q, a = design_space.PALM_QUANTUM, design_space.ANGLE_QUANTUM
    sites = []
    for i in range(int(round((design_space.MAX_MOUNT_RADIUS
                              - design_space.PALM_MIN_RADIUS) / q)) + 1):
        r = design_space.PALM_MIN_RADIUS + i * q
        for k in range(int(round(2 * math.pi / a))):
            b = k * a
            here = np.array([r * math.sin(b), r * math.cos(b)])
            if all(float(np.linalg.norm(here - s))
                   >= design_space.MIN_MOUNT_SEPARATION - 1e-12 for s in sites):
                sites.append(here)
    assert len(sites) > design_space.MAX_FINGERS, (
        f"the annulus holds {len(sites)} mounts against a cap of "
        f"{design_space.MAX_FINGERS}; if this ever inverts, geometry is the "
        f"bound again and the balance deviations must be re-read")

    # and the choke on add_finger at high complexity is the cap, not placement:
    # hands that still have room take a finger readily, hands at six never can.
    low_p, low_add = _balance(pop, 4)
    high_p, high_add = _balance(pop, 14)
    assert low_add > 0.80, f"add_finger should be easy at n=4, acted {low_add:.0%}"
    assert high_add < low_add / 2.0, (
        f"add_finger should choke as hands fill: {low_add:.0%} -> {high_add:.0%}")
    assert high_p < low_p, f"P(up) should sag at the cap: {low_p:.0%} -> {high_p:.0%}"


def test_deep_fingers_are_reachable(pop):
    """Depth must be reachable, not merely slower."""
    rng = random.Random(0)
    best = 0
    for s in range(12):
        h = pop[s]
        for _ in range(4000):
            c = mutate_design.mutate(rng, h, "split_link") or mutate_design.mutate(rng, h, "perturb_length")
            if c:
                h = c
        best = max(best, max(f.n_joints for f in h.fingers))
    assert best == design_space.MAX_JOINTS_PER_FINGER, f"deepest finger reached was {best}"

def test_identical_hands_compare_equal(pop):
    """Fitness memoisation and exact inverses both depend on this."""
    a = pop[0]
    b = design_space.Hand(palm=design_space.Palm(a.palm.thickness), fingers=a.fingers)
    assert a == b and hash(a) == hash(b)
def _closest_approach(hand, starts=4):
    """How near two fingertips can be brought, over all joint angles."""
    from scipy.optimize import minimize

    lo, hi = design_space.JOINT_LIMIT
    counts = [f.n_joints for f in hand.fingers]
    total = sum(counts)

    def separation(x):
        tips, k = [], 0
        for f, n in zip(hand.fingers, counts):
            tips.append(design_space.fingertip(f, hand.palm, {i: x[k + i] for i in range(n)}))
            k += n
        return min(float(np.linalg.norm(p - q))
                   for p, q in itertools.combinations(tips, 2))

    rng = random.Random(0)
    return min(float(minimize(separation,
                              [rng.uniform(lo, hi) for _ in range(total)],
                              bounds=[(lo, hi)] * total, method="L-BFGS-B").fun)
               for _ in range(starts))


def test_seeds_give_a_gradient_to_select_on(pop):
    """Most of generation 0 must be able to touch the object -- not all of it."""
    closes = [_closest_approach(h) < 0.040 for h in pop[:60]]
    rate = sum(closes) / len(closes)
    assert rate > 0.60, (
        f"only {rate:.0%} of seeds can reach the object; below this there is too "
        f"little signal to select on (the opposite-face regression sat at 42%)")

    # and the failures should be the cheap designs, not scattered at random
    by_motors = {}
    for h, ok in zip(pop[:60], closes):
        by_motors.setdefault(h.n_motors, []).append(ok)
    if 2 in by_motors and 4 in by_motors:
        cheap = sum(by_motors[2]) / len(by_motors[2])
        rich = sum(by_motors[4]) / len(by_motors[4])
        assert rich >= cheap, (
            f"4-motor seeds close {rich:.0%} of the time against 2-motor seeds' "
            f"{cheap:.0%}; closure should improve with motors, not worsen")

def _hand(*fingers):
    """Two fingers on adjacent faces, on a palm wide enough to separate them.

    80 mm leaves the +y and +z mounts 33.9 mm apart, which cleared the old
    15 mm across-face floor and does not clear MIN_MOUNT_SEPARATION."""
    return design_space.Hand(
        design_space.Palm(design_space.PALM_THICKNESS), tuple(fingers))


_KINDS = (design_space.FLEXION, design_space.ABDUCTION)


def _finger(bearing_deg, lengths, radius=0.040):
    return design_space.Finger(
        design_space.Mount.polar(radius, math.radians(bearing_deg) % (2 * math.pi),
                           math.radians(bearing_deg) % (2 * math.pi)),
        tuple(design_space.Segment(design_space.Joint(_KINDS[i % 2]), L)
              for i, L in enumerate(lengths)))


@pytest.mark.parametrize("lengths,note", [
    ([0.040, 0.040], "ordinary two-joint finger"),
    ([0.020, 0.020], "both links at MIN_LINK_LENGTH"),
    ([0.020, 0.075], "very uneven links"),
    ([0.045, 0.040], "merge overflows MAX_LINK_LENGTH by one quantum"),
    ([0.080, 0.080], "both links already at MAX_LINK_LENGTH"),
    ([0.045, 0.045, 0.030], "interior overflow -- distal neighbour fits"),
    ([0.080, 0.080, 0.040], "every merge overflows; the clamp is the only path"),
])
def test_merge_links_handles_every_merge_case(lengths, note):
    """A joint must be removable whatever the link lengths around it."""
    hand = _hand(_finger(90, lengths), _finger(0, [0.050]))
    assert validate_design.is_valid(hand), validate_design.check(hand)

    rng = random.Random(0)
    seen = set()
    for _ in range(200):
        child = mutate_design.apply(rng, hand, "merge_links")       # must never raise
        assert child.n_joints == hand.n_joints - 1, "not a unit step"
        assert validate_design.is_valid(child), validate_design.check(child)
        seen.add(tuple(f.segments for f in child.fingers))
    assert len(seen) >= 2, "every joint in the finger should be removable"


def test_structural_removal_refuses_at_the_floor():
    """MIN_FINGERS single-joint fingers is the floor: neither removal can act."""
    hand = _hand(_finger(90, [0.050]), _finger(0, [0.050]))
    assert hand.n_fingers == design_space.MIN_FINGERS
    rng = random.Random(0)
    for op in ("merge_links", "remove_finger"):
        for _ in range(25):
            with pytest.raises(mutate_design.MutationImpossible):
                mutate_design.apply(rng, hand, op)


def test_merge_links_preserves_reach_unless_it_must_clamp():
    """Reach is preserved on every path split_link can produce; the clamp is unreachable..."""
    hand = _hand(_finger(90, [0.040, 0.040]), _finger(0, [0.050]))
    rng = random.Random(0)
    before = sum(f.reach for f in hand.fingers)
    for _ in range(100):
        child = mutate_design.apply(rng, hand, "merge_links")
        assert sum(f.reach for f in child.fingers) == pytest.approx(before, abs=1e-12)


def test_one_joint_per_link():
    """No two joints share a point."""
    palm = design_space.Palm(design_space.PALM_THICKNESS)
    bad = design_space.Finger(design_space.Mount.polar(0.039, math.radians(90), math.radians(90)),
                   (design_space.Segment(design_space.Joint(design_space.FLEXION), 0.0),
                    design_space.Segment(design_space.Joint(design_space.FLEXION), 0.040)))
    assert validate_design.check_finger(bad, 0, palm), "a zero-length link must be rejected"

    rng = random.Random(1)
    hand = gen_init_pop.seed_population(0, 1)[0]
    for _ in range(4000):
        child = mutate_design.mutate(rng, hand)
        if child:
            hand = child
        for f in hand.fingers:
            # the floor is per position: the last link of a finger drives no
            # child joint, so it carries no motor and has its own lower bound
            for i, sg in enumerate(f.segments):
                floor = (design_space.MIN_DISTAL_LINK_LENGTH
                         if i == f.n_joints - 1 else design_space.MIN_LINK_LENGTH)
                assert sg.length >= floor - 1e-9

    # distinct joint positions, which is the geometric statement of the rule
    for f in hand.fingers:
        joints, _ = design_space.forward_kinematics(f, hand.palm)
        floors = ([design_space.MIN_LINK_LENGTH] * (f.n_joints - 1)
                  + [design_space.MIN_DISTAL_LINK_LENGTH])
        for a, b, floor in zip(joints, joints[1:], floors):
            assert np.linalg.norm(a - b) >= floor - 1e-9


def test_capsules_carry_their_segment_index():
    """Each capsule reports which segment it belongs to."""
    palm = design_space.Palm(design_space.PALM_THICKNESS)
    finger = design_space.Finger(design_space.Mount.polar(0.039, math.radians(90), math.radians(90)),
                      tuple(design_space.Segment(design_space.Joint(_KINDS[i % 2]), 0.030)
                            for i in range(3)))
    _, capsules = design_space.forward_kinematics(finger, palm)
    assert [c[3] for c in capsules] == [0, 1, 2]


def test_joint_axes_are_the_axes_the_joints_turn_about():
    """The viewer draws each joint as a cylinder along its reported axis, so the axis has..."""
    palm = design_space.Palm(design_space.PALM_THICKNESS)
    finger = design_space.Finger(design_space.Mount.polar(0.037, math.radians(90), math.radians(90)),
                      (design_space.Segment(design_space.Joint(design_space.FLEXION), 0.035),
                       design_space.Segment(design_space.Joint(design_space.ABDUCTION), 0.030,
                                            lean=3),
                       design_space.Segment(design_space.Joint(design_space.ROLL), 0.025)))
    rest, _ = design_space.forward_kinematics(finger, palm)
    axes = design_space.joint_axes(finger, palm)
    assert len(axes) == finger.n_joints
    assert all(abs(np.linalg.norm(a) - 1.0) < 1e-12 for a in axes)

    delta = 0.4
    for k, a in enumerate(axes):
        moved, _ = design_space.forward_kinematics(finger, palm, {k: delta})
        R = design_space.rodrigues(a, delta)
        for j in range(k + 1, len(rest)):
            assert np.allclose(moved[j], rest[k] + R @ (rest[j] - rest[k]), atol=1e-12)
        # a rotation fixes its own axis, so the marker is stable under its slider
        assert np.allclose(design_space.joint_axes(finger, palm, {k: delta})[k], a, atol=1e-12)


def test_links_do_not_overlap_at_rest(pop):
    """No two links may intersect at rest -- distal ones included.

    Measured on the AUTHORED capsule axis, which is inset one radius at each end
    because a capsule's tip-to-tip extent is the link length. This used to
    compare the full mount-to-tip span, treating every link as 2 * CAPSULE_RADIUS
    longer than the one the simulator builds; that is stricter, but strict about
    a link that does not exist, and it rejected 18.9% of a drifted population for
    contacts that never happen.
    """
    rng = random.Random(4)
    hand = pop[0]
    r = design_space.CAPSULE_RADIUS
    worst = float("inf")
    for _ in range(4000):
        child = mutate_design.mutate(rng, hand)
        if child:
            hand = child
        links = design_space.rest_capsules(hand)
        for (fi, si, a0, a1), (fj, sj, b0, b1) in itertools.combinations(links, 2):
            if fi == fj and abs(si - sj) <= 1:
                continue          # consecutive links meet at their shared joint
            worst = min(worst, design_space.segment_distance(a0, a1, b0, b1))
    assert worst >= 2 * r - 1e-9, (
        f"links came within {worst * 1000:.1f} mm; capsules intersect below "
        f"{2 * r * 1000:.0f} mm")


def test_mounts_keep_their_distance(pop):
    """One separation floor, whatever faces the two mounts are on.

    Was two rules, a loose across-face one and a tighter same-face one. They
    collapsed into MIN_MOUNT_SEPARATION = 2r + 5 mm once the capsule grew: below
    2r two parallel fingers interpenetrate outright, so the looser floor was not
    loose, it was wrong."""
    rng = random.Random(7)
    hand = pop[0]
    for _ in range(3000):
        child = mutate_design.mutate(rng, hand)
        if child:
            hand = child
        pos = [design_space.mount_position(f.mount) for f in hand.fingers]
        for pa, pb in itertools.combinations(pos, 2):
            d = float(np.linalg.norm(pa - pb))
            assert d >= design_space.MIN_MOUNT_SEPARATION - 1e-9, (
                f"two mounts {d * 1000:.1f} mm apart")

def test_crowding_does_not_block_new_fingers():
    """Placement must keep finding room until the cap, and must still respect
    the separation floor when it does.

    There is no small palm to contrast against: the annulus a mount may live in
    is one global ring, so every hand has the same room. What is left to check
    is that placement does not give up early, and that reaching the cap did not
    come of ignoring MIN_MOUNT_SEPARATION or putting a finger in the arm.
    """
    hand = gen_init_pop.seed_population(0, 1)[0]
    for k in range(30):
        if hand.n_fingers >= design_space.MAX_FINGERS:
            break
        out = mutate_design._new_finger(random.Random(k), hand)
        if out is None:
            break
        hand = out

    assert hand.n_fingers == design_space.MAX_FINGERS, (
        f"placement gave up at {hand.n_fingers} fingers with room left")
    assert validate_design.check(hand) == [], validate_design.check(hand)
    for a, b in itertools.combinations(hand.fingers, 2):
        d = float(np.linalg.norm(design_space.mount_position(a.mount)
                                 - design_space.mount_position(b.mount)))
        assert d >= design_space.MIN_MOUNT_SEPARATION - 1e-9, f"{d*1000:.1f} mm apart"


def test_min_link_length_allows_a_compact_knuckle():
    """Two axes may sit closer than a link's own diameter."""
    assert design_space.MIN_LINK_LENGTH < 2 * design_space.CAPSULE_RADIUS

    palm = design_space.Palm(design_space.PALM_THICKNESS)
    finger = design_space.Finger(design_space.Mount.polar(0.039, math.radians(90), math.radians(90)), (
        design_space.Segment(design_space.Joint(design_space.FLEXION),
                             design_space.MIN_LINK_LENGTH),
        design_space.Segment(design_space.Joint(design_space.ABDUCTION), 0.040)))
    assert not validate_design.check_finger(finger, 0, palm), validate_design.check_finger(finger, 0, palm)

    joints, _ = design_space.forward_kinematics(finger, palm)
    gap = float(np.linalg.norm(joints[1] - joints[0]))
    assert gap == pytest.approx(design_space.MIN_LINK_LENGTH)
    assert gap < 2 * design_space.CAPSULE_RADIUS, "axes are not closer than the link is wide"

def _distance_to_outline(ring, point):
    """Shortest distance from ``point`` to the closed polygon ``ring``.

    To the EDGES, not to the vertices: a vertex is as far as 0.1 mm from the
    true outline at the drawing resolution, and a nearest-vertex distance would
    report that gap as a margin the palm does not have.
    """
    a, b = ring, np.roll(ring, -1, axis=0)
    e = b - a
    t = np.clip(((point - a) * e).sum(axis=1)
                / np.maximum((e * e).sum(axis=1), 1e-18), 0.0, 1.0)
    return float(np.min(np.linalg.norm(a + t[:, None] * e - point, axis=1)))


def test_every_mount_stays_inside_the_palm():
    """What replaced test_mounts_stay_clear_of_face_edges.

    There are no faces and no edge margin: the palm is the hull of its own disc
    and of a PALM_RIM circle about each mount, so every mount is exactly one rim
    inside the outline by construction. This checks the construction holds under
    drift rather than checking a margin that no longer exists.

    The rim is also what stops the plate from burying the first link, so how far
    the outline reaches matters as much: it must stop one rim past the outermost
    mount, and never grow out over its own fingers the way it used to. A mount
    further in than the rim is fine and says nothing -- it is simply not on the
    hull, which some thumb always is not.
    """
    rng = random.Random(4)
    hand = gen_init_pop.seed_population(0, 1)[0]
    sag = 1e-4                        # the hull of 32 samples is inscribed
    worst, overgrown = float("inf"), 0.0
    for _ in range(1500):
        child = mutate_design.mutate(rng, hand)
        if child:
            hand = child
        ring = design_space.palm_outline(hand)
        for f in hand.fingers:
            here = design_space.mount_position(f.mount)[1:]
            worst = min(worst, _distance_to_outline(ring, here))
        reach = np.linalg.norm(ring - design_space.PALM_CENTRE[1:], axis=1).max()
        room = max(max(f.mount.radius for f in hand.fingers),
                   design_space.PALM_MIN_RADIUS) + design_space.PALM_RIM
        overgrown = max(overgrown, float(reach) - room)
    assert worst >= design_space.PALM_RIM - sag, (
        f"a mount came {worst * 1000:.2f} mm from the palm edge, inside the "
        f"{design_space.PALM_RIM * 1000:.0f} mm rim")
    assert overgrown <= sag, (
        f"the outline reached {overgrown * 1000:.1f} mm past the last mount's "
        f"own rim: the palm is growing out over its fingers again")


def test_move_mount_reaches_all_the_way_round_the_palm():
    """What replaced test_move_mount_still_crosses_faces_with_a_margin.

    There are no faces to cross, and no wedge left to go round either: a bearing
    wraps, so a finger walks the whole circle. What stops it behind the hand is
    not the bearing but the arm, and only when the finger actually reaches it --
    which is why a mount directly behind the palm is legal at a small radius.
    """
    rng = random.Random(8)
    hand = gen_init_pop.seed_population(0, 1)[0]
    seen = set()
    for _ in range(4000):
        child = mutate_design.mutate(rng, hand, "move_mount")
        if child is None:
            continue
        hand = child
        for f in hand.fingers:
            seen.add(round(math.degrees(f.mount.bearing)))
    assert len(seen) >= 12, f"only reached {len(seen)} bearings: {sorted(seen)}"
    assert any(135 <= b <= 225 for b in seen), (
        f"never reached a bearing behind the hand: {sorted(seen)} -- the wedge "
        f"that used to forbid those is gone")


def test_no_mutation_walk_can_put_a_hand_in_the_arm():
    """The regression this rule exists for.

    Seeds were always clean -- they are drawn pointing forward -- so nothing
    caught that 68% of the hands a walk reaches had something inside the arm,
    as deep as 178 mm. One step from a LEGAL parent the rate is only 5.0%, which
    is why the validator alone is enough: every operator already shuffles its
    candidates and takes the first that validates, so the rule turns a bad draw
    into a different draw rather than into a failure.

    add_finger is the one that needs it most, at 31.7% against 4% or less for
    everything else -- _free_mount_sites offers the whole ring, and a finger
    grown backward off the far rim lands in the arm. It is also the operator the
    complexity ratchet leans on, so the walk here mutates freely rather than
    picking operators.
    """
    rng = random.Random(3)
    deepest, where = float("inf"), None
    for s in range(12):
        hand = gen_init_pop.seed_population(s, 1)[0]
        for _ in range(400):
            child = mutate_design.mutate(rng, hand)
            if child is None or validate_design.check(child):
                continue
            hand = child
            z, who = design_space.rearmost(hand)
            if z < deepest:
                deepest, where = z, who
    # The validator's own tolerance: a hand CAN sit exactly on the face, since
    # the outer ring is tangent to it, and a bare >= then turns 5e-17 m of
    # float noise into a failure.
    assert deepest >= design_space.ARM_FACE_Z - 1e-9, (
        f"a legal hand reached z = {deepest*1000:.0f} mm, "
        f"{(design_space.ARM_FACE_Z - deepest)*1000:.0f} mm inside the arm, "
        f"owned by {'the palm' if where is None else f'finger {where}'}")


def test_a_palm_has_nothing_left_to_perturb():
    """What replaced test_perturb_palm_leaves_thickness_alone.

    The operator is gone because the thing it moved is gone: a palm stores only
    its thickness, which was never mutable, and its outline is derived from
    where the fingers start. move_mount changes the palm now, by changing them.
    """
    assert "perturb_palm" not in mutate_design.OPERATORS
    assert [f.name for f in dataclasses.fields(design_space.Palm)] == ["thickness"]

    rng = random.Random(10)
    hand = gen_init_pop.seed_population(0, 1)[0]
    before = design_space.palm_extents(hand)
    for _ in range(400):
        child = mutate_design.mutate(rng, hand, "move_mount")
        if child:
            hand = child
    assert hand.palm.thickness == design_space.PALM_THICKNESS
    assert design_space.palm_extents(hand) != before, \
        "moving a mount should have reshaped the palm"


def test_a_mount_means_the_same_place_on_any_palm():
    """What replaced test_check_finger_needs_the_real_palm.

    A mount used to be (face, u, v) -- an offset along a box -- so the same
    triple landed elsewhere on a palm of a different size. It is an offset from
    the palm's own centre now, and a palm has no size of its own to resolve
    against, so a mount means one place and finger legality never consults the
    palm.
    """
    mount = design_space.Mount(y=0.020, z=0.035, facing=math.radians(30))
    # Offset from PALM_CENTRE, which is NOT the frame origin: the frame's origin
    # is where the arm bolts on, and the fingers are centred WRIST_STANDOFF
    # forward of it so the hand clears the arm.
    assert np.allclose(design_space.mount_position(mount),
                       design_space.PALM_CENTRE + [0.0, 0.020, 0.035])
    # and Mount.polar is only a way of SAYING the same place
    assert design_space.Mount.polar(0.040, math.radians(30),
                                    math.radians(30)) == mount
    thin, thick = design_space.PALM_THICKNESS_RANGE
    for t in (thin, thick):
        hand = design_space.Hand(design_space.Palm(t), (design_space.Finger(
            mount, (design_space.Segment(design_space.Joint(design_space.FLEXION),
                                         0.040),)),))
        assert np.allclose(design_space.mount_position(hand.fingers[0].mount),
                           design_space.mount_position(mount))
    assert not validate_design.check_finger(design_space.Finger(
        mount, (design_space.Segment(design_space.Joint(design_space.FLEXION),
                                     0.040),)), 0, design_space.Palm(thin))


def test_require_valid_reports_every_reason():
    """The loud counterpart to is_valid."""
    palm = design_space.Palm(design_space.PALM_THICKNESS)
    good = gen_init_pop.seed_population(0, 1)[0]
    assert validate_design.require_valid(good) is good

    # one fault of each kind: a base off the position grid, and a link far over
    # MAX_LINK_LENGTH. Both must be named, not just the first one found.
    off_grid = design_space.Mount(y=0.0412, z=0.0, facing=math.radians(90))
    bad = design_space.Hand(palm, (
        design_space.Finger(off_grid,
                            (design_space.Segment(
                                design_space.Joint(design_space.FLEXION), 0.5),)),
        design_space.Finger(design_space.Mount.polar(0.040, 0.0, 0.0),
                            (design_space.Segment(
                                design_space.Joint(design_space.FLEXION), 0.040),))))
    with pytest.raises(ValueError) as e:
        validate_design.require_valid(bad)
    assert "length" in str(e.value) and "mount" in str(e.value), str(e.value)


def test_segment_distance_matches_brute_force():
    """The closed form must never OVERESTIMATE -- the dangerous direction, since an..."""
    rng = np.random.default_rng(0)
    for _ in range(400):
        p0, p1, q0, q1 = (rng.normal(size=3) for _ in range(4))
        closed = design_space.segment_distance(p0, p1, q0, q1)
        ts = np.linspace(0, 1, 160)
        a = p0 + np.outer(ts, p1 - p0)
        b = q0 + np.outer(ts, q1 - q0)
        brute = float(np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)))
        assert closed <= brute + 1e-9, f"overestimated: {closed} > {brute}"
        assert closed >= brute - 1e-2, f"underestimated badly: {closed} vs {brute}"


@pytest.mark.parametrize("p0,p1,q0,q1,want", [
    ((0, 0, 0), (0, 0, 0), (1, 1, 1), (1, 1, 1), math.sqrt(3)),   # point vs point
    ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), 1.0),            # parallel
    ((0, 0, 0), (1, 0, 0), (0, 0, 0), (1, 0, 0), 0.0),            # identical
    ((0, 0, 0), (1, 0, 0), (1, 0, 0), (2, 0, 0), 0.0),            # tip to tip
    ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), 1.0),            # collinear gap
    ((0, 0, 0), (1, 0, 0), (0.5, -1, 1), (0.5, 1, 1), 1.0),       # crossing, offset
])
def test_segment_distance_degenerate_cases(p0, p1, q0, q1, want):
    got = design_space.segment_distance(*(np.array(x, float) for x in (p0, p1, q0, q1)))
    assert got == pytest.approx(want, abs=1e-9)


def test_move_mount_keeps_a_finger_on_the_palm_plane():
    """What replaced test_move_mount_slides_along_the_face_and_never_off_the_midplane.

    A mount is polar now, so there is no midplane to fall off -- x is zero by
    construction. What is worth checking is that move_mount changes only WHERE a
    finger sits, never which way it points, which is aim_mount's job.
    """
    rng = random.Random(6)
    hand = gen_init_pop.seed_population(0, 1)[0]
    moved = 0
    for _ in range(300):
        child = mutate_design.mutate(rng, hand, "move_mount")
        if child is None:
            continue
        moved += 1
        for a, b in zip(hand.fingers, child.fingers):
            assert a.mount.facing == b.mount.facing, "move_mount turned a finger"
            assert design_space.mount_position(b.mount)[0] == 0.0
        hand = child
    assert moved > 100, f"move_mount only acted {moved} times in 300"

def test_every_genotype_field_is_validated():
    """No field may go out of bounds unnoticed."""
    from dataclasses import replace as _replace

    hand = gen_init_pop.seed_population(0, 1)[0]
    f0, palm = hand.fingers[0], hand.palm

    def variant(**kw):
        mount = _replace(f0.mount, **{k[2:]: v for k, v in kw.items()
                                      if k.startswith("m_")})
        seg = f0.segments[0]
        if "s_len" in kw:
            seg = _replace(seg, length=kw["s_len"])
        if "s_kind" in kw:
            seg = _replace(seg, joint=_replace(seg.joint, kind=kw["s_kind"]))
        if "s_lean" in kw:
            seg = _replace(seg, lean=kw["s_lean"])
        if "s_offset" in kw:
            seg = _replace(seg, joint=_replace(seg.joint, offset=kw["s_offset"]))
        new_palm = _replace(palm, **{k[2:]: v for k, v in kw.items()
                                     if k.startswith("p_")})
        return design_space.Hand(new_palm,
                      (_replace(f0, mount=mount, segments=(seg,) + f0.segments[1:]),)
                      + hand.fingers[1:])

    cases = {
        "palm thickness off grid": variant(p_thickness=0.0231),
        "mount past the ring": variant(m_y=0.200, m_z=0.0),
        "mount inside the palm's own disc": variant(m_y=0.010, m_z=0.005),
        "mount y off grid": variant(m_y=0.0412),
        "mount z off grid": variant(m_z=0.0412),
        "mount facing off grid": variant(m_facing=0.1),
        # No wedge any more; what is illegal behind the hand is reaching the
        # ARM, which takes the far rim and a long link pointed back at it.
        "finger reaching into the arm": variant(
            m_y=0.0, m_z=-design_space.MAX_MOUNT_RADIUS, m_facing=math.pi,
            s_len=design_space.MAX_LINK_LENGTH),
        "link too short": variant(s_len=0.010),
        "link too long": variant(s_len=0.200),
        "link off grid": variant(s_len=0.0431),
        # a generated joint has no assembly angle at all, so ANY offset is a fault
        "offset on a generated joint": variant(s_offset=math.radians(15)),
        "too many fingers": design_space.Hand(palm, hand.fingers * 4),
        "too few fingers": design_space.Hand(palm, hand.fingers[:1]),
    }
    missed = [name for name, bad in cases.items() if not validate_design.check(bad)]
    assert not missed, f"validator missed: {missed}"


def test_the_grammar_is_deterministic():
    """Same seed, same population and same walk -- the basis for a design being..."""
    assert gen_init_pop.seed_population(7, 20) == gen_init_pop.seed_population(7, 20)

    def walk(seed):
        rng = random.Random(seed)
        hand = gen_init_pop.seed_population(0, 1)[0]
        for _ in range(400):
            child = mutate_design.mutate(rng, hand)
            if child:
                hand = child
        return hand

    assert walk(3) == walk(3)
    assert walk(3) != walk(4)


def test_a_joint_kind_and_a_lean_do_different_things():
    """What replaced test_offset_subsumes_a_mount_pointing_direction and
    test_offset_moves_the_link_but_theta_does_not.

    Both were about the zero offset, which is gone: a generated joint has no
    assembly angle, and what used to aim a link is the LEAN. The property worth
    keeping is that the two knobs that remain are not two spellings of one
    thing. A kind turns the hinge and leaves the link where it was; a lean turns
    the link and leaves the hinge pointing the same way relative to it.
    """
    palm = design_space.Palm(design_space.PALM_THICKNESS)

    def tip(kind, lean):
        f = design_space.Finger(
            design_space.Mount.polar(0.035, math.radians(90), math.radians(90)),
            (design_space.Segment(design_space.Joint(kind), 0.050, lean=lean),))
        return design_space.fingertip(f, palm)

    # the kind never moves the link at rest -- only which way it will turn
    assert np.allclose(tip(design_space.FLEXION, 0), tip(design_space.ABDUCTION, 0))
    assert np.allclose(tip(design_space.FLEXION, 0), tip(design_space.ROLL, 0))
    # the lean does move it, and by the 45 degrees it says
    for lean in range(1, design_space.N_LEANS):
        assert not np.allclose(tip(design_space.FLEXION, 0),
                               tip(design_space.FLEXION, lean))

    # and the hinge follows the link it is bolted to -- EXCEPT when the lean
    # turns about the hinge's own axis, which leaves it pointing where it was.
    # Flexion's axis is +z, and lean 1 tips toward +y by rotating about +z.
    def hinge(lean):
        f = design_space.Finger(
            design_space.Mount.polar(0.035, math.radians(90), math.radians(90)),
            (design_space.Segment(design_space.Joint(design_space.FLEXION),
                                  0.050, lean=lean),))
        return design_space.joint_axes(f, palm)[0]

    def gap(a, b):
        return math.degrees(math.acos(float(np.clip(abs(a @ b), -1, 1))))

    assert gap(hinge(0), hinge(1)) == pytest.approx(0.0, abs=1e-6), \
        "a lean about the hinge's OWN axis should leave the hinge alone"
    assert gap(hinge(0), hinge(3)) == pytest.approx(45.0, abs=1e-6), \
        "a lean across the hinge should carry it by the 45 degrees it tips"


# --- one representation for generated and measured hands --------------------

SHARPA_URDF = ("assets/urdf/kuka_sharpa_description/"
               "iiwa14_left_sharpa_adjusted_restricted.urdf")


def _sharpa_joint_names():
    """SHARPA's controlled hand joints, in spec order, read from the URDF."""
    import xml.etree.ElementTree as ET
    from hand_sampler import resolve
    root = ET.parse(resolve(SHARPA_URDF)).getroot()
    return [j.get("name") for j in root.findall("joint")
            if j.get("type") == "revolute" and "iiwa" not in j.get("name")]


def test_sharpa_imports_as_a_hand():
    names = _sharpa_joint_names()
    hand = design_space.hand_from_urdf(SHARPA_URDF, names)
    assert hand.n_joints == len(names)
    assert hand.n_fingers == 5
    # thumb and pinky carry five controlled joints, the middle three carry four
    assert sorted(f.n_joints for f in hand.fingers) == [4, 4, 4, 5, 5]


def test_sharpa_tokens_match_its_urdf_exactly():
    """The tree is not an approximation of the asset: it reproduces it."""
    names = _sharpa_joint_names()
    _bodies, truth, _valid, scale = design_space.joint_link_boxes(SHARPA_URDF, names)
    boxes, valid, tree_scale = design_space.joint_boxes(
        design_space.hand_from_urdf(SHARPA_URDF, names))
    assert boxes.shape == truth.shape
    assert np.abs(boxes - truth).max() == 0.0
    assert tree_scale == pytest.approx(scale)
    assert valid.all()


def test_a_measured_hand_is_structural_but_not_in_the_design_space():
    """Why bounds live in validate_design and not in __post_init__."""
    hand = design_space.hand_from_urdf(SHARPA_URDF, _sharpa_joint_names())
    assert not validate_design.is_valid(hand)


def test_generated_hands_use_the_same_token_function():
    hand = gen_init_pop.seed_hand(random.Random(0))
    boxes, valid, scale = design_space.joint_boxes(hand)
    assert boxes.shape == (hand.n_joints, 4, 3)
    assert valid.all() and scale > 0.0


# --- many designs, one articulation ------------------------------------------

def test_population_projects_onto_one_template():
    from hand_sampler.robot_spec import population_spec
    hands = gen_init_pop.seed_population(seed=0, count=16)
    pop = population_spec(hands)
    J = design_space.MAX_FINGERS * design_space.MAX_JOINTS_PER_FINGER
    assert pop.spec.num_hand_joints == J
    assert pop.joint_link_boxes.shape == (16, J, 4, 3)
    assert pop.joint_valid.shape == (16, J)
    assert pop.joint_limits.shape == (16, J, 2)
    assert pop.fingertip_valid.shape == (16, design_space.MAX_FINGERS)


def test_a_designs_real_joints_land_in_its_own_finger_slots():
    from hand_sampler.robot_spec import population_spec
    hands = gen_init_pop.seed_population(seed=0, count=8)
    pop = population_spec(hands)
    D = design_space.MAX_JOINTS_PER_FINGER
    for i, hand in enumerate(hands):
        expected = [f * D + d for f, finger in enumerate(hand.fingers)
                    for d in range(finger.n_joints)]
        assert np.flatnonzero(pop.joint_valid[i]).tolist() == expected
        boxes, _valid, _scale = design_space.joint_boxes(hand)
        assert np.abs(pop.joint_link_boxes[i][pop.joint_valid[i]] - boxes).max() == 0.0


def test_ghost_joints_are_locked_so_joint_enabled_reads_zero():
    """reset.py derives joint_enabled as upper - lower > 1e-6."""
    from hand_sampler.robot_spec import population_spec
    pop = population_spec(gen_init_pop.seed_population(seed=1, count=8))
    ghosts = ~pop.joint_valid
    # (0, 1e-8), the convention the old multi-embodiment path used: locked, but
    # not an exactly coincident (degenerate) constraint.
    assert np.all(pop.joint_limits[ghosts][:, 0] == 0.0)
    assert np.all(pop.joint_limits[ghosts][:, 1] == np.float32(1e-8))
    enabled = pop.joint_limits[..., 1] - pop.joint_limits[..., 0] > 1e-6
    assert np.array_equal(enabled, pop.joint_valid)


def test_sharpa_fits_the_generated_envelope():
    """The reference hand is a point the template can hold: 5 fingers, <= 6 each."""
    from hand_sampler.robot_spec import population_spec
    hand = design_space.hand_from_urdf(SHARPA_URDF, _sharpa_joint_names())
    pop = population_spec([hand])
    assert pop.joint_valid.sum() == hand.n_joints == 22
    assert sorted(f.n_joints for f in hand.fingers) == [4, 4, 4, 5, 5]


def test_per_env_gather_is_consistent_with_the_design_index():
    from hand_sampler.robot_spec import design_index, population_spec
    pop = population_spec(gen_init_pop.seed_population(seed=3, count=7))
    idx = design_index(20, pop.n_designs)
    d = pop.per_env(idx)
    J = design_space.MAX_FINGERS * design_space.MAX_JOINTS_PER_FINGER
    assert d["joint_link_bbox_local"].shape == (20, J, 4, 3)
    assert d["hand_scale"].shape == (20, 1)
    # envs holding the same design get the same rows, and the wrap is i % n
    assert np.array_equal(d["joint_link_bbox_local"][0], d["joint_link_bbox_local"][7])
    enabled = d["joint_limits"][..., 1] - d["joint_limits"][..., 0] > 1e-6
    assert np.array_equal(enabled, d["joint_geometry_valid"])


def test_per_env_rejects_an_out_of_range_design():
    from hand_sampler.robot_spec import population_spec
    pop = population_spec(gen_init_pop.seed_population(seed=0, count=3))
    with pytest.raises(ValueError, match="out of range"):
        pop.per_env(np.array([0, 1, 3]))


def test_ranks_do_not_author_the_same_designs():
    """A 2-GPU run must instantiate its whole population, not half of it twice."""
    from hand_sampler.robot_spec import design_index
    n_envs, n_designs = 12288, 24576
    r0 = design_index(n_envs, n_designs, rank=0, world_size=2)
    r1 = design_index(n_envs, n_designs, rank=1, world_size=2)
    assert not set(r0.tolist()) & set(r1.tolist())
    assert len(set(r0.tolist()) | set(r1.tolist())) == n_designs


def test_a_population_smaller_than_the_scene_wraps():
    from hand_sampler.robot_spec import design_index
    idx = design_index(10, 4, rank=1, world_size=2)
    assert idx.tolist() == [(10 + i) % 4 for i in range(10)]


# --- self-intersection at rest, past the proximal links ----------------------

# A real hand from round 50 of the neutral-drift walk, taken before the check
# was extended. Its BASE links clear each other by 23.7 mm; finger 1's DISTAL
# link and finger 3's base link are 19.4 mm apart and therefore intersect. It
# cannot be reached by mutating any more -- the extended check rejects it at
# mutation time, which is the point -- so it is written down instead.
def _distal_overlap():
    """Two fingers whose BASE links clear each other and whose distal links do not.

    Built rather than stored. The previous fixture was a drifted hand whose base
    links cleared at the old 10 mm capsule radius; at 15 mm they overlap too, so
    it stopped isolating the distal case. Such a hand cannot be found by mutating
    either -- the validator rejects it, so it never survives into a population.
    """
    palm = design_space.Palm(design_space.PALM_THICKNESS)

    def finger(radius, bearing_deg, spec):
        return design_space.Finger(
            design_space.Mount.polar(radius, math.radians(bearing_deg) % (2 * math.pi),
                               math.radians(bearing_deg) % (2 * math.pi)),
            tuple(design_space.Segment(design_space.Joint(kind), 0.040, lean=lean)
                  for kind, lean in spec))

    # (kind, lean) per segment. Found by search: the BASE links clear by 48.7 mm
    # and finger 0's second link comes within 20.2 mm of finger 1's third.
    F, A, R = design_space.FLEXION, design_space.ABDUCTION, design_space.ROLL
    return design_space.Hand(palm, (
        finger(0.040, 345, [(R, 0), (R, 2), (R, 1)]),
        finger(0.030, 60, [(R, 4), (F, 4), (R, 4)]),
    ))


def test_clearance_checks_every_link_not_just_the_proximal_ones():
    """A distal link folding onto another finger used to be invisible.

    ``check_base_clearance`` looked at ``base_capsules`` -- one link per finger,
    which is every pair a two-joint hand HAS -- so generation 0 never noticed
    and 2.2% of a drifted population started in self-contact.
    """
    import itertools

    from hand_sampler import validate_design

    hand = _distal_overlap()

    # The base links are genuinely clear, so the old check saw nothing.
    base = design_space.base_capsules(hand)
    r = design_space.CAPSULE_RADIUS
    worst_base = min(
        design_space.segment_distance(*design_space.capsule_axis(p0, p1, r),
                                      *design_space.capsule_axis(q0, q1, r))
        for (p0, p1), (q0, q1) in itertools.combinations(base, 2))
    assert worst_base >= 2 * r

    complaints = validate_design.check_base_clearance(hand)
    assert complaints, "a distal link intersects another finger and was missed"
    assert "finger 0 link 1" in complaints[0] and "finger 1 link 2" in complaints[0], \
        complaints


def test_rest_capsules_uses_the_authored_capsule_axis():
    """Tip to tip is the link length, so the axis is inset a radius each end --
    the same decomposition author_hand and viewer.capsule_mesh use."""
    import numpy as np

    r = design_space.CAPSULE_RADIUS
    p0, p1 = np.zeros(3), np.array([0.05, 0.0, 0.0])
    a, b = design_space.capsule_axis(p0, p1, r)
    assert np.allclose(a, [r, 0, 0]) and np.allclose(b, [0.05 - r, 0, 0])
    # tip to tip is the full length again once the caps are added back
    assert np.isclose(np.linalg.norm(b - a) + 2 * r, 0.05)

    # A link shorter than a diameter is a sphere at its midpoint, as authored.
    a, b = design_space.capsule_axis(p0, np.array([0.015, 0.0, 0.0]), r)
    assert np.allclose(a, b) and np.allclose(a, [0.0075, 0, 0])


# --- object assignment ---------------------------------------------------------

def test_design_cycle_deals_every_object_to_every_design_once():
    """24576 envs on 2 ranks, 1024 designs, a 24-object pool: each design
    meets each object exactly once, and the two ranks hold disjoint halves of
    that deal instead of the same dozen twice."""
    import numpy as np
    from hand_sampler.robot_spec import design_index, object_index
    n_envs, n_designs, n_pool = 12288, 1024, 24
    pairs = set()
    for rank in (0, 1):
        d = design_index(n_envs, n_designs, rank=rank, world_size=2)
        o = object_index(n_envs, n_pool, "design_cycle", rank=rank, world_size=2, n_designs=n_designs)
        new = set(zip(d.tolist(), o.tolist()))
        assert not (pairs & new), "a (design, object) pair repeated across ranks"
        pairs |= new
    assert len(pairs) == n_designs * n_pool
    per_design = np.zeros((n_designs, n_pool), int)
    for d, o in pairs:
        per_design[d, o] += 1
    assert (per_design == 1).all()


def test_env_modulo_is_the_old_rule_and_repeats_across_ranks():
    from hand_sampler.robot_spec import object_index
    o0 = object_index(12288, 1200, "env_modulo", rank=0, world_size=2, n_designs=1024)
    o1 = object_index(12288, 1200, "env_modulo", rank=1, world_size=2, n_designs=1024)
    assert (o0 == o1).all() and o0[1201] == 1


def test_design_cycle_with_one_design_is_a_plain_cycle():
    from hand_sampler.robot_spec import object_index
    o = object_index(100, 24, "design_cycle", n_designs=1)
    assert list(o[:26]) == list(range(24)) + [0, 1]
