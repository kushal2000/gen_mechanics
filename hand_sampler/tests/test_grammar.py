"""Properties the grammar must hold, not examples of it working."""

from __future__ import annotations

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
    h = math.pi / 2
    assert np.allclose(design_space.axis_of(design_space.Joint(0.0, h)), [0, 0, 1])      # flexion
    assert np.allclose(design_space.axis_of(design_space.Joint(h, h)), [0, 1, 0])        # abduction
    assert np.allclose(design_space.axis_of(design_space.Joint(0.0, 1e-12)), [1, 0, 0])  # twist


def test_axis_is_unit():
    for t in np.linspace(0, math.pi, 13):
        for p in np.linspace(0.05, math.pi / 2, 7):
            assert abs(np.linalg.norm(design_space.axis_of(design_space.Joint(t, p))) - 1) < 1e-12


def test_mount_frame_orthonormal_on_every_face():
    """The frame degenerates when a finger points along GRASP_DIR, which a mount tilt can..."""
    palm = design_space.Palm(0.025, 0.060, 0.060)
    for face in design_space.FINGER_FACES:
        _, R = design_space.mount_frame(design_space.Mount(face, 0.5, 0.5), palm)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9), face
        assert abs(np.linalg.det(R) - 1.0) < 1e-9, face
def test_seeds_valid(pop):
    assert all(validate_design.is_valid(h) for h in pop)


def test_seeds_are_simple(pop):
    """Generation 0 starts at the conventional corner."""
    assert max(h.n_joints for h in pop) <= 4
    assert all(h.n_fingers == 2 for h in pop)
    assert all(s.joint.phi == pytest.approx(math.pi / 2)
               for h in pop for f in h.fingers for s in f.segments)


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


def test_operators_are_unbiased(pop):
    """Per move, complexity must be as likely to fall as to rise."""
    def at_least(h, target, seed):
        rng = random.Random(seed)
        for _ in range(8000):
            if h.n_joints >= target:
                return h
            c = mutate_design.mutate(rng, h, mutate_design.STRUCTURAL[rng.randrange(len(mutate_design.STRUCTURAL))])
            if c:
                h = c
        return h

    rng = random.Random(0)
    for target in (4, 6):
        up = down = 0
        for s in range(25):
            h = at_least(pop[s], target, s)
            assert h.n_joints >= target, f"could not reach n={target}"
            for _ in range(80):
                c = mutate_design.mutate(rng, h, mutate_design.STRUCTURAL[rng.randrange(len(mutate_design.STRUCTURAL))])
                if c is None:
                    continue
                up, down = (up + 1, down) if c.n_joints > h.n_joints else (up, down + 1)
        moved = up + down
        assert moved > 200, f"too few moves at n={target}"
        assert 0.40 < up / moved < 0.60, f"n={target}: P(up) = {up/moved:.1%}"


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
    b = design_space.Hand(palm=design_space.Palm(*a.palm.extents), fingers=a.fingers)
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
    return design_space.Hand(design_space.Palm(0.025, 0.060, 0.060), tuple(fingers))


def _finger(face, lengths, v=0.7):
    return design_space.Finger(
        design_space.Mount(face, 0.5, v),
        tuple(design_space.Segment(design_space.Joint((i * design_space.ANGLE_QUANTUM) % math.pi, math.pi / 2), L)
              for i, L in enumerate(lengths)))


@pytest.mark.parametrize("lengths,note", [
    ([0.040, 0.040], "ordinary two-joint finger"),
    ([0.020, 0.020], "both links at MIN_LINK_LENGTH"),
    ([0.020, 0.075], "very uneven links"),
    ([0.045, 0.040], "merge overflows MAX_LINK_LENGTH by one quantum"),
    ([0.080, 0.080], "both links already at MAX_LINK_LENGTH"),
    ([0.045, 0.045, 0.030], "interior overflow -- distal neighbour fits"),
    ([0.080, 0.080, 0.080], "every merge overflows; the clamp is the only path"),
])
def test_merge_links_handles_every_merge_case(lengths, note):
    """A joint must be removable whatever the link lengths around it."""
    hand = _hand(_finger("+y", lengths), _finger("+z", [0.050]))
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
    hand = _hand(_finger("+y", [0.050]), _finger("+z", [0.050]))
    assert hand.n_fingers == design_space.MIN_FINGERS
    rng = random.Random(0)
    for op in ("merge_links", "remove_finger"):
        for _ in range(25):
            with pytest.raises(mutate_design.MutationImpossible):
                mutate_design.apply(rng, hand, op)


def test_merge_links_preserves_reach_unless_it_must_clamp():
    """Reach is preserved on every path split_link can produce; the clamp is unreachable..."""
    hand = _hand(_finger("+y", [0.040, 0.040]), _finger("+z", [0.050]))
    rng = random.Random(0)
    before = sum(f.reach for f in hand.fingers)
    for _ in range(100):
        child = mutate_design.apply(rng, hand, "merge_links")
        assert sum(f.reach for f in child.fingers) == pytest.approx(before, abs=1e-12)


def test_one_joint_per_link():
    """No two joints share a point."""
    palm = design_space.Palm(0.025, 0.060, 0.060)
    bad = design_space.Finger(design_space.Mount("+y", 0.5, 0.7),
                   (design_space.Segment(design_space.Joint(0.0), 0.0), design_space.Segment(design_space.Joint(0.0), 0.040)))
    assert validate_design.check_finger(bad, 0, palm), "a zero-length link must be rejected"

    rng = random.Random(1)
    hand = gen_init_pop.seed_population(0, 1)[0]
    for _ in range(4000):
        child = mutate_design.mutate(rng, hand)
        if child:
            hand = child
        for f in hand.fingers:
            assert all(s.length >= design_space.MIN_LINK_LENGTH - 1e-9 for s in f.segments)

    # distinct joint positions, which is the geometric statement of the rule
    for f in hand.fingers:
        joints, _ = design_space.forward_kinematics(f, hand.palm)
        for a, b in zip(joints, joints[1:]):
            assert np.linalg.norm(a - b) >= design_space.MIN_LINK_LENGTH - 1e-9


def test_capsules_carry_their_segment_index():
    """Each capsule reports which segment it belongs to."""
    palm = design_space.Palm(0.025, 0.060, 0.060)
    finger = design_space.Finger(design_space.Mount("+y", 0.5, 0.7),
                      tuple(design_space.Segment(design_space.Joint(i * design_space.ANGLE_QUANTUM % math.pi), 0.030)
                            for i in range(3)))
    _, capsules = design_space.forward_kinematics(finger, palm)
    assert [c[3] for c in capsules] == [0, 1, 2]


def test_joint_axes_are_the_axes_the_joints_turn_about():
    """The viewer draws each joint as a cylinder along its reported axis, so the axis has..."""
    palm = design_space.Palm(0.025, 0.070, 0.070)
    finger = design_space.Finger(design_space.Mount("+y", 0.5, 0.6),
                      (design_space.Segment(design_space.Joint(0.0, offset=0.3), 0.035),
                       design_space.Segment(design_space.Joint(math.pi / 3), 0.030),
                       design_space.Segment(design_space.Joint(math.pi / 2, offset=-0.2), 0.025)))
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


def test_fingers_do_not_overlap_at_the_base(pop):
    """No two proximal links may intersect at rest."""
    rng = random.Random(4)
    hand = pop[0]
    worst = float("inf")
    for _ in range(4000):
        child = mutate_design.mutate(rng, hand)
        if child:
            hand = child
        caps = design_space.base_capsules(hand)
        for (p0, p1), (q0, q1) in itertools.combinations(caps, 2):
            worst = min(worst, design_space.segment_distance(p0, p1, q0, q1))
    assert worst >= 2 * design_space.CAPSULE_RADIUS - 1e-9, (
        f"base links came within {worst * 1000:.1f} mm; capsules intersect below "
        f"{2 * design_space.CAPSULE_RADIUS * 1000:.0f} mm")


def test_same_face_mounts_keep_their_distance(pop):
    rng = random.Random(7)
    hand = pop[0]
    for _ in range(3000):
        child = mutate_design.mutate(rng, hand)
        if child:
            hand = child
        pos = [(f.mount.face, design_space.mount_position(f.mount, hand.palm))
               for f in hand.fingers]
        for (fa, pa), (fb, pb) in itertools.combinations(pos, 2):
            if fa == fb:
                d = float(np.linalg.norm(pa - pb))
                assert d >= design_space.MIN_SAME_FACE_SEPARATION - 1e-9, (
                    f"two mounts {d * 1000:.1f} mm apart on {fa}")


def test_crowding_does_not_block_new_fingers():
    """Adding fingers must stop because the palm is FULL, not because placement gave up..."""
    def pack(palm):
        hand = design_space.Hand(palm, gen_init_pop.seed_population(0, 1)[0].fingers)
        for k in range(30):
            out = mutate_design._new_finger(random.Random(k), hand)
            if out is None or hand.n_fingers >= design_space.MAX_FINGERS:
                break
            hand = out
        return hand.n_fingers

    assert pack(design_space.Palm(0.025, 0.100, 0.100)) == design_space.MAX_FINGERS, \
        "a large palm should reach the cap"

    small = pack(design_space.Palm(0.025, 0.050, 0.050))
    assert 2 <= small < design_space.MAX_FINGERS, \
        f"a small palm packed {small}; separation should bind before the cap"


def test_min_link_length_allows_a_compact_knuckle():
    """Two axes may sit closer than a link's own diameter."""
    assert design_space.MIN_LINK_LENGTH < 2 * design_space.CAPSULE_RADIUS

    palm = design_space.Palm(0.025, 0.060, 0.060)
    finger = design_space.Finger(design_space.Mount("+y", 0.5, 0.7), (
        design_space.Segment(design_space.Joint(0.0), design_space.MIN_LINK_LENGTH),
        design_space.Segment(design_space.Joint(math.pi / 2), 0.040)))
    assert not validate_design.check_finger(finger, 0, palm), validate_design.check_finger(finger, 0, palm)

    joints, _ = design_space.forward_kinematics(finger, palm)
    gap = float(np.linalg.norm(joints[1] - joints[0]))
    assert gap == pytest.approx(design_space.MIN_LINK_LENGTH)
    assert gap < 2 * design_space.CAPSULE_RADIUS, "axes are not closer than the link is wide"

def test_mounts_stay_clear_of_face_edges(pop):
    """A mount within one capsule radius of an edge hangs the finger off the palm."""
    rng = random.Random(4)
    hand = pop[0]
    worst = float("inf")
    for _ in range(4000):
        child = mutate_design.mutate(rng, hand)
        if child:
            hand = child
        for f in hand.fingers:
            _, _, _, _, span_u, span_v = design_space.face_frame(f.mount.face, hand.palm)
            worst = min(worst,
                        min(f.mount.u, 1.0 - f.mount.u) * span_u,
                        min(f.mount.v, 1.0 - f.mount.v) * span_v)
    assert worst >= design_space.MOUNT_EDGE_MARGIN - 1e-9, (
        f"a mount came {worst * 1000:.1f} mm from a face edge, margin is "
        f"{design_space.MOUNT_EDGE_MARGIN * 1000:.0f} mm")


def test_move_mount_still_crosses_faces_with_a_margin(pop):
    """The margin must not disconnect the surface."""
    for seed in range(3):
        hand = pop[seed]
        rng = random.Random(seed)
        seen = {f.mount.face for f in hand.fingers}
        for _ in range(3000):
            child = mutate_design.mutate(rng, hand, "move_mount")
            if child:
                hand = child
                seen |= {f.mount.face for f in hand.fingers}
        assert seen == set(design_space.FINGER_FACES), f"only reached {sorted(seen)}"


def test_perturb_palm_leaves_thickness_alone(pop):
    """Thickness is seeded and never mutated; the step is twice the grid because..."""
    rng = random.Random(1)
    hand = pop[0]
    seen_steps = set()
    for _ in range(2000):
        child = mutate_design.mutate(rng, hand, "perturb_palm")
        if child is None:
            continue
        assert child.palm.thickness == hand.palm.thickness, "thickness moved"
        for dim in design_space.MUTABLE_PALM_DIMS:
            delta = getattr(child.palm, dim) - getattr(hand.palm, dim)
            if abs(delta) > 1e-9:
                seen_steps.add(round(abs(delta), 6))
        hand = child
    assert seen_steps == {round(design_space.PALM_STEP, 6)}, (
        f"steps seen: {sorted(seen_steps)}, expected only {design_space.PALM_STEP}")


def test_check_finger_needs_the_real_palm():
    """Mount bounds depend on the face spans, so the palm cannot be defaulted: the same..."""
    finger = design_space.Finger(design_space.Mount("+y", 0.5, 0.90),
                      (design_space.Segment(design_space.Joint(0.0), 0.040),))
    assert validate_design.check_finger(finger, 0, design_space.Palm(0.025, 0.060, 0.060))
    assert not validate_design.check_finger(finger, 0, design_space.Palm(0.025, 0.100, 0.100))
    with pytest.raises(TypeError):
        validate_design.check_finger(finger, 0)


def test_require_valid_reports_every_reason():
    """The loud counterpart to is_valid."""
    palm = design_space.Palm(0.025, 0.060, 0.060)
    good = gen_init_pop.seed_population(0, 1)[0]
    assert validate_design.require_valid(good) is good

    bad = design_space.Hand(palm, (design_space.Finger(design_space.Mount("+y", 0.5, 0.99),
                                 (design_space.Segment(design_space.Joint(0.0), 0.5),)),))
    with pytest.raises(ValueError) as e:
        validate_design.require_valid(bad)
    assert "length" in str(e.value) and "mount" in str(e.value)


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


def test_move_mount_moves_both_axes(pop):
    """Both face axes must move, not just the roomy one."""
    rng = random.Random(0)
    moved_u = moved_v = 0
    for hand in pop[:40]:
        for _ in range(40):
            child = mutate_design.mutate(rng, hand, "move_mount")
            if child is None:
                continue
            for a, b in zip(hand.fingers, child.fingers):
                moved_u += abs(a.mount.u - b.mount.u) > 1e-12
                moved_v += abs(a.mount.v - b.mount.v) > 1e-12
            hand = child
    assert moved_u > 0, "the thin axis never moved"
    assert moved_v > 0, "the long axis never moved"


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
        if "s_theta" in kw:
            seg = _replace(seg, joint=_replace(seg.joint, theta=kw["s_theta"]))
        if "s_phi" in kw:
            seg = _replace(seg, joint=_replace(seg.joint, phi=kw["s_phi"]))
        if "s_offset" in kw:
            seg = _replace(seg, joint=_replace(seg.joint, offset=kw["s_offset"]))
        new_palm = _replace(palm, **{k[2:]: v for k, v in kw.items()
                                     if k.startswith("p_")})
        return design_space.Hand(new_palm,
                      (_replace(f0, mount=mount, segments=(seg,) + f0.segments[1:]),)
                      + hand.fingers[1:])

    cases = {
        "palm width out of range": variant(p_width=0.200),
        "palm width off grid": variant(p_width=0.0623),
        "palm length out of range": variant(p_length=0.010),
        "link too short": variant(s_len=0.010),
        "link too long": variant(s_len=0.200),
        "link off grid": variant(s_len=0.0431),
        "theta out of range": variant(s_theta=math.pi + 0.3),
        "theta off grid": variant(s_theta=0.1),
        "phi out of range": variant(s_phi=math.pi),
        "phi off grid": variant(s_phi=0.1),
        "mount u past the margin": variant(m_u=0.99),
        "mount v past the margin": variant(m_v=0.99),
        "offset outside joint travel": variant(s_offset=math.radians(120)),
        "offset off grid": variant(s_offset=0.1),
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


def test_offset_subsumes_a_mount_pointing_direction():
    """A base-joint offset reproduces every rest direction a mount tilt could, and a..."""
    palm = design_space.Palm(0.025, 0.060, 0.060)
    face = "+y"
    _, normal, t_u, t_v, _, _ = design_space.face_frame(face, palm)
    _, R = design_space.mount_frame(design_space.Mount(face, 0.5, 0.5), palm)

    reachable = np.array([
        design_space.rodrigues(R @ design_space.axis_of(design_space.Joint(float(th), math.pi / 2)), float(d)) @ normal
        for th in np.arange(0, math.pi, design_space.ANGLE_QUANTUM)
        for d in np.arange(-math.pi / 2, math.pi / 2 + 1e-9, design_space.ANGLE_QUANTUM)])

    for a in np.arange(0, math.pi / 2 + 1e-9, design_space.ANGLE_QUANTUM):
        for b in np.arange(0, 2 * math.pi, design_space.ANGLE_QUANTUM):
            want = (math.cos(a) * normal
                    + math.sin(a) * (math.cos(b) * t_u + math.sin(b) * t_v))
            gap = float(np.min(np.linalg.norm(reachable - want, axis=1)))
            assert gap < 1e-9, f"alpha={math.degrees(a):.0f} deg uncovered"

    curled = design_space.Finger(design_space.Mount(face, 0.5, 0.7), (
        design_space.Segment(design_space.Joint(0.0, math.pi / 2), 0.04),
        design_space.Segment(design_space.Joint(0.0, math.pi / 2, math.radians(60)), 0.04)))
    straight = design_space.Finger(curled.mount, (curled.segments[0],
                                       design_space.Segment(design_space.Joint(0.0), 0.04)))
    assert float(np.linalg.norm(design_space.fingertip(curled, palm)
                                - design_space.fingertip(straight, palm))) > 0.03


def test_offset_moves_the_link_but_theta_does_not():
    """The two per-joint angles do different things, which is why both exist."""
    palm = design_space.Palm(0.025, 0.060, 0.060)

    def tip(theta_deg, offset_deg):
        f = design_space.Finger(design_space.Mount("+y", 0.5, 0.7),
                     (design_space.Segment(design_space.Joint(math.radians(theta_deg), math.pi / 2,
                                        math.radians(offset_deg)), 0.05),))
        return design_space.fingertip(f, palm)

    assert np.allclose(tip(0, 0), tip(60, 0)), "theta moved the link at rest"
    assert not np.allclose(tip(0, 0), tip(0, 45)), "offset did not move the link"


def test_crossing_a_face_rotates_the_finger_with_it():
    """A finger leaves along its face normal, so crossing an edge rotates its world..."""
    palm = design_space.Palm(0.025, 0.060, 0.060)
    mount = design_space.Mount("+y", 0.5, 0.80)
    before = design_space.mount_direction(mount, palm)

    for _ in range(8):
        nxt = mutate_design._step_mount(mount, palm, 0.0, +mutate_design.MOUNT_STEP_M)
        if nxt is None or nxt == mount:
            break
        mount = nxt
        if mount.face != "+y":
            break
    assert mount.face == "+z", "the walk never crossed"

    after = design_space.mount_direction(mount, palm)
    angle = math.degrees(math.acos(float(np.clip(before @ after, -1, 1))))
    assert angle == pytest.approx(90.0, abs=1e-6)


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
