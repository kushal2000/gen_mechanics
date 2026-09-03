"""Properties the grammar must hold, not examples of it working.

The load-bearing ones are ``test_operators_are_unbiased`` (the no-parsimony-
penalty position depends on additions and removals being equally available) and
``test_exact_inverse`` (which is what makes that measurable at all).
"""

from __future__ import annotations

import itertools
import math
import random

import numpy as np
import pytest

from hand_sampler import genotype as G
from hand_sampler import kinematics as K
from hand_sampler import mutate as M
from hand_sampler import sample as S
from hand_sampler import validate as V


@pytest.fixture(scope="module")
def pop():
    return S.seed_population(0, 120)


# --- kinematics -------------------------------------------------------------

def test_axis_endpoints():
    h = math.pi / 2
    assert np.allclose(K.axis_of(G.Joint(0.0, h)), [0, 0, 1])      # flexion
    assert np.allclose(K.axis_of(G.Joint(h, h)), [0, 1, 0])        # abduction
    assert np.allclose(K.axis_of(G.Joint(0.0, 1e-12)), [1, 0, 0])  # twist


def test_axis_is_unit():
    for t in np.linspace(0, math.pi, 13):
        for p in np.linspace(0.05, math.pi / 2, 7):
            assert abs(np.linalg.norm(K.axis_of(G.Joint(t, p))) - 1) < 1e-12


def test_mount_frame_orthonormal_on_every_face():
    """The frame degenerates when a finger points along GRASP_DIR, which a mount
    tilt can reach even though no face normal does."""
    palm = G.Palm(0.025, 0.060, 0.060)
    for face in G.FINGER_FACES:
        _, R = K.mount_frame(G.Mount(face, 0.5, 0.5, 0.0), palm)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9), face
        assert abs(np.linalg.det(R) - 1.0) < 1e-9, face
def test_seeds_valid(pop):
    assert all(V.is_valid(h) for h in pop)


def test_seeds_are_simple(pop):
    """Generation 0 starts at the conventional corner."""
    assert max(h.n_joints for h in pop) <= 4
    assert all(h.n_fingers == 2 for h in pop)
    assert all(s.joint.phi == pytest.approx(math.pi / 2)
               for h in pop for f in h.fingers for s in f.segments)


# --- operators --------------------------------------------------------------

def test_every_operator_can_act(pop):
    rng = random.Random(1)
    # remove_finger needs a hand above MIN_FINGERS carrying a single-joint
    # finger; a seed is at the floor, so it is given one that has grown.
    grown = [c for h in pop[:40] if (c := M.mutate(rng, h, "add_finger"))]
    for op in M.OPERATORS:
        source = grown if op == "remove_finger" else pop[:40]
        assert any(M.mutate(rng, h, op) for h in source), op


def test_mutation_is_closed(pop):
    """Every operator returns a valid hand or raises. Nothing downstream
    re-checks, so a leak here becomes an invalid design in a population file."""
    rng = random.Random(2)
    h = pop[0]
    for _ in range(3000):
        c = M.mutate(rng, h)
        if c is not None:
            assert V.is_valid(c), V.check(c)
            h = c


def test_split_link_preserves_reach(pop):
    """Splitting divides a link and merging restores it, so reach is unchanged --
    which is what makes merge_links an exact inverse rather than a shortening."""
    rng = random.Random(3)
    for h in pop[:60]:
        c = M.mutate(rng, h, "split_link")
        if c is not None:
            assert sum(f.reach for f in c.fingers) == pytest.approx(
                sum(f.reach for f in h.fingers), abs=1e-12)


@pytest.mark.parametrize("add,remove", [("split_link", "merge_links"),
                                        ("add_finger", "remove_finger")])
def test_exact_inverse(pop, add, remove):
    """Add then remove must return the ORIGINAL hand, not a nearby one. That is
    what the angle and length grids buy; continuous parameters would leak."""
    rng = random.Random(4)
    tried = recovered = 0
    for h in pop[:60]:
        child = M.mutate(rng, h, add)
        if child is None:
            continue
        tried += 1
        back = random.Random(99)
        if any(M.mutate(back, child, remove) == h for _ in range(60)):
            recovered += 1
    assert tried > 0
    assert recovered == tried, f"{recovered}/{tried} recovered"


def test_operators_are_unbiased(pop):
    """Per move, complexity must be as likely to fall as to rise.

    MEASURED PER MOVE, NOT AS ENDPOINT DRIFT -- earlier attempts measured where an
    unselected walk ends up, and both were artifacts of the starting hand.

    Tested where the space is interior. Balance falls with depth, and with four
    structural operators the reason is visible per-operator:

        n    split  merge  add_finger  remove_finger   P(up)
        4     83%    84%      100%          62%        55.8%
        6     96%    97%       74%          92%        47.2%
       10     78%   100%       34%          84%        38.3%
       14     77%   100%        4%          34%        38.2%

    ``add_finger`` collapses as the palm fills while ``merge_links`` stays near
    100%, so a deep hand drifts down. That is palm CAPACITY, not operator bias --
    ``perturb_palm`` is what relieves it. A single pooled add operator hid this,
    because splits kept succeeding under the same name once the palm was full.
    """
    def at_least(h, target, seed):
        rng = random.Random(seed)
        for _ in range(8000):
            if h.n_joints >= target:
                return h
            c = M.mutate(rng, h, M.STRUCTURAL[rng.randrange(len(M.STRUCTURAL))])
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
                c = M.mutate(rng, h, M.STRUCTURAL[rng.randrange(len(M.STRUCTURAL))])
                if c is None:
                    continue
                up, down = (up + 1, down) if c.n_joints > h.n_joints else (up, down + 1)
        moved = up + down
        assert moved > 200, f"too few moves at n={target}"
        assert 0.40 < up / moved < 0.60, f"n={target}: P(up) = {up/moved:.1%}"


def test_deep_fingers_are_reachable(pop):
    """Depth must be reachable, not merely slower. Removing the coincident-joint
    attach point left only 'split a long enough link' and 'start a new finger',
    which makes depth depend on length."""
    rng = random.Random(0)
    best = 0
    for s in range(12):
        h = pop[s]
        for _ in range(4000):
            c = M.mutate(rng, h, "split_link") or M.mutate(rng, h, "perturb_length")
            if c:
                h = c
        best = max(best, max(f.n_joints for f in h.fingers))
    assert best == G.MAX_JOINTS_PER_FINGER, f"deepest finger reached was {best}"

def test_identical_hands_compare_equal(pop):
    """Fitness memoisation and exact inverses both depend on this."""
    a = pop[0]
    b = G.Hand(palm=G.Palm(*a.palm.extents), fingers=a.fingers)
    assert a == b and hash(a) == hash(b)


def test_alpha_zero_forces_beta_zero():
    """At alpha = 0 every beta names the same finger; two spellings of one hand
    would break identity."""
    palm = G.Palm(0.025, 0.060, 0.060)
    reasons = V.check_finger(
        G.Finger(G.Mount("+z", 0.5, 0.5, alpha=0.0, beta=G.ANGLE_QUANTUM),
                 (G.Segment(G.Joint(0.0), 0.04),)), 0, palm)
    assert any("beta" in r for r in reasons)


def _closest_approach(hand, starts=4):
    """How near two fingertips can be brought, over all joint angles.

    By local optimisation, not random sampling: uniform sampling of a 4-D joint
    space is sparse enough to report false failures.
    """
    from scipy.optimize import minimize

    lo, hi = G.JOINT_LIMIT
    counts = [f.n_joints for f in hand.fingers]
    total = sum(counts)

    def separation(x):
        tips, k = [], 0
        for f, n in zip(hand.fingers, counts):
            tips.append(K.fingertip(f, hand.palm, {i: x[k + i] for i in range(n)}))
            k += n
        return min(float(np.linalg.norm(p - q))
                   for p, q in itertools.combinations(tips, 2))

    rng = random.Random(0)
    return min(float(minimize(separation,
                              [rng.uniform(lo, hi) for _ in range(total)],
                              bounds=[(lo, hi)] * total, method="L-BFGS-B").fun)
               for _ in range(starts))


def test_seeds_give_a_gradient_to_select_on(pop):
    """Most of generation 0 must be able to touch the object -- not all of it.

    The failure guarded is every seed scoring zero, leaving nothing to select on.
    Hit once for real: an earlier seed set paired fingers on opposite faces and
    58% of the population could not reach the object at any joint angles.

    It deliberately does NOT require that every seed closes. One-joint fingers
    often cannot, and forcing two joints everywhere would put the whole
    population at four motors -- deleting the cheap end of the
    performance-vs-motors curve, which is a result rather than a defect.
    """
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
    return G.Hand(G.Palm(0.025, 0.060, 0.060), tuple(fingers))


def _finger(face, lengths, v=0.7):
    return G.Finger(
        G.Mount(face, 0.5, v, 0.0),
        tuple(G.Segment(G.Joint((i * G.ANGLE_QUANTUM) % math.pi, math.pi / 2), L)
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
    """A joint must be removable whatever the link lengths around it.

    The overflow rows are regressions: folding a link only into its proximal
    neighbour left a finger whose adjacent links summed past the ceiling unable
    to shed that joint at all, and ``perturb_length`` walks fingers into that
    state routinely.
    """
    hand = _hand(_finger("+y", lengths), _finger("+z", [0.050]))
    assert V.is_valid(hand), V.check(hand)

    rng = random.Random(0)
    seen = set()
    for _ in range(200):
        child = M.apply(rng, hand, "merge_links")       # must never raise
        assert child.n_joints == hand.n_joints - 1, "not a unit step"
        assert V.is_valid(child), V.check(child)
        seen.add(tuple(f.segments for f in child.fingers))
    assert len(seen) >= 2, "every joint in the finger should be removable"


def test_structural_removal_refuses_at_the_floor():
    """MIN_FINGERS single-joint fingers is the floor: neither removal can act.

    merge_links needs a finger with two joints; remove_finger needs more than
    MIN_FINGERS. At the floor both are correctly impossible."""
    hand = _hand(_finger("+y", [0.050]), _finger("+z", [0.050]))
    assert hand.n_fingers == G.MIN_FINGERS
    rng = random.Random(0)
    for op in ("merge_links", "remove_finger"):
        for _ in range(25):
            with pytest.raises(M.MutationImpossible):
                M.apply(rng, hand, op)


def test_merge_links_preserves_reach_unless_it_must_clamp():
    """Reach is preserved on every path split_link can produce; the clamp is
    unreachable from a split, which is why it costs nothing in exactness."""
    hand = _hand(_finger("+y", [0.040, 0.040]), _finger("+z", [0.050]))
    rng = random.Random(0)
    before = sum(f.reach for f in hand.fingers)
    for _ in range(100):
        child = M.apply(rng, hand, "merge_links")
        assert sum(f.reach for f in child.fingers) == pytest.approx(before, abs=1e-12)


def test_one_joint_per_link():
    """No two joints share a point. Everything downstream depends on it: no
    zero-length case in the builder, and no way for capsule and segment indices
    to diverge in a renderer."""
    palm = G.Palm(0.025, 0.060, 0.060)
    bad = G.Finger(G.Mount("+y", 0.5, 0.7, 0.0),
                   (G.Segment(G.Joint(0.0), 0.0), G.Segment(G.Joint(0.0), 0.040)))
    assert V.check_finger(bad, 0, palm), "a zero-length link must be rejected"

    rng = random.Random(1)
    hand = S.seed_population(0, 1)[0]
    for _ in range(4000):
        child = M.mutate(rng, hand)
        if child:
            hand = child
        for f in hand.fingers:
            assert all(s.length >= G.MIN_LINK_LENGTH - 1e-9 for s in f.segments)

    # distinct joint positions, which is the geometric statement of the rule
    for f in hand.fingers:
        joints, _ = K.forward_kinematics(f, hand.palm)
        for a, b in zip(joints, joints[1:]):
            assert np.linalg.norm(a - b) >= G.MIN_LINK_LENGTH - 1e-9


def test_capsules_carry_their_segment_index():
    """Each capsule reports which segment it belongs to. The index equals its own
    position today, and is carried because ``build.py`` will skip geometry for
    ghosted joints -- at which point a positional zip reads the wrong joint."""
    palm = G.Palm(0.025, 0.060, 0.060)
    finger = G.Finger(G.Mount("+y", 0.5, 0.7, 0.0),
                      tuple(G.Segment(G.Joint(i * G.ANGLE_QUANTUM % math.pi), 0.030)
                            for i in range(3)))
    _, capsules = K.forward_kinematics(finger, palm)
    assert [c[3] for c in capsules] == [0, 1, 2]


def test_fingers_do_not_overlap_at_the_base(pop):
    """No two proximal links may intersect at rest.

    Two rules are needed. Capsules are tangent at 2 x CAPSULE_RADIUS, so a single
    15 mm mount floor allowed 5 mm of interpenetration. And separation alone
    constrains where a finger STARTS, not where it POINTS -- two rooted a legal
    25 mm apart can lean together until their base links cross.
    """
    rng = random.Random(4)
    hand = pop[0]
    worst = float("inf")
    for _ in range(4000):
        child = M.mutate(rng, hand)
        if child:
            hand = child
        caps = K.base_capsules(hand)
        for (p0, p1), (q0, q1) in itertools.combinations(caps, 2):
            worst = min(worst, K.segment_distance(p0, p1, q0, q1))
    assert worst >= 2 * G.CAPSULE_RADIUS - 1e-9, (
        f"base links came within {worst * 1000:.1f} mm; capsules intersect below "
        f"{2 * G.CAPSULE_RADIUS * 1000:.0f} mm")


def test_same_face_mounts_keep_their_distance(pop):
    rng = random.Random(7)
    hand = pop[0]
    for _ in range(3000):
        child = M.mutate(rng, hand)
        if child:
            hand = child
        pos = [(f.mount.face, K.mount_position(f.mount, hand.palm))
               for f in hand.fingers]
        for (fa, pa), (fb, pb) in itertools.combinations(pos, 2):
            if fa == fb:
                d = float(np.linalg.norm(pa - pb))
                assert d >= G.MIN_SAME_FACE_SEPARATION - 1e-9, (
                    f"two mounts {d * 1000:.1f} mm apart on {fa}")


def test_crowding_does_not_block_new_fingers(pop):
    """The envelope must stay reachable as the palm fills. Placement by rejection
    sampling made ``add_finger`` quietly stop being able to add fingers as space
    ran out -- a reachability hole wearing a timeout's clothing."""
    rng = random.Random(0)
    best = 0
    for s in range(10):
        hand = pop[s]
        for _ in range(2000):
            child = (M.mutate(rng, hand, "add_finger")
                     or M.mutate(rng, hand, "perturb_length")
                     or M.mutate(rng, hand, "perturb_palm"))
            if child:
                hand = child
        best = max(best, hand.n_fingers)
    assert best == G.MAX_FINGERS, f"only reached {best} fingers"


def test_mounts_stay_clear_of_face_edges(pop):
    """A mount within one capsule radius of an edge hangs the finger off the
    palm. Tight on the thin axis -- a 25 mm palm carrying a 20 mm finger leaves
    5 mm of play -- which is what that actually looks like."""
    rng = random.Random(4)
    hand = pop[0]
    worst = float("inf")
    for _ in range(4000):
        child = M.mutate(rng, hand)
        if child:
            hand = child
        for f in hand.fingers:
            _, _, _, _, span_u, span_v = K.face_frame(f.mount.face, hand.palm)
            worst = min(worst,
                        min(f.mount.u, 1.0 - f.mount.u) * span_u,
                        min(f.mount.v, 1.0 - f.mount.v) * span_v)
    assert worst >= G.MOUNT_EDGE_MARGIN - 1e-9, (
        f"a mount came {worst * 1000:.1f} mm from a face edge, margin is "
        f"{G.MOUNT_EDGE_MARGIN * 1000:.0f} mm")


def test_move_mount_still_crosses_faces_with_a_margin(pop):
    """The margin must not disconnect the surface. Forbidding a mount NEAR an
    edge forbids one AT it, so the crossing jumps the band; without that,
    ``remount`` would have to come back as its own operator."""
    for seed in range(3):
        hand = pop[seed]
        rng = random.Random(seed)
        seen = {f.mount.face for f in hand.fingers}
        for _ in range(3000):
            child = M.mutate(rng, hand, "move_mount")
            if child:
                hand = child
                seen |= {f.mount.face for f in hand.fingers}
        assert seen == set(G.FINGER_FACES), f"only reached {sorted(seen)}"


def test_perturb_palm_leaves_thickness_alone(pop):
    """Thickness is seeded and never mutated; the step is twice the grid because
    separation's optimum band is centimetres wide."""
    rng = random.Random(1)
    hand = pop[0]
    seen_steps = set()
    for _ in range(2000):
        child = M.mutate(rng, hand, "perturb_palm")
        if child is None:
            continue
        assert child.palm.thickness == hand.palm.thickness, "thickness moved"
        for dim in G.MUTABLE_PALM_DIMS:
            delta = getattr(child.palm, dim) - getattr(hand.palm, dim)
            if abs(delta) > 1e-9:
                seen_steps.add(round(abs(delta), 6))
        hand = child
    assert seen_steps == {round(G.PALM_STEP, 6)}, (
        f"steps seen: {sorted(seen_steps)}, expected only {G.PALM_STEP}")


def test_check_finger_needs_the_real_palm():
    """Mount bounds depend on the face spans, so the palm cannot be defaulted:
    the same mount is illegal on a 60 mm face and legal on a 100 mm one."""
    finger = G.Finger(G.Mount("+y", 0.5, 0.90, 0.0),
                      (G.Segment(G.Joint(0.0), 0.040),))
    assert V.check_finger(finger, 0, G.Palm(0.025, 0.060, 0.060))
    assert not V.check_finger(finger, 0, G.Palm(0.025, 0.100, 0.100))
    with pytest.raises(TypeError):
        V.check_finger(finger, 0)


def test_require_valid_reports_every_reason():
    """The loud counterpart to is_valid."""
    palm = G.Palm(0.025, 0.060, 0.060)
    good = S.seed_population(0, 1)[0]
    assert V.require_valid(good) is good

    bad = G.Hand(palm, (G.Finger(G.Mount("+y", 0.5, 0.99, 0.0),
                                 (G.Segment(G.Joint(0.0), 0.5),)),))
    with pytest.raises(ValueError) as e:
        V.require_valid(bad)
    assert "length" in str(e.value) and "mount" in str(e.value)


def test_segment_distance_matches_brute_force():
    """The closed form must never OVERESTIMATE -- the dangerous direction, since
    an overestimating clearance check reports parts as clear when they overlap.
    Clamping both parameters independently does exactly that, while leaving every
    degenerate case correct."""
    rng = np.random.default_rng(0)
    for _ in range(400):
        p0, p1, q0, q1 = (rng.normal(size=3) for _ in range(4))
        closed = K.segment_distance(p0, p1, q0, q1)
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
    got = K.segment_distance(*(np.array(x, float) for x in (p0, p1, q0, q1)))
    assert got == pytest.approx(want, abs=1e-9)


def test_move_mount_moves_both_axes(pop):
    """Both face axes must move, not just the roomy one. The thin axis has a 5 mm
    band against a 5 mm step, so every step overflows it toward a face that hosts
    no finger; refusing those froze the axis completely."""
    rng = random.Random(0)
    moved_u = moved_v = 0
    for hand in pop[:40]:
        for _ in range(40):
            child = M.mutate(rng, hand, "move_mount")
            if child is None:
                continue
            for a, b in zip(hand.fingers, child.fingers):
                moved_u += abs(a.mount.u - b.mount.u) > 1e-12
                moved_v += abs(a.mount.v - b.mount.v) > 1e-12
            hand = child
    assert moved_u > 0, "the thin axis never moved"
    assert moved_v > 0, "the long axis never moved"


def test_every_genotype_field_is_validated():
    """No field may go out of bounds unnoticed. Written as a sweep so that adding
    a genotype field without a rule shows up here."""
    from dataclasses import replace as _replace

    hand = S.seed_population(0, 1)[0]
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
        new_palm = _replace(palm, **{k[2:]: v for k, v in kw.items()
                                     if k.startswith("p_")})
        return G.Hand(new_palm,
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
        "mount alpha off grid": variant(m_alpha=0.1),
        "alpha zero with beta set": variant(m_alpha=0.0, m_beta=G.ANGLE_QUANTUM),
        "too many fingers": G.Hand(palm, hand.fingers * 4),
        "too few fingers": G.Hand(palm, hand.fingers[:1]),
    }
    missed = [name for name, bad in cases.items() if not V.check(bad)]
    assert not missed, f"validator missed: {missed}"


def test_the_grammar_is_deterministic():
    """Same seed, same population and same walk -- the basis for a design being
    identifiable, memoisable and re-evaluatable."""
    assert S.seed_population(7, 20) == S.seed_population(7, 20)

    def walk(seed):
        rng = random.Random(seed)
        hand = S.seed_population(0, 1)[0]
        for _ in range(400):
            child = M.mutate(rng, hand)
            if child:
                hand = child
        return hand

    assert walk(3) == walk(3)
    assert walk(3) != walk(4)


def test_crossing_a_face_rotates_the_finger_with_it():
    """A finger keeps its relationship to the face it is on, so crossing an edge
    rotates its world direction by the angle between the normals. Preserving the
    world direction instead lays the finger flat along the surface it is bolted
    to."""
    palm = G.Palm(0.025, 0.060, 0.060)
    mount = G.Mount("+y", 0.5, 0.80, alpha=0.0, beta=0.0)
    before = K.mount_direction(mount, palm)

    for _ in range(8):
        nxt = M._step_mount(mount, palm, 0.0, +M.MOUNT_STEP_M)
        if nxt is None or nxt == mount:
            break
        mount = nxt
        if mount.face != "+y":
            break
    assert mount.face == "+z", "the walk never crossed"
    assert mount.alpha == pytest.approx(0.0), "tilt relative to the face changed"

    after = K.mount_direction(mount, palm)
    angle = math.degrees(math.acos(float(np.clip(before @ after, -1, 1))))
    assert angle == pytest.approx(90.0, abs=1e-6), (
        f"world direction rotated {angle:.1f} deg, expected the 90 deg between "
        f"the two face normals")


def test_no_crossing_changes_the_tilt(pop):
    rng = random.Random(0)
    crossings = 0
    for hand in pop[:60]:
        for _ in range(60):
            child = M.mutate(rng, hand, "move_mount")
            if child is None:
                continue
            for a, b in zip(hand.fingers, child.fingers):
                if a.mount.face != b.mount.face:
                    crossings += 1
                    assert a.mount.alpha == pytest.approx(b.mount.alpha)
                    assert a.mount.beta == pytest.approx(b.mount.beta)
            hand = child
    assert crossings > 20, f"only {crossings} crossings seen; test is not exercising"
