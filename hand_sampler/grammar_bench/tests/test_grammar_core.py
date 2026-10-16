"""The grammar (GRAMMAR-LOCK-2026-10-07, revised 2026-10-14): genotype,
rules, derived kinematics and geometry, random hands and mutation
operators."""

from __future__ import annotations

import itertools
import math
import os
from dataclasses import replace

import numpy as np
import pytest

from hand_sampler.grammar import conform as CF
from hand_sampler.grammar import derive as dv
from hand_sampler.grammar import motor as mt
from hand_sampler.grammar import operators as ops
from hand_sampler.grammar import viability as V
from hand_sampler.grammar.adapters.urdf import load_urdf, to_urdf
from hand_sampler.grammar.fk import forward_kinematics, pose_to_matrix
from hand_sampler.grammar.hand import (
    EVOLUTION_RULES,
    FEETECH_RULES,
    KIND_AXIS,
    KIND_MIX,
    MIN_CUT_OFFSET_MM,
    NO_RULES,
    PALM_OUTLINE_MAX_VERTICES,
    PLATE_HALF_MM,
    Finger,
    Hand,
    Joint,
    PalmJoint,
    Rules,
    allowed_lengths,
    bend_first,
    canonical_axis,
    check,
    finger_length_mm,
    finger_section,
    hand_from_dict,
    hand_to_dict,
    parameter_count,
    turned_servo,
)

N_SAMPLES = int(os.environ.get("GRAMMAR_SAMPLES", "2000"))
N_CHAIN = int(os.environ.get("GRAMMAR_CHAIN", "2000"))

BEND, SPREAD, TWIST = KIND_AXIS["flexion"], KIND_AXIS["abduction"], KIND_AXIS["roll"]


def _finger(y, z, facing=0, tilt=0, joints=((BEND, 50), (BEND, 50), (BEND, 30))) -> Finger:
    return Finger(y=y, z=z, facing=facing, tilt=tilt, joints=tuple(Joint("hinge", a, L) for a, L in joints))


def simple_hand() -> Hand:
    """A buildable hand: a row of four bending fingers and a 2-2-1 thumb
    (spread-bend-spread-bend-bend, both spreads crossing the next bend)."""
    row = [_finger(y, 40) for y in (-80, -40, 0, 40)]
    thumb = _finger(80, -40, facing=90, joints=((SPREAD, 0), (BEND, 50), (SPREAD, 0), (BEND, 50), (BEND, 30)))
    return Hand(fingers=tuple(row) + (thumb,))


def general_hand() -> Hand:
    """A hand under the grammar's own limits only: coupled joints, a spread
    knuckle crossing a bend, a twisting thumb, short links."""
    row = [Finger(y=y, z=80, joints=(Joint("hinge", SPREAD, 0), Joint("hinge", BEND, 40),
                                      Joint("coupled", BEND, 30), Joint("hinge", BEND, 20)))
           for y in (-40, 0, 40)]
    thumb = Finger(y=60, z=30, facing=90, tilt=30, joints=(Joint("hinge", TWIST, 20), Joint("hinge", BEND, 40),
                                                        Joint("hinge", BEND, 30)))
    return Hand(fingers=tuple(row) + (thumb,))


# --------------------------------------------------------------------------
# Genotype and rules
# --------------------------------------------------------------------------


def test_simple_hand_is_buildable_clear_and_round_trips():
    h = simple_hand()
    assert check(h, FEETECH_RULES) == []
    assert V.c1_self_overlap(h).ok and V.c2_workspace_overlap(h).ok
    assert hand_from_dict(hand_to_dict(h)) == h


def test_general_hand_is_in_the_grammar_only():
    h = general_hand()
    assert check(h, NO_RULES) == []
    out = check(h, FEETECH_RULES)
    assert any("coupled joints not allowed" in o for o in out)
    assert any("off the 40 mm grid" in o for o in out)
    assert hand_from_dict(hand_to_dict(h)) == h


def test_canonical_axis_is_a_line():
    for az in range(-360, 361, 30):
        for el in range(-90, 91, 30):
            a, e = canonical_axis(az, el)
            assert 0 <= a < 180 and -90 <= e <= 90
            u, v = dv.direction(az, el), dv.direction(a, e)
            assert abs(abs(float(u @ v)) - 1.0) < 1e-12
            assert canonical_axis(a, e) == (a, e)


def _edit_finger(h: Hand, i: int, **kw) -> Hand:
    fs = list(h.fingers)
    fs[i] = replace(fs[i], **kw)
    return replace(h, fingers=tuple(fs))


def _edit_joints(h: Hand, i: int, joints) -> Hand:
    return _edit_finger(h, i, joints=tuple(Joint("hinge", a, L) for a, L in joints))


@pytest.mark.parametrize("edit,needle", [
    (lambda h: replace(h, fingers=h.fingers[:1]), "fingers, allowed"),
    (lambda h: replace(h, fingers=h.fingers + h.fingers[:2]), "fingers, allowed"),
    (lambda h: _edit_joints(h, 0, ((BEND, 50),) * 6), "joints, allowed"),
    (lambda h: _edit_finger(h, 0, y=-70), "off the 40 mm grid"),
    (lambda h: _edit_finger(h, 2, z=0), "from the palm centre, allowed 40-90"),
    (lambda h: _edit_finger(h, 0, y=-80, z=40, facing=90), "faces back across the palm"),
    (lambda h: _edit_joints(h, 0, ((BEND, 20), (BEND, 40))), "not allowed (30-90"),          # 30-40 turn the next bend servo
    (lambda h: _edit_joints(h, 0, ((BEND, 40), (BEND, 30))), "fingertip 40-90"),             # after a turned bend servo
    (lambda h: _edit_joints(h, 0, ((SPREAD, 30), (SPREAD, 30))), "not allowed (40-90"),
    (lambda h: _edit_joints(h, 0, ((BEND, 0), (TWIST, 30))), "link length 0 mm not allowed"),
    (lambda h: _edit_joints(h, 0, ((BEND, 0), (SPREAD, 0), (BEND, 30))), "link length 0 mm not allowed"),
    (lambda h: _edit_joints(h, 0, ((BEND, 50), (BEND, 95))), "link length 95"),
    (lambda h: _edit_joints(h, 0, ((BEND, 90), (BEND, 90), (BEND, 90))), "mm long, allowed 225"),
    (lambda h: _edit_joints(h, 0, ((BEND, 55), (BEND, 30))), "link length 55"),
    (lambda h: _edit_joints(h, 0, ((BEND, 50), ((5, 0), 30))), "axis off the 30 degree grid"),
    (lambda h: _edit_joints(h, 0, ((BEND, 50), ((180, 0), 30))), "not canonical"),
    (lambda h: _edit_finger(h, 0, facing=10), "grid"),
    (lambda h: _edit_finger(h, 0, tilt=120), "tilt 120 outside"),
    (lambda h: _edit_finger(h, 0, tilt=-30), "tilt -30 outside"),       # never under the palm
    (lambda h: _edit_finger(h, 0, tilt=30), "with a bend first"),       # the first bend already tilts it
    (lambda h: replace(h, palm_joints=(PalmJoint(0, 20),)), "through the mount"),
    (lambda h: replace(h, palm_joints=(PalmJoint(0, 15),)), "off the 10 mm grid"),
    (lambda h: replace(h, palm_joints=(PalmJoint(0, 300),)), "no finger beyond the cut"),
    (lambda h: replace(h, palm_joints=(PalmJoint(0, 30), PalmJoint(90, 30))), "palm joints, allowed at most 1"),
])
def test_check_catches(edit, needle):
    out = check(edit(simple_hand()), FEETECH_RULES)
    assert any(needle in o for o in out), out


def test_coupled_and_sliding_joints_only_without_rules():
    h = general_hand()
    f = h.fingers[3]
    h2 = _edit_finger(h, 3, joints=(Joint("sliding", SPREAD, 20),) + f.joints[1:])
    assert check(h2, NO_RULES) == []
    assert any("sliding joints not allowed" in o for o in check(h2, FEETECH_RULES))
    assert any("coupled joints not allowed" in o for o in check(h, FEETECH_RULES))


def test_allowed_lengths_follow_the_motor_floors():
    f = _finger(0, 40, joints=((SPREAD, 50), (BEND, 50), (TWIST, 50), (BEND, 30)))
    assert allowed_lengths(f, 0, True, 90) == [0] + list(range(30, 91, 10))    # spread then bend (30 turns it); may cross
    assert allowed_lengths(f, 1, True, 90) == list(range(40, 91, 10))           # bend then twist: no crossing
    assert allowed_lengths(f, 2, True, 90) == list(range(30, 91, 10))           # twist then bend (30-40 turn it)
    assert allowed_lengths(f, 3, True, 90) == list(range(30, 91, 10))           # the fingertip (its servo lies along 50 mm)
    assert not any(turned_servo(f, j) for j in range(4))
    b = _finger(0, 40, joints=((BEND, 40), (BEND, 40), (BEND, 40)))        # 40 mm links turn the bend servos after them
    assert [turned_servo(b, j) for j in range(3)] == [False, True, True]
    assert [allowed_lengths(b, j, True, 90)[0] for j in range(3)] == [30, 40, 40]
    assert allowed_lengths(f, 0, False, 90) == [0] + list(range(10, 91, 10))
    assert allowed_lengths(f, 3, False, 90) == list(range(10, 91, 10))
    assert allowed_lengths(f, 3, False)[-1] == NO_RULES.max_link_mm              # the grammar's own longest link
    g = _finger(0, 40, joints=((SPREAD, 0), (BEND, 50), (SPREAD, 50), (BEND, 30)))
    assert 0 not in allowed_lengths(g, 1, True)                              # bend then spread, but link 0 is 0 mm


def test_rules_clamp_never_widen_the_grammar():
    r = Rules(max_fingers=9, max_joints=8, max_palm_joints=9, max_finger_length_mm=10_000, base_pitch_mm=7,
              base_distance_mm=(0, 10_000), max_link_mm=10_000).clamp()
    assert r.max_fingers == 6 and r.max_joints == 5 and r.max_palm_joints == 1
    assert r.max_finger_length_mm == NO_RULES.max_finger_length_mm and r.base_pitch_mm == 10
    assert r.base_distance_mm == NO_RULES.base_distance_mm and r.max_link_mm == NO_RULES.max_link_mm
    assert (FEETECH_RULES.max_link_mm, FEETECH_RULES.max_finger_length_mm) == (90, 225)
    assert FEETECH_RULES.clamp() == FEETECH_RULES and EVOLUTION_RULES == FEETECH_RULES


def test_parameter_count():
    c = parameter_count(simple_hand())
    assert c["fingers"] == 5 and c["joints"] == 17 and c["numbers"] == 4 * 5 + 3 * 17
    c = parameter_count(replace(simple_hand(), palm_joints=(PalmJoint(0, 40),)))
    assert c["numbers"] == 4 * 5 + 3 * 17 + 2


# --------------------------------------------------------------------------
# Derived kinematics
# --------------------------------------------------------------------------


def _model_tips(hand: Hand, q: np.ndarray) -> np.ndarray:
    model = dv.derive(hand)
    ds = dv.dofs(hand)
    q = dv.tie(hand, q, ds)
    W = forward_kinematics(model, {d.name: float(q[k]) for k, d in enumerate(ds)})
    out = []
    for fr in model.frames:
        out.append((W[fr.body] @ pose_to_matrix(fr.pose))[:3, 3])
    return np.array(out)


def test_fk_matches_the_kinematic_model():
    rng = np.random.default_rng(0)
    for _ in range(40):
        h = ops.random_hand(rng, NO_RULES if _ % 2 else FEETECH_RULES)
        for _ in range(3):
            q = dv.sample_q(h, 1, rng)[0]
            assert np.abs(_model_tips(h, q) - dv.fk(h, q).tips).max() < 1e-12


def test_finger_frame_is_tilt_defined():
    """x along the finger, y = the palm normal turned by the tilt (the
    closing direction), z = x cross y in the plate; the frame depends on the
    finger alone."""
    for fa in range(0, 360, 30):
        for t in (-30, 0, 30, 60, 90):
            B = dv.frame_of(fa, t)
            assert np.allclose(B.T @ B, np.eye(3)) and np.linalg.det(B) > 0
            assert np.allclose(B[:, 0], dv.direction(fa, t))
            assert abs(B[0, 2]) < 1e-12                          # the bend axis lies in the plate
            assert B[0, 1] == pytest.approx(math.cos(math.radians(t)))


def test_derived_signs_close_or_spread():
    """+ moves the fingertip toward the link's closing direction (+y of the
    link frame); a joint that cannot (a spread) spreads toward +y of the
    palm; a pure twist turns right-handed about the link."""
    rng = np.random.default_rng(1)
    for _ in range(100):
        h = ops.random_hand(rng, NO_RULES)
        for i, f in enumerate(h.fingers):
            B = dv.finger_frame(f)
            for j, jt in enumerate(f.joints):
                a = dv.joint_axis(h, i, j)
                v = a if jt.type == "sliding" else np.array([0.0, a[2], -a[1]])   # tip velocity, link frame
                if abs(v[1]) > 1e-9:
                    assert v[1] > 0
                elif abs((B @ v)[1]) > 1e-9:
                    assert (B @ v)[1] > 0
                elif np.linalg.norm(v) < 1e-9:
                    assert a[0] > 0


def test_every_closing_joint_lifts_the_fingertip():
    """At the zero pose + on any joint that can close the finger moves its tip
    toward the palm's front face (straight up unless the finger already
    points straight up)."""
    rng = np.random.default_rng(15)
    n = 0
    for _ in range(150):
        h = ops.random_hand(rng, FEETECH_RULES)
        ds = dv.dofs(h)
        q0 = np.zeros(len(ds))
        tip0 = dv.fk(h, q0).tips
        for k, d in enumerate(ds):
            if d.finger < 0 or d.source >= 0:
                continue
            a = dv.direction(*h.fingers[d.finger].joints[d.index].axis)
            if abs(a[2]) < 1e-9:          # cannot close: spreads or twists
                continue
            q = q0.copy()
            q[k] = 1e-4
            dx = dv.fk(h, q).tips[d.finger][0] - tip0[d.finger][0]
            if h.fingers[d.finger].tilt == 90:     # straight up: the tip moves sideways (second order down)
                assert abs(dx) < 1e-8
            else:
                assert dx > 0
            n += 1
    assert n > 300


def test_ranges_by_kind_and_coupling():
    h = general_hand()
    ds = {d.name: d for d in dv.dofs(h)}
    assert ds["f0_j0"].kind == "abduction" and np.allclose(np.degrees(ds["f0_j0"].limits), (-30, 30))
    assert ds["f0_j1"].kind == "flexion" and np.allclose(np.degrees(ds["f0_j1"].limits), (-30, 90))
    assert ds["f3_j0"].kind == "roll" and np.allclose(np.degrees(ds["f3_j0"].limits), (-90, 90))
    assert ds["f0_j2"].type == "coupled" and np.allclose(np.degrees(ds["f0_j2"].limits), (-33, 99))
    assert ds["f0_j2"].source == list(ds).index("f0_j1")
    q = np.zeros(len(ds))
    k = list(ds).index("f0_j1")
    q[k] = 0.4
    assert dv.tie(h, q)[k + 1] == pytest.approx(0.44)


def test_links_are_straight_at_zero():
    rng = np.random.default_rng(2)
    for _ in range(50):
        h = ops.random_hand(rng, NO_RULES)
        p = dv.fk(h)
        for i, f in enumerate(h.fingers):
            tip = dv.mount_point_m(f) + dv.finger_direction(f) * finger_length_mm(f) * 1e-3
            assert np.allclose(p.tips[i], tip, atol=1e-12)


def test_urdf_round_trip():
    rng = np.random.default_rng(3)
    for _ in range(10):
        h = ops.random_hand(rng, NO_RULES)
        model = dv.derive(h)
        text, _ = to_urdf(model)
        back = load_urdf(text.encode()).model
        q = dv.sample_q(h, 1, rng)[0]
        ds = dv.dofs(h)
        qm = {d.name: float(v) for d, v in zip(ds, dv.tie(h, q))}
        A, B = forward_kinematics(model, qm), forward_kinematics(back, qm)
        for b in A:
            assert np.allclose(A[b], B[b], atol=1e-9)
        assert {c.dependent for c in back.couplings} == {c.dependent for c in model.couplings}


# --------------------------------------------------------------------------
# Derived geometry
# --------------------------------------------------------------------------


@pytest.mark.parametrize("axis,depth,below_back", [(BEND, 12.0, 0.0), (SPREAD, 16.5, 9.0), (TWIST, 11.5, 14.0)])
def test_base_depth_puts_the_motor_housing_flush(axis, depth, below_back):
    f = _finger(0, 40, joints=((axis, 30),))
    assert dv.base_depth_mm(f) == pytest.approx(depth)
    housing = dv.base_motor_boxes_mm(f)[0].grown(mt.WALL_MM).corners()
    assert housing[:, 0].max() == pytest.approx(PLATE_HALF_MM)                # flush with the front face
    assert -PLATE_HALF_MM - housing[:, 0].min() == pytest.approx(below_back)   # behind the back face


def test_crossed_base_slides_the_spread_motor_behind_the_palm():
    f = _finger(0, 40, joints=((SPREAD, 0), (BEND, 50), (BEND, 30)))
    assert dv.base_depth_mm(f) == pytest.approx(12.0)
    spread, bend = dv.base_motor_boxes_mm(f)
    assert -PLATE_HALF_MM - spread.grown(mt.WALL_MM).corners()[:, 0].min() == pytest.approx(31.0)
    assert bend.grown(mt.WALL_MM).corners()[:, 0].max() == pytest.approx(PLATE_HALF_MM)


def test_links_are_servo_sized():
    """A bending link is 33 mm wide (along its axis) and 24 mm deep, and runs
    from 12 mm behind its joint to the next motor's front."""
    f = _finger(0, 40, joints=((BEND, 50), (BEND, 30)))
    b = dv.link_box_mm(f, 0)
    assert np.allclose(2 * b.half[1:], (24.0, 33.0))
    assert b.centre[0] - b.half[0] == pytest.approx(-12.0) and b.centre[0] + b.half[0] == pytest.approx(59.5)
    tip = dv.link_box_mm(f, 1)
    assert tip.centre[0] + tip.half[0] == pytest.approx(30.0)


def test_palm_outlines():
    """Convex, at most the GPU hull's vertices; the main palm holds the mount;
    every base sits on its own plate; a cut leaves a seam."""
    rng = np.random.default_rng(4)
    for _ in range(150):
        h = ops.random_hand(rng, FEETECH_RULES)
        outs = dv.palm_outlines_mm(h)
        assert set(outs) == ({-1, 0} if h.palm_joints else {-1})
        for k, poly in outs.items():
            assert 3 <= len(poly) <= PALM_OUTLINE_MAX_VERTICES
            e = np.roll(poly, -1, axis=0) - poly
            cross = e[:, 0] * np.roll(e, -1, axis=0)[:, 1] - e[:, 1] * np.roll(e, -1, axis=0)[:, 0]
            assert (cross > -1e-9).all()
        assert dv.point_in_polygon(dv.mount_outline_mm(), outs[-1], 1e-6).all()
        for i, f in enumerate(h.fingers):
            assert dv.point_in_polygon(np.array([[f.y, f.z]]), outs[finger_section(h, i)])[0]
        if h.palm_joints:
            from hand_sampler.grammar.hand import SEAM_MM, cut_side_mm
            p = h.palm_joints[0]
            assert max(cut_side_mm(p, *v) for v in outs[-1]) == pytest.approx(-SEAM_MM / 2, abs=1e-6)
            assert min(cut_side_mm(p, *v) for v in outs[0]) == pytest.approx(SEAM_MM / 2, abs=1e-6)


def test_the_cut_hinges_along_its_line_and_lifts_the_part_beyond():
    h = replace(simple_hand(), palm_joints=(PalmJoint(facing=0, offset=40),))    # the line y = -40, beyond: y < -40
    assert check(h, FEETECH_RULES) == []
    assert [finger_section(h, i) for i in range(5)] == [0, -1, -1, -1, -1]
    a, p = dv.palm_axis(h), dv.palm_hinge_m(h) * 1e3
    assert np.allclose(a, (0, 0, 1)) and p[0] == pytest.approx(0.0) and p[1] == pytest.approx(-40.0)
    ds = dv.dofs(h)
    q = np.zeros(len(ds))
    q[0] = 0.3
    assert dv.fk(h, q).tips[0][0] > dv.fk(h).tips[0][0] + 0.01          # the part beyond rises
    assert np.allclose(dv.fk(h, q).tips[1:], dv.fk(h).tips[1:])          # the rest stays


def test_object_start_is_the_lowest_clear_height():
    """Over the outline's centroid, at the lowest height (from 5 mm above the
    front face, in 1 mm steps) where the cube clears every solid by 5 mm at
    the start pose: never inside the hand, no higher than it must be."""
    h = simple_hand()
    p = dv.object_start_m(h, 0.03)
    assert p[0] == pytest.approx((PLATE_HALF_MM + 30 + 5 + 1) * 1e-3)    # the row's curled first links: 1 mm up
    assert np.allclose(p[1:] * 1e3, dv.polygon_centroid(dv.outline_mm(h)))
    rng = np.random.default_rng(18)
    raised = 0
    for _ in range(N_SAMPLES // 20):
        h = ops.random_hand(rng, FEETECH_RULES)
        p = dv.object_start_m(h, 0.03) * 1e3
        solids = [s for s in dv.solids_at(h, dv.start_q(h)) if s.key[0] != "mount"]

        def gap(x):
            cube = dv.box_corners(np.array([x, p[1], p[2]]), np.full(3, 30.0)) * 1e-3
            return min(dv.gjk_distance(cube, s.points) - s.radius for s in solids) * 1e3

        assert gap(p[0]) >= 5.0 - 1e-6
        if p[0] > PLATE_HALF_MM + 35 + 1e-6:
            raised += 1
            assert gap(p[0] - 1.0) < 5.0 - 1e-6
    assert raised > 0


# --------------------------------------------------------------------------
# Random hands
# --------------------------------------------------------------------------

CUSTOM = Rules(joint_types=("hinge",), max_fingers=4, max_joints=3, max_palm_joints=0, max_finger_length_mm=150,
               base_pitch_mm=40, base_distance_mm=(40, 90), outward_facing=True, buildable_links=True)


@pytest.mark.parametrize("rules", [FEETECH_RULES, NO_RULES, CUSTOM], ids=["feetech", "none", "custom"])
def test_random_hands_keep_every_rule_and_are_clear(rules):
    rng = np.random.default_rng(5)
    for _ in range(N_SAMPLES // 20):
        h = ops.random_hand(rng, rules)
        assert check(h, rules) == []
        assert V.c1_self_overlap(h).ok
        for f in h.fingers:
            assert all(j.axis in KIND_AXIS.values() for j in f.joints)       # every drawn axis on a kind
            assert all(j.type == "hinge" for j in f.joints)


def test_random_hands_mix_and_counts():
    """Kinds 69/27/4; finger counts uniform over 2-6 (a hand keeps its
    drawn count); half the hands cut."""
    rng = np.random.default_rng(6)
    counts = {k: 0 for k in KIND_AXIS}
    nf = {n: 0 for n in range(2, 7)}
    cut = n = 0
    for _ in range(N_SAMPLES // 4):
        h = ops.random_hand(rng, FEETECH_RULES)
        n += 1
        cut += len(h.palm_joints) > 0
        nf[len(h.fingers)] += 1
        for f in h.fingers:
            for jt in f.joints:
                counts[next(k for k, a in KIND_AXIS.items() if a == jt.axis)] += 1
    total = sum(counts.values())
    assert abs(counts["flexion"] / total - 0.69) < 0.03
    assert abs(counts["abduction"] / total - 0.27) < 0.03
    assert abs(cut / n - ops.P_PALM_JOINT) < 0.08
    assert min(nf.values()) > 0.12 * n


def test_random_hands_are_uniform_over_the_limits():
    """No hand-shaped template: bases all around the palm centre, facings and
    tilts over their whole ranges, finger and joint counts over their ranges."""
    rng = np.random.default_rng(16)
    bearings, facings, tilts, nj, spots = [], set(), set(), set(), set()
    for _ in range(N_SAMPLES // 4):
        h = ops.random_hand(rng, FEETECH_RULES)
        for f in h.fingers:
            bearings.append(math.atan2(f.y, f.z))
            facings.add(f.facing)
            tilts.add(f.tilt)
            nj.add(len(f.joints))
            spots.add((f.y, f.z))
    R = abs(np.mean(np.exp(1j * np.array(bearings))))
    assert R < 0.1
    assert facings == set(range(0, 360, 30)) and tilts == {0, 30, 60, 90}     # fingers not starting with a bend
    assert nj == {1, 2, 3, 4, 5}
    assert spots == set(ops.base_spots(FEETECH_RULES)) and len(spots) == 20


def test_random_hand_never_drops_fingers():
    """A finger that cannot be placed restarts the hand with the same count;
    a count that never fits is an error, not a smaller hand."""
    rng = np.random.default_rng(19)
    tight = replace(FEETECH_RULES, min_fingers=6, max_fingers=6, base_distance_mm=(40, 60))
    with pytest.raises(ops.NoHand):
        ops.random_hand(rng, tight)
    six = replace(FEETECH_RULES, min_fingers=6, max_fingers=6)
    for _ in range(5):
        assert len(ops.random_hand(rng, six).fingers) == 6


# --------------------------------------------------------------------------
# Mutation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rules", [FEETECH_RULES, NO_RULES], ids=["feetech", "none"])
def test_long_mutation_chains_keep_every_rule(rules):
    rng = np.random.default_rng(7)
    used = set()
    done = calls = 0
    for c in range(10):
        h = ops.random_hand(rng, rules)
        for _ in range(N_CHAIN // 10):
            res = ops.mutate(h, rng, rules)
            calls += 1
            if res is None:
                continue
            h, name, _ = res
            done += 1
            used.add(name)
            assert check(h, rules) == [], (name, check(h, rules))
            assert V.c1_self_overlap(h).ok, name                 # every mutant keeps C1
    expected = {op.name for op in ops.OPERATORS} - ({"couple"} if "coupled" not in rules.joint_types else set())
    assert used == expected, expected - used
    assert done / calls > 0.8


def _diff(a: Hand, b: Hand):
    """Every value that differs between two hands with the same structure."""
    out = []
    for i, (fa, fb) in enumerate(zip(a.fingers, b.fingers)):
        for k in ("y", "z", "facing", "tilt"):
            if getattr(fa, k) != getattr(fb, k):
                out.append((i, k, getattr(fa, k), getattr(fb, k)))
        for j, (ja, jb) in enumerate(zip(fa.joints, fb.joints)):
            for k in ("type", "length"):
                if getattr(ja, k) != getattr(jb, k):
                    out.append((i, j, k, getattr(ja, k), getattr(jb, k)))
            if ja.axis != jb.axis:
                out.append((i, j, "axis", ja.axis, jb.axis))
    for k, (pa, pb) in enumerate(zip(a.palm_joints, b.palm_joints)):
        if pa != pb:
            out.append(("palm", k, pa, pb))
    return out


def _angle_between_lines(a, b) -> float:
    return math.degrees(math.acos(min(1.0, abs(float(dv.direction(*a) @ dv.direction(*b))))))


def test_parameter_steps_change_one_value_by_one_step():
    rng = np.random.default_rng(8)
    rules = FEETECH_RULES
    seen = set()
    for _ in range(12):
        h = ops.random_hand(rng, rules)
        for op in ops.OPERATORS:
            if op.structural:
                continue
            for mv in ops.candidates(h, op, rules):
                child = op.apply(h, mv, rules)
                d = _diff(h, child)
                seen.add(op.name)
                if op.name in ("axis_az", "axis_el"):
                    axes = [x for x in d if x[2] == "axis"]
                    assert len(axes) == 1 and _angle_between_lines(axes[0][3], axes[0][4]) <= 30 + 1e-9
                    assert all(x[2] in ("axis", "length") and x[0] == mv[0] for x in d)   # lengths refit to floors
                    continue
                if op.name == "length":       # one link one step; links after it may grow (a short link turns a servo)
                    (step,) = [x for x in d if x[1] == mv[1]]
                    assert all(x[2] == "length" and x[0] == mv[0] and x[1] > mv[1] for x in d if x is not step)
                    allowed = allowed_lengths(h.fingers[mv[0]], mv[1], True)
                    assert abs(allowed.index(step[3]) - allowed.index(step[4])) == 1
                    continue
                assert len(d) == 1, (op.name, d)
                (item,) = d
                if op.name == "move_finger":
                    assert abs(item[3] - item[2]) == rules.base_pitch_mm
                elif op.name == "turn_finger":
                    assert min((item[3] - item[2]) % 360, (item[2] - item[3]) % 360) == 30
                elif op.name == "tilt_finger":
                    assert abs(item[3] - item[2]) == 30
                elif op.name == "move_cut":
                    assert abs(item[2].offset - item[3].offset) == 10
                elif op.name == "turn_cut":
                    assert min((item[3].facing - item[2].facing) % 360, (item[2].facing - item[3].facing) % 360) == 30
    assert {"move_finger", "turn_finger", "tilt_finger", "length", "axis_az", "axis_el"} <= seen


def test_length_steps_follow_the_allowed_lengths():
    f = _finger(0, 40, joints=((SPREAD, 40), (BEND, 50), (BEND, 30)))
    assert ops.step_length(f, 0, -1, FEETECH_RULES) == 30          # 30 turns the bend servo
    assert ops.step_length(_finger(0, 40, joints=((SPREAD, 30), (BEND, 50), (BEND, 30))), 0, -1, FEETECH_RULES) == 0
    g = _finger(0, 40, joints=((SPREAD, 0), (BEND, 50), (BEND, 30)))
    assert ops.step_length(g, 0, 1, FEETECH_RULES) == 30           # a crossing up to the floor
    assert ops.step_length(f, 1, -1, FEETECH_RULES) == 40          # bend then bend: 40 turns the next servo
    assert ops.step_length(_finger(0, 40, joints=((SPREAD, 40), (BEND, 30), (BEND, 40))), 1, -1, FEETECH_RULES) is None
    assert ops.step_length(f, 2, -1, FEETECH_RULES) is None        # the fingertip floor
    assert ops.step_length(_finger(0, 40, joints=((BEND, 90), (BEND, 30))), 0, 1, FEETECH_RULES) is None
    assert ops.step_length(f, 1, -1, NO_RULES) == 40


def test_fit_lengths_moves_links_to_their_nearest_allowed_length():
    f = _finger(0, 40, joints=((BEND, 30), (TWIST, 0), (BEND, 10)))
    g = ops.fit_lengths(f, FEETECH_RULES)
    assert [j.length for j in g.joints] == [40, 30, 40]     # 30 turns the last bend servo, whose tip needs 40
    assert ops.fit_lengths(g, FEETECH_RULES) == g


def test_parameter_steps_are_inverse_pairs():
    rng = np.random.default_rng(10)
    rules = FEETECH_RULES
    checked = 0
    for _ in range(12):
        h = ops.random_hand(rng, rules)
        for op in ops.OPERATORS:
            if op.structural:
                continue
            for mv in ops.candidates(h, op, rules):
                child = op.apply(h, mv, rules)
                back = op.apply(child, mv[:-1] + (-mv[-1],), rules)
                if op.name in ("axis_az", "axis_el"):
                    if len(_diff(h, child)) > 1:
                        continue                   # lengths were refit: not a pure step
                    ax = child.fingers[mv[0]].joints[mv[1]].axis
                    if ax == (0, 90):              # through the pole, az is not kept
                        continue
                if not V.c1_self_overlap(back).ok or check(back, rules):
                    continue
                assert back == h, (op.name, mv)
                checked += 1
    assert checked > 200


def test_structural_inverse_pairs():
    rng = np.random.default_rng(11)
    rules = FEETECH_RULES
    for _ in range(12):
        h = ops.random_hand(rng, rules)
        for mv in ops.candidates(h, ops.OPERATOR_BY_NAME["add_finger"], rules):
            child = ops.OPERATOR_BY_NAME["add_finger"].apply(h, mv, rules)
            assert ops.OPERATOR_BY_NAME["remove_finger"].apply(child, (len(h.fingers),), rules) == h
            break
        for mv in ops.candidates(h, ops.OPERATOR_BY_NAME["add_cut"], rules):
            child = ops.OPERATOR_BY_NAME["add_cut"].apply(h, mv, rules)
            assert ops.OPERATOR_BY_NAME["remove_cut"].apply(child, (0,), rules) == h
            break
        for mv in ops.candidates(h, ops.OPERATOR_BY_NAME["split"], rules):
            child = ops.OPERATOR_BY_NAME["split"].apply(h, mv, rules)
            i, j = mv[0], mv[1]
            assert len(child.fingers[i].joints) == len(h.fingers[i].joints) + 1
            if finger_length_mm(child.fingers[i]) == finger_length_mm(h.fingers[i]) and mv[2] not in (0, ):
                back = ops.OPERATOR_BY_NAME["merge"].apply(child, (i, j + 1), rules)
                assert len(back.fingers[i].joints) == len(h.fingers[i].joints)
    g = general_hand()
    for mv in ops.candidates(g, ops.OPERATOR_BY_NAME["couple"], NO_RULES):
        child = ops.OPERATOR_BY_NAME["couple"].apply(g, mv, NO_RULES)
        assert ops.OPERATOR_BY_NAME["couple"].apply(child, mv, NO_RULES) == g


def test_split_keeps_the_finger_length_when_the_floors_allow():
    h = Hand(fingers=(_finger(0, 40, joints=((BEND, 90), (BEND, 30))), _finger(0, -40, facing=180)))
    child = ops.OPERATOR_BY_NAME["split"].apply(h, (0, 0, 50, "abduction"), FEETECH_RULES)
    assert [(j.axis, j.length) for j in child.fingers[0].joints] == [(BEND, 50), (SPREAD, 40), (BEND, 30)]
    child = ops.OPERATOR_BY_NAME["split"].apply(h, (0, 0, 40, "abduction"), FEETECH_RULES)
    assert [j.length for j in child.fingers[0].joints] == [50, 50, 30]      # bend then spread needs 50
    child = ops.OPERATOR_BY_NAME["split"].apply(h, (0, 1, 10, "flexion"), FEETECH_RULES)
    assert [j.length for j in child.fingers[0].joints] == [90, 30, 40]      # 30 turns the new servo; its tip needs 40
    moves = ops.OPERATOR_BY_NAME["split"].moves(h, FEETECH_RULES)
    assert {m[2] for m in moves if m[:2] == (0, 0)} == set(range(0, 91, 10))


def test_remove_a_finger_in_one_step():
    """Any finger can go in one step unless the hand, moved to put the mount
    back at the palm's centre, would leave the rules (a base inside the 40 mm
    ring): nearly every hand can lose some finger, and most fingers can go."""
    rng = np.random.default_rng(12)
    some = hands = fingers = removable = 0
    for _ in range(40):
        h = ops.random_hand(rng, FEETECH_RULES)
        if len(h.fingers) <= 2:
            continue
        hands += 1
        fingers += len(h.fingers)
        cands = sorted(c[0] for c in ops.candidates(h, ops.OPERATOR_BY_NAME["remove_finger"], FEETECH_RULES))
        some += bool(cands)
        removable += len(cands)
        for i in cands:
            child = ops.recentre(ops.OPERATOR_BY_NAME["remove_finger"].apply(h, (i,), FEETECH_RULES), FEETECH_RULES)
            assert len(child.fingers) == len(h.fingers) - 1
            assert check(child, FEETECH_RULES) == []          # a cut left without a finger on a side went too
    assert some >= 0.9 * hands and removable >= 0.7 * fingers


def test_add_finger_has_the_mean_joint_count_rounded_up_at_a_free_spot():
    """The added finger has as many joints as the hand's fingers on average,
    rounded up; its joints are drawn as a random hand's (kinds from the
    commercial mix), so different draws give different fingers."""
    rng = np.random.default_rng(13)
    h = ops.recentre(Hand(fingers=simple_hand().fingers[:2]), FEETECH_RULES)
    want = math.ceil(sum(len(f.joints) for f in h.fingers) / len(h.fingers))
    assert ops.added_finger_joints(h, FEETECH_RULES) == want
    added = set()
    for _ in range(8):
        child, mv = ops.apply_operator(h, "add_finger", rng, FEETECH_RULES)
        f = child.fingers[-1]
        assert len(f.joints) == want and all(j.type == "hinge" for j in f.joints)
        assert len({(g.y, g.z) for g in child.fingers}) == len(child.fingers) and f.tilt == 0    # a free spot
        assert check(child, FEETECH_RULES) == []
        added_hand = ops.OPERATOR_BY_NAME["add_finger"].apply(h, mv, FEETECH_RULES)
        assert ops.recentre(added_hand, FEETECH_RULES) == child       # the move fixes the draw; then centred
        added.add(f.joints)
    assert len(added) > 1
    h3 = Hand(fingers=(replace(h.fingers[0], joints=h.fingers[0].joints[:1]),) + h.fingers[1:])
    assert ops.added_finger_joints(h3, FEETECH_RULES) == math.ceil((1 + len(h.fingers[1].joints)) / 2)


def test_operators_that_cannot_act_have_no_candidates():
    h = Hand(fingers=simple_hand().fingers[:2])           # two fingers, no cut
    assert not ops.can_act(h, ops.OPERATOR_BY_NAME["remove_finger"], FEETECH_RULES)
    assert not ops.can_act(h, ops.OPERATOR_BY_NAME["remove_cut"], FEETECH_RULES)
    assert not ops.can_act(h, ops.OPERATOR_BY_NAME["move_cut"], FEETECH_RULES)
    assert not ops.can_act(h, ops.OPERATOR_BY_NAME["couple"], FEETECH_RULES)
    assert ops.can_act(h, ops.OPERATOR_BY_NAME["split"], FEETECH_RULES)
    assert not ops.can_act(h, ops.OPERATOR_BY_NAME["split"], replace(FEETECH_RULES, max_joints=3))
    assert not ops.can_act(h, ops.OPERATOR_BY_NAME["add_cut"], replace(FEETECH_RULES, max_palm_joints=0))


def test_mutating_a_hand_outside_the_rules_never_adds_violations():
    h = general_hand()
    n0 = len(check(h, FEETECH_RULES))
    rng = np.random.default_rng(14)
    for _ in range(150):
        res = ops.mutate(h, rng, FEETECH_RULES)
        if res is not None:
            h = res[0]
        assert len(check(h, FEETECH_RULES)) <= n0 and check(h, NO_RULES) == []


def test_search_space_counts():
    """The Feetech rules' search space, counted exactly (no geometry): about
    10^29.7 four-finger hands with joints on the three kinds, 10^44.2 with all
    31 axis directions (E16; since 2026-10-16 fingers at most 225 mm, tilts
    0-90 degrees, tilt 0 when the first joint is a bend, and links down to 30
    mm before a bend servo, which then turns)."""
    from hand_sampler.grammar import space

    assert len(space.AXES_ALL) == 31
    assert space.chains(FEETECH_RULES, space.AXES_KINDS) == (21, 401, 6731, 44189, 122448)
    assert space.hands_log10(FEETECH_RULES)[4] == pytest.approx(29.75, abs=0.01)
    assert space.hands_log10(FEETECH_RULES, space.AXES_ALL)[4] == pytest.approx(44.24, abs=0.01)
    assert space.hands_log10(NO_RULES)[4] > space.hands_log10(FEETECH_RULES)[4] + 20


def test_a_bend_first_finger_has_tilt_0():
    """A tilt turns a finger about its bend axis, so a finger whose first joint
    is a bend has tilt 0: random hands never tilt one (their first joints keep
    the 69/27/4 kind mix), the tilt operator cannot, and a commercial fit folds
    the tilt into that joint's zero-pose offset (the same chain)."""
    rng = np.random.default_rng(21)
    first = []
    for _ in range(N_SAMPLES // 8):
        h = ops.random_hand(rng, FEETECH_RULES)
        for f in h.fingers:
            assert not (f.tilt and bend_first(f))
            first.append(f.joints[0].axis)
    assert abs(first.count(BEND) / len(first) - KIND_MIX["flexion"]) < 0.06
    h = simple_hand()                     # four bend-first fingers and a thumb starting with a spread
    assert all(bend_first(f) for f in h.fingers[:4]) and not bend_first(h.fingers[4])
    assert {mv[0] for mv in ops.candidates(h, ops.OPERATOR_BY_NAME["tilt_finger"], FEETECH_RULES)} <= {4}
    for fa in (0, 90, 210):
        for t in (30, 60, 90):
            for q in (-0.4, 0.7):
                A = dv.frame_of(fa, t) @ CF._rot(dv.direction(*BEND), q)
                B = dv.frame_of(fa, 0) @ CF._rot(dv.direction(*BEND), q + math.radians(t))
                assert np.allclose(A, B)


def test_five_joint_fingers_fit_the_length_cap():
    """Five joints on the three kinds, each link at its shortest from the palm
    out (a link that short turns the next bend servo): at most 200 mm, inside
    the Feetech cap (225 mm); five bends need 190 (30 + 40 + 40 + 40 + a 40 mm
    fingertip)."""
    tot = {}
    for kinds in itertools.product(("flexion", "abduction", "roll"), repeat=5):
        js = [Joint("hinge", KIND_AXIS[k], 90) for k in kinds]
        tot[kinds] = ops._shortest_rest_mm(Finger(y=0, z=40), js, -1, 0, FEETECH_RULES)
    assert max(tot.values()) == 200 <= FEETECH_RULES.max_finger_length_mm and tot[("flexion",) * 5] == 190
