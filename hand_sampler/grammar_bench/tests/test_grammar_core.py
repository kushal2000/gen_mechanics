"""The locked grammar (GRAMMAR-LOCK-2026-10-07): genotype, derived kinematics,
rules, random hands and mutation operators."""

from __future__ import annotations

import math
import os
from dataclasses import replace

import numpy as np
import pytest

from hand_sampler.grammar import derive as dv
from hand_sampler.grammar import operators as ops
from hand_sampler.grammar.adapters.urdf import load_urdf, to_urdf
from hand_sampler.grammar.fk import forward_kinematics, pose_to_matrix
from hand_sampler.grammar.hand import (
    EVOLUTION_RULES,
    KIND_AXIS,
    NO_RULES,
    PALM_OUTLINE_MAX_VERTICES,
    STEPS,
    Finger,
    Hand,
    Joint,
    PalmJoint,
    Rules,
    canonical_axis,
    check,
    finger_length_mm,
    hand_from_dict,
    hand_to_dict,
    parameter_count,
)

N_SAMPLES = int(os.environ.get("GRAMMAR_SAMPLES", "2000"))
N_CHAIN = int(os.environ.get("GRAMMAR_CHAIN", "2000"))


def simple_hand() -> Hand:
    row = [Finger(y=y, z=90, joints=(Joint("hinge", (90, 0), 0), Joint("hinge", (0, 0), 45),
                                      Joint("coupled", (0, 0), 30), Joint("hinge", (0, 0), 25)))
           for y in (-30, 0, 30)]
    thumb = Finger(y=55, z=40, facing=90, tilt=45, joints=(Joint("hinge", (0, 90), 20), Joint("hinge", (0, 0), 40),
                                                        Joint("hinge", (0, 0), 30)))
    return Hand(fingers=tuple(row) + (thumb,))


# --------------------------------------------------------------------------
# Genotype and rules
# --------------------------------------------------------------------------


def test_simple_hand_is_valid_and_round_trips():
    h = simple_hand()
    assert check(h, EVOLUTION_RULES) == []
    assert hand_from_dict(hand_to_dict(h)) == h


def test_canonical_axis_is_a_line():
    for az in range(-360, 361, 5):
        for el in range(-90, 91, 5):
            a, e = canonical_axis(az, el)
            assert 0 <= a < 180 and -90 <= e <= 90
            u, v = dv.direction(az, el), dv.direction(a, e)
            assert abs(abs(float(u @ v)) - 1.0) < 1e-12
            assert canonical_axis(a, e) == (a, e)


@pytest.mark.parametrize("edit,needle", [
    (lambda h: replace(h, fingers=h.fingers[:1]), "fingers, allowed"),
    (lambda h: replace(h, fingers=h.fingers + h.fingers[:3]), "fingers, allowed"),
    (lambda h: replace(h, fingers=(replace(h.fingers[0], joints=h.fingers[0].joints * 2),) + h.fingers[1:]),
     "joints, allowed"),
    (lambda h: replace(h, fingers=(replace(h.fingers[0], y=-10),) + h.fingers[1:]), "apart, need 19"),
    (lambda h: replace(h, fingers=(replace(h.fingers[0], joints=(Joint("hinge", (0, 0), 7),) + h.fingers[0].joints[1:]),)
                       + h.fingers[1:]), "link length"),
    (lambda h: replace(h, fingers=(replace(h.fingers[0], joints=h.fingers[0].joints[:-1] + (Joint("hinge", (0, 0), 0),)),)
                       + h.fingers[1:]), "fingertip 10-90"),
    (lambda h: replace(h, fingers=(replace(h.fingers[0], joints=h.fingers[0].joints[:-1] + (Joint("hinge", (0, 0), 91),)),)
                       + h.fingers[1:]), "link length"),
    (lambda h: replace(h, fingers=(replace(h.fingers[0], joints=tuple(Joint("hinge", (0, 0), 90) for _ in range(3))),)
                       + h.fingers[1:]), "mm long, allowed 250"),
    (lambda h: replace(h, fingers=(replace(h.fingers[0], joints=(Joint("coupled", (0, 0), 40),)),) + h.fingers[1:]),
     "coupled joint must follow"),
    (lambda h: replace(h, fingers=(replace(h.fingers[0], joints=(Joint("hinge", (3, 0), 40),)),) + h.fingers[1:]),
     "grid"),
    (lambda h: replace(h, fingers=(replace(h.fingers[0], joints=(Joint("hinge", (180, 0), 40),)),) + h.fingers[1:]),
     "not canonical"),
    (lambda h: replace(h, fingers=(replace(h.fingers[0], facing=7),) + h.fingers[1:]), "grid"),
    (lambda h: replace(h, fingers=(replace(h.fingers[0], y=400),) + h.fingers[1:]), "from the wrist centre"),
    (lambda h: replace(h, palm_joints=(PalmJoint(0, 40),)), "carries no finger"),
    (lambda h: replace(h, fingers=(replace(h.fingers[0], palm_joint=2),) + h.fingers[1:]), "does not exist"),
])
def test_check_catches(edit, needle):
    out = check(edit(simple_hand()), NO_RULES)
    assert any(needle in o for o in out), out


def test_sliding_joints_only_without_rules():
    h = simple_hand()
    f = h.fingers[0]
    h2 = replace(h, fingers=(replace(f, joints=(Joint("sliding", (90, 0), 0),) + f.joints[1:]),) + h.fingers[1:])
    assert check(h2, NO_RULES) == []
    assert any("sliding joints not allowed" in o for o in check(h2, EVOLUTION_RULES))


def test_rules_clamp_never_widen_the_grammar():
    r = Rules(max_fingers=9, max_joints=8, max_palm_joints=9, max_finger_length_mm=400, min_spacing_mm=5,
              base_distance_mm=(0, 999)).clamp()
    assert r.max_fingers == 6 and r.max_joints == 5 and r.max_palm_joints == 6
    assert r.max_finger_length_mm == 250 and r.min_spacing_mm == 19
    assert r.base_distance_mm == NO_RULES.base_distance_mm


def test_parameter_count():
    c = parameter_count(simple_hand())
    assert c["fingers"] == 4 and c["joints"] == 15 and c["numbers"] == 4 * 4 + 3 * 15


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
    for _ in range(60):
        h = ops.random_hand(rng, NO_RULES)
        for _ in range(3):
            q = dv.sample_q(h, 1, rng)[0]
            assert np.abs(_model_tips(h, q) - dv.fk(h, q).tips).max() < 1e-12


def test_derived_signs_close_or_spread():
    """+ moves the fingertip toward the link's closing direction (+y of the
    link frame); a joint that cannot (abduction) spreads toward +y of the
    palm; a pure roll turns right-handed about the link."""
    rng = np.random.default_rng(1)
    for _ in range(200):
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


def test_flexion_closes_toward_the_palm_normal():
    h = simple_hand()
    q0 = np.zeros(len(dv.dofs(h)))
    ds = dv.dofs(h)
    k = next(k for k, d in enumerate(ds) if d.name == "f1_j1")
    q = q0.copy()
    q[k] = 0.3
    assert dv.fk(h, q).tips[1][0] > dv.fk(h, q0).tips[1][0] + 0.01   # toward +x (the grasp side)


def test_ranges_by_kind():
    h = simple_hand()
    ds = {d.name: d for d in dv.dofs(h)}
    assert ds["f0_j0"].kind == "abduction" and np.allclose(np.degrees(ds["f0_j0"].limits), (-30, 30))
    assert ds["f0_j1"].kind == "flexion" and np.allclose(np.degrees(ds["f0_j1"].limits), (-30, 90))
    assert ds["f3_j0"].kind == "roll" and np.allclose(np.degrees(ds["f3_j0"].limits), (-90, 90))
    assert ds["f0_j2"].type == "coupled" and np.allclose(np.degrees(ds["f0_j2"].limits), (-33, 99))
    assert ds["f0_j2"].source == list(ds).index("f0_j1")


def test_coupled_joint_follows_at_1_1():
    h = simple_hand()
    ds = dv.dofs(h)
    q = np.zeros(len(ds))
    k = [d.name for d in ds].index("f0_j1")
    q[k] = 0.4
    assert dv.tie(h, q)[k + 1] == pytest.approx(0.44)


def test_links_are_straight_at_zero():
    rng = np.random.default_rng(2)
    for _ in range(50):
        h = ops.random_hand(rng, NO_RULES)
        p = dv.fk(h)
        for i, f in enumerate(h.fingers):
            x = dv.finger_frame(f)[:, 0]
            tip = dv.mount_point_m(f) + x * finger_length_mm(f) * 1e-3
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


def test_palm_outlines_fit_the_gpu_hull_limits():
    rng = np.random.default_rng(4)
    for _ in range(200):
        h = ops.random_hand(rng, NO_RULES)
        outs = dv.palm_outlines_mm(h)
        assert set(outs) == {-1} | set(range(len(h.palm_joints)))
        for k, poly in outs.items():
            assert 3 <= len(poly) <= PALM_OUTLINE_MAX_VERTICES
            # convex and counter-clockwise
            e = np.roll(poly, -1, axis=0) - poly
            cross = e[:, 0] * np.roll(e, -1, axis=0)[:, 1] - e[:, 1] * np.roll(e, -1, axis=0)[:, 0]
            assert (cross > -1e-9).all()
        main = outs[-1]
        assert dv.point_in_polygon(np.array([[0.0, 0.0]]), main)[0]          # the heel disc at the wrist
        for f in h.fingers:
            poly = outs[f.palm_joint]
            assert dv.point_in_polygon(np.array([[f.y, f.z]]), poly)[0]     # every base on its plate


def test_link_cores_are_thick_enough_for_gpu_cooking():
    for L, tip in ((0, False), (15, False), (90, False), (10, True), (90, True)):
        x0, x1 = dv.link_core_x_mm(L, tip)
        assert x1 - x0 >= 1.0 - 1e-12
    assert dv.CORE_HALF_Y_MM * 2 >= 1.0 and dv.CORE_HALF_Z_MM * 2 >= 1.0


# --------------------------------------------------------------------------
# Random hands
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rules", [EVOLUTION_RULES, NO_RULES,
                                   Rules(joint_types=("hinge",), max_fingers=4, max_joints=3, max_palm_joints=1,
                                         max_finger_length_mm=150, min_spacing_mm=25)],
                         ids=["evolution", "none", "custom"])
@pytest.mark.parametrize("stage", ["coarse", "fine"])
def test_random_hands_keep_every_rule(rules, stage):
    rng = np.random.default_rng(5)
    mm, deg = STEPS[stage]
    for _ in range(N_SAMPLES // 4):
        h = ops.random_hand(rng, rules, stage)
        assert check(h, rules) == []
        if stage == "coarse":
            for f in h.fingers:
                assert f.facing % deg == 0 and f.tilt % deg == 0
                assert all(j.axis[0] % deg == 0 and j.axis[1] % deg == 0 for j in f.joints)


def test_random_hands_start_on_kinds_with_the_mix():
    rng = np.random.default_rng(6)
    counts = {k: 0 for k in KIND_AXIS}
    palm = n = 0
    for _ in range(N_SAMPLES):
        h = ops.random_hand(rng, EVOLUTION_RULES)
        n += 1
        palm += len(h.palm_joints) > 0
        for f in h.fingers:
            for jt in f.joints:
                kind = next(k for k, a in KIND_AXIS.items() if a == jt.axis)
                counts[kind] += 1
    total = sum(counts.values())
    assert abs(counts["flexion"] / total - 0.69) < 0.03
    assert abs(counts["abduction"] / total - 0.27) < 0.03
    assert abs(palm / n - ops.P_PALM_JOINT) < 0.05


def test_random_hands_are_uniform_over_the_limits():
    """No hand-shaped template: bases all around the wrist, facings and tilts
    over their whole ranges, finger and joint counts over their ranges."""
    rng = np.random.default_rng(16)
    bearings, facings, tilts, nf, nj = [], set(), set(), set(), set()
    for _ in range(N_SAMPLES // 2):
        h = ops.random_hand(rng, EVOLUTION_RULES)
        nf.add(len(h.fingers))
        for f in h.fingers:
            bearings.append(math.atan2(f.y, f.z))
            facings.add(f.facing)
            tilts.add(f.tilt)
            nj.add(len(f.joints))
    R = abs(np.mean(np.exp(1j * np.array(bearings))))
    assert R < 0.1                                  # bases spread all around the wrist centre
    assert facings == set(range(0, 360, 30)) and tilts == {-30, 0, 30, 60, 90}
    assert nf == {2, 3, 4, 5, 6} and nj == {1, 2, 3, 4, 5}


# --------------------------------------------------------------------------
# Mutation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rules", [EVOLUTION_RULES, NO_RULES], ids=["evolution", "none"])
@pytest.mark.parametrize("stage", ["coarse", "fine"])
def test_long_mutation_chains_keep_every_rule(rules, stage):
    rng = np.random.default_rng(7)
    used = set()
    for c in range(10):
        h = ops.random_hand(rng, rules, "coarse")
        for _ in range(N_CHAIN // 10):
            h, name, _ = ops.mutate(h, rng, rules, stage)
            used.add(name)
            assert check(h, rules) == [], (name, check(h, rules))
    expected = {op.name for op in ops.operators_for(stage)}
    assert used == expected, expected - used


def _diff(a: Hand, b: Hand):
    """Every value that differs between two hands with the same structure."""
    out = []
    for i, (fa, fb) in enumerate(zip(a.fingers, b.fingers)):
        for k in ("y", "z", "facing", "tilt", "palm_joint"):
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


@pytest.mark.parametrize("stage", ["coarse", "fine"])
def test_parameter_steps_change_one_value_by_one_step(stage):
    rng = np.random.default_rng(8)
    mm, deg = STEPS[stage]
    for _ in range(40):
        h = ops.random_hand(rng, EVOLUTION_RULES, "coarse")
        for op in ops.operators_for(stage):
            if op.structural:
                continue
            for mv in ops.candidates(h, op, EVOLUTION_RULES, stage):
                child = op.apply(h, mv, stage)
                d = _diff(h, child)
                assert len(d) == 1, (op.name, d)
                (item,) = d
                if op.name == "move_finger":
                    assert abs(item[3] - item[2]) == mm
                elif op.name in ("turn_finger",):
                    assert min((item[3] - item[2]) % 360, (item[2] - item[3]) % 360) == deg
                elif op.name == "tilt_finger":
                    assert abs(item[3] - item[2]) == deg
                elif op.name == "length":
                    a, b = item[3], item[4]
                    assert abs(b - a) == mm or {a, b} & {0} and max(a, b) <= 15 + mm
                elif op.name in ("axis_az", "axis_el"):
                    assert _angle_between_lines(item[3], item[4]) <= deg + 1e-9
                elif op.name == "move_hinge":
                    pa, pb = item[2], item[3]
                    assert abs(pa.y - pb.y) + abs(pa.z - pb.z) == mm
                elif op.name in ("hinge_az", "hinge_el"):
                    assert _angle_between_lines(item[2].axis, item[3].axis) <= deg + 1e-9


def test_length_steps_skip_the_gap():
    assert ops.new_length(15, -1, 1, False) == 0
    assert ops.new_length(0, 1, 1, False) == 15
    assert ops.new_length(20, -1, 10, False) == 0
    assert ops.new_length(0, 1, 10, False) == 15
    assert ops.new_length(10, -1, 1, True) is None
    assert ops.new_length(90, 1, 1, False) is None
    assert ops.new_length(16, -1, 1, False) == 15


def test_fine_stage_has_no_structural_operators():
    assert all(not op.structural for op in ops.operators_for("fine"))
    assert {op.name for op in ops.operators_for("coarse")} == {op.name for op in ops.OPERATORS}


def test_coarse_is_inside_fine():
    """Every coarse value lies on the fine grid, so every hand the coarse stage
    reaches is a fine hand, and a coarse step is a whole number of fine steps."""
    rng = np.random.default_rng(9)
    cm, cd = STEPS["coarse"]
    fm, fd = STEPS["fine"]
    assert cm % fm == 0 and cd % fd == 0
    h = ops.random_hand(rng, EVOLUTION_RULES, "coarse")
    for _ in range(300):
        h, _, _ = ops.mutate(h, rng, EVOLUTION_RULES, "coarse")
        assert check(h, NO_RULES) == []                 # NO_RULES checks the fine grid


@pytest.mark.parametrize("stage", ["coarse", "fine"])
def test_parameter_steps_are_inverse_pairs(stage):
    rng = np.random.default_rng(10)
    checked = 0
    for _ in range(30):
        h = ops.random_hand(rng, EVOLUTION_RULES, "coarse")
        for op in ops.operators_for(stage):
            if op.structural:
                continue
            for mv in ops.candidates(h, op, EVOLUTION_RULES, stage):
                child = op.apply(h, mv, stage)
                back = op.apply(child, mv[:-1] + (-mv[-1],), stage)
                if op.name == "length":
                    a = h.fingers[mv[0]].joints[mv[1]].length
                    b = child.fingers[mv[0]].joints[mv[1]].length
                    if 0 in (a, b):         # the gap: 0 <-> 15 only
                        continue
                if op.name in ("axis_el", "hinge_el"):
                    ax = (child.fingers[mv[0]].joints[mv[1]].axis if op.name == "axis_el"
                          else child.palm_joints[mv[0]].axis)
                    if ax == (0, 90):       # through the pole, az is not kept
                        continue
                assert back == h, (op.name, mv)
                checked += 1
    assert checked > 1000


def test_structural_inverse_pairs():
    rng = np.random.default_rng(11)
    for _ in range(40):
        h = ops.random_hand(rng, EVOLUTION_RULES, "coarse")
        for mv in ops.candidates(h, ops.OPERATOR_BY_NAME["split"], EVOLUTION_RULES):
            child = ops.OPERATOR_BY_NAME["split"].apply(h, mv, "coarse")
            i, j = mv[0], mv[1]
            assert finger_length_mm(child.fingers[i]) == finger_length_mm(h.fingers[i])
            assert ops.OPERATOR_BY_NAME["merge"].apply(child, (i, j + 1), "coarse") == h
        for mv in ops.candidates(h, ops.OPERATOR_BY_NAME["add_finger"], EVOLUTION_RULES):
            child = ops.OPERATOR_BY_NAME["add_finger"].apply(h, mv, "coarse")
            assert ops.OPERATOR_BY_NAME["remove_finger"].apply(child, (len(h.fingers),), "coarse") == h
            break
        for mv in ops.candidates(h, ops.OPERATOR_BY_NAME["couple"], EVOLUTION_RULES):
            child = ops.OPERATOR_BY_NAME["couple"].apply(h, mv, "coarse")
            assert ops.OPERATOR_BY_NAME["couple"].apply(child, mv, "coarse") == h
        for mv in ops.candidates(h, ops.OPERATOR_BY_NAME["own_palm_joint"], EVOLUTION_RULES):
            if h.fingers[mv[0]].palm_joint >= 0:
                continue
            child = ops.OPERATOR_BY_NAME["own_palm_joint"].apply(h, mv, "coarse")
            k = child.fingers[mv[0]].palm_joint
            assert ops.OPERATOR_BY_NAME["remove_palm_joint"].apply(child, (k,), "coarse") == h
            assert ops.OPERATOR_BY_NAME["move_to_section"].apply(child, (mv[0], -1), "coarse") == h


def test_remove_any_finger_in_one_step():
    rng = np.random.default_rng(12)
    for _ in range(50):
        h = ops.random_hand(rng, EVOLUTION_RULES)
        if len(h.fingers) <= 2:
            continue
        cands = ops.candidates(h, ops.OPERATOR_BY_NAME["remove_finger"], EVOLUTION_RULES)
        assert sorted(c[0] for c in cands) == list(range(len(h.fingers)))
        for (i,) in cands:
            child = ops.OPERATOR_BY_NAME["remove_finger"].apply(h, (i,), "coarse")
            assert len(child.fingers) == len(h.fingers) - 1
            assert check(child, EVOLUTION_RULES) == []      # a palm joint left without a finger went too


def test_add_finger_is_one_bending_joint_at_a_free_spot():
    rng = np.random.default_rng(13)
    h = Hand(fingers=simple_hand().fingers[:2])
    child, mv = ops.apply_operator(h, "add_finger", rng, EVOLUTION_RULES)
    f = child.fingers[-1]
    assert len(f.joints) == 1 and f.joints[0].axis == KIND_AXIS["flexion"] and f.joints[0].type == "hinge"
    assert check(child, EVOLUTION_RULES) == []


def test_operators_that_cannot_act_have_no_candidates():
    h = Hand(fingers=simple_hand().fingers[:2])           # two fingers, no palm joint
    assert not ops.can_act(h, ops.OPERATOR_BY_NAME["remove_finger"], EVOLUTION_RULES)
    assert not ops.can_act(h, ops.OPERATOR_BY_NAME["remove_palm_joint"], EVOLUTION_RULES)
    assert not ops.can_act(h, ops.OPERATOR_BY_NAME["move_hinge"], EVOLUTION_RULES)
    four = Hand(fingers=simple_hand().fingers[:2])        # both fingers have 4 joints
    assert ops.can_act(four, ops.OPERATOR_BY_NAME["split"], EVOLUTION_RULES)
    assert not ops.can_act(four, ops.OPERATOR_BY_NAME["split"], replace(EVOLUTION_RULES, max_joints=4))


def test_mutating_a_hand_outside_the_rules_never_adds_violations():
    h = simple_hand()
    f = h.fingers[0]
    h = replace(h, fingers=(replace(f, joints=(Joint("sliding", (90, 0), 0),) + f.joints[1:]),) + h.fingers[1:])
    n0 = len(check(h, EVOLUTION_RULES))
    rng = np.random.default_rng(14)
    for _ in range(200):
        h, _, _ = ops.mutate(h, rng, EVOLUTION_RULES)
        assert len(check(h, EVOLUTION_RULES)) <= n0 and check(h, NO_RULES) == []
