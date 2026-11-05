"""The derived mount (2026-11-04): a hand is its fingers' arrangement; the mount goes at the clear point inside the base
servos' hull nearest the palm centre, and built hands put the wrist frame at the palm centre
(`FEETECH_COARSE_NOCUT_RULES`)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hand_sampler.grammar import derive as dv
from hand_sampler.grammar import hand as gh
from hand_sampler.grammar import operators as ops
from hand_sampler.grammar import viability as V
from hand_sampler.grammar.hand import FEETECH_COARSE_NOCUT_RULES as R, Finger, Hand, Joint, check

FLEX, ABD, ROLL = (0, 0), (90, 0), (0, 90)
# g0058_c00, MAP-Elites' best hand of the 29 Oct runs (coarse rules, mount at the centre)
G58 = Hand(fingers=(
    Finger(y=-80, z=-40, facing=210, tilt=0, joints=(Joint("hinge", FLEX, 50), Joint("hinge", FLEX, 70),
                                                     Joint("hinge", FLEX, 70), Joint("hinge", ABD, 30))),
    Finger(y=-40, z=40, facing=330, tilt=0, joints=(Joint("hinge", ABD, 0), Joint("hinge", FLEX, 50),
                                                    Joint("hinge", FLEX, 50), Joint("hinge", FLEX, 30),
                                                    Joint("hinge", FLEX, 40))),
    Finger(y=0, z=-80, facing=180, tilt=0, joints=(Joint("hinge", FLEX, 90), Joint("hinge", FLEX, 30),
                                                   Joint("hinge", FLEX, 60), Joint("hinge", ROLL, 30))),
    Finger(y=40, z=0, facing=90, tilt=0, joints=(Joint("hinge", FLEX, 30), Joint("hinge", FLEX, 40),
                                                 Joint("hinge", FLEX, 60), Joint("hinge", FLEX, 50)))))


@pytest.fixture(scope="module")
def hands():
    rng = np.random.default_rng(0)
    return [ops.random_hand(rng, R) for _ in range(40)]


def test_random_hands_keep_the_rules_with_the_mount_at_its_place(hands):
    assert {len(h.fingers) for h in hands} == set(range(2, 7))
    for h in hands:
        assert check(h, R) == [], h
        assert V.clear(h)
        assert V.mount_position_mm(h) == (0, 0)
        assert not h.palm_joints
        assert ops.canonical(h, R) == h


def test_a_hand_is_its_arrangement_translating_it_changes_nothing(hands):
    for h in hands[:15]:
        for d in ((40, 0), (6, -14), (-80, 40), (1, 1)):
            assert ops.recentre(gh.shifted(h, *d), R) == h


def test_turned_and_mirrored_copies_are_one_hand(hands):
    """Every valid turned or mirrored copy canonicalises back to the hand. (A mirror image is not always valid: a servo
    slid at a crossing keeps its side, so the copy can fail C1 -- as under the coarse rules.)"""
    checked = 0
    for h in hands[:10]:
        for k in range(4):
            for mirror in (False, True):
                t = ops.recentre(ops._turned(h, k, mirror), R)
                if check(t, R) or not V.clear(t):
                    continue
                assert ops.canonical(t, R) == h
                checked += 1
    assert checked >= 70


def test_the_mount_is_clear_and_inside_the_servos_hull(hands):
    for h in hands:
        feet = np.concatenate([dv._footprint(b) for f in h.fingers for b in dv.base_motor_boxes_mm(f)])
        assert V._inside(dv._hull2d(feet), (0, 0))


def test_removing_a_finger_no_longer_drags_the_hand_into_its_mount():
    """Under the coarse rules only fingers 0 and 1 of g0058_c00 could be removed: re-centring by a whole 40 mm step put
    a base servo on the mount. With the mount derived, every finger can go."""
    g = ops.canonical(ops.recentre(G58, R), R)
    assert check(g, R) == [] and V.clear(g)
    removed = ops.candidates(g, ops.OPERATOR_BY_NAME["remove_finger"], R)
    assert len(removed) == len(g.fingers)


def test_mutants_keep_the_rules(hands):
    rng = np.random.default_rng(1)
    for h in hands[:12]:
        for _ in range(4):
            res = ops.mutate(h, rng, R, tries=8)
            if res is None:
                continue
            child = res[0]
            assert check(child, R) == [] and V.clear(child)
            assert V.mount_position_mm(child) == (0, 0)


def test_no_palm_cuts():
    h = ops.recentre(G58, R)
    assert not ops.OPERATOR_BY_NAME["add_cut"].moves(h, R)
    cut = Hand(fingers=h.fingers, palm_joints=(gh.PalmJoint(facing=0, offset=40),))
    assert any("palm joints" in p for p in check(cut, R))


def test_the_built_hand_hangs_from_its_palm_centre():
    from hand_sampler.grammar.urdf_export import build
    g = ops.canonical(ops.recentre(G58, R), R)
    c = dv.palm_centre_mm(g) * dv.MM
    with tempfile.TemporaryDirectory() as tmp:
        row = build(g, "g", Path(tmp) / "g", rules_name="FEETECH_COARSE_NOCUT_RULES")
        spec = json.loads(Path(row["spec"]).read_text())
        assert spec["rules"] == "FEETECH_COARSE_NOCUT_RULES"
        assert np.allclose(spec["mount_in_root_m"], [0.0, -c[0], -c[1]])
        import xml.etree.ElementTree as ET
        root = ET.parse(Path(spec["urdf"])).getroot()
        model = dv.derive(g, name="g")
        for jt in model.joints:                      # joints hung from the palm moved by the palm centre, no others
            el = root.find(f"joint[@name='{jt.name}']")
            xyz = np.array([float(v) for v in el.find("origin").get("xyz").split()])
            want = np.asarray(jt.origin.xyz) - (np.array([0.0, *c]) if jt.parent == "palm" else 0.0)
            assert np.allclose(xyz, want, atol=1e-9), jt.name


def test_the_old_rule_sets_are_unchanged():
    """The 29 Oct runs' rules: the mount at the centre, nothing derived."""
    assert not gh.FEETECH_COARSE_RULES.derived_mount and gh.FEETECH_COARSE_RULES.mount_at_centre
    assert check(G58, gh.FEETECH_COARSE_RULES) == []
