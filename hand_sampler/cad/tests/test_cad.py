"""The printable parts (hand_sampler/cad): every example hand renders, every
part is one closed piece, every joint turns over its range, every servo is
held, and the parts list adds up. Rendering needs OpenSCAD (skipped
without it).

    .venv_viewer/bin/python -m pytest -q hand_sampler/cad/tests
"""

from __future__ import annotations

import numpy as np
import pytest

from hand_sampler.grammar import derive as dv
from hand_sampler.grammar import motor as mt
from hand_sampler.grammar import operators as gops
from hand_sampler.grammar.hand import (EVOLUTION_RULES, FEETECH_RULES, Finger, Hand, Joint, PalmJoint, check)
from hand_sampler.grammar import viability as V

from hand_sampler.cad import package, render
from hand_sampler.cad.design import Design
from hand_sampler.cad.servo import HL3915

FLEX, ABD, ROLL = (0, 0), (90, 0), (0, 90)
BENDS = (("hinge", FLEX, 50), ("hinge", FLEX, 50), ("hinge", FLEX, 30))
THUMB_221 = (("hinge", ABD, 0), ("hinge", FLEX, 50), ("hinge", ABD, 0), ("hinge", FLEX, 50), ("hinge", FLEX, 30))


def _finger(y, z, joints, facing=0, tilt=0):
    return Finger(y=y, z=z, facing=facing, tilt=tilt, joints=tuple(Joint(t, a, L) for t, a, L in joints))


def cut_hand() -> Hand:
    """`make_grammar_population.cut_hand`: a row of three, two fingers beyond
    a palm cut, a 2-2-1 thumb with two crossings."""
    fingers = [_finger(y, 80, BENDS) for y in (-40, 0, 40)]
    fingers += [_finger(-80, 0, BENDS, facing=270), _finger(-80, -40, BENDS, facing=270)]
    fingers.append(_finger(80, -40, THUMB_221, facing=90))
    return Hand(fingers=tuple(fingers), palm_joints=(PalmJoint(facing=330, offset=30),))


def zero_link_hand() -> Hand:
    """`make_grammar_population.zero_link_hand`: knuckles crossing at the
    base (the spread servo slid out behind the palm) and a 2-2-1 thumb."""
    knuckle = (("hinge", ABD, 0), ("hinge", FLEX, 50), ("hinge", FLEX, 50), ("hinge", FLEX, 30))
    fingers = [_finger(y, 80, knuckle) for y in (-40, 0, 40)]
    fingers.append(_finger(40, -80, THUMB_221, facing=180))
    return Hand(fingers=tuple(fingers))


def oblique_hand() -> Hand:
    """`make_grammar_population.oblique_hand`: axes between the kinds."""
    fingers = [gops.fit_lengths(_finger(y, z, (("hinge", (150, 30), 50), ("hinge", (0, 30), 50), ("hinge", (0, -30), 50)),
                                        facing=fa), EVOLUTION_RULES)
               for y, z, fa in ((0, 80, 0), (-80, 0, 270), (80, 0, 90))]
    fingers.append(gops.fit_lengths(_finger(0, -80, (("hinge", (120, 30), 50), ("hinge", (30, 0), 50),
                                                     ("hinge", (0, 30), 50)), facing=180, tilt=30), EVOLUTION_RULES))
    return Hand(fingers=tuple(fingers))


def kinds_hand() -> Hand:
    """Every yoke: a U on bends and spreads, a turned bend after a spread's
    40 mm link, a one-arm yoke on a twist, a bend fingertip."""
    f = _finger(0, 40, (("hinge", FLEX, 50), ("hinge", ABD, 40), ("hinge", FLEX, 40), ("hinge", ROLL, 30)))
    g = _finger(0, -40, (("hinge", FLEX, 50), ("hinge", FLEX, 30)), facing=180)
    return Hand(fingers=(f, g))


def flipped_hand() -> Hand:
    """Bend servos flipped along the link after them (Martin, 10/29): a link
    with two yokes, one with a cradle and a yoke, one with two cradles, a
    fingertip holding its servo's case."""
    f = _finger(0, 40, (("hinge", FLEX, 50), ("hinge", FLEX, 40), ("hinge", FLEX, 50), ("hinge", FLEX, 40)))
    g = _finger(0, -40, (("hinge", FLEX, 50), ("hinge", FLEX, 40), ("hinge", FLEX, 60), ("hinge", FLEX, 30)),
                facing=180)
    return Hand(fingers=(f, g))


HANDS = {"kinds": kinds_hand, "cut": cut_hand, "zero_links": zero_link_hand, "oblique": oblique_hand,
         "flipped": flipped_hand}


def _openscad():
    try:
        render.openscad()
    except render.OpenSCADMissing:
        pytest.skip("OpenSCAD not installed")


@pytest.mark.parametrize("name", list(HANDS))
def test_the_example_hands_are_grammar_hands(name):
    h = HANDS[name]()
    assert check(h, FEETECH_RULES) == [] and V.c1_self_overlap(h).ok


def test_servo_frames_put_the_case_inside_the_grammar_s_motor_box():
    """The measured case (23 x 34 x 20, its horns out to 14.5) lies in the
    grammar's motor box (29 x 34 x 20) wherever a design puts a servo."""
    for h in (kinds_hand(), zero_link_hand(), oblique_hand(), flipped_hand()):
        d = Design(h)
        for sp in d.servos.values():
            if sp.key[0] != "f":
                continue
            _, i, j = sp.key
            box = dv.finger_geometry(h.fingers[i]).motors[j]
            C = mt.box_corners(np.array([0.0, 7.5, 0.0]), np.eye(3), np.array([14.5, 17.0, 10.0]))
            S = sp.S_parent if sp.flipped else sp.S_child          # in joint j's frame
            P = C @ S[:3, :3].T + S[:3, 3]
            local = (P - box.centre) @ box.axes
            assert np.all(np.abs(local) <= box.half + 1e-6)


def test_each_servo_has_one_yoke_and_one_holder():
    d = Design(cut_hand())
    assert len(d.servos) == sum(len(f.joints) for f in d.hand.fingers) + 1      # and the palm's hinge
    yokes = {sp.yoke for sp in d.servos.values()}
    assert yokes == {"u", "crossing"}
    for sp in d.servos.values():
        assert sp.parent in d.parts and sp.child in d.parts and sp.parent != sp.child
    d = Design(kinds_hand())
    assert [d.servos[("f", 0, j)].yoke for j in range(4)] == ["u", "u", "u", "single"]
    assert d.servos[("f", 0, 2)].turned


def test_a_flipped_servo_is_held_by_the_link_after_it():
    """A flipped servo's case is in link j and its horns in a yoke on link
    j - 1, which turns about it the other way: the links between carry two
    yokes, a cradle and a yoke, or two cradles."""
    d = Design(flipped_hand())
    assert [d.servos[("f", 0, j)].flipped for j in range(4)] == [False, False, True, True]
    assert [d.servos[("f", 1, j)].flipped for j in range(4)] == [False, False, True, False]
    for i, j in ((0, 2), (0, 3), (1, 2)):
        sp = d.servos[("f", i, j)]
        assert sp.parent == ("link", i, j) and sp.child == ("link", i, j - 1)
        lo, hi = (np.degrees(x) for x in dv.joint_limits(d.hand, i, j))
        assert sorted(abs(v) for v in sp.rot_deg) == sorted((abs(lo), abs(hi)))
    assert [sp.key for sp in d.parts[("link", 0, 1)].carries] == [("f", 0, 1), ("f", 0, 2)]
    assert d.parts[("link", 0, 1)].holds == []
    assert [sp.key for sp in d.parts[("link", 0, 2)].holds] == [("f", 0, 2)]
    assert [sp.key for sp in d.parts[("link", 1, 2)].holds] == [("f", 1, 2), ("f", 1, 3)]
    assert [sp.key for sp in d.parts[("link", 0, 3)].holds] == [("f", 0, 3)]


BOXED_IN = pytest.mark.xfail(strict=True, reason=(
    "a row of three base servos 40 mm apart, their axes along the row: the middle one's case screws face its "
    "neighbours' housings 13 mm away, so with every servo screwed straight into the palm (Martin, 10/29) no "
    "screwdriver reaches them; open (LOG, 10/29)"))


@pytest.mark.parametrize("name", [pytest.param(n, marks=BOXED_IN) if n == "cut" else n for n in HANDS])
def test_every_part_renders_and_every_check_passes(name, tmp_path):
    _openscad()
    kit = package.generate(HANDS[name](), tmp_path)
    assert kit.report.ok, kit.report.summary()
    assert kit.zip_path.exists()
    for n in kit.parts:
        assert (kit.folder / "print" / f"{n}.stl").exists()


def test_the_parts_list_adds_up(tmp_path):
    _openscad()
    h = kinds_hand()
    kit = package.generate(h, tmp_path)
    d = kit.design
    n = len(d.servos)
    bom = (kit.folder / "BOM.md").read_text()
    assert f"| {n} |" in bom
    used = package._count_screws(d)["box"]
    assert used <= HL3915.box_screws * n                      # the servos' own screws are enough
    assert package._count_screws(d)["M2x5"] == 4 * n
    steps = (kit.folder / "ASSEMBLY.md").read_text()
    assert "| 11 |" in steps and "CH1" in steps and "CH2" in steps


def test_every_placement_is_a_proper_rotation():
    """No transform in a hand file mirrors (a mirrored piece turns inside out
    in a hull, and a cut by it adds material)."""
    import re

    from hand_sampler.cad import scad

    for make in HANDS.values():
        txt = scad.hand_scad(Design(make()))
        mats = re.findall(r"multmatrix\(\[\[(.*?)\]\]\)", txt)
        for call in re.findall(r"tube\((.*?)\);", txt):              # both ends' frames
            mats += re.findall(r"\[\[(.*?)\]\]", call)
        for m in mats:
            R = np.array([list(map(float, r.split(","))) for r in m.split("],[")])[:3, :3]
            assert abs(np.linalg.det(R) - 1.0) < 1e-3
