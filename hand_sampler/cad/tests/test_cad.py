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
BENDS = (("hinge", FLEX, 50), ("hinge", FLEX, 50), ("hinge", FLEX, 40))      # fingertips at least 40 mm (2026-11-02)
THUMB_221 = (("hinge", ABD, 0), ("hinge", FLEX, 50), ("hinge", ABD, 0), ("hinge", FLEX, 50), ("hinge", FLEX, 40))


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
    knuckle = (("hinge", ABD, 0), ("hinge", FLEX, 50), ("hinge", FLEX, 50), ("hinge", FLEX, 40))
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
    """Every yoke: a U on bends and spreads, a side mount after a spread's
    40 mm link (a crossing's plate on the bend servo's side, 2026-11-04), a
    one-arm yoke on a twist, a bend fingertip; a turned bend after a 30 mm
    link."""
    f = _finger(0, 40, (("hinge", FLEX, 50), ("hinge", ABD, 40), ("hinge", FLEX, 40), ("hinge", ROLL, 40)))
    g = _finger(0, -40, (("hinge", FLEX, 30), ("hinge", FLEX, 45)), facing=180)
    return Hand(fingers=(f, g))


def flipped_hand() -> Hand:
    """Bend servos flipped along the link after them (Martin, 10/29): a link
    with two yokes, one with a cradle and a yoke, one with two cradles, a
    fingertip holding its servo's case (50 mm: room for its case, flare and dome)."""
    f = _finger(0, 40, (("hinge", FLEX, 50), ("hinge", FLEX, 40), ("hinge", FLEX, 50), ("hinge", FLEX, 50)))
    g = _finger(0, -40, (("hinge", FLEX, 50), ("hinge", FLEX, 40), ("hinge", FLEX, 60), ("hinge", FLEX, 40)),
                facing=180)
    return Hand(fingers=(f, g))


def startup_hand() -> Hand:
    """The viewer's first hand (its kit: `outputs/designs/d16535003d`)."""
    return Hand(fingers=(
        _finger(-80, -40, (("hinge", FLEX, 50),), facing=180),
        _finger(-40, 80, (("hinge", FLEX, 70), ("hinge", FLEX, 70)), facing=60),
        _finger(0, -40, (("hinge", FLEX, 60), ("hinge", FLEX, 40), ("hinge", ROLL, 30), ("hinge", FLEX, 40),
                         ("hinge", FLEX, 50)), facing=180),
        _finger(40, 0, (("hinge", ABD, 60), ("hinge", ABD, 50)), facing=90, tilt=90),
        _finger(80, 40, (("hinge", ABD, 0), ("hinge", FLEX, 70), ("hinge", FLEX, 90), ("hinge", ABD, 50)),
                facing=150, tilt=90)))


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
    """The measured case (23 x 34 x 20) lies in the grammar's motor box (29 x
    34 x 20, the datasheet's horns, 29 mm apart as Martin's servos measure,
    11/04) wherever a design puts a servo."""
    assert HL3915.driven_horn_x[1] == pytest.approx(mt.MOTOR_HALF_MM[0])
    for h in (kinds_hand(), zero_link_hand(), oblique_hand(), flipped_hand()):
        d = Design(h)
        for sp in d.servos.values():
            if sp.key[0] != "f":
                continue
            _, i, j = sp.key
            box = dv.finger_geometry(h.fingers[i]).motors[j]
            C = mt.box_corners(np.array([0.0, 7.5, 0.0]), np.eye(3), np.array([HL3915.driven_horn_x[1], 17.0, 10.0]))
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
    assert [d.servos[("f", 0, j)].yoke for j in range(4)] == ["u", "crossing", "u", "single"]
    assert d.servos[("f", 1, 1)].turned


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
    import trimesh

    for n in kit.parts:
        printed = trimesh.load(kit.folder / "print" / f"{n}.stl", force="mesh")     # its vertices joined, as a slicer
        assert printed.is_watertight and len(printed.split(only_watertight=False)) == 1
        P = np.asarray(printed.vertices)
        across = max(np.ptp(P[:, :2] @ [np.cos(t), np.sin(t)]) for t in np.linspace(0.0, np.pi, 12, endpoint=False))
        assert abs(printed.bounds[0, 2]) < 1e-6 and printed.extents[2] <= across + 1e-6     # never on its long side
        solids, faces, _, closed = _step_shell(kit.folder / "step" / f"{n}.step")
        assert solids == 1 and faces > 0 and closed


def test_each_part_prints_the_way_up_it_is_strongest():
    """Print orientation (Martin, 11/06: for strength, not supports): a link
    stands on the axis most of the torques it carries turn about (its own
    joint's and every joint's beyond it), never on its long side; the palm
    lies on its face."""
    h = kinds_hand()
    d = Design(h)
    ups = {}
    for p in d.part_list():
        if p.body[0] == "palm":
            ups[p.name] = package.print_up(d, p, np.zeros((1, 3)))
            continue
        _, i, j = p.body
        verts, _ = dv.shape_points(dv.link_shape_mm(h.fingers[i], j))     # the simulator's body stands in for the mesh
        ups[p.name] = package.print_up(d, p, np.asarray(verts, float))
    z, y = [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]
    assert np.allclose(ups["palm"], [1.0, 0.0, 0.0])
    # bend, spread (a side mount), bend, twist: the bends carry most, the twist's axis runs along the fingertip
    assert all(np.allclose(np.abs(ups[f"f0_l{j}"]), z) for j in range(3)) and np.allclose(np.abs(ups["f0_tip"]), z)
    # a turned bend, then a bend fingertip
    assert np.allclose(np.abs(ups["f1_l0"]), z) and np.allclose(np.abs(ups["f1_tip"]), z)
    # a spread-only finger stands on the spread axis
    g = _finger(0, 40, (("hinge", ABD, 60), ("hinge", ABD, 50)), facing=90, tilt=90)
    dg = Design(Hand(fingers=(g, _finger(0, -40, BENDS, facing=180))))
    p = dg.parts[("link", 0, 0)]
    assert np.allclose(np.abs(package.print_up(dg, p, np.asarray(dv.link_box_mm(g, 0).corners(), float))), y)


def _step_shell(path):
    """A STEP file's solids, faces and loops, and whether its shell is closed
    (every edge in two faces' loops, once each way)."""
    import re

    txt = path.read_text()
    uses = {}
    for e, o in re.findall(r"ORIENTED_EDGE\('',\*,\*,#(\d+),\.([TF])\.\)", txt):
        uses.setdefault(e, []).append(o)
    closed = bool(uses) and all(sorted(v) == ["F", "T"] for v in uses.values())
    return txt.count("MANIFOLD_SOLID_BREP("), txt.count("ADVANCED_FACE("), txt.count("EDGE_LOOP("), closed


def test_a_step_file_is_one_closed_solid_its_flat_faces_merged(tmp_path):
    """A 10 mm cube with a 16-sided hole through it: its six faces (two with
    the hole as a second loop) and the hole's 16, one closed shell."""
    import manifold3d as m3

    from hand_sampler.cad import step

    man = m3.Manifold.cube([10.0, 10.0, 10.0]) - m3.Manifold.cylinder(12.0, 2.0, 2.0, 16).translate([5.0, 5.0, -1.0])
    mesh = man.to_mesh()
    info = step.write(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), tmp_path / "c.step", "c")
    assert _step_shell(tmp_path / "c.step") == (1, 22, 24, True)
    assert info["faces"] == 22 and info["plane_error_mm"] < step.UNCERTAINTY_MM


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
    arms = sum(2 if sp.yoke == "u" else 1 for sp in d.servos.values())
    assert package._count_screws(d)["M2x5"] == 4 * arms                 # four on every arm (and crossing plate)
    steps = (kit.folder / "ASSEMBLY.md").read_text()
    assert "| 11 |" in steps and "CH1" in steps and "CH2" in steps
    assert "M5 heat-set insert" in bom and "floats on its cables" in bom and "column" in steps


def test_the_ribs_are_one_grid(monkeypatch):
    """The ribs under a palm's face (Martin, 11/03: "add the ribs in a grid
    inside"): every piece lies on a line of one square grid, the lines
    `RIB_PITCH_MM` apart along the palm's y and along its z."""
    from hand_sampler.cad import design as D

    seen = []
    ribs = D.Design._ribs

    def spy(outline, x0, walls, stops, lanes):
        out = ribs(outline, x0, walls, stops, lanes)
        seen.append(out[1])
        return out

    monkeypatch.setattr(D.Design, "_ribs", staticmethod(spy))
    for make in HANDS.values():
        Design(make())
    assert sum(len(feet) for feet in seen) > 0
    for feet in seen:
        lines = {0: set(), 1: set()}
        for f in feet:
            lo, hi = f.min(axis=0), f.max(axis=0)
            k = 0 if hi[0] - lo[0] < hi[1] - lo[1] else 1        # a line at y = c runs along z
            lines[k].add(float(lo[k] + hi[k]) / 2.0)
        for cs in lines.values():
            steps = np.diff(sorted(cs)) / D.RIB_PITCH_MM
            assert np.allclose(steps, np.round(steps)) and np.all(np.round(steps) >= 1)


def test_the_fills_under_the_face_leave_no_slit_and_no_sharp_inside_corner():
    """Under a palm's face (Martin, 11/04: "some internal corners are
    filleted while others are not", "thin gaps around some of the edges"):
    the fills round every inside corner where walls and ribs meet, close a
    slit between two walls even across a case screw's lane (keeping the rest
    of the lane clear), and reach into the walls they join (fills that only
    touched them left cracks)."""
    import re

    import manifold3d as m3

    from hand_sampler.cad import design as D

    def box(y0, z0, y1, z1):
        return np.array([[y0, z0], [y1, z0], [y1, z1], [y0, z1]], float)

    def cs(P):
        return m3.CrossSection([D._ccw(P).tolist()])

    outline = box(0.0, 0.0, 100.0, 60.0)
    walls = [box(30.0, 0.0, 32.5, 60.0), box(0.0, 30.0, 100.0, 32.5),    # a rib crossing, its ends on the rim
             box(60.0, 2.9, 90.0, 5.4)]                                   # a wall 0.4 mm inside the rim
    lane = box(70.0, 0.0, 80.0, 20.0)                                     # a lane across that slit
    tray = cs(outline)
    W = tray - tray.offset(-D.RIM_MM, m3.JoinType.Miter)
    for q in walls:
        W = W + cs(q)
    fills = []
    for p in D.Design._fills(outline, 10.5, walls, [], [lane]):
        pts, paths = re.match(r"prism_yz\((\[\[.*?\]\]), (\[\[.*?\]\]),", p("0")).groups()
        pts, paths = eval(pts), eval(paths)
        fills.append(m3.CrossSection([[pts[i] for i in path] for path in paths]))
    assert fills and all((f ^ W).area() > 0.1 for f in fills)
    U = W
    for f in fills:
        U = U + f
    r = 0.75                     # a sharp 90-degree inside corner leaves a crumb of 0.12 mm2 in this closing, a slit more
    gaps = U.offset(r, m3.JoinType.Round, 2.0, 64).offset(-r, m3.JoinType.Round, 2.0, 64) - U
    assert max([g.area() for g in gaps.decompose()] + [0.0]) < 0.05
    assert (cs(box(70.0, 2.5, 80.0, 2.9)) - U).area() < 0.01             # the slit, solid in the lane too
    assert (cs(box(70.0, 5.4 + D.REACH_MM + 0.1, 80.0, 20.0)) ^ U).area() < 0.01   # the lane past it, clear


def test_the_back_mount_and_the_board():
    """The back mount (Martin, 11/04): the grammar's mount column, 44 mm
    across and 40 mm long, behind the main palm, its four M5 inserts in its
    end 3 mm inside it. On the startup hand two case screws of finger 4's
    base servo are driven through it (it leaves their ways clear), and the
    board lies on the palm's back, its standoffs inside the outline and
    clear of the column. A knuckle crossing at its base goes in from behind
    with its finger on (its spread servo slid out behind the palm), so the
    column keeps off its way in: it moves aside, wholly on the palm, where
    there is room (a note); with none its inserts go into the back face (a
    note). The board floats where its standoffs find no room (a note)."""
    from hand_sampler.cad.design import MOUNT_INSERT_MM, MOUNT_R_MM, STANDOFF_MM, _ccw

    d = Design(startup_hand())
    assert d.mount == {"column": True, "at": [0.0, 0.0], "turn_deg": 0, "wall_mm": 3.0, "ways": 2} and not d.notes
    palm = d.parts[("palm", -1)]
    for p in d.board["holes"]:
        assert Design._dist_out(np.array(p), _ccw(palm.face)) <= -STANDOFF_MM[0] / 2.0
        assert np.hypot(*p) > MOUNT_R_MM + 2.0
    assert sum(f", {MOUNT_INSERT_MM[0] / 2.0}, " in n for n in palm.negative) == 4
    knuckle = (("hinge", ABD, 0), ("hinge", FLEX, 50), ("hinge", FLEX, 40))
    d = Design(Hand(fingers=(_finger(0, 40, knuckle), _finger(-80, -40, BENDS, facing=180),
                             _finger(80, -40, BENDS, facing=180))))
    assert d.mount["column"] and np.hypot(*d.mount["at"]) > MOUNT_R_MM and d.mount["wall_mm"] == 3.0
    assert any("off the palm's centre" in n for n in d.notes)
    d = Design(Hand(fingers=(_finger(0, 40, knuckle), _finger(0, -40, BENDS, facing=180))))
    assert not d.mount["column"] and any("back face" in n for n in d.notes)
    assert d.board is None and any("floats" in n for n in d.notes)


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
