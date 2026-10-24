"""The grammar viewer (viewer.py) driven headless against a real local viser
server: Random, the Rules dropdown and fields, the two viability checks,
operator buttons shown only when they can act, mutation and Back, commercial
hands over their meshes, pose."""

import socket

import numpy as np
import pytest

viser = pytest.importorskip("viser")

import viewer as V  # noqa: E402
from gviewer import draw  # noqa: E402
from hand_sampler.grammar import derive as gdv  # noqa: E402
from hand_sampler.grammar import operators as gops  # noqa: E402
from hand_sampler.grammar.hand import FEETECH_RULES, NO_RULES, check  # noqa: E402


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def app():
    server = viser.ViserServer(host="127.0.0.1", port=_free_port(), verbose=False)
    a = V.GrammarViewer(server, seed=0)
    yield a
    server.stop()


@pytest.fixture
def evo(app):
    app.choose_rules(V.FEETECH)
    app.gui_c1.value = True
    app.gui_c2.value = True
    return app


def test_random_finds_a_viable_hand_under_the_rules(evo):
    assert evo.random(wait=True)
    assert evo.hand is not None and check(evo.hand, FEETECH_RULES) == []
    assert evo.viability.ok
    assert evo.md_random.content.startswith("found after")
    assert evo.gui_c1.label.startswith("C1") and "PASS" in evo.gui_c1.label
    assert evo.gui_c2.label.startswith("C2") and "PASS" in evo.gui_c2.label


def test_rules_dropdown_and_custom_fields(evo):
    assert evo.rules() == FEETECH_RULES
    evo.choose_rules(V.NONE)
    assert evo.rules() == NO_RULES and evo.gui_sliding.value and not evo.gui_buildable.value
    evo.choose_rules(V.FEETECH)
    evo.gui_fingers.value = 3                      # editing a field: Custom Rules
    assert evo.gui_rules.value == V.CUSTOM and evo.rules().max_fingers == 3
    evo.random(wait=True)
    assert len(evo.hand.fingers) <= 3
    evo.choose_rules(V.FEETECH)
    assert evo.gui_fingers.value == 6 and evo.gui_rules.value == V.FEETECH
    assert evo.gui_pitch.value == 40 and evo.gui_outward.value and evo.gui_buildable.value


def test_fields_cannot_widen_the_grammar(evo):
    evo.gui_base.value = 10_000
    evo.gui_base_min.value = 0
    evo.gui_length.value = 10_000
    r = evo.rules()
    assert r.base_distance_mm == NO_RULES.base_distance_mm and r.max_finger_length_mm == NO_RULES.max_finger_length_mm
    assert r.max_palm_joints <= 1


def test_operator_buttons_follow_what_can_act(evo):
    evo.random(wait=True)
    for name, b in evo.op_buttons.items():
        assert b.visible == gops.can_act(evo.hand, gops.OPERATOR_BY_NAME[name], evo.rules())
    assert not evo.op_buttons["couple"].visible           # no coupled joints in the Feetech rules


def test_mutation_and_back(evo):
    evo.random(wait=True)
    before = evo.hand
    evo.mutate(wait=True)
    assert evo.hand != before and check(evo.hand, FEETECH_RULES) == []
    assert evo.md_mut.content
    evo.back(wait=True)
    assert evo.hand == before


def test_each_visible_operator_button_applies(evo):
    evo.random(wait=True)
    for name, b in list(evo.op_buttons.items()):
        if not b.visible:
            continue
        before = evo.hand
        evo.mutate(name, wait=True)
        assert evo.hand != before, name
        assert check(evo.hand, FEETECH_RULES) == [], name
        evo.back(wait=True)


def test_unticking_a_check_stops_random_requiring_it(evo):
    evo.gui_c1.value = False
    evo.gui_c2.value = False
    evo.random(wait=True)
    assert evo.md_random.content == "found after 1 try"


def test_commercial_hand_over_its_meshes(evo):
    hid = "dex3_left" if "dex3_left" in evo.records else sorted(evo.records)[0]
    assert evo.load_commercial(hid, wait=True)
    assert evo.commercial is not None and evo.hand == evo.records[hid]["hand"]
    assert "fit (at its real size):" in evo.md_fit.content and "x its size" in evo.md_fit.content
    assert evo.overlay.handles                      # its URDF meshes are drawn
    evo.set_curl(0.0)                                # curl 0 is the real hand's zero pose
    assert np.allclose(evo.q(), evo.commercial["q_off"])
    evo.mutate(wait=True)                            # a mutant is not the commercial hand
    assert evo.commercial is None and not evo.overlay.handles


def test_pose_slider_moves_the_links(evo):
    evo.random(wait=True)
    evo.set_curl(0.0)
    key = next(k for k in evo.drawing.bodies if k[0] == "link" and k[2] == len(evo.hand.fingers[k[1]].joints) - 1)
    pose = lambda: np.concatenate([evo.drawing.bodies[key].position, evo.drawing.bodies[key].wxyz])  # noqa: E731
    p0 = pose()
    evo.set_curl(1.0)
    moved = np.linalg.norm(pose() - p0)     # a bend at the link's own joint turns it in place
    assert moved > 1e-4 or len(evo.hand.fingers[key[1]].joints) == 1
    evo.set_curl(gdv.START_CURL)


def test_every_servo_is_drawn_with_its_horn_and_axis(evo):
    """Every solid the clearance check sees has a drawn body (each finger's
    base servo is its motor 0), and every joint has its servo, its output
    horn and a rod along its axis."""
    evo.random(wait=True)
    h = evo.hand
    keys = set(evo.drawing.bodies)
    for s in gdv.solids_at(h):
        k = ("motor", s.key[1], 0) if s.key[0] == "motor" else s.key
        assert k in keys, s.key
    for i, f in enumerate(h.fingers):
        for j in range(len(f.joints)):
            assert {("motor", i, j), ("horn", i, j), ("axis", i, j), ("link", i, j)} <= keys
    # the plates are see-through, so the servos in the palm show; the servos are solid
    for k in keys:
        if k[0] == "palm":
            assert evo.drawing.bodies[k].opacity == draw.PALM_OPACITY < 0.5, k
        if k[0] == "motor":
            assert evo.drawing.bodies[k].opacity is None, k


def test_a_horn_turns_with_its_joint_and_its_servo_does_not(evo):
    evo.random(wait=True)
    ds = gdv.dofs(evo.hand)
    k = next(k for k, d in enumerate(ds) if d.finger >= 0 and d.index == 0)
    i = ds[k].finger
    evo.set_curl(0.0)
    horn0 = np.array(evo.drawing.bodies[("horn", i, 0)].wxyz)
    motor0 = np.array(evo.drawing.bodies[("motor", i, 0)].wxyz)
    evo.set_joint(k, ds[k].limits[1] * 0.5)
    assert np.abs(np.array(evo.drawing.bodies[("horn", i, 0)].wxyz) - horn0).max() > 1e-3
    assert np.allclose(np.array(evo.drawing.bodies[("motor", i, 0)].wxyz), motor0)
    evo.set_curl(gdv.START_CURL)


def test_one_slider_per_joint(evo):
    """The Joints folder has a slider for every joint of the hand on screen,
    set by the presets; moving one moves only that joint (and the joints
    coupled to it), and a coupled joint's slider is disabled."""
    evo.random(wait=True)
    ds = gdv.dofs(evo.hand)
    assert [d.name for d, _ in evo.joint_sliders] == [d.name for d in ds]
    evo.set_curl(0.5)
    assert np.allclose(evo.q(), evo.preset_q(), atol=1e-9)
    k = next(k for k, d in enumerate(ds) if d.source < 0 and d.finger >= 0)
    before = evo.q()
    evo.set_joint(k, ds[k].limits[0])
    after = evo.q()
    assert after[k] == pytest.approx(ds[k].limits[0])
    others = [m for m, d in enumerate(ds) if m != k and d.source != k]
    assert np.allclose(after[others], before[others])
    evo.set_curl(gdv.START_CURL)


def test_link_bodies_follow_their_opacity_slider(evo):
    """The link bodies start see-through (the fingertips show) and follow
    the slider; 0 hides them."""
    evo.random(wait=True)
    links = [k for k in evo.drawing.bodies if k[0] == "link"]
    assert links and all(evo.drawing.bodies[k].visible for k in links)
    assert all(evo.drawing.bodies[k].opacity == draw.LINK_OPACITY for k in links if k not in evo.drawing.hot)
    evo.gui_links.value = 0.0
    assert all(not evo.drawing.bodies[k].visible for k in links if k not in evo.drawing.hot)
    evo.gui_links.value = draw.LINK_OPACITY
    assert all(evo.drawing.bodies[k].visible for k in links)


def test_box_mesh_has_the_extents():
    from hand_sampler.grammar.hand import Finger, Joint

    f = Finger(y=0, z=40, joints=(Joint("hinge", (0, 0), 50), Joint("hinge", (0, 0), 30)))
    V_, F = draw.box_mesh(gdv.link_box_mm(f, 0).corners() * 1e-3)
    ext = V_.max(0) - V_.min(0)
    assert ext[0] == pytest.approx(0.0715, abs=1e-6)          # 12 mm behind the joint to the next motor's front
    assert ext[1] == pytest.approx(0.024, abs=1e-6)           # the motor's width plus two walls
    assert ext[2] == pytest.approx(0.033, abs=1e-6)           # its length along the axis plus two walls


def test_palm_bend_folds_the_part_beyond_the_cut(evo):
    from hand_sampler.grammar.hand import Finger, Hand, Joint, PalmJoint

    h = Hand(fingers=(Finger(y=0, z=80, joints=(Joint("hinge", (0, 0), 50), Joint("hinge", (0, 0), 30))),
                      Finger(y=80, z=0, facing=90, joints=(Joint("hinge", (0, 0), 50), Joint("hinge", (0, 0), 30))),
                      Finger(y=-80, z=0, facing=270, joints=(Joint("hinge", (0, 0), 50), Joint("hinge", (0, 0), 30)))),
             palm_joints=(PalmJoint(facing=90, offset=40),))
    assert gdv.finger_section(h, 0) == 0
    evo._set_hand(h, history="reset", commercial=None, viability=V.gvb.viability(h), frame=False)
    assert evo.gui_palm_bend.visible
    evo.gui_palm_bend.value = 0.0
    evo._apply_presets()
    p0 = np.array(evo.drawing.bodies[("link", 0, 0)].position)
    q = evo.q()
    k = [d.type for d in gdv.dofs(h)].index("palm")
    evo.gui_palm_bend.value = 1.0
    evo._apply_presets()
    assert evo.q()[k] == pytest.approx(np.radians(30))
    assert np.linalg.norm(np.array(evo.drawing.bodies[("link", 0, 0)].position) - p0) > 1e-3   # its finger moves
    evo.gui_palm_bend.value = 0.0
    evo.random(wait=True)
    assert evo.gui_palm_bend.visible == bool(evo.hand.palm_joints)


def test_search_space_line_follows_the_rules(evo):
    """The line counts every finger count: the fine stage 10^65, the coarse
    stage 10^38, the whole grammar more."""
    assert "10^65 hands (2-6 fingers)" in evo.md_space.content
    evo.choose_rules(V.COARSE)
    assert "10^38 hands (2-6 fingers)" in evo.md_space.content
    evo.choose_rules(V.NONE)
    assert "hands (2-6 fingers)" in evo.md_space.content and "10^65 hands" not in evo.md_space.content
    evo.choose_rules(V.FEETECH)


def test_the_coarse_stage_draws_kinds_and_20_mm_steps(evo):
    """Under the coarse stage random hands have joints on the three kinds
    only and links on their floor plus 20 mm steps; they keep the fine rules
    too."""
    from hand_sampler.grammar.hand import FEETECH_COARSE_RULES

    evo.choose_rules(V.COARSE)
    assert evo.rules() == FEETECH_COARSE_RULES
    evo.random(wait=True)
    assert check(evo.hand, FEETECH_COARSE_RULES) == [] and check(evo.hand, FEETECH_RULES) == []
    assert {tuple(j.axis) for f in evo.hand.fingers for j in f.joints} <= {(0, 0), (90, 0), (0, 90)}
    evo.choose_rules(V.FEETECH)


def test_object_at_its_start(evo):
    evo.random(wait=True)
    assert evo.drawing.object is None
    evo.gui_object.value = True
    with evo.lock:
        evo._set_hand(evo.hand, history="keep", commercial=None, viability=evo.viability, frame=False)
    assert evo.drawing.object is not None
    want = gdv.object_start_m(evo.hand, V.OBJECT_HALF_M)
    V_ = np.asarray(evo.drawing.object.vertices)
    assert np.allclose((V_.max(0) + V_.min(0)) / 2, want, atol=1e-6)
    evo.gui_object.value = False


def test_generate_full_design_draws_the_parts_and_follows_the_joints(evo, tmp_path, monkeypatch):
    """The Build button's job: the hand's kit (zip, STLs), its parts and its
    servos (case and horns) drawn in place of the grammar's bodies, posed
    with the joint sliders; a new hand clears them and shows the bodies."""
    from hand_sampler.cad import package, render

    try:
        render.openscad()
    except render.OpenSCADMissing:
        pytest.skip("OpenSCAD not installed")
    monkeypatch.setattr(package, "DESIGNS_DIR", tmp_path)
    monkeypatch.setattr(package.generate, "__defaults__", (tmp_path, None))
    evo.random(wait=True)
    assert evo.generate_design(wait=True)
    assert evo.kit is not None and evo.kit.zip_path.exists()
    assert "parts in" in evo.md_build.content
    tags = ["_".join(str(k) for k in sp.key) for sp in evo.kit.design.servos.values()]
    assert set(evo.printed.handles) == (set(evo.kit.parts) | {f"servo_{t}" for t in tags}
                                        | {f"horns_{t}" for t in tags})
    assert not evo.drawing.shown and not any(h.visible for h in evo.drawing.handles)
    evo.gui_printed.value = False
    assert evo.drawing.shown and not any(h.visible for hs in evo.printed.handles.values() for h in hs)
    evo.gui_printed.value = True
    tips = [n for n in evo.kit.parts if n.endswith("_tip")]
    before = [np.array(evo.printed.handles[n][0].wxyz) for n in tips]
    evo.gui_curl.value = 0.8
    after = [np.array(evo.printed.handles[n][0].wxyz) for n in tips]
    assert max(np.linalg.norm(a - b) for a, b in zip(after, before)) > 1e-3     # the fingertips turned
    evo.random(wait=True)
    assert evo.kit is None and not evo.printed.handles
    assert evo.drawing.shown
