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
    evo.gui_base.value = 160
    evo.gui_base_min.value = 10
    evo.gui_length.value = 250
    r = evo.rules()
    assert r.base_distance_mm == (10, 160) and r.max_finger_length_mm <= 250 and r.max_palm_joints <= 1


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
    assert "fit:" in evo.md_fit.content
    assert evo.overlay.handles                      # its URDF meshes are drawn
    evo.set_curl(0.0)                                # curl 0 is the real hand's zero pose
    assert np.allclose(evo.q(), evo.commercial["q_off"])
    evo.mutate(wait=True)                            # a mutant is not the commercial hand
    assert evo.commercial is None and not evo.overlay.handles


def test_pose_slider_moves_the_links(evo):
    evo.random(wait=True)
    evo.set_curl(0.0)
    evo._render_pose()
    key = next(k for k in evo.drawing.bodies if k[0] == "link" and k[2] == len(evo.hand.fingers[k[1]].joints) - 1)
    p0 = np.array(evo.drawing.bodies[key].position)
    evo.set_curl(1.0)
    evo._render_pose()
    moved = np.linalg.norm(np.array(evo.drawing.bodies[key].position) - p0)
    assert moved > 1e-4 or len(evo.hand.fingers[key[1]].joints) == 1
    evo.set_curl(gdv.START_CURL)


def test_one_body_per_solid(evo):
    """The drawing has one body per solid the clearance check sees: the
    plates, the mount, the hinge motor (with a cut), each base motor and
    each link."""
    evo.random(wait=True)
    h = evo.hand
    keys = set(evo.drawing.bodies)
    want = {s.key for s in gdv.solids_at(h)}
    assert keys == want


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
    evo._render_pose()
    p0 = np.array(evo.drawing.bodies[("link", 0, 0)].position)
    q = evo.q()
    k = [d.type for d in gdv.dofs(h)].index("palm")
    evo.gui_palm_bend.value = 1.0
    evo._render_pose()
    assert evo.q()[k] == pytest.approx(np.radians(30))
    assert np.linalg.norm(np.array(evo.drawing.bodies[("link", 0, 0)].position) - p0) > 1e-3   # its finger moves
    evo.gui_palm_bend.value = 0.0
    evo.random(wait=True)
    assert evo.gui_palm_bend.visible == bool(evo.hand.palm_joints)


def test_search_space_line_follows_the_rules(evo):
    assert "10^31 four-finger hands" in evo.md_space.content
    evo.choose_rules(V.NONE)
    assert "10^54 four-finger hands" in evo.md_space.content
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
