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
from hand_sampler.grammar.hand import EVOLUTION_RULES, NO_RULES, check  # noqa: E402


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
    app.choose_rules(V.EVOLUTION)
    app.gui_c1.value = True
    app.gui_c2.value = True
    app.set_stage("coarse")
    return app


def test_random_finds_a_viable_hand_under_the_rules(evo):
    assert evo.random(wait=True)
    assert evo.hand is not None and check(evo.hand, EVOLUTION_RULES) == []
    assert evo.viability.ok
    assert evo.md_random.content.startswith("found after")
    assert evo.gui_c1.label.startswith("C1") and "PASS" in evo.gui_c1.label
    assert evo.gui_c2.label.startswith("C2") and "PASS" in evo.gui_c2.label


def test_rules_dropdown_and_custom_fields(evo):
    assert evo.rules() == EVOLUTION_RULES
    evo.choose_rules(V.NONE)
    assert evo.rules() == NO_RULES and evo.gui_sliding.value
    evo.choose_rules(V.EVOLUTION)
    evo.gui_fingers.value = 3                      # editing a field: Custom Rules
    assert evo.gui_rules.value == V.CUSTOM and evo.rules().max_fingers == 3
    evo.random(wait=True)
    assert len(evo.hand.fingers) <= 3
    evo.choose_rules(V.EVOLUTION)
    assert evo.gui_fingers.value == 6 and evo.gui_rules.value == V.EVOLUTION


def test_fields_cannot_widen_the_grammar(evo):
    evo.gui_spacing.value = 19
    evo.gui_length.value = 250
    r = evo.rules()
    assert r.min_spacing_mm >= 19 and r.max_finger_length_mm <= 250


def test_operator_buttons_follow_what_can_act(evo):
    evo.random(wait=True)
    for name, b in evo.op_buttons.items():
        assert b.visible == gops.can_act(evo.hand, gops.OPERATOR_BY_NAME[name], evo.rules(), "coarse")
    evo.set_stage("fine")
    for name, b in evo.op_buttons.items():
        if gops.OPERATOR_BY_NAME[name].structural:
            assert not b.visible


def test_mutation_and_back(evo):
    evo.random(wait=True)
    before = evo.hand
    evo.mutate(wait=True)
    assert evo.hand != before and check(evo.hand, EVOLUTION_RULES) == []
    assert evo.md_mut.content
    evo.back(wait=True)
    assert evo.hand == before


@pytest.mark.parametrize("stage", ["coarse", "fine"])
def test_each_visible_operator_button_applies(evo, stage):
    evo.random(wait=True)
    evo.set_stage(stage)
    for name, b in list(evo.op_buttons.items()):
        if not b.visible:
            continue
        before = evo.hand
        evo.mutate(name, wait=True)
        assert evo.hand != before, name
        assert check(evo.hand, EVOLUTION_RULES) == [], name
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
    h = next(iter(evo.drawing.links.values()))
    p0 = np.array(h.position)
    evo.set_curl(1.0)
    evo._render_pose()
    assert evo.drawing.links and evo.hand is not None
    evo.set_curl(gdv.START_CURL)


def test_rounded_box_mesh_has_the_cross_section():
    V_, F = draw.rounded_box(0.0, 30.0)
    ext = V_.max(0) - V_.min(0)
    assert ext[0] == pytest.approx(0.030 + 0.012, abs=2e-4)   # core 30 mm + two radii
    assert ext[1] == pytest.approx(0.018, abs=2e-4)          # height 18 mm
    assert ext[2] == pytest.approx(0.019, abs=2e-4)          # width 19 mm
