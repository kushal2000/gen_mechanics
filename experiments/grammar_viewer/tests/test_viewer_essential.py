"""The essential viewer (viewer.py): its callbacks driven against a real local
viser server: limits panel, viability toggles changing Random's tries,
operator buttons greyed out under the limits, mutation, pose, commercial."""

import functools
import socket
import urllib.request

import pytest

viser = pytest.importorskip("viser")

import viewer as V  # noqa: E402
from gviewer import checks as ck  # noqa: E402
from gviewer import limitsui as lui  # noqa: E402
from gviewer import model as gm  # noqa: E402
from gviewer import sources as src  # noqa: E402
from hand_sampler.grammar.derive import EVOLUTION_OPERATORS, EVOLUTION_OPERATORS_FINE, derive, sample_derivation  # noqa: E402,E501
from hand_sampler.grammar.kinematics import ModelError  # noqa: E402
from hand_sampler.grammar.limits import SIMULATOR, GenerationLimits, Structure, check  # noqa: E402
from hand_sampler.grammar.variants import RULES, build_distribution  # noqa: E402

ALL_ON = {r: True for r in RULES}
ALL_OFF = {r: False for r in RULES}
UNLIMITED = GenerationLimits()


def unlimit(app):
    """Set every limit field to 'no limit', as a user would in the panel."""
    for f in lui.INT_FIELDS:
        app.set_limit(f.key, lui.ANY)
    for key, _, _ in lui.BOOL_FIELDS:
        app.set_limit(key, True)
    app.set_limit("joint_types", "hinge + continuous + sliding")
    app.set_limit("coupled", True)
    assert app.limits() == UNLIMITED


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def app():
    port = _free_port()
    server = viser.ViserServer(host="127.0.0.1", port=port, verbose=False)
    a = V.EssentialViewer(server, start_seed=0)
    a._port = port
    yield a
    server.stop()


@pytest.fixture
def simulator(app):
    app.reset_limits()
    app.set_all_checks(True)
    app.set_rules(**ALL_ON)
    app.set_stage("coarse")
    yield app
    app.reset_limits()
    app.set_stage("coarse")
    app.set_all_checks(True)
    app.set_rules(**ALL_ON)


@functools.lru_cache(maxsize=None)
def failing_seed(key: str, only: bool = False):
    """(rules, seed) of a hand the grammar samples under SIMULATOR that fails
    `key` (with `only`: and passes the other checks)."""
    for rules in (ALL_ON, ALL_OFF):
        dist = build_distribution(**rules)
        for seed in range(600):
            try:
                m = derive(sample_derivation(seed, dist, limits=SIMULATOR))
            except ModelError:
                continue
            if only:
                if ck.evaluate(m).failing(ck.CHECK_KEYS) == [key]:
                    return tuple(sorted(rules.items())), seed
            elif ck.evaluate(m, {key}, stop_early=True).results[key].status == ck.FAIL:
                return tuple(sorted(rules.items())), seed
    raise AssertionError(f"no sampled design fails {key}")


def draw(app, rules, seed):
    app.set_rules(**dict(rules))
    return app.random(start_seed=seed, wait=True)


def test_http_and_initial_design(app):
    assert urllib.request.urlopen(f"http://127.0.0.1:{app._port}", timeout=10).status == 200
    assert app.shown.kind == "sampled"
    assert app.md_random.content.startswith("found after")
    assert app.md_status.content.endswith(": passes the checks")
    assert all(": PASS" in app.check_boxes[k].label for k in ck.CHECK_KEYS)
    assert all(cb.value for cb in app.rule_boxes.values()) and "the grammar (" in app.md_status.content
    assert app.md_limits.content == "this hand: follows the rules"


def test_panel_is_essential(app):
    assert set(app.check_boxes) == set(ck.CHECK_KEYS)
    assert not hasattr(app, "gui_preset") and app.limits() == SIMULATOR
    assert "require_digit_on_palm_body" not in app.limit_bools
    assert set(V.OPERATOR_INFO) == set(EVOLUTION_OPERATORS) | set(EVOLUTION_OPERATORS_FINE) == set(app.op_buttons)
    assert tuple(app.rule_boxes) == RULES and not hasattr(app, "gui_variant")
    assert "max_finger_length_mm" in app.limit_ints and app.stage() == "coarse"
    # no grammar codes or jargon in visible labels
    visible = [V.RULE_INFO[r][0] for r in RULES] + [V.OPERATOR_INFO[o][0] for o in V.OPERATOR_INFO]
    visible += [cb.label for cb in app.check_boxes.values()] + [lui_f.label for lui_f in lui.INT_FIELDS]
    for word in ("digit", "phalanx", "G_", "segment", "body", "capsule"):
        assert not any(word in v for v in visible), word
    for gone in ("gui_seed", "gallery", "sliders", "gui_orient", "md_analysis", "export_json", "gui_anim",
                 "md_variant", "md_verdict", "check_rows"):
        assert not hasattr(app, gone)
    # every visible line is short; explanations live in hover text
    assert all(len(cb.label) <= 50 for cb in app.check_boxes.values())


def test_limit_fields_and_reset(simulator):
    app = simulator
    assert app.limit_ints["max_finger_length_mm"].value == "250"
    app.set_limit("max_digits", 2)
    app.set_limit("max_finger_length_mm", "150")
    assert app.limits() == SIMULATOR.with_(max_digits=2, max_finger_length_mm=150.0)
    assert app.gui_rules.value == "Custom Rules"
    app.reset_limits()
    assert app.limits() == SIMULATOR and app.limit_ints["max_digits"].value == "5"
    assert app.gui_rules.value == "Evolution Rules"
    unlimit(app)
    assert app.gui_rules.value == "No Rules"
    app.reset_limits()
    assert app.limits() == SIMULATOR


def test_rules_dropdown(app):
    assert tuple(V.RULE_SETS) == ("Evolution Rules", "No Rules", "Custom Rules")
    assert app.gui_rules.value == "Evolution Rules" and app.limits() == SIMULATOR
    app.choose_rule_set("No Rules")
    assert app.limits() == UNLIMITED and app.limit_ints["max_digits"].value == lui.ANY
    app.set_limit("max_digits", 3)
    assert app.gui_rules.value == "Custom Rules" and app.limits() == UNLIMITED.with_(max_digits=3)
    app.choose_rule_set("Custom Rules")                      # keeps the fields as edited
    assert app.limits() == UNLIMITED.with_(max_digits=3)
    app.choose_rule_set("Evolution Rules")
    assert app.limits() == SIMULATOR and app.md_limits.content.startswith("this hand: ")


def test_random_uses_the_limits(simulator):
    app = simulator
    app.set_all_checks(False)
    unlimit(app)
    app.set_limit("max_digits", 2)
    app.set_limit("max_joints_per_digit", 2)
    for seed in range(5):
        assert app.random(start_seed=seed, wait=True)
        st = Structure.from_steps(app.shown.derivation.steps)
        assert len(st.top_digits()) <= 2 and max(st.joints_per_digit().values()) <= 2
        assert app.last_search.tries == 1
    assert app.md_limits.content == "this hand: follows the rules"


@pytest.mark.parametrize("key", ck.CHECK_KEYS)
def test_each_check_toggle_changes_random_tries(simulator, key):
    app = simulator
    rules, seed = failing_seed(key)
    app.set_all_checks(False)
    assert draw(app, rules, seed)
    assert app.last_search.tries == 1
    app.set_check(key, True)
    assert draw(app, rules, seed)
    assert app.last_search.tries > 1
    assert app.prep.ev.results[key].status == ck.PASS
    assert check(app.shown.derivation, SIMULATOR).ok


def test_switching_one_check_off_with_the_rest_on(simulator):
    app = simulator
    rules, seed = failing_seed("reach", only=True)
    assert draw(app, rules, seed)
    tries_all_on = app.last_search.tries
    app.set_check("reach", False)
    assert draw(app, rules, seed)
    assert app.last_search.tries == 1 < tries_all_on
    assert app.prep.ev.results["reach"].status == ck.FAIL
    app.set_check("reach", True)
    assert "fails: \u22652 fingertips reach object" in app.md_status.content
    assert "reach object*: FAIL" in app.check_boxes["reach"].label


def test_overlap_shown_in_red(simulator):
    app = simulator
    rules, seed = failing_seed("overlap_zero")
    app.set_all_checks(False)
    assert draw(app, rules, seed)
    assert app._highlight() == {}                           # red only where an ENABLED overlap check fails
    app.set_check("overlap_zero", True)
    hl = app._highlight()
    assert hl and all(c == gm.OVERLAP_RGB for c in hl.values())
    assert all(app.renderer.cur_rgb[b] == gm.OVERLAP_RGB for b in hl if b in app.renderer.caps)


def test_unbuildable_hand_reads_na(simulator):
    app = simulator
    unlimit(app)
    app.set_all_checks(True)
    ge = __import__("gviewer.envload", fromlist=["x"]).load_env_modules().grammar_envelope
    seed = next(s for s in range(100)
                if not ge._admit_structural(derive(sample_derivation(s, build_distribution(**ALL_OFF),
                                                                    limits=UNLIMITED))).ok)
    assert draw(app, tuple(ALL_OFF.items()), seed)
    assert app.last_search.tries == 1 and not app.prep.ev.buildable
    assert all(": n/a" in app.check_boxes[k].label for k in ck.CHECK_KEYS)
    assert "the simulator cannot build this hand" in app.md_status.content
    app.reset_limits()
    assert app.md_limits.content.startswith("this hand: breaks:")


def test_reach_display_toggle(simulator):
    app = simulator
    assert app.random(start_seed=0, wait=True)
    # off by default: no object, no fingertip dots
    assert app.spawn_handles and not any(h.visible for h in app.spawn_handles)
    assert app.renderer.tip_handles and not any(h.visible for h in app.renderer.tip_handles)
    app.gui_show_reach.value = True
    assert all(h.visible for h in app.spawn_handles) and all(h.visible for h in app.renderer.tip_handles)
    app.gui_show_reach.value = False
    assert not any(h.visible for h in app.spawn_handles)


def test_plain_colours_and_still_camera(simulator):
    app = simulator
    assert app.random(start_seed=0, wait=True)
    r = app.renderer
    assert r.arrows is None and r.root_axes is None and r.markers          # neutral joint markers only
    finger = {b: rgb for b, rgb in r.base_rgb.items() if b.startswith("d")}
    by_finger = {}
    for b, rgb in finger.items():
        by_finger.setdefault(b.split("p")[0], set()).add(rgb)
    assert all(len(c) == 1 for c in by_finger.values())                     # one colour per finger
    assert len({next(iter(c)) for c in by_finger.values()}) == len(by_finger)
    cam = tuple(app.server.initial_camera.position)
    keep = {f: next(iter(c)) for f, c in by_finger.items()}
    assert app.mutate("remove_digit", wait=True)                           # no reframe, no recolouring
    assert tuple(app.server.initial_camera.position) == cam
    for b, rgb in app.renderer.base_rgb.items():
        if b.startswith("d"):
            assert keep.get(b.split("p")[0], rgb) == rgb
    assert app.back(wait=True) and tuple(app.server.initial_camera.position) == cam
    app.recentre()


def test_operator_buttons_follow_the_limits(simulator):
    app = simulator
    assert app.random(start_seed=0, wait=True)
    b = app.op_buttons["add_branch_digit"]
    assert not b.visible and app.op_status["add_branch_digit"] == lui.NOT_ALLOWED
    assert app.op_status["step_coupling"] == lui.NOTHING and not app.op_buttons["step_coupling"].visible
    assert app.op_buttons["insert_phalanx"].visible
    assert app.op_buttons["add_palm_body"].label == "add a palm part with a short finger"
    assert app.mutate("add_branch_digit", wait=True)
    assert "not allowed by the rules" in app.md_mut.content and app.history.cursor == 0
    unlimit(app)
    # a palm part comes with a finger: a rule of the grammar, not a limit
    assert app.op_buttons["add_branch_digit"].visible
    assert app.op_buttons["add_palm_body"].label == "add a palm part with a short finger"
    assert app.mutate("add_branch_digit", wait=True) and app.history.cursor == 1


def test_mutation_back_and_ghost(simulator):
    app = simulator
    assert app.random(start_seed=0, wait=True)
    assert app.mutate("insert_phalanx", wait=True)
    assert app.history.cursor == 1 and app.parent_view is not None
    assert "add a joint to a finger" in app.md_mut.content and "joints" in app.md_mut.content
    assert app.mutate("step_coupling", wait=True)          # V3s hands have no coupled joints (button hidden)
    assert "nothing to act on" in app.md_mut.content and app.history.cursor == 1
    assert app.back(wait=True)
    assert app.history.cursor == 0 and app.parent_view is None and "undid" in app.md_mut.content
    for _ in range(10):
        assert app.mutate(None, wait=True)
        assert app.md_mut.content.startswith("random:")
        assert check(app.shown.derivation, SIMULATOR).ok


def test_curl_slider_moves_the_hand(app):
    assert app.random(start_seed=0, wait=True)
    app.set_curl(0.0)
    p0 = {n: h.position.copy() for n, h in app.renderer.caps.items()}
    app.set_curl(1.0)
    assert max(abs(h.position - p0[n]).max() for n, h in app.renderer.caps.items()) > 1e-3
    app.set_curl(V.RESET_CURL)


def test_commercial_hand(simulator):
    app = simulator
    from gviewer import commercial as com
    if com.hand_entry("allegro_right").availability != "available":
        pytest.skip("allegro URDF not available")
    assert app.random(start_seed=0, wait=True)
    # default: the projection snapped onto the grammar's fine grid
    assert app.gui_version.value == "fine grid"
    assert app.load_commercial("allegro_right", wait=True)
    assert app.shown.kind == "commercial" and app.shown.conformed is not None and app.overlay.handles
    line = app.md_fidelity.content
    assert line.startswith("snapped to the grammar's fine grid:") and "in the grammar: yes" in line
    assert "follows the rules: yes" in line
    pos = app.shown.conformed.fidelity["max_pos_mm"]
    assert pos <= 5.0                                                     # allegro: about 3 mm at the fine grid
    # the grid steps act on it
    assert app.op_buttons["step_limits"].visible and app.op_buttons["step_segment_length"].visible
    assert app.mutate("step_limits", wait=True) and app.history.cursor == 1
    assert app.mutate("insert_phalanx", wait=True) and app.history.cursor == 2
    # the coarse grid, and the exact projection, on request
    assert app.md_version.content.startswith("Fine grid")
    app.gui_version.value = "coarse grid"
    app.wait_idle()
    assert app.md_version.content.startswith("Coarse grid")
    assert app.md_fidelity.content.startswith("snapped to the grammar:") and app.history.cursor == 0
    assert app.shown.conformed.fidelity["max_pos_mm"] > pos
    app.gui_version.value = "exact (off-grid)"
    app.wait_idle()
    assert app.shown.conformed is None and app.history.cursor == 0
    line = app.md_fidelity.content
    assert line.startswith("exact projection:") and "in the grammar: no" in line
    app.gui_version.value = "fine grid"
    app.wait_idle()
    assert app.shown.conformed is not None
    app.gui_meshes.value = False
    assert not app.overlay.visible
    app.gui_meshes.value = True
    assert app.overlay.visible


def test_commercial_hand_outside_the_limits(simulator):
    app = simulator
    from gviewer import commercial as com
    if com.hand_entry("svh_right").availability != "available":
        pytest.skip("svh URDF not available")
    assert app.load_commercial("svh_right", wait=True)
    assert "follows the rules: no (fingers per palm joint 2 > 1)" in app.md_fidelity.content
    assert app.md_limits.content.startswith("this hand: breaks:")
    # it can still be mutated, as long as no limit gets worse
    assert app.mutate("step_segment_length", wait=True) and app.history.cursor == 1


def test_rule_toggles_redraw_with_the_rule(simulator):
    app = simulator
    assert app.random(start_seed=0, wait=True)
    r = next(s for s in app.shown.derivation.steps if s.path == "hand").params["capsule_radius_m"]
    top = [s for s in app.shown.derivation.steps if s.production == "Digit" and s.params["top_level"]]
    assert all(abs(abs(complex(*s.params["mount_offset"])) - r) < 1e-9 for s in top)       # on the surface
    app.rule_boxes["surface"].value = False                                                 # a new draw
    assert app.wait_idle() and app.rules["surface"] is False
    top = [s for s in app.shown.derivation.steps if s.production == "Digit" and s.params["top_level"]]
    assert all(s.params.get("mount_offset", (0.0, 0.0)) == (0.0, 0.0) for s in top)
    assert "sit on the palm surface" not in app.md_status.content


def test_coarse_and_fine_steps(simulator):
    app = simulator
    assert app.random(start_seed=0, wait=True)
    coarse = {op for op, b in app.op_buttons.items() if b.visible}
    assert coarse and coarse <= set(EVOLUTION_OPERATORS)
    app.set_stage("fine")
    fine = {op for op, b in app.op_buttons.items() if b.visible}
    assert fine and fine <= set(EVOLUTION_OPERATORS_FINE)
    assert {"fine_step_axis", "fine_step_segment_length", "fine_slide_mount"} <= fine
    assert app.op_buttons["fine_step_segment_length"].label == "lengthen/shorten a bone 1 mm"
    for _ in range(5):                                   # random mutation draws from the fine pool
        assert app.mutate(None, wait=True)
        assert app.history.current.operator in EVOLUTION_OPERATORS_FINE
        assert Structure.from_steps(app.shown.derivation.steps).digits == \
            Structure.from_steps(app.history.entries[0].derivation.steps).digits    # never the structure
    assert app.mutate("fine_step_axis", wait=True) and "tilt a joint axis 5°" in app.md_mut.content
    app.set_stage("coarse")
    assert {op for op, b in app.op_buttons.items() if b.visible} <= set(EVOLUTION_OPERATORS)


def test_fine_steps_on_a_commercial_hand(simulator):
    app = simulator
    from gviewer import commercial as com
    if com.hand_entry("leap_right").availability != "available":
        pytest.skip("leap URDF not available")
    assert app.load_commercial("leap_right", wait=True)
    app.set_stage("fine")
    shown = {op for op, b in app.op_buttons.items() if b.visible}
    assert shown == set(EVOLUTION_OPERATORS_FINE), set(EVOLUTION_OPERATORS_FINE) - shown
    assert app.mutate("fine_step_segment_length", wait=True) and app.history.cursor == 1
    app.set_stage("coarse")
    assert app.op_buttons["step_axis"].visible and app.op_buttons["insert_phalanx"].visible
