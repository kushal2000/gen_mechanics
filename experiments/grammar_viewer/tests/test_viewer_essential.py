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
from hand_sampler.grammar.derive import EVOLUTION_OPERATORS, derive, sample_derivation  # noqa: E402
from hand_sampler.grammar.kinematics import ModelError  # noqa: E402
from hand_sampler.grammar.limits import SIMULATOR, UNLIMITED, Structure, check  # noqa: E402


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
    a = V.EssentialViewer(server, variant="G_V3S", start_seed=0)
    a._port = port
    yield a
    server.stop()


@pytest.fixture
def simulator(app):
    app.set_preset("Simulator")
    app.set_all_checks(True)
    yield app
    app.set_preset("Simulator")
    app.set_all_checks(True)


@functools.lru_cache(maxsize=None)
def failing_seed(key: str, only: bool = False):
    """(variant, seed) of a design sampled under SIMULATOR that fails `key`
    (with `only`: and passes the other checks)."""
    for variant in ("G_V1", "G_V3S", "G_FULL"):
        dist = src.distribution(variant)
        for seed in range(600):
            try:
                m = derive(sample_derivation(seed, dist, limits=SIMULATOR))
            except ModelError:
                continue
            if only:
                if ck.evaluate(m).failing(ck.CHECK_KEYS) == [key]:
                    return variant, seed
            elif ck.evaluate(m, {key}, stop_early=True).results[key].status == ck.FAIL:
                return variant, seed
    raise AssertionError(f"no sampled design fails {key}")


def test_http_and_initial_design(app):
    assert urllib.request.urlopen(f"http://127.0.0.1:{app._port}", timeout=10).status == 200
    assert app.shown.kind == "sampled"
    assert app.md_random.content.startswith("found after")
    assert app.md_status.content.endswith(": viable")
    assert all(": PASS" in app.check_boxes[k].label for k in ck.CHECK_KEYS)
    assert app.gui_variant.hint == V.VARIANT_NOTES["G_V3S"]
    assert app.md_limits.content == "this hand: within limits"


def test_panel_is_essential(app):
    assert set(app.check_boxes) == set(ck.CHECK_KEYS)
    assert app.gui_preset.value == "Simulator" and app.limits() == SIMULATOR
    assert set(V.OPERATOR_INFO) == set(EVOLUTION_OPERATORS) == set(app.op_buttons)
    assert set(V.VARIANT_NOTES) == set(src.variant_names())
    for gone in ("gui_seed", "gallery", "sliders", "gui_orient", "md_analysis", "export_json", "gui_anim",
                 "md_variant", "md_verdict", "check_rows"):
        assert not hasattr(app, gone)
    # every visible line is short; explanations live in hover text
    assert all(len(cb.label) <= 40 for cb in app.check_boxes.values())


def test_presets_and_custom(simulator):
    app = simulator
    app.set_preset("Unlimited")
    assert app.limits() == UNLIMITED
    app.set_limit("max_digits", 2)
    assert app.gui_preset.value == "Custom" and app.limits().max_digits == 2
    app.set_preset("Simulator")
    assert app.limits() == SIMULATOR and app.limit_ints["max_digits"].value == "5"
    app.set_limit("max_digits", "5")                 # same values as a preset: shown as that preset
    assert app.gui_preset.value == "Simulator"


def test_random_uses_the_limits(simulator):
    app = simulator
    app.set_all_checks(False)
    app.set_preset("Unlimited")
    app.set_limit("max_digits", 2)
    app.set_limit("max_joints_per_digit", 2)
    for seed in range(5):
        assert app.random("G_FULL", start_seed=seed, wait=True)
        st = Structure.from_steps(app.shown.derivation.steps)
        assert len(st.top_digits()) <= 2 and max(st.joints_per_digit().values()) <= 2
        assert app.last_search.tries == 1
    assert app.md_limits.content == "this hand: within limits"


@pytest.mark.parametrize("key", ck.CHECK_KEYS)
def test_each_check_toggle_changes_random_tries(simulator, key):
    app = simulator
    variant, seed = failing_seed(key)
    app.set_all_checks(False)
    assert app.random(variant, start_seed=seed, wait=True)
    assert app.last_search.tries == 1
    app.set_check(key, True)
    assert app.random(variant, start_seed=seed, wait=True)
    assert app.last_search.tries > 1
    assert app.prep.ev.results[key].status == ck.PASS
    assert check(app.shown.derivation, SIMULATOR).ok


def test_switching_one_check_off_with_the_rest_on(simulator):
    app = simulator
    variant, seed = failing_seed("reach", only=True)
    assert app.random(variant, start_seed=seed, wait=True)
    tries_all_on = app.last_search.tries
    app.set_check("reach", False)
    assert app.random(variant, start_seed=seed, wait=True)
    assert app.last_search.tries == 1 < tries_all_on
    assert app.prep.ev.results["reach"].status == ck.FAIL
    app.set_check("reach", True)
    assert "fails 2 tips reach cube" in app.md_status.content
    assert "reach cube*: FAIL" in app.check_boxes["reach"].label


def test_overlap_shown_in_red(simulator):
    app = simulator
    variant, seed = failing_seed("overlap_zero")
    app.set_all_checks(False)
    assert app.random(variant, start_seed=seed, wait=True)
    hl = app._highlight()
    assert hl and all(c == gm.OVERLAP_RGB for c in hl.values())
    assert all(app.renderer.cur_rgb[b] == gm.OVERLAP_RGB for b in hl if b in app.renderer.caps)


def test_unbuildable_hand_reads_na(simulator):
    app = simulator
    app.set_preset("Unlimited")
    app.set_all_checks(True)
    ge = __import__("gviewer.envload", fromlist=["x"]).load_env_modules().grammar_envelope
    seed = next(s for s in range(100)
                if not ge._admit_structural(derive(sample_derivation(s, src.distribution("G_FULL")))).ok)
    assert app.random("G_FULL", start_seed=seed, wait=True)
    assert app.last_search.tries == 1 and not app.prep.ev.buildable
    assert all(": n/a" in app.check_boxes[k].label for k in ck.CHECK_KEYS)
    assert "outside the simulator" in app.md_status.content
    app.set_preset("Simulator")
    assert app.md_limits.content.startswith("this hand: outside:")


def test_reach_display_toggle(simulator):
    app = simulator
    assert app.random("G_V3S", start_seed=0, wait=True)
    assert app.spawn_handles and all(h.visible for h in app.spawn_handles)
    app.gui_show_reach.value = False
    assert not any(h.visible for h in app.spawn_handles)
    app.gui_show_reach.value = True
    assert all(h.visible for h in app.spawn_handles)


def test_operator_buttons_follow_the_limits(simulator):
    app = simulator
    assert app.random("G_V3S", start_seed=0, wait=True)
    b = app.op_buttons["add_branch_digit"]
    assert b.disabled and app.op_status["add_branch_digit"] == lui.NOT_ALLOWED and "not allowed by limits" in b.hint
    assert app.op_status["step_coupling"] == lui.NOTHING and app.op_buttons["step_coupling"].disabled
    assert not app.op_buttons["insert_phalanx"].disabled
    assert app.mutate("add_branch_digit", wait=True)
    assert "not allowed by limits" in app.md_mut.content and app.history.cursor == 0
    app.set_preset("Unlimited")
    assert not app.op_buttons["add_branch_digit"].disabled
    assert app.mutate("add_branch_digit", wait=True) and app.history.cursor == 1


def test_mutation_back_and_ghost(simulator):
    app = simulator
    assert app.random("G_V3S", start_seed=0, wait=True)
    assert app.mutate("insert_phalanx", wait=True)
    assert app.history.cursor == 1 and app.parent_view is not None
    assert "add a segment" in app.md_mut.content and "joints" in app.md_mut.content
    assert app.mutate("step_coupling", wait=True)          # V3s hands have no coupled joints
    assert "nothing to act on" in app.md_mut.content and app.history.cursor == 1
    assert app.back(wait=True)
    assert app.history.cursor == 0 and app.parent_view is None and "undid" in app.md_mut.content
    for _ in range(10):
        assert app.mutate(None, wait=True)
        assert app.md_mut.content.startswith("random:")
        assert check(app.shown.derivation, SIMULATOR).ok


def test_curl_slider_moves_the_hand(app):
    assert app.random("G_V3S", start_seed=0, wait=True)
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
    assert app.random("G_V1", start_seed=0, wait=True)
    # default: the projection snapped onto the variant's grids
    assert app.load_commercial("allegro_right", wait=True)
    assert app.shown.kind == "commercial" and app.shown.conformed is not None and app.overlay.handles
    line = app.md_fidelity.content
    assert line.startswith("snapped to G_V1:") and "within rules: yes" in line and "within limits: yes" in line
    # every grid step applies to it now (step_limits cannot act on the exact projection)
    assert not app.op_buttons["step_limits"].disabled and not app.op_buttons["step_segment_length"].disabled
    assert app.mutate("step_limits", wait=True) and app.history.cursor == 1
    assert app.mutate("insert_phalanx", wait=True) and app.history.cursor == 2
    # the exact projection on request
    app.gui_exact.value = True
    app.wait_idle()
    assert app.shown.conformed is None and app.history.cursor == 0
    line = app.md_fidelity.content
    assert line.startswith("exact projection:") and "within rules: no" in line
    assert app.op_buttons["step_limits"].disabled
    app.gui_exact.value = False
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
    assert "within limits: no (digits per jointed palm body 2 > 1)" in app.md_fidelity.content
    assert app.md_limits.content.startswith("this hand: outside:")
    # it can still be mutated, as long as no limit gets worse
    assert app.mutate("step_segment_length", wait=True) and app.history.cursor == 1
