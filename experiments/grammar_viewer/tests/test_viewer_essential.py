"""The essential viewer (viewer.py): its callbacks driven against a real local
viser server, and each viability check's toggle changing Random's tries."""

import functools
import socket
import urllib.request

import pytest

viser = pytest.importorskip("viser")

import viewer as V  # noqa: E402
from gviewer import checks as ck  # noqa: E402
from gviewer import model as gm  # noqa: E402
from gviewer import sources as src  # noqa: E402
from hand_sampler.grammar.derive import EVOLUTION_OPERATORS, derive, sample_derivation  # noqa: E402
from hand_sampler.grammar.kinematics import ModelError  # noqa: E402


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


@functools.lru_cache(maxsize=None)
def failing_seed(key: str, only: bool = False):
    """(variant, seed) of a design that fails `key` (with `only`: and passes
    every other check)."""
    order = ("G_V1", "G_V3S", "G_FULL") if key in ck.PHYSICAL_KEYS else ("G_FULL", "G_V1", "G_V3S")
    for variant in order:
        dist = src.distribution(variant)
        for seed in range(600):
            try:
                m = derive(sample_derivation(seed, dist))
            except ModelError:
                continue
            if only:
                ev = ck.evaluate(m)
                if ev.failing(ck.CHECK_KEYS) == [key]:
                    return variant, seed
            elif ck.evaluate(m, {key}, stop_early=True).results[key].status == ck.FAIL:
                return variant, seed
    raise AssertionError(f"no sampled design fails {key}")


def test_http_and_initial_design(app):
    assert urllib.request.urlopen(f"http://127.0.0.1:{app._port}", timeout=10).status == 200
    assert app.shown.kind == "sampled"
    assert "found after" in app.md_random.content
    assert "passes every enabled check" in app.md_verdict.content
    assert all(app.check_rows[k].content.startswith("PASS") for k in ck.CHECK_KEYS)
    assert app.md_variant.content == V.VARIANT_NOTES["G_V3S"]


def test_panel_is_essential(app):
    assert set(app.check_boxes) == set(ck.CHECK_KEYS)
    assert all(cb.value for cb in app.check_boxes.values())
    assert set(V.OPERATOR_INFO) == set(EVOLUTION_OPERATORS)
    assert set(V.VARIANT_NOTES) == set(src.variant_names())
    for gone in ("gui_seed", "gallery", "sliders", "gui_orient", "md_analysis", "export_json", "gui_anim"):
        assert not hasattr(app, gone)


@pytest.mark.parametrize("key", ck.CHECK_KEYS)
def test_each_check_toggle_changes_random_tries(app, key):
    variant, seed = failing_seed(key)
    try:
        app.set_all_checks(False)
        assert app.random(variant, start_seed=seed, wait=True)
        assert app.last_search.tries == 1
        app.set_check(key, True)
        assert app.random(variant, start_seed=seed, wait=True)
        assert app.last_search.tries > 1
        # PASS, or n/a on a hand the simulator cannot build (structural checks are off here)
        assert app.prep.ev.results[key].status != ck.FAIL
        assert "passes every enabled check" in app.md_verdict.content
    finally:
        app.set_all_checks(True)


@pytest.mark.parametrize("key", ["reach", "carrier_one_digit"])
def test_switching_one_check_off_with_the_rest_on(app, key):
    variant, seed = failing_seed(key, only=True)
    assert app.random(variant, start_seed=seed, wait=True)
    tries_all_on = app.last_search.tries
    try:
        app.set_check(key, False)
        assert app.random(variant, start_seed=seed, wait=True)
        assert app.last_search.tries == 1 < tries_all_on
        assert app.prep.ev.results[key].status == ck.FAIL
    finally:
        app.set_all_checks(True)
    assert "**fails:**" in app.md_verdict.content  # the shown design fails the re-enabled check


def test_overlap_shown_in_red(app):
    variant, seed = failing_seed("overlap_zero")
    try:
        app.set_all_checks(False)
        assert app.random(variant, start_seed=seed, wait=True)
    finally:
        app.set_all_checks(True)
    hl = app._highlight()
    assert hl and all(c == gm.OVERLAP_RGB for c in hl.values())
    assert all(app.renderer.cur_rgb[b] == gm.OVERLAP_RGB for b in hl if b in app.renderer.caps)


def test_reach_display_toggle(app):
    assert app.random("G_V3S", start_seed=0, wait=True)
    assert app.spawn_handles and all(h.visible for h in app.spawn_handles)
    app.gui_show_reach.value = False
    assert not any(h.visible for h in app.spawn_handles)
    app.gui_show_reach.value = True
    assert all(h.visible for h in app.spawn_handles)


def test_mutation_back_and_ghost(app):
    assert app.random("G_V3S", start_seed=0, wait=True)
    assert app.mutate("insert_phalanx", wait=True)
    assert app.history.cursor == 1 and app.parent_view is not None
    assert "add a segment" in app.md_mut.content and "joints" in app.md_mut.content
    assert app.mutate("step_coupling", wait=True)          # V3s hands have no coupled joints
    assert "cannot apply" in app.md_mut.content and app.history.cursor == 1
    assert app.back(wait=True)
    assert app.history.cursor == 0 and app.parent_view is None and "undid" in app.md_mut.content
    assert app.mutate(None, wait=True)
    assert app.history.cursor == 1 and app.md_mut.content.startswith("random:")


def test_curl_slider_moves_the_hand(app):
    assert app.random("G_V3S", start_seed=0, wait=True)
    app.set_curl(0.0)
    p0 = {n: h.position.copy() for n, h in app.renderer.caps.items()}
    app.set_curl(1.0)
    assert max(abs(h.position - p0[n]).max() for n, h in app.renderer.caps.items()) > 1e-3
    app.set_curl(V.RESET_CURL)


def test_commercial_hand(app):
    from gviewer import commercial as com
    if com.hand_entry("allegro_right").availability != "available":
        pytest.skip("allegro URDF not available")
    assert app.load_commercial("allegro_right", wait=True)
    assert app.shown.kind == "commercial" and app.overlay.handles
    assert "matches URDF within" in app.md_fidelity.content and "fits simulator: yes" in app.md_fidelity.content
    app.gui_meshes.value = False
    assert not app.overlay.visible
    app.gui_meshes.value = True
    assert app.overlay.visible
