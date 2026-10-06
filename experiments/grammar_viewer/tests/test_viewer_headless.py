"""The viewer's callbacks, driven directly against a real (local) viser server."""

import json
import socket
import urllib.request

import pytest

viser = pytest.importorskip("viser")

import viewer_full as V  # noqa: E402


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    port = _free_port()
    server = viser.ViserServer(host="127.0.0.1", port=port, verbose=False)
    a = V.GrammarViewer(server, variant="G_V3S", seed=0, until_viable=True,
                        out_dir=tmp_path_factory.mktemp("export"))
    a._port = port
    yield a
    a.set_animating(False)
    server.stop()


def test_http_and_initial_design(app):
    assert urllib.request.urlopen(f"http://127.0.0.1:{app._port}", timeout=10).status == 200
    assert app.cur is not None and app.cur.kind == "sampled"
    assert app.prep.analysis.viable
    assert "viable after" in app.md_sample.content
    assert set(app.sliders) == {r.name for r in app.prep.view.ranges}


def test_pose_buttons(app):
    app.pose_reset()
    reset = app.prep.analysis.reset_u()
    assert all(abs(app.u[k] - v) < 1e-12 for k, v in reset.items())
    app.pose_curl(1.0)
    assert all(abs(app.u[r.name] - r.hi) < 1e-12 for r in app.prep.view.ranges)
    app.pose_random()
    app.pose_zero()
    h, scale = next(iter(app.sliders.values()))
    assert abs(h.value) < 1e-9 or h.min > 0


def test_mutation_history(app):
    root_label = app.history.entries[0].label
    assert app.mutate("insert_phalanx", wait=True)
    assert app.history.cursor == 1 and app.history.current.operator == "insert_phalanx"
    assert "joints" in app.md_diff.content
    assert app.parent_view is not None  # ghost built
    assert app.mutate("remove_branch_digit", wait=True)  # inapplicable on V3s hands
    assert "VariationImpossible" in app.md_mut_status.content and app.history.cursor == 1
    assert app.back(wait=True) and app.history.cursor == 0 and app.cur.label == root_label
    assert app.forward(wait=True) and app.history.cursor == 1


def test_commercial_and_files(app):
    if com_available("allegro_right"):
        assert app.load_commercial("allegro_right", wait=True)
        assert app.cur.kind == "commercial"
        assert "PASS" in app.md_commercial.content and "fits" in app.md_commercial.content
        assert app.overlay.handles
    pop = V.REPO_ROOT / "outputs" / "viable" / "mixed32" / "population.json"
    if pop.is_file():
        assert app.load_file(str(pop), wait=True)
        assert app.cur.kind == "file" and "32 design" in app.md_file.content or "entry" in app.md_file.content


def test_gallery_spread_export(app):
    assert app.build_gallery("G_V1", "G_V3S", 2, 2, 0, 0.35, wait=True)
    assert len(app.gallery) == 4 and "viable" in app.md_gallery.content
    app.clear_gallery()
    assert not app.gallery
    assert app.build_spread(4, 0.4, wait=True)
    assert "random" in app.md_spread.content
    p = app.export_json()
    assert json.loads(p.read_text())["schema"].startswith("hand_grammar_derivation")
    out = app.export_urdf()
    assert out["urdf"].read_text().startswith("<?xml")
    h = app.export_history()
    assert json.loads(h.read_text())["schema"] == "grammar_viewer_history/0.1"


def com_available(hand_id):
    from gviewer import commercial as com
    return com.hand_entry(hand_id).availability == "available"
