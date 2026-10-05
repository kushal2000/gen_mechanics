"""The by-path loader of the simulator's numpy-only envelope modules."""

import sys

import numpy as np
import pytest

from gviewer import analysis as an
from gviewer import sources as src
from gviewer.envload import load_env_modules


def test_loads_without_isaac():
    mods = load_env_modules()
    assert mods.stubbed, "the real isaacsimenvs package should not be imported in this process"
    assert "isaaclab" not in sys.modules
    assert getattr(sys.modules["isaacsimenvs"], "__gviewer_stub__", False)
    ge = mods.grammar_envelope
    assert ge.N_SLOTS == 32
    assert ge.MIN_SPAWN_HEIGHT_ABOVE_PALM_M == mods.palm_calibration.MIN_SPAWN_HEIGHT_ABOVE_PALM_M
    # the relative `..palm_calibration` import resolved to the module loaded by path
    assert sys.modules["isaacsimenvs.inhand_reorient.palm_calibration"] is mods.palm_calibration
    assert load_env_modules() is mods  # cached


def test_archive_bins_match_driver():
    ar = load_env_modules().archive
    assert ar.N_CELLS == 30
    assert ar.descriptor(4, 14) == (3, 2)
    assert ar.cell_label((3, 2)) == "d4_j11-15"
    cell, label = an.descriptor_for(1, 5)
    assert cell == (0, 0) and label == "d1_j1-5"


@pytest.mark.parametrize("variant,seed", [("G_V3S", 8), ("G_V1", 3), ("G_SERIAL", 2)])
def test_analysis_matches_viability_report(variant, seed):
    ge = load_env_modules().grammar_envelope
    _d, m = src.sample(variant, seed)
    a = an.analyze(m)
    rep = ge.viability_report(m)
    assert a.report == rep
    if rep["digit_count"] is None:
        assert a.design is None and not a.structural_ok
        return
    # per-finger reach replay agrees with palm_up's count (analyze asserts it too)
    assert sum(r.reaches for r in a.finger_reach) == rep["fingertips_reachable"]
    # overlap pairs reproduce the report's worst overlap
    worst = max([p for *_x, p in a.pairs_q0 + a.pairs_reset], default=0.0) * 1000.0
    assert worst == pytest.approx(rep["max_rest_overlap_mm"], abs=1e-9)
    # reset pose round trip: slot vector -> model joints -> slot vector
    q = a.reset_u()
    assert np.allclose(an.model_q_to_slot(a.design, q)[a.design.slot_valid], a.pu.default_q[a.design.slot_valid])
    # overlaps_at at the reset pose equals the oracle's reset-pose pairs
    assert sorted(an.overlaps_at(a.design, q)) == sorted(a.pairs_reset)
    cell, label = an.descriptor_for(rep["digit_count"], rep["joint_count"])
    assert a.descriptor_label == label


def test_viable_seed_g_v3s():
    res = src.sample_until_viable("G_V3S", 0, max_tries=200)
    assert res.derivation is not None and res.viable
    assert res.report["admitted"] and res.report["fingertips_reachable"] >= 2
    assert res.tries == res.seed + 1  # seeds 0..seed tried in order
