"""The onboarded commercial hands: every spec is valid, and the pipeline reproduces the hand-written Allegro.

Kit-free: the scene_utils package __init__ imports pxr, so the robot modules are loaded by path.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG = "isaacsimenvs.pose_reaching_6d.scene_utils.robots"


def _load(name, rel):
    sp = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mods():
    import isaacsimenvs.pose_reaching_6d  # noqa: F401
    for n in ("isaacsimenvs.pose_reaching_6d.scene_utils", PKG, PKG + ".adjacency"):
        if n not in sys.modules:
            m = types.ModuleType(n); m.__path__ = []; sys.modules[n] = m
    _load(PKG + ".adjacency.allegro_handonly",
          "isaacsimenvs/pose_reaching_6d/scene_utils/robots/adjacency/allegro_handonly.py")
    allegro = _load(PKG + ".allegro_handonly",
                    "isaacsimenvs/pose_reaching_6d/scene_utils/robots/allegro_handonly.py")
    unified = _load(PKG + ".unified_hands",
                    "isaacsimenvs/pose_reaching_6d/scene_utils/robots/unified_hands.py")
    return allegro, unified


def test_every_onboarded_left_hand_is_a_valid_spec(mods):
    _, unified = mods
    assert set(unified.UNIFIED_LEFT) == {"allegro", "leap", "shadow", "dex3", "tesollo", "wuji2", "xhand"}
    for spec in unified.UNIFIED_LEFT.values():
        spec.validate()
        assert spec.name.endswith("_handonly")          # common.sh's hand-only naming rule
        assert all(spec.hand_default_joint_pos[n] is not None for n in spec.hand_joint_names)


@pytest.mark.parametrize("hand", ["allegro", "leap", "shadow", "dex3", "tesollo", "wuji2", "xhand"])
def test_palm_frame_is_a_proper_rotation_and_every_check_passed(hand):
    s = json.loads((ROOT / f"assets/urdf/unified_commercial_hands/{hand}/{hand}_left.spec.json").read_text())
    F = np.asarray(s["palm_frame"])
    assert np.allclose(F.T @ F, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(F), 1.0)
    report = (ROOT / f"assets/urdf/unified_commercial_hands/{hand}/{hand}_left.report.md").read_text()
    assert "**FAIL**" not in report


def test_pipeline_reproduces_the_hand_written_allegro(mods):
    """allegro_handonly.py was measured by hand; onboard.py + unified_hands.py must agree with it."""
    allegro, unified = mods
    A = allegro.ALLEGRO_HANDONLY
    G = unified.spec_from_json(unified.HANDS_DIR / "allegro" / "allegro_right.spec.json", name="check")
    assert tuple(G.hand_joint_names) == tuple(A.hand_joint_names)
    assert tuple(G.fingertip_body_names) == tuple(A.fingertip_body_names)
    for f in ("base_pos", "base_rot", "palm_center_offset", "palm_keypoints"):
        assert np.allclose(np.array(getattr(G, f), float), np.array(getattr(A, f), float), atol=2e-4), f
    # the pad point differs by the method (vendor tip frame vs mesh) but must be within 1 cm
    assert np.abs(np.array(G.fingertip_offsets) - np.array(A.fingertip_offsets)).max() < 0.01
