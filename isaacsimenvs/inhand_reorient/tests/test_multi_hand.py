"""Multi-hand scene layout (robots/multi_hand.py): slots, per-env tables, env blocks. Kit-free."""
from __future__ import annotations

import importlib.util
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
def mh():
    import isaacsimenvs.pose_reaching_6d  # noqa: F401
    for n in ("isaacsimenvs.pose_reaching_6d.scene_utils", PKG + ".adjacency"):
        if n not in sys.modules:
            m = types.ModuleType(n); m.__path__ = []; sys.modules[n] = m
    pkg = types.ModuleType(PKG); pkg.__path__ = [str(ROOT / "isaacsimenvs/pose_reaching_6d/scene_utils/robots")]
    sys.modules[PKG] = pkg
    for mod in ("sharpa_iiwa14", "sharpa_handonly"):
        _load(f"{PKG}.adjacency.{mod}", f"isaacsimenvs/pose_reaching_6d/scene_utils/robots/adjacency/{mod}.py")
    _load(PKG + ".sharpa_handonly", "isaacsimenvs/pose_reaching_6d/scene_utils/robots/sharpa_handonly.py")
    _load(PKG + ".unified_hands", "isaacsimenvs/pose_reaching_6d/scene_utils/robots/unified_hands.py")
    uni = _load(PKG + ".unified_dynamics_hands", "isaacsimenvs/pose_reaching_6d/scene_utils/robots/unified_dynamics_hands.py")
    pkg.REGISTRY = {s.name: s for s in uni.UNIFORM_LEFT.values()}
    return _load(PKG + ".multi_hand", "isaacsimenvs/pose_reaching_6d/scene_utils/robots/multi_hand.py")


def test_template_pads_to_the_largest_hand(mh):
    hs = mh.hand_set("multi:uniform")
    assert hs.n_hands == 8
    assert hs.n_joints == max(s.num_hand_joints for s in hs.specs) == 22
    assert hs.n_tips == 5
    t = hs.template
    assert t.num_hand_joints == 22 and t.num_fingertips == 5 and t.num_arm_joints == 0
    assert t.hand_joint_names == tuple(f"s{k}" for k in range(22))
    t.validate()


def test_per_env_tables_follow_each_hand(mh):
    hs = mh.hand_set("multi:allegro_left_uniform_handonly+dex3_left_uniform_handonly")
    idx = hs.hand_of_env(10)
    assert idx.tolist() == [0, 1] * 5                         # round-robin across envs
    t = hs.per_env(idx)
    allegro, dex3 = hs.specs
    assert t["joint_valid"][0].sum() == allegro.num_hand_joints == 16
    assert t["joint_valid"][9].sum() == dex3.num_hand_joints == 7
    assert t["fingertip_valid"][0].sum() == 4 and t["fingertip_valid"][9].sum() == 3
    np.testing.assert_allclose(t["joint_link_bbox_local"][9, :7], np.asarray(dex3.joint_link_boxes, np.float32))
    assert not t["joint_link_bbox_local"][9, 7:].any()          # ghost slots carry nothing
    # every 2048-env SAPG block holds every hand
    full = mh.hand_set("multi:uniform").hand_of_env(12288)
    for b in range(6):
        assert len(np.unique(full[b * 2048:(b + 1) * 2048])) == 8
    np.testing.assert_allclose(t["palm_keypoints"][0], np.asarray(allegro.palm_keypoints, np.float32))
    np.testing.assert_allclose(t["fingertip_offsets"][9, :3], np.asarray(dex3.fingertip_offsets, np.float32))


def test_uneven_split_and_bad_refs(mh):
    hs = mh.hand_set("multi:uniform")
    sizes = np.bincount(hs.hand_of_env(12289), minlength=8)
    assert sizes.max() - sizes.min() <= 1 and sizes.sum() == 12289
    with pytest.raises(KeyError):
        mh.hand_set("multi:not_a_hand")
